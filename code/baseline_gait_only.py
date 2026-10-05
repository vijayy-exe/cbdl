"""
CBDL Phase 7 — "gait-only" baseline for PADS's classification task.

Per the plan's baseline list: finger-only, gait-only, naive-concat fusion,
and finger-waveform-only vs. finger-waveform-plus-mPower — together isolate
how much each component actually contributes.

"Gait-only" for PADS's own label is a degenerate but informative baseline:
since PADS and GaitRec share no subjects, there IS no real per-patient gait
signal to give a PADS classifier — the only gait information available for
any PADS sample is the same fixed population gait prototype used everywhere
else in this pipeline (see train_phase4_clinical_head.py). Feeding a
classifier ONLY that constant, patient-independent input should collapse to
predicting the majority class for every sample, by construction — this
script verifies that empirically rather than asserting it, and reports it as
the "0% real signal" floor the fused/finger-only results should be compared
against.

Usage:
    python baseline_gait_only.py --data-root "."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from train_phase4_clinical_head import ClassifierHead, get_device, load_pads, majority_baseline


def run(args) -> None:
    device = get_device()
    root = Path(args.data_root)

    pads_w, pads_m, pads_y, pads_split = load_pads(root)
    phase4_ckpt = torch.load(root / "phase4_checkpoint.pt", map_location=device)
    gait_prototype = phase4_ckpt["gait_prototype"].to(device).mean(dim=1)  # [1, 128] pooled

    head = ClassifierHead(128, 64, 3).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    counts = np.bincount(pads_y[pads_split == "train"], minlength=3).astype(np.float64)
    weight = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weight)

    def run_split(split, train):
        head.train(train)
        idx = np.where(pads_split == split)[0]
        y = torch.from_numpy(pads_y[idx]).long().to(device)
        x = gait_prototype.expand(len(idx), -1)
        with torch.set_grad_enabled(train):
            logits = head(x)
            loss = ce(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        return logits.argmax(-1).detach().cpu().numpy(), y.cpu().numpy()

    for epoch in range(1, args.epochs + 1):
        run_split("train", True)

    test_preds, test_targets = run_split("test", False)
    acc = (test_preds == test_targets).mean()
    f1 = f1_score(test_targets, test_preds, average="macro")

    print("=" * 70)
    print("PHASE 7 BASELINE — GAIT-ONLY (constant population prototype, no per-patient signal)")
    print("=" * 70)
    print(f"Test accuracy: {acc:.3f}  macro-F1: {f1:.3f}  (majority baseline: {majority_baseline(pads_y[pads_split=='test']):.3f})")
    print(f"Predictions (should all collapse to one class): {np.unique(test_preds, return_counts=True)}")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
