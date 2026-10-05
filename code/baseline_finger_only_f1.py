"""
Computes macro-F1 for the Phase 2 "finger-only" PADS probes (waveform-only and
fused-with-absent-mPower-token) on the full, correctly-batched test set, for a
fair apples-to-apples comparison against the Phase 4 Cross-Body ablation table
(both metrics, same test set, same code path avoiding the batch-drop bug fixed
in §6.4 of the documentation).

Usage:
    python baseline_finger_only_f1.py --data-root "."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from model import CBDLPhase2Model
from train_probes import LinearProbe, load_pads, get_device, majority_baseline


def run(args) -> None:
    device = get_device()
    root = Path(args.data_root)

    pads_w, pads_m, pads_y, pads_split, pads_classes = load_pads(root)

    model = CBDLPhase2Model().to(device)
    ckpt = torch.load(root / "phase2_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    probe_fused = LinearProbe(128, 3).to(device)
    probe_fused.load_state_dict(ckpt["probe_pads"])
    probe_wf = LinearProbe(128, 3).to(device)
    probe_wf.load_state_dict(ckpt["probe_pads_waveform_only"])
    model.eval(); probe_fused.eval(); probe_wf.eval()

    test_idx = np.where(pads_split == "test")[0]
    x = torch.from_numpy(pads_w[test_idx]).float().to(device)
    m = torch.from_numpy(pads_m[test_idx]).bool().to(device)
    y = pads_y[test_idx]

    with torch.no_grad():
        wf_embed = model.encode_finger_waveform(x, m, source="pads")
        fused = model.encode_finger_fused(wf_embed, None, x.size(0), device)
        preds_fused = probe_fused(fused).argmax(-1).cpu().numpy()
        preds_wf = probe_wf(wf_embed).argmax(-1).cpu().numpy()

    print("=" * 70)
    print("PHASE 2 FINGER-ONLY BASELINE — full test set (71 samples), macro-F1")
    print("=" * 70)
    print(f"Waveform-only (no mPower fusion): acc={(preds_wf==y).mean():.3f}  macro-F1={f1_score(y, preds_wf, average='macro'):.3f}")
    print(f"Fused (with absent-mPower token):  acc={(preds_fused==y).mean():.3f}  macro-F1={f1_score(y, preds_fused, average='macro'):.3f}")
    print(f"Majority baseline: {majority_baseline(y):.3f}")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
