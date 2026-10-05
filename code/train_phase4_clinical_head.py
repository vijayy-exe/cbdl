"""
CBDL Phase 4 — Clinical Grounding (PADS-supervised).

Per CBDL_Development_Plan.md Phase 4: a classification head on top of the
FUSED embedding (Phase 3's Cross-Body Dependency Module output), supervised
PRIMARILY by PADS's real diagnostic labels (Healthy/PD/Other), with
GaitRec's own pathology label used as a secondary gait-side check.

A real design problem this script has to solve, stated plainly: classifying
a PADS sample requires fusing it with SOME gait representation (that's what
"the fused embedding" means), but no PADS sample has a genuine paired gait
recording — GaitRec and PADS share no subjects (see cross_body_module.py).
Fusing each PADS sample against a same-tier GaitRec sample (as Phase 3's
training did) is fine for a self-supervised contrastive loss, but doing that
at classification time would leak the very label being predicted into the
pairing choice.

Resolution: fuse every PADS sample against ONE FIXED, tier-agnostic
"population gait prototype" — the mean-pooled GaitRec window embedding
across the TRAIN split only (not label-conditioned) — so no PADS sample's
predicted class can influence which gait representation it's fused with.
The mirror-image secondary check for GaitRec does the same with a fixed
finger prototype from PADS's train split.

Both Phase 2 encoders and the Phase 3 Cross-Body module are loaded and kept
FROZEN — only the two new classifier heads are trained here, protecting the
already-validated upstream components (same rationale as Phase 3 freezing
Phase 2's encoders).

FiLM personalization (Track B in the plan) is explicitly DEFERRED, not
silently dropped — see the printed note at the end of this script's output
and CBDL_PROJECT_DOCUMENTATION.md §10.

Usage:
    python train_phase4_clinical_head.py --data-root "." --epochs 40
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from cross_body_module import LaggedCrossAttention, NaiveConcatFusion, gaitrec_label_to_tier, pads_label_to_tier


def build_cbdm(variant: str, device):
    if variant == "naive_concat":
        return NaiveConcatFusion(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
    return LaggedCrossAttention(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
from model import CBDLPhase2Model

K_WINDOWS = 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pads(root: Path):
    w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    m = np.load(root / "pads_preprocessed" / "pads_masks.npy")
    labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")
    return w, m, labels["label"].to_numpy(), labels["split"].to_numpy()


def load_gaitrec(root: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    classes = sorted(labels["ClassLabel"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = labels["ClassLabel"].map(class_to_idx).to_numpy()
    return w, y, labels["Split"].to_numpy(), class_to_idx


def majority_baseline(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run(args) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    root = Path(args.data_root)
    pads_w, pads_m, pads_y, pads_split = load_pads(root)
    gait_w, gait_y, gait_split, gait_class_to_idx = load_gaitrec(root)
    n_gait_classes = len(gait_class_to_idx)

    model = CBDLPhase2Model().to(device)
    model.load_state_dict(torch.load(root / "phase2_checkpoint.pt", map_location=device)["model"])
    phase3_path = root / args.phase3_checkpoint
    cbdm = build_cbdm(args.variant, device)
    cbdm.load_state_dict(torch.load(phase3_path, map_location=device)["cbdm"])
    print(f"Loaded Cross-Body module variant='{args.variant}' from {phase3_path}")
    model.eval()
    cbdm.eval()
    for p in list(model.parameters()) + list(cbdm.parameters()):
        p.requires_grad = False
    print("Loaded and froze Phase 2 encoders + Phase 3 Cross-Body module.")

    def pads_windows(idx: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(pads_w[idx]).float().to(device)
        m = torch.from_numpy(pads_m[idx]).bool().to(device)
        with torch.no_grad():
            return model.encode_finger_windows(x, m, source="pads", k=K_WINDOWS)

    def gait_windows(idx: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(gait_w[idx]).float().to(device)
        with torch.no_grad():
            return model.encode_gait_windows(x, k=K_WINDOWS)

    # ── Fixed, tier-agnostic prototypes computed from TRAIN data only ──
    train_gait_idx = np.where(gait_split == "train")[0]
    rng = np.random.RandomState(args.seed)
    proto_sample_idx = rng.choice(train_gait_idx, size=min(2000, len(train_gait_idx)), replace=False)
    with torch.no_grad():
        gait_prototype = gait_windows(proto_sample_idx).mean(dim=0, keepdim=True)  # [1, K, D]

    train_pads_idx = np.where(pads_split == "train")[0]
    with torch.no_grad():
        finger_prototype = pads_windows(train_pads_idx).mean(dim=0, keepdim=True)  # [1, K, D]

    print(f"Gait prototype computed from {len(proto_sample_idx)} train GaitRec trials.")
    print(f"Finger prototype computed from {len(train_pads_idx)} train PADS subjects.")

    pads_head = ClassifierHead(128, 64, 3).to(device)
    gait_head = ClassifierHead(128, 64, n_gait_classes).to(device)
    optimizer = torch.optim.Adam(list(pads_head.parameters()) + list(gait_head.parameters()), lr=args.lr)

    def pads_class_weights():
        counts = np.bincount(pads_y[pads_split == "train"], minlength=3).astype(np.float64)
        w = counts.sum() / (3 * counts)
        return torch.tensor(w, dtype=torch.float32, device=device)

    def gait_class_weights():
        counts = np.bincount(gait_y[gait_split == "train"], minlength=n_gait_classes).astype(np.float64)
        w = counts.sum() / (n_gait_classes * counts)
        return torch.tensor(w, dtype=torch.float32, device=device)

    ce_pads = nn.CrossEntropyLoss(weight=pads_class_weights())
    ce_gait = nn.CrossEntropyLoss(weight=gait_class_weights())

    def fused_for_pads(idx: np.ndarray) -> torch.Tensor:
        fw = pads_windows(idx)
        gp = gait_prototype.expand(fw.size(0), -1, -1)
        with torch.no_grad():
            fused, _ = cbdm(fw, gp)
        return fused

    def fused_for_gait(idx: np.ndarray) -> torch.Tensor:
        gw = gait_windows(idx)
        fp = finger_prototype.expand(gw.size(0), -1, -1)
        with torch.no_grad():
            fused, _ = cbdm(fp, gw)
        return fused

    def run_epoch(split: str, train: bool):
        pads_head.train(train)
        gait_head.train(train)

        p_idx_pool = np.where(pads_split == split)[0]
        g_idx_pool = np.where(gait_split == split)[0]
        rng_local = np.random.RandomState(args.seed + (1 if train else 0))
        p_order = p_idx_pool.copy()
        rng_local.shuffle(p_order)
        g_order = g_idx_pool.copy()
        rng_local.shuffle(g_order)

        pads_preds, pads_targets, gait_preds, gait_targets = [], [], [], []

        # range(0, len, batch_size) — NOT len//batch_size — so the final
        # partial batch is included, not silently dropped from eval metrics.
        for start in range(0, len(p_order), args.batch_size):
            idx = p_order[start : start + args.batch_size]
            if len(idx) == 0:
                continue
            y = torch.from_numpy(pads_y[idx]).long().to(device)
            with torch.set_grad_enabled(train):
                fused = fused_for_pads(idx)
                logits = pads_head(fused)
                loss = ce_pads(logits, y)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            pads_preds.append(logits.argmax(-1).detach().cpu().numpy())
            pads_targets.append(y.cpu().numpy())

        for start in range(0, len(g_order), args.batch_size_gait):
            idx = g_order[start : start + args.batch_size_gait]
            if len(idx) == 0:
                continue
            y = torch.from_numpy(gait_y[idx]).long().to(device)
            with torch.set_grad_enabled(train):
                fused = fused_for_gait(idx)
                logits = gait_head(fused)
                loss = ce_gait(logits, y)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            gait_preds.append(logits.argmax(-1).detach().cpu().numpy())
            gait_targets.append(y.cpu().numpy())

        pads_preds = np.concatenate(pads_preds) if pads_preds else np.array([])
        pads_targets = np.concatenate(pads_targets) if pads_targets else np.array([])
        gait_preds = np.concatenate(gait_preds) if gait_preds else np.array([])
        gait_targets = np.concatenate(gait_targets) if gait_targets else np.array([])

        pads_acc = (pads_preds == pads_targets).mean() if len(pads_targets) else float("nan")
        gait_acc = (gait_preds == gait_targets).mean() if len(gait_targets) else float("nan")
        return pads_acc, gait_acc, (pads_preds, pads_targets), (gait_preds, gait_targets)

    for epoch in range(1, args.epochs + 1):
        train_pads_acc, train_gait_acc, _, _ = run_epoch("train", train=True)
        val_pads_acc, val_gait_acc, _, _ = run_epoch("val", train=False)
        print(f"Epoch {epoch:2d} | PADS clinical head {train_pads_acc:.3f} -> val {val_pads_acc:.3f} | "
              f"GaitRec head (secondary) {train_gait_acc:.3f} -> val {val_gait_acc:.3f}")

    test_pads_acc, test_gait_acc, (pp, pt), (gp, gt) = run_epoch("test", train=False)
    pads_f1 = f1_score(pt, pp, average="macro") if len(pt) else float("nan")
    gait_f1 = f1_score(gt, gp, average="macro") if len(gt) else float("nan")

    print("\n" + "=" * 70)
    print("PHASE 4 CLINICAL GROUNDING — RESULTS (test split)")
    print("=" * 70)
    print(f"PADS clinical head  (PRIMARY)  : acc={test_pads_acc:.3f}  macro-F1={pads_f1:.3f}  "
          f"(majority baseline {majority_baseline(pads_y[pads_split=='test']):.3f})")
    print(f"GaitRec head (secondary check) : acc={test_gait_acc:.3f}  macro-F1={gait_f1:.3f}  "
          f"(majority baseline {majority_baseline(gait_y[gait_split=='test']):.3f})")
    print("=" * 70)
    print(
        "\nNote on methodology: both heads classify from the Cross-Body fused embedding, "
        "produced by fusing each real sample against a FIXED, tier-agnostic population "
        "prototype of the other modality (mean-pooled train-split window embeddings) — "
        "never a label-selected pairing — so no test-time label leakage occurs through "
        "the fusion step. See module docstring."
    )
    print(
        "\nFiLM personalization (Track B) — DEFERRED, not implemented, due to time "
        "constraints, consistent with the development plan's own instruction to note "
        "this explicitly rather than drop it silently."
    )

    out_name = "phase4_checkpoint.pt" if args.run_label == "attention" else f"phase4_checkpoint_{args.run_label}.pt"
    torch.save({"pads_head": pads_head.state_dict(), "gait_head": gait_head.state_dict(),
                "gait_prototype": gait_prototype.cpu(), "finger_prototype": finger_prototype.cpu(),
                "variant": args.variant, "test_pads_acc": test_pads_acc, "test_pads_f1": pads_f1},
               root / out_name)
    print(f"\nSaved checkpoint to {root / out_name}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batch-size-gait", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", choices=["attention", "naive_concat"], default="attention", help="Phase 7 ablation: which Cross-Body fusion module to load.")
    p.add_argument("--phase3-checkpoint", type=str, default="phase3_checkpoint.pt", help="Which Phase 3 checkpoint file to load (must match --variant).")
    p.add_argument("--run-label", type=str, default=None, help="Checkpoint filename suffix (default: same as --variant).")
    args = p.parse_args()
    if args.run_label is None:
        args.run_label = args.variant
    return args


if __name__ == "__main__":
    run(parse_args())
