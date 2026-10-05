"""
CBDL Phase 3 — trains the Cross-Body Dependency Module on top of the
Phase 2 checkpoint's already-validated Finger/Gait encoders (loaded and
FROZEN here — Phase 3 trains only the new Lagged Cross-Attention module,
so it can't undo Phase 2's PADS-probe gate pass; a defensible, standard
transfer-learning choice given the time budget).

Population-level pairing (see cross_body_module.py docstring): each
training pair is one PADS finger sample + one GaitRec gait sample, matched
only by a coarse binary tier (Healthy vs Pathological) — never the same
subject, since none of these datasets share subjects.

Usage:
    python train_phase3_cbdm.py --data-root "." --epochs 30
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pysdtw
import torch

from cross_body_module import (
    LaggedCrossAttention,
    NaiveConcatFusion,
    build_positive_indices,
    gaitrec_label_to_tier,
    pads_label_to_tier,
    summarize_lag_weights,
)
from model import CBDLPhase2Model

K_WINDOWS = 8
LAGS = [0, 1, 2, 3]


def build_cbdm(variant: str, device):
    if variant == "naive_concat":
        return NaiveConcatFusion(embed_dim=128, lags=LAGS).to(device)
    return LaggedCrossAttention(embed_dim=128, lags=LAGS).to(device)


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
    tier = labels["label"].map(pads_label_to_tier).to_numpy()
    return w, m, tier, labels["split"].to_numpy()


def load_gaitrec(root: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    tier = labels["ClassLabel"].map(gaitrec_label_to_tier).to_numpy()
    return w, tier, labels["Split"].to_numpy()


def run(args) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    root = Path(args.data_root)
    pads_w, pads_m, pads_tier, pads_split = load_pads(root)
    gait_w, gait_tier, gait_split = load_gaitrec(root)

    print(f"PADS tiers (train): {np.bincount(pads_tier[pads_split == 'train'])} (0=Healthy, 1=Pathological)")
    print(f"GaitRec tiers (train): {np.bincount(gait_tier[gait_split == 'train'])} (0=Healthy, 1=Pathological)")

    model = CBDLPhase2Model().to(device)
    ckpt_path = root / "phase2_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded Phase 2 encoder weights from {ckpt_path}")
    else:
        print("WARNING: no Phase 2 checkpoint found — training Cross-Body module on RANDOM encoders.")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    cbdm = build_cbdm(args.variant, device)
    optimizer = torch.optim.Adam(cbdm.parameters(), lr=args.lr)
    sdtw = pysdtw.SoftDTW(gamma=1.0, dist_func=pysdtw.distance.pairwise_l2_squared, use_cuda=(device.type == "cuda"))

    def get_pads_windows(idx: np.ndarray):
        x = torch.from_numpy(pads_w[idx]).float().to(device)
        m = torch.from_numpy(pads_m[idx]).bool().to(device)
        with torch.no_grad():
            return model.encode_finger_windows(x, m, source="pads", k=K_WINDOWS)

    def get_gait_windows(idx: np.ndarray):
        x = torch.from_numpy(gait_w[idx]).float().to(device)
        with torch.no_grad():
            return model.encode_gait_windows(x, k=K_WINDOWS)

    def run_epoch(split: str, train: bool, batch_size: int):
        cbdm.train(train)
        pads_idx_pool = np.where(pads_split == split)[0]
        gait_idx_pool = np.where(gait_split == split)[0]
        rng = np.random.RandomState(args.seed + (0 if not train else 1))

        pads_order = pads_idx_pool.copy()
        rng.shuffle(pads_order)

        infonce_losses, dtw_losses, lag_weight_batches = [], [], []

        # range(0, len, batch_size) — NOT len//batch_size — so the final
        # partial batch is included, not silently dropped from eval metrics.
        for start in range(0, len(pads_order), batch_size):
            f_idx = pads_order[start : start + batch_size]
            if len(f_idx) == 0:
                continue
            g_idx = rng.choice(gait_idx_pool, size=len(f_idx), replace=True)

            finger_windows = get_pads_windows(f_idx)  # [B, K, D], frozen encoder output (no grad)
            gait_windows = get_gait_windows(g_idx)    # [B, K, D], frozen encoder output (no grad)

            tier_f = pads_tier[f_idx]
            tier_g = gait_tier[g_idx]
            pos_idx = build_positive_indices(tier_f, tier_g)
            valid = pos_idx >= 0
            if valid.sum() == 0:
                continue

            finger_v = finger_windows[valid]
            gait_matched_v = gait_windows[pos_idx[valid]]  # same-tier match, aligned 1:1 with finger_v

            with torch.set_grad_enabled(train):
                # fused DOES depend on cbdm's parameters (attn/lag_scorer/out_proj) —
                # this is what actually gets trained by the InfoNCE loss below.
                fused, lag_weights = cbdm(finger_v, gait_matched_v)

                # In-batch InfoNCE: fused[i] (finger_v[i] cross-attended against its
                # OWN tier-matched gait) should be closer to gait_matched_v[i]'s
                # pooled embedding than to any other row's — the diagonal is
                # correct by construction (see module docstring for the caveat
                # that off-diagonal rows aren't guaranteed different-tier).
                gait_pooled_matched = gait_matched_v.mean(dim=1)
                fused_n = torch.nn.functional.normalize(fused, dim=-1)
                gait_n = torch.nn.functional.normalize(gait_pooled_matched, dim=-1)
                sim = fused_n @ gait_n.T / args.temperature
                targets = torch.arange(sim.size(0), device=device)
                loss_infonce = torch.nn.functional.cross_entropy(sim, targets)

                # Soft-DTW on a learned projection (align_proj) of the frozen
                # window embeddings — gives Soft-DTW a trainable target, since
                # the raw encoder outputs are frozen (see LaggedCrossAttention
                # docstring). Computed on CPU: pysdtw's CPU backend requires it.
                finger_aligned = cbdm.project_for_alignment(finger_v)
                gait_aligned = cbdm.project_for_alignment(gait_matched_v)
                loss_dtw = sdtw(finger_aligned.cpu(), gait_aligned.cpu()).mean().to(device)

                loss = loss_infonce + args.dtw_weight * loss_dtw

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(cbdm.parameters(), 5.0)
                    optimizer.step()

            infonce_losses.append(loss_infonce.item())
            dtw_losses.append(loss_dtw.item())
            lag_weight_batches.append(lag_weights.detach().cpu().numpy())

        return (
            float(np.mean(infonce_losses)) if infonce_losses else float("nan"),
            float(np.mean(dtw_losses)) if dtw_losses else float("nan"),
            lag_weight_batches,
        )

    for epoch in range(1, args.epochs + 1):
        train_infonce, train_dtw, _ = run_epoch("train", train=True, batch_size=args.batch_size)
        val_infonce, val_dtw, _ = run_epoch("val", train=False, batch_size=args.batch_size)
        print(f"Epoch {epoch:2d} | InfoNCE {train_infonce:.4f} -> val {val_infonce:.4f} | "
              f"SoftDTW {train_dtw:.3f} -> val {val_dtw:.3f}")

    test_infonce, test_dtw, test_lag_weights = run_epoch("test", train=False, batch_size=args.batch_size)
    heatmap = summarize_lag_weights(test_lag_weights, LAGS)

    print("\n" + "=" * 70)
    print("PHASE 3 CROSS-BODY DEPENDENCY MODULE — RESULTS (test split)")
    print("=" * 70)
    print(f"InfoNCE loss (population-level, tier-matched): {test_infonce:.4f}")
    print(f"Soft-DTW alignment loss (positive pairs only): {test_dtw:.3f}")
    print("\nAttention-lag heatmap (mean softmax weight per lag, across the cohort):")
    for k, v in heatmap.items():
        print(f"  {k}: {v:.3f}")
    best_lag = max(heatmap, key=heatmap.get)
    print(f"\nMost-weighted lag across the cohort: {best_lag}")
    print("=" * 70)

    out_name = "phase3_checkpoint.pt" if args.run_label == "attention" else f"phase3_checkpoint_{args.run_label}.pt"
    torch.save({"cbdm": cbdm.state_dict(), "lag_heatmap": heatmap, "variant": args.variant, "run_label": args.run_label}, root / out_name)
    print(f"\nSaved checkpoint to {root / out_name}")

    print(
        "\nLIMITATION (state in paper): pairs are population-level (tier-matched "
        "PADS finger + GaitRec gait samples, never the same subject — GaitRec/PADS "
        "share no subject IDs). The lag that gets the most attention weight reflects "
        "which offset best aligns tier-consistent structure ACROSS the cohort, not a "
        "per-patient physiological delay. State this explicitly, consistent with the "
        "same framing already used for Phase 1's dataset-alignment note."
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--dtw-weight", type=float, default=0.01, help="Soft-DTW cost is on a much larger scale than InfoNCE; downweighted accordingly.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", choices=["attention", "naive_concat"], default="attention", help="Phase 7 ablation: which fusion module to train.")
    p.add_argument("--run-label", type=str, default=None, help="Checkpoint filename suffix (default: same as --variant; 'attention' maps to the default phase3_checkpoint.pt for backward compatibility).")
    args = p.parse_args()
    if args.run_label is None:
        args.run_label = args.variant
    return args


if __name__ == "__main__":
    run(parse_args())
