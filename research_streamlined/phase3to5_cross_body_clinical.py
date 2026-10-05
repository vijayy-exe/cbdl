"""
CBDL Phase 3-5 — Cross-Body Dependency Module, Clinical Grounding, and
Calibration, plus the Phase 7 gait-only baseline (consolidated
research-release version).

Merges five originally-separate files from ../code/:
  - cross_body_module.py         architecture: LaggedCrossAttention (the
                                  core novelty), NaiveConcatFusion (Phase 7
                                  ablation), tier-mapping helpers, InfoNCE —
                                  unchanged.
  - train_phase3_cbdm.py         Phase 3 training (Lagged Cross-Attention +
                                  Soft-DTW + Contrastive Lag Learning on
                                  frozen Phase 2 encoders) — unchanged logic;
                                  renamed run -> run_phase3, and its
                                  set_seed/get_device/build_cbdm/K_WINDOWS
                                  were IDENTICAL to Phase 4's copies of the
                                  same, so those duplicates were dropped here
                                  in favor of the single copy kept in the
                                  Phase 4 section below (Python resolves
                                  these at call time, so section order
                                  doesn't matter). Its load_pads/load_gaitrec
                                  return TIER-mapped labels, which is NOT the
                                  same as Phase 4's same-named functions (raw
                                  class labels) — renamed to
                                  load_pads_tiers_phase3 /
                                  load_gaitrec_tiers_phase3 to keep both
                                  semantics distinct and unambiguous.
  - train_phase4_clinical_head.py Phase 4 training (classifier heads on the
                                  Cross-Body fused embedding, frozen
                                  Phase2+3) — unchanged, renamed
                                  run -> run_phase4. This section is the
                                  canonical home of set_seed / get_device /
                                  build_cbdm / K_WINDOWS / ClassifierHead /
                                  majority_baseline / load_pads / load_gaitrec,
                                  matching how the original scripts already
                                  imported these FROM this file.
  - train_phase5_calibration.py   Phase 5 isotonic-regression calibration of
                                  the Phase 4 PADS head — unchanged, renamed
                                  run -> run_phase5.
  - baseline_gait_only.py         Phase 7 gait-only baseline (evaluates the
                                  already-trained Phase 4 checkpoint against
                                  a constant population-prototype input —
                                  does not retrain Phase 2-4) — unchanged,
                                  renamed run -> run_baseline_gait.

See ../code/ for the original, unmerged scripts and
../CBDL_PROJECT_DOCUMENTATION.md §3.4/§3.5, §5.4/§5.5/§5.6 for the writeups.

Usage:
    python phase3to5_cross_body_clinical.py phase3   --data-root "." --epochs 30
    python phase3to5_cross_body_clinical.py phase4   --data-root "." --epochs 40
    python phase3to5_cross_body_clinical.py phase5   --data-root "."
    python phase3to5_cross_body_clinical.py baseline-gait --data-root "."
Run `python phase3to5_cross_body_clinical.py <task> --help` for that task's full flag list.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysdtw
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

from phase2_encoders import CBDLPhase2Model


# ============================================================================
# CROSS-BODY MODULE ARCHITECTURE (from cross_body_module.py)
# ============================================================================

TIER_HEALTHY, TIER_PATHOLOGICAL = 0, 1


def pads_label_to_tier(label: int) -> int:
    """PADS: 0=Healthy, 1=Parkinson's, 2=Other Movement Disorder."""
    return TIER_HEALTHY if label == 0 else TIER_PATHOLOGICAL


def gaitrec_label_to_tier(class_label: str) -> int:
    """GaitRec: HC=healthy control; A/C/H/K are ankle/calcaneus/hip/knee pathology groups."""
    return TIER_HEALTHY if class_label == "HC" else TIER_PATHOLOGICAL


# ─── Lagged Cross-Attention ─────────────────────────────────────────────────


class LaggedCrossAttention(nn.Module):
    """
    finger_windows: [B, K, D]   gait_windows: [B, K, D]  (same B — population-
    level pairs constructed by the training loop, not per-subject; same K —
    both encoders window-pool to the same K, see model.py).

    For each candidate lag d in `lags`, attends finger window t against gait
    window t+d (dropping the last d gait windows / first d finger windows so
    shapes match), producing a lag-specific fused representation. A learned
    scorer combines all lags via softmax into one fused embedding, and the
    per-lag softmax weights are returned for the attention-lag heatmap.
    """

    def __init__(self, embed_dim: int, lags: list[int] = (0, 1, 2, 3), n_heads: int = 4):
        super().__init__()
        self.lags = list(lags)
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.lag_scorer = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Learnable projection used ONLY for the Soft-DTW pathway (see module
        # docstring / train_phase3_cbdm.py note): the Finger/Gait encoders are
        # frozen (already validated in Phase 2), so Soft-DTW's alignment cost
        # on raw encoder outputs would be a fixed number with no gradient path
        # to anything trainable. This projection gives Soft-DTW something to
        # actually optimize — a learned re-embedding that Soft-DTW can push
        # toward making tier-matched finger/gait sequences more alignable.
        self.align_proj = nn.Linear(embed_dim, embed_dim)

    def project_for_alignment(self, x: torch.Tensor) -> torch.Tensor:
        return self.align_proj(x)

    def forward(self, finger_windows: torch.Tensor, gait_windows: torch.Tensor):
        """Returns (fused_embedding [B, D], lag_weights [B, n_lags])."""
        b, k, d = finger_windows.shape
        lag_reprs = []
        lag_logits = []

        for lag in self.lags:
            if lag == 0:
                f, g = finger_windows, gait_windows
            else:
                if lag >= k:
                    # Degenerate lag for this K — fall back to lag 0's slice so shapes stay valid.
                    f, g = finger_windows, gait_windows
                else:
                    f = finger_windows[:, : k - lag, :]
                    g = gait_windows[:, lag:, :]

            attended, _ = self.attn(query=f, key=g, value=g)   # [B, K-lag, D]
            combined = (f + attended) / 2.0
            pooled = combined.mean(dim=1)                        # [B, D]
            lag_reprs.append(pooled)
            lag_logits.append(self.lag_scorer(pooled))           # [B, 1]

        lag_logits = torch.cat(lag_logits, dim=1)                # [B, n_lags]
        lag_weights = F.softmax(lag_logits, dim=1)               # [B, n_lags]
        stacked = torch.stack(lag_reprs, dim=1)                  # [B, n_lags, D]
        fused = (lag_weights.unsqueeze(-1) * stacked).sum(dim=1) # [B, D]
        return self.out_proj(fused), lag_weights


class NaiveConcatFusion(nn.Module):
    """
    Phase 7 baseline — "naive concatenation fusion" (no lagged attention):
    mean-pools each side's windows to a single vector and projects their
    concatenation. Same public interface as LaggedCrossAttention (forward
    returns (fused, lag_weights) and exposes project_for_alignment) so the
    Phase 3/4 training scripts can swap between the two with one flag,
    isolating exactly what the attention mechanism itself contributes.
    `lag_weights` here is always uniform (1/n_lags) — there IS no lag
    mechanism in this variant; it's returned only for interface parity.
    """

    def __init__(self, embed_dim: int, lags: list[int] = (0, 1, 2, 3)):
        super().__init__()
        self.n_lags = len(lags)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.align_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, finger_windows: torch.Tensor, gait_windows: torch.Tensor):
        f = finger_windows.mean(dim=1)
        g = gait_windows.mean(dim=1)
        fused = self.proj(torch.cat([f, g], dim=-1))
        uniform_weights = torch.full((finger_windows.size(0), self.n_lags), 1.0 / self.n_lags, device=finger_windows.device)
        return fused, uniform_weights

    def project_for_alignment(self, x: torch.Tensor) -> torch.Tensor:
        return self.align_proj(x)


# ─── Contrastive Lag Learning (population-level InfoNCE) ───────────────────


def build_positive_indices(tier_f: np.ndarray, tier_g: np.ndarray) -> np.ndarray:
    """
    For each finger sample i, returns the index of one gait sample with the
    SAME tier (its positive). Returns -1 for finger samples with no
    same-tier gait sample anywhere in the batch (excluded from the loss).
    """
    pos_idx = np.full(len(tier_f), -1, dtype=np.int64)
    for t in np.unique(tier_f):
        f_idx = np.where(tier_f == t)[0]
        g_idx = np.where(tier_g == t)[0]
        if len(g_idx) == 0:
            continue
        chosen = np.random.choice(g_idx, size=len(f_idx))
        pos_idx[f_idx] = chosen
    return pos_idx


def info_nce_loss(finger_embed: torch.Tensor, gait_embed: torch.Tensor, positive_idx: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    finger_embed, gait_embed: [B, D] (L2-normalized inside).
    positive_idx: [B] long, index into gait_embed's batch dim (-1 = skip).
    In-batch negatives: every other gait sample (regardless of tier) serves
    as a candidate negative in the softmax denominator — standard InfoNCE.
    """
    valid = positive_idx >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=finger_embed.device, requires_grad=True)

    f = F.normalize(finger_embed[valid], dim=-1)
    g = F.normalize(gait_embed, dim=-1)
    sim = f @ g.T / temperature  # [B_valid, B]
    targets = positive_idx[valid]
    return F.cross_entropy(sim, targets)


# ─── Cohort-level attention-lag heatmap ─────────────────────────────────────


def summarize_lag_weights(all_lag_weights: list[np.ndarray], lags: list[int]) -> dict:
    """Aggregates per-batch lag_weights [B, n_lags] arrays into a cohort-level mean per lag."""
    stacked = np.concatenate(all_lag_weights, axis=0)  # [N, n_lags]
    means = stacked.mean(axis=0)
    return {f"lag_{d}": float(m) for d, m in zip(lags, means)}

# ============================================================================
# PHASE 3 — TRAINING (from train_phase3_cbdm.py)
# ============================================================================

LAGS = [0, 1, 2, 3]


def load_pads_tiers_phase3(root: Path):
    w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    m = np.load(root / "pads_preprocessed" / "pads_masks.npy")
    labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")
    tier = labels["label"].map(pads_label_to_tier).to_numpy()
    return w, m, tier, labels["split"].to_numpy()


def load_gaitrec_tiers_phase3(root: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    tier = labels["ClassLabel"].map(gaitrec_label_to_tier).to_numpy()
    return w, tier, labels["Split"].to_numpy()


def run_phase3(args) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    root = Path(args.data_root)
    pads_w, pads_m, pads_tier, pads_split = load_pads_tiers_phase3(root)
    gait_w, gait_tier, gait_split = load_gaitrec_tiers_phase3(root)

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


def parse_args_phase3():
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

# ============================================================================
# PHASE 4 — CLINICAL GROUNDING (from train_phase4_clinical_head.py)
# ============================================================================

def build_cbdm(variant: str, device):
    if variant == "naive_concat":
        return NaiveConcatFusion(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
    return LaggedCrossAttention(embed_dim=128, lags=[0, 1, 2, 3]).to(device)

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


def run_phase4(args) -> None:
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


def parse_args_phase4():
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

# ============================================================================
# PHASE 5 — CALIBRATION (from train_phase5_calibration.py)
# ============================================================================

def get_pads_probs_and_labels(model, cbdm, pads_head, gait_prototype, pads_w, pads_m, pads_y, pads_split, split, device):
    idx = np.where(pads_split == split)[0]
    x = torch.from_numpy(pads_w[idx]).float().to(device)
    m = torch.from_numpy(pads_m[idx]).bool().to(device)
    with torch.no_grad():
        fw = model.encode_finger_windows(x, m, source="pads", k=K_WINDOWS)
        gp = gait_prototype.to(device).expand(fw.size(0), -1, -1)
        fused, _ = cbdm(fw, gp)
        logits = pads_head(fused)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs, pads_y[idx]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Standard multi-class ECE: bins by the model's top predicted-class confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "acc": None, "conf": None})
            continue
        acc = correct[in_bin].mean()
        conf = confidences[in_bin].mean()
        ece += (n_in_bin / len(confidences)) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "n": int(n_in_bin), "acc": float(acc), "conf": float(conf)})
    return float(ece), bins


def fit_isotonic_calibrators(val_probs: np.ndarray, val_labels: np.ndarray, n_classes: int) -> list[IsotonicRegression]:
    """One-vs-rest isotonic regression per class, fit on VALIDATION only."""
    calibrators = []
    for c in range(n_classes):
        targets = (val_labels == c).astype(np.float64)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(val_probs[:, c], targets)
        calibrators.append(ir)
    return calibrators


def apply_calibrators(calibrators: list[IsotonicRegression], probs: np.ndarray) -> np.ndarray:
    calibrated = np.stack([ir.predict(probs[:, c]) for c, ir in enumerate(calibrators)], axis=1)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return calibrated / row_sums  # renormalize to sum to 1, standard post-isotonic step


def run_phase5(args) -> None:
    device = get_device()
    root = Path(args.data_root)

    pads_w, pads_m, pads_y, pads_split = load_pads(root)

    model = CBDLPhase2Model().to(device)
    model.load_state_dict(torch.load(root / "phase2_checkpoint.pt", map_location=device)["model"])
    cbdm = LaggedCrossAttention(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
    cbdm.load_state_dict(torch.load(root / "phase3_checkpoint.pt", map_location=device)["cbdm"])
    phase4_ckpt = torch.load(root / "phase4_checkpoint.pt", map_location=device)
    pads_head = ClassifierHead(128, 64, 3).to(device)
    pads_head.load_state_dict(phase4_ckpt["pads_head"])
    gait_prototype = phase4_ckpt["gait_prototype"]

    model.eval(); cbdm.eval(); pads_head.eval()
    print("Loaded Phase 2/3/4 checkpoints (frozen).")

    val_probs, val_labels = get_pads_probs_and_labels(model, cbdm, pads_head, gait_prototype, pads_w, pads_m, pads_y, pads_split, "val", device)
    test_probs, test_labels = get_pads_probs_and_labels(model, cbdm, pads_head, gait_prototype, pads_w, pads_m, pads_y, pads_split, "test", device)

    ece_before, bins_before = expected_calibration_error(test_probs, test_labels, n_bins=args.n_bins)
    print(f"ECE before calibration (test): {ece_before:.4f}")

    calibrators = fit_isotonic_calibrators(val_probs, val_labels, n_classes=3)
    test_probs_calibrated = apply_calibrators(calibrators, test_probs)
    ece_after, bins_after = expected_calibration_error(test_probs_calibrated, test_labels, n_bins=args.n_bins)
    print(f"ECE after calibration (test):  {ece_after:.4f}")

    acc_before = (test_probs.argmax(1) == test_labels).mean()
    acc_after = (test_probs_calibrated.argmax(1) == test_labels).mean()
    print(f"Accuracy before/after calibration: {acc_before:.3f} / {acc_after:.3f}. Note: per-class "
          f"isotonic regression is fit independently per class, so it is NOT guaranteed to preserve "
          f"argmax ranking — accuracy can shift as a side effect, especially with a validation set "
          f"this small (70 samples / 3 classes). This is a real limitation of calibrating on so little "
          f"data, not a bug — see documentation.")

    with open(root / "phase5_calibration.pkl", "wb") as f:
        pickle.dump({
            "calibrators": calibrators,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "bins_before": bins_before,
            "bins_after": bins_after,
            "test_probs_raw": test_probs,
            "test_probs_calibrated": test_probs_calibrated,
            "test_labels": test_labels,
        }, f)
    print(f"\nSaved calibration model + reliability data to {root / 'phase5_calibration.pkl'}")

    print("\n" + "=" * 70)
    print("PHASE 5 CALIBRATION — SUMMARY")
    print("=" * 70)
    print(f"PADS clinical head Expected Calibration Error: {ece_before:.4f} -> {ece_after:.4f} "
          f"({'improved' if ece_after < ece_before else 'no improvement'})")
    print("Method: per-class (one-vs-rest) Isotonic Regression, fit on VALIDATION split only, "
          "applied to TEST split. Bootstrap confidence intervals skipped per the development plan.")
    print("=" * 70)


def parse_args_phase5():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--n-bins", type=int, default=10)
    return p.parse_args()

# ============================================================================
# PHASE 7 — GAIT-ONLY BASELINE (from baseline_gait_only.py)
# ============================================================================

def run_baseline_gait(args) -> None:
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


def parse_args_baseline_gait():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=40)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────
# Unified CLI — dispatches to whichever Phase 3/4/5/baseline task was
# requested. Each keeps its own parse_args_<task>() with its own flags,
# exactly as in the original standalone scripts.
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    tasks = {
        "phase3": (run_phase3, parse_args_phase3),
        "phase4": (run_phase4, parse_args_phase4),
        "phase5": (run_phase5, parse_args_phase5),
        "baseline-gait": (run_baseline_gait, parse_args_baseline_gait),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in tasks:
        print(f"usage: {sys.argv[0]} {{{','.join(tasks)}}} [task-specific args...]", file=sys.stderr)
        print(f"Run '{sys.argv[0]} <task> --help' to see that task's arguments.", file=sys.stderr)
        sys.exit(1)
    task = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    run_fn, parse_fn = tasks[task]
    run_fn(parse_fn())


if __name__ == "__main__":
    main()
