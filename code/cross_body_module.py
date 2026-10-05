"""
CBDL Phase 3 — Cross-Body Dependency Module (the core novelty).

Per CBDL_Development_Plan.md Phase 3:
  1. Lagged Cross-Attention: attention between finger-window embeddings at
     window t and gait-window embeddings at window t+lag, for a small set
     of candidate lags. Combined across lags via a learned, softmax-weighted
     mixture — the mixture weights ARE the "which lag best explains
     cross-body relationships across the cohort" signal, aggregated into
     the attention-lag heatmap deliverable.
  2. Soft-DTW alignment (pysdtw — an existing, tested differentiable
     implementation, not written from scratch): a loss term that rewards
     tier-matched finger/gait window sequences for being alignable under
     flexible time-warping, tolerating tempo differences between the two
     modalities.
  3. Contrastive Lag Learning: population-level InfoNCE. Positive pairs =
     one finger sequence and one gait sequence from the SAME diagnosis
     tier; negatives = different-tier pairs, all drawn from the same batch.

Population-level pairing, not per-subject (per Phase 1.6 / Phase 3.1 of the
plan): GaitRec, Tappy, PADS, and mPower share no subject IDs. A "pair" here
is (one PADS finger sample, one GaitRec gait sample) matched only by a
coarse, common DIAGNOSIS TIER — not the same person. See `to_tier()` below
for the two source-specific label->tier mappings used to make this
cross-dataset matching possible at all.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Cross-dataset tier mapping ─────────────────────────────────────────────
# PADS's 3-class diagnosis and GaitRec's 5-class pathology-location label use
# different taxonomies with no shared meaning at fine granularity. The common
# ground both datasets actually share is binary: healthy vs. not.
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
