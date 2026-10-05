"""
CBDL Phase 6 (Track B) — Explainability.

Per CBDL_Development_Plan.md Phase 6: SHAP attribution on the classifier
head's inputs, plus attention visualization (explicitly reusing Phase 3's
lag-attention heatmap — "don't build a second explainability artifact",
already rendered as figures/attention_lag_heatmap.png).

This computes SHAP attributions on the PADS clinical head's RAW INPUT —
the 6-channel, 2928-timestep wrist-IMU waveform — by wrapping the entire
frozen pipeline (adapter -> shared trunk -> windowing -> Cross-Body fusion
against the fixed gait prototype -> classifier head) as one differentiable
function and running shap.GradientExplainer end-to-end through it. This is
more informative than attributing the abstract 128-dim fused embedding
(which has no interpretable meaning on its own) back to something a
clinician-facing figure can actually show: which of PointFinger / TouchIndex
/ TouchNose (the three PADS tasks), and which of the 6 IMU channels, drove
the prediction.

Usage:
    python explain_shap.py --data-root "."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn

from cross_body_module import LaggedCrossAttention
from model import CBDLPhase2Model
from train_phase4_clinical_head import ClassifierHead, get_device, load_pads, K_WINDOWS

CHANNEL_NAMES = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]
TASK_BOUNDARIES = [("PointFinger", 0, 976), ("TouchIndex", 976, 1952), ("TouchNose", 1952, 2928)]


class PADSClinicalPipeline(nn.Module):
    """Wraps the whole frozen Phase 2+3+4 PADS pathway as one differentiable
    function of the raw waveform, for SHAP to attribute through end-to-end."""

    def __init__(self, model: CBDLPhase2Model, cbdm: LaggedCrossAttention, pads_head: ClassifierHead, gait_prototype: torch.Tensor):
        super().__init__()
        self.model = model
        self.cbdm = cbdm
        self.pads_head = pads_head
        self.register_buffer("gait_prototype", gait_prototype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 6, 2928], no padding for PADS (mask = all real)
        mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
        fw = self.model.encode_finger_windows(x, mask, source="pads", k=K_WINDOWS)
        gp = self.gait_prototype.expand(fw.size(0), -1, -1)
        fused, _ = self.cbdm(fw, gp)
        return self.pads_head(fused)


def run(args) -> None:
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
    gait_prototype = phase4_ckpt["gait_prototype"].to(device)

    pipeline = PADSClinicalPipeline(model, cbdm, pads_head, gait_prototype).to(device)
    pipeline.eval()
    for p in pipeline.parameters():
        p.requires_grad = False
    print("Loaded and froze the full Phase 2+3+4 PADS pipeline for SHAP attribution.")

    train_idx = np.where(pads_split == "train")[0]
    test_idx = np.where(pads_split == "test")[0]
    rng = np.random.RandomState(42)
    background_idx = rng.choice(train_idx, size=min(args.n_background, len(train_idx)), replace=False)

    background = torch.from_numpy(pads_w[background_idx]).float().to(device)
    test_x = torch.from_numpy(pads_w[test_idx]).float().to(device)
    test_y = pads_y[test_idx]

    print(f"Running GradientExplainer: {len(background)} background samples, explaining {len(test_x)} test samples...")
    explainer = shap.GradientExplainer(pipeline, background)
    shap_values = explainer.shap_values(test_x, nsamples=args.nsamples)
    # shap_values: [B, 6, 2928, n_classes] (or list of [B,6,2928] per class, depending on shap version)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    print(f"SHAP values shape: {shap_values.shape}")

    # Attribute to the TRUE predicted class per sample (what actually drove each sample's own prediction)
    with torch.no_grad():
        preds = pipeline(test_x).argmax(-1).cpu().numpy()
    per_sample_shap = np.stack([shap_values[i, :, :, preds[i]] for i in range(len(preds))], axis=0)  # [B, 6, 2928]
    abs_shap = np.abs(per_sample_shap)

    # ── Channel importance (summed |SHAP| over all timesteps and samples) ──
    channel_importance = abs_shap.sum(axis=(0, 2))
    channel_importance = channel_importance / channel_importance.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CHANNEL_NAMES, channel_importance, color="#a0523b")
    for b, v in zip(bars, channel_importance):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Share of total |SHAP| attribution")
    ax.set_title("SHAP Channel Importance — PADS Clinical Head\n(attributed to each sample's own predicted class)", fontsize=10)
    plt.xticks(rotation=20)
    fig.tight_layout()
    (root / "figures").mkdir(exist_ok=True)
    fig.savefig(root / "figures" / "shap_channel_importance.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/shap_channel_importance.png")

    # ── Task importance (PointFinger / TouchIndex / TouchNose) ──
    task_importance = []
    for name, lo, hi in TASK_BOUNDARIES:
        task_importance.append(abs_shap[:, :, lo:hi].sum())
    task_importance = np.array(task_importance)
    task_importance = task_importance / task_importance.sum()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar([t[0] for t in TASK_BOUNDARIES], task_importance, color="#3b8f6e")
    for b, v in zip(bars, task_importance):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Share of total |SHAP| attribution")
    ax.set_title("SHAP Task Importance — PADS Clinical Head\n(which finger/hand task drove predictions most)", fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "figures" / "shap_task_importance.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/shap_task_importance.png")

    print("\n" + "=" * 70)
    print("PHASE 6 SHAP EXPLAINABILITY — SUMMARY")
    print("=" * 70)
    for name, val in zip(CHANNEL_NAMES, channel_importance):
        print(f"  {name:10s}: {val:.3f}")
    print("-" * 40)
    for (name, _, _), val in zip(TASK_BOUNDARIES, task_importance):
        print(f"  {name:12s}: {val:.3f}")
    print("=" * 70)
    print(
        "\nAttention visualization: reusing figures/attention_lag_heatmap.png from Phase 3 "
        "(per the plan's instruction not to build a second explainability artifact)."
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--n-background", type=int, default=40)
    p.add_argument("--nsamples", type=int, default=50, help="GradientExplainer Monte Carlo samples per explained input")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
