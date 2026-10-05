"""
CBDL Phase 6-7 — Explainability (SHAP), the Patient Report data pipeline,
and every figure/log generator, consolidated (research-release version).

Merges eight originally-separate files from ../code/:
  - explain_shap.py                    Phase 6 SHAP attribution on the PADS
                                        clinical head's raw waveform input.
  - generate_embedding_logs.py         dumps Phase 2/3 embedding vectors to
                                        CSV for inspection.
  - generate_figures.py                training-curve / confusion-matrix /
                                        reliability-diagram / lag-heatmap
                                        figures from real logs+checkpoints.
  - generate_patient_report_data.py    per-subject clinical report JSON
                                        (diagnostic probabilities, SHAP,
                                        DSP motor-pattern proxies).
  - generate_pipeline_shape_figures.py  architecture/tensor-shape diagrams.
  - generate_preprocessing_figures.py   before/after preprocessing figures
                                        for all four datasets.
  - generate_wavelet_channel_structure.py  per-channel wavelet scalogram
                                        grids (illustrative only).
  - generate_wavelet_figures.py         single-channel wavelet scalograms
                                        (illustrative only).

All unchanged in logic. Renames applied only to resolve collisions from
putting eight independent scripts in one namespace:
  - run/parse_args -> run_<task>/parse_args_<task> per task (all 8 defined
    these names).
  - explain_shap.py and generate_patient_report_data.py each defined a
    PADSClinicalPipeline class with the SAME name but a DIFFERENT forward()
    (logits-only vs. (logits, lag_weights)) -> renamed
    PADSClinicalPipelineLogitsOnly / PADSClinicalPipelineWithLagWeights so
    neither silently shadows the other.
  - fig_gaitrec/fig_pads/fig_tappy/fig_mpower were each defined in 2-3 of
    the figure-generator files for genuinely different plots (preprocessing
    before/after vs. wavelet channel grids vs. wavelet scalograms) ->
    suffixed per origin (e.g. fig_gaitrec_preprocessing /
    fig_gaitrec_wavelet_channels / fig_gaitrec_wavelet).
  - Color constants (BLUE/ORANGE/GRAY/RED/...) were defined with the SAME
    names but DIFFERENT hex values in generate_pipeline_shape_figures.py,
    generate_preprocessing_figures.py, and generate_wavelet_figures.py ->
    prefixed per file (PSF_/PPF_/WVF_) so one file's palette can never
    silently leak into another's plots.
  - generate_embedding_logs.py's own get_device()/K_WINDOWS were IDENTICAL
    to Phase 3-5's canonical copies -> dropped here, imported instead.
  - generate_wavelet_channel_structure.py and generate_wavelet_figures.py
    both defined WAVELET = "cmor1.5-1.0" (identical) -> kept once.
  - The GAITREC_*/TAPPY_*/MPOWER_* normalization-stat constants and
    CHANNEL_NAMES/TASK_BOUNDARIES were previously cross-file imports
    (`from generate_preprocessing_figures import ...`,
    `from explain_shap import ...`-equivalent pattern) -> now plain
    same-file references (the import lines were dropped); a couple of tiny
    identical constant lists were simply left duplicated once each (noted
    inline) rather than threaded across sections, since that is clearer for
    independently-runnable CLI tasks and changes no behavior.

See ../code/ for the original, unmerged scripts and
../CBDL_PROJECT_DOCUMENTATION.md §3.6/§5.6, §9, §11 for the writeups.

Usage:
    python phase6to7_explainability_reporting.py shap                 --data-root "."
    python phase6to7_explainability_reporting.py embedding-logs       --data-root "."
    python phase6to7_explainability_reporting.py figures              --data-root "."
    python phase6to7_explainability_reporting.py patient-report       --subject-id 037 --data-root "."
    python phase6to7_explainability_reporting.py pipeline-shapes      --data-root "."
    python phase6to7_explainability_reporting.py preprocessing-figs   --data-root "."
    python phase6to7_explainability_reporting.py wavelet-channels     --data-root "."
    python phase6to7_explainability_reporting.py wavelet-figs         --data-root "."
Run `python phase6to7_explainability_reporting.py <task> --help` for that task's full flag list.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import pywt
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9

from phase2_encoders import CBDLPhase2Model
from phase3to5_cross_body_clinical import (
    LaggedCrossAttention,
    ClassifierHead,
    get_device,
    load_pads,
    load_gaitrec,
    K_WINDOWS,
    apply_calibrators,
    build_positive_indices,
    gaitrec_label_to_tier,
    pads_label_to_tier,
)


# ============================================================================
# SHAP EXPLAINABILITY (from explain_shap.py)
# ============================================================================

CHANNEL_NAMES = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]
TASK_BOUNDARIES = [("PointFinger", 0, 976), ("TouchIndex", 976, 1952), ("TouchNose", 1952, 2928)]


class PADSClinicalPipelineLogitsOnly(nn.Module):
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


def run_shap(args) -> None:
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

    pipeline = PADSClinicalPipelineLogitsOnly(model, cbdm, pads_head, gait_prototype).to(device)
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


def parse_args_shap():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--n-background", type=int, default=40)
    p.add_argument("--nsamples", type=int, default=50, help="GradientExplainer Monte Carlo samples per explained input")
    return p.parse_args()

# ============================================================================
# EMBEDDING LOGS (from generate_embedding_logs.py)
# ============================================================================

BATCH_SIZE = 512


def emb_columns(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(dim)]


def write_csv(path: Path, meta: pd.DataFrame, log_lines: list[str], **named_embeddings: np.ndarray) -> None:
    frames = [meta.reset_index(drop=True)]
    for prefix, arr in named_embeddings.items():
        cols = emb_columns(prefix, arr.shape[1])
        frames.append(pd.DataFrame(arr, columns=cols))
    out = pd.concat(frames, axis=1)
    out.to_csv(path, index=False)
    log_lines.append(f"  wrote {path.name}: {out.shape[0]} rows x {out.shape[1]} cols ({path.stat().st_size / 1e6:.2f} MB)")
    for prefix, arr in named_embeddings.items():
        sample = np.array2string(arr[0, : min(6, arr.shape[1])], precision=5, separator=", ")
        log_lines.append(f"    {prefix}: shape={tuple(arr.shape)} row0[:6]={sample}")


@torch.no_grad()
def batched(fn, n: int, batch_size: int = BATCH_SIZE):
    outs = []
    for start in range(0, n, batch_size):
        outs.append(fn(start, min(start + batch_size, n)))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,))


def run_embedding_logs(args) -> None:
    root = Path(args.data_root)
    out_dir = root / "embedding_logs"
    out_dir.mkdir(exist_ok=True)
    device = get_device()
    log_lines = [
        f"CBDL embedding extraction log — {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Data root: {root.resolve()}",
        f"Device: {device}",
    ]

    model = CBDLPhase2Model().to(device)
    ckpt2_path = root / "phase2_checkpoint.pt"
    ckpt2 = torch.load(ckpt2_path, map_location=device)
    model.load_state_dict(ckpt2["model"])
    model.eval()
    log_lines.append(f"Loaded Phase 2 checkpoint: {ckpt2_path}")

    # ── PADS ────────────────────────────────────────────────────────────────
    pads_w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    pads_m = np.load(root / "pads_preprocessed" / "pads_masks.npy")
    pads_labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")

    def pads_waveform_batch(lo, hi):
        x = torch.from_numpy(pads_w[lo:hi]).float().to(device)
        m = torch.from_numpy(pads_m[lo:hi]).bool().to(device)
        return model.encode_finger_waveform(x, m, source="pads").cpu().numpy()

    pads_waveform_embed = batched(pads_waveform_batch, len(pads_labels))

    def pads_fused_batch(lo, hi):
        wf = torch.from_numpy(pads_waveform_embed[lo:hi]).to(device)
        return model.encode_finger_fused(wf, None, hi - lo, device).cpu().numpy()

    pads_finger_embed = batched(pads_fused_batch, len(pads_labels))

    write_csv(
        out_dir / "phase2_pads_embeddings.csv",
        pads_labels[["subject_id", "split", "label", "condition", "session_index"]],
        log_lines,
        waveform_embed=pads_waveform_embed,
        finger_embed=pads_finger_embed,
    )

    # ── Tappy ───────────────────────────────────────────────────────────────
    tappy_w = np.load(root / "tappy_preprocessed" / "tappy_waveforms.npy")
    tappy_m = np.load(root / "tappy_preprocessed" / "tappy_masks.npy")
    tappy_labels = pd.read_csv(root / "tappy_preprocessed" / "tappy_labels.csv")

    def tappy_waveform_batch(lo, hi):
        x = torch.from_numpy(tappy_w[lo:hi]).float().to(device)
        m = torch.from_numpy(tappy_m[lo:hi]).bool().to(device)
        return model.encode_finger_waveform(x, m, source="tappy").cpu().numpy()

    tappy_waveform_embed = batched(tappy_waveform_batch, len(tappy_labels))

    def tappy_fused_batch(lo, hi):
        wf = torch.from_numpy(tappy_waveform_embed[lo:hi]).to(device)
        return model.encode_finger_fused(wf, None, hi - lo, device).cpu().numpy()

    tappy_finger_embed = batched(tappy_fused_batch, len(tappy_labels))

    write_csv(
        out_dir / "phase2_tappy_embeddings.csv",
        tappy_labels[["UserKey", "Split", "SessionIndex", "Parkinsons"]],
        log_lines,
        waveform_embed=tappy_waveform_embed,
        finger_embed=tappy_finger_embed,
    )

    # ── GaitRec ─────────────────────────────────────────────────────────────
    gait_w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    gait_labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")

    def gait_batch(lo, hi):
        x = torch.from_numpy(gait_w[lo:hi]).float().to(device)
        return model.encode_gait(x).cpu().numpy()

    gait_embed = batched(gait_batch, len(gait_labels))

    write_csv(
        out_dir / "phase2_gaitrec_embeddings.csv",
        gait_labels[["SubjectID", "SessionID", "trial_id", "Split", "ClassLabel"]],
        log_lines,
        gait_embed=gait_embed,
    )

    # ── mPower ──────────────────────────────────────────────────────────────
    mpower_f = np.load(root / "mpower_preprocessed" / "mpower_features.npy")
    mpower_labels = pd.read_csv(root / "mpower_preprocessed" / "mpower_labels.csv")

    def mpower_batch(lo, hi):
        x = torch.from_numpy(mpower_f[lo:hi]).float().to(device)
        return model.mpower_branch(x).cpu().numpy()

    mpower_embed = batched(mpower_batch, len(mpower_labels))

    write_csv(
        out_dir / "phase2_mpower_embeddings.csv",
        mpower_labels[["recordId", "healthCode", "Split", "PD"]],
        log_lines,
        mpower_embed=mpower_embed,
    )

    log_lines.append("\nPhase 2 embedding extraction complete (all splits, all four datasets).\n")

    # ── Phase 3 — Cross-Body Dependency Module ────────────────────────────────
    ckpt3_path = root / "phase3_checkpoint.pt"
    ckpt3 = torch.load(ckpt3_path, map_location=device)
    cbdm = LaggedCrossAttention(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
    cbdm.load_state_dict(ckpt3["cbdm"])
    cbdm.eval()
    log_lines.append(f"Loaded Phase 3 checkpoint: {ckpt3_path} (lag heatmap: {ckpt3.get('lag_heatmap')})")

    pads_tier = pads_labels["label"].map(pads_label_to_tier).to_numpy()
    gait_tier = gait_labels["ClassLabel"].map(gaitrec_label_to_tier).to_numpy()

    rng = np.random.RandomState(args.seed)
    rows_meta = []
    fused_all = []
    lag_all = []

    for split in ["train", "val", "test"]:
        pads_idx = np.where(pads_labels["split"].to_numpy() == split)[0]
        gait_idx_pool = np.where(gait_labels["Split"].to_numpy() == split)[0]
        if len(pads_idx) == 0 or len(gait_idx_pool) == 0:
            continue

        g_choice = rng.choice(gait_idx_pool, size=len(pads_idx), replace=True)
        tier_f = pads_tier[pads_idx]
        tier_g = gait_tier[g_choice]
        pos_idx = build_positive_indices(tier_f, tier_g)  # indices into g_choice, per train_phase3_cbdm.py's pairing
        valid = pos_idx >= 0
        if valid.sum() == 0:
            continue

        f_idx_valid = pads_idx[valid]
        g_idx_valid = g_choice[pos_idx[valid]]

        for start in range(0, len(f_idx_valid), BATCH_SIZE):
            f_batch = f_idx_valid[start : start + BATCH_SIZE]
            g_batch = g_idx_valid[start : start + BATCH_SIZE]

            with torch.no_grad():
                fx = torch.from_numpy(pads_w[f_batch]).float().to(device)
                fm = torch.from_numpy(pads_m[f_batch]).bool().to(device)
                finger_windows = model.encode_finger_windows(fx, fm, source="pads", k=K_WINDOWS)

                gx = torch.from_numpy(gait_w[g_batch]).float().to(device)
                gait_windows = model.encode_gait_windows(gx, k=K_WINDOWS)

                fused, lag_weights = cbdm(finger_windows, gait_windows)

            fused_all.append(fused.cpu().numpy())
            lag_all.append(lag_weights.cpu().numpy())

            for fi, gi in zip(f_batch, g_batch):
                rows_meta.append(
                    {
                        "pads_subject_id": pads_labels.iloc[fi]["subject_id"],
                        "split": split,
                        "pads_tier": int(pads_tier[fi]),
                        "gaitrec_subject_id": gait_labels.iloc[gi]["SubjectID"],
                        "gaitrec_trial_id": gait_labels.iloc[gi]["trial_id"],
                        "gaitrec_tier": int(gait_tier[gi]),
                    }
                )

    fused_all = np.concatenate(fused_all, axis=0)
    lag_all = np.concatenate(lag_all, axis=0)
    meta_df = pd.DataFrame(rows_meta)

    write_csv(
        out_dir / "phase3_cbdm_embeddings.csv",
        meta_df,
        log_lines,
        fused_embed=fused_all,
        lag_weight=lag_all,
    )

    log_lines.append("\nPhase 3 Cross-Body embedding extraction complete (population-level tier-matched pairs, train/val/test).")
    log_lines.append(
        "NOTE: pairs are population-level (PADS finger sample x GaitRec gait sample matched only by "
        "Healthy/Pathological tier) — GaitRec and PADS share no subject IDs, so this is not a per-patient pairing."
    )

    log_path = out_dir / "generate_embedding_logs.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))
    print(f"\nAll outputs written to {out_dir.resolve()}")


def parse_args_embedding_logs():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ============================================================================
# RESULT FIGURES (from generate_figures.py)
# ============================================================================

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9


# ─── Log parsing ─────────────────────────────────────────────────────────────

def parse_phase2_log(path: Path) -> dict:
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+PADS\(fused/wf\)\s+([\d.]+)/([\d.]+)\s+->\s+val\s+([\d.]+)/([\d.]+)\s+\|\s+"
        r"Tappy\s+([\d.]+)\s+->\s+val\s+([\d.]+)\s+\|\s+Gait\s+([\d.]+)\s+->\s+val\s+([\d.]+)\s+\|\s+"
        r"mPower recon\s+([\d.]+)\s+->\s+val\s+([\d.]+)"
    )
    rows = {k: [] for k in ["epoch", "pads_train", "pads_val", "tappy_train", "tappy_val",
                              "gait_train", "gait_val", "mpower_train", "mpower_val"]}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if not m:
            continue
        rows["epoch"].append(int(m.group(1)))
        rows["pads_train"].append(float(m.group(2)))
        rows["pads_val"].append(float(m.group(4)))
        rows["tappy_train"].append(float(m.group(6)))
        rows["tappy_val"].append(float(m.group(7)))
        rows["gait_train"].append(float(m.group(8)))
        rows["gait_val"].append(float(m.group(9)))
        rows["mpower_train"].append(float(m.group(10)))
        rows["mpower_val"].append(float(m.group(11)))
    return rows


def parse_phase3_log(path: Path) -> dict:
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+InfoNCE\s+([\d.]+)\s+->\s+val\s+([\d.]+)\s+\|\s+SoftDTW\s+(-?[\d.]+)\s+->\s+val\s+(-?[\d.]+)"
    )
    rows = {k: [] for k in ["epoch", "infonce_train", "infonce_val", "dtw_train", "dtw_val"]}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if not m:
            continue
        rows["epoch"].append(int(m.group(1)))
        rows["infonce_train"].append(float(m.group(2)))
        rows["infonce_val"].append(float(m.group(3)))
        rows["dtw_train"].append(float(m.group(4)))
        rows["dtw_val"].append(float(m.group(5)))
    return rows


def parse_phase4_log(path: Path) -> dict:
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+PADS clinical head\s+([\d.]+)\s+->\s+val\s+([\d.]+)\s+\|\s+"
        r"GaitRec head \(secondary\)\s+([\d.]+)\s+->\s+val\s+([\d.]+)"
    )
    rows = {k: [] for k in ["epoch", "pads_train", "pads_val", "gait_train", "gait_val"]}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if not m:
            continue
        rows["epoch"].append(int(m.group(1)))
        rows["pads_train"].append(float(m.group(2)))
        rows["pads_val"].append(float(m.group(3)))
        rows["gait_train"].append(float(m.group(4)))
        rows["gait_val"].append(float(m.group(5)))
    return rows


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig_phase2(rows: dict, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Phase 2 — Modality Encoder Training", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(rows["epoch"], rows["pads_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["pads_val"], label="val", marker="o", ms=3)
    ax.axhline(0.592, color="gray", ls="--", lw=1, label="majority baseline")
    ax.set_title("PADS finger probe (PRIMARY)"); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(rows["epoch"], rows["tappy_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["tappy_val"], label="val", marker="o", ms=3)
    ax.axhline(0.845, color="gray", ls="--", lw=1, label="majority baseline")
    ax.set_title("Tappy finger probe (secondary)"); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(rows["epoch"], rows["gait_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["gait_val"], label="val", marker="o", ms=3)
    ax.axhline(0.295, color="gray", ls="--", lw=1, label="majority baseline")
    ax.set_title("GaitRec gait probe"); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(rows["epoch"], rows["mpower_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["mpower_val"], label="val", marker="o", ms=3)
    ax.set_title("mPower reconstruction (unsupervised)"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_phase3(rows: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Phase 3 — Cross-Body Dependency Module Training", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(rows["epoch"], rows["infonce_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["infonce_val"], label="val", marker="o", ms=3)
    ax.axhline(np.log(32), color="gray", ls="--", lw=1, label="random chance (ln 32)")
    ax.set_title("InfoNCE (population-level contrastive)"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(rows["epoch"], rows["dtw_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["dtw_val"], label="val", marker="o", ms=3)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("Soft-DTW alignment (positive pairs)"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_phase4(rows: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Phase 4 — Clinical Grounding Head Training", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(rows["epoch"], rows["pads_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["pads_val"], label="val", marker="o", ms=3)
    ax.axhline(0.592, color="gray", ls="--", lw=1, label="majority baseline")
    ax.set_title("PADS clinical head (PRIMARY)"); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(rows["epoch"], rows["gait_train"], label="train", marker="o", ms=3)
    ax.plot(rows["epoch"], rows["gait_val"], label="val", marker="o", ms=3)
    ax.axhline(0.295, color="gray", ls="--", lw=1, label="majority baseline")
    ax.set_title("GaitRec head (secondary check)"); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_lag_heatmap(lag_heatmap: dict, out: Path):
    lags = list(lag_heatmap.keys())
    weights = [lag_heatmap[k] for k in lags]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([l.replace("lag_", "Δ=") for l in lags], weights, color="#3b6fa0")
    for b, w in zip(bars, weights):
        ax.text(b.get_x() + b.get_width() / 2, w + 0.01, f"{w:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(weights) * 1.25)
    ax.set_ylabel("Mean softmax weight (test cohort)")
    ax.set_title("Attention-Lag Heatmap\n(which lag best explains cross-body structure)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds, targets):
        cm[t, p] += 1
    return cm


def fig_confusion(cm: np.ndarray, class_names: list[str], title: str, out: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(bins_before: list[dict], bins_after: list[dict], ece_before: float, ece_after: float, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, bins, ece, title in [
        (axes[0], bins_before, ece_before, "Before calibration"),
        (axes[1], bins_after, ece_after, "After calibration (Isotonic)"),
    ]:
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins if b["n"] > 0]
        accs = [b["acc"] for b in bins if b["n"] > 0]
        confs = [b["conf"] for b in bins if b["n"] > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.bar(centers, accs, width=0.08, alpha=0.7, label="observed accuracy")
        ax.scatter(confs, accs, color="red", zorder=5, s=15, label="bin (conf, acc)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=10)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("PADS Clinical Head — Reliability Diagram", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run_figures(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)
    device = get_device()

    # ── Training curves ──
    if (root / "train_run_v2.log").exists():
        fig_phase2(parse_phase2_log(root / "train_run_v2.log"), figdir / "phase2_training_curves.png")
        print("Saved figures/phase2_training_curves.png")
    p3log = root / "train_phase3_run2.log" if (root / "train_phase3_run2.log").exists() else root / "train_phase3_run1.log"
    if p3log.exists():
        fig_phase3(parse_phase3_log(p3log), figdir / "phase3_training_curves.png")
        print("Saved figures/phase3_training_curves.png")
    p4log = root / "train_phase4_run2.log" if (root / "train_phase4_run2.log").exists() else root / "train_phase4_run1.log"
    if p4log.exists():
        fig_phase4(parse_phase4_log(p4log), figdir / "phase4_training_curves.png")
        print("Saved figures/phase4_training_curves.png")

    # ── Attention-lag heatmap ──
    phase3_ckpt = torch.load(root / "phase3_checkpoint.pt", map_location="cpu")
    fig_lag_heatmap(phase3_ckpt["lag_heatmap"], figdir / "attention_lag_heatmap.png")
    print("Saved figures/attention_lag_heatmap.png")

    # ── Confusion matrices (recomputed fresh, full test set, no batch-drop) ──
    model = CBDLPhase2Model().to(device)
    model.load_state_dict(torch.load(root / "phase2_checkpoint.pt", map_location=device)["model"])
    cbdm = LaggedCrossAttention(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
    cbdm.load_state_dict(phase3_ckpt["cbdm"])
    phase4_ckpt = torch.load(root / "phase4_checkpoint.pt", map_location=device)
    pads_head = ClassifierHead(128, 64, 3).to(device); pads_head.load_state_dict(phase4_ckpt["pads_head"])
    gait_head = ClassifierHead(128, 64, 5).to(device); gait_head.load_state_dict(phase4_ckpt["gait_head"])
    gait_prototype = phase4_ckpt["gait_prototype"].to(device)
    finger_prototype = phase4_ckpt["finger_prototype"].to(device)
    model.eval(); cbdm.eval(); pads_head.eval(); gait_head.eval()

    pads_w, pads_m, pads_y, pads_split = load_pads(root)
    gait_w, gait_y, gait_split, gait_class_to_idx = load_gaitrec(root)
    idx_to_gait_class = {v: k for k, v in gait_class_to_idx.items()}

    test_idx = np.where(pads_split == "test")[0]
    x = torch.from_numpy(pads_w[test_idx]).float().to(device)
    m = torch.from_numpy(pads_m[test_idx]).bool().to(device)
    with torch.no_grad():
        fw = model.encode_finger_windows(x, m, source="pads", k=K_WINDOWS)
        gp = gait_prototype.expand(fw.size(0), -1, -1)
        fused, _ = cbdm(fw, gp)
        pads_preds = pads_head(fused).argmax(-1).cpu().numpy()
    pads_cm = confusion_matrix(pads_preds, pads_y[test_idx], 3)
    fig_confusion(pads_cm, ["Healthy", "Parkinson's", "Other"], "PADS Clinical Head (test, full 71 samples)",
                  figdir / "pads_confusion_matrix.png")
    print("Saved figures/pads_confusion_matrix.png")

    gtest_idx = np.where(gait_split == "test")[0]
    gait_preds_chunks = []
    chunk_size = 256
    for start in range(0, len(gtest_idx), chunk_size):
        chunk_idx = gtest_idx[start:start + chunk_size]
        gx = torch.from_numpy(gait_w[chunk_idx]).float().to(device)
        with torch.no_grad():
            gw = model.encode_gait_windows(gx, k=K_WINDOWS)
            fp = finger_prototype.expand(gw.size(0), -1, -1)
            fused_g, _ = cbdm(fp, gw)
            gait_preds_chunks.append(gait_head(fused_g).argmax(-1).cpu().numpy())
    gait_preds = np.concatenate(gait_preds_chunks)
    gait_cm = confusion_matrix(gait_preds, gait_y[gtest_idx], 5)
    class_names = [idx_to_gait_class[i] for i in range(5)]
    fig_confusion(gait_cm, class_names, f"GaitRec Secondary Head (test, {len(gtest_idx)} trials)",
                  figdir / "gaitrec_confusion_matrix.png")
    print("Saved figures/gaitrec_confusion_matrix.png")

    # ── Reliability diagram ──
    if (root / "phase5_calibration.pkl").exists():
        with open(root / "phase5_calibration.pkl", "rb") as f:
            cal = pickle.load(f)
        fig_reliability(cal["bins_before"], cal["bins_after"], cal["ece_before"], cal["ece_after"],
                         figdir / "pads_reliability_diagram.png")
        print("Saved figures/pads_reliability_diagram.png")

    print(f"\nAll figures saved to {figdir}/")


def parse_args_figures():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()

# ============================================================================
# PATIENT REPORT DATA (from generate_patient_report_data.py)
# ============================================================================

CHANNEL_NAMES = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]
TASK_BOUNDARIES = [("PointFinger", 0, 976), ("TouchIndex", 976, 1952), ("TouchNose", 1952, 2928)]
CLASS_NAMES = ["Healthy", "Parkinson's Disease", "Other Movement Disorder"]
SAMPLE_RATE_HZ = 976 / (2928 / 3) * (2928 / 3) / 1  # placeholder overwritten below if a real rate is known
# PADS raw tasks are 1024 samples before the 48-sample trim; sampling rate is
# documented by PADS as 100 Hz for the movement sensors.
FS_HZ = 100.0


class PADSClinicalPipelineWithLagWeights(nn.Module):
    def __init__(self, model, cbdm, pads_head, gait_prototype):
        super().__init__()
        self.model = model
        self.cbdm = cbdm
        self.pads_head = pads_head
        self.register_buffer("gait_prototype", gait_prototype)

    def forward(self, x):
        mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
        fw = self.model.encode_finger_windows(x, mask, source="pads", k=K_WINDOWS)
        gp = self.gait_prototype.expand(fw.size(0), -1, -1)
        fused, lag_weights = self.cbdm(fw, gp)
        return self.pads_head(fused), lag_weights


def load_demographics(root: Path) -> pd.DataFrame:
    path = root / "physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/file_list.csv"
    df = pd.read_csv(path)
    df = df[df["resource_type"] == "patient"].copy()
    df["subject_id"] = df["id"].astype(str).str.zfill(3)
    return df.set_index("subject_id")


def tremor_band_power_ratio(x: np.ndarray, fs: float = FS_HZ, band=(3.5, 7.0)) -> float:
    """Share of accelerometer-magnitude spectral power in the classic 3.5-7Hz
    parkinsonian resting-tremor band, out of total power (0-fs/2)."""
    accel = x[0:3, :]  # Accel X/Y/Z
    mag = np.sqrt((accel ** 2).sum(axis=0))
    mag = mag - mag.mean()
    freqs = np.fft.rfftfreq(len(mag), d=1.0 / fs)
    power = np.abs(np.fft.rfft(mag)) ** 2
    total = power.sum() + 1e-12
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(power[band_mask].sum() / total)


def movement_amplitude_rms(x: np.ndarray) -> float:
    """RMS of gyroscope magnitude — a movement-amplitude/speed proxy (lower = slower/more bradykinetic)."""
    gyro = x[3:6, :]
    mag = np.sqrt((gyro ** 2).sum(axis=0))
    return float(np.sqrt((mag ** 2).mean()))


def coordination_cross_correlation(x: np.ndarray) -> float:
    """Mean |correlation| between the 3 accelerometer and 3 gyroscope channels
    — a rough inter-channel coordination proxy (higher = more coupled movement)."""
    corrs = []
    for i in range(6):
        for j in range(i + 1, 6):
            c = np.corrcoef(x[i], x[j])[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))
    return float(np.mean(corrs)) if corrs else 0.0


def percentile_of(value: float, population: np.ndarray) -> float:
    return float((population < value).mean() * 100)


def grade_from_percentile(pct: float) -> str:
    if pct >= 75:
        return "High"
    if pct >= 40:
        return "Moderate"
    return "Low"


def run_patient_report(args):
    device = get_device()
    root = Path(args.data_root)

    pads_w, pads_m, pads_y, pads_split = load_pads(root)
    labels_df = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv", dtype={"subject_id": str})
    labels_df["subject_id"] = labels_df["subject_id"].str.zfill(3)
    demo = load_demographics(root)

    subject_id = args.subject_id.zfill(3)
    row_idx = labels_df.index[labels_df["subject_id"] == subject_id]
    if len(row_idx) == 0:
        raise ValueError(f"Subject {subject_id} not found in pads_labels.csv")
    idx = row_idx[0]
    if labels_df.loc[idx, "split"] != "test":
        print(f"WARNING: subject {subject_id} is in split='{labels_df.loc[idx, 'split']}', not 'test'. Proceeding anyway.")

    model = CBDLPhase2Model().to(device)
    model.load_state_dict(torch.load(root / "phase2_checkpoint.pt", map_location=device)["model"])
    cbdm = LaggedCrossAttention(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
    cbdm.load_state_dict(torch.load(root / "phase3_checkpoint.pt", map_location=device)["cbdm"])
    phase4_ckpt = torch.load(root / "phase4_checkpoint.pt", map_location=device)
    pads_head = ClassifierHead(128, 64, 3).to(device)
    pads_head.load_state_dict(phase4_ckpt["pads_head"])
    gait_prototype = phase4_ckpt["gait_prototype"].to(device)

    pipeline = PADSClinicalPipelineWithLagWeights(model, cbdm, pads_head, gait_prototype).to(device)
    pipeline.eval()
    for p in pipeline.parameters():
        p.requires_grad = False

    x_np = pads_w[idx]  # [6, 2928]
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits, lag_weights = pipeline(x)
        raw_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        lag_weights_np = lag_weights.cpu().numpy()[0]

    with open(root / "phase5_calibration.pkl", "rb") as f:
        cal = pickle.load(f)
    calibrated_probs = apply_calibrators(cal["calibrators"], raw_probs[None, :])[0]

    # ── Per-subject SHAP (small nsamples — one subject, needs to be fast) ──
    train_idx = np.where(pads_split == "train")[0]
    rng = np.random.RandomState(42)
    background_idx = rng.choice(train_idx, size=min(30, len(train_idx)), replace=False)
    background = torch.from_numpy(pads_w[background_idx]).float().to(device)

    class LogitsOnly(nn.Module):
        def __init__(self, pipe):
            super().__init__()
            self.pipe = pipe

        def forward(self, x):
            logits, _ = self.pipe(x)
            return logits

    logits_only = LogitsOnly(pipeline).to(device)
    explainer = shap.GradientExplainer(logits_only, background)
    shap_values = explainer.shap_values(x, nsamples=40)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    pred_class = int(raw_probs.argmax())
    sample_shap = shap_values[0, :, :, pred_class]  # [6, 2928]
    abs_shap = np.abs(sample_shap)

    channel_importance = abs_shap.sum(axis=1)
    channel_importance = (channel_importance / channel_importance.sum()).tolist()

    task_importance = []
    for name, lo, hi in TASK_BOUNDARIES:
        task_importance.append(float(abs_shap[:, lo:hi].sum()))
    total = sum(task_importance)
    task_importance = [v / total for v in task_importance]

    top_feature_idx = int(np.argmax(channel_importance))

    # ── DSP-derived motor pattern proxies, percentile-normed against TRAIN population ──
    train_waveforms = pads_w[train_idx]
    tremor_pop = np.array([tremor_band_power_ratio(w) for w in train_waveforms])
    amplitude_pop = np.array([movement_amplitude_rms(w) for w in train_waveforms])
    coord_pop = np.array([coordination_cross_correlation(w) for w in train_waveforms])

    tremor_val = tremor_band_power_ratio(x_np)
    amplitude_val = movement_amplitude_rms(x_np)
    coord_val = coordination_cross_correlation(x_np)

    tremor_pct = percentile_of(tremor_val, tremor_pop)
    amplitude_pct = percentile_of(amplitude_val, amplitude_pop)
    coord_pct = percentile_of(coord_val, coord_pop)
    # Lower movement amplitude percentile => more bradykinetic => report as inverted severity
    bradykinesia_severity_pct = 100 - amplitude_pct

    demographics = {}
    if subject_id in demo.index:
        d = demo.loc[subject_id]
        demographics = {
            "age": int(d["age"]) if pd.notna(d["age"]) else None,
            "gender": str(d["gender"]) if pd.notna(d["gender"]) else None,
            "height_cm": int(d["height"]) if pd.notna(d["height"]) else None,
            "weight_kg": int(d["weight"]) if pd.notna(d["weight"]) else None,
            "handedness": str(d["handedness"]) if pd.notna(d["handedness"]) else None,
            "condition_notes": str(d["disease_comment"]) if pd.notna(d.get("disease_comment")) and d["disease_comment"] != "-" else None,
        }

    true_label = labels_df.loc[idx, "condition"]

    n_sessions_for_subject = int((labels_df["subject_id"] == subject_id).sum())

    output = {
        "subject_id": subject_id,
        "demographics": demographics,
        "n_sessions_available": n_sessions_for_subject,
        "true_label_for_validation_only": true_label,  # kept for internal validation, not necessarily shown
        "diagnostic": {
            "class_names": CLASS_NAMES,
            "raw_probabilities": raw_probs.tolist(),
            "calibrated_probabilities": calibrated_probs.tolist(),
            "predicted_class": CLASS_NAMES[pred_class],
            "predicted_class_confidence_calibrated": float(calibrated_probs[pred_class]),
        },
        "cross_body": {
            "lags": [0, 1, 2, 3],
            "this_subject_lag_weights": lag_weights_np.tolist(),
            "dominant_lag": int(np.argmax(lag_weights_np)),
            "cohort_lag_weights_reference": None,  # filled from phase3_checkpoint.pt below
        },
        "explainability": {
            "channel_names": CHANNEL_NAMES,
            "channel_importance": channel_importance,
            "task_names": [t[0] for t in TASK_BOUNDARIES],
            "task_importance": task_importance,
            "top_channel": CHANNEL_NAMES[top_feature_idx],
        },
        "motor_patterns": {
            "tremor_band_power_ratio": tremor_val,
            "tremor_percentile_vs_population": tremor_pct,
            "tremor_severity": grade_from_percentile(tremor_pct),
            "movement_amplitude_rms": amplitude_val,
            "movement_amplitude_percentile_vs_population": amplitude_pct,
            "bradykinesia_severity_percentile": bradykinesia_severity_pct,
            "bradykinesia_severity": grade_from_percentile(bradykinesia_severity_pct),
            "coordination_cross_correlation": coord_val,
            "coordination_percentile_vs_population": coord_pct,
        },
        "data_coverage": {
            "pads_finger_imu": "Available — real sensor data for this subject",
            "gaitrec_gait": "NOT available for this subject — GaitRec and PADS share no subjects. "
                             "Cross-body analysis uses a fixed, tier-agnostic population gait prototype, not this subject's own gait.",
            "mpower_tapping_features": "NOT available for this subject — mPower's branch is trained but not "
                                         "connected to this classifier's decision path (see documentation §3.3/§7.2).",
        },
        "trend": {
            "available": n_sessions_for_subject > 1,
            "note": "Single-session dataset — PADS records exactly one session per subject, so no "
                    "longitudinal trend can be computed for this or any subject." if n_sessions_for_subject <= 1
                    else "Multiple sessions found.",
        },
    }

    phase3_ckpt = torch.load(root / "phase3_checkpoint.pt", map_location="cpu")
    output["cross_body"]["cohort_lag_weights_reference"] = phase3_ckpt["lag_heatmap"]

    out_path = root / f"patient_report_{subject_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")
    print(json.dumps(output, indent=2))


def parse_args_patient_report():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", type=str, required=True)
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()

# ============================================================================
# PIPELINE SHAPE DIAGRAMS (from generate_pipeline_shape_figures.py)
# ============================================================================

PSF_BLUE, PSF_LBLUE = "#3b6fa0", "#dbe7f5"
PSF_ORANGE, PSF_LORANGE = "#c96a2e", "#f6ddc9"
PSF_GREEN, PSF_LGREEN = "#4c8c5c", "#dcedd9"
PSF_GRAY, PSF_LGRAY = "#666666", "#e8e8e8"
PSF_RED = "#c0392b"


def box(ax, x, y, w, h, text, fc=PSF_LBLUE, ec=PSF_BLUE, fontsize=8, lw=1.3, style="round,pad=0.012"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)
    return x, y, w, h


def arrow(ax, p1, p2, color="#444444", lw=1.3, ls="-", **kw):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0), zorder=1, **kw)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def new_canvas(figsize, xlim, ylim, title):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    return fig, ax


# ─── Figure 1: Phase 2 — per-dataset shape flow through the modality encoders ──

def fig_encoders(out: Path):
    fig, ax = new_canvas((14, 9.5), (0, 148), (0, 100), "CBDL Phase 2 — Dataset Shapes Through the Modality Encoders")

    # GaitRec row
    g_in = box(ax, 2, 84, 22, 9, "GaitRec input\n[B, 18, 101]", fc=PSF_LGREEN, ec=PSF_GREEN)
    g_conv = box(ax, 28, 84, 24, 9, "Conv1d×2\n18→32→64ch, k=5", fc="white", ec=PSF_GRAY)
    g_lstm = box(ax, 56, 84, 22, 9, "BiLSTM\nhidden=64", fc="white", ec=PSF_GRAY)
    g_pool = box(ax, 82, 84, 20, 9, "mean-pool\n+ Linear→128", fc="white", ec=PSF_GRAY)
    g_out = box(ax, 106, 84, 26, 9, "Gait Embedding\n[B, 128]", fc=PSF_LGREEN, ec=PSF_GREEN, fontsize=9)
    for a, b_ in [(g_in, g_conv), (g_conv, g_lstm), (g_lstm, g_pool), (g_pool, g_out)]:
        arrow(ax, right(a), left(b_))

    # PADS row
    p_in = box(ax, 2, 62, 22, 9, "PADS input\n[B, 6, 2928]", fc=PSF_LBLUE, ec=PSF_BLUE)
    p_ad = box(ax, 28, 62, 24, 9, "WaveformAdapter\nConv1d(6→64), k=1", fc="white", ec=PSF_GRAY)
    arrow(ax, right(p_in), left(p_ad))

    # Tappy row
    t_in = box(ax, 2, 40, 22, 9, "Tappy input\n[B, 3, 888]", fc=PSF_LORANGE, ec=PSF_ORANGE)
    t_ad = box(ax, 28, 40, 24, 9, "WaveformAdapter\nConv1d(3→64), k=1", fc="white", ec=PSF_GRAY)
    arrow(ax, right(t_in), left(t_ad))

    # Shared trunk (converges PADS + Tappy)
    trunk = box(ax, 56, 48, 26, 13, "Shared\nFingerWaveformEncoder\nBiGRU(hidden=64)\nmasked mean-pool → Linear→128", fc="white", ec=PSF_BLUE, fontsize=7.8)
    arrow(ax, right(p_ad), (56, 62 + 4.5))
    arrow(ax, right(t_ad), (56, 40 + 4.5))
    ax.text(52, 61, "same\nweights", fontsize=6.5, color=PSF_BLUE, ha="center", style="italic")

    p_out = box(ax, 88, 62, 24, 9, "PADS waveform\nembedding [B, 128]", fc=PSF_LBLUE, ec=PSF_BLUE, fontsize=8)
    t_out = box(ax, 88, 40, 24, 9, "Tappy waveform\nembedding [B, 128]", fc=PSF_LORANGE, ec=PSF_ORANGE, fontsize=8)
    arrow(ax, (82, 62 + 4.5), left(p_out))
    arrow(ax, (82, 40 + 4.5), left(t_out))

    # mPower row
    m_in = box(ax, 2, 18, 22, 9, "mPower input\n[B, 41]", fc=PSF_LGRAY, ec=PSF_GRAY)
    m_mlp = box(ax, 28, 18, 24, 9, "MPowerMLPBranch\n41→64→128 (MLP)", fc="white", ec=PSF_GRAY)
    m_out = box(ax, 56, 18, 24, 9, "mPower embedding\n[B, 128]", fc=PSF_LGRAY, ec=PSF_GRAY, fontsize=8)
    arrow(ax, right(m_in), left(m_mlp))
    arrow(ax, right(m_mlp), left(m_out))
    m_dec = box(ax, 56, 3, 24, 9, "MPowerDecoder\n128→64→41\n(reconstruction loss —\nno diagnosis label available)", fc="white", ec=PSF_GRAY, fontsize=6.8)
    arrow(ax, bottom(m_out), top(m_dec), ls="--")

    # Fusion layer (PADS waveform embed + mPower embed -> Finger Embedding)
    fusion = box(ax, 116, 30, 30, 15, "FusionLayer\nconcat[256] → Linear → 128\n(learned 'absent' token\nsubstituted for the missing side —\nno sample has both a waveform\nand mPower features)", fc="white", ec=PSF_RED, fontsize=7)
    arrow(ax, right(p_out), left(fusion))
    arrow(ax, right(m_out), left(fusion))

    fe_out = box(ax, 116, 10, 30, 9, "Phase 2 Finger Embedding\n[B, 128]  (probe-only;\nsee note)", fc="white", ec=PSF_RED, fontsize=7.2)
    arrow(ax, bottom(fusion), top(fe_out))

    ax.text(74, 1.5,
            "Note: Phase 3 (CBDL) attends over the pre-fusion, WINDOWED trunk/gait outputs directly (Fig. 2) — the pooled Fusion-Layer\n"
            "embedding above is used only for the Phase 2 linear-probe sanity check, per model.py's encode_finger_windows / encode_gait_windows.",
            fontsize=7, color=PSF_GRAY, ha="center")

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 2: Phase 3 — Cross-Body Dependency Module (CBDL) shape flow ────────

def window_strip(ax, x, y, w, h, n, fc, ec):
    cell_w = w / n
    for i in range(n):
        box(ax, x + i * cell_w, y, cell_w * 0.92, h, "", fc=fc, ec=ec, lw=0.8)
    ax.text(x + w / 2, y + h + 2, f"K={n} windows", fontsize=7, ha="center", color=ec)


def fig_cbdl(out: Path):
    fig, ax = new_canvas((14, 10), (0, 148), (0, 100), "CBDL Phase 3 — Cross-Body Dependency Module")

    # Windowed encodings feeding in (population-level pairing, not per-subject)
    box(ax, 2, 84, 40, 8, "PADS window embeddings [B, 8, 128]\n(masked_window_pool, K=8)", fc=PSF_LBLUE, ec=PSF_BLUE, fontsize=7.6)
    window_strip(ax, 2, 76, 40, 5, 8, PSF_LBLUE, PSF_BLUE)
    box(ax, 2, 55, 40, 8, "GaitRec window embeddings [B, 8, 128]\n(masked_window_pool, K=8)", fc=PSF_LGREEN, ec=PSF_GREEN, fontsize=7.6)
    window_strip(ax, 2, 47, 40, 5, 8, PSF_LGREEN, PSF_GREEN)

    ax.text(22, 66.5, "population-level pairing by\nDIAGNOSIS TIER (Healthy vs.\nPathological) — not subject\nidentity (0 shared subject IDs)",
            fontsize=6.8, color=PSF_RED, ha="center", style="italic")

    arrow(ax, (42, 80), (52, 74))
    arrow(ax, (42, 51), (52, 68))

    # Lagged cross-attention: 4 lag branches
    lag_y = [88, 74, 60, 46]
    lag_boxes = []
    for d, y in zip([0, 1, 2, 3], lag_y):
        b_ = box(ax, 52, y, 42, 11,
                  f"Δ={d}: finger[0:{8-d}] attends gait[{d}:8]\n"
                  f"MultiheadAttention(4 heads)\n(f + attended)/2 → mean-pool → [B,128]",
                  fc="white", ec=PSF_GRAY, fontsize=6.6)
        lag_boxes.append(b_)

    scorer = box(ax, 100, 60, 20, 26, "Lag Scorer\nLinear(128→64)\nReLU\nLinear(64→1)\nper lag → 4 logits\n\nsoftmax → lag\nweights [B, 4]", fc="white", ec=PSF_ORANGE, fontsize=7)
    for b_ in lag_boxes:
        arrow(ax, right(b_), left(scorer))

    weighted = box(ax, 100, 30, 20, 18, "Weighted sum\nΣ weight_d · repr_d\n→ [B, 128]\n\n(this produces the\nAttention-Lag Heatmap\nwhen averaged over\nthe test cohort)", fc="white", ec=PSF_ORANGE, fontsize=6.8)
    arrow(ax, bottom(scorer), top(weighted))
    for b_ in lag_boxes:
        arrow(ax, (94, b_[1] + b_[3] / 2), (100, 39), lw=0.7, color="#bbbbbb")

    outproj = box(ax, 124, 40, 22, 8, "out_proj\nLinear(128→128)", fc="white", ec=PSF_BLUE, fontsize=7.5)
    arrow(ax, right(weighted), left(outproj))

    fused = box(ax, 108, 12, 38, 10, "Fused Cross-Body Embedding\n[B, 128]  →  Phase 4 (Fig. 3)", fc=PSF_LBLUE, ec=PSF_BLUE, fontsize=8.5)
    arrow(ax, bottom(outproj), (135, 22))

    # Auxiliary training-only losses (dashed, off to the side)
    dtw = box(ax, 2, 22, 40, 10, "Soft-DTW alignment loss\n(align_proj re-embeds BOTH sides —\nencoders are frozen in Phase 3, so\nSoft-DTW needs a trainable target)", fc="white", ec=PSF_GRAY, fontsize=6.6, style="round,pad=0.012")
    infonce = box(ax, 2, 6, 40, 10, "InfoNCE contrastive loss\n(in-batch negatives; positive =\nsame-tier gait sample; computed on\nthe FUSED embedding, not raw encoder\noutput, so attention gets gradient)", fc="white", ec=PSF_GRAY, fontsize=6.4)
    ax.text(22, 34.5, "training-only auxiliary losses\n(not part of the forward path used at inference)", fontsize=6.5, color=PSF_GRAY, ha="center", style="italic")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 3: Phase 4 — final classification output shape flow ───────────────

def fig_final_output(out: Path):
    fig, ax = new_canvas((13, 6.5), (0, 140), (0, 85), "CBDL Phase 4 — Final Classification Output")

    fused = box(ax, 2, 44, 30, 12, "Fused Cross-Body\nEmbedding [B, 128]\n(from Fig. 2)", fc=PSF_LBLUE, ec=PSF_BLUE, fontsize=9)

    # PADS primary head
    p_head = box(ax, 42, 66, 34, 11, "PADS Clinical Head (PRIMARY)\nLinear(128→64)→ReLU→Dropout(0.2)\n→Linear(64→3)", fc="white", ec=PSF_BLUE, fontsize=7.5)
    p_logit = box(ax, 82, 66, 20, 11, "logits\n[B, 3]", fc=PSF_LBLUE, ec=PSF_BLUE, fontsize=8)
    p_soft = box(ax, 108, 66, 26, 11, "softmax → argmax\nHealthy / Parkinson's /\nOther Movement Disorder", fc="white", ec=PSF_BLUE, fontsize=7)
    arrow(ax, right(fused), (42, 71.5))
    arrow(ax, right(p_head), left(p_logit))
    arrow(ax, right(p_logit), left(p_soft))

    p_cal = box(ax, 108, 50, 26, 11, "Phase 5: isotonic\ncalibration recalibrates\nthese probabilities\n(see reliability diagram)", fc="white", ec=PSF_GRAY, fontsize=6.8)
    arrow(ax, bottom(p_soft), top(p_cal), ls="--")

    # GaitRec secondary head
    g_head = box(ax, 42, 20, 34, 11, "GaitRec Head (secondary check)\nLinear(128→64)→ReLU→Dropout(0.2)\n→Linear(64→5)", fc="white", ec=PSF_GREEN, fontsize=7.5)
    g_logit = box(ax, 82, 20, 20, 11, "logits\n[B, 5]", fc=PSF_LGREEN, ec=PSF_GREEN, fontsize=8)
    g_soft = box(ax, 108, 20, 26, 11, "softmax → argmax\nHC / A / C / H / K\n(pathology location)", fc="white", ec=PSF_GREEN, fontsize=7)
    arrow(ax, right(fused), (42, 25.5))
    arrow(ax, right(g_head), left(g_logit))
    arrow(ax, right(g_logit), left(g_soft))

    ax.text(70, 4, "At test time, the side without a real paired sample is filled with a fixed population PROTOTYPE (mean of train-split windows) —\n"
                   "PADS test samples pair with a gait_prototype, GaitRec test samples pair with a finger_prototype (no sample has both, §1).",
            fontsize=7, color=PSF_RED, ha="center", style="italic")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run_pipeline_shapes(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_encoders(figdir / "pipeline_encoders.png")
    print("Saved figures/pipeline_encoders.png")
    fig_cbdl(figdir / "pipeline_cbdl.png")
    print("Saved figures/pipeline_cbdl.png")
    fig_final_output(figdir / "pipeline_final_output.png")
    print("Saved figures/pipeline_final_output.png")


def parse_args_pipeline_shapes():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()

# ============================================================================
# PREPROCESSING FIGURES (from generate_preprocessing_figures.py)
# ============================================================================

GAITREC_CHANNELS = [
    "F_V_RAW_L", "F_V_RAW_R", "F_V_PRO_R", "F_ML_RAW_L", "F_ML_RAW_R",
    "F_ML_PRO_L", "F_ML_PRO_R", "F_AP_RAW_L", "F_AP_RAW_R", "F_AP_PRO_R",
    "COP_ML_RAW_L", "COP_ML_RAW_R", "COP_ML_PRO_L", "COP_ML_PRO_R",
    "COP_AP_RAW_L", "COP_AP_RAW_R", "COP_AP_PRO_R", "COP_AP_PRO_L",
]
GAITREC_MEAN = np.array([637.114013671875, 636.0904541015625, 0.7786163687705994, 18.694671630859375,
    18.2748966217041, 0.02246806025505066, 0.02197427488863468, 3.402470350265503, 2.19914174079895,
    0.0027927705086767673, 0.05621721222996712, 0.05718270316720009, -9.529247108730488e-06,
    -8.657619218865875e-06, 0.013265073299407959, 0.013461710885167122, 0.13937710225582123,
    0.1385464370250702], dtype=np.float32)
GAITREC_STD = np.array([276.57122802734375, 277.1957092285156, 0.2982848584651947, 22.62496566772461,
    22.631778717041016, 0.026182014495134354, 0.02623864822089672, 81.16717529296875, 81.36552429199219,
    0.09826258569955826, 0.030105339363217354, 0.030528105795383453, 0.00979784969240427,
    0.010680989362299442, 0.09587275236845016, 0.09601877629756927, 0.07332467287778854,
    0.07315360754728317], dtype=np.float32)

TAPPY_CHANNELS = ["hold_time", "latency_time", "flight_time"]
TAPPY_MEAN = np.array([119.33480072021484, 262.592529296875, 192.0791473388672], dtype=np.float32)
TAPPY_STD = np.array([67.53646850585938, 144.2754364013672, 131.09173583984375], dtype=np.float32)

MPOWER_FEATURES = ["meanTapInter", "medianTapInter", "iqrTapInter", "minTapInter", "maxTapInter",
    "skewTapInter", "kurTapInter", "sdTapInter", "madTapInter", "cvTapInter", "rangeTapInter",
    "tkeoTapInter", "ar1TapInter", "ar2TapInter", "fatigue10TapInter", "fatigue25TapInter",
    "fatigue50TapInter", "meanDriftLeft", "medianDriftLeft", "iqrDriftLeft", "minDriftLeft",
    "maxDriftLeft", "skewDriftLeft", "kurDriftLeft", "sdDriftLeft", "madDriftLeft", "cvDriftLeft",
    "rangeDriftLeft", "meanDriftRight", "medianDriftRight", "iqrDriftRight", "minDriftRight",
    "maxDriftRight", "skewDriftRight", "kurDriftRight", "sdDriftRight", "madDriftRight",
    "cvDriftRight", "rangeDriftRight", "numberTaps", "buttonNoneFreq"]
MPOWER_MEAN = np.array([0.18978873051934178, 0.1960459328245129, 0.18505561316118269, 0.018624785548315446,
    0.6170732550426022, 0.5961410717994132, 4.721036447580436, 0.12824153756443504, 0.12516086792213293,
    75.83087690536358, 0.5984484694942867, 0.014270979908866065, -0.4780340490282895, 0.39282595438153955,
    -0.0004565639704958843, -0.00474728049683401, -0.003973049880860573, 12.027709820616018,
    10.398606920391167, 9.342291504542862, 1.3156176088573832, 38.56056412339032, 1.2244355838522856,
    5.204768014802605, 7.945954570471145, 6.630566896522515, 66.30259469674104, 37.244946514532764,
    12.383300573495529, 10.49699076458324, 9.535462225201057, 1.3353567398193358, 42.0456406647591,
    1.2794731647224944, 5.5432537125346055, 8.74225503930565, 6.677406276423131, 68.24932187981294,
    40.71028392493977, 133.39532306328272, 0.027615731129093053], dtype=np.float32)
MPOWER_STD = np.array([0.11214945617218494, 0.10493382825233931, 0.09174045386248711, 0.06928902345224595,
    0.47302598120018, 1.0417080565884562, 7.003980608496841, 0.0730841312283163, 0.07015471780821977,
    26.153974483890316, 0.46800672131686355, 0.04720994781334695, 0.2812427259761467, 0.2761058353046568,
    0.09003438749557227, 0.06273030618682598, 0.04213877518737384, 6.100310079792625, 5.364503982221159,
    6.23263580385101, 1.1330941949665347, 24.139537374415102, 0.7448743588900684, 3.9183250483345464,
    5.45597301228183, 3.7439124993889727, 15.683993763699778, 23.907925293690617, 6.816380317534866,
    5.994531275372201, 7.286267343260951, 1.1305435385125542, 34.83674016233986, 0.8535238958239708,
    4.881753013708619, 8.693956652658315, 3.883584692321453, 23.258388457516293, 34.67074357631933,
    53.874484067508206, 0.05774845049767557], dtype=np.float32)

PPF_BLUE, PPF_ORANGE, PPF_GRAY, PPF_RED = "#3b6fa0", "#d97a3f", "#888888", "#c0392b"


# ─── GaitRec ─────────────────────────────────────────────────────────────────

def fig_gaitrec_preprocessing(root: Path, out: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    raw = w * GAITREC_STD[None, :, None] + GAITREC_MEAN[None, :, None]

    show_ch = [GAITREC_CHANNELS.index(c) for c in ["F_V_RAW_L", "F_AP_RAW_L", "COP_AP_RAW_L"]]
    trial = np.where(labels["ClassLabel"].values == "HC")[0][0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("GaitRec — Raw Force-Plate Signal vs. Preprocessed Waveform", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    for i, ci in enumerate(show_ch):
        ax.plot(raw[trial, ci], label=GAITREC_CHANNELS[ci], color=[PPF_BLUE, PPF_ORANGE, "#5aa469"][i])
    ax.set_title("Before: raw sensor units\n(one HC trial, 3 of 20 recorded channels)")
    ax.set_xlabel("% gait cycle (0-100)"); ax.set_ylabel("raw units (N, mm)"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    for i, ci in enumerate(show_ch):
        ax.plot(w[trial, ci], label=GAITREC_CHANNELS[ci], color=[PPF_BLUE, PPF_ORANGE, "#5aa469"][i])
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_title("After: per-channel z-score\n(same trial, unified scale)")
    ax.set_xlabel("% gait cycle (0-100)"); ax.set_ylabel("z-score"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    bars = ax.bar(["Recorded\n(20 channels)", "Kept\n(18 channels)"], [20, 18], color=[PPF_GRAY, PPF_BLUE], width=0.5)
    for b, v in zip(bars, [20, 18]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center", fontweight="bold")
    ax.set_ylim(0, 23)
    ax.set_title("Channel drop: 2 corrupted channels excluded")
    ax.text(0.5, -0.32, "GRF_F_V_PRO_left (0% present) · GRF_F_AP_PRO_left (97% missing)\n"
                         "verified: not a row drop — same 75,732-trial key set across the other 18",
            transform=ax.transAxes, ha="center", fontsize=7, color=PPF_RED)

    ax = axes[1, 1]
    splits = ["train", "val", "test"]
    trials = [53456, 11338, 10938]
    subjects = [1607, 345, 343]
    x = np.arange(3)
    ax.bar(x - 0.18, trials, width=0.36, color=PPF_BLUE, label="trials")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, subjects, width=0.36, color=PPF_ORANGE, label="subjects")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("trials", color=PPF_BLUE); ax2.set_ylabel("subjects", color=PPF_ORANGE)
    ax.set_title("Subject-disjoint split (75,732 trials / 2,295 subjects total)")
    for xi, v in zip(x - 0.18, trials):
        ax.text(xi, v + 800, str(v), ha="center", fontsize=7, color=PPF_BLUE)
    for xi, v in zip(x + 0.18, subjects):
        ax2.text(xi, v + 20, str(v), ha="center", fontsize=7, color=PPF_ORANGE)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── Tappy ───────────────────────────────────────────────────────────────────

def fig_tappy_preprocessing(root: Path, out: Path):
    w = np.load(root / "tappy_preprocessed" / "tappy_waveforms.npy")
    m = np.load(root / "tappy_preprocessed" / "tappy_masks.npy")
    labels = pd.read_csv(root / "tappy_preprocessed" / "tappy_labels.csv")
    raw_lens = labels["SessionLength"].values
    T = w.shape[2]

    short_idx = np.argmin(np.abs(raw_lens - 60))   # gets padded
    long_idx = np.argmax(raw_lens)                 # gets truncated

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Tappy — Raw Keystroke Event Stream vs. Preprocessed Fixed-Length Waveform", fontsize=12, fontweight="bold")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.55, 0.55], hspace=0.55, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.logspace(np.log10(20), np.log10(raw_lens.max()), 40)
    ax.hist(raw_lens, bins=bins, color=PPF_BLUE, alpha=0.85)
    ax.set_xscale("log")
    for val, lbl, y in [(109, "p50=109", 0.75), (888, "T=888 (p90)", 0.55), (3636, "p99=3636", 0.35)]:
        ax.axvline(val, color=PPF_RED, ls="--", lw=1)
        ax.text(val * 1.15, ax.get_ylim()[1] * y, lbl, fontsize=7, color=PPF_RED)
    ax.set_title("Before: raw session length distribution\n(min=20, max=46,101 keystrokes, n=23,752 sessions)", fontsize=9)
    ax.set_xlabel("keystrokes per session (log scale)"); ax.set_ylabel("# sessions")

    ax = fig.add_subplot(gs[0, 1])
    cats = ["kept\n(≤10,000ms)", "dropped\n(implausible, >10,000ms)"]
    vals = [9_300_000 - 615, 615]
    bars = ax.bar(cats, vals, color=[PPF_BLUE, PPF_RED])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:,}", ha="center", fontsize=8)
    ax.set_title("Hold-time outlier filter\n(max raw value ≈ 13.6M ms — stuck-key artifact)", fontsize=9)
    ax.set_ylabel("# raw keystroke\nevents (log scale)")

    ax = fig.add_subplot(gs[1, :])
    hold_short_raw = w[short_idx, 0][m[short_idx]] * TAPPY_STD[0] + TAPPY_MEAN[0]
    ax.plot(hold_short_raw, color=PPF_BLUE, marker="o", ms=3)
    ax.axvspan(len(hold_short_raw), T, color=PPF_BLUE, alpha=0.12)
    ax.set_xlim(0, T)
    ax.set_title(f"After — short session ({raw_lens[short_idx]} events): zero-padded to T={T} (shaded = padded, masked out)", fontsize=9)
    ax.set_ylabel("hold_time (ms)")

    ax = fig.add_subplot(gs[2, :])
    hold_long_raw = w[long_idx, 0] * TAPPY_STD[0] + TAPPY_MEAN[0]
    ax.plot(hold_long_raw, color=PPF_ORANGE, lw=0.6)
    ax.axvline(T, color=PPF_RED, ls="--", lw=1)
    ax.set_xlim(0, len(hold_long_raw))
    ax.set_title(f"After — long session ({raw_lens[long_idx]} events): truncated at T={T} (red line = cutoff, tail discarded)", fontsize=9)
    ax.set_xlabel("raw timestep"); ax.set_ylabel("hold_time (ms)")

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── PADS ────────────────────────────────────────────────────────────────────

def fig_pads_preprocessing(root: Path, out: Path):
    w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")

    subj = 0
    example = w[subj, 0]  # Accelerometer_X, z-scored (data card does not list per-channel mean/std separately from split fit; shown in z-score space)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("PADS — Raw Wrist-IMU Recording vs. Preprocessed Concatenated Waveform", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    tasks = ["PointFinger", "TouchIndex", "TouchNose"]
    starts = [0, 1024, 2048]
    for s, t, c in zip(starts, tasks, [PPF_BLUE, PPF_ORANGE, "#5aa469"]):
        ax.barh(0, 1024, left=s, height=0.5, color=c, alpha=0.35, label=f"{t} (1024 raw samples)")
        ax.barh(0, 48, left=s, height=0.5, color=PPF_RED, alpha=0.8)
    ax.set_yticks([]); ax.set_xlim(0, 3072)
    ax.set_title("Before: 3 tasks × 1024 raw samples each\n(red = first 48 samples trimmed per task — startup vibration artifact)")
    ax.set_xlabel("raw sample index"); ax.legend(fontsize=6.5, loc="upper center", ncol=1)

    ax = axes[0, 1]
    ax.barh(0, 2928, left=0, height=0.5, color=PPF_BLUE, alpha=0.6)
    for s in [976, 1952]:
        ax.axvline(s, color="black", ls="--", lw=1)
    ax.set_yticks([]); ax.set_xlim(0, 3072)
    ax.set_title("After: concatenated, trimmed → T=2,928\n(976 samples/task × 3 tasks)")
    ax.set_xlabel("preprocessed timestep")

    ax = axes[1, 0]
    cats = ["Shipped .bin\n(run_preprocessing.py)", "Re-derived from\nraw .txt (used here)"]
    pointfinger = [0, 469]
    touchindex = [0, 469]
    touchnose = [469, 469]
    x = np.arange(2)
    w_ = 0.25
    ax.bar(x - w_, pointfinger, width=w_, color=PPF_BLUE, label="PointFinger")
    ax.bar(x, touchindex, width=w_, color=PPF_ORANGE, label="TouchIndex")
    ax.bar(x + w_, touchnose, width=w_, color="#5aa469", label="TouchNose")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("subjects with task data (of 469)")
    ax.set_title("Why raw .txt, not shipped .bin\n(.bin drops PointFinger + TouchIndex entirely)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(example, color=PPF_BLUE, lw=0.7)
    for s in [976, 1952]:
        ax.axvline(s, color="black", ls="--", lw=0.8)
    ax.set_title(f"Preprocessed Accelerometer_X, subject {labels.iloc[subj]['subject_id']}\n"
                 "(z-scored; dashed lines = task boundaries)")
    ax.set_xlabel("timestep"); ax.set_ylabel("z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── mPower ──────────────────────────────────────────────────────────────────

def fig_mpower_preprocessing(root: Path, out: Path):
    labels = pd.read_csv(root / "mpower_preprocessed" / "mpower_labels.csv")
    feats = np.load(root / "mpower_preprocessed" / "mpower_features.npy")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("mPower — Raw Tapping-Session Features vs. Preprocessed Feature Table", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    cats = ["tapFeatures.tsv\n(broader, no subject id)", "data_for_treat_vs_tod\n_paper.csv (chosen)"]
    rows = [78873, 12915]
    bars = ax.bar(cats, rows, color=[PPF_GRAY, PPF_BLUE])
    for b, v in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 1000, f"{v:,}", ha="center", fontsize=8)
    ax.set_title("Source file choice\n(only the chosen file carries healthCode → subject-disjoint split)")
    ax.set_ylabel("raw rows")

    ax = axes[1, 0]
    stages = ["Raw records", "− missing\nfeature values", "− numberTaps\n< 10", "Final"]
    vals = [12915, 12915 - 1, 12915 - 1 - 4, 12910]
    ax.plot(stages, vals, "o-", color=PPF_BLUE)
    for s, v in zip(stages, vals):
        ax.text(s, v + 5, str(v), ha="center", fontsize=8)
    ax.set_title("Cleaning funnel (104 subjects preserved throughout)")
    ax.set_ylabel("# records"); ax.set_ylim(12905, 12920)

    show = ["numberTaps", "cvTapInter", "sdTapInter", "iqrDriftLeft", "buttonNoneFreq"]
    idx = [MPOWER_FEATURES.index(f) for f in show]

    ax = axes[0, 1]
    x = np.arange(len(show))
    ax.bar(x, MPOWER_MEAN[idx], yerr=MPOWER_STD[idx], color=PPF_GRAY, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=30, ha="right", fontsize=7)
    ax.set_title("Before: raw feature scales differ by orders\nof magnitude (mean ± std, 5 of 41 features shown)")
    ax.set_yscale("symlog")

    ax = axes[1, 1]
    ax.bar(x, np.zeros(len(show)), yerr=np.ones(len(show)), color=PPF_BLUE, capsize=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=30, ha="right", fontsize=7)
    ax.set_title("After: z-scored per feature\n(all mean=0, std=1 — comparable to the MLP branch)")
    ax.set_ylim(-1.5, 1.5)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run_preprocessing_figs(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_gaitrec_preprocessing(root, figdir / "preprocessing_gaitrec.png")
    print("Saved figures/preprocessing_gaitrec.png")
    fig_tappy_preprocessing(root, figdir / "preprocessing_tappy.png")
    print("Saved figures/preprocessing_tappy.png")
    fig_pads_preprocessing(root, figdir / "preprocessing_pads.png")
    print("Saved figures/preprocessing_pads.png")
    fig_mpower_preprocessing(root, figdir / "preprocessing_mpower.png")
    print("Saved figures/preprocessing_mpower.png")


def parse_args_preprocessing_figs():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()

# ============================================================================
# WAVELET CHANNEL STRUCTURE (from generate_wavelet_channel_structure.py)
# ============================================================================

WAVELET = "cmor1.5-1.0"
PADS_CHANNELS = ["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z", "Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z"]


def plot_channel_grid(fig, outer_gs, signals: list[np.ndarray], names: list[str], scales: np.ndarray,
                       ncols: int, header: str, header_color: str, header_gap: float):
    n = len(names)
    nrows = math.ceil(n / ncols)
    inner = outer_gs.subgridspec(nrows, ncols, hspace=0.85, wspace=0.35)
    for i, (sig, name) in enumerate(zip(signals, names)):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(inner[r, c])
        coeffs, _ = pywt.cwt(sig, scales, WAVELET)
        power = np.abs(coeffs)
        ax.imshow(power, extent=[0, len(sig), scales[-1], scales[0]], aspect="auto",
                  cmap="viridis", interpolation="bilinear")
        ax.set_title(name, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    # blank out any unused grid cells
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        fig.add_subplot(inner[r, c]).axis("off")

    # header label centered above this block
    pos = outer_gs.get_position(fig)
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + header_gap, header, ha="center", va="bottom",
              fontsize=11, fontweight="bold", color=header_color)


def dataset_figure(out: Path, title: str, before_signals, after_signals, names, scales, ncols, figsize, grid_top):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.995)
    outer = fig.add_gridspec(1, 2, wspace=0.18, left=0.03, right=0.99, top=grid_top, bottom=0.03)
    header_gap = 0.4 / figsize[1]  # fixed ~0.4in gap, expressed as a figure-height fraction
    plot_channel_grid(fig, outer[0, 0], before_signals, names, scales, ncols, "BEFORE — raw", "#3b6fa0", header_gap)
    plot_channel_grid(fig, outer[0, 1], after_signals, names, scales, ncols, "AFTER — preprocessed", "#d97a3f", header_gap)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── GaitRec — all 18 GRF channels ─────────────────────────────────────────────

def fig_gaitrec_wavelet_channels(root: Path, out: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    trial = np.where(labels["ClassLabel"].values == "HC")[0][0]

    names = [c.replace(".csv", "") for c in GAITREC_CHANNELS]
    after = [w[trial, c] for c in range(18)]
    before = [w[trial, c] * GAITREC_STD[c] + GAITREC_MEAN[c] for c in range(18)]
    scales = np.arange(1, 33)

    dataset_figure(out, "GaitRec — All 18 GRF Channels, Wavelet Structure Before vs. After Preprocessing\n(one HC trial; each tile = one channel's CWT, % gait cycle on x-axis, scale on y-axis)",
                   before, after, names, scales, ncols=3, figsize=(13, 11.5), grid_top=0.87)


# ─── PADS — all 6 IMU channels ─────────────────────────────────────────────────

def fig_pads_wavelet_channels(root: Path, out: Path):
    raw_txt = root / "physionet.org" / "files" / "parkinsons-disease-smartwatch" / "1.0.0" / \
        "movement" / "timeseries" / "113_PointFinger_RightWrist.txt"
    raw = np.loadtxt(raw_txt, delimiter=",")
    before = [raw[:, c] for c in range(1, 7)]  # AccX,AccY,AccZ,GyroX,GyroY,GyroZ

    w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")
    idx = labels.index[labels["subject_id"] == 113][0]
    after = [w[idx, c, :976] for c in range(6)]

    scales = np.arange(1, 65)
    dataset_figure(out, "PADS — All 6 Wrist-IMU Channels, Wavelet Structure Before vs. After Preprocessing\n(subject 113, PointFinger task; before = raw .txt, after = gravity-corrected + trimmed + z-scored)",
                   before, after, PADS_CHANNELS, scales, ncols=3, figsize=(13, 8.5), grid_top=0.80)


# ─── Tappy — all 3 keystroke-timing channels ───────────────────────────────────

def fig_tappy_wavelet_channels(root: Path, out: Path):
    w = np.load(root / "tappy_preprocessed" / "tappy_waveforms.npy")
    m = np.load(root / "tappy_preprocessed" / "tappy_masks.npy")
    labels = pd.read_csv(root / "tappy_preprocessed" / "tappy_labels.csv")
    lens = labels["SessionLength"].values

    idx = np.argmin(np.abs(lens - 400))
    n = int(m[idx].sum())
    after = [w[idx, c, :n] for c in range(3)]
    before = [w[idx, c, :n] * TAPPY_STD[c] + TAPPY_MEAN[c] for c in range(3)]

    scales = np.arange(1, 49)
    dataset_figure(out, f"Tappy — All 3 Keystroke-Timing Channels, Wavelet Structure Before vs. After Preprocessing\n(one {n}-keystroke session; before = raw ms, after = z-scored)",
                   before, after, TAPPY_CHANNELS, scales, ncols=3, figsize=(11, 4.6), grid_top=0.60)


def run_wavelet_channels(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_gaitrec_wavelet_channels(root, figdir / "wavelet_channels_gaitrec.png")
    print("Saved figures/wavelet_channels_gaitrec.png")
    fig_pads_wavelet_channels(root, figdir / "wavelet_channels_pads.png")
    print("Saved figures/wavelet_channels_pads.png")
    fig_tappy_wavelet_channels(root, figdir / "wavelet_channels_tappy.png")
    print("Saved figures/wavelet_channels_tappy.png")


def parse_args_wavelet_channels():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()

# ============================================================================
# WAVELET FIGURES (from generate_wavelet_figures.py)
# ============================================================================

WVF_BLUE, WVF_ORANGE = "#3b6fa0", "#d97a3f"


def scalogram(ax, signal: np.ndarray, scales: np.ndarray, xlabel: str, title: str):
    coeffs, _ = pywt.cwt(signal, scales, WAVELET)
    power = np.abs(coeffs)
    im = ax.imshow(power, extent=[0, len(signal), scales[-1], scales[0]], cmap="viridis",
                    aspect="auto", interpolation="bilinear")
    ax.set_ylabel("wavelet scale\n(large = low frequency)")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
    return im


def trace_and_scalogram(fig, gs_col, signal: np.ndarray, scales: np.ndarray, color: str,
                         trace_title: str, scalo_title: str, xlabel: str, ylabel: str):
    ax_t = fig.add_subplot(gs_col[0])
    ax_t.plot(signal, color=color, lw=0.8)
    ax_t.set_title(trace_title, fontsize=9)
    ax_t.set_ylabel(ylabel)
    ax_t.set_xlim(0, len(signal))

    ax_s = fig.add_subplot(gs_col[1])
    im = scalogram(ax_s, signal, scales, xlabel, scalo_title)
    fig.colorbar(im, ax=ax_s, fraction=0.046, pad=0.04, label="|CWT|")


# ─── GaitRec ─────────────────────────────────────────────────────────────────

def fig_gaitrec_wavelet(root: Path, out: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    ch = GAITREC_CHANNELS.index("F_V_RAW_L")
    trial = np.where(labels["ClassLabel"].values == "HC")[0][0]

    raw_sig = w[trial, ch] * GAITREC_STD[ch] + GAITREC_MEAN[ch]
    pre_sig = w[trial, ch]
    scales = np.arange(1, 33)

    fig = plt.figure(figsize=(12, 7.5))
    fig.suptitle("GaitRec — Wavelet Scalogram, Raw vs. Preprocessed\n(GRF_F_V_RAW_left, one HC trial — illustrative CWT, not a pipeline step)",
                 fontsize=11, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.4, wspace=0.3)

    trace_and_scalogram(fig, [gs[0, 0], gs[1, 0]], raw_sig, scales, WVF_BLUE,
                         "Before: raw sensor units", "CWT of raw signal",
                         "% gait cycle", "raw units (N)")
    trace_and_scalogram(fig, [gs[0, 1], gs[1, 1]], pre_sig, scales, WVF_ORANGE,
                         "After: z-scored", "CWT of z-scored signal",
                         "% gait cycle", "z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── PADS ────────────────────────────────────────────────────────────────────

def fig_pads_wavelet(root: Path, out: Path):
    raw_txt = root / "physionet.org" / "files" / "parkinsons-disease-smartwatch" / "1.0.0" / \
        "movement" / "timeseries" / "113_PointFinger_RightWrist.txt"
    raw = np.loadtxt(raw_txt, delimiter=",")
    raw_sig = raw[:, 1]  # column 0 = Time, column 1 = Accelerometer X (dominant/right wrist here)

    w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")
    idx = labels.index[labels["subject_id"] == 113][0]
    pre_sig = w[idx, 0, :976]  # Accelerometer_X, PointFinger task segment (post gravity-correction, trim, z-score)

    scales = np.arange(1, 65)

    fig = plt.figure(figsize=(12, 7.5))
    fig.suptitle("PADS — Wavelet Scalogram, Raw vs. Preprocessed\n(Accelerometer_X, subject 113, PointFinger task — illustrative CWT, not a pipeline step)",
                 fontsize=11, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.4, wspace=0.3)

    trace_and_scalogram(fig, [gs[0, 0], gs[1, 0]], raw_sig, scales, WVF_BLUE,
                         "Before: raw .txt (1024 samples,\npre-gravity-correction)", "CWT of raw signal",
                         "raw sample index", "acceleration (g)")
    trace_and_scalogram(fig, [gs[0, 1], gs[1, 1]], pre_sig, scales, WVF_ORANGE,
                         "After: gravity-corrected, trimmed,\nz-scored (976 samples)", "CWT of preprocessed signal",
                         "timestep", "z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── Tappy ───────────────────────────────────────────────────────────────────

def fig_tappy_wavelet(root: Path, out: Path):
    w = np.load(root / "tappy_preprocessed" / "tappy_waveforms.npy")
    m = np.load(root / "tappy_preprocessed" / "tappy_masks.npy")
    labels = pd.read_csv(root / "tappy_preprocessed" / "tappy_labels.csv")
    lens = labels["SessionLength"].values

    idx = np.argmin(np.abs(lens - 400))  # a session long enough for a meaningful scalogram, unpadded
    n = int(m[idx].sum())
    raw_sig = w[idx, 0, :n] * TAPPY_STD[0] + TAPPY_MEAN[0]
    pre_sig = w[idx, 0, :n]
    scales = np.arange(1, 49)

    fig = plt.figure(figsize=(12, 7.5))
    fig.suptitle(f"Tappy — Wavelet Scalogram, Raw vs. Preprocessed\n(hold_time, one {n}-keystroke session — illustrative CWT, not a pipeline step)",
                 fontsize=11, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.4, wspace=0.3)

    trace_and_scalogram(fig, [gs[0, 0], gs[1, 0]], raw_sig, scales, WVF_BLUE,
                         "Before: raw hold-time (ms)", "CWT of raw signal",
                         "keystroke index", "hold_time (ms)")
    trace_and_scalogram(fig, [gs[0, 1], gs[1, 1]], pre_sig, scales, WVF_ORANGE,
                         "After: z-scored", "CWT of z-scored signal",
                         "keystroke index", "z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── mPower (not applicable) ────────────────────────────────────────────────

def fig_mpower_wavelet(out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    fig.suptitle("mPower — Wavelet Scalogram: Not Applicable", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.6,
            "mpower_features.npy is a flat [N, 41] table of pre-computed\n"
            "handcrafted statistics (mean/median/skew/... of tap intervals\n"
            "and finger-drift) — there is no time axis to decompose.\n"
            "A wavelet transform requires a sequence; mPower's raw source\n"
            "(tapFeatures.tsv) never shipped one.",
            ha="center", va="center", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.15,
            "No wavelet transform exists anywhere in the CBDL pipeline for any dataset —\n"
            "preprocess_pads.py's apply_wavelet_transform() is an explicit documented no-op / future seam.",
            ha="center", va="center", fontsize=8, color="#888888", style="italic", transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run_wavelet_figs(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_gaitrec_wavelet(root, figdir / "wavelet_gaitrec.png")
    print("Saved figures/wavelet_gaitrec.png")
    fig_pads_wavelet(root, figdir / "wavelet_pads.png")
    print("Saved figures/wavelet_pads.png")
    fig_tappy_wavelet(root, figdir / "wavelet_tappy.png")
    print("Saved figures/wavelet_tappy.png")
    fig_mpower_wavelet(figdir / "wavelet_mpower.png")
    print("Saved figures/wavelet_mpower.png")


def parse_args_wavelet_figs():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────
# Unified CLI — dispatches to whichever explainability/reporting/figure
# task was requested. Each keeps its own parse_args_<task>() with its own
# flags, exactly as in the original standalone scripts.
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    tasks = {
        "shap": (run_shap, parse_args_shap),
        "embedding-logs": (run_embedding_logs, parse_args_embedding_logs),
        "figures": (run_figures, parse_args_figures),
        "patient-report": (run_patient_report, parse_args_patient_report),
        "pipeline-shapes": (run_pipeline_shapes, parse_args_pipeline_shapes),
        "preprocessing-figs": (run_preprocessing_figs, parse_args_preprocessing_figs),
        "wavelet-channels": (run_wavelet_channels, parse_args_wavelet_channels),
        "wavelet-figs": (run_wavelet_figs, parse_args_wavelet_figs),
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
