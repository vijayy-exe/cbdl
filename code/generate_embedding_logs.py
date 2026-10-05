"""
Generates log files containing the actual embedding vectors produced by the
CBDL pipeline at two points:

  1. Phase 2 (per-modality encoders, `model.py`) — the raw encoder outputs
     (waveform-trunk / gait / mPower-branch embeddings) and, where the
     architecture actually forms one, the fused Finger Embedding.
  2. Phase 3 (Cross-Body Dependency Module, `cross_body_module.py`) — the
     lag-fused embedding and per-lag attention weights for each population-
     level tier-matched (PADS finger, GaitRec gait) pair, using the same
     tier-matching procedure as `train_phase3_cbdm.py`'s eval loop.

Loads the already-trained checkpoints (`phase2_checkpoint.pt`,
`phase3_checkpoint.pt` — the "attention" variant) rather than retraining
anything. Writes one CSV per source (metadata columns + emb_0..emb_{D-1})
under `embedding_logs/`, plus a text run log summarizing shapes and a value
preview for sanity-checking.

Usage:
    python generate_embedding_logs.py --data-root "."
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cross_body_module import (
    LaggedCrossAttention,
    build_positive_indices,
    gaitrec_label_to_tier,
    pads_label_to_tier,
)
from model import CBDLPhase2Model

K_WINDOWS = 8
BATCH_SIZE = 512


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def run(args) -> None:
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
