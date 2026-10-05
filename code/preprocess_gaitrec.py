"""
Preprocessing pipeline for the GaitRec dataset (PhysioNet):
"A large-scale ground reaction force dataset of healthy and impaired gait".

Converts the pre-merged long-format GRF signal table plus GRF_metadata.csv
into a fixed-length, multichannel 1D waveform tensor per trial (already
resampled to 101 samples/trial upstream — this script pivots, filters, joins
labels, and normalizes; it does not resample), matching the storage
convention used by preprocess_tappy.py / preprocess_pads.py.

Known, deliberate data-quality exclusion (see gaitrec_data_card.md for the
full story): 2 of the 20 canonical GRF channels — GRF_F_V_PRO_left.csv
(entirely absent) and GRF_F_AP_PRO_left.csv (97% of rows missing) — were
corrupted in the source long-format CSV. All 18 remaining channels were
verified to share an identical (subject_id, session_id, trial_id) key set
(75,732 trials each) before this script was written, so dropping the 2 is a
clean column-drop, not a row-drop. Re-attempting to source those 2 channels
is future work; this run proceeds on 18/20 channels by explicit decision.

Dataset layout actually found on disk:
    Universal_Cleaned_Gait_Signals_v2.csv   long format: subject_id,
                                             session_id, trial_id,
                                             source_file, t_0..t_100
    GRF_metadata.csv                        one row per (SUBJECT_ID,
                                             SESSION_ID): CLASS_LABEL
                                             (HC/A/C/H/K), BODY_WEIGHT,
                                             BODY_MASS, AGE, SEX, native
                                             TRAIN/TEST columns (not used —
                                             this pipeline uses its own
                                             subject-disjoint split for
                                             convention parity with
                                             Tappy/PADS, see data card).

Usage:
    python preprocess_gaitrec.py \
        --long-csv "Universal_Cleaned_Gait_Signals_v2.csv" \
        --metadata-csv "GRF_metadata.csv" \
        --output-dir "gaitrec_preprocessed"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("preprocess_gaitrec")

# All 20 canonical GaitRec GRF channels, in a fixed, documented order.
# The 2 excluded ones are listed for transparency but never loaded.
EXCLUDED_CHANNELS = ["GRF_F_V_PRO_left.csv", "GRF_F_AP_PRO_left.csv"]
CHANNEL_ORDER = [
    "GRF_F_V_RAW_left.csv", "GRF_F_V_RAW_right.csv", "GRF_F_V_PRO_right.csv",
    "GRF_F_ML_RAW_left.csv", "GRF_F_ML_RAW_right.csv", "GRF_F_ML_PRO_left.csv", "GRF_F_ML_PRO_right.csv",
    "GRF_F_AP_RAW_left.csv", "GRF_F_AP_RAW_right.csv", "GRF_F_AP_PRO_right.csv",
    "GRF_COP_ML_RAW_left.csv", "GRF_COP_ML_RAW_right.csv", "GRF_COP_ML_PRO_left.csv", "GRF_COP_ML_PRO_right.csv",
    "GRF_COP_AP_RAW_left.csv", "GRF_COP_AP_RAW_right.csv", "GRF_COP_AP_PRO_right.csv", "GRF_COP_AP_PRO_left.csv",
]
T = 101  # fixed, already resampled upstream (t_0..t_100)


@dataclass
class Config:
    long_csv: Path
    metadata_csv: Path
    output_dir: Path
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


# ─── 1. Load + pivot the long-format signal table ──────────────────────────


def load_and_pivot(long_csv: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Streams the long-format CSV once, keeping only rows for CHANNEL_ORDER,
    and pivots into a [N_trials, 18, 101] float32 tensor plus a key
    DataFrame (subject_id, session_id, trial_id) aligned by row index.
    """
    t_cols = [f"t_{i}" for i in range(T)]
    usecols = ["subject_id", "session_id", "trial_id", "source_file"] + t_cols

    log.info("Streaming %s (this is a multi-GB file; reading in chunks)...", long_csv)
    chunks = []
    for chunk in pd.read_csv(long_csv, usecols=usecols, chunksize=500_000):
        chunk = chunk[chunk["source_file"].isin(CHANNEL_ORDER)]
        for c in ["subject_id", "session_id", "trial_id"]:
            chunk[c] = chunk[c].astype(np.int64)
        chunks.append(chunk)
    long_df = pd.concat(chunks, ignore_index=True)
    log.info("Loaded %d rows across %d channels.", len(long_df), long_df["source_file"].nunique())

    missing = set(CHANNEL_ORDER) - set(long_df["source_file"].unique())
    if missing:
        raise RuntimeError(f"Expected channels missing from long CSV: {missing}")

    key_cols = ["subject_id", "session_id", "trial_id"]
    keys = long_df.loc[long_df["source_file"] == CHANNEL_ORDER[0], key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)
    n_trials = len(keys)
    key_index = {tuple(row): i for i, row in enumerate(keys[key_cols].itertuples(index=False, name=None))}

    waveforms = np.zeros((n_trials, len(CHANNEL_ORDER), T), dtype=np.float32)
    for ch_idx, channel in enumerate(CHANNEL_ORDER):
        ch_df = long_df[long_df["source_file"] == channel]
        if len(ch_df) != n_trials:
            raise RuntimeError(f"Channel {channel} has {len(ch_df)} rows, expected {n_trials} (key misalignment).")
        row_idx = [key_index[t] for t in ch_df[key_cols].itertuples(index=False, name=None)]
        waveforms[row_idx, ch_idx, :] = ch_df[t_cols].to_numpy(dtype=np.float32)

    return waveforms, keys


# ─── 2. Join labels ─────────────────────────────────────────────────────────


def join_labels(keys: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    meta = meta.rename(columns={"SUBJECT_ID": "subject_id", "SESSION_ID": "session_id"})
    meta = meta[["subject_id", "session_id", "CLASS_LABEL", "CLASS_LABEL_DETAILED", "SEX", "AGE",
                 "BODY_WEIGHT", "BODY_MASS", "AFFECTED_SIDE"]]

    merged = keys.merge(meta, on=["subject_id", "session_id"], how="left")
    n_unmatched = merged["CLASS_LABEL"].isna().sum()
    if n_unmatched:
        log.warning("%d/%d trials have no matching metadata row and will be dropped.", n_unmatched, len(merged))
    return merged


# ─── 3. Subject-disjoint stratified split ──────────────────────────────────


def split_subjects(meta: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict[int, str]:
    """Splits SUBJECTS (not sessions/trials) into train/val/test, stratified by CLASS_LABEL."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)

    subject_label = meta.groupby("subject_id")["CLASS_LABEL"].agg(lambda s: s.mode().iat[0])
    assignment: dict[int, str] = {}
    for label, group in subject_label.groupby(subject_label):
        keys = list(group.index)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        for k in keys[:n_train]:
            assignment[k] = "train"
        for k in keys[n_train:n_train + n_val]:
            assignment[k] = "val"
        for k in keys[n_train + n_val:]:
            assignment[k] = "test"
    return assignment


# ─── 4. Normalization ───────────────────────────────────────────────────────


def fit_and_apply_normalization(waveforms: np.ndarray, is_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score, fit on train-split trials only (all timesteps real, no padding)."""
    means = waveforms[is_train].mean(axis=(0, 2))
    stds = waveforms[is_train].std(axis=(0, 2))
    stds[stds < 1e-8] = 1.0
    normalized = (waveforms - means[None, :, None]) / stds[None, :, None]
    return normalized.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


# ─── main ────────────────────────────────────────────────────────────────────


def run(cfg: Config) -> None:
    waveforms, keys = load_and_pivot(cfg.long_csv)
    meta = join_labels(keys, cfg.metadata_csv)

    valid = meta["CLASS_LABEL"].notna().to_numpy()
    if not valid.all():
        waveforms = waveforms[valid]
        meta = meta[valid].reset_index(drop=True)

    split_assignment = split_subjects(meta, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
    meta["split"] = meta["subject_id"].map(split_assignment)

    is_train = (meta["split"] == "train").to_numpy()
    waveforms, means, stds = fit_and_apply_normalization(waveforms, is_train)
    masks = np.ones((waveforms.shape[0], T), dtype=bool)  # all real, no padding

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "gaitrec_waveforms.npy", waveforms)
    np.save(cfg.output_dir / "gaitrec_masks.npy", masks)

    labels_csv = meta.rename(columns={
        "subject_id": "SubjectID", "session_id": "SessionID", "CLASS_LABEL": "ClassLabel",
        "CLASS_LABEL_DETAILED": "ClassLabelDetailed", "split": "Split",
    })[["SubjectID", "SessionID", "trial_id", "ClassLabel", "ClassLabelDetailed", "Split", "SEX", "AGE", "BODY_WEIGHT", "BODY_MASS"]]
    labels_csv.to_csv(cfg.output_dir / "gaitrec_labels.csv", index=False)

    split_counts = meta.groupby("split").agg(trials=("subject_id", "count"), subjects=("subject_id", "nunique"))
    class_balance = meta.groupby(["split", "CLASS_LABEL"]).size().unstack(fill_value=0)

    card_lines = [
        "# GaitRec Dataset — Preprocessing Data Card",
        "",
        "## Source",
        "PhysioNet: \"A large-scale ground reaction force dataset of healthy and "
        "impaired gait\" (GaitRec). Input here is a pre-merged long-format table "
        "(`Universal_Cleaned_Gait_Signals_v2.csv`) already resampled to 101 "
        "samples/trial, plus `GRF_metadata.csv` for labels.",
        "",
        "## Known data-quality exclusion",
        "2 of the 20 canonical channels were corrupted in the source long CSV and "
        "are excluded by explicit decision:",
        f"- `{EXCLUDED_CHANNELS[0]}` — entirely absent (0/75,732 expected rows).",
        f"- `{EXCLUDED_CHANNELS[1]}` — 97% missing (2,472/75,732 expected rows present, "
        "and those present did not reconstruct cleanly against the known-good "
        "RAW-normalization relationship, so the partial data was not trusted either).",
        "All 18 remaining channels were verified byte-for-byte to share an identical "
        "(subject_id, session_id, trial_id) key set (75,732 trials) before this "
        "pipeline was written — this is a clean column drop, not a row drop. State "
        "this explicitly in the paper's Limitations per the CBDL development plan.",
        "",
        f"## Channels kept (in gaitrec_waveforms.npy, shape [N, {len(CHANNEL_ORDER)}, {T}], float32, z-scored)",
        *[f"{i}. `{c}`" for i, c in enumerate(CHANNEL_ORDER)],
        "",
        "## Normalization",
        "Z-score per channel, fit on TRAIN-split trials only (no padding — every "
        "trial already has all 101 real timesteps).",
        f"mean = {means.tolist()}",
        f"std  = {stds.tolist()}",
        "",
        "## Label",
        "`CLASS_LABEL` from GRF_metadata.csv: HC (healthy control), A (ankle), "
        "K (knee), H (hip), C (calcaneus/other) pathology group. Used as the "
        "secondary gait-side clinical-grounding check per the CBDL plan (PADS "
        "remains primary).",
        "",
        "## Split (subject-disjoint, stratified by CLASS_LABEL)",
        f"Target fractions: train={cfg.train_frac}, val={cfg.val_frac}, test={cfg.test_frac}, seed={cfg.seed}. "
        "Note: GRF_metadata.csv ships its own native TRAIN/TEST columns — not used here, "
        "in favor of the same subject-disjoint convention applied to Tappy/PADS/mPower.",
        "",
        "```",
        split_counts.to_string(),
        "```",
        "",
        "### Class balance per split (trial count)",
        "```",
        class_balance.to_string(),
        "```",
    ]
    (cfg.output_dir / "gaitrec_data_card.md").write_text("\n".join(card_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("GAITREC PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Trials: {len(meta)}   Subjects: {meta['subject_id'].nunique()}   Channels: {len(CHANNEL_ORDER)}/20")
    print("\nTrials per split:")
    print(split_counts.to_string())
    print("\nClass balance per split:")
    print(class_balance.to_string())
    print("=" * 60)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--long-csv", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return Config(
        long_csv=args.long_csv, metadata_csv=args.metadata_csv, output_dir=args.output_dir,
        train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
