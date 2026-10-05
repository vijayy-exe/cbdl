"""
CBDL Phase 1 — Data Preprocessing (consolidated research-release version).

Merges the four originally-separate preprocessing pipelines
(preprocess_gaitrec.py, preprocess_mpower.py, preprocess_pads.py,
preprocess_tappy.py from ../code/) into one file, one per dataset section
below, unchanged in logic — only identifiers that collided across the four
original modules (Config, run, parse_args, log, split_subjects,
fit_normalization_stats, apply_normalization) were namespaced per dataset
(e.g. GaitrecConfig, run_gaitrec, parse_args_gaitrec) so they can coexist in
one namespace. See ../code/ for the original, unmerged scripts and
../CBDL_PROJECT_DOCUMENTATION.md §2 for the full dataset writeup.

Usage (one dataset per invocation, same flags as the original scripts):
    python phase1_preprocessing.py gaitrec --long-csv ... --metadata-csv ... --output-dir ...
    python phase1_preprocessing.py mpower  --source-csv ... --output-dir ...
    python phase1_preprocessing.py pads    --pads-root ... --output-dir ...
    python phase1_preprocessing.py tappy   --dataset-root ... --output-dir ...
Run `python phase1_preprocessing.py <dataset> --help` for that dataset's full flag list.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import multiprocessing
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log_gaitrec = logging.getLogger("preprocess_gaitrec")
log_mpower = logging.getLogger("preprocess_mpower")
log_pads = logging.getLogger("preprocess_pads")
log_tappy = logging.getLogger("preprocess_tappy")


# ============================================================================
# GAITREC
# ============================================================================

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
class GaitrecConfig:
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

    log_gaitrec.info("Streaming %s (this is a multi-GB file; reading in chunks)...", long_csv)
    chunks = []
    for chunk in pd.read_csv(long_csv, usecols=usecols, chunksize=500_000):
        chunk = chunk[chunk["source_file"].isin(CHANNEL_ORDER)]
        for c in ["subject_id", "session_id", "trial_id"]:
            chunk[c] = chunk[c].astype(np.int64)
        chunks.append(chunk)
    long_df = pd.concat(chunks, ignore_index=True)
    log_gaitrec.info("Loaded %d rows across %d channels.", len(long_df), long_df["source_file"].nunique())

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
        log_gaitrec.warning("%d/%d trials have no matching metadata row and will be dropped.", n_unmatched, len(merged))
    return merged


# ─── 3. Subject-disjoint stratified split ──────────────────────────────────


def split_subjects_gaitrec(meta: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict[int, str]:
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


def run_gaitrec(cfg: GaitrecConfig) -> None:
    waveforms, keys = load_and_pivot(cfg.long_csv)
    meta = join_labels(keys, cfg.metadata_csv)

    valid = meta["CLASS_LABEL"].notna().to_numpy()
    if not valid.all():
        waveforms = waveforms[valid]
        meta = meta[valid].reset_index(drop=True)

    split_assignment = split_subjects_gaitrec(meta, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
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


def parse_args_gaitrec() -> GaitrecConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--long-csv", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return GaitrecConfig(
        long_csv=args.long_csv, metadata_csv=args.metadata_csv, output_dir=args.output_dir,
        train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )

# ============================================================================
# MPOWER
# ============================================================================

FEATURE_COLUMNS = [
    "meanTapInter", "medianTapInter", "iqrTapInter", "minTapInter", "maxTapInter",
    "skewTapInter", "kurTapInter", "sdTapInter", "madTapInter", "cvTapInter",
    "rangeTapInter", "tkeoTapInter", "ar1TapInter", "ar2TapInter",
    "fatigue10TapInter", "fatigue25TapInter", "fatigue50TapInter",
    "meanDriftLeft", "medianDriftLeft", "iqrDriftLeft", "minDriftLeft", "maxDriftLeft",
    "skewDriftLeft", "kurDriftLeft", "sdDriftLeft", "madDriftLeft", "cvDriftLeft", "rangeDriftLeft",
    "meanDriftRight", "medianDriftRight", "iqrDriftRight", "minDriftRight", "maxDriftRight",
    "skewDriftRight", "kurDriftRight", "sdDriftRight", "madDriftRight", "cvDriftRight", "rangeDriftRight",
    "numberTaps", "buttonNoneFreq",
]


@dataclass
class MpowerConfig:
    source_csv: Path
    output_dir: Path
    min_number_taps: int = 10
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


def split_subjects_mpower(subjects: list[str], train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    """Plain subject-disjoint split (no stratification — mPower has no usable label here)."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    keys = list(subjects)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    assignment = {}
    for k in keys[:n_train]:
        assignment[k] = "train"
    for k in keys[n_train:n_train + n_val]:
        assignment[k] = "val"
    for k in keys[n_train + n_val:]:
        assignment[k] = "test"
    return assignment


def run_mpower(cfg: MpowerConfig) -> None:
    df = pd.read_csv(cfg.source_csv)
    n_raw = len(df)
    log_mpower.info("Loaded %d raw records, %d subjects (healthCode).", n_raw, df["healthCode"].nunique())

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Expected feature columns missing from source: {missing_cols}")

    # ── Cleaning ──
    n_nan_dropped = df[FEATURE_COLUMNS].isna().any(axis=1).sum()
    df = df[~df[FEATURE_COLUMNS].isna().any(axis=1)].copy()

    n_low_taps_dropped = (df["numberTaps"] < cfg.min_number_taps).sum()
    df = df[df["numberTaps"] >= cfg.min_number_taps].copy()

    df = df.reset_index(drop=True)
    log_mpower.info(
        "After cleaning: %d records (%d dropped for missing features, %d dropped for numberTaps < %d).",
        len(df), n_nan_dropped, n_low_taps_dropped, cfg.min_number_taps,
    )

    # ── Subject-disjoint split ──
    subjects = sorted(df["healthCode"].unique().tolist())
    split_assignment = split_subjects_mpower(subjects, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
    df["Split"] = df["healthCode"].map(split_assignment)

    # ── Normalize (z-score per feature, fit on train only) ──
    is_train = (df["Split"] == "train").to_numpy()
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    means = features[is_train].mean(axis=0)
    stds = features[is_train].std(axis=0)
    stds[stds < 1e-8] = 1.0
    normalized = ((features - means) / stds).astype(np.float32)

    # ── Save ──
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "mpower_features.npy", normalized)

    labels_csv = df[["recordId", "healthCode", "Split", "PD", "numberTaps"]].copy()
    labels_csv.to_csv(cfg.output_dir / "mpower_labels.csv", index=False)

    with open(cfg.output_dir / "mpower_feature_names.txt", "w") as f:
        f.write("\n".join(FEATURE_COLUMNS))

    split_counts = df.groupby("Split").agg(records=("healthCode", "count"), subjects=("healthCode", "nunique"))

    card_lines = [
        "# mPower Dataset — Preprocessing Data Card",
        "",
        "## Source",
        f"`{cfg.source_csv.name}` (Synapse/Sage Bionetworks mPower study) — chosen over "
        "`tapFeatures.tsv` specifically because it carries `healthCode` (a real subject "
        "id), enabling a subject-disjoint split; `tapFeatures.tsv` is keyed only by "
        "`filehandle` with no subject id at all. See module docstring for the full "
        "reasoning.",
        "",
        "## No usable diagnosis label",
        "Every one of this file's 104 subjects has `PD == True` (it is drawn from a "
        "medication-timing study enrolling only diagnosed PD patients — no healthy-"
        "control arm). `PD` is carried through to `mpower_labels.csv` for reference "
        "but is NOT a usable binary classification label (zero negative-class "
        "examples). Per the CBDL plan (Phase 0.1), this is acceptable: mPower's "
        "architectural role is a handcrafted-feature MLP branch fused into the Finger "
        "Embedding; PADS remains the primary clinical-grounding label source for the "
        "fused embedding, not mPower itself.",
        "",
        f"## Features kept (in mpower_features.npy, shape [N, {len(FEATURE_COLUMNS)}], float32, z-scored)",
        "Pre-computed handcrafted tapping-session statistics — NOT a waveform, no time "
        "axis (see module docstring). Full list in `mpower_feature_names.txt`.",
        "`" + "`, `".join(FEATURE_COLUMNS) + "`",
        "",
        "## Cleaning",
        f"- Records with any missing value across the {len(FEATURE_COLUMNS)} feature "
        f"columns dropped: {n_nan_dropped} (of {n_raw} raw records).",
        f"- Records with `numberTaps` < {cfg.min_number_taps} dropped as degenerate/"
        f"invalid tapping sessions: {n_low_taps_dropped}.",
        f"- Records remaining: {len(df)}.",
        "",
        "## Normalization",
        "Z-score per feature column, fit on TRAIN-split records only.",
        f"mean = {means.tolist()}",
        f"std  = {stds.tolist()}",
        "",
        "## Split (subject-disjoint by healthCode — no stratification, single-class label)",
        f"Target fractions: train={cfg.train_frac}, val={cfg.val_frac}, test={cfg.test_frac}, seed={cfg.seed}",
        "```",
        split_counts.to_string(),
        "```",
    ]
    (cfg.output_dir / "mpower_data_card.md").write_text("\n".join(card_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("MPOWER PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Records: {len(df)}   Subjects: {df['healthCode'].nunique()}   Features: {len(FEATURE_COLUMNS)}")
    print("\nRecords/subjects per split:")
    print(split_counts.to_string())
    print("\nNote: PD label is single-class (all True) — unusable for classification, kept for reference only.")
    print("=" * 60)


def parse_args_mpower() -> MpowerConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-number-taps", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return MpowerConfig(
        source_csv=args.source_csv, output_dir=args.output_dir, min_number_taps=args.min_number_taps,
        train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )

# ============================================================================
# PADS
# ============================================================================

# Channel order within one task-block, matching PADS's own sorted order
# (utils/... is not reused for this constant since it's just a naming
# convention, but it matches run_preprocessing.py's per-wrist ordering:
# Accelerometer X,Y,Z then Gyroscope X,Y,Z, with Time dropped).
SENSOR_AXES = [
    ("Accelerometer", "X"),
    ("Accelerometer", "Y"),
    ("Accelerometer", "Z"),
    ("Gyroscope", "X"),
    ("Gyroscope", "Y"),
    ("Gyroscope", "Z"),
]

# Official PADS task order used by plot_example_pd_signal_processed.py to
# reconstruct channel names for the ALREADY-SHIPPED .bin files (Step 2
# verification only — PointFinger/LiftHold/TouchIndex are absent here
# because run_preprocessing.py strips them before saving).
OFFICIAL_BIN_TASKS = [
    "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold",
    "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1",
    "Entrainment2",
]
OFFICIAL_BIN_WRISTS = ["LeftWrist", "RightWrist"]
OFFICIAL_BIN_SENSORS = ["Acceleration", "Rotation"]
OFFICIAL_BIN_AXES = ["X", "Y", "Z"]

LABEL_NAMES = {0: "Healthy", 1: "Parkinson's", 2: "Other movement disorder"}


# ─── PadsConfig ──────────────────────────────────────────────────────────────


@dataclass
class PadsConfig:
    pads_root: Path  # .../parkinsons-disease-smartwatch/1.0.0
    output_dir: Path
    tasks: list[str] = field(default_factory=lambda: ["PointFinger", "TouchIndex", "TouchNose"])
    wrist_mode: str = "dominant"  # "dominant" | "LeftWrist" | "RightWrist"
    vlambda: float = 50.0  # PADS's own l1_trend_filter regularization (official default)
    trim_samples: int = 48  # PADS's own "drop first 0.5s vibration notification" trim
    task_length_percentile: float = 100.0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    n_jobs: int = 1
    limit: Optional[int] = None

    @property
    def preprocessed_dir(self) -> Path:
        return self.pads_root / "preprocessed"

    @property
    def file_list_csv(self) -> Path:
        return self.preprocessed_dir / "file_list.csv"

    @property
    def bin_dir(self) -> Path:
        return self.preprocessed_dir / "movement"

    @property
    def raw_movement_dir(self) -> Path:
        return self.pads_root / "movement"

    @property
    def scripts_dir(self) -> Path:
        return self.pads_root / "scripts"


def _pads_utils(cfg: PadsConfig):
    """
    Imports PADS's own scripts/utils package (data_handling, l1_trend_filter)
    by path, so this script reuses PADS's own loading/gravity-correction
    logic instead of reimplementing it (see module docstring, point 1).
    """
    scripts_dir = str(cfg.scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from utils.data_handling import load_all_files, get_data_from_observation  # noqa: E402
    from utils.l1_trend_filter import l1_trend_filter  # noqa: E402

    return load_all_files, get_data_from_observation, l1_trend_filter


# ─── Step 1: verify official preprocessing already ran ─────────────────────


def verify_official_preprocessing(cfg: PadsConfig) -> None:
    """
    Confirms preprocessed/movement/*.bin and preprocessed/file_list.csv
    already exist, as PADS's own run_preprocessing.py would have produced.
    Does NOT reimplement PADS's gravity-correction/offset-removal logic —
    if the official outputs are missing, tells the user to run
    scripts/run_preprocessing.py themselves and stops.
    """
    if not cfg.file_list_csv.exists():
        raise FileNotFoundError(
            f"{cfg.file_list_csv} not found. Run PADS's own "
            f"scripts/run_preprocessing.py (from within {cfg.scripts_dir}) first — "
            f"this script does not reimplement that logic."
        )
    bin_files = sorted(cfg.bin_dir.glob("*_ml.bin"))
    if not bin_files:
        raise FileNotFoundError(
            f"No *_ml.bin files found under {cfg.bin_dir}. Run PADS's own "
            f"scripts/run_preprocessing.py first."
        )
    log_pads.info(
        "Official PADS preprocessing already ran: %s exists, %d .bin files found under %s.",
        cfg.file_list_csv, len(bin_files), cfg.bin_dir,
    )


# ─── Step 2: reconstruct the channel order of the ALREADY-SHIPPED .bin files ──


def reconstruct_official_channel_order() -> list[str]:
    """
    Reproduces the exact channel list construction from
    scripts/plot_example_pd_signal_processed.py: task -> wrist -> sensor ->
    axis, e.g. "TouchNose_Acceleration_RightWrist_X". This is the channel
    order of preprocessed/movement/*.bin as shipped (11 tasks, PointFinger/
    LiftHold/TouchIndex already removed by PADS's own run_preprocessing.py).
    """
    channels = []
    for task in OFFICIAL_BIN_TASKS:
        for wrist in OFFICIAL_BIN_WRISTS:
            for sensor in OFFICIAL_BIN_SENSORS:
                for axis in OFFICIAL_BIN_AXES:
                    channels.append(f"{task}_{sensor}_{wrist}_{axis}")
    return channels


def load_bin_file(path: Path, n_channels: int) -> np.ndarray:
    """Loads and reshapes one subject's *_ml.bin file to [n_channels, T]."""
    return np.fromfile(path, dtype=np.float32).reshape((n_channels, -1))


def verify_bin_shape(cfg: PadsConfig) -> None:
    """
    Step 2 sanity check: load one subject's .bin file, reshape it using the
    channel order above, and print the resulting shape and channel count —
    verifying it matches before processing (don't assume).
    """
    channels = reconstruct_official_channel_order()
    example_bin = sorted(cfg.bin_dir.glob("*_ml.bin"))[0]
    n_floats = example_bin.stat().st_size // 4
    if n_floats % len(channels) != 0:
        raise AssertionError(
            f"{example_bin.name}: {n_floats} floats is not divisible by the "
            f"reconstructed channel count {len(channels)} — channel-order "
            f"reconstruction does not match this .bin layout, stopping "
            f"rather than guessing."
        )
    data = load_bin_file(example_bin, len(channels))
    log_pads.info(
        "Step 2 verification (%s): reconstructed %d channels, .bin reshapes to %s "
        "(%d channels x %d timesteps). Matches expected channel count: %s",
        example_bin.name, len(channels), data.shape, data.shape[0], data.shape[1],
        data.shape[0] == len(channels),
    )
    assert data.shape[0] == len(channels)


# ─── Step 3: task/wrist selection ───────────────────────────────────────────


def select_dominant_wrist(handedness: str) -> str:
    """
    Maps file_list.csv's `handedness` field to the PADS device_location
    naming used in the raw session metadata. Falls back to RightWrist (the
    cohort majority, 437/469) with a warning for any unexpected value —
    none are expected since handedness has no missing values in this
    release, but subjects are not dropped over a metadata quirk here.
    """
    value = str(handedness).strip().lower()
    if value == "left":
        return "LeftWrist"
    if value == "right":
        return "RightWrist"
    log_pads.warning("Unexpected handedness value %r; defaulting to RightWrist.", handedness)
    return "RightWrist"


def resolve_wrist(cfg: PadsConfig, handedness: str) -> str:
    if cfg.wrist_mode == "dominant":
        return select_dominant_wrist(handedness)
    if cfg.wrist_mode in ("LeftWrist", "RightWrist"):
        return cfg.wrist_mode
    raise ValueError(f"Unknown wrist_mode {cfg.wrist_mode!r}")


# ─── Step 4a: per-subject raw extraction + PADS's own gravity correction ───


def _extract_one_subject(
    scripts_dir: str,
    subject_meta: pd.DataFrame,
    raw_movement_dir: str,
    tasks: list[str],
    wrist: str,
    vlambda: float,
    trim_samples: int,
) -> dict:
    """
    Runs in a worker process. Loads the 3 target-task raw recordings for one
    subject's dominant wrist, applies PADS's own l1_trend_filter to the
    Accelerometer rows only (matching run_preprocessing.py's process_mask),
    trims the first `trim_samples` samples (matching run_preprocessing.py's
    vibration-notification trim), and concatenates the tasks along time.

    Returns a dict with either:
      {"subject_id", "ok": True, "array" [6, sum(task_lengths)], "task_lengths": {...}}
    or:
      {"subject_id", "ok": False, "reason": "..."}
    """
    subject_id = subject_meta["subject_id"].iloc[0]
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from utils.data_handling import get_data_from_observation
    from utils.l1_trend_filter import l1_trend_filter

    try:
        task_frames = []
        for task in tasks:
            rows = subject_meta[
                (subject_meta["record_name"] == task) & (subject_meta["device_location"] == wrist)
            ]
            if len(rows) != 1:
                return {
                    "subject_id": subject_id, "ok": False,
                    "reason": f"expected exactly 1 raw recording for task={task} wrist={wrist}, found {len(rows)}",
                }
            task_frames.append(rows)
        ordered_meta = pd.concat(task_frames, axis=0).reset_index(drop=True)

        data, channel_names = get_data_from_observation(raw_movement_dir, ordered_meta)
        # data: [len(tasks) * 7, L] in task-major order, per-task rows =
        # [Time, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z]
        keep_mask = ~pd.Series(channel_names).str.contains("_Time$", regex=True)
        data = data[keep_mask.to_numpy()]
        channel_names = list(np.array(channel_names)[keep_mask.to_numpy()])

        n_tasks = len(tasks)
        n_ch_per_task = len(SENSOR_AXES)  # 6
        if data.shape[0] != n_tasks * n_ch_per_task:
            return {
                "subject_id": subject_id, "ok": False,
                "reason": f"expected {n_tasks * n_ch_per_task} rows after dropping Time, got {data.shape[0]}",
            }

        task_lengths = {}
        for t_idx, task in enumerate(tasks):
            block = data[t_idx * n_ch_per_task:(t_idx + 1) * n_ch_per_task]
            task_lengths[task] = block.shape[1]

        process_mask = pd.Series(channel_names).str.contains("Accelerometer")
        with contextlib.redirect_stdout(io.StringIO()):
            data[process_mask.to_numpy()] = np.apply_along_axis(
                lambda x: x - l1_trend_filter(x, vlambda=vlambda, verbose=False),
                1, data[process_mask.to_numpy()],
            )

        data = data[:, trim_samples:]

        return {
            "subject_id": subject_id, "ok": True,
            "array": data.reshape(n_tasks, n_ch_per_task, -1),  # [n_tasks, 6, T_task]
            "task_lengths": task_lengths,
        }
    except Exception as exc:  # noqa: BLE001 — reported per-subject, not fatal to the run
        return {"subject_id": subject_id, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def report_raw_task_length_distribution(all_subject_meta: list[pd.DataFrame], tasks: list[str]) -> dict:
    """
    Step 4: prints the raw (pre-trim) per-task recording-length distribution
    across the whole cohort, from metadata alone (cheap — no gravity
    correction needed yet), so the common-length choice below is justified
    rather than arbitrary.
    """
    lengths_by_task: dict[str, list[int]] = {t: [] for t in tasks}
    for meta in all_subject_meta:
        for task in tasks:
            rows = meta.loc[meta["record_name"] == task, "rows"]
            if len(rows):
                lengths_by_task[task].append(int(rows.iloc[0]))

    summary = {}
    for task, lengths in lengths_by_task.items():
        arr = np.asarray(lengths)
        summary[task] = {
            "n": int(arr.size), "min": int(arr.min()), "max": int(arr.max()),
            "mean": float(arr.mean()), "n_unique_values": int(np.unique(arr).size),
        }
        log_pads.info("Raw length distribution for task=%s: %s", task, summary[task])
    return summary


def choose_common_task_length(length_summary: dict, percentile: float) -> int:
    """
    Picks one common raw recording length shared by all three tasks (PADS's
    protocol records each task for a fixed duration, so this is expected to
    equal every task's own min==max, not an arbitrary compromise — verified,
    not assumed, via report_raw_task_length_distribution above).
    """
    all_mins = [s["min"] for s in length_summary.values()]
    all_maxs = [s["max"] for s in length_summary.values()]
    if len(set(all_mins)) == 1 and all_mins == all_maxs:
        return all_mins[0]
    # Fallback for a cohort that turns out not to be perfectly fixed-length:
    pooled = np.concatenate([[s["min"]] * 1 for s in length_summary.values()])
    return int(round(float(np.percentile(pooled, percentile))))


# ─── Step 4b: pad/truncate + concatenate tasks into one per-subject sequence ─


def pad_or_truncate_task_block(block: np.ndarray, t_task: int) -> tuple[np.ndarray, np.ndarray]:
    """
    block: [6, L] float32 (post gravity-correction, post-trim, single task)
    Returns (fixed [6, t_task] float32, mask [t_task] bool).
    """
    _, length = block.shape
    if length >= t_task:
        return block[:, :t_task].astype(np.float32), np.ones(t_task, dtype=bool)
    fixed = np.zeros((block.shape[0], t_task), dtype=np.float32)
    fixed[:, :length] = block
    mask = np.zeros(t_task, dtype=bool)
    mask[:length] = True
    return fixed, mask


def concatenate_tasks(array: np.ndarray, t_task: int) -> tuple[np.ndarray, np.ndarray]:
    """
    array: [n_tasks, 6, L] for one subject (post gravity-correction/trim).
    Returns (sequence [6, n_tasks * t_task], mask [n_tasks * t_task]) with
    the tasks concatenated along time in the fixed order they were given.
    """
    fixed_blocks, mask_blocks = [], []
    for t_idx in range(array.shape[0]):
        fixed, mask = pad_or_truncate_task_block(array[t_idx], t_task)
        fixed_blocks.append(fixed)
        mask_blocks.append(mask)
    sequence = np.concatenate(fixed_blocks, axis=1)  # [6, n_tasks * t_task]
    mask = np.concatenate(mask_blocks, axis=0)  # [n_tasks * t_task]
    return sequence, mask


# ─── Orchestration: Steps 3-4 across the whole cohort ───────────────────────


def build_all_subjects(cfg: PadsConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """
    Runs Steps 3-4 for every subject in file_list.csv: resolve dominant
    wrist, extract+gravity-correct+trim the 3 target tasks from raw data
    (Step 4a, optionally parallel across subjects via cfg.n_jobs), then pad/
    truncate+concatenate into one fixed-length sequence per subject
    (Step 4b).

    Returns (waveforms [N, 6, T], masks [N, T], subject_table, drop_stats).
    subject_table has one row per subject actually produced: subject_id,
    wrist, task_lengths (dict), any per-task truncation/padding applied.
    """
    load_all_files, _, _ = _pads_utils(cfg)
    all_subject_meta_full = load_all_files(str(cfg.raw_movement_dir) + "/")
    all_subject_meta = all_subject_meta_full if cfg.limit is None else all_subject_meta_full[: cfg.limit]

    file_list = pd.read_csv(cfg.file_list_csv)
    file_list["subject_id"] = file_list["id"].apply(lambda i: f"{int(i):03d}")
    handedness_lookup = file_list.set_index("subject_id")["handedness"].to_dict()

    length_summary = report_raw_task_length_distribution(all_subject_meta, cfg.tasks)
    t_task = choose_common_task_length(length_summary, cfg.task_length_percentile)
    log_pads.info(
        "Chosen common per-task length = %d samples (percentile=%.0f). "
        "Post gravity-correction trim of %d samples -> %d samples/task, "
        "%d tasks concatenated -> final T = %d.",
        t_task, cfg.task_length_percentile, cfg.trim_samples,
        t_task - cfg.trim_samples, len(cfg.tasks),
        (t_task - cfg.trim_samples) * len(cfg.tasks),
    )
    t_task_trimmed = t_task - cfg.trim_samples

    jobs = []
    for meta in all_subject_meta:
        subject_id = meta["subject_id"].iloc[0]
        handedness = handedness_lookup.get(subject_id)
        if handedness is None:
            continue  # not in file_list.csv; reported as a drop below
        wrist = resolve_wrist(cfg, handedness)
        jobs.append((str(cfg.scripts_dir), meta, str(cfg.raw_movement_dir) + "/", cfg.tasks, wrist, cfg.vlambda, cfg.trim_samples))

    log_pads.info("Extracting %d subjects (n_jobs=%d)...", len(jobs), cfg.n_jobs)
    if cfg.n_jobs > 1:
        with multiprocessing.Pool(cfg.n_jobs) as pool:
            results = pool.starmap(_extract_one_subject, jobs)
    else:
        results = [_extract_one_subject(*j) for j in jobs]

    # Computed against the FULL raw cohort (not the --limit slice used for
    # `jobs` below), so a test run with --limit never misreports subjects
    # outside its slice as "missing".
    subjects_in_file_list = set(file_list["subject_id"])
    subjects_in_raw = {m["subject_id"].iloc[0] for m in all_subject_meta_full}
    missing_from_raw = [] if cfg.limit is not None else sorted(subjects_in_file_list - subjects_in_raw)

    waveforms, masks, rows = [], [], []
    drop_reasons: list[str] = []
    for res in results:
        if not res["ok"]:
            drop_reasons.append(f"{res['subject_id']}: {res['reason']}")
            continue
        sequence, mask = concatenate_tasks(res["array"], t_task_trimmed)
        waveforms.append(sequence)
        masks.append(mask)
        rows.append({"subject_id": res["subject_id"], "task_lengths": res["task_lengths"]})

    for subject_id in missing_from_raw:
        drop_reasons.append(f"{subject_id}: no raw movement/*.json observation found")

    waveforms_arr = np.stack(waveforms, axis=0).astype(np.float32) if waveforms else np.zeros((0, 6, 0), np.float32)
    masks_arr = np.stack(masks, axis=0) if masks else np.zeros((0, 0), bool)
    subject_table = pd.DataFrame(rows)

    drop_stats = {
        "n_attempted": len(jobs) + len(missing_from_raw),
        "n_succeeded": len(rows),
        "n_dropped": len(drop_reasons),
        "drop_reasons": drop_reasons,
        "t_task_raw": t_task,
        "t_task_trimmed": t_task_trimmed,
        "T": t_task_trimmed * len(cfg.tasks),
        "length_summary": length_summary,
    }
    return waveforms_arr, masks_arr, subject_table, drop_stats


# ─── Step 5: attach labels ──────────────────────────────────────────────────


def merge_labels(cfg: PadsConfig, subject_table: pd.DataFrame) -> pd.DataFrame:
    """Merges label/condition from file_list.csv by subject id; reports class balance."""
    file_list = pd.read_csv(cfg.file_list_csv)
    file_list["subject_id"] = file_list["id"].apply(lambda i: f"{int(i):03d}")
    merged = subject_table.merge(
        file_list[["subject_id", "label", "condition"]], on="subject_id", how="left"
    )
    if merged["label"].isna().any():
        missing = merged.loc[merged["label"].isna(), "subject_id"].tolist()
        raise AssertionError(f"Subjects with no label after merge (should be impossible): {missing}")
    merged["label"] = merged["label"].astype(int)
    merged["parkinsons"] = merged["label"] == 1

    balance = merged["label"].map(LABEL_NAMES).value_counts()
    log_pads.info("Class balance after label merge:\n%s", balance.to_string())
    return merged


# ─── Step 6: wavelet transform (placeholder — see module docstring) ────────


def apply_wavelet_transform(
    waveforms: np.ndarray,
    wavelet: Optional[str] = None,
    level: Optional[int] = None,
) -> np.ndarray:
    """
    SEAM FOR A FUTURE WAVELET STAGE.

    No wavelet transform exists anywhere in this project today — the real
    Tappy pipeline (preprocess_tappy.py) outputs raw z-scored waveforms, not
    wavelet coefficients (see module docstring, point 2). This function is
    therefore a documented no-op so pads_waveforms.npy matches the Tappy
    pipeline's ACTUAL current output shape ([N, C, T] z-scored floats).

    If a wavelet-coefficient schema is later confirmed for both datasets,
    fill this in (e.g. via pywavelets: `pywt.wavedec(x, wavelet, level=level)`
    per-channel, per-subject) and change only this function plus the column
    names written in save_outputs — no other step needs to change.
    """
    if wavelet is not None or level is not None:
        raise NotImplementedError(
            "Wavelet parameters were provided but no wavelet transform is "
            "implemented yet — see this function's docstring."
        )
    return waveforms


# ─── Normalization (mirrors preprocess_tappy.py's fit/apply) ───────────────


def fit_normalization_stats_pads(waveforms: np.ndarray, masks: np.ndarray, is_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_waveforms = waveforms[is_train]
    train_masks = masks[is_train]
    n_channels = waveforms.shape[1]
    means = np.zeros(n_channels, dtype=np.float64)
    stds = np.ones(n_channels, dtype=np.float64)
    for c in range(n_channels):
        values = train_waveforms[:, c, :][train_masks]
        if values.size == 0:
            log_pads.warning("Channel %d has no valid training timesteps; leaving mean=0, std=1.", c)
            continue
        means[c] = values.mean()
        std = values.std()
        stds[c] = std if std > 1e-8 else 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def apply_normalization_pads(waveforms: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    normalized = waveforms.copy()
    for c in range(waveforms.shape[1]):
        normalized[:, c, :] = (waveforms[:, c, :] - means[c]) / stds[c]
    return normalized.astype(np.float32)


# ─── Subject-disjoint stratified split (mirrors preprocess_tappy.py) ───────


def split_subjects_pads(labels: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    """Stratifies by the 3-class `label` (Healthy/PD/Other), analogous to Tappy's boolean-Parkinsons stratification."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    assignment: dict[str, str] = {}
    for label_value in sorted(labels["label"].unique()):
        keys = labels.loc[labels["label"] == label_value, "subject_id"].tolist()
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


# ─── Step 7: save + report ──────────────────────────────────────────────────


def export_schema(
    cfg: PadsConfig,
    waveforms: np.ndarray,
    masks: np.ndarray,
    labels: pd.DataFrame,
    drop_stats: dict,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "pads_waveforms.npy", waveforms)
    np.save(cfg.output_dir / "pads_masks.npy", masks)

    labels_csv = labels[["subject_id", "label", "parkinsons", "condition", "split"]].copy()
    labels_csv["session_index"] = 0
    labels_csv["sequence_length"] = [int(m.sum()) for m in masks]
    labels_csv = labels_csv[
        ["subject_id", "label", "parkinsons", "condition", "split", "session_index", "sequence_length"]
    ]
    labels_csv.to_csv(cfg.output_dir / "pads_labels.csv", index=False)

    tappy_schema_cols = ["UserKey", "Parkinsons", "Split", "SessionIndex", "SessionLength"]
    pads_schema_cols = ["subject_id", "parkinsons", "split", "session_index", "sequence_length"]
    log_pads.info(
        "Schema parity check vs Tappy: tappy_labels.csv columns %s <-> "
        "pads_labels.csv equivalents %s (plus PADS-native `label`/`condition` "
        "kept for the 3-class breakdown Tappy has no equivalent of).",
        tappy_schema_cols, pads_schema_cols,
    )

    card_lines = [
        "# PADS Finger/Hand Movement Dataset — Preprocessing Data Card",
        "",
        "## Source",
        "PhysioNet: \"A public dataset of body-worn sensor data for the "
        "assessment of Parkinson's Disease\" (parkinsons-disease-smartwatch). "
        "Movement channels reconstructed via PADS's own scripts/utils "
        "(data_handling.get_data_from_observation, l1_trend_filter) applied "
        "directly to the raw movement/timeseries/*.txt recordings, NOT the "
        "shipped preprocessed/movement/*.bin files — see module docstring.",
        "",
        "## Why not the shipped .bin files",
        "preprocessed/movement/*.bin is missing PointFinger and TouchIndex: "
        "PADS's own run_preprocessing.py strips them "
        "(`to_remove = 'Time|LiftHold|PointFinger|TouchIndex'`) before saving. "
        "Only TouchNose survives there. This pipeline re-derives all three "
        "target tasks from the raw per-task .txt files instead, reusing "
        "PADS's own loading and gravity-correction code unchanged.",
        "",
        "## Tasks kept",
        f"{cfg.tasks} (finger/hand micro-movement tasks). Dropped: all "
        "Relaxed*/StretchHold/LiftHold/HoldWeight/DrinkGlas/CrossArms/"
        "Entrainment* gait/postural/whole-arm tasks.",
        "",
        "## Wrist",
        f"wrist_mode={cfg.wrist_mode}: dominant wrist per subject, from "
        "file_list.csv's `handedness` column (complete for all subjects: "
        "437 right / 32 left). Only that wrist's 6 channels "
        "(Accelerometer X/Y/Z, Gyroscope X/Y/Z) are kept.",
        "",
        "## Channels (in pads_waveforms.npy, shape [N, 6, T], float32, z-scored)",
        *[f"{i}. `{sensor}_{axis}` (dominant wrist)" for i, (sensor, axis) in enumerate(SENSOR_AXES)],
        "",
        "## Task concatenation",
        f"Tasks concatenated along time in order {cfg.tasks}. Each task's raw "
        f"recording is a fixed {drop_stats['t_task_raw']} samples; PADS's own "
        f"gravity correction (l1_trend_filter, vlambda={cfg.vlambda}) is "
        f"applied to the Accelerometer rows only (matching run_preprocessing.py), "
        f"then the first {cfg.trim_samples} samples are dropped (vibration-"
        f"notification trim, matching run_preprocessing.py), leaving "
        f"{drop_stats['t_task_trimmed']} samples/task -> T = {drop_stats['T']}.",
        "",
        "## Raw per-task length distribution (samples, before trim)",
        *[f"- {task}: {stats}" for task, stats in drop_stats["length_summary"].items()],
        "",
        "## Wavelet transform",
        "None. apply_wavelet_transform() is a documented no-op — see that "
        "function's docstring. pads_waveforms.npy holds raw (gravity-"
        "corrected, z-scored) values, matching what the real Tappy pipeline "
        "actually outputs today (tappy_waveforms.npy), not a hypothetical "
        "wavelet-coefficient CSV.",
        "",
        "## Normalization",
        "Z-score per channel, fit on TRAIN-split real (non-padded) timesteps only.",
        "",
        "## Split (subject-disjoint, stratified by `label`)",
        f"Target fractions: train={cfg.train_frac}, val={cfg.val_frac}, test={cfg.test_frac}, seed={cfg.seed}",
        "",
        "## Subjects dropped",
        f"- Attempted: {drop_stats['n_attempted']}, succeeded: {drop_stats['n_succeeded']}, dropped: {drop_stats['n_dropped']}",
        *[f"  - {reason}" for reason in drop_stats["drop_reasons"]],
        "",
        "## Class balance (final)",
        labels["label"].map(LABEL_NAMES).value_counts().to_string(),
        "",
        "## Schema parity with Tappy",
        f"tappy columns {tappy_schema_cols} <-> pads_labels.csv columns "
        f"{pads_schema_cols} (plus PADS-native `label`/`condition`). Both "
        "datasets use the [N, C, T] float32 z-scored waveform + [N, T] bool "
        "mask convention.",
    ]
    (cfg.output_dir / "pads_data_card.md").write_text("\n".join(card_lines), encoding="utf-8")
    log_pads.info("Saved outputs to %s", cfg.output_dir)


# ─── main ────────────────────────────────────────────────────────────────


def run_pads(cfg: PadsConfig) -> None:
    verify_official_preprocessing(cfg)
    verify_bin_shape(cfg)

    waveforms, masks, subject_table, drop_stats = build_all_subjects(cfg)
    if subject_table.empty:
        raise RuntimeError("No usable subjects were produced — check dataset paths and drop_stats.")

    labels = merge_labels(cfg, subject_table)

    waveforms = apply_wavelet_transform(waveforms)  # no-op today; see docstring

    split_assignment = split_subjects_pads(labels, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
    labels["split"] = labels["subject_id"].map(split_assignment)

    is_train = (labels["split"] == "train").to_numpy()
    means, stds = fit_normalization_stats_pads(waveforms, masks, is_train)
    waveforms = apply_normalization_pads(waveforms, means, stds)

    export_schema(cfg, waveforms, masks, labels, drop_stats)

    print("\n" + "=" * 60)
    print("PADS FINGER/HAND MOVEMENT PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Subjects attempted : {drop_stats['n_attempted']}")
    print(f"Subjects succeeded : {drop_stats['n_succeeded']}")
    print(f"Subjects dropped   : {drop_stats['n_dropped']}")
    for reason in drop_stats["drop_reasons"]:
        print(f"  - {reason}")
    print(f"Sequence length T  : {drop_stats['T']} (tasks={cfg.tasks}, wrist_mode={cfg.wrist_mode})")
    print("\nClass balance:")
    print(labels["label"].map(LABEL_NAMES).value_counts().to_string())
    print("\nSubjects per split:")
    print(labels.groupby("split").size().to_string())
    print("\nClass balance per split:")
    print(labels.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns=LABEL_NAMES).to_string())
    print(f"\nColumn schema matches Tappy convention (subject_id/parkinsons/split/session_index/sequence_length "
          f"<-> UserKey/Parkinsons/Split/SessionIndex/SessionLength): confirmed in pads_data_card.md")
    print("=" * 60)


def parse_args_pads() -> PadsConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pads-root", type=Path, required=True, help="Path to parkinsons-disease-smartwatch/1.0.0")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write the .npy/.csv/.md outputs")
    p.add_argument("--tasks", nargs="+", default=["PointFinger", "TouchIndex", "TouchNose"])
    p.add_argument("--wrist-mode", choices=["dominant", "LeftWrist", "RightWrist"], default="dominant")
    p.add_argument("--vlambda", type=float, default=50.0, help="l1_trend_filter regularization (PADS official default)")
    p.add_argument("--trim-samples", type=int, default=48, help="Startup-vibration trim (PADS official default)")
    p.add_argument("--task-length-percentile", type=float, default=100.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel worker processes for gravity correction")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N subjects (for testing)")
    args = p.parse_args()

    return PadsConfig(
        pads_root=args.pads_root,
        output_dir=args.output_dir,
        tasks=args.tasks,
        wrist_mode=args.wrist_mode,
        vlambda=args.vlambda,
        trim_samples=args.trim_samples,
        task_length_percentile=args.task_length_percentile,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        n_jobs=args.n_jobs,
        limit=args.limit,
    )

# ============================================================================
# TAPPY
# ============================================================================

CHANNEL_NAMES = ["hold_time", "latency_time", "flight_time"]
VALID_HANDS = {"L", "R", "S"}
VALID_DIRECTIONS = {h1 + h2 for h1 in VALID_HANDS for h2 in VALID_HANDS}
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{1,3}$")
DATE_RE = re.compile(r"^\d{6}$")
BOOL_MAP = {"True": True, "False": False}


# ─── TappyConfig ────────────────────────────────────────────────────────────────


@dataclass
class TappyConfig:
    dataset_root: Path
    output_dir: Path
    archived_dirname: str = "Archived users"
    tappy_dirname: str = "Tappy Data"
    session_gap_minutes: float = 15.0
    min_session_events: int = 20
    session_length_percentile: float = 90.0
    max_hold_time_ms: float = 10_000.0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    @property
    def archived_dir(self) -> Path:
        return self.dataset_root / self.archived_dirname

    @property
    def tappy_dir(self) -> Path:
        return self.dataset_root / self.tappy_dirname


# ─── 1. Loading: subject-level label table ─────────────────────────────────


def _parse_metadata_file(path: Path) -> dict:
    """Parses one 'Field: value' metadata file into a dict of raw strings."""
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _coerce_bool(value: str) -> Optional[bool]:
    return BOOL_MAP.get(value)


def _coerce_int(value: str) -> Optional[int]:
    return int(value) if value.isdigit() else None


def load_labels(archived_dir: Path) -> pd.DataFrame:
    """
    Loads every 'User_<KEY>.txt' metadata file into one subject-level table.

    Returns a DataFrame indexed by row with columns:
    UserKey, BirthYear, Gender, Parkinsons, Tremors, DiagnosisYear, Sided,
    UPDRS, Impact, Levadopa, DA, MAOB, Other.

    Parkinsons/Tremors/Levadopa/DA/MAOB/Other are booleans (None if the
    source field was blank/unparseable). BirthYear/DiagnosisYear are
    nullable ints. Gender/Sided/UPDRS/Impact are left as free-text strings
    since their value sets aren't guaranteed stable across releases.
    """
    if not archived_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {archived_dir}")

    rows = []
    for path in sorted(archived_dir.glob("User_*.txt")):
        user_key = path.stem[len("User_") :]
        raw = _parse_metadata_file(path)
        rows.append(
            {
                "UserKey": user_key,
                "BirthYear": _coerce_int(raw.get("BirthYear", "")),
                "Gender": raw.get("Gender") or None,
                "Parkinsons": _coerce_bool(raw.get("Parkinsons", "")),
                "Tremors": _coerce_bool(raw.get("Tremors", "")),
                "DiagnosisYear": _coerce_int(raw.get("DiagnosisYear", "")),
                "Sided": raw.get("Sided") or None,
                "UPDRS": raw.get("UPDRS") or None,
                "Impact": raw.get("Impact") or None,
                "Levadopa": _coerce_bool(raw.get("Levadopa", "")),
                "DA": _coerce_bool(raw.get("DA", "")),
                "MAOB": _coerce_bool(raw.get("MAOB", "")),
                "Other": _coerce_bool(raw.get("Other", "")),
            }
        )

    labels = pd.DataFrame(rows)
    n_missing_pd = labels["Parkinsons"].isna().sum()
    if n_missing_pd:
        log_tappy.warning(
            "%d/%d subjects have an unparseable Parkinsons field and will be "
            "dropped downstream (no usable label).",
            n_missing_pd,
            len(labels),
        )
    log_tappy.info("Loaded metadata for %d subjects from %s", len(labels), archived_dir)
    return labels


# ─── 2. Loading: per-subject keystroke event logs ──────────────────────────


def _parse_event_line(
    line: str, expected_user_key: str, max_hold_time_ms: float
) -> tuple[Optional[dict], Optional[str]]:
    """
    Validates and parses one tab-separated keystroke event row.

    Returns (row_dict, None) on success, or (None, reason) on failure, where
    reason is "malformed" (wrong field count, corrupted/glued values,
    out-of-range hand/direction codes, non-numeric/negative timing fields,
    unparseable date/timestamp) or "implausible_hold" (syntactically valid
    but physically impossible Hold-time value — see module docstring note
    on the app-backgrounding artifact this dataset has).

    This dataset has a small amount of corrupted rows (~a few hundred out of
    ~9.3M in the source release) where a tab is missing and two field values
    are concatenated together (e.g. "0105.0EA27ICBLF" instead of separate
    Hold-time and UserKey values). Requiring field[0] == expected_user_key
    catches most of these for free, since a glued value never equals the
    clean key.

    Separately, a small number of rows (~600-800 out of ~9.3M, confirmed by
    direct inspection) have a Hold time in the tens of seconds to multiple
    hours (max observed: ~13.6M ms, i.e. ~3.8 hours) — physically impossible
    for a single keystroke, almost certainly a stuck-key/app-backgrounding
    artifact rather than genuine (if severely impaired) typing. Latency time
    and Flight time show no equivalent contamination (0 rows > 5000 ms in
    either, checked directly). `max_hold_time_ms` drops only the Hold-time
    channel's implausible tail; it does not touch Latency/Flight.
    """
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 8:
        return None, "malformed"

    user_key, date, timestamp, hand, hold, direction, latency, flight = fields[:8]

    if user_key != expected_user_key:
        return None, "malformed"
    if not DATE_RE.match(date):
        return None, "malformed"
    if not TIMESTAMP_RE.match(timestamp):
        return None, "malformed"
    if hand not in VALID_HANDS:
        return None, "malformed"
    if direction not in VALID_DIRECTIONS:
        return None, "malformed"

    try:
        hold_v = float(hold)
        latency_v = float(latency)
        flight_v = float(flight)
    except ValueError:
        return None, "malformed"
    if hold_v < 0 or latency_v < 0 or flight_v < 0:
        return None, "malformed"

    if hold_v > max_hold_time_ms:
        return None, "implausible_hold"

    try:
        dt = pd.to_datetime(date + timestamp, format="%y%m%d%H:%M:%S.%f")
    except ValueError:
        return None, "malformed"

    return {
        "UserKey": user_key,
        "Timestamp": dt,
        "Hand": hand,
        "HoldTime": hold_v,
        "Direction": direction,
        "LatencyTime": latency_v,
        "FlightTime": flight_v,
    }, None


def load_user_events(
    tappy_dir: Path, user_key: str, max_hold_time_ms: float
) -> tuple[pd.DataFrame, int, int]:
    """
    Loads and concatenates all monthly event-log_tappy files for one subject,
    validates every row, and returns them sorted by timestamp.

    Returns (events_df, n_malformed_dropped, n_implausible_hold_dropped).
    """
    paths = sorted(tappy_dir.glob(f"{user_key}_*.txt"))
    parsed: list[dict] = []
    n_malformed = 0
    n_implausible_hold = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row, reason = _parse_event_line(line, user_key, max_hold_time_ms)
            if row is not None:
                parsed.append(row)
            elif reason == "malformed":
                n_malformed += 1
            elif reason == "implausible_hold":
                n_implausible_hold += 1

    if not parsed:
        empty = pd.DataFrame(columns=["UserKey", "Timestamp", "Hand", "HoldTime", "Direction", "LatencyTime", "FlightTime"])
        return empty, n_malformed, n_implausible_hold

    events = pd.DataFrame(parsed).sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    return events, n_malformed, n_implausible_hold


# ─── 3. Session segmentation ────────────────────────────────────────────────


def segment_sessions(
    events: pd.DataFrame, gap_minutes: float, min_session_events: int
) -> tuple[list[pd.DataFrame], int]:
    """
    Splits a time-sorted event log_tappy into sessions: contiguous runs of
    keystrokes with no gap larger than `gap_minutes` between consecutive
    events. Sessions shorter than `min_session_events` are dropped (treated
    as accidental/noise taps rather than real typing activity).

    Returns (sessions, n_dropped_short_sessions).
    """
    if events.empty:
        return [], 0

    gaps = events["Timestamp"].diff().dt.total_seconds() / 60.0
    new_session = (gaps > gap_minutes).fillna(False)
    session_id = new_session.cumsum()

    sessions = []
    n_dropped_short = 0
    for _, group in events.groupby(session_id, sort=True):
        if len(group) < min_session_events:
            n_dropped_short += 1
            continue
        sessions.append(group.reset_index(drop=True))

    return sessions, n_dropped_short


# ─── 4. Channel extraction ──────────────────────────────────────────────────


def extract_channels(session: pd.DataFrame) -> np.ndarray:
    """
    Builds the ordered [3, L] channel array for one session:
    Hold time, Latency time, Flight time — one value per keystroke event,
    in chronological order (session is assumed pre-sorted by segment_sessions).
    """
    return np.stack(
        [
            session["HoldTime"].to_numpy(dtype=np.float32),
            session["LatencyTime"].to_numpy(dtype=np.float32),
            session["FlightTime"].to_numpy(dtype=np.float32),
        ],
        axis=0,
    )


# ─── 5. Session-length distribution and T selection ────────────────────────


def summarize_session_lengths(lengths: list[int]) -> dict:
    arr = np.asarray(lengths)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    summary = {
        "n_sessions": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        **{f"p{p}": float(np.percentile(arr, p)) for p in percentiles},
    }
    return summary


def choose_sequence_length(lengths: list[int], percentile: float) -> int:
    return int(round(float(np.percentile(np.asarray(lengths), percentile))))


# ─── Padding / truncation ───────────────────────────────────────────────────


def pad_or_truncate(channels: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """
    channels: [3, L] float32
    Returns (fixed [3, T] float32, mask [T] bool — True where real, False
    where zero-padded).
    """
    _, length = channels.shape
    if length >= T:
        return channels[:, :T].astype(np.float32), np.ones(T, dtype=bool)

    fixed = np.zeros((channels.shape[0], T), dtype=np.float32)
    fixed[:, :length] = channels
    mask = np.zeros(T, dtype=bool)
    mask[:length] = True
    return fixed, mask


# ─── Orchestration: build every session for every usable subject ───────────


def build_all_sessions(cfg: TappyConfig, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray], dict]:
    """
    For every subject that has both a usable label and at least one
    keystroke-log_tappy file, loads events, segments sessions, and extracts raw
    (un-padded) channel arrays.

    Returns:
      meta: DataFrame, one row per session (UserKey, SessionIndex,
            SessionLength, StartTime, EndTime, Parkinsons — and other
            label columns carried through for convenience)
      raw_channels: list of [3, L] float32 arrays, aligned by index with meta
      drop_stats: dict of counts for the data card
    """
    tappy_user_keys = sorted(
        {p.name.rsplit("_", 1)[0] for p in cfg.tappy_dir.glob("*_*.txt")}
    )
    labeled_keys = set(labels.loc[labels["Parkinsons"].notna(), "UserKey"])
    event_keys = set(tappy_user_keys)

    usable_keys = sorted(labeled_keys & event_keys)
    label_only = sorted(labeled_keys - event_keys)
    events_only = sorted(event_keys - labeled_keys)

    log_tappy.info(
        "%d subjects have both a usable label and keystroke data; "
        "%d dropped (label but no keystroke files); "
        "%d dropped (keystroke files but no usable label).",
        len(usable_keys),
        len(label_only),
        len(events_only),
    )

    meta_rows = []
    raw_channels: list[np.ndarray] = []
    n_malformed_total = 0
    n_implausible_hold_total = 0
    n_short_sessions_dropped_total = 0
    n_subjects_zero_sessions = 0

    label_lookup = labels.set_index("UserKey")

    for user_key in usable_keys:
        events, n_malformed, n_implausible_hold = load_user_events(
            cfg.tappy_dir, user_key, cfg.max_hold_time_ms
        )
        n_malformed_total += n_malformed
        n_implausible_hold_total += n_implausible_hold

        sessions, n_short_dropped = segment_sessions(
            events, cfg.session_gap_minutes, cfg.min_session_events
        )
        n_short_sessions_dropped_total += n_short_dropped

        if not sessions:
            n_subjects_zero_sessions += 1
            continue

        label_row = label_lookup.loc[user_key]
        for session_idx, session in enumerate(sessions):
            channels = extract_channels(session)
            raw_channels.append(channels)
            meta_rows.append(
                {
                    "UserKey": user_key,
                    "SessionIndex": session_idx,
                    "SessionLength": channels.shape[1],
                    "StartTime": session["Timestamp"].iloc[0],
                    "EndTime": session["Timestamp"].iloc[-1],
                    "Parkinsons": bool(label_row["Parkinsons"]),
                    "Gender": label_row["Gender"],
                    "BirthYear": label_row["BirthYear"],
                    "DiagnosisYear": label_row["DiagnosisYear"],
                    "Impact": label_row["Impact"],
                }
            )

    meta = pd.DataFrame(meta_rows)
    drop_stats = {
        "subjects_label_only_dropped": len(label_only),
        "subjects_events_only_dropped": len(events_only),
        "subjects_with_zero_valid_sessions": n_subjects_zero_sessions,
        "malformed_rows_dropped": n_malformed_total,
        "implausible_hold_rows_dropped": n_implausible_hold_total,
        "short_sessions_dropped": n_short_sessions_dropped_total,
        "usable_subjects": len(usable_keys),
        "label_only_keys": label_only,
        "events_only_keys": events_only,
    }
    return meta, raw_channels, drop_stats


# ─── 8. Subject-disjoint stratified split ───────────────────────────────────


def split_subjects_tappy(
    labels: pd.DataFrame,
    usable_keys: list[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    """
    Splits subjects (not sessions) into train/val/test, stratified by
    Parkinsons status so class balance is preserved in each split. Every
    session belonging to a subject inherits that subject's split, so no
    subject's sessions ever cross a split boundary.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    label_lookup = labels.set_index("UserKey")["Parkinsons"]

    assignment: dict[str, str] = {}
    for pd_status in [True, False]:
        keys = [k for k in usable_keys if label_lookup.loc[k] == pd_status]
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # remainder goes to test, guarantees all subjects are assigned
        train_keys = keys[:n_train]
        val_keys = keys[n_train : n_train + n_val]
        test_keys = keys[n_train + n_val :]
        for k in train_keys:
            assignment[k] = "train"
        for k in val_keys:
            assignment[k] = "val"
        for k in test_keys:
            assignment[k] = "test"

    return assignment


# ─── 9. Normalization ────────────────────────────────────────────────────────


def fit_normalization_stats_tappy(
    waveforms: np.ndarray, masks: np.ndarray, is_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    waveforms: [N, 3, T] float32
    masks:     [N, T] bool (True = real timestep)
    is_train:  [N] bool

    Computes per-channel mean/std using only real (non-padded) timesteps
    from train sessions. Returns (mean [3], std [3]).
    """
    train_waveforms = waveforms[is_train]
    train_masks = masks[is_train]

    means = np.zeros(waveforms.shape[1], dtype=np.float64)
    stds = np.ones(waveforms.shape[1], dtype=np.float64)
    for c in range(waveforms.shape[1]):
        values = train_waveforms[:, c, :][train_masks]
        if values.size == 0:
            log_tappy.warning("Channel %d has no valid training timesteps; leaving mean=0, std=1.", c)
            continue
        means[c] = values.mean()
        std = values.std()
        stds[c] = std if std > 1e-8 else 1.0

    return means.astype(np.float32), stds.astype(np.float32)


def apply_normalization_tappy(waveforms: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    normalized = waveforms.copy()
    for c in range(waveforms.shape[1]):
        normalized[:, c, :] = (waveforms[:, c, :] - means[c]) / stds[c]
    return normalized.astype(np.float32)


# ─── 10. Saving ──────────────────────────────────────────────────────────────


def save_outputs(
    cfg: TappyConfig,
    waveforms: np.ndarray,
    masks: np.ndarray,
    meta: pd.DataFrame,
    length_summary: dict,
    T: int,
    drop_stats: dict,
    norm_means: np.ndarray,
    norm_stds: np.ndarray,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(cfg.output_dir / "tappy_waveforms.npy", waveforms)
    np.save(cfg.output_dir / "tappy_masks.npy", masks)

    labels_csv = meta[["UserKey", "Parkinsons", "Split", "SessionIndex", "SessionLength"]].copy()
    labels_csv.to_csv(cfg.output_dir / "tappy_labels.csv", index=False)

    split_counts = meta.groupby("Split").agg(
        sessions=("UserKey", "count"), subjects=("UserKey", "nunique")
    )
    class_balance = meta.groupby(["Split", "Parkinsons"]).size().unstack(fill_value=0)

    card_lines = [
        "# Tappy Keystroke Dataset — Preprocessing Data Card",
        "",
        "## Source",
        "PhysioNet: \"Keystroke logs collected from subjects with and without "
        "Parkinson's disease\". This release has no single `Users.txt`; "
        "per-subject demographics/diagnosis live under `Archived users/` "
        "(one file per subject), and per-keystroke event logs live under "
        "`Tappy Data/` (one file per subject per month).",
        "",
        "## Channels (in tappy_waveforms.npy, shape [N, 3, T], float32, z-scored)",
        "1. `hold_time` — dwell time of each keystroke (ms)",
        "2. `latency_time` — time from previous key event to this one (ms)",
        "3. `flight_time` — time from previous key press to this key's press (ms)",
        "",
        "Each channel is an ORDERED sequence of per-keystroke values across "
        "the session, not a summary statistic. Padded (past the real session "
        "length) or truncated timesteps are included in `tappy_masks.npy` "
        "(shape [N, T], bool — True = real event, False = pad). Padding was "
        "applied as zeros BEFORE z-score normalization, so padded positions "
        "are not exactly 0 after normalization — always gate on the mask, "
        "not on the value.",
        "",
        "## Known data-quality issue: Hold-time outliers",
        f"Direct inspection of the raw files found a small number of rows "
        f"(confirmed by scanning all ~9.3M raw events) with Hold-time values "
        f"in the tens of seconds to multiple hours (max observed ≈13.6M ms, "
        f"~3.8 hours) — physically impossible for one keystroke, almost "
        f"certainly a stuck-key/app-backgrounding artifact rather than "
        f"genuine typing. Latency time and Flight time showed no equivalent "
        f"contamination (0 rows > 5000 ms in either, checked directly), so "
        f"only Hold time is filtered. Rows with Hold time above "
        f"**{cfg.max_hold_time_ms:.0f} ms** (configurable via "
        f"--max-hold-time-ms) are dropped before session segmentation — see "
        f"`implausible_hold_rows_dropped` below. This matters: leaving them "
        f"in inflates the Hold-time channel's std by orders of magnitude "
        f"and would compress genuine keystroke variation toward 0 after "
        f"z-scoring.",
        "",
        f"## Session segmentation rule",
        f"A session is a run of a subject's keystrokes with no gap larger "
        f"than **{cfg.session_gap_minutes} minutes** between consecutive "
        f"events (configurable via --session-gap-minutes). Sessions with "
        f"fewer than **{cfg.min_session_events}** keystrokes were dropped as "
        f"noise (configurable via --min-session-events).",
        "",
        "## Session length distribution (raw event count, all usable sessions, pre-split)",
        f"n_sessions={length_summary['n_sessions']}, min={length_summary['min']}, "
        f"max={length_summary['max']}, mean={length_summary['mean']:.1f}",
        f"p10={length_summary['p10']:.0f}  p25={length_summary['p25']:.0f}  "
        f"p50={length_summary['p50']:.0f}  p75={length_summary['p75']:.0f}  "
        f"p90={length_summary['p90']:.0f}  p95={length_summary['p95']:.0f}  "
        f"p99={length_summary['p99']:.0f}",
        "",
        f"## Chosen sequence length T = {T}",
        f"Set to the {cfg.session_length_percentile:.0f}th percentile of raw "
        f"session lengths across the whole usable dataset (computed before "
        f"the train/val/test split, since T is a shape hyperparameter, not "
        f"a statistic that can leak label information the way normalization "
        f"stats can). This keeps most sessions unpadded/lightly-padded while "
        f"bounding the truncation applied to the small tail of very long "
        f"sessions.",
        "",
        "## Normalization",
        "Z-score per channel, fit on TRAIN-split real (non-padded) timesteps "
        "only, then applied unchanged to val/test.",
        f"mean = {norm_means.tolist()}",
        f"std  = {norm_stds.tolist()}",
        "",
        "## Split (subject-disjoint, stratified by Parkinsons status)",
        f"Target fractions: train={cfg.train_frac}, val={cfg.val_frac}, test={cfg.test_frac}",
        "",
        "```",
        split_counts.to_string(),
        "```",
        "",
        "### Class balance per split (session count)",
        "```",
        class_balance.to_string(),
        "```",
        "",
        "## Subjects/rows dropped",
        f"- Subjects with keystroke files but no usable label (missing/unparseable "
        f"`Parkinsons` field in their metadata file): {drop_stats['subjects_events_only_dropped']}",
        f"- Subjects with a usable label but no keystroke files at all: "
        f"{drop_stats['subjects_label_only_dropped']}",
        f"- Subjects with a usable label and keystroke files, but zero sessions "
        f"survived segmentation/min-length filtering: {drop_stats['subjects_with_zero_valid_sessions']}",
        f"- Individual malformed keystroke-log_tappy rows dropped during parsing "
        f"(bad field count, corrupted/glued values, invalid hand/direction "
        f"code, non-numeric or negative timing fields, unparseable date/"
        f"timestamp): {drop_stats['malformed_rows_dropped']}",
        f"- Individual rows dropped for an implausible Hold-time value "
        f"(> {cfg.max_hold_time_ms:.0f} ms — see data-quality note above): "
        f"{drop_stats['implausible_hold_rows_dropped']}",
        f"- Sessions dropped for having fewer than {cfg.min_session_events} "
        f"keystrokes: {drop_stats['short_sessions_dropped']}",
        f"- Subjects used: {drop_stats['usable_subjects']}",
        "",
        "## TappyConfig used",
        f"session_gap_minutes={cfg.session_gap_minutes}, "
        f"min_session_events={cfg.min_session_events}, "
        f"session_length_percentile={cfg.session_length_percentile}, "
        f"max_hold_time_ms={cfg.max_hold_time_ms}, "
        f"seed={cfg.seed}",
    ]
    (cfg.output_dir / "tappy_data_card.md").write_text("\n".join(card_lines), encoding="utf-8")

    log_tappy.info("Saved outputs to %s", cfg.output_dir)


# ─── main ────────────────────────────────────────────────────────────────────


def run_tappy(cfg: TappyConfig) -> None:
    labels = load_labels(cfg.archived_dir)

    meta, raw_channels, drop_stats = build_all_sessions(cfg, labels)
    if meta.empty:
        raise RuntimeError("No usable sessions were built — check dataset paths and thresholds.")

    length_summary = summarize_session_lengths(meta["SessionLength"].tolist())
    log_tappy.info("Session length distribution: %s", length_summary)

    T = choose_sequence_length(meta["SessionLength"].tolist(), cfg.session_length_percentile)
    log_tappy.info("Chosen T = %d (p%.0f of session lengths)", T, cfg.session_length_percentile)

    fixed_list, mask_list = [], []
    for channels in raw_channels:
        fixed, mask = pad_or_truncate(channels, T)
        fixed_list.append(fixed)
        mask_list.append(mask)
    waveforms = np.stack(fixed_list, axis=0).astype(np.float32)  # [N, 3, T]
    masks = np.stack(mask_list, axis=0)  # [N, T] bool

    usable_keys = sorted(meta["UserKey"].unique().tolist())
    split_assignment = split_subjects_tappy(
        labels, usable_keys, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed
    )
    meta["Split"] = meta["UserKey"].map(split_assignment)

    is_train = (meta["Split"] == "train").to_numpy()
    norm_means, norm_stds = fit_normalization_stats_tappy(waveforms, masks, is_train)
    waveforms = apply_normalization_tappy(waveforms, norm_means, norm_stds)

    save_outputs(cfg, waveforms, masks, meta, length_summary, T, drop_stats, norm_means, norm_stds)

    # ─── Summary ───
    n_subjects = meta["UserKey"].nunique()
    n_sessions = len(meta)
    print("\n" + "=" * 60)
    print("TAPPY KEYSTROKE PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total usable subjects : {n_subjects}")
    print(f"Total sessions        : {n_sessions}")
    print(f"Sequence length T     : {T} (p{cfg.session_length_percentile:.0f} of session lengths, "
          f"min={length_summary['min']}, max={length_summary['max']}, mean={length_summary['mean']:.1f})")
    print("\nSessions per split:")
    print(meta.groupby("Split").size().to_string())
    print("\nSubjects per split:")
    print(meta.groupby("Split")["UserKey"].nunique().to_string())
    print("\nClass balance (session count) per split:")
    print(meta.groupby(["Split", "Parkinsons"]).size().unstack(fill_value=0).to_string())
    print("\nClass balance (subject count) per split:")
    subj_balance = meta.drop_duplicates("UserKey").groupby(["Split", "Parkinsons"]).size().unstack(fill_value=0)
    print(subj_balance.to_string())
    print("=" * 60)


def parse_args_tappy() -> TappyConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True, help="Path to tappy-keystroke-data-*.* directory")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write the .npy/.csv/.md outputs")
    p.add_argument("--session-gap-minutes", type=float, default=15.0, help="Gap (minutes) that starts a new session")
    p.add_argument("--min-session-events", type=int, default=20, help="Drop sessions with fewer keystrokes than this")
    p.add_argument("--session-length-percentile", type=float, default=90.0, help="Percentile of session lengths used to pick T")
    p.add_argument("--max-hold-time-ms", type=float, default=10_000.0, help="Drop rows with Hold time above this (ms) as physically implausible")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    return TappyConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        session_gap_minutes=args.session_gap_minutes,
        min_session_events=args.min_session_events,
        session_length_percentile=args.session_length_percentile,
        max_hold_time_ms=args.max_hold_time_ms,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

# ─────────────────────────────────────────────────────────────────────────
# Unified CLI — dispatches to whichever dataset's pipeline was requested.
# Each dataset keeps its own parse_args_<dataset>() with its own flags,
# exactly as in the original standalone scripts; only the outer dataset
# selector is new (replaces having 4 separate files to invoke).
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    datasets = {"gaitrec": (run_gaitrec, parse_args_gaitrec),
                "mpower": (run_mpower, parse_args_mpower),
                "pads": (run_pads, parse_args_pads),
                "tappy": (run_tappy, parse_args_tappy)}
    if len(sys.argv) < 2 or sys.argv[1] not in datasets:
        print(f"usage: {sys.argv[0]} {{{','.join(datasets)}}} [dataset-specific args...]", file=sys.stderr)
        print(f"Run '{sys.argv[0]} <dataset> --help' to see that dataset's arguments.", file=sys.stderr)
        sys.exit(1)
    dataset = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # let the dataset's own parser see only its own flags
    run_fn, parse_fn = datasets[dataset]
    run_fn(parse_fn())


if __name__ == "__main__":
    main()
