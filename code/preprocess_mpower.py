"""
Preprocessing pipeline for the mPower dataset (Synapse/Sage Bionetworks):
"mPower: A study of Parkinson's disease using smartphones".

mPower's finger-tapping data arrived as PRE-COMPUTED handcrafted features
(not raw touch/accelerometer signal), so unlike Tappy/PADS/GaitRec this
stays a flat [N_records, num_features] TABLE — no waveform reshaping is
attempted (there's no genuine time axis in pre-aggregated features). See
Phase 0.1 / Phase 1.5 of the CBDL development plan for why.

Two candidate source files were inspected before choosing one:
  - tapFeatures.tsv (78,873 rows): broader population, but keyed only by
    `filehandle` (a Synapse file-object id) — NO subject identifier at all,
    so a subject-disjoint split is impossible from this file alone, and no
    diagnosis label could be joined to it with the files on hand.
  - mpower/data_for_treat_vs_tod_paper.csv (12,915 rows, 104 subjects):
    the SAME 41 tapping features (a strict subset — missing dfaTapInter and
    corXY, which tapFeatures.tsv has) PLUS `healthCode` (a real subject id)
    and a `PD` column.

**Chosen: data_for_treat_vs_tod_paper.csv**, for the subject-disjoint split
it enables. Its `PD` column is not usable as a classification label here —
every one of its 104 subjects is PD==True (it's a medication-timing study
enrolling only diagnosed patients, no healthy-control arm) — so this run
produces mPower as an UNLABELED feature table. This is consistent with the
CBDL plan's own fallback ("mPower's label is a secondary signal — PADS
remains the primary clinical grounding source... either way"): mPower's
role in the architecture is a handcrafted-feature MLP branch fused into the
Finger Embedding, not a labeled probe target, so an unlabeled feature table
does not block Phase 2.

Usage:
    python preprocess_mpower.py \
        --source-csv "mpower/data_for_treat_vs_tod_paper.csv" \
        --output-dir "mpower_preprocessed"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("preprocess_mpower")

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
class Config:
    source_csv: Path
    output_dir: Path
    min_number_taps: int = 10
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


def split_subjects(subjects: list[str], train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict[str, str]:
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


def run(cfg: Config) -> None:
    df = pd.read_csv(cfg.source_csv)
    n_raw = len(df)
    log.info("Loaded %d raw records, %d subjects (healthCode).", n_raw, df["healthCode"].nunique())

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Expected feature columns missing from source: {missing_cols}")

    # ── Cleaning ──
    n_nan_dropped = df[FEATURE_COLUMNS].isna().any(axis=1).sum()
    df = df[~df[FEATURE_COLUMNS].isna().any(axis=1)].copy()

    n_low_taps_dropped = (df["numberTaps"] < cfg.min_number_taps).sum()
    df = df[df["numberTaps"] >= cfg.min_number_taps].copy()

    df = df.reset_index(drop=True)
    log.info(
        "After cleaning: %d records (%d dropped for missing features, %d dropped for numberTaps < %d).",
        len(df), n_nan_dropped, n_low_taps_dropped, cfg.min_number_taps,
    )

    # ── Subject-disjoint split ──
    subjects = sorted(df["healthCode"].unique().tolist())
    split_assignment = split_subjects(subjects, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
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


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-number-taps", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return Config(
        source_csv=args.source_csv, output_dir=args.output_dir, min_number_taps=args.min_number_taps,
        train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
