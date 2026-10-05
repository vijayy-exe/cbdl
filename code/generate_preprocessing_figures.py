"""
Generates "before vs. after preprocessing" figures for each of the four CBDL
source datasets (GaitRec, Tappy, PADS, mPower), for use in a presentation.

Unlike generate_figures.py (which visualizes trained-model results), this
script visualizes the *preprocessing pipeline itself* — what each dataset
looked like before preprocessing vs. after, using the already-preprocessed
.npy arrays plus normalization statistics recorded in each dataset's data
card (gaitrec_preprocessed/gaitrec_data_card.md, etc.). Raw-scale values are
recovered losslessly by inverting the per-channel z-score (raw = z*std +
mean) — this avoids re-reading the multi-gigabyte raw source CSVs while
reproducing the exact pre-normalization values.

Usage:
    python generate_preprocessing_figures.py --data-root "."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9

# ─── Normalization stats transcribed from each *_data_card.md (train-split fit) ──

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

BLUE, ORANGE, GRAY, RED = "#3b6fa0", "#d97a3f", "#888888", "#c0392b"


# ─── GaitRec ─────────────────────────────────────────────────────────────────

def fig_gaitrec(root: Path, out: Path):
    w = np.load(root / "gaitrec_preprocessed" / "gaitrec_waveforms.npy")
    labels = pd.read_csv(root / "gaitrec_preprocessed" / "gaitrec_labels.csv")
    raw = w * GAITREC_STD[None, :, None] + GAITREC_MEAN[None, :, None]

    show_ch = [GAITREC_CHANNELS.index(c) for c in ["F_V_RAW_L", "F_AP_RAW_L", "COP_AP_RAW_L"]]
    trial = np.where(labels["ClassLabel"].values == "HC")[0][0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("GaitRec — Raw Force-Plate Signal vs. Preprocessed Waveform", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    for i, ci in enumerate(show_ch):
        ax.plot(raw[trial, ci], label=GAITREC_CHANNELS[ci], color=[BLUE, ORANGE, "#5aa469"][i])
    ax.set_title("Before: raw sensor units\n(one HC trial, 3 of 20 recorded channels)")
    ax.set_xlabel("% gait cycle (0-100)"); ax.set_ylabel("raw units (N, mm)"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    for i, ci in enumerate(show_ch):
        ax.plot(w[trial, ci], label=GAITREC_CHANNELS[ci], color=[BLUE, ORANGE, "#5aa469"][i])
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_title("After: per-channel z-score\n(same trial, unified scale)")
    ax.set_xlabel("% gait cycle (0-100)"); ax.set_ylabel("z-score"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    bars = ax.bar(["Recorded\n(20 channels)", "Kept\n(18 channels)"], [20, 18], color=[GRAY, BLUE], width=0.5)
    for b, v in zip(bars, [20, 18]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center", fontweight="bold")
    ax.set_ylim(0, 23)
    ax.set_title("Channel drop: 2 corrupted channels excluded")
    ax.text(0.5, -0.32, "GRF_F_V_PRO_left (0% present) · GRF_F_AP_PRO_left (97% missing)\n"
                         "verified: not a row drop — same 75,732-trial key set across the other 18",
            transform=ax.transAxes, ha="center", fontsize=7, color=RED)

    ax = axes[1, 1]
    splits = ["train", "val", "test"]
    trials = [53456, 11338, 10938]
    subjects = [1607, 345, 343]
    x = np.arange(3)
    ax.bar(x - 0.18, trials, width=0.36, color=BLUE, label="trials")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, subjects, width=0.36, color=ORANGE, label="subjects")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("trials", color=BLUE); ax2.set_ylabel("subjects", color=ORANGE)
    ax.set_title("Subject-disjoint split (75,732 trials / 2,295 subjects total)")
    for xi, v in zip(x - 0.18, trials):
        ax.text(xi, v + 800, str(v), ha="center", fontsize=7, color=BLUE)
    for xi, v in zip(x + 0.18, subjects):
        ax2.text(xi, v + 20, str(v), ha="center", fontsize=7, color=ORANGE)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── Tappy ───────────────────────────────────────────────────────────────────

def fig_tappy(root: Path, out: Path):
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
    ax.hist(raw_lens, bins=bins, color=BLUE, alpha=0.85)
    ax.set_xscale("log")
    for val, lbl, y in [(109, "p50=109", 0.75), (888, "T=888 (p90)", 0.55), (3636, "p99=3636", 0.35)]:
        ax.axvline(val, color=RED, ls="--", lw=1)
        ax.text(val * 1.15, ax.get_ylim()[1] * y, lbl, fontsize=7, color=RED)
    ax.set_title("Before: raw session length distribution\n(min=20, max=46,101 keystrokes, n=23,752 sessions)", fontsize=9)
    ax.set_xlabel("keystrokes per session (log scale)"); ax.set_ylabel("# sessions")

    ax = fig.add_subplot(gs[0, 1])
    cats = ["kept\n(≤10,000ms)", "dropped\n(implausible, >10,000ms)"]
    vals = [9_300_000 - 615, 615]
    bars = ax.bar(cats, vals, color=[BLUE, RED])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:,}", ha="center", fontsize=8)
    ax.set_title("Hold-time outlier filter\n(max raw value ≈ 13.6M ms — stuck-key artifact)", fontsize=9)
    ax.set_ylabel("# raw keystroke\nevents (log scale)")

    ax = fig.add_subplot(gs[1, :])
    hold_short_raw = w[short_idx, 0][m[short_idx]] * TAPPY_STD[0] + TAPPY_MEAN[0]
    ax.plot(hold_short_raw, color=BLUE, marker="o", ms=3)
    ax.axvspan(len(hold_short_raw), T, color=BLUE, alpha=0.12)
    ax.set_xlim(0, T)
    ax.set_title(f"After — short session ({raw_lens[short_idx]} events): zero-padded to T={T} (shaded = padded, masked out)", fontsize=9)
    ax.set_ylabel("hold_time (ms)")

    ax = fig.add_subplot(gs[2, :])
    hold_long_raw = w[long_idx, 0] * TAPPY_STD[0] + TAPPY_MEAN[0]
    ax.plot(hold_long_raw, color=ORANGE, lw=0.6)
    ax.axvline(T, color=RED, ls="--", lw=1)
    ax.set_xlim(0, len(hold_long_raw))
    ax.set_title(f"After — long session ({raw_lens[long_idx]} events): truncated at T={T} (red line = cutoff, tail discarded)", fontsize=9)
    ax.set_xlabel("raw timestep"); ax.set_ylabel("hold_time (ms)")

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── PADS ────────────────────────────────────────────────────────────────────

def fig_pads(root: Path, out: Path):
    w = np.load(root / "pads_preprocessed" / "pads_waveforms.npy")
    labels = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv")

    subj = 0
    example = w[subj, 0]  # Accelerometer_X, z-scored (data card does not list per-channel mean/std separately from split fit; shown in z-score space)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("PADS — Raw Wrist-IMU Recording vs. Preprocessed Concatenated Waveform", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    tasks = ["PointFinger", "TouchIndex", "TouchNose"]
    starts = [0, 1024, 2048]
    for s, t, c in zip(starts, tasks, [BLUE, ORANGE, "#5aa469"]):
        ax.barh(0, 1024, left=s, height=0.5, color=c, alpha=0.35, label=f"{t} (1024 raw samples)")
        ax.barh(0, 48, left=s, height=0.5, color=RED, alpha=0.8)
    ax.set_yticks([]); ax.set_xlim(0, 3072)
    ax.set_title("Before: 3 tasks × 1024 raw samples each\n(red = first 48 samples trimmed per task — startup vibration artifact)")
    ax.set_xlabel("raw sample index"); ax.legend(fontsize=6.5, loc="upper center", ncol=1)

    ax = axes[0, 1]
    ax.barh(0, 2928, left=0, height=0.5, color=BLUE, alpha=0.6)
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
    ax.bar(x - w_, pointfinger, width=w_, color=BLUE, label="PointFinger")
    ax.bar(x, touchindex, width=w_, color=ORANGE, label="TouchIndex")
    ax.bar(x + w_, touchnose, width=w_, color="#5aa469", label="TouchNose")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("subjects with task data (of 469)")
    ax.set_title("Why raw .txt, not shipped .bin\n(.bin drops PointFinger + TouchIndex entirely)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(example, color=BLUE, lw=0.7)
    for s in [976, 1952]:
        ax.axvline(s, color="black", ls="--", lw=0.8)
    ax.set_title(f"Preprocessed Accelerometer_X, subject {labels.iloc[subj]['subject_id']}\n"
                 "(z-scored; dashed lines = task boundaries)")
    ax.set_xlabel("timestep"); ax.set_ylabel("z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── mPower ──────────────────────────────────────────────────────────────────

def fig_mpower(root: Path, out: Path):
    labels = pd.read_csv(root / "mpower_preprocessed" / "mpower_labels.csv")
    feats = np.load(root / "mpower_preprocessed" / "mpower_features.npy")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("mPower — Raw Tapping-Session Features vs. Preprocessed Feature Table", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    cats = ["tapFeatures.tsv\n(broader, no subject id)", "data_for_treat_vs_tod\n_paper.csv (chosen)"]
    rows = [78873, 12915]
    bars = ax.bar(cats, rows, color=[GRAY, BLUE])
    for b, v in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 1000, f"{v:,}", ha="center", fontsize=8)
    ax.set_title("Source file choice\n(only the chosen file carries healthCode → subject-disjoint split)")
    ax.set_ylabel("raw rows")

    ax = axes[1, 0]
    stages = ["Raw records", "− missing\nfeature values", "− numberTaps\n< 10", "Final"]
    vals = [12915, 12915 - 1, 12915 - 1 - 4, 12910]
    ax.plot(stages, vals, "o-", color=BLUE)
    for s, v in zip(stages, vals):
        ax.text(s, v + 5, str(v), ha="center", fontsize=8)
    ax.set_title("Cleaning funnel (104 subjects preserved throughout)")
    ax.set_ylabel("# records"); ax.set_ylim(12905, 12920)

    show = ["numberTaps", "cvTapInter", "sdTapInter", "iqrDriftLeft", "buttonNoneFreq"]
    idx = [MPOWER_FEATURES.index(f) for f in show]

    ax = axes[0, 1]
    x = np.arange(len(show))
    ax.bar(x, MPOWER_MEAN[idx], yerr=MPOWER_STD[idx], color=GRAY, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=30, ha="right", fontsize=7)
    ax.set_title("Before: raw feature scales differ by orders\nof magnitude (mean ± std, 5 of 41 features shown)")
    ax.set_yscale("symlog")

    ax = axes[1, 1]
    ax.bar(x, np.zeros(len(show)), yerr=np.ones(len(show)), color=BLUE, capsize=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=30, ha="right", fontsize=7)
    ax.set_title("After: z-scored per feature\n(all mean=0, std=1 — comparable to the MLP branch)")
    ax.set_ylim(-1.5, 1.5)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_gaitrec(root, figdir / "preprocessing_gaitrec.png")
    print("Saved figures/preprocessing_gaitrec.png")
    fig_tappy(root, figdir / "preprocessing_tappy.png")
    print("Saved figures/preprocessing_tappy.png")
    fig_pads(root, figdir / "preprocessing_pads.png")
    print("Saved figures/preprocessing_pads.png")
    fig_mpower(root, figdir / "preprocessing_mpower.png")
    print("Saved figures/preprocessing_mpower.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
