"""
Generates wavelet scalogram (time-frequency) visualizations of a
representative signal from each dataset, raw vs. preprocessed, for use in a
presentation.

IMPORTANT — these are illustrative signal-analysis visualizations, not a
step CBDL actually performs. No wavelet transform exists anywhere in the
real CBDL pipeline: preprocess_pads.py ships an `apply_wavelet_transform()`
function that is an explicitly documented no-op / seam for a hypothetical
future stage (see that function's docstring) — pads_waveforms.npy and every
other *_waveforms.npy hold plain z-scored floats, never wavelet
coefficients. This script computes a continuous wavelet transform (CWT,
complex Morlet) purely for visualization, to show how each dataset's
signal's time-frequency content looks before/after preprocessing.

mPower has no time axis at all (it is a flat handcrafted-feature table, see
generate_preprocessing_figures.py / mpower_data_card.md) — a wavelet
transform needs a sequence to decompose, so mPower gets an explanatory
panel instead of a scalogram.

Usage:
    python generate_wavelet_figures.py --data-root "."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from generate_preprocessing_figures import GAITREC_MEAN, GAITREC_STD, GAITREC_CHANNELS, TAPPY_MEAN, TAPPY_STD

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 9

WAVELET = "cmor1.5-1.0"
BLUE, ORANGE = "#3b6fa0", "#d97a3f"


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

def fig_gaitrec(root: Path, out: Path):
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

    trace_and_scalogram(fig, [gs[0, 0], gs[1, 0]], raw_sig, scales, BLUE,
                         "Before: raw sensor units", "CWT of raw signal",
                         "% gait cycle", "raw units (N)")
    trace_and_scalogram(fig, [gs[0, 1], gs[1, 1]], pre_sig, scales, ORANGE,
                         "After: z-scored", "CWT of z-scored signal",
                         "% gait cycle", "z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── PADS ────────────────────────────────────────────────────────────────────

def fig_pads(root: Path, out: Path):
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

    trace_and_scalogram(fig, [gs[0, 0], gs[1, 0]], raw_sig, scales, BLUE,
                         "Before: raw .txt (1024 samples,\npre-gravity-correction)", "CWT of raw signal",
                         "raw sample index", "acceleration (g)")
    trace_and_scalogram(fig, [gs[0, 1], gs[1, 1]], pre_sig, scales, ORANGE,
                         "After: gravity-corrected, trimmed,\nz-scored (976 samples)", "CWT of preprocessed signal",
                         "timestep", "z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── Tappy ───────────────────────────────────────────────────────────────────

def fig_tappy(root: Path, out: Path):
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

    trace_and_scalogram(fig, [gs[0, 0], gs[1, 0]], raw_sig, scales, BLUE,
                         "Before: raw hold-time (ms)", "CWT of raw signal",
                         "keystroke index", "hold_time (ms)")
    trace_and_scalogram(fig, [gs[0, 1], gs[1, 1]], pre_sig, scales, ORANGE,
                         "After: z-scored", "CWT of z-scored signal",
                         "keystroke index", "z-score")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── mPower (not applicable) ────────────────────────────────────────────────

def fig_mpower(out: Path):
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


def run(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_gaitrec(root, figdir / "wavelet_gaitrec.png")
    print("Saved figures/wavelet_gaitrec.png")
    fig_pads(root, figdir / "wavelet_pads.png")
    print("Saved figures/wavelet_pads.png")
    fig_tappy(root, figdir / "wavelet_tappy.png")
    print("Saved figures/wavelet_tappy.png")
    fig_mpower(figdir / "wavelet_mpower.png")
    print("Saved figures/wavelet_mpower.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
