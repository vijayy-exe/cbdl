"""
Generates per-CHANNEL wavelet scalogram grids for each dataset, before vs.
after preprocessing — one small scalogram per actual column/channel of the
dataset (all 18 GaitRec GRF channels, all 6 PADS IMU channels, all 3 Tappy
keystroke-timing channels), laid out as small multiples so the dataset's
full channel structure is visible in one image: which channels look alike
(e.g. left/right symmetry, RAW vs. PRO pairs), which look different (e.g.
Accelerometer vs. Gyroscope, hold vs. latency vs. flight), and whether
preprocessing changes that structure or just rescales it.

Companion to generate_wavelet_figures.py (single-channel before/after) and
generate_preprocessing_figures.py (raw data-shape changes). Same caveat
applies: this is an illustrative signal-analysis view, not a step CBDL's
pipeline actually performs (see generate_wavelet_figures.py docstring).

mPower is excluded — its 41 columns are pre-aggregated scalar statistics
with no time axis, so there is no per-channel sequence to run a wavelet
transform over (see mpower_data_card.md / wavelet_mpower.png).

Usage:
    python generate_wavelet_channel_structure.py --data-root "."
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from generate_preprocessing_figures import GAITREC_MEAN, GAITREC_STD, GAITREC_CHANNELS, TAPPY_MEAN, TAPPY_STD, TAPPY_CHANNELS

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 8.5

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

def fig_gaitrec(root: Path, out: Path):
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

def fig_pads(root: Path, out: Path):
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

def fig_tappy(root: Path, out: Path):
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


def run(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_gaitrec(root, figdir / "wavelet_channels_gaitrec.png")
    print("Saved figures/wavelet_channels_gaitrec.png")
    fig_pads(root, figdir / "wavelet_channels_pads.png")
    print("Saved figures/wavelet_channels_pads.png")
    fig_tappy(root, figdir / "wavelet_channels_tappy.png")
    print("Saved figures/wavelet_channels_tappy.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
