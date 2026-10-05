"""
Generates architecture / tensor-shape flow diagrams for the CBDL pipeline,
for use in a presentation: how each dataset's tensor shape changes through
the Phase 2 modality encoders, through the Phase 3 Cross-Body Dependency
Module (CBDL's core novelty), and into the Phase 4 final classification
output. These are block diagrams (shapes/architecture), not data plots —
companion to generate_preprocessing_figures.py (which shows how the raw
DATA changes) and generate_figures.py (which shows trained-model results).

Shapes and module names are taken directly from model.py, cross_body_module.py,
and train_phase4_clinical_head.py (K_WINDOWS=8, embed_dim=128, etc.) — this
script draws no numbers that aren't in that code.

Usage:
    python generate_pipeline_shape_figures.py --data-root "."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 8.5

BLUE, LBLUE = "#3b6fa0", "#dbe7f5"
ORANGE, LORANGE = "#c96a2e", "#f6ddc9"
GREEN, LGREEN = "#4c8c5c", "#dcedd9"
GRAY, LGRAY = "#666666", "#e8e8e8"
RED = "#c0392b"


def box(ax, x, y, w, h, text, fc=LBLUE, ec=BLUE, fontsize=8, lw=1.3, style="round,pad=0.012"):
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
    g_in = box(ax, 2, 84, 22, 9, "GaitRec input\n[B, 18, 101]", fc=LGREEN, ec=GREEN)
    g_conv = box(ax, 28, 84, 24, 9, "Conv1d×2\n18→32→64ch, k=5", fc="white", ec=GRAY)
    g_lstm = box(ax, 56, 84, 22, 9, "BiLSTM\nhidden=64", fc="white", ec=GRAY)
    g_pool = box(ax, 82, 84, 20, 9, "mean-pool\n+ Linear→128", fc="white", ec=GRAY)
    g_out = box(ax, 106, 84, 26, 9, "Gait Embedding\n[B, 128]", fc=LGREEN, ec=GREEN, fontsize=9)
    for a, b_ in [(g_in, g_conv), (g_conv, g_lstm), (g_lstm, g_pool), (g_pool, g_out)]:
        arrow(ax, right(a), left(b_))

    # PADS row
    p_in = box(ax, 2, 62, 22, 9, "PADS input\n[B, 6, 2928]", fc=LBLUE, ec=BLUE)
    p_ad = box(ax, 28, 62, 24, 9, "WaveformAdapter\nConv1d(6→64), k=1", fc="white", ec=GRAY)
    arrow(ax, right(p_in), left(p_ad))

    # Tappy row
    t_in = box(ax, 2, 40, 22, 9, "Tappy input\n[B, 3, 888]", fc=LORANGE, ec=ORANGE)
    t_ad = box(ax, 28, 40, 24, 9, "WaveformAdapter\nConv1d(3→64), k=1", fc="white", ec=GRAY)
    arrow(ax, right(t_in), left(t_ad))

    # Shared trunk (converges PADS + Tappy)
    trunk = box(ax, 56, 48, 26, 13, "Shared\nFingerWaveformEncoder\nBiGRU(hidden=64)\nmasked mean-pool → Linear→128", fc="white", ec=BLUE, fontsize=7.8)
    arrow(ax, right(p_ad), (56, 62 + 4.5))
    arrow(ax, right(t_ad), (56, 40 + 4.5))
    ax.text(52, 61, "same\nweights", fontsize=6.5, color=BLUE, ha="center", style="italic")

    p_out = box(ax, 88, 62, 24, 9, "PADS waveform\nembedding [B, 128]", fc=LBLUE, ec=BLUE, fontsize=8)
    t_out = box(ax, 88, 40, 24, 9, "Tappy waveform\nembedding [B, 128]", fc=LORANGE, ec=ORANGE, fontsize=8)
    arrow(ax, (82, 62 + 4.5), left(p_out))
    arrow(ax, (82, 40 + 4.5), left(t_out))

    # mPower row
    m_in = box(ax, 2, 18, 22, 9, "mPower input\n[B, 41]", fc=LGRAY, ec=GRAY)
    m_mlp = box(ax, 28, 18, 24, 9, "MPowerMLPBranch\n41→64→128 (MLP)", fc="white", ec=GRAY)
    m_out = box(ax, 56, 18, 24, 9, "mPower embedding\n[B, 128]", fc=LGRAY, ec=GRAY, fontsize=8)
    arrow(ax, right(m_in), left(m_mlp))
    arrow(ax, right(m_mlp), left(m_out))
    m_dec = box(ax, 56, 3, 24, 9, "MPowerDecoder\n128→64→41\n(reconstruction loss —\nno diagnosis label available)", fc="white", ec=GRAY, fontsize=6.8)
    arrow(ax, bottom(m_out), top(m_dec), ls="--")

    # Fusion layer (PADS waveform embed + mPower embed -> Finger Embedding)
    fusion = box(ax, 116, 30, 30, 15, "FusionLayer\nconcat[256] → Linear → 128\n(learned 'absent' token\nsubstituted for the missing side —\nno sample has both a waveform\nand mPower features)", fc="white", ec=RED, fontsize=7)
    arrow(ax, right(p_out), left(fusion))
    arrow(ax, right(m_out), left(fusion))

    fe_out = box(ax, 116, 10, 30, 9, "Phase 2 Finger Embedding\n[B, 128]  (probe-only;\nsee note)", fc="white", ec=RED, fontsize=7.2)
    arrow(ax, bottom(fusion), top(fe_out))

    ax.text(74, 1.5,
            "Note: Phase 3 (CBDL) attends over the pre-fusion, WINDOWED trunk/gait outputs directly (Fig. 2) — the pooled Fusion-Layer\n"
            "embedding above is used only for the Phase 2 linear-probe sanity check, per model.py's encode_finger_windows / encode_gait_windows.",
            fontsize=7, color=GRAY, ha="center")

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
    box(ax, 2, 84, 40, 8, "PADS window embeddings [B, 8, 128]\n(masked_window_pool, K=8)", fc=LBLUE, ec=BLUE, fontsize=7.6)
    window_strip(ax, 2, 76, 40, 5, 8, LBLUE, BLUE)
    box(ax, 2, 55, 40, 8, "GaitRec window embeddings [B, 8, 128]\n(masked_window_pool, K=8)", fc=LGREEN, ec=GREEN, fontsize=7.6)
    window_strip(ax, 2, 47, 40, 5, 8, LGREEN, GREEN)

    ax.text(22, 66.5, "population-level pairing by\nDIAGNOSIS TIER (Healthy vs.\nPathological) — not subject\nidentity (0 shared subject IDs)",
            fontsize=6.8, color=RED, ha="center", style="italic")

    arrow(ax, (42, 80), (52, 74))
    arrow(ax, (42, 51), (52, 68))

    # Lagged cross-attention: 4 lag branches
    lag_y = [88, 74, 60, 46]
    lag_boxes = []
    for d, y in zip([0, 1, 2, 3], lag_y):
        b_ = box(ax, 52, y, 42, 11,
                  f"Δ={d}: finger[0:{8-d}] attends gait[{d}:8]\n"
                  f"MultiheadAttention(4 heads)\n(f + attended)/2 → mean-pool → [B,128]",
                  fc="white", ec=GRAY, fontsize=6.6)
        lag_boxes.append(b_)

    scorer = box(ax, 100, 60, 20, 26, "Lag Scorer\nLinear(128→64)\nReLU\nLinear(64→1)\nper lag → 4 logits\n\nsoftmax → lag\nweights [B, 4]", fc="white", ec=ORANGE, fontsize=7)
    for b_ in lag_boxes:
        arrow(ax, right(b_), left(scorer))

    weighted = box(ax, 100, 30, 20, 18, "Weighted sum\nΣ weight_d · repr_d\n→ [B, 128]\n\n(this produces the\nAttention-Lag Heatmap\nwhen averaged over\nthe test cohort)", fc="white", ec=ORANGE, fontsize=6.8)
    arrow(ax, bottom(scorer), top(weighted))
    for b_ in lag_boxes:
        arrow(ax, (94, b_[1] + b_[3] / 2), (100, 39), lw=0.7, color="#bbbbbb")

    outproj = box(ax, 124, 40, 22, 8, "out_proj\nLinear(128→128)", fc="white", ec=BLUE, fontsize=7.5)
    arrow(ax, right(weighted), left(outproj))

    fused = box(ax, 108, 12, 38, 10, "Fused Cross-Body Embedding\n[B, 128]  →  Phase 4 (Fig. 3)", fc=LBLUE, ec=BLUE, fontsize=8.5)
    arrow(ax, bottom(outproj), (135, 22))

    # Auxiliary training-only losses (dashed, off to the side)
    dtw = box(ax, 2, 22, 40, 10, "Soft-DTW alignment loss\n(align_proj re-embeds BOTH sides —\nencoders are frozen in Phase 3, so\nSoft-DTW needs a trainable target)", fc="white", ec=GRAY, fontsize=6.6, style="round,pad=0.012")
    infonce = box(ax, 2, 6, 40, 10, "InfoNCE contrastive loss\n(in-batch negatives; positive =\nsame-tier gait sample; computed on\nthe FUSED embedding, not raw encoder\noutput, so attention gets gradient)", fc="white", ec=GRAY, fontsize=6.4)
    ax.text(22, 34.5, "training-only auxiliary losses\n(not part of the forward path used at inference)", fontsize=6.5, color=GRAY, ha="center", style="italic")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 3: Phase 4 — final classification output shape flow ───────────────

def fig_final_output(out: Path):
    fig, ax = new_canvas((13, 6.5), (0, 140), (0, 85), "CBDL Phase 4 — Final Classification Output")

    fused = box(ax, 2, 44, 30, 12, "Fused Cross-Body\nEmbedding [B, 128]\n(from Fig. 2)", fc=LBLUE, ec=BLUE, fontsize=9)

    # PADS primary head
    p_head = box(ax, 42, 66, 34, 11, "PADS Clinical Head (PRIMARY)\nLinear(128→64)→ReLU→Dropout(0.2)\n→Linear(64→3)", fc="white", ec=BLUE, fontsize=7.5)
    p_logit = box(ax, 82, 66, 20, 11, "logits\n[B, 3]", fc=LBLUE, ec=BLUE, fontsize=8)
    p_soft = box(ax, 108, 66, 26, 11, "softmax → argmax\nHealthy / Parkinson's /\nOther Movement Disorder", fc="white", ec=BLUE, fontsize=7)
    arrow(ax, right(fused), (42, 71.5))
    arrow(ax, right(p_head), left(p_logit))
    arrow(ax, right(p_logit), left(p_soft))

    p_cal = box(ax, 108, 50, 26, 11, "Phase 5: isotonic\ncalibration recalibrates\nthese probabilities\n(see reliability diagram)", fc="white", ec=GRAY, fontsize=6.8)
    arrow(ax, bottom(p_soft), top(p_cal), ls="--")

    # GaitRec secondary head
    g_head = box(ax, 42, 20, 34, 11, "GaitRec Head (secondary check)\nLinear(128→64)→ReLU→Dropout(0.2)\n→Linear(64→5)", fc="white", ec=GREEN, fontsize=7.5)
    g_logit = box(ax, 82, 20, 20, 11, "logits\n[B, 5]", fc=LGREEN, ec=GREEN, fontsize=8)
    g_soft = box(ax, 108, 20, 26, 11, "softmax → argmax\nHC / A / C / H / K\n(pathology location)", fc="white", ec=GREEN, fontsize=7)
    arrow(ax, right(fused), (42, 25.5))
    arrow(ax, right(g_head), left(g_logit))
    arrow(ax, right(g_logit), left(g_soft))

    ax.text(70, 4, "At test time, the side without a real paired sample is filled with a fixed population PROTOTYPE (mean of train-split windows) —\n"
                   "PADS test samples pair with a gait_prototype, GaitRec test samples pair with a finger_prototype (no sample has both, §1).",
            fontsize=7, color=RED, ha="center", style="italic")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run(args):
    root = Path(args.data_root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    fig_encoders(figdir / "pipeline_encoders.png")
    print("Saved figures/pipeline_encoders.png")
    fig_cbdl(figdir / "pipeline_cbdl.png")
    print("Saved figures/pipeline_cbdl.png")
    fig_final_output(figdir / "pipeline_final_output.png")
    print("Saved figures/pipeline_final_output.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
