"""
Generates every figure referenced in CBDL_PROJECT_DOCUMENTATION.md, saved as
PNG files under figures/. Run this after Phases 2-5 have all produced their
checkpoints/logs.

Usage:
    python generate_figures.py --data-root "."
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cross_body_module import LaggedCrossAttention
from model import CBDLPhase2Model
from train_phase4_clinical_head import ClassifierHead, get_device, load_pads, load_gaitrec, K_WINDOWS

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


def run(args):
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
