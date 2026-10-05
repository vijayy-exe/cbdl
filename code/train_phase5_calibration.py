"""
CBDL Phase 5 (Track B) — Calibration & Uncertainty.

Per CBDL_Development_Plan.md Phase 5: Isotonic Regression or Platt Scaling
on validation-set predicted probabilities, plus one reliability-diagram
figure. Bootstrap confidence intervals explicitly skipped per the plan
("skip unless there's spare time").

Calibrates the PADS clinical head (the PRIMARY classifier from Phase 4) —
a raw softmax output is NOT the same as a calibrated probability (a model
that says "70% confident" should be right about 70% of the time it says
that; an uncalibrated softmax usually isn't). This fits a per-class
Isotonic Regression mapping raw softmax probability -> calibrated
probability, using the VALIDATION split only (never test, to keep the test
numbers honest), then evaluates Expected Calibration Error (ECE) and a
reliability diagram before vs. after, on the held-out TEST split.

Usage:
    python train_phase5_calibration.py --data-root "."
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from cross_body_module import LaggedCrossAttention
from model import CBDLPhase2Model
from train_phase4_clinical_head import ClassifierHead, get_device, load_pads, K_WINDOWS


def get_pads_probs_and_labels(model, cbdm, pads_head, gait_prototype, pads_w, pads_m, pads_y, pads_split, split, device):
    idx = np.where(pads_split == split)[0]
    x = torch.from_numpy(pads_w[idx]).float().to(device)
    m = torch.from_numpy(pads_m[idx]).bool().to(device)
    with torch.no_grad():
        fw = model.encode_finger_windows(x, m, source="pads", k=K_WINDOWS)
        gp = gait_prototype.to(device).expand(fw.size(0), -1, -1)
        fused, _ = cbdm(fw, gp)
        logits = pads_head(fused)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs, pads_y[idx]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Standard multi-class ECE: bins by the model's top predicted-class confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "acc": None, "conf": None})
            continue
        acc = correct[in_bin].mean()
        conf = confidences[in_bin].mean()
        ece += (n_in_bin / len(confidences)) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "n": int(n_in_bin), "acc": float(acc), "conf": float(conf)})
    return float(ece), bins


def fit_isotonic_calibrators(val_probs: np.ndarray, val_labels: np.ndarray, n_classes: int) -> list[IsotonicRegression]:
    """One-vs-rest isotonic regression per class, fit on VALIDATION only."""
    calibrators = []
    for c in range(n_classes):
        targets = (val_labels == c).astype(np.float64)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(val_probs[:, c], targets)
        calibrators.append(ir)
    return calibrators


def apply_calibrators(calibrators: list[IsotonicRegression], probs: np.ndarray) -> np.ndarray:
    calibrated = np.stack([ir.predict(probs[:, c]) for c, ir in enumerate(calibrators)], axis=1)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return calibrated / row_sums  # renormalize to sum to 1, standard post-isotonic step


def run(args) -> None:
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
    gait_prototype = phase4_ckpt["gait_prototype"]

    model.eval(); cbdm.eval(); pads_head.eval()
    print("Loaded Phase 2/3/4 checkpoints (frozen).")

    val_probs, val_labels = get_pads_probs_and_labels(model, cbdm, pads_head, gait_prototype, pads_w, pads_m, pads_y, pads_split, "val", device)
    test_probs, test_labels = get_pads_probs_and_labels(model, cbdm, pads_head, gait_prototype, pads_w, pads_m, pads_y, pads_split, "test", device)

    ece_before, bins_before = expected_calibration_error(test_probs, test_labels, n_bins=args.n_bins)
    print(f"ECE before calibration (test): {ece_before:.4f}")

    calibrators = fit_isotonic_calibrators(val_probs, val_labels, n_classes=3)
    test_probs_calibrated = apply_calibrators(calibrators, test_probs)
    ece_after, bins_after = expected_calibration_error(test_probs_calibrated, test_labels, n_bins=args.n_bins)
    print(f"ECE after calibration (test):  {ece_after:.4f}")

    acc_before = (test_probs.argmax(1) == test_labels).mean()
    acc_after = (test_probs_calibrated.argmax(1) == test_labels).mean()
    print(f"Accuracy before/after calibration: {acc_before:.3f} / {acc_after:.3f}. Note: per-class "
          f"isotonic regression is fit independently per class, so it is NOT guaranteed to preserve "
          f"argmax ranking — accuracy can shift as a side effect, especially with a validation set "
          f"this small (70 samples / 3 classes). This is a real limitation of calibrating on so little "
          f"data, not a bug — see documentation.")

    with open(root / "phase5_calibration.pkl", "wb") as f:
        pickle.dump({
            "calibrators": calibrators,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "bins_before": bins_before,
            "bins_after": bins_after,
            "test_probs_raw": test_probs,
            "test_probs_calibrated": test_probs_calibrated,
            "test_labels": test_labels,
        }, f)
    print(f"\nSaved calibration model + reliability data to {root / 'phase5_calibration.pkl'}")

    print("\n" + "=" * 70)
    print("PHASE 5 CALIBRATION — SUMMARY")
    print("=" * 70)
    print(f"PADS clinical head Expected Calibration Error: {ece_before:.4f} -> {ece_after:.4f} "
          f"({'improved' if ece_after < ece_before else 'no improvement'})")
    print("Method: per-class (one-vs-rest) Isotonic Regression, fit on VALIDATION split only, "
          "applied to TEST split. Bootstrap confidence intervals skipped per the development plan.")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--n-bins", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
