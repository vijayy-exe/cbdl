"""
Computes every real, model-derived value needed for one PADS test subject's
CBDL Motor Symptom Monitoring Report, and writes them to a JSON file that
the HTML report template reads.

Every number in the output is either:
  (a) a genuine model output for this specific subject (calibrated
      classification probabilities, this subject's own Cross-Body lag
      weights, this subject's own SHAP attribution), or
  (b) a real signal-processing measurement on this subject's own raw
      waveform (tremor-band FFT power, movement-amplitude RMS,
      cross-channel correlation), reported as a percentile against the
      TRAIN-split population so it has a stated reference frame, or
  (c) explicitly marked as unavailable / population-substituted when it
      genuinely is (no gait data for this subject, single-session dataset
      so no trend, mPower branch not connected downstream) — never
      fabricated to fill a template slot.

Usage:
    python generate_patient_report_data.py --subject-id 037 --data-root "."
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from cross_body_module import LaggedCrossAttention
from model import CBDLPhase2Model
from train_phase4_clinical_head import ClassifierHead, get_device, load_pads, K_WINDOWS
from train_phase5_calibration import apply_calibrators

CHANNEL_NAMES = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]
TASK_BOUNDARIES = [("PointFinger", 0, 976), ("TouchIndex", 976, 1952), ("TouchNose", 1952, 2928)]
CLASS_NAMES = ["Healthy", "Parkinson's Disease", "Other Movement Disorder"]
SAMPLE_RATE_HZ = 976 / (2928 / 3) * (2928 / 3) / 1  # placeholder overwritten below if a real rate is known
# PADS raw tasks are 1024 samples before the 48-sample trim; sampling rate is
# documented by PADS as 100 Hz for the movement sensors.
FS_HZ = 100.0


class PADSClinicalPipeline(nn.Module):
    def __init__(self, model, cbdm, pads_head, gait_prototype):
        super().__init__()
        self.model = model
        self.cbdm = cbdm
        self.pads_head = pads_head
        self.register_buffer("gait_prototype", gait_prototype)

    def forward(self, x):
        mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
        fw = self.model.encode_finger_windows(x, mask, source="pads", k=K_WINDOWS)
        gp = self.gait_prototype.expand(fw.size(0), -1, -1)
        fused, lag_weights = self.cbdm(fw, gp)
        return self.pads_head(fused), lag_weights


def load_demographics(root: Path) -> pd.DataFrame:
    path = root / "physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/file_list.csv"
    df = pd.read_csv(path)
    df = df[df["resource_type"] == "patient"].copy()
    df["subject_id"] = df["id"].astype(str).str.zfill(3)
    return df.set_index("subject_id")


def tremor_band_power_ratio(x: np.ndarray, fs: float = FS_HZ, band=(3.5, 7.0)) -> float:
    """Share of accelerometer-magnitude spectral power in the classic 3.5-7Hz
    parkinsonian resting-tremor band, out of total power (0-fs/2)."""
    accel = x[0:3, :]  # Accel X/Y/Z
    mag = np.sqrt((accel ** 2).sum(axis=0))
    mag = mag - mag.mean()
    freqs = np.fft.rfftfreq(len(mag), d=1.0 / fs)
    power = np.abs(np.fft.rfft(mag)) ** 2
    total = power.sum() + 1e-12
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(power[band_mask].sum() / total)


def movement_amplitude_rms(x: np.ndarray) -> float:
    """RMS of gyroscope magnitude — a movement-amplitude/speed proxy (lower = slower/more bradykinetic)."""
    gyro = x[3:6, :]
    mag = np.sqrt((gyro ** 2).sum(axis=0))
    return float(np.sqrt((mag ** 2).mean()))


def coordination_cross_correlation(x: np.ndarray) -> float:
    """Mean |correlation| between the 3 accelerometer and 3 gyroscope channels
    — a rough inter-channel coordination proxy (higher = more coupled movement)."""
    corrs = []
    for i in range(6):
        for j in range(i + 1, 6):
            c = np.corrcoef(x[i], x[j])[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))
    return float(np.mean(corrs)) if corrs else 0.0


def percentile_of(value: float, population: np.ndarray) -> float:
    return float((population < value).mean() * 100)


def grade_from_percentile(pct: float) -> str:
    if pct >= 75:
        return "High"
    if pct >= 40:
        return "Moderate"
    return "Low"


def run(args):
    device = get_device()
    root = Path(args.data_root)

    pads_w, pads_m, pads_y, pads_split = load_pads(root)
    labels_df = pd.read_csv(root / "pads_preprocessed" / "pads_labels.csv", dtype={"subject_id": str})
    labels_df["subject_id"] = labels_df["subject_id"].str.zfill(3)
    demo = load_demographics(root)

    subject_id = args.subject_id.zfill(3)
    row_idx = labels_df.index[labels_df["subject_id"] == subject_id]
    if len(row_idx) == 0:
        raise ValueError(f"Subject {subject_id} not found in pads_labels.csv")
    idx = row_idx[0]
    if labels_df.loc[idx, "split"] != "test":
        print(f"WARNING: subject {subject_id} is in split='{labels_df.loc[idx, 'split']}', not 'test'. Proceeding anyway.")

    model = CBDLPhase2Model().to(device)
    model.load_state_dict(torch.load(root / "phase2_checkpoint.pt", map_location=device)["model"])
    cbdm = LaggedCrossAttention(embed_dim=128, lags=[0, 1, 2, 3]).to(device)
    cbdm.load_state_dict(torch.load(root / "phase3_checkpoint.pt", map_location=device)["cbdm"])
    phase4_ckpt = torch.load(root / "phase4_checkpoint.pt", map_location=device)
    pads_head = ClassifierHead(128, 64, 3).to(device)
    pads_head.load_state_dict(phase4_ckpt["pads_head"])
    gait_prototype = phase4_ckpt["gait_prototype"].to(device)

    pipeline = PADSClinicalPipeline(model, cbdm, pads_head, gait_prototype).to(device)
    pipeline.eval()
    for p in pipeline.parameters():
        p.requires_grad = False

    x_np = pads_w[idx]  # [6, 2928]
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits, lag_weights = pipeline(x)
        raw_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        lag_weights_np = lag_weights.cpu().numpy()[0]

    with open(root / "phase5_calibration.pkl", "rb") as f:
        cal = pickle.load(f)
    calibrated_probs = apply_calibrators(cal["calibrators"], raw_probs[None, :])[0]

    # ── Per-subject SHAP (small nsamples — one subject, needs to be fast) ──
    train_idx = np.where(pads_split == "train")[0]
    rng = np.random.RandomState(42)
    background_idx = rng.choice(train_idx, size=min(30, len(train_idx)), replace=False)
    background = torch.from_numpy(pads_w[background_idx]).float().to(device)

    class LogitsOnly(nn.Module):
        def __init__(self, pipe):
            super().__init__()
            self.pipe = pipe

        def forward(self, x):
            logits, _ = self.pipe(x)
            return logits

    logits_only = LogitsOnly(pipeline).to(device)
    explainer = shap.GradientExplainer(logits_only, background)
    shap_values = explainer.shap_values(x, nsamples=40)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    pred_class = int(raw_probs.argmax())
    sample_shap = shap_values[0, :, :, pred_class]  # [6, 2928]
    abs_shap = np.abs(sample_shap)

    channel_importance = abs_shap.sum(axis=1)
    channel_importance = (channel_importance / channel_importance.sum()).tolist()

    task_importance = []
    for name, lo, hi in TASK_BOUNDARIES:
        task_importance.append(float(abs_shap[:, lo:hi].sum()))
    total = sum(task_importance)
    task_importance = [v / total for v in task_importance]

    top_feature_idx = int(np.argmax(channel_importance))

    # ── DSP-derived motor pattern proxies, percentile-normed against TRAIN population ──
    train_waveforms = pads_w[train_idx]
    tremor_pop = np.array([tremor_band_power_ratio(w) for w in train_waveforms])
    amplitude_pop = np.array([movement_amplitude_rms(w) for w in train_waveforms])
    coord_pop = np.array([coordination_cross_correlation(w) for w in train_waveforms])

    tremor_val = tremor_band_power_ratio(x_np)
    amplitude_val = movement_amplitude_rms(x_np)
    coord_val = coordination_cross_correlation(x_np)

    tremor_pct = percentile_of(tremor_val, tremor_pop)
    amplitude_pct = percentile_of(amplitude_val, amplitude_pop)
    coord_pct = percentile_of(coord_val, coord_pop)
    # Lower movement amplitude percentile => more bradykinetic => report as inverted severity
    bradykinesia_severity_pct = 100 - amplitude_pct

    demographics = {}
    if subject_id in demo.index:
        d = demo.loc[subject_id]
        demographics = {
            "age": int(d["age"]) if pd.notna(d["age"]) else None,
            "gender": str(d["gender"]) if pd.notna(d["gender"]) else None,
            "height_cm": int(d["height"]) if pd.notna(d["height"]) else None,
            "weight_kg": int(d["weight"]) if pd.notna(d["weight"]) else None,
            "handedness": str(d["handedness"]) if pd.notna(d["handedness"]) else None,
            "condition_notes": str(d["disease_comment"]) if pd.notna(d.get("disease_comment")) and d["disease_comment"] != "-" else None,
        }

    true_label = labels_df.loc[idx, "condition"]

    n_sessions_for_subject = int((labels_df["subject_id"] == subject_id).sum())

    output = {
        "subject_id": subject_id,
        "demographics": demographics,
        "n_sessions_available": n_sessions_for_subject,
        "true_label_for_validation_only": true_label,  # kept for internal validation, not necessarily shown
        "diagnostic": {
            "class_names": CLASS_NAMES,
            "raw_probabilities": raw_probs.tolist(),
            "calibrated_probabilities": calibrated_probs.tolist(),
            "predicted_class": CLASS_NAMES[pred_class],
            "predicted_class_confidence_calibrated": float(calibrated_probs[pred_class]),
        },
        "cross_body": {
            "lags": [0, 1, 2, 3],
            "this_subject_lag_weights": lag_weights_np.tolist(),
            "dominant_lag": int(np.argmax(lag_weights_np)),
            "cohort_lag_weights_reference": None,  # filled from phase3_checkpoint.pt below
        },
        "explainability": {
            "channel_names": CHANNEL_NAMES,
            "channel_importance": channel_importance,
            "task_names": [t[0] for t in TASK_BOUNDARIES],
            "task_importance": task_importance,
            "top_channel": CHANNEL_NAMES[top_feature_idx],
        },
        "motor_patterns": {
            "tremor_band_power_ratio": tremor_val,
            "tremor_percentile_vs_population": tremor_pct,
            "tremor_severity": grade_from_percentile(tremor_pct),
            "movement_amplitude_rms": amplitude_val,
            "movement_amplitude_percentile_vs_population": amplitude_pct,
            "bradykinesia_severity_percentile": bradykinesia_severity_pct,
            "bradykinesia_severity": grade_from_percentile(bradykinesia_severity_pct),
            "coordination_cross_correlation": coord_val,
            "coordination_percentile_vs_population": coord_pct,
        },
        "data_coverage": {
            "pads_finger_imu": "Available — real sensor data for this subject",
            "gaitrec_gait": "NOT available for this subject — GaitRec and PADS share no subjects. "
                             "Cross-body analysis uses a fixed, tier-agnostic population gait prototype, not this subject's own gait.",
            "mpower_tapping_features": "NOT available for this subject — mPower's branch is trained but not "
                                         "connected to this classifier's decision path (see documentation §3.3/§7.2).",
        },
        "trend": {
            "available": n_sessions_for_subject > 1,
            "note": "Single-session dataset — PADS records exactly one session per subject, so no "
                    "longitudinal trend can be computed for this or any subject." if n_sessions_for_subject <= 1
                    else "Multiple sessions found.",
        },
    }

    phase3_ckpt = torch.load(root / "phase3_checkpoint.pt", map_location="cpu")
    output["cross_body"]["cohort_lag_weights_reference"] = phase3_ckpt["lag_heatmap"]

    out_path = root / f"patient_report_{subject_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")
    print(json.dumps(output, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", type=str, required=True)
    p.add_argument("--data-root", type=str, default=".")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
