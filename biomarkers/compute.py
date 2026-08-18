"""
Stage 6 — Novel Digital Biomarker Computation for the CBDL Pipeline.

Computes 6 biomarkers per subject from the trained cross-body model:
  1. Cross-Body Dependency Index (CBDi)
  2. Finger–Gait Coupling Score
  3. Neuromotor Stability Score
  4. Coordination Index
  5. Motor Variability Index
  6. Synchronization Score

Aggregates per-subject biomarkers and saves to /output/biomarkers.csv.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy as scipy_entropy


def compute_cbdi(dep_matrix: np.ndarray) -> float:
    """
    Cross-Body Dependency Index: mean absolute off-diagonal energy of 4×4 matrix.

    Args:
        dep_matrix: Dependency matrix of shape (4, 4) or (N_windows, 4, 4).

    Returns:
        Scalar CBDi value.
    """
    if dep_matrix.ndim == 3:
        dep_matrix = dep_matrix.mean(axis=0)
    D = dep_matrix.shape[0]
    mask = ~np.eye(D, dtype=bool)
    return float(np.mean(np.abs(dep_matrix[mask])))


def compute_finger_gait_coupling(
    finger_emb: np.ndarray,
    gait_emb: np.ndarray,
    lag_weights: np.ndarray,
) -> float:
    """
    Finger–Gait Coupling Score: cosine similarity weighted by learned lag attention.

    Args:
        finger_emb: Finger embeddings of shape (N_windows, D) or (D,).
        gait_emb: Gait embeddings of shape (N_windows, D) or (D,).
        lag_weights: Lag attention weights of shape (N_windows, L) or (L,).

    Returns:
        Scalar coupling score.
    """
    if finger_emb.ndim == 1:
        finger_emb = finger_emb[np.newaxis]
        gait_emb = gait_emb[np.newaxis]
        lag_weights = lag_weights[np.newaxis]

    # Normalize
    fn = finger_emb / (np.linalg.norm(finger_emb, axis=-1, keepdims=True) + 1e-8)
    gn = gait_emb / (np.linalg.norm(gait_emb, axis=-1, keepdims=True) + 1e-8)
    cos_sim = (fn * gn).sum(axis=-1)  # (N_windows,)

    # Weight by maximum lag attention
    lag_confidence = lag_weights.max(axis=-1)  # (N_windows,) — confidence in lag estimate
    weighted_coupling = (cos_sim * lag_confidence).sum() / (lag_confidence.sum() + 1e-8)
    return float(weighted_coupling)


def compute_neuromotor_stability(cross_body_embs: np.ndarray) -> float:
    """
    Neuromotor Stability Score: 1 / variance of cross-body embedding across windows.

    Args:
        cross_body_embs: Cross-body embeddings of shape (N_windows, D).

    Returns:
        Scalar stability score (higher = more stable).
    """
    if cross_body_embs.ndim == 1 or cross_body_embs.shape[0] < 2:
        return 0.0
    var = np.var(cross_body_embs, axis=0).mean()
    return float(1.0 / (var + 1e-6))


def compute_coordination_index(attn_weights: np.ndarray) -> float:
    """
    Coordination Index: entropy of the cross-limb attention weight distribution.

    Higher entropy = more distributed attention = better coordination.

    Args:
        attn_weights: Attention weights of shape (4, 4) or (N_windows, 4, 4).

    Returns:
        Scalar coordination index.
    """
    if attn_weights.ndim == 3:
        attn_weights = attn_weights.mean(axis=0)
    flat = attn_weights.flatten()
    flat = flat / (flat.sum() + 1e-8)  # normalize to probability
    return float(scipy_entropy(flat + 1e-10))


def compute_motor_variability(
    stream_embs: Dict[str, np.ndarray],
) -> float:
    """
    Motor Variability Index: mean std of per-stream embeddings across windows.

    Args:
        stream_embs: Dict mapping stream name → (N_windows, D) embeddings.

    Returns:
        Scalar motor variability index.
    """
    stds = []
    for emb in stream_embs.values():
        if emb.ndim == 2 and emb.shape[0] > 1:
            stds.append(np.std(emb, axis=0).mean())
    return float(np.mean(stds)) if stds else 0.0


def compute_synchronization_score(
    finger_emb: np.ndarray,
    gait_emb: np.ndarray,
    lag_weights: np.ndarray,
) -> float:
    """
    Synchronization Score: max cross-correlation peak between finger and gait
    embedding norms at the learned lag.

    Args:
        finger_emb: (N_windows, D) or (D,) finger embeddings.
        gait_emb: (N_windows, D) or (D,) gait embeddings.
        lag_weights: (N_windows, L) or (L,) lag weights.

    Returns:
        Scalar synchronization score.
    """
    if finger_emb.ndim == 1:
        return 0.0

    f_norm = np.linalg.norm(finger_emb, axis=-1)  # (N_windows,)
    g_norm = np.linalg.norm(gait_emb, axis=-1)    # (N_windows,)

    # Normalize signals
    f_norm = (f_norm - f_norm.mean()) / (f_norm.std() + 1e-8)
    g_norm = (g_norm - g_norm.mean()) / (g_norm.std() + 1e-8)

    # Cross-correlation
    corr = np.correlate(f_norm, g_norm, mode="full")
    max_corr = float(np.max(np.abs(corr))) / (len(f_norm) + 1e-8)

    # Weight by lag confidence
    if lag_weights.ndim == 2:
        lag_confidence = lag_weights.max(axis=-1).mean()
    else:
        lag_confidence = lag_weights.max()

    return float(max_corr * lag_confidence)


def compute_subject_biomarkers(
    subject_id: int,
    cross_body_embs: np.ndarray,
    finger_embs: np.ndarray,
    gait_embs: np.ndarray,
    balance_embs: np.ndarray,
    sensor_embs: np.ndarray,
    dep_matrices: np.ndarray,
    attn_weights: np.ndarray,
    lag_weights: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all 6 biomarkers for a single subject.

    Args:
        subject_id: Integer subject identifier.
        cross_body_embs: (N_windows, 512) cross-body embeddings.
        finger_embs: (N_windows, 128) finger stream embeddings.
        gait_embs: (N_windows, 128) gait stream embeddings.
        balance_embs: (N_windows, 128) balance stream embeddings.
        sensor_embs: (N_windows, 128) sensor/wrist stream embeddings.
        dep_matrices: (N_windows, 4, 4) dependency matrices.
        attn_weights: (N_windows, 4, 4) attention weight matrices.
        lag_weights: (N_windows, L) lag attention weights.

    Returns:
        Dict with keys: subject_id, cbdi, finger_gait_coupling,
        neuromotor_stability, coordination_index, motor_variability,
        synchronization_score.
    """
    stream_embs = {
        "finger": finger_embs,
        "gait": gait_embs,
        "balance": balance_embs,
        "sensor": sensor_embs,
    }

    return {
        "subject_id": subject_id,
        "cbdi": compute_cbdi(dep_matrices),
        "finger_gait_coupling": compute_finger_gait_coupling(finger_embs, gait_embs, lag_weights),
        "neuromotor_stability": compute_neuromotor_stability(cross_body_embs),
        "coordination_index": compute_coordination_index(attn_weights),
        "motor_variability": compute_motor_variability(stream_embs),
        "synchronization_score": compute_synchronization_score(finger_embs, gait_embs, lag_weights),
    }


def save_biomarkers_csv(
    records: List[Dict],
    output_path: str = "output/biomarkers.csv",
) -> pd.DataFrame:
    """
    Save biomarker records to a CSV file.

    Args:
        records: List of dicts, each from compute_subject_biomarkers plus metadata.
        output_path: Path to save the CSV.

    Returns:
        DataFrame of all biomarkers.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"[Stage 6] Saved biomarkers for {len(df)} subjects → {output_path}")
    return df
