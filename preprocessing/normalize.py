"""
Stage 2d — Per-Subject Z-Score Normalization for the CBDL Pipeline.

Computes per-channel mean and std from the subject's baseline windows
(first 20% of windows) and applies Z-score normalization.
"""

import numpy as np
from typing import Dict, Tuple


def compute_baseline_stats(
    windowed_streams: Dict[str, np.ndarray],
    baseline_fraction: float = 0.2,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-channel mean and std from baseline (first N%) windows.

    Args:
        windowed_streams: Dict mapping stream name → (N_windows, C, T).
        baseline_fraction: Fraction of windows used as baseline (default 0.2).

    Returns:
        Dict mapping stream name → (mean, std), each of shape (C,).
    """
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, windows in windowed_streams.items():
        n_baseline = max(1, int(windows.shape[0] * baseline_fraction))
        baseline = windows[:n_baseline]  # (n_baseline, C, T)
        # Flatten time and baseline dimensions together
        flat = baseline.reshape(n_baseline, windows.shape[1], -1)
        mean = np.nanmean(flat, axis=(0, 2))   # (C,)
        std = np.nanstd(flat, axis=(0, 2))     # (C,)
        std = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
        stats[name] = (mean, std)
    return stats


def normalize_streams(
    windowed_streams: Dict[str, np.ndarray],
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Apply Z-score normalization to each stream using precomputed stats.

    Args:
        windowed_streams: Dict mapping stream name → (N_windows, C, T).
        stats: Dict mapping stream name → (mean (C,), std (C,)).

    Returns:
        Dict mapping stream name → normalized (N_windows, C, T).
    """
    normalized: Dict[str, np.ndarray] = {}
    for name, windows in windowed_streams.items():
        mean, std = stats[name]
        # Broadcast (C,) → (1, C, 1) for correct subtraction
        normalized[name] = (windows - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
    return normalized


def normalize_subject(
    windowed_streams: Dict[str, np.ndarray],
    baseline_fraction: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute baseline stats and normalize all streams for a single subject.

    Args:
        windowed_streams: Dict mapping stream name → (N_windows, C, T).
        baseline_fraction: Fraction of windows used as baseline.

    Returns:
        Tuple of:
          - normalized_streams: Dict of normalized (N_windows, C, T) arrays.
          - stats: Dict of (mean, std) tuples per stream for later use.
    """
    stats = compute_baseline_stats(windowed_streams, baseline_fraction)
    normalized = normalize_streams(windowed_streams, stats)
    return normalized, stats
