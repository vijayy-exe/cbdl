"""
Stage 2e — Missing Data Handling for the CBDL Pipeline.

Linearly interpolates NaN gaps ≤ 200 ms; flags windows with longer gaps
as "unreliable" and excludes them from training.
"""

import numpy as np
from typing import Dict, Tuple


def interpolate_short_gaps(
    data: np.ndarray,
    max_gap_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate NaN gaps shorter than max_gap_samples in a 2-D array.

    Gaps longer than max_gap_samples are left as NaN and should be flagged.

    Args:
        data: Array of shape (C, T) with possible NaNs.
        max_gap_samples: Maximum gap length (in samples) to interpolate.

    Returns:
        Tuple of:
          - filled: Array of shape (C, T) with short NaN gaps filled.
          - gap_lengths: Array of shape (C,) with the max NaN gap per channel.
    """
    filled = data.copy()
    gap_lengths = np.zeros(data.shape[0], dtype=np.int64)

    for ch in range(data.shape[0]):
        channel = filled[ch]
        nan_mask = np.isnan(channel)
        if not nan_mask.any():
            continue
        indices = np.arange(len(channel))
        # Find runs of NaN
        runs = _find_nan_runs(nan_mask)
        max_gap = 0
        for (start, end) in runs:
            run_len = end - start
            max_gap = max(max_gap, run_len)
            if run_len <= max_gap_samples:
                # Linear interpolation across this gap
                left_idx = start - 1
                right_idx = end
                if left_idx >= 0 and right_idx < len(channel):
                    left_val = channel[left_idx]
                    right_val = channel[right_idx]
                    interp_vals = np.linspace(left_val, right_val, run_len + 2)[1:-1]
                    channel[start:end] = interp_vals
                elif left_idx < 0 and right_idx < len(channel):
                    channel[start:end] = channel[right_idx]
                elif left_idx >= 0:
                    channel[start:end] = channel[left_idx]
                else:
                    channel[start:end] = 0.0
        gap_lengths[ch] = max_gap

    return filled, gap_lengths


def _find_nan_runs(nan_mask: np.ndarray) -> list:
    """
    Identify contiguous runs of True in a boolean array.

    Args:
        nan_mask: 1-D boolean array.

    Returns:
        List of (start, end) tuples (half-open intervals).
    """
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(nan_mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(nan_mask)))
    return runs


def handle_missing_data(
    windowed_streams: Dict[str, np.ndarray],
    reliable_mask: np.ndarray,
    max_gap_ms: float = 200.0,
    fs: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Interpolate short NaN gaps and apply reliable window mask to all streams.

    Args:
        windowed_streams: Dict mapping stream name → (N_windows, C, T).
        reliable_mask: Boolean array (N_windows,). True = keep window.
        max_gap_ms: Max NaN gap in ms to interpolate.
        fs: Sampling frequency in Hz.

    Returns:
        Dict mapping stream name → (N_reliable, C, T) with short NaNs filled.
    """
    max_gap_samples = int(max_gap_ms / 1000.0 * fs)
    result: Dict[str, np.ndarray] = {}

    for name, windows in windowed_streams.items():
        # Apply reliable mask first
        windows = windows[reliable_mask]
        filled_windows = np.empty_like(windows)
        for i in range(len(windows)):
            filled, _ = interpolate_short_gaps(windows[i], max_gap_samples)
            filled_windows[i] = filled
        # Replace remaining NaNs with 0 (very long gaps already excluded by mask)
        filled_windows = np.nan_to_num(filled_windows, nan=0.0)
        result[name] = filled_windows

    return result
