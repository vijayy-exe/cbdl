"""
Stage 2c — Sliding Window Segmentation for the CBDL Pipeline.

Segments each sensor stream into overlapping windows of 2 s (200 samples)
with a stride of 0.5 s (50 samples).
"""

import numpy as np
from typing import Dict, List, Tuple


def window_stream(
    data: np.ndarray,
    window_size: int = 200,
    stride: int = 50,
) -> np.ndarray:
    """
    Segment a single sensor stream into overlapping windows.

    Args:
        data: Array of shape (C, T).
        window_size: Number of samples per window.
        stride: Number of samples between window starts.

    Returns:
        Array of shape (N_windows, C, window_size).
    """
    C, T = data.shape
    n_windows = (T - window_size) // stride + 1
    windows = np.zeros((n_windows, C, window_size), dtype=data.dtype)
    for i in range(n_windows):
        start = i * stride
        windows[i] = data[:, start: start + window_size]
    return windows


def window_all_streams(
    streams: Dict[str, np.ndarray],
    window_size: int = 200,
    stride: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Apply sliding window segmentation to all streams.

    Args:
        streams: Dictionary mapping stream name → (C, T).
        window_size: Samples per window.
        stride: Stride in samples.

    Returns:
        Dictionary mapping stream name → (N_windows, C, window_size).
    """
    return {
        name: window_stream(data, window_size, stride)
        for name, data in streams.items()
    }


def compute_reliable_mask(
    streams: Dict[str, np.ndarray],
    window_size: int = 200,
    stride: int = 50,
    max_nan_gap_ms: float = 200.0,
    fs: float = 100.0,
) -> np.ndarray:
    """
    Compute a reliability mask: True for windows that have no NaN gap
    longer than `max_nan_gap_ms` milliseconds in any stream.

    Args:
        streams: Dictionary mapping stream name → (C, T) with possible NaNs.
        window_size: Samples per window.
        stride: Stride in samples.
        max_nan_gap_ms: Maximum tolerable NaN gap in milliseconds.
        fs: Sampling frequency in Hz.

    Returns:
        Boolean array of shape (N_windows,). True = reliable.
    """
    max_gap_samples = int(max_nan_gap_ms / 1000.0 * fs)
    C_total = sum(v.shape[0] for v in streams.values())
    T = next(iter(streams.values())).shape[1]
    n_windows = (T - window_size) // stride + 1
    reliable = np.ones(n_windows, dtype=bool)

    for name, data in streams.items():
        for ch in range(data.shape[0]):
            for i in range(n_windows):
                start = i * stride
                seg = data[ch, start: start + window_size]
                nan_flags = np.isnan(seg)
                if not nan_flags.any():
                    continue
                # Find longest consecutive NaN run
                max_run = _max_consecutive_true(nan_flags)
                if max_run > max_gap_samples:
                    reliable[i] = False
    return reliable


def _max_consecutive_true(arr: np.ndarray) -> int:
    """
    Return the length of the longest consecutive True run in a boolean array.

    Args:
        arr: 1-D boolean array.

    Returns:
        Integer length of longest True run.
    """
    if not arr.any():
        return 0
    max_run = 0
    current_run = 0
    for v in arr:
        if v:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run
