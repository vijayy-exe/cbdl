"""
Stage 2b — Stream Synchronization for the CBDL Pipeline.

Aligns all sensor streams to a common 100 Hz clock using
cross-correlation of shared gait heel-strike events.
Returns a NaN mask for missing samples after alignment.
"""

import numpy as np
from typing import Dict, Tuple


def detect_heel_strikes(
    gait_data: np.ndarray,
    fs: float = 100.0,
    threshold_factor: float = 1.5,
) -> np.ndarray:
    """
    Detect heel-strike events from gait IMU data using threshold crossing.

    Uses the vertical acceleration channel (channel 2) to find
    impact events exceeding threshold_factor × std above mean.

    Args:
        gait_data: Array of shape (C, T) — gait stream.
        fs: Sampling frequency in Hz.
        threshold_factor: Multiplier on std for heel-strike detection.

    Returns:
        Array of sample indices where heel strikes occur.
    """
    # Use z-axis accelerometer (channel 2) — vertical impact
    acc_z = gait_data[2].copy()
    # Replace NaN with 0 for peak detection
    acc_z = np.nan_to_num(acc_z, nan=0.0)
    threshold = np.mean(np.abs(acc_z)) + threshold_factor * np.std(acc_z)
    # Find peaks above threshold with minimum distance of 0.3s
    min_dist = int(0.3 * fs)
    peaks = []
    i = 0
    while i < len(acc_z):
        if np.abs(acc_z[i]) > threshold:
            # Find local max in window
            window_end = min(i + min_dist, len(acc_z))
            local_max = i + np.argmax(np.abs(acc_z[i:window_end]))
            peaks.append(local_max)
            i = local_max + min_dist
        else:
            i += 1
    return np.array(peaks, dtype=np.int64)


def compute_cross_correlation_lag(
    ref_signal: np.ndarray,
    target_signal: np.ndarray,
    max_lag_samples: int = 50,
) -> int:
    """
    Compute the lag between ref_signal and target_signal via cross-correlation.

    Args:
        ref_signal: Reference 1-D signal.
        target_signal: Target 1-D signal to align.
        max_lag_samples: Maximum lag to consider (in samples).

    Returns:
        Integer lag (positive = target leads ref, negative = target lags ref).
    """
    ref = np.nan_to_num(ref_signal)
    tgt = np.nan_to_num(target_signal)
    # Normalize
    ref = (ref - ref.mean()) / (ref.std() + 1e-8)
    tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-8)
    corr = np.correlate(ref, tgt, mode="full")
    center = len(ref) - 1
    search_range = corr[center - max_lag_samples: center + max_lag_samples + 1]
    lag = np.argmax(search_range) - max_lag_samples
    return int(lag)


def shift_stream(
    data: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shift a stream by lag samples and produce a NaN mask for the gap.

    Args:
        data: Array of shape (C, T).
        lag: Integer sample lag (positive shifts right, negative shifts left).

    Returns:
        Tuple of:
          - shifted: Array of shape (C, T) with boundary NaN-padded.
          - nan_mask: Boolean array of shape (T,) marking shifted-in NaN positions.
    """
    C, T = data.shape
    shifted = np.full_like(data, np.nan)
    nan_mask = np.zeros(T, dtype=bool)

    if lag >= 0:
        shifted[:, lag:] = data[:, :T - lag] if lag < T else data
        nan_mask[:lag] = True
    else:
        abs_lag = -lag
        shifted[:, :T - abs_lag] = data[:, abs_lag:]
        nan_mask[T - abs_lag:] = True

    return shifted, nan_mask


def synchronize_streams(
    streams: Dict[str, np.ndarray],
    fs: float = 100.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Align all sensor streams to the gait stream's heel-strike timing.

    Uses cross-correlation between each stream's first channel and the
    gait reference channel to estimate and correct temporal offsets.

    Args:
        streams: Dictionary mapping stream name → np.ndarray (C, T).
        fs: Sampling frequency in Hz.

    Returns:
        Tuple of:
          - synced_streams: Dict with same keys, streams shifted to align.
          - nan_masks: Dict mapping stream name → bool mask (T,) of NaN gaps.
    """
    reference = streams["gait"][0]  # Use gait acc-x as reference
    synced: Dict[str, np.ndarray] = {}
    nan_masks: Dict[str, np.ndarray] = {}

    for name, data in streams.items():
        if name == "gait":
            synced[name] = data.copy()
            nan_masks[name] = np.isnan(data[0])
            continue
        lag = compute_cross_correlation_lag(reference, data[0])
        shifted, mask = shift_stream(data, lag)
        synced[name] = shifted
        nan_masks[name] = mask | np.isnan(data[0])

    return synced, nan_masks
