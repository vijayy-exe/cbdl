"""
Stage 2a — Noise Filtering for the CBDL Pipeline.

Applies a Butterworth bandpass filter (0.5–20 Hz) and a notch filter
at 50 Hz to each sensor channel using scipy.signal.
"""

import numpy as np
from scipy import signal
from typing import Tuple


def design_bandpass_filter(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter.

    Args:
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Tuple (b, a) of filter coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def design_notch_filter(
    notch_freq: float,
    fs: float,
    quality_factor: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design an IIR notch filter.

    Args:
        notch_freq: Frequency to notch out in Hz.
        fs: Sampling frequency in Hz.
        quality_factor: Quality factor of the notch.

    Returns:
        Tuple (b, a) of filter coefficients.
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    return b, a


def apply_filter(
    data: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """
    Apply a zero-phase filter (filtfilt) to each channel, handling NaNs.

    NaN gaps are linearly interpolated before filtering, then restored.

    Args:
        data: Array of shape (C, T) with possible NaNs.
        b: Numerator filter coefficients.
        a: Denominator filter coefficients.

    Returns:
        Filtered array of shape (C, T), NaNs restored to original positions.
    """
    result = np.empty_like(data)
    for ch in range(data.shape[0]):
        channel = data[ch].copy()
        nan_mask = np.isnan(channel)
        if nan_mask.any():
            indices = np.arange(len(channel))
            valid = ~nan_mask
            if valid.sum() >= 2:
                channel[nan_mask] = np.interp(indices[nan_mask], indices[valid], channel[valid])
            else:
                result[ch] = data[ch]
                continue
        filtered = signal.filtfilt(b, a, channel)
        filtered[nan_mask] = np.nan
        result[ch] = filtered
    return result


def filter_stream(
    data: np.ndarray,
    fs: float = 100.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 20.0,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """
    Apply bandpass (0.5–20 Hz) and notch (50 Hz) filters to a sensor stream.

    Args:
        data: Array of shape (C, T) with possible NaNs.
        fs: Sampling frequency in Hz.
        bandpass_low: Lower bandpass cutoff in Hz.
        bandpass_high: Upper bandpass cutoff in Hz.
        notch_freq: Notch filter frequency in Hz.

    Returns:
        Filtered array of shape (C, T).
    """
    b_bp, a_bp = design_bandpass_filter(bandpass_low, bandpass_high, fs)
    b_notch, a_notch = design_notch_filter(notch_freq, fs)
    data = apply_filter(data, b_bp, a_bp)
    data = apply_filter(data, b_notch, a_notch)
    return data
