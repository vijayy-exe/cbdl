"""
Stage 1 — Synthetic Data Generator for the CBDL Pipeline.

Generates 6 sensor streams for N virtual subjects (healthy + symptomatic)
at 100 Hz, 60 seconds each, with realistic noise and NaN dropouts.
Saves one .npz file per subject under data/synthetic/.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
from tqdm import tqdm


# ──────────────────────────────────────────────
#  Constants / stream config
# ──────────────────────────────────────────────

STREAM_CHANNELS: Dict[str, int] = {
    "finger": 6,           # acc xyz + gyro xyz
    "wrist": 6,            # acc xyz + gyro xyz
    "gait": 6,             # acc xyz + gyro xyz
    "insole_pressure": 4,  # heel, toe, lat, med
    "phone_acc_gyro": 6,   # acc xyz + gyro xyz
    "physio": 4,           # HR, HRV, skin_temp, resp
}


def _rng(seed: int) -> np.random.Generator:
    """Return a seeded numpy Generator."""
    return np.random.default_rng(seed)


# ──────────────────────────────────────────────
#  Per-stream signal generators
# ──────────────────────────────────────────────

def _generate_finger(
    n_samples: int,
    fs: float,
    symptomatic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate finger IMU (acc + gyro) time-series.

    Symptomatic subjects have a dominant tremor frequency of 3–7 Hz;
    healthy subjects have a smooth low-amplitude motion.

    Args:
        n_samples: Number of time samples.
        fs: Sampling frequency in Hz.
        symptomatic: True if the subject is symptomatic.
        rng: Seeded random generator.

    Returns:
        Array of shape (6, n_samples).
    """
    t = np.arange(n_samples) / fs
    data = np.zeros((6, n_samples))

    if symptomatic:
        tremor_freq = rng.uniform(3.0, 7.0)
        tremor_amp = rng.uniform(0.3, 0.8)
        for ch in range(3):
            phase = rng.uniform(0, 2 * np.pi)
            data[ch] = tremor_amp * np.sin(2 * np.pi * tremor_freq * t + phase)
            # Add harmonics
            data[ch] += 0.2 * tremor_amp * np.sin(4 * np.pi * tremor_freq * t)
        for ch in range(3, 6):
            phase = rng.uniform(0, 2 * np.pi)
            data[ch] = 0.5 * tremor_amp * np.sin(2 * np.pi * tremor_freq * t + phase)
    else:
        for ch in range(6):
            base_freq = rng.uniform(0.5, 1.5)
            amp = rng.uniform(0.05, 0.15)
            data[ch] = amp * np.sin(2 * np.pi * base_freq * t + rng.uniform(0, 2 * np.pi))

    # Gaussian sensor noise
    data += rng.normal(0, 0.02, size=data.shape)
    return data.astype(np.float32)


def _generate_wrist(
    n_samples: int,
    fs: float,
    symptomatic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate wrist IMU time-series with bradykinesia-style slow ramps
    and dyskinesia bursts for symptomatic subjects.

    Args:
        n_samples: Number of time samples.
        fs: Sampling frequency in Hz.
        symptomatic: True if the subject is symptomatic.
        rng: Seeded random generator.

    Returns:
        Array of shape (6, n_samples).
    """
    t = np.arange(n_samples) / fs
    data = np.zeros((6, n_samples))

    if symptomatic:
        # Slow movement ramp
        ramp_period = rng.uniform(3.0, 8.0)
        for ch in range(3):
            data[ch] = 0.3 * np.sin(2 * np.pi * t / ramp_period)
        # Inject dyskinesia bursts
        n_bursts = rng.integers(3, 8)
        burst_starts = rng.integers(0, n_samples - 50, n_bursts)
        for burst in burst_starts:
            burst_len = rng.integers(20, 60)
            end = min(burst + burst_len, n_samples)
            burst_freq = rng.uniform(4.0, 8.0)
            burst_t = np.arange(end - burst) / fs
            for ch in range(3):
                data[ch, burst:end] += 0.5 * np.sin(2 * np.pi * burst_freq * burst_t)
        for ch in range(3, 6):
            data[ch] = 0.3 * np.sin(2 * np.pi * t / (ramp_period * 1.2) + 0.5)
    else:
        for ch in range(6):
            data[ch] = 0.1 * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t)

    data += rng.normal(0, 0.02, size=data.shape)
    return data.astype(np.float32)


def _generate_gait(
    n_samples: int,
    fs: float,
    symptomatic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate shoe-IMU gait time-series with stride cycles.

    Symptomatic subjects have shuffled cadence and reduced swing amplitude.

    Args:
        n_samples: Number of time samples.
        fs: Sampling frequency in Hz.
        symptomatic: True if the subject is symptomatic.
        rng: Seeded random generator.

    Returns:
        Array of shape (6, n_samples).
    """
    t = np.arange(n_samples) / fs
    data = np.zeros((6, n_samples))

    if symptomatic:
        cadence = rng.uniform(0.8, 1.2)   # steps/s (slow)
        swing_amp = rng.uniform(0.2, 0.5)
    else:
        cadence = rng.uniform(1.5, 2.0)   # steps/s (normal)
        swing_amp = rng.uniform(0.8, 1.2)

    # Primary stride cycle
    for ch in range(3):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = swing_amp * np.sin(2 * np.pi * cadence * t + phase)
        # Shuffle cadence — add small frequency variation
        if symptomatic:
            jitter = 0.05 * np.cumsum(rng.normal(0, 0.001, n_samples))
            data[ch] += 0.2 * np.sin(2 * np.pi * cadence * (t + jitter) + phase)
    for ch in range(3, 6):
        data[ch] = 0.5 * swing_amp * np.sin(2 * np.pi * 2 * cadence * t + rng.uniform(0, np.pi))

    data += rng.normal(0, 0.025, size=data.shape)
    return data.astype(np.float32)


def _generate_insole_pressure(
    n_samples: int,
    fs: float,
    symptomatic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate insole pressure (heel, toe, lat, med) time-series.

    Symptomatic subjects exhibit left/right weight asymmetry.

    Args:
        n_samples: Number of time samples.
        fs: Sampling frequency in Hz.
        symptomatic: True if the subject is symptomatic.
        rng: Seeded random generator.

    Returns:
        Array of shape (4, n_samples).
    """
    t = np.arange(n_samples) / fs
    data = np.zeros((4, n_samples))

    gait_freq = rng.uniform(0.8, 1.8) if not symptomatic else rng.uniform(0.5, 1.0)

    if symptomatic:
        # Asymmetric weighting: heel-heavy
        weights = [0.8, 0.3, 0.6, 0.4]
        asym = rng.uniform(0.3, 0.6)   # asymmetry factor
    else:
        weights = [0.5, 0.5, 0.5, 0.5]
        asym = rng.uniform(0.0, 0.1)

    for ch, w in enumerate(weights):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = w * (0.5 + 0.5 * np.sin(2 * np.pi * gait_freq * t + phase))
        data[ch] += asym * np.sin(2 * np.pi * gait_freq * 0.5 * t + phase * 0.5)

    data = np.clip(data, 0, None)  # Pressure is non-negative
    data += rng.normal(0, 0.01, size=data.shape)
    return data.astype(np.float32)


def _generate_phone_acc_gyro(
    n_samples: int,
    fs: float,
    symptomatic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate smartphone accelerometer and gyroscope (trunk sway) time-series.

    Args:
        n_samples: Number of time samples.
        fs: Sampling frequency in Hz.
        symptomatic: True if the subject is symptomatic.
        rng: Seeded random generator.

    Returns:
        Array of shape (6, n_samples).
    """
    t = np.arange(n_samples) / fs
    data = np.zeros((6, n_samples))

    if symptomatic:
        sway_amp = rng.uniform(0.3, 0.7)
        drift_rate = rng.uniform(0.05, 0.15)
    else:
        sway_amp = rng.uniform(0.05, 0.15)
        drift_rate = rng.uniform(0.0, 0.02)

    for ch in range(3):
        freq = rng.uniform(0.1, 0.5)
        data[ch] = sway_amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        data[ch] += drift_rate * t / (t[-1] + 1e-6)  # slow postural drift

    for ch in range(3, 6):
        data[ch] = 0.5 * sway_amp * np.sin(2 * np.pi * rng.uniform(0.08, 0.3) * t)

    data += rng.normal(0, 0.015, size=data.shape)
    return data.astype(np.float32)


def _generate_physio(
    n_samples: int,
    fs: float,
    symptomatic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate physiological signals: HR (bpm), HRV (ms), skin_temp (°C), resp (Hz).

    Args:
        n_samples: Number of time samples.
        fs: Sampling frequency in Hz.
        symptomatic: True if the subject is symptomatic.
        rng: Seeded random generator.

    Returns:
        Array of shape (4, n_samples).
    """
    t = np.arange(n_samples) / fs
    data = np.zeros((4, n_samples))

    # HR: slow drift around mean
    hr_mean = rng.uniform(65, 90) if not symptomatic else rng.uniform(75, 100)
    data[0] = hr_mean + 5 * np.sin(2 * np.pi * 0.02 * t) + rng.normal(0, 1.0, n_samples)

    # HRV: inversely related to HR
    hrv_base = rng.uniform(20, 60) if not symptomatic else rng.uniform(10, 30)
    data[1] = hrv_base + 3 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 2.0, n_samples)

    # Skin temp: very slow drift
    temp_base = rng.uniform(32.0, 36.0)
    data[2] = temp_base + 0.2 * np.sin(2 * np.pi * 0.005 * t) + rng.normal(0, 0.05, n_samples)

    # Respiration: 0.2–0.4 Hz
    resp_freq = rng.uniform(0.2, 0.4)
    data[3] = np.sin(2 * np.pi * resp_freq * t) + rng.normal(0, 0.05, n_samples)

    return data.astype(np.float32)


# ──────────────────────────────────────────────
#  NaN dropout + timestamp jitter
# ──────────────────────────────────────────────

def _inject_dropout(
    data: np.ndarray,
    dropout_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly set packet dropouts (NaNs) in time-series data.

    Args:
        data: Array of shape (C, T).
        dropout_rate: Fraction of samples to drop (e.g. 0.02 = 2%).
        rng: Seeded random generator.

    Returns:
        Array of same shape with NaN dropout injected.
    """
    data = data.copy()
    mask = rng.random(data.shape[1]) < dropout_rate
    data[:, mask] = np.nan
    return data


def _generate_timestamps(
    n_samples: int,
    fs: float,
    jitter_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate timestamps with small random jitter.

    Args:
        n_samples: Number of time samples.
        fs: Sampling frequency in Hz.
        jitter_ms: Standard deviation of jitter in milliseconds.
        rng: Seeded random generator.

    Returns:
        Array of shape (n_samples,) with timestamps in seconds.
    """
    base = np.arange(n_samples) / fs
    jitter = rng.normal(0, jitter_ms / 1000.0, n_samples)
    jitter[0] = 0.0
    return (base + jitter).astype(np.float32)


# ──────────────────────────────────────────────
#  Per-subject generation
# ──────────────────────────────────────────────

def generate_subject(
    subject_id: int,
    symptomatic: bool,
    fs: float = 100.0,
    duration: float = 60.0,
    rng: np.random.Generator = None,
) -> Dict[str, Any]:
    """
    Generate all sensor streams and metadata for a single virtual subject.

    Args:
        subject_id: Integer subject index.
        symptomatic: True if the subject is symptomatic (used for signal shaping
                     only, NOT stored as a training label).
        fs: Sampling frequency in Hz.
        duration: Recording duration in seconds.
        rng: Seeded random generator (if None, creates from subject_id).

    Returns:
        Dictionary with keys:
          - 'finger', 'wrist', 'gait', 'insole_pressure',
            'phone_acc_gyro', 'physio': np.ndarray (C, T)
          - 'timestamps': np.ndarray (T,)
          - 'metadata': dict with age, gender, dominant_hand,
            baseline_profile, updrs, fall_risk, motor_severity, label
    """
    if rng is None:
        rng = _rng(subject_id)

    n_samples = int(fs * duration)
    dropout_rate = rng.uniform(0.02, 0.05)

    streams = {
        "finger":          _inject_dropout(_generate_finger(n_samples, fs, symptomatic, rng), dropout_rate, rng),
        "wrist":           _inject_dropout(_generate_wrist(n_samples, fs, symptomatic, rng), dropout_rate, rng),
        "gait":            _inject_dropout(_generate_gait(n_samples, fs, symptomatic, rng), dropout_rate, rng),
        "insole_pressure": _inject_dropout(_generate_insole_pressure(n_samples, fs, symptomatic, rng), dropout_rate, rng),
        "phone_acc_gyro":  _inject_dropout(_generate_phone_acc_gyro(n_samples, fs, symptomatic, rng), dropout_rate, rng),
        "physio":          _inject_dropout(_generate_physio(n_samples, fs, symptomatic, rng), dropout_rate, rng),
    }

    timestamps = _generate_timestamps(n_samples, fs, jitter_ms=5.0, rng=rng)

    # Clinical metadata (EVALUATION ONLY — not used during training)
    age = float(rng.integers(40, 81))
    gender = float(rng.integers(0, 2))              # 0=M, 1=F
    dominant_hand = float(rng.integers(0, 2))       # 0=R, 1=L
    baseline_profile = rng.normal(0, 1, 8).astype(np.float32)

    if symptomatic:
        updrs = float(rng.uniform(15.0, 60.0))
        fall_risk = float(rng.uniform(4.0, 10.0))
        motor_severity = float(rng.integers(2, 5))
    else:
        updrs = float(rng.uniform(0.0, 14.0))
        fall_risk = float(rng.uniform(0.0, 3.9))
        motor_severity = float(rng.integers(0, 2))

    metadata = {
        "subject_id": subject_id,
        "label": int(symptomatic),              # stored for evaluation ONLY
        "age": age,
        "gender": gender,
        "dominant_hand": dominant_hand,
        "baseline_profile": baseline_profile,
        "updrs": updrs,
        "fall_risk": fall_risk,
        "motor_severity": motor_severity,
    }

    return {**streams, "timestamps": timestamps, "metadata": metadata}


# ──────────────────────────────────────────────
#  Dataset-level generator
# ──────────────────────────────────────────────

def generate_synthetic_data(
    n_subjects: int = 200,
    fs: float = 100.0,
    duration: float = 60.0,
    output_dir: str = "data/synthetic",
    seed: int = 42,
) -> None:
    """
    Generate synthetic sensor data for all virtual subjects and save to disk.

    Half of the subjects are healthy, half are symptomatic.
    Each subject is saved as a .npz file under `output_dir`.

    Args:
        n_subjects: Total number of subjects (half healthy, half symptomatic).
        fs: Sampling frequency in Hz.
        duration: Recording duration in seconds per subject.
        output_dir: Directory path to save .npz files.
        seed: Global random seed.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    master_rng = _rng(seed)

    n_symptomatic = n_subjects // 2
    labels = [False] * (n_subjects - n_symptomatic) + [True] * n_symptomatic
    # Shuffle so healthy and symptomatic are interleaved
    master_rng.shuffle(labels)

    print(f"\n[Stage 1] Generating synthetic data for {n_subjects} subjects…")
    for i, symptomatic in tqdm(enumerate(labels), total=n_subjects, desc="Subjects"):
        subject_rng = _rng(seed * 1000 + i)
        subject = generate_subject(
            subject_id=i,
            symptomatic=symptomatic,
            fs=fs,
            duration=duration,
            rng=subject_rng,
        )
        meta = subject.pop("metadata")
        save_path = os.path.join(output_dir, f"subject_{i:03d}.npz")
        np.savez_compressed(
            save_path,
            **subject,
            **{f"meta_{k}": v for k, v in meta.items()},
        )

    print(f"[Stage 1] Saved {n_subjects} .npz files to '{output_dir}'")
    print(f"          Symptomatic: {sum(labels)}, Healthy: {n_subjects - sum(labels)}")


if __name__ == "__main__":
    generate_synthetic_data()
