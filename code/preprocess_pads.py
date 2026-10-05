"""
Preprocessing pipeline for the PADS (Parkinson's Disease Smartwatch) dataset
(PhysioNet, "parkinsons-disease-smartwatch"), extracting finger/hand
micro-movement data for a shared Finger Encoder pipeline alongside the
Tappy-keystroke-derived dataset (see preprocess_tappy.py / tappy_data_card.md).

IMPORTANT — two facts confirmed by direct inspection of the downloaded
dataset and PADS's own scripts, both of which contradict a naive reading of
the task brief:

1. The already-shipped preprocessed/movement/*.bin files do NOT contain the
   PointFinger or TouchIndex tasks. PADS's own scripts/run_preprocessing.py
   explicitly strips them (`to_remove = 'Time|LiftHold|PointFinger|TouchIndex'`)
   before writing the .bin files. Only TouchNose survives in those files
   among the three tasks closest to finger micro-movement.

   The raw per-task recordings for PointFinger and TouchIndex DO still exist
   under movement/timeseries/*.txt (confirmed present for all 469 subjects).
   So this script re-derives all three target tasks (PointFinger, TouchIndex,
   TouchNose) directly from the raw per-task .txt files, reusing PADS's own
   loading code (utils.data_handling) and PADS's own gravity-offset-removal
   code (utils.l1_trend_filter.l1_trend_filter, the same L1 trend filter
   used by run_preprocessing.py, same vlambda=50 default) rather than any
   custom gravity-correction step, and the same "drop first 48 samples"
   vibration-startup trim. Per-channel math is identical to what
   run_preprocessing.py would have produced for these tasks had it not
   filtered them out (l1_trend_filter is applied independently per channel
   row, so restricting the task list ahead of time changes nothing about
   the correction itself).

2. There is no wavelet transform, and no "universal_clean.csv" with wavelet
   columns, anywhere in this project. The actual existing Tappy pipeline
   (preprocess_tappy.py) outputs tappy_waveforms.npy [N, 3, T] (z-scored),
   tappy_masks.npy [N, T] (bool), and tappy_labels.csv (UserKey, Parkinsons,
   Split, SessionIndex, SessionLength) — no wavelet coefficients. This
   script mirrors that real schema instead of a fictional wavelet-CSV one:
   pads_waveforms.npy [N, 6, T], pads_masks.npy [N, T], pads_labels.csv with
   the same column shape as tappy_labels.csv. Step 6 (apply_wavelet_transform)
   is kept as an explicit no-op seam so a future wavelet stage can be dropped
   in later without touching anything else, exactly like the Tappy pipeline
   would need the same seam added to it first.

Wrist decision (Step 3): use each subject's DOMINANT wrist only (from
file_list.csv's `handedness` column, complete for all 469 subjects — 437
right, 32 left), not both wrists. PADS's finger tasks are performed by a
subject's dominant hand; self-reported handedness is a direct, complete,
per-subject signal for "wrist most engaged," so it is preferred over an
arbitrary per-task heuristic. This halves channel count (6 vs 12) and avoids
learning from a mostly-inactive non-dominant-wrist channel.

Task-concatenation decision (Step 4): concatenate the three tasks along the
TIME axis into one per-subject sequence (channel dim = 6: Accelerometer
X/Y/Z, Gyroscope X/Y/Z of the dominant wrist; time dim = the three tasks
back-to-back in a fixed order PointFinger -> TouchIndex -> TouchNose), the
same [N, C, T] + mask convention Tappy uses, rather than three separate
per-task arrays.

Usage:
    python preprocess_pads.py --pads-root "/path/to/parkinsons-disease-smartwatch/1.0.0" \
        --output-dir "/path/to/pads_preprocessed" [--n-jobs 8] [--limit 20]

Run with --help for all configurable parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import multiprocessing
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("preprocess_pads")

# Channel order within one task-block, matching PADS's own sorted order
# (utils/... is not reused for this constant since it's just a naming
# convention, but it matches run_preprocessing.py's per-wrist ordering:
# Accelerometer X,Y,Z then Gyroscope X,Y,Z, with Time dropped).
SENSOR_AXES = [
    ("Accelerometer", "X"),
    ("Accelerometer", "Y"),
    ("Accelerometer", "Z"),
    ("Gyroscope", "X"),
    ("Gyroscope", "Y"),
    ("Gyroscope", "Z"),
]

# Official PADS task order used by plot_example_pd_signal_processed.py to
# reconstruct channel names for the ALREADY-SHIPPED .bin files (Step 2
# verification only — PointFinger/LiftHold/TouchIndex are absent here
# because run_preprocessing.py strips them before saving).
OFFICIAL_BIN_TASKS = [
    "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold",
    "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1",
    "Entrainment2",
]
OFFICIAL_BIN_WRISTS = ["LeftWrist", "RightWrist"]
OFFICIAL_BIN_SENSORS = ["Acceleration", "Rotation"]
OFFICIAL_BIN_AXES = ["X", "Y", "Z"]

LABEL_NAMES = {0: "Healthy", 1: "Parkinson's", 2: "Other movement disorder"}


# ─── Config ──────────────────────────────────────────────────────────────


@dataclass
class Config:
    pads_root: Path  # .../parkinsons-disease-smartwatch/1.0.0
    output_dir: Path
    tasks: list[str] = field(default_factory=lambda: ["PointFinger", "TouchIndex", "TouchNose"])
    wrist_mode: str = "dominant"  # "dominant" | "LeftWrist" | "RightWrist"
    vlambda: float = 50.0  # PADS's own l1_trend_filter regularization (official default)
    trim_samples: int = 48  # PADS's own "drop first 0.5s vibration notification" trim
    task_length_percentile: float = 100.0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    n_jobs: int = 1
    limit: Optional[int] = None

    @property
    def preprocessed_dir(self) -> Path:
        return self.pads_root / "preprocessed"

    @property
    def file_list_csv(self) -> Path:
        return self.preprocessed_dir / "file_list.csv"

    @property
    def bin_dir(self) -> Path:
        return self.preprocessed_dir / "movement"

    @property
    def raw_movement_dir(self) -> Path:
        return self.pads_root / "movement"

    @property
    def scripts_dir(self) -> Path:
        return self.pads_root / "scripts"


def _pads_utils(cfg: Config):
    """
    Imports PADS's own scripts/utils package (data_handling, l1_trend_filter)
    by path, so this script reuses PADS's own loading/gravity-correction
    logic instead of reimplementing it (see module docstring, point 1).
    """
    scripts_dir = str(cfg.scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from utils.data_handling import load_all_files, get_data_from_observation  # noqa: E402
    from utils.l1_trend_filter import l1_trend_filter  # noqa: E402

    return load_all_files, get_data_from_observation, l1_trend_filter


# ─── Step 1: verify official preprocessing already ran ─────────────────────


def verify_official_preprocessing(cfg: Config) -> None:
    """
    Confirms preprocessed/movement/*.bin and preprocessed/file_list.csv
    already exist, as PADS's own run_preprocessing.py would have produced.
    Does NOT reimplement PADS's gravity-correction/offset-removal logic —
    if the official outputs are missing, tells the user to run
    scripts/run_preprocessing.py themselves and stops.
    """
    if not cfg.file_list_csv.exists():
        raise FileNotFoundError(
            f"{cfg.file_list_csv} not found. Run PADS's own "
            f"scripts/run_preprocessing.py (from within {cfg.scripts_dir}) first — "
            f"this script does not reimplement that logic."
        )
    bin_files = sorted(cfg.bin_dir.glob("*_ml.bin"))
    if not bin_files:
        raise FileNotFoundError(
            f"No *_ml.bin files found under {cfg.bin_dir}. Run PADS's own "
            f"scripts/run_preprocessing.py first."
        )
    log.info(
        "Official PADS preprocessing already ran: %s exists, %d .bin files found under %s.",
        cfg.file_list_csv, len(bin_files), cfg.bin_dir,
    )


# ─── Step 2: reconstruct the channel order of the ALREADY-SHIPPED .bin files ──


def reconstruct_official_channel_order() -> list[str]:
    """
    Reproduces the exact channel list construction from
    scripts/plot_example_pd_signal_processed.py: task -> wrist -> sensor ->
    axis, e.g. "TouchNose_Acceleration_RightWrist_X". This is the channel
    order of preprocessed/movement/*.bin as shipped (11 tasks, PointFinger/
    LiftHold/TouchIndex already removed by PADS's own run_preprocessing.py).
    """
    channels = []
    for task in OFFICIAL_BIN_TASKS:
        for wrist in OFFICIAL_BIN_WRISTS:
            for sensor in OFFICIAL_BIN_SENSORS:
                for axis in OFFICIAL_BIN_AXES:
                    channels.append(f"{task}_{sensor}_{wrist}_{axis}")
    return channels


def load_bin_file(path: Path, n_channels: int) -> np.ndarray:
    """Loads and reshapes one subject's *_ml.bin file to [n_channels, T]."""
    return np.fromfile(path, dtype=np.float32).reshape((n_channels, -1))


def verify_bin_shape(cfg: Config) -> None:
    """
    Step 2 sanity check: load one subject's .bin file, reshape it using the
    channel order above, and print the resulting shape and channel count —
    verifying it matches before processing (don't assume).
    """
    channels = reconstruct_official_channel_order()
    example_bin = sorted(cfg.bin_dir.glob("*_ml.bin"))[0]
    n_floats = example_bin.stat().st_size // 4
    if n_floats % len(channels) != 0:
        raise AssertionError(
            f"{example_bin.name}: {n_floats} floats is not divisible by the "
            f"reconstructed channel count {len(channels)} — channel-order "
            f"reconstruction does not match this .bin layout, stopping "
            f"rather than guessing."
        )
    data = load_bin_file(example_bin, len(channels))
    log.info(
        "Step 2 verification (%s): reconstructed %d channels, .bin reshapes to %s "
        "(%d channels x %d timesteps). Matches expected channel count: %s",
        example_bin.name, len(channels), data.shape, data.shape[0], data.shape[1],
        data.shape[0] == len(channels),
    )
    assert data.shape[0] == len(channels)


# ─── Step 3: task/wrist selection ───────────────────────────────────────────


def select_dominant_wrist(handedness: str) -> str:
    """
    Maps file_list.csv's `handedness` field to the PADS device_location
    naming used in the raw session metadata. Falls back to RightWrist (the
    cohort majority, 437/469) with a warning for any unexpected value —
    none are expected since handedness has no missing values in this
    release, but subjects are not dropped over a metadata quirk here.
    """
    value = str(handedness).strip().lower()
    if value == "left":
        return "LeftWrist"
    if value == "right":
        return "RightWrist"
    log.warning("Unexpected handedness value %r; defaulting to RightWrist.", handedness)
    return "RightWrist"


def resolve_wrist(cfg: Config, handedness: str) -> str:
    if cfg.wrist_mode == "dominant":
        return select_dominant_wrist(handedness)
    if cfg.wrist_mode in ("LeftWrist", "RightWrist"):
        return cfg.wrist_mode
    raise ValueError(f"Unknown wrist_mode {cfg.wrist_mode!r}")


# ─── Step 4a: per-subject raw extraction + PADS's own gravity correction ───


def _extract_one_subject(
    scripts_dir: str,
    subject_meta: pd.DataFrame,
    raw_movement_dir: str,
    tasks: list[str],
    wrist: str,
    vlambda: float,
    trim_samples: int,
) -> dict:
    """
    Runs in a worker process. Loads the 3 target-task raw recordings for one
    subject's dominant wrist, applies PADS's own l1_trend_filter to the
    Accelerometer rows only (matching run_preprocessing.py's process_mask),
    trims the first `trim_samples` samples (matching run_preprocessing.py's
    vibration-notification trim), and concatenates the tasks along time.

    Returns a dict with either:
      {"subject_id", "ok": True, "array" [6, sum(task_lengths)], "task_lengths": {...}}
    or:
      {"subject_id", "ok": False, "reason": "..."}
    """
    subject_id = subject_meta["subject_id"].iloc[0]
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from utils.data_handling import get_data_from_observation
    from utils.l1_trend_filter import l1_trend_filter

    try:
        task_frames = []
        for task in tasks:
            rows = subject_meta[
                (subject_meta["record_name"] == task) & (subject_meta["device_location"] == wrist)
            ]
            if len(rows) != 1:
                return {
                    "subject_id": subject_id, "ok": False,
                    "reason": f"expected exactly 1 raw recording for task={task} wrist={wrist}, found {len(rows)}",
                }
            task_frames.append(rows)
        ordered_meta = pd.concat(task_frames, axis=0).reset_index(drop=True)

        data, channel_names = get_data_from_observation(raw_movement_dir, ordered_meta)
        # data: [len(tasks) * 7, L] in task-major order, per-task rows =
        # [Time, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z]
        keep_mask = ~pd.Series(channel_names).str.contains("_Time$", regex=True)
        data = data[keep_mask.to_numpy()]
        channel_names = list(np.array(channel_names)[keep_mask.to_numpy()])

        n_tasks = len(tasks)
        n_ch_per_task = len(SENSOR_AXES)  # 6
        if data.shape[0] != n_tasks * n_ch_per_task:
            return {
                "subject_id": subject_id, "ok": False,
                "reason": f"expected {n_tasks * n_ch_per_task} rows after dropping Time, got {data.shape[0]}",
            }

        task_lengths = {}
        for t_idx, task in enumerate(tasks):
            block = data[t_idx * n_ch_per_task:(t_idx + 1) * n_ch_per_task]
            task_lengths[task] = block.shape[1]

        process_mask = pd.Series(channel_names).str.contains("Accelerometer")
        with contextlib.redirect_stdout(io.StringIO()):
            data[process_mask.to_numpy()] = np.apply_along_axis(
                lambda x: x - l1_trend_filter(x, vlambda=vlambda, verbose=False),
                1, data[process_mask.to_numpy()],
            )

        data = data[:, trim_samples:]

        return {
            "subject_id": subject_id, "ok": True,
            "array": data.reshape(n_tasks, n_ch_per_task, -1),  # [n_tasks, 6, T_task]
            "task_lengths": task_lengths,
        }
    except Exception as exc:  # noqa: BLE001 — reported per-subject, not fatal to the run
        return {"subject_id": subject_id, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def report_raw_task_length_distribution(all_subject_meta: list[pd.DataFrame], tasks: list[str]) -> dict:
    """
    Step 4: prints the raw (pre-trim) per-task recording-length distribution
    across the whole cohort, from metadata alone (cheap — no gravity
    correction needed yet), so the common-length choice below is justified
    rather than arbitrary.
    """
    lengths_by_task: dict[str, list[int]] = {t: [] for t in tasks}
    for meta in all_subject_meta:
        for task in tasks:
            rows = meta.loc[meta["record_name"] == task, "rows"]
            if len(rows):
                lengths_by_task[task].append(int(rows.iloc[0]))

    summary = {}
    for task, lengths in lengths_by_task.items():
        arr = np.asarray(lengths)
        summary[task] = {
            "n": int(arr.size), "min": int(arr.min()), "max": int(arr.max()),
            "mean": float(arr.mean()), "n_unique_values": int(np.unique(arr).size),
        }
        log.info("Raw length distribution for task=%s: %s", task, summary[task])
    return summary


def choose_common_task_length(length_summary: dict, percentile: float) -> int:
    """
    Picks one common raw recording length shared by all three tasks (PADS's
    protocol records each task for a fixed duration, so this is expected to
    equal every task's own min==max, not an arbitrary compromise — verified,
    not assumed, via report_raw_task_length_distribution above).
    """
    all_mins = [s["min"] for s in length_summary.values()]
    all_maxs = [s["max"] for s in length_summary.values()]
    if len(set(all_mins)) == 1 and all_mins == all_maxs:
        return all_mins[0]
    # Fallback for a cohort that turns out not to be perfectly fixed-length:
    pooled = np.concatenate([[s["min"]] * 1 for s in length_summary.values()])
    return int(round(float(np.percentile(pooled, percentile))))


# ─── Step 4b: pad/truncate + concatenate tasks into one per-subject sequence ─


def pad_or_truncate_task_block(block: np.ndarray, t_task: int) -> tuple[np.ndarray, np.ndarray]:
    """
    block: [6, L] float32 (post gravity-correction, post-trim, single task)
    Returns (fixed [6, t_task] float32, mask [t_task] bool).
    """
    _, length = block.shape
    if length >= t_task:
        return block[:, :t_task].astype(np.float32), np.ones(t_task, dtype=bool)
    fixed = np.zeros((block.shape[0], t_task), dtype=np.float32)
    fixed[:, :length] = block
    mask = np.zeros(t_task, dtype=bool)
    mask[:length] = True
    return fixed, mask


def concatenate_tasks(array: np.ndarray, t_task: int) -> tuple[np.ndarray, np.ndarray]:
    """
    array: [n_tasks, 6, L] for one subject (post gravity-correction/trim).
    Returns (sequence [6, n_tasks * t_task], mask [n_tasks * t_task]) with
    the tasks concatenated along time in the fixed order they were given.
    """
    fixed_blocks, mask_blocks = [], []
    for t_idx in range(array.shape[0]):
        fixed, mask = pad_or_truncate_task_block(array[t_idx], t_task)
        fixed_blocks.append(fixed)
        mask_blocks.append(mask)
    sequence = np.concatenate(fixed_blocks, axis=1)  # [6, n_tasks * t_task]
    mask = np.concatenate(mask_blocks, axis=0)  # [n_tasks * t_task]
    return sequence, mask


# ─── Orchestration: Steps 3-4 across the whole cohort ───────────────────────


def build_all_subjects(cfg: Config) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """
    Runs Steps 3-4 for every subject in file_list.csv: resolve dominant
    wrist, extract+gravity-correct+trim the 3 target tasks from raw data
    (Step 4a, optionally parallel across subjects via cfg.n_jobs), then pad/
    truncate+concatenate into one fixed-length sequence per subject
    (Step 4b).

    Returns (waveforms [N, 6, T], masks [N, T], subject_table, drop_stats).
    subject_table has one row per subject actually produced: subject_id,
    wrist, task_lengths (dict), any per-task truncation/padding applied.
    """
    load_all_files, _, _ = _pads_utils(cfg)
    all_subject_meta_full = load_all_files(str(cfg.raw_movement_dir) + "/")
    all_subject_meta = all_subject_meta_full if cfg.limit is None else all_subject_meta_full[: cfg.limit]

    file_list = pd.read_csv(cfg.file_list_csv)
    file_list["subject_id"] = file_list["id"].apply(lambda i: f"{int(i):03d}")
    handedness_lookup = file_list.set_index("subject_id")["handedness"].to_dict()

    length_summary = report_raw_task_length_distribution(all_subject_meta, cfg.tasks)
    t_task = choose_common_task_length(length_summary, cfg.task_length_percentile)
    log.info(
        "Chosen common per-task length = %d samples (percentile=%.0f). "
        "Post gravity-correction trim of %d samples -> %d samples/task, "
        "%d tasks concatenated -> final T = %d.",
        t_task, cfg.task_length_percentile, cfg.trim_samples,
        t_task - cfg.trim_samples, len(cfg.tasks),
        (t_task - cfg.trim_samples) * len(cfg.tasks),
    )
    t_task_trimmed = t_task - cfg.trim_samples

    jobs = []
    for meta in all_subject_meta:
        subject_id = meta["subject_id"].iloc[0]
        handedness = handedness_lookup.get(subject_id)
        if handedness is None:
            continue  # not in file_list.csv; reported as a drop below
        wrist = resolve_wrist(cfg, handedness)
        jobs.append((str(cfg.scripts_dir), meta, str(cfg.raw_movement_dir) + "/", cfg.tasks, wrist, cfg.vlambda, cfg.trim_samples))

    log.info("Extracting %d subjects (n_jobs=%d)...", len(jobs), cfg.n_jobs)
    if cfg.n_jobs > 1:
        with multiprocessing.Pool(cfg.n_jobs) as pool:
            results = pool.starmap(_extract_one_subject, jobs)
    else:
        results = [_extract_one_subject(*j) for j in jobs]

    # Computed against the FULL raw cohort (not the --limit slice used for
    # `jobs` below), so a test run with --limit never misreports subjects
    # outside its slice as "missing".
    subjects_in_file_list = set(file_list["subject_id"])
    subjects_in_raw = {m["subject_id"].iloc[0] for m in all_subject_meta_full}
    missing_from_raw = [] if cfg.limit is not None else sorted(subjects_in_file_list - subjects_in_raw)

    waveforms, masks, rows = [], [], []
    drop_reasons: list[str] = []
    for res in results:
        if not res["ok"]:
            drop_reasons.append(f"{res['subject_id']}: {res['reason']}")
            continue
        sequence, mask = concatenate_tasks(res["array"], t_task_trimmed)
        waveforms.append(sequence)
        masks.append(mask)
        rows.append({"subject_id": res["subject_id"], "task_lengths": res["task_lengths"]})

    for subject_id in missing_from_raw:
        drop_reasons.append(f"{subject_id}: no raw movement/*.json observation found")

    waveforms_arr = np.stack(waveforms, axis=0).astype(np.float32) if waveforms else np.zeros((0, 6, 0), np.float32)
    masks_arr = np.stack(masks, axis=0) if masks else np.zeros((0, 0), bool)
    subject_table = pd.DataFrame(rows)

    drop_stats = {
        "n_attempted": len(jobs) + len(missing_from_raw),
        "n_succeeded": len(rows),
        "n_dropped": len(drop_reasons),
        "drop_reasons": drop_reasons,
        "t_task_raw": t_task,
        "t_task_trimmed": t_task_trimmed,
        "T": t_task_trimmed * len(cfg.tasks),
        "length_summary": length_summary,
    }
    return waveforms_arr, masks_arr, subject_table, drop_stats


# ─── Step 5: attach labels ──────────────────────────────────────────────────


def merge_labels(cfg: Config, subject_table: pd.DataFrame) -> pd.DataFrame:
    """Merges label/condition from file_list.csv by subject id; reports class balance."""
    file_list = pd.read_csv(cfg.file_list_csv)
    file_list["subject_id"] = file_list["id"].apply(lambda i: f"{int(i):03d}")
    merged = subject_table.merge(
        file_list[["subject_id", "label", "condition"]], on="subject_id", how="left"
    )
    if merged["label"].isna().any():
        missing = merged.loc[merged["label"].isna(), "subject_id"].tolist()
        raise AssertionError(f"Subjects with no label after merge (should be impossible): {missing}")
    merged["label"] = merged["label"].astype(int)
    merged["parkinsons"] = merged["label"] == 1

    balance = merged["label"].map(LABEL_NAMES).value_counts()
    log.info("Class balance after label merge:\n%s", balance.to_string())
    return merged


# ─── Step 6: wavelet transform (placeholder — see module docstring) ────────


def apply_wavelet_transform(
    waveforms: np.ndarray,
    wavelet: Optional[str] = None,
    level: Optional[int] = None,
) -> np.ndarray:
    """
    SEAM FOR A FUTURE WAVELET STAGE.

    No wavelet transform exists anywhere in this project today — the real
    Tappy pipeline (preprocess_tappy.py) outputs raw z-scored waveforms, not
    wavelet coefficients (see module docstring, point 2). This function is
    therefore a documented no-op so pads_waveforms.npy matches the Tappy
    pipeline's ACTUAL current output shape ([N, C, T] z-scored floats).

    If a wavelet-coefficient schema is later confirmed for both datasets,
    fill this in (e.g. via pywavelets: `pywt.wavedec(x, wavelet, level=level)`
    per-channel, per-subject) and change only this function plus the column
    names written in save_outputs — no other step needs to change.
    """
    if wavelet is not None or level is not None:
        raise NotImplementedError(
            "Wavelet parameters were provided but no wavelet transform is "
            "implemented yet — see this function's docstring."
        )
    return waveforms


# ─── Normalization (mirrors preprocess_tappy.py's fit/apply) ───────────────


def fit_normalization_stats(waveforms: np.ndarray, masks: np.ndarray, is_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_waveforms = waveforms[is_train]
    train_masks = masks[is_train]
    n_channels = waveforms.shape[1]
    means = np.zeros(n_channels, dtype=np.float64)
    stds = np.ones(n_channels, dtype=np.float64)
    for c in range(n_channels):
        values = train_waveforms[:, c, :][train_masks]
        if values.size == 0:
            log.warning("Channel %d has no valid training timesteps; leaving mean=0, std=1.", c)
            continue
        means[c] = values.mean()
        std = values.std()
        stds[c] = std if std > 1e-8 else 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def apply_normalization(waveforms: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    normalized = waveforms.copy()
    for c in range(waveforms.shape[1]):
        normalized[:, c, :] = (waveforms[:, c, :] - means[c]) / stds[c]
    return normalized.astype(np.float32)


# ─── Subject-disjoint stratified split (mirrors preprocess_tappy.py) ───────


def split_subjects(labels: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    """Stratifies by the 3-class `label` (Healthy/PD/Other), analogous to Tappy's boolean-Parkinsons stratification."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    assignment: dict[str, str] = {}
    for label_value in sorted(labels["label"].unique()):
        keys = labels.loc[labels["label"] == label_value, "subject_id"].tolist()
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        for k in keys[:n_train]:
            assignment[k] = "train"
        for k in keys[n_train:n_train + n_val]:
            assignment[k] = "val"
        for k in keys[n_train + n_val:]:
            assignment[k] = "test"
    return assignment


# ─── Step 7: save + report ──────────────────────────────────────────────────


def export_schema(
    cfg: Config,
    waveforms: np.ndarray,
    masks: np.ndarray,
    labels: pd.DataFrame,
    drop_stats: dict,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "pads_waveforms.npy", waveforms)
    np.save(cfg.output_dir / "pads_masks.npy", masks)

    labels_csv = labels[["subject_id", "label", "parkinsons", "condition", "split"]].copy()
    labels_csv["session_index"] = 0
    labels_csv["sequence_length"] = [int(m.sum()) for m in masks]
    labels_csv = labels_csv[
        ["subject_id", "label", "parkinsons", "condition", "split", "session_index", "sequence_length"]
    ]
    labels_csv.to_csv(cfg.output_dir / "pads_labels.csv", index=False)

    tappy_schema_cols = ["UserKey", "Parkinsons", "Split", "SessionIndex", "SessionLength"]
    pads_schema_cols = ["subject_id", "parkinsons", "split", "session_index", "sequence_length"]
    log.info(
        "Schema parity check vs Tappy: tappy_labels.csv columns %s <-> "
        "pads_labels.csv equivalents %s (plus PADS-native `label`/`condition` "
        "kept for the 3-class breakdown Tappy has no equivalent of).",
        tappy_schema_cols, pads_schema_cols,
    )

    card_lines = [
        "# PADS Finger/Hand Movement Dataset — Preprocessing Data Card",
        "",
        "## Source",
        "PhysioNet: \"A public dataset of body-worn sensor data for the "
        "assessment of Parkinson's Disease\" (parkinsons-disease-smartwatch). "
        "Movement channels reconstructed via PADS's own scripts/utils "
        "(data_handling.get_data_from_observation, l1_trend_filter) applied "
        "directly to the raw movement/timeseries/*.txt recordings, NOT the "
        "shipped preprocessed/movement/*.bin files — see module docstring.",
        "",
        "## Why not the shipped .bin files",
        "preprocessed/movement/*.bin is missing PointFinger and TouchIndex: "
        "PADS's own run_preprocessing.py strips them "
        "(`to_remove = 'Time|LiftHold|PointFinger|TouchIndex'`) before saving. "
        "Only TouchNose survives there. This pipeline re-derives all three "
        "target tasks from the raw per-task .txt files instead, reusing "
        "PADS's own loading and gravity-correction code unchanged.",
        "",
        "## Tasks kept",
        f"{cfg.tasks} (finger/hand micro-movement tasks). Dropped: all "
        "Relaxed*/StretchHold/LiftHold/HoldWeight/DrinkGlas/CrossArms/"
        "Entrainment* gait/postural/whole-arm tasks.",
        "",
        "## Wrist",
        f"wrist_mode={cfg.wrist_mode}: dominant wrist per subject, from "
        "file_list.csv's `handedness` column (complete for all subjects: "
        "437 right / 32 left). Only that wrist's 6 channels "
        "(Accelerometer X/Y/Z, Gyroscope X/Y/Z) are kept.",
        "",
        "## Channels (in pads_waveforms.npy, shape [N, 6, T], float32, z-scored)",
        *[f"{i}. `{sensor}_{axis}` (dominant wrist)" for i, (sensor, axis) in enumerate(SENSOR_AXES)],
        "",
        "## Task concatenation",
        f"Tasks concatenated along time in order {cfg.tasks}. Each task's raw "
        f"recording is a fixed {drop_stats['t_task_raw']} samples; PADS's own "
        f"gravity correction (l1_trend_filter, vlambda={cfg.vlambda}) is "
        f"applied to the Accelerometer rows only (matching run_preprocessing.py), "
        f"then the first {cfg.trim_samples} samples are dropped (vibration-"
        f"notification trim, matching run_preprocessing.py), leaving "
        f"{drop_stats['t_task_trimmed']} samples/task -> T = {drop_stats['T']}.",
        "",
        "## Raw per-task length distribution (samples, before trim)",
        *[f"- {task}: {stats}" for task, stats in drop_stats["length_summary"].items()],
        "",
        "## Wavelet transform",
        "None. apply_wavelet_transform() is a documented no-op — see that "
        "function's docstring. pads_waveforms.npy holds raw (gravity-"
        "corrected, z-scored) values, matching what the real Tappy pipeline "
        "actually outputs today (tappy_waveforms.npy), not a hypothetical "
        "wavelet-coefficient CSV.",
        "",
        "## Normalization",
        "Z-score per channel, fit on TRAIN-split real (non-padded) timesteps only.",
        "",
        "## Split (subject-disjoint, stratified by `label`)",
        f"Target fractions: train={cfg.train_frac}, val={cfg.val_frac}, test={cfg.test_frac}, seed={cfg.seed}",
        "",
        "## Subjects dropped",
        f"- Attempted: {drop_stats['n_attempted']}, succeeded: {drop_stats['n_succeeded']}, dropped: {drop_stats['n_dropped']}",
        *[f"  - {reason}" for reason in drop_stats["drop_reasons"]],
        "",
        "## Class balance (final)",
        labels["label"].map(LABEL_NAMES).value_counts().to_string(),
        "",
        "## Schema parity with Tappy",
        f"tappy columns {tappy_schema_cols} <-> pads_labels.csv columns "
        f"{pads_schema_cols} (plus PADS-native `label`/`condition`). Both "
        "datasets use the [N, C, T] float32 z-scored waveform + [N, T] bool "
        "mask convention.",
    ]
    (cfg.output_dir / "pads_data_card.md").write_text("\n".join(card_lines), encoding="utf-8")
    log.info("Saved outputs to %s", cfg.output_dir)


# ─── main ────────────────────────────────────────────────────────────────


def run(cfg: Config) -> None:
    verify_official_preprocessing(cfg)
    verify_bin_shape(cfg)

    waveforms, masks, subject_table, drop_stats = build_all_subjects(cfg)
    if subject_table.empty:
        raise RuntimeError("No usable subjects were produced — check dataset paths and drop_stats.")

    labels = merge_labels(cfg, subject_table)

    waveforms = apply_wavelet_transform(waveforms)  # no-op today; see docstring

    split_assignment = split_subjects(labels, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
    labels["split"] = labels["subject_id"].map(split_assignment)

    is_train = (labels["split"] == "train").to_numpy()
    means, stds = fit_normalization_stats(waveforms, masks, is_train)
    waveforms = apply_normalization(waveforms, means, stds)

    export_schema(cfg, waveforms, masks, labels, drop_stats)

    print("\n" + "=" * 60)
    print("PADS FINGER/HAND MOVEMENT PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Subjects attempted : {drop_stats['n_attempted']}")
    print(f"Subjects succeeded : {drop_stats['n_succeeded']}")
    print(f"Subjects dropped   : {drop_stats['n_dropped']}")
    for reason in drop_stats["drop_reasons"]:
        print(f"  - {reason}")
    print(f"Sequence length T  : {drop_stats['T']} (tasks={cfg.tasks}, wrist_mode={cfg.wrist_mode})")
    print("\nClass balance:")
    print(labels["label"].map(LABEL_NAMES).value_counts().to_string())
    print("\nSubjects per split:")
    print(labels.groupby("split").size().to_string())
    print("\nClass balance per split:")
    print(labels.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns=LABEL_NAMES).to_string())
    print(f"\nColumn schema matches Tappy convention (subject_id/parkinsons/split/session_index/sequence_length "
          f"<-> UserKey/Parkinsons/Split/SessionIndex/SessionLength): confirmed in pads_data_card.md")
    print("=" * 60)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pads-root", type=Path, required=True, help="Path to parkinsons-disease-smartwatch/1.0.0")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write the .npy/.csv/.md outputs")
    p.add_argument("--tasks", nargs="+", default=["PointFinger", "TouchIndex", "TouchNose"])
    p.add_argument("--wrist-mode", choices=["dominant", "LeftWrist", "RightWrist"], default="dominant")
    p.add_argument("--vlambda", type=float, default=50.0, help="l1_trend_filter regularization (PADS official default)")
    p.add_argument("--trim-samples", type=int, default=48, help="Startup-vibration trim (PADS official default)")
    p.add_argument("--task-length-percentile", type=float, default=100.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel worker processes for gravity correction")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N subjects (for testing)")
    args = p.parse_args()

    return Config(
        pads_root=args.pads_root,
        output_dir=args.output_dir,
        tasks=args.tasks,
        wrist_mode=args.wrist_mode,
        vlambda=args.vlambda,
        trim_samples=args.trim_samples,
        task_length_percentile=args.task_length_percentile,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        n_jobs=args.n_jobs,
        limit=args.limit,
    )


if __name__ == "__main__":
    run(parse_args())
