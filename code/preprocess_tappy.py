"""
Preprocessing pipeline for the Tappy Keystroke dataset (PhysioNet):
"Keystroke logs collected from subjects with and without Parkinson's disease".

Converts raw per-keystroke event logs into fixed-length, multichannel 1D
waveform tensors per typing session (NOT summary statistics) suitable for a
TCN/BiGRU sequence encoder.

Dataset layout actually found on disk (confirmed by direct inspection, not
assumed from any changelog — see tappy_data_card.md for details):

    <dataset_root>/
      Archived users/
        User_<KEY>.txt        one file per subject, 12 "Field: value" lines:
                               BirthYear, Gender, Parkinsons, Tremors,
                               DiagnosisYear, Sided, UPDRS, Impact, Levadopa,
                               DA, MAOB, Other
      Tappy Data/
        <KEY>_<YYMM>.txt       one file per subject per month of keystroke
                                events, tab-separated, 8 real columns + a
                                trailing empty column from a trailing tab:
                                UserKey, Date (YYMMDD), Timestamp
                                (HH:MM:SS.mmm), Hand (L/R/S), Hold time,
                                Direction (2-char hand-pair code), Latency
                                time, Flight time

There is no single "Users.txt" in this release — per-subject metadata is
one file per subject under "Archived users" instead. This script treats
that directory as the label source the prompt calls "Users.txt".

Usage:
    python preprocess_tappy.py --dataset-root "/path/to/tappy-keystroke-data-1.0.0" \
        --output-dir "/path/to/output"

Run with --help for all configurable parameters.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("preprocess_tappy")

CHANNEL_NAMES = ["hold_time", "latency_time", "flight_time"]
VALID_HANDS = {"L", "R", "S"}
VALID_DIRECTIONS = {h1 + h2 for h1 in VALID_HANDS for h2 in VALID_HANDS}
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{1,3}$")
DATE_RE = re.compile(r"^\d{6}$")
BOOL_MAP = {"True": True, "False": False}


# ─── Config ────────────────────────────────────────────────────────────────


@dataclass
class Config:
    dataset_root: Path
    output_dir: Path
    archived_dirname: str = "Archived users"
    tappy_dirname: str = "Tappy Data"
    session_gap_minutes: float = 15.0
    min_session_events: int = 20
    session_length_percentile: float = 90.0
    max_hold_time_ms: float = 10_000.0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    @property
    def archived_dir(self) -> Path:
        return self.dataset_root / self.archived_dirname

    @property
    def tappy_dir(self) -> Path:
        return self.dataset_root / self.tappy_dirname


# ─── 1. Loading: subject-level label table ─────────────────────────────────


def _parse_metadata_file(path: Path) -> dict:
    """Parses one 'Field: value' metadata file into a dict of raw strings."""
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _coerce_bool(value: str) -> Optional[bool]:
    return BOOL_MAP.get(value)


def _coerce_int(value: str) -> Optional[int]:
    return int(value) if value.isdigit() else None


def load_labels(archived_dir: Path) -> pd.DataFrame:
    """
    Loads every 'User_<KEY>.txt' metadata file into one subject-level table.

    Returns a DataFrame indexed by row with columns:
    UserKey, BirthYear, Gender, Parkinsons, Tremors, DiagnosisYear, Sided,
    UPDRS, Impact, Levadopa, DA, MAOB, Other.

    Parkinsons/Tremors/Levadopa/DA/MAOB/Other are booleans (None if the
    source field was blank/unparseable). BirthYear/DiagnosisYear are
    nullable ints. Gender/Sided/UPDRS/Impact are left as free-text strings
    since their value sets aren't guaranteed stable across releases.
    """
    if not archived_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {archived_dir}")

    rows = []
    for path in sorted(archived_dir.glob("User_*.txt")):
        user_key = path.stem[len("User_") :]
        raw = _parse_metadata_file(path)
        rows.append(
            {
                "UserKey": user_key,
                "BirthYear": _coerce_int(raw.get("BirthYear", "")),
                "Gender": raw.get("Gender") or None,
                "Parkinsons": _coerce_bool(raw.get("Parkinsons", "")),
                "Tremors": _coerce_bool(raw.get("Tremors", "")),
                "DiagnosisYear": _coerce_int(raw.get("DiagnosisYear", "")),
                "Sided": raw.get("Sided") or None,
                "UPDRS": raw.get("UPDRS") or None,
                "Impact": raw.get("Impact") or None,
                "Levadopa": _coerce_bool(raw.get("Levadopa", "")),
                "DA": _coerce_bool(raw.get("DA", "")),
                "MAOB": _coerce_bool(raw.get("MAOB", "")),
                "Other": _coerce_bool(raw.get("Other", "")),
            }
        )

    labels = pd.DataFrame(rows)
    n_missing_pd = labels["Parkinsons"].isna().sum()
    if n_missing_pd:
        log.warning(
            "%d/%d subjects have an unparseable Parkinsons field and will be "
            "dropped downstream (no usable label).",
            n_missing_pd,
            len(labels),
        )
    log.info("Loaded metadata for %d subjects from %s", len(labels), archived_dir)
    return labels


# ─── 2. Loading: per-subject keystroke event logs ──────────────────────────


def _parse_event_line(
    line: str, expected_user_key: str, max_hold_time_ms: float
) -> tuple[Optional[dict], Optional[str]]:
    """
    Validates and parses one tab-separated keystroke event row.

    Returns (row_dict, None) on success, or (None, reason) on failure, where
    reason is "malformed" (wrong field count, corrupted/glued values,
    out-of-range hand/direction codes, non-numeric/negative timing fields,
    unparseable date/timestamp) or "implausible_hold" (syntactically valid
    but physically impossible Hold-time value — see module docstring note
    on the app-backgrounding artifact this dataset has).

    This dataset has a small amount of corrupted rows (~a few hundred out of
    ~9.3M in the source release) where a tab is missing and two field values
    are concatenated together (e.g. "0105.0EA27ICBLF" instead of separate
    Hold-time and UserKey values). Requiring field[0] == expected_user_key
    catches most of these for free, since a glued value never equals the
    clean key.

    Separately, a small number of rows (~600-800 out of ~9.3M, confirmed by
    direct inspection) have a Hold time in the tens of seconds to multiple
    hours (max observed: ~13.6M ms, i.e. ~3.8 hours) — physically impossible
    for a single keystroke, almost certainly a stuck-key/app-backgrounding
    artifact rather than genuine (if severely impaired) typing. Latency time
    and Flight time show no equivalent contamination (0 rows > 5000 ms in
    either, checked directly). `max_hold_time_ms` drops only the Hold-time
    channel's implausible tail; it does not touch Latency/Flight.
    """
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 8:
        return None, "malformed"

    user_key, date, timestamp, hand, hold, direction, latency, flight = fields[:8]

    if user_key != expected_user_key:
        return None, "malformed"
    if not DATE_RE.match(date):
        return None, "malformed"
    if not TIMESTAMP_RE.match(timestamp):
        return None, "malformed"
    if hand not in VALID_HANDS:
        return None, "malformed"
    if direction not in VALID_DIRECTIONS:
        return None, "malformed"

    try:
        hold_v = float(hold)
        latency_v = float(latency)
        flight_v = float(flight)
    except ValueError:
        return None, "malformed"
    if hold_v < 0 or latency_v < 0 or flight_v < 0:
        return None, "malformed"

    if hold_v > max_hold_time_ms:
        return None, "implausible_hold"

    try:
        dt = pd.to_datetime(date + timestamp, format="%y%m%d%H:%M:%S.%f")
    except ValueError:
        return None, "malformed"

    return {
        "UserKey": user_key,
        "Timestamp": dt,
        "Hand": hand,
        "HoldTime": hold_v,
        "Direction": direction,
        "LatencyTime": latency_v,
        "FlightTime": flight_v,
    }, None


def load_user_events(
    tappy_dir: Path, user_key: str, max_hold_time_ms: float
) -> tuple[pd.DataFrame, int, int]:
    """
    Loads and concatenates all monthly event-log files for one subject,
    validates every row, and returns them sorted by timestamp.

    Returns (events_df, n_malformed_dropped, n_implausible_hold_dropped).
    """
    paths = sorted(tappy_dir.glob(f"{user_key}_*.txt"))
    parsed: list[dict] = []
    n_malformed = 0
    n_implausible_hold = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row, reason = _parse_event_line(line, user_key, max_hold_time_ms)
            if row is not None:
                parsed.append(row)
            elif reason == "malformed":
                n_malformed += 1
            elif reason == "implausible_hold":
                n_implausible_hold += 1

    if not parsed:
        empty = pd.DataFrame(columns=["UserKey", "Timestamp", "Hand", "HoldTime", "Direction", "LatencyTime", "FlightTime"])
        return empty, n_malformed, n_implausible_hold

    events = pd.DataFrame(parsed).sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    return events, n_malformed, n_implausible_hold


# ─── 3. Session segmentation ────────────────────────────────────────────────


def segment_sessions(
    events: pd.DataFrame, gap_minutes: float, min_session_events: int
) -> tuple[list[pd.DataFrame], int]:
    """
    Splits a time-sorted event log into sessions: contiguous runs of
    keystrokes with no gap larger than `gap_minutes` between consecutive
    events. Sessions shorter than `min_session_events` are dropped (treated
    as accidental/noise taps rather than real typing activity).

    Returns (sessions, n_dropped_short_sessions).
    """
    if events.empty:
        return [], 0

    gaps = events["Timestamp"].diff().dt.total_seconds() / 60.0
    new_session = (gaps > gap_minutes).fillna(False)
    session_id = new_session.cumsum()

    sessions = []
    n_dropped_short = 0
    for _, group in events.groupby(session_id, sort=True):
        if len(group) < min_session_events:
            n_dropped_short += 1
            continue
        sessions.append(group.reset_index(drop=True))

    return sessions, n_dropped_short


# ─── 4. Channel extraction ──────────────────────────────────────────────────


def extract_channels(session: pd.DataFrame) -> np.ndarray:
    """
    Builds the ordered [3, L] channel array for one session:
    Hold time, Latency time, Flight time — one value per keystroke event,
    in chronological order (session is assumed pre-sorted by segment_sessions).
    """
    return np.stack(
        [
            session["HoldTime"].to_numpy(dtype=np.float32),
            session["LatencyTime"].to_numpy(dtype=np.float32),
            session["FlightTime"].to_numpy(dtype=np.float32),
        ],
        axis=0,
    )


# ─── 5. Session-length distribution and T selection ────────────────────────


def summarize_session_lengths(lengths: list[int]) -> dict:
    arr = np.asarray(lengths)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    summary = {
        "n_sessions": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        **{f"p{p}": float(np.percentile(arr, p)) for p in percentiles},
    }
    return summary


def choose_sequence_length(lengths: list[int], percentile: float) -> int:
    return int(round(float(np.percentile(np.asarray(lengths), percentile))))


# ─── Padding / truncation ───────────────────────────────────────────────────


def pad_or_truncate(channels: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """
    channels: [3, L] float32
    Returns (fixed [3, T] float32, mask [T] bool — True where real, False
    where zero-padded).
    """
    _, length = channels.shape
    if length >= T:
        return channels[:, :T].astype(np.float32), np.ones(T, dtype=bool)

    fixed = np.zeros((channels.shape[0], T), dtype=np.float32)
    fixed[:, :length] = channels
    mask = np.zeros(T, dtype=bool)
    mask[:length] = True
    return fixed, mask


# ─── Orchestration: build every session for every usable subject ───────────


def build_all_sessions(cfg: Config, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray], dict]:
    """
    For every subject that has both a usable label and at least one
    keystroke-log file, loads events, segments sessions, and extracts raw
    (un-padded) channel arrays.

    Returns:
      meta: DataFrame, one row per session (UserKey, SessionIndex,
            SessionLength, StartTime, EndTime, Parkinsons — and other
            label columns carried through for convenience)
      raw_channels: list of [3, L] float32 arrays, aligned by index with meta
      drop_stats: dict of counts for the data card
    """
    tappy_user_keys = sorted(
        {p.name.rsplit("_", 1)[0] for p in cfg.tappy_dir.glob("*_*.txt")}
    )
    labeled_keys = set(labels.loc[labels["Parkinsons"].notna(), "UserKey"])
    event_keys = set(tappy_user_keys)

    usable_keys = sorted(labeled_keys & event_keys)
    label_only = sorted(labeled_keys - event_keys)
    events_only = sorted(event_keys - labeled_keys)

    log.info(
        "%d subjects have both a usable label and keystroke data; "
        "%d dropped (label but no keystroke files); "
        "%d dropped (keystroke files but no usable label).",
        len(usable_keys),
        len(label_only),
        len(events_only),
    )

    meta_rows = []
    raw_channels: list[np.ndarray] = []
    n_malformed_total = 0
    n_implausible_hold_total = 0
    n_short_sessions_dropped_total = 0
    n_subjects_zero_sessions = 0

    label_lookup = labels.set_index("UserKey")

    for user_key in usable_keys:
        events, n_malformed, n_implausible_hold = load_user_events(
            cfg.tappy_dir, user_key, cfg.max_hold_time_ms
        )
        n_malformed_total += n_malformed
        n_implausible_hold_total += n_implausible_hold

        sessions, n_short_dropped = segment_sessions(
            events, cfg.session_gap_minutes, cfg.min_session_events
        )
        n_short_sessions_dropped_total += n_short_dropped

        if not sessions:
            n_subjects_zero_sessions += 1
            continue

        label_row = label_lookup.loc[user_key]
        for session_idx, session in enumerate(sessions):
            channels = extract_channels(session)
            raw_channels.append(channels)
            meta_rows.append(
                {
                    "UserKey": user_key,
                    "SessionIndex": session_idx,
                    "SessionLength": channels.shape[1],
                    "StartTime": session["Timestamp"].iloc[0],
                    "EndTime": session["Timestamp"].iloc[-1],
                    "Parkinsons": bool(label_row["Parkinsons"]),
                    "Gender": label_row["Gender"],
                    "BirthYear": label_row["BirthYear"],
                    "DiagnosisYear": label_row["DiagnosisYear"],
                    "Impact": label_row["Impact"],
                }
            )

    meta = pd.DataFrame(meta_rows)
    drop_stats = {
        "subjects_label_only_dropped": len(label_only),
        "subjects_events_only_dropped": len(events_only),
        "subjects_with_zero_valid_sessions": n_subjects_zero_sessions,
        "malformed_rows_dropped": n_malformed_total,
        "implausible_hold_rows_dropped": n_implausible_hold_total,
        "short_sessions_dropped": n_short_sessions_dropped_total,
        "usable_subjects": len(usable_keys),
        "label_only_keys": label_only,
        "events_only_keys": events_only,
    }
    return meta, raw_channels, drop_stats


# ─── 8. Subject-disjoint stratified split ───────────────────────────────────


def split_subjects(
    labels: pd.DataFrame,
    usable_keys: list[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    """
    Splits subjects (not sessions) into train/val/test, stratified by
    Parkinsons status so class balance is preserved in each split. Every
    session belonging to a subject inherits that subject's split, so no
    subject's sessions ever cross a split boundary.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    label_lookup = labels.set_index("UserKey")["Parkinsons"]

    assignment: dict[str, str] = {}
    for pd_status in [True, False]:
        keys = [k for k in usable_keys if label_lookup.loc[k] == pd_status]
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # remainder goes to test, guarantees all subjects are assigned
        train_keys = keys[:n_train]
        val_keys = keys[n_train : n_train + n_val]
        test_keys = keys[n_train + n_val :]
        for k in train_keys:
            assignment[k] = "train"
        for k in val_keys:
            assignment[k] = "val"
        for k in test_keys:
            assignment[k] = "test"

    return assignment


# ─── 9. Normalization ────────────────────────────────────────────────────────


def fit_normalization_stats(
    waveforms: np.ndarray, masks: np.ndarray, is_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    waveforms: [N, 3, T] float32
    masks:     [N, T] bool (True = real timestep)
    is_train:  [N] bool

    Computes per-channel mean/std using only real (non-padded) timesteps
    from train sessions. Returns (mean [3], std [3]).
    """
    train_waveforms = waveforms[is_train]
    train_masks = masks[is_train]

    means = np.zeros(waveforms.shape[1], dtype=np.float64)
    stds = np.ones(waveforms.shape[1], dtype=np.float64)
    for c in range(waveforms.shape[1]):
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


# ─── 10. Saving ──────────────────────────────────────────────────────────────


def save_outputs(
    cfg: Config,
    waveforms: np.ndarray,
    masks: np.ndarray,
    meta: pd.DataFrame,
    length_summary: dict,
    T: int,
    drop_stats: dict,
    norm_means: np.ndarray,
    norm_stds: np.ndarray,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(cfg.output_dir / "tappy_waveforms.npy", waveforms)
    np.save(cfg.output_dir / "tappy_masks.npy", masks)

    labels_csv = meta[["UserKey", "Parkinsons", "Split", "SessionIndex", "SessionLength"]].copy()
    labels_csv.to_csv(cfg.output_dir / "tappy_labels.csv", index=False)

    split_counts = meta.groupby("Split").agg(
        sessions=("UserKey", "count"), subjects=("UserKey", "nunique")
    )
    class_balance = meta.groupby(["Split", "Parkinsons"]).size().unstack(fill_value=0)

    card_lines = [
        "# Tappy Keystroke Dataset — Preprocessing Data Card",
        "",
        "## Source",
        "PhysioNet: \"Keystroke logs collected from subjects with and without "
        "Parkinson's disease\". This release has no single `Users.txt`; "
        "per-subject demographics/diagnosis live under `Archived users/` "
        "(one file per subject), and per-keystroke event logs live under "
        "`Tappy Data/` (one file per subject per month).",
        "",
        "## Channels (in tappy_waveforms.npy, shape [N, 3, T], float32, z-scored)",
        "1. `hold_time` — dwell time of each keystroke (ms)",
        "2. `latency_time` — time from previous key event to this one (ms)",
        "3. `flight_time` — time from previous key press to this key's press (ms)",
        "",
        "Each channel is an ORDERED sequence of per-keystroke values across "
        "the session, not a summary statistic. Padded (past the real session "
        "length) or truncated timesteps are included in `tappy_masks.npy` "
        "(shape [N, T], bool — True = real event, False = pad). Padding was "
        "applied as zeros BEFORE z-score normalization, so padded positions "
        "are not exactly 0 after normalization — always gate on the mask, "
        "not on the value.",
        "",
        "## Known data-quality issue: Hold-time outliers",
        f"Direct inspection of the raw files found a small number of rows "
        f"(confirmed by scanning all ~9.3M raw events) with Hold-time values "
        f"in the tens of seconds to multiple hours (max observed ≈13.6M ms, "
        f"~3.8 hours) — physically impossible for one keystroke, almost "
        f"certainly a stuck-key/app-backgrounding artifact rather than "
        f"genuine typing. Latency time and Flight time showed no equivalent "
        f"contamination (0 rows > 5000 ms in either, checked directly), so "
        f"only Hold time is filtered. Rows with Hold time above "
        f"**{cfg.max_hold_time_ms:.0f} ms** (configurable via "
        f"--max-hold-time-ms) are dropped before session segmentation — see "
        f"`implausible_hold_rows_dropped` below. This matters: leaving them "
        f"in inflates the Hold-time channel's std by orders of magnitude "
        f"and would compress genuine keystroke variation toward 0 after "
        f"z-scoring.",
        "",
        f"## Session segmentation rule",
        f"A session is a run of a subject's keystrokes with no gap larger "
        f"than **{cfg.session_gap_minutes} minutes** between consecutive "
        f"events (configurable via --session-gap-minutes). Sessions with "
        f"fewer than **{cfg.min_session_events}** keystrokes were dropped as "
        f"noise (configurable via --min-session-events).",
        "",
        "## Session length distribution (raw event count, all usable sessions, pre-split)",
        f"n_sessions={length_summary['n_sessions']}, min={length_summary['min']}, "
        f"max={length_summary['max']}, mean={length_summary['mean']:.1f}",
        f"p10={length_summary['p10']:.0f}  p25={length_summary['p25']:.0f}  "
        f"p50={length_summary['p50']:.0f}  p75={length_summary['p75']:.0f}  "
        f"p90={length_summary['p90']:.0f}  p95={length_summary['p95']:.0f}  "
        f"p99={length_summary['p99']:.0f}",
        "",
        f"## Chosen sequence length T = {T}",
        f"Set to the {cfg.session_length_percentile:.0f}th percentile of raw "
        f"session lengths across the whole usable dataset (computed before "
        f"the train/val/test split, since T is a shape hyperparameter, not "
        f"a statistic that can leak label information the way normalization "
        f"stats can). This keeps most sessions unpadded/lightly-padded while "
        f"bounding the truncation applied to the small tail of very long "
        f"sessions.",
        "",
        "## Normalization",
        "Z-score per channel, fit on TRAIN-split real (non-padded) timesteps "
        "only, then applied unchanged to val/test.",
        f"mean = {norm_means.tolist()}",
        f"std  = {norm_stds.tolist()}",
        "",
        "## Split (subject-disjoint, stratified by Parkinsons status)",
        f"Target fractions: train={cfg.train_frac}, val={cfg.val_frac}, test={cfg.test_frac}",
        "",
        "```",
        split_counts.to_string(),
        "```",
        "",
        "### Class balance per split (session count)",
        "```",
        class_balance.to_string(),
        "```",
        "",
        "## Subjects/rows dropped",
        f"- Subjects with keystroke files but no usable label (missing/unparseable "
        f"`Parkinsons` field in their metadata file): {drop_stats['subjects_events_only_dropped']}",
        f"- Subjects with a usable label but no keystroke files at all: "
        f"{drop_stats['subjects_label_only_dropped']}",
        f"- Subjects with a usable label and keystroke files, but zero sessions "
        f"survived segmentation/min-length filtering: {drop_stats['subjects_with_zero_valid_sessions']}",
        f"- Individual malformed keystroke-log rows dropped during parsing "
        f"(bad field count, corrupted/glued values, invalid hand/direction "
        f"code, non-numeric or negative timing fields, unparseable date/"
        f"timestamp): {drop_stats['malformed_rows_dropped']}",
        f"- Individual rows dropped for an implausible Hold-time value "
        f"(> {cfg.max_hold_time_ms:.0f} ms — see data-quality note above): "
        f"{drop_stats['implausible_hold_rows_dropped']}",
        f"- Sessions dropped for having fewer than {cfg.min_session_events} "
        f"keystrokes: {drop_stats['short_sessions_dropped']}",
        f"- Subjects used: {drop_stats['usable_subjects']}",
        "",
        "## Config used",
        f"session_gap_minutes={cfg.session_gap_minutes}, "
        f"min_session_events={cfg.min_session_events}, "
        f"session_length_percentile={cfg.session_length_percentile}, "
        f"max_hold_time_ms={cfg.max_hold_time_ms}, "
        f"seed={cfg.seed}",
    ]
    (cfg.output_dir / "tappy_data_card.md").write_text("\n".join(card_lines), encoding="utf-8")

    log.info("Saved outputs to %s", cfg.output_dir)


# ─── main ────────────────────────────────────────────────────────────────────


def run(cfg: Config) -> None:
    labels = load_labels(cfg.archived_dir)

    meta, raw_channels, drop_stats = build_all_sessions(cfg, labels)
    if meta.empty:
        raise RuntimeError("No usable sessions were built — check dataset paths and thresholds.")

    length_summary = summarize_session_lengths(meta["SessionLength"].tolist())
    log.info("Session length distribution: %s", length_summary)

    T = choose_sequence_length(meta["SessionLength"].tolist(), cfg.session_length_percentile)
    log.info("Chosen T = %d (p%.0f of session lengths)", T, cfg.session_length_percentile)

    fixed_list, mask_list = [], []
    for channels in raw_channels:
        fixed, mask = pad_or_truncate(channels, T)
        fixed_list.append(fixed)
        mask_list.append(mask)
    waveforms = np.stack(fixed_list, axis=0).astype(np.float32)  # [N, 3, T]
    masks = np.stack(mask_list, axis=0)  # [N, T] bool

    usable_keys = sorted(meta["UserKey"].unique().tolist())
    split_assignment = split_subjects(
        labels, usable_keys, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed
    )
    meta["Split"] = meta["UserKey"].map(split_assignment)

    is_train = (meta["Split"] == "train").to_numpy()
    norm_means, norm_stds = fit_normalization_stats(waveforms, masks, is_train)
    waveforms = apply_normalization(waveforms, norm_means, norm_stds)

    save_outputs(cfg, waveforms, masks, meta, length_summary, T, drop_stats, norm_means, norm_stds)

    # ─── Summary ───
    n_subjects = meta["UserKey"].nunique()
    n_sessions = len(meta)
    print("\n" + "=" * 60)
    print("TAPPY KEYSTROKE PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total usable subjects : {n_subjects}")
    print(f"Total sessions        : {n_sessions}")
    print(f"Sequence length T     : {T} (p{cfg.session_length_percentile:.0f} of session lengths, "
          f"min={length_summary['min']}, max={length_summary['max']}, mean={length_summary['mean']:.1f})")
    print("\nSessions per split:")
    print(meta.groupby("Split").size().to_string())
    print("\nSubjects per split:")
    print(meta.groupby("Split")["UserKey"].nunique().to_string())
    print("\nClass balance (session count) per split:")
    print(meta.groupby(["Split", "Parkinsons"]).size().unstack(fill_value=0).to_string())
    print("\nClass balance (subject count) per split:")
    subj_balance = meta.drop_duplicates("UserKey").groupby(["Split", "Parkinsons"]).size().unstack(fill_value=0)
    print(subj_balance.to_string())
    print("=" * 60)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True, help="Path to tappy-keystroke-data-*.* directory")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write the .npy/.csv/.md outputs")
    p.add_argument("--session-gap-minutes", type=float, default=15.0, help="Gap (minutes) that starts a new session")
    p.add_argument("--min-session-events", type=int, default=20, help="Drop sessions with fewer keystrokes than this")
    p.add_argument("--session-length-percentile", type=float, default=90.0, help="Percentile of session lengths used to pick T")
    p.add_argument("--max-hold-time-ms", type=float, default=10_000.0, help="Drop rows with Hold time above this (ms) as physically implausible")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    return Config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        session_gap_minutes=args.session_gap_minutes,
        min_session_events=args.min_session_events,
        session_length_percentile=args.session_length_percentile,
        max_hold_time_ms=args.max_hold_time_ms,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
