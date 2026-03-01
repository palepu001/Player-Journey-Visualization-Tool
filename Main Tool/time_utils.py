from __future__ import annotations

import pandas as pd


def parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the event_time column into pandas datetime.

    This keeps the original rows intact and only converts the column if present.
    """
    out = df.copy()

    if "event_time" not in out.columns:
        out["event_time"] = pd.NaT
        return out

    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    return out


def apply_timeline_filter(
    df: pd.DataFrame,
    progress_pct: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Filter rows up to the playback cutoff.

    Returns:
    - filtered dataframe
    - start_time
    - end_time
    - cutoff_time
    """
    if df.empty or "event_time" not in df.columns:
        return df.copy(), None, None, None

    working = df.copy()
    working = working.dropna(subset=["event_time"]).copy()

    if working.empty:
        return df.iloc[0:0].copy(), None, None, None

    working = working.sort_values(["event_time"], ascending=True)

    start_time = working["event_time"].min()
    end_time = working["event_time"].max()

    if pd.isna(start_time) or pd.isna(end_time):
        return df.iloc[0:0].copy(), None, None, None

    safe_progress = max(0, min(int(progress_pct), 100))

    if safe_progress == 0:
        cutoff_time = start_time
        filtered = working[working["event_time"] <= cutoff_time].copy()
        return filtered, start_time, end_time, cutoff_time

    if start_time == end_time or safe_progress == 100:
        cutoff_time = end_time
        return working.copy(), start_time, end_time, cutoff_time

    total_span = end_time - start_time
    cutoff_time = start_time + (total_span * (safe_progress / 100.0))

    filtered = working[working["event_time"] <= cutoff_time].copy()
    return filtered, start_time, end_time, cutoff_time


def format_ts(value) -> str:
    """
    Format timestamps consistently for UI tables and captions.
    """
    if value is None or pd.isna(value):
        return "N/A"

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "N/A"

    return ts.strftime("%Y-%m-%d %H:%M:%S")