from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _infer_interval_minutes(df: pd.DataFrame, time_col: str, patient_col: str) -> float:
    if time_col not in df.columns:
        return 5.0
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if ts.isna().all():
        return 5.0

    intervals = (
        df.assign(_ts=ts)
        .sort_values([patient_col, "_ts"], kind="mergesort")
        .groupby(patient_col)["_ts"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    intervals = intervals.replace([np.inf, -np.inf], np.nan).dropna()
    intervals = intervals[intervals > 0]
    if intervals.empty:
        return 5.0
    return float(intervals.median())


def add_rolling_features(df: pd.DataFrame, patient_col: str = "patient_id", time_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.sort_values([patient_col, time_col], kind="mergesort")

    interval_minutes = _infer_interval_minutes(df, time_col=time_col, patient_col=patient_col)

    vital_cols: List[str] = [
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "temperature",
        "spo2",
        "respiratory_rate",
    ]

    windows = [3, 12, 48]
    stats = ["mean", "std", "min", "max"]

    added_cols: List[str] = []

    for vital in vital_cols:
        if vital not in df.columns:
            continue

        grouped = df.groupby(patient_col, sort=False)[vital]

        for w in windows:
            minutes = int(round(w * interval_minutes))
            suffix = f"{minutes}m"

            rolling = grouped.rolling(window=w, min_periods=1)

            mean_col = f"{vital}_mean_{suffix}"
            std_col = f"{vital}_std_{suffix}"
            min_col = f"{vital}_min_{suffix}"
            max_col = f"{vital}_max_{suffix}"

            df[mean_col] = rolling.mean().reset_index(level=0, drop=True)
            df[std_col] = rolling.std(ddof=0).reset_index(level=0, drop=True)
            df[min_col] = rolling.min().reset_index(level=0, drop=True)
            df[max_col] = rolling.max().reset_index(level=0, drop=True)

            added_cols.extend([mean_col, std_col, min_col, max_col])

        # Trend: diff(1) / interval_minutes, named as 15m trend by spec
        trend_col = f"{vital}_trend_{int(round(3 * interval_minutes))}m"
        df[trend_col] = grouped.diff(1).reset_index(level=0, drop=True) / float(interval_minutes)
        added_cols.append(trend_col)

    # Fill initial NaNs in new rolling/trend columns
    if added_cols:
        df[added_cols] = (
            df.groupby(patient_col, sort=False)[added_cols]
            .apply(lambda g: g.bfill())
            .reset_index(level=0, drop=True)
        )

    return df
