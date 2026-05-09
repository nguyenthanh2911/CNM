from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

import joblib


def _infer_interval_minutes(df: pd.DataFrame, time_col: str, patient_col: Optional[str] = None) -> float:
    if time_col not in df.columns:
        return 5.0

    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if ts.isna().all():
        return 5.0

    if patient_col and patient_col in df.columns:
        # Median over all patients
        intervals = (
            df.assign(_ts=ts)
            .sort_values([patient_col, "_ts"], kind="mergesort")
            .groupby(patient_col)["_ts"]
            .diff()
            .dt.total_seconds()
            .div(60.0)
        )
    else:
        intervals = ts.sort_values().diff().dt.total_seconds().div(60.0)

    intervals = intervals.replace([np.inf, -np.inf], np.nan).dropna()
    intervals = intervals[intervals > 0]
    if intervals.empty:
        return 5.0

    return float(intervals.median())


@dataclass
class ICUPreprocessor:
    patient_col: str = "patient_id"
    time_col: str = "timestamp"
    vital_columns: List[str] = field(
        default_factory=lambda: [
            "heart_rate",
            "systolic_bp",
            "diastolic_bp",
            "temperature",
            "spo2",
            "respiratory_rate",
        ]
    )
    lab_columns: List[str] = field(
        default_factory=lambda: [
            "lactate",
            "wbc",
            "creatinine",
            "bilirubin",
            "platelet",
        ]
    )

    scaler: Optional[StandardScaler] = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce", utc=True)

        sort_cols = [c for c in [self.patient_col, self.time_col] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort")

        # 1) Forward-fill missing vitals if gap < 10 minutes.
        interval_min = _infer_interval_minutes(df, self.time_col, self.patient_col)
        ffill_limit = int(np.floor(10.0 / max(interval_min, 1e-6)))
        if ffill_limit > 0:
            if self.patient_col in df.columns:
                df[self.vital_columns] = (
                    df.groupby(self.patient_col, sort=False)[self.vital_columns]
                    .apply(lambda g: g.fillna(method="ffill", limit=ffill_limit))
                    .reset_index(level=0, drop=True)
                )
            else:
                df[self.vital_columns] = df[self.vital_columns].fillna(method="ffill", limit=ffill_limit)

        # 2) KNN imputation for lab columns
        present_labs = [c for c in self.lab_columns if c in df.columns]
        if present_labs:
            imputer = KNNImputer(n_neighbors=3)
            df[present_labs] = imputer.fit_transform(df[present_labs])

        # 3) Outlier IQR clipping/replacement with median
        numeric_cols = [c for c in (self.vital_columns + self.lab_columns) if c in df.columns]
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            low = q1 - 3.0 * iqr
            high = q3 + 3.0 * iqr
            median = series.median()
            mask = (series < low) | (series > high)
            if mask.any() and not pd.isna(median):
                df.loc[mask, col] = median

        # 4) Fit StandardScaler on vital columns
        present_vitals = [c for c in self.vital_columns if c in df.columns]
        self.scaler = StandardScaler()
        if present_vitals:
            df[present_vitals] = self.scaler.fit_transform(df[present_vitals].astype(float))

        # 5) Return processed df
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError("Scaler is not fitted. Call fit_transform() or load() first.")

        df = df.copy()
        present_vitals = [c for c in self.vital_columns if c in df.columns]
        if present_vitals:
            df[present_vitals] = self.scaler.transform(df[present_vitals].astype(float))
        return df

    def save(self, path: str) -> None:
        if self.scaler is None:
            raise ValueError("Scaler is not fitted. Nothing to save.")
        joblib.dump(self.scaler, path)

    def load(self, path: str) -> "ICUPreprocessor":
        self.scaler = joblib.load(path)
        return self
