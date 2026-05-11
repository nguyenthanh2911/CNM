from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .clinical_scores import calculate_news2, calculate_qsofa, calculate_sofa
from .vitals_features import add_rolling_features


@dataclass
class FeatureBuilder:
    patient_col: str = "patient_id"
    time_col: str = "timestamp"
    label_col: str = "sepsis_label"
    _feature_columns: Optional[List[str]] = field(default=None, init=False)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce", utc=True)
        df = df.sort_values([self.patient_col, self.time_col], kind="mergesort")

        # 1) Rolling features
        df = add_rolling_features(df, patient_col=self.patient_col, time_col=self.time_col)

        # 2) Clinical scores
        df["sofa_score"] = df.apply(calculate_sofa, axis=1)
        df["news2_score"] = df.apply(calculate_news2, axis=1)
        df["qsofa_score"] = df.apply(calculate_qsofa, axis=1)

        # 3) Time since last abnormal HR (minutes)
        if "heart_rate" in df.columns:
            abnormal = (df["heart_rate"] > 100) | (df["heart_rate"] < 60)
            last_abnormal_time = (
                df[self.time_col]
                .where(abnormal)
                .groupby(df[self.patient_col], sort=False)
                .ffill()
            )
            df["time_since_last_abnormal_hr"] = (
                (df[self.time_col] - last_abnormal_time).dt.total_seconds() / 60.0
            )
        else:
            df["time_since_last_abnormal_hr"] = np.nan

        # 4) Drop raw columns not needed
        raw_cols = [
            "heart_rate",
            "systolic_bp",
            "diastolic_bp",
            "temperature",
            "spo2",
            "respiratory_rate",
            "lactate",
            "wbc",
            "creatinine",
            "bilirubin",
            "platelet",
        ]
        drop_cols = [c for c in raw_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Keep id/time/label plus features
        keep_front = [c for c in [self.patient_col, self.time_col, self.label_col] if c in df.columns]
        remaining = [c for c in df.columns if c not in keep_front]
        df = df[keep_front + remaining]

        # 5) Feature columns list
        self._feature_columns = [c for c in df.columns if c not in keep_front]
        return df

    def get_feature_columns(self) -> List[str]:
        return list(self._feature_columns or [])
