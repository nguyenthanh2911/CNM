from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine, text


@dataclass
class DriftResult:
    drift_score: float
    is_drift: bool
    drifted_columns: List[str]
    checked_at: str


class DriftDetector:
    def __init__(self, reference_data_path: str, db_url: str | None):
        self.reference_data_path = str(reference_data_path)

        if not db_url:
            db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

        if not db_url:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            user = os.getenv("POSTGRES_USER", "sepsis_user")
            password = os.getenv("POSTGRES_PASSWORD", "sepsis_pass")
            db = os.getenv("POSTGRES_DB", "sepsis_db")
            db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

        self.db_url = str(db_url)

        self.reference_data = pd.read_parquet(self.reference_data_path)
        self._engine = create_engine(self.db_url, pool_pre_ping=True)
        self._report: Optional[Report] = None
        self._last_result: Optional[Dict[str, Any]] = None

    def get_recent_production_data(self, hours: int = 24) -> pd.DataFrame:
        """Query vital_records in the last N hours from PostgreSQL."""
        since = datetime.now(timezone.utc) - timedelta(hours=int(hours))

        query = text(
            """
            SELECT
              patient_id,
              timestamp,
              heart_rate,
              systolic_bp,
              diastolic_bp,
              temperature,
              spo2,
              respiratory_rate,
              lactate,
              wbc,
              creatinine,
              bilirubin,
              platelet
            FROM vital_records
            WHERE timestamp >= :since
            ORDER BY timestamp DESC
            """
        )

        with self._engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"since": since.replace(tzinfo=None)})
        return df

    def _extract_drift_metrics(self, report_dict: Dict[str, Any]) -> tuple[bool, float, List[str]]:
        dataset_drift = False
        share_of_drifted_columns = 0.0
        drifted_columns: List[str] = []

        for m in report_dict.get("metrics", []):
            if m.get("metric") != "DataDriftPreset":
                continue
            result = m.get("result") or {}

            dataset_drift = bool(result.get("dataset_drift", False))
            share_of_drifted_columns = float(result.get("share_of_drifted_columns", 0.0) or 0.0)

            drift_by_columns = result.get("drift_by_columns") or {}
            if isinstance(drift_by_columns, dict):
                for col, col_res in drift_by_columns.items():
                    if isinstance(col_res, dict) and bool(col_res.get("drift_detected", False)):
                        drifted_columns.append(str(col))

            break

        drifted_columns = sorted(set(drifted_columns))
        return dataset_drift, share_of_drifted_columns, drifted_columns

    def detect_drift(self) -> Dict[str, Any]:
        reference = self.reference_data.sample(min(len(self.reference_data), 1000), random_state=42)
        current = self.get_recent_production_data(hours=24)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report_dict = report.as_dict()

        _dataset_drift, drift_score, drifted_columns = self._extract_drift_metrics(report_dict)

        result = DriftResult(
            drift_score=float(drift_score),
            is_drift=bool(drift_score > 0.7),
            drifted_columns=drifted_columns,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

        self._report = report
        self._last_result = {
            "drift_score": result.drift_score,
            "is_drift": result.is_drift,
            "drifted_columns": result.drifted_columns,
            "checked_at": result.checked_at,
        }
        return dict(self._last_result)

    def save_report(self, output_path: str = "reports/drift/") -> str:
        """Save the last Evidently HTML report to output_path and return the file path."""
        if self._report is None:
            self.detect_drift()

        assert self._report is not None

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"drift_report_{ts}.html"
        self._report.save_html(str(out_file))
        return str(out_file)
