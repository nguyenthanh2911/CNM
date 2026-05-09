from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _rand_uniform(low: float, high: float) -> float:
    return float(np.random.uniform(low, high))


def _noisy(value: float, std: float) -> float:
    return float(value + np.random.normal(0.0, std))


@dataclass
class PhysiologicalModel:
    patient_id: str
    age: int
    has_sepsis: bool
    current_hour: int = 0

    def _severity_factor(self) -> float:
        """Returns a 0..1 factor; after hour 12 sepsis worsens gradually."""
        if not self.has_sepsis:
            return 0.0
        if self.current_hour <= 12:
            return 0.0
        # ramp from hour 12 -> 24 (or beyond) smoothly
        return _clamp((self.current_hour - 12) / 12.0, 0.0, 1.0)

    def generate_vitals(self, timestamp: datetime) -> Dict[str, Any]:
        _ = timestamp  # timestamp can be used later for circadian effects

        if not self.has_sepsis:
            heart_rate = _rand_uniform(60, 100)
            systolic_bp = _rand_uniform(105, 135)
            diastolic_bp = _rand_uniform(65, 85)
            temperature = _rand_uniform(36.3, 37.5)
            spo2 = _rand_uniform(95, 100)
            respiratory_rate = _rand_uniform(12, 20)
        else:
            sev = self._severity_factor()

            # Start near normal and drift towards septic ranges after hour 12.
            heart_rate = (1 - sev) * _rand_uniform(70, 105) + sev * _rand_uniform(110, 130)
            temperature = (1 - sev) * _rand_uniform(36.8, 37.8) + sev * _rand_uniform(38.5, 40.0)
            spo2 = (1 - sev) * _rand_uniform(94, 99) + sev * _rand_uniform(88, 94)
            respiratory_rate = (1 - sev) * _rand_uniform(14, 20) + sev * _rand_uniform(22, 30)
            systolic_bp = (1 - sev) * _rand_uniform(100, 130) + sev * _rand_uniform(80, 95)
            diastolic_bp = (1 - sev) * _rand_uniform(60, 85) + sev * _rand_uniform(45, 65)

        # Add small noise
        heart_rate = _clamp(_noisy(heart_rate, std=2.5), 30, 220)
        systolic_bp = _clamp(_noisy(systolic_bp, std=3.0), 50, 250)
        diastolic_bp = _clamp(_noisy(diastolic_bp, std=2.0), 30, 150)
        temperature = _clamp(_noisy(temperature, std=0.08), 34.0, 42.0)
        spo2 = _clamp(_noisy(spo2, std=0.8), 70.0, 100.0)
        respiratory_rate = _clamp(_noisy(respiratory_rate, std=1.0), 4.0, 60.0)

        return {
            "heart_rate": round(heart_rate, 1),
            "systolic_bp": round(systolic_bp, 1),
            "diastolic_bp": round(diastolic_bp, 1),
            "temperature": round(temperature, 2),
            "spo2": round(spo2, 1),
            "respiratory_rate": round(respiratory_rate, 1),
        }


class LabResultModel:
    def generate_labs(self, timestamp: datetime, has_sepsis: bool, hour: int) -> Dict[str, Any]:
        _ = timestamp

        if not has_sepsis:
            lactate = _rand_uniform(0.5, 2.0)
            wbc = _rand_uniform(4.0, 11.0)
            creatinine = _rand_uniform(0.6, 1.2)
            bilirubin = _rand_uniform(0.1, 1.2)
            platelet = _rand_uniform(150, 450)
        else:
            sev = _clamp(max(0, hour - 12) / 12.0, 0.0, 1.0)
            lactate = (1 - sev) * _rand_uniform(1.8, 3.0) + sev * _rand_uniform(2.5, 6.0)

            # WBC can be high or low in sepsis
            if np.random.rand() < 0.8:
                wbc = (1 - sev) * _rand_uniform(8.0, 12.0) + sev * _rand_uniform(12.0, 22.0)
            else:
                wbc = (1 - sev) * _rand_uniform(4.0, 8.0) + sev * _rand_uniform(2.0, 4.0)

            creatinine = (1 - sev) * _rand_uniform(1.0, 1.5) + sev * _rand_uniform(1.5, 3.5)
            bilirubin = (1 - sev) * _rand_uniform(0.8, 1.8) + sev * _rand_uniform(1.2, 6.0)
            platelet = (1 - sev) * _rand_uniform(130, 220) + sev * _rand_uniform(50, 150)

        lactate = _clamp(_noisy(lactate, std=0.15), 0.0, 20.0)
        wbc = _clamp(_noisy(wbc, std=0.6), 0.1, 80.0)
        creatinine = _clamp(_noisy(creatinine, std=0.08), 0.0, 20.0)
        bilirubin = _clamp(_noisy(bilirubin, std=0.12), 0.0, 50.0)
        platelet = _clamp(_noisy(platelet, std=8.0), 1.0, 1000.0)

        return {
            "lactate": round(lactate, 2),
            "wbc": round(wbc, 2),
            "creatinine": round(creatinine, 2),
            "bilirubin": round(bilirubin, 2),
            "platelet": round(platelet, 0),
        }


class ICUSepsisGenerator:
    def __init__(self, n_patients: int = 20, hours: int = 24, interval_minutes: int = 5):
        self.n_patients = int(n_patients)
        self.hours = int(hours)
        self.interval_minutes = int(interval_minutes)

        self._lab_model = LabResultModel()
        self._patients: Dict[str, PhysiologicalModel] = {}
        self._sepsis_by_patient: Dict[str, bool] = {}

        # streaming state
        self._stream_start: datetime = datetime.now(timezone.utc)
        self._stream_index_by_patient: Dict[str, int] = {}
        self._last_record_by_patient: Dict[str, Dict[str, Any]] = {}

        self._init_patients()

    def _init_patients(self) -> None:
        n_sepsis = int(round(self.n_patients * 0.30))
        sepsis_flags = [True] * n_sepsis + [False] * (self.n_patients - n_sepsis)
        np.random.shuffle(sepsis_flags)

        for i in range(self.n_patients):
            patient_id = f"P{(i + 1):04d}"
            has_sepsis = bool(sepsis_flags[i])
            age = int(np.random.randint(18, 90))
            self._patients[patient_id] = PhysiologicalModel(
                patient_id=patient_id,
                age=age,
                has_sepsis=has_sepsis,
                current_hour=0,
            )
            self._sepsis_by_patient[patient_id] = has_sepsis
            self._stream_index_by_patient[patient_id] = 0

    def _build_record(self, patient_id: str, timestamp: datetime, hour: int) -> Dict[str, Any]:
        patient = self._patients[patient_id]
        patient.current_hour = int(hour)

        vitals = patient.generate_vitals(timestamp)
        labs = self._lab_model.generate_labs(timestamp, has_sepsis=patient.has_sepsis, hour=hour)

        record: Dict[str, Any] = {
            "patient_id": patient_id,
            "timestamp": timestamp.isoformat(),
            **vitals,
            **labs,
            "sepsis_label": int(patient.has_sepsis),
        }
        return record

    def generate_csv(self, output_path: str) -> None:
        output_path = str(output_path)
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        records: List[Dict[str, Any]] = []
        start = datetime.now(timezone.utc)
        total_records_per_patient = int(self.hours * 60 / self.interval_minutes)
        delta = timedelta(minutes=self.interval_minutes)

        for patient_id in sorted(self._patients.keys()):
            for idx in range(total_records_per_patient):
                timestamp = start + idx * delta
                hour = int((idx * self.interval_minutes) // 60)
                record = self._build_record(patient_id, timestamp, hour)
                records.append(record)

            # Keep last record around for optional stream usage
            self._last_record_by_patient[patient_id] = records[-1]

        df = pd.DataFrame.from_records(records)
        desired_cols = [
            "patient_id",
            "timestamp",
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
            "sepsis_label",
        ]
        df = df[desired_cols]
        df.to_csv(output_path, index=False)

    def stream_one_record(self, patient_id: str) -> Dict[str, Any]:
        if patient_id not in self._patients:
            raise KeyError(f"Unknown patient_id: {patient_id}")

        idx = self._stream_index_by_patient.get(patient_id, 0)
        timestamp = self._stream_start + timedelta(minutes=idx * self.interval_minutes)
        hour = int((idx * self.interval_minutes) // 60)

        record = self._build_record(patient_id, timestamp, hour)

        self._stream_index_by_patient[patient_id] = idx + 1
        self._last_record_by_patient[patient_id] = record
        return record


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic ICU Sepsis data generator")
    parser.add_argument("--mode", choices=["csv", "stream"], default="csv")
    parser.add_argument("--patients", type=int, default=20)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/icu_data_synthetic.csv",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    gen = ICUSepsisGenerator(
        n_patients=args.patients,
        hours=args.hours,
        interval_minutes=args.interval,
    )

    if args.mode == "csv":
        gen.generate_csv(args.output)
        print(f"Wrote synthetic ICU data to: {args.output}")
    else:
        patient_ids = sorted(gen._patients.keys())
        while True:
            # Print exactly one record per tick, for a simple stream simulation.
            patient_id = str(np.random.choice(patient_ids))
            rec = gen.stream_one_record(patient_id)
            print(rec, flush=True)
            time.sleep(args.interval)
