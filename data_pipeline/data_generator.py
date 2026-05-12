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
    age_vitals_multiplier: float = 1.0
    severity_level: str = "moderate"  # mild/moderate/severe
    nonsepsis_bad_spikes: bool = False
    sepsis_recovery: bool = False
    current_hour: int = 0
    hr_baseline: float = 0.0
    temp_baseline: float = 0.0
    spo2_baseline: float = 0.0

    def _severity_factor(self) -> float:
        """Returns a 0..1 factor; after hour 12 sepsis worsens gradually."""
        if not self.has_sepsis:
            return 0.0
        if self.current_hour <= 12:
            return 0.0
        # ramp from hour 12 -> 24 (or beyond) smoothly
        return _clamp((self.current_hour - 12) / 12.0, 0.0, 1.0)

    def _severity_scale(self) -> float:
        # Severity affects how strongly the patient drifts into septic ranges.
        if self.severity_level == "mild":
            return 0.8
        if self.severity_level == "severe":
            return 1.25
        return 1.0

    def _age_vitals_multiplier(self) -> float:
        # Confounder: older patients have shifted baselines (~±10%).
        return float(self.age_vitals_multiplier)

    def _apply_equipment_noise(self, vitals: Dict[str, float]) -> Dict[str, float]:
        # Confounder: occasional sensor/equipment spikes per vital.
        if np.random.rand() < 0.02:
            vitals["heart_rate"] = float(vitals["heart_rate"] + np.random.normal(0.0, 40.0))
        if np.random.rand() < 0.02:
            vitals["systolic_bp"] = float(vitals["systolic_bp"] + np.random.normal(0.0, 35.0))
        if np.random.rand() < 0.02:
            vitals["diastolic_bp"] = float(vitals["diastolic_bp"] + np.random.normal(0.0, 25.0))
        if np.random.rand() < 0.02:
            vitals["temperature"] = float(vitals["temperature"] + np.random.normal(0.0, 1.2))
        if np.random.rand() < 0.02:
            vitals["spo2"] = float(vitals["spo2"] + np.random.normal(0.0, 8.0))
        if np.random.rand() < 0.02:
            vitals["respiratory_rate"] = float(vitals["respiratory_rate"] + np.random.normal(0.0, 10.0))
        return vitals

    def generate_vitals(self, timestamp: datetime) -> Dict[str, Any]:
        _ = timestamp  # timestamp can be used later for circadian effects

        age_mult = self._age_vitals_multiplier()

        if not self.has_sepsis:
            heart_rate = _rand_uniform(60, 100)
            systolic_bp = _rand_uniform(105, 135)
            diastolic_bp = _rand_uniform(65, 85)
            temperature = _rand_uniform(36.3, 37.5)
            spo2 = _rand_uniform(95, 100)
            respiratory_rate = _rand_uniform(12, 20)

            # Add overlap: some non-sepsis patients have transiently bad vitals.
            if self.nonsepsis_bad_spikes and np.random.rand() < 0.10:
                heart_rate = (heart_rate + _rand_uniform(110, 135)) / 2
                temperature = (temperature + _rand_uniform(38.2, 39.5)) / 2
                spo2 = (spo2 + _rand_uniform(88, 94)) / 2
                respiratory_rate = (respiratory_rate + _rand_uniform(22, 32)) / 2
                systolic_bp = (systolic_bp + _rand_uniform(85, 100)) / 2
                diastolic_bp = (diastolic_bp + _rand_uniform(45, 65)) / 2
        else:
            sev = _clamp(self._severity_factor() * self._severity_scale(), 0.0, 1.0)

            # Dùng baseline riêng thay vì range cố định
            heart_rate = self.hr_baseline + sev * np.random.uniform(15, 40)
            temperature = self.temp_baseline + sev * np.random.uniform(0.8, 2.5)
            spo2 = self.spo2_baseline - sev * np.random.uniform(3, 12)
            respiratory_rate = (1 - sev) * _rand_uniform(14, 20) + sev * _rand_uniform(22, 30)
            systolic_bp = (1 - sev) * _rand_uniform(100, 130) + sev * _rand_uniform(80, 95)
            diastolic_bp = (1 - sev) * _rand_uniform(60, 85) + sev * _rand_uniform(45, 65)

            # Add overlap: some sepsis patients temporarily look normal (recovery).
            if self.sepsis_recovery and np.random.rand() < 0.10:
                heart_rate = (heart_rate + _rand_uniform(60, 95)) / 2
                temperature = (temperature + _rand_uniform(36.4, 37.4)) / 2
                spo2 = (spo2 + _rand_uniform(95, 100)) / 2
                respiratory_rate = (respiratory_rate + _rand_uniform(12, 18)) / 2
                systolic_bp = (systolic_bp + _rand_uniform(105, 135)) / 2
                diastolic_bp = (diastolic_bp + _rand_uniform(65, 85)) / 2

        # Confounder: older patients tend to run higher baselines (exclude SpO2).
        heart_rate *= age_mult
        systolic_bp *= age_mult
        diastolic_bp *= age_mult
        respiratory_rate *= age_mult
        if self.age > 70:
            temperature = float(temperature + np.random.uniform(0.0, 0.25))

        # Add higher noise (harder separation)
        heart_rate = _clamp(_noisy(heart_rate, std=10.0), 30, 220)
        systolic_bp = _clamp(_noisy(systolic_bp, std=14.0), 50, 250)
        diastolic_bp = _clamp(_noisy(diastolic_bp, std=9.0), 30, 150)
        temperature = _clamp(_noisy(temperature, std=0.35), 34.0, 42.0)
        spo2 = _clamp(_noisy(spo2, std=3.0), 70.0, 100.0)
        respiratory_rate = _clamp(_noisy(respiratory_rate, std=4.0), 4.0, 60.0)

        vitals_f = {
            "heart_rate": float(heart_rate),
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "temperature": float(temperature),
            "spo2": float(spo2),
            "respiratory_rate": float(respiratory_rate),
        }
        vitals_f = self._apply_equipment_noise(vitals_f)

        # Clamp again after equipment noise
        heart_rate = _clamp(vitals_f["heart_rate"], 30, 220)
        systolic_bp = _clamp(vitals_f["systolic_bp"], 50, 250)
        diastolic_bp = _clamp(vitals_f["diastolic_bp"], 30, 150)
        temperature = _clamp(vitals_f["temperature"], 34.0, 42.0)
        spo2 = _clamp(vitals_f["spo2"], 70.0, 100.0)
        respiratory_rate = _clamp(vitals_f["respiratory_rate"], 4.0, 60.0)

        return {
            "heart_rate": round(heart_rate, 1),
            "systolic_bp": round(systolic_bp, 1),
            "diastolic_bp": round(diastolic_bp, 1),
            "temperature": round(temperature, 2),
            "spo2": round(spo2, 1),
            "respiratory_rate": round(respiratory_rate, 1),
        }


class LabResultModel:
    def generate_labs(
        self,
        timestamp: datetime,
        has_sepsis: bool,
        hour: int,
        severity_level: str = "moderate",
        nonsepsis_mild_abnormal: bool = False,
        sepsis_early_normal: bool = False,
    ) -> Dict[str, Any]:
        _ = timestamp

        def _severity_scale() -> float:
            if severity_level == "mild":
                return 0.8
            if severity_level == "severe":
                return 1.25
            return 1.0

        if not has_sepsis:
            lactate = _rand_uniform(0.5, 2.0)
            wbc = _rand_uniform(4.0, 11.0)
            creatinine = _rand_uniform(0.6, 1.2)
            bilirubin = _rand_uniform(0.1, 1.2)
            platelet = _rand_uniform(150, 450)

            # Add overlap: some non-sepsis patients show mildly abnormal labs.
            if nonsepsis_mild_abnormal:
                lactate = (lactate + _rand_uniform(1.8, 3.0)) / 2
                wbc = (wbc + _rand_uniform(10.0, 14.0)) / 2
                creatinine = (creatinine + _rand_uniform(1.0, 1.8)) / 2
                bilirubin = (bilirubin + _rand_uniform(0.8, 2.0)) / 2
                platelet = (platelet + _rand_uniform(120, 220)) / 2
        else:
            sev = _clamp((_clamp(max(0, hour - 12) / 12.0, 0.0, 1.0)) * _severity_scale(), 0.0, 1.0)
            lactate = (1 - sev) * _rand_uniform(1.8, 3.0) + sev * _rand_uniform(2.5, 6.0)

            # WBC can be high or low in sepsis
            if np.random.rand() < 0.8:
                wbc = (1 - sev) * _rand_uniform(8.0, 12.0) + sev * _rand_uniform(12.0, 22.0)
            else:
                wbc = (1 - sev) * _rand_uniform(4.0, 8.0) + sev * _rand_uniform(2.0, 4.0)

            creatinine = (1 - sev) * _rand_uniform(1.0, 1.5) + sev * _rand_uniform(1.5, 3.5)
            bilirubin = (1 - sev) * _rand_uniform(0.8, 1.8) + sev * _rand_uniform(1.2, 6.0)
            platelet = (1 - sev) * _rand_uniform(130, 220) + sev * _rand_uniform(50, 150)

            # Add overlap: some sepsis patients have near-normal labs early on.
            if sepsis_early_normal and hour < 8:
                lactate = (lactate + _rand_uniform(0.8, 2.0)) / 2
                wbc = (wbc + _rand_uniform(4.0, 11.0)) / 2
                creatinine = (creatinine + _rand_uniform(0.6, 1.2)) / 2
                bilirubin = (bilirubin + _rand_uniform(0.1, 1.2)) / 2
                platelet = (platelet + _rand_uniform(150, 450)) / 2

        # Higher lab noise (harder separation)
        lactate = _clamp(_noisy(lactate, std=0.5), 0.0, 20.0)
        wbc = _clamp(_noisy(wbc, std=2.0), 0.1, 80.0)
        creatinine = _clamp(_noisy(creatinine, std=0.3), 0.0, 20.0)
        bilirubin = _clamp(_noisy(bilirubin, std=0.35), 0.0, 50.0)
        platelet = _clamp(_noisy(platelet, std=35.0), 1.0, 1000.0)

        labs: Dict[str, Any] = {
            "lactate": round(lactate, 2),
            "wbc": round(wbc, 2),
            "creatinine": round(creatinine, 2),
            "bilirubin": round(bilirubin, 2),
            "platelet": round(platelet, 0),
        }

        # Confounder: random missing labs (~5%)
        for k in list(labs.keys()):
            if np.random.rand() < 0.05:
                labs[k] = None

        return labs


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
        n_sepsis = int(round(self.n_patients * 0.40))
        sepsis_flags = [True] * n_sepsis + [False] * (self.n_patients - n_sepsis)
        np.random.shuffle(sepsis_flags)

        # Severity distribution: mild (40%), moderate (40%), severe (20%)
        def _pick_severity() -> str:
            r = float(np.random.rand())
            if r < 0.40:
                return "mild"
            if r < 0.80:
                return "moderate"
            return "severe"

        for i in range(self.n_patients):
            patient_id = f"P{(i + 1):04d}"
            has_sepsis = bool(sepsis_flags[i])
            age = int(np.random.randint(18, 90))
            severity_level = _pick_severity()

            # Confounder: age baseline multiplier (constant per patient)
            age_vitals_multiplier = 1.0
            if age > 70:
                age_vitals_multiplier = float(1.0 + np.random.uniform(-0.10, 0.10))
                age_vitals_multiplier = float(_clamp(age_vitals_multiplier, 0.85, 1.15))

            # Overlap flags (per patient)
            nonsepsis_bad_spikes = (not has_sepsis) and (np.random.rand() < 0.20)
            sepsis_recovery = has_sepsis and (np.random.rand() < 0.15)

            nonsepsis_mild_abnormal_labs = (not has_sepsis) and (np.random.rand() < 0.25)
            sepsis_early_normal_labs = has_sepsis and (np.random.rand() < 0.20)

            self._patients[patient_id] = PhysiologicalModel(
                patient_id=patient_id,
                age=age,
                has_sepsis=has_sepsis,
                age_vitals_multiplier=age_vitals_multiplier,
                severity_level=severity_level,
                nonsepsis_bad_spikes=nonsepsis_bad_spikes,
                sepsis_recovery=sepsis_recovery,
                current_hour=0,
                hr_baseline=float(np.random.uniform(58, 92)),
                temp_baseline=float(np.random.uniform(36.1, 37.6)),
                spo2_baseline=float(np.random.uniform(95, 99)),
            )
            self._sepsis_by_patient[patient_id] = has_sepsis
            self._stream_index_by_patient[patient_id] = 0

            # attach lab overlap flags to patient object (avoids extra dicts)
            setattr(self._patients[patient_id], "nonsepsis_mild_abnormal_labs", nonsepsis_mild_abnormal_labs)
            setattr(self._patients[patient_id], "sepsis_early_normal_labs", sepsis_early_normal_labs)

    def _build_record(self, patient_id: str, timestamp: datetime, hour: int) -> Dict[str, Any]:
        patient = self._patients[patient_id]
        patient.current_hour = int(hour)

        vitals = patient.generate_vitals(timestamp)
        labs = self._lab_model.generate_labs(
            timestamp,
            has_sepsis=patient.has_sepsis,
            hour=hour,
            severity_level=str(getattr(patient, "severity_level", "moderate")),
            nonsepsis_mild_abnormal=bool(getattr(patient, "nonsepsis_mild_abnormal_labs", False)),
            sepsis_early_normal=bool(getattr(patient, "sepsis_early_normal_labs", False)),
        )

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

    def generate_dataframe(self) -> pd.DataFrame:
        """Generate a full synthetic dataset and return as a DataFrame (no file I/O)."""
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

            if records:
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
        return df[desired_cols]

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
        import requests
        ml_url = os.environ.get("ML_SERVICE_URL", "http://localhost:8001").rstrip("/")
        patient_ids = sorted(gen._patients.keys())
        print(f"Streaming data to {ml_url}/vitals ...")
        while True:
            patient_id = str(np.random.choice(patient_ids))
            rec = gen.stream_one_record(patient_id)
            print(f"[{datetime.now().isoformat()}] Sending vitals for {patient_id}...")
            try:
                resp = requests.post(f"{ml_url}/vitals", json=rec, timeout=5.0)
                if resp.status_code == 200:
                    print(f"Success: {resp.json().get('risk_level')} risk")
                else:
                    print(f"Failed: HTTP {resp.status_code} - {resp.text}")
            except Exception as e:
                print(f"Error connecting to ml_service: {e}")
            time.sleep(args.interval)
