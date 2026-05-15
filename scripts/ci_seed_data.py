#!/usr/bin/env python3
"""CI-only: generate reference parquet + seed vital_records for retrain flow."""
import os, numpy as np, pandas as pd
from sqlalchemy import create_engine, text

# ── 1. Generate reference parquet ────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
np.random.seed(42)
n = 1000
cols = [
    "patient_id", "timestamp", "heart_rate", "systolic_bp", "diastolic_bp",
    "temperature", "spo2", "respiratory_rate", "lactate", "wbc",
    "creatinine", "bilirubin", "platelet", "sepsis_label", "early_warning_label",
]
data = {
    "patient_id": [f"P{i:04d}" for i in range(1, n + 1)],
    "timestamp": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
    "heart_rate":       np.random.normal(75, 15, n).clip(40, 160),
    "systolic_bp":      np.random.normal(120, 18, n).clip(70, 180),
    "diastolic_bp":     np.random.normal(78, 12, n).clip(40, 110),
    "temperature":      np.random.normal(37.0, 0.8, n).clip(35.5, 40.5),
    "spo2":             np.random.normal(97, 3, n).clip(80, 100),
    "respiratory_rate": np.random.normal(16, 5, n).clip(8, 40),
    "lactate":          np.random.exponential(1.5, n).clip(0.3, 12),
    "wbc":              np.random.normal(9, 4, n).clip(1, 35),
    "creatinine":       np.random.exponential(1.2, n).clip(0.3, 10),
    "bilirubin":        np.random.exponential(0.8, n).clip(0.1, 20),
    "platelet":         np.random.normal(250, 80, n).clip(20, 600),
    "sepsis_label":     np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "early_warning_label": np.random.choice([0, 1, 2], n, p=[0.6, 0.25, 0.15]),
}
pd.DataFrame(data, columns=cols).to_parquet("data/processed/features_train.parquet", index=False)
print("Created data/processed/features_train.parquet")

# ── 2. Seed vital_records ────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL") or "postgresql+psycopg2://sepsis_user:sepsis_pass@localhost:5432/sepsis_db"
engine = create_engine(DB_URL)

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vital_records (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(16) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            heart_rate FLOAT, systolic_bp FLOAT, diastolic_bp FLOAT,
            temperature FLOAT, spo2 FLOAT, respiratory_rate FLOAT,
            lactate FLOAT, wbc FLOAT, creatinine FLOAT,
            bilirubin FLOAT, platelet FLOAT
        )
    """))
    conn.commit()
print("Table vital_records ready")

np.random.seed(123)
now = pd.Timestamp.now("UTC")
rows = [
    {
        "patient_id": f"P{(i % 20) + 1:04d}",
        "timestamp": (now - pd.Timedelta(hours=i)).isoformat(),
        "heart_rate":       float(np.random.normal(85, 20).clip(40, 160)),
        "systolic_bp":      float(np.random.normal(110, 20).clip(70, 180)),
        "diastolic_bp":     float(np.random.normal(70, 15).clip(40, 110)),
        "temperature":      float(np.random.normal(37.5, 1.0).clip(35.5, 40.5)),
        "spo2":             float(np.random.normal(96, 4).clip(80, 100)),
        "respiratory_rate": float(np.random.normal(18, 6).clip(8, 40)),
        "lactate":          float(np.random.exponential(2.0).clip(0.3, 12)),
        "wbc":              float(np.random.normal(10, 5).clip(1, 35)),
        "creatinine":       float(np.random.exponential(1.5).clip(0.3, 10)),
        "bilirubin":        float(np.random.exponential(1.0).clip(0.1, 20)),
        "platelet":         float(np.random.normal(220, 90).clip(20, 600)),
    }
    for i in range(200)
]
pd.DataFrame(rows).to_sql("vital_records", engine, if_exists="append", index=False, method="multi")
print(f"Seeded {len(rows)} vital_records")
