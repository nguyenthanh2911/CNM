#!/usr/bin/env python3
"""
Generate data/processed/features_current.parquet với drift mạnh so với features_train.parquet.

Chiến lược drift (mean shift lớn để Evidently KS-test detect):
  heart_rate:       75  → 120  (+45, nhịp tim nhanh nặng)
  systolic_bp:      120 → 78   (-42, tụt huyết áp)
  diastolic_bp:     78  → 48   (-30, tụt huyết áp)
  temperature:      37.0→ 39.8 (+2.8, sốt cao)
  spo2:             97  → 84   (-13, thiếu oxy nặng)
  respiratory_rate: 16  → 32   (+16, thở nhanh)
  lactate:          1.5 → 7.0  (4.7×, nhiễm toan lactate)
  wbc:              9   → 24   (+15, tăng bạch cầu)
  creatinine:       1.2 → 5.0  (4×, suy thận)
  bilirubin:        0.8 → 4.5  (5.6×, vàng da)
  platelet:         250 → 85   (-165, giảm tiểu cầu)

=> share_of_drifted_columns ~= 1.0 >> threshold 0.3 → is_drift = True
"""
import os
import numpy as np
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

np.random.seed(2026)
n = 1000

data = {
    "patient_id":        [f"C{i:04d}" for i in range(1, n + 1)],
    "timestamp":         pd.date_range("2026-05-01", periods=n, freq="h").astype(str),
    # --- vitals drifted mạnh ---
    "heart_rate":        np.random.normal(120, 18, n).clip(60, 220),
    "systolic_bp":       np.random.normal(78,  14, n).clip(40, 130),
    "diastolic_bp":      np.random.normal(48,  10, n).clip(25,  85),
    "temperature":       np.random.normal(39.8, 0.5, n).clip(38.0, 42.0),
    "spo2":              np.random.normal(84,   6,  n).clip(60, 100),
    "respiratory_rate":  np.random.normal(32,   7,  n).clip(15,  70),
    "lactate":           np.random.exponential(7.0, n).clip(2.5, 20.0),
    "wbc":               np.random.normal(24,   7,  n).clip(8,   55),
    "creatinine":        np.random.exponential(5.0, n).clip(1.5, 18),
    "bilirubin":         np.random.exponential(4.5, n).clip(0.8, 25),
    "platelet":          np.random.normal(85,  35,  n).clip(10, 250),
}

df = pd.DataFrame(data)
out = "data/processed/features_current.parquet"
df.to_parquet(out, index=False)

print(f"[make_drift_data] Saved  : {out}")
print(f"[make_drift_data] Shape  : {df.shape}")
print(f"[make_drift_data] Columns: {list(df.columns)}")
print("[make_drift_data] Drift summary (current means):")
for col in ["heart_rate","systolic_bp","temperature","spo2","lactate","respiratory_rate","wbc"]:
    print(f"  {col:20s}: {df[col].mean():.2f}")
