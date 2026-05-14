#!/usr/bin/env python3
"""
ICU Realtime Simulation — 20 bệnh nhân, 40 giờ mô phỏng.
Mỗi 10 giây gửi 1 data point (= 1 giờ mô phỏng).
Tổng: 40 giờ x 6 rounds/giờ x 10s = 400 giây thực tế.

Nhóm A (P0002,P0004,P0007,P0008,P0011,P0013,P0015,P0017,P0020, P0006):
  LOW → gần WARNING → dao động sin, không vượt WARNING

Nhóm B (P0001,P0003,P0005,P0009,P0010,P0012,P0014,P0016,P0018,P0019):
  LOW → WARNING → HIGH (Sepsis), tăng từ từ
"""

import math
import time
import concurrent.futures
from datetime import datetime, timezone

import httpx

ML_SERVICE_URL = "http://ml_service:8001/vitals"
TOTAL_STEPS = 240   # 40h x 6 steps/h
STEP_SLEEP  = 10    # giây thực giữa mỗi step

# ---------- helpers ----------

def _clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))

def _lerp(a, b, t):
    """Linear interpolate a→b tại t∈[0,1]."""
    return a + (b - a) * t

def sanitize(v: dict) -> dict:
    v["heart_rate"]       = _clamp(v["heart_rate"],       30,  200)
    v["systolic_bp"]      = _clamp(v["systolic_bp"],      60,  200)
    v["diastolic_bp"]     = _clamp(v["diastolic_bp"],     30,  130)
    v["temperature"]      = _clamp(v["temperature"],      34.0, 41.5)
    v["spo2"]             = _clamp(v["spo2"],             70.0, 100.0)
    v["respiratory_rate"] = _clamp(v["respiratory_rate"], 6,   50)
    v["lactate"]          = _clamp(v["lactate"],          0.1, 15.0)
    v["wbc"]              = _clamp(v["wbc"],              0.5, 40.0)
    v["creatinine"]       = _clamp(v["creatinine"],       0.1, 12.0)
    v["bilirubin"]        = _clamp(v["bilirubin"],        0.0, 25.0)
    v["platelet"]         = _clamp(v["platelet"],         10,  800)
    if v["systolic_bp"] <= v["diastolic_bp"] + 5:
        v["systolic_bp"] = min(200.0, v["diastolic_bp"] + 10.0)
    return {k: round(float(val), 2) for k, val in v.items()}

# ---------- vitals profiles ----------

# Mức LOW bình thường (ban đầu tất cả bắt đầu ở đây)
LOW = {
    "heart_rate": 72, "systolic_bp": 120, "diastolic_bp": 78,
    "temperature": 36.8, "spo2": 98.5, "respiratory_rate": 14,
    "lactate": 1.0, "wbc": 7.0, "creatinine": 0.8,
    "bilirubin": 0.5, "platelet": 280,
}

# Gần WARNING nhưng chưa vượt (đỉnh sin nhóm A)
NEAR_WARNING = {
    "heart_rate": 98, "systolic_bp": 95, "diastolic_bp": 62,
    "temperature": 37.9, "spo2": 94.5, "respiratory_rate": 20,
    "lactate": 2.0, "wbc": 11.5, "creatinine": 1.2,
    "bilirubin": 1.2, "platelet": 175,
}

# WARNING rõ ràng
WARNING = {
    "heart_rate": 108, "systolic_bp": 88, "diastolic_bp": 56,
    "temperature": 38.5, "spo2": 93.0, "respiratory_rate": 23,
    "lactate": 2.8, "wbc": 14.0, "creatinine": 1.6,
    "bilirubin": 2.0, "platelet": 135,
}

# HIGH / Sepsis rõ ràng
HIGH = {
    "heart_rate": 128, "systolic_bp": 75, "diastolic_bp": 45,
    "temperature": 39.4, "spo2": 89.5, "respiratory_rate": 29,
    "lactate": 4.5, "wbc": 20.0, "creatinine": 2.8,
    "bilirubin": 3.5, "platelet": 72,
}

# ---------- bệnh nhân ----------

GROUP_A = ["P0002","P0004","P0007","P0008","P0011","P0013","P0015","P0017","P0020","P0006"]
GROUP_B = ["P0001","P0003","P0005","P0009","P0010","P0012","P0014","P0016","P0018","P0019"]

ALL_PATIENTS = GROUP_A + GROUP_B

import random

def _noise(base: dict, scale: float = 0.02) -> dict:
    """Thêm nhiễu nhỏ ±scale% vào mỗi chỉ số."""
    return {k: v * (1 + random.uniform(-scale, scale)) for k, v in base.items()}

def vitals_group_a(step: int) -> dict:
    """
    Nhóm A: LOW → NEAR_WARNING rồi dao động sin.
    Phase 1 (step 0-60): tăng dần LOW → NEAR_WARNING
    Phase 2 (step 60-240): sin giữa LOW và NEAR_WARNING
    """
    if step < 60:
        t = step / 60.0
        # easing: tăng chậm lúc đầu
        t = t * t
        base = {k: _lerp(LOW[k], NEAR_WARNING[k], t) for k in LOW}
    else:
        # sin từ LOW đến NEAR_WARNING, chu kỳ ~60 steps
        phase = (step - 60) / 60.0 * 2 * math.pi
        t = (math.sin(phase - math.pi / 2) + 1) / 2  # 0→1 sin
        t = t * 0.85  # không chạm đỉnh NEAR_WARNING hoàn toàn
        base = {k: _lerp(LOW[k], NEAR_WARNING[k], t) for k in LOW}
    return sanitize(_noise(base, 0.015))

def vitals_group_b(step: int) -> dict:
    """
    Nhóm B: LOW → WARNING → HIGH, tăng từ từ.
    Phase 1 (step 0-80):   LOW → WARNING (easing)
    Phase 2 (step 80-160): WARNING (dao động nhẹ)
    Phase 3 (step 160-240): WARNING → HIGH (easing)
    """
    if step < 80:
        t = (step / 80.0) ** 1.5  # tăng chậm đầu
        base = {k: _lerp(LOW[k], WARNING[k], t) for k in LOW}
    elif step < 160:
        t = (step - 80) / 80.0
        # dao động nhẹ quanh WARNING
        wobble = math.sin(t * 4 * math.pi) * 0.08
        base = {k: WARNING[k] * (1 + wobble * 0.1) for k in LOW}
    else:
        t = ((step - 160) / 80.0) ** 1.3
        base = {k: _lerp(WARNING[k], HIGH[k], min(t, 1.0)) for k in LOW}
    return sanitize(_noise(base, 0.02))

# ---------- send ----------

def send_patient(patient_id: str, step: int):
    if patient_id in GROUP_A:
        vitals = vitals_group_a(step)
    else:
        vitals = vitals_group_b(step)

    payload = {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **vitals,
    }
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.post(ML_SERVICE_URL, json=payload)
            if resp.status_code == 200:
                r = resp.json()
                ew = r.get("early_warning", {})
                return (f"{patient_id} | {r.get('risk_level','?'):8s} "
                        f"score={r.get('risk_score',0):.3f} "
                        f"EW={ew.get('early_warning_level','?')}({ew.get('early_warning_probability',0)*100:.0f}%)")
            return f"{patient_id} | HTTP {resp.status_code}"
    except Exception as e:
        return f"{patient_id} | ERROR: {e}"

def send_all(step: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(send_patient, pid, step): pid for pid in ALL_PATIENTS}
        for f in concurrent.futures.as_completed(futures):
            print(" ", f.result())

# ---------- main ----------

def main():
    print("=" * 65)
    print(f" ICU Simulation | {TOTAL_STEPS} steps x {STEP_SLEEP}s = {TOTAL_STEPS*STEP_SLEEP}s thực tế")
    print(f" Nhóm A (10 BN): LOW → sin(NEAR_WARNING)")
    print(f" Nhóm B (10 BN): LOW → WARNING → HIGH")
    print("=" * 65)

    for step in range(TOTAL_STEPS):
        hour_sim = step / 6.0
        print(f"\n[Step {step+1:03d}/{TOTAL_STEPS} | Giờ mô phỏng {hour_sim:.1f}h | {datetime.now().strftime('%H:%M:%S')}]")
        send_all(step)
        if step < TOTAL_STEPS - 1:
            time.sleep(STEP_SLEEP)

    print("\n Simulation complete.")

if __name__ == "__main__":
    main()
