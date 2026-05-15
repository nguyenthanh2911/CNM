#!/usr/bin/env python3
""" ICU Realtime Simulation — 20 bệnh nhân, 40 giờ mô phỏng.
Mỗi 10 giây = 1 giờ mô phỏng. Tổng 240 steps = 400 giây thực.

Nhóm LOW  (10 BN): dao động quanh mức bình thường, risk < 0.3
Nhóm WARN  (4 BN): LOW → WARNING từ từ, risk 0.3~0.6
Nhóm HIGH  (6 BN): LOW → WARNING → HIGH từ từ, risk lên đến 0.8+
"""

import math, random, time, concurrent.futures
from datetime import datetime, timezone

import httpx

ML_SERVICE_URL = "http://ml_service:8001/vitals"
TOTAL_STEPS    = 240
STEP_SLEEP     = 10

# ── helpers ──────────────────────────────────────────────────────────

def _clamp(x, lo, hi): return max(lo, min(hi, float(x)))
def _lerp(a, b, t):    return a + (b - a) * float(t)

def sanitize(v: dict) -> dict:
    rules = {
        "heart_rate":       (40,  160),
        "systolic_bp":      (70,  180),
        "diastolic_bp":     (40,  110),
        "temperature":      (35.5, 40.5),
        "spo2":             (80.0, 100.0),
        "respiratory_rate": (8,   40),
        "lactate":          (0.3, 12.0),
        "wbc":              (1.0, 35.0),
        "creatinine":       (0.3, 10.0),
        "bilirubin":        (0.1, 20.0),
        "platelet":         (20,  600),
    }
    out = {k: _clamp(v[k], *rules[k]) for k in rules}
    if out["systolic_bp"] <= out["diastolic_bp"] + 5:
        out["systolic_bp"] = min(180.0, out["diastolic_bp"] + 12.0)
    return {k: round(val, 2) for k, val in out.items()}

def _noise(base: dict, pct: float) -> dict:
    return {k: val * (1 + random.uniform(-pct, pct)) for k, val in base.items()}

# ── vitals profiles ───────────────────────────────────────────────────

NORMAL = {
    "heart_rate": 72,  "systolic_bp": 118, "diastolic_bp": 76,
    "temperature": 36.7, "spo2": 98.5, "respiratory_rate": 14,
    "lactate": 0.9,  "wbc": 6.5,  "creatinine": 0.75,
    "bilirubin": 0.4, "platelet": 290,
}
NEAR_WARN = {
    "heart_rate": 96,  "systolic_bp": 96,  "diastolic_bp": 63,
    "temperature": 37.8, "spo2": 95.0, "respiratory_rate": 19,
    "lactate": 1.9,  "wbc": 11.0, "creatinine": 1.15,
    "bilirubin": 1.1, "platelet": 185,
}
WARNING = {
    "heart_rate": 106, "systolic_bp": 88,  "diastolic_bp": 57,
    "temperature": 38.4, "spo2": 93.5, "respiratory_rate": 22,
    "lactate": 2.6,  "wbc": 13.5, "creatinine": 1.55,
    "bilirubin": 1.8, "platelet": 140,
}
HIGH = {
    "heart_rate": 124, "systolic_bp": 76,  "diastolic_bp": 46,
    "temperature": 39.3, "spo2": 90.0, "respiratory_rate": 28,
    "lactate": 4.2,  "wbc": 19.0, "creatinine": 2.7,
    "bilirubin": 3.2, "platelet": 78,
}

# ── nhóm bệnh nhân ───────────────────────────────────────────────────

GROUP_LOW  = ["P0002","P0004","P0007","P0008","P0011","P0013","P0015","P0017","P0020","P0006"]
GROUP_WARN = ["P0003","P0009","P0012","P0016"]
GROUP_HIGH = ["P0001","P0005","P0010","P0014","P0018","P0019"]
ALL        = GROUP_LOW + GROUP_WARN + GROUP_HIGH

# ── easing: smooth-step ──────────────────────────────────────────────

def _smooth(t: float) -> float:
    """Smooth-step: tăng tự nhiên, không giật."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

# ── vitals theo nhóm ─────────────────────────────────────────────────

def vitals_low(step: int) -> dict:
    """Dao động sin nhẹ quanh NORMAL, không lên WARNING."""
    phase = step / 40.0 * 2 * math.pi
    t = (math.sin(phase) + 1) / 2 * 0.45   # max 45% về phía NEAR_WARN
    base = {k: _lerp(NORMAL[k], NEAR_WARN[k], t) for k in NORMAL}
    return sanitize(_noise(base, 0.012))

def vitals_warn(step: int) -> dict:
    """
    LOW → WARNING từ từ.
    step 0-120:  NORMAL → WARNING (smooth)
    step 120-240: dao động quanh WARNING
    """
    if step < 120:
        t = _smooth(step / 120.0)
        base = {k: _lerp(NORMAL[k], WARNING[k], t) for k in NORMAL}
    else:
        s = (step - 120) / 120.0
        wobble = math.sin(s * 3 * math.pi) * 0.06
        base = {k: WARNING[k] * (1 + wobble) for k in NORMAL}
    return sanitize(_noise(base, 0.015))

def vitals_high(step: int) -> dict:
    """
    LOW → WARNING → HIGH từ từ.
    step 0-80:   NORMAL → WARNING
    step 80-160: WARNING (dao động nhẹ)
    step 160-240: WARNING → HIGH
    """
    if step < 80:
        t = _smooth(step / 80.0)
        base = {k: _lerp(NORMAL[k], WARNING[k], t) for k in NORMAL}
    elif step < 160:
        s = (step - 80) / 80.0
        wobble = math.sin(s * 4 * math.pi) * 0.05
        base = {k: WARNING[k] * (1 + wobble) for k in NORMAL}
    else:
        t = _smooth((step - 160) / 80.0)
        base = {k: _lerp(WARNING[k], HIGH[k], t) for k in NORMAL}
    return sanitize(_noise(base, 0.018))

# ── send ─────────────────────────────────────────────────────────────

def send_patient(pid: str, step: int) -> str:
    if pid in GROUP_LOW:
        vitals = vitals_low(step)
    elif pid in GROUP_WARN:
        vitals = vitals_warn(step)
    else:
        vitals = vitals_high(step)

    payload = {"patient_id": pid, "timestamp": datetime.now(timezone.utc).isoformat(), **vitals}
    try:
        with httpx.Client(timeout=8.0) as c:
            r = c.post(ML_SERVICE_URL, json=payload)
            if r.status_code == 200:
                d  = r.json()
                ew = d.get("early_warning", {})
                return (f"{pid} | {d.get('risk_level','?'):8s} "
                        f"score={d.get('risk_score',0):.3f} "
                        f"EW={ew.get('early_warning_level','?')}"
                        f"({ew.get('early_warning_probability',0)*100:.0f}%)")
            return f"{pid} | HTTP {r.status_code}"
    except Exception as e:
        return f"{pid} | ERROR: {e}"

def send_all(step: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(send_patient, pid, step): pid for pid in ALL}
        for f in concurrent.futures.as_completed(futs):
            print(" ", f.result())

# ── main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f" ICU Sim | {TOTAL_STEPS} steps x {STEP_SLEEP}s = {TOTAL_STEPS*STEP_SLEEP}s")
    print(f" LOW  (10): dao động bình thường")
    print(f" WARN  (4): LOW → WARNING")
    print(f" HIGH  (6): LOW → WARNING → HIGH")
    print("=" * 65)

    for step in range(TOTAL_STEPS):
        print(f"\n[Step {step+1:03d}/{TOTAL_STEPS} | {step/6:.1f}h sim | {datetime.now().strftime('%H:%M:%S')}]")
        send_all(step)
        if step < TOTAL_STEPS - 1:
            time.sleep(STEP_SLEEP)

    print("\nSimulation complete.")

if __name__ == "__main__":
    main()
