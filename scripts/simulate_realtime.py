#!/usr/bin/env python3
"""
Simulate 20 ICU patients sending vitals to ML service in realtime.
Each patient sends new vitals every 10 seconds, indefinitely.
"""

import random
import time
import threading
from datetime import datetime, timezone

import httpx

ML_SERVICE_URL = "http://localhost:8001/vitals"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def sanitize_vitals(vitals: dict) -> dict:
    """Clamp vitals into plausible physiological ranges to avoid API 422."""
    v = dict(vitals)

    v["heart_rate"] = _clamp(v.get("heart_rate", 0), 30, 220)
    v["systolic_bp"] = _clamp(v.get("systolic_bp", 0), 60, 220)
    v["diastolic_bp"] = _clamp(v.get("diastolic_bp", 0), 30, 140)
    v["temperature"] = _clamp(v.get("temperature", 0), 34.0, 42.0)
    v["spo2"] = _clamp(v.get("spo2", 0), 70.0, 100.0)
    v["respiratory_rate"] = _clamp(v.get("respiratory_rate", 0), 6, 60)
    v["lactate"] = _clamp(v.get("lactate", 0), 0.0, 15.0)
    v["wbc"] = _clamp(v.get("wbc", 0), 0.5, 50.0)
    v["creatinine"] = _clamp(v.get("creatinine", 0), 0.1, 15.0)
    v["bilirubin"] = _clamp(v.get("bilirubin", 0), 0.0, 30.0)
    v["platelet"] = _clamp(v.get("platelet", 0), 1.0, 1000.0)

    # Ensure systolic stays above diastolic by a small margin
    if v["systolic_bp"] <= v["diastolic_bp"] + 5:
        v["systolic_bp"] = min(220.0, v["diastolic_bp"] + 10.0)

    # Round to match existing output style
    for k in list(v.keys()):
        if isinstance(v[k], (int, float)):
            v[k] = round(float(v[k]), 2)
    return v


# 20 bệnh nhân với profile sinh lý khác nhau
PATIENTS = [
    # (patient_id, profile)
    # profile: base vitals + severity (0=stable, 1=warning, 2=critical)
    {"id": "P0001", "severity": 2, "name": "Nguyễn Văn An"},
    {"id": "P0002", "severity": 0, "name": "Trần Thị Bình"},
    {"id": "P0003", "severity": 1, "name": "Lê Văn Cường"},
    {"id": "P0004", "severity": 0, "name": "Phạm Thị Dung"},
    {"id": "P0005", "severity": 2, "name": "Hoàng Văn Em"},
    {"id": "P0006", "severity": 1, "name": "Vũ Thị Phương"},
    {"id": "P0007", "severity": 0, "name": "Đặng Văn Giang"},
    {"id": "P0008", "severity": 0, "name": "Bùi Thị Hoa"},
    {"id": "P0009", "severity": 1, "name": "Ngô Văn Inh"},
    {"id": "P0010", "severity": 2, "name": "Dương Thị Kim"},
    {"id": "P0011", "severity": 0, "name": "Trịnh Văn Long"},
    {"id": "P0012", "severity": 1, "name": "Lý Thị Mai"},
    {"id": "P0013", "severity": 0, "name": "Phan Văn Nam"},
    {"id": "P0014", "severity": 2, "name": "Đinh Thị Oanh"},
    {"id": "P0015", "severity": 0, "name": "Hồ Văn Phúc"},
    {"id": "P0016", "severity": 1, "name": "Cao Thị Quỳnh"},
    {"id": "P0017", "severity": 0, "name": "Mai Văn Rồng"},
    {"id": "P0018", "severity": 2, "name": "Tô Thị Sen"},
    {"id": "P0019", "severity": 1, "name": "Lưu Văn Tâm"},
    {"id": "P0020", "severity": 0, "name": "Kiều Thị Uyên"},
]


def base_vitals(severity: int) -> dict:
    """Sinh vitals nền theo mức độ bệnh."""
    if severity == 0:  # STABLE
        return {
            "heart_rate":        random.uniform(60, 85),
            "systolic_bp":       random.uniform(110, 130),
            "diastolic_bp":      random.uniform(70, 85),
            "temperature":       random.uniform(36.5, 37.2),
            "spo2":              random.uniform(97, 99),
            "respiratory_rate":  random.uniform(12, 16),
            "lactate":           random.uniform(0.5, 1.5),
            "wbc":               random.uniform(4.5, 10.0),
            "creatinine":        random.uniform(0.6, 1.1),
            "bilirubin":         random.uniform(0.2, 0.8),
            "platelet":          random.uniform(180, 350),
        }
    elif severity == 1:  # WARNING
        return {
            "heart_rate":        random.uniform(95, 115),
            "systolic_bp":       random.uniform(90, 110),
            "diastolic_bp":      random.uniform(55, 70),
            "temperature":       random.uniform(37.8, 38.8),
            "spo2":              random.uniform(93, 96),
            "respiratory_rate":  random.uniform(20, 25),
            "lactate":           random.uniform(2.0, 3.5),
            "wbc":               random.uniform(12.0, 18.0),
            "creatinine":        random.uniform(1.3, 2.0),
            "bilirubin":         random.uniform(1.0, 2.5),
            "platelet":          random.uniform(100, 170),
        }
    else:  # CRITICAL
        return {
            "heart_rate":        random.uniform(120, 145),
            "systolic_bp":       random.uniform(70, 88),
            "diastolic_bp":      random.uniform(40, 55),
            "temperature":       random.uniform(39.0, 40.5),
            "spo2":              random.uniform(88, 92),
            "respiratory_rate":  random.uniform(26, 35),
            "lactate":           random.uniform(4.0, 8.0),
            "wbc":               random.uniform(18.0, 30.0),
            "creatinine":        random.uniform(2.5, 5.0),
            "bilirubin":         random.uniform(3.0, 8.0),
            "platelet":          random.uniform(40, 90),
        }


# Lưu state vitals hiện tại của mỗi bệnh nhân (drift theo thời gian)
_state: dict[str, dict] = {}


def get_next_vitals(patient: dict) -> dict:
    """Lấy vitals tiếp theo, drift dần từ state trước và sanitize."""
    pid = patient["id"]
    sev = patient["severity"]

    if pid not in _state:
        _state[pid] = base_vitals(sev)

    # Drift nhẹ về phía base mỗi lần
    target = base_vitals(sev)
    current = _state[pid]
    drifted = {}
    for k in current:
        drift = (target[k] - current[k]) * 0.15
        drifted[k] = round(current[k] + drift + (target[k] * 0.02 * random.uniform(-1, 1)), 2)

    _state[pid] = drifted
    return sanitize_vitals(drifted)


def send_vitals(patient: dict) -> None:
    """Gửi 1 lần vitals cho 1 bệnh nhân."""
    vitals = get_next_vitals(patient)
    payload = {
        "patient_id": patient["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **vitals,
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(ML_SERVICE_URL, json=payload)
            # FastAPI returns JSON for 4xx too; make failures explicit.
            if resp.status_code != 200:
                try:
                    err = resp.json()
                except Exception:
                    err = {"text": resp.text}
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"{patient['id']:6s} | HTTP {resp.status_code} | {err}"
                )
                return

            result = resp.json()
            level = result.get("risk_level", "?")
            score = float(result.get("risk_score", 0.0) or 0.0)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"{patient['id']:6s} | {level:8s} | score={score:.3f} | "
                  f"HR={vitals['heart_rate']:.0f} BP={vitals['systolic_bp']:.0f}/"
                  f"{vitals['diastolic_bp']:.0f} SpO2={vitals['spo2']:.1f}%")
    except Exception as e:
        print(f"[ERROR] {patient['id']}: {e}")


def patient_loop(patient: dict, interval: int, stagger: float) -> None:
    """Vòng lặp gửi vitals cho 1 bệnh nhân, stagger để không gửi cùng lúc."""
    time.sleep(stagger)
    while True:
        send_vitals(patient)
        time.sleep(interval)


def main():
    print("=" * 65)
    print(" ICU Sepsis — Realtime Simulation — 20 Patients")
    print(" Ctrl+C để dừng")
    print("=" * 65)

    INTERVAL = 10    # giây giữa mỗi lần gửi vitals
    threads = []

    for i, patient in enumerate(PATIENTS):
        stagger = i * 0.5   # mỗi bệnh nhân cách nhau 0.5s để không flood
        t = threading.Thread(
            target=patient_loop,
            args=(patient, INTERVAL, stagger),
            daemon=True,
        )
        t.start()
        threads.append(t)
        sev_label = "STABLE" if patient["severity"] == 0 else "WARNING" if patient["severity"] == 1 else "CRITICAL"
        print(f"  Started thread for {patient['id']} ({patient['name']}) — severity={sev_label}")

    print(f"\n✅ {len(PATIENTS)} patient threads running. Sending every {INTERVAL}s...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⛔ Simulation stopped.")


if __name__ == "__main__":
    main()
