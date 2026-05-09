from __future__ import annotations

from typing import Any


def _get_float(row: Any, key: str, default: float | None = None) -> float | None:
    try:
        val = row[key]
    except Exception:
        val = getattr(row, key, default)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


def calculate_sofa(row) -> int:
    spo2 = _get_float(row, "spo2")
    platelet = _get_float(row, "platelet")
    bilirubin = _get_float(row, "bilirubin")
    creatinine = _get_float(row, "creatinine")
    systolic_bp = _get_float(row, "systolic_bp")

    score = 0

    # respiratory (SpO2 proxy)
    if spo2 is not None:
        if spo2 < 90:
            score += 4
        elif spo2 < 94:
            score += 3
        elif spo2 < 96:
            score += 2
        elif spo2 < 98:
            score += 1

    # coagulation
    if platelet is not None:
        if platelet < 50:
            score += 4
        elif platelet < 100:
            score += 3
        elif platelet < 150:
            score += 2
        elif platelet < 200:
            score += 1

    # liver
    if bilirubin is not None:
        if bilirubin > 12:
            score += 4
        elif bilirubin > 6:
            score += 3
        elif bilirubin > 2:
            score += 2
        elif bilirubin > 1.2:
            score += 1

    # renal
    if creatinine is not None:
        if creatinine > 5:
            score += 4
        elif creatinine > 3.5:
            score += 3
        elif creatinine > 2:
            score += 2
        elif creatinine > 1.2:
            score += 1

    # cardiovascular
    if systolic_bp is not None:
        if systolic_bp < 70:
            score += 4
        elif systolic_bp < 80:
            score += 3
        elif systolic_bp < 90:
            score += 2
        elif systolic_bp < 100:
            score += 1

    return int(score)


def calculate_news2(row) -> int:
    resp_rate = _get_float(row, "respiratory_rate")
    spo2 = _get_float(row, "spo2")
    temperature = _get_float(row, "temperature")
    systolic_bp = _get_float(row, "systolic_bp")
    heart_rate = _get_float(row, "heart_rate")

    score = 0

    # resp_rate
    if resp_rate is not None:
        if resp_rate >= 25:
            score += 3
        elif resp_rate >= 21:
            score += 2
        elif resp_rate >= 18:
            score += 1
        elif 12 <= resp_rate <= 17:
            score += 0
        elif 9 <= resp_rate <= 11:
            score += 1
        elif resp_rate <= 8:
            score += 3

    # spo2
    if spo2 is not None:
        if spo2 <= 91:
            score += 3
        elif 92 <= spo2 <= 93:
            score += 2
        elif 94 <= spo2 <= 95:
            score += 1
        elif spo2 >= 96:
            score += 0

    # temperature
    if temperature is not None:
        if temperature <= 35:
            score += 3
        elif 35.1 <= temperature <= 36:
            score += 1
        elif 36.1 <= temperature <= 38:
            score += 0
        elif 38.1 <= temperature <= 39:
            score += 1
        elif temperature > 39:
            score += 2

    # systolic_bp
    if systolic_bp is not None:
        if systolic_bp <= 90:
            score += 3
        elif 91 <= systolic_bp <= 100:
            score += 2
        elif 101 <= systolic_bp <= 110:
            score += 1
        elif 111 <= systolic_bp <= 219:
            score += 0
        elif systolic_bp >= 220:
            score += 3

    # heart_rate
    if heart_rate is not None:
        if heart_rate <= 40:
            score += 3
        elif 41 <= heart_rate <= 50:
            score += 1
        elif 51 <= heart_rate <= 90:
            score += 0
        elif 91 <= heart_rate <= 110:
            score += 1
        elif 111 <= heart_rate <= 130:
            score += 2
        elif heart_rate >= 131:
            score += 3

    return int(score)


def calculate_qsofa(row) -> int:
    resp_rate = _get_float(row, "respiratory_rate")
    systolic_bp = _get_float(row, "systolic_bp")

    score = 0
    if resp_rate is not None and resp_rate >= 22:
        score += 1
    if systolic_bp is not None and systolic_bp <= 100:
        score += 1

    # No GCS in dataset; max 2.
    return int(min(score, 2))
