from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from django.conf import settings
from django.db.models import QuerySet
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_GET

from .models import Alert, Prediction


def _risk_badge(level: str) -> str:
    level = (level or "").upper()
    if level == "CRITICAL":
        return "CRITICAL"
    if level == "WARNING":
        return "WARNING"
    return "STABLE"


def patient_list(request: HttpRequest) -> HttpResponse:
    # Option A: query DB directly for latest prediction per patient (Postgres DISTINCT ON)
    latest: QuerySet[Prediction] = Prediction.objects.order_by("patient_id", "-timestamp").distinct("patient_id")

    patients: List[Dict[str, Any]] = []
    critical = 0
    warning = 0

    for row in latest:
        badge = _risk_badge(row.risk_level)
        if badge == "CRITICAL":
            critical += 1
        elif badge == "WARNING":
            warning += 1

        # Synthetic name/room (since not stored)
        digits = "".join([c for c in row.patient_id if c.isdigit()])
        room = f"ICU-{int(digits or 0) % 10:02d}"

        patients.append(
            {
                "patient_id": row.patient_id,
                "name": f"Patient {row.patient_id}",
                "room": room,
                "risk_score": float(row.risk_score),
                "risk_pct": float(row.risk_score) * 100.0,
                "risk_level": badge,
                "timestamp": row.timestamp,
            }
        )

    total = len(patients)
    stable = max(total - critical - warning, 0)

    return render(
        request,
        "dashboard/patient_list.html",
        {
            "patients": patients,
            "total": total,
            "critical": critical,
            "warning": warning,
            "stable": stable,
        },
    )


@require_GET
def api_dashboard_data(request: HttpRequest) -> JsonResponse:
    """JSON API trả về dữ liệu dashboard mới nhất – dùng cho AJAX polling."""
    latest: QuerySet[Prediction] = Prediction.objects.order_by("patient_id", "-timestamp").distinct("patient_id")

    patients: List[Dict[str, Any]] = []
    critical = 0
    warning = 0

    for row in latest:
        badge = _risk_badge(row.risk_level)
        if badge == "CRITICAL":
            critical += 1
        elif badge == "WARNING":
            warning += 1

        digits = "".join([c for c in row.patient_id if c.isdigit()])
        room = f"ICU-{int(digits or 0) % 10:02d}"

        patients.append(
            {
                "patient_id": row.patient_id,
                "name": f"Patient {row.patient_id}",
                "room": room,
                "risk_score": round(float(row.risk_score), 4),
                "risk_pct": round(float(row.risk_score) * 100.0, 1),
                "risk_level": badge,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            }
        )

    total = len(patients)
    stable = max(total - critical - warning, 0)

    return JsonResponse(
        {
            "total": total,
            "critical": critical,
            "warning": warning,
            "stable": stable,
            "patients": patients,
        }
    )


@require_GET
def api_patient_detail(request: HttpRequest, patient_id: str) -> JsonResponse:
    """JSON API trả về dữ liệu chi tiết của 1 bệnh nhân – dùng cho AJAX polling trang detail."""
    qs = Prediction.objects.filter(patient_id=patient_id).order_by("-timestamp")[:24]
    records = list(reversed(list(qs)))

    risk_series = [
        {
            "t": r.timestamp.isoformat(),
            "risk": round(float(r.risk_score), 4),
            "level": _risk_badge(r.risk_level),
            "sofa": int(r.sofa_score),
            "news2": int(r.news2_score),
        }
        for r in records
    ]

    latest_pred = records[-1] if records else None
    risk_score = round(float(latest_pred.risk_score), 4) if latest_pred else 0.0
    level = _risk_badge(latest_pred.risk_level) if latest_pred else "STABLE"

    return JsonResponse(
        {
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_level": level,
            "sofa_score": int(latest_pred.sofa_score) if latest_pred else 0,
            "news2_score": int(latest_pred.news2_score) if latest_pred else 0,
            "risk_series": risk_series,
        }
    )


def patient_detail(request: HttpRequest, patient_id: str) -> HttpResponse:
    # 24 latest prediction records for chart
    qs = Prediction.objects.filter(patient_id=patient_id).order_by("-timestamp")[:24]
    records = list(reversed(list(qs)))

    risk_series = [
        {
            "t": r.timestamp.isoformat(),
            "risk": float(r.risk_score),
            "level": _risk_badge(r.risk_level),
            "sofa": int(r.sofa_score),
            "news2": int(r.news2_score),
        }
        for r in records
    ]

    latest_pred = records[-1] if records else None

    # Try to fetch vitals/shap from ML service history endpoint (optional; may not exist)
    vitals: Dict[str, Any] = {}
    shap_features: List[Dict[str, Any]] = []

    ml_url = getattr(settings, "ML_SERVICE_URL", "http://localhost:8001").rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{ml_url}/vitals/{patient_id}/history")
            if resp.status_code == 200:
                payload = resp.json()
                vitals = payload.get("latest_vitals", {}) or {}
                shap_features = payload.get("top_features", []) or []
    except Exception:
        pass

    # Fallback: nếu ML service không trả vitals, lấy từ DB
    if not vitals and latest_pred:
        vitals = {
            "heart_rate": latest_pred.heart_rate,
            "systolic_bp": latest_pred.systolic_bp,
            "diastolic_bp": latest_pred.diastolic_bp,
            "temperature": latest_pred.temperature,
            "spo2": latest_pred.spo2,
            "respiratory_rate": latest_pred.respiratory_rate,
            "lactate": latest_pred.lactate,
            "wbc": latest_pred.wbc,
            "creatinine": latest_pred.creatinine,
            "bilirubin": latest_pred.bilirubin,
            "platelet": latest_pred.platelet,
        }
        # Bỏ None values
        vitals = {k: v for k, v in vitals.items() if v is not None}

    # Fallback: use latest alert (if any) for SHAP features
    if not shap_features:
        alert = Alert.objects.filter(patient_id=patient_id).order_by("-created_at").first()
        if alert and isinstance(alert.top_features, list):
            shap_features = alert.top_features[:5]

    # Determine current risk
    risk_score = float(latest_pred.risk_score) if latest_pred else 0.0
    level = _risk_badge(latest_pred.risk_level) if latest_pred else "STABLE"

    # Determine pending critical alert for acknowledge button
    pending_alert = (
        Alert.objects.filter(patient_id=patient_id, acknowledged=False, risk_level="CRITICAL")
        .order_by("-created_at")
        .first()
    )

    return render(
        request,
        "dashboard/patient_detail.html",
        {
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_level": level,
            "sofa_score": int(latest_pred.sofa_score) if latest_pred else 0,
            "news2_score": int(latest_pred.news2_score) if latest_pred else 0,
            "risk_series_json": mark_safe(json.dumps(risk_series, ensure_ascii=False)),
            "shap_features_json": mark_safe(json.dumps(shap_features, ensure_ascii=False)),
            "vitals": vitals,
            "pending_alert": pending_alert,
        },
    )


def alerts_page(request: HttpRequest) -> HttpResponse:
    status = request.GET.get("status", "all")
    if status not in {"all", "pending", "confirmed"}:
        status = "all"

    alert_url = getattr(settings, "ALERT_SERVICE_URL", "http://localhost:8002").rstrip("/")

    alerts: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{alert_url}/alerts", params={"limit": 50, "status": status})
            if resp.status_code == 200:
                alerts = resp.json()
    except Exception:
        alerts = []

    return render(
        request,
        "dashboard/alerts.html",
        {
            "alerts": alerts,
            "status": status,
            "alert_service_url": alert_url,
        },
    )


def acknowledge_alert(request: HttpRequest, alert_id: str) -> HttpResponse:
    if request.method != "POST":
        return redirect("alerts_page")

    ack_by = request.POST.get("ack_by", "dashboard")

    alert_url = getattr(settings, "ALERT_SERVICE_URL", "http://localhost:8002").rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as client:
            client.patch(
                f"{alert_url}/alerts/{alert_id}/acknowledge",
                json={"ack_by": ack_by},
            )
    except Exception:
        pass

    return redirect(f"/alerts/?status=pending")
