from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from django.conf import settings
from django.db.models import QuerySet
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils.safestring import mark_safe

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

        # Nếu chưa CRITICAL nhưng early_warning HIGH → tính vào WARNING
        ew_level = getattr(row, 'early_warning_level', None) or ''
        if badge != "CRITICAL" and ew_level == "HIGH":
            badge = "WARNING"

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
        Alert.objects.filter(
            patient_id=patient_id,
            acknowledged=False,
            risk_level__iexact="CRITICAL"
        )
        .order_by("-created_at")
        .first()
    )

    # --- Lấy early warning từ ML service history ---
    early_warning = None
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{ml_url}/vitals/{patient_id}/history")
            if resp.status_code == 200:
                payload = resp.json()
                ew_data = payload.get("early_warning")
                if ew_data:
                    early_warning = ew_data
    except Exception:
        pass

    # Fallback: lấy từ DB predictions mới nhất
    if not early_warning and latest_pred:
        if hasattr(latest_pred, 'early_warning_probability'):
            early_warning = {
                "early_warning_probability": float(latest_pred.early_warning_probability or 0),
                "early_warning_level": str(latest_pred.early_warning_level or "LOW"),
                "trend_score": float(latest_pred.trend_score or 0),
                "rate_of_change_score": float(latest_pred.rate_of_change_score or 0),
                "threshold_score": float(latest_pred.threshold_score or 0),
                "contributing_factors": [],
                "time_window_minutes": 30,
            }

    # Tính các giá trị phần trăm và màu sắc cho template
    ew_prob = 0.0
    ew_level = "LOW"
    if early_warning:
        ew_prob = float(early_warning.get("early_warning_probability", 0))
        ew_level = str(early_warning.get("early_warning_level", "LOW"))

    if ew_level == "HIGH":
        ew_badge_color = "danger"
        ew_border_color = "danger"
        ew_text_color = "danger"
    elif ew_level == "MEDIUM":
        ew_badge_color = "warning"
        ew_border_color = "warning"
        ew_text_color = "warning"
    else:
        ew_badge_color = "success"
        ew_border_color = "success"
        ew_text_color = "success"

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
            # Early warning context
            "early_warning": early_warning,
            "ew_probability_pct": int(ew_prob * 100),
            "ew_trend_pct": int(float(early_warning.get("trend_score", 0)) * 100) if early_warning else 0,
            "ew_roc_pct": int(float(early_warning.get("rate_of_change_score", 0)) * 100) if early_warning else 0,
            "ew_thresh_pct": int(float(early_warning.get("threshold_score", 0)) * 100) if early_warning else 0,
                        "ew_badge_color": ew_badge_color,
            "ew_border_color": ew_border_color,
            "ew_text_color": ew_text_color,
            "ew_level": ew_level,
        },
    )


def alerts_page(request: HttpRequest) -> HttpResponse:
    status = request.GET.get("status", "all")
    if status not in {"all", "pending", "confirmed"}:
        status = "all"

    alert_url = getattr(settings, "ALERT_SERVICE_URL", "http://localhost:8002").rstrip("/")

    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"[alerts_page] status={status} alert_url={alert_url}")

    alerts: List[Dict[str, Any]] = []
    try:
        params: Dict[str, Any] = {"limit": 50}
        if status != "all":
            params["status"] = status
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{alert_url}/alerts", params=params)
            if resp.status_code == 200:
                data = resp.json()
                # Normalize risk_level về uppercase để hiển thị đồng nhất
                for a in data:
                    if "risk_level" in a:
                        a["risk_level"] = str(a["risk_level"]).upper()
                alerts = data
    except Exception:
        alerts = []

    # Fallback: nếu alert_service không có data, đọc từ Django DB
    if not alerts:
        db_alerts = Alert.objects.all().order_by("-created_at")[:50]
        if status == "pending":
            db_alerts = Alert.objects.filter(acknowledged=False).order_by("-created_at")[:50]
        elif status == "confirmed":
            db_alerts = Alert.objects.filter(acknowledged=True).order_by("-created_at")[:50]
        alerts = [
            {
                "alert_id": str(a.alert_id),
                "patient_id": a.patient_id,
                "risk_score": float(a.risk_score) if a.risk_score else 0.0,
                "risk_level": str(a.risk_level or "").upper(),
                "alert_type": getattr(a, "alert_type", "sepsis"),
                "created_at": a.created_at.isoformat() if a.created_at else "",
                "acknowledged": bool(a.acknowledged),
                "ack_by": a.ack_by,
                "ack_at": a.ack_at.isoformat() if a.ack_at else None,
            }
            for a in db_alerts
        ]

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
    now = datetime.now(timezone.utc)

    # 1. Gọi alert_service PATCH
    alert_url = getattr(settings, "ALERT_SERVICE_URL", "http://localhost:8002").rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as client:
            client.patch(
                f"{alert_url}/alerts/{alert_id}/acknowledge",
                json={"ack_by": ack_by},
            )
    except Exception:
        pass

    # 2. Cập nhật Django DB đồng bộ
    try:
        from django.utils import timezone as dj_tz
        Alert.objects.filter(alert_id=alert_id).update(
            acknowledged=True,
            ack_by=ack_by,
            ack_at=dj_tz.now(),
        )
    except Exception:
        pass

    # 3. Redirect về đúng chỗ
    referer = request.META.get("HTTP_REFERER", "")
    if "/patients/" in referer:
        return redirect(referer)
    return redirect("/alerts/?status=pending")


def patient_latest_api(request: HttpRequest, patient_id: str) -> JsonResponse:
    p = Prediction.objects.filter(patient_id=patient_id).order_by('-timestamp').first()
    if not p:
        return JsonResponse({}, status=404)

    ml_url = getattr(settings, 'ML_SERVICE_URL', 'http://localhost:8001').rstrip('/')
    shap_features = []
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f'{ml_url}/vitals/{patient_id}/history')
            if resp.status_code == 200:
                shap_features = resp.json().get('top_features', [])
    except Exception:
        pass

    return JsonResponse({
        'risk_score': p.risk_score,
        'risk_level': p.risk_level,
        'sofa_score': p.sofa_score,
        'news2_score': p.news2_score,
        'timestamp': p.timestamp.isoformat(),
        'heart_rate': p.heart_rate,
        'systolic_bp': p.systolic_bp,
        'diastolic_bp': p.diastolic_bp,
        'temperature': p.temperature,
        'spo2': p.spo2,
        'respiratory_rate': p.respiratory_rate,
        'early_warning': {
            'early_warning_probability': p.early_warning_probability or 0,
            'early_warning_level': p.early_warning_level or 'LOW',
            'trend_score': p.trend_score or 0,
            'threshold_score': p.threshold_score or 0,
        },
        'shap_features': shap_features,
    })
