from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_client.exposition import make_asgi_app
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .predictor import SepsisPredictor
from .schemas import HealthResponse, PredictionResponse, VitalRequest


predictions_total = Counter("predictions_total", "Total number of predictions")
predictions_by_risk_total = Counter(
    "predictions_by_risk_total",
    "Total number of predictions by risk level",
    ["risk_level"],
)
inference_seconds = Histogram("inference_seconds", "Inference request latency in seconds")

Base = declarative_base()


class PredictionORM(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    alert_triggered = Column(Boolean, nullable=False)

    sofa_score = Column(Integer, nullable=False)
    news2_score = Column(Integer, nullable=False)
    inference_time_ms = Column(Float, nullable=False)

    # THÊM MỚI — raw vitals
    heart_rate = Column(Float, nullable=True)
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)
    respiratory_rate = Column(Float, nullable=True)
    lactate = Column(Float, nullable=True)
    wbc = Column(Float, nullable=True)
    creatinine = Column(Float, nullable=True)
    bilirubin = Column(Float, nullable=True)
    platelet = Column(Float, nullable=True)

    # THÊM MỚI — early warning scores
    early_warning_probability = Column(Float, nullable=True)
    early_warning_level       = Column(String(16), nullable=True)
    trend_score               = Column(Float, nullable=True)
    rate_of_change_score      = Column(Float, nullable=True)
    threshold_score           = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False)


def _build_db_url() -> str:
    db_url = os.getenv("DB_URL")
    if db_url:
        return db_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "sepsis_user")
    password = os.getenv("POSTGRES_PASSWORD", "sepsis_pass")
    db = os.getenv("POSTGRES_DB", "sepsis_db")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


engine = create_engine(_build_db_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

app = FastAPI(title="CNM Sepsis ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/metrics", make_asgi_app())

_start_time: Optional[float] = None


def _ensure_columns(table: str, columns: Dict[str, str]) -> None:
    """Best-effort schema upgrade for existing tables.

    `create_all()` will not add columns to an existing table. If users reuse a
    persisted Postgres volume created with an older schema, inserts can fail
    (while Prometheus metrics still increase), making the Django dashboard look
    empty. This helper makes upgrades idempotent using `ADD COLUMN IF NOT EXISTS`.
    """
    try:
        inspector = inspect(engine)
        if not inspector.has_table(table):
            return
        existing = {c.get("name") for c in inspector.get_columns(table)}
    except Exception:
        return

    stmts = []
    for name, sql_type in columns.items():
        if name in existing:
            continue
        stmts.append(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {sql_type}"))

    if not stmts:
        return

    with engine.begin() as conn:
        for stmt in stmts:
            conn.execute(stmt)


@app.middleware("http")
async def prometheus_latency_middleware(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.perf_counter() - start
        if request.url.path == "/vitals":
            inference_seconds.observe(elapsed)


@app.on_event("startup")
def startup_event() -> None:
    global _start_time
    _start_time = time.time()

    # init predictor singleton
    SepsisPredictor.get_instance()

    # create tables
    Base.metadata.create_all(bind=engine)

    # ensure schema is compatible with current PredictionORM
    _ensure_columns(
        "predictions",
        {
            # Raw vitals columns (nullable)
            "heart_rate": "DOUBLE PRECISION",
            "systolic_bp": "DOUBLE PRECISION",
            "diastolic_bp": "DOUBLE PRECISION",
            "temperature": "DOUBLE PRECISION",
            "spo2": "DOUBLE PRECISION",
            "respiratory_rate": "DOUBLE PRECISION",
            "lactate": "DOUBLE PRECISION",
            "wbc": "DOUBLE PRECISION",
            "creatinine": "DOUBLE PRECISION",
            "bilirubin": "DOUBLE PRECISION",
            "platelet": "DOUBLE PRECISION",
            # Early warning columns (nullable)
            "early_warning_probability": "DOUBLE PRECISION",
            "early_warning_level": "VARCHAR(16)",
            "trend_score": "DOUBLE PRECISION",
            "rate_of_change_score": "DOUBLE PRECISION",
            "threshold_score": "DOUBLE PRECISION",
        },
    )


@app.post("/vitals", response_model=PredictionResponse)
async def post_vitals(payload: VitalRequest) -> PredictionResponse:
    predictor = SepsisPredictor.get_instance()

    try:
        result = predictor.predict(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    predictions_total.inc()
    predictions_by_risk_total.labels(risk_level=str(result.risk_level)).inc()

    # persist to DB
    now = datetime.now(timezone.utc)
    ts = result.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    db = SessionLocal()
    try:
        row = PredictionORM(
            patient_id=result.patient_id,
            timestamp=ts,
            risk_score=float(result.risk_score),
            risk_level=str(result.risk_level),
            alert_triggered=bool(result.alert_triggered),
            sofa_score=int(result.sofa_score),
            news2_score=int(result.news2_score),
            inference_time_ms=float(result.inference_time_ms),
            # THÊM MỚI — lưu raw vitals từ payload gốc
            heart_rate=payload.heart_rate,
            systolic_bp=payload.systolic_bp,
            diastolic_bp=payload.diastolic_bp,
            temperature=payload.temperature,
            spo2=payload.spo2,
            respiratory_rate=payload.respiratory_rate,
            lactate=payload.lactate,
            wbc=payload.wbc,
            creatinine=payload.creatinine,
            bilirubin=payload.bilirubin,
            platelet=payload.platelet,
            # THÊM MỚI — early warning scores
            early_warning_probability=result.early_warning.early_warning_probability,
            early_warning_level=result.early_warning.early_warning_level,
            trend_score=result.early_warning.trend_score,
            rate_of_change_score=result.early_warning.rate_of_change_score,
            threshold_score=result.early_warning.threshold_score,
            created_at=now,
        )
        db.add(row)
        db.commit()
    finally:
        db.close()

        # alert service call
    alert_url = os.getenv("ALERT_SERVICE_URL", "http://alert_service:8002/alerts")

    # Gửi alert cho sepsis CRITICAL
    if result.alert_triggered:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.post(
                    alert_url,
                    json={
                        "patient_id": result.patient_id,
                        "risk_score": float(result.risk_score),
                        "risk_level": str(result.risk_level),
                        "alert_type": "sepsis",
                        "top_features": [f.model_dump() for f in result.top_features],
                        "sofa_score": int(result.sofa_score),
                        "news2_score": int(result.news2_score),
                    },
                )
            except Exception:
                # Do not fail inference if alert service is down
                pass

    # Gửi alert sớm khi early_warning HIGH nhưng chưa CRITICAL
    if result.early_warning.early_warning_level == "HIGH" and not result.alert_triggered:
        ew_msg = (
            f"Nguy cơ sepsis {result.early_warning.early_warning_probability*100:.0f}% "
            f"trong 30 phút tới"
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.post(
                    alert_url,
                    json={
                        "patient_id": result.patient_id,
                        "risk_score": float(result.early_warning.early_warning_probability),
                        "risk_level": "EARLY_WARNING",
                        "alert_type": "early_warning",
                        "top_features": [],
                        "sofa_score": 0,
                        "news2_score": 0,
                    },
                )
            except Exception:
                pass

    return result


@app.get("/vitals/{patient_id}/history")
async def get_patient_history(patient_id: str) -> Dict[str, Any]:
    predictor = SepsisPredictor.get_instance()
    try:
        return predictor.get_history(patient_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    predictor = SepsisPredictor.get_instance()

    uptime = 0.0
    if _start_time is not None:
        uptime = max(0.0, time.time() - _start_time)

    return HealthResponse(
        status="ok",
        model_version=str(predictor.model_version),
        model_auroc=float(predictor.model_auroc),
        uptime_seconds=float(uptime),
    )
