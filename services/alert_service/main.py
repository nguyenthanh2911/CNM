from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from prometheus_client import Gauge, generate_latest
from starlette.responses import Response
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, func, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .schemas import AlertCreate, AlertResponse, AlertStats
from .websocket_manager import ConnectionManager


Base = declarative_base()


class AlertORM(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), unique=True, nullable=False, index=True)

    patient_id = Column(String(64), nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False, index=True)

    top_features = Column(JSONB, nullable=False)
    sofa_score = Column(Integer, nullable=False)
    news2_score = Column(Integer, nullable=False)
    alert_type = Column(String(32), nullable=False, default="sepsis")  # "sepsis" | "early_warning"

    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    acknowledged = Column(Boolean, nullable=False, default=False, index=True)
    ack_by = Column(String(128), nullable=True)
    ack_at = Column(DateTime(timezone=True), nullable=True)


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

app = FastAPI(title="CNM Alert Service")
manager = ConnectionManager()

active_alerts_gauge = Gauge("active_alerts", "Number of active (unacknowledged) alerts")


def _ensure_columns(table: str, columns: Dict[str, str]) -> None:
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


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    # Update active alerts gauge on scrape (cheap query, default scrape_interval is 15s)
    db: Session = SessionLocal()
    try:
        active = int(db.query(func.count(AlertORM.id)).filter(AlertORM.acknowledged.is_(False)).scalar() or 0)
        active_alerts_gauge.set(active)
    finally:
        db.close()

    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)

    _ensure_columns(
        "alerts",
        {
            "acknowledged": "BOOLEAN DEFAULT FALSE",
            "ack_by": "VARCHAR(128)",
            "ack_at": "TIMESTAMPTZ",
            "alert_type": "VARCHAR(32) DEFAULT 'sepsis'",
        },
        )


def _to_response(row: AlertORM) -> AlertResponse:
    return AlertResponse(
        alert_id=row.alert_id,
        patient_id=row.patient_id,
        risk_score=float(row.risk_score),
        risk_level=str(row.risk_level),
        alert_type=str(row.alert_type),
        created_at=row.created_at,
        acknowledged=bool(row.acknowledged),
        ack_by=row.ack_by,
        ack_at=row.ack_at,
    )


@app.post("/alerts", response_model=AlertResponse)
async def create_alert(payload: AlertCreate) -> AlertResponse:
    now = datetime.now(timezone.utc)

    db: Session = SessionLocal()
    try:
        alert = AlertORM(
            alert_id=str(uuid.uuid4()),
            patient_id=payload.patient_id,
            risk_score=float(payload.risk_score),
            risk_level=str(payload.risk_level),
            top_features=payload.top_features,
                        sofa_score=int(payload.sofa_score),
            news2_score=int(payload.news2_score),
            alert_type=payload.alert_type,
            created_at=now,
            acknowledged=False,
            ack_by=None,
            ack_at=None,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
    finally:
        db.close()

    alert_data: Dict[str, Any] = {
        "alert_id": alert.alert_id,
        "patient_id": alert.patient_id,
        "risk_score": alert.risk_score,
        "risk_level": alert.risk_level,
        "created_at": alert.created_at.isoformat(),
        "acknowledged": alert.acknowledged,
        "ack_by": alert.ack_by,
        "ack_at": alert.ack_at.isoformat() if alert.ack_at else None,
        "top_features": payload.top_features,
        "sofa_score": alert.sofa_score,
        "news2_score": alert.news2_score,
    }

    await manager.send_alert(payload.patient_id, alert_data)
    return _to_response(alert)


@app.get("/alerts", response_model=List[AlertResponse])
async def list_alerts(
    patient_id: Optional[str] = Query(default=None),
    status: str = Query(default="all", pattern="^(pending|confirmed|all)$"),
    limit: int = Query(default=50, ge=1, le=500),
) -> List[AlertResponse]:
    db: Session = SessionLocal()
    try:
        q = db.query(AlertORM)
        if patient_id:
            q = q.filter(AlertORM.patient_id == patient_id)

        if status == "pending":
            q = q.filter(AlertORM.acknowledged.is_(False))
        elif status == "confirmed":
            q = q.filter(AlertORM.acknowledged.is_(True))

        rows = q.order_by(AlertORM.created_at.desc()).limit(int(limit)).all()
        return [_to_response(r) for r in rows]
    finally:
        db.close()


@app.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str) -> AlertResponse:
    db: Session = SessionLocal()
    try:
        row = db.query(AlertORM).filter(AlertORM.alert_id == alert_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        return _to_response(row)
    finally:
        db.close()


@app.patch("/alerts/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(alert_id: str, body: Dict[str, str] = Body(...)) -> AlertResponse:
    ack_by = body.get("ack_by")
    if not ack_by:
        raise HTTPException(status_code=422, detail="ack_by is required")

    now = datetime.now(timezone.utc)

    db: Session = SessionLocal()
    try:
        row = db.query(AlertORM).filter(AlertORM.alert_id == alert_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        row.acknowledged = True
        row.ack_by = str(ack_by)
        row.ack_at = now
        db.commit()
        db.refresh(row)
        return _to_response(row)
    finally:
        db.close()


@app.get("/alerts/stats", response_model=AlertStats)
async def alert_stats() -> AlertStats:
    today = datetime.now(timezone.utc).date()
    start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)

    db: Session = SessionLocal()
    try:
        base_q = db.query(AlertORM).filter(AlertORM.created_at >= start)

        total_today = int(base_q.count())
        critical_today = int(base_q.filter(AlertORM.risk_level == "CRITICAL").count())
        warning_today = int(base_q.filter(AlertORM.risk_level == "WARNING").count())

        acked = base_q.filter(AlertORM.acknowledged.is_(True), AlertORM.ack_at.isnot(None)).all()
        if not acked:
            avg_minutes = 0.0
        else:
            deltas = [
                (r.ack_at - r.created_at).total_seconds() / 60.0
                for r in acked
                if r.ack_at is not None and r.created_at is not None
            ]
            avg_minutes = float(sum(deltas) / max(len(deltas), 1)) if deltas else 0.0

        return AlertStats(
            total_today=total_today,
            critical_today=critical_today,
            warning_today=warning_today,
            avg_response_time_minutes=avg_minutes,
        )
    finally:
        db.close()


@app.websocket("/ws/all")
async def ws_all(websocket: WebSocket):
    await manager.connect(websocket, patient_id="all")
    try:
        while True:
            msg = await websocket.receive_text()
            if msg.lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, patient_id="all")


@app.websocket("/ws/{patient_id}")
async def ws_patient(websocket: WebSocket, patient_id: str):
    await manager.connect(websocket, patient_id=patient_id)
    try:
        while True:
            msg = await websocket.receive_text()
            if msg.lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, patient_id=patient_id)
        if msg.lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, patient_id=patient_id)
