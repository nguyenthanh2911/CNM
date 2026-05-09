from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AlertCreate(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    top_features: List[Dict[str, Any]]
    sofa_score: int
    news2_score: int


class AlertResponse(BaseModel):
    alert_id: str
    patient_id: str
    risk_score: float
    risk_level: str
    created_at: datetime
    acknowledged: bool
    ack_by: Optional[str] = None
    ack_at: Optional[datetime] = None


class AlertStats(BaseModel):
    total_today: int
    critical_today: int
    warning_today: int
    avg_response_time_minutes: float
