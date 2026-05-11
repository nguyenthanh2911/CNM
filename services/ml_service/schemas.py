from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class VitalRequest(BaseModel):
    patient_id: str
    timestamp: datetime

    heart_rate: float = Field(ge=20, le=250)
    systolic_bp: float = Field(ge=40, le=300)
    diastolic_bp: float = Field(ge=20, le=200)
    temperature: float = Field(ge=30, le=45)
    spo2: float = Field(ge=50, le=100)
    respiratory_rate: float = Field(ge=4, le=60)

    lactate: Optional[float] = None
    wbc: Optional[float] = None
    creatinine: Optional[float] = None
    bilirubin: Optional[float] = None
    platelet: Optional[float] = None


class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: datetime
    risk_score: float
    risk_level: str  # LOW / WARNING / CRITICAL
    alert_triggered: bool
    top_features: List[FeatureExplanation]
    sofa_score: int
    news2_score: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    # Pydantic v2 reserves the `model_` prefix; allow it for response fields.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    model_auroc: float
    uptime_seconds: float
