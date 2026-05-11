"""Unit tests for FastAPI ML Service endpoints.

Strategy: import the FastAPI `app` object **once** per pytest session using a
session-scoped fixture.  Re-importing / reloading the module each test would
re-execute the Prometheus Counter/Histogram definitions at module level, which
raises "Duplicated timeseries in CollectorRegistry".
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_VITALS_PAYLOAD = {
    "patient_id": "P001",
    "timestamp": "2024-01-15T08:30:00",
    "heart_rate": 85,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "temperature": 37.0,
    "spo2": 98,
    "respiratory_rate": 16,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_predictor(risk_score: float = 0.25):
    """Return a MagicMock that mimics SepsisPredictor for a given risk score."""
    from services.ml_service.schemas import FeatureExplanation, PredictionResponse

    if risk_score < 0.3:
        level = "LOW"
    elif risk_score < 0.7:
        level = "WARNING"
    else:
        level = "CRITICAL"

    mock_response = PredictionResponse(
        patient_id="P001",
        timestamp=datetime(2024, 1, 15, 8, 30, 0),
        risk_score=risk_score,
        risk_level=level,
        alert_triggered=(risk_score >= 0.7),
        top_features=[FeatureExplanation(feature="heart_rate_mean_15m", shap_value=0.12)],
        sofa_score=2,
        news2_score=3,
        inference_time_ms=45.0,
    )

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = mock_response
    mock_predictor.model_version = "test-1.0"
    mock_predictor.model_auroc = 0.91
    return mock_predictor


# ---------------------------------------------------------------------------
# Session-scoped app import (avoids re-registering Prometheus metrics)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _app():
    """Import the FastAPI app exactly once per test session."""
    os.environ.setdefault("DB_URL", "sqlite:///:memory:")

    with (
        patch("services.ml_service.predictor.SepsisPredictor._load_artifacts"),
        patch("services.ml_service.main.Base.metadata.create_all"),
    ):
        from services.ml_service.main import app  # noqa: PLC0415
        return app


@pytest.fixture
def client(_app):
    """TestClient backed by a LOW-risk mock predictor."""
    from fastapi.testclient import TestClient

    mock_pred = _make_mock_predictor(risk_score=0.25)
    with (
        patch("services.ml_service.main.SepsisPredictor.get_instance", return_value=mock_pred),
        patch("services.ml_service.main.SessionLocal"),
    ):
        yield TestClient(_app)


@pytest.fixture
def client_critical(_app):
    """TestClient backed by a CRITICAL-risk mock predictor."""
    from fastapi.testclient import TestClient

    mock_pred = _make_mock_predictor(risk_score=0.85)
    with (
        patch("services.ml_service.main.SepsisPredictor.get_instance", return_value=mock_pred),
        patch("services.ml_service.main.SessionLocal"),
    ):
        yield TestClient(_app)


# ---------------------------------------------------------------------------
# Tests — /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_has_status_field(self, client):
        assert "status" in client.get("/health").json()

    def test_health_has_model_version(self, client):
        assert "model_version" in client.get("/health").json()

    def test_health_has_uptime(self, client):
        assert "uptime_seconds" in client.get("/health").json()


# ---------------------------------------------------------------------------
# Tests — POST /vitals
# ---------------------------------------------------------------------------

class TestPostVitals:
    def test_valid_payload_returns_200(self, client):
        assert client.post("/vitals", json=VALID_VITALS_PAYLOAD).status_code == 200

    def test_response_has_risk_score(self, client):
        assert "risk_score" in client.post("/vitals", json=VALID_VITALS_PAYLOAD).json()

    def test_risk_score_in_range(self, client):
        data = client.post("/vitals", json=VALID_VITALS_PAYLOAD).json()
        assert 0 <= data["risk_score"] <= 1

    def test_risk_level_valid_values(self, client):
        data = client.post("/vitals", json=VALID_VITALS_PAYLOAD).json()
        assert data["risk_level"] in {"LOW", "WARNING", "CRITICAL"}

    def test_response_has_sofa_news2(self, client):
        data = client.post("/vitals", json=VALID_VITALS_PAYLOAD).json()
        assert "sofa_score" in data and "news2_score" in data

    def test_critical_risk_triggers_alert(self, client_critical):
        data = client_critical.post("/vitals", json=VALID_VITALS_PAYLOAD).json()
        assert data["alert_triggered"] is True
        assert data["risk_level"] == "CRITICAL"

    def test_low_risk_no_alert(self, client):
        assert client.post("/vitals", json=VALID_VITALS_PAYLOAD).json()["alert_triggered"] is False

    def test_missing_required_field_returns_422(self, client):
        """No timestamp → Pydantic validation error."""
        response = client.post("/vitals", json={"patient_id": "P001", "heart_rate": 85})
        assert response.status_code == 422

    def test_heart_rate_out_of_range_returns_422(self, client):
        """heart_rate=999 exceeds Field(le=250)."""
        response = client.post("/vitals", json={**VALID_VITALS_PAYLOAD, "heart_rate": 999})
        assert response.status_code == 422

    def test_spo2_below_minimum_returns_422(self, client):
        """spo2=10 is below Field(ge=50)."""
        response = client.post("/vitals", json={**VALID_VITALS_PAYLOAD, "spo2": 10})
        assert response.status_code == 422

    def test_temperature_out_of_range_returns_422(self, client):
        """temperature=60 exceeds Field(le=45)."""
        response = client.post("/vitals", json={**VALID_VITALS_PAYLOAD, "temperature": 60})
        assert response.status_code == 422
