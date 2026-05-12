from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from feature_engineering.clinical_scores import calculate_news2, calculate_sofa
from ml.explain import SepsisExplainer
from ml.mlflow_utils import load_production_model_with_metadata
from ml.models.xgboost_model import SepsisXGBModel

from .schemas import FeatureExplanation, PredictionResponse, VitalRequest


FEATURE_COLS = [
    'heart_rate', 'systolic_bp', 'diastolic_bp',
    'temperature', 'spo2', 'respiratory_rate',
    'lactate', 'wbc', 'creatinine', 'bilirubin', 'platelet'
]


@dataclass
class _LoadedArtifacts:
    model: Any | None
    preprocess_pipeline: Any | None  # sklearn Pipeline
    explainer: SepsisExplainer | None


class SepsisPredictor:
    _instance: "SepsisPredictor | None" = None

    def __init__(self) -> None:
        # Keep default consistent with demo scripts and README.
        self.model_name = os.getenv("MODEL_NAME", "sepsis_xgboost")
        self.model_version = os.getenv("MODEL_VERSION", "unknown")
        self.model_auroc = float(os.getenv("MODEL_AUROC", "0.0"))

        mlflow_uri = os.getenv("MLFLOW_URI") or os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(mlflow_uri)

        self._artifacts = self._load_artifacts()

    @classmethod
    def get_instance(cls) -> "SepsisPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self) -> _LoadedArtifacts:
        model_obj: Any | None = None
        explainer: SepsisExplainer | None = None
        preprocess_pipeline: Any | None = None

        # Load model from MLflow registry
        try:
            model_obj, meta = load_production_model_with_metadata(self.model_name)

            # Populate health metadata from MLflow (best-effort)
            mv = meta.get("model_version")
            if mv:
                self.model_version = str(mv)
            auroc = meta.get("model_auroc")
            if auroc is not None:
                self.model_auroc = float(auroc)

            # adapt to SepsisXGBModel for SHAP explainer
            wrapped = SepsisXGBModel()
            wrapped.model = model_obj
            explainer = SepsisExplainer(wrapped)
        except Exception:
            model_obj = None
            explainer = None

        # Load sklearn Pipeline (imputer + scaler)
        try:
            preprocess_pipeline = joblib.load(
                os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib")
            )
            print("Loaded preprocessor pipeline OK")
        except FileNotFoundError:
            print("WARNING: preprocessor.joblib not found, will use raw features")
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            preprocess_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
            ])

        return _LoadedArtifacts(
            model=model_obj,
            preprocess_pipeline=preprocess_pipeline,
            explainer=explainer,
        )

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        model = self._artifacts.model
        if model is None:
            return np.zeros((len(X), 2), dtype=float)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return np.asarray(proba)

        # MLflow xgboost flavor often returns xgboost.Booster
        try:
            import xgboost as xgb

            dm = xgb.DMatrix(X)
            p1 = np.asarray(model.predict(dm)).reshape(-1)
            p1 = np.clip(p1, 0.0, 1.0)
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        except Exception as e:
            raise RuntimeError(f"Unsupported model type for predict_proba: {type(model)}") from e

    def predict(self, vital_request: VitalRequest) -> PredictionResponse:
        req = vital_request.model_dump()
        patient_id = req["patient_id"]

        # Ensure timestamp is python datetime
        ts = req["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        # Tính clinical scores từ raw vitals
        sofa_score  = calculate_sofa(req)
        news2_score = calculate_news2(req)

        # Tạo DataFrame 11 raw features
        X = pd.DataFrame([{col: req.get(col) for col in FEATURE_COLS}])

        # Ép kiểu numeric
        for col in FEATURE_COLS:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Transform qua sklearn Pipeline
        pipeline = self._artifacts.preprocess_pipeline
        if pipeline is not None:
            try:
                X_processed = pipeline.transform(X)
            except Exception:
                # Pipeline chưa fit → fit tạm trên X
                X_processed = pipeline.fit_transform(X)
        else:
            X_processed = X.values

        # Inference
        start = time.perf_counter()
        proba = self._predict_proba(
            pd.DataFrame(X_processed, columns=FEATURE_COLS)
        )
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        risk_score = float(proba[0, 1]) if proba.size else 0.0

        # Risk level
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.7:
            risk_level = "WARNING"
        else:
            risk_level = "CRITICAL"

        alert_triggered = risk_score >= 0.7

        # SHAP explain
        top_features: List[FeatureExplanation] = []
        if self._artifacts.model is not None and self._artifacts.explainer is not None:
            try:
                X_df = pd.DataFrame(X_processed, columns=FEATURE_COLS)
                exp = self._artifacts.explainer.explain(
                    X_df, feature_names=FEATURE_COLS
                )
                top_features = [FeatureExplanation(**d) for d in exp]
            except Exception:
                top_features = []

        if not hasattr(self, "_shap_cache"):
            self._shap_cache = {}
        self._shap_cache[patient_id] = [f.model_dump() for f in top_features]

        # THÊM MỚI — cache raw vitals cho get_history()
        if not hasattr(self, "_vitals_cache"):
            self._vitals_cache = {}
        self._vitals_cache[patient_id] = {
            "heart_rate": req.get("heart_rate"),
            "systolic_bp": req.get("systolic_bp"),
            "diastolic_bp": req.get("diastolic_bp"),
            "temperature": req.get("temperature"),
            "spo2": req.get("spo2"),
            "respiratory_rate": req.get("respiratory_rate"),
            "lactate": req.get("lactate"),
            "wbc": req.get("wbc"),
            "creatinine": req.get("creatinine"),
            "bilirubin": req.get("bilirubin"),
            "platelet": req.get("platelet"),
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        }

        return PredictionResponse(
            patient_id=patient_id,
            timestamp=ts,
            risk_score=risk_score,
            risk_level=risk_level,
            alert_triggered=alert_triggered,
            top_features=top_features,
            sofa_score=int(sofa_score),
            news2_score=int(news2_score),
            inference_time_ms=float(inference_time_ms),
        )

    def get_history(self, patient_id: str) -> Dict[str, Any]:
        return {
            "latest_vitals": getattr(self, "_vitals_cache", {}).get(patient_id, {}),
            "top_features": getattr(self, "_shap_cache", {}).get(patient_id, [])
        }
