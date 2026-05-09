from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from data_pipeline.preprocessor import ICUPreprocessor
from feature_engineering.clinical_scores import calculate_news2, calculate_sofa
from feature_engineering.feature_builder import FeatureBuilder
from ml.explain import SepsisExplainer
from ml.mlflow_utils import load_production_model
from ml.models.xgboost_model import SepsisXGBModel

from .schemas import FeatureExplanation, PredictionResponse, VitalRequest


@dataclass
class _LoadedArtifacts:
    model: Any | None
    preprocessor: ICUPreprocessor | None
    explainer: SepsisExplainer | None
    feature_builder: FeatureBuilder


class SepsisPredictor:
    _instance: "SepsisPredictor | None" = None

    def __init__(self) -> None:
        self.model_name = os.getenv("MODEL_NAME", "SepsisXGB")
        self.model_version = os.getenv("MODEL_VERSION", "unknown")
        self.model_auroc = float(os.getenv("MODEL_AUROC", "0.0"))

        mlflow_uri = os.getenv("MLFLOW_URI") or os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(mlflow_uri)

        self.preprocessor_path = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib")

        self._buffer: Dict[str, Deque[Dict[str, Any]]] = {}
        self._maxlen = 48  # keep 4h history (48 x 5min)

        self._artifacts = self._load_artifacts()

    @classmethod
    def get_instance(cls) -> "SepsisPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self) -> _LoadedArtifacts:
        feature_builder = FeatureBuilder()

        model_obj: Any | None = None
        explainer: SepsisExplainer | None = None
        pre: ICUPreprocessor | None = None

        # Load model from MLflow registry
        try:
            model_obj = load_production_model(self.model_name)
            # adapt to SepsisXGBModel for SHAP explainer
            wrapped = SepsisXGBModel()
            wrapped.model = model_obj
            explainer = SepsisExplainer(wrapped)
        except Exception:
            model_obj = None
            explainer = None

        # Load preprocessor (scaler)
        try:
            pre = ICUPreprocessor().load(self.preprocessor_path)
        except Exception:
            pre = None

        return _LoadedArtifacts(
            model=model_obj,
            preprocessor=pre,
            explainer=explainer,
            feature_builder=feature_builder,
        )

    def _append_to_buffer(self, patient_id: str, record: Dict[str, Any]) -> None:
        if patient_id not in self._buffer:
            self._buffer[patient_id] = deque(maxlen=self._maxlen)
        self._buffer[patient_id].append(record)

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
        req["timestamp"] = ts

        self._append_to_buffer(patient_id, req)

        history = list(self._buffer[patient_id])
        df_hist = pd.DataFrame(history)

        # Compute clinical scores on current raw row (before feature drop)
        sofa_score = calculate_sofa(req)
        news2_score = calculate_news2(req)

        # 3) feature builder
        df_feat = self._artifacts.feature_builder.build(df_hist)

        # 4) preprocessor transform
        pre = self._artifacts.preprocessor
        if pre is not None:
            df_feat = pre.transform(df_feat)

        # 5) last row
        id_cols = [self._artifacts.feature_builder.patient_col, self._artifacts.feature_builder.time_col]
        label_col = self._artifacts.feature_builder.label_col
        feature_cols = [c for c in df_feat.columns if c not in id_cols + [label_col]]

        X_last = df_feat.iloc[[-1]][feature_cols]

        # 6) inference time
        start = time.perf_counter()
        proba = self._predict_proba(X_last)
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        # 7) risk score
        risk_score = float(proba[0, 1]) if proba.size else 0.0

        # 8) risk level
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.7:
            risk_level = "WARNING"
        else:
            risk_level = "CRITICAL"

        # 9) alert
        alert_triggered = risk_score >= 0.7

        # 10) SHAP explain
        top_features: List[FeatureExplanation] = []
        if self._artifacts.model is not None and self._artifacts.explainer is not None:
            try:
                exp = self._artifacts.explainer.explain(X_last, feature_names=feature_cols)
                top_features = [FeatureExplanation(**d) for d in exp]
            except Exception:
                top_features = []

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
