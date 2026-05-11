from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import joblib
from xgboost import XGBClassifier


DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "auc",
    "random_state": 42,
}


@dataclass
class SepsisXGBModel:
    params: Optional[Dict[str, Any]] = None
    model: Optional[XGBClassifier] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = dict(DEFAULT_PARAMS)
        else:
            merged = dict(DEFAULT_PARAMS)
            merged.update(self.params)
            self.params = merged

    def fit(self, X_train, y_train, X_val, y_val) -> "SepsisXGBModel":
        y_train_arr = np.asarray(y_train)
        neg = int((y_train_arr == 0).sum())
        pos = int((y_train_arr == 1).sum())
        scale_pos_weight = float(neg / max(pos, 1))

        params = dict(self.params or {})
        params["scale_pos_weight"] = scale_pos_weight

        self.model = XGBClassifier(**params)

        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }

        # XGBoost compatibility: some versions accept early_stopping_rounds directly,
        # newer versions may require callbacks.
        try:
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=20,
                **fit_kwargs,
            )
        except TypeError:
            try:
                from xgboost.callback import EarlyStopping

                self.model.fit(
                    X_train,
                    y_train,
                    callbacks=[EarlyStopping(rounds=20, save_best=True)],
                    **fit_kwargs,
                )
            except Exception:
                # Last resort: train without early stopping
                self.model.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )
        self.params = params
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained/loaded.")
        proba = self.model.predict_proba(X)
        return np.asarray(proba)

    def predict(self, X, threshold: float = 0.4) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= float(threshold)).astype(int)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained/loaded. Nothing to save.")
        joblib.dump({"model": self.model, "params": self.params}, path)

    @classmethod
    def load(cls, path: str) -> "SepsisXGBModel":
        payload = joblib.load(path)
        obj = cls(params=(payload.get("params") if isinstance(payload, dict) else None))
        if isinstance(payload, dict) and "model" in payload:
            obj.model = payload["model"]
        else:
            obj.model = payload
        return obj


