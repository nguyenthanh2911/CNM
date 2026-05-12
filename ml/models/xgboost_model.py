from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import joblib
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators":     150,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.65,
    "colsample_bytree": 0.65,
    "min_child_weight": 20,
    "gamma":            2.0,
    "reg_alpha":        1.0,
    "reg_lambda":       3.0,
    "max_delta_step":   1,
    "eval_metric":      ["auc", "logloss"],
    "random_state":     42,
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
                early_stopping_rounds=30,
                **fit_kwargs,
            )
        except TypeError:
            try:
                from xgboost.callback import EarlyStopping

                self.model.fit(
                    X_train,
                    y_train,
                    callbacks=[EarlyStopping(rounds=30, save_best=True)],
                    **fit_kwargs,
                )
            except Exception:
                # Last resort: train without early stopping
                self.model.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )

        # Diagnostics: best_iteration + train/val AUC gap
        best_it = getattr(self.model, "best_iteration", None)
        try:
            train_auc = float(roc_auc_score(np.asarray(y_train).astype(int), self.model.predict_proba(X_train)[:, 1]))
        except Exception:
            train_auc = float("nan")
        try:
            val_auc = float(roc_auc_score(np.asarray(y_val).astype(int), self.model.predict_proba(X_val)[:, 1]))
        except Exception:
            val_auc = float("nan")

        print(f"best_iteration: {best_it}")
        print(f"train_AUC: {train_auc:.4f}")
        print(f"val_AUC: {val_auc:.4f}")
        if np.isfinite(train_auc) and np.isfinite(val_auc) and (train_auc - val_auc) > 0.15:
            print("POSSIBLE OVERFIT DETECTED")

        self.params = params
        return self

    def cross_validate(self, X, y, n_folds: int = 5) -> Dict[str, float]:
        skf = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=42)

        aucs: list[float] = []
        f1s: list[float] = []

        y_arr = np.asarray(y).astype(int)
        for train_idx, val_idx in skf.split(X, y_arr):
            X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            y_tr = y_arr[train_idx]
            X_va = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            y_va = y_arr[val_idx]

            fold_model = SepsisXGBModel(params=dict(self.params or {}))
            fold_model.fit(X_tr, y_tr, X_va, y_va)

            y_proba = np.asarray(fold_model.predict_proba(X_va))[:, 1]
            y_pred = (y_proba >= 0.4).astype(int)

            try:
                aucs.append(float(roc_auc_score(y_va, y_proba)))
            except Exception:
                # If fold has single class (rare), skip AUC
                continue
            f1s.append(float(f1_score(y_va, y_pred)))

        mean_auroc = float(np.mean(aucs)) if aucs else 0.0
        std_auroc = float(np.std(aucs)) if aucs else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        std_f1 = float(np.std(f1s)) if f1s else 0.0

        if std_auroc > 0.05:
            print("HIGH VARIANCE - possible overfit")

        return {
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        }

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


