from __future__ import annotations

import argparse
import os
from typing import List

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluate import evaluate_model
from ml.mlflow_utils import log_training_run, register_model
from ml.models.xgboost_model import SepsisXGBModel


FEATURE_COLS = [
    'heart_rate', 'systolic_bp', 'diastolic_bp',
    'temperature', 'spo2', 'respiratory_rate',
    'lactate', 'wbc', 'creatinine', 'bilirubin', 'platelet'
]
LABEL_COL = 'sepsis_label'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sepsis XGBoost model + MLflow logging")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--experiment-name", type=str, default="CNM-Sepsis")
    parser.add_argument("--model-name", type=str, default="sepsis_xgboost")
    parser.add_argument("--fast", action="store_true", help="Faster training (fewer estimators)")
    parser.add_argument("--augment", action="store_true", help="Apply SMOTE to oversample minority class")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    df = pd.read_csv(args.data)

    # Ép kiểu numeric như notebook
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    print(f"Total records  : {len(df):,}")
    print(f"Sepsis ratio   : {y.mean():.4f}")
    print(f"Features       : {len(FEATURE_COLS)}")

    # Split giống notebook: stratified 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Train sepsis: {y_train.mean():.4f} | Val: {y_val.mean():.4f} | Test: {y_test.mean():.4f}")

    # sklearn Pipeline: impute median + scale
    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    X_train_p = preprocess_pipeline.fit_transform(X_train)
    X_val_p   = preprocess_pipeline.transform(X_val)
    X_test_p  = preprocess_pipeline.transform(X_test)

    # Lưu pipeline
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(preprocess_pipeline, "artifacts/preprocessor.joblib")
    print("Saved: artifacts/preprocessor.joblib")

    feature_cols = FEATURE_COLS

    # 7) Cross-validation before main training
    params = None
    if args.fast:
        params = {"n_estimators": 50, "max_depth": 4}

    cv_probe = SepsisXGBModel(params=params)
    cv_metrics = cv_probe.cross_validate(X_train_p, y_train, n_folds=5)
    mean_auc = float(cv_metrics.get("mean_auroc", 0.0))
    std_auc = float(cv_metrics.get("std_auroc", 0.0))
    print(f"CV AUROC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Auto-regularize if high variance
    if std_auc > 0.08:
        print("CV variance high -> tightening regularization")
        tuned = dict(params or {})
        tuned["max_depth"] = 3
        tuned["reg_lambda"] = 3.0
        params = tuned

    # Optional: SMOTE augmentation on training set only
    X_train_fit = X_train_p
    y_train_fit = y_train
    if args.augment:
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
        except Exception as e:
            raise SystemExit(
                "--augment requires imbalanced-learn. Install it (pip install imbalanced-learn)"
            ) from e

        y_arr = np.asarray(y_train).astype(int)
        before_neg = int((y_arr == 0).sum())
        before_pos = int((y_arr == 1).sum())
        print(f"SMOTE before: neg={before_neg}, pos={before_pos}")

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train_p, y_train)

        # Keep pandas columns if possible
        if isinstance(X_res, np.ndarray):
            X_res = pd.DataFrame(X_res, columns=feature_cols)

        y_res_arr = np.asarray(y_res).astype(int)
        after_neg = int((y_res_arr == 0).sum())
        after_pos = int((y_res_arr == 1).sum())
        print(f"SMOTE after:  neg={after_neg}, pos={after_pos}")

        X_train_fit = X_res
        y_train_fit = y_res

    model = SepsisXGBModel(params=params)
    model.fit(X_train_fit, y_train_fit, X_val_p, y_val)

    # 8) Overfit detection + metrics across splits
    print("\n=== Split metrics ===")
    train_metrics = evaluate_model(model, X_train_p, y_train)
    val_metrics   = evaluate_model(model, X_val_p,   y_val)
    test_metrics  = evaluate_model(model, X_test_p,  y_test)

    train_auroc = float(train_metrics.get("auroc", 0.0))
    val_auroc = float(val_metrics.get("auroc", 0.0))
    test_auroc = float(test_metrics.get("auroc", 0.0))
    gap = float(train_auroc - test_auroc)

    print("\n=== AUROC comparison ===")
    print(f"Train AUROC: {train_auroc:.4f}")
    print(f"Val   AUROC: {val_auroc:.4f}")
    print(f"Test  AUROC: {test_auroc:.4f}")
    print(f"Gap (Train-Test): {gap:.4f}")
    if gap > 0.10:
        print(f"WARNING: Model may be overfitting (gap={gap:.4f})")

    # Metrics to log (includes train/val/test AUROC)
    metrics = {
        **{f"train_{k}": float(v) for k, v in train_metrics.items() if isinstance(v, (int, float))},
        **{f"val_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        **{f"test_{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        "gap_train_test": float(gap),
    }

    # 9) Log to MLflow
    run_id = log_training_run(
        params=model.params or {},
        metrics=metrics,
        model=model,
        feature_names=FEATURE_COLS,
        run_name="train-xgb",
        experiment_name=args.experiment_name,
    )

    # 10) Register if good
    stage = "Not Registered"
    if test_auroc > 0.80 and gap < 0.10:
        stage = "Production"
        register_model(run_id, args.model_name, stage=stage)
    elif test_auroc > 0.75 and gap < 0.15:
        stage = "Staging"
        register_model(run_id, args.model_name, stage=stage)
    else:
        print("Model quality insufficient")

    # 11) Final results
    print("\n=== Training finished ===")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Run ID: {run_id}")
    print(f"Model name: {args.model_name}")
    print(f"Stage decision: {stage}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
