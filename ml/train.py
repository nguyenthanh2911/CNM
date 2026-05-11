from __future__ import annotations

import argparse
import os
from typing import List

import mlflow
import numpy as np
import pandas as pd

from data_pipeline.preprocessor import ICUPreprocessor
from feature_engineering.feature_builder import FeatureBuilder

from ml.evaluate import evaluate_model
from ml.mlflow_utils import log_training_run, register_model
from ml.models.xgboost_model import SepsisXGBModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sepsis XGBoost model + MLflow logging")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--experiment-name", type=str, default="CNM-Sepsis")
    parser.add_argument("--model-name", type=str, default="SepsisXGB")
    parser.add_argument("--fast", action="store_true", help="Faster training (fewer estimators)")
    return parser.parse_args()


def _chronological_split(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


if __name__ == "__main__":
    args = _parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    df = pd.read_csv(args.data)

    # 3) Preprocess (imputation, scaling)
    pre = ICUPreprocessor()
    df_clean = pre.fit_transform(df)
    os.makedirs("artifacts", exist_ok=True)
    pre.save("artifacts/preprocessor.joblib")

    # 4) Feature building
    builder = FeatureBuilder()
    df_feat = builder.build(df_clean)

    # 5) Chronological split
    time_col = builder.time_col
    train_df, val_df, test_df = _chronological_split(df_feat, time_col=time_col)

    # 6) X, y
    id_cols: List[str] = [builder.patient_col, builder.time_col]
    label_col = builder.label_col
    feature_cols = [c for c in df_feat.columns if c not in id_cols + [label_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_val = val_df[feature_cols]
    y_val = val_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    # 7) Train model
    params = None
    if args.fast:
        params = {"n_estimators": 50, "max_depth": 4}

    model = SepsisXGBModel(params=params)
    model.fit(X_train, y_train, X_val, y_val)

    # 8) Evaluate on test
    metrics = evaluate_model(model, X_test, y_test)

    # 9) Log to MLflow
    run_id = log_training_run(
        params=model.params or {},
        metrics=metrics,
        model=model,
        feature_names=feature_cols,
        run_name="train-xgb",
        experiment_name=args.experiment_name,
    )

    # 10) Register if good
    stage = "Staging"
    if metrics.get("auroc", 0.0) > 0.75:
        stage = "Production"
        register_model(run_id, args.model_name, stage=stage)

    # 11) Final results
    print("\n=== Training finished ===")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Run ID: {run_id}")
    print(f"Model name: {args.model_name}")
    print(f"Stage decision: {stage}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
