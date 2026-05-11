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
    parser.add_argument("--model-name", type=str, default="sepsis_xgboost")
    parser.add_argument("--fast", action="store_true", help="Faster training (fewer estimators)")
    parser.add_argument("--augment", action="store_true", help="Apply SMOTE to oversample minority class")
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

    # Distribution check
    train_ratio = float(train_df[builder.label_col].mean()) if len(train_df) else 0.0
    val_ratio = float(val_df[builder.label_col].mean()) if len(val_df) else 0.0
    test_ratio = float(test_df[builder.label_col].mean()) if len(test_df) else 0.0
    print(f"Train sepsis ratio: {train_ratio:.4f}")
    print(f"Val   sepsis ratio: {val_ratio:.4f}")
    print(f"Test  sepsis ratio: {test_ratio:.4f}")
    if abs(train_ratio - test_ratio) > 0.15:
        print("LABEL DISTRIBUTION SKEW")

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

    # 7) Cross-validation before main training
    params = None
    if args.fast:
        params = {"n_estimators": 50, "max_depth": 4}

    cv_probe = SepsisXGBModel(params=params)
    cv_metrics = cv_probe.cross_validate(X_train, y_train, n_folds=5)
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
    X_train_fit = X_train
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
        X_res, y_res = smote.fit_resample(X_train, y_train)

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
    model.fit(X_train_fit, y_train_fit, X_val, y_val)

    # 8) Overfit detection + metrics across splits
    print("\n=== Split metrics ===")
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

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
        feature_names=feature_cols,
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
