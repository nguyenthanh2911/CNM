from __future__ import annotations

import argparse
import os
from typing import List

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline.labeling import (
    LABEL_COL_T6H,
    create_t6h_labels,
    get_label_stats,
    split_by_patient,
)
from ml.evaluate import evaluate_model
from ml.mlflow_utils import log_training_run, register_model
from ml.models.xgboost_model import SepsisXGBModel


# ── Feature columns (giữ nguyên như cũ) ─────────────────────────────
FEATURE_COLS: List[str] = [
    "heart_rate", "systolic_bp", "diastolic_bp",
    "temperature", "spo2", "respiratory_rate",
    "lactate", "wbc", "creatinine", "bilirubin", "platelet",
]

# ── Label mới: T+6h thay vì label tĩnh ──────────────────────────────
LABEL_COL = LABEL_COL_T6H          # "sepsis_in_next_6h"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Sepsis XGBoost model T+6h + MLflow logging"
    )
    parser.add_argument("--data",            type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default="CNM-Sepsis-T6H")
    parser.add_argument("--model-name",      type=str, default="sepsis_xgboost_t6h")
    parser.add_argument("--horizon",         type=int, default=6,
                        help="Prediction horizon in hours (default: 6)")
    parser.add_argument("--fast",    action="store_true")
    parser.add_argument("--augment", action="store_true",
                        help="Apply SMOTE (recommended vì label imbalance cao)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # ── 1. Load data ─────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded: {len(df):,} rows, {df['patient_id'].nunique()} patients")

    # ── 2. Tạo label T+6h ───────────────────────────────────────────
    print(f"\nCreating T+{args.horizon}h labels...")
    df = create_t6h_labels(df, horizon_hours=args.horizon)

    stats = get_label_stats(df, label_col=LABEL_COL)
    print(f"Label stats:")
    print(f"  positive_ratio : {stats['positive_ratio']}")
    print(f"  imbalance_ratio: {stats['imbalance_ratio']}:1")
    print(f"  positive rows  : {stats['positive']:,}")
    print(f"  negative rows  : {stats['negative']:,}")

    # ── 3. Patient-based split (tránh leakage) ───────────────────────
    print("\nSplitting by patient (no leakage)...")
    train_df, val_df, test_df = split_by_patient(
        df,
        test_ratio=0.2,
        val_ratio=0.2,
        random_seed=42,
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL]
    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df[LABEL_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[LABEL_COL]

    # ── 4. Preprocessing pipeline ────────────────────────────────────
    preprocess_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_train_p = preprocess_pipeline.fit_transform(X_train)
    X_val_p   = preprocess_pipeline.transform(X_val)
    X_test_p  = preprocess_pipeline.transform(X_test)

    # Convert numpy arrays back to DataFrames để giữ feature names (XGBoost cần)
    X_train_p = pd.DataFrame(X_train_p, columns=FEATURE_COLS) if not isinstance(X_train_p, pd.DataFrame) else X_train_p
    X_val_p   = pd.DataFrame(X_val_p, columns=FEATURE_COLS) if not isinstance(X_val_p, pd.DataFrame) else X_val_p
    X_test_p  = pd.DataFrame(X_test_p, columns=FEATURE_COLS) if not isinstance(X_test_p, pd.DataFrame) else X_test_p

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(preprocess_pipeline, "artifacts/preprocessor_t6h.joblib")
    print("\nSaved: artifacts/preprocessor_t6h.joblib")

    # ── 5. Cross-validation ──────────────────────────────────────────
    params = None
    if args.fast:
        params = {"n_estimators": 50, "max_depth": 4}

    print("\nRunning 5-fold cross-validation...")
    cv_probe = SepsisXGBModel(params=params)
    cv_metrics = cv_probe.cross_validate(X_train_p, y_train, n_folds=5)
    mean_auc = float(cv_metrics.get("mean_auroc", 0.0))
    std_auc  = float(cv_metrics.get("std_auroc",  0.0))
    print(f"CV AUROC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Auto-regularize nếu variance cao
    if std_auc > 0.08:
        print("CV variance high → tightening regularization")
        tuned = dict(params or {})
        tuned["max_depth"]   = 3
        tuned["reg_lambda"]  = 3.0
        params = tuned

    # ── 6. SMOTE — QUAN TRỌNG vì imbalance_ratio ~9:1 ───────────────
    X_train_fit = X_train_p
    y_train_fit = y_train

    # Tự động bật SMOTE nếu imbalance_ratio > 5 (hoặc dùng --augment)
    should_smote = args.augment or stats["imbalance_ratio"] > 5
    if should_smote:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as e:
            raise SystemExit(
                "imbalanced-learn chưa cài. Chạy: pip install imbalanced-learn"
            ) from e

        y_arr = np.asarray(y_train).astype(int)
        print(f"\nSMOTE before: neg={int((y_arr==0).sum())}, pos={int((y_arr==1).sum())}")

        # sampling_strategy=0.4: tăng minority lên 40% của majority
        # (không cân bằng hoàn toàn vì imbalance thực tế trong ICU)
        smote = SMOTE(
            sampling_strategy=0.4,
            random_state=42,
            k_neighbors=min(5, int((y_arr == 1).sum()) - 1),
        )
        X_res, y_res = smote.fit_resample(X_train_p, y_train)

        if isinstance(X_res, np.ndarray):
            X_res = pd.DataFrame(X_res, columns=FEATURE_COLS)

        y_res_arr = np.asarray(y_res).astype(int)
        print(f"SMOTE after:  neg={int((y_res_arr==0).sum())}, pos={int((y_res_arr==1).sum())}")

        X_train_fit = X_res
        y_train_fit = y_res

    # ── 7. Train model ───────────────────────────────────────────────
    print("\nTraining XGBoost T+6h model...")
    model = SepsisXGBModel(params=params)
    model.fit(X_train_fit, y_train_fit, X_val_p, y_val)

    # ── 8. Evaluate ──────────────────────────────────────────────────
    print("\n=== Split metrics ===")
    train_metrics = evaluate_model(model, X_train_p, y_train)
    val_metrics   = evaluate_model(model, X_val_p,   y_val)
    test_metrics  = evaluate_model(model, X_test_p,  y_test)

    train_auroc = float(train_metrics.get("auroc", 0.0))
    val_auroc   = float(val_metrics.get("auroc",   0.0))
    test_auroc  = float(test_metrics.get("auroc",  0.0))
    gap         = float(train_auroc - test_auroc)

    print(f"\n=== AUROC ===")
    print(f"Train : {train_auroc:.4f}")
    print(f"Val   : {val_auroc:.4f}")
    print(f"Test  : {test_auroc:.4f}")
    print(f"Gap   : {gap:.4f}")
    if gap > 0.10:
        print(f"WARNING: Possible overfitting (gap={gap:.4f})")

    metrics = {
        **{f"train_{k}": float(v) for k, v in train_metrics.items()
           if isinstance(v, (int, float))},
        **{f"val_{k}":   float(v) for k, v in val_metrics.items()
           if isinstance(v, (int, float))},
        **{f"test_{k}":  float(v) for k, v in test_metrics.items()
           if isinstance(v, (int, float))},
        "gap_train_test":        float(gap),
        "label_positive_ratio":  float(stats["positive_ratio"]),
        "label_imbalance_ratio": float(stats["imbalance_ratio"]),
        "horizon_hours":         float(args.horizon),
        "cv_mean_auroc":         float(mean_auc),
        "cv_std_auroc":          float(std_auc),
    }

    # ── 9. MLflow logging ────────────────────────────────────────────
    run_id = log_training_run(
        params={**(model.params or {}), "horizon_hours": args.horizon},
        metrics=metrics,
        model=model,
        feature_names=FEATURE_COLS,
        run_name="train-xgb-t6h",
        experiment_name=args.experiment_name,
    )

    # ── 10. Register model ───────────────────────────────────────────
    # T+6h task khó hơn → hạ ngưỡng xuống 0.75/0.70
    stage = "Not Registered"
    if test_auroc > 0.75 and gap < 0.12:
        stage = "Production"
        register_model(run_id, args.model_name, stage=stage)
    elif test_auroc > 0.70 and gap < 0.18:
        stage = "Staging"
        register_model(run_id, args.model_name, stage=stage)
    else:
        print("Model quality insufficient for registration")

    # ── 11. Summary ──────────────────────────────────────────────────
    print("\n=== Training finished ===")
    print(f"Label         : {LABEL_COL} (T+{args.horizon}h)")
    print(f"MLflow run    : {run_id}")
    print(f"Model name    : {args.model_name}")
    print(f"Stage         : {stage}")
    print(f"Test AUROC    : {test_auroc:.4f}")
    print(f"Imbalance     : {stats['imbalance_ratio']}:1 → SMOTE applied: {should_smote}")