from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task

from monitoring.drift_detector import DriftDetector


def _get_tracking_uri() -> str:
    return (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("MLFLOW_URI")
        or "http://localhost:5000"
    )


def _get_latest_run_metrics(experiment_name: str) -> Tuple[Optional[str], Dict[str, float]]:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    if exp is None:
        return None, {}

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        return None, {}

    run = runs[0]
    return run.info.run_id, {
        k: float(v) for k, v in (run.data.metrics or {}).items()
    }


def _get_production_auroc(model_name: str) -> Tuple[Optional[str], float]:
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception as exc:
        print(f"[RetrainFlow] Could not get Production model: {exc}")
        return None, 0.0

    if not versions:
        return None, 0.0

    version = versions[0]
    run = client.get_run(version.run_id)
    auroc = float((run.data.metrics or {}).get("auroc", 0.0))

    return version.run_id, auroc


def _promote_run_to_production(run_id: str, model_name: str) -> bool:
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    candidates = [v for v in versions if getattr(v, "run_id", None) == run_id]

    if not candidates:
        print(f"[RetrainFlow] No model version found for run_id={run_id}")
        return False

    def _version_int(version: Any) -> int:
        try:
            return int(version.version)
        except Exception:
            return 0

    chosen = sorted(candidates, key=_version_int)[-1]

    client.transition_model_version_stage(
        name=model_name,
        version=chosen.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(
        f"[RetrainFlow] Promoted model={model_name}, "
        f"version={chosen.version}, run_id={run_id} to Production"
    )

    return True


@task
def check_drift(
    reference_path: str,
    current_path: str,
    db_url: str | None = None,
) -> dict:
    print("[RetrainFlow] Checking data drift...")
    print(f"[RetrainFlow] Reference path: {reference_path}")
    print(f"[RetrainFlow] Current path  : {current_path}")

    reference_file = Path(reference_path)
    current_file = Path(current_path)

    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    if not current_file.exists():
        raise FileNotFoundError(f"Current file not found: {current_path}")

    detector = DriftDetector(
        reference_data_path=reference_path,
        db_url=db_url,
        current_path=current_path,
    )

    result = detector.detect_drift()
    detector.save_report(output_path="reports/drift/")

    print(f"[RetrainFlow] Drift result: {result}")

    return result


@task
def run_training(data_path: str, experiment_name: str, model_name: str) -> dict:
    print("[RetrainFlow] Retraining model...")
    print(f"[RetrainFlow] Training data path: {data_path}")
    print(f"[RetrainFlow] Experiment name   : {experiment_name}")
    print(f"[RetrainFlow] Model name        : {model_name}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    cmd = [
        "python",
        "ml/train.py",
        "--data",
        data_path,
        "--experiment-name",
        experiment_name,
        "--model-name",
        model_name,
    ]

    subprocess.run(cmd, check=True)

    mlflow.set_tracking_uri(_get_tracking_uri())
    run_id, metrics = _get_latest_run_metrics(experiment_name)

    print(f"[RetrainFlow] Training completed")
    print(f"[RetrainFlow] Latest run_id: {run_id}")
    print(f"[RetrainFlow] Metrics      : {metrics}")

    return {
        "auroc": float(metrics.get("auroc", 0.0)),
        "run_id": str(run_id or ""),
    }


@task
def compare_and_promote(new_metrics: dict, model_name: str) -> bool:
    mlflow.set_tracking_uri(_get_tracking_uri())

    new_auroc = float(new_metrics.get("auroc", 0.0))
    new_run_id = str(new_metrics.get("run_id", "") or "")

    print("[RetrainFlow] Comparing new model with Production model...")
    print(f"[RetrainFlow] New AUROC: {new_auroc}")
    print(f"[RetrainFlow] New run_id: {new_run_id}")

    _prod_run_id, current_auroc = _get_production_auroc(model_name)

    print(f"[RetrainFlow] Current Production AUROC: {current_auroc}")

    if current_auroc <= 0.0 and new_run_id:
        print("[RetrainFlow] No Production model found. Promoting new model...")
        return _promote_run_to_production(new_run_id, model_name)

    if new_auroc > float(current_auroc) + 0.01 and new_run_id:
        print("[RetrainFlow] New model is better. Promoting...")
        return _promote_run_to_production(new_run_id, model_name)

    print("[RetrainFlow] New model is not better enough. Keeping current Production model.")
    return False


@flow(name="sepsis-retrain-flow")
def retrain_flow(
    reference_path: str = "data/processed/features_train.parquet",
    current_path: str = "data/processed/features_current.parquet",
    data_path: str = "data/synthetic/icu_data_synthetic.csv",
    experiment_name: str = "sepsis_retrain",
    model_name: str = "sepsis_xgboost",
):
    print("[RetrainFlow] Starting sepsis retrain flow")

    mlflow_uri = _get_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    print(f"[RetrainFlow] MLflow tracking URI: {mlflow_uri}")
    print(f"[RetrainFlow] Reference path     : {reference_path}")
    print(f"[RetrainFlow] Current path       : {current_path}")
    print(f"[RetrainFlow] Training data path : {data_path}")

    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

    drift_result = check_drift(
        reference_path=reference_path,
        current_path=current_path,
        db_url=db_url,
    )

    drift_score = float(drift_result.get("drift_score", 0.0))
    is_drift = bool(drift_result.get("is_drift", False))

    print(f"Drift score: {drift_score}")

    if is_drift:
        print("Significant drift detected")
        print("Retraining model...")

        metrics = run_training(
            data_path=data_path,
            experiment_name=experiment_name,
            model_name=model_name,
        )

        print("Training completed")

        promoted = compare_and_promote(
            new_metrics=metrics,
            model_name=model_name,
        )

        if promoted:
            print(f"New model promoted! AUROC: {metrics['auroc']}")
        else:
            print("New model not better, keeping current production model")
    else:
        print("No significant drift detected. Skipping retrain.")

    print("[RetrainFlow] Workflow completed")

    return drift_result


if __name__ == "__main__":
    retrain_flow()