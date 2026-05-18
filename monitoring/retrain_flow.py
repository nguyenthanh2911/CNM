from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task

from monitoring.drift_detector import DriftDetector


def _get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_URI") or "http://localhost:5000"


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
    return run.info.run_id, {k: float(v) for k, v in (run.data.metrics or {}).items()}


def _get_production_auroc(model_name: str) -> Tuple[Optional[str], float]:
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None, 0.0

    v = versions[0]
    run = client.get_run(v.run_id)
    auroc = float((run.data.metrics or {}).get("test_auroc", 0.0))
    return v.run_id, auroc


def _promote_run_to_production(run_id: str, model_name: str) -> bool:
    client = MlflowClient()

    # Find the model version logged from this run
    versions = client.search_model_versions(f"name='{model_name}'")
    candidates = [v for v in versions if getattr(v, "run_id", None) == run_id]
    if not candidates:
        return False

    # Pick highest version if multiple
    def _version_int(v: Any) -> int:
        try:
            return int(v.version)
        except Exception:
            return 0

    chosen = sorted(candidates, key=_version_int)[-1]

    client.transition_model_version_stage(
        name=model_name,
        version=chosen.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return True


@task
def check_drift(reference_path: str, db_url: str | None) -> dict:
    detector = DriftDetector(reference_path, db_url)
    result = detector.detect_drift()
    detector.save_report(output_path="reports/drift/")
    return result


@task
def run_training(data_path: str, experiment_name: str, model_name: str) -> dict:
    # Run training as a subprocess
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

    # Read latest run metrics from MLflow
    mlflow.set_tracking_uri(_get_tracking_uri())
    run_id, metrics = _get_latest_run_metrics(experiment_name)

    return {"auroc": float(metrics.get("test_auroc", 0.0)), "run_id": str(run_id or "")}


@task
def compare_and_promote(new_metrics: dict, model_name: str) -> bool:
    mlflow.set_tracking_uri(_get_tracking_uri())

    new_auroc = float(new_metrics.get("auroc", 0.0))
    new_run_id = str(new_metrics.get("run_id", "") or "")

    _prod_run_id, current_auroc = _get_production_auroc(model_name)

    # If there's no production model yet, promote immediately
    if current_auroc <= 0.0 and new_run_id:
        return _promote_run_to_production(new_run_id, model_name)

    if new_auroc > float(current_auroc) + 0.01 and new_run_id:
        return _promote_run_to_production(new_run_id, model_name)

    return False


@flow(name="sepsis-retrain-flow")
def retrain_flow(
    reference_path: str = "data/processed/features_train.parquet",
    data_path: str = "data/synthetic/icu_data_synthetic.csv",
    experiment_name: str = "CNM-Sepsis-T6H",
    model_name: str = "sepsis_xgboost_t6h",
):
    mlflow.set_tracking_uri(_get_tracking_uri())

    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    drift_result = check_drift(reference_path, db_url=db_url)

    print(f"Drift score: {drift_result['drift_score']}")

    if drift_result["is_drift"]:
        print("Drift detected! Starting retraining...")
        metrics = run_training(data_path, experiment_name, model_name)
        promoted = compare_and_promote(metrics, model_name)
        if promoted:
            print(f"New model promoted! AUROC: {metrics['auroc']}")
        else:
            print("New model not better, keeping current production model")
    else:
        print("No significant drift detected. Skipping retrain.")

    return drift_result


if __name__ == "__main__":
    retrain_flow()
