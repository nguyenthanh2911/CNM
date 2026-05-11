from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


def get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name)


def log_training_run(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model,
    feature_names: List[str],
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> str:
    if experiment_name:
        exp_id = get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    else:
        exp_id = None

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model (xgboost flavor)
        xgb_model = getattr(model, "model", model)
        mlflow.xgboost.log_model(xgb_model=xgb_model, artifact_path="model")

        # Log feature names as artifact json
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "feature_names.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"feature_names": list(feature_names)}, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(path)

        return mlflow.active_run().info.run_id


def register_model(run_id: str, model_name: str, stage: str = "Staging") -> str:
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Transition stage
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
        archive_existing_versions=False,
    )

    return str(mv.version)


def load_production_model_with_metadata(model_name: str) -> Tuple[Any, Dict[str, Any]]:
    """Load a model from the MLflow Model Registry and return basic metadata.

    Metadata keys (best-effort):
    - model_name
    - model_version
    - model_stage
    - model_auroc
    - model_uri
    - run_id
    """
    client = MlflowClient()

    def _pick_model_version() -> Any | None:
        def _version_key(v: Any) -> int:
            try:
                return int(getattr(v, "version", 0))
            except Exception:
                return 0

        # Prefer Production -> Staging, then fall back to latest.
        # Use `search_model_versions` to avoid deprecated `get_latest_versions`.
        try:
            versions = list(client.search_model_versions(f"name='{model_name}'"))
        except Exception:
            versions = []

        if versions:
            for preferred_stage in ("Production", "Staging"):
                stage_versions = [v for v in versions if getattr(v, "current_stage", None) == preferred_stage]
                if stage_versions:
                    return max(stage_versions, key=_version_key)
            return max(versions, key=_version_key)

        # Fallback for older/alternative backends
        try:
            latest = client.get_latest_versions(model_name)
            if latest:
                return latest[0]
        except Exception:
            return None

        return None

    mv = _pick_model_version()
    if mv is None:
        raise RuntimeError(f"No registered model versions found for {model_name}")

    version = str(getattr(mv, "version", "unknown"))
    stage = getattr(mv, "current_stage", None)
    run_id = getattr(mv, "run_id", None)
    model_uri = f"models:/{model_name}/{version}"

    model = mlflow.xgboost.load_model(model_uri)

    auroc: float | None = None
    if run_id:
        try:
            run = client.get_run(run_id)
            auroc_val = run.data.metrics.get("auroc")
            if auroc_val is not None:
                auroc = float(auroc_val)
        except Exception:
            auroc = None

    meta: Dict[str, Any] = {
        "model_name": model_name,
        "model_version": version,
        "model_stage": stage,
        "model_auroc": auroc,
        "model_uri": model_uri,
        "run_id": run_id,
    }
    return model, meta


def load_production_model(model_name: str):
    model, _meta = load_production_model_with_metadata(model_name)
    return model
