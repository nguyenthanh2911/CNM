from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

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


def load_production_model(model_name: str):
    client = MlflowClient()

    def _try(stage: str):
        try:
            uri = f"models:/{model_name}/{stage}"
            return mlflow.xgboost.load_model(uri)
        except Exception:
            return None

    model = _try("Production")
    if model is not None:
        return model

    model = _try("Staging")
    if model is not None:
        return model

    # Last resort: try latest versions directly
    for stage in ["Production", "Staging", "None"]:
        try:
            versions = client.get_latest_versions(model_name, stages=[stage] if stage != "None" else None)
            if versions:
                uri = f"models:/{model_name}/{versions[0].version}"
                return mlflow.xgboost.load_model(uri)
        except Exception:
            continue

    raise RuntimeError(f"No model found for {model_name} in Production/Staging")
