from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from real_estate.config import TrainingConfig
from real_estate.features import build_preprocessor


@dataclass(frozen=True)
class ModelArtifacts:
    model_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class ModelMetrics:
    mae: float
    rmse: float
    r2: float


def build_model(config: TrainingConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def train_model(features, target, config: TrainingConfig) -> Pipeline:
    preprocessor = build_preprocessor(features)
    model = build_model(config)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate_model(model: Pipeline, x_test, y_test) -> ModelMetrics:
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    return ModelMetrics(mae=mae, rmse=rmse, r2=r2)


def save_model(model: Pipeline, artifacts: ModelArtifacts) -> None:
    artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts.model_path)


def save_metrics(metrics: ModelMetrics, artifacts: ModelArtifacts) -> None:
    payload = {"mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2}
    artifacts.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.metrics_path.write_text(
        "{\n" + ",\n".join(f"  \"{k}\": {v:.4f}" for k, v in payload.items()) + "\n}"
    )
