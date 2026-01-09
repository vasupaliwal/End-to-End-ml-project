from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from real_estate.config import ProjectPaths, TrainingConfig
from real_estate.data import load_ames_housing
from real_estate.modeling import ModelArtifacts, evaluate_model, save_metrics, save_model, train_model


def run_training(output_dir: Path, config: TrainingConfig) -> None:
    dataset = load_ames_housing()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    pipeline = train_model(x_train, y_train, config)
    pipeline.fit(x_train, y_train)

    metrics = evaluate_model(pipeline, x_test, y_test)
    artifacts = ModelArtifacts(
        model_path=output_dir / "model.joblib",
        metrics_path=output_dir / "metrics.json",
    )

    save_model(pipeline, artifacts)
    save_metrics(metrics, artifacts)

    metrics_summary = asdict(metrics)
    print("Training complete. Metrics:")
    for key, value in metrics_summary.items():
        print(f"- {key}: {value:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a real estate price model.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ProjectPaths.default().artifacts,
        help="Directory to write model artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = TrainingConfig()
    run_training(args.output_dir, config)


if __name__ == "__main__":
    main()
