from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from real_estate.config import TrainingConfig
from real_estate.data import load_ames_housing
from real_estate.modeling import evaluate_model


def run_evaluation(model_path: Path, config: TrainingConfig) -> None:
    dataset = load_ames_housing()
    _, x_test, _, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = joblib.load(model_path)
    metrics = evaluate_model(model, x_test, y_test)

    print("Evaluation metrics:")
    print(f"- mae: {metrics.mae:.4f}")
    print(f"- rmse: {metrics.rmse:.4f}")
    print(f"- r2: {metrics.r2:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved model.")
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the saved model.joblib.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = TrainingConfig()
    run_evaluation(args.model_path, config)


if __name__ == "__main__":
    main()
