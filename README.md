# Real Estate Price Prediction

A production-minded, end-to-end machine learning project that predicts home sale prices using the Ames Housing dataset from OpenML. It demonstrates real-world ML workflows: data ingestion, feature engineering, model training, evaluation, and artifact persistence.

## Why this project stands out

- **End-to-end ML pipeline**: From raw data to a trained model and metrics artifacts.
- **Reproducible CLI workflows**: Train and evaluate from the command line.
- **ML best practices**: Train/test split, preprocessing pipelines, and saved artifacts for deployment readiness.
- **Real estate domain**: Structured data with mixed numeric + categorical features, mirroring real production datasets.

If you're hiring for data science or ML engineering roles, this repo showcases practical skills in:
- Data wrangling and feature preprocessing
- Scikit-learn pipelines and model evaluation
- Experiment structure and reproducibility
- Clean, modular Python package design

## Project Structure

- `src/real_estate/data.py`: Downloads the Ames Housing dataset from OpenML.
- `src/real_estate/features.py`: Builds preprocessing pipelines for numeric and categorical features.
- `src/real_estate/modeling.py`: Model training, evaluation, and artifact persistence.
- `src/real_estate/train.py`: CLI entry point for training.
- `src/real_estate/evaluate.py`: CLI entry point for evaluating a saved model.
- `artifacts/`: Output directory for trained models and metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirments.txt
```

## Train the model

```bash
python -m real_estate.train --output-dir artifacts
```

Artifacts will be written to `artifacts/model.joblib` and `artifacts/metrics.json`.

## Evaluate the model

```bash
python -m real_estate.evaluate artifacts/model.joblib
```

## Notes

- The dataset is fetched from OpenML on first run and cached locally by scikit-learn.
- Modify `TrainingConfig` in `src/real_estate/config.py` to adjust model hyperparameters.

## Next steps (easy extensions)

- Add cross-validation and hyperparameter tuning.
- Track experiments with MLflow.
- Serve predictions via a lightweight API.
