"""
src/qubo_project/model.py

Train a binary classification model using a reduced feature dataset and save
both the trained model and training metadata.

Python 3.11+
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def _get_classifier(name: str, seed: int, params: dict[str, Any] | None = None):
    """
    Return the requested classifier instance initialized with default or user-specified parameters.

    Parameters
    ----------
    name : str
        Classifier name (case-insensitive).
    seed : int
        Random seed.
    params : dict[str, Any], optional
        Additional parameter overrides passed from CLI or function calls.

    Returns
    -------
    tuple[str, Any]
    """
    params = params or {}
    key = name.strip().lower()

    if key == "xgboost":
        model_kwargs = {
            "random_state": seed,
            "eval_metric": "logloss",
            "n_estimators": params.get("n_estimators") or 100,
            "learning_rate": params.get("learning_rate") or 0.1,
            "max_depth": params.get("max_depth") or 6,
            "n_jobs": params.get("n_jobs") or -1,
        }
        # Filter out keys if explicitly set to None
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        return key, XGBClassifier(**model_kwargs)

    elif key == "knn":
        model_kwargs = {
            "n_neighbors": params.get("n_neighbors") or 5,
            "n_jobs": params.get("n_jobs") or -1,
            "weights": params.get("weights") or "uniform",
        }
        return key, KNeighborsClassifier(**model_kwargs)

    elif key == "lightgbm":
        model_kwargs = {
            "random_state": seed,
            "verbose": -1,
            "n_estimators": params.get("n_estimators") or 100,
            "learning_rate": params.get("learning_rate") or 0.1,
            "max_depth": params.get("max_depth") or -1,
            "n_jobs": params.get("n_jobs") or -1,
        }
        return key, LGBMClassifier(**model_kwargs)

    else:
        supported = "xgboost, knn, lightgbm"
        raise ValueError(
            f"Unsupported classifier '{name}'. "
            f"Supported classifiers: {supported}"
        )


def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
    **clf_params: Any,
):
    """
    Train a binary classification model.

    Parameters
    ----------
    classifier : str
        Model name (e.g., xgboost, knn, lightgbm).
    reducedTrain_csv : str
        Path to reduced training CSV.
    target_column : str
        Target column name.
    model_path : str
        Output path for trained model (.joblib).
    metrics_json : str
        Output path for metrics JSON.
    seed : int, default=42
        Random seed.
    **clf_params : Any
        Optional hyperparameter overrides for the chosen classifier.
    """
    csv_path = "outputs"/Path(reducedTrain_csv)
    model_file = "outputs"/Path(model_path)
    metrics_file = "outputs"/Path(metrics_json)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Load dataset
    start = time.perf_counter()
    df = pd.read_csv(csv_path)
    dataset_input_time = time.perf_counter() - start

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    n_samples = int(len(df))
    n_features = int(X.shape[1])
    target_1_percentage = round(float((y == 1).mean() * 100.0), 2)

    classifier_name, model = _get_classifier(classifier, seed, clf_params)

    # Train model
    start = time.perf_counter()
    model.fit(X, y)
    training_time = time.perf_counter() - start

    # Save model
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)

    # Save metrics JSON matching specification section 11.3
    metrics = {
        "classifier": classifier_name,
        "seed": seed,
        "training_dataset": csv_path.name,
        "target_column": target_column,
        "model_path": model_file.name,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": target_1_percentage,
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2),
        "classifier_metrics": [
            clf_params
        ]
    }

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a binary classification model."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a classifier.",
    )

    # Mandatory CLI flags
    train_parser.add_argument(
        "--classifier",
        required=True,
        type=str,
        help="Classifier to use (xgboost, knn, lightgbm).",
    )

    train_parser.add_argument(
        "--in-reduced",
        required=True,
        dest="reduced_train",
        type=str,
        help="Reduced training CSV.",
    )

    train_parser.add_argument(
        "--target",
        required=True,
        type=str,
        help="Target column.",
    )

    train_parser.add_argument(
        "--out-model",
        required=True,
        dest="model_path",
        type=str,
        help="Output joblib model.",
    )

    train_parser.add_argument(
        "--out-metrics",
        required=True,
        dest="metrics_json",
        type=str,
        help="Output metrics JSON.",
    )

    train_parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed.",
    )

    # Optional Classifier-Specific Arguments
    train_parser.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="Number of neighbors for KNN.",
    )

    train_parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for CPU processing.",
    )

    train_parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of boosting trees (XGBoost/LightGBM).",
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (XGBoost/LightGBM).",
    )

    train_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (XGBoost/LightGBM).",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            # Extract optional classifier parameter overrides
            clf_params = {
                "n_neighbors": args.n_neighbors,
                "n_jobs": args.n_jobs,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
            }

            train(
                classifier=args.classifier,
                reducedTrain_csv=args.reduced_train,
                target_column=args.target,
                model_path=args.model_path,
                metrics_json=args.metrics_json,
                seed=args.seed,
                **clf_params,
            )
            print("Training completed successfully.")

    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()