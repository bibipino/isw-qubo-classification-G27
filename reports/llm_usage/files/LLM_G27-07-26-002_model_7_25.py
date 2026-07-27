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
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
):
    outputs = Path("outputs")

    csv_path = outputs / Path(reduced_Test_csv)
    model_file = outputs / Path(model_path)
    predictions_file = outputs / Path(predictions_csv)
    stats_file = outputs / Path(classif_stats_json)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = joblib.load(model_file)

    prediction = model.predict(X)

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]

    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)

        decision = np.asarray(decision)

        if decision.ndim > 1:
            decision = decision[:, -1]

        score = decision

    else:
        score = np.asarray(prediction, dtype=float)

    predictions = pd.DataFrame(
        {
            "row_n": range(len(df)),
            "target": y.astype(int),
            "prediction": prediction.astype(int),
            "score": score.astype(float),
        }
    )

    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_file, index=False)

    accuracy = accuracy_score(y, prediction)

    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        prediction,
        labels=[0, 1],
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y, score)
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y, prediction, labels=[0, 1])

    model_type = type(model).__name__.lower()
    if "kneighbors" in model_type or "knn" in model_type:
        classifier_name = "knn"
    elif "xgb" in model_type:
        classifier_name = "xgboost"
    elif "randomforest" in model_type or "random_forest" in model_type:
        classifier_name = "random_forest"
    else:
        classifier_name = model_type

    stats = {
        "classifier": classifier_name,
        "n_samples": int(len(df)),
        "target_1_count": int((y == 1).sum()),
        "target_1_percentage": float(round((y == 1).mean() * 100, 2)),
        "accuracy": float(accuracy),
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.astype(int).tolist(),
        },
    }

    stats_file.parent.mkdir(parents=True, exist_ok=True)

    with stats_file.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    return predictions

def _get_classifier(name: str, seed: int, params: dict[str, Any] | None = None):
    """
    Return the requested classifier instance initialized with default or user-specified parameters.
    """
    params = params or {}
    key = name.strip().lower()

    if key in ["random_forest", "randomforest"]:
        model_kwargs = {
            "random_state": seed,
            "n_estimators": params.get("n_estimators") or 100,
            "max_depth": params.get("max_depth"),
            "n_jobs": params.get("n_jobs") or -1,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        return "random_forest", RandomForestClassifier(**model_kwargs)

    elif key == "xgboost":
        model_kwargs = {
            "random_state": seed,
            "eval_metric": "logloss",
            "n_estimators": params.get("n_estimators") or 100,
            "learning_rate": params.get("learning_rate") or 0.1,
            "max_depth": params.get("max_depth") or 6,
            "n_jobs": params.get("n_jobs") or -1,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        return "xgboost", XGBClassifier(**model_kwargs)

    elif key == "knn":
        model_kwargs = {
            "n_neighbors": params.get("n_neighbors") or 5,
            "n_jobs": params.get("n_jobs") or -1,
            "weights": params.get("weights") or "uniform",
        }
        return "knn", KNeighborsClassifier(**model_kwargs)

    else:
        supported = "random_forest, xgboost, knn"
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
    csv_path = Path("outputs") / Path(reducedTrain_csv)
    model_file = Path("outputs") / Path(model_path)
    metrics_file = Path("outputs") / Path(metrics_json)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

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

    start = time.perf_counter()
    model.fit(X, y)
    training_time = time.perf_counter() - start

    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)

    active_params = {k: v for k, v in clf_params.items() if v is not None}

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
            active_params
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

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction using a trained classifier.",
    )

    predict_parser.add_argument(
        "--input-testset",
        required=True,
        dest="reduced_test",
        type=str,
    )

    predict_parser.add_argument(
        "--target",
        required=True,
        type=str,
    )

    predict_parser.add_argument(
        "--model",
        required=True,
        dest="model_path",
        type=str,
    )

    predict_parser.add_argument(
        "--out-predictions",
        required=True,
        dest="predictions_csv",
        type=str,
    )

    predict_parser.add_argument(
        "--out-stats",
        required=True,
        dest="stats_json",
        type=str,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a classifier.",
    )

    train_parser.add_argument(
        "--classifier",
        required=True,
        type=str,
        help="Classifier to use (random_forest, xgboost, knn).",
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
        help="Number of trees (RandomForest/XGBoost).",
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (XGBoost).",
    )

    train_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (RandomForest/XGBoost).",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
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
        elif args.command == "predict":
            predict(
                reduced_Test_csv=args.reduced_test,
                target_column=args.target,
                model_path=args.model_path,
                predictions_csv=args.predictions_csv,
                classif_stats_json=args.stats_json,
            )

            print("Prediction completed successfully.")
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()