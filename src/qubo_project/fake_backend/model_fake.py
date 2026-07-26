from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict


def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int,
) -> Dict[str, object]:

    random.seed(seed)

    time.sleep(3)

    accuracy = round(random.uniform(0.82, 0.98), 3)
    precision = round(random.uniform(0.80, accuracy), 3)
    recall = round(random.uniform(0.80, accuracy), 3)
    f1 = round((2 * precision * recall) / (precision + recall), 3)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    Path(model_path).write_text(
        "Fake Joblib model placeholder",
        encoding="utf-8",
    )

    metrics = {
        "classifier": classifier,
        "target_column": target_column,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "seed": seed,
        "status": "success",
    }

    with open(metrics_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    return metrics


def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
) -> Dict[str, object]:
    """
    Fake prediction.

    Signature intentionally matches the future backend.
    """

    time.sleep(2)

    Path(predictions_csv).parent.mkdir(parents=True, exist_ok=True)

    predictions = [
        "id,prediction",
        "1,0",
        "2,1",
        "3,1",
        "4,0",
        "5,1",
    ]

    Path(predictions_csv).write_text(
        "\n".join(predictions),
        encoding="utf-8",
    )

    stats = {
        "accuracy": 0.94,
        "precision": 0.93,
        "recall": 0.95,
        "f1_score": 0.94,
        "roc_auc": 0.97,
        "confusion_matrix": [
            [48, 2],
            [3, 47],
        ],
        "status": "success",
        "target_column": target_column,
        "model_path": model_path,
    }

    with open(classif_stats_json, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4)

    return stats