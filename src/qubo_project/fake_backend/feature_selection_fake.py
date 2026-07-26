from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict


def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float,
    percSelected: float,
    allowance: float,
    seed: int,
    alpha_computations: int,
) -> Dict[str, object]:
    """
    Fake feature selection.

    Signature intentionally matches the future backend.
    """

    random.seed(seed)

    time.sleep(3)

    selected_features = [
        f"feature_{i}"
        for i in random.sample(range(1, 21), 8)
    ]

    Path(reducedTrain_csv).parent.mkdir(parents=True, exist_ok=True)

    Path(reducedTrain_csv).write_text(
        ",".join(selected_features + [target_column]) + "\n",
        encoding="utf-8",
    )

    Path(reducedTest_csv).write_text(
        ",".join(selected_features + [target_column]) + "\n",
        encoding="utf-8",
    )

    Path(output_ottim_csv).write_text(
        "feature,score\n"
        + "\n".join(
            f"{feature},{random.uniform(0.70,1.00):.3f}"
            for feature in selected_features
        ),
        encoding="utf-8",
    )

    report = {
        "status": "success",
        "selected_features": selected_features,
        "num_selected": len(selected_features),
        "percSelected": percSelected,
        "percTest": percTest,
        "allowance": allowance,
        "seed": seed,
        "alpha_computations": alpha_computations,
    }

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report