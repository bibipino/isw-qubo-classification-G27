from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict


def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitialRes_json: str,
    minPercValid: float,
) -> Dict[str, object]:
    """
    Fake implementation of the preprocessing stage.

    This function has exactly the same signature as the future backend.
    It simulates preprocessing by waiting a few seconds and generating
    a fake CSV file and a JSON report.
    """

    time.sleep(2)

    normalized_path = Path(normalized_csv)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_path.write_text(
        "feature_1,feature_2,target\n"
        "0.15,0.91,0\n"
        "0.43,0.72,1\n"
        "0.84,0.10,0\n",
        encoding="utf-8",
    )

    report = {
        "input_csv": input_csv,
        "target_column": target_column,
        "minPercValid": minPercValid,
        "rows_before": 150,
        "rows_after": 148,
        "removed_rows": 2,
        "normalized_columns": 12,
        "status": "success",
    }

    json_path = Path(outInitialRes_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report