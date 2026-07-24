# File: src/qubo_project/preprocessing.py

"""
Preprocessing module for the QUBO Project.

Pipeline:
1. Load a CSV dataset.
2. Separate the target column.
3. Remove feature columns whose percentage of:
   - missing values OR
   - zero values
   strictly exceeds the specified threshold.
4. Apply Z-score normalization to all remaining features.
5. Deterministically split the dataset (no shuffling).
6. Save the resulting training and test datasets.

The implementation is designed to be efficient for very large datasets
(>1.5 million samples) by relying on vectorized pandas/numpy operations.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def preprocess_dataset(
    dataset_path: str | Path,
    target_column: str,
    drop_threshold_percent: float,
    test_size_percent: float,
    output_dir: str | Path = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess a dataset according to the project specifications.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the input CSV dataset.

    target_column : str
        Name of the binary target column.

    drop_threshold_percent : float
        Threshold percentage used to remove feature columns.
        A feature is removed if:
            %missing > threshold
            OR
            %zeros > threshold

    test_size_percent : float
        Percentage of samples assigned to the test set.

    output_dir : str or Path, default="data"
        Directory where training.csv and test.csv will be saved.

    Returns
    -------
    training_df : pandas.DataFrame
        Training dataset (features + target).

    test_df : pandas.DataFrame
        Test dataset (features + target).
    """

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(dataset_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    # ------------------------------------------------------------------
    # Separate features and target
    # ------------------------------------------------------------------
    target = df[target_column]
    features = df.drop(columns=[target_column])

    n_samples = len(features)

    # ------------------------------------------------------------------
    # Remove columns with too many missing values
    # ------------------------------------------------------------------
    missing_percentage = features.isna().mean() * 100.0

    # ------------------------------------------------------------------
    # Remove columns with too many zero values
    # NaNs are not counted as zeros.
    # ------------------------------------------------------------------
    zero_percentage = features.eq(0).mean() * 100.0

    keep_columns = (
        (missing_percentage <= drop_threshold_percent)
        & (zero_percentage <= drop_threshold_percent)
    )

    features = features.loc[:, keep_columns]

    # ------------------------------------------------------------------
    # Z-score normalization
    #
    # Constant columns (std = 0) become all zeros.
    # ------------------------------------------------------------------
    means = features.mean()

    stds = features.std(ddof=0)

    stds = stds.replace(0, 1)

    features = (features - means) / stds

    # Replace possible NaNs generated during normalization
    features = features.fillna(0.0)

    # ------------------------------------------------------------------
    # Recombine features and target
    # ------------------------------------------------------------------
    processed = features.copy()
    processed[target_column] = target.values

    # ------------------------------------------------------------------
    # Deterministic split (NO SHUFFLING)
    # ------------------------------------------------------------------
    test_fraction = test_size_percent / 100.0

    M = int(n_samples * (1.0 - test_fraction))

    training_df = processed.iloc[:M].reset_index(drop=True)
    test_df = processed.iloc[M:].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    training_path = output_dir / "training.csv"
    test_path = output_dir / "test.csv"

    training_df.to_csv(training_path, index=False)
    test_df.to_csv(test_path, index=False)

    return training_df, test_df


if __name__ == "__main__":
    """
    Example execution using the development dataset.

    Repository structure:

    project_root/
    ├── data/
    │   ├── trial_dataset_ISW.csv
    │   ├── training.csv
    │   └── test.csv
    └── src/
        └── qubo_project/
            └── preprocessing.py
    """

    preprocess_dataset(
        dataset_path="data/trial_dataset_ISW.csv",
        target_column="target",          # Replace with the actual target column name
        drop_threshold_percent=95.0,
        test_size_percent=20.0,
        output_dir="data",
    )

    print("Preprocessing completed successfully.")
    print("Training dataset saved to: data/training.csv")
    print("Test dataset saved to: data/test.csv")