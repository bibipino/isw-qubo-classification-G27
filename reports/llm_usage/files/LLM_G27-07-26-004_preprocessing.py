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
from time import perf_counter

import argparse
import os
import numpy as np
import pandas as pd
import json


def fit_normalize(
    input_csv: str | Path,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str,
    minPercValid: float = 0.05,
    output_data_dir: str | Path = "data",
    output_json_dir: str | Path = "outputs",
) -> Tuple[pd.DataFrame, dict]:

    output_data_dir = Path(output_data_dir)
    output_json_dir = Path(output_json_dir)
    
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_csv)

    if not input_path.exists():
        fallback_path = output_data_dir / input_path.name
        if fallback_path.exists():
            input_path = fallback_path

    print("Current working directory:", os.getcwd())
    print("Dataset exists:", input_path.exists())
    print("Dataset path:", input_path.resolve())

    if not input_path.exists():
        raise FileNotFoundError(f"Could not locate input CSV file at {input_path.resolve()}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    start_input_time = perf_counter()

    df = pd.read_csv(input_path)

    end_input_time = perf_counter() - start_input_time
    start_proc_time = perf_counter()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    # ------------------------------------------------------------------
    # Separate features and target
    # ------------------------------------------------------------------
    target = df[target_column]
    features = df.drop(columns=[target_column])

    n_features_start = features.shape[1]
    n_samples = len(features)

    invalid_ratio = (features.isna() | (features == 0)).mean()
    
    # Identify features to drop (strictly exceeding the threshold)
    drop_mask = invalid_ratio < minPercValid
    dropped_columns = features.columns[drop_mask].tolist()
    
    # Keep valid feature columns
    features = features.loc[:, ~drop_mask]

    # Fill any remaining NaNs with 0 before normalization if needed
    features = features.fillna(0.0)

    # ------------------------------------------------------------------
    # Z-score normalization
    # ------------------------------------------------------------------
    means = features.mean()
    stds = features.std(ddof=0).replace(0, 1)

    features = (features - means) / stds
    features = features.fillna(0.0)

    # ------------------------------------------------------------------
    # Recombine features and target
    # ------------------------------------------------------------------
    processed = features.copy()
    processed[target_column] = target.values

    n_features_end = processed.shape[1]
    end_proc_time = perf_counter() - start_proc_time

    normalized_csv = normalized_csv if normalized_csv.endswith(".csv") else f"{normalized_csv}.csv"
    processedPath = output_data_dir / normalized_csv

    processed.to_csv(processedPath, index=False)

    jsonOutput = {
        "n_input_features": n_features_start,
        "n_kept_features": n_features_end,
        "dataset_size": n_samples,
        "dataset_input_time": end_input_time,
        "dataset_processing_time": end_proc_time,
        "dropped_feature_names": dropped_columns
    }

    json_path = output_json_dir / outInitalRes_json
    with open(json_path, "w") as f:
        json.dump(jsonOutput, f, indent=4)

    return processed, jsonOutput


def divide_csvs(
    reduced_df: pd.DataFrame,
    train_size_percent: float,
    output_dir: str | Path = "outputs",
    train_csv_name: str = "train.csv",
    test_csv_name: str = "test.csv"
):
    start_input_time = perf_counter()

    if not (0.0 < train_size_percent < 1.0):
        raise ValueError("train_size_percent must be a float between 0 and 1.")

    total_rows = len(reduced_df)
    split_index = int(total_rows * train_size_percent)

    train_df = reduced_df.iloc[:split_index]
    test_df = reduced_df.iloc[split_index:]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file_name = train_csv_name if train_csv_name.endswith('.csv') else f"{train_csv_name}.csv"
    test_file_name = test_csv_name if test_csv_name.endswith('.csv') else f"{test_csv_name}.csv"

    train_path = output_path / train_file_name
    test_path = output_path / test_file_name

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    end_input_time = perf_counter() - start_input_time

    print(f"Split complete: {len(train_df)} train rows, {len(test_df)} test rows. Time: {end_input_time:.4f}s")

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run data preprocessing or CSV division pipeline for QUBO project."
    )

    # Flag to switch to division mode
    parser.add_argument(
        "-c", "--divide",
        action="store_true",
        help="If set, run divide_csvs() instead of fit_normalize()."
    )

    # Arguments for fit_normalize (with defaults so no typing is required)
    parser.add_argument(
        "--input",
        type=str,
        default="trial_dataset_ISW.csv",
        help="Path or filename of the input raw CSV dataset."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Name of the target variable column."
    )
    parser.add_argument(
        "--out-data",
        type=str,
        default="normalized.csv",
        help="Filename for the output normalized CSV data."
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="preprocessing_result.json",
        help="Filename for output execution metrics and statistics."
    )
    parser.add_argument(
        "--min-perc-valid",
        type=float,
        default=0.06,
        help="Minimum required percentage of valid non-zero data for a column."
    )
    parser.add_argument(
        "--out-data-dir",
        type=str,
        default="data",
        help="Directory for data files (defaults to 'data')."
    )
    parser.add_argument(
        "--out-json-dir",
        type=str,
        default="outputs",
        help="Directory for output files (defaults to 'outputs')."
    )

    args = parser.parse_args()

    if args.divide:
        divide_csvs(
            normalized_csv="normalized.csv",
            train_size_percent=0.3,
        )
    else:
        processed_df, summary_json = fit_normalize(
            input_csv=args.input,
            target_column=args.target,
            normalized_csv=args.out_data,
            outInitalRes_json=args.out_json,
            minPercValid=args.min_perc_valid,
            output_data_dir=args.out_data_dir,
            output_json_dir=args.out_json_dir
        )

        print(f"\n[SUCCESS] Preprocessing completed.")
        print(f"Data saved to: {Path(args.out_json_dir) / args.out_data}")
        print(f"JSON saved to: {Path(args.out_json_dir) / args.out_json}")