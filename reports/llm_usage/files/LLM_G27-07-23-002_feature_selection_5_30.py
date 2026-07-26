# File: src/qubo_project/feature_selection.py

"""Utilities for loading tabular data, computing Spearman correlations, and
constructing a QUBO matrix for feature selection.
"""

from __future__ import annotations

from preprocessing import divide_csvs

from pathlib import Path
from time import perf_counter
from typing import Any

import json
import numpy as np
import pandas as pd
import argparse

def validate_inputs(
    normalized_csv: str,
    output_ottim_csv: str,
    output_json: str,
    percTest: float,
    percSelected: float,
    allowance: int,
    alpha_computations: int
):
    #------------------------------------------
    #- Check of all parameters that can fail. -
    #------------------------------------------
    
    #normalized_csv
    candidate_paths = [
        Path("data") / normalized_csv,
        Path(output_ottim_csv) / normalized_csv,
        Path(output_json) / normalized_csv,
    ]

    dataset_path = next((p for p in candidate_paths if p.exists()), None)

    if dataset_path is None:
        raise FileNotFoundError(
            f"Input normalized dataset not found: {normalized_csv}"
        )

    # Target-column we hope it exists.

    if not 0.0 < percTest < 1.0:
        raise ValueError("percTest must be strictly between 0 and 1.")

    if not 0.0 < percSelected <= 1.0:
        raise ValueError("percSelected must be > 0 and <= 1.")

    if allowance < 0:
        raise ValueError("allowance must be >= 0.")

    if alpha_computations < 1:
        raise ValueError("alpha_computations must be >= 1.")

    return dataset_path

def load_and_correlate(
    dataset_path: str,
    target_column: str
):
    load_time = perf_counter()
    
    df = pd.read_csv(dataset_path)
    
    load_time = perf_counter() - load_time

    #Error Here if it doesn't exists, we didn't wait long
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in input dataset.")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    feature_names = [column for column in df.columns if column != target_column]

    if len(feature_names) == 0:
        raise ValueError("No feature columns found.")

    df_ntc = df.drop(columns=[target_column])
    target = df[target_column]

    # Ensure numeric values.
    if not all(pd.api.types.is_numeric_dtype(df_ntc[column]) for column in df_ntc.columns):
        raise ValueError("You did something wrong in the normalization.")

    # Check for NaN/Inf because the input is expected to be
    # already preprocessed.
    if not np.isfinite(df_ntc.to_numpy(dtype=np.float64)).all():
        raise ValueError(
            "Input feature data contains NaN or infinite values. "
            "The function expects an already preprocessed "
            "normalized dataset."
        )

    if not np.isfinite(
        target.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Target data contains NaN or infinite values."
        )
    
    creation_time = perf_counter()

    rho_m = (df_ntc.corr(method="spearman").abs().fillna(0.0).to_numpy(dtype=np.float64))

    # Feature-Target Spearman Correlation
    correlations = [
        abs(float(df_ntc[col].corr(target, method="spearman")))
            for col in df_ntc.columns
        ]
    
    rho_v = np.nan_to_num(np.array(correlations, dtype=np.float64), nan=0.0)

    creation_time = perf_counter() - creation_time

    return load_time, creation_time, df, df_ntc, target, feature_names, rho_m, rho_v

def build_qubo_matrix(
    rho_v: np.ndarray,
    rho_m: np.ndarray,
    alpha: float,
):
    rho_v = np.asarray(rho_v, dtype=np.float64)
    rho_m = np.asarray(rho_m, dtype=np.float64)

    beta = 1.0 - alpha

    # Off-diagonal entries.
    qubo = -beta * rho_m

    # Diagonal entries.
    diagonal = alpha * rho_v - beta * np.diag(rho_m)
    np.fill_diagonal(qubo, diagonal)

    return qubo

def solve_qubo(
    Q: np.ndarray,
    seed: int = 42,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, float, float]:
    """
    Solves an unconstrained binary QUBO (minimizing x.T @ Q @ x) using
    a direct local search / greedy hill-climbing approach without 
    cooling schedules or temperature mechanics.
    """
    Q = np.asarray(Q, dtype=np.float64)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q matrix must be square.")

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    start_time = perf_counter()
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    # 1. Initialize random solution
    x = rng.integers(0, 2, size=n, dtype=np.int8)
    current_cost = float(x @ Q @ x)

    best_x = x.copy()
    best_cost = current_cost

    # 2. Precompute symmetric matrix Q_sym = Q + Q.T for O(N) delta computation
    Q_sym = Q + Q.T

    # 3. Local Search Loop (Greedy / Hill-Climbing)
    for _ in range(max_steps):
        i = rng.integers(0, n)

        # Direction of flip: +1 if bit 0->1, -1 if bit 1->0
        direction = 1 if x[i] == 0 else -1

        # Calculate exact change in cost from flipping bit i:
        # delta = direction * (Q_sym[i] @ x) - direction * Q[i, i]
        delta = direction * (Q_sym[i] @ x) - direction * Q[i, i]

        # Greedy acceptance: strictly accept non-increasing moves
        if delta <= 0.0:
            x[i] = 1 - x[i]
            current_cost += delta

            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()

    # Recompute best cost directly at the end to eliminate floating-point drift
    best_cost = float(best_x @ Q @ best_x)
    execution_time = perf_counter() - start_time

    return best_x, best_cost, execution_time

def search_optimal_alpha(
    rho_v: np.ndarray,
    rho_m: np.ndarray,
    target_k: int,
    allowance: int = 1,
    max_computations: int = 100,
    seed: int = 42,
):
    alpha_steps = np.linspace(0.0, 1.0, max_computations)
    results = []
    best_record = None

    for idx, alpha in enumerate(alpha_steps):
        # 1. Build QUBO matrix for current alpha
        Q = build_qubo_matrix(rho_v, rho_m, alpha)

        # 2. Solve QUBO using local search
        solution, cost_value, opt_time = solve_qubo(
            Q,
            seed=seed + idx,
            max_steps=10_000,
        )

        n_selected = int(np.sum(solution))
        k_diff = abs(n_selected - target_k)
        is_valid = k_diff <= allowance

        record = {
            "alpha": float(alpha),
            "solution": solution,
            "cost_value": cost_value,
            "n_selected": n_selected,
            "k_diff": k_diff,
            "is_valid": is_valid,
            "optimisation_time": opt_time,
        }
        results.append(record)

        if is_valid:
            if best_record is None or k_diff < best_record["k_diff"]:
                best_record = record

    if best_record is None:
        # Fallback to closest solution if no solution satisfied allowance
        best_record = min(results, key=lambda r: r["k_diff"])

    return best_record, results

def write_outputs(
    history_list: list[dict],
    metrics_dict: dict,
    csv_out: str | Path,
    json_out: str | Path,
):
    csv_path = "outputs"/Path(csv_out)
    json_path = "outputs"/Path(json_out)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    csv_records = []
    for record in history_list:
        rec_copy = record.copy()
        if isinstance(rec_copy.get("solution"), np.ndarray):
            rec_copy["solution"] = rec_copy["solution"].tolist()
        rec_copy["solution"] = str(rec_copy["solution"])
        csv_records.append(rec_copy)

    history_df = pd.DataFrame(csv_records)
    history_df.to_csv(csv_path, index=False)


    formatted_metrics = {}
    for k, v in metrics_dict.items():
        if isinstance(v, np.ndarray):
            formatted_metrics[k] = v.tolist()
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            formatted_metrics[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            formatted_metrics[k] = float(v)
        else:
            formatted_metrics[k] = v

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(formatted_metrics, f, indent=4)

def select_features(
    normalized_csv: str, # Input dataset name
    reducedTrain_csv: str, # Name of output training dataset with reduced feat.
    reducedTest_csv: str, # Name of output test dataset with reduced features
    output_ottim_csv: str, # Name of output optimization data varying alpha
    output_json: str, # Name of output statistics and data file
    target_column: str, # Column name of target
    percTest: float = 0.30, # % of test data with respect to the dataset size
    percSelected: float = 0.20, # percentage of features to select
    allowance: int = 1, # Allowance of features to select
    seed: int = 42, # Seed for random repeatibility
    alpha_computations: int = 100 # Max. n. of optimizations varying alpha
):
    """
    Complete feature-selection pipeline.
    """

    dataset_path = validate_inputs(
        normalized_csv = normalized_csv,
        output_ottim_csv = output_ottim_csv,
        output_json = output_json,
        percTest = percTest,
        percSelected = percSelected,
        allowance = allowance,
        alpha_computations = alpha_computations
    )

    total_start = perf_counter()

    load_time, creation_time, df, df_ntc, target, feature_names, rho_m, rho_v = \
    load_and_correlate(
        dataset_path,target_column
    )

    # CALCULATE NUMBER OF FEATURES

    target_k = int(round(len(feature_names) * percSelected))

    best_record, history_list = search_optimal_alpha(
        rho_v, rho_m, target_k, allowance, alpha_computations, seed
    )

    selected_vector = best_record["solution"]
    selected_feature_names = [name for name, bit in zip(feature_names, selected_vector) if bit == 1]

    reduced_df = df[selected_feature_names + [target_column]]

    train_df, test_df = divide_csvs(
        reduced_df, percTest, "outputs", reducedTrain_csv, reducedTest_csv
    )

    opt_times = [record["optimisation_time"] for record in history_list]

    mean_opt_time = float(np.mean(opt_times))
    std_opt_time = float(np.std(opt_times))
    
    total_end = perf_counter() - total_start

    metrics_dict = {
        "n_features": len(feature_names),
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": len(selected_feature_names),
        "alpha": best_record["alpha"],
        "selected_vector": selected_vector,
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(history_list),
        "percTest": percTest,
        "training_dataset_size": len(train_df),
        "test_dataset_size": len(test_df),
        "q_matrix_creation_time": creation_time,
        "mean_optimization_time": mean_opt_time,
        "std_dev_optimization_time": std_opt_time
    }

    write_outputs(
        history_list=history_list,
        metrics_dict=metrics_dict,
        csv_out=output_ottim_csv,
        json_out=output_json
    )



def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QUBO-based feature selection."
    )

    parser.add_argument(
        "--in-normalized",
        required=True,
        dest="normalized_csv",
    )

    parser.add_argument(
        "--out-train",
        required=True,
        dest="reducedTrain_csv",
    )

    parser.add_argument(
        "--out-test",
        required=True,
        dest="reducedTest_csv",
    )

    parser.add_argument(
        "--out-optimizations",
        required=True,
        dest="output_ottim_csv",
    )

    parser.add_argument(
        "--out-json",
        required=True,
        dest="output_json",
    )

    parser.add_argument(
        "--target",
        required=True,
        dest="target_column",
    )

    parser.add_argument(
        "--perc-selected",
        type=float,
        default=0.20,
        dest="percSelected",
    )

    parser.add_argument(
        "--allowance",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--perc-test",
        type=float,
        default=0.30,
        dest="percTest",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--alpha-computations",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    select_features(**vars(args))


if __name__ == "__main__":
    main()