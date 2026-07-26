"""Utilities for loading tabular data, computing Spearman correlations, and
constructing a QUBO matrix for feature selection.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def build_qubo_matrix(
    rho_v: np.ndarray,
    rho_m: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Construct the dense QUBO matrix for feature selection.

    The QUBO formulation is:

        Q[j, j] = alpha * |rho(V, j)| - (1 - alpha) * |rho(j, j)|

        Q[j, k] = -(1 - alpha) * |rho(j, k)|    for j != k

    Parameters
    ----------
    rho_v
        Absolute Spearman correlation vector between each feature and the
        target. Shape: (n_features,).
    rho_m
        Absolute Spearman correlation matrix between feature columns.
        Shape: (n_features, n_features).
    alpha
        Trade-off parameter in the interval [0, 1]. Larger values favour
        relevance to the target, while smaller values penalise redundancy.

    Returns
    -------
    np.ndarray
        Dense QUBO matrix of shape (n_features, n_features).

    Raises
    ------
    ValueError
        If the input shapes are inconsistent or alpha is outside [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in the interval [0, 1].")

    rho_v = np.asarray(rho_v, dtype=np.float64)
    rho_m = np.asarray(rho_m, dtype=np.float64)

    if rho_m.ndim != 2 or rho_m.shape[0] != rho_m.shape[1]:
        raise ValueError("rho_m must be a square matrix.")

    if rho_v.ndim != 1:
        raise ValueError("rho_v must be a one-dimensional array.")

    if rho_m.shape[0] != rho_v.shape[0]:
        raise ValueError(
            "rho_v length must match the dimensions of rho_m."
        )

    beta = 1.0 - alpha

    # Off-diagonal entries.
    qubo = -beta * rho_m

    # Diagonal entries.
    diagonal = alpha * rho_v - beta * np.diag(rho_m)
    np.fill_diagonal(qubo, diagonal)

    return qubo


def load_and_correlate(
    csv_path: str,
    target_column: str
):
    """Load a dataset, compute Spearman correlations, and build the QUBO matrix.

    The function:

    1. Loads the CSV file.
    2. Separates feature columns from the target column.
    3. Computes absolute Spearman feature-feature correlations.
    4. Computes absolute Spearman feature-target correlations.
    5. Builds the dense QUBO matrix.
    6. Measures the execution time required to generate the QUBO matrix and computing the Spearman's correlations.

    Parameters
    ----------
    csv_path
        Path to the input CSV dataset.
    target_column
        Name of the target column.
    alpha
        Trade-off parameter used in the QUBO formulation.
    """
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    feature_df = df.drop(columns=[target_column])
    target = df[target_column]

    if feature_df.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")

    start_time = perf_counter()

    feature_names = feature_df.columns.tolist()

    # Feature-Feature Spearman Correlation
    rho_m = (
        feature_df.corr(method="spearman")
        .abs()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    # Feature-Target Spearman Correlation
    correlations = [
        abs(float(feature_df[col].corr(target, method="spearman")))
        for col in feature_df.columns
    ]
    rho_v = np.nan_to_num(np.array(correlations, dtype=np.float64), nan=0.0)

    creation_time = perf_counter() - start_time

    return feature_names, rho_v, rho_m, creation_time

def solve_qubo(
    Q: np.ndarray,
    seed: int = 42,
    max_steps: int = 1000,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
):
    """Solve a dense QUBO using Simulated Annealing.

    The objective is to maximise

        x.T @ Q @ x

    where x is a binary vector.

    Parameters
    ----------
    Q
        Dense square QUBO matrix of shape (n_features, n_features).
    seed
        Seed used to initialise the random number generator.
    max_steps
        Maximum number of annealing iterations.
    initial_temp
        Initial temperature.
    cooling_rate
        Multiplicative cooling factor applied after each iteration.
        Must satisfy 0 < cooling_rate < 1.

    Returns
    -------
    tuple[np.ndarray, float, float]
        Tuple containing:

        - Best binary solution vector.
        - Best objective value (x.T @ Q @ x).
        - Execution time in seconds.

    Raises
    ------
    ValueError
        If the QUBO matrix or optimisation parameters are invalid.
    """
    Q = np.asarray(Q, dtype=np.float64)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    if initial_temp <= 0.0:
        raise ValueError("initial_temp must be positive.")

    if not 0.0 < cooling_rate < 1.0:
        raise ValueError("cooling_rate must satisfy 0 < cooling_rate < 1.")

    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    start_time = perf_counter()

    # Initial random solution.
    current_x = rng.integers(0, 2, size=n, dtype=np.int8)
    current_value = float(current_x @ Q @ current_x)

    best_x = current_x.copy()
    best_value = current_value

    temperature = initial_temp

    # Symmetrize matrix once to simplify bit-flip math: Q_sym = Q + Q.T
    Q_sym = Q + Q.T

    for _ in range(max_steps):
        index = rng.integers(n)
        
        # Delta value calculation in O(N) instead of O(N^2)
        # If bit flips from 0 to 1, d_x = 1. If 1 to 0, d_x = -1.
        d_x = 1 - 2 * current_x[index] 
        
        # Change in objective: d_x * (Q_jj + sum_{j != index} Q_sym[index, j] * x_j)
        delta_val = d_x * (Q[index, index] + np.dot(Q_sym[index], current_x) - Q_sym[index, index] * current_x[index])
        delta_energy = -delta_val

        if delta_energy <= 0.0 or rng.random() < np.exp(-delta_energy / temperature):
            current_x[index] ^= 1
            current_value += delta_val

            if current_value > best_value:
                best_value = current_value
                best_x = current_x.copy()

        temperature *= cooling_rate
        if temperature < 1e-12:
            break

    execution_time = perf_counter() - start_time

    return best_x.astype(np.int8), best_value, execution_time

def search_optimal_alpha(
    rho_v: np.ndarray,
    rho_m: np.ndarray,
    target_k: int,
    allowance: int,
    max_computations: int = 100,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Search for an alpha value that yields the desired number of features.

    For each candidate alpha in the interval [0, 1], a QUBO matrix is
    constructed and solved using Simulated Annealing. The search terminates
    early if the number of selected features lies within the specified
    allowance of the target cardinality.

    If no feasible solution is found, the candidate whose number of selected
    features is closest to ``target_k`` is returned.

    Parameters
    ----------
    rho_v
        Absolute Spearman correlation vector between the target and features.
    rho_m
        Absolute Spearman correlation matrix between features.
    target_k
        Desired number of selected features.
    allowance
        Acceptable deviation from ``target_k``.
    max_computations
        Maximum number of alpha values to evaluate.
    seed
        Random seed passed to the QUBO solver.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        A tuple containing

        - The selected candidate.
        - A list of all evaluated candidates sorted by alpha.

    Raises
    ------
    ValueError
        If the input parameters are invalid.
    """
    if target_k < 0:
        raise ValueError("target_k must be non-negative.")

    if allowance < 0:
        raise ValueError("allowance must be non-negative.")

    if max_computations <= 0:
        raise ValueError("max_computations must be positive.")

    low_a, high_a = 0.0, 1.0

    history: list[dict[str, Any]] = []

    best_candidate: dict[str, Any] | None = None
    best_distance = float("inf")

    lower = target_k - allowance
    upper = target_k + allowance

    # Evaluate boundaries first or run adaptive search
    for i in range(max_computations):
        # Sample mid-point or linear step
        alpha = (low_a + high_a) / 2.0 if i > 1 else (0.0 if i == 0 else 1.0)
        Q = build_qubo_matrix(rho_v, rho_m, float(alpha))

        solution, cost_value, opt_time = solve_qubo(Q, seed=seed)
        n_selected = int(solution.sum())

        record = {
            "alpha": float(alpha),
            "solution": solution,
            "qubo_matrix": Q,
            "cost_value": float(cost_value),
            "optimization_time": opt_time,
            "n_features": n_selected,
        }

        history.append(record)

        # Early exit if within the requested bounds.
        if lower <= n_selected <= upper:
            history.sort(key=lambda item: item["alpha"])
            return record, history

        # Binary search directional adjustments
        if n_selected < lower:
            low_a = alpha  # need higher alpha to select more features
        else:
            high_a = alpha  # need lower alpha to reduce features

        distance = abs(n_selected - target_k)

        if best_candidate is None or distance < best_distance:
            best_candidate = record
            best_distance = distance

    history.sort(key=lambda item: item["alpha"])
    return best_candidate, history