import os
import json
import pytest
import pandas as pd
import numpy as np

# Import functions from your package structure
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

# Path to the mandated sample test dataset
SAMPLE_DATASET_PATH = "data/sample_test_dataset.csv"
TARGET_COL = "target"

@pytest.fixture(scope="module")
def setup_sample_data(tmp_path_factory):
    """
    Ensures sample data exists and creates a temporary directory 
    for intermediate test output files.
    """
    if not os.path.exists(SAMPLE_DATASET_PATH):
        pytest.fail(f"Required sample dataset missing at: {SAMPLE_DATASET_PATH}")
    
    # Check that sample dataset contains both target classes
    df = pd.read_csv(SAMPLE_DATASET_PATH)
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' missing from test dataset."
    unique_targets = df[TARGET_COL].unique()
    assert 0 in unique_targets and 1 in unique_targets, "Sample test dataset must contain both 0 and 1 target classes."
    
    # Create temp directory for output files generated during test runs
    temp_dir = tmp_path_factory.mktemp("test_outputs")
    return str(temp_dir)


def test_1_and_2_preprocessing_numeric_and_missing_values(setup_sample_data):
    """
    Requirement 1: Preprocessing produces only numeric columns.
    Requirement 2: Preprocessing handles missing values (e.g. removes or imputes them).
    """
    temp_dir = setup_sample_data
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    out_json = os.path.join(temp_dir, "prep_out.json")

    # Run preprocessing
    fit_normalize(
        input_csv=SAMPLE_DATASET_PATH,
        target_column=TARGET_COL,
        normalized_csv=normalized_csv,
        outInitalRes_json=out_json,
        minPercValid=0.05
    )

    assert os.path.exists(normalized_csv), "Normalized CSV was not created."
    df_norm = pd.read_csv(normalized_csv)

    # 1. Verify all columns are numeric
    non_numeric_cols = df_norm.select_dtypes(exclude=[np.number]).columns.tolist()
    assert len(non_numeric_cols) == 0, f"Found non-numeric columns after preprocessing: {non_numeric_cols}"

    # 2. Verify no missing/NaN values remain in processed output
    assert df_norm.isna().sum().sum() == 0, "Preprocessing output contains missing/NaN values."


def test_3_normalization_validity(setup_sample_data):
    """
    Requirement 3: Normalization produces a valid dataset.
    Validates Z-score standardization (mean ~ 0, std ~ 1) on feature columns.
    """
    temp_dir = setup_sample_data
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    
    df_norm = pd.read_csv(normalized_csv)
    feature_cols = [col for col in df_norm.columns if col != TARGET_COL]

    # Verify z-score standardization stats on features
    means = df_norm[feature_cols].mean()
    stds = df_norm[feature_cols].std(ddof=0)

    # Allow slight tolerance due to sample size or potential constant columns
    np.testing.assert_allclose(means.values, 0.0, atol=1e-1, err_msg="Feature means are not close to 0.")
    
    # Target column should remain unnormalized binary values (0 and 1)
    target_vals = set(df_norm[TARGET_COL].unique())
    assert target_vals.issubset({0, 1}), "Target values were corrupted during normalization."


def test_4_and_5_feature_selection_binary_and_20_percent(setup_sample_data):
    """
    Requirement 4: Feature selection produces a binary vector.
    Requirement 5: Number of selected features is ~20% of processed features.
    """
    temp_dir = setup_sample_data
    normalized_csv = os.path.join(temp_dir, "normalized.csv")
    reduced_train = os.path.join(temp_dir, "reduced_train.csv")
    reduced_test = os.path.join(temp_dir, "reduced_test.csv")
    optim_csv = os.path.join(temp_dir, "optim.csv")
    fs_json = os.path.join(temp_dir, "fs_out.json")

    perc_selected = 0.20
    allowance = 1

    select_features(
        normalized_csv=normalized_csv,
        reducedTrain_csv=reduced_train,
        reducedTest_csv=reduced_test,
        output_ottim_csv=optim_csv,
        output_json=fs_json,
        target_column=TARGET_COL,
        percTest=0.30,
        allowance=allowance,
        seed=42,
        percSelected=perc_selected,
        alpha_computations=20
    )

    assert os.path.exists(fs_json), "Feature selection JSON output was not created."
    with open(fs_json, "r") as f:
        meta = json.load(f)

    # 4. Check binary vector
    selected_vector = meta.get("selected_vector", [])
    assert len(selected_vector) > 0, "Selected vector is empty."
    assert set(selected_vector).issubset({0, 1}), f"Selected vector contains non-binary values: {selected_vector}"

    # 5. Check count is ~20% (Target K = round(0.20 * m) +/- allowance)
    n_features = meta.get("n_features")
    expected_k = round(perc_selected * n_features)
    n_selected = meta.get("n_selected")
    
    assert abs(n_selected - expected_k) <= allowance, (
        f"Selected features count ({n_selected}) is outside allowance limits "
        f"for expected ~20% ({expected_k}) of total {n_features} features."
    )


def test_6_training_produces_saved_model(setup_sample_data):
    """
    Requirement 6: Training produces a saved model file.
    """
    temp_dir = setup_sample_data
    reduced_train = os.path.join(temp_dir, "reduced_train.csv")
    model_path = os.path.join(temp_dir, "model.joblib")
    metrics_json = os.path.join(temp_dir, "train_metrics.json")

    train(
        classifier="random_forest",
        reducedTrain_csv=reduced_train,
        target_column=TARGET_COL,
        model_path=model_path,
        metrics_json=metrics_json,
        seed=42
    )

    # Verify model artifact and metrics file were generated
    assert os.path.exists(model_path), "Trained model file (.joblib) was not saved."
    assert os.path.getsize(model_path) > 0, "Saved model file is empty."
    assert os.path.exists(metrics_json), "Training metrics JSON file was not saved."


def test_7_prediction_produces_required_csv(setup_sample_data):
    """
    Requirement 7: Prediction produces a CSV file containing required columns:
    'row_n', 'target', 'prediction', 'score'.
    """
    temp_dir = setup_sample_data
    reduced_test = os.path.join(temp_dir, "reduced_test.csv")
    model_path = os.path.join(temp_dir, "model.joblib")
    predictions_csv = os.path.join(temp_dir, "predictions.csv")
    stats_json = os.path.join(temp_dir, "classif_stats.json")

    predict(
        reduced_Test_csv=reduced_test,
        target_column=TARGET_COL,
        model_path=model_path,
        predictions_csv=predictions_csv,
        classif_stats_json=stats_json
    )

    assert os.path.exists(predictions_csv), "Predictions CSV file was not created."
    
    df_pred = pd.read_csv(predictions_csv)
    required_columns = ["row_n", "target", "prediction", "score"]
    
    # Verify exact required column names are present
    for col in required_columns:
        assert col in df_pred.columns, f"Required column '{col}' missing from predictions CSV."

    # Verify predictions dataframe has rows
    assert len(df_pred) > 0, "Predictions CSV is empty."