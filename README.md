# QUBO Binary Classification with Feature Reduction

## Project Overview

This project implements an end-to-end Python software system for binary classification on tabular datasets, featuring a Quadratic Unconstrained Binary Optimization (QUBO) mathematical framework for feature selection. The primary objective is to handle high-dimensional binary classification tasks (such as credit risk assessment) by identifying a minimal, optimal subset of informative features while discarding redundant or non-informative columns.

The pipeline comprises four sequential phases:
1. Data Preprocessing and Standardization: Cleaning low-variance or missing data and scaling remaining feature values using Z-score normalization.
2. QUBO-Based Feature Selection: Formulating feature relevance, redundancy minimization, and target subset cardinality as a quadratic pseudo-Boolean matrix optimization problem, solved via a greedy hill-climbing search across varying weighting factors.
3. Machine Learning Model Training: Training binary classifiers (Random Forest, XGBoost, or K-Nearest Neighbors) on the selected feature subset.
4. Model Evaluation and Inference: Generating target predictions, confidence scores, and comprehensive diagnostic metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix) on the holdout testing partition.

The software includes both a Command Line Interface (CLI) compliant with automated evaluation harnesses and an interactive Streamlit Graphical User Interface (GUI) for visual analysis, parameter configuration, and artifact downloading.

---

## Directory Structure and File Mapping

The repository structure follows a strict modular organization to segregate source code, datasets, execution outputs, and project documentation. Below is the file mapping describing where artifacts reside and where output files are generated:

```
isw-qubo-classification-GXX/
├── README.md                          # Comprehensive system documentation
├── requirements.txt                   # Third-party Python dependencies
├── group_info.yaml                    # Group metadata and submission info
├── data/                              # Input datasets and intermediate normalized data
│   ├── input_dataset.csv              # Initial raw dataset for feature reduction and training
│   ├── sample_test_dataset.csv        # Minimal test dataset for automated unit tests
│   └── normalized.csv                 # Z-score normalized dataset generated during preprocessing
├── src/                               # Application source code root
│   └── qubo_project/                  # Core python package
│       ├── __init__.py                # Package initialization file
│       ├── preprocessing.py           # Preprocessing and dataset splitting functions
│       ├── feature_selection.py       # QUBO formulation, solver, and search loops
│       ├── model.py                   # Classifier training, persistence, and prediction
│       ├── gui.py                     # Streamlit multi-page web application
│       └── gui_utils.py               # Streamlit state management and sidebar utilities
├── outputs/                           # Output directory for runtime generated artifacts
│   ├── preprocessing_result.json      # Metadata from dataset cleaning and standardization
│   ├── train.csv                      # Feature-reduced training partition
│   ├── test.csv                       # Feature-reduced test partition
│   ├── optimizations.csv              # Alpha sweep history from QUBO optimizations
│   ├── feature_selection_result.json  # QUBO execution summary and selected feature indices
│   ├── model.joblib                   # Serialized trained classifier model artifact
│   ├── training_metrics.json          # Classifier hyperparameter and execution metrics
│   ├── predictions.csv                # Instance-level predictions and confidence scores
│   └── classification_stats.json      # Evaluation metrics and confusion matrix
├── tests/                             # Automated test suite
│   └── test_pipeline.py               # Pytest automated test scripts
├── reports/                           # Academic and project reports
│   └── project_report.yaml            # Final structured project report
└── llm_usage/                         # Documentation of LLM prompts and generated code
    ├── LOG-GXX-01.md                  # Log of interactions with AI tools
    └── files/                         # Prompts and intermediate snippets
```

---

## Technical Specifications and Mathematical Background

### 1. Preprocessing and Normalization

Raw tabular datasets often contain columns with missing values or zero variance. The preprocessing module filters out features whose fraction of valid non-zero values falls below a user-defined threshold (`minPercValid`).

For remaining features, Z-score normalization transforms each column $X_j$ to zero mean and unit variance:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

where $\mu_j$ is the sample mean of feature $j$, and $\sigma_j$ is the sample standard deviation (with zero standard deviation replaced by 1 to avoid division by zero). The target column is isolated during transformation and re-attached without modification.

### 2. QUBO Matrix Formulation

Feature selection balances two competing objectives: maximizing feature-target correlation (relevance) and minimizing pairwise inter-feature correlation (redundancy), while enforcing a constraint on the number of selected features $K$.

Let $x \in \{0, 1\}^n$ be a binary decision vector, where $x_i = 1$ indicates that feature $i$ is selected.

Let $\rho_v \in [0, 1]^n$ be the vector of absolute Spearman rank correlations between each feature and the binary target $y$.
Let $\rho_m \in [0, 1]^{n \times n}$ be the matrix of absolute pairwise Spearman rank correlations between features.

The objective function is parameterized by $\alpha \in [0, 1]$ and penalty parameter $\gamma > 0$:

$$E(x) = - \alpha \sum_{i=1}^n \rho_{v, i} x_i + (1 - \alpha) \sum_{i=1}^n \sum_{j \neq i}^n \rho_{m, ij} x_i x_j + \gamma \left( \sum_{i=1}^n x_i - K \right)^2$$

Expanding the soft cardinality penalty term $\gamma (\sum x_i - K)^2$:

$$\gamma \left( \sum_{i=1}^n x_i - K \right)^2 = \gamma \left( \sum_{i=1}^n \sum_{j=1}^n x_i x_j - 2K \sum_{i=1}^n x_i + K^2 \right)$$

Because $x_i^2 = x_i$ for binary variables, the diagonal entries of the overall QUBO matrix $Q$ combine relevance, self-redundancy, and the linear contribution of the cardinality constraint:

$$Q_{ii} = - \alpha \rho_{v, i} + \gamma (1 - 2K)$$

The off-diagonal entries $Q_{ij}$ ($i \neq j$) represent redundancy and quadratic cardinality penalties:

$$Q_{ij} = (1 - \alpha) \rho_{m, ij} + \gamma$$

Minimizing $x^T Q x$ yields a binary vector $x^*$ balancing relevance and redundancy. By sweeping $\alpha$ from $0.0$ to $1.0$, the solver identifies an optimal $\alpha$ that yields exactly $K \pm \text{allowance}$ selected features.

---

## Detailed Code and Function API Reference

### Module: `src/qubo_project/preprocessing.py`

#### `fit_normalize(input_csv, target_column, normalized_csv, outInitalRes_json, minPercValid=0.05, output_data_dir="data", output_json_dir="outputs")`
- **Purpose**: Reads a raw CSV dataset, verifies the existence of the target column, filters out columns with excessive missing/zero entries, standardizes valid numeric features using Z-score normalization, and writes output files.
- **Parameters**:
  - `input_csv` (str | Path): Path or filename of the raw input CSV file.
  - `target_column` (str): Header name of the binary classification target column.
  - `normalized_csv` (str): Filename for saving the normalized dataset.
  - `outInitalRes_json` (str): Filename for saving execution statistics in JSON format.
  - `minPercValid` (float): Minimum proportion of non-zero/valid entries required for a feature column to be retained (default: 0.05).
  - `output_data_dir` (str | Path): Directory path for saving processed CSV files (default: "data").
  - `output_json_dir` (str | Path): Directory path for saving metric JSON files (default: "outputs").
- **Returns**: A tuple `(processed_df, json_output_dict)` containing the normalized DataFrame and summary statistics dictionary.
- **Exceptions Raised**: `FileNotFoundError` if the input file does not exist; `ValueError` if the specified target column is missing.

#### `divide_csvs(reduced_df, train_size_percent, output_dir="outputs", train_csv_name="train.csv", test_csv_name="test.csv")`
- **Purpose**: Splits a DataFrame sequentially (without shuffling, taking the first $M$ rows) into training and testing partitions, saving both as CSV files.
- **Parameters**:
  - `reduced_df` (pd.DataFrame): DataFrame containing selected features and target column.
  - `train_size_percent` (float): Fraction of rows assigned to the training set (between 0.0 and 1.0).
  - `output_dir` (str | Path): Target folder for exported CSV partitions.
  - `train_csv_name` (str): Export filename for training partition.
  - `test_csv_name` (str): Export filename for test partition.
- **Returns**: Tuple `(train_df, test_df)`.

---

### Module: `src/qubo_project/feature_selection.py`

#### `validate_inputs(normalized_csv, output_ottim_csv, output_json, percTest, percSelected, allowance, alpha_computations)`
- **Purpose**: Validates system arguments and checks file paths across candidate directories (`.`, `data/`, `outputs/`).
- **Returns**: Validated `Path` object pointing to the input dataset.
- **Exceptions Raised**: `FileNotFoundError` or `ValueError` if boundary constraints are violated.

#### `load_and_correlate(dataset_path, target_column)`
- **Purpose**: Loads a dataset, checks numeric data integrity, and computes Spearman correlation matrices.
- **Returns**: Tuple `(load_time, creation_time, df, df_ntc, target, feature_names, rho_m, rho_v)`.

#### `clean_constant_columns(df_ntc)`
- **Purpose**: Drops invariant feature columns (columns with $\le 1$ unique value) to prevent zero-variance NaN issues in correlation calculations.

#### `build_qubo_matrix(rho_v, rho_m, target_k, alpha, gamma=2.0)`
- **Purpose**: Constructs the $n \times n$ QUBO cost matrix $Q$ combining linear target correlation benefits, quadratic feature-feature redundancy penalties, and soft cardinality constraints.
- **Returns**: Square symmetric/upper-triangular `np.ndarray` of type `float64`.

#### `solve_qubo(Q, seed=42, max_steps=10000)`
- **Purpose**: Solves the unconstrained QUBO problem $\min_x x^T Q x$ using a greedy hill-climbing local search algorithm. Calculates bit flip energy deltas in $O(N)$ time using symmetric matrix precomputation.
- **Returns**: Tuple `(best_x, best_cost, execution_time)`.

#### `search_optimal_alpha(rho_v, rho_m, target_k, allowance=1, max_computations=100, seed=42)`
- **Purpose**: Performs a grid search over $\alpha \in [0, 1]$ across `max_computations` steps to find an optimal $\alpha$ value that yields a feature subset size within $K \pm \text{allowance}$.
- **Returns**: Tuple `(best_record, results_list)`.

#### `write_outputs(history_list, metrics_dict, csv_out, json_out)`
- **Purpose**: Serializes optimization step histories to CSV and pipeline execution metadata to JSON inside the `outputs/` directory.

#### `select_features(...)`
- **Purpose**: High-level execution entry point orchestrating validation, correlation analysis, QUBO matrix building, alpha search, dataset splitting via `divide_csvs`, and output writing.

---

### Module: `src/qubo_project/model.py`

#### `_get_classifier(name, seed, params=None)`
- **Purpose**: Instantiates and returns a Scikit-Learn or XGBoost classification model according to the provided algorithm name (`random_forest`, `xgboost`, or `knn`) and hyperparameter dictionary.

#### `train(classifier, reducedTrain_csv, target_column, model_path, metrics_json, seed=42, **clf_params)`
- **Purpose**: Trains the selected binary classifier on the reduced feature training set, saves the serialized model using `joblib`, and outputs performance metadata to JSON.
- **Returns**: Trained classifier model object.

#### `predict(reduced_Test_csv, target_column, model_path, predictions_csv, classif_stats_json)`
- **Purpose**: Loads a serialized model, performs prediction on a reduced test set, computes predicted classes and probabilities, exports instance-level results to CSV, and writes summary metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix) to JSON.
- **Returns**: `pd.DataFrame` containing predictions and scores.

---

### Module: `src/qubo_project/gui_utils.py`

#### `initialize_session_state()`
- **Purpose**: Initializes default configuration values and operational flags in Streamlit `st.session_state` to maintain UI state persistence across page rerenders.

#### `pipeline_status()`
- **Purpose**: Renders visual indicator icons in the Streamlit sidebar demonstrating completion status across Preprocessing, Feature Selection, Training, and Prediction steps.

---

## Command Line Interface (CLI) Execution Guide

All execution modules support standalone CLI execution via terminal commands.

### Step 1: Preprocessing

Run dataset cleaning and Z-score standardization:

```bash
python src/qubo_project/preprocessing.py   --input data/input_dataset.csv   --target target   --out-data normalized.csv   --out-json preprocessing_result.json   --min-perc-valid 0.05
```

Generated outputs:
- `data/normalized.csv`
- `outputs/preprocessing_result.json`

### Step 2: QUBO Feature Selection

Execute QUBO feature selection to find optimal alpha and partition datasets:

```bash
python src/qubo_project/feature_selection.py   --in-normalized normalized.csv   --out-train train.csv   --out-test test.csv   --out-optimizations optimizations.csv   --out-json feature_selection_result.json   --target target   --perc-selected 0.20   --allowance 1   --perc-test 0.30   --seed 42   --alpha-computations 100
```

Generated outputs:
- `outputs/train.csv`
- `outputs/test.csv`
- `outputs/optimizations.csv`
- `outputs/feature_selection_result.json`

### Step 3: Classifier Model Training

Train a classifier (e.g., Random Forest, XGBoost, or KNN) on the reduced dataset:

```bash
python src/qubo_project/model.py train   --classifier xgboost   --in-reduced train.csv   --target target   --out-model model.joblib   --out-metrics training_metrics.json   --seed 42   --n-estimators 100   --learning-rate 0.1   --max-depth 6
```

Generated outputs:
- `outputs/model.joblib`
- `outputs/training_metrics.json`

### Step 4: Model Prediction and Evaluation

Run prediction on the holdout testing partition:

```bash
python src/qubo_project/model.py predict   --input-testset test.csv   --target target   --model model.joblib   --out-predictions predictions.csv   --out-stats classification_stats.json
```

Generated outputs:
- `outputs/predictions.csv`
- `outputs/classification_stats.json`

---

## Graphical User Interface (GUI) Guide

The system includes an interactive web interface constructed with Streamlit.

### How to Launch the GUI

Execute the following command from the project root directory:

```bash
streamlit run src/qubo_project/gui.py
```

Upon execution, Streamlit starts a local web server (typically accessible at `http://localhost:8501`).

### GUI Navigation Pages and Workflow

1. Home: Overview of the application pipeline and setup summary.
2. Dataset / File Preview: Allows interactive loading, inspecting, rendering, and downloading of CSV and JSON data files.
3. Preprocessing Page:
   - Input dataset path selection and target column specification.
   - Interactive configuration of the minimum valid data percentage slider (`minPercValid`).
   - Execution button for `fit_normalize`.
   - Rendered table view and download button for `normalized.csv` and metadata.
4. Feature Selection Page:
   - Parameters for target feature ratio, allowance tolerance, seed, and alpha sweep steps.
   - Execution button triggering `select_features`.
   - Display of optimal alpha, selected feature list, training/testing partition tables, and download controls.
5. Model Training Page:
   - Algorithm selection (`random_forest`, `xgboost`, `knn`) and target selection.
   - Dynamic hyperparameter controls (tree depth, estimator counts, learning rates, neighbor counts).
   - Execution button for `train` with model artifact download (`.joblib`).
6. Prediction Page:
   - Input field configuration for test set path and model binary file path.
   - Execution button triggering `predict`.
   - Display of evaluation metrics (Accuracy, ROC-AUC, Precision, Recall, F1-scores, Confusion Matrix) and instance predictions CSV download.

---

## Automated Testing Suite

Automated system tests are managed via `pytest`.

To run all automated tests, execute:

```bash
pytest
```

The test suite verifies:
1. Pure numeric feature output following preprocessing.
2. Handling of missing or NaN values during dataset standardization.
3. Correct Z-score scaling properties on valid columns.
4. Generation of binary vectors ($x_i \in \{0, 1\}$) by the QUBO solver.
5. Selected feature count conformance to target percentage constraints ($K \pm \text{allowance}$).
6. Serialization and persistence of model binary artifacts (`.joblib`).
7. Structural correctness of generated prediction CSV and evaluation metrics JSON files.
