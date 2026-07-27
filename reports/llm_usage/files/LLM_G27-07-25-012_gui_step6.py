from pathlib import Path
import json

import pandas as pd
import streamlit as st

from preprocessing import fit_normalize
from feature_selection import select_features
from model import train
from model import predict

from gui_utils import (
    initialize_session_state,
    pipeline_status,
)

# ----------------------------------------------------
# Streamlit page configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="QUBO Binary Classification",
    page_icon="🧩",
    layout="wide",
)

initialize_session_state()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("QUBO Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Preprocessing",
        "Feature Selection",
        "Training",
        "Prediction",
        "Results",
    ],
)

pipeline_status()

# ----------------------------------------------------
# Pages
# ----------------------------------------------------

if page == "Home":

    st.title("QUBO Binary Classification with Feature Reduction")

    st.info(
        """
Welcome!

Use the sidebar to execute the complete pipeline.

The backend functions will be called directly from this GUI.
"""
    )

elif page == "Dataset":

    st.title("Dataset Selection")

    st.write("Select the dataset that will be used for the pipeline.")

    default_path = "data/input_dataset.csv"

    dataset_path = st.text_input(
        "Dataset path",
        value=st.session_state.dataset_path or default_path,
    )

    if st.button("Load Dataset"):

        try:

            path = Path(dataset_path)

            if not path.exists():
                st.error("Dataset not found.")
                st.stop()

            df = pd.read_csv(path)

            st.session_state.dataset_path = str(path)

            st.success("Dataset loaded successfully.")

            st.session_state.available_columns = list(df.columns)

        except Exception as e:
            st.error(str(e))

    if st.session_state.dataset_path:

        try:

            df = pd.read_csv(st.session_state.dataset_path)

            st.subheader("Dataset Information")

            col1, col2 = st.columns(2)

            col1.metric("Rows", len(df))
            col2.metric("Columns", len(df.columns))

            target = st.selectbox(
                "Target column",
                options=df.columns,
                index=list(df.columns).index(
                    st.session_state.target_column
                ) if st.session_state.target_column in df.columns else 0,
            )

            st.session_state.target_column = target

            st.subheader("Dataset Preview")

            st.dataframe(df.head(20), use_container_width=True)

        except Exception as e:
            st.error(str(e))

elif page == "Preprocessing":

    st.title("Preprocessing")

    if st.session_state.dataset_path is None:

        st.warning("Please load a dataset first.")

    else:

        st.write("Run preprocessing using the selected dataset.")

        min_valid = st.slider(
            "Minimum valid percentage",
            min_value=0.01,
            max_value=0.50,
            value=0.05,
            step=0.01,
        )

        if st.button("Run Preprocessing"):

            try:

                with st.spinner("Running preprocessing..."):

                    fit_normalize(
                        input_csv=st.session_state.dataset_path,
                        target_column=st.session_state.target_column,
                        normalized_csv="normalized.csv",
                        outInitalRes_json="preprocessing_result.json",
                        minPercValid=min_valid,
                    )

                st.session_state.preprocessing_done = True
                st.session_state.normalized_csv = "normalized.csv"
                st.session_state.preprocessing_json = "preprocessing_result.json"

                st.success("Preprocessing completed successfully.")

            except Exception as e:

                st.error(str(e))

        if st.session_state.preprocessing_done:

            json_path = Path("outputs") / "preprocessing_result.json"

            if json_path.exists():

                with open(json_path) as f:
                    summary = json.load(f)

                st.subheader("Preprocessing Summary")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Input Features",
                    summary["n_input_features"],
                )

                c2.metric(
                    "Kept Features",
                    summary["n_kept_features"],
                )

                c3.metric(
                    "Samples",
                    summary["dataset_size"],
                )

                st.subheader("Dropped Features")

                dropped = summary.get("dropped_feature_names", [])

                if dropped:
                    st.write(dropped)
                else:
                    st.success("No features were removed.")

                st.subheader("Execution Information")

                st.json(summary)

                normalized = Path("data") / "normalized.csv"

                if normalized.exists():

                    df = pd.read_csv(normalized)

                    st.subheader("Normalized Dataset Preview")

                    st.dataframe(
                        df.head(20),
                        use_container_width=True,
                    )

elif page == "Feature Selection":

    st.title("Feature Selection")

    if not st.session_state.preprocessing_done:

        st.warning("Run preprocessing first.")

    else:

        st.write("Run QUBO Feature Selection.")

        col1, col2 = st.columns(2)

        with col1:

            perc_selected = st.slider(
                "Selected Feature Percentage",
                0.05,
                0.50,
                0.20,
                0.01,
            )

            allowance = st.number_input(
                "Allowance",
                min_value=0,
                max_value=20,
                value=1,
            )

            perc_test = st.slider(
                "Test Percentage",
                0.10,
                0.50,
                0.30,
                0.05,
            )

        with col2:

            seed = st.number_input(
                "Seed",
                value=42,
                step=1,
            )

            alpha_computations = st.number_input(
                "Maximum Alpha Computations",
                min_value=1,
                max_value=500,
                value=100,
            )

        if st.button("Run Feature Selection"):

            try:

                with st.spinner("Running Feature Selection..."):

                    select_features(
                        normalized_csv="normalized.csv",
                        reducedTrain_csv="training_reduced.csv",
                        reducedTest_csv="test_reduced.csv",
                        output_ottim_csv="optimizations.csv",
                        output_json="feature_selection_result.json",
                        target_column=st.session_state.target_column,
                        percTest=perc_test,
                        percSelected=perc_selected,
                        allowance=allowance,
                        seed=seed,
                        alpha_computations=alpha_computations,
                    )

                st.session_state.feature_selection_done = True

                st.session_state.train_csv = "training_reduced.csv"
                st.session_state.test_csv = "test_reduced.csv"
                st.session_state.feature_selection_json = "feature_selection_result.json"

                st.success("Feature Selection completed.")

            except Exception as e:

                st.error(str(e))

        if st.session_state.feature_selection_done:

            json_path = Path("outputs") / "feature_selection_result.json"

            if json_path.exists():

                with open(json_path) as f:
                    results = json.load(f)

                st.subheader("Feature Selection Summary")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Selected Features",
                    results["n_selected"],
                )

                c2.metric(
                    "Alpha",
                    round(results["alpha"], 4),
                )

                c3.metric(
                    "Optimization Runs",
                    results["alpha_computations"],
                )

                st.subheader("Selected Feature Names")

                st.write(results["selected_feature_names"])

                st.subheader("Execution Statistics")

                st.json(results)

            train_path = Path("outputs") / "training_reduced.csv"

            if train_path.exists():

                st.subheader("Reduced Training Dataset")

                train_df = pd.read_csv(train_path)

                st.dataframe(
                    train_df.head(20),
                    use_container_width=True,
                )

            optim_path = Path("outputs") / "optimizations.csv"

            if optim_path.exists():

                st.subheader("Optimization History")

                optim_df = pd.read_csv(optim_path)

                st.dataframe(
                    optim_df,
                    use_container_width=True,
                )

elif page == "Training":

    st.title("Training")

    if not st.session_state.feature_selection_done:

        st.warning("Run Feature Selection first.")

    else:

        st.write("Train the classifier using the reduced training dataset.")

        classifier = st.selectbox(
            "Classifier",
            [
                "lightgbm",
                "xgboost",
                "knn",
            ],
        )

        seed = st.number_input(
            "Seed",
            min_value=0,
            value=42,
            step=1,
        )

        st.info(
            "The remaining classifier parameters use the backend defaults."
        )

        if st.button("Train Model"):

            try:

                with st.spinner("Training model..."):

                    train(
                        classifier=classifier,
                        reducedTrain_csv=st.session_state.train_csv,
                        target_column=st.session_state.target_column,
                        model_path="model.joblib",
                        metrics_json="training_metrics.json",
                        seed=seed,
                    )

                st.session_state.training_done = True
                st.session_state.model_path = "model.joblib"
                st.session_state.training_metrics = "training_metrics.json"

                st.success("Training completed successfully.")

            except Exception as e:

                st.error(str(e))

        if st.session_state.training_done:

            metrics_path = Path("outputs") / "training_metrics.json"

            if metrics_path.exists():

                with open(metrics_path) as f:
                    metrics = json.load(f)

                st.subheader("Training Summary")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Classifier",
                    metrics["classifier"],
                )

                c2.metric(
                    "Samples",
                    metrics["n_samples"],
                )

                c3, c4 = st.columns(2)

                c3.metric(
                    "Features",
                    metrics["n_features"],
                )

                c4.metric(
                    "Training Time (s)",
                    round(metrics["training_time"], 3),
                )

                st.subheader("Training Statistics")

                st.json(metrics)

                model_file = Path("outputs") / "model.joblib"

                if model_file.exists():

                    st.success("Model successfully saved.")

                    st.code(str(model_file))

elif page == "Prediction":

    st.title("Prediction")

    if not st.session_state.training_done:

        st.warning("Train a model first.")

    else:

        st.write("Generate predictions using the trained model.")

        if st.button("Run Prediction"):

            try:

                with st.spinner("Running prediction..."):

                    predict(
                        reduced_Test_csv=st.session_state.test_csv,
                        target_column=st.session_state.target_column,
                        model_path=st.session_state.model_path,
                        predictions_csv="predictions.csv",
                        classif_stats_json="classification_stats.json",
                    )

                st.session_state.prediction_done = True
                st.session_state.predictions_csv = "predictions.csv"
                st.session_state.classification_stats = "classification_stats.json"

                st.success("Prediction completed successfully.")

            except Exception as e:

                st.error(str(e))

        if st.session_state.prediction_done:

            stats_path = Path("outputs") / "classification_stats.json"

            if stats_path.exists():

                with open(stats_path) as f:
                    stats = json.load(f)

                st.subheader("Classification Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Accuracy",
                    f"{stats['accuracy']:.4f}"
                )

                col2.metric(
                    "ROC AUC",
                    f"{stats['roc_auc']:.4f}"
                )

                col3.metric(
                    "Samples",
                    stats["n_samples"]
                )

                st.subheader("Class 0")

                class0 = stats["class_0"]

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Precision",
                    f"{class0['precision']:.4f}"
                )

                c2.metric(
                    "Recall",
                    f"{class0['recall']:.4f}"
                )

                c3.metric(
                    "F1",
                    f"{class0['f1']:.4f}"
                )

                st.subheader("Class 1")

                class1 = stats["class_1"]

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Precision",
                    f"{class1['precision']:.4f}"
                )

                c2.metric(
                    "Recall",
                    f"{class1['recall']:.4f}"
                )

                c3.metric(
                    "F1",
                    f"{class1['f1']:.4f}"
                )

                st.subheader("Confusion Matrix")

                matrix = stats["confusion_matrix"]["matrix"]

                matrix_df = pd.DataFrame(
                    matrix,
                    columns=["Predicted 0", "Predicted 1"],
                    index=["Actual 0", "Actual 1"],
                )

                st.dataframe(
                    matrix_df,
                    use_container_width=True,
                )

                st.subheader("Complete Statistics")

                st.json(stats)

            prediction_file = Path("outputs") / "predictions.csv"

            if prediction_file.exists():

                st.subheader("Prediction Preview")

                predictions = pd.read_csv(prediction_file)

                st.dataframe(
                    predictions.head(20),
                    use_container_width=True,
                )

elif page == "Results":

    st.title("Results")

    st.write("Results page coming in Step 7.")