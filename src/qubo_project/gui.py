import streamlit as st
from pathlib import Path
import pandas as pd
import json
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
    ],
)

#pipeline_status()

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

    st.title("Dataset / File Preview")

    st.write("Select a CSV or JSON file to visualize.")

    default_path = "data/input_dataset.csv"

    dataset_path = st.text_input(
        "File path (CSV or JSON)",
        value=st.session_state.dataset_path or default_path,
    )

    if st.button("Load File"):
        try:
            path = Path(dataset_path)

            if not path.exists():
                st.error("File not found at specified path.")
                st.stop()

            ext = path.suffix.lower()
            if ext not in [".csv", ".json"]:
                st.error("Unsupported file extension. Please select a .csv or .json file.")
                st.stop()

            st.session_state.dataset_path = str(path)
            st.session_state.dataset_type = ext
            st.success(f"{ext.upper()} file loaded successfully.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    if st.session_state.get("dataset_path"):
        current_path = Path(st.session_state.dataset_path)

        if current_path.exists():
            ext = current_path.suffix.lower()

            # ----------------------------------------------------
            # JSON File Rendering
            # ----------------------------------------------------
            if ext == ".json":
                try:
                    with open(current_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                    st.subheader("JSON Information")
                    
                    col1, col2 = st.columns(2)
                    if isinstance(json_data, dict):
                        col1.metric("Top-level Keys", len(json_data.keys()))
                    elif isinstance(json_data, list):
                        col1.metric("Total Items", len(json_data))
                    
                    file_size_kb = round(current_path.stat().st_size / 1024, 2)
                    col2.metric("File Size", f"{file_size_kb} KB")

                    st.subheader("JSON Viewer")
                    st.json(json_data)

                    # Download button for JSON
                    with open(current_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {current_path.name}",
                            data=f,
                            file_name=current_path.name,
                            mime="application/json",
                        )

                except Exception as e:
                    st.error(f"Error parsing JSON file: {e}")

            # ----------------------------------------------------
            # CSV File Rendering
            # ----------------------------------------------------
            elif ext == ".csv":
                try:
                    df = pd.read_csv(current_path)
                    st.session_state.available_columns = list(df.columns)

                    st.subheader("Dataset Information")
                    col1, col2 = st.columns(2)
                    col1.metric("Rows", len(df))
                    col2.metric("Columns", len(df.columns))

                    st.subheader("Dataset Preview")
                    st.dataframe(df.head(20), use_container_width=True)

                    # Download button for CSV
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download {current_path.name}",
                        data=csv_bytes,
                        file_name=current_path.name,
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")

elif page == "Preprocessing":

    st.title("Preprocessing")

    default_path = "data/input_dataset.csv"
    
    preprocess_dataset_path = st.text_input(
        "Dataset path",
        value=st.session_state.preprocess_dataset_path or default_path,
    )

    if st.button("Load Dataset"):

        try:

            path = Path(preprocess_dataset_path)

            if not path.exists() or not preprocess_dataset_path.endswith(".csv"):
                st.error("Dataset not found or wrong file selected.")
                st.stop()

            df = pd.read_csv(path)

            st.session_state.preprocess_dataset_path = str(path)

            st.success("Dataset loaded successfully.")

            st.session_state.available_columns = list(df.columns)

        except Exception as e:
            st.error(str(e))
            
    if st.session_state.preprocess_dataset_path is None:

        st.warning("Please load a dataset first.")

    else:
        
        st.write("Run preprocessing using the selected dataset.")

        df = pd.read_csv(st.session_state.preprocess_dataset_path)

        target = st.selectbox(
            "Target column",
            options=df.columns,
            index=list(df.columns).index(
                st.session_state.target_column
            ) if st.session_state.target_column in df.columns else 0,
        )

        st.session_state.target_column = target
        
        min_valid = st.slider(
            "Minimum valid percentage",
            min_value=0.01,
            max_value=0.50,
            value=0.05,
            step=0.01,
        )

        col1, col2 = st.columns(2)

        with col1:

            normalized_name = st.text_input(
                "Normalized dataset name",
                value=st.session_state.normalized_csv_name or "normalized.csv",
            )

            csv_preprocess_out = st.text_input(
                "Normalized dataset output dir",
                value=st.session_state.preprocessing_out_data_dir,
            )

        with col2:
            json_preprocess_name = st.text_input(
                "Result JSON name",
                value=st.session_state.preprocessing_json_name or "preprocessing_result.json",
            )

            json_preprocess_out = st.text_input(
                "Result JSON output dir",
                value=st.session_state.preprocessing_out_json_dir,
            )

        if st.button("Run Preprocessing"):

            try:

                with st.spinner("Running preprocessing..."):

                    fit_normalize(
                        input_csv=st.session_state.preprocess_dataset_path,
                        target_column=st.session_state.target_column,
                        normalized_csv=normalized_name,
                        outInitalRes_json=json_preprocess_name,
                        minPercValid=min_valid,
                        output_data_dir=csv_preprocess_out,
                        output_json_dir=json_preprocess_out
                    )

                st.session_state.preprocessing_done = True

                st.success("Preprocessing completed successfully.")

            except Exception as e:

                st.error(str(e))

        if st.session_state.get("preprocessing_done"):

            # Using 'or' guarantees a fallback string if session state holds None
            out_dir_json = st.session_state.get("preprocessing_out_json_dir")
            out_dir_csv = st.session_state.get("preprocessing_out_data_dir")
            json_name = st.session_state.get("preprocessing_json_name") or "preprocessing_result.json"
            csv_name = st.session_state.get("normalized_csv_name") or "normalized.csv"

            json_path = Path(out_dir_json) / json_name
            csv_path = Path(out_dir_csv) / csv_name
            # ----------------------------------------------------
            # 1. JSON Execution Information & Download
            # ----------------------------------------------------
            if json_path.exists():
                st.subheader("Execution Information (JSON)")

                with open(json_path, "r") as f:
                    json_data = json.load(f)

                # Display full JSON
                st.json(json_data)

                # Download button for JSON
                with open(json_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {json_name}",
                        data=f,
                        file_name=json_name,
                        mime="application/json",
                    )
            else:
                st.error(f"JSON result file not found at `{json_path}`")

            st.markdown("---")

            # ----------------------------------------------------
            # 2. Normalized Dataset & Download
            # ----------------------------------------------------
            if csv_path.exists():
                st.subheader("Normalized Dataset (CSV)")

                df = pd.read_csv(csv_path)

                # Display full dataframe (entirety)
                st.dataframe(df, use_container_width=True)

                # Download button for CSV
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download {csv_name}",
                    data=csv_bytes,
                    file_name=csv_name,
                    mime="text/csv",
                )
            else:
                st.error(f"Normalized CSV file not found at `{csv_path}`")

elif page == "Feature Selection":

    st.title("Feature Selection")

    default_path = "data/normalized.csv"
    
    normalized_path = st.text_input(
        "Dataset path",
        value=st.session_state.get("normalized_path") or default_path,
    )

    if st.button("Load Dataset"):

        try:

            path = Path(normalized_path)

            if not path.exists() or not normalized_path.endswith(".csv"):
                st.error("Dataset not found or wrong file selected.")
                st.stop()

            df = pd.read_csv(path)

            st.session_state.normalized_path = str(path)

            st.success("Dataset loaded successfully.")

            st.session_state.available_columns = list(df.columns)

        except Exception as e:
            st.error(str(e))

    if not st.session_state.get("normalized_path"):

        st.warning("Please load a dataset first.")

    else:

        st.write("Run QUBO Feature Selection.")

        col1, col2 = st.columns(2)

        with col1:

            train_csv = st.text_input(
                "Normalized training dataset name",
                value=st.session_state.get("train_csv") or "train.csv",
            )

            test_csv = st.text_input(
                "Normalized testing dataset name",
                value=st.session_state.get("test_csv") or "test.csv",
            )

            perc_selected = st.slider(
                "Selected Feature Percentage",
                0.00,
                1.00,
                0.20,
                0.01,
            )

            allowance = st.number_input(
                "Allowance",
                min_value=0,
                max_value=100,
                value=1,
            )

            perc_test = st.slider(
                "Test Percentage",
                0.00,
                1.00,
                0.30,
                0.01,
            )

        with col2:

            optimization_csv = st.text_input(
                "Optimization CSV output name",
                value=st.session_state.get("optimization_csv") or "optimizations.csv",
            )

            json_name_feature_selection = st.text_input(
                "Feature Selection JSON output name",
                value=st.session_state.get("feature_selection_json") or "feature_selection_result.json",
            )

            seed = st.number_input(
                "Seed",
                value=42,
                step=1,
            )

            alpha_computations = st.number_input(
                "Maximum Alpha Computations",
                min_value=1,
                max_value=99999,
                value=100,
            )

        if st.button("Run Feature Selection"):

            try:

                with st.spinner("Running Feature Selection..."):

                    select_features(
                        normalized_csv=st.session_state.normalized_path,
                        reducedTrain_csv=train_csv,
                        reducedTest_csv=test_csv,
                        output_ottim_csv=optimization_csv,
                        output_json=json_name_feature_selection,
                        target_column=st.session_state.get("target_column", "target"),
                        percTest=perc_test,
                        percSelected=perc_selected,
                        allowance=allowance,
                        seed=seed,
                        alpha_computations=alpha_computations,
                    )

                # Save precise file names to session state
                st.session_state.feature_selection_done = True
                st.session_state.train_csv = train_csv
                st.session_state.test_csv = test_csv
                st.session_state.optimization_csv = optimization_csv
                st.session_state.feature_selection_json = json_name_feature_selection

                st.success("Feature Selection completed successfully.")

            except Exception as e:

                st.error(str(e))

        # ----------------------------------------------------
        # Display Outputs (1 JSON + 3 CSVs)
        # ----------------------------------------------------
        if st.session_state.get("feature_selection_done"):

            out_dir_str = st.session_state.get("feature_selection_out_dir") or "outputs"
            out_dir = Path(out_dir_str)

            json_name = st.session_state.get("feature_selection_json") or "feature_selection_result.json"
            train_name = st.session_state.get("train_csv") or "train.csv"
            test_name = st.session_state.get("test_csv") or "test.csv"
            optim_name = st.session_state.get("optimization_csv") or "optimizations.csv"

            json_path = out_dir / json_name
            train_path = out_dir / train_name
            test_path = out_dir / test_name
            optim_path = out_dir / optim_name

            st.markdown("---")

            # 1. Feature Selection JSON
            if json_path.exists():
                st.subheader("Feature Selection Summary & Execution Info (JSON)")

                with open(json_path, "r") as f:
                    results = json.load(f)

                c1, c2, c3 = st.columns(3)
                c1.metric("Selected Features", results.get("n_selected", "N/A"))
                c2.metric("Alpha", round(results.get("alpha", 0), 4) if "alpha" in results else "N/A")
                c3.metric("Optimization Runs", results.get("alpha_computations", "N/A"))

                st.write("**Selected Feature Names:**", results.get("selected_feature_names", []))
                
                st.json(results)

                with open(json_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {json_name}",
                        data=f,
                        file_name=json_name,
                        mime="application/json",
                    )
            else:
                st.error(f"JSON result file not found at `{json_path}`")

            st.markdown("---")

            # 2. Reduced Training Dataset CSV
            if train_path.exists():
                st.subheader("Reduced Training Dataset (CSV)")

                train_df = pd.read_csv(train_path)
                st.dataframe(train_df, use_container_width=True)

                st.download_button(
                    label=f"📥 Download {train_name}",
                    data=train_df.to_csv(index=False).encode("utf-8"),
                    file_name=train_name,
                    mime="text/csv",
                )
            else:
                st.error(f"Training dataset file not found at `{train_path}`")

            st.markdown("---")

            # 3. Reduced Testing Dataset CSV
            if test_path.exists():
                st.subheader("Reduced Testing Dataset (CSV)")

                test_df = pd.read_csv(test_path)
                st.dataframe(test_df, use_container_width=True)

                st.download_button(
                    label=f"📥 Download {test_name}",
                    data=test_df.to_csv(index=False).encode("utf-8"),
                    file_name=test_name,
                    mime="text/csv",
                )
            else:
                st.error(f"Testing dataset file not found at `{test_path}`")

            st.markdown("---")

            # 4. Optimization Log CSV
            if optim_path.exists():
                st.subheader("Optimization History (CSV)")

                optim_df = pd.read_csv(optim_path)
                st.dataframe(optim_df, use_container_width=True)

                st.download_button(
                    label=f"📥 Download {optim_name}",
                    data=optim_df.to_csv(index=False).encode("utf-8"),
                    file_name=optim_name,
                    mime="text/csv",
                )
            else:
                st.error(f"Optimization CSV file not found at `{optim_path}`")

elif page == "Training":

    st.title("Model Training")

    default_train_path = "outputs/training_reduced.csv"
    data_dir = Path("outputs")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Upload / Path Configuration
    if not st.session_state.get("train_csv_path"):
        train_path_input = st.text_input(
            "Or enter training dataset path",
            value=default_train_path,
        )
        if st.button("Load Dataset Path"):
            path = Path(train_path_input)
            if path.exists():
                st.session_state.train_csv_path = str(path)
                df = pd.read_csv(path)
                st.session_state.available_columns = list(df.columns)
                st.success("Dataset path loaded.")
            else:
                st.error("File not found at specified path.")

    if not st.session_state.get("train_csv_path"):
        st.warning("Please load a dataset first.")

    # 2. Main Training Parameters & Classifier Hyperparameters
    if st.session_state.get("train_csv_path"):

        st.markdown("---")
        st.subheader("Training Options")

        col1, col2 = st.columns(2)

        with col1:
            classifier = st.selectbox(
                "Classifier Algorithm",
                options=["random_forest", "xgboost", "knn"],
                index=["random_forest", "xgboost", "knn"].index(
                    st.session_state.get("selected_classifier") if st.session_state.get("selected_classifier") in ["random_forest", "xgboost", "knn"] else "random_forest"
                ),
            )
            st.session_state.selected_classifier = classifier

            # Read columns dynamically for target selection
            train_df_preview = pd.read_csv(st.session_state.train_csv_path)
            target = st.selectbox(
                "Target Column",
                options=train_df_preview.columns,
                index=list(train_df_preview.columns).index(
                    st.session_state.get("target_column")
                ) if st.session_state.get("target_column") in train_df_preview.columns else 0,
            )
            st.session_state.target_column = target

            seed = st.number_input(
                "Random Seed",
                value=int(st.session_state.get("training_seed") or 42),
                step=1,
            )

        with col2:
            model_out_name = st.text_input(
                "Output Model Name (.joblib)",
                value=st.session_state.get("model_path_name") or "model.joblib",
            )

            metrics_out_name = st.text_input(
                "Output Metrics JSON Name",
                value=st.session_state.get("training_metrics_json") or "training_metrics.json",
            )

        # ----------------------------------------------------
        # Dynamic Hyperparameters per Classifier
        # ----------------------------------------------------
        st.subheader(f"Hyperparameters ({classifier.upper()})")
        clf_params = {}

        h1, h2 = st.columns(2)

        if classifier == "random_forest":
            with h1:
                clf_params["n_estimators"] = st.number_input(
                    "n_estimators (Trees)", min_value=1, max_value=2000, value=100
                )
                max_depth_val = st.number_input(
                    "max_depth (0 for None)", min_value=0, max_value=100, value=0
                )
                clf_params["max_depth"] = max_depth_val if max_depth_val > 0 else None
            with h2:
                clf_params["n_jobs"] = st.number_input(
                    "n_jobs (-1 for all CPUs)", min_value=-1, max_value=64, value=-1
                )

        elif classifier == "xgboost":
            with h1:
                clf_params["n_estimators"] = st.number_input(
                    "n_estimators (Trees)", min_value=1, max_value=2000, value=100
                )
                clf_params["learning_rate"] = st.number_input(
                    "learning_rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01
                )
            with h2:
                clf_params["max_depth"] = st.number_input(
                    "max_depth", min_value=1, max_value=100, value=6
                )
                clf_params["n_jobs"] = st.number_input(
                    "n_jobs (-1 for all CPUs)", min_value=-1, max_value=64, value=-1
                )

        elif classifier == "knn":
            with h1:
                clf_params["n_neighbors"] = st.number_input(
                    "n_neighbors", min_value=1, max_value=100, value=5
                )
                clf_params["weights"] = st.selectbox(
                    "weights", options=["uniform", "distance"], index=0
                )
            with h2:
                clf_params["n_jobs"] = st.number_input(
                    "n_jobs (-1 for all CPUs)", min_value=-1, max_value=64, value=-1
                )

        # 3. Execution
        if st.button("Run Training"):
            try:
                with st.spinner(f"Training {classifier.upper()} model..."):
                    train_filename = Path(st.session_state.train_csv_path).name

                    train(
                        classifier=classifier,
                        reducedTrain_csv=train_filename,
                        target_column=st.session_state.target_column,
                        model_path=model_out_name,
                        metrics_json=metrics_out_name,
                        seed=seed,
                        **clf_params,
                    )

                st.session_state.training_done = True
                st.session_state.model_path_name = model_out_name
                st.session_state.training_metrics_json = metrics_out_name
                st.success("Model training completed successfully!")

            except Exception as e:
                st.error(f"Training Error: {e}")

        # 4. Results Display & Download Section (.joblib + .json)
        if st.session_state.get("training_done"):

            st.markdown("---")
            out_dir = Path("outputs")

            m_json_name = st.session_state.get("training_metrics_json") or "training_metrics.json"
            m_joblib_name = st.session_state.get("model_path_name") or "model.joblib"

            json_path = out_dir / m_json_name
            joblib_path = out_dir / m_joblib_name

            if json_path.exists():
                st.subheader("Training Metrics & Execution Summary (JSON)")

                with open(json_path, "r") as f:
                    metrics_data = json.load(f)

                c1, c2, c3 = st.columns(3)
                c1.metric("Classifier", metrics_data.get("classifier", "N/A").upper())
                c2.metric("Samples", metrics_data.get("n_samples", "N/A"))
                c3.metric("Training Time (s)", f"{metrics_data.get('training_time', 0)}s")

                st.json(metrics_data)

                with open(json_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {m_json_name}",
                        data=f,
                        file_name=m_json_name,
                        mime="application/json",
                    )
            else:
                st.error(f"Metrics JSON file not found at `{json_path}`")

            st.markdown("---")

            if joblib_path.exists():
                st.subheader("Trained Model File (.joblib)")
                st.info(f"Model saved successfully to `{joblib_path}`.")

                with open(joblib_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {m_joblib_name}",
                        data=f,
                        file_name=m_joblib_name,
                        mime="application/octet-stream",
                    )
            else:
                st.error(f"Model binary file not found at `{joblib_path}`")

elif page == "Prediction":

    st.title("Prediction")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Path Configuration Inputs
    st.subheader("Input File Paths")

    col1, col2 = st.columns(2)

    default_test_name = st.session_state.get("test_csv") or "test.csv"
    default_model_name = st.session_state.get("model_path_name") or "model.joblib"

    with col1:
        test_path_input = st.text_input(
            "Reduced test dataset path (CSV)",
            value=st.session_state.get("test_csv_path") or str(out_dir / default_test_name),
        )
        st.session_state.test_csv_path = test_path_input

    with col2:
        model_path_input = st.text_input(
            "Trained model path (.joblib)",
            value=st.session_state.get("model_file_path") or str(out_dir / default_model_name),
        )
        st.session_state.model_file_path = model_path_input

    # 2. Target Column & Output File Names Configuration
    st.markdown("---")
    st.subheader("Prediction Options")

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        # Dynamically load columns from the specified test CSV path if available
        available_cols = ["target"]
        test_file_path = Path(st.session_state.get("test_csv_path", ""))
        
        if test_file_path.exists() and test_file_path.is_file():
            try:
                available_cols = list(pd.read_csv(test_file_path, nrows=1).columns)
            except Exception:
                pass

        target = st.selectbox(
            "Target Column",
            options=available_cols,
            index=available_cols.index(
                st.session_state.get("target_column")
            ) if st.session_state.get("target_column") in available_cols else 0,
        )
        st.session_state.target_column = target

    with cfg_col2:
        predictions_out_name = st.text_input(
            "Predictions CSV output name",
            value=st.session_state.get("predictions_csv") or "predictions.csv",
        )
        stats_out_name = st.text_input(
            "Classification stats JSON name",
            value=st.session_state.get("classification_stats") or "classification_stats.json",
        )

    # 3. Execution Section
    if st.button("Run Prediction"):
        try:
            test_path = Path(st.session_state.get("test_csv_path", ""))
            model_path = Path(st.session_state.get("model_file_path", ""))

            if not test_path.exists():
                st.error(f"Test dataset file not found at path: `{test_path}`")
                st.stop()

            if not model_path.exists():
                st.error(f"Trained model file not found at path: `{model_path}`")
                st.stop()

            with st.spinner("Running prediction..."):
                # Backend predict function expects filenames relative to the outputs directory
                predict(
                    reduced_Test_csv=test_path.name,
                    target_column=st.session_state.target_column,
                    model_path=model_path.name,
                    predictions_csv=predictions_out_name,
                    classif_stats_json=stats_out_name,
                )

            st.session_state.prediction_done = True
            st.session_state.predictions_csv = predictions_out_name
            st.session_state.classification_stats = stats_out_name
            st.success("Prediction completed successfully.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # ----------------------------------------------------
    # 4. Results Display & Download Section (JSON + CSV)
    # ----------------------------------------------------
    if st.session_state.get("prediction_done"):

        st.markdown("---")

        stats_name = st.session_state.get("classification_stats") or "classification_stats.json"
        preds_name = st.session_state.get("predictions_csv") or "predictions.csv"

        stats_path = out_dir / stats_name
        prediction_file = out_dir / preds_name

        # 1. Classification Statistics JSON
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)

            st.subheader("Classification Results Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{stats.get('accuracy', 0):.4f}")
            roc_auc_val = stats.get("roc_auc")
            c2.metric("ROC AUC", f"{roc_auc_val:.4f}" if roc_auc_val is not None else "N/A")
            c3.metric("Samples", stats.get("n_samples", "N/A"))

            # Class metrics breakdown
            col_c0, col_c1 = st.columns(2)

            with col_c0:
                st.markdown("##### Class 0 Metrics")
                class0 = stats.get("class_0", {})
                st.write(f"**Precision:** {class0.get('precision', 0):.4f}")
                st.write(f"**Recall:** {class0.get('recall', 0):.4f}")
                st.write(f"**F1 Score:** {class0.get('f1', 0):.4f}")

            with col_c1:
                st.markdown("##### Class 1 Metrics")
                class1 = stats.get("class_1", {})
                st.write(f"**Precision:** {class1.get('precision', 0):.4f}")
                st.write(f"**Recall:** {class1.get('recall', 0):.4f}")
                st.write(f"**F1 Score:** {class1.get('f1', 0):.4f}")

            # Confusion Matrix
            if "confusion_matrix" in stats and "matrix" in stats["confusion_matrix"]:
                st.markdown("##### Confusion Matrix")
                matrix = stats["confusion_matrix"]["matrix"]
                matrix_df = pd.DataFrame(
                    matrix,
                    columns=["Predicted 0", "Predicted 1"],
                    index=["Actual 0", "Actual 1"],
                )
                st.dataframe(matrix_df, use_container_width=True)

            st.markdown("##### Complete Stats JSON")
            st.json(stats)

            with open(stats_path, "rb") as f:
                st.download_button(
                    label=f"📥 Download {stats_name}",
                    data=f,
                    file_name=stats_name,
                    mime="application/json",
                )
        else:
            st.error(f"Classification stats JSON not found at `{stats_path}`")

        st.markdown("---")

        # 2. Predictions Dataframe CSV (Entirety)
        if prediction_file.exists():
            st.subheader("Complete Predictions Dataset (CSV)")

            predictions_df = pd.read_csv(prediction_file)
            st.dataframe(predictions_df, use_container_width=True)

            st.download_button(
                label=f"📥 Download {preds_name}",
                data=predictions_df.to_csv(index=False).encode("utf-8"),
                file_name=preds_name,
                mime="text/csv",
            )
        else:
            st.error(f"Predictions CSV not found at `{prediction_file}`")