import streamlit as st


def initialize_session_state():
    """Initialize session state variables only once."""

    defaults = {
        "page": "Home",
        "dataset_path": None,
        "target_column": "target",

        "preprocess_dataset_path": None,
        "normalized_csv_name": None,
        "preprocessing_json_name": None,
        "preprocessing_out_data_dir": "data",
        "preprocessing_out_json_dir": "outputs",
        "normalized_csv": None,
        "preprocessing_json": None,

        "normalized_path": None,
        "train_csv": "training_reduced.csv",
        "test_csv": "test_reduced.csv",
        "optimization_csv": "optimization.csv",
        "feature_selection_json": "feature_selection_result.json",

        "train_csv_path": None,
        "selected_classifier": "xgboost",
        "model_path_name": "model.joblib",
        "training_metrics_json": "training_metrics.json",
        "training_seed": 42,
        "training_done": False,

        "predictions_csv": None,
        "classification_stats": None,

        "preprocessing_done": False,
        "feature_selection_done": False,
        "training_done": False,
        "prediction_done": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def pipeline_status():
    """Display pipeline progress in the sidebar."""

    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline Status")

    def icon(done):
        return "✅" if done else "⬜"

    st.sidebar.write(f"{icon(st.session_state.preprocessing_done)} Preprocessing")
    st.sidebar.write(f"{icon(st.session_state.feature_selection_done)} Feature Selection")
    st.sidebar.write(f"{icon(st.session_state.training_done)} Training")
    st.sidebar.write(f"{icon(st.session_state.prediction_done)} Prediction")