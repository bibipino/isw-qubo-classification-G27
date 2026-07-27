import streamlit as st


def initialize_session_state():
    """Initialize session state variables only once."""

    defaults = {
        "page": "Home",
        "dataset_path": None,
        "target_column": "target",

        "normalized_csv": None,
        "preprocessing_json": None,

        "train_csv": None,
        "test_csv": None,
        "feature_selection_json": None,

        "model_path": None,
        "training_metrics": None,

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