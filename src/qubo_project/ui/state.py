from __future__ import annotations

import streamlit as st


DEFAULT_STATE = {
    "dataset_path": None,
    "target_column": None,
    "classifier": None,
    "parameters": {},
    "preprocessing_completed": False,
    "feature_selection_completed": False,
    "training_completed": False,
    "prediction_completed": False,
    "generated_files": {},
    "logs": [],
}


def initialize_state() -> None:
    """
    Initialize the Streamlit session state.
    """
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value