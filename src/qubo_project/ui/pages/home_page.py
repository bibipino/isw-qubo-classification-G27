from __future__ import annotations

import streamlit as st


def render() -> None:
    """
    Render the Home page.
    """

    st.title("🧩 QUBO Binary Classification")

    st.caption("Feature Reduction using QUBO Optimization")

    st.markdown(
        """
Welcome to the graphical interface.

The GUI orchestrates the complete machine learning workflow while remaining
independent from the backend implementation.

Use the navigation menu on the left to execute the pipeline step by step.
"""
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Preprocessing",
        "Done" if st.session_state.preprocessing_completed else "Pending",
    )

    col2.metric(
        "Feature Selection",
        "Done" if st.session_state.feature_selection_completed else "Pending",
    )

    col3.metric(
        "Training",
        "Done" if st.session_state.training_completed else "Pending",
    )

    col4.metric(
        "Prediction",
        "Done" if st.session_state.prediction_completed else "Pending",
    )

    st.divider()

    with st.expander("Current Session State"):

        st.json(st.session_state)