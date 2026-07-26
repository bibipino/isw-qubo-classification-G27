from __future__ import annotations

import streamlit as st


PAGES = (
    "Home",
    "Dataset",
    "Preprocessing",
    "Feature Selection",
    "Training",
    "Prediction",
    "Results",
)


def render_sidebar() -> str:
    """
    Render the application sidebar.

    Returns
    -------
    str
        Selected page.
    """

    st.sidebar.title("🧩 QUBO Pipeline")

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        PAGES,
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Pipeline Status")

    st.sidebar.write(
        f"Preprocessing: {'✅' if st.session_state.preprocessing_completed else '❌'}"
    )

    st.sidebar.write(
        f"Feature Selection: {'✅' if st.session_state.feature_selection_completed else '❌'}"
    )

    st.sidebar.write(
        f"Training: {'✅' if st.session_state.training_completed else '❌'}"
    )

    st.sidebar.write(
        f"Prediction: {'✅' if st.session_state.prediction_completed else '❌'}"
    )

    return page