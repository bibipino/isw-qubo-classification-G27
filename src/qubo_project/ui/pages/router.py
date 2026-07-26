from __future__ import annotations

import streamlit as st

from ui.pages import (
    dataset_page,
    feature_selection_page,
    home_page,
    prediction_page,
    preprocessing_page,
    results_page,
    training_page,
)


def render_page(page: str) -> None:

    if page == "Home":
        home_page.render()
        return

    if page == "Dataset":
        dataset_page.render()
        return

    if page == "Preprocessing":
        preprocessing_page.render()
        return

    if page == "Feature Selection":
        feature_selection_page.render()
        return

    if page == "Training":
        training_page.render()
        return

    if page == "Prediction":
        prediction_page.render()
        return

    if page == "Results":
        results_page.render()
        return

    st.error("Unknown page.")