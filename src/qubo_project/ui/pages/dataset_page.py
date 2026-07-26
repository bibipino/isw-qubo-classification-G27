from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st


def render() -> None:
    """
    Dataset selection page.
    """

    st.title("📁 Dataset")

    st.write(
        "Upload a CSV dataset and select the target column before continuing."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a dataset to begin.")
        return

    try:
        dataframe = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))

    except Exception as exc:
        st.error(f"Unable to read the CSV file.\n\n{exc}")
        return

    st.success("Dataset loaded successfully.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", dataframe.shape[0])
    col2.metric("Columns", dataframe.shape[1])
    col3.metric("Missing Values", int(dataframe.isna().sum().sum()))

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(
            dataframe,
            width="stretch",
        )

    target_column = st.selectbox(
        "Target column",
        dataframe.columns.tolist(),
        index=0,
    )

    st.markdown("---")

    if st.button("Save Dataset Configuration", type="primary"):

        st.session_state.dataset_path = uploaded_file.name
        st.session_state.target_column = target_column

        st.success("Dataset configuration saved.")

    st.markdown("---")

    st.subheader("Current Selection")

    st.write(
        f"**Dataset:** {st.session_state.dataset_path or 'Not selected'}"
    )

    st.write(
        f"**Target column:** {st.session_state.target_column or 'Not selected'}"
    )