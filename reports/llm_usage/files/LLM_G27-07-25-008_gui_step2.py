import streamlit as st
from pathlib import Path
import pandas as pd

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

    st.write("Preprocessing page coming in Step 3.")

elif page == "Feature Selection":

    st.title("Feature Selection")

    st.write("Feature Selection page coming in Step 4.")

elif page == "Training":

    st.title("Training")

    st.write("Training page coming in Step 5.")

elif page == "Prediction":

    st.title("Prediction")

    st.write("Prediction page coming in Step 6.")

elif page == "Results":

    st.title("Results")

    st.write("Results page coming in Step 7.")