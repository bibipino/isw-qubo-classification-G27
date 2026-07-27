import streamlit as st

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

    st.title("Dataset")

    st.write("Dataset page coming in Step 2.")

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