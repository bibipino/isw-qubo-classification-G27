from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.pipeline_service import PipelineService


OUTPUT_DIR = Path("outputs")


def render() -> None:
    """
    Render the preprocessing page.
    """

    st.title("⚙️ Preprocessing")

    if st.session_state.dataset_path is None:
        st.warning("Please select a dataset first.")
        return

    if st.session_state.target_column is None:
        st.warning("Please select a target column first.")
        return

    st.success("Dataset configuration found.")

    st.write(f"**Dataset:** {st.session_state.dataset_path}")
    st.write(f"**Target column:** {st.session_state.target_column}")

    st.divider()

    min_perc_valid = st.number_input(
        "Minimum valid percentage",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        format="%.2f",
    )

    if st.button(
        "Run Preprocessing",
        type="primary",
    ):
        OUTPUT_DIR.mkdir(exist_ok=True)

        normalized_csv = OUTPUT_DIR / "normalized.csv"
        preprocessing_json = OUTPUT_DIR / "preprocessing_report.json"

        service = PipelineService()

        with st.spinner("Running preprocessing..."):

            report = service.run_preprocessing(
                input_csv=st.session_state.dataset_path,
                target_column=st.session_state.target_column,
                normalized_csv=str(normalized_csv),
                out_initial_res_json=str(preprocessing_json),
                min_perc_valid=min_perc_valid,
            )

        st.session_state.preprocessing_completed = True

        st.session_state.generated_files["normalized_csv"] = str(
            normalized_csv
        )

        st.session_state.generated_files["preprocessing_json"] = str(
            preprocessing_json
        )

        st.session_state.logs.append(
            "Preprocessing completed successfully."
        )

        st.success("Preprocessing completed.")

        st.subheader("Generated Report")

        st.json(report)

    st.divider()

    st.subheader("Status")

    if st.session_state.preprocessing_completed:
        st.success("Completed")
    else:
        st.info("Waiting for execution")