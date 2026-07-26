from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.pipeline_service import PipelineService


OUTPUT_DIR = Path("outputs")


def render() -> None:

    st.title("🧬 Feature Selection")

    if not st.session_state.preprocessing_completed:
        st.error("Complete preprocessing before feature selection.")
        return

    perc_test = st.slider(
        "Test percentage",
        0.1,
        0.5,
        0.2,
        0.05,
    )

    perc_selected = st.slider(
        "Selected feature percentage",
        0.05,
        1.0,
        0.3,
        0.05,
    )

    allowance = st.number_input(
        "Allowance",
        value=0.05,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )

    seed = st.number_input(
        "Random seed",
        value=42,
        step=1,
    )

    alpha = st.number_input(
        "Alpha computations",
        value=100,
        step=1,
    )

    if st.button("Run Feature Selection", type="primary"):

        OUTPUT_DIR.mkdir(exist_ok=True)

        service = PipelineService()

        report = service.run_feature_selection(
            normalized_csv=st.session_state.generated_files["normalized_csv"],
            reduced_train_csv=str(OUTPUT_DIR / "reduced_train.csv"),
            reduced_test_csv=str(OUTPUT_DIR / "reduced_test.csv"),
            optimization_csv=str(OUTPUT_DIR / "optimization.csv"),
            optimization_json=str(OUTPUT_DIR / "feature_selection.json"),
            target_column=st.session_state.target_column,
            perc_test=perc_test,
            perc_selected=perc_selected,
            allowance=allowance,
            seed=int(seed),
            alpha_computations=int(alpha),
        )

        st.session_state.feature_selection_completed = True

        st.session_state.generated_files["reduced_train_csv"] = str(
            OUTPUT_DIR / "reduced_train.csv"
        )

        st.session_state.generated_files["reduced_test_csv"] = str(
            OUTPUT_DIR / "reduced_test.csv"
        )

        st.session_state.generated_files["optimization_csv"] = str(
            OUTPUT_DIR / "optimization.csv"
        )

        st.session_state.generated_files["feature_selection_json"] = str(
            OUTPUT_DIR / "feature_selection.json"
        )

        st.session_state.logs.append(
            "Feature selection completed."
        )

        st.success("Feature selection completed.")

        st.subheader("Selected Features")

        st.table(report["selected_features"])

        st.subheader("Report")

        st.json(report)