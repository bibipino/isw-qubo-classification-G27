from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.pipeline_service import PipelineService

OUTPUT_DIR = Path("outputs")


def render() -> None:

    st.title("🔮 Prediction")

    if not st.session_state.training_completed:
        st.error("Complete Training before prediction.")
        return

    if st.button("Run Prediction", type="primary"):

        OUTPUT_DIR.mkdir(exist_ok=True)

        service = PipelineService()

        report = service.run_prediction(
            reduced_test_csv=st.session_state.generated_files["reduced_test_csv"],
            target_column=st.session_state.target_column,
            model_path=st.session_state.generated_files["model"],
            predictions_csv=str(OUTPUT_DIR / "predictions.csv"),
            classif_stats_json=str(OUTPUT_DIR / "prediction_stats.json"),
        )

        st.session_state.prediction_completed = True

        st.session_state.generated_files["predictions_csv"] = str(
            OUTPUT_DIR / "predictions.csv"
        )

        st.session_state.generated_files["prediction_stats"] = str(
            OUTPUT_DIR / "prediction_stats.json"
        )

        st.session_state.logs.append("Prediction completed.")

        st.success("Prediction completed successfully.")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", report["accuracy"])
        col2.metric("Precision", report["precision"])
        col3.metric("ROC AUC", report["roc_auc"])

        st.subheader("Confusion Matrix")

        st.table(report["confusion_matrix"])

        st.subheader("Statistics")

        st.json(report)