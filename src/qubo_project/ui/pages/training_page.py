from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.pipeline_service import PipelineService


OUTPUT_DIR = Path("outputs")


CLASSIFIERS = [
    "Logistic Regression",
    "Random Forest",
    "Support Vector Machine",
]


def render() -> None:

    st.title("🤖 Training")

    if not st.session_state.feature_selection_completed:
        st.error("Complete Feature Selection before training.")
        return

    classifier = st.selectbox(
        "Classifier",
        CLASSIFIERS,
    )

    seed = st.number_input(
        "Random Seed",
        value=42,
        step=1,
    )

    if st.button("Train Model", type="primary"):

        OUTPUT_DIR.mkdir(exist_ok=True)

        service = PipelineService()

        report = service.run_training(
            classifier=classifier,
            reduced_train_csv=st.session_state.generated_files["reduced_train_csv"],
            target_column=st.session_state.target_column,
            model_path=str(OUTPUT_DIR / "model.joblib"),
            metrics_json=str(OUTPUT_DIR / "training_metrics.json"),
            seed=int(seed),
        )

        st.session_state.training_completed = True
        st.session_state.classifier = classifier

        st.session_state.generated_files["model"] = str(
            OUTPUT_DIR / "model.joblib"
        )

        st.session_state.generated_files["training_metrics"] = str(
            OUTPUT_DIR / "training_metrics.json"
        )

        st.session_state.logs.append("Training completed.")

        st.success("Training completed successfully.")

        col1, col2 = st.columns(2)

        col1.metric("Accuracy", report["accuracy"])
        col1.metric("Precision", report["precision"])

        col2.metric("Recall", report["recall"])
        col2.metric("F1 Score", report["f1_score"])

        st.json(report)