from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def _load_json(path: str):
    file = Path(path)

    if not file.exists():
        return None

    with file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _download_button(label: str, path: str):

    file = Path(path)

    if not file.exists():
        return

    with file.open("rb") as f:
        st.download_button(
            label=label,
            data=f,
            file_name=file.name,
        )


def render() -> None:

    st.title("📊 Results")

    progress = sum(
        [
            st.session_state.preprocessing_completed,
            st.session_state.feature_selection_completed,
            st.session_state.training_completed,
            st.session_state.prediction_completed,
        ]
    )

    st.progress(progress / 4)

    st.subheader("Pipeline Status")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Preprocessing",
        "✅" if st.session_state.preprocessing_completed else "❌",
    )

    col2.metric(
        "Selection",
        "✅" if st.session_state.feature_selection_completed else "❌",
    )

    col3.metric(
        "Training",
        "✅" if st.session_state.training_completed else "❌",
    )

    col4.metric(
        "Prediction",
        "✅" if st.session_state.prediction_completed else "❌",
    )

    st.divider()

    tabs = st.tabs(
        [
            "Training",
            "Prediction",
            "Generated Files",
            "Execution Log",
        ]
    )

    #
    # TRAINING
    #

    with tabs[0]:

        metrics = None

        if "training_metrics" in st.session_state.generated_files:
            metrics = _load_json(
                st.session_state.generated_files["training_metrics"]
            )

        if metrics is None:
            st.info("Training has not been executed.")
        else:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", metrics["accuracy"])
            c2.metric("Precision", metrics["precision"])
            c3.metric("Recall", metrics["recall"])
            c4.metric("F1", metrics["f1_score"])

            st.json(metrics)

    #
    # PREDICTION
    #

    with tabs[1]:

        stats = None

        if "prediction_stats" in st.session_state.generated_files:
            stats = _load_json(
                st.session_state.generated_files["prediction_stats"]
            )

        if stats is None:

            st.info("Prediction has not been executed.")

        else:

            c1, c2, c3 = st.columns(3)

            c1.metric("Accuracy", stats["accuracy"])
            c2.metric("Precision", stats["precision"])
            c3.metric("ROC AUC", stats["roc_auc"])

            st.subheader("Confusion Matrix")

            st.table(stats["confusion_matrix"])

            st.json(stats)

    #
    # GENERATED FILES
    #

    with tabs[2]:

        if not st.session_state.generated_files:

            st.info("No files generated.")

        else:

            for name, path in st.session_state.generated_files.items():

                col1, col2 = st.columns([3, 1])

                col1.write(f"**{name}**")

                col1.caption(path)

                _download_button(
                    f"Download {name}",
                    path,
                )

    #
    # LOG
    #

    with tabs[3]:

        if not st.session_state.logs:

            st.info("No execution log.")

        else:

            for entry in st.session_state.logs:

                st.write("•", entry)