from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from fake_backend.feature_selection_fake import select_features
from fake_backend.model_fake import predict, train
from fake_backend.preprocessing_fake import fit_normalize


@dataclass(slots=True)
class PipelineService:

    def run_preprocessing(
        self,
        input_csv: str,
        target_column: str,
        normalized_csv: str,
        out_initial_res_json: str,
        min_perc_valid: float,
    ) -> Dict[str, object]:
        return fit_normalize(
            input_csv=input_csv,
            target_column=target_column,
            normalized_csv=normalized_csv,
            outInitialRes_json=out_initial_res_json,
            minPercValid=min_perc_valid,
        )

    def run_feature_selection(
        self,
        normalized_csv: str,
        reduced_train_csv: str,
        reduced_test_csv: str,
        optimization_csv: str,
        optimization_json: str,
        target_column: str,
        perc_test: float,
        perc_selected: float,
        allowance: float,
        seed: int,
        alpha_computations: int,
    ) -> Dict[str, object]:
        return select_features(
            normalized_csv=normalized_csv,
            reducedTrain_csv=reduced_train_csv,
            reducedTest_csv=reduced_test_csv,
            output_ottim_csv=optimization_csv,
            output_json=optimization_json,
            target_column=target_column,
            percTest=perc_test,
            percSelected=perc_selected,
            allowance=allowance,
            seed=seed,
            alpha_computations=alpha_computations,
        )

    def run_training(
        self,
        classifier: str,
        reduced_train_csv: str,
        target_column: str,
        model_path: str,
        metrics_json: str,
        seed: int,
    ) -> Dict[str, object]:
        return train(
            classifier=classifier,
            reducedTrain_csv=reduced_train_csv,
            target_column=target_column,
            model_path=model_path,
            metrics_json=metrics_json,
            seed=seed,
        )

    def run_prediction(
        self,
        reduced_test_csv: str,
        target_column: str,
        model_path: str,
        predictions_csv: str,
        classif_stats_json: str,
    ) -> Dict[str, object]:
        return predict(
            reduced_Test_csv=reduced_test_csv,
            target_column=target_column,
            model_path=model_path,
            predictions_csv=predictions_csv,
            classif_stats_json=classif_stats_json,
        )