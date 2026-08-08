import numpy as np
import pandas as pd

from irocket_model_selection import NestedCVResult, OuterFoldResult


def _outer_fold(fold_index, score, n_selected, stability):
    return OuterFoldResult(
        fold_index=fold_index,
        train_indices=np.array([2, 3]),
        test_indices=np.array([0, 1]),
        inner_search=None,
        best_consensus_threshold=0.7,
        best_alpha=1.0,
        predictions=np.array([0, 1]),
        decision_scores=np.array([-1.0, 1.0]),
        metrics={"balanced_accuracy": score, "accuracy": score},
        primary_score=score,
        n_transform_features=100,
        n_selected_features=n_selected,
        nogueira_stability=stability,
        selector_n_resamples=4,
        selector_requested_n_resamples=4,
        cutoff_sizes=np.array([4, 5]),
        selection_probabilities=np.array([0.8, 0.2]),
        selected_indices=np.array([0]),
        feature_metadata=(),
    )


def _nested_result():
    outer_folds = (
        _outer_fold(0, 0.75, 5, 0.80),
        _outer_fold(1, 0.85, 7, 0.90),
    )
    return NestedCVResult(
        outer_fold_results=outer_folds,
        outer_scores=np.array([0.75, 0.85]),
        outer_predictions=np.array([0, 1, 0, 1]),
        outer_decision_scores=np.array([-1.0, 1.0, -0.5, 0.5]),
        outer_test_counts=np.ones(4, dtype=int),
        mean_metrics={"balanced_accuracy": 0.80, "accuracy": 0.80},
        std_metrics={"balanced_accuracy": 0.07, "accuracy": 0.07},
        pooled_metrics={"balanced_accuracy": 0.80, "accuracy": 0.80},
        selected_counts=np.array([5, 7]),
        nogueira_stabilities=np.array([0.80, 0.90]),
        cutoff_distributions=(np.array([4, 5]), np.array([5, 6])),
        classes=np.array([0, 1]),
        scoring="balanced_accuracy",
        resample_groups=False,
        final_search=None,
        final_model=None,
        best_parameters={"consensus_threshold": 0.7, "alpha": 1.0},
        feature_metadata=(),
    )


def test_nested_result_reporting_tables_and_printing(capsys):
    result = _nested_result()

    metrics, metadata = result.summary_tables()
    assert isinstance(metrics, pd.DataFrame)
    assert list(metrics.columns) == ["Mean", "Std", "Pooled"]
    assert metadata["n_outer_folds"] == 2

    folds = result.outer_fold_table()
    assert folds["fold"].tolist() == [1, 2]
    assert folds["n_selected_features"].tolist() == [5, 7]

    returned_metrics, returned_metadata = result.print_summary(
        digits=3,
        include_outer_folds=True,
    )
    pd.testing.assert_frame_equal(returned_metrics, metrics)
    pd.testing.assert_series_equal(returned_metadata, metadata)
    output = capsys.readouterr().out
    assert "--- CLASSIFICATION METRICS ---" in output
    assert "--- OUTER FOLDS ---" in output
