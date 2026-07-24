"""Tests for leakage-free nested I-ROCKET model selection."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from _irocket_selection import ResampledShrinkageSelector
from interp_rocket import InterpRocketTransform
from irocket_model_selection import (
    InnerCandidateResult,
    StableRocketClassifier,
    _select_candidate,
    nested_stability_cv,
)


class FixedCV:
    """Deterministic splitter for explicit leakage tests."""

    def __init__(self, splits):
        self.splits = tuple(
            (
                np.asarray(train, dtype=np.int64),
                np.asarray(test, dtype=np.int64),
            )
            for train, test in splits
        )

    def split(self, X, y=None, groups=None):
        del X, y, groups
        for train, test in self.splits:
            yield train.copy(), test.copy()


class RowKFold:
    """Intentionally ignores groups, without sklearn's warning side effect."""

    def __init__(self, n_splits=3, random_state=4):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        del groups
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        yield from splitter.split(X, y)


class RecordingTransformer(TransformerMixin, BaseEstimator):
    """Fast deterministic transform with a class-level fit audit log."""

    fit_log = []

    def __init__(self, n_output_features=16):
        self.n_output_features = n_output_features

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        type(self).fit_log.append(
            tuple(sorted(X[:, 0].astype(np.int64).tolist()))
        )
        self.n_features_in_ = X.shape[1]
        self.center_ = X[:, 1:].mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["center_"])
        X = np.asarray(X, dtype=np.float32)
        base = X[:, 1:] - self.center_
        candidates = np.column_stack(
            [
                X[:, 0],
                base,
                base**2,
                np.sin(base),
                np.cos(base),
            ]
        )
        repeats = int(np.ceil(self.n_output_features / candidates.shape[1]))
        return np.tile(candidates, (1, repeats))[:, : self.n_output_features]


class CountingSelector(TransformerMixin, BaseEstimator):
    """Small selector used to test orchestration rather than statistics."""

    fit_log = []

    def __init__(self, consensus_threshold=0.7, random_state=0):
        self.consensus_threshold = consensus_threshold
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        group_record = None if groups is None else tuple(np.unique(groups))
        row_ids = tuple(sorted(X[:, 0].astype(np.int64).tolist()))
        type(self).fit_log.append((row_ids, group_record))
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("CountingSelector test helper requires two classes.")
        scores = np.abs(
            X[y == classes[1]].mean(axis=0)
            - X[y == classes[0]].mean(axis=0)
        )
        ranking = np.argsort(-scores, kind="mergesort")
        probabilities = np.empty(X.shape[1], dtype=float)
        probabilities[ranking] = np.linspace(1.0, 0.1, X.shape[1])
        self.n_features_in_ = X.shape[1]
        self.selection_probabilities_ = probabilities
        self.full_scores_ = scores
        self.consensus_ranking_ = ranking
        self.nogueira_stability_ = 0.75
        self.cutoff_sizes_ = np.asarray([max(2, X.shape[1] // 3)] * 4)
        self.cutoff_improvements_ = np.asarray([0.8] * 4)
        self.selected_indices_ = self.get_support_at_threshold(
            self.consensus_threshold, indices=True
        )
        return self

    def get_support_at_threshold(self, threshold, *, indices=False):
        check_is_fitted(self, attributes=["selection_probabilities_"])
        ranking = self.consensus_ranking_
        selected = ranking[self.selection_probabilities_[ranking] >= threshold]
        if selected.size == 0:
            selected = ranking[:1]
        if indices:
            return selected.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected] = True
        return mask

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]


def _raw_binary_data(n_samples=36, seed=12):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n_samples // 2)
    X = rng.normal(size=(n_samples, 6)).astype(np.float32)
    X[:, 0] = np.arange(n_samples)  # immutable row identifier for fit auditing
    X[y == 1, 1:3] += 1.8
    return X, y


def test_one_standard_error_rule_prefers_simpler_eligible_model():
    candidates = [
        InnerCandidateResult(
            consensus_threshold=0.5,
            alpha=1.0,
            fold_scores=np.asarray([0.92, 0.88, 0.90]),
            selected_counts=np.asarray([100, 100, 100]),
            mean_score=0.90,
            std_score=0.02,
            standard_error=0.03,
            mean_selected_features=100.0,
        ),
        InnerCandidateResult(
            consensus_threshold=0.8,
            alpha=10.0,
            fold_scores=np.asarray([0.89, 0.87, 0.88]),
            selected_counts=np.asarray([20, 20, 20]),
            mean_score=0.88,
            std_score=0.01,
            standard_error=0.01,
            mean_selected_features=20.0,
        ),
        InnerCandidateResult(
            consensus_threshold=0.9,
            alpha=10.0,
            fold_scores=np.asarray([0.85, 0.85, 0.85]),
            selected_counts=np.asarray([10, 10, 10]),
            mean_score=0.85,
            std_score=0.0,
            standard_error=0.0,
            mean_selected_features=10.0,
        ),
    ]

    best_index, cutoff = _select_candidate(candidates, "one_se")
    assert cutoff == pytest.approx(0.87)
    assert best_index == 1

    empirical_index, empirical_cutoff = _select_candidate(candidates, "best")
    assert empirical_index == 0
    assert empirical_cutoff is None


def test_nested_cv_fit_log_matches_only_declared_training_partitions():
    X, y = _raw_binary_data()
    RecordingTransformer.fit_log.clear()
    CountingSelector.fit_log.clear()

    result = nested_stability_cv(
        X,
        y,
        transformer=RecordingTransformer(n_output_features=16),
        selector=CountingSelector(),
        outer_cv=3,
        inner_cv=2,
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        random_state=17,
        refit=True,
    )

    expected_fit_sets = []
    for outer_fold in result.outer_fold_results:
        for inner_fold in outer_fold.inner_search.fold_results:
            expected_fit_sets.append(tuple(sorted(inner_fold.train_indices)))
            assert np.intersect1d(
                inner_fold.train_indices, inner_fold.validation_indices
            ).size == 0
            assert set(inner_fold.train_indices).issubset(
                set(outer_fold.train_indices)
            )
            assert set(inner_fold.validation_indices).issubset(
                set(outer_fold.train_indices)
            )
        expected_fit_sets.append(tuple(sorted(outer_fold.train_indices)))
        assert np.intersect1d(
            outer_fold.train_indices, outer_fold.test_indices
        ).size == 0

    for inner_fold in result.final_search.fold_results:
        expected_fit_sets.append(tuple(sorted(inner_fold.train_indices)))
    expected_fit_sets.append(tuple(range(len(y))))

    assert RecordingTransformer.fit_log == expected_fit_sets
    assert [rows for rows, _ in CountingSelector.fit_log] == expected_fit_sets
    np.testing.assert_array_equal(result.outer_test_counts, np.ones(len(y)))
    assert result.outer_predictions.shape == y.shape
    assert np.isfinite(result.outer_scores).all()
    assert result.final_model is not None


def test_outer_test_labels_do_not_affect_that_folds_tuning_or_feature_fit():
    X, y = _raw_binary_data(n_samples=60, seed=29)
    outer = FixedCV(
        [
            (np.arange(0, 30), np.arange(30, 60)),
            (np.arange(30, 60), np.arange(0, 30)),
        ]
    )
    common = dict(
        transformer=RecordingTransformer(n_output_features=16),
        selector=CountingSelector(),
        outer_cv=outer,
        inner_cv=2,
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        random_state=13,
        refit=False,
    )

    first = nested_stability_cv(X, y, **common)
    changed_y = y.copy()
    changed_y[30:60] = 1 - changed_y[30:60]
    second = nested_stability_cv(X, changed_y, **common)

    first_fold = first.outer_fold_results[0]
    second_fold = second.outer_fold_results[0]
    assert first_fold.best_consensus_threshold == (
        second_fold.best_consensus_threshold
    )
    assert first_fold.best_alpha == second_fold.best_alpha
    np.testing.assert_array_equal(
        first_fold.selected_indices, second_fold.selected_indices
    )
    np.testing.assert_allclose(
        first_fold.selection_probabilities,
        second_fold.selection_probabilities,
    )
    np.testing.assert_array_equal(
        first_fold.predictions, second_fold.predictions
    )
    for first_candidate, second_candidate in zip(
        first_fold.inner_search.candidates,
        second_fold.inner_search.candidates,
    ):
        np.testing.assert_allclose(
            first_candidate.fold_scores, second_candidate.fold_scores
        )



def test_threshold_and_alpha_grid_reuses_one_selector_fit_per_split():
    X, y = _raw_binary_data()
    CountingSelector.fit_log.clear()

    nested_stability_cv(
        X,
        y,
        transformer=RecordingTransformer(n_output_features=16),
        selector=CountingSelector(),
        outer_cv=3,
        inner_cv=2,
        consensus_thresholds=(0.4, 0.6, 0.8, 1.0),
        classifier_alphas=(0.01, 0.1, 1.0, 10.0),
        random_state=5,
        refit=False,
    )

    # Three outer folds, each with two inner fits plus one outer refit.
    assert len(CountingSelector.fit_log) == 3 * (2 + 1)


def test_default_group_splitters_keep_groups_disjoint_everywhere():
    rng = np.random.default_rng(25)
    n_groups = 12
    rows_per_group = 6
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    y = np.tile([0, 0, 0, 1, 1, 1], n_groups)
    X = rng.normal(size=(groups.size, 6)).astype(np.float32)
    X[:, 0] = np.arange(groups.size)
    X[y == 1, 1:3] += 1.5

    result = nested_stability_cv(
        X,
        y,
        groups=groups,
        transformer=RecordingTransformer(n_output_features=16),
        selector=CountingSelector(),
        outer_cv=3,
        inner_cv=2,
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        random_state=9,
        refit=False,
    )

    for outer_fold in result.outer_fold_results:
        assert np.intersect1d(
            np.unique(groups[outer_fold.train_indices]),
            np.unique(groups[outer_fold.test_indices]),
        ).size == 0
        for inner_fold in outer_fold.inner_search.fold_results:
            assert np.intersect1d(
                np.unique(groups[inner_fold.train_indices]),
                np.unique(groups[inner_fold.validation_indices]),
            ).size == 0


def test_group_leakage_in_user_splitter_is_rejected():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(6), 4)
    y = np.tile([0, 0, 1, 1], 6)
    X = rng.normal(size=(len(y), 6)).astype(np.float32)
    X[:, 0] = np.arange(len(y))

    with pytest.raises(ValueError, match="leaks.*group"):
        nested_stability_cv(
            X,
            y,
            groups=groups,
            transformer=RecordingTransformer(),
            selector=CountingSelector(),
            outer_cv=RowKFold(n_splits=3, random_state=4),
            inner_cv=2,
            consensus_thresholds=(0.5,),
            classifier_alphas=(1.0,),
            refit=False,
        )


def test_outer_cv_must_be_a_complete_nonrepeated_partition():
    class IncompleteCV:
        def split(self, X, y):
            yield np.arange(12, len(y)), np.arange(0, 12)
            yield np.r_[0:12, 24:len(y)], np.arange(12, 24)

    X, y = _raw_binary_data()
    with pytest.raises(ValueError, match="partition.*exactly once"):
        nested_stability_cv(
            X,
            y,
            transformer=RecordingTransformer(),
            selector=CountingSelector(),
            outer_cv=IncompleteCV(),
            inner_cv=2,
            consensus_thresholds=(0.5,),
            classifier_alphas=(1.0,),
            refit=False,
        )


def test_stable_rocket_classifier_is_cloneable_and_checks_fitted_state():
    estimator = StableRocketClassifier(
        transformer=RecordingTransformer(),
        selector=CountingSelector(),
        consensus_threshold=0.8,
        alpha=1.0,
    )
    cloned = clone(estimator)
    assert cloned.alpha == estimator.alpha
    assert cloned.consensus_threshold == estimator.consensus_threshold
    assert cloned.transformer.get_params() == estimator.transformer.get_params()
    assert cloned.selector.get_params() == estimator.selector.get_params()
    X, _ = _raw_binary_data()
    with pytest.raises(NotFittedError):
        cloned.predict(X)


def test_real_transform_selector_nested_integration():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(48, 24)).astype(np.float32)
    y = np.repeat([0, 1], 24)
    X[y == 1, 8:13] += 1.4

    result = nested_stability_cv(
        X,
        y,
        transformer=InterpRocketTransform(
            num_features=84,
            max_dilations_per_kernel=2,
            representations="raw",
            random_state=5,
        ),
        selector=ResampledShrinkageSelector(
            n_resamples=6,
            sample_fraction=0.6,
            consensus_threshold=0.5,
            cutoff_min_size=3,
            min_features=2,
            random_state=5,
        ),
        outer_cv=3,
        inner_cv=2,
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        random_state=5,
        refit=True,
    )

    assert result.outer_scores.shape == (3,)
    assert np.isfinite(result.outer_scores).all()
    assert result.pooled_metrics["balanced_accuracy"] >= 0.6
    assert result.final_model.n_transform_features_ == 336
    assert result.final_model.n_selected_features_ >= 2
    assert len(result.feature_metadata) == result.final_model.n_selected_features_
    assert all("padding_mode" in item for item in result.feature_metadata)
    assert np.isfinite(result.nogueira_stabilities).all()
    assert result.summary()["n_outer_folds"] == 3


def test_group_disjoint_validation_can_use_row_level_selector_resampling():
    rng = np.random.default_rng(63)
    groups = np.repeat(np.arange(12), 6)
    y = np.tile([0, 0, 0, 1, 1, 1], 12)
    X = rng.normal(size=(groups.size, 6)).astype(np.float32)
    X[:, 0] = np.arange(groups.size)
    X[y == 1, 1:3] += 1.3
    CountingSelector.fit_log.clear()

    result = nested_stability_cv(
        X,
        y,
        groups=groups,
        transformer=RecordingTransformer(n_output_features=16),
        selector=CountingSelector(),
        outer_cv=3,
        inner_cv=2,
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        random_state=9,
        resample_groups=False,
        refit=False,
    )

    assert result.resample_groups is False
    assert all(group_record is None for _, group_record in CountingSelector.fit_log)
    for outer_fold in result.outer_fold_results:
        assert np.intersect1d(
            np.unique(groups[outer_fold.train_indices]),
            np.unique(groups[outer_fold.test_indices]),
        ).size == 0
