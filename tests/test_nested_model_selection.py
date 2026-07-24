"""Tests for the leakage-free nested I-ROCKET model-selection engine."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.model_selection import StratifiedKFold

from _irocket_selection import ResampledShrinkageSelector
from interp_rocket import InterpRocketTransform
from irocket_model_selection import (
    InnerCandidateResult,
    StableRocketClassifier,
    _select_candidate,
    nested_stability_cv,
)


class IdentityTransformer(TransformerMixin, BaseEstimator):
    """Small cloneable transformer used to isolate nested-CV behavior."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = int(X.shape[1])
        self.n_timepoints_in_ = int(X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature count changed")
        return X


class RecordingTransformer(IdentityTransformer):
    """Record row identifiers received by every cloned fit."""

    fit_rows = []

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        type(self).fit_rows.append(tuple(sorted(X[:, 0].astype(int).tolist())))
        return super().fit(X, y)


class RecordingSelector(ResampledShrinkageSelector):
    """Record transformed row identifiers received by selector fitting."""

    fit_rows = []

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        type(self).fit_rows.append(tuple(sorted(X[:, 0].astype(int).tolist())))
        return super().fit(X, y, groups=groups)


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


def _time_series_data(seed=44, n_samples=54):
    rng = np.random.default_rng(seed)
    y = np.tile(np.asarray([0, 1]), n_samples // 2)
    X = rng.normal(size=(n_samples, 24)).astype(np.float32)
    X[y == 1, 8:14] += 1.4
    return X, y


def _selector(**overrides):
    parameters = dict(
        n_resamples=4,
        sample_fraction=0.65,
        consensus_threshold=0.5,
        cutoff_min_size=3,
        min_features=2,
        random_state=None,
    )
    parameters.update(overrides)
    return ResampledShrinkageSelector(**parameters)


def test_stable_classifier_is_cloneable_and_exposes_selected_features():
    X, y = _time_series_data(n_samples=40)
    estimator = StableRocketClassifier(
        transformer=InterpRocketTransform(
            num_features=84,
            max_dilations_per_kernel=2,
            representations="raw",
            random_state=2,
        ),
        selector=_selector(),
        consensus_threshold=0.7,
        alpha=1.0,
    )
    cloned = clone(estimator)
    assert is_classifier(estimator)
    assert cloned.consensus_threshold == estimator.consensus_threshold
    assert cloned.alpha == estimator.alpha
    assert isinstance(cloned.transformer, InterpRocketTransform)
    assert isinstance(cloned.selector, ResampledShrinkageSelector)

    estimator.fit(X, y)
    predictions = estimator.predict(X)
    assert predictions.shape == y.shape
    assert estimator.n_selected_features_ == estimator.get_support(indices=True).size
    assert estimator.get_support().sum() == estimator.n_selected_features_
    decoded = estimator.get_selected_feature_metadata()
    assert len(decoded) == estimator.n_selected_features_
    assert all("selection_probability" in item for item in decoded)


def test_nested_cv_runs_outer_evaluation_and_full_refit():
    X, y = _time_series_data()
    thresholds = (0.5, 0.8)
    alphas = (0.1, 1.0)
    result = nested_stability_cv(
        X,
        y,
        transformer=InterpRocketTransform(
            num_features=84,
            max_dilations_per_kernel=2,
            representations="raw",
            random_state=5,
        ),
        selector=_selector(),
        consensus_thresholds=thresholds,
        classifier_alphas=alphas,
        outer_cv=3,
        inner_cv=2,
        random_state=8,
        refit=True,
    )

    assert len(result.outer_fold_results) == 3
    assert result.outer_scores.shape == (3,)
    assert np.isfinite(result.outer_scores).all()
    assert result.outer_predictions.shape == y.shape
    np.testing.assert_array_equal(
        result.outer_test_counts, np.ones(len(y), dtype=int)
    )
    assert {
        fold.best_consensus_threshold for fold in result.outer_fold_results
    }.issubset(thresholds)
    assert {
        fold.best_alpha for fold in result.outer_fold_results
    }.issubset(alphas)
    assert np.all(result.selected_counts > 0)
    assert np.isfinite(result.nogueira_stabilities).all()
    assert result.final_search is not None
    assert result.final_model is not None
    assert result.best_parameters == {
        "consensus_threshold": result.final_search.best_consensus_threshold,
        "alpha": result.final_search.best_alpha,
    }
    assert result.final_model.n_selected_features_ > 0
    assert result.final_model.predict(X[:4]).shape == (4,)

    for fold in result.outer_fold_results:
        assert len(fold.inner_search.candidates) == 4
        assert len(fold.inner_search.fold_results) == 2
        assert fold.selection_probabilities.ndim == 1
        assert len(fold.feature_metadata) == fold.n_selected_features


def test_outer_test_labels_do_not_change_tuning_or_fitted_feature_set():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(60, 30)).astype(np.float32)
    y = np.concatenate(
        [
            np.repeat([0, 1], 20),
            np.repeat([0, 1], 10),
        ]
    )
    X[y == 1, 4:8] += 1.0

    outer = FixedCV(
        [
            (np.arange(40), np.arange(40, 60)),
            (np.arange(40, 60), np.arange(40)),
        ]
    )
    inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
    common = dict(
        transformer=IdentityTransformer(),
        selector=_selector(),
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        outer_cv=outer,
        inner_cv=inner,
        random_state=19,
        refit=False,
    )
    first = nested_stability_cv(X, y, **common)
    changed_y = y.copy()
    changed_y[40:60] = changed_y[40:60][::-1]
    second = nested_stability_cv(X, changed_y, **common)

    first_fold = first.outer_fold_results[0]
    second_fold = second.outer_fold_results[0]
    for first_candidate, second_candidate in zip(
        first_fold.inner_search.candidates,
        second_fold.inner_search.candidates,
    ):
        np.testing.assert_allclose(
            first_candidate.fold_scores, second_candidate.fold_scores
        )
    assert (
        first_fold.best_consensus_threshold
        == second_fold.best_consensus_threshold
    )
    assert first_fold.best_alpha == second_fold.best_alpha
    np.testing.assert_array_equal(
        first_fold.selected_indices, second_fold.selected_indices
    )
    np.testing.assert_allclose(
        first_fold.selection_probabilities,
        second_fold.selection_probabilities,
    )
    np.testing.assert_array_equal(first_fold.predictions, second_fold.predictions)
    assert first_fold.primary_score != second_fold.primary_score


def test_transform_and_selector_never_fit_on_outer_test_rows():
    RecordingTransformer.fit_rows = []
    RecordingSelector.fit_rows = []
    rng = np.random.default_rng(3)
    X = rng.normal(size=(48, 24)).astype(np.float32)
    X[:, 0] = np.arange(len(X), dtype=np.float32)
    y = np.tile([0, 1], 24)
    X[y == 1, 5:9] += 1.0

    outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=4)
    inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    result = nested_stability_cv(
        X,
        y,
        transformer=RecordingTransformer(),
        selector=RecordingSelector(
            n_resamples=4,
            sample_fraction=0.65,
            consensus_threshold=0.5,
            cutoff_min_size=3,
            min_features=2,
            random_state=None,
        ),
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        outer_cv=outer,
        inner_cv=inner,
        random_state=6,
        refit=False,
    )

    # Two inner fits plus one outer-training refit for each outer fold.
    assert len(RecordingTransformer.fit_rows) == 6
    assert len(RecordingSelector.fit_rows) == 6
    for fold_index, fold in enumerate(result.outer_fold_results):
        train_ids = set(X[fold.train_indices, 0].astype(int).tolist())
        test_ids = set(X[fold.test_indices, 0].astype(int).tolist())
        start = 3 * fold_index
        for recorded in RecordingTransformer.fit_rows[start : start + 3]:
            recorded = set(recorded)
            assert recorded.issubset(train_ids)
            assert recorded.isdisjoint(test_ids)
        for recorded in RecordingSelector.fit_rows[start : start + 3]:
            recorded = set(recorded)
            assert recorded.issubset(train_ids)
            assert recorded.isdisjoint(test_ids)
        assert set(RecordingTransformer.fit_rows[start + 2]) == train_ids
        assert set(RecordingSelector.fit_rows[start + 2]) == train_ids


def test_group_aware_nested_splits_keep_units_disjoint():
    rng = np.random.default_rng(12)
    n_groups = 12
    rows_per_group = 6
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    y = np.tile(np.asarray([0, 0, 0, 1, 1, 1]), n_groups)
    X = rng.normal(size=(groups.size, 24)).astype(np.float32)
    X[y == 1, 4:8] += 1.2

    result = nested_stability_cv(
        X,
        y,
        groups=groups,
        transformer=IdentityTransformer(),
        selector=_selector(sample_fraction=0.5),
        consensus_thresholds=(0.5, 0.8),
        classifier_alphas=(0.1, 1.0),
        outer_cv=3,
        inner_cv=2,
        random_state=22,
        refit=False,
    )

    for fold in result.outer_fold_results:
        outer_train_groups = set(groups[fold.train_indices])
        outer_test_groups = set(groups[fold.test_indices])
        assert outer_train_groups.isdisjoint(outer_test_groups)
        for inner_fold in fold.inner_search.fold_results:
            assert set(groups[inner_fold.train_indices]).isdisjoint(
                set(groups[inner_fold.validation_indices])
            )


def test_one_standard_error_rule_prefers_sparse_strongly_regularized_candidate():
    candidates = []
    means = {
        (0.5, 0.1): 0.900,
        (0.5, 1.0): 0.895,
        (0.5, 10.0): 0.890,
        (0.8, 0.1): 0.880,
        (0.8, 1.0): 0.880,
        (0.8, 10.0): 0.880,
    }
    for threshold, selected_count in ((0.5, 100.0), (0.8, 20.0)):
        for alpha in (0.1, 1.0, 10.0):
            mean = means[(threshold, alpha)]
            candidates.append(
                InnerCandidateResult(
                    consensus_threshold=threshold,
                    alpha=alpha,
                    fold_scores=np.asarray([mean, mean, mean]),
                    selected_counts=np.asarray([selected_count] * 3),
                    mean_score=mean,
                    std_score=0.0,
                    standard_error=0.03 if (threshold, alpha) == (0.5, 0.1) else 0.0,
                    mean_selected_features=selected_count,
                )
            )

    selected, cutoff = _select_candidate(candidates, "one_se")
    assert candidates[selected].consensus_threshold == 0.8
    assert candidates[selected].alpha == 10.0
    assert cutoff == pytest.approx(0.87)


def test_nested_cv_validates_grids_and_split_class_coverage():
    X, y = _time_series_data(n_samples=40)
    with pytest.raises(ValueError, match="duplicate"):
        nested_stability_cv(
            X,
            y,
            transformer=IdentityTransformer(),
            selector=_selector(),
            consensus_thresholds=(0.5, 0.5),
            classifier_alphas=(1.0,),
            outer_cv=2,
            inner_cv=2,
            refit=False,
        )

    even = np.arange(0, 40, 2)
    odd = np.arange(1, 40, 2)
    bad_outer = FixedCV([(even, odd), (odd, even)])
    with pytest.raises(ValueError, match="every class"):
        nested_stability_cv(
            X,
            y,
            transformer=IdentityTransformer(),
            selector=_selector(),
            consensus_thresholds=(0.5,),
            classifier_alphas=(1.0,),
            outer_cv=bad_outer,
            inner_cv=2,
            refit=False,
        )


def test_stable_classifier_uses_declared_ridge_solver():
    X, y = _time_series_data(n_samples=40)
    estimator = StableRocketClassifier(
        transformer=IdentityTransformer(),
        selector=_selector(),
        consensus_threshold=0.7,
        alpha=1.0,
        ridge_solver="lsqr",
        random_state=9,
    ).fit(X, y)

    assert estimator.ridge_solver == "lsqr"
    assert estimator.classifier_.solver == "lsqr"


def test_nested_cv_rejects_empty_ridge_solver():
    X, y = _time_series_data(n_samples=40)
    with pytest.raises(TypeError, match="ridge_solver"):
        nested_stability_cv(
            X,
            y,
            transformer=IdentityTransformer(),
            selector=_selector(),
            consensus_thresholds=(0.5,),
            classifier_alphas=(1.0,),
            outer_cv=2,
            inner_cv=2,
            ridge_solver="",
            refit=False,
        )


class InvalidIndexSelector:
    """Minimal selector helper returning a prescribed malformed index array."""

    def __init__(self, indices):
        self.indices = indices

    def get_support_at_threshold(self, threshold, *, indices=False):
        del threshold
        assert indices
        return self.indices


@pytest.mark.parametrize(
    "bad_indices, error_type, message",
    [
        ([0.5, 1.0], TypeError, "integer feature indices"),
        ([1, 1], ValueError, "duplicate"),
        ([-1, 1], ValueError, "outside"),
        ([0, 99], ValueError, "outside"),
    ],
)
def test_selector_index_validation_rejects_malformed_outputs(
    bad_indices, error_type, message
):
    from irocket_model_selection import _selector_indices

    with pytest.raises(error_type, match=message):
        _selector_indices(
            InvalidIndexSelector(bad_indices),
            0.7,
            n_features=10,
        )


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("consensus_thresholds", (True,)),
        ("classifier_alphas", (False,)),
    ],
)
def test_nested_cv_rejects_boolean_grid_values(keyword, value):
    X, y = _time_series_data(n_samples=40)
    parameters = dict(
        transformer=IdentityTransformer(),
        selector=_selector(),
        consensus_thresholds=(0.5,),
        classifier_alphas=(1.0,),
        outer_cv=2,
        inner_cv=2,
        refit=False,
    )
    parameters[keyword] = value
    with pytest.raises(TypeError, match=keyword):
        nested_stability_cv(X, y, **parameters)
