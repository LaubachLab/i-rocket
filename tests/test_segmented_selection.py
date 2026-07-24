"""Tests for segmented cutoffs and resampled consensus selection."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _irocket_selection import (
    ResampledShrinkageSelector,
    ShrinkageFeatureSelector,
    nogueira_stability,
    segmented_cutoff,
)
from interp_rocket import InterpRocketTransform
from stability_nogueira import selection_stability


def _line_sse(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return float(residual @ residual)


def _brute_force_breakpoint(y, min_size):
    x = np.linspace(0.0, 1.0, len(y))
    candidates = range(min_size, len(y) - min_size + 1)
    losses = [
        _line_sse(x[:cut], y[:cut]) + _line_sse(x[cut:], y[cut:])
        for cut in candidates
    ]
    return list(candidates)[int(np.argmin(losses))], min(losses)


def _binary_feature_data(seed=4):
    rng = np.random.default_rng(seed)
    n_samples = 160
    n_features = 60
    y = np.repeat([0, 1], n_samples // 2)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    X[y == 1, 0] += 2.5
    X[y == 1, 1] -= 2.0
    X[y == 1, 2] += 1.4
    return X, y


def test_segmented_cutoff_recovers_known_two_line_breakpoint():
    n_features = 100
    positions = np.arange(n_features)
    curve = np.empty(n_features, dtype=float)
    curve[:20] = 10.0 - 0.25 * positions[:20]
    curve[20:] = 5.0 - 0.01 * (positions[20:] - 20)

    result = segmented_cutoff(curve, min_size=5)

    assert result.breakpoint == 20
    assert result.cutoff_idx == 19
    assert result.n_selected == 20
    np.testing.assert_array_equal(result.selected, np.arange(20))
    assert result.relative_improvement > 0.99
    assert result.tail_slope_ratio < 0.05


def test_segmented_cutoff_matches_brute_force_ordinary_least_squares():
    rng = np.random.default_rng(19)
    curve = np.sort(rng.gamma(shape=2.0, scale=1.0, size=75))[::-1]
    expected_breakpoint, expected_loss = _brute_force_breakpoint(curve, 6)

    result = segmented_cutoff(curve, min_size=6)

    assert result.breakpoint == expected_breakpoint
    assert result.segmented_sse == pytest.approx(expected_loss, abs=1e-10)


def test_segmented_cutoff_ranks_signed_scores_by_absolute_magnitude():
    scores = np.asarray([1.0, -8.0, 6.0, -4.0, 2.0, 0.5, 0.4, 0.3, 0.2, 0.1])
    result = segmented_cutoff(scores, min_size=2)
    np.testing.assert_array_equal(result.ranking[:4], [1, 2, 3, 4])
    assert set(result.selected).issubset(set(result.ranking))


def test_squared_scores_are_an_explicit_sensitivity_option():
    curve = np.asarray([9.0, 7.0, 5.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8])
    absolute_result = segmented_cutoff(curve, min_size=2, score_power=1.0)
    squared_result = segmented_cutoff(curve, min_size=2, score_power=2.0)
    np.testing.assert_array_equal(absolute_result.ranking, squared_result.ranking)
    np.testing.assert_allclose(squared_result.ranked_scores, curve**2)
    assert squared_result.score_power == 2.0


@pytest.mark.parametrize(
    "scores,kwargs,match",
    [
        (np.ones(12), {}, "constant"),
        (np.arange(8.0), {"min_size": 5}, r"2 \* min_size"),
        (np.asarray([1.0, np.nan, 0.0, 0.0]), {"min_size": 2}, "finite"),
    ],
)
def test_segmented_cutoff_rejects_undefined_inputs(scores, kwargs, match):
    with pytest.raises(ValueError, match=match):
        segmented_cutoff(scores, **kwargs)


def test_segmented_cutoff_validates_supplied_ranking():
    curve = np.arange(12.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="permutation"):
        segmented_cutoff(
            curve,
            ranked_indices=np.asarray([0] * 12),
            min_size=3,
        )


def test_nogueira_stability_is_one_for_identical_nontrivial_sets():
    matrix = np.tile(np.asarray([1, 1, 0, 0]), (8, 1))
    result = nogueira_stability(matrix)
    assert result.stability == pytest.approx(1.0)
    assert result.mean_selected == pytest.approx(2.0)
    np.testing.assert_allclose(result.selection_probabilities, [1, 1, 0, 0])


def test_nogueira_stability_matches_manual_finite_sample_example():
    matrix = np.eye(4, dtype=int)
    result = nogueira_stability(matrix)
    assert result.stability == pytest.approx(-1.0 / 3.0)
    np.testing.assert_array_equal(result.selected_counts, np.ones(4, dtype=int))


def test_nogueira_stability_matches_direct_definition_on_random_matrix():
    rng = np.random.default_rng(23)
    matrix = (rng.random((17, 41)) < 0.18).astype(int)
    probabilities = matrix.mean(axis=0)
    mean_selected = matrix.sum(axis=1).mean()
    denominator = (mean_selected / matrix.shape[1]) * (
        1.0 - mean_selected / matrix.shape[1]
    )
    expected = 1.0 - (
        matrix.shape[0]
        / (matrix.shape[0] - 1.0)
        * np.mean(probabilities * (1.0 - probabilities))
        / denominator
    )
    assert nogueira_stability(matrix).stability == pytest.approx(expected)


def test_legacy_stability_wrapper_uses_same_point_estimate():
    matrix = np.asarray(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
        ]
    )
    direct = nogueira_stability(matrix)
    wrapped = selection_stability(matrix)
    assert wrapped["stability"] == pytest.approx(direct.stability)
    np.testing.assert_allclose(
        wrapped["selection_probabilities"], direct.selection_probabilities
    )


@pytest.mark.parametrize(
    "matrix,match",
    [
        (np.asarray([[1, 0, 1]]), "two"),
        (np.asarray([[1, 0], [0, 2]]), "zeros and ones"),
        (np.asarray([[1.0, 0.0], [np.nan, 1.0]]), "NaN"),
        (np.zeros((3, 4), dtype=int), "undefined"),
        (np.ones((3, 4), dtype=int), "undefined"),
    ],
)
def test_nogueira_validation(matrix, match):
    with pytest.raises(ValueError, match=match):
        nogueira_stability(matrix)


def test_single_fit_segmented_selector_exposes_breakpoint_diagnostics():
    X, y = _binary_feature_data()
    selector = ShrinkageFeatureSelector(
        threshold="segmented",
        cutoff_min_size=3,
    ).fit(X, y)
    assert selector.breakpoint_ == selector.cutoff_idx_ + 1
    assert selector.cutoff_result_.relative_improvement > 0.0
    assert {0, 1, 2}.issubset(set(selector.selected_indices_))


def test_resampled_selector_recovers_stable_signal_features():
    X, y = _binary_feature_data()
    selector = ResampledShrinkageSelector(
        n_resamples=20,
        sample_fraction=0.5,
        consensus_threshold=0.7,
        cutoff_min_size=3,
        random_state=42,
    ).fit(X, y)

    assert selector.selection_matrix_.shape == (20, X.shape[1])
    np.testing.assert_array_equal(
        selector.selection_matrix_.sum(axis=1), selector.cutoff_sizes_
    )
    np.testing.assert_allclose(
        selector.selection_probabilities_,
        selector.selection_matrix_.mean(axis=0),
    )
    assert {0, 1, 2}.issubset(set(selector.selected_indices_))
    assert np.all(selector.selection_probabilities_[:3] >= 0.7)
    assert selector.nogueira_stability_ > 0.7
    assert np.all(selector.cutoff_improvements_ > 0.0)
    assert selector.transform(X).shape == (X.shape[0], selector.n_selected_)

    strict_indices = selector.get_support_at_threshold(1.0, indices=True)
    relaxed_indices = selector.get_support_at_threshold(0.5, indices=True)
    assert strict_indices.size <= relaxed_indices.size
    assert selector.get_support_at_threshold(0.7).dtype == bool


def test_resampled_selector_is_cloneable_deterministic_and_refittable():
    X, y = _binary_feature_data()
    selector = ResampledShrinkageSelector(
        n_resamples=12,
        consensus_threshold=0.6,
        cutoff_min_size=3,
        random_state=7,
    )
    cloned = clone(selector)
    assert cloned.get_params(deep=False) == selector.get_params(deep=False)

    first = selector.fit(X, y)
    first_matrix = first.selection_matrix_.copy()
    first_selected = first.selected_indices_.copy()
    second = clone(selector).fit(X, y)
    np.testing.assert_array_equal(first_matrix, second.selection_matrix_)
    np.testing.assert_array_equal(first_selected, second.selected_indices_)

    selector.set_params(consensus_threshold=1.0, min_features=5)
    selector.fit(X, y)
    assert selector.n_selected_ >= 5
    assert selector.support_mask_.sum() == selector.n_selected_


def test_group_aware_resampling_keeps_groups_intact():
    rng = np.random.default_rng(9)
    n_groups = 12
    rows_per_group = 10
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    y = np.tile(np.repeat([0, 1], rows_per_group // 2), n_groups)
    X = rng.normal(size=(groups.size, 40)).astype(np.float32)
    X[y == 1, :2] += 1.5

    selector = ResampledShrinkageSelector(
        n_resamples=6,
        sample_fraction=0.5,
        consensus_threshold=0.5,
        cutoff_min_size=3,
        random_state=11,
        store_subsample_indices=True,
    ).fit(X, y, groups=groups)

    for row_indices in selector.resample_indices_:
        retained_groups = np.unique(groups[row_indices])
        for group in retained_groups:
            np.testing.assert_array_equal(
                np.flatnonzero(groups == group),
                np.intersect1d(np.flatnonzero(groups == group), row_indices),
            )


def test_transform_and_resampled_selector_run_inside_cross_validation():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(36, 24)).astype(np.float32)
    y = np.repeat([0, 1], 18)
    X[y == 1, 8:13] += 1.4

    pipeline = Pipeline(
        [
            (
                "rocket",
                InterpRocketTransform(
                    num_features=84,
                    max_dilations_per_kernel=2,
                    representations="raw",
                    random_state=5,
                ),
            ),
            (
                "selector",
                ResampledShrinkageSelector(
                    n_resamples=6,
                    sample_fraction=0.6,
                    consensus_threshold=0.5,
                    cutoff_min_size=3,
                    random_state=5,
                ),
            ),
            ("scale", StandardScaler()),
            ("classifier", RidgeClassifier(alpha=1.0)),
        ]
    )
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5),
    )
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert scores.mean() >= 0.65


def test_group_resampling_uses_all_valid_unique_subsets_when_request_is_larger():
    rng = np.random.default_rng(101)
    n_groups = 6
    rows_per_group = 8
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    y = np.tile(np.asarray([0, 0, 0, 0, 1, 1, 1, 1]), n_groups)
    X = rng.normal(size=(groups.size, 40)).astype(np.float32)
    X[y == 1, :3] += 1.2

    selector = ResampledShrinkageSelector(
        n_resamples=50,
        sample_fraction=0.5,
        consensus_threshold=0.5,
        cutoff_min_size=3,
        random_state=8,
        store_subsample_indices=True,
    )
    with pytest.warns(RuntimeWarning, match="valid unique whole-group"):
        selector.fit(X, y, groups=groups)

    # Six choose three gives exactly 20 unique half-group subsets.
    assert selector.requested_n_resamples_ == 50
    assert selector.n_resamples_ == 20
    assert selector.selection_matrix_.shape[0] == 20
    observed = {
        tuple(np.unique(groups[row_indices]).tolist())
        for row_indices in selector.resample_indices_
    }
    assert len(observed) == 20
