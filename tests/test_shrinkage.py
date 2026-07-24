"""Numerical and estimator tests for I-ROCKET's internal shrinkage scoring."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_selection import screen_features, shrinkage_t, shrinkage_t_ovr
from _irocket_selection import ShrinkageFeatureSelector


def _manual_shrinkage_t(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    active = X.var(axis=0) > 0
    scores = np.zeros(X.shape[1], dtype=float)
    Xa = X[:, active]
    masks = [y == cls for cls in classes]
    ns = [int(mask.sum()) for mask in masks]
    means = [Xa[mask].mean(axis=0) for mask in masks]
    variances = [Xa[mask].var(axis=0, ddof=1) for mask in masks]
    pooled = (
        (ns[0] - 1) * variances[0] + (ns[1] - 1) * variances[1]
    ) / (sum(ns) - 2)
    grand = Xa.mean(axis=0)
    fourth = np.mean((Xa - grand) ** 4, axis=0)
    n = sum(ns)
    variance_of_variance = (1.0 / n) * (
        fourth - pooled**2 * (n - 3) / (n - 1)
    )
    variance_of_variance = np.maximum(variance_of_variance, 0.0)
    target = float(np.median(pooled))
    denominator = float(np.sum((pooled - target) ** 2))
    shrinkage = 0.0 if denominator < 1e-12 else float(
        np.clip(np.sum(variance_of_variance) / denominator, 0.0, 1.0)
    )
    shrunk = np.maximum((1.0 - shrinkage) * pooled + shrinkage * target, 1e-12)
    standard_error = np.sqrt(shrunk * (1.0 / ns[0] + 1.0 / ns[1]))
    scores[active] = (means[1] - means[0]) / standard_error
    return scores, shrinkage, target


def _binary_data():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 8))
    y = np.repeat([0, 1], 20)
    X[y == 1, 2] += 2.5
    X[y == 1, 5] -= 1.5
    X[:, 7] = 3.0
    return X, y


def test_shrinkage_t_matches_independent_formula():
    X, y = _binary_data()
    expected_scores, expected_lambda, expected_target = _manual_shrinkage_t(X, y)
    result = shrinkage_t(X, y)
    np.testing.assert_allclose(result.scores, expected_scores, rtol=1e-12, atol=1e-12)
    assert result.metadata["lambda_var"] == pytest.approx(expected_lambda)
    assert result.metadata["target_variance"] == pytest.approx(expected_target)
    np.testing.assert_array_equal(
        result.ranking, np.argsort(np.abs(expected_scores))[::-1]
    )


def test_constant_columns_receive_zero_scores():
    X, y = _binary_data()
    result = shrinkage_t(X, y)
    assert result.scores[7] == 0.0
    assert not result.metadata["active_mask"][7]


def test_signal_features_rank_near_the_top():
    X, y = _binary_data()
    top = set(shrinkage_t(X, y).ranking[:3].tolist())
    assert 2 in top
    assert 5 in top


def test_multiclass_ovr_returns_one_score_per_feature():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(45, 6))
    y = np.repeat([0, 1, 2], 15)
    X[y == 0, 0] += 2.0
    X[y == 1, 1] += 2.0
    X[y == 2, 2] += 2.0
    result = shrinkage_t_ovr(X, y)
    assert result.scores.shape == (6,)
    assert result.metadata["t_scores_ovr"].shape == (3, 6)
    assert set(result.ranking[:3]) == {0, 1, 2}


def test_screen_features_selects_top_k_from_internal_implementation():
    X, y = _binary_data()
    result = screen_features(X, y, top_k=3)
    expected = shrinkage_t(X, y).ranking[:3]
    np.testing.assert_array_equal(result.selected, expected)


def test_selector_is_cloneable_and_works_in_pipeline():
    X, y = _binary_data()
    selector = ShrinkageFeatureSelector(top_k=4)
    cloned = clone(selector)
    assert cloned.get_params(deep=False) == selector.get_params(deep=False)
    pipeline = Pipeline(
        [
            ("selector", selector),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    pipeline.fit(X, y)
    assert pipeline.named_steps["selector"].get_support(indices=True).size == 4
    assert pipeline.score(X, y) >= 0.8


@pytest.mark.parametrize(
    "bad_X,bad_y,match",
    [
        (np.zeros((3, 2, 1)), np.asarray([0, 1, 1]), "2D|dimensional"),
        (np.zeros((3, 2)), np.asarray([0, 1]), "different|inconsistent"),
        (np.zeros((3, 2)), np.asarray([0, 0, 0]), "2 classes|two classes"),
    ],
)
def test_shrinkage_input_validation(bad_X, bad_y, match):
    with pytest.raises(ValueError, match=match):
        shrinkage_t(bad_X, bad_y)


def test_internal_validation_does_not_promote_entire_float32_matrix():
    from _irocket_selection._utils import as_arrays

    X, y = _binary_data()
    X = X.astype(np.float32)
    validated_X, validated_y = as_arrays(X, y)
    assert validated_X.dtype == np.float32
    assert np.shares_memory(validated_X, X)
    np.testing.assert_array_equal(validated_y, y)


def test_float32_high_dimensional_shrinkage_smoke():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(24, 5000)).astype(np.float32)
    y = np.repeat([0, 1], 12)
    X[y == 1, :5] += 1.5
    result = shrinkage_t(X, y)
    assert result.scores.shape == (5000,)
    assert np.isfinite(result.scores).all()


def test_selector_refit_removes_stale_cutoff_attribute():
    from _irocket_selection import ShrinkageFeatureSelector

    X, y = _binary_data()
    selector = ShrinkageFeatureSelector(
        threshold="segmented", cutoff_min_size=2
    ).fit(X, y)
    assert hasattr(selector, "cutoff_idx_")

    selector.set_params(threshold="top_k", top_k=3)
    selector.fit(X, y)
    assert not hasattr(selector, "cutoff_idx_")
    assert selector.get_support(indices=True).size == 3


def test_selector_preserves_numeric_input_dtype_on_transform():
    from _irocket_selection import ShrinkageFeatureSelector

    X, y = _binary_data()
    X = X.astype(np.float32)
    selector = ShrinkageFeatureSelector(top_k=4).fit(X, y)
    transformed = selector.transform(X)
    assert transformed.dtype == np.float32
