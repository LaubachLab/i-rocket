"""Regression tests for the scikit-learn estimator contract.

These tests protect the scikit-learn estimator behavior required by the
leakage-free model-selection pipeline.
"""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from interp_rocket import InterpRocket


def _small_classification_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(16, 20)).astype(np.float32)
    y = np.repeat([0, 1], 8)
    X[y == 1, 8:12] += 2.0
    return X, y


def test_interp_rocket_is_recognized_as_classifier():
    assert is_classifier(InterpRocket())


def test_default_estimator_is_cloneable():
    estimator = InterpRocket()
    cloned = clone(estimator)
    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)


def test_array_alpha_range_is_accepted_and_cloneable():
    alphas = np.array([0.1, 1.0, 10.0])
    estimator = InterpRocket(alpha_range=alphas)
    cloned = clone(estimator)
    np.testing.assert_array_equal(cloned.alpha_range, alphas)


def test_unfitted_estimator_is_reported_as_unfitted():
    estimator = InterpRocket()
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)


def test_learned_attributes_are_created_only_during_fit():
    estimator = InterpRocket()
    learned_names = [
        "base_kernels_",
        "base_indices_",
        "dilations_raw_",
        "dilations_diff_",
        "biases_raw_",
        "biases_diff_",
        "classifier_",
        "scaler_",
        "classes_",
    ]
    assert all(not hasattr(estimator, name) for name in learned_names)


def test_parameter_validation_occurs_in_fit_not_init():
    estimator = InterpRocket(representations="invalid")
    X, y = _small_classification_data()
    with pytest.raises(ValueError, match="representations"):
        estimator.fit(X, y)


def test_transform_before_fit_raises_not_fitted_error():
    X, _ = _small_classification_data()
    with pytest.raises(NotFittedError):
        InterpRocket().transform(X)


def test_fit_rejects_nonfinite_input():
    X, y = _small_classification_data()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite|NaN|infinity"):
        InterpRocket(num_features=84, representations="raw").fit(X, y)


def test_fit_rejects_series_shorter_than_kernel():
    X = np.zeros((10, 8), dtype=np.float32)
    y = np.repeat([0, 1], 5)
    with pytest.raises(ValueError, match="9|timepoints|length"):
        InterpRocket(num_features=84, representations="raw").fit(X, y)


def test_transform_rejects_different_series_length():
    X, y = _small_classification_data()
    estimator = InterpRocket(num_features=84, representations="raw")
    estimator.fit(X, y)
    with pytest.raises(ValueError, match="length|timepoints|features"):
        estimator.transform(np.zeros((2, X.shape[1] + 1), dtype=np.float32))


def test_fit_is_quiet_by_default(capsys):
    X, y = _small_classification_data()
    InterpRocket(
        num_features=84,
        representations="raw",
        alpha_range=np.asarray([1.0]),
    ).fit(X, y)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_verbose_fit_reports_progress(capsys):
    X, y = _small_classification_data()
    InterpRocket(
        num_features=84,
        representations="raw",
        alpha_range=np.asarray([1.0]),
        verbose=True,
    ).fit(X, y)
    captured = capsys.readouterr()
    assert "InterpRocket.fit" in captured.out
    assert "Feature matrix" in captured.out


def test_balanced_class_weight_does_not_change_transform_fit():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(20, 20)).astype(np.float32)
    y = np.array([0] * 15 + [1] * 5)
    params = dict(
        num_features=84,
        representations="raw",
        random_state=9,
        alpha_range=np.asarray([1.0]),
    )
    unweighted = InterpRocket(**params, class_weight=None).fit(X, y)
    balanced = InterpRocket(**params, class_weight="balanced").fit(X, y)

    np.testing.assert_array_equal(unweighted.dilations_raw_, balanced.dilations_raw_)
    np.testing.assert_array_equal(unweighted.biases_raw_, balanced.biases_raw_)
    assert balanced.classifier_.class_weight == "balanced"


def test_dictionary_class_weight_is_passed_to_ridge_classifier():
    X, y = _small_classification_data()
    class_weight = {0: 1.0, 1: 2.0}
    estimator = InterpRocket(
        num_features=84,
        representations="raw",
        random_state=3,
        alpha_range=np.asarray([1.0]),
        class_weight=class_weight,
    ).fit(X, y)
    assert estimator.classifier_.class_weight == class_weight
