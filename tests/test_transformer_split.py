"""Tests for the transformer/classifier separation introduced in Patch 03."""

import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interp_rocket import InterpRocket, InterpRocketTransform


def _data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(24, 24)).astype(np.float32)
    y = np.repeat([0, 1], 12)
    X[y == 1, 9:14] += 1.5
    return X, y


def _transformer():
    return InterpRocketTransform(
        num_features=168,
        max_dilations_per_kernel=4,
        representations="both",
        random_state=7,
    )


def test_transformer_is_cloneable_and_not_a_classifier():
    transformer = _transformer()
    cloned = clone(transformer)
    assert cloned.get_params(deep=False) == transformer.get_params(deep=False)
    assert not is_classifier(transformer)
    assert not is_regressor(transformer)


def test_transformer_creates_no_classifier_state():
    X, y = _data()
    transformer = _transformer().fit(X, y)
    assert not hasattr(transformer, "classifier_")
    assert not hasattr(transformer, "scaler_")
    assert transformer.n_output_features_ == transformer.transform(X[:2]).shape[1]


def test_fit_transform_matches_transform_after_fit():
    X, y = _data()
    first = _transformer().fit_transform(X, y)
    second_model = _transformer().fit(X, y)
    second = second_model.transform(X)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_transformer_is_label_agnostic():
    X, y = _data()
    first = _transformer().fit(X, y)
    second = _transformer().fit(X, y[::-1])
    np.testing.assert_array_equal(first.biases_raw_, second.biases_raw_)
    np.testing.assert_array_equal(first.biases_diff_, second.biases_diff_)
    np.testing.assert_allclose(first.transform(X), second.transform(X))


def test_classifier_and_transformer_fit_the_same_feature_universe():
    X, y = _data()
    params = dict(
        num_features=168,
        max_dilations_per_kernel=4,
        representations="both",
        random_state=7,
    )
    transformer = InterpRocketTransform(**params).fit(X)
    classifier = InterpRocket(
        **params,
        alpha_range=np.asarray([1.0]),
    ).fit(X, y)

    np.testing.assert_array_equal(
        transformer.dilations_raw_, classifier.dilations_raw_
    )
    np.testing.assert_array_equal(
        transformer.dilations_diff_, classifier.dilations_diff_
    )
    np.testing.assert_array_equal(transformer.biases_raw_, classifier.biases_raw_)
    np.testing.assert_array_equal(transformer.biases_diff_, classifier.biases_diff_)
    np.testing.assert_allclose(transformer.transform(X), classifier.transform(X))


def test_transformer_works_in_a_standard_sklearn_pipeline():
    X, y = _data()
    pipeline = Pipeline(
        [
            ("rocket", _transformer()),
            ("scale", StandardScaler()),
            ("classifier", RidgeClassifier(alpha=1.0)),
        ]
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == y.shape
    assert np.mean(predictions == y) >= 0.75


def test_feature_names_and_decoder_cover_all_columns():
    X, _ = _data()
    transformer = _transformer().fit(X)
    names = transformer.get_feature_names_out()
    assert names.shape == (transformer.n_output_features_,)
    assert names[0] == "irocket_0"
    assert names[-1] == f"irocket_{transformer.n_output_features_ - 1}"
    first = transformer.decode_feature_index(0)
    last = transformer.decode_feature_index(transformer.n_output_features_ - 1)
    assert first["representation"] == "raw"
    assert last["representation"] == "diff"


def test_refit_after_representation_change_clears_stale_state():
    X, _ = _data()
    transformer = _transformer().fit(X)
    assert transformer.biases_diff_.size > 0

    transformer.set_params(representations="raw")
    transformer.fit(X)

    assert transformer.dilations_diff_.size == 0
    assert transformer.num_features_per_dilation_diff_.size == 0
    assert transformer.biases_diff_.size == 0
    assert transformer.n_features_per_rep_[1] == 0
    assert transformer.transform(X[:2]).shape[1] == transformer.n_output_features_


def test_feature_names_validate_input_timepoint_names():
    X, _ = _data()
    transformer = _transformer().fit(X)
    names = transformer.get_feature_names_out(
        [f"t{index}" for index in range(X.shape[1])]
    )
    assert names.size == transformer.n_output_features_

    import pytest

    with pytest.raises(ValueError, match="timepoint"):
        transformer.get_feature_names_out(["too", "short"])


def test_transformer_cross_validates_inside_pipeline():
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = _data()
    pipeline = Pipeline(
        [
            (
                "rocket",
                InterpRocketTransform(
                    num_features=84,
                    max_dilations_per_kernel=2,
                    representations="raw",
                    random_state=7,
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
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=4),
    )
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
