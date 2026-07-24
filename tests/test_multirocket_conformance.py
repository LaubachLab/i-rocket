"""Numerical conformance tests for the repaired univariate transform."""

import numpy as np
import pytest

from interp_rocket import (
    InterpRocket,
    _fit_biases,
    _fit_dilations,
    _pool_convolution,
    _quantiles,
    _transform,
    compute_activation_map,
)
from tests.reference_multirocket import (
    aeon_to_contiguous,
    convolution as reference_convolution,
    fit_biases as reference_fit_biases,
    pool as reference_pool,
    quantiles as reference_quantiles,
    transform as reference_transform,
)


def test_low_discrepancy_quantiles_match_reference():
    np.testing.assert_array_equal(_quantiles(257), reference_quantiles(257))


def test_pooling_values_match_reference_implementation():
    C = np.asarray([-2.0, 1.0, 3.0, -1.0, 4.0], dtype=np.float32)
    bias = np.float32(0.5)
    expected = reference_pool(C, float(bias), 0, len(C))
    actual = np.asarray(_pool_convolution(C, bias, 0, len(C)), dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        actual,
        np.asarray([0.6, 19.0 / 6.0, 7.0 / 3.0, 3.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_bias_fitting_uses_one_example_per_kernel_dilation():
    X = np.vstack(
        [
            np.linspace(-1.0, 1.0, 12),
            np.linspace(2.0, 5.0, 12),
            np.linspace(-4.0, -2.0, 12),
        ]
    ).astype(np.float32)
    dilations = np.asarray([1], dtype=np.int32)
    features_per_dilation = np.asarray([2], dtype=np.int32)
    assigned_quantiles = _quantiles(84 * 2)

    expected = reference_fit_biases(
        X,
        dilations,
        features_per_dilation,
        assigned_quantiles,
        seed=17,
    )
    actual = _fit_biases(
        X,
        dilations,
        features_per_dilation,
        assigned_quantiles,
        17,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("is_first_difference", [False, True])
def test_transform_matches_readable_reference(is_first_difference):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(3, 12)).astype(np.float32)
    if is_first_difference:
        X = np.diff(X, axis=1).astype(np.float32)

    dilations = np.asarray([1], dtype=np.int32)
    features_per_dilation = np.asarray([1], dtype=np.int32)
    assigned_quantiles = _quantiles(84)
    biases = _fit_biases(
        X,
        dilations,
        features_per_dilation,
        assigned_quantiles,
        11,
    )

    expected = reference_transform(
        X,
        dilations,
        features_per_dilation,
        biases,
        is_first_difference=is_first_difference,
    )
    actual = _transform(
        X,
        dilations,
        features_per_dilation,
        biases,
        is_first_difference=is_first_difference,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("representation", ["raw", "diff"])
@pytest.mark.parametrize("padding", ["same", "valid"])
def test_activation_map_matches_reference_convolution(representation, padding):
    x = np.linspace(-1.5, 2.5, 20, dtype=np.float32)
    kernel_index = 37
    dilation = 2
    bias = 0.25
    is_diff = representation == "diff"

    full = reference_convolution(
        x,
        kernel_index,
        dilation,
        is_first_difference=is_diff,
    )
    if padding == "same":
        expected = full
        expected_indices = np.arange(len(full), dtype=np.float32)
    else:
        trim = 4 * dilation
        expected = full[trim:-trim]
        expected_indices = np.arange(trim, len(full) - trim, dtype=np.float32)

    conv, activation, indices = compute_activation_map(
        x,
        kernel_index,
        dilation,
        bias,
        padding=padding,
        representation=representation,
    )
    np.testing.assert_allclose(conv, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(activation, (expected > bias).astype(np.float32))
    np.testing.assert_array_equal(indices, expected_indices)


def test_decoded_features_report_alternating_padding_and_round_trip():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 20)).astype(np.float32)
    y = np.repeat([0, 1], 6)
    model = InterpRocket(
        num_features=84,
        max_dilations_per_kernel=1,
        representations="raw",
        alpha_range=np.asarray([1.0]),
        random_state=3,
    ).fit(X, y)

    assert model.transform(X[:2]).shape == (2, 84 * 4)
    assert model.n_output_features_ == 84 * 4

    # One bias per kernel at one dilation: four columns per kernel.
    for kernel_index in (0, 1, 42, 83):
        feature_index = kernel_index * 4
        info = model.decode_feature_index(feature_index)
        assert info["kernel_index"] == kernel_index
        assert info["bias_index"] == kernel_index
        assert info["pooling_op"] == "PPV"
        expected_mode = "same" if kernel_index % 2 == 0 else "valid"
        assert info["padding_mode"] == expected_mode
        assert info["padding"] == (expected_mode == "same")


def test_transform_matches_aeon_when_aeon_is_available():
    pytest.importorskip("aeon")
    from aeon.transformations.collection.convolution_based._multirocket import (
        MultiRocket,
        _fit_biases_univariate,
        _transform_uni,
    )

    rng = np.random.default_rng(8)
    X = rng.normal(size=(2, 50)).astype(np.float32)
    X_diff = np.diff(X, axis=1).astype(np.float32)
    seed = 19
    n_kernels = 504

    # This budget produces multiple dilations and multiple biases per kernel,
    # so the comparison exercises allocation, padding alternation, and feature
    # indexing rather than only the one-bias/one-dilation special case.
    dilations, counts = _fit_dilations(50, n_kernels, 8)
    dilations_diff, counts_diff = _fit_dilations(49, n_kernels, 8)
    q = _quantiles(84 * int(np.sum(counts)))
    q_diff = _quantiles(84 * int(np.sum(counts_diff)))

    biases = _fit_biases(X, dilations, counts, q, seed)
    biases_diff = _fit_biases(
        X_diff,
        dilations_diff,
        counts_diff,
        q_diff,
        seed,
    )

    indices = MultiRocket._indices
    aeon_fit_biases = getattr(
        _fit_biases_univariate,
        "py_func",
        _fit_biases_univariate,
    )
    aeon_transform = getattr(_transform_uni, "py_func", _transform_uni)

    aeon_biases = aeon_fit_biases(
        X,
        dilations,
        counts,
        q,
        indices,
        seed,
    )
    aeon_biases_diff = aeon_fit_biases(
        X_diff,
        dilations_diff,
        counts_diff,
        q_diff,
        indices,
        seed,
    )
    np.testing.assert_allclose(biases, aeon_biases, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(
        biases_diff,
        aeon_biases_diff,
        rtol=3e-6,
        atol=3e-6,
    )

    actual = np.concatenate(
        [
            _transform(X, dilations, counts, biases),
            _transform(
                X_diff,
                dilations_diff,
                counts_diff,
                biases_diff,
                is_first_difference=True,
            ),
        ],
        axis=1,
    )
    aeon_features = aeon_transform(
        X,
        X_diff,
        (dilations, counts, aeon_biases),
        (dilations_diff, counts_diff, aeon_biases_diff),
        4,
        indices,
        seed,
    )
    expected = aeon_to_contiguous(
        aeon_features,
        84 * int(np.sum(counts)),
        84 * int(np.sum(counts_diff)),
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_decoded_feature_round_trips_to_its_transform_column():
    """Decoded metadata must reproduce the exact stored feature value."""
    rng = np.random.default_rng(29)
    X = rng.normal(size=(12, 24)).astype(np.float32)
    y = np.repeat([0, 1], 6)
    model = InterpRocket(
        num_features=168,
        max_dilations_per_kernel=2,
        representations="both",
        alpha_range=np.asarray([1.0]),
        random_state=7,
    ).fit(X, y)

    transformed = model.transform(X[:1])[0]
    n_raw_columns = 4 * int(model.n_features_per_rep_[0])

    # Cover raw/diff, same/valid padding, and all four pooling operators.
    candidate_indices = [
        0, 1, 2, 3,          # raw, kernel 0, same padding
        4, 5, 6, 7,          # raw, kernel 1, valid padding
        n_raw_columns,
        n_raw_columns + 1,
        n_raw_columns + 2,
        n_raw_columns + 3,   # diff, kernel 0, same padding
        n_raw_columns + 4,
        n_raw_columns + 5,
        n_raw_columns + 6,
        n_raw_columns + 7,   # diff, kernel 1, valid padding
    ]

    for feature_index in candidate_indices:
        info = model.decode_feature_index(feature_index)
        x = X[0]
        if info["representation"] == "diff":
            x = np.diff(x).astype(np.float32)

        convolution, _, _ = compute_activation_map(
            x,
            info["kernel_index"],
            info["dilation"],
            info["bias"],
            padding=info["padding_mode"],
            representation=info["representation"],
        )
        pooled = _pool_convolution(
            convolution,
            np.float32(info["bias"]),
            0,
            len(convolution),
        )
        expected = pooled[info["pooling_index"]]
        np.testing.assert_allclose(
            transformed[feature_index],
            expected,
            rtol=3e-6,
            atol=3e-6,
        )


def test_every_decoded_component_reconstructs_its_feature_value():
    """Decoded metadata must be sufficient to reproduce transformed columns."""
    rng = np.random.default_rng(41)
    X = rng.normal(size=(12, 30)).astype(np.float32)
    y = np.repeat([0, 1], 6)
    model = InterpRocket(
        num_features=336,
        max_dilations_per_kernel=6,
        representations="both",
        random_state=31,
        alpha_range=np.asarray([1.0]),
    ).fit(X, y)
    transformed = model.transform(X[:1])[0]

    wanted = {
        (representation, pooling, padding)
        for representation in ("raw", "diff")
        for pooling in model.POOLING_NAMES
        for padding in ("same", "valid")
    }
    candidates = {}
    for feature_index in range(transformed.size):
        info = model.decode_feature_index(feature_index)
        key = (
            info["representation"],
            info["pooling_op"],
            info["padding_mode"],
        )
        if key in wanted and key not in candidates:
            candidates[key] = feature_index
        if len(candidates) == len(wanted):
            break

    assert set(candidates) == wanted

    for key, feature_index in candidates.items():
        info = model.decode_feature_index(feature_index)
        x = X[0] if info["representation"] == "raw" else np.diff(X[0])
        conv, _, _ = compute_activation_map(
            x.astype(np.float32),
            info["kernel_index"],
            info["dilation"],
            info["bias"],
            padding=info["padding_mode"],
            representation=info["representation"],
        )
        expected = reference_pool(conv, info["bias"], 0, len(conv))[
            info["pooling_index"]
        ]
        assert transformed[feature_index] == pytest.approx(
            float(expected), abs=3e-6
        ), key


@pytest.mark.parametrize(
    ("representations", "multiplier"),
    [("raw", 4), ("diff", 4), ("both", 8)],
)
def test_model_feature_count_follows_canonical_budget(
    representations, multiplier
):
    rng = np.random.default_rng(52)
    X = rng.normal(size=(10, 21)).astype(np.float32)
    y = np.repeat([0, 1], 5)
    requested = 500
    per_representation = 84 * (requested // 84)

    model = InterpRocket(
        num_features=requested,
        max_dilations_per_kernel=16,
        representations=representations,
        random_state=7,
        alpha_range=np.asarray([1.0]),
    ).fit(X, y)

    expected = multiplier * per_representation
    assert model.n_output_features_ == expected
    assert model.transform(X[:2]).shape == (2, expected)


@pytest.mark.parametrize("bad_index", [-1, 10**9, 1.5, True])
def test_decode_feature_index_rejects_invalid_values(bad_index):
    rng = np.random.default_rng(61)
    X = rng.normal(size=(10, 20)).astype(np.float32)
    y = np.repeat([0, 1], 5)
    model = InterpRocket(
        num_features=84,
        representations="raw",
        random_state=5,
        alpha_range=np.asarray([1.0]),
    ).fit(X, y)

    expected_error = TypeError if isinstance(bad_index, (float, bool)) else IndexError
    with pytest.raises(expected_error):
        model.decode_feature_index(bad_index)
