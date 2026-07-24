"""Numerical contract tests for the I-ROCKET convolutional transform."""

import numpy as np
import pytest

from interp_rocket import _fit_dilations, _generate_base_kernels


def _reference_fit_dilations(input_length, num_features, max_dilations_per_kernel):
    """Reference MultiRocket dilation allocation (Tan et al./aeon)."""
    n_kernels = 84
    n_features_per_kernel = num_features // n_kernels
    if n_features_per_kernel < 1:
        raise ValueError("num_features must be at least 84")
    if input_length < 9:
        raise ValueError("input_length must be at least 9")
    if max_dilations_per_kernel < 1:
        raise ValueError("max_dilations_per_kernel must be at least 1")

    true_max = min(n_features_per_kernel, max_dilations_per_kernel)
    multiplier = n_features_per_kernel / true_max
    max_exponent = np.log2((input_length - 1) / (9 - 1))

    dilations, counts = np.unique(
        np.logspace(0, max_exponent, true_max, base=2).astype(np.int32),
        return_counts=True,
    )
    counts = (counts * multiplier).astype(np.int32)

    remainder = n_features_per_kernel - int(np.sum(counts))
    i = 0
    while remainder > 0:
        counts[i] += 1
        remainder -= 1
        i = (i + 1) % len(counts)

    return dilations.astype(np.int32), counts.astype(np.int32)


def test_base_kernel_bank_is_exactly_84_unique_length_nine_kernels():
    kernels, positive_indices = _generate_base_kernels()
    assert kernels.shape == (84, 9)
    assert positive_indices.shape == (84, 3)
    assert np.unique(kernels, axis=0).shape[0] == 84
    assert np.all(np.sum(kernels == 2.0, axis=1) == 3)
    assert np.all(np.sum(kernels == -1.0, axis=1) == 6)
    np.testing.assert_allclose(kernels.sum(axis=1), 0.0)


@pytest.mark.parametrize("input_length", [9, 20, 21, 50, 100, 500, 2000])
@pytest.mark.parametrize("num_features", [84, 500, 1000, 10000])
def test_dilation_allocation_matches_reference(
    input_length, num_features
):
    expected_dilations, expected_counts = _reference_fit_dilations(
        input_length, num_features, 16
    )
    actual_dilations, actual_counts = _fit_dilations(
        input_length, num_features, 16
    )
    np.testing.assert_array_equal(actual_dilations, expected_dilations)
    np.testing.assert_array_equal(actual_counts, expected_counts)


@pytest.mark.parametrize("input_length", [9, 20, 100, 500])
@pytest.mark.parametrize("num_features", [84, 500, 1000, 10000])
def test_dilation_allocation_preserves_requested_budget(
    input_length, num_features
):
    _, counts = _fit_dilations(input_length, num_features, 16)
    assert int(np.sum(counts)) == num_features // 84


def test_dilation_allocation_rejects_too_few_features():
    with pytest.raises(ValueError, match="84|features"):
        _fit_dilations(100, 83, 16)


def test_dilation_allocation_rejects_too_short_series():
    with pytest.raises(ValueError, match="9|length|timepoints"):
        _fit_dilations(8, 1000, 16)


@pytest.mark.parametrize("input_length", [10, 21, 150, 500, 1024])
def test_both_representations_preserve_full_multirocket_output_budget(input_length):
    """Short signals must not silently reduce the requested feature budget."""
    requested = 10_000
    _, raw_counts = _fit_dilations(input_length, requested, 32)
    _, diff_counts = _fit_dilations(input_length - 1, requested, 32)

    n_raw_biases = 84 * int(np.sum(raw_counts))
    n_diff_biases = 84 * int(np.sum(diff_counts))
    n_output_columns = 4 * (n_raw_biases + n_diff_biases)

    # 10,000 is rounded down to 119 biases per kernel: 84 * 119 = 9,996
    # biases per representation, with four pooling operators and two
    # representations.
    assert n_raw_biases == 9_996
    assert n_diff_biases == 9_996
    assert n_output_columns == 79_968
