"""Interpretable MultiRocket for univariate time-series classification.

The module provides a transparent, numerically validated univariate
MultiRocket transform.  Every transformed column can be decoded to its base
kernel, dilation, padding mode, bias, pooling operator, and signal
representation.

The recommended leakage-free workflow is implemented in ``irocket_model_selection``:
fit ``InterpRocketTransform`` inside each training partition, perform
resampled shrinkage-*t* consensus selection on that fixed feature universe,
measure selection reproducibility with the Nogueira statistic, and tune the
ridge classifier inside nested cross-validation.

Public surface
--------------
InterpRocketTransform
    Classifier-agnostic MultiRocket transformer.
InterpRocket
    Convenience transform-plus-ridge classifier.
compute_activation_map
    Per-timepoint activation for a decoded kernel.
mutual_information
    Confusion-matrix mutual information in bits.
OI, POOLING_COLORS, INFO_COLORS
    Shared colorblind-safe plotting constants.

References
----------
Brunner, F. (2024). Explainable time series classification with X-ROCKET.
Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket.

Author
------
Mark Laubach, American University, Department of Neuroscience.

License
-------
BSD-3-Clause.
"""

import numpy as np
from itertools import combinations
from numba import njit, prange
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

__all__ = [
    "InterpRocketTransform",
    "InterpRocket",
    "compute_activation_map",
    "mutual_information",
    "OI",
    "POOLING_COLORS",
    "INFO_COLORS",
]

# ============================================================================
# COLORBLIND-SAFE PALETTES
# ============================================================================


# Okabe-Ito categorical palette shared by all categorical plots.
OI = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Pooling operators use the requested high-contrast Okabe-Ito subset.
POOLING_COLORS = {
    "PPV": "#0072B2",   # blue
    "MPV": "#56B4E9",   # sky blue
    "MIPV": "#D55E00",  # vermillion
    "LSPV": "#E69F00",  # orange
}

INFO_COLORS = {
    "redundant": "#E69F00",    # orange
    "synergistic": "#0072B2",  # blue
    "independent": "#7f7f7f",  # gray
}

def _validate_feature_index_array(feature_indices, n_features, *, name="feature_mask"):
    """Validate a one-dimensional, unique set of transformed feature indices."""
    indices = np.asarray(feature_indices)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{name} must contain integers.")
    indices = indices.astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if np.any(indices < 0) or np.any(indices >= int(n_features)):
        raise ValueError(f"{name} contains an out-of-range feature index.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    return indices


def _kernel_configuration_key(feature_info):
    """Return the pre-bias convolutional kernel identity for one feature.

    Multiple MultiRocket columns can share this identity while differing in
    bias threshold and/or pooling operator.  It is the appropriate grouping
    key for displays whose unit is a unique convolutional kernel configuration
    rather than an individual transformed feature.
    """
    return (
        str(feature_info["representation"]),
        int(feature_info["kernel_index"]),
        int(feature_info["dilation"]),
        str(feature_info.get("padding_mode", "")),
    )


def _format_feature_label(feature_info, *, compact=False):
    """Format a feature label that cannot hide distinct MultiRocket columns.

    A complete feature identity includes representation, base kernel, dilation,
    padding, bias threshold, and pooling operator.  The transformed column index
    and within-kernel bias rank are included so that two rows never appear to
    describe the same feature when their thresholds differ.
    """
    parts = []
    if "feature_index" in feature_info:
        parts.append(f"F{int(feature_info['feature_index'])}")
    parts.append(f"K{int(feature_info['kernel_index'])}")
    parts.append(f"d={int(feature_info['dilation'])}")

    bias_rank = feature_info.get("bias_rank_within_kernel")
    if bias_rank is not None:
        parts.append(f"b{int(bias_rank)}")

    parts.append(str(feature_info["pooling_op"]))
    parts.append(str(feature_info["representation"]))

    padding_mode = feature_info.get("padding_mode")
    if padding_mode:
        parts.append(
            "S" if compact and padding_mode == "same"
            else "V" if compact and padding_mode == "valid"
            else str(padding_mode)
        )

    if not compact and "bias" in feature_info:
        parts.append(f"bias={float(feature_info['bias']):.4g}")
    return " ".join(parts)


# ============================================================================
# SECTION 1: THE 84 BASE KERNELS
# ============================================================================
#
# MiniRocket/MultiRocket use 84 deterministic kernels of length 9.
# Each kernel has weights from {-1, 2}: six positions get -1, three get 2.
# The 84 kernels enumerate all C(9,3) = 84 ways to choose which 3 of 9
# positions receive the weight 2 (the rest get -1).


# -------------------------------------------------------------------------
# Kernel core
# -------------------------------------------------------------------------

def _generate_base_kernels():
    """
    Generate the 84 deterministic MiniRocket base kernels.

    Returns
    -------
    kernels : ndarray, shape (84, 9), dtype float32
        Each row is a length-9 kernel with weights in {-1, 2}.
    indices : ndarray, shape (84, 3), dtype int32
        The 3 positions (of 9) that receive weight 2 in each kernel.
    """
    indices = np.array([combo for combo in combinations(range(9), 3)], dtype=np.int32)
    kernels = np.full((84, 9), -1.0, dtype=np.float32)
    for i, idx in enumerate(indices):
        kernels[i, idx] = 2.0
    return kernels, indices


def _fit_dilations(input_length, num_features, max_dilations_per_kernel):
    """
    Determine dilations and features-per-dilation for given series length.

    Follows the MiniRocket/MultiRocket algorithm exactly:
    - max dilation = (input_length - 1) / (9 - 1), ensuring receptive field
      fits within the series
    - dilations are exponentially spaced: 2^0, 2^1, ..., 2^(num_dilations-1)
    - features are distributed across dilations as evenly as possible

    Parameters
    ----------
    input_length : int
        Length of input time series.
    num_features : int
        Target number of features (will be rounded to multiple of 84).
    max_dilations_per_kernel : int
        Maximum number of distinct dilations to use.

    Returns
    -------
    dilations : ndarray of int32
        The dilation values to use.
    num_features_per_dilation : ndarray of int32
        How many features (biases) to generate per dilation.
    """
    if not isinstance(input_length, (int, np.integer)) or input_length < 9:
        raise ValueError("input_length must be an integer of at least 9")
    if not isinstance(num_features, (int, np.integer)) or num_features < 84:
        raise ValueError("num_features must be an integer of at least 84")
    if (
        not isinstance(max_dilations_per_kernel, (int, np.integer))
        or max_dilations_per_kernel < 1
    ):
        raise ValueError("max_dilations_per_kernel must be a positive integer")

    num_kernels = 84
    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    # Canonical MiniRocket/MultiRocket allocation: start from an exponentially
    # spaced grid, collapse duplicate integer dilations, scale their counts,
    # then distribute any remainder so the requested per-kernel budget is
    # preserved exactly.
    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(
            0,
            max_exponent,
            true_max_dilations_per_kernel,
            base=2,
        ).astype(np.int32),
        return_counts=True,
    )
    num_features_per_dilation = (
        num_features_per_dilation * multiplier
    ).astype(np.int32)

    remainder = num_features_per_kernel - int(
        np.sum(num_features_per_dilation)
    )
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations.astype(np.int32), num_features_per_dilation.astype(np.int32)


def _quantiles(n):
    """Generate the canonical low-discrepancy MultiRocket quantiles.

    The sequence is calculated in float64 and cast to float32 only after each
    value has been generated. This matches the reference MultiRocket/aeon
    implementation and avoids cumulative differences from a float32 golden
    ratio constant.
    """
    phi = (np.sqrt(5.0) + 1.0) / 2.0
    return np.array(
        [((i + 1) * phi) % 1.0 for i in range(n)],
        dtype=np.float32,
    )


@njit(fastmath=True, cache=True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles, random_state_seed):
    """Fit MultiRocket bias thresholds from the training data.

    MultiRocket draws one training example independently for every
    kernel--dilation combination, computes that combination's zero-padded
    convolution output, and takes the assigned low-discrepancy quantiles from
    that one output.  Bias fitting always uses the padded convolution, even
    though the transform alternates padded and unpadded pooling regions.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints), dtype float32
        Training time series for one representation.
    dilations : ndarray of int32
        Dilation values.
    num_features_per_dilation : ndarray of int32
        Number of bias thresholds per kernel at each dilation.
    quantiles : ndarray of float32
        Low-discrepancy quantile positions, one per bias.
    random_state_seed : int
        Seed for the legacy NumPy random stream used by the reference
        MultiRocket implementation.

    Returns
    -------
    biases : ndarray of float32
        Biases ordered by dilation, kernel, and within-combination quantile.
    """
    np.random.seed(random_state_seed)

    num_instances, input_length = X.shape

    indices_raw = np.zeros((84, 3), dtype=np.int32)
    count = 0
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                indices_raw[count, 0] = i
                indices_raw[count, 1] = j
                indices_raw[count, 2] = k
                count += 1

    num_kernels = 84
    num_dilations = len(dilations)
    num_features_total = num_kernels * np.sum(num_features_per_dilation)
    biases = np.zeros(num_features_total, dtype=np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2
        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + num_features_this_dilation

            # One randomly selected training example per kernel--dilation pair.
            x = X[np.random.randint(num_instances)]
            A = -x
            G = x + x + x

            # Shared MiniRocket convolution construction.
            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            shift_start = dilation
            shift_end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-shift_end:] = C_alpha[-shift_end:] + A[:shift_end]
                C_gamma[gamma_index, -shift_end:] = G[:shift_end]
                shift_end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-shift_start] = C_alpha[:-shift_start] + A[shift_start:]
                C_gamma[gamma_index, :-shift_start] = G[shift_start:]
                shift_start += dilation

            i0 = indices_raw[kernel_index, 0]
            i1 = indices_raw[kernel_index, 1]
            i2 = indices_raw[kernel_index, 2]
            C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]

            biases[feature_index_start:feature_index_end] = np.quantile(
                C,
                quantiles[feature_index_start:feature_index_end],
            )
            feature_index_start = feature_index_end

    return biases


@njit(fastmath=True, inline="always")
def _pool_convolution(C, bias, start, stop):
    """Return PPV, MPV, MIPV, and LSPV for one convolution region.

    The implementation intentionally follows the reference MultiRocket source
    and aeon, including its exact MPV and LSPV calculations.  ``start`` is
    inclusive and ``stop`` is exclusive.
    """
    ppv = 0
    last_val = 0
    max_stretch = 0.0
    mean_index = 0
    mean = 0.0

    n_values = stop - start
    for local_index in range(n_values):
        value = C[start + local_index]
        if value > bias:
            ppv += 1
            mean_index += local_index
            # This is the operation used in the original and aeon
            # MultiRocket implementations.
            mean += value + bias
        elif value < bias:
            stretch = local_index - last_val
            if stretch > max_stretch:
                max_stretch = stretch
            last_val = local_index

    stretch = n_values - 1 - last_val
    if stretch > max_stretch:
        max_stretch = stretch

    ppv_value = ppv / n_values
    mpv_value = mean / ppv if ppv > 0 else 0.0
    mipv_value = mean_index / ppv if ppv > 0 else -1.0

    return ppv_value, mpv_value, mipv_value, max_stretch


@njit(fastmath=True, parallel=True, cache=True)
def _transform(
    X,
    dilations,
    num_features_per_dilation,
    biases,
    is_first_difference=False,
):
    """Apply one MultiRocket representation with transparent feature ordering.

    Numerical calculations match the univariate reference implementation.  The
    only intentional difference is column order: I-ROCKET stores the four
    pooling values contiguously for each bias as ``PPV, MPV, MIPV, LSPV`` so a
    feature index can be decoded without a global column permutation.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints), dtype float32
        Raw signals or first-differenced signals.
    dilations, num_features_per_dilation, biases : ndarray
        Fitted parameters for this representation.
    is_first_difference : bool, default=False
        Reproduce the reference MultiRocket alignment used when transforming
        first differences.  Bias fitting remains symmetric for both
        representations, as in the reference implementation.

    Returns
    -------
    features : ndarray, shape (n_instances, n_biases * 4)
        Four contiguous pooling values per fitted bias.
    """
    num_instances, input_length = X.shape
    num_kernels = 84
    num_dilations = len(dilations)
    num_biases = num_kernels * np.sum(num_features_per_dilation)
    features = np.zeros((num_instances, num_biases * 4), dtype=np.float32)

    indices_raw = np.zeros((84, 3), dtype=np.int32)
    count = 0
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                indices_raw[count, 0] = i
                indices_raw[count, 1] = j
                indices_raw[count, 2] = k
                count += 1

    for instance_index in prange(num_instances):
        x = X[instance_index]
        A = -x
        G = x + x + x
        feature_index_start = 0

        for dilation_index in range(num_dilations):
            padding_selector = dilation_index % 2
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            shift_start = dilation
            # The original MultiRocket implementation uses the pre-difference
            # length in this expression.  For an already differenced array that
            # is input_length + 1 and produces the established asymmetric
            # alignment for positions left of the kernel center.
            shift_end = input_length + (1 if is_first_difference else 0) - padding

            for gamma_index in range(9 // 2):
                C_alpha[-shift_end:] = C_alpha[-shift_end:] + A[:shift_end]
                C_gamma[gamma_index, -shift_end:] = G[:shift_end]
                shift_end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-shift_start] = C_alpha[:-shift_start] + A[shift_start:]
                C_gamma[gamma_index, :-shift_start] = G[shift_start:]
                shift_start += dilation

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation
                uses_same_padding = ((padding_selector + kernel_index) % 2) == 0

                i0 = indices_raw[kernel_index, 0]
                i1 = indices_raw[kernel_index, 1]
                i2 = indices_raw[kernel_index, 2]
                C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]

                if uses_same_padding:
                    pool_start = 0
                    pool_stop = C.shape[0]
                else:
                    pool_start = padding
                    pool_stop = C.shape[0] - padding

                for feature_count in range(num_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    bias = biases[feature_index]
                    ppv, mpv, mipv, lspv = _pool_convolution(
                        C,
                        bias,
                        pool_start,
                        pool_stop,
                    )

                    output_index = feature_index * 4
                    features[instance_index, output_index] = ppv
                    features[instance_index, output_index + 1] = mpv
                    features[instance_index, output_index + 2] = mipv
                    features[instance_index, output_index + 3] = lspv

                feature_index_start = feature_index_end

    return features


@njit(fastmath=True, cache=True)
def _compute_activation_map_core(
    x,
    kernel_index,
    dilation,
    bias,
    uses_same_padding,
    is_first_difference,
):
    input_length = len(x)
    padding = ((9 - 1) * dilation) // 2

    indices_raw = np.zeros((84, 3), dtype=np.int32)
    count = 0
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                indices_raw[count, 0] = i
                indices_raw[count, 1] = j
                indices_raw[count, 2] = k
                count += 1

    kernel = np.full(9, -1.0, dtype=np.float32)
    i0 = indices_raw[kernel_index, 0]
    i1 = indices_raw[kernel_index, 1]
    i2 = indices_raw[kernel_index, 2]
    kernel[i0] = 2.0
    kernel[i1] = 2.0
    kernel[i2] = 2.0

    offsets = np.empty(9, dtype=np.int32)
    for position in range(9):
        if is_first_difference and position < 4:
            offsets[position] = (position - 4) * dilation + 1
        else:
            offsets[position] = (position - 4) * dilation

    full_conv = np.zeros(input_length, dtype=np.float32)
    for output_index in range(input_length):
        value = np.float32(0.0)
        for position in range(9):
            input_index = output_index + offsets[position]
            if 0 <= input_index < input_length:
                value += kernel[position] * x[input_index]
        full_conv[output_index] = value

    if uses_same_padding:
        start = 0
        stop = input_length
    else:
        start = padding
        stop = input_length - padding

    n_output = stop - start
    conv_output = np.zeros(n_output, dtype=np.float32)
    activation = np.zeros(n_output, dtype=np.float32)
    time_indices = np.zeros(n_output, dtype=np.float32)

    for local_index in range(n_output):
        full_index = start + local_index
        conv_value = full_conv[full_index]
        conv_output[local_index] = conv_value
        activation[local_index] = 1.0 if conv_value > bias else 0.0
        time_indices[local_index] = full_index

    return conv_output, activation, time_indices


def compute_activation_map(
    x,
    kernel_index,
    dilation,
    bias,
    padding="same",
    representation="raw",
):
    """Compute the convolution and binary activation for one decoded feature.

    Parameters
    ----------
    x : ndarray, shape (n_timepoints,)
        A raw signal when ``representation='raw'`` or an already
        first-differenced signal when ``representation='diff'``.
    kernel_index : int
        Index of one of the 84 deterministic kernels.
    dilation : int
        Kernel dilation.
    bias : float
        Fitted bias threshold.
    padding : {'same', 'valid'}, default='same'
        Pooling region used by the decoded feature.
    representation : {'raw', 'diff'}, default='raw'
        Selects the established MultiRocket convolution alignment.

    Returns
    -------
    conv_output, activation, time_indices : ndarray
        Values from the exact region used by the feature.  ``time_indices``
        refer to indices in ``x``; callers mapping differences back to the
        original signal may apply their preferred half-sample convention.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    if not isinstance(kernel_index, (int, np.integer)) or not 0 <= kernel_index < 84:
        raise ValueError("kernel_index must be an integer from 0 through 83")
    if not isinstance(dilation, (int, np.integer)) or dilation < 1:
        raise ValueError("dilation must be a positive integer")
    if padding not in ("same", "valid"):
        raise ValueError("padding must be 'same' or 'valid'")
    if representation not in ("raw", "diff"):
        raise ValueError("representation must be 'raw' or 'diff'")

    required_length = 1 + 8 * int(dilation)
    if padding == "valid" and len(x) < required_length:
        raise ValueError(
            "x is too short for valid convolution at the requested dilation"
        )

    return _compute_activation_map_core(
        x,
        int(kernel_index),
        int(dilation),
        np.float32(bias),
        padding == "same",
        representation == "diff",
    )


def mutual_information(y_true=None, y_pred=None, cm=None, base=2):
    """
    Calculate mutual information between true and predicted labels.

    Parameters
    ----------
    y_true : array-like, optional
        True class labels.
    y_pred : array-like, optional
        Predicted class labels.
    cm : array-like, optional
        Pre-computed confusion matrix (rows=true, cols=predicted).
    base : int or float, default=2
        Logarithm base. Use 2 for bits, np.e for nats.

    Returns
    -------
    mi : float
        Mutual information in specified units (bits if base=2).
    """
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Must provide either (y_true, y_pred) or cm")
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = np.asarray(cm)

    total = cm.sum()
    if total == 0:
        return 0.0

    p_joint = cm / total
    p_true = p_joint.sum(axis=1)
    p_pred = p_joint.sum(axis=0)

    mi = 0.0
    n_classes_true, n_classes_pred = p_joint.shape
    for i in range(n_classes_true):
        for j in range(n_classes_pred):
            if p_joint[i, j] > 0 and p_true[i] > 0 and p_pred[j] > 0:
                mi += (
                    p_joint[i, j]
                    * np.log(p_joint[i, j] / (p_true[i] * p_pred[j]))
                    / np.log(base)
                )
    return mi


def _compute_all_metrics(y_true, y_pred):
    """
    Compute all classification metrics.

    Returns
    -------
    metrics : dict with keys:
        'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
        'mcc', 'mutual_info'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(y_true))

    avg = "binary" if n_classes == 2 else "macro"

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "mutual_info": float(mutual_information(y_true=y_true, y_pred=y_pred)),
    }



# -------------------------------------------------------------------------
# InterpRocket
# -------------------------------------------------------------------------

class InterpRocketTransform(TransformerMixin, BaseEstimator):
    """Transparent univariate MultiRocket feature transformer.

    This estimator fits only the unsupervised convolutional transform.  It is
    the component intended for scikit-learn pipelines and nested validation:
    the transform is fitted on each training partition, and downstream
    selectors and classifiers operate on the resulting feature matrix.

    Parameters
    ----------
    max_dilations_per_kernel : int, default=16
        Maximum number of dilation values per kernel.
    num_features : int, default=10000
        Target number of bias features per representation.  MultiRocket emits
        four pooling values per bias and rounds the bias budget down to a
        multiple of the 84 deterministic kernels.
    random_state : int, default=0
        Seed used for the training-example draws in bias fitting.
    representations : {'both', 'raw', 'diff'}, default='both'
        Signal representations included in the transform.
    verbose : bool or int, default=False
        Print fit progress when truthy.
    """

    POOLING_NAMES = ["PPV", "MPV", "MIPV", "LSPV"]

    _TRANSFORM_LEARNED_ATTRIBUTES = (
        "n_features_in_",
        "n_timepoints_in_",
        "base_kernels_",
        "base_indices_",
        "dilations_raw_",
        "num_features_per_dilation_raw_",
        "biases_raw_",
        "dilations_diff_",
        "num_features_per_dilation_diff_",
        "biases_diff_",
        "n_features_per_rep_",
        "n_output_features_",
        "_n_features_out",
    )

    def __init__(
        self,
        max_dilations_per_kernel=16,
        num_features=10000,
        random_state=0,
        representations="both",
        verbose=False,
    ):
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_features = num_features
        self.random_state = random_state
        self.representations = representations
        self.verbose = verbose

    def _reset_transform_state(self):
        """Remove learned transform state before refitting."""
        for attribute in self._TRANSFORM_LEARNED_ATTRIBUTES:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def _validate_parameters(self):
        if self.representations not in ("both", "raw", "diff"):
            raise ValueError(
                "representations must be 'both', 'raw', or 'diff', "
                f"got {self.representations!r}"
            )
        if (
            isinstance(self.max_dilations_per_kernel, (bool, np.bool_))
            or not isinstance(
                self.max_dilations_per_kernel, (int, np.integer)
            )
            or self.max_dilations_per_kernel < 1
        ):
            raise ValueError(
                "max_dilations_per_kernel must be a positive integer"
            )
        if (
            isinstance(self.num_features, (bool, np.bool_))
            or not isinstance(self.num_features, (int, np.integer))
            or self.num_features < 84
        ):
            raise ValueError("num_features must be an integer of at least 84")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer")
        if not isinstance(self.verbose, (bool, np.bool_, int, np.integer)):
            raise ValueError("verbose must be a boolean or integer")

    def _validate_X(self, X, *, reset):
        X = check_array(
            X,
            accept_sparse=False,
            ensure_2d=True,
            allow_nd=False,
            dtype=np.float32,
        )
        minimum_length = 10 if self.representations in ("both", "diff") else 9
        if X.shape[1] < minimum_length:
            if minimum_length == 10:
                raise ValueError(
                    "at least 10 timepoints are required when first "
                    "differences are used"
                )
            raise ValueError("at least 9 timepoints are required")

        if reset:
            self.n_features_in_ = X.shape[1]
            self.n_timepoints_in_ = X.shape[1]
        elif X.shape[1] != self.n_timepoints_in_:
            raise ValueError(
                "X has a different number of timepoints than the data used "
                f"during fit: got {X.shape[1]}, expected "
                f"{self.n_timepoints_in_}"
            )
        return X

    def _log(self, message):
        if self.verbose:
            print(message)

    def _fit_transform_parameters(self, X):
        """Fit kernels, dilations, and biases on a validated matrix."""
        n_instances, input_length = X.shape
        self._log(
            f"{self.__class__.__name__}.fit: {n_instances} instances x "
            f"{input_length} timepoints"
        )

        self.base_kernels_, self.base_indices_ = _generate_base_kernels()
        use_raw = self.representations in ("both", "raw")
        use_diff = self.representations in ("both", "diff")

        # Define all representation attributes on every successful fit. This
        # prevents stale parameters when an estimator is refit after changing
        # ``representations`` with ``set_params``.
        self.dilations_raw_ = np.empty(0, dtype=np.int32)
        self.num_features_per_dilation_raw_ = np.empty(0, dtype=np.int32)
        self.biases_raw_ = np.empty(0, dtype=np.float32)
        self.dilations_diff_ = np.empty(0, dtype=np.int32)
        self.num_features_per_dilation_diff_ = np.empty(0, dtype=np.int32)
        self.biases_diff_ = np.empty(0, dtype=np.float32)

        if use_raw:
            self._log("  Fitting dilations (raw)...")
            (
                self.dilations_raw_,
                self.num_features_per_dilation_raw_,
            ) = _fit_dilations(
                input_length,
                self.num_features,
                self.max_dilations_per_kernel,
            )
            n_features_raw = 84 * int(
                np.sum(self.num_features_per_dilation_raw_)
            )
            self._log(
                f"  Fitting biases (raw): {n_features_raw} biases across "
                f"{len(self.dilations_raw_)} dilations..."
            )
            self.biases_raw_ = _fit_biases(
                X,
                self.dilations_raw_,
                self.num_features_per_dilation_raw_,
                _quantiles(n_features_raw),
                self.random_state,
            )
        else:
            n_features_raw = 0

        if use_diff:
            X_diff = np.diff(X, axis=1).astype(np.float32)
            self._log("  Fitting dilations (diff)...")
            (
                self.dilations_diff_,
                self.num_features_per_dilation_diff_,
            ) = _fit_dilations(
                X_diff.shape[1],
                self.num_features,
                self.max_dilations_per_kernel,
            )
            n_features_diff = 84 * int(
                np.sum(self.num_features_per_dilation_diff_)
            )
            self._log(
                f"  Fitting biases (diff): {n_features_diff} biases across "
                f"{len(self.dilations_diff_)} dilations..."
            )
            self.biases_diff_ = _fit_biases(
                X_diff,
                self.dilations_diff_,
                self.num_features_per_dilation_diff_,
                _quantiles(n_features_diff),
                self.random_state,
            )
        else:
            n_features_diff = 0

        self.n_features_per_rep_ = (
            int(n_features_raw),
            int(n_features_diff),
        )
        self.n_output_features_ = int(4 * (n_features_raw + n_features_diff))
        self._n_features_out = self.n_output_features_

    def fit(self, X, y=None):
        """Fit the convolutional transform; ``y`` is accepted and ignored."""
        self._reset_transform_state()
        self._validate_parameters()
        X = self._validate_X(X, reset=True)
        self._fit_transform_parameters(X)
        return self

    def _transform(self, X):
        """Apply the fitted raw and/or differenced MultiRocket transform."""
        check_is_fitted(
            self,
            attributes=[
                "base_kernels_",
                "base_indices_",
                "n_features_per_rep_",
                "n_timepoints_in_",
            ],
        )
        X = self._validate_X(X, reset=False)
        blocks = []
        if self.representations in ("both", "raw"):
            blocks.append(
                _transform(
                    X,
                    self.dilations_raw_,
                    self.num_features_per_dilation_raw_,
                    self.biases_raw_,
                )
            )
        if self.representations in ("both", "diff"):
            blocks.append(
                _transform(
                    np.diff(X, axis=1).astype(np.float32),
                    self.dilations_diff_,
                    self.num_features_per_dilation_diff_,
                    self.biases_diff_,
                    is_first_difference=True,
                )
            )
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)

    def transform(self, X):
        """Return the unscaled MultiRocket feature matrix."""
        return self._transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return deterministic names for transformed columns."""
        check_is_fitted(self, attributes=["n_output_features_"])
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if (
                input_features.ndim != 1
                or input_features.size != self.n_features_in_
            ):
                raise ValueError(
                    "input_features must contain one name per input "
                    "timepoint."
                )
        return np.asarray(
            [f"irocket_{index}" for index in range(self.n_output_features_)],
            dtype=object,
        )

    def decode_feature_index(self, feature_index):
        """Map a transformed column to its complete generating parameters.

        I-ROCKET stores four pooling values contiguously for every bias. The
        returned metadata therefore identifies the representation, dilation,
        kernel, bias, pooling operator, and the alternating MultiRocket padding
        mode used to calculate that exact column.
        """
        check_is_fitted(
            self,
            attributes=[
                "base_kernels_",
                "base_indices_",
                "n_features_per_rep_",
            ],
        )
        if isinstance(feature_index, (bool, np.bool_)) or not isinstance(
            feature_index, (int, np.integer)
        ):
            raise TypeError("feature_index must be an integer")

        feature_index = int(feature_index)
        n_raw_total = int(self.n_features_per_rep_[0]) * 4
        n_diff_total = int(self.n_features_per_rep_[1]) * 4
        n_total = n_raw_total + n_diff_total
        if feature_index < 0 or feature_index >= n_total:
            raise IndexError(
                f"feature_index must be in [0, {n_total}), got {feature_index}"
            )

        if feature_index < n_raw_total:
            representation = "raw"
            local_index = feature_index
            dilations = self.dilations_raw_
            features_per_dilation = self.num_features_per_dilation_raw_
            biases = self.biases_raw_
        else:
            representation = "diff"
            local_index = feature_index - n_raw_total
            dilations = self.dilations_diff_
            features_per_dilation = self.num_features_per_dilation_diff_
            biases = self.biases_diff_

        bias_index = local_index // 4
        pooling_index = local_index % 4

        remaining = bias_index
        dilation_index = -1
        kernel_index = -1
        dilation = -1
        bias_rank_within_kernel = -1

        for candidate_dilation_index in range(len(dilations)):
            n_biases_per_kernel = int(
                features_per_dilation[candidate_dilation_index]
            )
            n_biases_this_dilation = 84 * n_biases_per_kernel
            if remaining < n_biases_this_dilation:
                dilation_index = candidate_dilation_index
                kernel_index = remaining // n_biases_per_kernel
                bias_rank_within_kernel = remaining % n_biases_per_kernel
                dilation = int(dilations[candidate_dilation_index])
                break
            remaining -= n_biases_this_dilation

        if dilation_index < 0:
            raise RuntimeError(
                "Internal feature indexing error: unable to decode feature "
                f"{feature_index}"
            )

        use_padding = ((dilation_index % 2 + kernel_index) % 2) == 0
        receptive_field = 1 + 8 * dilation

        return {
            "feature_index": int(feature_index),
            "representation": representation,
            "kernel_index": int(kernel_index),
            "kernel_weights": self.base_kernels_[kernel_index].copy(),
            "kernel_positive_indices": self.base_indices_[kernel_index].copy(),
            "dilation_index": int(dilation_index),
            "dilation": int(dilation),
            "receptive_field": int(receptive_field),
            "padding": bool(use_padding),
            "padding_mode": "same" if use_padding else "valid",
            "bias_index": int(bias_index),
            "bias_rank_within_kernel": int(bias_rank_within_kernel),
            "bias": float(biases[bias_index]),
            "pooling_op": self.POOLING_NAMES[pooling_index],
            "pooling_index": int(pooling_index),
        }


    def plot_kernel_properties(
        self,
        selected_indices,
        *,
        selection_probabilities=None,
        consensus_threshold=None,
        figsize=(14, 8),
    ):
        """Summarize properties of consensus-selected I-ROCKET features.

        This plot describes the feature set retained by resampled consensus
        selection. It does not rank arbitrary transform columns by the
        convenience classifier's coefficients.

        Parameters
        ----------
        selected_indices : array-like of int
            Full-transform indices retained by the fitted selector.
        selection_probabilities : array-like, optional
            One consensus probability per transformed feature. When supplied,
            the final panel shows probabilities within the retained set.
        consensus_threshold : float, optional
            Fitted consensus threshold. When supplied with probabilities, it is
            drawn as a reference line in the final panel.
        figsize : tuple, default=(14, 8)

        Returns
        -------
        matplotlib.figure.Figure
        """
        check_is_fitted(self, attributes=["n_output_features_"])
        indices = np.asarray(selected_indices)
        if indices.ndim != 1:
            raise ValueError("selected_indices must be one-dimensional.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("selected_indices must contain integers.")
        indices = indices.astype(np.int64, copy=False)
        if indices.size == 0:
            raise ValueError("selected_indices must contain at least one feature.")
        if np.any(indices < 0) or np.any(indices >= self.n_output_features_):
            raise ValueError("selected_indices contains an out-of-range index.")
        if np.unique(indices).size != indices.size:
            raise ValueError("selected_indices must not contain duplicates.")

        probabilities = None
        if selection_probabilities is not None:
            probabilities = np.asarray(selection_probabilities, dtype=float)
            if (
                probabilities.ndim != 1
                or probabilities.size != self.n_output_features_
            ):
                raise ValueError(
                    "selection_probabilities must contain one value per "
                    "transformed feature."
                )
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("selection_probabilities must be finite.")
            if np.any((probabilities < 0.0) | (probabilities > 1.0)):
                raise ValueError("selection_probabilities must be in [0, 1].")

        if consensus_threshold is not None:
            if not np.isscalar(consensus_threshold):
                raise TypeError("consensus_threshold must be a scalar.")
            consensus_threshold = float(consensus_threshold)
            if not np.isfinite(consensus_threshold) or not 0.0 < consensus_threshold <= 1.0:
                raise ValueError("consensus_threshold must be in (0, 1].")
            if probabilities is None:
                raise ValueError(
                    "selection_probabilities are required when "
                    "consensus_threshold is supplied."
                )

        decoded = [self.decode_feature_index(int(index)) for index in indices]
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        dilations = np.asarray([item["dilation"] for item in decoded])
        axes[0, 0].hist(
            dilations,
            bins=min(20, max(5, np.unique(dilations).size)),
        )
        axes[0, 0].set_title("Selected dilations")
        axes[0, 0].set_xlabel("Dilation")
        axes[0, 0].set_ylabel("Feature count")

        receptive_fields = np.asarray(
            [item["receptive_field"] for item in decoded]
        )
        axes[0, 1].hist(
            receptive_fields,
            bins=min(20, max(5, np.unique(receptive_fields).size)),
        )
        axes[0, 1].set_title("Selected receptive fields")
        axes[0, 1].set_xlabel("Width (timepoints)")
        axes[0, 1].set_ylabel("Feature count")

        pooling = [item["pooling_op"] for item in decoded]
        pooling_counts = [pooling.count(name) for name in self.POOLING_NAMES]
        axes[0, 2].bar(
            self.POOLING_NAMES,
            pooling_counts,
            color=[POOLING_COLORS[name] for name in self.POOLING_NAMES],
        )
        axes[0, 2].set_title("Selected pooling operators")
        axes[0, 2].set_ylabel("Feature count")

        representations = [item["representation"] for item in decoded]
        rep_names = [
            name for name in ("raw", "diff") if name in representations
        ]
        rep_counts = [representations.count(name) for name in rep_names]
        rep_colors = {"raw": OI[0], "diff": OI[5]}
        axes[1, 0].bar(
            rep_names,
            rep_counts,
            color=[rep_colors[name] for name in rep_names],
        )
        axes[1, 0].set_title("Selected representations")
        axes[1, 0].set_ylabel("Feature count")

        kernel_indices = np.asarray(
            [item["kernel_index"] for item in decoded]
        )
        axes[1, 1].hist(kernel_indices, bins=np.arange(-0.5, 84.5, 4.0))
        axes[1, 1].set_title("Selected base kernels")
        axes[1, 1].set_xlabel("Kernel index")
        axes[1, 1].set_ylabel("Feature count")

        if probabilities is not None:
            axes[1, 2].hist(
                probabilities[indices], bins=np.linspace(0.0, 1.0, 21)
            )
            if consensus_threshold is not None:
                axes[1, 2].axvline(
                    consensus_threshold,
                    color="#000000",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Threshold = {consensus_threshold:.2f}",
                )
                axes[1, 2].legend(fontsize=8)
            axes[1, 2].set_xlabel("Selection probability")
            axes[1, 2].set_title("Consensus strength")
        else:
            axes[1, 2].hist(
                [item["bias"] for item in decoded], bins=20
            )
            axes[1, 2].set_xlabel("Bias threshold")
            axes[1, 2].set_title("Selected bias thresholds")
        axes[1, 2].set_ylabel("Feature count")

        fig.suptitle(
            f"Consensus-selected I-ROCKET features (n={indices.size})",
            fontsize=13,
            y=1.01,
        )
        fig.tight_layout()
        return fig



class InterpRocket(ClassifierMixin, InterpRocketTransform):
    """Convenience classifier combining the I-ROCKET transform and ridge CV.

    ``InterpRocketTransform`` is the preferred component for nested pipelines.
    This wrapper preserves the established one-estimator API for direct fitting,
    prediction, feature decoding, and the existing interpretation methods.
    """

    def __init__(
        self,
        max_dilations_per_kernel=16,
        num_features=10000,
        random_state=0,
        alpha_range=None,
        class_weight=None,
        representations="both",
        verbose=False,
    ):
        super().__init__(
            max_dilations_per_kernel=max_dilations_per_kernel,
            num_features=num_features,
            random_state=random_state,
            representations=representations,
            verbose=verbose,
        )
        self.alpha_range = alpha_range
        self.class_weight = class_weight

    def _reset_classifier_state(self):
        """Remove learned classifier state before refitting."""
        for attribute in ("classes_", "scaler_", "classifier_"):
            if hasattr(self, attribute):
                delattr(self, attribute)

    def _validate_parameters(self):
        super()._validate_parameters()
        if self.class_weight is not None and self.class_weight != "balanced":
            if not isinstance(self.class_weight, dict):
                raise ValueError(
                    "class_weight must be None, 'balanced', or a dictionary."
                )
        if self.alpha_range is not None:
            alphas = np.asarray(self.alpha_range, dtype=float)
            if alphas.ndim != 1 or alphas.size == 0:
                raise ValueError("alpha_range must be a non-empty 1D array")
            if not np.all(np.isfinite(alphas)) or np.any(alphas <= 0):
                raise ValueError(
                    "alpha_range must contain only finite positive values"
                )

    def fit(self, X, y):
        """Fit the transform, standardizer, and ridge classifier."""
        self._reset_classifier_state()
        self._reset_transform_state()
        self._validate_parameters()
        X = self._validate_X(X, reset=True)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "X and y contain different numbers of observations: "
                f"{X.shape[0]} and {y.shape[0]}"
            )
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("y must contain at least two classes")

        alpha_range = (
            np.logspace(-10, 10, 20)
            if self.alpha_range is None
            else np.asarray(self.alpha_range, dtype=float)
        )


        self._fit_transform_parameters(X)
        self._log(f"  Classes: {self.classes_}")
        self._log("  Transforming training data...")
        X_features = self._transform(X)
        self._log(f"  Feature matrix: {X_features.shape}")

        self._log("  Standardizing features...")
        self.scaler_ = StandardScaler(with_mean=True)
        X_features = self.scaler_.fit_transform(X_features)

        self._log("  Fitting RidgeClassifierCV...")
        self.classifier_ = RidgeClassifierCV(
            alphas=alpha_range,
            class_weight=self.class_weight,
        )
        self.classifier_.fit(X_features, y)
        self._log(
            f"  Training accuracy: "
            f"{self.classifier_.score(X_features, y):.4f}"
        )
        self._log(f"  Selected alpha: {self.classifier_.alpha_:.4f}")
        return self

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self, attributes=["classifier_", "scaler_"])
        features = self.scaler_.transform(self._transform(X))
        return self.classifier_.predict(features)

    def score(self, X, y):
        """Return classification accuracy, following scikit-learn."""
        return float(accuracy_score(np.asarray(y), self.predict(X)))

    def evaluate(self, X, y):
        """Return the package's full set of classification metrics."""
        return _compute_all_metrics(np.asarray(y), self.predict(X))

    def get_feature_importance(self, feature_mask=None):
        """Return normalized ridge-coefficient importance for transformed columns.

        Binary models use absolute coefficients. Multiclass models use the L2
        norm across class coefficient vectors. When ``feature_mask`` is supplied,
        columns outside that explicit set receive zero importance.
        """
        check_is_fitted(self, attributes=["classifier_", "n_output_features_"])
        coefficients = np.asarray(self.classifier_.coef_)
        if coefficients.ndim == 1:
            importance = np.abs(coefficients)
        else:
            importance = np.linalg.norm(coefficients, axis=0, ord=2)
        importance = np.asarray(importance, dtype=float)

        if feature_mask is not None:
            indices = _validate_feature_index_array(
                feature_mask, importance.size, name="feature_mask"
            )
            masked = np.zeros_like(importance)
            masked[indices] = importance[indices]
            importance = masked

        maximum = float(np.max(importance)) if importance.size else 0.0
        if maximum > 0.0:
            importance = importance / maximum
        return importance


    def get_top_features(self, n=None, feature_mask=None):
        """Return the highest-weight decoded features from an explicit universe.

        ``feature_mask`` is commonly the consensus-selected set produced by
        ``ResampledShrinkageSelector``. Ranking within that set uses the fitted
        ridge coefficients; the selection step itself is not coefficient based.
        """
        importance = self.get_feature_importance(feature_mask=feature_mask)
        if feature_mask is None:
            candidates = np.arange(importance.size, dtype=np.int64)
        else:
            candidates = _validate_feature_index_array(
                feature_mask, importance.size, name="feature_mask"
            )

        if n is None:
            n = len(candidates) if feature_mask is not None else min(20, len(candidates))
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer or None.")
        if n < 1:
            raise ValueError("n must be positive.")
        n = min(int(n), len(candidates))
        order = np.argsort(importance[candidates], kind="stable")[::-1]
        top_indices = candidates[order[:n]]

        results = []
        for index in top_indices:
            info = self.decode_feature_index(int(index))
            info["importance"] = float(importance[index])
            info["feature_index"] = int(index)
            results.append(info)
        return results


    def plot_top_kernels(
        self,
        X,
        y,
        n_kernels=None,
        n_examples=3,
        figsize=None,
        feature_mask=None,
        show_difference=False,
        colors=None,
        unique_kernels=True,
    ):
        """Visualize the highest-ranked selected kernel configurations.

        Each row contains a kernel weight pattern and class-specific activation
        rates.  A MultiRocket feature is defined by its representation, kernel,
        dilation, padding mode, bias threshold, and pooling operator.  Several
        selected features can therefore share one convolutional kernel while
        using different thresholds or pooling summaries.

        By default, this method shows one representative feature--the one with
        the largest ridge importance--for each unique convolutional kernel
        configuration.  This restores the historical meaning of "top kernels"
        and prevents repeated rows for different biases from masquerading as
        the same kernel.  Set ``unique_kernels=False`` to inspect every selected
        feature.  In that mode, the title includes the transformed feature
        index and bias rank/value so distinct activation patterns are explicit.

        The activation curve is the pre-pooling threshold response.  Pooling
        labels identify the selected transformed feature, but pooling itself is
        applied after this binary activation has been calculated.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like
        n_kernels : int or None
            Number of rows to display. ``None`` uses five.
        n_examples : int, default=3
            Number of example signals shown per class. Activation rates use all
            examples in each class.
        figsize : tuple, optional
        feature_mask : array-like of int, optional
            Candidate transformed columns. StableRocketClassifier supplies its
            consensus-selected set automatically.
        show_difference : bool, default=False
            Add a final column containing max-minus-min activation rate across
            classes.
        colors : list of str, optional
            One line color per class. Defaults to the Okabe-Ito palette.
        unique_kernels : bool, default=True
            If True, retain only the highest-ranked feature for each
            ``(representation, kernel, dilation, padding)`` configuration. If
            False, display individual transformed features, including multiple
            bias thresholds or pooling operators from the same kernel.

        Returns
        -------
        fig : matplotlib Figure
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or y.size != X.shape[0]:
            raise ValueError("X must be 2D and y must contain one label per row.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")
        if isinstance(unique_kernels, (np.bool_, bool)):
            unique_kernels = bool(unique_kernels)
        else:
            raise TypeError("unique_kernels must be boolean.")

        classes = np.unique(y)
        n_classes = len(classes)
        n_timepoints = X.shape[1]

        if n_kernels is None:
            n_kernels = 5
        if isinstance(n_kernels, (bool, np.bool_)) or not isinstance(
            n_kernels, (int, np.integer)
        ):
            raise TypeError("n_kernels must be an integer or None.")
        if n_kernels < 1:
            raise ValueError("n_kernels must be positive.")
        n_kernels = int(n_kernels)

        importance = self.get_feature_importance(feature_mask=feature_mask)
        if feature_mask is None:
            candidates = np.arange(importance.size, dtype=np.int64)
        else:
            candidates = _validate_feature_index_array(
                feature_mask, importance.size, name="feature_mask"
            )
        order = np.argsort(importance[candidates], kind="stable")[::-1]
        ranked_indices = candidates[order]

        plotted_features = []
        seen = set()
        for feature_index in ranked_indices:
            info = self.decode_feature_index(int(feature_index))
            info["importance"] = float(importance[feature_index])
            if unique_kernels:
                key = _kernel_configuration_key(info)
                if key in seen:
                    continue
                seen.add(key)
            plotted_features.append(info)
            if len(plotted_features) >= n_kernels:
                break

        if not plotted_features:
            raise ValueError("No features are available for plotting.")
        n_rows = len(plotted_features)

        n_cols = 1 + n_classes
        width_ratios = [1] + [3] * n_classes
        if show_difference:
            n_cols += 1
            width_ratios.append(3)

        if figsize is None:
            figsize = (4.5 * n_cols, 3.5 * n_rows)

        if colors is not None:
            class_colors = list(colors)
            if len(class_colors) < n_classes:
                raise ValueError(
                    f"colors has {len(class_colors)} entries but there are "
                    f"{n_classes} classes."
                )
        else:
            class_colors = [OI[i % len(OI)] for i in range(n_classes)]

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
        )
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row, kinfo in enumerate(plotted_features):
            ki = kinfo["kernel_index"]
            dil = kinfo["dilation"]
            rep = kinfo["representation"]
            bias = kinfo["bias"]
            imp = kinfo["importance"]

            ax = axes[row, 0]
            weights = kinfo["kernel_weights"]
            positions = np.arange(9) * dil
            ax.bar(
                positions,
                weights,
                width=max(dil * 0.6, 0.6),
                color="#7f7f7f",
                edgecolor="#2c2c2c",
                linewidth=0.5,
            )
            ax.set_title(
                f"{_format_feature_label(kinfo, compact=False)}\n"
                f"importance={imp:.4f}",
                fontsize=8,
            )
            ax.set_xlabel("Dilated position")
            ax.axhline(0, color="#7f7f7f", linewidth=0.5)
            ax.set_ylabel("Weight")

            class_act_rates = []
            for cls_idx, cls in enumerate(classes):
                ax = axes[row, 1 + cls_idx]
                X_cls = X[y == cls]
                X_use = (
                    np.diff(X_cls, axis=1).astype(np.float32)
                    if rep == "diff"
                    else X_cls
                )
                n_use = len(X_use)
                n_plot = min(int(n_examples), n_use)
                act_count = np.zeros(n_timepoints, dtype=np.float64)
                total_count = np.zeros(n_timepoints, dtype=np.float64)

                for x in X_use:
                    _, act, time_idx = compute_activation_map(
                        x,
                        ki,
                        np.int32(dil),
                        np.float32(bias),
                        kinfo["padding_mode"],
                        rep,
                    )
                    for local_index, active in enumerate(act):
                        center = int(round(time_idx[local_index]))
                        if rep == "diff":
                            center = min(center + 1, n_timepoints - 1)
                        if 0 <= center < n_timepoints:
                            total_count[center] += 1.0
                            if active > 0:
                                act_count[center] += 1.0

                act_rate = np.zeros(n_timepoints, dtype=np.float64)
                valid = total_count > 0
                act_rate[valid] = act_count[valid] / total_count[valid]
                class_act_rates.append(act_rate)

                for ex_idx in range(n_plot):
                    x = X_use[ex_idx]
                    ax.plot(
                        np.arange(len(x)),
                        x,
                        alpha=0.3,
                        linewidth=0.8,
                        color="#7f7f7f",
                    )

                ax2 = ax.twinx()
                ax2.plot(
                    np.arange(n_timepoints),
                    act_rate,
                    color=class_colors[cls_idx],
                    linewidth=1.5,
                    alpha=0.85,
                )
                ax2.set_ylim(0, 1)
                if cls_idx == n_classes - 1:
                    ax2.set_ylabel("Activation rate", fontsize=8)
                else:
                    ax2.set_yticklabels([])

                ax.set_title(f"Class {cls} ({rep} signal)", fontsize=9)
                ax.set_xlabel("Timepoint")
                if cls_idx == 0:
                    ax.set_ylabel("Amplitude")

            if show_difference:
                ax_diff = axes[row, -1]
                class_act_array = np.asarray(class_act_rates)
                diff_rate = np.max(class_act_array, axis=0) - np.min(
                    class_act_array, axis=0
                )
                ax_diff.plot(
                    np.arange(n_timepoints),
                    diff_rate,
                    color="#2c2c2c",
                    linewidth=1.5,
                )
                ax_diff.set_xlabel("Timepoint")
                ax_diff.set_title(
                    "Differential\n(max - min across classes)"
                    if row == 0
                    else "Differential",
                    fontsize=9,
                )
                ax_diff.set_ylabel("Delta act. rate", fontsize=8)
                for cls_idx, cls in enumerate(classes):
                    ax_diff.plot(
                        np.arange(n_timepoints),
                        class_act_rates[cls_idx],
                        color=class_colors[cls_idx],
                        linewidth=1.5,
                        alpha=0.3,
                        label=f"Class {cls}" if row == 0 else None,
                    )
                if row == 0:
                    ax_diff.legend(fontsize=7, loc="upper right")

        figure_unit = "unique kernel configurations" if unique_kernels else "features"
        fig.suptitle(f"Top {n_rows} {figure_unit}", fontsize=11, y=1.002)
        plt.tight_layout()
        return fig


    def plot_feature_distributions(
        self, X, y, n_top=None, figsize=None, feature_mask=None
    ):
        """
        Plot class-conditional distributions of top features.

        Shows histograms of feature values split by class, revealing
        whether features actually separate classes.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like
        n_top : int or None
            Number of top features to plot. If None, defaults to
            len(feature_mask) when a mask is provided, or 12 otherwise.
        figsize : tuple, optional
        feature_mask : array-like of int, optional
            If provided, only these feature indices are eligible for ranking.
            Typically the consensus-selected feature indices.

        Returns
        -------
        fig : matplotlib Figure
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)

        # Resolve n_top: default to all survivors if mask provided
        if n_top is None:
            if feature_mask is not None:
                n_top = len(feature_mask)
            else:
                n_top = 12

        X_features = self._transform(X)
        top_features = self.get_top_features(n=n_top, feature_mask=feature_mask)

        ncols = 4
        nrows = (n_top + ncols - 1) // ncols
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        for i, finfo in enumerate(top_features):
            if i >= len(axes):
                break
            ax = axes[i]
            fidx = finfo["feature_index"]
            vals = X_features[:, fidx]

            for cls in classes:
                mask = y == cls
                ax.hist(
                    vals[mask], bins=30, alpha=0.5, density=True, label=f"Class {cls}"
                )

            ax.set_title(
                f"{_format_feature_label(finfo, compact=True)}\n"
                f"importance={finfo['importance']:.3f}",
                fontsize=8,
            )
            ax.legend(fontsize=7)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Class-Conditional Feature Distributions", fontsize=12, y=1.01)
        plt.tight_layout()
        return fig




    def summary(self):
        """Print a summary of the fitted model."""
        if not hasattr(self, "classifier_"):
            print("Model not fitted yet.")
            return

        n_raw = self.n_features_per_rep_[0]
        n_diff = self.n_features_per_rep_[1]
        total = (n_raw + n_diff) * 4

        print("=" * 60)
        print("InterpRocket Model Summary")
        print("=" * 60)
        print(f"  Base kernels: 84 (length 9, weights {{-1, 2}})")
        print(f"  Dilations (raw):  {self.dilations_raw_}")
        print(f"  Dilations (diff): {self.dilations_diff_}")
        print(f"  Features per representation (biases):")
        print(f"    Raw:  {n_raw} biases × 4 pooling ops = {n_raw * 4}")
        print(f"    Diff: {n_diff} biases × 4 pooling ops = {n_diff * 4}")
        print(f"  Total features: {total}")
        print(f"  Classifier: RidgeClassifierCV (alpha={self.classifier_.alpha_:.4f})")
        print(f"  Classes: {self.classes_}")
        print()
        print("  Top 10 features:")
        for i, f in enumerate(self.get_top_features(10)):
            print(
                f"    {i+1}. {_format_feature_label(f, compact=False)} "
                f"importance={f['importance']:.4f} RF={f['receptive_field']}"
            )
        print("=" * 60)


