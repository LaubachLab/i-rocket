"""
interp_rocket.py - Interpretable ROCKET for Time Series Classification

A standalone, fully transparent reimplementation of MultiRocket (Tan et al.,
2022) with complete kernel-level interpretability. Designed for scientific
applications where understanding why a classifier makes its decisions is as
important as the accuracy.

Inspired by the transparent parameter storage in msROCKET (Lundy and O'Toole,
2021), which was based on the vanilla ROCKET classifier.

WHAT THIS DOES:
    MultiRocket uses 84 deterministic base kernels (from MiniRocket) applied
    at multiple dilations to both the raw signal and its first-order
    difference. Four pooling operators (PPV, MPV, MIPV, LSPV) extract
    features from each convolution output. A linear classifier (Ridge) trains
    on these features.

    Unlike sktime/aeon implementations, every parameter is stored in plain
    numpy arrays with documented indexing, making it possible to:
    1. Trace any feature back to its kernel, dilation, pooling op, and
       signal type via decode_feature_index()
    2. Visualize what each important kernel detects in a time series
    3. Map classifier importance to temporal regions of the input
    4. Identify robust features via cross-validation stability analysis
    5. Test feature significance via permutation importance (PIMP)
    6. Decompose feature contributions as redundant, synergistic, or
       independent using information-theoretic methods
    7. Assess temporal sensitivity via model-agnostic occlusion

ARCHITECTURE:
    84 base kernels x D dilations x 2 representations x 4 pooling ops
    where D depends on series length (controlled by max_dilations_per_kernel,
    default 16). The distribution across dilations is fitted to the data.

FEATURE SELECTION:
    Feature stability analysis (FSA) is the recommended method. It
    identifies features that are consistently ranked as important across
    cross-validation folds (Meinshausen and Buhlmann, 2010; Saeys et al.,
    2008). Permutation importance (PIMP) provides an independent statistical
    test using RandomForest to confirm that feature importance exceeds
    chance (Altmann et al., 2010). Recursive feature elimination (RFE) is
    available but not recommended as primary method due to sensitivity to
    random seed and data split.

INTERPRETABILITY TOOLS:
    - Temporal importance profiles (differential activation method)
    - Receptive field diagrams (feature RF at peak discriminative location)
    - Class-mean activation maps (kernel response on class-averaged signals)
    - Aggregate activation (importance-weighted sum with differential)
    - Multi-kernel summary (binary activation heatmap across features)
    - Temporal occlusion sensitivity (per-trial and aggregate)
    - Confusion-conditioned activation maps (correct vs. misclassified)
    - Information decomposition (redundant/synergistic/independent)
    - Kernel similarity network (correlation structure among features)
    - Feature distribution analysis (per-class histograms)

COLOR PALETTE:
    All plotting functions use a consistent tab10 hex palette defined at
    module level (TAB10, POOLING_COLORS, INFO_COLORS).

KEY DIFFERENCES FROM SKTIME/AEON:
    - All kernel weights, dilations, biases stored as accessible numpy arrays
    - Complete feature to kernel to timepoint traceability
    - Integrated visualization and analysis suite
    - Feature stability analysis for robust feature selection
    - Permutation importance with statistical testing (PIMP)
    - Information-theoretic feature decomposition
    - Class balancing via random oversampling for imbalanced data
    - NumPy 2.x compatible
    - Single-file, no framework dependencies beyond numpy/numba/sklearn/matplotlib

EXTENSIONS (in extensions/ directory):
    - AMEE evaluation: perturbation-based saliency map ranking
    - TSHAP integration: instance-level Shapley value attributions
    - Channel selection: classifier-agnostic multivariate channel selection
    - Kernel explorer: interactive tool for exploring kernels and pooling

REFERENCES:
    Altmann, A., Tolosi, L., Sander, O., & Lengauer, T. (2010). Permutation
    importance: a corrected feature importance measure. Bioinformatics,
    26(10), 1340-1347.

    Meinshausen, N. & Buhlmann, P. (2010). Stability selection. Journal of
    the Royal Statistical Society: Series B, 72(4), 417-473.

    Narayanan, N. S., Kimchi, E. Y., & Laubach, M. (2005). Redundancy and
    synergy of neuronal ensembles in motor cortex. Journal of Neuroscience,
    25(17), 4207-4216.

    Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022).
    MultiRocket: multiple pooling operators and transformations for fast and
    effective time series classification. Data Mining and Knowledge
    Discovery, 36(5), 1623-1646.

    Lundy, C., & O'Toole, J. M. (2021). Random convolution kernels with
    multi-scale decomposition for preterm EEG inter-burst detection. In 2021
    29th European Signal Processing Conference (EUSIPCO) (pp. 1182-1186).

    Uribarri, G., Barone, F., Ansuini, A., & Fransen, E. (2024).
    Detach-ROCKET: sequential feature selection for time series
    classification with random convolutional kernels. Data Mining and
    Knowledge Discovery, 38(6), 3922-3947.

USAGE:
    import interp_rocket as IR

    model = IR.InterpRocket(max_dilations_per_kernel=16)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    # Feature stability analysis
    stability = IR.cv_feature_stability(X_train, y_train)
    stable = IR.get_stable_features(stability, threshold=0.8)

    # Visualization (constrained by stable features)
    model.plot_temporal_importance(X_test, y_test, feature_mask=stable)
    IR.plot_receptive_field_diagram(model, X_test, y_test, feature_mask=stable)

    # Permutation importance
    pimp = IR.permutation_importance_test(model, X_train, y_train)
    IR.plot_permutation_importance(pimp, model=model)

    # Cross-validation
    results = IR.cross_validate(X, y, n_repeats=10, n_folds=10, n_jobs=-2)

REQUIREMENTS:
    numpy, numba (>=0.50), scikit-learn, matplotlib
    Compatible with: NumPy 2.x, Python 3.10+

Author: Mark Laubach (American University, Department of Neuroscience)
        Developed with Claude (Anthropic) as AI coding assistant.
License: BSD-3-Clause
"""

__version__ = "0.6.1"

import numpy as np
from itertools import combinations
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ============================================================================
# COLOR PALETTE (tab10 as hex, used throughout all plotting functions)
# ============================================================================

TAB10 = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

POOLING_COLORS = {
    "PPV": "#1f77b4",   # blue
    "MPV": "#ff7f0e",   # orange
    "MIPV": "#2ca02c",  # green
    "LSPV": "#9467bd",  # purple
}

INFO_COLORS = {
    "redundant": "#ff7f0e",    # orange
    "synergistic": "#1f77b4",  # blue
    "independent": "#7f7f7f",  # gray
}

# ============================================================================
# SECTION 1: THE 84 BASE KERNELS
# ============================================================================
#
# MiniRocket/MultiRocket use 84 deterministic kernels of length 9.
# Each kernel has weights from {-1, 2}: six positions get -1, three get 2.
# The 84 kernels enumerate all C(9,3) = 84 ways to choose which 3 of 9
# positions receive the weight 2 (the rest get -1).

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

# ============================================================================
# SECTION 2: DILATION FITTING
# ============================================================================
#
# Dilations control the temporal scale each kernel operates at.
# For a kernel of length 9 with dilation d, the receptive field spans
# 1 + 8*d timepoints. The set of dilations is chosen so that the largest
# dilation produces a receptive field just under the series length.

@njit(cache=True)
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
    num_kernels = 84
    num_features_per_kernel = num_features // num_kernels

    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )

    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    # Maximum dilation such that receptive field fits in input
    max_exponent = np.log2((input_length - 1) / (9 - 1))
    max_exponent = max(max_exponent, 0)

    num_dilations = min(true_max_dilations_per_kernel, int(max_exponent) + 1)

    # Exponentially spaced dilations
    dilations = np.zeros(num_dilations, dtype=np.int32)
    for i in range(num_dilations):
        dilations[i] = np.int32(2 ** (i * max_exponent / max(num_dilations - 1, 1)))

    # Distribute features across dilations
    num_features_per_dilation = np.zeros(num_dilations, dtype=np.int32)
    for i in range(num_dilations):
        num_features_per_dilation[i] = np.int32(
            (i + 1) * multiplier - np.sum(num_features_per_dilation)
        )

    return dilations, num_features_per_dilation

# ============================================================================
# SECTION 3: QUANTILE GENERATION (for bias selection)
# ============================================================================

@njit(cache=True)
def _quantiles(n):
    """
    Generate low-discrepancy quantiles using the golden ratio sequence.

    These quantiles are used to sample biases from the distribution of
    convolution outputs on training data. The golden ratio sequence
    produces quasi-random, well-distributed quantiles.

    Parameters
    ----------
    n : int
        Number of quantiles to generate.

    Returns
    -------
    quantiles : ndarray of float32
        Values in (0, 1), well-distributed.
    """
    phi = (np.sqrt(np.float32(5.0)) + 1.0) / 2.0
    quantiles = np.zeros(n, dtype=np.float32)
    for i in range(n):
        quantiles[i] = ((i + 1) * phi) % 1.0
    return quantiles

# ============================================================================
# SECTION 4: BIAS FITTING
# ============================================================================
#
# Biases are the only truly data-dependent parameters. They are set by
# convolving a sample of training instances with each kernel at each
# dilation, then selecting quantiles of the resulting convolution outputs
# as bias thresholds.

@njit(fastmath=True, cache=True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles, random_state_seed):
    """
    Fit biases from training data for all 84 kernels × all dilations.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints), dtype float32
        Training time series.
    dilations : ndarray of int32
        Dilation values.
    num_features_per_dilation : ndarray of int32
        Number of features (biases) per dilation.
    quantiles : ndarray of float32
        Quantile positions for bias selection.
    random_state_seed : int
        Seed for reproducibility.

    Returns
    -------
    biases : ndarray of float32
        One bias per feature. Length = 84 * sum(num_features_per_dilation).
    """
    np.random.seed(random_state_seed)

    num_instances, input_length = X.shape

    # The 84 index patterns (positions of weight=2 in each kernel)
    # Regenerated here inside numba context
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

    # Use min(n_instances, 10) examples for bias fitting (like MiniRocket)
    num_examples = min(num_instances, 100)

    feature_idx = 0

    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):
            # Indices of the 3 positions with weight 2
            i0 = indices_raw[kernel_index, 0]
            i1 = indices_raw[kernel_index, 1]
            i2 = indices_raw[kernel_index, 2]

            # Collect convolution outputs from sample of training instances
            # For each example, compute the convolution output at each position
            n_conv = input_length + 2 * padding - (9 - 1) * dilation
            if n_conv < 1:
                n_conv = 1

            C_all = np.zeros(num_examples * n_conv, dtype=np.float32)
            c_idx = 0

            for example_index in range(num_examples):
                # Random selection of training examples
                ex = np.random.randint(num_instances)
                x = X[ex]

                for t in range(n_conv):
                    # Sum of all 9 positions at this dilation
                    total = np.float32(0.0)
                    for pos in range(9):
                        input_idx = t + pos * dilation - padding
                        if 0 <= input_idx < input_length:
                            total += x[input_idx]

                    # The kernel: -1 at 6 positions, +2 at 3 positions
                    # = -sum_all + 3 * sum_at_indices
                    # Because: -1*sum_all + 2*sum_indices + 1*sum_indices
                    # Wait, let's be precise:
                    # w = -1 everywhere, then w[i0,i1,i2] = 2
                    # conv = sum(w * x_dilated) = -sum_all + 3*sum_at_indices
                    sum_at_indices = np.float32(0.0)
                    for idx_val in (i0, i1, i2):
                        input_idx = t + idx_val * dilation - padding
                        if 0 <= input_idx < input_length:
                            sum_at_indices += x[input_idx]

                    conv_val = -total + 3.0 * sum_at_indices
                    C_all[c_idx] = conv_val
                    c_idx += 1

            # Select biases as quantiles of the convolution output
            C_sorted = np.sort(C_all[:c_idx])

            for feature_count in range(num_features_this_dilation):
                q = quantiles[feature_idx]
                bias_index = int(q * (c_idx - 1))
                if bias_index < 0:
                    bias_index = 0
                if bias_index >= c_idx:
                    bias_index = c_idx - 1
                biases[feature_idx] = C_sorted[bias_index]
                feature_idx += 1

    return biases

# ============================================================================
# SECTION 5: TRANSFORM — The Core Convolution + Pooling
# ============================================================================
#
# This is where the features are actually extracted. For each series:
# 1. Convolve with each of 84 kernels at each dilation
# 2. Subtract each bias → get binary indicator (>0 or not)
# 3. Compute 4 pooling operators: PPV, MPV, MIPV, LSPV
#
# MultiRocket does this for both raw signal and first-order difference.

@njit(fastmath=True, parallel=True, cache=True)
def _transform(X, dilations, num_features_per_dilation, biases):
    """
    Transform time series using MiniRocket-style convolution + 4 pooling ops.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints), dtype float32
    dilations : ndarray of int32
    num_features_per_dilation : ndarray of int32
    biases : ndarray of float32

    Returns
    -------
    features : ndarray, shape (n_instances, n_features * 4)
        4 features per bias: PPV, MPV, MIPV, LSPV
    """
    num_instances, input_length = X.shape
    num_kernels = 84
    num_dilations = len(dilations)
    num_features_per_rep = num_kernels * np.sum(num_features_per_dilation)

    # 4 pooling operators per feature
    features = np.zeros((num_instances, num_features_per_rep * 4), dtype=np.float32)

    # Regenerate indices inside numba
    indices_raw = np.zeros((84, 3), dtype=np.int32)
    count = 0
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                indices_raw[count, 0] = i
                indices_raw[count, 1] = j
                indices_raw[count, 2] = k
                count += 1

    for instance_idx in prange(num_instances):
        x = X[instance_idx]
        feature_idx = 0

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            n_conv = input_length + 2 * padding - (9 - 1) * dilation
            if n_conv < 1:
                n_conv = 1

            for kernel_index in range(num_kernels):
                i0 = indices_raw[kernel_index, 0]
                i1 = indices_raw[kernel_index, 1]
                i2 = indices_raw[kernel_index, 2]

                # Compute full convolution output
                C = np.zeros(n_conv, dtype=np.float32)
                for t in range(n_conv):
                    total = np.float32(0.0)
                    for pos in range(9):
                        input_idx = t + pos * dilation - padding
                        if 0 <= input_idx < input_length:
                            total += x[input_idx]

                    sum_at_indices = np.float32(0.0)
                    for idx_val in (i0, i1, i2):
                        input_idx = t + idx_val * dilation - padding
                        if 0 <= input_idx < input_length:
                            sum_at_indices += x[input_idx]

                    C[t] = -total + 3.0 * sum_at_indices

                # For each bias, compute the 4 pooling operators
                for feature_count in range(num_features_this_dilation):
                    bias = biases[feature_idx]

                    # ---- PPV: Proportion of Positive Values ----
                    ppv_count = 0
                    # ---- MPV: Mean of Positive Values ----
                    mpv_sum = np.float32(0.0)
                    mpv_count = 0
                    # ---- MIPV: Mean of Indices of Positive Values ----
                    mipv_sum = np.float32(0.0)
                    # ---- LSPV: Longest Stretch of Positive Values ----
                    lspv_max = 0
                    lspv_current = 0

                    for t in range(n_conv):
                        val = C[t] - bias  # shifted value
                        if val > 0:
                            ppv_count += 1
                            mpv_sum += val
                            mpv_count += 1
                            mipv_sum += t
                            lspv_current += 1
                            if lspv_current > lspv_max:
                                lspv_max = lspv_current
                        else:
                            lspv_current = 0

                    ppv = np.float32(ppv_count) / np.float32(n_conv)
                    mpv = (
                        mpv_sum / np.float32(mpv_count)
                        if mpv_count > 0
                        else np.float32(0.0)
                    )
                    mipv = (
                        mipv_sum / np.float32(ppv_count)
                        if ppv_count > 0
                        else np.float32(-1.0)
                    )
                    lspv = np.float32(lspv_max)

                    # Store: 4 features per bias, contiguously
                    base = feature_idx * 4
                    features[instance_idx, base] = ppv
                    features[instance_idx, base + 1] = mpv
                    features[instance_idx, base + 2] = mipv
                    features[instance_idx, base + 3] = lspv

                    feature_idx += 1

    return features

# ============================================================================
# SECTION 6: PER-TIMEPOINT ACTIVATION MAP
# ============================================================================

@njit(fastmath=True, cache=True)
def compute_activation_map(x, kernel_index, dilation, bias):
    """
    Compute the per-timepoint convolution output for a single kernel+dilation+bias.

    Returns the raw convolution output (before bias subtraction) and the
    binary activation (after bias subtraction), allowing visualization of
    exactly where this kernel "fires" on the input.

    Parameters
    ----------
    x : ndarray, shape (n_timepoints,), dtype float32
    kernel_index : int
        Which of the 84 base kernels (0-83).
    dilation : int
    bias : float32

    Returns
    -------
    conv_output : ndarray, shape (n_conv,)
        Raw convolution values.
    activation : ndarray, shape (n_conv,)
        Binary: 1 where conv_output > bias, else 0.
    time_indices : ndarray, shape (n_conv,)
        The center timepoint each convolution position maps to.
    """
    input_length = len(x)
    padding = ((9 - 1) * dilation) // 2

    # Regenerate indices
    indices_raw = np.zeros((84, 3), dtype=np.int32)
    count = 0
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                indices_raw[count, 0] = i
                indices_raw[count, 1] = j
                indices_raw[count, 2] = k
                count += 1

    i0 = indices_raw[kernel_index, 0]
    i1 = indices_raw[kernel_index, 1]
    i2 = indices_raw[kernel_index, 2]

    n_conv = input_length + 2 * padding - (9 - 1) * dilation
    if n_conv < 1:
        n_conv = 1

    conv_output = np.zeros(n_conv, dtype=np.float32)
    activation = np.zeros(n_conv, dtype=np.float32)
    time_indices = np.zeros(n_conv, dtype=np.float32)

    for t in range(n_conv):
        # Center of receptive field
        center = t - padding + 4 * dilation
        time_indices[t] = np.float32(center)

        total = np.float32(0.0)
        for pos in range(9):
            input_idx = t + pos * dilation - padding
            if 0 <= input_idx < input_length:
                total += x[input_idx]

        sum_at_indices = np.float32(0.0)
        for idx_val in (i0, i1, i2):
            input_idx = t + idx_val * dilation - padding
            if 0 <= input_idx < input_length:
                sum_at_indices += x[input_idx]

        conv_val = -total + 3.0 * sum_at_indices
        conv_output[t] = conv_val
        activation[t] = np.float32(1.0) if conv_val > bias else np.float32(0.0)

    return conv_output, activation, time_indices

# ============================================================================
# SECTION 7: MUTUAL INFORMATION
# ============================================================================
#
# Information-theoretic classification metric. Measures how much knowing
# the predicted label reduces uncertainty about the true label.
# Ported from R code by Mark Laubach (version from 2005).

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

# ============================================================================
# SECTION 8: THE INTERPRETABLE ROCKET CLASS
# ============================================================================

class InterpRocket(BaseEstimator, ClassifierMixin):
    """
    Interpretable MultiRocket classifier.

    Provides full traceability from classifier decision → feature importance →
    kernel identity → temporal activation pattern.

    Inherits from sklearn.base.BaseEstimator and ClassifierMixin, providing
    get_params(), set_params(), and a standard score(X, y) that returns
    accuracy as a scalar for compatibility with sklearn pipelines. For the
    full multi-metric evaluation, use evaluate(X, y).

    Parameters
    ----------
    max_dilations_per_kernel : int, default=32
        Maximum number of dilation values per kernel.
    num_features : int, default=10000
        Target number of features per representation.
        Actual count: 2 representations × 4 pooling ops × (rounded to 84 multiple).
    random_state : int, default=0
        Seed for reproducibility (only affects bias fitting and class balancing).
    alpha_range : ndarray, optional
        Range of Ridge regularization parameters.
    class_weight : str or None, default=None
        If 'balanced', randomly oversample minority class(es) to match
        the majority class count before fitting. Resampling is applied to
        the raw time series before the transform.
    """

    # The 4 pooling operator names, in feature order
    POOLING_NAMES = ["PPV", "MPV", "MIPV", "LSPV"]

    def __init__(
        self,
        max_dilations_per_kernel=16,  # suggested for use with I-ROCKET
        num_features=10000,
        random_state=0,
        alpha_range=None,
        class_weight=None,
    ):
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_features = num_features
        self.random_state = random_state
        self.alpha_range = alpha_range or np.logspace(-10, 10, 20)
        self.class_weight = class_weight

        # Will be set during fit()
        self.base_kernels_ = None
        self.base_indices_ = None
        self.dilations_raw_ = None
        self.dilations_diff_ = None
        self.num_features_per_dilation_raw_ = None
        self.num_features_per_dilation_diff_ = None
        self.biases_raw_ = None
        self.biases_diff_ = None
        self.classifier_ = None
        self.scaler_ = None
        self.n_features_per_rep_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the MultiRocket transform and Ridge classifier.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
            Training time series. Will be converted to float32.
        y : array-like, shape (n_instances,)
            Class labels.

        Returns
        -------
        self

        Notes
        -----
        If class_weight='balanced', the minority class(es) are randomly
        oversampled (with replacement) to match the majority class count
        before fitting. This resampling is applied to the raw time series
        before the transform, ensuring the convolution outputs and bias
        quantiles reflect the balanced distribution.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # --- Class balancing via random oversampling ---
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            max_count = counts.max()
            oversample_idx = []
            rng = np.random.default_rng(self.random_state)
            for cls, cnt in zip(classes, counts):
                cls_idx = np.where(y == cls)[0]
                if cnt < max_count:
                    extra = rng.choice(cls_idx, size=max_count - cnt, replace=True)
                    cls_idx = np.concatenate([cls_idx, extra])
                oversample_idx.append(cls_idx)
            oversample_idx = np.concatenate(oversample_idx)
            rng.shuffle(oversample_idx)
            X = X[oversample_idx]
            y = y[oversample_idx]
            print(
                f"  Class balancing: oversampled to {len(y)} instances "
                f"({max_count} per class)"
            )

        n_instances, input_length = X.shape
        print(f"InterpRocket.fit: {n_instances} instances × {input_length} timepoints")
        print(f"  Classes: {self.classes_}")

        # Generate the 84 base kernels
        self.base_kernels_, self.base_indices_ = _generate_base_kernels()

        # --- Raw representation ---
        print("  Fitting dilations (raw)...")
        self.dilations_raw_, self.num_features_per_dilation_raw_ = _fit_dilations(
            input_length, self.num_features, self.max_dilations_per_kernel
        )

        n_features_raw = 84 * np.sum(self.num_features_per_dilation_raw_)
        quantiles_raw = _quantiles(n_features_raw)

        print(
            f"  Fitting biases (raw): {n_features_raw} biases across "
            f"{len(self.dilations_raw_)} dilations..."
        )
        self.biases_raw_ = _fit_biases(
            X,
            self.dilations_raw_,
            self.num_features_per_dilation_raw_,
            quantiles_raw,
            self.random_state,
        )

        # --- First-difference representation ---
        X_diff = np.diff(X, axis=1).astype(np.float32)
        diff_length = X_diff.shape[1]

        print("  Fitting dilations (diff)...")
        self.dilations_diff_, self.num_features_per_dilation_diff_ = _fit_dilations(
            diff_length, self.num_features, self.max_dilations_per_kernel
        )

        n_features_diff = 84 * np.sum(self.num_features_per_dilation_diff_)
        quantiles_diff = _quantiles(n_features_diff)

        print(
            f"  Fitting biases (diff): {n_features_diff} biases across "
            f"{len(self.dilations_diff_)} dilations..."
        )
        self.biases_diff_ = _fit_biases(
            X_diff,
            self.dilations_diff_,
            self.num_features_per_dilation_diff_,
            quantiles_diff,
            self.random_state + 1,  # different seed for diff
        )

        self.n_features_per_rep_ = (n_features_raw, n_features_diff)

        # --- Transform ---
        print("  Transforming training data...")
        X_features = self._transform(X)

        print(f"  Feature matrix: {X_features.shape}")

        # --- Standardize features ---
        # Ridge regression's L2 penalty is sensitive to feature scale.
        # PPV features are in [0,1] while LSPV and MPV can have much larger
        # ranges. Without standardization, the penalty disproportionately
        # shrinks large-scale features regardless of their discriminative
        # value. This matches the standard ROCKET pipeline.
        print("  Standardizing features...")
        self.scaler_ = StandardScaler(with_mean=True)
        X_features = self.scaler_.fit_transform(X_features)

        # --- Fit classifier ---
        print("  Fitting RidgeClassifierCV...")
        self.classifier_ = RidgeClassifierCV(alphas=self.alpha_range)
        self.classifier_.fit(X_features, y)

        train_acc = self.classifier_.score(X_features, y)
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Selected alpha: {self.classifier_.alpha_:.4f}")

        return self

    def _transform(self, X):
        """
        Transform time series to feature vectors.

        Returns concatenated features: [raw_PPV, raw_MPV, raw_MIPV, raw_LSPV,
                                         diff_PPV, diff_MPV, diff_MIPV, diff_LSPV]

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints), dtype float32

        Returns
        -------
        features : ndarray, shape (n_instances, n_total_features)
        """
        X = np.asarray(X, dtype=np.float32)

        # Raw features
        features_raw = _transform(
            X,
            self.dilations_raw_,
            self.num_features_per_dilation_raw_,
            self.biases_raw_,
        )

        # First-difference features
        X_diff = np.diff(X, axis=1).astype(np.float32)
        features_diff = _transform(
            X_diff,
            self.dilations_diff_,
            self.num_features_per_dilation_diff_,
            self.biases_diff_,
        )

        return np.concatenate([features_raw, features_diff], axis=1)

    def transform(self, X):
        """Public transform method. Returns raw (unscaled) features."""
        return self._transform(X)

    def predict(self, X):
        """Predict class labels."""
        X_features = self._transform(X)
        X_features = self.scaler_.transform(X_features)
        return self.classifier_.predict(X_features)

    def score(self, X, y):
        """
        Return classification accuracy as a scalar (sklearn convention).

        This method exists for compatibility with sklearn pipelines,
        GridSearchCV, and cross_val_score. For the full multi-metric
        evaluation, use evaluate(X, y).

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like

        Returns
        -------
        accuracy : float
        """
        y_pred = self.predict(X)
        return float(accuracy_score(np.asarray(y), y_pred))

    def evaluate(self, X, y):
        """
        Evaluate on test data, returning multiple metrics.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like

        Returns
        -------
        metrics : dict with keys:
            'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
            'mcc' (Matthews correlation coefficient),
            'mutual_info' (bits)
        """
        y_pred = self.predict(X)
        return _compute_all_metrics(np.asarray(y), y_pred)

    # ====================================================================
    # SECTION 9: FEATURE INDEX DECODING
    # ====================================================================

    def decode_feature_index(self, feature_index):
        """
        Map a feature index back to its generating components.

        This is the core interpretability function: given a column index in the
        feature matrix, returns exactly which kernel, dilation, pooling operator,
        and signal representation produced it.

        Parameters
        ----------
        feature_index : int
            Column index in the feature matrix.

        Returns
        -------
        info : dict with keys:
            'representation': str, 'raw' or 'diff'
            'kernel_index': int, 0-83
            'kernel_weights': ndarray, the 9 weights
            'kernel_positive_indices': ndarray, the 3 positions with weight 2
            'dilation': int
            'receptive_field': int, total span in timepoints
            'bias': float
            'pooling_op': str, one of 'PPV', 'MPV', 'MIPV', 'LSPV'
            'pooling_index': int, 0-3
        """
        n_raw_total = self.n_features_per_rep_[0] * 4
        n_diff_total = self.n_features_per_rep_[1] * 4

        if feature_index < n_raw_total:
            rep = "raw"
            local_idx = feature_index
            dilations = self.dilations_raw_
            nfpd = self.num_features_per_dilation_raw_
            biases = self.biases_raw_
        else:
            rep = "diff"
            local_idx = feature_index - n_raw_total
            dilations = self.dilations_diff_
            nfpd = self.num_features_per_dilation_diff_
            biases = self.biases_diff_

        # Each bias produces 4 features (PPV, MPV, MIPV, LSPV)
        bias_index = local_idx // 4
        pooling_index = local_idx % 4

        # Walk through dilation/kernel structure to find which kernel+dilation
        remaining = bias_index
        found = False
        dilation_val = 0
        kernel_idx = 0

        for d_idx in range(len(dilations)):
            n_features_this_dil = nfpd[d_idx]
            n_biases_this_dil = 84 * n_features_this_dil

            if remaining < n_biases_this_dil:
                dilation_val = int(dilations[d_idx])
                kernel_idx = remaining // n_features_this_dil
                found = True
                break
            remaining -= n_biases_this_dil

        if not found:
            raise ValueError(f"Feature index {feature_index} out of range")

        receptive_field = 1 + 8 * dilation_val

        return {
            "representation": rep,
            "kernel_index": kernel_idx,
            "kernel_weights": self.base_kernels_[kernel_idx].copy(),
            "kernel_positive_indices": self.base_indices_[kernel_idx].copy(),
            "dilation": dilation_val,
            "receptive_field": receptive_field,
            "bias": float(biases[bias_index]),
            "pooling_op": self.POOLING_NAMES[pooling_index],
            "pooling_index": pooling_index,
        }

    def get_feature_importance(self, feature_mask=None):
        """
        Get per-feature importance from the Ridge classifier.

        For multi-class problems, importance is the L2 norm of the
        coefficient vector across classes, matching the default
        ('norm') method in Detach-ROCKET (Uribarri et al., 2024).
        For binary problems, importance is the absolute coefficient.

        The raw importance values are L2 norms of Ridge regression
        coefficients, which depend on the scale of the features and
        the regularization strength. They're meaningful for ranking
        but the absolute magnitudes are arbitrary. Here, we normalize
        the values by dividing by the maximum, so the most important
        feature is 1.0 and everything else is a proportion of that.

        Parameters
        ----------
        feature_mask : array-like of int, optional
            If provided, only these feature indices are eligible.
            All other features receive importance = 0.
            Typically the surviving indices from RFE.

        Returns
        -------
        importance : ndarray, shape (n_features,)
        """
        coefs = self.classifier_.coef_
        if coefs.ndim == 1:
            importance = np.abs(coefs)
        else:
            importance = np.linalg.norm(coefs, axis=0, ord=2)

        if feature_mask is not None:
            mask = np.zeros_like(importance)
            feature_mask = np.asarray(feature_mask)
            valid = feature_mask[feature_mask < len(importance)]
            mask[valid] = importance[valid]
            importance = mask

        if importance.max() > 0:
            importance = importance / importance.max()

        return importance

    def get_top_features(self, n=None, feature_mask=None):
        """
        Get the n most important features, fully decoded.

        Parameters
        ----------
        n : int
            Number of top features to return.
        feature_mask : array-like of int, optional
            If provided, only these feature indices are eligible.
            Typically the surviving indices from RFE.

        Returns
        -------
        top_features : list of dicts
            Each dict has the decode_feature_index output plus 'importance'
            and 'feature_index'.
        """

        importance = self.get_feature_importance(feature_mask=feature_mask)
        if n is None:
            if feature_mask is not None:
                n = len(feature_mask)
            else:
                n = 20
        top_idx = np.argsort(importance)[::-1][:n]

        results = []
        for idx in top_idx:
            info = self.decode_feature_index(int(idx))
            info["importance"] = float(importance[idx])
            info["feature_index"] = int(idx)
            results.append(info)

        return results

    # ====================================================================
    # SECTION 10: VISUALIZATION
    # ====================================================================

    def plot_top_kernels(
        self, X, y, n_kernels=None, n_examples=3, figsize=None,
        feature_mask=None, show_difference=False, colors=None,
    ):
        """
        Visualize the top-n most important kernels: their weight patterns
        and per-class activation rates.

        Layout: one row per kernel.
            Column 0: kernel weight pattern (bar chart at dilated positions).
            Columns 1..n_classes: per-class activation rate with example
            signal traces in gray. Each colored line shows the fraction of
            examples in that class where the kernel fires at each timepoint.

        When show_difference=True, an additional final column shows the
        differential activation (max minus min activation rate across
        classes at each timepoint). Peaks in this curve indicate WHERE
        the kernel discriminates between classes, not just where it fires.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like
        n_kernels : int or None
            Number of top kernels to visualize. If None, defaults to
            len(feature_mask) when a mask is provided, or 5 otherwise.
        n_examples : int
            Number of example instances per class to show as signal traces.
            Activation rates are computed from ALL examples in each class.
        figsize : tuple, optional
        feature_mask : array-like of int, optional
            If provided, only these feature indices are eligible for ranking.
            Typically the surviving indices from FSA or RFE.
        show_difference : bool, default=False
            If True, add a final column showing the differential activation
            (max minus min across classes) at each timepoint. Useful for
            identifying where the kernel discriminates between classes.
        colors : list of str, optional
            Colors for per-class activation rate lines, one per class.
            Defaults to TAB10[:n_classes].

        Returns
        -------
        fig : matplotlib Figure
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        n_classes = len(classes)
        n_timepoints = X.shape[1]

        # Resolve n_kernels: default to all survivors if mask provided
        if n_kernels is None:
            if feature_mask is not None:
                n_kernels = len(feature_mask)
            else:
                n_kernels = 5

        # Get all candidate features (limited to mask if provided)
        if feature_mask is not None:
            n_candidates = len(feature_mask)
        else:
            n_candidates = n_kernels * 4
        top_features = self.get_top_features(n=n_candidates, feature_mask=feature_mask)

        # Deduplicate by (kernel_index, dilation, representation) when
        # browsing the full model. When a feature_mask is provided (e.g.,
        # RFE survivors), show every feature -- the user curated the set
        # and different pooling ops from the same kernel carry distinct
        # information.
        if feature_mask is not None:
            unique_kernels = top_features[:n_kernels]
        else:
            seen = set()
            unique_kernels = []
            for f in top_features:
                key = (f["representation"], f["kernel_index"], f["dilation"])
                if key not in seen:
                    seen.add(key)
                    unique_kernels.append(f)
                if len(unique_kernels) >= n_kernels:
                    break
        n_kernels = len(unique_kernels)

        # Layout: kernel weights | class 0 | class 1 | ... | [differential]
        n_cols = 1 + n_classes
        width_ratios = [1] + [3] * n_classes
        if show_difference:
            n_cols += 1
            width_ratios.append(3)

        if figsize is None:
            figsize = (4.5 * n_cols, 3.5 * n_kernels)

        # Class-specific colors for activation rate lines
        if colors is not None:
            class_colors = list(colors)
        else:
            class_colors = [TAB10[i] for i in range(n_classes)]

        fig, axes = plt.subplots(
            n_kernels,
            n_cols,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
        )
        if n_kernels == 1:
            axes = axes.reshape(1, -1)

        for row, kinfo in enumerate(unique_kernels):
            ki = kinfo["kernel_index"]
            dil = kinfo["dilation"]
            rep = kinfo["representation"]
            bias = kinfo["bias"]
            imp = kinfo["importance"]
            pooling = kinfo["pooling_op"]

            # --- Column 0: Kernel weight pattern ---
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
                f"Kernel {ki} (d={dil}, {rep})\nimp={imp:.4f} [{pooling}]", fontsize=9
            )
            ax.set_xlabel("Dilated position")
            ax.axhline(0, color="#7f7f7f", linewidth=0.5)
            ax.set_ylabel("Weight")

            # --- Compute per-class activation rates ---
            class_act_rates = []

            for cls_idx, cls in enumerate(classes):
                ax = axes[row, 1 + cls_idx]
                mask = y == cls
                X_cls = X[mask]

                if rep == "diff":
                    X_use = np.diff(X_cls, axis=1).astype(np.float32)
                else:
                    X_use = X_cls

                n_use = len(X_use)
                n_plot = min(n_examples, n_use)

                # Compute per-timepoint activation rate across ALL examples
                act_count = np.zeros(n_timepoints, dtype=np.float64)
                total_count = np.zeros(n_timepoints, dtype=np.float64)

                for ex_idx in range(n_use):
                    x = X_use[ex_idx]
                    conv_out, act, time_idx = compute_activation_map(
                        x, ki, np.int32(dil), np.float32(bias)
                    )

                    for t in range(len(act)):
                        center = int(round(time_idx[t]))
                        if rep == "diff":
                            center = min(center + 1, n_timepoints - 1)
                        if 0 <= center < n_timepoints:
                            total_count[center] += 1.0
                            if act[t] > 0:
                                act_count[center] += 1.0

                act_rate = np.zeros(n_timepoints, dtype=np.float64)
                valid = total_count > 0
                act_rate[valid] = act_count[valid] / total_count[valid]
                class_act_rates.append(act_rate)

                # Plot example signal traces
                for ex_idx in range(n_plot):
                    x = X_use[ex_idx]
                    t_signal = np.arange(len(x))
                    ax.plot(t_signal, x, alpha=0.3, linewidth=0.8, color="#7f7f7f")

                # Overlay this class's activation rate
                ax2 = ax.twinx()
                ax2.plot(
                    range(n_timepoints),
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

            # --- Optional last column: Differential activation ---
            if show_difference:
                ax_diff = axes[row, -1]
                class_act_array = np.array(class_act_rates)
                diff_rate = np.max(class_act_array, axis=0) - np.min(
                    class_act_array, axis=0
                )

                ax_diff.plot(
                    range(n_timepoints), diff_rate, color="#2c2c2c", linewidth=1.5
                )
                ax_diff.set_xlabel("Timepoint")
                if row == 0:
                    ax_diff.set_title(
                        "Differential\n(max - min across classes)", fontsize=9
                    )
                else:
                    ax_diff.set_title("Differential", fontsize=9)
                ax_diff.set_ylabel("Delta act. rate", fontsize=8)

                # Also overlay per-class lines lightly for reference
                for cls_idx, cls in enumerate(classes):
                    ax_diff.plot(
                        range(n_timepoints),
                        class_act_rates[cls_idx],
                        color=class_colors[cls_idx],
                        linewidth=1.5,
                        alpha=0.3,
                        label=f"Class {cls}" if row == 0 else None,
                    )
                if row == 0:
                    ax_diff.legend(fontsize=7, loc="upper right")

        plt.tight_layout()
        return fig


    def plot_temporal_importance(
        self,
        X,
        y,
        n_top=None,
        n_examples=20,
        method="differential",
        figsize=(14, 6),
        feature_mask=None,
    ):
        """
        Aggregate activation maps of top features to show which time regions
        are most important for classification.

        Two methods are available:

        'center' — attributes importance to the convolution center point only.
            Fast, and produces the most spatially concentrated peaks. However,
            a dilated kernel's output at position t actually depends on
            timepoints spread across its receptive field (1 + 8d timepoints
            for dilation d), so the apparent precision overstates what the
            kernel resolves. Useful for initial exploration; interpret peak
            locations as approximate.

        'differential' (default) — for each top feature (kernel+dilation+bias),
            computes the mean activation rate per timepoint for each class, then
            weights by the absolute difference in activation rates across classes.
            This isolates *where classes differ* rather than where kernels fire
            in general. Importance is attributed to the convolution center point.
            Best for identifying discriminative trial epochs.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like
        n_top : int or None
            Number of top features to aggregate. If None, defaults to
            len(feature_mask) when a mask is provided, or 50 otherwise.
        n_examples : int
            Number of examples per class to average over (default 20).
        method : str, 'center' or 'differential'
        figsize : tuple
        feature_mask : array-like of int, optional
            If provided, only these feature indices are eligible for ranking.

        Returns
        -------
        fig : matplotlib Figure
        importance_by_time : ndarray, shape (n_timepoints,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        n_timepoints = X.shape[1]

        # Resolve n_top: default to all survivors if mask provided
        if n_top is None:
            if feature_mask is not None:
                n_top = len(feature_mask)
            else:
                n_top = 50

        top_features = self.get_top_features(n=n_top, feature_mask=feature_mask)

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        if method == "differential":
            # ---------------------------------------------------------
            # DIFFERENTIAL METHOD
            # For each top feature (kernel+dilation+bias), compute the
            # mean activation map per class, then weight by the max
            # absolute difference across classes at each timepoint.
            # This highlights WHERE classes differ, not just where
            # kernels fire.
            # ---------------------------------------------------------
            overall_importance = np.zeros(n_timepoints, dtype=np.float64)

            # Collect per-class activation maps for all top features
            # Shape per feature: (n_classes, n_timepoints)
            class_importances = [
                np.zeros(n_timepoints, dtype=np.float64) for _ in classes
            ]

            for finfo in top_features:
                ki = finfo["kernel_index"]
                dil = finfo["dilation"]
                rep = finfo["representation"]
                bias = finfo["bias"]
                imp = finfo["importance"]

                # Compute mean activation map for each class
                class_act_maps = []

                for cls in classes:
                    mask = y == cls
                    X_cls = X[mask]
                    n_ex = min(n_examples, len(X_cls))

                    if rep == "diff":
                        X_use = np.diff(X_cls[:n_ex], axis=1).astype(np.float32)
                    else:
                        X_use = X_cls[:n_ex]

                    # Accumulate activation at center points
                    act_map = np.zeros(n_timepoints, dtype=np.float64)
                    count_map = np.zeros(n_timepoints, dtype=np.float64)

                    for ex_idx in range(len(X_use)):
                        x = X_use[ex_idx]
                        conv_out, act, time_idx = compute_activation_map(
                            x, ki, np.int32(dil), np.float32(bias)
                        )

                        for t in range(len(act)):
                            center = int(round(time_idx[t]))
                            # For diff representation, shift by +0.5 to map
                            # between the two original timepoints
                            if rep == "diff":
                                center = min(center + 1, n_timepoints - 1)
                            if 0 <= center < n_timepoints:
                                act_map[center] += act[t]
                                count_map[center] += 1.0

                    # Mean activation rate at each timepoint
                    valid = count_map > 0
                    act_map[valid] /= count_map[valid]
                    class_act_maps.append(act_map)

                # Differential: max pairwise difference at each timepoint
                class_act_array = np.array(class_act_maps)  # (n_classes, n_timepoints)
                diff_map = np.max(class_act_array, axis=0) - np.min(
                    class_act_array, axis=0
                )

                # Weight by classifier importance and add to per-class maps
                weighted_diff = diff_map * imp
                overall_importance += weighted_diff

                # Also add per-class contribution (activation × importance × differential)
                for cls_idx in range(len(classes)):
                    class_importances[cls_idx] += (
                        class_act_maps[cls_idx] * imp * diff_map
                    )

            # Plot per-class
            for cls_idx, cls in enumerate(classes):
                ci = class_importances[cls_idx]
                if ci.max() > 0:
                    ci = ci / ci.max()
                axes[0].plot(range(n_timepoints), ci, linewidth=1.2)
                axes[0].set_ylabel("Importance per class", fontsize=10)
                axes[0].set_ylim(0, 1.1)

        else:
            # ---------------------------------------------------------
            # CENTER METHOD: when a kernel fires,
            # attribute importance to the center of its convolution
            # position only (no receptive field smearing).
            # ---------------------------------------------------------
            overall_importance = np.zeros(n_timepoints, dtype=np.float64)
            class_importances = [
                np.zeros(n_timepoints, dtype=np.float64) for _ in classes
            ]

            for cls_idx, cls in enumerate(classes):
                mask = y == cls
                X_cls = X[mask]
                n_ex = min(n_examples, len(X_cls))

                for finfo in top_features:
                    ki = finfo["kernel_index"]
                    dil = finfo["dilation"]
                    rep = finfo["representation"]
                    bias = finfo["bias"]
                    imp = finfo["importance"]

                    if rep == "diff":
                        X_use = np.diff(X_cls[:n_ex], axis=1).astype(np.float32)
                    else:
                        X_use = X_cls[:n_ex]

                    for ex_idx in range(len(X_use)):
                        x = X_use[ex_idx]
                        conv_out, act, time_idx = compute_activation_map(
                            x, ki, np.int32(dil), np.float32(bias)
                        )

                        for t in range(len(act)):
                            if act[t] > 0:
                                center = int(round(time_idx[t]))
                                if rep == "diff":
                                    center = min(center + 1, n_timepoints - 1)
                                if 0 <= center < n_timepoints:
                                    class_importances[cls_idx][center] += imp

                ci = class_importances[cls_idx]
                if ci.max() > 0:
                    ci = ci / ci.max()
                axes[0].plot(range(n_timepoints), ci, linewidth=1.2)
                axes[0].set_ylabel("Importance per class", fontsize=10)
                axes[0].set_ylim(0, 1.1)

            overall_importance = sum(class_importances)

        # Overall panel
        if overall_importance.max() > 0:
            overall_importance /= overall_importance.max()
        axes[-1].plot(
            range(n_timepoints), overall_importance, linewidth=1.2, color="#2c2c2c"
        )
        axes[-1].set_ylabel("Overall importance", fontsize=10)
        axes[-1].set_xlabel("Timepoint")
        axes[-1].set_ylim(0, 1.1)

        method_label = "differential" if method == "differential" else "center-point"
        fig.suptitle(
            f"Temporal Importance Map — {method_label} attribution", fontsize=12, y=1.01
        )
        plt.tight_layout()

        return fig, overall_importance

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
            Typically the surviving indices from RFE.

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
                f"K{finfo['kernel_index']} d={finfo['dilation']} "
                f"{finfo['pooling_op']} ({finfo['representation']})\n"
                f"imp={finfo['importance']:.3f}",
                fontsize=8,
            )
            ax.legend(fontsize=7)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Class-Conditional Feature Distributions", fontsize=12, y=1.01)
        plt.tight_layout()
        return fig

    def plot_kernel_properties(self, n_top=100, figsize=(14, 8)):
        """
        Compare properties (dilation, receptive field, pooling op distribution)
        of top vs. bottom features.

        Answers: "what temporal scales and summary statistics drive classification?"

        Parameters
        ----------
        n_top : int
            Number of top and bottom features to compare.
        figsize : tuple

        Returns
        -------
        fig : matplotlib Figure
        """
        importance = self.get_feature_importance()
        n_features = len(importance)
        sorted_idx = np.argsort(importance)[::-1]

        top_idx = sorted_idx[:n_top]
        bottom_idx = sorted_idx[-n_top:]

        # Decode all
        top_props = [self.decode_feature_index(int(i)) for i in top_idx]
        bottom_props = [self.decode_feature_index(int(i)) for i in bottom_idx]

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # --- Dilation distribution ---
        ax = axes[0, 0]
        top_dilations = [p["dilation"] for p in top_props]
        bottom_dilations = [p["dilation"] for p in bottom_props]
        bins = np.unique(top_dilations + bottom_dilations)
        if len(bins) > 1:
            ax.hist(top_dilations, bins=20, alpha=0.6, label="Top", density=True)
            ax.hist(bottom_dilations, bins=20, alpha=0.6, label="Bottom", density=True)
        else:
            ax.bar(
                [0, 1],
                [len(top_dilations), len(bottom_dilations)],
                tick_label=["Top", "Bottom"],
            )
        ax.set_title("Dilation Distribution")
        ax.set_xlabel("Dilation")
        ax.legend()

        # --- Receptive field distribution ---
        ax = axes[0, 1]
        top_rf = [p["receptive_field"] for p in top_props]
        bottom_rf = [p["receptive_field"] for p in bottom_props]
        ax.hist(top_rf, bins=20, alpha=0.6, label="Top", density=True)
        ax.hist(bottom_rf, bins=20, alpha=0.6, label="Bottom", density=True)
        ax.set_title("Receptive Field Distribution")
        ax.set_xlabel("Receptive field (timepoints)")
        ax.legend()

        # --- Pooling operator distribution ---
        ax = axes[0, 2]
        top_pools = [p["pooling_op"] for p in top_props]
        bottom_pools = [p["pooling_op"] for p in bottom_props]
        pool_names = self.POOLING_NAMES
        expanded_pool_names = [
            "PPV\n(Proportion)","MPV\n(Amplitude)",
            "MIPV\n(Timing)","LSPV\n(Persistence)"
        ]
        top_counts = [top_pools.count(p) for p in pool_names]
        bottom_counts = [bottom_pools.count(p) for p in pool_names]
        x_pos = np.arange(len(pool_names))
        ax.bar(x_pos - 0.2, top_counts, 0.4, label="Top", alpha=0.7)
        ax.bar(x_pos + 0.2, bottom_counts, 0.4, label="Bottom", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(expanded_pool_names)
        ax.set_title("Pooling Operator Distribution")
        ax.legend()

        # --- Representation distribution ---
        ax = axes[1, 0]
        top_reps = [p["representation"] for p in top_props]
        bottom_reps = [p["representation"] for p in bottom_props]
        for label, reps in [("Top", top_reps), ("Bottom", bottom_reps)]:
            raw_count = reps.count("raw")
            diff_count = reps.count("diff")
            ax.bar(
                label,
                raw_count,
                label="Raw" if label == "Top" else "",
                color="#1f77b4",
                alpha=0.7,
            )
            ax.bar(
                label,
                diff_count,
                bottom=raw_count,
                label="Diff" if label == "Top" else "",
                color="#1f77b4",
                alpha=0.7,
            )
        ax.set_title("Representation (Raw vs Diff)")
        ax.legend()

        # --- Kernel index distribution ---
        ax = axes[1, 1]
        top_ki = [p["kernel_index"] for p in top_props]
        bottom_ki = [p["kernel_index"] for p in bottom_props]
        ax.hist(top_ki, bins=range(0, 85, 4), alpha=0.6, label="Top", density=True)
        ax.hist(
            bottom_ki, bins=range(0, 85, 4), alpha=0.6, label="Bottom", density=True
        )
        ax.set_title("Base Kernel Index Distribution")
        ax.set_xlabel("Kernel index (0-83)")
        ax.legend()

        # --- Importance histogram ---
        ax = axes[1, 2]
        ax.hist(importance, bins=50, alpha=0.7, color="#7f7f7f")
        threshold = importance[sorted_idx[n_top - 1]]
        ax.axvline(
            threshold, color="#d62728", linestyle="--", label=f"Top-{n_top} threshold"
        )
        ax.set_title("Feature Importance Distribution")
        ax.set_xlabel("Importance (mean |coef|)")
        ax.set_yscale("log")
        ax.legend()

        fig.suptitle(
            "Kernel Property Analysis: Top vs Bottom Features", fontsize=13, y=1.01
        )
        plt.tight_layout()
        return fig

    def summary(self):
        """Print a summary of the fitted model."""
        if self.classifier_ is None:
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
                f"    {i+1}. K{f['kernel_index']:02d} d={f['dilation']:>4d} "
                f"{f['pooling_op']:>4s} ({f['representation']:>4s}) "
                f"imp={f['importance']:.4f} "
                f"RF={f['receptive_field']} bias={f['bias']:.3f}"
            )
        print("=" * 60)

# ============================================================================
# SECTION 11: RECURSIVE FEATURE ELIMINATION
# ============================================================================

def kneedle(y):
    """
    Find the knee point in a curve using the Kneedle algorithm.

    For the RFE accuracy curve (which generally increases as more features
    are retained, reading right to left), the knee is the point where
    accuracy gains flatten — i.e., adding more features yields diminishing
    returns. The algorithm normalizes the curve to [0,1] x [0,1] and finds
    the index of maximum difference from the diagonal connecting the
    endpoints.

    Parameters
    ----------
    y : array-like
        Sequence of values (e.g., test accuracies from RFE steps).
        The curve should be roughly monotone in the region of interest.

    Returns
    -------
    knee_idx : int
        Index of the knee point.

    References
    ----------
    Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011).
    Finding a" kneedle" in a haystack: Detecting knee points in
    system behavior. In 2011 31st international conference on
    distributed computing systems workshops (pp. 166-171). IEEE.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return 0
    x = np.arange(n, dtype=float)

    # Normalize to [0, 1]
    x_norm = (x - x[0]) / (x[-1] - x[0] + 1e-15)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-15)

    # Difference from the line connecting first and last points
    line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    diff = y_norm - line

    # The knee is where the curve deviates most from the straight line
    knee_idx = int(np.argmax(np.abs(diff)))
    return knee_idx


def recursive_feature_elimination(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    drop_percentage=0.05,
    total_number_steps=150,
    alpha_range=None,
    knee_method="kneedle",
    verbose=True,
):
    """
    Recursive Feature Elimination for InterpRocket.

    Starting from the full feature set, iteratively removes the least
    important features (by Ridge coefficient magnitude) and retrains the
    scaler + classifier on the reduced set. Features are re-ranked at each
    step so that importance reflects the current redundancy structure rather
    than the initial ranking (Guyon et al., 2002; Uribarri et al., 2024).

    The transform itself (convolutions, biases, pooling) is NOT recomputed —
    only the downstream column selection, scaling, and classifier change at
    each step.

    The step schedule follows Detach-ROCKET (Uribarri et al., 2024):
    at each step, a fraction (drop_percentage) of the CURRENT features are
    removed. This produces an exponential decay in feature count, with
    finer resolution at small feature counts where the elimination curve
    is most informative.

    Feature importance uses the L2 norm of Ridge coefficients across classes,
    matching Detach-ROCKET's default ('norm') method.

    Parameters
    ----------
    model : InterpRocket
        A fitted model. Must have been fit() already.
    X_train, y_train : arrays
        Training data used for refitting the classifier at each step.
    X_test, y_test : arrays
        Held-out test data for evaluation at each step.
    drop_percentage : float, default=0.05
        Fraction of features to drop at each step (0.05 = 5%).
        Matches the default in Detach-ROCKET.
    total_number_steps : int, default=150
        Maximum number of elimination steps. The actual number of steps
        is determined by the number of unique feature counts produced
        by the exponential schedule.
    alpha_range : array, optional
        Ridge regularization range. Defaults to model's alpha_range.
    knee_method : str, default='threshold'
        Method for detecting the knee in the RFE accuracy curve.
        'threshold' — Original method: smallest feature set within 1%
            of peak test accuracy (walks backward from most aggressive
            reduction).
        'kneedle' — Kneedle algorithm (Satopaa et al., 2011): finds the
            point of maximum curvature in the test accuracy curve. More
            principled than an arbitrary threshold, but may be sensitive
            to noisy accuracy curves.
        'both' — Compute both methods. The 'threshold' result is stored
            in the standard knee_* keys; the Kneedle result is stored
            in 'kneedle_idx', 'kneedle_fraction', 'kneedle_n_features',
            and 'kneedle_accuracy'.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict with keys:
        'fractions' : list of float — fraction of features retained
        'n_features' : list of int — number of features at each step
        'train_accuracies' : list of float
        'test_accuracies' : list of float
        'surviving_indices' : list of ndarray — feature indices at each step
        'full_feature_ranking' : ndarray — all feature indices ranked by
            importance from the full model
        'knee_idx', 'knee_fraction', 'knee_n_features', 'knee_accuracy',
        'peak_accuracy', 'peak_idx'
    """
    if model.classifier_ is None:
        raise ValueError("Model must be fitted before RFE.")

    if alpha_range is None:
        alpha_range = model.alpha_range

    # Transform once — this is the expensive step
    if verbose:
        print("Transforming training data...")
    X_train_features = model._transform(np.asarray(X_train, dtype=np.float32))
    if verbose:
        print("Transforming test data...")
    X_test_features = model._transform(np.asarray(X_test, dtype=np.float32))

    n_total = X_train_features.shape[1]

    # --- Build the exponential step schedule (matching Detach-ROCKET) ---
    keep_percentage = 1.0 - drop_percentage
    powers_vector = np.arange(total_number_steps)
    percentage_vector_unif = np.power(keep_percentage, powers_vector)
    num_feat_per_step = np.unique(
        (percentage_vector_unif * n_total).astype(int)
    )
    num_feat_per_step = num_feat_per_step[::-1]           # descending
    num_feat_per_step = num_feat_per_step[num_feat_per_step > 0]

    n_steps = len(num_feat_per_step)

    # --- Initial feature importance from the fitted model (L2 norm) ---
    importance = model.get_feature_importance()
    feature_importance = importance.copy()

    if verbose:
        print(f"\nRecursive Feature Elimination: {n_total} total features")
        print(f"  Drop rate: {drop_percentage:.0%} per step, {n_steps} steps")
        print(
            f"  {'Step':>6s}  {'Fraction':>10s}  {'N features':>10s}  "
            f"{'Train acc':>10s}  {'Test acc':>10s}"
        )
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    fractions = []
    n_features_list = []
    train_accs = []
    test_accs = []
    surviving_indices = []

    for count, feat_num in enumerate(num_feat_per_step):
        frac = feat_num / n_total

        # Select top features by current importance ranking
        drop_features = n_total - feat_num
        selected_idxs = np.argsort(feature_importance)[drop_features:]
        selection_mask = np.full(n_total, False)
        selection_mask[selected_idxs] = True

        # Subset features
        X_tr = X_train_features[:, selection_mask]
        X_te = X_test_features[:, selection_mask]

        # Refit scaler + classifier on the reduced set
        scaler = StandardScaler(with_mean=True)
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        clf = RidgeClassifierCV(alphas=alpha_range)
        clf.fit(X_tr_scaled, np.asarray(y_train))

        train_acc = clf.score(X_tr_scaled, np.asarray(y_train))
        test_acc = clf.score(X_te_scaled, np.asarray(y_test))

        fractions.append(frac)
        n_features_list.append(feat_num)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        # Store survivors sorted by importance (descending)
        surv_idx = np.where(selection_mask)[0]
        surv_importance = feature_importance[surv_idx]
        sorted_order = np.argsort(surv_importance)[::-1]
        surviving_indices.append(surv_idx[sorted_order].copy())

        if verbose:
            print(
                f"  {count+1:>6d}  {frac:>10.4f}  {feat_num:>10d}  "
                f"{train_acc:>10.4f}  {test_acc:>10.4f}"
            )

        # Zero out dropped features, then re-rank survivors using
        # the reduced classifier's coefficients. This is the key RFE
        # insight (Guyon et al., 2002): importance changes as redundant
        # features are removed.
        feature_importance[~selection_mask] = 0

        coefs = clf.coef_
        if coefs.ndim > 1:
            new_importance = np.linalg.norm(coefs, axis=0, ord=2)
        else:
            new_importance = np.abs(coefs)
        feature_importance[selection_mask] = new_importance

    # ------------------------------------------------------------------
    # Knee detection
    # ------------------------------------------------------------------
    max_acc = max(test_accs)
    peak_idx = test_accs.index(max_acc)

    # Method 1: Threshold (within 1% of peak)
    threshold_knee_idx = 0
    for i in range(len(test_accs) - 1, -1, -1):
        if test_accs[i] >= max_acc - 0.01:
            threshold_knee_idx = i
            break

    # Method 2: Kneedle (maximum curvature)
    # Apply to test accuracy curve, which generally rises as we move
    # from aggressive elimination (right/end) toward full model (left/start).
    # Reverse it so kneedle sees a decreasing curve, then map back.
    accs_array = np.array(test_accs)
    kneedle_knee_idx = kneedle(accs_array)

    # Select primary knee based on method
    if knee_method == "kneedle":
        knee_idx = kneedle_knee_idx
    else:
        knee_idx = threshold_knee_idx

    if verbose:
        print(
            f"\n  Peak test accuracy: {max_acc:.4f} "
            f"at {fractions[peak_idx]:.1%} retention"
        )
        if knee_method in ("threshold", "both"):
            print(
                f"  Knee (threshold, within 1% of peak): "
                f"{fractions[threshold_knee_idx]:.1%} retention "
                f"({n_features_list[threshold_knee_idx]} features, "
                f"acc={test_accs[threshold_knee_idx]:.4f})"
            )
        if knee_method in ("kneedle", "both"):
            print(
                f"  Knee (Kneedle): "
                f"{fractions[kneedle_knee_idx]:.1%} retention "
                f"({n_features_list[kneedle_knee_idx]} features, "
                f"acc={test_accs[kneedle_knee_idx]:.4f})"
            )

    results = {
        "fractions": fractions,
        "n_features": n_features_list,
        "train_accuracies": train_accs,
        "test_accuracies": test_accs,
        "surviving_indices": surviving_indices,
        "full_feature_ranking": surviving_indices[0].copy(),
        "knee_idx": knee_idx,
        "knee_fraction": fractions[knee_idx],
        "knee_n_features": n_features_list[knee_idx],
        "knee_accuracy": test_accs[knee_idx],
        "peak_accuracy": max_acc,
        "peak_idx": peak_idx,
        "knee_method": knee_method,
    }

    # When 'both', also store the alternative
    if knee_method == "both":
        results["kneedle_idx"] = kneedle_knee_idx
        results["kneedle_fraction"] = fractions[kneedle_knee_idx]
        results["kneedle_n_features"] = n_features_list[kneedle_knee_idx]
        results["kneedle_accuracy"] = test_accs[kneedle_knee_idx]
        results["threshold_idx"] = threshold_knee_idx
        results["threshold_fraction"] = fractions[threshold_knee_idx]
        results["threshold_n_features"] = n_features_list[threshold_knee_idx]
        results["threshold_accuracy"] = test_accs[threshold_knee_idx]

    return results

def plot_elimination_curve(rfe_results, start_fraction=0.5, figsize=(10, 5)):
    """
    Plot the Recursive Feature Elimination curve.

    Parameters
    ----------
    rfe_results : dict
        Output from recursive_feature_elimination().
    start_fraction : float, default=0.5
        Show only steps where the fraction retained is at or below this
        value. Set to 1.0 to show the full curve.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    fracs = rfe_results["fractions"]
    train_acc = rfe_results["train_accuracies"]
    test_acc = rfe_results["test_accuracies"]
    n_feats = rfe_results["n_features"]

    # Filter to steps at or below start_fraction
    start_idx = 0
    for i, f in enumerate(fracs):
        if f <= start_fraction:
            start_idx = i
            break
    fracs = fracs[start_idx:]
    train_acc = train_acc[start_idx:]
    test_acc = test_acc[start_idx:]
    n_feats = n_feats[start_idx:]

    # Adjust knee index relative to the trimmed lists
    knee_idx = rfe_results["knee_idx"] - start_idx

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot accuracies
    ax1.plot(
        fracs,
        train_acc,
        "o-",
        color="#1f77b4",
        linewidth=1.5,
        markersize=5,
        label="Train accuracy",
    )
    ax1.plot(
        fracs,
        test_acc,
        "s-",
        color="#ff7f0e",
        linewidth=1.5,
        markersize=5,
        label="Test accuracy",
    )

    ax1.set_xlabel("Fraction of features retained")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(max(fracs) + 0.02, min(fracs) - 0.005)  # reversed x-axis
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    # Mark the knee point(s) — adjust index for trimmed lists
    knee_idx = rfe_results["knee_idx"] - start_idx
    if 0 <= knee_idx < len(fracs):
        method = rfe_results.get("knee_method", "threshold")
        label = "Knee" if method != "both" else "Knee (threshold)"
        ax1.axvline(fracs[knee_idx], color="#7f7f7f", linestyle="--", alpha=0.5)

        # Place annotation near the bottom of the plot, left of the knee
        y_min, y_max = ax1.get_ylim()
        ax1.annotate(
            f"{label}: {fracs[knee_idx]:.1%}\n({n_feats[knee_idx]} features)\n"
            f"acc={test_acc[knee_idx]:.3f}",
            xy=(fracs[knee_idx], test_acc[knee_idx]),
            xytext=(fracs[knee_idx] + 0.08, y_min + 0.02 * (y_max - y_min)),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#7f7f7f"),
            va='bottom',
        )

    # If 'both' method was used, also show the Kneedle knee
    if "kneedle_idx" in rfe_results:
        kn_idx = rfe_results["kneedle_idx"] - start_idx
        if 0 <= kn_idx < len(fracs) and kn_idx != knee_idx:
            ax1.axvline(fracs[kn_idx], color="#2ca02c",
                        linestyle="-.", alpha=0.5)
            y_min, y_max = ax1.get_ylim()
            ax1.annotate(
                f"Kneedle: {fracs[kn_idx]:.1%}\n"
                f"({n_feats[kn_idx]} features)\n"
                f"acc={test_acc[kn_idx]:.3f}",
                xy=(fracs[kn_idx], test_acc[kn_idx]),
                xytext=(fracs[kn_idx] - 0.08,
                        y_min + 0.12 * (y_max - y_min)),
                fontsize=9,
                color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c"),
                va='bottom',
            )

    # Secondary x-axis with absolute feature counts
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    candidate_ticks = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    tick_fracs = [f for f in candidate_ticks if min(fracs) <= f <= max(fracs)]
    if not tick_fracs:
        tick_fracs = fracs[:: max(1, len(fracs) // 6)]
    tick_labels = []
    for f in tick_fracs:
        # Find the closest fraction in the data
        closest_idx = min(range(len(fracs)), key=lambda i: abs(fracs[i] - f))
        tick_labels.append(str(n_feats[closest_idx]))
    ax2.set_xticks(tick_fracs)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel("Number of features", fontsize=9)

    #fig.suptitle("Recursive Feature Elimination", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig

# ============================================================================
# SECTION 12: REPEATED K-FOLD CROSS-VALIDATION
# ============================================================================
#
# Evaluates the full InterpRocket pipeline with repeated stratified k-fold CV.
# RepeatedStratifiedKFold preserves class proportions in each fold, which is
# essential for imbalanced data.

def cross_validate(
    X, y, n_repeats=10, n_folds=10, n_jobs=None, random_state=42,
    verbose=True, **model_kwargs
):
    """
    Evaluate InterpRocket with repeated stratified k-fold cross-validation.

    The full pipeline (class balancing → bias fitting → transform →
    standardize → classify) is refit on each training fold, so no
    information leaks across the fold boundary. RepeatedStratifiedKFold
    preserves class proportions in each fold.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints)
        Time series data.
    y : array-like, shape (n_instances,)
        Class labels.
    n_repeats : int, default=10
        Number of repetitions (each with a fresh shuffle).
    n_folds : int, default=10
        Number of folds per repetition.
    n_jobs : int or None, default=None
        Number of CPU cores for numba's parallel transform.
        None — use all available cores (numba default).
        -1   — all cores.
        -2   — all cores minus one (recommended for interactive use).
        -3   — all cores minus two.
        Positive int — use exactly that many cores.
    random_state : int, default=42
        Seed for the fold assignments in RepeatedStratifiedKFold.
    verbose : bool, default=True
        Print progress and per-repeat summaries.
    **model_kwargs
        Passed to InterpRocket (e.g., max_dilations_per_kernel,
        num_features, random_state, class_weight).

    Returns
    -------
    results : dict with keys:
        'accuracy' : dict with 'mean', 'std', 'values' (per-fold)
        'balanced_accuracy' : dict with 'mean', 'std', 'values'
        'f1_macro' : dict with 'mean', 'std', 'values'
        'f1_weighted' : dict with 'mean', 'std', 'values'
        'mcc' : dict with 'mean', 'std', 'values'
        'mutual_info' : dict with 'mean', 'std', 'values'
        'per_repeat_means' : ndarray, shape (n_repeats,) — accuracy
        'confusion_matrices' : list of ndarray
        'confusion_matrix_total' : ndarray
        'n_repeats' : int
        'n_folds' : int
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    import os, io, contextlib

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes = np.unique(y)

    # --- Limit numba thread count ---
    if n_jobs is not None:
        from numba import set_num_threads

        total_cores = os.cpu_count()
        if n_jobs < 0:
            target = max(1, total_cores + n_jobs + 1)
        else:
            target = max(1, min(n_jobs, total_cores))
        set_num_threads(target)
        if verbose:
            print(f"  Numba threads: {target} of {total_cores} available")

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)

    # Accumulators for all metrics
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "mcc",
        "mutual_info",
    ]
    all_metrics = {m: [] for m in metric_names}
    conf_matrices = []
    total_folds = n_repeats * n_folds

    if verbose:
        print(
            f"Cross-validation: {n_repeats} repeats × {n_folds} folds "
            f"= {total_folds} evaluations"
        )
        print(
            f"  Data: {X.shape[0]} instances × {X.shape[1]} timepoints, "
            f"{len(classes)} classes"
        )
        class_counts = {c: int(np.sum(y == c)) for c in classes}
        print(f"  Class counts: {class_counts}")

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = InterpRocket(**model_kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train, y_train)

        fold_metrics = model.evaluate(X_test, y_test)
        for m in metric_names:
            all_metrics[m].append(fold_metrics[m])

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        conf_matrices.append(cm)

        if verbose and (fold_idx + 1) % n_folds == 0:
            repeat_num = (fold_idx + 1) // n_folds
            repeat_accs = all_metrics["accuracy"][-n_folds:]
            repeat_bal = all_metrics["balanced_accuracy"][-n_folds:]
            print(
                f"  Repeat {repeat_num:2d}/{n_repeats}: "
                f"acc = {np.mean(repeat_accs):.4f}  "
                f"bal_acc = {np.mean(repeat_bal):.4f}  "
                f"mcc = {np.mean(all_metrics['mcc'][-n_folds:]):.4f}"
            )

    # Compile results
    results = {}
    for m in metric_names:
        vals = np.array(all_metrics[m])
        results[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "values": vals,
        }

    acc_vals = results["accuracy"]["values"]
    per_repeat_means = np.array(
        [np.mean(acc_vals[i * n_folds : (i + 1) * n_folds]) for i in range(n_repeats)]
    )
    total_cm = sum(conf_matrices)

    results["per_repeat_means"] = per_repeat_means
    results["confusion_matrices"] = conf_matrices
    results["confusion_matrix_total"] = total_cm
    results["n_repeats"] = n_repeats
    results["n_folds"] = n_folds

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Results (mean ± std across {total_folds} folds):")
        for m in metric_names:
            print(f"    {m:20s}: {results[m]['mean']:.4f} ± {results[m]['std']:.4f}")
        print(f"\n  Aggregated confusion matrix:")
        header = "       " + "  ".join(f"{c:>6}" for c in classes)
        print(header)
        for i, cls in enumerate(classes):
            row = "  ".join(f"{total_cm[i, j]:6d}" for j in range(len(classes)))
            print(f"  {cls:>4}  {row}")
        print(f"{'='*60}")

    return results

# ============================================================================
# SECTION 13: TEMPORAL OCCLUSION SENSITIVITY
# ============================================================================

def temporal_occlusion(
    model,
    X_test,
    y_test,
    n_samples=5,
    sample_indices=None,
    window_size=None,
    stride=None,
    feature_mask=None,
    class_names=None,
    verbose=True,
):
    """
    Model-agnostic temporal occlusion sensitivity analysis.

    For each sample, systematically zeros out a sliding window of the input
    time series and measures the change in the classifier's decision function.
    Regions where occlusion causes a large change are important for
    classification.

    The method's lineage traces to Zeiler & Fergus (2014) in computer vision.

    Parameters
    ----------
    model : InterpRocket
        A fitted model.
    X_test : ndarray, shape (n_instances, n_timepoints)
        Test data.
    y_test : array-like
        True labels.
    n_samples : int
        Number of samples to analyze (picks one per class, then fills).
    window_size : int or None
        Width of the occlusion window. Default: n_timepoints // 20.
    stride : int or None
        Step size for the sliding window. Default: window_size // 2.
    feature_mask : array-like of int, optional
        If provided, only these feature indices contribute to the decision
        function. All other features are zeroed out before scaling and
        classification. This allows occlusion analysis restricted to RFE
        survivors, e.g. feature_mask=rfe['surviving_indices'][rfe['knee_idx']].
    class_names : dict or list, optional
        Human-readable class labels.
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'sample_indices' : list of int
        'signals' : list of ndarray — original time series
        'sensitivities' : list of ndarray — per-timepoint occlusion sensitivity
        'true_labels' : list — true class labels
        'predicted_labels' : list — baseline predictions
        'window_size' : int
        'stride' : int
        'class_names' : dict
    """
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)
    classes = np.unique(y_test)
    n_timepoints = X_test.shape[1]

    # Build a boolean mask for feature selection if provided
    if feature_mask is not None:
        feature_mask = np.asarray(feature_mask, dtype=int)

    if class_names is None:
        class_names = {c: str(c) for c in classes}
    elif isinstance(class_names, (list, tuple)):
        class_names = {c: n for c, n in zip(classes, class_names)}

    if window_size is None:
        window_size = max(3, n_timepoints // 20)
    if stride is None:
        stride = max(1, window_size // 2)

    if sample_indices is not None:
        sample_indices = list(sample_indices)
    else:
        # Select samples: one per class, then pad
        sample_indices = []
        for cls in classes:
            cls_idx = np.where(y_test == cls)[0]
            if len(cls_idx) > 0:
                sample_indices.append(cls_idx[0])
        while len(sample_indices) < n_samples:
            remaining = [i for i in range(len(y_test)) if i not in sample_indices]
            if remaining:
                sample_indices.append(remaining[0])

    # Get the decision function, optionally restricted to feature_mask
    def _decision_function(X_single):
        """Get decision function values for a single instance."""
        features = model._transform(X_single)
        if feature_mask is not None:
            mask = np.zeros(features.shape[1], dtype=bool)
            mask[feature_mask[feature_mask < features.shape[1]]] = True
            features[:, ~mask] = 0.0
        features_scaled = model.scaler_.transform(features)
        return model.classifier_.decision_function(features_scaled)

    if verbose:
        print(f"Temporal occlusion: window={window_size}, stride={stride}")
        print(f"  Analyzing {len(sample_indices)} samples...")

    all_signals = []
    all_sensitivities = []
    all_true = []
    all_pred = []

    for s_idx in sample_indices:
        X_single = X_test[s_idx : s_idx + 1].copy()
        signal = X_single[0].copy()
        n = len(signal)

        # Baseline
        base_pred = model.predict(X_single)[0]
        base_decision = _decision_function(X_single)

        # Occlude and measure
        positions = list(range(0, n - window_size + 1, stride))
        sensitivity = np.zeros(n)
        counts = np.zeros(n)

        for pos in positions:
            X_occluded = X_single.copy()
            X_occluded[0, pos : pos + window_size] = 0.0

            try:
                occl_decision = _decision_function(X_occluded)
                impact = np.sum(np.abs(base_decision - occl_decision))
            except Exception:
                occl_pred = model.predict(X_occluded)[0]
                impact = float(occl_pred != base_pred)

            sensitivity[pos : pos + window_size] += impact
            counts[pos : pos + window_size] += 1

        counts[counts == 0] = 1
        sensitivity /= counts

        all_signals.append(signal)
        all_sensitivities.append(sensitivity)
        all_true.append(y_test[s_idx])
        all_pred.append(base_pred)

        if verbose:
            print(
                f"  Sample {s_idx}: true={class_names[y_test[s_idx]]}, "
                f"pred={class_names[base_pred]}, "
                f"max_sensitivity={sensitivity.max():.4f}"
            )

    return {
        "sample_indices": sample_indices,
        "signals": all_signals,
        "sensitivities": all_sensitivities,
        "true_labels": all_true,
        "predicted_labels": all_pred,
        "window_size": window_size,
        "stride": stride,
        "class_names": class_names,
    }

def plot_occlusion(occ_results, figsize=(12, None)):
    """
    Plot temporal occlusion sensitivity results.

    Each row shows one sample: the signal in gray with a red sensitivity
    curve overlaid on a twin axis.

    Parameters
    ----------
    occ_results : dict
        Output from temporal_occlusion().
    figsize : tuple
        Width is fixed; height scales with number of samples if None.

    Returns
    -------
    fig : matplotlib Figure
    """
    signals = occ_results["signals"]
    sensitivities = occ_results["sensitivities"]
    true_labels = occ_results["true_labels"]
    pred_labels = occ_results["predicted_labels"]
    sample_indices = occ_results["sample_indices"]
    class_names = occ_results["class_names"]
    window_size = occ_results["window_size"]
    stride = occ_results["stride"]

    n_samples = len(signals)
    if figsize[1] is None:
        figsize = (figsize[0], 2.5 * n_samples)

    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, squeeze=False)

    for row in range(n_samples):
        ax = axes[row, 0]
        signal = signals[row]
        sensitivity = sensitivities[row]
        t = np.arange(len(signal))
        true_lbl = class_names[true_labels[row]]
        pred_lbl = class_names[pred_labels[row]]

        # Signal
        ax.plot(t, signal, color="#7f7f7f", linewidth=0.8, alpha=0.7)
        pred_lbl = occ_results["predicted_labels"][row]
        ax.set_ylabel(f"Sample {sample_indices[row]}\ntrue={true_lbl}\npred={pred_lbl}", fontsize=9)

        # Sensitivity on twin axis
        ax2 = ax.twinx()
        ax2.plot(t, sensitivity, color="#ff7f0e", linewidth=1.2, alpha=0.85)
        ax2.set_ylabel("Sensitivity", fontsize=8, color="#ff7f0e")

        if row == 0:
            correct = "✓" if true_labels[row] == pred_labels[row] else "✗"
            ax.set_title(
                f"Temporal Occlusion Sensitivity "
                f"(window={window_size}, stride={stride})",
                fontsize=11,
            )
        if row == n_samples - 1:
            ax.set_xlabel("Timepoint")

    plt.tight_layout()
    return fig

# ============================================================================
# SECTION 14: RFE SURVIVING FEATURE ANALYSIS
# ============================================================================

def plot_rfe_survivors(rfe_results, model, step=-1, figsize=(14, 8)):
    """
    Analyze the properties of features that survive RFE to a given step.

    Produces a 2×3 panel figure comparing the properties of surviving
    features against the full feature population, using InterpRocket's
    exact feature decoding.

    Parameters
    ----------
    rfe_results : dict
        Output from recursive_feature_elimination().
    model : InterpRocket
        The fitted model (for decode_feature_index).
    step : int
        Which RFE step to analyze. Supports both positive indices
        (0 = full model, 1 = first reduction, ...) and negative indices
        (-1 = most aggressive, -2 = second-to-last, etc.).
        Common usage: step=rfe_results['knee_idx'] for the knee.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    from matplotlib.gridspec import GridSpec

    # Resolve step index
    n_steps = len(rfe_results["fractions"])
    frac_idx = step if step >= 0 else n_steps + step
    frac = rfe_results["fractions"][frac_idx]
    n_total = rfe_results["n_features"][0]

    # Decode surviving features at the requested step on demand
    survivor_indices = rfe_results["surviving_indices"][frac_idx]
    importance = model.get_feature_importance()
    decoded = []
    for feat_idx in survivor_indices:
        try:
            info = model.decode_feature_index(int(feat_idx))
            info["feature_index"] = int(feat_idx)
            info["importance"] = float(importance[feat_idx])
            decoded.append(info)
        except Exception:
            pass

    n_survivors = len(decoded)

    # Also decode ALL features for comparison (the "pruned" population)
    all_decoded = []
    for feat_idx in range(n_total):
        try:
            all_decoded.append(model.decode_feature_index(feat_idx))
        except Exception:
            pass

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"RFE Surviving Feature Properties — "
        f"{frac:.1%} retention ({n_survivors} of {n_total} features)",
        fontsize=13,
    )

    # --- Panel A: Importance distribution of survivors ---
    ax_a = fig.add_subplot(gs[0, 0])
    importances = [d["importance"] for d in decoded]
    ax_a.hist(
        importances,
        bins=min(30, max(5, n_survivors // 3)),
        color="#1f77b4",
        edgecolor="white",
        alpha=0.8,
    )
    ax_a.set_xlabel("|Classifier coefficient|")
    ax_a.set_ylabel("Count")
    ax_a.set_title("A. Importance Distribution\n(surviving features)")
    if importances:
        ax_a.axvline(
            np.median(importances),
            color="#d62728",
            linestyle="--",
            label=f"Median: {np.median(importances):.3f}",
        )
        ax_a.legend(fontsize=8)

    # --- Panel B: Signal type (raw vs diff) ---
    ax_b = fig.add_subplot(gs[0, 1])
    surv_raw = sum(1 for d in decoded if d["representation"] == "raw")
    surv_diff = sum(1 for d in decoded if d["representation"] == "diff")
    all_raw = sum(1 for d in all_decoded if d["representation"] == "raw")
    all_diff = sum(1 for d in all_decoded if d["representation"] == "diff")

    x = np.arange(2)
    width = 0.35
    surv_total = surv_raw + surv_diff
    all_total = all_raw + all_diff
    surv_props = (
        [surv_raw / surv_total, surv_diff / surv_total] if surv_total > 0 else [0, 0]
    )
    all_props = [all_raw / all_total, all_diff / all_total] if all_total > 0 else [0, 0]

    ax_b.bar(
        x - width / 2,
        surv_props,
        width,
        label="Surviving",
        color="#1f77b4",
        alpha=0.8,
    )
    ax_b.bar(
        x + width / 2, all_props, width, label="All features", color="#ff7f0e", alpha=0.8
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Raw signal", "1st Difference"])
    ax_b.set_ylabel("Proportion")
    ax_b.set_title("B. Signal Type")
    ax_b.legend(fontsize=8)

    # --- Panel C: Pooling type ---
    ax_c = fig.add_subplot(gs[0, 2])
    pooling_names = [
        "PPV\n(Proportion)",
        "MPV\n(Amplitude)",
        "MIPV\n(Timing)",
        "LSPV\n(Persistence)",
    ]
    surv_pooling = [0, 0, 0, 0]
    all_pooling = [0, 0, 0, 0]
    pool_map = {"PPV": 0, "MPV": 1, "MIPV": 2, "LSPV": 3}

    for d in decoded:
        surv_pooling[pool_map[d["pooling_op"]]] += 1
    for d in all_decoded:
        all_pooling[pool_map[d["pooling_op"]]] += 1

    surv_p_total = sum(surv_pooling)
    all_p_total = sum(all_pooling)
    surv_p = [c / surv_p_total for c in surv_pooling] if surv_p_total > 0 else [0] * 4
    all_p = [c / all_p_total for c in all_pooling] if all_p_total > 0 else [0] * 4

    x4 = np.arange(4)
    ax_c.bar(
        x4 - width / 2, surv_p, width, label="Surviving", color="#1f77b4", alpha=0.8
    )
    ax_c.bar(
        x4 + width / 2, all_p, width, label="All features", color="#ff7f0e", alpha=0.8
    )
    ax_c.set_xticks(x4)
    ax_c.set_xticklabels(pooling_names, fontsize=8)
    ax_c.set_ylabel("Proportion")
    ax_c.set_title("C. Pooling Type")
    ax_c.legend(fontsize=8)

    # --- Panel D: Dilation / receptive field distribution ---
    ax_d = fig.add_subplot(gs[1, 0])
    surv_rf = [d["receptive_field"] for d in decoded]
    all_rf = [d["receptive_field"] for d in all_decoded]

    if surv_rf:
        bins = np.linspace(0, max(max(surv_rf), max(all_rf)) * 1.1, 20)
        ax_d.hist(
            surv_rf,
            bins=bins,
            alpha=0.6,
            density=True,
            label="Surviving",
            color="#1f77b4",
        )
        ax_d.hist(
            all_rf,
            bins=bins,
            alpha=0.6,
            density=True,
            label="All features",
            color="#1f77b4",
        )
    ax_d.set_xlabel("Receptive field (timepoints)")
    ax_d.set_ylabel("Density")
    ax_d.set_title("D. Timescale Distribution")
    ax_d.legend(fontsize=8)

    # --- Panel E: Kernel index distribution ---
    ax_e = fig.add_subplot(gs[1, 1])
    surv_kernels = [d["kernel_index"] for d in decoded]
    all_kernels = [d["kernel_index"] for d in all_decoded]
    bins_k = np.arange(-0.5, 84.5, 1)

    if surv_kernels:
        # Show as histogram
        ax_e.hist(
            surv_kernels, bins=bins_k, alpha=0.7, color="#1f77b4", label="Surviving"
        )
    ax_e.set_xlabel("Kernel index (0–83)")
    ax_e.set_ylabel("Count")
    ax_e.set_title("E. Kernel Usage")
    ax_e.legend(fontsize=8)

    # --- Panel F: Summary ---
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    # Count unique kernels
    unique_kernels = len(set(surv_kernels)) if surv_kernels else 0
    unique_dilations = len(set(d["dilation"] for d in decoded)) if decoded else 0

    summary = (
        f"RFE Retention: {frac:.1%}\n"
        f"Surviving features: {n_survivors}\n"
        f"Unique kernels: {unique_kernels} / 84\n"
        f"Unique dilations: {unique_dilations}\n"
        f"Test accuracy: {rfe_results['test_accuracies'][frac_idx]:.4f}\n\n"
        f"Top 5 surviving features:\n"
    )
    for i, d in enumerate(sorted(decoded, key=lambda x: -x["importance"])[:5]):
        summary += (
            f"  K{d['kernel_index']:02d} d={d['dilation']:>3d} "
            f"{d['pooling_op']:>4s} ({d['representation']:>4s}) "
            f"imp={d['importance']:.4f}\n"
        )

    ax_f.text(
        0.05,
        0.95,
        summary,
        transform=ax_f.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#fffde7", alpha=0.8),
    )
    ax_f.set_title("F. Summary")

    fig.subplots_adjust(top=0.90)
    return fig

# ============================================================================
# SECTION 15: FEATURE INFORMATION DECOMPOSITION
# ============================================================================
#
# Partial information decomposition of feature contributions, adapted from
# methods used for neural ensemble analysis (Laubach et al., 2000; Narayanan
# & Laubach, 2009). Classifies each feature's information as redundant,
# synergistic, or independent relative to the full ensemble.
#
# The approach computes mutual information (MI) between class labels and
# predictions from: (1) the full feature set, (2) each individual feature,
# and (3) the ensemble minus each feature. The contribution and partial
# information terms reveal whether a feature carries unique information,
# duplicates information available elsewhere, or only contributes when
# combined with other features.

def information_decomposition(
    model,
    X_test,
    y_test,
    feature_mask=None,
    group_by="kernel",
    n_shuffles=100,
    alpha_range=None,
    random_state=42,
    verbose=True,
):
    """
    Decompose feature information into redundant, synergistic, and independent.

    For each feature (or feature group), computes:
        I_single  = MI from classifying with that feature alone
        I_ensemble = MI from classifying with all features
        I_contrib = I_ensemble - I_leave_one_out  (contribution of this feature)
        P_feature = I_contrib - I_single  (partial information)

    Features are classified as:
        Redundant:   P_feature < -shuffle_threshold
        Synergistic: P_feature > +shuffle_threshold
        Independent: |P_feature| <= shuffle_threshold

    Parameters
    ----------
    model : InterpRocket
        A fitted model.
    X_test : ndarray, shape (n_instances, n_timepoints)
        Test data.
    y_test : array-like
        True labels.
    feature_mask : ndarray, optional
        Indices of features to analyze. If None, uses all features.
        Typically set to RFE survivors: rfe['surviving_indices'][rfe['knee_idx']].
    group_by : str, default='kernel'
        How to group features for analysis.
        'kernel' — group by base kernel index (84 groups max). Each group
            includes all dilations and pooling operators for that kernel.
        'kernel_dilation' — group by (kernel, dilation) pair.
        'individual' — analyze each feature individually (slow).
    n_shuffles : int, default=100
        Number of label shuffles for null distribution.
    alpha_range : array, optional
        Ridge regularization range. Defaults to model's alpha_range.
    random_state : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'group_labels' : list of str — label for each group
        'group_indices' : list of ndarray — feature indices per group
        'I_ensemble' : float — MI from full feature set
        'I_single' : ndarray — MI from each group alone
        'I_leave_one_out' : ndarray — MI from ensemble minus each group
        'I_contrib' : ndarray — contribution of each group
        'P_feature' : ndarray — partial information of each group
        'I_shuffle' : float — mean MI under shuffled labels (null)
        'I_shuffle_std' : float — std of shuffled MI
        'classification' : ndarray of str — 'redundant', 'synergistic', 'independent'
        'n_redundant' : int
        'n_synergistic' : int
        'n_independent' : int
    """
    if alpha_range is None:
        alpha_range = model.alpha_range

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    # Get the full feature matrix
    features = model.transform(X_test)

    if feature_mask is not None:
        features = features[:, feature_mask]
        all_indices = np.arange(len(feature_mask))
    else:
        all_indices = np.arange(features.shape[1])
        feature_mask = all_indices

    n_features = features.shape[1]

    # Group features
    groups = _build_feature_groups(model, feature_mask, group_by)
    n_groups = len(groups['labels'])

    if verbose:
        print(f"Information decomposition: {n_features} features in "
              f"{n_groups} groups ({group_by})")

    # Scale features once
    scaler_full = StandardScaler()
    features_scaled = scaler_full.fit_transform(features)

    # Ensemble MI (all features)
    clf_full = RidgeClassifierCV(alphas=alpha_range)
    clf_full.fit(features_scaled, y_test)  # Note: using test as both fit and eval
    # To avoid data leakage, we refit on the same data the model was originally
    # evaluated on. This measures discriminability, not generalization.
    # For a proper estimate, use the training data for fitting:
    train_features = model.transform(
        np.asarray(model.X_train_ if hasattr(model, 'X_train_') else X_test,
                   dtype=np.float32)
    )
    if feature_mask is not None and len(feature_mask) < train_features.shape[1]:
        train_features = train_features[:, feature_mask]

    # Use the model's own classifier for the ensemble prediction
    y_pred_full = model.predict(X_test)
    I_ensemble = mutual_information(y_true=y_test, y_pred=y_pred_full)

    if verbose:
        print(f"  Ensemble MI: {I_ensemble:.4f} bits")

    # Single-feature MI and leave-one-out MI
    I_single = np.zeros(n_groups)
    I_leave_one_out = np.zeros(n_groups)

    rng = np.random.default_rng(random_state)

    for g_idx in range(n_groups):
        g_features = groups['indices'][g_idx]

        # Single group: classify using only these features
        single_feats = features_scaled[:, g_features]
        if single_feats.shape[1] > 0:
            clf_single = RidgeClassifierCV(alphas=alpha_range)
            clf_single.fit(single_feats, y_test)
            y_pred_single = clf_single.predict(single_feats)
            I_single[g_idx] = mutual_information(
                y_true=y_test, y_pred=y_pred_single
            )

        # Leave-one-out: classify using all features except this group
        loo_mask = np.ones(n_features, dtype=bool)
        loo_mask[g_features] = False
        loo_feats = features_scaled[:, loo_mask]
        if loo_feats.shape[1] > 0:
            clf_loo = RidgeClassifierCV(alphas=alpha_range)
            clf_loo.fit(loo_feats, y_test)
            y_pred_loo = clf_loo.predict(loo_feats)
            I_leave_one_out[g_idx] = mutual_information(
                y_true=y_test, y_pred=y_pred_loo
            )
        else:
            I_leave_one_out[g_idx] = 0.0

        if verbose and (g_idx + 1) % max(1, n_groups // 10) == 0:
            print(f"  Group {g_idx + 1}/{n_groups}: "
                  f"I_single={I_single[g_idx]:.4f}, "
                  f"I_loo={I_leave_one_out[g_idx]:.4f}")

    # Contribution and partial information
    I_contrib = I_ensemble - I_leave_one_out
    P_feature = I_contrib - I_single

    # Shuffle null distribution
    if verbose:
        print(f"  Computing null distribution ({n_shuffles} shuffles)...")

    shuffle_mi = np.zeros(n_shuffles)
    for s in range(n_shuffles):
        y_shuf = rng.permutation(y_test)
        clf_shuf = RidgeClassifierCV(alphas=alpha_range)
        clf_shuf.fit(features_scaled, y_shuf)
        y_pred_shuf = clf_shuf.predict(features_scaled)
        shuffle_mi[s] = mutual_information(y_true=y_test, y_pred=y_pred_shuf)

    I_shuffle_mean = shuffle_mi.mean()
    I_shuffle_std = shuffle_mi.std()

    # Classify features
    threshold = I_shuffle_mean + 2 * I_shuffle_std  # 2 SD above shuffle mean
    classification = np.array(['independent'] * n_groups)
    classification[P_feature < -threshold] = 'redundant'
    classification[P_feature > threshold] = 'synergistic'

    n_redundant = int(np.sum(classification == 'redundant'))
    n_synergistic = int(np.sum(classification == 'synergistic'))
    n_independent = int(np.sum(classification == 'independent'))

    if verbose:
        print(f"\n  Results ({n_groups} groups):")
        print(f"    Redundant:   {n_redundant} "
              f"({100 * n_redundant / n_groups:.1f}%)")
        print(f"    Synergistic: {n_synergistic} "
              f"({100 * n_synergistic / n_groups:.1f}%)")
        print(f"    Independent: {n_independent} "
              f"({100 * n_independent / n_groups:.1f}%)")
        print(f"    Shuffle MI:  {I_shuffle_mean:.4f} ± {I_shuffle_std:.4f}")
        print(f"    Ensemble MI: {I_ensemble:.4f}")
        sum_single = I_single.sum()
        P_ens = I_ensemble - sum_single
        print(f"    Sum(I_single): {sum_single:.4f}")
        print(f"    P_ensemble (I_ens - sum(I_single)): {P_ens:.4f}")

    return {
        'group_labels': groups['labels'],
        'group_indices': groups['indices'],
        'I_ensemble': I_ensemble,
        'I_single': I_single,
        'I_leave_one_out': I_leave_one_out,
        'I_contrib': I_contrib,
        'P_feature': P_feature,
        'I_shuffle_mean': I_shuffle_mean,
        'I_shuffle_std': I_shuffle_std,
        'classification': classification,
        'n_redundant': n_redundant,
        'n_synergistic': n_synergistic,
        'n_independent': n_independent,
    }


def _build_feature_groups(model, feature_mask, group_by):
    """Build feature groups for information decomposition."""
    labels = []
    indices = []

    if group_by == 'individual':
        for i, fi in enumerate(feature_mask):
            info = model.decode_feature_index(int(fi))
            labels.append(
                f"K{info['kernel_index']}_d{info['dilation']}_"
                f"{info['pooling_op']}_{info['representation']}"
            )
            indices.append(np.array([i]))

    elif group_by == 'kernel':
        kernel_groups = {}
        for i, fi in enumerate(feature_mask):
            info = model.decode_feature_index(int(fi))
            key = info['kernel_index']
            if key not in kernel_groups:
                kernel_groups[key] = []
            kernel_groups[key].append(i)
        for k_idx in sorted(kernel_groups.keys()):
            labels.append(f"Kernel {k_idx}")
            indices.append(np.array(kernel_groups[k_idx]))

    elif group_by == 'kernel_dilation':
        kd_groups = {}
        for i, fi in enumerate(feature_mask):
            info = model.decode_feature_index(int(fi))
            key = (info['kernel_index'], info['dilation'])
            if key not in kd_groups:
                kd_groups[key] = []
            kd_groups[key].append(i)
        for (k_idx, dil) in sorted(kd_groups.keys()):
            labels.append(f"K{k_idx}_d{dil}")
            indices.append(np.array(kd_groups[(k_idx, dil)]))

    else:
        raise ValueError(f"Unknown group_by: {group_by}")

    return {'labels': labels, 'indices': indices}


def plot_information_decomposition(info_results, figsize=(14, 6)):
    """
    Plot feature information decomposition.

    Left panel: scatter of single-feature MI vs partial information,
    colored by classification (redundant/synergistic/independent).
    Right panel: summary bar chart.

    Parameters
    ----------
    info_results : dict
        Output from information_decomposition().
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    I_single = info_results['I_single']
    P_feature = info_results['P_feature']
    classification = info_results['classification']
    I_shuffle_mean = info_results['I_shuffle_mean']
    I_shuffle_std = info_results['I_shuffle_std']
    threshold = I_shuffle_mean + 2 * I_shuffle_std

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: scatter plot ---
    marker_map = {
        'redundant': 'v',
        'synergistic': '^',
        'independent': 'o',
    }

    for cls_type in ['redundant', 'independent', 'synergistic']:
        mask = classification == cls_type
        if mask.any():
            ax1.scatter(
                I_single[mask], P_feature[mask],
                c=INFO_COLORS[cls_type],
                marker=marker_map[cls_type],
                s=40, alpha=0.7, edgecolors='white', linewidths=0.5,
                label=f'{cls_type.capitalize()} ({mask.sum()})',
            )

    # Threshold lines
    ax1.axhline(threshold, color='#7f7f7f', linestyle='--', alpha=0.5,
                linewidth=0.8, label=f'±threshold ({threshold:.4f})')
    ax1.axhline(-threshold, color='#7f7f7f', linestyle='--', alpha=0.5,
                linewidth=0.8)
    ax1.axhline(0, color='#2c2c2c', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='#2c2c2c', linewidth=0.5, alpha=0.3)

    ax1.set_xlabel('Single-feature MI (bits)')
    ax1.set_ylabel('Partial information (bits)')
    ax1.set_title('Feature Information Decomposition')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.2)

    # --- Right: summary bars ---
    categories = ['Redundant', 'Independent', 'Synergistic']
    counts = [
        info_results['n_redundant'],
        info_results['n_independent'],
        info_results['n_synergistic'],
    ]
    bar_colors = [INFO_COLORS['redundant'], INFO_COLORS['independent'],
                  INFO_COLORS['synergistic']]

    n_total = len(classification)
    bars = ax2.bar(categories, counts, color=bar_colors, alpha=0.7,
                   edgecolor='white')

    # Add percentage labels
    for bar, count in zip(bars, counts):
        pct = 100 * count / n_total if n_total > 0 else 0
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{count}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Number of feature groups')
    ax2.set_title(f'Information Classification\n'
                  f'(I_ensemble = {info_results["I_ensemble"]:.3f} bits)')

    # Add ensemble vs sum annotation
    sum_single = I_single.sum()
    P_ens = info_results['I_ensemble'] - sum_single
    ax2.text(0.98, 0.95,
             f'Σ I_single = {sum_single:.3f}\n'
             f'P_ensemble = {P_ens:+.3f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5deb3', alpha=0.5))

    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 16: KERNEL SIMILARITY NETWORK
# ============================================================================
#
# Reveals the correlation structure among surviving features. Features that
# fire on the same temporal regions are correlated; RFE exploits this
# redundancy. The correlation matrix and optional network graph show which
# kernels form clusters and which are truly independent.

def plot_kernel_similarity(
    model, X_test, feature_mask=None, method="correlation",
    threshold=0.5, n_top=None, figsize=(12, 5),
):
    """
    Visualize similarity among kernel features.

    Parameters
    ----------
    model : InterpRocket
        A fitted model.
    X_test : ndarray, shape (n_instances, n_timepoints)
    feature_mask : ndarray, optional
        Feature indices to analyze. If None, uses top features by importance.
    method : str, default='correlation'
        'correlation' — Pearson correlation of feature values across instances.
    threshold : float, default=0.5
        For the network panel, edges are drawn for |r| > threshold.
    n_top : int, optional
        Number of top features if feature_mask is None. Default: 50.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    corr_matrix : ndarray — the correlation matrix
    """

    from matplotlib.colors import LinearSegmentedColormap

    # Custome colormap: tab:blue = #1f77b4, tab:orange = #ff7f0e
    blue_orange = LinearSegmentedColormap.from_list(
        'BlueOrange', ['#1f77b4', '#f0f0f0', '#ff7f0e'], N=256
    )

    X_test = np.asarray(X_test, dtype=np.float32)
    features = model.transform(X_test)

    if feature_mask is None:
        importance = model.get_feature_importance()
        if n_top is None:
            n_top = min(50, len(importance))
        top_idx = np.argsort(importance)[::-1][:n_top]
        feature_mask = top_idx

    features = features[:, feature_mask]
    n_feats = features.shape[1]

    # Compute correlation matrix
    corr = np.corrcoef(features.T)
    np.fill_diagonal(corr, 0)  # zero diagonal for visualization

    # Decode labels
    labels = []
    for fi in feature_mask:
        info = model.decode_feature_index(int(fi))
        labels.append(f"K{info['kernel_index']}d{info['dilation']}"
                      f"{info['pooling_op'][0]}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: correlation matrix ---
    im = ax1.imshow(corr, cmap=blue_orange, vmin=-1, vmax=1,
                    interpolation='nearest')
    ax1.set_title(f'Feature Correlation ({n_feats} features)')
    plt.colorbar(im, ax=ax1, shrink=0.8, label='Pearson r')

    if n_feats <= 30:
        ax1.set_xticks(range(n_feats))
        ax1.set_xticklabels(labels, rotation=90, fontsize=8)
        ax1.set_yticks(range(n_feats))
        ax1.set_yticklabels(labels, fontsize=8)

    # --- Right: network/adjacency summary ---
    abs_corr = np.abs(corr)
    n_edges = np.sum(abs_corr > threshold) // 2  # symmetric, no diagonal

    # Degree: number of strong connections per feature
    degree = np.sum(abs_corr > threshold, axis=1)

    # Group by kernel to show cluster structure
    kernel_ids = []
    for fi in feature_mask:
        info = model.decode_feature_index(int(fi))
        kernel_ids.append(info['kernel_index'])
    kernel_ids = np.array(kernel_ids)

    unique_kernels = np.unique(kernel_ids)
    n_unique = len(unique_kernels)

    # Between-kernel vs within-kernel correlation
    within_corrs = []
    between_corrs = []
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            if kernel_ids[i] == kernel_ids[j]:
                within_corrs.append(abs_corr[i, j])
            else:
                between_corrs.append(abs_corr[i, j])

    bins = np.linspace(0, 1, 30)
    if within_corrs:
        ax2.hist(within_corrs, bins=bins, alpha=0.6, color='#1f77b4',
                 label=f'Within-kernel ({len(within_corrs)})', density=True)
    if between_corrs:
        ax2.hist(between_corrs, bins=bins, alpha=0.6, color='#ff7f0e',
                 label=f'Between-kernel ({len(between_corrs)})', density=True)
    ax2.axvline(threshold, color='#7f7f7f', linestyle='--', alpha=0.7,
                label=f'Threshold ({threshold})')
    ax2.set_xlabel('|Correlation|')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Correlation Distribution\n'
                  f'{n_unique} unique kernels, {n_edges} edges > {threshold}')
    ax2.legend(fontsize=8)

    #fig.suptitle('Kernel Similarity Network', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig, corr


# ============================================================================
# SECTION 17: CONFUSION-CONDITIONED ACTIVATION MAPS
# ============================================================================
#
# Shows activation maps separately for correctly classified trials and
# misclassified trials. Reveals where the model fails and what temporal
# patterns are associated with errors.

def plot_confusion_conditioned_maps(
    model,
    X_test,
    y_test,
    feature_mask=None,
    n_top=50,
    figsize=(12, None),
):
    """
    Temporal importance maps conditioned on classification outcome.

    Computes separate activation-rate profiles for correctly classified
    trials and misclassified trials per class, revealing which temporal
    regions are associated with errors.

    Parameters
    ----------
    model : InterpRocket
        A fitted model.
    X_test : ndarray, shape (n_instances, n_timepoints)
    y_test : array-like
    feature_mask : ndarray, optional
        Feature indices to analyze.
    n_top : int, default=50
        Number of top features if feature_mask is None.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)

    classes = model.classes_
    n_classes = len(classes)
    n_timepoints = X_test.shape[1]

    # Get importance and feature mask
    importance = model.get_feature_importance(feature_mask=feature_mask)
    if feature_mask is None:
        n_use = min(n_top, len(importance))
        feature_mask = np.argsort(importance)[::-1][:n_use]

    if figsize[1] is None:
        figsize = (figsize[0], 3 * n_classes)

    fig, axes = plt.subplots(n_classes, 1, figsize=figsize, sharex=True)
    if n_classes == 1:
        axes = [axes]

    colors_correct = [TAB10[i] for i in range(n_classes)]

    for k, cls in enumerate(classes):
        ax = axes[k]

        # Split trials for this class
        cls_mask = y_test == cls
        correct_mask = cls_mask & (y_pred == cls)
        wrong_mask = cls_mask & (y_pred != cls)

        n_correct = correct_mask.sum()
        n_wrong = wrong_mask.sum()

        # Compute temporal activation profiles for correct and wrong
        for mask, label, color, ls in [
            (correct_mask, f"Correct ({n_correct})", colors_correct[k], "-"),
            (wrong_mask, f"Misclassified ({n_wrong})", "#7f7f7f", "--"),
        ]:
            if mask.sum() == 0:
                continue

            X_subset = X_test[mask]
            profile = np.zeros(n_timepoints)
            total_activations = 0.0

            for fi in feature_mask:
                info = model.decode_feature_index(int(fi))
                dilation = info["dilation"]
                rep = info["representation"]
                imp = importance[fi] if fi < len(importance) else 0

                for trial_idx in range(len(X_subset)):
                    if rep == "diff":
                        x_use = np.diff(X_subset[trial_idx]).astype(np.float32)
                    else:
                        x_use = X_subset[trial_idx]

                    conv_out, activated, t_indices = compute_activation_map(
                        x_use,
                        info["kernel_index"],
                        dilation,
                        info["bias"],
                    )
                    for j in range(len(activated)):
                        if activated[j] > 0:
                            center = int(round(t_indices[j]))
                            if rep == "diff":
                                center = min(center + 1, n_timepoints - 1)
                            if 0 <= center < n_timepoints:
                                profile[center] += imp
                                total_activations += imp

            if total_activations > 0:
                profile /= total_activations

            ax.plot(
                range(n_timepoints),
                profile,
                color=color,
                linewidth=1.5,
                linestyle=ls,
                label=label,
            )

        ax.set_ylabel(f"Class {cls}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Timepoint")
    #fig.suptitle("Confusion-Conditioned Activation Maps", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 18: CROSS-VALIDATION FEATURE STABILITY
# ============================================================================
#
# Assesses which features are consistently important across CV folds versus
# fold-specific. A feature that ranks highly in every fold is robustly
# interpretable; one that appears in only some folds may be fitting noise.

def cv_feature_stability(
    X, y, n_repeats=5, n_folds=5, n_top=50,
    random_state=42, verbose=True, **model_kwargs
):
    """
    Measure feature importance stability across cross-validation folds.

    Fits InterpRocket on each fold and records the top features ranked
    by importance. Returns a frequency matrix showing how often each
    feature appears in the top set across folds.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints)
    y : array-like
    n_repeats : int
    n_folds : int
    n_top : int, default=50
        Number of top features to track per fold.
    random_state : int
    verbose : bool
    **model_kwargs : passed to InterpRocket

    Returns
    -------
    results : dict with keys:
        'feature_counts' : ndarray — how many folds each feature appeared
            in the top set
        'top_features_per_fold' : list of ndarray — top feature indices per fold
        'importance_per_fold' : list of ndarray — full importance vector per fold
        'decoded_stable' : list of dict — decoded info for the most stable features
        'n_folds_total' : int
        'ref_model' : InterpRocket — reference model for feature decoding
    """
    import contextlib, io

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                  random_state=random_state)
    total_folds = n_folds * n_repeats

    # We need a reference model to know the total feature count
    ref_model = InterpRocket(**model_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        ref_model.fit(X, y)
    n_features_total = ref_model.transform(X[:1]).shape[1]

    feature_counts = np.zeros(n_features_total, dtype=int)
    top_features_per_fold = []
    importance_per_fold = []

    if verbose:
        print(f"CV feature stability: {n_repeats}x{n_folds} folds, "
              f"tracking top {n_top} features per fold")

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model = InterpRocket(**model_kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X[train_idx], y[train_idx])

        imp = model.get_feature_importance()
        top_idx = np.argsort(imp)[::-1][:n_top]

        feature_counts[top_idx] += 1
        top_features_per_fold.append(top_idx)
        importance_per_fold.append(imp)

        if verbose and (fold_idx + 1) % n_folds == 0:
            r = (fold_idx + 1) // n_folds
            n_stable = np.sum(feature_counts >= r)
            print(f"  Round {r}/{n_repeats}: "
                  f"{n_stable} features appeared in every round so far")

    # Decode the most stable features
    stable_order = np.argsort(feature_counts)[::-1]
    decoded_stable = []
    for fi in stable_order[:n_top]:
        info = ref_model.decode_feature_index(int(fi))
        info['stability_count'] = int(feature_counts[fi])
        info['stability_fraction'] = feature_counts[fi] / total_folds
        info['feature_index'] = int(fi)
        decoded_stable.append(info)

    if verbose:
        n_always = np.sum(feature_counts == total_folds)
        n_most = np.sum(feature_counts >= total_folds * 0.8)
        n_never = np.sum(feature_counts == 0)
        print(f"\n  Features in ALL folds:  {n_always}")
        print(f"  Features in ≥80% folds: {n_most}")
        print(f"  Features in NO folds:   {n_never}")

    return {
        'feature_counts': feature_counts,
        'top_features_per_fold': top_features_per_fold,
        'importance_per_fold': importance_per_fold,
        'decoded_stable': decoded_stable,
        'n_folds_total': total_folds,
        'ref_model': ref_model,
    }


def plot_feature_stability(stability, model=None, figsize=(14, None)):
    """
    Visualize cross-validation feature stability.

    Parameters
    ----------
    stability : dict
        Output from cv_feature_stability().
    model : InterpRocket, optional
        If provided, feature labels use decoded kernel names.
        If None, uses the ref_model stored in stability results.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    from matplotlib.colors import ListedColormap

    # Use the ref_model from stability results if no model provided,
    # or if the user's model has a different feature count
    if 'ref_model' in stability:
        decode_model = stability['ref_model']
    else:
        decode_model = model

    # Fall back to user model if ref_model not available
    if decode_model is None:
        decode_model = model

    top_sets = stability['top_features_per_fold']
    feature_counts = stability['feature_counts']
    n_folds = stability['n_folds_total']
    all_features = np.where(feature_counts > 0)[0]
    scores = feature_counts[all_features] / n_folds

    # Keep only top 50 most stable
    n_show = min(50, len(all_features))
    top_idx = np.argsort(scores)[::-1][:n_show]
    all_features = all_features[top_idx]
    scores = scores[top_idx]

    n_features = len(all_features)

    if figsize[1] is None:
        figsize = (figsize[0], max(6, min(16, n_features * 0.25)))

    # Build presence matrix
    matrix = np.zeros((n_features, n_folds), dtype=int)
    for j, ts in enumerate(top_sets):
        for fi in ts:
            if fi in all_features:
                row = np.where(all_features == fi)[0][0]
                matrix[row, j] = 1

    # Sort by stability
    order = np.argsort(scores)[::-1]
    matrix = matrix[order]
    sorted_features = all_features[order]
    sorted_scores = scores[order]

    # Feature labels
    if decode_model is not None:
        labels = []
        for fi in sorted_features:
            info = decode_model.decode_feature_index(int(fi))
            labels.append(
                f"K{info['kernel_index']}d{info['dilation']}"
                f"{info['pooling_op'][0].upper()}"
            )
        # Disambiguate duplicates by appending pooling index
        seen = {}
        for i, label in enumerate(labels):
            if labels.count(label) > 1:
                info = decode_model.decode_feature_index(int(sorted_features[i]))
                labels[i] = f"{label}_{info['pooling_index']}"
    else:
        labels = [str(fi) for fi in sorted_features]

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=figsize, width_ratios=[3, 1],
        gridspec_kw={'wspace': 0.05}
    )

    # --- Heatmap ---
    cmap = ListedColormap(['#f0f0f0', '#1f77b4'])
    ax_heat.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    ax_heat.set_facecolor('white')

    # Gridlines
    ax_heat.set_xticks(np.arange(-0.5, n_folds, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
    ax_heat.grid(which='minor', color='white', linewidth=0.5)
    ax_heat.tick_params(which='minor', length=0)

    ax_heat.set_xticks(np.arange(0, n_folds, max(1, n_folds // 10)))
    ax_heat.set_yticks(range(n_features))
    ax_heat.set_yticklabels(labels, fontsize=7)
    ax_heat.set_xlabel('CV Fold')
    ax_heat.set_ylabel('Feature (ranked by stability)')
    ax_heat.set_title(f'Feature Presence in Top Set ({n_folds} folds)')

    # --- Bar chart ---
    colors = ['#1f77b4' if s >= 0.8 else '#a3c4e0' for s in sorted_scores]
    ax_bar.barh(range(n_features), sorted_scores, color=colors,
                edgecolor='white', linewidth=0.3)
    ax_bar.axvline(0.8, color='#7f7f7f', linestyle='--', linewidth=0.8,
                   label='80% threshold')
    ax_bar.set_ylim(ax_heat.get_ylim())
    ax_bar.set_yticklabels([])
    ax_bar.set_xlabel('Fraction of folds in top set')
    ax_bar.set_title('Feature Stability')
    ax_bar.legend(fontsize=7, loc='lower right')

    #fig.suptitle('Feature Stability', fontsize=13, y=0.9)
    return fig


# ============================================================================
# SECTION 19: RECEPTIVE FIELD DIAGRAM
# ============================================================================
#
# Shows the temporal footprint of each surviving feature overlaid on the
# signal. Each feature's receptive field (determined by dilation) is drawn
# as a horizontal bar, making it immediately clear which time regions are
# covered by the classifier's feature set.

def plot_receptive_field_diagram(
    model, X_test=None, y_test=None, feature_mask=None, n_top=30,
    figsize=(12, 8),
):
    """
    Plot the receptive field footprint of top features on the signal.

    Top panel: class means (if X_test and y_test provided).
    Bottom panel: horizontal bars showing each feature's receptive field,
    colored by pooling operator, with vertical position ordered by importance.

    Parameters
    ----------
    model : InterpRocket
        A fitted model.
    X_test : ndarray, optional
        Test data for plotting class means and computing activation peaks.
    y_test : array-like, optional
        Test labels for class means.
    feature_mask : ndarray, optional
        Feature indices to show. If None, uses top by importance.
    n_top : int, default=30
        Number of features if feature_mask is None.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    importance = model.get_feature_importance(feature_mask=feature_mask)

    if feature_mask is None:
        n_use = min(n_top, len(importance))
        feature_mask = np.argsort(importance)[::-1][:n_use]
    else:
        # Sort mask by importance
        mask_importance = importance[feature_mask]
        sort_order = np.argsort(mask_importance)[::-1][:n_top]
        feature_mask = feature_mask[sort_order]

    # Decode all features
    decoded = []
    for fi in feature_mask:
        info = model.decode_feature_index(int(fi))
        info['feature_index'] = int(fi)
        info['importance'] = float(importance[fi])
        decoded.append(info)

    ## Sort by dilation (ascending) for visual clarity
    # decoded.sort(key=lambda x: (x['dilation'], x['kernel_index']))

    # Sort by importance from the classifier
    decoded.sort(key=lambda x: -x['importance'])

    n_timepoints = X_test.shape[1]
    n_feats = len(decoded)

    has_signal = X_test is not None and y_test is not None
    if has_signal:
        fig, (ax_sig, ax_rf) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={'height_ratios': [1, 2.5]}
        )
    else:
        fig, ax_rf = plt.subplots(figsize=figsize)
        ax_sig = None

    # --- Top panel: class means ---
    if has_signal:
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test)
        classes = np.unique(y_test)
        n_classes = len(classes)
        colors_class = [TAB10[i] for i in range(n_classes)]
        t = np.arange(n_timepoints)

        # Compute class means (used for both the plot and activation peaks)
        class_means = []
        for cls in classes:
            mask = y_test == cls
            class_means.append(X_test[mask].mean(axis=0))

        for k, cls in enumerate(classes):
            ax_sig.plot(t, class_means[k], color=colors_class[k],
                        linewidth=1.5, label=f'Class {cls}')
        ax_sig.legend(fontsize=8, ncol=n_classes)
        ax_sig.set_ylabel('Amplitude')
        ax_sig.set_title('Class Means')
        ax_sig.grid(True, alpha=0.2)

    # --- Compute peak differential activation for each feature ---
    # For each feature, run compute_activation_map on each class mean,
    # find the timepoint of maximum (max - min) activation across classes,
    # and use that as the center position for the RF bar.
    peak_centers = []
    for info in decoded:
        if not has_signal:
            # Fallback: center of series
            peak_centers.append(n_timepoints / 2)
            continue

        ki = info['kernel_index']
        dil = info['dilation']
        rep = info['representation']
        bias = info['bias']
        rf = info['receptive_field']

        # Compute activation on each class mean
        class_act_rates = []
        for k, cls in enumerate(classes):
            if rep == 'diff':
                x_use = np.diff(class_means[k]).astype(np.float32)
            else:
                x_use = class_means[k]

            conv_out, act, time_idx = compute_activation_map(
                x_use, ki, np.int32(dil), np.float32(bias)
            )

            # Build per-timepoint activation profile
            act_profile = np.zeros(n_timepoints, dtype=np.float64)
            for t_i in range(len(act)):
                center = int(round(time_idx[t_i]))
                if rep == 'diff':
                    center = min(center + 1, n_timepoints - 1)
                if 0 <= center < n_timepoints:
                    act_profile[center] = act[t_i]
            class_act_rates.append(act_profile)

        # Differential activation: max - min across classes at each timepoint
        class_act_array = np.array(class_act_rates)
        diff_act = np.max(class_act_array, axis=0) - np.min(class_act_array, axis=0)

        # Find peak via center of mass — robust to plateaus where
        # argmax would return the leftmost edge (e.g., MIPV features
        # that fire broadly across the bump class).
        total = diff_act.sum()
        if total > 0:
            peak_t = int(round(
                np.dot(np.arange(n_timepoints, dtype=np.float64), diff_act) / total
            ))
            peak_t = max(0, min(peak_t, n_timepoints - 1))
        else:
            # Secondary fallback: use convolution output magnitude
            # on the overall mean signal
            overall_mean = np.mean(class_means, axis=0).astype(np.float32)
            if rep == 'diff':
                overall_mean = np.diff(overall_mean).astype(np.float32)
            conv_out, act, time_idx = compute_activation_map(
                overall_mean, ki, np.int32(dil), np.float32(bias)
            )
            if len(conv_out) > 0 and np.max(np.abs(conv_out)) > 0:
                best_t_idx = int(np.argmax(np.abs(conv_out)))
                peak_t = int(round(time_idx[best_t_idx]))
                if rep == 'diff':
                    peak_t = min(peak_t + 1, n_timepoints - 1)
                peak_t = max(0, min(peak_t, n_timepoints - 1))
            else:
                peak_t = n_timepoints // 2

        peak_centers.append(peak_t)

    # --- Bottom panel: receptive fields ---

    for row, info in enumerate(decoded):
        rf = info['receptive_field']
        dil = info['dilation']
        color = POOLING_COLORS.get(info['pooling_op'], '#7f7f7f')
        imp_alpha = 0.3 + 0.7 * info['importance']

        # RF bar centered on peak differential activation
        peak_t = peak_centers[row]
        rf_left = max(0, peak_t - rf / 2)
        # Ensure the bar doesn't extend beyond the series
        if rf_left + rf > n_timepoints:
            rf_left = max(0, n_timepoints - rf)
        ax_rf.barh(row, rf, left=rf_left, height=0.7,
                   color=color, alpha=imp_alpha, edgecolor='white',
                   linewidth=0.3)

    # Labels
    y_labels = [
        f"K{d['kernel_index']}d{d['dilation']} {d['pooling_op']} "
        f"(RF={d['receptive_field']})"
        for d in decoded
    ]
    ax_rf.set_yticks(range(n_feats))
    ax_rf.set_yticklabels(y_labels, fontsize=8)
    ax_rf.set_xlabel('Timepoint')
    ax_rf.set_ylabel('Feature (sorted by dilation)')
    ax_rf.set_xlim(0, n_timepoints)
    ax_rf.set_ylim(-0.5, n_feats - 0.5)
    ax_rf.invert_yaxis()

    # Legend for pooling operators
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=op)
                       for op, c in POOLING_COLORS.items()]
    ax_rf.legend(handles=legend_elements, fontsize=8, loc='lower right',
                 title='Pooling', title_fontsize=8)

    ax_rf.set_title(f'Receptive Field Diagram ({n_feats} features)')
    #fig.suptitle('Feature Receptive Fields', fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 20: FEATURE STABILITY SELECTION
# ============================================================================
#
# Provides a helper to extract stable features from cv_feature_stability
# results using a threshold on the fraction of folds in which each feature
# appeared in the top set. This is an alternative to RFE that is robust
# to random seed and split variability.
#
# References:
#   Meinshausen, N. and Bühlmann, P. (2010). Stability selection.
#       JRSS-B, 72(4):417-473. doi:10.1111/j.1467-9868.2010.00740.x
#   Saeys, Y., Abeel, T., and Van de Peer, Y. (2008). Robust feature
#       selection using ensemble feature selection techniques.
#       Proc. ECML PKDD, 313-325. doi:10.1007/978-3-540-87481-2_21

def get_stable_features(stability, threshold=0.8):
    """
    Extract feature indices that appear in the top set in at least
    `threshold` fraction of CV folds.

    Parameters
    ----------
    stability : dict
        Output from cv_feature_stability().
    threshold : float, default=0.8
        Minimum fraction of folds a feature must appear in.
        1.0 = present in every fold. 0.5 = present in half.

    Returns
    -------
    stable_features : ndarray of int
        Feature indices meeting the threshold, sorted by frequency
        (most stable first).
    """
    counts = stability['feature_counts']
    n_folds = stability['n_folds_total']
    mask = counts >= threshold * n_folds

    # Sort by frequency (descending), then by index (ascending) for ties
    indices = np.where(mask)[0]
    order = np.argsort(-counts[indices])
    stable_features = indices[order]

    print(f"Stable features (≥{threshold:.0%} of {n_folds} folds): "
          f"{len(stable_features)}")
    return stable_features


# ============================================================================
# SECTION 21: CLASS-MEAN VISUALIZATION
# ============================================================================
#
# Functions for applying decoded kernels to class-averaged signals.
# These strip trial-to-trial variability and show what each kernel
# detects on the idealized signal for each class.

def plot_class_mean_activation(
    model, X, y, feature_mask=None, feature_rank=0,
    figsize=None,
):
    """
    Side-by-side visualization of kernel activation (left) and raw
    convolution output (right) on class-mean signals for a single feature.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
        Data (typically X_test).
    y : array-like
        Class labels.
    feature_mask : array-like of int, optional
        Subset of feature indices (e.g., from get_stable_features).
    feature_rank : int, default=0
        Which feature to plot, ranked by importance within feature_mask.
        0 = top feature, 1 = second, etc.
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    top = model.get_top_features(feature_mask=feature_mask)
    if feature_rank >= len(top):
        raise ValueError(
            f"feature_rank={feature_rank} but only {len(top)} features available"
        )
    f = top[feature_rank]
    ki = f['kernel_index']
    dil = f['dilation']
    bias = f['bias']
    rep = f['representation']
    pooling = f['pooling_op']

    classes = np.unique(y)
    n_classes = len(classes)
    colors = [TAB10[i] for i in range(n_classes)]

    if figsize is None:
        figsize = (14, 3 * n_classes)

    fig, axes = plt.subplots(n_classes, 2, figsize=figsize, sharex=True)
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for k, cls in enumerate(classes):
        mask = y == cls
        if rep == 'diff':
            class_mean = np.diff(X[mask].mean(axis=0)).astype(np.float32)
        else:
            class_mean = X[mask].mean(axis=0).astype(np.float32)

        conv_out, act, t_idx = compute_activation_map(
            class_mean, ki, np.int32(dil), np.float32(bias)
        )

        # Left: activation map
        ax_act = axes[k, 0]
        ax_act.plot(class_mean, color='#7f7f7f', alpha=0.5, label='Class mean')
        ax_act.fill_between(
            t_idx, 0, act * class_mean.max() * 0.3,
            color=colors[k], alpha=0.3, label='Activation'
        )
        ax_act.set_ylabel(f'Class {cls}')
        ax_act.legend(fontsize=7)
        ax_act.grid(True, alpha=0.2)

        # Right: convolution output with bias line
        ax_conv = axes[k, 1]
        ax_conv.plot(class_mean, color='#7f7f7f', alpha=0.5, label='Class mean')
        ax2 = ax_conv.twinx()
        ax2.plot(t_idx, conv_out, color=colors[k], linewidth=1.5,
                 label='Conv output')
        ax2.axhline(bias, color='#2c2c2c', linestyle='--', linewidth=0.8,
                     label=f'Bias={bias:.2f}')
        ax2.fill_between(t_idx, bias, conv_out, where=conv_out > bias,
                         color=colors[k], alpha=0.2)
        ax2.set_ylabel('Conv output')
        if k == 0:
            ax2.legend(fontsize=7, loc='upper right')
        ax_conv.grid(True, alpha=0.2)

    axes[0, 0].set_title('Activation on class mean')
    axes[0, 1].set_title('Convolution output on class mean')
    axes[-1, 0].set_xlabel('Timepoint')
    axes[-1, 1].set_xlabel('Timepoint')

    fig.suptitle(
        f'K{ki} d={dil} {pooling} ({rep}) — rank {feature_rank + 1}',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    return fig


def plot_multi_kernel_summary(
    model, X, y, feature_mask=None, n_show=15, figsize=None,
):
    """
    Heatmap of binary activation across top features on class-mean signals.

    Rows are features sorted by importance, columns are timepoints, panels
    are classes. Dark cells indicate the kernel fires at that position on
    the class mean. Features that never fire on any class mean are marked
    with † and dimmed.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    feature_mask : array-like of int, optional
    n_show : int, default=15
        Maximum number of features to display.
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    top = model.get_top_features(feature_mask=feature_mask)
    n_show = min(len(top), n_show)
    classes = np.unique(y)
    n_classes = len(classes)
    n_timepoints = X.shape[1]

    if figsize is None:
        figsize = (5 * n_classes, 0.4 * n_show + 1.5)

    fig, axes = plt.subplots(1, n_classes, figsize=figsize, sharey=True)
    if n_classes == 1:
        axes = [axes]

    # Pre-check: does each feature fire on any class mean?
    fires_on_mean = []
    for f in top[:n_show]:
        any_fires = False
        for cls in classes:
            mask = y == cls
            if f['representation'] == 'diff':
                cm = np.diff(X[mask].mean(axis=0)).astype(np.float32)
            else:
                cm = X[mask].mean(axis=0).astype(np.float32)
            _, act, _ = compute_activation_map(
                cm, f['kernel_index'], f['dilation'], f['bias']
            )
            if act.max() > 0:
                any_fires = True
                break
        fires_on_mean.append(any_fires)

    for k, cls in enumerate(classes):
        mask = y == cls
        act_matrix = []
        labels = []

        for row_idx, f in enumerate(top[:n_show]):
            if f['representation'] == 'diff':
                cm = np.diff(X[mask].mean(axis=0)).astype(np.float32)
            else:
                cm = X[mask].mean(axis=0).astype(np.float32)

            _, act, t_idx = compute_activation_map(
                cm, f['kernel_index'], f['dilation'], f['bias']
            )
            act_full = np.zeros(n_timepoints)
            for i, t in enumerate(t_idx):
                ti = int(round(t))
                if 0 <= ti < n_timepoints:
                    act_full[ti] = act[i]
            act_matrix.append(act_full)

            tag = "" if fires_on_mean[row_idx] else " †"
            labels.append(
                f"K{f['kernel_index']} d={f['dilation']} "
                f"{f['pooling_op']}{tag}"
            )

        ax = axes[k]
        from matplotlib.colors import ListedColormap
        cmap_binary = ListedColormap(['#f0f0f0', '#1f77b4'])
        ax.imshow(act_matrix, aspect='auto', cmap=cmap_binary, alpha=0.6,
                  interpolation='nearest', vmin=0, vmax=0.1)

        # Dim subthreshold rows
        for row_idx in range(n_show):
            if not fires_on_mean[row_idx]:
                ax.axhspan(row_idx - 0.5, row_idx + 0.5,
                           color='white', alpha=0.6, zorder=2)

        ax.set_xlabel('Timepoint')
        ax.set_title(f'Class {cls}')
        if k == 0:
            ax.set_yticks(range(n_show))
            ax.set_yticklabels(labels, fontsize=8)
            for row_idx, label_obj in enumerate(ax.get_yticklabels()):
                if not fires_on_mean[row_idx]:
                    label_obj.set_alpha(0.4)

    fig.suptitle(
        'Activation on class means (top features)\n'
        '† = subthreshold on class means (fires on individual trials only)',
        fontsize=11, y=1.04
    )
    plt.tight_layout()
    return fig


def plot_aggregate_activation(
    model, X, y, feature_mask=None, figsize=(8, 6),
):
    """
    Importance-weighted sum of kernel activations on class means,
    collapsed across all features.

    Top panel shows the aggregate activation curve per class.
    Bottom panel shows the max-min differential across classes at each
    timepoint, highlighting where features collectively discriminate.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    feature_mask : array-like of int, optional
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib Figure
    class_activation : ndarray, shape (n_classes, n_timepoints)
    differential : ndarray, shape (n_timepoints,)
    """
    import matplotlib.pyplot as plt

    top = model.get_top_features(feature_mask=feature_mask)
    classes = np.unique(y)
    n_classes = len(classes)
    n_timepoints = X.shape[1]
    colors = [TAB10[i] for i in range(n_classes)]

    class_activation = np.zeros((n_classes, n_timepoints))

    for f in top:
        imp = f['importance']
        for k, cls in enumerate(classes):
            mask = y == cls
            if f['representation'] == 'diff':
                cm = np.diff(X[mask].mean(axis=0)).astype(np.float32)
            else:
                cm = X[mask].mean(axis=0).astype(np.float32)

            _, act, t_idx = compute_activation_map(
                cm, f['kernel_index'], f['dilation'], f['bias']
            )
            for i, t in enumerate(t_idx):
                ti = int(round(t))
                if 0 <= ti < n_timepoints:
                    class_activation[k, ti] += act[i] * imp

    # Normalize by number of features
    class_activation /= len(top)

    differential = (np.max(class_activation, axis=0)
                    - np.min(class_activation, axis=0))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    for k, cls in enumerate(classes):
        ax.plot(class_activation[k], color=colors[k], linewidth=1.5,
                label=f'Class {cls}')
    ax.legend(fontsize=8)
    ax.set_ylabel('Activation')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.fill_between(range(n_timepoints), differential,
                    color='#7f7f7f', alpha=0.4)
    ax.plot(differential, color='#2c2c2c', linewidth=1.5)
    ax.set_ylabel('Differential')
    ax.set_xlabel('Timepoint')
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f'Aggregate kernel activation on class means '
        f'({len(top)} features)',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    return fig, class_activation, differential


def aggregate_temporal_occlusion(
    model, X, y, feature_mask=None, window_size=None,
    stride=None, verbose=True,
):
    """
    Temporal occlusion sensitivity computed over all trials, grouped by class.

    A sliding window of zeros is passed across each trial. The change in
    the classifier's decision function is measured at each position and
    averaged within each class. Returns per-class mean ± SEM profiles
    and plots them with a differential panel.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    feature_mask : array-like of int, optional
        If provided, zero out non-masked features before computing
        the decision function.
    window_size : int, optional
        Width of the occlusion window. Default: max(3, n_timepoints // 20).
    stride : int, optional
        Step size. Default: max(1, window_size // 2).
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'class_sensitivities' : dict of class → ndarray (n_trials, n_timepoints)
        'window_size' : int
        'stride' : int
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n_timepoints = X.shape[1]
    classes = np.unique(y)
    n_classes = len(classes)

    if window_size is None:
        window_size = max(3, n_timepoints // 20)
    if stride is None:
        stride = max(1, window_size // 2)

    if verbose:
        print(f"Aggregate temporal occlusion: window={window_size}, "
              f"stride={stride}, {len(y)} trials")

    def _decision(X_single):
        features = _transform(
            X_single,
            model.dilations_raw_, model.num_features_per_dilation_raw_,
            model.biases_raw_,
        )
        if hasattr(model, 'dilations_diff_'):
            diff_feats = _transform(
                np.diff(X_single, axis=1),
                model.dilations_diff_, model.num_features_per_dilation_diff_,
                model.biases_diff_,
            )
            features = np.hstack([features, diff_feats])
        if feature_mask is not None:
            mask_arr = np.zeros(features.shape[1], dtype=bool)
            valid = feature_mask[feature_mask < features.shape[1]]
            mask_arr[valid] = True
            features[:, ~mask_arr] = 0.0
        features_scaled = model.scaler_.transform(features)
        return model.classifier_.decision_function(features_scaled)

    class_sensitivities = {cls: [] for cls in classes}

    for trial_idx in range(len(X)):
        X_single = X[trial_idx:trial_idx + 1].astype(np.float32)
        base_decision = _decision(X_single)

        sensitivity = np.zeros(n_timepoints)
        counts = np.zeros(n_timepoints)

        for pos in range(0, n_timepoints - window_size + 1, stride):
            X_occ = X_single.copy()
            X_occ[0, pos:pos + window_size] = 0.0
            occ_decision = _decision(X_occ)
            impact = np.sum(np.abs(base_decision - occ_decision))
            sensitivity[pos:pos + window_size] += impact
            counts[pos:pos + window_size] += 1

        counts[counts == 0] = 1
        sensitivity /= counts

        cls = y[trial_idx]
        class_sensitivities[cls].append(sensitivity)

        if verbose and (trial_idx + 1) % 100 == 0:
            print(f"  {trial_idx + 1}/{len(X)} trials")

    for cls in classes:
        class_sensitivities[cls] = np.array(class_sensitivities[cls])

    if verbose:
        print("Done.")

    # Plot
    colors = [TAB10[i] for i in range(n_classes)]
    fig, axes = plt.subplots(
        n_classes + 1, 1,
        figsize=(12, 2.5 * (n_classes + 1)),
        sharex=True,
    )

    for k, cls in enumerate(classes):
        ax = axes[k]
        sens = class_sensitivities[cls]
        mean_s = sens.mean(axis=0)
        sem_s = sens.std(axis=0) / np.sqrt(len(sens))

        ax.fill_between(range(n_timepoints),
                        mean_s - sem_s, mean_s + sem_s,
                        alpha=0.2, color=colors[k])
        ax.plot(range(n_timepoints), mean_s, color=colors[k], linewidth=1.5)
        ax.set_ylabel(f'Class {cls}\n(n={len(sens)})')
        ax.grid(True, alpha=0.2)

    # Differential panel
    ax = axes[-1]
    all_means = np.array([class_sensitivities[cls].mean(axis=0)
                          for cls in classes])
    differential = all_means.max(axis=0) - all_means.min(axis=0)
    ax.fill_between(range(n_timepoints), differential,
                    alpha=0.3, color='#7f7f7f')
    ax.plot(range(n_timepoints), differential, color='#2c2c2c', linewidth=1.5)
    ax.set_ylabel('Differential')
    ax.set_xlabel('Timepoint')
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f'Aggregate Temporal Occlusion '
        f'(window={window_size}, stride={stride}, {len(y)} trials)',
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    results = {
        'class_sensitivities': class_sensitivities,
        'window_size': window_size,
        'stride': stride,
    }
    return results, fig


# ============================================================================
# SECTION 23: PERMUTATION IMPORTANCE (PIMP)
# ============================================================================
#
# Implements the PIMP algorithm (Altmann et al., 2010) for computing
# statistically corrected feature importance with p-values. The method
# permutes class labels, refits a classifier, and compares observed
# importance against the null distribution.
#
# By default, a RandomForestClassifier is used for the
# importance computation (not the main InterpRocket Ridge classifier).
# Tree-based methods assign zero importance to genuinely uninformative
# features, producing a meaningful null distribution. Ridge assigns
# non-trivial coefficients to all features due to regularization,
# causing every feature to appear significant. The ROCKET transform
# is label-independent, so it is computed once and reused.
#
# The InterpRocket Ridge classifier remains the basis for all
# visualization and interpretability tools (temporal importance,
# RF diagrams, activation maps). PIMP provides an independent
# statistical test using a different classifier architecture.
#
# Reference:
#   Altmann, A., Tolosi, L., Sander, O., and Lengauer, T. (2010).
#       Permutation importance: a corrected feature importance measure.
#       Bioinformatics, 26(10):1340-1347.

def permutation_importance_test(
    model, X, y, n_permutations=100, classifier=None,
    feature_mask=None, random_state=42, verbose=True,
):
    """
    PIMP: Permutation Importance with p-values (Altmann et al., 2010).

    Computes feature importance on the real data using a tree-based
    classifier, then builds a null distribution by permuting class
    labels and refitting. Returns p-values for each feature.

    By default uses RandomForestClassifier, which produces meaningful
    null distributions because uninformative features receive zero
    importance. An alternative classifier can be passed via the
    `classifier` parameter.

    The ROCKET transform is label-independent, so it is computed once
    and reused across all permutations.

    Parameters
    ----------
    model : InterpRocket
        Fitted model. Must have been fitted with fit() before calling.
    X : ndarray, shape (n_samples, n_timepoints)
        Training data (same data used to fit the model).
    y : array-like
        Class labels.
    n_permutations : int, default=100
        Number of label permutations for the null distribution.
    classifier : sklearn classifier, optional
        Classifier to use for importance computation. Must provide
        either `feature_importances_` (tree-based) or `coef_` (linear)
        after fitting. Default: RandomForestClassifier.
    feature_mask : array-like of int, optional
        Subset of feature indices to test. If None, tests all features.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'observed_importance' : ndarray, shape (n_features,)
            Real feature importance (normalized to max=1).
        'null_importances' : ndarray, shape (n_permutations, n_features)
            Importance values from each permutation.
        'p_values' : ndarray, shape (n_features,)
            Fraction of null importances >= observed importance.
        'significant_mask' : ndarray of bool, shape (n_features,)
            Features with p < 0.05.
        'n_significant' : int
        'feature_mask' : ndarray or None
        'n_permutations' : int
        'classifier_type' : str
    """
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler

    if classifier is None:
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=8, n_jobs=-1,
            random_state=random_state,
        )

    rng = np.random.RandomState(random_state)

    # Transform data once (label-independent)
    features = model.transform(X)
    n_total_features = features.shape[1]

    if feature_mask is not None:
        feature_mask = np.asarray(feature_mask)
    else:
        feature_mask = np.arange(n_total_features)

    n_test = len(feature_mask)

    def _get_importance(clf):
        """Extract importance from fitted classifier."""
        if hasattr(clf, 'feature_importances_'):
            return clf.feature_importances_.copy()
        elif hasattr(clf, 'coef_'):
            coefs = clf.coef_
            if coefs.ndim > 1:
                return np.linalg.norm(coefs, axis=0, ord=2)
            else:
                return np.abs(coefs.ravel())
        else:
            raise ValueError(
                "Classifier must provide feature_importances_ or coef_"
            )

    # Fit on real labels
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clf_real = clone(classifier)
    clf_real.fit(features_scaled, y)
    obs_imp = _get_importance(clf_real)

    obs_max = obs_imp.max()
    if obs_max > 0:
        obs_imp = obs_imp / obs_max

    clf_type = type(classifier).__name__

    if verbose:
        print(f"PIMP: {n_permutations} permutations, "
              f"{n_test} features, classifier={clf_type}")

    # Build null distribution
    null_importances = np.zeros((n_permutations, n_total_features))

    for perm_idx in range(n_permutations):
        y_perm = rng.permutation(y)

        clf_perm = clone(classifier)
        clf_perm.fit(features_scaled, y_perm)

        perm_imp = _get_importance(clf_perm)

        # Normalize by the same max as observed (comparable scale)
        if obs_max > 0:
            perm_imp = perm_imp / obs_max

        null_importances[perm_idx] = perm_imp

        if verbose and (perm_idx + 1) % 25 == 0:
            print(f"  Permutation {perm_idx + 1}/{n_permutations}")

    # Compute p-values: fraction of null >= observed
    p_values = np.ones(n_total_features)
    for fi in feature_mask:
        p_values[fi] = np.mean(null_importances[:, fi] >= obs_imp[fi])

    significant = p_values[feature_mask] < 0.05
    n_sig = int(significant.sum())

    if verbose:
        print(f"\n  Significant features (p < 0.05): {n_sig} / {n_test}")
        print(f"  Non-significant: {n_test - n_sig} / {n_test}")

    results = {
        'observed_importance': obs_imp,
        'null_importances': null_importances,
        'p_values': p_values,
        'significant_mask': p_values < 0.05,
        'n_significant': n_sig,
        'feature_mask': feature_mask,
        'n_permutations': n_permutations,
        'classifier_type': clf_type,
    }
    return results


def plot_permutation_importance(
    pimp_results, model=None, n_show=30, alpha=0.05, figsize=(12, 8),
):
    """
    Visualize PIMP results: observed importance vs null distribution.

    Left panel: bar chart of observed importance for top features,
    colored by significance. Right panel: p-value distribution.

    Parameters
    ----------
    pimp_results : dict
        Output from permutation_importance_test().
    model : InterpRocket, optional
        If provided, feature labels include kernel/dilation/pooling info.
    n_show : int, default=30
        Maximum features to display in the bar chart.
    alpha : float, default=0.05
        Significance threshold.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    obs = pimp_results['observed_importance']
    pvals = pimp_results['p_values']
    null = pimp_results['null_importances']
    fm = pimp_results['feature_mask']

    # Sort features by observed importance
    order = np.argsort(-obs[fm])
    top_idx = fm[order[:n_show]]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [3, 1]})

    # Left: importance bar chart with null distribution overlay
    ax = axes[0]
    y_pos = np.arange(len(top_idx))

    # Null distribution summary per feature (mean + 95th percentile)
    null_means = null[:, top_idx].mean(axis=0)
    null_95 = np.percentile(null[:, top_idx], 95, axis=0)

    # Bars colored by significance
    colors = ['#1f77b4' if pvals[fi] < alpha else '#c7c7c7'
              for fi in top_idx]

    ax.barh(y_pos, obs[top_idx], color=colors, edgecolor='white',
            linewidth=0.5, zorder=3)
    ax.set_ylim(-0.5, len(top_idx) - 0.5)
    ax.scatter(null_95, y_pos, color='#d62728', marker='|', s=80,
               zorder=4, label='Null 95th pctl')
    ax.scatter(null_means, y_pos, color='#7f7f7f', marker='|', s=80,
               zorder=4, alpha=0.5, label='Null mean')

    # Labels
    if model is not None:
        labels = []
        for fi in top_idx:
            info = model.decode_feature_index(int(fi))
            p_str = f"p={pvals[fi]:.3f}" if pvals[fi] >= 0.001 else "p<0.001"
            labels.append(
                f"K{info['kernel_index']}d{info['dilation']} "
                f"{info['pooling_op']} ({p_str})"
            )
    else:
        labels = [f"F{fi} (p={pvals[fi]:.3f})" for fi in top_idx]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Permutation Importance ({len(top_idx)} features)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2, axis='x')

    # Right: p-value histogram
    ax2 = axes[1]
    p_tested = pvals[fm]
    ax2.hist(p_tested, bins=20, color='#1f77b4', edgecolor='white',
             orientation='horizontal')
    ax2.axhline(alpha, color='#d62728', linestyle='--', linewidth=1.5,
                label=f'alpha={alpha}')
    ax2.set_xlabel('Count')
    ax2.set_ylabel('p-value')
    ax2.set_title('p-value distribution')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.invert_yaxis()

    n_sig = pimp_results['n_significant']
    n_total = len(fm)
    clf_type = pimp_results.get('classifier_type', 'unknown')
    fig.suptitle(
        f'PIMP ({clf_type}): {n_sig}/{n_total} features significant '
        f'at p < {alpha} ({pimp_results["n_permutations"]} permutations)',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 20: CONVENIENCE / DEMO
# ============================================================================

def demo_with_synthetic_data():
    """
    Run a quick demonstration with synthetic data.

    Creates a 3-class problem where classes differ in which temporal region
    contains a discriminative pattern, then shows the model correctly
    identifies these regions.
    """
    np.random.seed(42)
    n_per_class = 200
    n_timepoints = 200
    noise = 0.3

    X_all = []
    y_all = []

    for cls in range(3):
        for _ in range(n_per_class):
            x = np.random.randn(n_timepoints).astype(np.float32) * noise
            # Each class has a bump in a different region
            if cls == 0:
                x[30:50] += 1.5
            elif cls == 1:
                x[100:120] += 1.5
            else:
                x[160:180] += 1.5
            X_all.append(x)
            y_all.append(cls)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all)

    # Shuffle and split
    idx = np.random.permutation(len(y_all))
    X_all, y_all = X_all[idx], y_all[idx]
    split = int(0.7 * len(y_all))
    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]

    print("=" * 60)
    print("InterpRocket Demo: Synthetic 3-class temporal bump data")
    print(f"  {n_timepoints} timepoints, bumps at [30:50], [100:120], [160:180]")
    print("=" * 60)

    model = InterpRocket(
        max_dilations_per_kernel=40, num_features=5000, random_state=666
    )
    model.fit(X_train, y_train)

    test_acc = model.score(X_test, y_test)
    print(f"\n  Test accuracy: {test_acc:.4f}")

    model.summary()

    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = demo_with_synthetic_data()

    print("\nGenerating visualizations...")
    fig1 = model.plot_top_kernels(X_test, y_test, n_kernels=3)
    fig1.savefig("multirocket_top_kernels.png", dpi=150, bbox_inches="tight")

    fig2, imp_by_time = model.plot_temporal_importance(X_test, y_test)
    fig2.savefig("multirocket_temporal_importance.png", dpi=150, bbox_inches="tight")

    fig3 = model.plot_feature_distributions(X_test, y_test)
    fig3.savefig("multirocket_distributions.png", dpi=150, bbox_inches="tight")

    fig4 = model.plot_kernel_properties()
    fig4.savefig("multirocket_properties.png", dpi=150, bbox_inches="tight")

    print("Done. Saved 4 visualization figures.")
