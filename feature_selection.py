"""Focused feature-selection tools for I-ROCKET.

The canonical selection path is:

``InterpRocketTransform`` -> shrinkage-*t* -> segmented cutoff within repeated
subsamples -> consensus selection probabilities -> Nogueira stability.

The public selectors in this module operate on one fixed transformed feature
matrix.  Leakage-free threshold and classifier tuning are implemented in
``irocket_model_selection``.

AUTHOR: Mark Laubach (American University, Department of Neuroscience)
LICENSE: BSD-3-Clause
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from interp_rocket import mutual_information
from _irocket_selection import (
    ResampledShrinkageSelector,
    ShrinkageFeatureSelector,
    nogueira_stability,
    screen_features,
    segmented_cutoff,
    shrinkage_t,
    shrinkage_t_ovr,
)

__all__ = [
    "ResampledShrinkageSelector",
    "ShrinkageFeatureSelector",
    "shrinkage_t",
    "shrinkage_t_ovr",
    "screen_features",
    "segmented_cutoff",
    "nogueira_stability",
    "filter_features_by_type",
    "information_decomposition",
]


def _resolve_transformer(model):
    """Return an object that can transform and decode I-ROCKET features."""
    if hasattr(model, "transformer_"):
        transformer = model.transformer_
    else:
        transformer = model
    if not hasattr(transformer, "transform"):
        raise TypeError(
            "model must be a fitted InterpRocket, InterpRocketTransform, "
            "or StableRocketClassifier."
        )
    if not hasattr(transformer, "decode_feature_index"):
        raise TypeError("model does not expose I-ROCKET feature metadata.")
    return transformer


def _validate_feature_indices(feature_indices, n_features):
    indices = np.asarray(feature_indices)
    if indices.ndim != 1:
        raise ValueError("feature_indices must be one-dimensional.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("feature_indices must contain integers.")
    indices = indices.astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError("feature_indices must contain at least one feature.")
    if np.any(indices < 0) or np.any(indices >= n_features):
        raise ValueError("feature_indices contains an out-of-range index.")
    if np.unique(indices).size != indices.size:
        raise ValueError("feature_indices must not contain duplicates.")
    return indices


def filter_features_by_type(
    model,
    feature_indices,
    *,
    pooling=None,
    representation=None,
):
    """Filter decoded I-ROCKET features by pooling operator or representation.

    The input order is preserved, so filtering a consensus ranking preserves
    that ranking.  This function does not refit a classifier or selector.

    Parameters
    ----------
    model : fitted I-ROCKET transformer or classifier
        ``InterpRocketTransform``, ``InterpRocket``, or a fitted
        ``StableRocketClassifier``.
    feature_indices : array-like of int
        Full-transform feature indices.
    pooling : str or iterable of str, optional
        Any subset of ``PPV``, ``MPV``, ``MIPV``, and ``LSPV``.
    representation : str or iterable of str, optional
        Any subset of ``raw`` and ``diff``.

    Returns
    -------
    ndarray of int
        Matching feature indices in their original order.
    """
    transformer = _resolve_transformer(model)
    n_features = int(getattr(transformer, "n_output_features_", 0))
    if n_features <= 0:
        raise ValueError("The I-ROCKET transformer has not been fitted.")
    indices = _validate_feature_indices(feature_indices, n_features)

    if isinstance(pooling, str):
        pooling = {pooling}
    elif pooling is not None:
        pooling = set(pooling)
    if isinstance(representation, str):
        representation = {representation}
    elif representation is not None:
        representation = set(representation)

    valid_pooling = {"PPV", "MPV", "MIPV", "LSPV"}
    valid_representations = {"raw", "diff"}
    if pooling is not None and not pooling <= valid_pooling:
        raise ValueError(
            "pooling may contain only PPV, MPV, MIPV, and LSPV."
        )
    if representation is not None and not representation <= valid_representations:
        raise ValueError("representation may contain only raw and diff.")

    kept = []
    for feature_index in indices:
        info = transformer.decode_feature_index(int(feature_index))
        if pooling is not None and info["pooling_op"] not in pooling:
            continue
        if (
            representation is not None
            and info["representation"] not in representation
        ):
            continue
        kept.append(int(feature_index))
    return np.asarray(kept, dtype=np.int64)


def _build_feature_groups(transformer, feature_indices, group_by):
    """Return labels and subset-relative column indices for decomposition."""
    labels = []
    columns = []

    if group_by == "individual":
        for subset_column, feature_index in enumerate(feature_indices):
            info = transformer.decode_feature_index(int(feature_index))
            labels.append(
                f"F{feature_index}:K{info['kernel_index']}_"
                f"d{info['dilation']}_{info['pooling_op']}_"
                f"{info['representation']}"
            )
            columns.append(np.asarray([subset_column], dtype=np.int64))
        return {"labels": labels, "indices": columns}

    grouped = {}
    for subset_column, feature_index in enumerate(feature_indices):
        info = transformer.decode_feature_index(int(feature_index))
        if group_by == "kernel":
            key = (int(info["kernel_index"]),)
        elif group_by == "kernel_dilation":
            key = (
                int(info["kernel_index"]),
                int(info["dilation"]),
                str(info["representation"]),
            )
        else:
            raise ValueError(
                "group_by must be 'individual', 'kernel', or "
                "'kernel_dilation'."
            )
        grouped.setdefault(key, []).append(subset_column)

    for key in sorted(grouped):
        if group_by == "kernel":
            labels.append(f"Kernel {key[0]}")
        else:
            labels.append(f"K{key[0]}_d{key[1]}_{key[2]}")
        columns.append(np.asarray(grouped[key], dtype=np.int64))
    return {"labels": labels, "indices": columns}


def information_decomposition(
    model,
    X,
    y,
    feature_mask=None,
    group_by="individual",
    n_shuffles=100,
    alpha_range=None,
    random_state=42,
    verbose=True,
):
    """Describe redundant, synergistic, and individual feature information.

    This is a post-hoc descriptive analysis of a fixed selected feature set. It
    fits and evaluates auxiliary ridge classifiers on the same supplied data,
    so its mutual-information values are *resubstitution diagnostics*, not
    estimates of generalization. Predictive performance must come from the
    leakage-free validation functions in :mod:`irocket_model_selection`.

    Parameters
    ----------
    model : fitted I-ROCKET transformer or classifier
        A fitted ``StableRocketClassifier`` is preferred. When ``feature_mask``
        is omitted, its consensus-selected feature indices are used.
    X : ndarray of shape (n_samples, n_timepoints)
    y : array-like of shape (n_samples,)
    feature_mask : array-like of int, optional
        Full-transform feature indices. Defaults to the fitted consensus set
        when available, otherwise all transformed columns.
    group_by : {'individual', 'kernel', 'kernel_dilation'}, default='individual'
        Unit of the decomposition. Individual features are the default because
        pooling operators and bias thresholds are distinct predictors even when
        they share a base kernel.
    n_shuffles : int, default=100
        Label permutations used to form a descriptive null scale.
    alpha_range : array-like, optional
        Ridge regularization values. Defaults to the classifier's configured
        values when available, otherwise ``logspace(-3, 3, 10)``.
    random_state : int, default=42
    verbose : bool, default=True

    Returns
    -------
    dict
        Group labels, mutual-information diagnostics, partial-information
        values, and the descriptive classification of each group.
    """
    transformer = _resolve_transformer(model)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional time-series matrix.")
    if y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("y must contain one label per row of X.")
    if np.unique(y).size < 2:
        raise ValueError("At least two classes are required.")
    if isinstance(n_shuffles, (bool, np.bool_)) or not isinstance(
        n_shuffles, (int, np.integer)
    ):
        raise TypeError("n_shuffles must be an integer.")
    if n_shuffles < 0:
        raise ValueError("n_shuffles must be nonnegative.")

    features_all = transformer.transform(X)
    if feature_mask is None:
        if hasattr(model, "selected_indices_"):
            feature_mask = np.asarray(model.selected_indices_, dtype=np.int64)
        else:
            feature_mask = np.arange(features_all.shape[1], dtype=np.int64)
    feature_mask = _validate_feature_indices(
        feature_mask, features_all.shape[1]
    )
    features = features_all[:, feature_mask]

    groups = _build_feature_groups(transformer, feature_mask, group_by)
    n_groups = len(groups["labels"])
    if n_groups == 0:
        raise ValueError("No feature groups were produced.")

    if alpha_range is None:
        if hasattr(model, "alpha_range") and model.alpha_range is not None:
            alpha_range = model.alpha_range
        elif hasattr(model, "alpha"):
            alpha_range = [float(model.alpha)]
        else:
            alpha_range = np.logspace(-3, 3, 10)
    alpha_range = np.asarray(alpha_range, dtype=float)
    if (
        alpha_range.ndim != 1
        or alpha_range.size == 0
        or not np.all(np.isfinite(alpha_range))
        or np.any(alpha_range <= 0)
    ):
        raise ValueError("alpha_range must contain positive finite values.")

    if verbose:
        print(
            f"Information decomposition: {features.shape[1]} features in "
            f"{n_groups} groups ({group_by})"
        )

    scaled = StandardScaler().fit_transform(features)

    def _fit_mi(matrix, labels):
        classifier = RidgeClassifierCV(alphas=alpha_range)
        classifier.fit(matrix, labels)
        predictions = classifier.predict(matrix)
        return mutual_information(y_true=labels, y_pred=predictions)

    I_ensemble = _fit_mi(scaled, y)
    I_single = np.zeros(n_groups, dtype=float)
    I_leave_one_out = np.zeros(n_groups, dtype=float)

    for group_index, group_columns in enumerate(groups["indices"]):
        I_single[group_index] = _fit_mi(scaled[:, group_columns], y)
        keep = np.ones(scaled.shape[1], dtype=bool)
        keep[group_columns] = False
        if np.any(keep):
            I_leave_one_out[group_index] = _fit_mi(scaled[:, keep], y)
        if verbose and (group_index + 1) % max(1, n_groups // 10) == 0:
            print(f"  Group {group_index + 1}/{n_groups}")

    I_contrib = I_ensemble - I_leave_one_out
    P_feature = I_contrib - I_single

    rng = np.random.default_rng(random_state)
    shuffle_mi = np.empty(n_shuffles, dtype=float)
    for shuffle_index in range(n_shuffles):
        shuffle_mi[shuffle_index] = _fit_mi(
            scaled, rng.permutation(y)
        )
    if n_shuffles:
        I_shuffle_mean = float(np.mean(shuffle_mi))
        I_shuffle_std = float(np.std(shuffle_mi))
        threshold = I_shuffle_mean + 2.0 * I_shuffle_std
    else:
        I_shuffle_mean = float("nan")
        I_shuffle_std = float("nan")
        threshold = 0.0

    classification = np.full(n_groups, "independent", dtype=object)
    classification[P_feature < -threshold] = "redundant"
    classification[P_feature > threshold] = "synergistic"

    if verbose:
        print(f"  Ensemble MI: {I_ensemble:.4f} bits")
        print(
            "  Groups: "
            f"{np.sum(classification == 'redundant')} redundant, "
            f"{np.sum(classification == 'synergistic')} synergistic, "
            f"{np.sum(classification == 'independent')} independent"
        )

    return {
        "feature_indices": feature_mask.copy(),
        "group_labels": groups["labels"],
        "group_indices": groups["indices"],
        "group_by": group_by,
        "I_ensemble": float(I_ensemble),
        "I_single": I_single,
        "I_leave_one_out": I_leave_one_out,
        "I_contrib": I_contrib,
        "P_feature": P_feature,
        "I_shuffle_mean": I_shuffle_mean,
        "I_shuffle_std": I_shuffle_std,
        "classification": classification,
        "n_redundant": int(np.sum(classification == "redundant")),
        "n_synergistic": int(np.sum(classification == "synergistic")),
        "n_independent": int(np.sum(classification == "independent")),
        "resubstitution": True,
    }
