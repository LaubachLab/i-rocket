"""Shrinkage-*t* feature scores used inside I-ROCKET.

The implementation follows Opgen-Rhein and Strimmer (2007) and operates on an
existing feature matrix.  Binary labels use a two-class shrinkage-*t* score;
multiclass labels use one-versus-rest scoring with a documented aggregation.

References
----------
Opgen-Rhein, R., & Strimmer, K. (2007). Accurate ranking of differentially
    expressed genes by a distribution-free shrinkage approach. Statistical
    Applications in Genetics and Molecular Biology, 6(1), Article 9.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._utils import (
    active_mask,
    as_arrays,
    check_binary,
    check_classes,
    rank_scores,
    safe_top_k,
)
from .results import ScoreResult, SelectionResult


def _pooled_stats(X, y, classes, active_indices, chunk_size=32):
    """Compute class statistics without copying the full feature matrix.

    Raw moments are accumulated in float64 over small row chunks.  This avoids
    the otherwise substantial temporary float64 copy of a large MultiRocket
    feature matrix while reproducing the pooled variance and fourth central
    moment used by the shrinkage estimator.
    """
    masks = [y == class_label for class_label in classes]
    class_counts = [int(mask.sum()) for mask in masks]
    if min(class_counts) < 2:
        raise ValueError(
            "Each binary class must have at least 2 samples, got "
            f"{class_counts}."
        )

    n_samples = X.shape[0]
    n_active = len(active_indices)
    class_sums = np.zeros((2, n_active), dtype=np.float64)
    class_sums_sq = np.zeros((2, n_active), dtype=np.float64)
    raw_moment_1 = np.zeros(n_active, dtype=np.float64)
    raw_moment_2 = np.zeros(n_active, dtype=np.float64)
    raw_moment_3 = np.zeros(n_active, dtype=np.float64)
    raw_moment_4 = np.zeros(n_active, dtype=np.float64)

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        block = np.asarray(
            X[start:stop, active_indices], dtype=np.float64
        )
        block_sq = block * block
        raw_moment_1 += block.sum(axis=0)
        raw_moment_2 += block_sq.sum(axis=0)
        raw_moment_3 += (block_sq * block).sum(axis=0)
        raw_moment_4 += (block_sq * block_sq).sum(axis=0)

        for class_index, mask in enumerate(masks):
            local_mask = mask[start:stop]
            if np.any(local_mask):
                class_block = block[local_mask]
                class_sums[class_index] += class_block.sum(axis=0)
                class_sums_sq[class_index] += (
                    class_block * class_block
                ).sum(axis=0)

    class_means = class_sums / np.asarray(class_counts)[:, None]
    class_variances = np.empty_like(class_means)
    for class_index, count in enumerate(class_counts):
        numerator = (
            class_sums_sq[class_index]
            - class_sums[class_index] ** 2 / count
        )
        class_variances[class_index] = np.maximum(
            numerator / (count - 1), 0.0
        )

    pooled_variance = (
        (class_counts[0] - 1) * class_variances[0]
        + (class_counts[1] - 1) * class_variances[1]
    ) / (n_samples - 2)

    mean = raw_moment_1 / n_samples
    moment_2 = raw_moment_2 / n_samples
    moment_3 = raw_moment_3 / n_samples
    moment_4 = raw_moment_4 / n_samples
    fourth_central_moment = (
        moment_4
        - 4.0 * mean * moment_3
        + 6.0 * mean**2 * moment_2
        - 3.0 * mean**4
    )
    fourth_central_moment = np.maximum(fourth_central_moment, 0.0)

    mean_difference = class_means[1] - class_means[0]
    return (
        mean_difference,
        pooled_variance,
        fourth_central_moment,
        class_counts,
    )


def _analytical_lambda_var(pooled_var, fourth_moments, n_samples):
    variance_of_variance = (1.0 / n_samples) * (
        fourth_moments
        - pooled_var**2 * (n_samples - 3) / (n_samples - 1)
    )
    variance_of_variance = np.maximum(variance_of_variance, 0.0)
    target = float(np.median(pooled_var))
    numerator = float(np.sum(variance_of_variance))
    denominator = float(np.sum((pooled_var - target) ** 2))
    if denominator < 1e-12:
        return 0.0, target
    return float(np.clip(numerator / denominator, 0.0, 1.0)), target


def shrinkage_t(X, y, *, verbose=False):
    """Compute binary shrinkage-t scores for every feature.

    Positive scores indicate a larger mean in the second sorted class.  The
    ranking is based on descending absolute score.
    """
    X, y = as_arrays(X, y)
    classes = check_binary(y)
    n_features = X.shape[1]
    active = active_mask(X)
    active_indices = np.flatnonzero(active)

    scores = np.zeros(n_features, dtype=np.float64)
    if active_indices.size == 0:
        return ScoreResult(
            scores=scores,
            ranking=np.arange(n_features, dtype=np.int64),
            method="shrinkage_t",
            metadata={
                "classes": classes,
                "lambda_var": 0.0,
                "target_variance": 0.0,
                "active_mask": active,
                "class_counts": {
                    label: int(np.sum(y == label))
                    for label in classes.tolist()
                },
            },
        )

    mean_difference, pooled_var, fourth_moments, counts = _pooled_stats(
        X, y, classes, active_indices
    )
    n_samples = int(sum(counts))
    lambda_var, target = _analytical_lambda_var(
        pooled_var, fourth_moments, n_samples
    )
    shrunk_variance = (
        (1.0 - lambda_var) * pooled_var + lambda_var * target
    )
    shrunk_variance = np.maximum(shrunk_variance, 1e-12)
    standard_error = np.sqrt(
        shrunk_variance * (1.0 / counts[0] + 1.0 / counts[1])
    )
    scores[active_indices] = mean_difference / standard_error
    ranking = rank_scores(scores, absolute=True)

    if verbose:
        print(
            "shrinkage_t: "
            f"lambda_var={lambda_var:.4f}, target={target:.4e}, "
            f"active={active_indices.size}/{n_features}"
        )

    return ScoreResult(
        scores=scores,
        ranking=ranking,
        method="shrinkage_t",
        metadata={
            "classes": classes,
            "lambda_var": lambda_var,
            "target_variance": target,
            "active_mask": active,
            "class_counts": dict(zip(classes.tolist(), counts)),
        },
    )


def shrinkage_t_ovr(X, y, *, aggregate="max_abs", verbose=False):
    """Compute one-vs-rest shrinkage-t scores for multiclass labels."""
    X, y = as_arrays(X, y)
    classes = check_classes(y)
    if classes.size == 2:
        return shrinkage_t(X, y, verbose=verbose)

    scores_by_class = []
    lambda_vars = []
    for class_label in classes:
        result = shrinkage_t(
            X, (y == class_label).astype(np.int8), verbose=False
        )
        scores_by_class.append(result.scores)
        lambda_vars.append(result.metadata.get("lambda_var"))
    scores_by_class = np.asarray(scores_by_class, dtype=np.float64)

    if aggregate == "max_abs":
        scores = np.max(np.abs(scores_by_class), axis=0)
    elif aggregate == "l2":
        scores = np.sqrt(np.sum(scores_by_class**2, axis=0)) / np.sqrt(
            classes.size
        )
    else:
        raise ValueError("aggregate must be 'max_abs' or 'l2'.")

    if verbose:
        print(
            f"shrinkage_t_ovr: K={classes.size}, aggregate={aggregate}"
        )
    return ScoreResult(
        scores=scores,
        ranking=rank_scores(scores, absolute=False),
        method=f"shrinkage_t_ovr:{aggregate}",
        metadata={
            "classes": classes,
            "t_scores_ovr": scores_by_class,
            "lambda_var_per_class": lambda_vars,
        },
    )


def screen_features(
    X,
    y,
    *,
    top_k: Optional[int] = 500,
    multiclass="auto",
    aggregate: Optional[str] = None,
    verbose=False,
):
    """Rank and optionally retain features using shrinkage-*t* scores.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    top_k : int or None, default=500
        Number of leading features to retain. ``None`` retains the complete
        ranking while still returning the score vector.
    multiclass : {'auto', 'binary', 'ovr'}, default='auto'
        ``auto`` uses one-versus-rest scoring when more than two classes are
        present. ``binary`` requires exactly two classes.
    aggregate : {'max_abs', 'l2'} or None, default=None
        Multiclass score aggregation. ``None`` uses ``max_abs``.
    verbose : bool, default=False

    Returns
    -------
    SelectionResult
        Selected indices, complete ranking, scores, and method metadata.
    """
    X, y = as_arrays(X, y)
    classes = check_classes(y)
    if multiclass not in {"auto", "binary", "ovr"}:
        raise ValueError("multiclass must be 'auto', 'binary', or 'ovr'.")
    if multiclass == "binary" and classes.size != 2:
        raise ValueError("multiclass='binary' requires exactly two classes.")
    use_ovr = multiclass == "ovr" or (
        multiclass == "auto" and classes.size > 2
    )

    if use_ovr:
        score_result = shrinkage_t_ovr(
            X,
            y,
            aggregate=aggregate or "max_abs",
            verbose=verbose,
        )
    else:
        score_result = shrinkage_t(X, y, verbose=verbose)

    n_keep = safe_top_k(top_k, X.shape[1])
    selected = score_result.ranking[:n_keep]
    return SelectionResult(
        selected=selected,
        ranking=score_result.ranking,
        scores=score_result.scores,
        method=score_result.method,
        metadata=score_result.metadata,
    )
