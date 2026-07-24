"""Data-driven cutoffs for ranked I-ROCKET feature scores.

The primary method is a one-break segmented-regression cutoff.  Ranked
absolute scores are split into two regions, each fitted with an ordinary
least-squares line.  The breakpoint minimizing the combined residual sum of
squares is retained.  The implementation uses cumulative sums, so scanning
all possible breakpoints is linear in the number of features.
"""

from __future__ import annotations

import numpy as np

from .results import SegmentedCutoffResult


def _validate_ranking(ranked_indices, n_features):
    ranking = np.asarray(ranked_indices)
    if ranking.ndim != 1 or ranking.size != n_features:
        raise ValueError(
            "ranked_indices must be one-dimensional and contain one index "
            "per feature."
        )
    if not np.issubdtype(ranking.dtype, np.integer):
        raise TypeError("ranked_indices must contain integer feature indices.")
    ranking = ranking.astype(np.int64, copy=False)
    if np.any(ranking < 0) or np.any(ranking >= n_features):
        raise ValueError("ranked_indices contains an out-of-range index.")
    if np.unique(ranking).size != n_features:
        raise ValueError("ranked_indices must be a permutation of all features.")
    return ranking


def _line_fit_from_sums(count, sum_x, sum_y, sum_xx, sum_xy, sum_yy):
    """Return OLS intercept, slope, and SSE from vectorized sufficient sums."""
    count = np.asarray(count, dtype=np.float64)
    sum_x = np.asarray(sum_x, dtype=np.float64)
    sum_y = np.asarray(sum_y, dtype=np.float64)
    sum_xx = np.asarray(sum_xx, dtype=np.float64)
    sum_xy = np.asarray(sum_xy, dtype=np.float64)
    sum_yy = np.asarray(sum_yy, dtype=np.float64)

    centered_xx = sum_xx - (sum_x * sum_x) / count
    centered_xy = sum_xy - (sum_x * sum_y) / count
    centered_yy = sum_yy - (sum_y * sum_y) / count

    tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(sum_xx))
    nondegenerate = centered_xx > tolerance
    slope = np.zeros_like(centered_xx, dtype=np.float64)
    np.divide(centered_xy, centered_xx, out=slope, where=nondegenerate)
    intercept = (sum_y - slope * sum_x) / count

    reduction = np.zeros_like(centered_yy, dtype=np.float64)
    np.divide(
        centered_xy * centered_xy,
        centered_xx,
        out=reduction,
        where=nondegenerate,
    )
    sse = np.maximum(centered_yy - reduction, 0.0)
    return intercept, slope, sse


def segmented_cutoff(
    scores,
    *,
    ranked_indices=None,
    absolute=True,
    score_power=1.0,
    min_size=5,
):
    """Estimate one breakpoint in a ranked feature-score curve.

    Parameters
    ----------
    scores : array-like of shape (n_features,)
        Feature scores. Shrinkage-*t* scores are normally supplied here.
    ranked_indices : array-like of shape (n_features,), optional
        Complete strongest-to-weakest feature ranking. If omitted, the ranking
        is calculated from ``scores``.
    absolute : bool, default=True
        Rank and fit score magnitudes. This should remain ``True`` for signed
        shrinkage-*t* scores because their signs encode direction, not strength.
    score_power : float, default=1.0
        Positive power applied after taking the score magnitude. ``1.0`` uses
        absolute scores. ``2.0`` fits squared scores as a sensitivity analysis.
    min_size : int, default=5
        Minimum number of ranked features in each fitted segment.

    Returns
    -------
    SegmentedCutoffResult
        Selected indices and fit diagnostics. ``breakpoint`` equals the number
        selected; ``cutoff_idx`` is the last selected rank position.

    Notes
    -----
    Exactly one breakpoint is estimated. The two segments are ordinary
    least-squares lines with independently fitted intercepts and slopes; this is
    not the endpoint-constrained ``ruptures`` ``CostCLinear`` model. The result
    includes the proportional reduction in residual error and both slopes so
    the strength of the breakpoint can be assessed rather than assumed.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional.")
    if values.size == 0:
        raise ValueError("scores must contain at least one feature.")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values.")
    if not isinstance(absolute, (bool, np.bool_)):
        raise TypeError("absolute must be a boolean.")
    if isinstance(min_size, (bool, np.bool_)) or not isinstance(
        min_size, (int, np.integer)
    ):
        raise TypeError("min_size must be an integer.")
    min_size = int(min_size)
    if min_size < 2:
        raise ValueError("min_size must be at least 2.")
    if not np.isscalar(score_power) or isinstance(score_power, (bool, np.bool_)):
        raise TypeError("score_power must be a positive finite number.")
    score_power = float(score_power)
    if not np.isfinite(score_power) or score_power <= 0.0:
        raise ValueError("score_power must be a positive finite number.")

    n_features = int(values.size)
    if n_features < 2 * min_size:
        raise ValueError(
            "At least 2 * min_size features are required; got "
            f"{n_features} features and min_size={min_size}."
        )

    ranking_values = np.abs(values) if absolute else values
    if ranked_indices is None:
        ranking = np.argsort(-ranking_values, kind="mergesort")
    else:
        ranking = _validate_ranking(ranked_indices, n_features)

    ranked_scores = ranking_values[ranking]
    if absolute:
        ranked_scores = np.abs(ranked_scores)
    if score_power != 1.0:
        ranked_scores = np.power(ranked_scores, score_power)
    ranked_scores = np.asarray(ranked_scores, dtype=np.float64)

    spread = float(np.ptp(ranked_scores))
    scale = max(1.0, float(np.max(np.abs(ranked_scores))))
    if spread <= np.finfo(np.float64).eps * scale:
        raise ValueError(
            "The ranked score curve is constant; a breakpoint is undefined."
        )

    # Normalize rank positions only. Scaling y is unnecessary because it does
    # not change the least-squares breakpoint and would obscure diagnostics.
    x = np.linspace(0.0, 1.0, n_features, dtype=np.float64)
    y = ranked_scores

    cumulative_x = np.concatenate(([0.0], np.cumsum(x)))
    cumulative_y = np.concatenate(([0.0], np.cumsum(y)))
    cumulative_xx = np.concatenate(([0.0], np.cumsum(x * x)))
    cumulative_xy = np.concatenate(([0.0], np.cumsum(x * y)))
    cumulative_yy = np.concatenate(([0.0], np.cumsum(y * y)))

    candidates = np.arange(
        min_size, n_features - min_size + 1, dtype=np.int64
    )
    left_count = candidates.astype(np.float64)
    right_count = (n_features - candidates).astype(np.float64)

    left_intercept, left_slope, left_sse = _line_fit_from_sums(
        left_count,
        cumulative_x[candidates],
        cumulative_y[candidates],
        cumulative_xx[candidates],
        cumulative_xy[candidates],
        cumulative_yy[candidates],
    )
    right_intercept, right_slope, right_sse = _line_fit_from_sums(
        right_count,
        cumulative_x[-1] - cumulative_x[candidates],
        cumulative_y[-1] - cumulative_y[candidates],
        cumulative_xx[-1] - cumulative_xx[candidates],
        cumulative_xy[-1] - cumulative_xy[candidates],
        cumulative_yy[-1] - cumulative_yy[candidates],
    )

    total_sse = left_sse + right_sse
    best_position = int(np.argmin(total_sse))
    breakpoint = int(candidates[best_position])

    single_intercept, single_slope, single_sse_array = _line_fit_from_sums(
        float(n_features),
        cumulative_x[-1],
        cumulative_y[-1],
        cumulative_xx[-1],
        cumulative_xy[-1],
        cumulative_yy[-1],
    )
    del single_intercept, single_slope
    single_sse = float(np.asarray(single_sse_array))
    segmented_sse = float(total_sse[best_position])
    if single_sse <= np.finfo(np.float64).eps * max(1.0, np.sum(y * y)):
        relative_improvement = 0.0
    else:
        relative_improvement = float(
            np.clip((single_sse - segmented_sse) / single_sse, 0.0, 1.0)
        )

    selected = ranking[:breakpoint].copy()
    return SegmentedCutoffResult(
        selected=selected,
        ranking=ranking.copy(),
        ranked_scores=ranked_scores.copy(),
        breakpoint=breakpoint,
        cutoff_idx=breakpoint - 1,
        single_sse=single_sse,
        segmented_sse=segmented_sse,
        relative_improvement=relative_improvement,
        slope_before=float(left_slope[best_position]),
        slope_after=float(right_slope[best_position]),
        intercept_before=float(left_intercept[best_position]),
        intercept_after=float(right_intercept[best_position]),
        score_power=score_power,
        min_size=min_size,
    )
