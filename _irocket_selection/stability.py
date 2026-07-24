"""Feature-selection stability on a fixed feature universe.

This module implements the chance-corrected stability point estimate of
Nogueira, Sechidis, and Brown (2018).  It intentionally does not expose the
paper's asymptotic confidence intervals or hypothesis tests because repeated,
overlapping subsamples from one dataset should not be described as independent
population draws without additional justification.

Reference
---------
Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature
selection algorithms. Journal of Machine Learning Research, 18(174), 1-54.
"""

from __future__ import annotations

import numpy as np

from .results import NogueiraStabilityResult


def _check_selection_matrix(selection_matrix):
    matrix = np.asarray(selection_matrix)
    if matrix.ndim != 2:
        raise ValueError(
            "selection_matrix must have shape (n_resamples, n_features)."
        )
    n_resamples, n_features = matrix.shape
    if n_resamples < 2:
        raise ValueError("At least two resampled feature sets are required.")
    if n_features < 1:
        raise ValueError("The feature universe must contain at least one feature.")
    if not np.issubdtype(matrix.dtype, np.number) and matrix.dtype != np.bool_:
        raise ValueError("selection_matrix must contain binary numeric values.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("selection_matrix contains NaN or infinite values.")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("selection_matrix must contain only zeros and ones.")
    return matrix.astype(np.float64, copy=False)


def nogueira_stability(selection_matrix):
    """Calculate Nogueira feature-selection stability.

    Parameters
    ----------
    selection_matrix : array-like of shape (n_resamples, n_features)
        Binary matrix on one fixed feature universe. A value of one indicates
        that the feature was selected in that resample.

    Returns
    -------
    NogueiraStabilityResult
        Chance-corrected stability, feature selection probabilities, and
        selected-set sizes.

    Notes
    -----
    The measure is undefined if the average selected-set size is zero or equals
    the complete feature universe. Finite-sample estimates can be negative when
    a selector is less reproducible than chance.
    """
    matrix = _check_selection_matrix(selection_matrix)
    n_resamples, n_features = matrix.shape
    selection_probabilities = matrix.mean(axis=0)
    selected_counts = matrix.sum(axis=1)
    mean_selected = float(selected_counts.mean())

    selection_fraction = mean_selected / n_features
    random_variance = selection_fraction * (1.0 - selection_fraction)
    if random_variance <= 0.0:
        raise ValueError(
            "Nogueira stability is undefined when the mean selected-set size "
            "is zero or equals the complete feature universe."
        )

    observed_variance = (
        n_resamples
        / (n_resamples - 1.0)
        * np.mean(
            selection_probabilities * (1.0 - selection_probabilities)
        )
    )
    stability = 1.0 - observed_variance / random_variance

    return NogueiraStabilityResult(
        stability=float(stability),
        selection_probabilities=selection_probabilities.copy(),
        selected_counts=selected_counts.astype(np.int64),
        mean_selected=mean_selected,
        n_resamples=int(n_resamples),
        n_features=int(n_features),
        random_selection_variance=float(random_variance),
    )
