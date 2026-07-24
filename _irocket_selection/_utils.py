"""Internal validation and numerical helpers for I-ROCKET selection."""

from __future__ import annotations

import numpy as np


def as_arrays(X, y=None):
    """Return validated dense arrays without forcing a float64 matrix copy."""
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_arr.shape}.")
    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature.")
    if not np.issubdtype(X_arr.dtype, np.number):
        raise ValueError("X must contain numeric values.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")

    if y is None:
        return X_arr

    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            "X and y have inconsistent sample counts: "
            f"{X_arr.shape[0]} and {y_arr.shape[0]}."
        )
    return X_arr, y_arr


def check_binary(y):
    """Return sorted binary classes or raise ``ValueError``."""
    classes = np.unique(y)
    if classes.size != 2:
        noun = "class" if classes.size == 1 else "classes"
        raise ValueError(
            f"Expected exactly 2 classes; got {classes.size} {noun}: {classes}."
        )
    return classes


def check_classes(y, min_classes=2):
    """Return sorted classes after checking the minimum class count."""
    classes = np.unique(y)
    if classes.size < min_classes:
        noun = "class" if classes.size == 1 else "classes"
        raise ValueError(
            f"Expected at least {min_classes} classes; got "
            f"{classes.size} {noun}."
        )
    return classes


def active_mask(X):
    """Return features with nonzero finite variance."""
    return np.var(X, axis=0, dtype=np.float64) > 0.0


def rank_scores(scores, absolute=True):
    """Return feature indices sorted from strongest to weakest."""
    values = np.abs(scores) if absolute else np.asarray(scores)
    # Stable sorting gives deterministic index order for exact ties.
    return np.argsort(-values, kind="mergesort")


def safe_top_k(top_k, n_features):
    """Normalize a requested top-k value."""
    if top_k is None:
        return int(n_features)
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(
        top_k, (int, np.integer)
    ):
        raise TypeError("top_k must be an integer or None.")
    if top_k <= 0:
        raise ValueError("top_k must be positive or None.")
    return min(int(top_k), int(n_features))
