"""
channel_selection.py -- Channel Selection for Multivariate Time Series

Implements classifier-agnostic channel selection based on class prototype
distances, following the approach of Dhariyal et al. (2023). Channels
where class prototypes are well-separated carry more discriminative
information and are retained; channels where classes overlap are discarded.

This serves as a preprocessing step before I-ROCKET classification.
For univariate time series, this module is not needed.

Usage:
    selected = select_channels(X_train, y_train, method='ecp')
    X_train_sel = X_train[:, selected, :]
    X_test_sel = X_test[:, selected, :]
    # Then flatten and pass to I-ROCKET

Reference:
    Dhariyal, B., Le Nguyen, T., and Ifrim, G. (2023).
    Scalable classifier-agnostic channel selection for multivariate
    time series classification. Data Mining and Knowledge Discovery,
    37:1010-1054.

License: BSD-3-Clause
"""

import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations


# ============================================================================
# CLASS PROTOTYPES
# ============================================================================

def compute_class_prototypes(X, y):
    """
    Compute the class prototype (mean time series) per channel.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_timepoints)
        Multivariate time series data.
    y : array-like, shape (n_samples,)
        Class labels.

    Returns
    -------
    prototypes : dict of class_label -> ndarray, shape (n_channels, n_timepoints)
        Mean time series per class.
    """
    classes = np.unique(y)
    prototypes = {}
    for cls in classes:
        prototypes[cls] = X[y == cls].mean(axis=0)
    return prototypes


# ============================================================================
# DISTANCE METRICS
# ============================================================================

def channel_distances_euclidean(prototypes):
    """
    Compute pairwise Euclidean distance between class prototypes per channel.

    Parameters
    ----------
    prototypes : dict of class_label -> ndarray, shape (n_channels, n_timepoints)

    Returns
    -------
    distances : ndarray, shape (n_channels,)
        Sum of pairwise Euclidean distances between class prototypes
        for each channel. Higher values indicate more discriminative channels.
    """
    classes = list(prototypes.keys())
    n_channels = prototypes[classes[0]].shape[0]

    distances = np.zeros(n_channels)
    for c1, c2 in combinations(classes, 2):
        for ch in range(n_channels):
            distances[ch] += euclidean(
                prototypes[c1][ch], prototypes[c2][ch]
            )

    return distances


def channel_distances_dtw(prototypes):
    """
    Compute pairwise DTW distance between class prototypes per channel.

    Falls back to Euclidean if dtw-python is not installed.

    Parameters
    ----------
    prototypes : dict of class_label -> ndarray, shape (n_channels, n_timepoints)

    Returns
    -------
    distances : ndarray, shape (n_channels,)
    """
    try:
        from scipy.spatial.distance import cdist

        def dtw_distance(a, b):
            """Simple DTW via dynamic programming."""
            n, m = len(a), len(b)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0.0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = (a[i - 1] - b[j - 1]) ** 2
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1],
                    )
            return np.sqrt(dtw_matrix[n, m])

    except ImportError:
        print("Warning: using Euclidean distance (dtw not available)")
        return channel_distances_euclidean(prototypes)

    classes = list(prototypes.keys())
    n_channels = prototypes[classes[0]].shape[0]

    distances = np.zeros(n_channels)
    for c1, c2 in combinations(classes, 2):
        for ch in range(n_channels):
            distances[ch] += dtw_distance(
                prototypes[c1][ch], prototypes[c2][ch]
            )

    return distances


# ============================================================================
# CHANNEL SELECTION METHODS
# ============================================================================

def _find_elbow(values):
    """
    Find the elbow point in a sorted (descending) sequence of values.

    Uses the maximum perpendicular distance from the line connecting
    the first and last points.

    Parameters
    ----------
    values : ndarray
        Sorted in descending order.

    Returns
    -------
    elbow_idx : int
        Index of the elbow point.
    """
    n = len(values)
    if n <= 2:
        return n - 1

    # Line from first to last point
    x = np.arange(n)
    p1 = np.array([0, values[0]])
    p2 = np.array([n - 1, values[-1]])

    # Perpendicular distance from each point to the line
    line_vec = p2 - p1
    line_len = np.sqrt(line_vec[0] ** 2 + line_vec[1] ** 2)

    if line_len == 0:
        return 0

    distances = np.abs(
        line_vec[1] * x - line_vec[0] * values
        + p2[0] * p1[1] - p2[1] * p1[0]
    ) / line_len

    return int(np.argmax(distances))


def select_channels_ecp(X, y, distance='euclidean', verbose=True):
    """
    Elbow Class Pairwise (ECP) channel selection.

    Ranks channels by the sum of pairwise distances between class
    prototypes, then selects channels above the elbow point.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_timepoints)
    y : array-like
    distance : str, 'euclidean' or 'dtw'
    verbose : bool

    Returns
    -------
    selected_channels : ndarray of int
        Indices of selected channels, sorted by discriminative power
        (most discriminative first).
    channel_scores : ndarray, shape (n_channels,)
        Distance score for each channel.
    """
    prototypes = compute_class_prototypes(X, y)

    if distance == 'dtw':
        scores = channel_distances_dtw(prototypes)
    else:
        scores = channel_distances_euclidean(prototypes)

    # Rank channels by score (descending)
    ranked = np.argsort(-scores)
    sorted_scores = scores[ranked]

    # Find elbow
    elbow = _find_elbow(sorted_scores)
    n_selected = elbow + 1

    selected = ranked[:n_selected]

    if verbose:
        print(f"Channel selection (ECP, {distance}):")
        print(f"  {len(scores)} channels, {n_selected} selected "
              f"({100 * n_selected / len(scores):.0f}%)")
        print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
        print(f"  Elbow at rank {elbow}, threshold score = "
              f"{sorted_scores[elbow]:.2f}")

    return selected, scores


def select_channels_topk(X, y, k=None, fraction=0.5,
                          distance='euclidean', verbose=True):
    """
    Top-k channel selection.

    Simpler alternative to ECP: retain the top k channels by
    prototype distance, or a fixed fraction.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_timepoints)
    y : array-like
    k : int, optional
        Number of channels to keep. Overrides fraction.
    fraction : float, default=0.5
        Fraction of channels to keep (if k is None).
    distance : str
    verbose : bool

    Returns
    -------
    selected_channels : ndarray of int
    channel_scores : ndarray, shape (n_channels,)
    """
    prototypes = compute_class_prototypes(X, y)

    if distance == 'dtw':
        scores = channel_distances_dtw(prototypes)
    else:
        scores = channel_distances_euclidean(prototypes)

    n_channels = len(scores)
    if k is None:
        k = max(1, int(fraction * n_channels))
    k = min(k, n_channels)

    ranked = np.argsort(-scores)
    selected = ranked[:k]

    if verbose:
        print(f"Channel selection (top-{k}, {distance}):")
        print(f"  {n_channels} channels, {k} selected "
              f"({100 * k / n_channels:.0f}%)")

    return selected, scores


def select_channels(X, y, method='ecp', **kwargs):
    """
    Select discriminative channels from multivariate time series.

    Convenience function that dispatches to ECP or top-k methods.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_timepoints)
    y : array-like
    method : str, 'ecp' or 'topk'
    **kwargs : passed to the selected method

    Returns
    -------
    selected_channels : ndarray of int
    channel_scores : ndarray, shape (n_channels,)
    """
    if method == 'ecp':
        return select_channels_ecp(X, y, **kwargs)
    elif method == 'topk':
        return select_channels_topk(X, y, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ecp' or 'topk'.")


# ============================================================================
# MULTIVARIATE -> UNIVARIATE FLATTENING
# ============================================================================

def flatten_channels(X, selected_channels=None):
    """
    Flatten selected channels into a single long univariate time series.

    This is the simplest approach for feeding multivariate data into
    I-ROCKET, which operates on univariate time series. Each sample's
    selected channels are concatenated end-to-end.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_timepoints)
    selected_channels : array-like of int, optional
        If None, uses all channels.

    Returns
    -------
    X_flat : ndarray, shape (n_samples, n_selected * n_timepoints)
    """
    if selected_channels is not None:
        X = X[:, selected_channels, :]
    n_samples, n_channels, n_timepoints = X.shape
    return X.reshape(n_samples, n_channels * n_timepoints)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_channel_scores(scores, selected=None, channel_names=None,
                        figsize=(10, 4)):
    """
    Visualize channel discriminative scores with selection threshold.

    Parameters
    ----------
    scores : ndarray, shape (n_channels,)
    selected : ndarray of int, optional
        Selected channel indices.
    channel_names : list of str, optional
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n_channels = len(scores)
    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(n_channels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    gridspec_kw={'width_ratios': [2, 1]})

    # Left: scores sorted by rank
    ranked = np.argsort(-scores)
    sorted_scores = scores[ranked]

    colors = ['#1f77b4' if i in (selected if selected is not None else [])
              else '#c7c7c7' for i in ranked]

    ax1.bar(range(n_channels), sorted_scores, color=colors,
            edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Channel rank')
    ax1.set_ylabel('Prototype distance')
    ax1.set_title('Channel scores (ranked)')
    ax1.grid(True, alpha=0.15, axis='y')

    if selected is not None:
        # Mark the elbow
        n_sel = len(selected)
        if n_sel < n_channels:
            ax1.axvline(n_sel - 0.5, color='#d62728', linestyle='--',
                         linewidth=1, label=f'Cutoff ({n_sel} channels)')
            ax1.legend(fontsize=8)

    # Right: original channel order with scores
    bar_colors = ['#1f77b4' if i in (selected if selected is not None else [])
                  else '#c7c7c7' for i in range(n_channels)]

    ax2.barh(range(n_channels), scores, color=bar_colors,
             edgecolor='white', height=0.7)
    ax2.set_yticks(range(n_channels))
    ax2.set_yticklabels(channel_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Prototype distance')
    ax2.set_title('By channel')
    ax2.grid(True, alpha=0.15, axis='x')

    plt.tight_layout()
    return fig


def plot_channel_prototypes(X, y, selected=None, n_show=None,
                             figsize=(12, None)):
    """
    Plot class prototype time series for each channel.

    Selected channels are highlighted; non-selected are grayed out.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_timepoints)
    y : array-like
    selected : ndarray of int, optional
    n_show : int, optional
        Max channels to show. Default: all.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    TAB10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    prototypes = compute_class_prototypes(X, y)
    classes = list(prototypes.keys())
    n_channels = X.shape[2] if X.ndim == 3 else 1
    n_channels = prototypes[classes[0]].shape[0]
    n_timepoints = prototypes[classes[0]].shape[1]

    if n_show is None:
        n_show = n_channels
    n_show = min(n_show, n_channels)

    if figsize[1] is None:
        figsize = (figsize[0], 1.8 * n_show)

    fig, axes = plt.subplots(n_show, 1, figsize=figsize, sharex=True)
    if n_show == 1:
        axes = [axes]

    t = np.arange(n_timepoints)

    for ch_idx in range(n_show):
        ax = axes[ch_idx]
        is_selected = selected is None or ch_idx in selected

        for k, cls in enumerate(classes):
            color = TAB10[k % len(TAB10)] if is_selected else '#c7c7c7'
            alpha = 1.0 if is_selected else 0.4
            ax.plot(t, prototypes[cls][ch_idx], color=color,
                    linewidth=1.2 if is_selected else 0.8,
                    alpha=alpha,
                    label=f'Class {cls}' if ch_idx == 0 else None)

        label = f'Ch {ch_idx}'
        if is_selected:
            label += ' *'
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.1)

        if not is_selected:
            ax.set_facecolor('#f8f8f8')

    axes[0].legend(fontsize=8, loc='upper right', ncol=len(classes))
    axes[-1].set_xlabel('Timepoint')
    fig.suptitle('Class prototypes per channel (* = selected)',
                  fontsize=12, y=0.95)
    plt.tight_layout()
    return fig
