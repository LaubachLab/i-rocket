"""Visualization and post-hoc interpretation for I-ROCKET.

The retained functions operate on a fitted ``InterpRocket`` or
``StableRocketClassifier`` and never refit the transform or selector.
Trial-level activation plots, class-mean kernel inspection, feature-correlation
summaries, composite traces, and quantitative localization diagnostics are
kept because they answer distinct questions about selected kernels.

Public functions
----------------
plot_activation_map
plot_kernel_pattern
plot_class_mean_activation
feature_trial_activation
plot_feature_trial_heatmap
composite_activation_trace
plot_activation_trace
plot_kernel_similarity
plot_feature_stability
localization_profile
localization_table
plot_localization_summary
plot_localization_diagnostic

Author
------
Mark Laubach, American University, Department of Neuroscience.

License
-------
BSD-3-Clause.
"""

import warnings

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_rgba

from interp_rocket import (
    compute_activation_map,
    OI,
    POOLING_COLORS,
    _pool_convolution,
    _validate_feature_index_array,
    _format_feature_label,
)

__all__ = [
    "plot_activation_map",
    "plot_kernel_pattern",
    "plot_class_mean_activation",
    "feature_trial_activation",
    "plot_feature_trial_heatmap",
    "composite_activation_trace",
    "plot_activation_trace",
    "plot_kernel_similarity",
    "plot_feature_stability",
    "localization_profile",
    "localization_table",
    "plot_localization_summary",
    "plot_localization_diagnostic",
]

# Weight tick colors for kernel pattern barcodes.
WEIGHT_COLORS = {2: "black", -1: "gray"}

# Activation-map background and fire alpha.
_BG_COLOR = np.array([240, 240, 240, 255], dtype=np.float32) / 255.0
_FIRE_ALPHA = 0.7


def _full_feature_matrix(model, X):
    """Transform X into the complete fitted I-ROCKET feature universe."""
    if hasattr(model, "transformer_"):
        return model.transformer_.transform(X)
    if hasattr(model, "transform"):
        return model.transform(X)
    raise TypeError(
        "model must be a fitted InterpRocket or StableRocketClassifier."
    )


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------

def _resolve_time_axis(n, time_vector=None, sampling_rate=None,
                       t_start=0.0, t_label=None):
    """Build the per-sample time axis and x-axis label."""
    if time_vector is not None:
        t = np.asarray(time_vector, dtype=float)
        if len(t) != n:
            raise ValueError(
                f"time_vector length {len(t)} != n_timepoints {n}"
            )
        label = "Time" if t_label is None else t_label
    elif sampling_rate is not None:
        t = np.arange(n, dtype=float) / float(sampling_rate) + float(t_start)
        label = "Time (s)" if t_label is None else t_label
    else:
        t = np.arange(n, dtype=float)
        label = "Timepoint (samples)" if t_label is None else t_label
    return t, label


def _samples_to_time(samples, t_axis):
    """
    Map (possibly fractional) sample indices to time values via linear
    interpolation against the integer-sample time axis.
    """
    s = np.asarray(samples, dtype=float)
    n = len(t_axis)
    s_clip = np.clip(s, 0, n - 1)
    i_lo = np.floor(s_clip).astype(int)
    i_hi = np.minimum(i_lo + 1, n - 1)
    frac = s_clip - i_lo
    return t_axis[i_lo] * (1.0 - frac) + t_axis[i_hi] * frac


def _full_classifier_coefficients(model):
    """Return classifier coefficients indexed by full transform columns."""
    if hasattr(model, "get_full_classifier_coefficients"):
        return np.asarray(model.get_full_classifier_coefficients())
    if not hasattr(model, "classifier_"):
        raise TypeError("model must expose a fitted linear classifier.")
    return np.asarray(model.classifier_.coef_)


def _resolve_feature_order(model, *, feature_mask=None, rank_order=None):
    """Return a validated display order in the full transformed universe."""
    n_features = int(getattr(model, "n_output_features_", 0))
    if n_features <= 0:
        raise ValueError("model must be fitted before plotting features.")
    if rank_order is not None:
        ranking = rank_order
    elif feature_mask is not None:
        ranking = feature_mask
    else:
        # Keep the complete eligible ranking available so display filters can
        # skip ineligible rows and backfill from lower-ranked features before
        # applying n_show. StableRocketClassifier is restricted to its
        # consensus-selected set; InterpRocket ranks its full transform.
        selected = getattr(model, "selected_indices_", None)
        if selected is not None and len(selected) > 0:
            candidates = _validate_feature_index_array(
                selected,
                n_features,
                name="selected_indices_",
            )
        else:
            candidates = np.arange(n_features, dtype=np.int64)

        if hasattr(model, "get_feature_importance"):
            importance = np.asarray(
                model.get_feature_importance(feature_mask=candidates),
                dtype=float,
            )
            if importance.ndim != 1 or importance.size != n_features:
                raise ValueError(
                    "model.get_feature_importance() returned an invalid shape."
                )
            order = np.argsort(
                importance[candidates],
                kind="stable",
            )[::-1]
            ranking = candidates[order]
        else:
            top = model.get_top_features(n=len(candidates))
            ranking = [item["feature_index"] for item in top]
    return _validate_feature_index_array(
        ranking, n_features, name="feature ordering"
    )


def _validate_n_show(n_show):
    if isinstance(n_show, (bool, np.bool_)) or not isinstance(
        n_show, (int, np.integer)
    ):
        raise TypeError("n_show must be an integer.")
    if n_show < 1:
        raise ValueError("n_show must be positive.")
    return int(n_show)



def _require_pandas():
    """Import pandas only for public table-producing helpers."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "This table-producing function requires pandas. Install pandas "
            "or install I-ROCKET with its notebook dependencies."
        ) from exc
    return pd


def _resolve_single_feature(
    model,
    *,
    feature_index=None,
    feature_rank=0,
    feature_mask=None,
    rank_order=None,
):
    """Resolve one full transformed-feature index and its optional rank."""
    n_features = int(getattr(model, "n_output_features_", 0))
    if n_features <= 0:
        raise ValueError("model must be fitted before selecting a feature.")

    if feature_index is not None:
        if isinstance(feature_index, (bool, np.bool_)) or not isinstance(
            feature_index, (int, np.integer)
        ):
            raise TypeError("feature_index must be an integer or None.")
        feature_index = int(feature_index)
        if feature_index < 0 or feature_index >= n_features:
            raise ValueError(
                f"feature_index must be in [0, {n_features}); "
                f"got {feature_index}."
            )
        return feature_index, None

    ranking = _resolve_feature_order(
        model,
        feature_mask=feature_mask,
        rank_order=rank_order,
    )
    if isinstance(feature_rank, (bool, np.bool_)) or not isinstance(
        feature_rank, (int, np.integer)
    ):
        raise TypeError("feature_rank must be an integer.")
    feature_rank = int(feature_rank)
    if feature_rank < 0 or feature_rank >= len(ranking):
        raise ValueError(
            f"feature_rank={feature_rank} but only {len(ranking)} "
            "features are available."
        )
    return int(ranking[feature_rank]), feature_rank


def _longest_positive_run(binary_values):
    """Return inclusive bounds and length of the longest True run."""
    positive = np.asarray(binary_values, dtype=bool)
    if positive.ndim != 1:
        raise ValueError("binary_values must be one-dimensional.")
    if positive.size == 0 or not np.any(positive):
        return -1, -1, 0

    padded = np.concatenate(([False], positive, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    lengths = stops - starts + 1
    best = int(np.argmax(lengths))
    return int(starts[best]), int(stops[best]), int(lengths[best])


def _validate_trial_order(trial_order, n_trials):
    """Validate a complete permutation of trial indices."""
    order = np.asarray(trial_order)
    if order.ndim != 1 or not np.issubdtype(order.dtype, np.integer):
        raise ValueError("trial_order must be a one-dimensional integer array.")
    order = order.astype(np.int64, copy=False)
    if order.size != n_trials:
        raise ValueError(
            "trial_order must contain exactly one entry per trial."
        )
    if np.any(order < 0) or np.any(order >= n_trials):
        raise ValueError("trial_order contains an out-of-range trial index.")
    if np.unique(order).size != n_trials:
        raise ValueError("trial_order must be a permutation without duplicates.")
    return order


def _validate_filter_flag(value, name):
    """Validate a public boolean display-filter option."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def _resolve_localized_feature_set(model, localized_features):
    """Validate an optional externally supplied localized-feature set."""
    if localized_features is None:
        return None

    try:
        values = list(localized_features)
    except TypeError as exc:
        raise TypeError(
            "localized_features must be an iterable of integers."
        ) from exc

    n_features = int(getattr(model, "n_output_features_", 0))
    local_set = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("localized_features must contain integers.")
        index = int(value)
        if index < 0 or index >= n_features:
            raise ValueError(
                "localized_features contains an out-of-range feature index."
            )
        local_set.add(index)
    return local_set


def _class_mean_signals(X, y, classes):
    """Return float64 class-average waveforms in class order."""
    return [
        X[y == cls].mean(axis=0).astype(np.float64)
        for cls in classes
    ]


def _activation_profiles_on_class_means(class_means, feature, n_timepoints):
    """Map one decoded feature's class-mean activations to input time."""
    profiles = np.zeros(
        (len(class_means), n_timepoints),
        dtype=np.float64,
    )

    for class_index, class_mean in enumerate(class_means):
        x_use = (
            np.diff(class_mean).astype(np.float32)
            if feature["representation"] == "diff"
            else class_mean.astype(np.float32)
        )
        _, activation, time_indices = compute_activation_map(
            x_use,
            feature["kernel_index"],
            feature["dilation"],
            feature["bias"],
            feature["padding_mode"],
            feature["representation"],
        )
        for value, time_index in zip(activation, time_indices):
            center = int(round(time_index))
            if feature["representation"] == "diff":
                center = min(center + 1, n_timepoints - 1)
            if 0 <= center < n_timepoints:
                profiles[class_index, center] = max(
                    profiles[class_index, center],
                    float(value),
                )

    return profiles


def _localization_from_profiles(
    profiles,
    feature,
    n_timepoints,
    localization_frac,
):
    """Compute differential centroid and localization from class profiles."""
    diff_act = np.max(profiles, axis=0) - np.min(profiles, axis=0)
    total = float(diff_act.sum())

    if total > 0.0:
        t_grid = np.arange(n_timepoints, dtype=np.float64)
        peak_t = int(round(np.dot(t_grid, diff_act) / total))
        peak_t = max(0, min(peak_t, n_timepoints - 1))
        half = int(feature["receptive_field"]) // 2
        lo = max(0, peak_t - half)
        hi = min(n_timepoints, peak_t + half + 1)
        mass_in_rf = float(diff_act[lo:hi].sum() / total)
        is_local = mass_in_rf >= localization_frac
    else:
        peak_t = n_timepoints // 2
        is_local = False
        mass_in_rf = 0.0

    return (
        peak_t,
        is_local,
        feature["receptive_field"] / 2.0,
        mass_in_rf,
    )


def _select_display_features(
    model,
    ranking,
    n_show,
    *,
    X=None,
    y=None,
    localization_frac=0.5,
    localized_features=None,
    localized_only=False,
    suprathreshold_only=False,
):
    """Filter a ranked feature list, then apply n_show.

    ``suprathreshold`` means that the decoded kernel exceeds its fitted bias
    on at least one class-average waveform.  This is the same criterion used
    for the dagger-marked rows in the original activation map.
    """
    localized_only = _validate_filter_flag(localized_only, "localized_only")
    suprathreshold_only = _validate_filter_flag(
        suprathreshold_only,
        "suprathreshold_only",
    )
    requested = min(len(ranking), _validate_n_show(n_show))
    local_set = _resolve_localized_feature_set(model, localized_features)

    has_signal = X is not None and y is not None
    if suprathreshold_only and not has_signal:
        raise ValueError(
            "X and y are required when suprathreshold_only is True."
        )
    if localized_only and not has_signal and local_set is None:
        raise ValueError(
            "X and y, or an explicit localized_features set, are required "
            "when localized_only is True."
        )

    if has_signal:
        classes = np.unique(y)
        class_means = _class_mean_signals(X, y, classes)
        n_timepoints = int(X.shape[1])
    else:
        class_means = None
        n_timepoints = int(getattr(model, "n_features_in_", 0))

    selected = []
    for feature_index in ranking:
        feature_index = int(feature_index)

        # An external local set can reject a feature without computing its
        # class-mean activation maps.
        if (
            localized_only
            and local_set is not None
            and feature_index not in local_set
        ):
            continue

        feature = model.decode_feature_index(feature_index)

        if has_signal:
            profiles = _activation_profiles_on_class_means(
                class_means,
                feature,
                n_timepoints,
            )
            fires_on_mean = bool(np.any(profiles > 0.0))
            peak_t, is_local, rf_half, mass_in_rf = _localization_from_profiles(
                profiles,
                feature,
                n_timepoints,
                localization_frac,
            )
        else:
            profiles = None
            fires_on_mean = None
            peak_t = n_timepoints // 2
            is_local = False
            rf_half = feature["receptive_field"] / 2.0
            mass_in_rf = 0.0

        if local_set is not None:
            is_local = feature_index in local_set

        if suprathreshold_only and not fires_on_mean:
            continue
        if localized_only and not is_local:
            continue

        selected.append(
            {
                "feature_index": feature_index,
                "feature": feature,
                "profiles": profiles,
                "fires_on_mean": fires_on_mean,
                "peak_t": peak_t,
                "is_local": is_local,
                "rf_half": rf_half,
                "mass_in_rf": mass_in_rf,
            }
        )
        if len(selected) >= requested:
            break

    if not selected:
        filters = []
        if localized_only:
            filters.append("localized")
        if suprathreshold_only:
            filters.append("suprathreshold")
        requested_filter = " and ".join(filters) or "display"
        raise ValueError(
            f"No ranked features satisfy the requested {requested_filter} "
            "feature filter."
        )

    return selected


def _compute_differential_centroid(model, X, y, f, n_timepoints,
                                   localization_frac=0.5):
    """
    Compute the differential activation centroid and RF half-width for one
    feature.

    The centroid is the center of mass of the inter-class activation
    difference: sum(t * diff_act[t]) / sum(diff_act), where
    diff_act[t] = max_class_act[t] - min_class_act[t].

    This is a differential quantity. It does not belong to either class
    individually. The RF half-width is a fixed kernel property, not a
    statistical spread.

    A center of mass exists whenever the classes differ anywhere. It is
    only meaningful when the differential activation concentrates near it.
    A kernel can produce a well-defined centroid that lands in a quiet
    region between two separated activation lobes. The localization test
    measures the fraction of differential mass inside one receptive field
    window centered on the centroid. A kernel that fails the test is
    treated as not localized and is drawn as a full-width line with no
    centroid marker.

    Parameters
    ----------
    localization_frac : float, default=0.5
        Minimum fraction of differential activation mass that must fall
        within the receptive field window centered on the centroid for the
        centroid to be considered localized.

    Returns
    -------
    peak_t : int
        Centroid timepoint index.
    centroid_valid : bool
        True when the centroid is temporally localized.
    rf_half : float
        Half the kernel receptive field in timepoints.
    mass_in_rf : float
        Fraction of total differential activation mass that falls inside
        one receptive field window centered on the centroid. The quantity
        the localization test thresholds. Ranges from near the uniform-fire
        baseline (receptive field width over window length) for a global
        kernel to near 1.0 for a tightly localized kernel.
    """
    classes = np.unique(y)
    # Force float64 for the mean regardless of X dtype. Smooth signals can
    # lose precision when a float32 class mean is first-differenced.
    # float32 is only needed at the compute_activation_map call.
    class_means = _class_mean_signals(X, y, classes)
    profiles = _activation_profiles_on_class_means(
        class_means,
        f,
        n_timepoints,
    )
    return _localization_from_profiles(
        profiles,
        f,
        n_timepoints,
        localization_frac,
    )


def _build_rgba_frame(act_matrix, decoded, n_show, n_timepoints, fires_on_mean):
    """
    Build an RGBA image array colored by pooling operator type.

    Active timepoints in each row are colored by that row's pooling type.
    Inactive timepoints are set to the background color. Subthreshold rows
    are dimmed.

    Returns
    -------
    rgba : ndarray, shape (n_show, n_timepoints, 4)
    """
    rgba = np.tile(_BG_COLOR, (n_show, n_timepoints, 1))

    for row, f in enumerate(decoded):
        color_hex = POOLING_COLORS.get(f["pooling_op"], "#7f7f7f")
        r, g, b, _ = to_rgba(color_hex)
        active = act_matrix[row] > 0
        alpha = _FIRE_ALPHA if fires_on_mean[row] else _FIRE_ALPHA * 0.35
        rgba[row, active] = [r, g, b, alpha]
        if not fires_on_mean[row]:
            rgba[row, ~active] = [*_BG_COLOR[:3], 0.35]

    return rgba


# -------------------------------------------------------------------------
# Activation map
# -------------------------------------------------------------------------

def plot_activation_map(
    model,
    X,
    y,
    feature_mask=None,
    rank_order=None,
    n_show=15,
    figsize=(12, 8),
    time_vector=None,
    sampling_rate=None,
    t_start=0.0,
    t_label=None,
    class_names=None,
    rf_color="black",
    rf_alpha=0.85,
    rf_markersize=6,
    rf_linewidth=1.5,
    localization_frac=0.5,
    localized_features=None,
    *,
    localized_only=False,
    suprathreshold_only=False,
):
    """
    Activation heatmap colored by pooling type, with receptive field overlay.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
        Class labels.
    feature_mask : array-like of int, optional
        Subset of feature indices in display order.
    rank_order : array-like of int, optional
        Feature indices in display order. Overrides feature_mask.
    n_show : int, default=15
        Maximum number of features to display.
    figsize : tuple, default=(12, 8)
    time_vector : 1D array, optional
        Explicit per-sample time values. Takes precedence over sampling_rate.
    sampling_rate : float, optional
        Sampling rate in Hz.
    t_start : float, default=0.0
        Start time for the sampling_rate path.
    t_label : str, optional
        X-axis label override.
    class_names : list of str, optional
        Display names for each class panel. Defaults to 'Class {label}'.
    rf_color : str, default='black'
        Color for the RF centroid marker and span line.
    rf_alpha : float, default=0.85
        Opacity for the RF overlay.
    rf_markersize : float, default=6
        Size of the centroid circle marker.
    rf_linewidth : float, default=1.5
        Line width for the RF span.
    localization_frac : float, default=0.5
        Minimum fraction of differential activation mass within one
        receptive field of the centroid for the centroid to be drawn.
        Kernels below this threshold are not temporally localized. They
        get a full-width line and no centroid marker.
    localized_features : iterable of int, optional
        Feature indices to treat as localized, overriding the internal
        localization_frac test. When given, only features in this set draw
        a centroid and all others draw a full-width line. Use this to drive
        the plot from an external criterion, for example the local set from
        localization_profile under the excess-over-baseline rule.
    localized_only : bool, default=False
        If True, display only temporally localized features. The filter uses
        ``localized_features`` when supplied and otherwise uses the
        ``localization_frac`` criterion. Filtering occurs before ``n_show``,
        so lower-ranked eligible features backfill excluded rows.
    suprathreshold_only : bool, default=False
        If True, display only features whose kernel exceeds its fitted bias on
        at least one class-average waveform. This removes the blank,
        dagger-marked rows. Filtering occurs before ``n_show``. When both
        display filters are True, their intersection is shown.

    Returns
    -------
    fig : matplotlib Figure

    Notes
    -----
    Heatmap coloring
        Each row is colored by pooling operator: PPV (blue), MPV (sky blue),
        MIPV (vermillion), and LSPV (orange). Active timepoints carry the pooling
        color. Inactive timepoints show the background. Subthreshold rows
        (marked †) are dimmed.

    RF overlay
        The circle marks the centroid of the inter-class differential
        activation: the center of mass of timepoints where the two class
        means differ in activation magnitude. The line spans the kernel's
        receptive field centered on that point. Both quantities are computed
        from both classes together and are identical across panels. The line
        width is not a confidence interval. It reflects the kernel's temporal
        scale.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("X must be 2D and y must contain one label per row.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")

    classes = np.unique(y)
    n_classes = len(classes)
    n_timepoints = X.shape[1]

    # ---- Resolve ordering and filter before applying n_show ----
    ranking = _resolve_feature_order(
        model, feature_mask=feature_mask, rank_order=rank_order
    )
    selected = _select_display_features(
        model,
        ranking,
        n_show,
        X=X,
        y=y,
        localization_frac=localization_frac,
        localized_features=localized_features,
        localized_only=localized_only,
        suprathreshold_only=suprathreshold_only,
    )
    decoded = [item["feature"] for item in selected]
    fires_on_mean = [item["fires_on_mean"] for item in selected]
    n_show = len(selected)

    if class_names is None:
        class_names = [f"Class {cls}" for cls in classes]
    if len(class_names) < n_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but there are "
            f"{n_classes} classes"
        )

    t_signal, xlabel = _resolve_time_axis(
        n_timepoints, time_vector=time_vector,
        sampling_rate=sampling_rate, t_start=t_start, t_label=t_label,
    )
    dt = (t_signal[-1] - t_signal[0]) / (n_timepoints - 1)

    if figsize is None:
        figsize = (5 * n_classes, 0.45 * n_show + 1.8)

    # ---- RF overlays computed during feature filtering ----
    peak_times = [t_signal[item["peak_t"]] for item in selected]
    rf_spans = [item["rf_half"] * dt for item in selected]
    centroid_valids = [item["is_local"] for item in selected]

    # ---- Build figure ----
    fig, axes = plt.subplots(1, n_classes, figsize=figsize, sharey=True)
    if n_classes == 1:
        axes = [axes]

    x_lo, x_hi = t_signal[0], t_signal[-1]

    for k, cls in enumerate(classes):
        act_matrix = np.asarray(
            [item["profiles"][k] for item in selected],
            dtype=np.float64,
        )

        # Build per-row colored RGBA image
        rgba = _build_rgba_frame(
            act_matrix, decoded, n_show, n_timepoints, fires_on_mean
        )

        ax = axes[k]
        ax.imshow(
            rgba,
            aspect="auto",
            interpolation="nearest",
            extent=(x_lo, x_hi, n_show - 0.5, -0.5),
        )

        # ---- RF overlay ----
        for row_idx in range(n_show):
            pt = peak_times[row_idx]
            half = rf_spans[row_idx]
            valid = centroid_valids[row_idx]

            if valid:
                left = max(x_lo, pt - half)
                right = min(x_hi, pt + half)
            else:
                left = x_lo
                right = x_hi

            ax.plot(
                [left, right], [row_idx, row_idx],
                color=rf_color, alpha=rf_alpha,
                linewidth=rf_linewidth,
                solid_capstyle="butt",
                zorder=3,
            )
            if valid:
                ax.plot(
                    pt, row_idx,
                    marker="o", color=rf_color,
                    markersize=rf_markersize,
                    alpha=rf_alpha,
                    linestyle="none",
                    zorder=4,
                )

        ax.set_xlabel(xlabel)
        ax.set_title(class_names[k])
        ax.set_xlim(x_lo, x_hi)

        if k == 0:
            labels = []
            for row_idx, f in enumerate(decoded):
                tag = "" if fires_on_mean[row_idx] else " †"
                labels.append(
                    f"{_format_feature_label(f, compact=True)}{tag}"
                )
            ax.set_yticks(range(n_show))
            ax.set_yticklabels(labels, fontsize=8)
            for row_idx, lbl in enumerate(ax.get_yticklabels()):
                if not fires_on_mean[row_idx]:
                    lbl.set_alpha(0.4)

    # ---- Legend ----
    pooling_ops_shown = {f["pooling_op"] for f in decoded}
    legend_patches = [
        mpatches.Patch(color=POOLING_COLORS[op], alpha=_FIRE_ALPHA, label=op)
        for op in ["PPV", "MPV", "MIPV", "LSPV"]
        if op in pooling_ops_shown
    ]
    axes[-1].legend(
        handles=legend_patches,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        title="Pooling",
        title_fontsize=8,
        framealpha=0.9,
    )

    title_parts = [
        "Circle = differential activation centroid (shown only when temporally localized)",
        "Line = kernel receptive field",
    ]
    if not suprathreshold_only and not all(fires_on_mean):
        title_parts.append(
            "† = subthreshold on class means "
            "(fires on some individual trials only)"
        )
    filters = []
    if localized_only:
        filters.append("localized")
    if suprathreshold_only:
        filters.append("suprathreshold")
    if filters:
        title_parts.append(f"Displayed: {' and '.join(filters)} features only")
    fig.suptitle("  |  ".join(title_parts), fontsize=9, y=1.02)
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Kernel weight patterns
# -------------------------------------------------------------------------

def plot_kernel_pattern(
    model,
    X_test=None,
    y_test=None,
    feature_mask=None,
    rank_order=None,
    rank_label=None,
    n_show=15,
    time_vector=None,
    sampling_rate=None,
    t_start=0.0,
    t_label=None,
    event_time=None,
    colors=None,
    class_names=None,
    tick_height=0.35,
    tick_width=3,
    localization_frac=0.5,
    localized_features=None,
    figsize=(12, 8),
    *,
    localized_only=True,
    suprathreshold_only=False,
):
    """
    Plot kernel weight patterns at their dilated temporal positions.

    Top panel: class means (when X_test and y_test are provided).
    Bottom panel: each row shows one kernel's weight pattern as a barcode
    of colored ticks (+2 = black, -1 = gray) at the dilated sample positions,
    centered on the differential activation centroid. Kernels that are not
    temporally localized are drawn as a full-width span with no ticks.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Fitted model.
    X_test, y_test : optional
        Data for the class-means panel and centroid computation.
    feature_mask : array of int, optional
        Feature indices. Treated as already ordered when rank_order is None.
    rank_order : array of int, optional
        Feature indices in display order (row 0 = top).
    rank_label : str, optional
        Y-axis label for the ordering method.
    n_show : int, default=15
        Maximum number of features.
    time_vector : 1D array, optional
        Explicit per-sample time values.
    sampling_rate : float, optional
        Sampling rate in Hz.
    t_start : float, default=0.0
        Start time for the sampling_rate path.
    t_label : str, optional
        X-axis label override.
    event_time : float, optional
        Vertical dashed line at this time.
    colors : list of str, optional
        Per-class colors for the top panel.
    class_names : list of str, optional
        Per-class display names.
    tick_height : float, default=0.35
        Half-height of each weight tick in row units.
    tick_width : float, default=3
        Line width for weight ticks.
    localization_frac : float, default=0.5
        Minimum fraction of differential activation mass within one
        receptive field of the centroid for the kernel to be drawn with
        weight ticks. Kernels below this threshold are not temporally
        localized and are drawn as a full-width span only.
    localized_features : iterable of int, optional
        Feature indices to treat as localized, overriding the internal
        localization_frac test. When given, only features in this set draw
        weight ticks and all others draw a full-width span. Use this to
        drive the plot from an external criterion, for example the local
        set from localization_profile under the excess-over-baseline rule.
    figsize : tuple
    localized_only : bool, default=True
        If True, display only temporally localized features. The filter uses
        ``localized_features`` when supplied and otherwise uses the
        ``localization_frac`` criterion. Filtering occurs before ``n_show``,
        so lower-ranked eligible features backfill excluded rows.
    suprathreshold_only : bool, default=False
        If True, display only features whose kernel exceeds its fitted bias on
        at least one class-average waveform. Filtering occurs before
        ``n_show``. When both display filters are True, their intersection is
        shown.

    Returns
    -------
    fig : matplotlib Figure
    """
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must be supplied together.")
    if rank_label is None:
        rank_label = "Feature rank"

    # --- Time axis ---
    has_signal = X_test is not None
    if has_signal:
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test)
        if X_test.ndim != 2 or y_test.ndim != 1 or y_test.size != X_test.shape[0]:
            raise ValueError(
                "X_test must be 2D and y_test must contain one label per row."
            )
        if not np.all(np.isfinite(X_test)):
            raise ValueError("X_test must contain only finite values.")
        n_timepoints = int(X_test.shape[1])
    else:
        n_timepoints = int(getattr(model, "n_features_in_", 0))
        if n_timepoints <= 0:
            raise ValueError(
                "A fitted model with n_features_in_ is required when X_test "
                "and y_test are omitted."
            )

    # --- Resolve ordering and filter before applying n_show ---
    ranking = _resolve_feature_order(
        model, feature_mask=feature_mask, rank_order=rank_order
    )
    effective_localized_only = bool(localized_only)
    if (
        effective_localized_only
        and not has_signal
        and localized_features is None
    ):
        # Localization is a data-dependent property. Keep the historical
        # no-data kernel-pattern view usable while defaulting to localized
        # features whenever X_test and y_test are supplied.
        effective_localized_only = False

    selected = _select_display_features(
        model,
        ranking,
        n_show,
        X=X_test if has_signal else None,
        y=y_test if has_signal else None,
        localization_frac=localization_frac,
        localized_features=localized_features,
        localized_only=effective_localized_only,
        suprathreshold_only=suprathreshold_only,
    )
    decoded = [item["feature"] for item in selected]
    n_show = len(selected)

    t_axis, x_label = _resolve_time_axis(
        n_timepoints,
        time_vector=time_vector,
        sampling_rate=sampling_rate,
        t_start=t_start,
        t_label=t_label,
    )
    explicit_time = time_vector is not None or sampling_rate is not None

    if n_timepoints is not None:
        dt = (t_axis[-1] - t_axis[0]) / (n_timepoints - 1)
    else:
        dt = 1.0

    # --- Layout ---
    if has_signal:
        fig, (ax_sig, ax_kp) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [1, 2.5]},
        )
    else:
        fig, ax_kp = plt.subplots(figsize=figsize)
        ax_sig = None

    # --- Top panel: class means ---
    if has_signal:
        y_test = np.asarray(y_test)
        classes = np.unique(y_test)

        if colors is None:
            colors_use = [OI[i % len(OI)] for i in range(len(classes))]
        else:
            colors_use = list(colors)

        if class_names is None:
            names_use = [f"Class {cls}" for cls in classes]
        else:
            names_use = list(class_names)

        for k, cls in enumerate(classes):
            cm = X_test[y_test == cls].mean(axis=0)
            ax_sig.plot(t_axis, cm, color=colors_use[k], linewidth=3,
                        label=names_use[k])

        if event_time is not None and explicit_time:
            ax_sig.axvline(event_time, color="k", linestyle="--",
                           linewidth=0.8, label="Event")

        ax_sig.legend(
            fontsize=8,
            ncol=len(classes) + (1 if event_time is not None and explicit_time else 0),
        )
        ax_sig.set_ylabel("Amplitude")
        ax_sig.set_title("Class Means")
        ax_sig.grid(True, alpha=0.2)

    # --- Bottom panel: kernel weight patterns ---
    for row, info in enumerate(decoded):
        kernel_weights = np.array(info["kernel_weights"])
        dilation = info["dilation"]
        kernel_len = len(kernel_weights)

        peak_t = selected[row]["peak_t"]
        centroid_valid = selected[row]["is_local"]

        if centroid_valid:
            rf = (kernel_len - 1) * dilation + 1
            start_sample = peak_t - rf // 2
        else:
            # No temporal localization: span the full window
            start_sample = 0
            rf = n_timepoints

        sample_positions = []
        for wi in range(kernel_len):
            s = start_sample + wi * dilation
            sample_positions.append(s)

        # Draw weight ticks only when centroid is valid
        if centroid_valid:
            for wi in range(kernel_len):
                s = sample_positions[wi]
                w = kernel_weights[wi]
                if explicit_time and t_axis is not None:
                    s_clip = max(0, min(s, n_timepoints - 1))
                    t_pos = t_axis[0] + s_clip * dt
                else:
                    t_pos = float(s)
                color = WEIGHT_COLORS.get(int(round(w)), "#7f7f7f")
                ax_kp.plot(
                    [t_pos, t_pos],
                    [row - tick_height, row + tick_height],
                    color=color,
                    linewidth=tick_width,
                    solid_capstyle='butt',
                    alpha=0.8,
                    zorder=3,
                )

        # Draw horizontal span line
        pooling_color = POOLING_COLORS.get(info["pooling_op"], "#7f7f7f")
        if centroid_valid:
            t_positions_time = []
            for s in sample_positions:
                if explicit_time and t_axis is not None:
                    s_clip = max(0, min(s, n_timepoints - 1))
                    t_positions_time.append(t_axis[0] + s_clip * dt)
                else:
                    t_positions_time.append(float(s))
            span_left = min(t_positions_time)
            span_right = max(t_positions_time)
        else:
            span_left = t_axis[0] if explicit_time else 0.0
            span_right = t_axis[-1] if explicit_time else float(n_timepoints - 1)

        ax_kp.plot(
            [span_left, span_right],
            [row, row],
            color=pooling_color,
            linewidth=3,
            alpha=0.6,
            zorder=1,
        )

    # Event line
    if event_time is not None and explicit_time:
        ax_kp.axvline(event_time, color="k", linestyle="--",
                       linewidth=0.8, alpha=0.6)

    # Labels
    y_labels = [
        _format_feature_label(feature, compact=True)
        for feature in decoded
    ]
    ax_kp.set_yticks(range(n_show))
    ax_kp.set_yticklabels(y_labels, fontsize=8)
    ax_kp.set_xlabel(x_label)
    ax_kp.set_ylabel(f"Feature (sorted by {rank_label})")

    if explicit_time and t_axis is not None:
        ax_kp.set_xlim(t_axis[0], t_axis[-1])
    elif n_timepoints is not None:
        ax_kp.set_xlim(0, n_timepoints)

    ax_kp.set_ylim(-0.5, n_show - 0.5)
    ax_kp.invert_yaxis()
    ax_kp.grid(True, alpha=0.15, axis="x")

    # Legend
    legend_elements = [
        Patch(facecolor=WEIGHT_COLORS[2], alpha=0.8, label="+2"),
        Patch(facecolor=WEIGHT_COLORS[-1], alpha=0.8, label="-1"),
    ]
    # Add pooling colors that appear in the data
    pooling_shown = {d["pooling_op"] for d in decoded}
    for op in ["PPV", "MPV", "MIPV", "LSPV"]:
        if op in pooling_shown:
            legend_elements.append(
                Patch(facecolor=POOLING_COLORS[op], alpha=0.6,
                      label=f"{op}")
            )

    ax_kp.legend(
        handles=legend_elements, fontsize=7,
        loc="best", title="Kernel weights", title_fontsize=8,
    )
    filters = []
    if effective_localized_only:
        filters.append("localized")
    if suprathreshold_only:
        filters.append("suprathreshold")
    filter_suffix = (
        f", {' and '.join(filters)} only"
        if filters
        else ""
    )
    ax_kp.set_title(
        f"Kernel Weight Patterns ({n_show} features, sorted by {rank_label}"
        f"{filter_suffix})"
    )

    plt.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Class-mean activation
# -------------------------------------------------------------------------

def plot_class_mean_activation(
    model,
    X,
    y,
    feature_mask=None,
    rank_order=None,
    feature_rank=0,
    figsize=None,
    time_vector=None,
    sampling_rate=None,
    t_start=0.0,
    t_label=None,
    colors=None,
    class_names=None,
):
    """
    Inspect one selected feature on class-mean signals.

    This is the detailed companion to ``InterpRocket.plot_top_kernels``.
    ``plot_top_kernels`` gives a ranked overview of unique kernel
    configurations by default; this function shows the exact convolution
    output, bias threshold, and activation pattern for one transformed feature
    on each class-average signal. Features that share a base kernel and pooling
    operator can still differ because their representation or fitted bias
    threshold differs; the full identity is included in the figure title.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
        Data, typically X_test.
    y : array-like
        Class labels.
    feature_mask : array-like of int, optional
        Subset of feature indices. When ``rank_order`` is ``None``, the
        mask is treated as already ordered (rank zero is first). Ignored when
        ``rank_order`` is supplied.
    rank_order : array-like of int, optional
        Feature indices already sorted in display order. When provided,
        ``feature_mask`` and coefficient-based sorting are ignored. A fitted
        selector's ``consensus_ranking_`` is the canonical input.
    feature_rank : int, default=0
        Which feature to plot from the resolved ordering. 0 = top.
    figsize : tuple, optional
    time_vector : 1D array, optional
        Explicit time values, one per integer sample index. Length must
        equal n_timepoints. Takes precedence over `sampling_rate`.
    sampling_rate : float, optional
        Sampling rate in Hz. Used only if `time_vector` is None.
    t_start : float, default=0.0
        Start time in seconds for the `sampling_rate` path.
    t_label : str, optional
        Override the x-axis label. Defaults are 'Time' (time_vector path),
        'Time (s)' (sampling_rate path), or 'Timepoint (samples)' (no
        time info given).
    colors : list of str, optional
        One color per class, in the order they appear in np.unique(y).
        Defaults to OI.
    class_names : list of str, optional
        Display names for each class, in np.unique(y) order. Defaults to
        'Class {label}'.

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

    # ---- Resolve which feature to plot ----
    ranking = _resolve_feature_order(
        model, feature_mask=feature_mask, rank_order=rank_order
    )
    if isinstance(feature_rank, (bool, np.bool_)) or not isinstance(
        feature_rank, (int, np.integer)
    ):
        raise TypeError("feature_rank must be an integer.")
    if feature_rank < 0:
        raise ValueError("feature_rank must be nonnegative.")
    if feature_rank >= len(ranking):
        raise ValueError(
            f"feature_rank={feature_rank} but only {len(ranking)} "
            f"features available"
        )

    fi = int(ranking[feature_rank])
    f = model.decode_feature_index(fi)
    ki = f["kernel_index"]
    dil = f["dilation"]
    bias = f["bias"]
    rep = f["representation"]
    pooling = f["pooling_op"]

    # ---- Resolve classes, colors, and labels ----
    classes = np.unique(y)
    n_classes = len(classes)

    if colors is None:
        colors = [OI[i % len(OI)] for i in range(n_classes)]
    if len(colors) < n_classes:
        raise ValueError(
            f"colors has {len(colors)} entries but there are "
            f"{n_classes} classes"
        )

    if class_names is None:
        class_names = [f"Class {cls}" for cls in classes]
    if len(class_names) < n_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but there are "
            f"{n_classes} classes"
        )

    # ---- Resolve time axis ----
    n_timepoints = X.shape[1]
    t_signal, xlabel = _resolve_time_axis(
        n_timepoints, time_vector=time_vector,
        sampling_rate=sampling_rate, t_start=t_start, t_label=t_label,
    )

    if figsize is None:
        figsize = (14, 3 * n_classes)

    fig, axes = plt.subplots(n_classes, 2, figsize=figsize, sharex=True)
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for k, cls in enumerate(classes):
        mask = y == cls
        if rep == "diff":
            class_mean = np.diff(X[mask].mean(axis=0)).astype(np.float32)
            # diff loses one sample; place values at midpoints between
            # the original integer samples.
            cm_samples = np.arange(len(class_mean), dtype=float) + 0.5
            cm_time = _samples_to_time(cm_samples, t_signal)
        else:
            class_mean = X[mask].mean(axis=0).astype(np.float32)
            cm_time = t_signal

        conv_out, act, t_idx = compute_activation_map(
            class_mean,
            ki,
            np.int32(dil),
            np.float32(bias),
            f["padding_mode"],
            rep,
        )
        # t_idx is given in samples of the (possibly diffed) class mean.
        # For 'diff', t_idx indexes the diffed series, so we map through
        # the same midpoint convention.
        if rep == "diff":
            act_time = _samples_to_time(np.asarray(t_idx) + 0.5, t_signal)
        else:
            act_time = _samples_to_time(t_idx, t_signal)

        # ---- Left panel: activation map ----
        ax_act = axes[k, 0]
        ax_act.plot(cm_time, class_mean,
                    color="#7f7f7f", alpha=0.5, label="Class mean")
        ax_act.fill_between(
            act_time, 0, act * class_mean.max() * 0.3,
            color=colors[k], alpha=0.3, label="Activation",
        )
        ax_act.set_ylabel(class_names[k])
        ax_act.legend(fontsize=7)
        ax_act.grid(True, alpha=0.2)

        # ---- Right panel: convolution output ----
        ax_conv = axes[k, 1]
        ax_conv.plot(cm_time, class_mean,
                     color="#7f7f7f", alpha=0.5, label="Class mean")
        ax2 = ax_conv.twinx()
        ax2.plot(act_time, conv_out,
                 color=colors[k], linewidth=1.5, label="Conv output")
        ax2.axhline(bias, color="#2c2c2c", linestyle="--", linewidth=0.8,
                    label=f"Bias={bias:.2f}")
        ax2.fill_between(act_time, bias, conv_out,
                         where=conv_out > bias,
                         color=colors[k], alpha=0.2)
        ax2.set_ylabel("Conv output")
        if k == 0:
            ax2.legend(fontsize=7, loc="upper right")
        ax_conv.grid(True, alpha=0.2)

    axes[0, 0].set_title("Activation on class mean")
    axes[0, 1].set_title("Convolution output on class mean")
    axes[-1, 0].set_xlabel(xlabel)
    axes[-1, 1].set_xlabel(xlabel)

    fig.suptitle(
        f"{_format_feature_label(f, compact=False)}, rank {feature_rank + 1}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    return fig



# -------------------------------------------------------------------------
# Trial-by-trial selected-feature activation
# -------------------------------------------------------------------------

def feature_trial_activation(
    model,
    X,
    *,
    feature_index=None,
    feature_rank=0,
    feature_mask=None,
    rank_order=None,
):
    """Calculate one fitted feature across every trial.

    This is the trial-level counterpart to :func:`plot_class_mean_activation`.
    It preserves the local convolution domain of the decoded feature and maps
    raw- and first-difference activations back to the original input time axis.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Supplies feature decoding and the fitted bias threshold.
    X : ndarray, shape (n_trials, n_timepoints)
        Raw univariate time-series trials.
    feature_index : int, optional
        Full transformed-column index. When supplied, it overrides
        ``feature_rank``, ``feature_mask``, and ``rank_order``.
    feature_rank : int, default=0
        Rank within the resolved feature ordering. Rank zero is the top-ranked
        feature.
    feature_mask : array-like of int, optional
        Candidate feature indices. Preserved as the ordering when
        ``rank_order`` is omitted.
    rank_order : array-like of int, optional
        Explicit feature ordering.

    Returns
    -------
    dict
        ``feature_index`` and ``feature`` identify the decoded column.
        ``activation``, ``convolution``, and ``centered_response`` have shape
        ``(n_trials, n_timepoints)``. Positions outside a valid convolution
        region are NaN. ``pooling_value`` contains the exact I-ROCKET
        transformed value for the decoded pooling operator on each trial.
        Conventional longest-positive-run bounds and counts are also returned
        for interpreting LSPV features.

    Notes
    -----
    Each activation cell is local: it comes from one dilated nine-tap kernel
    placement. The pooled feature is global over the complete eligible
    convolution domain. This distinction is especially important for LSPV,
    whose value can be determined by a sustained run far from a visually
    prominent receptive-field location.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional (n_trials, n_timepoints).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if X.shape[1] < 9:
        raise ValueError("Each trial must contain at least nine timepoints.")

    resolved_index, resolved_rank = _resolve_single_feature(
        model,
        feature_index=feature_index,
        feature_rank=feature_rank,
        feature_mask=feature_mask,
        rank_order=rank_order,
    )
    feature = dict(model.decode_feature_index(resolved_index))

    n_trials, n_timepoints = X.shape
    activation = np.full(
        (n_trials, n_timepoints),
        np.nan,
        dtype=np.float32,
    )
    convolution = np.full(
        (n_trials, n_timepoints),
        np.nan,
        dtype=np.float32,
    )
    centered_response = np.full(
        (n_trials, n_timepoints),
        np.nan,
        dtype=np.float32,
    )
    pooling_value = np.zeros(n_trials, dtype=np.float64)
    positive_count = np.zeros(n_trials, dtype=np.int64)
    valid_count = np.zeros(n_trials, dtype=np.int64)
    longest_run_start = np.full(n_trials, -1, dtype=np.int64)
    longest_run_stop = np.full(n_trials, -1, dtype=np.int64)
    longest_run_length = np.zeros(n_trials, dtype=np.int64)

    pooling_names = ("PPV", "MPV", "MIPV", "LSPV")
    pooling_name = str(feature["pooling_op"])
    if pooling_name not in pooling_names:
        raise ValueError(f"Unknown pooling operator {pooling_name!r}.")
    pooling_index = pooling_names.index(pooling_name)

    representation = str(feature["representation"])
    bias = float(feature["bias"])

    for trial_index in range(n_trials):
        raw_trial = X[trial_index]
        if representation == "diff":
            if raw_trial.size < 10:
                raise ValueError(
                    "First-difference features require at least ten raw "
                    "timepoints."
                )
            x_use = np.diff(raw_trial).astype(np.float32)
        else:
            x_use = raw_trial

        conv, active, time_indices = compute_activation_map(
            x_use,
            int(feature["kernel_index"]),
            int(feature["dilation"]),
            bias,
            str(feature["padding_mode"]),
            representation,
        )
        conv = np.asarray(conv, dtype=np.float32)
        active = np.asarray(active, dtype=np.float32)
        time_indices = np.asarray(time_indices, dtype=float)

        exact_values = _pool_convolution(
            conv,
            np.float32(bias),
            0,
            len(conv),
        )
        pooling_value[trial_index] = float(exact_values[pooling_index])
        positive_count[trial_index] = int(np.sum(active > 0.0))
        valid_count[trial_index] = int(active.size)

        run_start, run_stop, run_length = _longest_positive_run(active > 0.0)
        longest_run_length[trial_index] = run_length
        if run_length > 0:
            start_center = int(round(float(time_indices[run_start])))
            stop_center = int(round(float(time_indices[run_stop])))
            if representation == "diff":
                start_center += 1
                stop_center += 1
            longest_run_start[trial_index] = int(
                np.clip(start_center, 0, n_timepoints - 1)
            )
            longest_run_stop[trial_index] = int(
                np.clip(stop_center, 0, n_timepoints - 1)
            )

        for local_index, time_index in enumerate(time_indices):
            center = int(round(float(time_index)))
            if representation == "diff":
                center += 1
            if 0 <= center < n_timepoints:
                activation[trial_index, center] = active[local_index]
                convolution[trial_index, center] = conv[local_index]
                centered_response[trial_index, center] = (
                    conv[local_index] - bias
                )

    return {
        "feature_index": resolved_index,
        "feature_rank": resolved_rank,
        "feature": feature,
        "activation": activation,
        "convolution": convolution,
        "centered_response": centered_response,
        "valid_mask": np.isfinite(activation),
        "pooling_value": pooling_value,
        "pooling_op": pooling_name,
        "positive_count": positive_count,
        "valid_count": valid_count,
        "positive_fraction": np.divide(
            positive_count,
            valid_count,
            out=np.zeros(n_trials, dtype=np.float64),
            where=valid_count > 0,
        ),
        "longest_run_start": longest_run_start,
        "longest_run_stop": longest_run_stop,
        "longest_run_length": longest_run_length,
        "trial_index": np.arange(n_trials, dtype=np.int64),
    }


def plot_feature_trial_heatmap(
    model,
    X,
    y=None,
    *,
    feature_index=None,
    feature_rank=0,
    feature_mask=None,
    rank_order=None,
    trial_order=None,
    sort_by=None,
    value="activation",
    show_pooling_values=True,
    highlight_longest_run=None,
    time_vector=None,
    sampling_rate=None,
    t_start=0.0,
    t_label=None,
    class_names=None,
    trial_labels=None,
    figsize=(12, 8),
):
    """Plot one selected feature as a trial-by-time activation heatmap.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Fitted model.
    X : ndarray, shape (n_trials, n_timepoints)
        Raw univariate trials.
    y : array-like, optional
        Class labels. When supplied, the default ordering groups trials by
        class and the right-side pooled-value panel is colored by class.
    feature_index : int, optional
        Full transformed-column index. Overrides rank-based selection.
    feature_rank : int, default=0
        Feature rank when ``feature_index`` is omitted.
    feature_mask, rank_order : array-like of int, optional
        Feature candidate set or explicit ordering.
    trial_order : array-like of int, optional
        Explicit complete permutation of trial indices. Overrides ``sort_by``.
    sort_by : {None, 'trial', 'class', 'pooling', 'longest_run'}, optional
        Trial ordering. ``None`` uses ``'class'`` when ``y`` is supplied and
        ``'trial'`` otherwise.
    value : {'activation', 'centered'}, default='activation'
        Heatmap quantity. ``activation`` displays the binary threshold crossing;
        ``centered`` displays convolution minus fitted bias.
    show_pooling_values : bool, default=True
        Add a right-side panel containing the exact transformed value for the
        selected pooling operator on every trial.
    highlight_longest_run : bool or None, default=None
        Overlay each trial's longest positive run. ``None`` enables the overlay
        automatically for LSPV features.
    time_vector, sampling_rate, t_start, t_label
        Time-axis options shared with the other interpretability plots.
    class_names : sequence or mapping, optional
        Display names for classes in ``np.unique(y)`` order, or a mapping from
        class values to names.
    trial_labels : sequence, optional
        Labels for individual trials. They are displayed when there are at most
        30 rows and trials are not grouped by class.
    figsize : tuple, default=(12, 8)

    Returns
    -------
    fig : matplotlib Figure

    Notes
    -----
    Gray cells were not evaluated because the selected feature uses valid
    padding or the first-difference representation. The adjacent panel reports
    the globally pooled feature value. For an LSPV column, the black overlay
    marks the conventional longest contiguous activation run on each trial.
    """
    if not isinstance(show_pooling_values, (bool, np.bool_)):
        raise TypeError("show_pooling_values must be a boolean.")
    if highlight_longest_run is not None and not isinstance(
        highlight_longest_run, (bool, np.bool_)
    ):
        raise TypeError("highlight_longest_run must be a boolean or None.")
    if value not in {"activation", "centered"}:
        raise ValueError("value must be 'activation' or 'centered'.")

    data = feature_trial_activation(
        model,
        X,
        feature_index=feature_index,
        feature_rank=feature_rank,
        feature_mask=feature_mask,
        rank_order=rank_order,
    )
    n_trials, n_timepoints = data["activation"].shape

    if y is not None:
        y = np.asarray(y)
        if y.ndim != 1 or y.size != n_trials:
            raise ValueError("y must contain one class label per trial.")
    if trial_labels is not None:
        trial_labels = np.asarray(trial_labels, dtype=object)
        if trial_labels.ndim != 1 or trial_labels.size != n_trials:
            raise ValueError("trial_labels must contain one value per trial.")

    if trial_order is not None:
        order = _validate_trial_order(trial_order, n_trials)
        resolved_sort = "custom"
    else:
        resolved_sort = (
            "class" if sort_by is None and y is not None
            else "trial" if sort_by is None
            else str(sort_by)
        )
        if resolved_sort == "trial":
            order = np.arange(n_trials, dtype=np.int64)
        elif resolved_sort == "class":
            if y is None:
                raise ValueError("sort_by='class' requires y.")
            order = np.argsort(y, kind="stable")
        elif resolved_sort == "pooling":
            order = np.argsort(
                data["pooling_value"],
                kind="stable",
            )[::-1]
        elif resolved_sort == "longest_run":
            order = np.argsort(
                data["longest_run_length"],
                kind="stable",
            )[::-1]
        else:
            raise ValueError(
                "sort_by must be None, 'trial', 'class', 'pooling', or "
                "'longest_run'."
            )

    t_axis, xlabel = _resolve_time_axis(
        n_timepoints,
        time_vector=time_vector,
        sampling_rate=sampling_rate,
        t_start=t_start,
        t_label=t_label,
    )
    x_lo = float(t_axis[0])
    x_hi = float(t_axis[-1])

    if show_pooling_values:
        fig, (ax_heat, ax_pool) = plt.subplots(
            1,
            2,
            figsize=figsize,
            sharey=True,
            gridspec_kw={"width_ratios": [5.0, 1.25], "wspace": 0.08},
        )
    else:
        fig, ax_heat = plt.subplots(figsize=figsize)
        ax_pool = None

    feature = data["feature"]
    pooling_color = POOLING_COLORS.get(
        str(feature["pooling_op"]),
        OI[0],
    )

    if value == "activation":
        shown = np.ma.masked_invalid(data["activation"][order])
        cmap = ListedColormap(["#F0F0F0", pooling_color])
        cmap.set_bad("#B3B3B3")
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        image = ax_heat.imshow(
            shown,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=(x_lo, x_hi, n_trials - 0.5, -0.5),
        )
        legend_handles = [
            Patch(facecolor=pooling_color, label="Above fitted bias"),
            Patch(facecolor="#F0F0F0", edgecolor="#B3B3B3", label="Below bias"),
            Patch(facecolor="#B3B3B3", label="Not evaluated"),
        ]
    else:
        shown = np.ma.masked_invalid(data["centered_response"][order])
        finite = shown.compressed()
        limit = (
            float(np.percentile(np.abs(finite), 99.0))
            if finite.size
            else 1.0
        )
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        cmap = LinearSegmentedColormap.from_list(
            "irocket_centered_response",
            [OI[1], "#FFFFFF", OI[0]],
        )
        cmap.set_bad("#B3B3B3")
        image = ax_heat.imshow(
            shown,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            extent=(x_lo, x_hi, n_trials - 0.5, -0.5),
        )
        fig.colorbar(
            image,
            ax=ax_heat,
            pad=0.01,
            label="Convolution - fitted bias",
        )
        legend_handles = [
            Patch(facecolor="#B3B3B3", label="Not evaluated"),
        ]

    if highlight_longest_run is None:
        highlight_longest_run = str(feature["pooling_op"]) == "LSPV"
    if highlight_longest_run:
        for display_row, original_trial in enumerate(order):
            start = int(data["longest_run_start"][original_trial])
            stop = int(data["longest_run_stop"][original_trial])
            if start >= 0 and stop >= start:
                ax_heat.plot(
                    [t_axis[start], t_axis[stop]],
                    [display_row, display_row],
                    color="#000000",
                    linewidth=0.65,
                    alpha=0.8,
                    solid_capstyle="butt",
                )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#000000",
                linewidth=1.2,
                label="Longest positive run",
            )
        )

    ax_heat.set_xlabel(xlabel)
    ax_heat.set_ylabel("Trial")
    ax_heat.set_xlim(x_lo, x_hi)
    ax_heat.set_ylim(n_trials - 0.5, -0.5)
    ax_heat.legend(
        handles=legend_handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=min(len(legend_handles), 4),
        framealpha=0.9,
    )

    ordered_y = None if y is None else y[order]
    if resolved_sort == "class" and ordered_y is not None:
        classes = np.unique(ordered_y)
        if class_names is None:
            class_display = {cls: f"Class {cls}" for cls in classes}
        elif isinstance(class_names, dict):
            class_display = {
                cls: str(class_names.get(cls, cls))
                for cls in classes
            }
        else:
            if len(class_names) < len(classes):
                raise ValueError(
                    "class_names has fewer entries than the number of classes."
                )
            class_display = {
                cls: str(class_names[index])
                for index, cls in enumerate(classes)
            }

        changes = np.flatnonzero(ordered_y[1:] != ordered_y[:-1]) + 1
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [n_trials]))
        centers = 0.5 * (starts + stops - 1)
        group_values = [ordered_y[start] for start in starts]
        ax_heat.set_yticks(centers)
        ax_heat.set_yticklabels(
            [class_display[value] for value in group_values],
            fontsize=8,
        )
        for boundary in changes:
            ax_heat.axhline(
                boundary - 0.5,
                color="#000000",
                linewidth=0.8,
                alpha=0.6,
            )
            if ax_pool is not None:
                ax_pool.axhline(
                    boundary - 0.5,
                    color="#000000",
                    linewidth=0.8,
                    alpha=0.6,
                )
    elif n_trials <= 30:
        if trial_labels is None:
            labels = [str(index) for index in order]
        else:
            labels = [str(trial_labels[index]) for index in order]
        ax_heat.set_yticks(np.arange(n_trials))
        ax_heat.set_yticklabels(labels, fontsize=7)
    else:
        ax_heat.set_yticks([])

    if ax_pool is not None:
        rows = np.arange(n_trials)
        pooled = data["pooling_value"][order]
        if ordered_y is None:
            ax_pool.scatter(
                pooled,
                rows,
                s=16,
                color=pooling_color,
                alpha=0.8,
            )
        else:
            classes = np.unique(ordered_y)
            for class_index, cls in enumerate(classes):
                mask = ordered_y == cls
                ax_pool.scatter(
                    pooled[mask],
                    rows[mask],
                    s=16,
                    color=OI[class_index % len(OI)],
                    alpha=0.8,
                    label=str(cls),
                )
        ax_pool.set_xlabel(f"{feature['pooling_op']} value")
        ax_pool.set_title("Pooled feature", fontsize=10)
        ax_pool.grid(True, axis="x", alpha=0.2)
        ax_pool.set_ylim(n_trials - 0.5, -0.5)
        ax_pool.tick_params(axis="y", left=False, labelleft=False)

    rank_text = (
        ""
        if data["feature_rank"] is None
        else f" | rank {data['feature_rank'] + 1}"
    )
    fig.suptitle(
        "Selected feature across trials\n"
        f"{_format_feature_label(feature, compact=False)}{rank_text}",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.10,
        top=0.89,
        wspace=0.08,
    )
    return fig


# -------------------------------------------------------------------------
# Composite activation trace
# -------------------------------------------------------------------------

def composite_activation_trace(
    X,
    model,
    selected_features=None,
    class_index=0,
    show_progress=None,
    progress_threshold=500,
):
    """Build an approximate classifier-weighted pre-pooling activation trace.

    The trace groups selected columns that share the same convolutional kernel,
    dilation, bias, representation, and padding mode, then weights that shared
    pre-pooling response by the sum of their signed ridge coefficients. Exact
    convolution regions from :func:`interp_rocket.compute_activation_map` are
    used, including valid padding and the first-difference alignment.

    This remains a descriptive visualization: PPV, MPV, MIPV, and LSPV do not
    have a unique inverse mapping to time, so the trace should not be treated as
    an additive decomposition of the classifier decision.

    Parameters
    ----------
    X : ndarray of shape (n_trials, n_timepoints)
        Raw univariate time series.
    model : fitted InterpRocket or StableRocketClassifier
        Supplies decoded feature metadata and ridge coefficients.
    selected_features : array-like of int, optional
        Full-transform feature indices. Defaults to ``model.selected_indices_``
        for a fitted ``StableRocketClassifier``.
    class_index : int, default=0
        Coefficient row used for multiclass ridge models. Binary models have a
        single row and ignore this value.
    show_progress : bool or None, default=None
        Display a ``tqdm`` progress bar over unique convolution/bias groups.
        ``None`` enables the bar automatically when ``X`` contains at least
        ``progress_threshold`` trials. Set explicitly to ``True`` or ``False``
        to override the automatic behavior. If ``tqdm`` is unavailable, the
        calculation continues without a bar and emits a warning.
    progress_threshold : int, default=500
        Minimum number of trials that enables the progress bar when
        ``show_progress=None``.

    Returns
    -------
    ndarray of shape (n_trials, n_timepoints)
        Classifier-weighted approximate activation trace.
    """
    from interp_rocket import compute_activation_map

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if show_progress is not None and not isinstance(
        show_progress, (bool, np.bool_)
    ):
        raise TypeError("show_progress must be a boolean or None.")
    if isinstance(progress_threshold, (bool, np.bool_)) or not isinstance(
        progress_threshold, (int, np.integer)
    ):
        raise TypeError("progress_threshold must be an integer.")
    if int(progress_threshold) < 1:
        raise ValueError("progress_threshold must be at least 1.")
    progress_threshold = int(progress_threshold)

    n_output_features = int(getattr(model, "n_output_features_", 0))
    if n_output_features <= 0:
        raise ValueError("model must be fitted before computing a trace.")

    if selected_features is None:
        if not hasattr(model, "selected_indices_"):
            raise ValueError(
                "selected_features is required unless model exposes "
                "selected_indices_."
            )
        selected_features = model.selected_indices_
    selected_features = np.asarray(selected_features)
    if selected_features.ndim != 1 or not np.issubdtype(
        selected_features.dtype, np.integer
    ):
        raise ValueError(
            "selected_features must be a one-dimensional integer array."
        )
    selected_features = selected_features.astype(np.int64, copy=False)
    if selected_features.size == 0:
        raise ValueError("selected_features must not be empty.")
    if np.any(selected_features < 0) or np.any(
        selected_features >= n_output_features
    ):
        raise ValueError("selected_features contains an out-of-range index.")
    if np.unique(selected_features).size != selected_features.size:
        raise ValueError("selected_features must not contain duplicates.")

    if hasattr(model, "selected_indices_"):
        fitted_selected = set(
            np.asarray(model.selected_indices_, dtype=np.int64).tolist()
        )
        missing = [
            int(index)
            for index in selected_features
            if int(index) not in fitted_selected
        ]
        if missing:
            raise ValueError(
                "selected_features includes columns that were not retained by "
                f"the fitted selector: {missing[:5]}."
            )

    coefficients = _full_classifier_coefficients(model)
    if coefficients.ndim == 1:
        coefficients = coefficients[np.newaxis, :]
    if coefficients.ndim != 2 or coefficients.shape[1] != n_output_features:
        raise ValueError(
            "Classifier coefficients do not match the full transformed "
            "feature universe."
        )
    if coefficients.shape[0] == 1:
        coefficient_row = coefficients[0]
    else:
        if isinstance(class_index, (bool, np.bool_)) or not isinstance(
            class_index, (int, np.integer)
        ):
            raise TypeError("class_index must be an integer.")
        if class_index < 0 or class_index >= coefficients.shape[0]:
            raise ValueError(
                "class_index is outside the classifier coefficient rows."
            )
        coefficient_row = coefficients[int(class_index)]

    # Pooling columns that share a convolution and bias contribute to the same
    # approximate pre-pooling trace. Sum their signed ridge coefficients once.
    grouped_weights = defaultdict(float)
    grouped_metadata = {}
    for feature_index in selected_features:
        feature_index = int(feature_index)
        info = model.decode_feature_index(feature_index)
        key = (
            int(info["kernel_index"]),
            int(info["dilation"]),
            float(info["bias"]),
            str(info["representation"]),
            str(info["padding_mode"]),
        )
        grouped_weights[key] += float(coefficient_row[feature_index])
        grouped_metadata[key] = info

    n_trials, n_timepoints = X.shape
    trace = np.zeros((n_trials, n_timepoints), dtype=np.float64)

    active_groups = [
        (key, combined_weight)
        for key, combined_weight in grouped_weights.items()
        if combined_weight != 0.0
    ]
    progress_enabled = (
        n_trials >= progress_threshold
        if show_progress is None
        else bool(show_progress)
    )
    group_iterator = active_groups
    if progress_enabled:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            warnings.warn(
                "Progress display requires tqdm. Continuing without a "
                "progress bar; install tqdm to enable it.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            group_iterator = tqdm(
                active_groups,
                total=len(active_groups),
                desc=f"Composite activation trace ({n_trials} trials)",
                unit="kernel group",
                dynamic_ncols=True,
            )

    for key, combined_weight in group_iterator:
        info = grouped_metadata[key]
        representation = str(info["representation"])

        for trial_index in range(n_trials):
            if representation == "diff":
                x_input = np.diff(X[trial_index]).astype(np.float32)
            else:
                x_input = X[trial_index]
            convolution, _, time_indices = compute_activation_map(
                x_input,
                int(info["kernel_index"]),
                int(info["dilation"]),
                float(info["bias"]),
                str(info["padding_mode"]),
                representation,
            )
            positive_response = np.maximum(
                convolution.astype(np.float64) - float(info["bias"]),
                0.0,
            )
            for local_index, time_index in enumerate(time_indices):
                center = int(round(float(time_index)))
                if representation == "diff":
                    center = min(center + 1, n_timepoints - 1)
                if 0 <= center < n_timepoints:
                    trace[trial_index, center] += (
                        combined_weight * positive_response[local_index]
                    )

    return trace


def plot_activation_trace(
    composite_trace,
    y,
    time_vector=None,
    sampling_rate=None,
    t_start=0.0,
    t_label=None,
    event_time=0.0,
    colors=None,
    class_names=None,
    figsize=(10, 6),
):
    """
    Plot the trial-averaged composite activation trace split by class,
    with shaded standard-error bands.

    Parameters
    ----------
    composite_trace : ndarray, shape (n_trials, n_timepoints)
        Output of `composite_activation_trace`.
    y : array-like
        Class labels.
    time_vector : 1D array, optional
        Explicit per-sample time values. Length must equal n_timepoints.
        Takes precedence over `sampling_rate`.
    sampling_rate : float, optional
        Sampling rate in Hz. Used only when `time_vector` is None.
    t_start : float, default=0.0
        Start time for the `sampling_rate` path.
    t_label : str, optional
        X-axis label override. Defaults are 'Time' (time_vector path),
        'Time (s)' (sampling_rate path), or 'Timepoint (samples)'.
    event_time : float, default=0.0
        Vertical dashed line at this time value. The line is drawn only when
        ``t_label`` is not None; this suppresses an arbitrary line at sample 0
        for unlabeled sample-index plots. Set to None to disable explicitly.
    colors : list of str, optional
        One color per class, in np.unique(y) order. Defaults to the
        Okabe-Ito palette.
    class_names : list of str, optional
        Display names for each class. Defaults to 'Class {label}'.
    figsize : tuple, default=(10, 6)

    Returns
    -------
    fig : matplotlib Figure
    """
    composite_trace = np.asarray(composite_trace, dtype=float)
    y = np.asarray(y)
    if composite_trace.ndim != 2:
        raise ValueError(
            f"composite_trace must be 2D (n_trials, n_timepoints); "
            f"got shape {composite_trace.shape}"
        )
    if not np.all(np.isfinite(composite_trace)):
        raise ValueError("composite_trace must contain only finite values.")
    if y.ndim != 1 or y.size != composite_trace.shape[0]:
        raise ValueError(
            "y must be one-dimensional with one label per trace."
        )

    n_timepoints = composite_trace.shape[1]
    classes = np.unique(y)
    if classes.size < 1:
        raise ValueError("y must contain at least one class label.")
    n_classes = len(classes)

    # Time axis
    t_axis, xlabel = _resolve_time_axis(
        n_timepoints, time_vector=time_vector,
        sampling_rate=sampling_rate, t_start=t_start, t_label=t_label,
    )

    # Colors and names
    if colors is None:
        colors_use = [OI[i % len(OI)] for i in range(n_classes)]
    else:
        if len(colors) < n_classes:
            raise ValueError(
                f"colors has {len(colors)} entries but there are "
                f"{n_classes} classes"
            )
        colors_use = list(colors)

    if class_names is None:
        names_use = [f"Class {c}" for c in classes]
    else:
        if len(class_names) < n_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but there "
                f"are {n_classes} classes"
            )
        names_use = list(class_names)

    fig, ax = plt.subplots(figsize=figsize)

    for k, c in enumerate(classes):
        traces_c = composite_trace[y == c]
        mean_trace = np.mean(traces_c, axis=0)
        sem_trace = np.std(traces_c, axis=0) / np.sqrt(len(traces_c))

        ax.plot(t_axis, mean_trace, color=colors_use[k],
                label=names_use[k], linewidth=2)
        ax.fill_between(
            t_axis,
            mean_trace - sem_trace,
            mean_trace + sem_trace,
            color=colors_use[k], alpha=0.3,
        )

    if event_time is not None and t_label is not None:
        ax.axvline(
            x=event_time,
            color="k",
            linestyle="--",
            alpha=0.7,
            label="Event",
        )

    ax.set_title("Composite Activation Trace")
    ax.set_ylabel("Weighted Activation")
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig


# -------------------------------------------------------------------------
# Model-level interpretability plots
# -------------------------------------------------------------------------

def plot_kernel_similarity(
    model, X_test, feature_mask=None,
    threshold=0.5, n_top=None, figsize=(12, 5),
):
    """
    Visualize similarity among kernel features.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Fitted model.
    X_test : ndarray, shape (n_instances, n_timepoints)
    feature_mask : ndarray, optional
        Feature indices to analyze. If None, uses top features by importance.
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

    # Diverging colorblind-safe map for negative and positive correlations.
    blue_orange = LinearSegmentedColormap.from_list(
        'BlueOrange', ['#0072B2', '#f0f0f0', '#E69F00'], N=256
    )

    if not np.isscalar(threshold):
        raise TypeError("threshold must be a scalar in [0, 1].")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    X_test = np.asarray(X_test, dtype=np.float32)
    if X_test.ndim != 2 or not np.all(np.isfinite(X_test)):
        raise ValueError("X_test must be a finite two-dimensional matrix.")
    features = _full_feature_matrix(model, X_test)

    importance = model.get_feature_importance()
    if feature_mask is None:
        if hasattr(model, "selected_indices_"):
            candidates = np.asarray(model.selected_indices_, dtype=np.int64)
        else:
            candidates = np.arange(len(importance), dtype=np.int64)
        if n_top is None:
            n_top = min(50, len(candidates))
        if isinstance(n_top, (bool, np.bool_)) or not isinstance(
            n_top, (int, np.integer)
        ):
            raise TypeError("n_top must be an integer or None.")
        if n_top < 2:
            raise ValueError("n_top must be at least two.")
        order = np.argsort(importance[candidates], kind="stable")[::-1]
        feature_mask = candidates[order[: int(n_top)]]
    else:
        feature_mask = _validate_feature_index_array(
            feature_mask, features.shape[1], name="feature_mask"
        )
        if n_top is not None:
            if isinstance(n_top, (bool, np.bool_)) or not isinstance(
                n_top, (int, np.integer)
            ):
                raise TypeError("n_top must be an integer or None.")
            if n_top < 2:
                raise ValueError("n_top must be at least two.")
            if len(feature_mask) > int(n_top):
                order = np.argsort(
                    importance[feature_mask], kind="stable"
                )[::-1]
                feature_mask = feature_mask[order[: int(n_top)]]

    if feature_mask.size < 2:
        raise ValueError("At least two features are required for correlation.")
    if np.any(feature_mask < 0) or np.any(feature_mask >= features.shape[1]):
        raise ValueError("feature_mask contains an out-of-range index.")
    features = features[:, feature_mask]
    n_feats = features.shape[1]

    # Compute correlation matrix. Constant selected columns have undefined
    # Pearson correlations; treat those entries as zero for visualization.
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(features.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    # Decode labels
    labels = []
    for fi in feature_mask:
        info = model.decode_feature_index(int(fi))
        labels.append(_format_feature_label(info, compact=True))

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
        ax2.hist(within_corrs, bins=bins, alpha=0.6, color='#0072B2',
                 label=f'Within-kernel ({len(within_corrs)})', density=True)
    if between_corrs:
        ax2.hist(between_corrs, bins=bins, alpha=0.6, color='#E69F00',
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


def plot_feature_stability(selection, model=None, n_show=50, figsize=(14, None)):
    """Visualize fixed-universe selection masks and consensus probabilities.

    Parameters
    ----------
    selection : fitted ResampledShrinkageSelector or StableRocketClassifier
        A final classifier contributes its fitted ``selector_`` and
        ``transformer_`` automatically.
    model : fitted I-ROCKET transformer or classifier, optional
        Decoder used when ``selection`` is a selector rather than a final
        classifier.
    n_show : int, default=50
        Maximum number of features, ranked by selection probability.
    figsize : tuple, default=(14, None)

    Returns
    -------
    matplotlib.figure.Figure
    """
    if hasattr(selection, "selector_"):
        selector = selection.selector_
        decoder = selection.transformer_
        threshold = float(selection.consensus_threshold)
    else:
        selector = selection
        decoder = getattr(model, "transformer_", model)
        threshold = float(getattr(selector, "consensus_threshold", 0.7))

    required = (
        "selection_matrix_",
        "selection_probabilities_",
        "consensus_ranking_",
    )
    missing = [name for name in required if not hasattr(selector, name)]
    if missing:
        raise TypeError(
            "selection must be fitted and expose " + ", ".join(required) + "."
        )
    matrix = np.asarray(selector.selection_matrix_)
    probabilities = np.asarray(selector.selection_probabilities_, dtype=float)
    ranking = np.asarray(selector.consensus_ranking_, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != probabilities.size:
        raise ValueError("The fitted selector contains inconsistent stability arrays.")
    if isinstance(n_show, (bool, np.bool_)) or not isinstance(
        n_show, (int, np.integer)
    ):
        raise TypeError("n_show must be an integer.")
    if n_show < 1:
        raise ValueError("n_show must be positive.")
    shown = ranking[: min(int(n_show), ranking.size)]
    heatmap = matrix[:, shown].T
    shown_probabilities = probabilities[shown]

    if figsize[1] is None:
        figsize = (figsize[0], max(5, min(16, shown.size * 0.25)))
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=figsize, width_ratios=[3, 1],
        gridspec_kw={"wspace": 0.05},
    )
    cmap = plt.matplotlib.colors.ListedColormap(["#f0f0f0", OI[0]])
    ax_heat.imshow(
        heatmap,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax_heat.set_xlabel("Selector resample")
    ax_heat.set_ylabel("Feature (ranked by selection probability)")
    ax_heat.set_title(f"Feature selections across {matrix.shape[0]} resamples")

    if decoder is not None and hasattr(decoder, "decode_feature_index"):
        labels = []
        for feature_index in shown:
            info = decoder.decode_feature_index(int(feature_index))
            labels.append(_format_feature_label(info, compact=True))
        ax_heat.set_yticks(np.arange(shown.size))
        ax_heat.set_yticklabels(labels, fontsize=7)

    selected_colors = [
        OI[0] if value >= threshold else "#B3B3B3"
        for value in shown_probabilities
    ]
    ax_bar.barh(np.arange(shown.size), shown_probabilities, color=selected_colors)
    ax_bar.axvline(
        threshold, color="#000000", linestyle="--", linewidth=1.0,
        label=f"Consensus = {threshold:.2f}",
    )
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(ax_heat.get_ylim())
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Selection probability")
    ax_bar.set_title("Consensus frequency")
    ax_bar.legend(fontsize=8, loc="lower right")
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.12, top=0.90, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# Localization threshold diagnostic
# ---------------------------------------------------------------------------


def localization_profile(model, X, y, feature_mask, localization_frac=0.5):
    """Compute class-mean localization quantities for a feature set.

    For each feature, the function measures the fraction of differential
    class-mean activation mass that falls within one receptive-field window
    centered on the differential centroid. It also reports the
    receptive-field-specific uniform baseline.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
        One class label per row.
    feature_mask : array-like of int
        Full transformed-feature indices, in the desired output order.
    localization_frac : float, default=0.5
        Raw ``mass_in_rf`` threshold used for the returned ``is_local`` field.
        For comparisons across dilations, the excess-over-baseline criterion
        used by :func:`localization_table` is usually preferable.

    Returns
    -------
    dict
        Keys are ``feature_index``, ``mass_in_rf``, ``baseline``, ``excess``,
        ``receptive_field``, ``dilation``, ``pooling_op``,
        ``fires_on_class_mean``, ``has_signal``, ``is_local``, and
        ``localization_frac``.

    Notes
    -----
    ``fires_on_class_mean`` means that the feature exceeds its fitted bias on at
    least one class-average waveform. A False value is the subthreshold or
    dagger case; the feature may still activate on individual trials.

    ``has_signal`` is stricter and asks whether class-mean activations differ
    anywhere in time. A feature can fire on all class means but have no
    differential class-mean activation.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("X must be 2D and y must contain one label per row.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.isscalar(localization_frac):
        raise TypeError("localization_frac must be a scalar in [0, 1].")
    localization_frac = float(localization_frac)
    if (
        not np.isfinite(localization_frac)
        or localization_frac < 0.0
        or localization_frac > 1.0
    ):
        raise ValueError("localization_frac must be in [0, 1].")

    feature_mask = _validate_feature_index_array(
        feature_mask,
        int(getattr(model, "n_output_features_", 0)),
        name="feature_mask",
    )
    n_timepoints = X.shape[1]
    classes = np.unique(y)
    class_means = _class_mean_signals(X, y, classes)

    mass = np.zeros(len(feature_mask), dtype=np.float64)
    baseline = np.zeros(len(feature_mask), dtype=np.float64)
    receptive_field = np.zeros(len(feature_mask), dtype=np.int64)
    dilation = np.zeros(len(feature_mask), dtype=np.int64)
    pooling = []
    fires_on_class_mean = np.zeros(len(feature_mask), dtype=bool)
    has_signal = np.zeros(len(feature_mask), dtype=bool)

    for row, feature_index in enumerate(feature_mask):
        feature = model.decode_feature_index(int(feature_index))
        profiles = _activation_profiles_on_class_means(
            class_means,
            feature,
            n_timepoints,
        )
        fires_on_class_mean[row] = bool(np.any(profiles > 0.0))
        differential = np.max(profiles, axis=0) - np.min(profiles, axis=0)
        has_signal[row] = bool(float(differential.sum()) > 0.0)

        _, _, _, mass_in_rf = _localization_from_profiles(
            profiles,
            feature,
            n_timepoints,
            localization_frac,
        )
        mass[row] = mass_in_rf
        receptive_field[row] = int(feature["receptive_field"])
        dilation[row] = int(feature["dilation"])
        pooling.append(str(feature["pooling_op"]))
        window = min(receptive_field[row], n_timepoints)
        baseline[row] = window / float(n_timepoints)

    return {
        "feature_index": feature_mask,
        "mass_in_rf": mass,
        "baseline": baseline,
        "excess": mass - baseline,
        "receptive_field": receptive_field,
        "dilation": dilation,
        "pooling_op": pooling,
        "fires_on_class_mean": fires_on_class_mean,
        "has_signal": has_signal,
        "is_local": has_signal & (mass >= localization_frac),
        "localization_frac": localization_frac,
    }


def localization_table(
    model,
    X,
    y,
    feature_mask=None,
    *,
    rank_order=None,
    n_top=15,
    margin=0.10,
    localization_frac=0.5,
):
    """Return a ranked DataFrame classifying feature localization.

    The table implements the dilation-fair criterion used in the July 25
    notebook workflow. A feature is called local when it has differential
    class-mean activation and its ``mass_in_rf`` exceeds the feature-specific
    uniform baseline by at least ``margin``.

    Three mutually exclusive categories are reported:

    ``local``
        Suprathreshold on at least one class mean, differential by class, and
        ``excess_over_baseline >= margin``.
    ``non-local``
        Suprathreshold on at least one class mean but not called local.
    ``subthreshold``
        Does not exceed the fitted bias on any class-average waveform. Such a
        feature may still activate on individual trials.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    feature_mask : array-like of int, optional
        Candidate indices. When omitted, features are ranked by the model's
        classifier importance within the eligible selected set.
    rank_order : array-like of int, optional
        Explicit feature ordering; overrides ``feature_mask``.
    n_top : int or None, default=15
        Number of ranked features to report. ``None`` reports the complete
        resolved ordering.
    margin : float, default=0.10
        Minimum excess of ``mass_in_rf`` over the uniform receptive-field
        baseline required for a local call.
    localization_frac : float, default=0.5
        Raw threshold retained in the underlying localization profile. It does
        not determine the table's ``local`` category, which uses ``margin``.

    Returns
    -------
    pandas.DataFrame
        Ranked feature metadata and localization quantities.
    """
    pd = _require_pandas()

    if not np.isscalar(margin) or isinstance(margin, (bool, np.bool_)):
        raise TypeError("margin must be a nonnegative finite scalar.")
    margin = float(margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be a nonnegative finite scalar.")

    ranking = _resolve_feature_order(
        model,
        feature_mask=feature_mask,
        rank_order=rank_order,
    )
    if n_top is not None:
        if isinstance(n_top, (bool, np.bool_)) or not isinstance(
            n_top, (int, np.integer)
        ):
            raise TypeError("n_top must be an integer or None.")
        if int(n_top) < 1:
            raise ValueError("n_top must be positive.")
        ranking = ranking[: min(int(n_top), len(ranking))]

    profile = localization_profile(
        model,
        X,
        y,
        feature_mask=ranking,
        localization_frac=localization_frac,
    )

    importance = np.full(
        int(getattr(model, "n_output_features_", 0)),
        np.nan,
        dtype=float,
    )
    if hasattr(model, "get_feature_importance"):
        importance = np.asarray(model.get_feature_importance(), dtype=float)

    probabilities = np.full_like(importance, np.nan, dtype=float)
    selector = getattr(model, "selector_", None)
    if selector is not None and hasattr(selector, "selection_probabilities_"):
        candidate = np.asarray(
            selector.selection_probabilities_,
            dtype=float,
        )
        if candidate.shape == probabilities.shape:
            probabilities = candidate

    fires = np.asarray(profile["fires_on_class_mean"], dtype=bool)
    differential = np.asarray(profile["has_signal"], dtype=bool)
    excess = np.asarray(profile["excess"], dtype=float)
    local = fires & differential & (excess >= margin)
    category = np.full(len(ranking), "non-local", dtype=object)
    category[~fires] = "subthreshold"
    category[local] = "local"

    decoded = [
        dict(model.decode_feature_index(int(index)))
        for index in ranking
    ]
    table = pd.DataFrame(
        {
            "rank": np.arange(1, len(ranking) + 1, dtype=int),
            "feature_index": np.asarray(
                profile["feature_index"],
                dtype=int,
            ),
            "kernel_index": [
                int(item["kernel_index"]) for item in decoded
            ],
            "receptive_field": np.asarray(
                profile["receptive_field"],
                dtype=int,
            ),
            "dilation": np.asarray(profile["dilation"], dtype=int),
            "pooling": list(profile["pooling_op"]),
            "representation": [
                str(item["representation"]) for item in decoded
            ],
            "padding": [
                str(item["padding_mode"]) for item in decoded
            ],
            "bias_rank": [
                int(item.get("bias_rank_within_kernel", -1))
                for item in decoded
            ],
            "bias": [float(item["bias"]) for item in decoded],
            "importance": importance[ranking],
            "selection_probability": probabilities[ranking],
            "rf_fraction": np.asarray(profile["baseline"], dtype=float),
            "mass_in_rf": np.asarray(profile["mass_in_rf"], dtype=float),
            "excess_over_baseline": excess,
            "fires_on_class_mean": fires,
            "differential_by_class": differential,
            "localized": local,
            "category": category,
        }
    )
    table.attrs["localization_margin"] = margin
    table.attrs["localization_frac"] = float(localization_frac)
    return table


def plot_localization_summary(
    table,
    *,
    margin=None,
    figsize=(12, 5),
):
    """Plot local, non-local, and subthreshold feature categories.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`localization_table`.
    margin : float, optional
        Localization margin to display. Defaults to the value stored in
        ``table.attrs`` and then to 0.10.
    figsize : tuple, default=(12, 5)

    Returns
    -------
    fig : matplotlib Figure

    Notes
    -----
    The left panel reports category counts split by pooling operator. The right
    panel shows each ranked feature's excess localization mass. Subthreshold
    features remain visible as a separate category rather than being mistaken
    for diffuse non-local features.
    """
    pd = _require_pandas()
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame.")
    required = {
        "rank",
        "pooling",
        "excess_over_baseline",
        "category",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            "table is missing required columns: " + ", ".join(missing)
        )

    if margin is None:
        margin = float(table.attrs.get("localization_margin", 0.10))
    elif not np.isscalar(margin) or isinstance(margin, (bool, np.bool_)):
        raise TypeError("margin must be a nonnegative finite scalar.")
    margin = float(margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be a nonnegative finite scalar.")

    categories = ("local", "non-local", "subthreshold")
    category_colors = {
        "local": OI[0],
        "non-local": OI[1],
        "subthreshold": "#7F7F7F",
    }
    markers = {
        "local": "o",
        "non-local": "s",
        "subthreshold": "x",
    }

    unexpected = sorted(set(table["category"]).difference(categories))
    if unexpected:
        raise ValueError(
            "table contains unknown feature categories: "
            + ", ".join(map(str, unexpected))
        )

    fig, (ax_count, ax_rank) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [0.9, 1.6]},
    )

    bottoms = np.zeros(len(categories), dtype=float)
    pooling_order = [
        name
        for name in ("PPV", "MPV", "MIPV", "LSPV")
        if name in set(table["pooling"])
    ]
    for pooling_name in pooling_order:
        counts = np.asarray(
            [
                int(
                    (
                        (table["category"] == category)
                        & (table["pooling"] == pooling_name)
                    ).sum()
                )
                for category in categories
            ],
            dtype=float,
        )
        ax_count.bar(
            categories,
            counts,
            bottom=bottoms,
            color=POOLING_COLORS.get(pooling_name, "#7F7F7F"),
            label=pooling_name,
        )
        bottoms += counts

    for position, total in enumerate(bottoms):
        ax_count.text(
            position,
            total + max(0.2, 0.02 * max(1.0, bottoms.max())),
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_count.set_ylabel("Feature count")
    ax_count.set_title("Localization categories by pooling")
    if pooling_order:
        ax_count.legend(
            title="Pooling",
            fontsize=8,
            title_fontsize=8,
        )
    ax_count.grid(True, axis="y", alpha=0.15)

    for category in categories:
        subset = table[table["category"] == category]
        ax_rank.scatter(
            subset["rank"],
            subset["excess_over_baseline"],
            color=category_colors[category],
            marker=markers[category],
            s=38,
            alpha=0.85,
            label=f"{category} (n={len(subset)})",
        )
    ax_rank.axhline(
        0.0,
        color="#7F7F7F",
        linewidth=1.0,
        label="Uniform-fire baseline",
    )
    ax_rank.axhline(
        margin,
        color="#000000",
        linestyle="--",
        linewidth=1.2,
        label=f"Local margin = {margin:g}",
    )
    ax_rank.set_xlabel("Classifier-importance rank")
    ax_rank.set_ylabel("Mass in RF - RF fraction")
    ax_rank.set_title("Ranked localization excess")
    ax_rank.grid(True, alpha=0.15)
    ax_rank.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig


def plot_localization_diagnostic(profile, method="excess",
                                 candidate_fracs=(0.2, 0.5),
                                 candidate_margins=(0.05, 0.10),
                                 exclude_no_signal=True,
                                 figsize=(12, 5)):
    """
    Justify a localization threshold from a localization_profile.

    Two methods.

    method="raw" thresholds mass_in_rf directly with localization_frac.
    A kernel is local when mass_in_rf >= localization_frac. This ignores
    receptive field size, so a global high-dilation kernel with a wide RF
    can clear a low flat threshold.

    method="excess" thresholds the excess of mass_in_rf over the per-kernel
    uniform-fire baseline with a margin. A kernel is local when
    mass_in_rf - baseline >= margin. This is dilation-fair. A kernel firing
    exactly as a uniform kernel would sits at excess 0 regardless of RF.

    No-signal kernels are excluded by default. A kernel that does not fire
    differentially on the class means has mass_in_rf 0 and excess equal to
    minus its baseline. Folding these into the threshold analysis produces a
    false negative tail sorted by dilation that looks like diffuse structure
    but is not. They are a separate category, the dagger cases, and the
    count of them is reported in the title rather than mixed into the curve.

    Two panels in both methods.

    Left panel ranks the features by the thresholded quantity and overlays
    the reference. For raw, the reference is each kernel's baseline. For
    excess, the reference is the zero line, which marks uniform firing.
    Candidate thresholds are drawn as horizontal lines.

    Right panel sweeps the threshold and plots the number of features called
    local. A flat region means the partition does not change across that
    range, so any value there is equivalent and the choice is robust. A
    smooth slope with no plateau means the partition is sensitive, so the
    value must be reported and defended. A candidate sitting on a step edge
    is the least robust choice.

    Parameters
    ----------
    profile : dict
        Output from localization_profile.
    method : str, default="excess"
        ``excess`` thresholds mass above the receptive-field-specific uniform
        baseline and is the recommended dilation-fair diagnostic. ``raw`` is
        retained for direct inspection of ``mass_in_rf``.
    candidate_fracs : tuple of float
        Raw thresholds to mark. Used when method="raw".
    candidate_margins : tuple of float
        Excess margins to mark. Used when method="excess".
    exclude_no_signal : bool, default=True
        Drop kernels with mass_in_rf 0 from the threshold analysis. These
        are no-signal kernels, not global ones.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    if method not in ("raw", "excess"):
        raise ValueError(f"method must be 'raw' or 'excess', got {method!r}")

    mass_all = np.asarray(profile["mass_in_rf"], dtype=float)
    baseline_all = np.asarray(profile["baseline"], dtype=float)

    if exclude_no_signal:
        keep = mass_all > 0.0
    else:
        keep = np.ones(len(mass_all), dtype=bool)
    n_no_signal = int((~keep).sum())
    mass = mass_all[keep]
    baseline = baseline_all[keep]
    n = len(mass)
    palette = [OI[3], OI[0], OI[2], OI[4]]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    rank = np.arange(n)

    if n == 0:
        ax_l.text(0.5, 0.5, "No firing kernels to rank",
                  ha="center", va="center", transform=ax_l.transAxes)
        ax_r.text(0.5, 0.5, f"All {n_no_signal} kernels are no-signal",
                  ha="center", va="center", transform=ax_r.transAxes)
        fig.tight_layout()
        return fig

    if method == "raw":
        quantity = mass
        candidates = candidate_fracs
        sweep = np.linspace(0.0, 1.0, 201)
        x_label = "localization_frac"
        q_label = "mass_in_rf"
        y_label = "Fraction of differential mass in RF"
        left_title = "Ranked localization vs baseline"
    else:
        quantity = mass - baseline
        candidates = candidate_margins
        lo = float(min(quantity.min(), 0.0)) - 0.05
        hi = float(quantity.max()) + 0.05
        sweep = np.linspace(lo, hi, 201)
        x_label = "margin (mass - baseline)"
        q_label = "excess (mass - baseline)"
        y_label = "Excess of mass over uniform baseline"
        left_title = "Ranked excess over baseline"

    order = np.argsort(quantity)
    q_sorted = quantity[order]

    # Left: ranked quantity with reference.
    ax_l.plot(rank, q_sorted, "o-", color="#2c2c2c", markersize=4,
              linewidth=1, label=q_label, zorder=3)
    if method == "raw":
        ax_l.plot(rank, baseline[order], "s", color="#999999", markersize=4,
                  alpha=0.8, label="uniform-fire baseline", zorder=2)
        ax_l.set_ylim(0, 1.02)
    else:
        ax_l.axhline(0.0, color="#999999", linestyle="-", linewidth=1,
                     alpha=0.8, label="uniform firing (excess 0)", zorder=2)
    for c, thr in zip(palette, candidates):
        n_local = int((quantity >= thr).sum())
        ax_l.axhline(thr, color=c, linestyle="--", linewidth=1.2,
                     label=f"thr={thr:g} -> {n_local} local")
    ax_l.set_xlabel(f"Feature rank (by {q_label})")
    ax_l.set_ylabel(y_label)
    if exclude_no_signal and n_no_signal > 0:
        ax_l.set_title(f"{left_title}  ({n} firing, {n_no_signal} no-signal)")
    else:
        ax_l.set_title(left_title)
    ax_l.grid(True, alpha=0.15)
    ax_l.legend(fontsize=8, loc="upper left")

    # Right: count of local features across the threshold sweep.
    counts = np.array([(quantity >= g).sum() for g in sweep])
    ax_r.plot(sweep, counts, color="#2c2c2c", linewidth=1.5)
    for c, thr in zip(palette, candidates):
        ax_r.axvline(thr, color=c, linestyle="--", linewidth=1.2,
                     label=f"thr={thr:g}")
    ax_r.set_xlabel(x_label)
    ax_r.set_ylabel("Number of features called local")
    ax_r.set_title("Local count vs threshold")
    ax_r.set_xlim(sweep[0], sweep[-1])
    ax_r.set_ylim(0, n + 0.5)
    ax_r.grid(True, alpha=0.15)
    ax_r.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig
