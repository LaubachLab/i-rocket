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
composite_activation_trace
plot_activation_trace
plot_kernel_similarity
plot_feature_stability
localization_profile
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
from matplotlib.colors import to_rgba

from interp_rocket import (
    compute_activation_map,
    OI,
    POOLING_COLORS,
    _validate_feature_index_array,
    _format_feature_label,
)

__all__ = [
    "plot_activation_map",
    "plot_kernel_pattern",
    "plot_class_mean_activation",
    "composite_activation_trace",
    "plot_activation_trace",
    "plot_kernel_similarity",
    "plot_feature_stability",
    "localization_profile",
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
        top = model.get_top_features()
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
    from interp_rocket import compute_activation_map

    classes = np.unique(y)
    # Force float64 for the mean regardless of X dtype. Smooth signals can
    # lose precision when a float32 class mean is first-differenced.
    # float32 is only needed at the compute_activation_map call.
    class_means = [X[y == cls].mean(axis=0).astype(np.float64) for cls in classes]

    profiles = []
    for cm in class_means:
        x_use = (np.diff(cm).astype(np.float32)
                 if f["representation"] == "diff"
                 else cm.astype(np.float32))
        _, act, t_idx = compute_activation_map(
            x_use,
            f["kernel_index"],
            f["dilation"],
            f["bias"],
            f["padding_mode"],
            f["representation"],
        )
        profile = np.zeros(n_timepoints, dtype=np.float64)
        for i, ti in enumerate(t_idx):
            center = int(round(ti))
            if f["representation"] == "diff":
                center = min(center + 1, n_timepoints - 1)
            if 0 <= center < n_timepoints:
                profile[center] = act[i]
        profiles.append(profile)

    diff_act = np.max(profiles, axis=0) - np.min(profiles, axis=0)
    total = diff_act.sum()

    if total > 0:
        t_grid = np.arange(n_timepoints, dtype=np.float64)
        peak_t = int(round(np.dot(t_grid, diff_act) / total))
        peak_t = max(0, min(peak_t, n_timepoints - 1))
        half = int(f["receptive_field"]) // 2
        lo = max(0, peak_t - half)
        hi = min(n_timepoints, peak_t + half + 1)
        mass_in_rf = diff_act[lo:hi].sum() / total
        centroid_valid = mass_in_rf >= localization_frac
    else:
        peak_t = n_timepoints // 2
        centroid_valid = False
        mass_in_rf = 0.0

    return peak_t, centroid_valid, f["receptive_field"] / 2.0, mass_in_rf


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
    figsize=None,
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
    figsize : tuple, optional
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
    from interp_rocket import compute_activation_map

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("X must be 2D and y must contain one label per row.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")

    # ---- Resolve feature ordering ----
    ranking = _resolve_feature_order(
        model, feature_mask=feature_mask, rank_order=rank_order
    )
    n_show = min(len(ranking), _validate_n_show(n_show))
    feat_indices = ranking[:n_show]
    decoded = [model.decode_feature_index(int(fi)) for fi in feat_indices]

    classes = np.unique(y)
    n_classes = len(classes)
    n_timepoints = X.shape[1]

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

    # ---- Pre-check: fires on any class mean ----
    fires_on_mean = []
    for f in decoded:
        fires = False
        for cls in classes:
            cm = X[y == cls].mean(axis=0).astype(np.float32)
            x_use = np.diff(cm) if f["representation"] == "diff" else cm
            _, act, _ = compute_activation_map(
                x_use,
                f["kernel_index"],
                f["dilation"],
                f["bias"],
                f["padding_mode"],
                f["representation"],
            )
            if act.max() > 0:
                fires = True
                break
        fires_on_mean.append(fires)

    # ---- Compute RF overlays once ----
    # When localized_features is given, it overrides the internal mass test.
    # A feature draws a centroid only if it is in that set. This lets an
    # external criterion such as excess over baseline drive the partition.
    if localized_features is not None:
        local_set = set(int(i) for i in localized_features)
    else:
        local_set = None

    peak_times = []
    rf_spans = []
    centroid_valids = []
    for f, fi in zip(decoded, feat_indices):
        peak_t, centroid_valid, rf_half, _ = _compute_differential_centroid(
            model, X, y, f, n_timepoints, localization_frac=localization_frac
        )
        if local_set is not None:
            centroid_valid = int(fi) in local_set
        peak_times.append(t_signal[peak_t])
        rf_spans.append(rf_half * dt)
        centroid_valids.append(centroid_valid)

    # ---- Build figure ----
    fig, axes = plt.subplots(1, n_classes, figsize=figsize, sharey=True)
    if n_classes == 1:
        axes = [axes]

    x_lo, x_hi = t_signal[0], t_signal[-1]

    for k, cls in enumerate(classes):
        mask = y == cls
        act_matrix = []
        labels = []

        for f in decoded:
            cm = X[mask].mean(axis=0)
            x_use = (np.diff(cm).astype(np.float32)
                     if f["representation"] == "diff"
                     else cm.astype(np.float32))
            _, act, t_idx = compute_activation_map(
                x_use,
                f["kernel_index"],
                f["dilation"],
                f["bias"],
                f["padding_mode"],
                f["representation"],
            )
            act_full = np.zeros(n_timepoints)
            for i, ti in enumerate(t_idx):
                tii = int(round(ti))
                if f["representation"] == "diff":
                    tii = min(tii + 1, n_timepoints - 1)
                if 0 <= tii < n_timepoints:
                    act_full[tii] = act[i]
            act_matrix.append(act_full)

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

    fig.suptitle(
        "Circle = differential activation centroid (shown only when temporally localized)  |  "
        "Line = kernel receptive field  |  "
        "† = subthreshold on class means (fires on some individual trials only)",
        fontsize=9, y=1.02,
    )
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

    Returns
    -------
    fig : matplotlib Figure
    """
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must be supplied together.")

    # --- Resolve feature ordering ---
    ranking = _resolve_feature_order(
        model, feature_mask=feature_mask, rank_order=rank_order
    )
    if rank_label is None:
        rank_label = "Feature rank"

    n_show = min(len(ranking), _validate_n_show(n_show))
    feat_indices = ranking[:n_show]
    decoded = [model.decode_feature_index(int(fi)) for fi in feat_indices]

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
    # localized_features, when given, overrides the internal mass test so an
    # external criterion such as excess over baseline drives the partition.
    if localized_features is not None:
        local_set = set(int(i) for i in localized_features)
    else:
        local_set = None

    for row, info in enumerate(decoded):
        kernel_weights = np.array(info["kernel_weights"])
        dilation = info["dilation"]
        kernel_len = len(kernel_weights)

        # Compute centroid position
        if has_signal:
            peak_t, centroid_valid, _, _ = _compute_differential_centroid(
                model, X_test, y_test, info, n_timepoints,
                localization_frac=localization_frac,
            )
        else:
            peak_t = n_timepoints // 2
            centroid_valid = False

        if local_set is not None:
            centroid_valid = int(feat_indices[row]) in local_set

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
    ax_kp.set_title(
        f"Kernel Weight Patterns ({n_show} features, sorted by {rank_label})"
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
        Vertical dashed line at this time value. Set to None to disable.
        Drawn against the resolved time axis (which may be sample indices
        if no time information was provided).
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

    if event_time is not None:
        ax.axvline(x=event_time, color="k", linestyle="--",
                   alpha=0.7, label="Event")

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
    """
    Compute the localization quantity mass_in_rf for a set of features.

    For each feature this returns the fraction of total differential
    activation mass that falls inside one receptive field of the centroid,
    along with the uniform-fire baseline for that kernel. The baseline is
    the receptive field width over the window length. A global kernel sits
    near its baseline. A localized kernel sits well above it. The threshold
    localization_frac separates the two.

    This is the quantity the activation-map and kernel-pattern plots
    threshold. Use plot_localization_diagnostic to choose a defensible
    localization_frac from these values.

    Parameters
    ----------
    model : fitted InterpRocket or StableRocketClassifier
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    feature_mask : array of int
        Feature indices to profile, usually the selected set.
    localization_frac : float, default=0.5
        Threshold used to fill the is_local column. Does not affect the
        mass_in_rf values themselves.

    Returns
    -------
    dict with keys
        'feature_index' : ndarray of int
        'mass_in_rf'    : ndarray of float, the localization quantity
        'baseline'      : ndarray of float, uniform-fire mass_in_rf
        'excess'        : ndarray of float, mass_in_rf - baseline
        'receptive_field' : ndarray of int
        'dilation'      : ndarray of int
        'pooling_op'    : list of str
        'has_signal'    : ndarray of bool, True when the kernel fires
            differentially on the class means at all (mass_in_rf > 0)
        'is_local'      : ndarray of bool at the given localization_frac
        'localization_frac' : float

    Three categories follow from these fields. A kernel with has_signal
    False does not fire differentially on the class means. It is a
    no-signal kernel, the dagger case, not a global kernel. Among kernels
    with has_signal True, those near their baseline are diffuse and those
    well above are local. Keep the no-signal kernels separate when judging
    localization, since mass_in_rf cannot tell a no-signal kernel from a
    diffuse one and both sit at the low end.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("X must be 2D and y must contain one label per row.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    feature_mask = _validate_feature_index_array(
        feature_mask,
        int(getattr(model, "n_output_features_", 0)),
        name="feature_mask",
    )
    n_timepoints = X.shape[1]

    mass = np.zeros(len(feature_mask), dtype=np.float64)
    baseline = np.zeros(len(feature_mask), dtype=np.float64)
    rf = np.zeros(len(feature_mask), dtype=int)
    dil = np.zeros(len(feature_mask), dtype=int)
    pool = []

    for i, fi in enumerate(feature_mask):
        f = model.decode_feature_index(int(fi))
        _, _, _, m = _compute_differential_centroid(
            model, X, y, f, n_timepoints, localization_frac=localization_frac
        )
        mass[i] = m
        rf[i] = int(f["receptive_field"])
        dil[i] = int(f["dilation"])
        pool.append(f["pooling_op"])
        win = min(rf[i], n_timepoints)
        baseline[i] = win / float(n_timepoints)

    return {
        "feature_index": feature_mask,
        "mass_in_rf": mass,
        "baseline": baseline,
        "excess": mass - baseline,
        "receptive_field": rf,
        "dilation": dil,
        "pooling_op": pool,
        "has_signal": mass > 0.0,
        "is_local": mass >= localization_frac,
        "localization_frac": localization_frac,
    }


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
