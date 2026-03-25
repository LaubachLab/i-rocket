"""
tshap_integration.py -- TSHAP Integration for I-ROCKET

Bridges I-ROCKET with the TSHAP package (Le Nguyen and Ifrim, 2025)
for instance-level Shapley value attributions on time series.

TSHAP provides exact SHAP values by grouping timepoints into sliding
windows, making Shapley computation tractable for long time series.
This module wraps I-ROCKET's prediction pipeline into the format
TSHAP expects and provides utilities for extracting, aggregating,
and comparing TSHAP attributions with I-ROCKET's built-in
interpretability tools.

Requirements:
    pip install tshap

References:
    Le Nguyen, T. and Ifrim, G. (2025). TSHAP: Fast and Exact SHAP
    for Explaining Time Series Classification and Regression.
    ECML-PKDD 2025.

License: BSD-3-Clause
"""

import numpy as np


# ============================================================================
# PREDICTION WRAPPERS
# ============================================================================

def make_predict_fn(model, mode='confidence', target_class=None):
    """
    Create a TSHAP-compatible prediction function from an InterpRocket model.

    TSHAP expects a function that takes a batch of 3D arrays
    (n_samples, n_channels, n_timepoints) and returns a 1D array
    of scalar values (one per sample).

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    mode : str, default='confidence'
        How to produce scalar predictions:
        'confidence' -- max decision function value across classes.
            Higher values indicate stronger classification confidence.
        'target' -- decision function value for a specific class.
            Requires target_class parameter.
        'predict' -- predicted class label (integer).
            Less informative for SHAP but works as a fallback.
    target_class : int or str, optional
        Class to explain when mode='target'. Must be one of
        model.classes_.

    Returns
    -------
    predict_fn : callable
        Function mapping (n, 1, T) arrays to (n,) scalar arrays.
    """
    if mode == 'target' and target_class is None:
        raise ValueError("target_class required when mode='target'")

    if mode == 'target':
        target_idx = np.where(model.classes_ == target_class)[0]
        if len(target_idx) == 0:
            raise ValueError(
                f"target_class={target_class} not in model.classes_="
                f"{model.classes_}"
            )
        target_idx = target_idx[0]

    def predict_fn(X_batch):
        # TSHAP sends (n, n_channels, T); reshape to (n, T) for I-ROCKET
        X_2d = X_batch.reshape(X_batch.shape[0], -1).astype(np.float32)
        features = model.transform(X_2d)
        features_scaled = model.scaler_.transform(features)
        decision = model.classifier_.decision_function(features_scaled)

        if mode == 'confidence':
            if decision.ndim == 1:
                return decision
            return decision.max(axis=1)
        elif mode == 'target':
            if decision.ndim == 1:
                return decision
            return decision[:, target_idx]
        elif mode == 'predict':
            return model.classifier_.predict(features_scaled).astype(float)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return predict_fn


# ============================================================================
# TSHAP SALIENCY EXTRACTION
# ============================================================================

def extract_tshap_saliency(model, X, window_length=None, stride=None,
                            n_background=20, mode='confidence',
                            target_class=None, random_state=42,
                            verbose=True):
    """
    Compute TSHAP attributions for I-ROCKET predictions.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
        Samples to explain.
    window_length : int, optional
        TSHAP window size. Default: n_timepoints // 10.
    stride : int, optional
        TSHAP stride. Default: window_length // 3.
    n_background : int, default=20
        Number of background samples for Shapley computation.
    mode : str, default='confidence'
        Prediction mode (see make_predict_fn).
    target_class : int or str, optional
        Class to explain when mode='target'.
    random_state : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'window_attributions' : ndarray, shape (n_samples, n_timepoints)
            Per-timepoint TSHAP Window attributions.
        'roi_attributions' : ndarray, shape (n_samples, n_timepoints)
            Per-timepoint TSHAP ROI attributions.
        'mean_window' : ndarray, shape (n_timepoints,)
            Mean absolute window attribution (dataset-level saliency).
        'mean_roi' : ndarray, shape (n_timepoints,)
            Mean absolute ROI attribution (dataset-level saliency).
        'window_length' : int
        'stride' : int
        'n_background' : int
        'mode' : str
    """
    from tshap.tshap import TSHAPExplainer

    X = np.asarray(X, dtype=np.float32)
    n_samples, n_timepoints = X.shape

    if window_length is None:
        window_length = max(5, n_timepoints // 10)
    if stride is None:
        stride = max(1, window_length // 3)

    # Reshape to 3D for TSHAP: (n, 1, T)
    X_3d = X[:, np.newaxis, :]

    # Select background samples
    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(n_samples, size=min(n_background, n_samples),
                         replace=False)
    baselines = X_3d[bg_idx]

    predict_fn = make_predict_fn(model, mode=mode, target_class=target_class)

    if verbose:
        print(f"TSHAP: {n_samples} samples, window={window_length}, "
              f"stride={stride}, {len(baselines)} background samples")

    explainer = TSHAPExplainer(
        window_length=window_length,
        stride=stride,
        interpolation=True,
        roi=True,
    )

    window_exp, roi_exp = explainer.explain(X_3d, baselines, predict_fn)

    # Squeeze channel dimension: (n, 1, T) -> (n, T)
    window_attr = window_exp.squeeze(axis=1)
    roi_attr = roi_exp.squeeze(axis=1)

    # Dataset-level saliency: mean absolute attribution
    mean_window = np.abs(window_attr).mean(axis=0)
    mean_roi = np.abs(roi_attr).mean(axis=0)

    # Normalize to [0, 1]
    if mean_window.max() > 0:
        mean_window /= mean_window.max()
    if mean_roi.max() > 0:
        mean_roi /= mean_roi.max()

    if verbose:
        print(f"  Window attr range: [{window_attr.min():.4f}, "
              f"{window_attr.max():.4f}]")
        print(f"  ROI attr range: [{roi_attr.min():.4f}, "
              f"{roi_attr.max():.4f}]")

    return {
        'window_attributions': window_attr,
        'roi_attributions': roi_attr,
        'mean_window': mean_window,
        'mean_roi': mean_roi,
        'window_length': window_length,
        'stride': stride,
        'n_background': len(baselines),
        'mode': mode,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_tshap_attributions(X, y, tshap_results, n_show=3, sample_indices=None,
                             figsize=(12, None)):
    """
    Plot TSHAP attributions for individual instances.

    Each row shows one sample: the signal with TSHAP Window and ROI
    attributions overlaid as colored backgrounds.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    tshap_results : dict
        Output from extract_tshap_saliency().
    n_show : int
        Number of instances to display.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    TAB10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    if sample_indices is not None:
        sample_indices = np.asarray(sample_indices)
        n_show = len(sample_indices)
    else:
        sample_indices = np.arange(min(n_show, len(X)))

    if figsize[1] is None:
        figsize = (figsize[0], 3 * n_show)

    window_attr = tshap_results['window_attributions']
    roi_attr = tshap_results['roi_attributions']
    n_timepoints = X.shape[1]
    t = np.arange(n_timepoints)

    # Colormap for attributions
    cmap = LinearSegmentedColormap.from_list(
        'bwr_custom', ['#1f77b4', '#f0f0f0', '#d62728']
    )

    fig, axes = plt.subplots(n_show, 2, figsize=figsize, sharex=True)
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i, si in enumerate(sample_indices):
        # Window attributions
        ax_w = axes[i, 0]
        vmax = max(np.abs(window_attr[si]).max(), 1e-6)
        for ti in range(n_timepoints - 1):
            color = cmap((window_attr[si, ti] / vmax + 1) / 2)
            ax_w.axvspan(ti, ti + 1, color=color, alpha=0.5)
        ax_w.plot(t, X[si], color='#2c2c2c', linewidth=1)
        ax_w.set_ylabel(f'Sample {si}\n(class {y[si]})', fontsize=9)
        if i == 0:
            ax_w.set_title('TSHAP Window')
        ax_w.grid(True, alpha=0.1)

        # ROI attributions
        ax_r = axes[i, 1]
        vmax_r = max(np.abs(roi_attr[si]).max(), 1e-6)
        for ti in range(n_timepoints - 1):
            color = cmap((roi_attr[si, ti] / vmax_r + 1) / 2)
            ax_r.axvspan(ti, ti + 1, color=color, alpha=0.5)
        ax_r.plot(t, X[si], color='#2c2c2c', linewidth=1)
        if i == 0:
            ax_r.set_title('TSHAP ROI')
        ax_r.grid(True, alpha=0.1)

    axes[-1, 0].set_xlabel('Timepoint')
    axes[-1, 1].set_xlabel('Timepoint')
    plt.tight_layout()
    return fig


def plot_tshap_vs_irocket(X, y, tshap_results, irocket_saliency,
                           aggregate_diff=None, occlusion_saliency=None,
                           figsize=(12, 10)):
    """
    Compare I-ROCKET and TSHAP saliency maps side by side.

    Row 1: I-ROCKET temporal importance
    Row 2: I-ROCKET aggregate activation differential
    Row 3: I-ROCKET occlusion
    Row 4: TSHAP Window attribution
    Row 5: TSHAP ROI attribution

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    tshap_results : dict
        Output from extract_tshap_saliency().
    irocket_saliency : ndarray, shape (n_timepoints,)
        I-ROCKET temporal importance profile.
    aggregate_diff : ndarray, shape (n_timepoints,), optional
        Differential output from plot_aggregate_activation().
    occlusion_saliency : ndarray, shape (n_timepoints,), optional
        Output from extract_occlusion_saliency().
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    classes = np.unique(y)
    n_timepoints = X.shape[1]
    t = np.arange(n_timepoints)

    # Normalize aggregate_diff to [0, 1] if provided
    if aggregate_diff is not None:
        agg = np.abs(aggregate_diff)
        if agg.max() > 0:
            agg = agg / agg.max()
    else:
        agg = np.zeros(n_timepoints)

    if occlusion_saliency is None:
        occlusion_saliency = np.zeros(n_timepoints)

    saliency_maps = {
        'I-ROCKET temporal imp.': irocket_saliency,
        'I-ROCKET aggregate diff': agg,
        'I-ROCKET occlusion': occlusion_saliency,
        'TSHAP Window (mean |attr|)': tshap_results['mean_window'],
        'TSHAP ROI (mean |attr|)': tshap_results['mean_roi'],
    }

    colors = ['#1f77b4', '#2ca02c', '#8c564b', '#ff7f0e', '#d62728']
    n_rows = len(saliency_maps)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)

    for ax, (name, saliency), color in zip(axes, saliency_maps.items(),
                                            colors):
        for cls in classes:
            cls_mean = X[y == cls].mean(axis=0)
            ax.plot(t, cls_mean, color='#c7c7c7', linewidth=0.8, alpha=0.6)

        scale = X.std()
        ax.fill_between(t, 0, saliency * scale, alpha=0.3, color=color)
        ax.plot(t, saliency * scale, color=color, linewidth=1.5)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel('Timepoint')
    fig.suptitle('Saliency map comparison: I-ROCKET vs TSHAP',
                  fontsize=12, y=0.95)
    plt.tight_layout()
    return fig
