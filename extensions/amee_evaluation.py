"""
amee_evaluation.py -- AMEE-style Explanation Evaluation for I-ROCKET

Implements the perturbation-based evaluation framework from:
    Nguyen, T.T., Nguyen, T.L., and Ifrim, G. (2024).
    Robust explainer recommendation for time series classification.
    Data Mining and Knowledge Discovery, 38:3372-3413.

Given a saliency map (importance per timepoint) from any explainer,
this module evaluates its informativeness by perturbing the most
important time regions and measuring the resulting accuracy drop.
A larger drop indicates a more informative explanation.

Multiple perturbation strategies and baseline explainers (random,
inverse) are provided for robust comparison.

License: BSD-3-Clause
"""

import numpy as np
from sklearn.metrics import accuracy_score


# ============================================================================
# SALIENCY MAP EXTRACTION FROM I-ROCKET
# ============================================================================

def extract_temporal_importance(model, X, y, feature_mask=None, n_examples=20):
    """
    Extract the temporal importance profile as a saliency map.

    This is the differential method from plot_temporal_importance,
    returning the raw importance array without plotting.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    feature_mask : array-like of int, optional
    n_examples : int, default=20
        Examples per class for activation rate estimation.

    Returns
    -------
    saliency : ndarray, shape (n_timepoints,)
        Importance at each timepoint, normalized to [0, 1].
    """
    from interp_rocket import compute_activation_map

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes = np.unique(y)
    n_timepoints = X.shape[1]

    top_features = model.get_top_features(feature_mask=feature_mask)
    importance_profile = np.zeros(n_timepoints, dtype=np.float64)

    for finfo in top_features:
        ki = finfo['kernel_index']
        dil = finfo['dilation']
        rep = finfo['representation']
        bias = finfo['bias']
        imp = finfo['importance']

        class_act_maps = []
        for cls in classes:
            mask = y == cls
            X_cls = X[mask]
            n_ex = min(n_examples, len(X_cls))

            if rep == 'diff':
                X_use = np.diff(X_cls[:n_ex], axis=1).astype(np.float32)
            else:
                X_use = X_cls[:n_ex]

            act_map = np.zeros(n_timepoints, dtype=np.float64)
            count_map = np.zeros(n_timepoints, dtype=np.float64)

            for ex_idx in range(len(X_use)):
                conv_out, act, time_idx = compute_activation_map(
                    X_use[ex_idx], ki, np.int32(dil), np.float32(bias)
                )
                for t in range(len(act)):
                    center = int(round(time_idx[t]))
                    if rep == 'diff':
                        center = min(center + 1, n_timepoints - 1)
                    if 0 <= center < n_timepoints:
                        act_map[center] += act[t]
                        count_map[center] += 1.0

            valid = count_map > 0
            act_map[valid] /= count_map[valid]
            class_act_maps.append(act_map)

        class_act_array = np.array(class_act_maps)
        diff_map = np.max(class_act_array, axis=0) - np.min(class_act_array, axis=0)
        importance_profile += diff_map * imp

    if importance_profile.max() > 0:
        importance_profile /= importance_profile.max()

    return importance_profile


def extract_occlusion_saliency(model, X, y, feature_mask=None,
                                window_size=None, stride=None):
    """
    Extract temporal occlusion sensitivity as a saliency map.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    feature_mask : array-like of int, optional
    window_size : int, optional
    stride : int, optional

    Returns
    -------
    saliency : ndarray, shape (n_timepoints,)
        Occlusion sensitivity at each timepoint, normalized to [0, 1].
    """
    from interp_rocket import temporal_occlusion

    occ = temporal_occlusion(
        model, X, y, n_samples=min(50, len(X)),
        window_size=window_size, stride=stride,
        feature_mask=feature_mask, verbose=False,
    )

    mean_sens = np.mean(occ['sensitivities'], axis=0)

    if mean_sens.max() > 0:
        mean_sens /= mean_sens.max()

    return mean_sens


def random_saliency(n_timepoints, random_state=42):
    """
    Random baseline saliency map.

    Parameters
    ----------
    n_timepoints : int
    random_state : int

    Returns
    -------
    saliency : ndarray, shape (n_timepoints,)
    """
    rng = np.random.RandomState(random_state)
    s = rng.rand(n_timepoints)
    return s / s.max()


def inverse_saliency(saliency):
    """
    Inverse of a saliency map: highlights the least important regions.

    Parameters
    ----------
    saliency : ndarray, shape (n_timepoints,)

    Returns
    -------
    inv : ndarray, shape (n_timepoints,)
    """
    inv = 1.0 - saliency
    if inv.max() > 0:
        inv /= inv.max()
    return inv


# ============================================================================
# PERTURBATION STRATEGIES
# ============================================================================

def perturb_zero(X, mask):
    """Replace masked timepoints with zero."""
    X_pert = X.copy()
    X_pert[:, mask] = 0.0
    return X_pert


def perturb_mean(X, mask):
    """Replace masked timepoints with the global mean."""
    X_pert = X.copy()
    global_mean = X.mean()
    X_pert[:, mask] = global_mean
    return X_pert


def perturb_noise(X, mask, random_state=42):
    """Replace masked timepoints with Gaussian noise (same mean/std as data)."""
    rng = np.random.RandomState(random_state)
    X_pert = X.copy()
    X_pert[:, mask] = rng.normal(X.mean(), X.std(), size=(X.shape[0], mask.sum()))
    return X_pert


def perturb_inverse(X, mask):
    """Replace masked timepoints with the mean of unmasked timepoints."""
    X_pert = X.copy()
    unmasked_mean = X[:, ~mask].mean(axis=1, keepdims=True)
    X_pert[:, mask] = unmasked_mean
    return X_pert


PERTURBATION_METHODS = {
    'zero': perturb_zero,
    'mean': perturb_mean,
    'noise': perturb_noise,
    'inverse': perturb_inverse,
}


# ============================================================================
# AMEE EVALUATION
# ============================================================================

def evaluate_saliency(model, X_test, y_test, saliency,
                      fractions=None, perturbation='zero',
                      random_state=42):
    """
    Evaluate a saliency map by perturbing top-k% timepoints.

    For each fraction, the top-k% most important timepoints (according
    to the saliency map) are perturbed, and the accuracy is measured.
    A larger accuracy drop indicates a more informative saliency map.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X_test : ndarray, shape (n_samples, n_timepoints)
    y_test : array-like
    saliency : ndarray, shape (n_timepoints,)
        Importance at each timepoint (higher = more important).
    fractions : array-like of float, optional
        Fractions of timepoints to perturb (default: 0.05 to 0.50).
    perturbation : str
        Perturbation method: 'zero', 'mean', 'noise', or 'inverse'.
    random_state : int

    Returns
    -------
    results : dict with keys:
        'fractions' : ndarray
        'accuracies' : ndarray -- accuracy at each perturbation level
        'accuracy_drops' : ndarray -- drop from baseline at each level
        'baseline_accuracy' : float
        'auc_drop' : float -- area under the accuracy drop curve
        'perturbation' : str
    """
    if fractions is None:
        fractions = np.arange(0.05, 0.55, 0.05)
    fractions = np.asarray(fractions)

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)
    n_timepoints = X_test.shape[1]

    # Ensure saliency map matches the time series length
    if len(saliency) != n_timepoints:
        import warnings
        warnings.warn(
            f"Saliency length ({len(saliency)}) does not match "
            f"n_timepoints ({n_timepoints}). Resampling via interpolation. "
            f"Check that random_saliency() uses X_test.shape[1]."
        )
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(saliency))
        x_new = np.linspace(0, 1, n_timepoints)
        saliency = interp1d(x_old, saliency, kind='linear')(x_new)

    perturb_fn = PERTURBATION_METHODS[perturbation]

    # Baseline accuracy (no perturbation)
    baseline_acc = accuracy_score(y_test, model.predict(X_test))

    # Rank timepoints by importance (descending)
    ranked = np.argsort(-saliency)

    accuracies = np.zeros(len(fractions))

    for i, frac in enumerate(fractions):
        k = max(1, int(frac * n_timepoints))
        top_k_mask = np.zeros(n_timepoints, dtype=bool)
        top_k_mask[ranked[:k]] = True

        if perturbation == 'noise':
            X_pert = perturb_fn(X_test, top_k_mask, random_state=random_state)
        else:
            X_pert = perturb_fn(X_test, top_k_mask)

        y_pred = model.predict(X_pert)
        accuracies[i] = accuracy_score(y_test, y_pred)

    accuracy_drops = baseline_acc - accuracies

    # Anchor at (0, 0) for proper AUC computation:
    # zero perturbation means zero accuracy drop
    fractions_full = np.concatenate([[0.0], fractions])
    drops_full = np.concatenate([[0.0], accuracy_drops])

    # Area under the accuracy drop curve (higher = more informative)
    try:
        auc_drop = float(np.trapezoid(drops_full, fractions_full))
    except AttributeError:
        auc_drop = float(np.trapz(drops_full, fractions_full))

    return {
        'fractions': fractions,
        'accuracies': accuracies,
        'accuracy_drops': accuracy_drops,
        'baseline_accuracy': baseline_acc,
        'auc_drop': auc_drop,
        'perturbation': perturbation,
    }


def amee_evaluate(model, X_test, y_test, saliency_maps,
                  fractions=None,
                  perturbations=None,
                  random_state=42, verbose=True):
    """
    Full AMEE evaluation: compare multiple saliency maps across
    multiple perturbation strategies.

    Parameters
    ----------
    model : InterpRocket
        Fitted model.
    X_test : ndarray, shape (n_samples, n_timepoints)
    y_test : array-like
    saliency_maps : dict of str -> ndarray
        Named saliency maps to evaluate. Each value is shape (n_timepoints,).
    fractions : array-like of float, optional
    perturbations : list of str, optional
        Perturbation methods to use. Default: all four.
    random_state : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'per_explainer' : dict of str -> dict of str -> eval_results
            results[explainer_name][perturbation_method] = evaluate_saliency output
        'auc_summary' : dict of str -> dict of str -> float
            AUC drop for each explainer/perturbation combination
        'mean_auc' : dict of str -> float
            Mean AUC drop across perturbations (the AMEE ranking metric)
        'ranking' : list of (str, float)
            Explainers sorted by mean AUC drop (best first)
    """
    if perturbations is None:
        perturbations = list(PERTURBATION_METHODS.keys())

    if fractions is None:
        fractions = np.arange(0.05, 0.55, 0.05)

    per_explainer = {}
    auc_summary = {}

    for name, saliency in saliency_maps.items():
        per_explainer[name] = {}
        auc_summary[name] = {}

        for pert in perturbations:
            res = evaluate_saliency(
                model, X_test, y_test, saliency,
                fractions=fractions,
                perturbation=pert,
                random_state=random_state,
            )
            per_explainer[name][pert] = res
            auc_summary[name][pert] = res['auc_drop']

            if verbose:
                print(f"  {name:>25s} + {pert:<8s}: "
                      f"AUC drop = {res['auc_drop']:.4f}")

    # Mean AUC across perturbations
    mean_auc = {}
    for name in saliency_maps:
        mean_auc[name] = float(np.mean(list(auc_summary[name].values())))

    ranking = sorted(mean_auc.items(), key=lambda x: -x[1])

    if verbose:
        print(f"\nRanking (higher = more informative):")
        for rank, (name, auc) in enumerate(ranking, 1):
            print(f"  {rank}. {name}: mean AUC drop = {auc:.4f}")

    return {
        'per_explainer': per_explainer,
        'auc_summary': auc_summary,
        'mean_auc': mean_auc,
        'ranking': ranking,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_amee_results(results, figsize=(14, 5)):
    """
    Visualize AMEE evaluation results.

    Left: accuracy drop curves for each explainer (averaged across
    perturbation methods). Right: mean AUC drop ranking bar chart.

    Parameters
    ----------
    results : dict
        Output from amee_evaluate().
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

    per_explainer = results['per_explainer']
    ranking = results['ranking']
    explainer_names = [name for name, _ in ranking]
    n_explainers = len(explainer_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    gridspec_kw={'width_ratios': [2, 1]})

    # Left: accuracy drop curves (mean across perturbation methods)
    for i, name in enumerate(explainer_names):
        pert_results = per_explainer[name]
        fracs = list(pert_results.values())[0]['fractions']

        # Average accuracy drop across perturbations
        all_drops = np.array([r['accuracy_drops'] for r in pert_results.values()])
        mean_drop = all_drops.mean(axis=0)
        sem_drop = all_drops.std(axis=0) / np.sqrt(len(all_drops))

        color = TAB10[i % len(TAB10)]
        ax1.plot(fracs * 100, mean_drop, color=color, linewidth=1.5,
                 marker='o', markersize=4, label=name)
        ax1.fill_between(fracs * 100, mean_drop - sem_drop,
                          mean_drop + sem_drop,
                          alpha=0.15, color=color)

    ax1.set_xlabel('Timepoints perturbed (%)')
    ax1.set_ylabel('Accuracy drop')
    ax1.set_title('Perturbation sensitivity by explainer')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, fracs[-1] * 100 + 2)

    # Right: ranking bar chart
    names = [name for name, _ in ranking]
    aucs = [auc for _, auc in ranking]
    colors = [TAB10[explainer_names.index(n) % len(TAB10)] for n in names]

    ax2.barh(range(len(names)), aucs, color=colors, edgecolor='white',
             height=0.6)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Mean AUC drop')
    ax2.set_title('AMEE ranking')
    ax2.grid(True, alpha=0.2, axis='x')

    # for i, auc in enumerate(aucs):
    #     ax2.text(auc + 0.001, i, f'{auc:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    return fig


def plot_saliency_comparison(X, y, saliency_maps, figsize=(12, None)):
    """
    Plot saliency maps overlaid on class means for visual comparison.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_timepoints)
    y : array-like
    saliency_maps : dict of str -> ndarray
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

    n_maps = len(saliency_maps)
    if figsize[1] is None:
        figsize = (figsize[0], 2.5 * n_maps)

    classes = np.unique(y)
    n_timepoints = X.shape[1]
    t = np.arange(n_timepoints)

    fig, axes = plt.subplots(n_maps, 1, figsize=figsize, sharex=True)
    if n_maps == 1:
        axes = [axes]

    for i, (name, saliency) in enumerate(saliency_maps.items()):
        ax = axes[i]

        # Plot class means
        for k, cls in enumerate(classes):
            cls_mean = X[y == cls].mean(axis=0)
            ax.plot(t, cls_mean, color='#c7c7c7', linewidth=0.8, alpha=0.6)

        # Plot saliency as filled area
        color = TAB10[i % len(TAB10)]
        ax.fill_between(t, 0, saliency * X.std(), alpha=0.3, color=color)
        ax.plot(t, saliency * X.std(), color=color, linewidth=1.5)

        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim(bottom=min(0, X.min()))
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel('Timepoint')
    fig.suptitle('Saliency map comparison', fontsize=12)
    plt.tight_layout()
    return fig
