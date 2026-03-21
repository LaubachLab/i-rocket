"""
interp_rocket_regressor.py — Interpretable ROCKET for Time Series Regression

A parallel companion to interp_rocket.py (classification), providing the same
MultiRocket transform with full kernel-level interpretability for continuous
target variables. All feature extraction, decoding, and kernel visualization
infrastructure is shared with the classifier; only the downstream estimator,
metrics, and target-aware visualizations differ.

ARCHITECTURE:
    Identical to InterpRocket: 84 base kernels × D dilations × 2
    representations (raw, first-difference) × 4 pooling ops (PPV, MPV,
    MIPV, LSPV). The feature matrix is passed to RidgeCV (regression)
    instead of RidgeClassifierCV.

KEY DIFFERENCES FROM InterpRocket (classifier):
    - Estimator: RidgeCV instead of RidgeClassifierCV
    - score() returns R² (sklearn RegressorMixin convention)
    - evaluate() returns R², MSE, RMSE, MAE, explained variance, Pearson r
    - No class_weight / oversampling (continuous target)
    - Cross-validation uses RepeatedKFold (not stratified)
    - Visualizations show target correlations instead of per-class activations
    - No mutual information or confusion matrices

USAGE:
    from interp_rocket_regressor import InterpRocketRegressor, cross_validate

    model = InterpRocketRegressor(max_dilations_per_kernel=40)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    # Visualization
    model.plot_top_kernels(X_test, y_test, n_kernels=5)
    model.plot_temporal_importance(X_test, y_test)
    model.plot_feature_distributions(X_test, y_test)
    model.plot_kernel_properties()

    # Recursive feature elimination
    from interp_rocket_regressor import recursive_feature_elimination
    from interp_rocket_regressor import plot_elimination_curve
    rfe = recursive_feature_elimination(model, X_train, y_train, X_test, y_test)
    plot_elimination_curve(rfe)

    # Cross-validation
    results = cross_validate(X, y, n_repeats=10, n_folds=10, n_jobs=-2)

REQUIREMENTS:
    numpy, numba (>=0.50), scikit-learn, matplotlib
    Requires interp_rocket.py in the import path (shared transform engine).

Author: Mark Laubach (American University, Department of Neuroscience)
        Developed with Claude (Anthropic) as AI coding assistant.
License: BSD-3-Clause
"""

__version__ = "0.1.0"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

# Shared infrastructure from the classifier module.
# If interp_rocket is not installed as a package, look for it in the same
# directory as this file, or in the parent directory (e.g., when this file
# lives in an extensions/ subdirectory of the main repo).
try:
    from interp_rocket import (
        _generate_base_kernels, _fit_dilations, _quantiles,
        _fit_biases, _transform, compute_activation_map, kneedle,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    _this_dir = Path(__file__).resolve().parent
    for _candidate in [_this_dir, _this_dir.parent]:
        if (_candidate / "interp_rocket.py").exists():
            sys.path.insert(0, str(_candidate))
            break
    from interp_rocket import (
        _generate_base_kernels, _fit_dilations, _quantiles,
        _fit_biases, _transform, compute_activation_map, kneedle,
    )

# ============================================================================
# COLOR PALETTE (tab10 as hex, consistent with interp_rocket.py)
# ============================================================================

TAB10 = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# ============================================================================
# SECTION 1: REGRESSION METRICS
# ============================================================================

def _compute_regression_metrics(y_true, y_pred):
    """
    Compute all regression metrics.

    Returns
    -------
    metrics : dict with keys:
        'r2', 'adjusted_r2', 'mse', 'rmse', 'mae',
        'explained_variance', 'pearson_r'

    Notes
    -----
    adjusted_r2 is not computed here (requires n_features); it is added
    by evaluate() where the feature count is known.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = float(mean_squared_error(y_true, y_pred))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "pearson_r": r,
    }


# ============================================================================
# SECTION 2: THE INTERPRETABLE ROCKET REGRESSOR CLASS
# ============================================================================

class InterpRocketRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable MultiRocket regressor for time series.

    Provides full traceability from regressor decision → feature importance →
    kernel identity → temporal activation pattern. Identical transform engine
    to InterpRocket (classifier); differs only in the downstream estimator
    (RidgeCV) and evaluation metrics (R², MSE, etc.).

    Inherits from sklearn.base.BaseEstimator and RegressorMixin, providing
    get_params(), set_params(), and a standard score(X, y) that returns
    R² as a scalar for compatibility with sklearn pipelines. For the
    full multi-metric evaluation, use evaluate(X, y).

    Parameters
    ----------
    max_dilations_per_kernel : int, default=32
        Maximum number of dilation values per kernel.
    num_features : int, default=10000
        Target number of features per representation.
        Actual count: 2 representations × 4 pooling ops × (rounded to 84 multiple).
    random_state : int, default=0
        Seed for reproducibility (only affects bias fitting).
    alpha_range : ndarray, optional
        Range of Ridge regularization parameters.
    """

    # The 4 pooling operator names, in feature order
    POOLING_NAMES = ["PPV", "MPV", "MIPV", "LSPV"]

    def __init__(
        self,
        max_dilations_per_kernel=32,
        num_features=10000,
        random_state=0,
        alpha_range=None,
    ):
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.num_features = num_features
        self.random_state = random_state
        self.alpha_range = alpha_range or np.logspace(-10, 10, 20)

        # Will be set during fit()
        self.base_kernels_ = None
        self.base_indices_ = None
        self.dilations_raw_ = None
        self.dilations_diff_ = None
        self.num_features_per_dilation_raw_ = None
        self.num_features_per_dilation_diff_ = None
        self.biases_raw_ = None
        self.biases_diff_ = None
        self.regressor_ = None
        self.scaler_ = None
        self.n_features_per_rep_ = None

    def fit(self, X, y):
        """
        Fit the MultiRocket transform and Ridge regressor.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
            Training time series. Will be converted to float32.
        y : array-like, shape (n_instances,)
            Continuous target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)

        n_instances, input_length = X.shape
        print(
            f"InterpRocketRegressor.fit: {n_instances} instances "
            f"× {input_length} timepoints"
        )
        print(
            f"  Target range: [{y.min():.4f}, {y.max():.4f}], "
            f"mean={y.mean():.4f}, std={y.std():.4f}"
        )

        # Generate the 84 base kernels
        self.base_kernels_, self.base_indices_ = _generate_base_kernels()

        # --- Raw representation ---
        print("  Fitting dilations (raw)...")
        self.dilations_raw_, self.num_features_per_dilation_raw_ = _fit_dilations(
            input_length, self.num_features, self.max_dilations_per_kernel
        )

        n_features_raw = 84 * np.sum(self.num_features_per_dilation_raw_)
        quantiles_raw = _quantiles(n_features_raw)

        print(
            f"  Fitting biases (raw): {n_features_raw} biases across "
            f"{len(self.dilations_raw_)} dilations..."
        )
        self.biases_raw_ = _fit_biases(
            X,
            self.dilations_raw_,
            self.num_features_per_dilation_raw_,
            quantiles_raw,
            self.random_state,
        )

        # --- First-difference representation ---
        X_diff = np.diff(X, axis=1).astype(np.float32)
        diff_length = X_diff.shape[1]

        print("  Fitting dilations (diff)...")
        self.dilations_diff_, self.num_features_per_dilation_diff_ = _fit_dilations(
            diff_length, self.num_features, self.max_dilations_per_kernel
        )

        n_features_diff = 84 * np.sum(self.num_features_per_dilation_diff_)
        quantiles_diff = _quantiles(n_features_diff)

        print(
            f"  Fitting biases (diff): {n_features_diff} biases across "
            f"{len(self.dilations_diff_)} dilations..."
        )
        self.biases_diff_ = _fit_biases(
            X_diff,
            self.dilations_diff_,
            self.num_features_per_dilation_diff_,
            quantiles_diff,
            self.random_state + 1,
        )

        self.n_features_per_rep_ = (n_features_raw, n_features_diff)

        # --- Transform ---
        print("  Transforming training data...")
        X_features = self._transform(X)
        print(f"  Feature matrix: {X_features.shape}")

        # --- Standardize features ---
        print("  Standardizing features...")
        self.scaler_ = StandardScaler(with_mean=True)
        X_features = self.scaler_.fit_transform(X_features)

        # --- Fit regressor ---
        print("  Fitting RidgeCV...")
        self.regressor_ = RidgeCV(alphas=self.alpha_range)
        self.regressor_.fit(X_features, y)

        train_r2 = self.regressor_.score(X_features, y)
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Selected alpha: {self.regressor_.alpha_:.4f}")

        return self

    def _transform(self, X):
        """
        Transform time series to feature vectors.

        Returns concatenated features: [raw_PPV, raw_MPV, raw_MIPV, raw_LSPV,
                                         diff_PPV, diff_MPV, diff_MIPV, diff_LSPV]
        """
        X = np.asarray(X, dtype=np.float32)

        features_raw = _transform(
            X,
            self.dilations_raw_,
            self.num_features_per_dilation_raw_,
            self.biases_raw_,
        )

        X_diff = np.diff(X, axis=1).astype(np.float32)
        features_diff = _transform(
            X_diff,
            self.dilations_diff_,
            self.num_features_per_dilation_diff_,
            self.biases_diff_,
        )

        return np.concatenate([features_raw, features_diff], axis=1)

    def transform(self, X):
        """Public transform method. Returns raw (unscaled) features."""
        return self._transform(X)

    def predict(self, X):
        """Predict continuous target values."""
        X_features = self._transform(X)
        X_features = self.scaler_.transform(X_features)
        return self.regressor_.predict(X_features)

    def score(self, X, y):
        """
        Return R² as a scalar (sklearn RegressorMixin convention).

        For the full multi-metric evaluation, use evaluate(X, y).
        """
        y_pred = self.predict(X)
        return float(r2_score(np.asarray(y), y_pred))

    def evaluate(self, X, y):
        """
        Evaluate on test data, returning multiple regression metrics.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like

        Returns
        -------
        metrics : dict with keys:
            'r2', 'mse', 'rmse', 'mae', 'explained_variance', 'pearson_r'
        """
        y_pred = self.predict(X)
        return _compute_regression_metrics(np.asarray(y), y_pred)

    # ====================================================================
    # SECTION 3: FEATURE INDEX DECODING (shared logic)
    # ====================================================================

    def decode_feature_index(self, feature_index):
        """
        Map a feature index back to its generating components.

        This is the core interpretability function: given a column index in
        the feature matrix, returns exactly which kernel, dilation, pooling
        operator, and signal representation produced it.

        Parameters
        ----------
        feature_index : int
            Column index in the feature matrix.

        Returns
        -------
        info : dict with keys:
            'representation': str, 'raw' or 'diff'
            'kernel_index': int, 0-83
            'kernel_weights': ndarray, the 9 weights
            'kernel_positive_indices': ndarray, the 3 positions with weight 2
            'dilation': int
            'receptive_field': int, total span in timepoints
            'bias': float
            'pooling_op': str, one of 'PPV', 'MPV', 'MIPV', 'LSPV'
            'pooling_index': int, 0-3
        """
        n_raw_total = self.n_features_per_rep_[0] * 4
        n_diff_total = self.n_features_per_rep_[1] * 4

        if feature_index < n_raw_total:
            rep = "raw"
            local_idx = feature_index
            dilations = self.dilations_raw_
            nfpd = self.num_features_per_dilation_raw_
            biases = self.biases_raw_
        else:
            rep = "diff"
            local_idx = feature_index - n_raw_total
            dilations = self.dilations_diff_
            nfpd = self.num_features_per_dilation_diff_
            biases = self.biases_diff_

        bias_index = local_idx // 4
        pooling_index = local_idx % 4

        remaining = bias_index
        found = False
        dilation_val = 0
        kernel_idx = 0

        for d_idx in range(len(dilations)):
            n_features_this_dil = nfpd[d_idx]
            n_biases_this_dil = 84 * n_features_this_dil

            if remaining < n_biases_this_dil:
                dilation_val = int(dilations[d_idx])
                kernel_idx = remaining // n_features_this_dil
                found = True
                break
            remaining -= n_biases_this_dil

        if not found:
            raise ValueError(f"Feature index {feature_index} out of range")

        receptive_field = 1 + 8 * dilation_val

        return {
            "representation": rep,
            "kernel_index": kernel_idx,
            "kernel_weights": self.base_kernels_[kernel_idx].copy(),
            "kernel_positive_indices": self.base_indices_[kernel_idx].copy(),
            "dilation": dilation_val,
            "receptive_field": receptive_field,
            "bias": float(biases[bias_index]),
            "pooling_op": self.POOLING_NAMES[pooling_index],
            "pooling_index": pooling_index,
        }

    def get_feature_importance(self, feature_mask=None):
        """
        Get per-feature importance from the Ridge regressor.

        For regression, importance is the absolute value of the Ridge
        coefficient, normalized so the maximum is 1.0. This parallels
        the classifier's approach (L2 norm across classes for multi-class;
        absolute value for binary / regression).

        Parameters
        ----------
        feature_mask : array-like of int, optional
            If provided, only these feature indices are eligible.
            All other features receive importance = 0.

        Returns
        -------
        importance : ndarray, shape (n_features,)
        """
        coefs = self.regressor_.coef_
        importance = np.abs(coefs)

        if feature_mask is not None:
            mask = np.zeros_like(importance)
            feature_mask = np.asarray(feature_mask)
            valid = feature_mask[feature_mask < len(importance)]
            mask[valid] = importance[valid]
            importance = mask

        if importance.max() > 0:
            importance = importance / importance.max()

        return importance

    def get_top_features(self, n=None, feature_mask=None):
        """
        Get the n most important features, fully decoded.

        Parameters
        ----------
        n : int
            Number of top features to return.
        feature_mask : array-like of int, optional

        Returns
        -------
        top_features : list of dicts
            Each dict has the decode_feature_index output plus 'importance'
            and 'feature_index'.
        """
        importance = self.get_feature_importance(feature_mask=feature_mask)
        if n is None:
            if feature_mask is not None:
                n = len(feature_mask)
            else:
                n = 20
        top_idx = np.argsort(importance)[::-1][:n]

        results = []
        for idx in top_idx:
            info = self.decode_feature_index(int(idx))
            info["importance"] = float(importance[idx])
            info["feature_index"] = int(idx)
            results.append(info)

        return results

    # ====================================================================
    # SECTION 4: VISUALIZATION
    # ====================================================================

    def plot_top_kernels(
        self, X, y, n_kernels=None, n_examples=3, figsize=None,
        feature_mask=None,
    ):
        """
        Visualize the top-n most important kernels for regression.

        Layout: one row per kernel.
            Column 0: kernel weight pattern (bar chart at dilated positions).
            Column 1: scatter of mean activation value vs target, showing
                how this kernel's output correlates with the target.
            Column 2: per-timepoint correlation between activation and target,
                showing WHERE in the series this kernel's firing relates
                to the target variable.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like, continuous target
        n_kernels : int or None
        n_examples : int
            Not used directly (kept for API consistency with classifier).
        figsize : tuple, optional
        feature_mask : array-like of int, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        n_timepoints = X.shape[1]
        n_instances = X.shape[0]

        if n_kernels is None:
            if feature_mask is not None:
                n_kernels = len(feature_mask)
            else:
                n_kernels = 5

        if feature_mask is not None:
            n_candidates = len(feature_mask)
        else:
            n_candidates = n_kernels * 4
        top_features = self.get_top_features(
            n=n_candidates, feature_mask=feature_mask
        )

        # Deduplicate: unique (kernel_index, dilation, representation)
        seen = set()
        unique_kernels = []
        for f in top_features:
            key = (f["representation"], f["kernel_index"], f["dilation"])
            if key not in seen:
                seen.add(key)
                unique_kernels.append(f)
            if len(unique_kernels) >= n_kernels:
                break

        n_kernels = len(unique_kernels)
        n_cols = 3  # weights | feature vs target | temporal correlation
        if figsize is None:
            figsize = (4.5 * n_cols, 3.5 * n_kernels)

        fig, axes = plt.subplots(
            n_kernels,
            n_cols,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 3, 3]},
        )
        if n_kernels == 1:
            axes = axes.reshape(1, -1)

        for row, kinfo in enumerate(unique_kernels):
            ki = kinfo["kernel_index"]
            dil = kinfo["dilation"]
            rep = kinfo["representation"]
            bias = kinfo["bias"]
            imp = kinfo["importance"]
            pooling = kinfo["pooling_op"]

            # --- Column 0: Kernel weight pattern ---
            ax = axes[row, 0]
            weights = kinfo["kernel_weights"]
            positions = np.arange(9) * dil
            ax.bar(
                positions,
                weights,
                width=max(dil * 0.6, 0.6),
                color="#7f7f7f",
                edgecolor="#2c2c2c",
                linewidth=0.5,
            )
            ax.set_title(
                f"Kernel {ki} (d={dil}, {rep})\nimp={imp:.4f} [{pooling}]",
                fontsize=9,
            )
            ax.set_xlabel("Dilated position")
            ax.axhline(0, color="#7f7f7f", linewidth=0.5)
            ax.set_ylabel("Weight")

            # --- Compute feature values for this kernel ---
            # Get the pooling feature value for each instance
            features = self._transform(X)
            features_scaled = self.scaler_.transform(features)

            # Find the specific feature index for this kernel+dilation+pooling
            fi = kinfo["feature_index"]
            feature_vals = features_scaled[:, fi]

            # --- Column 1: Feature value vs target scatter ---
            ax = axes[row, 1]
            ax.scatter(feature_vals, y, s=8, alpha=0.4, color="#1f77b4",
                       edgecolors="none")
            # Regression line
            if np.std(feature_vals) > 0:
                m, b = np.polyfit(feature_vals, y, 1)
                x_line = np.linspace(feature_vals.min(), feature_vals.max(), 50)
                ax.plot(x_line, m * x_line + b, color="#d62728", linewidth=1.5)
                r_val = np.corrcoef(feature_vals, y)[0, 1]
                ax.set_title(f"Feature vs Target (r={r_val:.3f})", fontsize=9)
            else:
                ax.set_title("Feature vs Target (constant)", fontsize=9)
            ax.set_xlabel(f"{pooling} (scaled)")
            ax.set_ylabel("Target")

            # --- Column 2: Per-timepoint activation–target correlation ---
            ax = axes[row, 2]

            corr_by_time = np.zeros(n_timepoints, dtype=np.float64)
            count_by_time = np.zeros(n_timepoints, dtype=np.float64)

            # Accumulate activation values per timepoint across all instances
            act_by_time = np.zeros((n_instances, n_timepoints), dtype=np.float64)

            for ex_idx in range(n_instances):
                if rep == "diff":
                    x = np.diff(X[ex_idx]).astype(np.float32)
                else:
                    x = X[ex_idx]

                conv_out, act, time_idx = compute_activation_map(
                    x, ki, np.int32(dil), np.float32(bias)
                )

                for t in range(len(act)):
                    center = int(round(time_idx[t]))
                    if rep == "diff":
                        center = min(center + 1, n_timepoints - 1)
                    if 0 <= center < n_timepoints:
                        act_by_time[ex_idx, center] = conv_out[t]
                        count_by_time[center] += 1.0

            # Compute correlation at each timepoint
            for t in range(n_timepoints):
                if count_by_time[t] >= 3:
                    col = act_by_time[:, t]
                    if np.std(col) > 0:
                        corr_by_time[t] = np.corrcoef(col, y)[0, 1]

            ax.plot(range(n_timepoints), corr_by_time, color="#2ca02c",
                    linewidth=1.2)
            ax.axhline(0, color="#7f7f7f", linewidth=0.5, alpha=0.5)
            ax.set_xlabel("Timepoint")
            ax.set_ylabel("Correlation with target")
            ax.set_title("Temporal activation–target correlation", fontsize=9)
            ax.set_ylim(-1.05, 1.05)

        plt.tight_layout()
        return fig

    def plot_temporal_importance(
        self,
        X,
        y,
        n_top=None,
        n_examples=None,
        figsize=(14, 6),
        feature_mask=None,
    ):
        """
        Aggregate activation maps of top features to show which time regions
        are most important for predicting the target.

        For regression, importance at each timepoint is the absolute
        correlation between the kernel's activation and the target,
        weighted by the feature's Ridge coefficient importance. This
        highlights WHERE temporal patterns most strongly predict the
        continuous outcome.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like, continuous target
        n_top : int or None
        n_examples : int or None
            Maximum instances to use for correlation computation.
            None uses all instances.
        figsize : tuple
        feature_mask : array-like of int, optional

        Returns
        -------
        fig : matplotlib Figure
        importance_by_time : ndarray, shape (n_timepoints,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        n_instances = X.shape[0]
        n_timepoints = X.shape[1]

        if n_examples is not None:
            n_use = min(n_examples, n_instances)
        else:
            n_use = n_instances

        if n_top is None:
            if feature_mask is not None:
                n_top = len(feature_mask)
            else:
                n_top = 50

        top_features = self.get_top_features(n=n_top, feature_mask=feature_mask)

        importance_by_time = np.zeros(n_timepoints, dtype=np.float64)
        abs_corr_by_time = np.zeros(n_timepoints, dtype=np.float64)

        for finfo in top_features:
            ki = finfo["kernel_index"]
            dil = finfo["dilation"]
            rep = finfo["representation"]
            bias = finfo["bias"]
            imp = finfo["importance"]

            # Accumulate activation values per timepoint
            act_by_time = np.zeros((n_use, n_timepoints), dtype=np.float64)
            count_by_time = np.zeros(n_timepoints, dtype=np.float64)

            for ex_idx in range(n_use):
                if rep == "diff":
                    x = np.diff(X[ex_idx]).astype(np.float32)
                else:
                    x = X[ex_idx]

                conv_out, act, time_idx = compute_activation_map(
                    x, ki, np.int32(dil), np.float32(bias)
                )

                for t in range(len(act)):
                    center = int(round(time_idx[t]))
                    if rep == "diff":
                        center = min(center + 1, n_timepoints - 1)
                    if 0 <= center < n_timepoints:
                        act_by_time[ex_idx, center] = conv_out[t]
                        count_by_time[center] += 1.0

            # Compute |correlation| at each timepoint, weighted by importance
            for t in range(n_timepoints):
                if count_by_time[t] >= 3:
                    col = act_by_time[:n_use, t]
                    if np.std(col) > 0:
                        r = np.corrcoef(col, y[:n_use])[0, 1]
                        importance_by_time[t] += imp * abs(r)

        # Normalize
        if importance_by_time.max() > 0:
            importance_by_time /= importance_by_time.max()

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Top panel: mean signal with target-colored scatter
        ax = axes[0]
        mean_signal = X[:n_use].mean(axis=0)
        ax.plot(range(n_timepoints), mean_signal, color="#7f7f7f", linewidth=1.0,
                label="Mean signal")
        ax.set_ylabel("Amplitude")
        ax.set_title("Mean Signal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Bottom panel: importance profile
        ax = axes[1]
        ax.fill_between(
            range(n_timepoints), importance_by_time,
            alpha=0.3, color="#1f77b4",
        )
        ax.plot(
            range(n_timepoints), importance_by_time,
            color="#1f77b4", linewidth=1.2,
        )
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Importance\n(weighted |correlation|)")
        ax.set_title("Temporal Importance for Regression")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.2)

        fig.suptitle("Temporal Importance Profile (Regression)", fontsize=13,
                      y=1.01)
        plt.tight_layout()
        return fig, importance_by_time

    def plot_feature_distributions(
        self, X, y, n_features=12, figsize=(14, 10), feature_mask=None,
    ):
        """
        Scatter plots of top features against the continuous target.

        Parameters
        ----------
        X : ndarray, shape (n_instances, n_timepoints)
        y : array-like, continuous target
        n_features : int
        figsize : tuple
        feature_mask : array-like of int, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)

        top_features = self.get_top_features(n=n_features,
                                              feature_mask=feature_mask)
        features = self._transform(X)
        features_scaled = self.scaler_.transform(features)

        n_cols = min(4, n_features)
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, finfo in enumerate(top_features):
            if i >= len(axes):
                break
            ax = axes[i]
            fi = finfo["feature_index"]
            fvals = features_scaled[:, fi]

            ax.scatter(fvals, y, s=8, alpha=0.4, color="#1f77b4",
                       edgecolors="none")

            # Regression line
            if np.std(fvals) > 0:
                m, b = np.polyfit(fvals, y, 1)
                x_line = np.linspace(fvals.min(), fvals.max(), 50)
                ax.plot(x_line, m * x_line + b, color="#d62728",
                        linewidth=1.2, alpha=0.8)
                r_val = np.corrcoef(fvals, y)[0, 1]
                r_label = f"r={r_val:.3f}"
            else:
                r_label = "constant"

            ki = finfo["kernel_index"]
            dil = finfo["dilation"]
            pooling = finfo["pooling_op"]
            rep = finfo["representation"]
            imp = finfo["importance"]

            ax.set_title(
                f"K{ki} d={dil} {pooling} ({rep})\n"
                f"imp={imp:.3f}  {r_label}",
                fontsize=8,
            )
            ax.set_xlabel(f"{pooling} (scaled)", fontsize=7)
            ax.set_ylabel("Target", fontsize=7)
            ax.tick_params(labelsize=6)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature–Target Distributions (Regression)", fontsize=12,
                      y=1.01)
        plt.tight_layout()
        return fig

    def plot_kernel_properties(self, n_top=100, figsize=(14, 8)):
        """
        Compare properties (dilation, receptive field, pooling op distribution)
        of top vs. bottom features.

        Identical to the classifier version — importance ranking is the same
        concept regardless of whether the downstream task is classification
        or regression.
        """
        importance = self.get_feature_importance()
        n_features = len(importance)
        sorted_idx = np.argsort(importance)[::-1]

        top_idx = sorted_idx[:n_top]
        bottom_idx = sorted_idx[-n_top:]

        top_props = [self.decode_feature_index(int(i)) for i in top_idx]
        bottom_props = [self.decode_feature_index(int(i)) for i in bottom_idx]

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # --- Dilation distribution ---
        ax = axes[0, 0]
        top_dilations = [p["dilation"] for p in top_props]
        bottom_dilations = [p["dilation"] for p in bottom_props]
        bins = np.unique(top_dilations + bottom_dilations)
        if len(bins) > 1:
            ax.hist(top_dilations, bins=20, alpha=0.6, label="Top", density=True)
            ax.hist(bottom_dilations, bins=20, alpha=0.6, label="Bottom",
                    density=True)
        else:
            ax.bar(
                [0, 1],
                [len(top_dilations), len(bottom_dilations)],
                tick_label=["Top", "Bottom"],
            )
        ax.set_title("Dilation Distribution")
        ax.set_xlabel("Dilation")
        ax.legend()

        # --- Receptive field distribution ---
        ax = axes[0, 1]
        top_rf = [p["receptive_field"] for p in top_props]
        bottom_rf = [p["receptive_field"] for p in bottom_props]
        ax.hist(top_rf, bins=20, alpha=0.6, label="Top", density=True)
        ax.hist(bottom_rf, bins=20, alpha=0.6, label="Bottom", density=True)
        ax.set_title("Receptive Field Distribution")
        ax.set_xlabel("Receptive field (timepoints)")
        ax.legend()

        # --- Pooling operator distribution ---
        ax = axes[0, 2]
        top_pools = [p["pooling_op"] for p in top_props]
        bottom_pools = [p["pooling_op"] for p in bottom_props]
        pool_names = self.POOLING_NAMES
        expanded_pool_names = [
            "PPV\n(Proportion)", "MPV\n(Amplitude)",
            "MIPV\n(Timing)", "LSPV\n(Persistence)",
        ]
        top_counts = [top_pools.count(p) for p in pool_names]
        bottom_counts = [bottom_pools.count(p) for p in pool_names]
        x_pos = np.arange(len(pool_names))
        ax.bar(x_pos - 0.2, top_counts, 0.4, label="Top", alpha=0.7)
        ax.bar(x_pos + 0.2, bottom_counts, 0.4, label="Bottom", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(expanded_pool_names)
        ax.set_title("Pooling Operator Distribution")
        ax.legend()

        # --- Representation distribution ---
        ax = axes[1, 0]
        top_reps = [p["representation"] for p in top_props]
        bottom_reps = [p["representation"] for p in bottom_props]
        for label, reps in [("Top", top_reps), ("Bottom", bottom_reps)]:
            raw_count = reps.count("raw")
            diff_count = reps.count("diff")
            ax.bar(
                label, raw_count,
                label="Raw" if label == "Top" else "",
                color="#1f77b4", alpha=0.7,
            )
            ax.bar(
                label, diff_count, bottom=raw_count,
                label="Diff" if label == "Top" else "",
                color="#ff7f0e", alpha=0.7,
            )
        ax.set_title("Representation (Raw vs Diff)")
        ax.legend()

        # --- Kernel index distribution ---
        ax = axes[1, 1]
        top_ki = [p["kernel_index"] for p in top_props]
        bottom_ki = [p["kernel_index"] for p in bottom_props]
        ax.hist(top_ki, bins=range(0, 85, 4), alpha=0.6, label="Top",
                density=True)
        ax.hist(bottom_ki, bins=range(0, 85, 4), alpha=0.6, label="Bottom",
                density=True)
        ax.set_title("Base Kernel Index Distribution")
        ax.set_xlabel("Kernel index (0-83)")
        ax.legend()

        # --- Importance histogram ---
        ax = axes[1, 2]
        ax.hist(importance, bins=50, alpha=0.7, color="#7f7f7f")
        threshold = importance[sorted_idx[n_top - 1]]
        ax.axvline(
            threshold, color="#d62728", linestyle="--",
            label=f"Top-{n_top} threshold",
        )
        ax.set_title("Feature Importance Distribution")
        ax.set_xlabel("Importance (|coefficient|)")
        ax.set_yscale("log")
        ax.legend()

        fig.suptitle(
            "Kernel Property Analysis: Top vs Bottom Features", fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        return fig

    def summary(self):
        """Print a summary of the fitted model."""
        if self.regressor_ is None:
            print("Model not fitted yet.")
            return

        n_raw = self.n_features_per_rep_[0]
        n_diff = self.n_features_per_rep_[1]
        total = (n_raw + n_diff) * 4

        print("=" * 60)
        print("InterpRocketRegressor Model Summary")
        print("=" * 60)
        print(f"  Base kernels: 84 (length 9, weights {{-1, 2}})")
        print(f"  Dilations (raw):  {self.dilations_raw_}")
        print(f"  Dilations (diff): {self.dilations_diff_}")
        print(f"  Features per representation (biases):")
        print(f"    Raw:  {n_raw} biases × 4 pooling ops = {n_raw * 4}")
        print(f"    Diff: {n_diff} biases × 4 pooling ops = {n_diff * 4}")
        print(f"  Total features: {total}")
        print(f"  Regressor: RidgeCV (alpha={self.regressor_.alpha_:.4f})")
        print()
        print("  Top 10 features:")
        for i, f in enumerate(self.get_top_features(10)):
            print(
                f"    {i+1}. K{f['kernel_index']:02d} d={f['dilation']:>4d} "
                f"{f['pooling_op']:>4s} ({f['representation']:>4s}) "
                f"imp={f['importance']:.4f} "
                f"RF={f['receptive_field']} bias={f['bias']:.3f}"
            )
        print("=" * 60)


# ============================================================================
# SECTION 5: RECURSIVE FEATURE ELIMINATION (Regression)
# ============================================================================

def recursive_feature_elimination(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    drop_percentage=0.05,
    total_number_steps=150,
    alpha_range=None,
    knee_method="kneedle",
    verbose=True,
):
    """
    Recursive Feature Elimination for InterpRocketRegressor.

    Identical algorithm to the classifier version, but uses RidgeCV and
    R² instead of RidgeClassifierCV and accuracy. See interp_rocket.py
    for full documentation of the RFE algorithm.

    Parameters
    ----------
    model : InterpRocketRegressor
        A fitted model.
    X_train, y_train : arrays
        Training data for refitting at each step.
    X_test, y_test : arrays
        Held-out test data for evaluation.
    drop_percentage : float, default=0.05
    total_number_steps : int, default=150
    alpha_range : array, optional
    knee_method : str, default='kneedle'
        'kneedle', 'threshold', or 'both'.
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'fractions', 'n_features', 'train_r2', 'test_r2',
        'surviving_indices', 'full_feature_ranking',
        'knee_idx', 'knee_fraction', 'knee_n_features', 'knee_r2',
        'peak_r2', 'peak_idx'
    """
    if model.regressor_ is None:
        raise ValueError("Model must be fitted before RFE.")

    if alpha_range is None:
        alpha_range = model.alpha_range

    if verbose:
        print("Transforming training data...")
    X_train_features = model._transform(np.asarray(X_train, dtype=np.float32))
    if verbose:
        print("Transforming test data...")
    X_test_features = model._transform(np.asarray(X_test, dtype=np.float32))

    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    n_total = X_train_features.shape[1]

    # Exponential step schedule
    keep_percentage = 1.0 - drop_percentage
    powers_vector = np.arange(total_number_steps)
    percentage_vector_unif = np.power(keep_percentage, powers_vector)
    num_feat_per_step = np.unique(
        (percentage_vector_unif * n_total).astype(int)
    )
    num_feat_per_step = num_feat_per_step[::-1]
    num_feat_per_step = num_feat_per_step[num_feat_per_step > 0]
    n_steps = len(num_feat_per_step)

    # Initial feature importance
    importance = model.get_feature_importance()
    feature_importance = importance.copy()

    if verbose:
        print(f"\nRecursive Feature Elimination: {n_total} total features")
        print(f"  Drop rate: {drop_percentage:.0%} per step, {n_steps} steps")
        print(
            f"  {'Step':>6s}  {'Fraction':>10s}  {'N features':>10s}  "
            f"{'Train R²':>10s}  {'Test R²':>10s}"
        )
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    fractions = []
    n_features_list = []
    train_r2s = []
    test_r2s = []
    surviving_indices = []

    for count, feat_num in enumerate(num_feat_per_step):
        frac = feat_num / n_total

        drop_features = n_total - feat_num
        selected_idxs = np.argsort(feature_importance)[drop_features:]
        selection_mask = np.full(n_total, False)
        selection_mask[selected_idxs] = True

        X_tr = X_train_features[:, selection_mask]
        X_te = X_test_features[:, selection_mask]

        scaler = StandardScaler(with_mean=True)
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        reg = RidgeCV(alphas=alpha_range)
        reg.fit(X_tr_scaled, y_train)

        train_r2 = reg.score(X_tr_scaled, y_train)
        test_r2 = reg.score(X_te_scaled, y_test)

        fractions.append(frac)
        n_features_list.append(feat_num)
        train_r2s.append(train_r2)
        test_r2s.append(test_r2)

        surv_idx = np.where(selection_mask)[0]
        surv_importance = feature_importance[surv_idx]
        sorted_order = np.argsort(surv_importance)[::-1]
        surviving_indices.append(surv_idx[sorted_order].copy())

        if verbose:
            print(
                f"  {count+1:>6d}  {frac:>10.4f}  {feat_num:>10d}  "
                f"{train_r2:>10.4f}  {test_r2:>10.4f}"
            )

        feature_importance[~selection_mask] = 0
        coefs = reg.coef_
        new_importance = np.abs(coefs)
        feature_importance[selection_mask] = new_importance

    # ------------------------------------------------------------------
    # Knee detection
    # ------------------------------------------------------------------
    max_r2 = max(test_r2s)
    peak_idx = int(np.argmax(test_r2s))

    results = {
        "fractions": fractions,
        "n_features": n_features_list,
        "train_r2": train_r2s,
        "test_r2": test_r2s,
        "surviving_indices": surviving_indices,
        "full_feature_ranking": np.argsort(importance)[::-1],
        "peak_r2": max_r2,
        "peak_idx": peak_idx,
    }

    use_threshold = knee_method in ("threshold", "both")
    use_kneedle = knee_method in ("kneedle", "both")

    # Threshold method: smallest feature set within 1% of peak R²
    if use_threshold:
        threshold = max_r2 - 0.01 * abs(max_r2)
        knee_idx = len(test_r2s) - 1
        for i in range(len(test_r2s) - 1, -1, -1):
            if test_r2s[i] >= threshold:
                knee_idx = i
            else:
                break
        results["knee_idx"] = knee_idx
        results["knee_fraction"] = fractions[knee_idx]
        results["knee_n_features"] = n_features_list[knee_idx]
        results["knee_r2"] = test_r2s[knee_idx]

    if use_kneedle:
        kneedle_idx = kneedle(np.array(test_r2s))
        results["kneedle_idx"] = kneedle_idx
        results["kneedle_fraction"] = fractions[kneedle_idx]
        results["kneedle_n_features"] = n_features_list[kneedle_idx]
        results["kneedle_r2"] = test_r2s[kneedle_idx]
        if not use_threshold:
            results["knee_idx"] = kneedle_idx
            results["knee_fraction"] = fractions[kneedle_idx]
            results["knee_n_features"] = n_features_list[kneedle_idx]
            results["knee_r2"] = test_r2s[kneedle_idx]

    if verbose:
        print(f"\n  Peak test R²: {max_r2:.4f} at "
              f"{n_features_list[peak_idx]} features "
              f"({fractions[peak_idx]:.1%})")
        ki = results["knee_idx"]
        print(f"  Knee: {n_features_list[ki]} features "
              f"({fractions[ki]:.1%}), R²={test_r2s[ki]:.4f}")

    return results


def plot_elimination_curve(rfe_results, figsize=(10, 5)):
    """
    Plot the RFE elimination curve for regression.

    Parameters
    ----------
    rfe_results : dict
        Output from recursive_feature_elimination().
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    fracs = rfe_results["fractions"]
    n_feats = rfe_results["n_features"]
    train_r2 = rfe_results["train_r2"]
    test_r2 = rfe_results["test_r2"]
    knee_idx = rfe_results.get("knee_idx")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fracs, train_r2, "o-", markersize=2, linewidth=1.0,
            label="Train R²", alpha=0.7)
    ax.plot(fracs, test_r2, "o-", markersize=2, linewidth=1.0,
            label="Test R²", alpha=0.9)

    # Mark peak
    peak_idx = rfe_results["peak_idx"]
    ax.axvline(fracs[peak_idx], color="#2ca02c", linestyle=":", alpha=0.6,
               label=f"Peak ({n_feats[peak_idx]} features)")

    # Mark knee
    if knee_idx is not None:
        ax.axvline(fracs[knee_idx], color="#d62728", linestyle="--", alpha=0.6,
                   label=f"Knee ({n_feats[knee_idx]} features)")
        ax.plot(fracs[knee_idx], test_r2[knee_idx], "rv", markersize=10)

    # Kneedle if present
    if "kneedle_idx" in rfe_results and "knee_idx" in rfe_results:
        ki = rfe_results["kneedle_idx"]
        if ki != knee_idx:
            ax.axvline(fracs[ki], color="#9467bd", linestyle="-.", alpha=0.6,
                       label=f"Kneedle ({n_feats[ki]} features)")

    ax.set_xlabel("Fraction of features retained")
    ax.set_ylabel("R²")
    ax.set_title("Recursive Feature Elimination (Regression)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    # Secondary x-axis with feature counts
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    candidate_ticks = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    tick_fracs = [f for f in candidate_ticks if min(fracs) <= f <= max(fracs)]
    if not tick_fracs:
        tick_fracs = fracs[:: max(1, len(fracs) // 6)]
    tick_labels = []
    for f in tick_fracs:
        closest_idx = min(range(len(fracs)), key=lambda i: abs(fracs[i] - f))
        tick_labels.append(str(n_feats[closest_idx]))
    ax2.set_xticks(tick_fracs)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel("Number of features", fontsize=9)

    fig.suptitle("Recursive Feature Elimination", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 6: REPEATED K-FOLD CROSS-VALIDATION (Regression)
# ============================================================================

def cross_validate(
    X, y, n_repeats=10, n_folds=10, n_jobs=None, random_state=42,
    verbose=True, **model_kwargs
):
    """
    Evaluate InterpRocketRegressor with repeated k-fold cross-validation.

    Uses RepeatedKFold (not stratified, since target is continuous).
    The full pipeline (bias fitting → transform → standardize → regress)
    is refit on each training fold.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints)
    y : array-like, shape (n_instances,), continuous target
    n_repeats : int, default=10
    n_folds : int, default=10
    n_jobs : int or None
    random_state : int, default=42
    verbose : bool, default=True
    **model_kwargs : passed to InterpRocketRegressor

    Returns
    -------
    results : dict with keys:
        'r2', 'mse', 'rmse', 'mae', 'explained_variance', 'pearson_r'
            — each a dict with 'mean', 'std', 'values'
        'per_repeat_means' : ndarray, shape (n_repeats,) — R² per repeat
        'n_repeats', 'n_folds'
    """
    import os, io, contextlib

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)

    # Limit numba threads
    if n_jobs is not None:
        from numba import set_num_threads

        total_cores = os.cpu_count()
        if n_jobs < 0:
            target = max(1, total_cores + n_jobs + 1)
        else:
            target = max(1, min(n_jobs, total_cores))
        set_num_threads(target)
        if verbose:
            print(f"  Numba threads: {target} of {total_cores} available")

    cv = RepeatedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
    )

    metric_names = [
        "r2", "mse", "rmse", "mae", "explained_variance", "pearson_r",
    ]
    all_metrics = {m: [] for m in metric_names}
    total_folds = n_repeats * n_folds

    if verbose:
        print(
            f"Cross-validation: {n_repeats} repeats × {n_folds} folds "
            f"= {total_folds} evaluations"
        )
        print(
            f"  Data: {X.shape[0]} instances × {X.shape[1]} timepoints"
        )
        print(
            f"  Target: mean={y.mean():.4f}, std={y.std():.4f}, "
            f"range=[{y.min():.4f}, {y.max():.4f}]"
        )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = InterpRocketRegressor(**model_kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train, y_train)

        fold_metrics = model.evaluate(X_test, y_test)
        for m in metric_names:
            all_metrics[m].append(fold_metrics[m])

        if verbose and (fold_idx + 1) % n_folds == 0:
            repeat_num = (fold_idx + 1) // n_folds
            repeat_r2 = all_metrics["r2"][-n_folds:]
            repeat_rmse = all_metrics["rmse"][-n_folds:]
            print(
                f"  Repeat {repeat_num:2d}/{n_repeats}: "
                f"R² = {np.mean(repeat_r2):.4f}  "
                f"RMSE = {np.mean(repeat_rmse):.4f}  "
                f"r = {np.mean(all_metrics['pearson_r'][-n_folds:]):.4f}"
            )

    # Compile results
    results = {}
    for m in metric_names:
        vals = np.array(all_metrics[m])
        results[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "values": vals,
        }

    r2_vals = results["r2"]["values"]
    per_repeat_means = np.array(
        [np.mean(r2_vals[i * n_folds : (i + 1) * n_folds])
         for i in range(n_repeats)]
    )

    results["per_repeat_means"] = per_repeat_means
    results["n_repeats"] = n_repeats
    results["n_folds"] = n_folds

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Results (mean ± std across {total_folds} folds):")
        for m in metric_names:
            print(
                f"    {m:22s}: {results[m]['mean']:.4f} "
                f"± {results[m]['std']:.4f}"
            )
        print(f"{'='*60}")

    return results


# ============================================================================
# SECTION 7: TEMPORAL OCCLUSION SENSITIVITY (Regression)
# ============================================================================

def temporal_occlusion(
    model,
    X_test,
    y_test,
    n_samples=5,
    window_size=None,
    stride=None,
    feature_mask=None,
    verbose=True,
):
    """
    Model-agnostic temporal occlusion sensitivity for regression.

    For each sample, zeros out a sliding window and measures the change
    in the predicted value. Regions where occlusion causes a large change
    are important for the prediction.

    Parameters
    ----------
    model : InterpRocketRegressor
    X_test : ndarray
    y_test : array-like, continuous target
    n_samples : int
    window_size : int or None
    stride : int or None
    feature_mask : array-like of int, optional
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'sample_indices', 'signals', 'sensitivities',
        'true_values', 'predicted_values', 'window_size', 'stride'
    """
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float64)
    n_timepoints = X_test.shape[1]

    if feature_mask is not None:
        feature_mask = np.asarray(feature_mask, dtype=int)

    if window_size is None:
        window_size = max(3, n_timepoints // 20)
    if stride is None:
        stride = max(1, window_size // 2)

    # Select samples spread across the target range
    sorted_by_target = np.argsort(y_test)
    step = max(1, len(sorted_by_target) // n_samples)
    sample_indices = sorted_by_target[::step][:n_samples].tolist()

    def _predict_with_mask(X_single):
        """Predict with optional feature masking."""
        features = model._transform(X_single)
        if feature_mask is not None:
            mask = np.zeros(features.shape[1], dtype=bool)
            mask[feature_mask[feature_mask < features.shape[1]]] = True
            features[:, ~mask] = 0.0
        features_scaled = model.scaler_.transform(features)
        return model.regressor_.predict(features_scaled)[0]

    if verbose:
        print(f"Temporal occlusion: window={window_size}, stride={stride}")
        print(f"  Analyzing {len(sample_indices)} samples...")

    all_signals = []
    all_sensitivities = []
    all_true = []
    all_pred = []

    for s_idx in sample_indices:
        X_single = X_test[s_idx : s_idx + 1].copy()
        signal = X_single[0].copy()
        n = len(signal)

        base_pred = _predict_with_mask(X_single)

        positions = list(range(0, n - window_size + 1, stride))
        sensitivity = np.zeros(n)
        counts = np.zeros(n)

        for pos in positions:
            X_occluded = X_single.copy()
            X_occluded[0, pos : pos + window_size] = 0.0

            occl_pred = _predict_with_mask(X_occluded)
            impact = abs(base_pred - occl_pred)

            sensitivity[pos : pos + window_size] += impact
            counts[pos : pos + window_size] += 1

        counts[counts == 0] = 1
        sensitivity /= counts

        all_signals.append(signal)
        all_sensitivities.append(sensitivity)
        all_true.append(float(y_test[s_idx]))
        all_pred.append(float(base_pred))

        if verbose:
            print(
                f"  Sample {s_idx}: true={y_test[s_idx]:.4f}, "
                f"pred={base_pred:.4f}, "
                f"max_sensitivity={sensitivity.max():.4f}"
            )

    return {
        "sample_indices": sample_indices,
        "signals": all_signals,
        "sensitivities": all_sensitivities,
        "true_values": all_true,
        "predicted_values": all_pred,
        "window_size": window_size,
        "stride": stride,
    }


def plot_occlusion(occ_results, figsize=(12, None)):
    """
    Plot temporal occlusion sensitivity results for regression.

    Each row shows one sample: the signal in gray with a sensitivity
    curve overlaid on a twin axis.
    """
    signals = occ_results["signals"]
    sensitivities = occ_results["sensitivities"]
    true_values = occ_results["true_values"]
    pred_values = occ_results["predicted_values"]
    sample_indices = occ_results["sample_indices"]
    window_size = occ_results["window_size"]
    stride = occ_results["stride"]

    n_samples = len(signals)
    if figsize[1] is None:
        figsize = (figsize[0], 2.5 * n_samples)

    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, squeeze=False)

    for row in range(n_samples):
        ax = axes[row, 0]
        signal = signals[row]
        sensitivity = sensitivities[row]
        t = np.arange(len(signal))

        ax.plot(t, signal, color="#7f7f7f", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(
            f"Sample {sample_indices[row]}\n"
            f"true={true_values[row]:.3f}\n"
            f"pred={pred_values[row]:.3f}",
            fontsize=9,
        )

        ax2 = ax.twinx()
        ax2.plot(t, sensitivity, color="#ff7f0e", linewidth=1.2, alpha=0.85)
        ax2.set_ylabel("Sensitivity", fontsize=8, color="#ff7f0e")

        if row == 0:
            ax.set_title(
                f"Temporal Occlusion Sensitivity "
                f"(window={window_size}, stride={stride})",
                fontsize=11,
            )
        if row == n_samples - 1:
            ax.set_xlabel("Timepoint")

    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 8: CROSS-VALIDATION FEATURE STABILITY (Regression)
# ============================================================================

def cv_feature_stability(
    X, y, n_repeats=5, n_folds=5, n_top=50,
    random_state=42, verbose=True, **model_kwargs
):
    """
    Measure feature importance stability across cross-validation folds.

    Identical logic to the classifier version, but uses RepeatedKFold
    and InterpRocketRegressor.

    Parameters
    ----------
    X : ndarray, shape (n_instances, n_timepoints)
    y : array-like, continuous target
    n_repeats, n_folds, n_top, random_state, verbose : as in classifier
    **model_kwargs : passed to InterpRocketRegressor

    Returns
    -------
    results : dict (same structure as classifier version)
    """
    import contextlib, io

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)

    cv = RepeatedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
    )
    total_folds = n_folds * n_repeats

    ref_model = InterpRocketRegressor(**model_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        ref_model.fit(X, y)
    n_features_total = ref_model.transform(X[:1]).shape[1]

    feature_counts = np.zeros(n_features_total, dtype=int)
    top_features_per_fold = []
    importance_per_fold = []

    if verbose:
        print(
            f"CV feature stability: {n_repeats}x{n_folds} folds, "
            f"tracking top {n_top} features per fold"
        )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        model = InterpRocketRegressor(**model_kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X[train_idx], y[train_idx])

        imp = model.get_feature_importance()
        top_idx = np.argsort(imp)[::-1][:n_top]

        feature_counts[top_idx] += 1
        top_features_per_fold.append(top_idx)
        importance_per_fold.append(imp)

        if verbose and (fold_idx + 1) % n_folds == 0:
            r = (fold_idx + 1) // n_folds
            n_stable = np.sum(feature_counts >= r)
            print(
                f"  Round {r}/{n_repeats}: "
                f"{n_stable} features appeared in every round so far"
            )

    stable_order = np.argsort(feature_counts)[::-1]
    decoded_stable = []
    for fi in stable_order[:n_top]:
        info = ref_model.decode_feature_index(int(fi))
        info["stability_count"] = int(feature_counts[fi])
        info["stability_fraction"] = feature_counts[fi] / total_folds
        info["feature_index"] = int(fi)
        decoded_stable.append(info)

    if verbose:
        n_always = np.sum(feature_counts == total_folds)
        n_most = np.sum(feature_counts >= total_folds * 0.8)
        n_never = np.sum(feature_counts == 0)
        print(f"\n  Features in ALL folds:  {n_always}")
        print(f"  Features in ≥80% folds: {n_most}")
        print(f"  Features in NO folds:   {n_never}")

    return {
        "feature_counts": feature_counts,
        "top_features_per_fold": top_features_per_fold,
        "importance_per_fold": importance_per_fold,
        "decoded_stable": decoded_stable,
        "n_folds_total": total_folds,
    }


# ============================================================================
# SECTION 8B: FEATURE STABILITY SELECTION
# ============================================================================

def get_stable_features(stability, threshold=0.8):
    """
    Extract feature indices that appear in the top set in at least
    `threshold` fraction of CV folds.

    Parameters
    ----------
    stability : dict
        Output from cv_feature_stability().
    threshold : float, default=0.8
        Minimum fraction of folds a feature must appear in.

    Returns
    -------
    stable_features : ndarray of int
        Feature indices meeting the threshold, sorted by frequency.
    """
    counts = stability['feature_counts']
    n_folds = stability['n_folds_total']
    mask = counts >= threshold * n_folds

    indices = np.where(mask)[0]
    order = np.argsort(-counts[indices])
    stable_features = indices[order]

    print(f"Stable features (≥{threshold:.0%} of {n_folds} folds): "
          f"{len(stable_features)}")
    return stable_features


# Note: Class-mean visualization functions (plot_class_mean_activation,
# plot_multi_kernel_summary, plot_aggregate_activation) are specific to
# classification and are available in interp_rocket.py only. For
# regression, target-conditional analysis would require binning the
# continuous target, which is application-specific.


# ============================================================================
# SECTION 9: CONVENIENCE / DEMO
# ============================================================================

def demo_with_synthetic_data():
    """
    Run a quick demonstration with synthetic regression data.

    Creates a problem where the target is determined by the amplitude
    of a pattern in a specific temporal region, plus noise.
    """
    np.random.seed(42)
    n_samples = 400
    n_timepoints = 200
    noise = 0.3

    X_all = []
    y_all = []

    for i in range(n_samples):
        x = np.random.randn(n_timepoints).astype(np.float32) * noise
        # Target is determined by the amplitude of a bump at t=[80:100]
        amplitude = np.random.uniform(0.5, 3.0)
        x[80:100] += amplitude
        # Add a secondary weaker pattern at t=[140:155]
        secondary = np.random.uniform(-1.0, 1.0)
        x[140:155] += secondary * 0.5
        # Target is a linear combination of both patterns
        target = 2.0 * amplitude + 0.5 * secondary + np.random.randn() * 0.2
        X_all.append(x)
        y_all.append(target)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float64)

    # Shuffle and split
    idx = np.random.permutation(len(y_all))
    X_all, y_all = X_all[idx], y_all[idx]
    split = int(0.7 * len(y_all))
    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]

    print("=" * 60)
    print("InterpRocketRegressor Demo: Synthetic regression data")
    print(f"  {n_timepoints} timepoints, target determined by bump at [80:100]")
    print(f"  Training: {len(y_train)}, Testing: {len(y_test)}")
    print("=" * 60)

    model = InterpRocketRegressor(
        max_dilations_per_kernel=40, num_features=5000, random_state=666
    )
    model.fit(X_train, y_train)

    test_r2 = model.score(X_test, y_test)
    print(f"\n  Test R²: {test_r2:.4f}")

    metrics = model.evaluate(X_test, y_test)
    print(f"  Test RMSE: {metrics['rmse']:.4f}")
    print(f"  Test Pearson r: {metrics['pearson_r']:.4f}")

    model.summary()

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = demo_with_synthetic_data()

    print("\nGenerating visualizations...")
    fig1 = model.plot_top_kernels(X_test, y_test, n_kernels=3)
    fig1.savefig("regression_top_kernels.png", dpi=150, bbox_inches="tight")

    fig2, imp_by_time = model.plot_temporal_importance(X_test, y_test)
    fig2.savefig("regression_temporal_importance.png", dpi=150,
                 bbox_inches="tight")

    fig3 = model.plot_feature_distributions(X_test, y_test)
    fig3.savefig("regression_distributions.png", dpi=150,
                 bbox_inches="tight")

    fig4 = model.plot_kernel_properties()
    fig4.savefig("regression_properties.png", dpi=150, bbox_inches="tight")

    print("Done. Saved 4 visualization figures.")
