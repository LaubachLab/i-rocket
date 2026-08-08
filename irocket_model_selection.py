"""Leakage-free model selection for the I-ROCKET pipeline.

The canonical validation path is deliberately explicit:

1. Fit ``InterpRocketTransform`` on an inner-training partition only.
2. Fit ``ResampledShrinkageSelector`` on that fixed transformed matrix only.
3. Reuse its selection probabilities to compare predeclared consensus
   thresholds and ridge regularization values on the inner-validation data.
4. Refit the complete transform-selector-classifier sequence on the outer-
   training partition with the selected settings.
5. Evaluate exactly once on the untouched outer-test partition.

After outer cross-validation estimates generalization, an optional final model
is tuned again by inner cross-validation on the complete development dataset
and refitted for kernel interpretation.  Its training performance is not an
estimate of generalization.

The module uses resampled consensus selection and the Nogueira stability
measure.  It does not claim the false-positive guarantees of formal
Meinshausen-Buehlmann stability selection.

AUTHOR: Mark Laubach (American University, Department of Neuroscience)
LICENSE: BSD-3-Clause
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    get_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
    has_fit_parameter,
)

try:  # introduced after the oldest supported scikit-learn releases
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - exercised only on older sklearn
    StratifiedGroupKFold = None

from interp_rocket import InterpRocket, InterpRocketTransform, _validate_feature_index_array
from _irocket_selection import ResampledShrinkageSelector


def _require_pandas():
    """Import pandas for result-reporting helpers."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "I-ROCKET result tables require pandas. Install pandas or install "
            "I-ROCKET with its notebook dependencies."
        ) from exc
    return pd


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class InnerCandidateResult:
    """Cross-validated performance for one threshold/alpha combination."""

    consensus_threshold: float
    alpha: float
    fold_scores: np.ndarray
    selected_counts: np.ndarray
    mean_score: float
    std_score: float
    standard_error: float
    mean_selected_features: float


@dataclass
class InnerFoldResult:
    """Audit information for one inner split."""

    fold_index: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    n_transform_features: int
    selector_stability: float
    selector_n_resamples: int
    selector_requested_n_resamples: int
    selector_cutoff_sizes: np.ndarray
    selector_cutoff_improvements: np.ndarray
    selected_counts: np.ndarray


@dataclass
class InnerSearchResult:
    """Complete result of one inner model-selection search."""

    candidates: Tuple[InnerCandidateResult, ...]
    fold_results: Tuple[InnerFoldResult, ...]
    consensus_thresholds: np.ndarray
    classifier_alphas: np.ndarray
    best_candidate_index: int
    best_consensus_threshold: float
    best_alpha: float
    best_mean_score: float
    best_standard_error: float
    one_se_cutoff: Optional[float]
    selection_rule: str
    scoring: Union[str, Callable]

    @property
    def best_candidate(self) -> InnerCandidateResult:
        return self.candidates[self.best_candidate_index]


@dataclass
class OuterFoldResult:
    """Prediction, performance, and selection diagnostics for one outer fold."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    inner_search: InnerSearchResult
    best_consensus_threshold: float
    best_alpha: float
    predictions: np.ndarray
    decision_scores: np.ndarray
    metrics: Dict[str, float]
    primary_score: float
    n_transform_features: int
    n_selected_features: int
    nogueira_stability: float
    selector_n_resamples: int
    selector_requested_n_resamples: int
    cutoff_sizes: np.ndarray
    selection_probabilities: np.ndarray
    selected_indices: np.ndarray
    feature_metadata: Tuple[Dict[str, Any], ...]
    model: Optional[Any] = None


@dataclass
class NestedCVResult:
    """Output of :func:`nested_stability_cv`."""

    outer_fold_results: Tuple[OuterFoldResult, ...]
    outer_scores: np.ndarray
    outer_predictions: np.ndarray
    outer_decision_scores: np.ndarray
    outer_test_counts: np.ndarray
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    pooled_metrics: Dict[str, float]
    selected_counts: np.ndarray
    nogueira_stabilities: np.ndarray
    cutoff_distributions: Tuple[np.ndarray, ...]
    classes: np.ndarray
    scoring: Union[str, Callable]
    resample_groups: bool
    final_search: Optional[InnerSearchResult]
    final_model: Optional[Any]
    best_parameters: Optional[Dict[str, float]]
    feature_metadata: Tuple[Dict[str, Any], ...]

    def summary(self) -> Dict[str, Any]:
        """Return a compact, serialization-friendly summary."""
        return {
            "n_outer_folds": len(self.outer_fold_results),
            "outer_score_mean": float(np.mean(self.outer_scores)),
            "outer_score_std": float(np.std(self.outer_scores, ddof=1))
            if self.outer_scores.size > 1
            else 0.0,
            "mean_metrics": dict(self.mean_metrics),
            "std_metrics": dict(self.std_metrics),
            "pooled_metrics": dict(self.pooled_metrics),
            "selected_count_mean": float(np.mean(self.selected_counts)),
            "selected_count_range": (
                int(np.min(self.selected_counts)),
                int(np.max(self.selected_counts)),
            ),
            "nogueira_stability_mean": float(
                np.mean(self.nogueira_stabilities)
            ),
            "resample_groups": bool(self.resample_groups),
            "best_parameters": None
            if self.best_parameters is None
            else dict(self.best_parameters),
        }

    def summary_tables(self):
        """Return aggregate classification metrics and metadata tables.

        Returns
        -------
        metrics : pandas.DataFrame
            Rows are classification metrics and columns are ``Mean``, ``Std``,
            and ``Pooled`` across the untouched outer folds.
        metadata : pandas.Series
            Remaining nested-validation metadata. ``outer_score_mean`` and
            ``outer_score_std`` are omitted because they duplicate the primary
            score already present in the classification-metrics table.

        Notes
        -----
        The pooled values are calculated from all outer-fold predictions. The
        final interpretation model is not evaluated on its training data here.
        """
        pd = _require_pandas()
        summary = self.summary()
        metrics = pd.DataFrame(
            {
                "Mean": summary["mean_metrics"],
                "Std": summary["std_metrics"],
                "Pooled": summary["pooled_metrics"],
            }
        )
        metadata = {
            key: value
            for key, value in summary.items()
            if not key.endswith("_metrics")
        }
        metadata_series = pd.Series(metadata, name="Value").drop(
            ["outer_score_mean", "outer_score_std"],
            errors="ignore",
        )
        return metrics, metadata_series

    def outer_fold_table(self):
        """Return one row per untouched outer-test fold.

        The table combines fold-specific classification metrics with the
        selected consensus threshold, ridge alpha, feature count, and Nogueira
        stability. Fold numbers are reported as one-based values for display;
        ``fold_index`` preserves the stored zero-based index.
        """
        pd = _require_pandas()
        rows = []
        for result in self.outer_fold_results:
            row = {
                "fold": int(result.fold_index) + 1,
                "fold_index": int(result.fold_index),
                "n_train": int(len(result.train_indices)),
                "n_test": int(len(result.test_indices)),
                "consensus_threshold": float(
                    result.best_consensus_threshold
                ),
                "alpha": float(result.best_alpha),
                "n_transform_features": int(result.n_transform_features),
                "n_selected_features": int(result.n_selected_features),
                "nogueira_stability": float(result.nogueira_stability),
                "primary_score": float(result.primary_score),
            }
            row.update(
                {
                    str(name): float(value)
                    for name, value in result.metrics.items()
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def print_summary(self, digits=4, *, include_outer_folds=False):
        """Print cleaned aggregate results and optionally the fold table.

        Parameters
        ----------
        digits : int, default=4
            Number of decimal places used for the metrics display.
        include_outer_folds : bool, default=False
            Print the one-row-per-fold table after the aggregate summary.

        Returns
        -------
        metrics, metadata : tuple
            The same objects returned by :meth:`summary_tables`.
        """
        if isinstance(digits, (bool, np.bool_)) or not isinstance(
            digits, (int, np.integer)
        ):
            raise TypeError("digits must be an integer.")
        if int(digits) < 0:
            raise ValueError("digits must be nonnegative.")
        if not isinstance(include_outer_folds, (bool, np.bool_)):
            raise TypeError("include_outer_folds must be a boolean.")

        metrics, metadata = self.summary_tables()
        print("--- CLASSIFICATION METRICS ---")
        print(metrics.round(int(digits)))
        print("\n--- METADATA ---")
        print(metadata)
        if include_outer_folds:
            print("\n--- OUTER FOLDS ---")
            print(self.outer_fold_table().round(int(digits)))
        return metrics, metadata


# ---------------------------------------------------------------------------
# Final fitted classifier
# ---------------------------------------------------------------------------


class StableRocketClassifier(ClassifierMixin, BaseEstimator):
    """Fit a fixed I-ROCKET transform, consensus selector, and ridge model.

    This estimator does not tune its own parameters.  It is the refit component
    used after an inner search has chosen ``consensus_threshold`` and ``alpha``.
    ``groups`` may be supplied to :meth:`fit` so that the selector's internal
    resampling preserves complete groups.

    Parameters
    ----------
    transformer : estimator or None, default=None
        Cloneable transformer. ``None`` creates ``InterpRocketTransform``.
    selector : estimator or None, default=None
        Cloneable selector exposing ``get_support_at_threshold``. ``None``
        creates ``ResampledShrinkageSelector``.
    consensus_threshold : float, default=0.7
        Selection-probability threshold used for the final consensus set.
    alpha : float, default=1.0
        RidgeClassifier regularization parameter.
    class_weight : dict, 'balanced', or None, default=None
        Passed directly to ``RidgeClassifier``.
    ridge_solver : str, default='lsqr'
        Ridge solver. ``lsqr`` is the default because the selected ROCKET
        matrix can remain wide and highly correlated.
    random_state : int, default=0
        Used only when default transformer or selector objects are created.
    """

    _LEARNED_ATTRIBUTES = (
        "classes_",
        "classifier_",
        "feature_metadata_",
        "n_features_in_",
        "n_output_features_",
        "n_selected_features_",
        "n_selected_",
        "nogueira_stability_",
        "n_transform_features_",
        "scaler_",
        "selected_indices_",
        "selector_",
        "transformer_",
    )

    def __init__(
        self,
        transformer=None,
        selector=None,
        consensus_threshold=0.7,
        alpha=1.0,
        class_weight=None,
        ridge_solver="lsqr",
        random_state=0,
    ):
        self.transformer = transformer
        self.selector = selector
        self.consensus_threshold = consensus_threshold
        self.alpha = alpha
        self.class_weight = class_weight
        self.ridge_solver = ridge_solver
        self.random_state = random_state

    def _reset(self):
        for attribute in self._LEARNED_ATTRIBUTES:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def _validate_parameters(self):
        if not np.isscalar(self.consensus_threshold):
            raise TypeError("consensus_threshold must be a number in (0, 1].")
        threshold = float(self.consensus_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("consensus_threshold must be in (0, 1].")
        if not np.isscalar(self.alpha) or isinstance(
            self.alpha, (bool, np.bool_)
        ):
            raise TypeError("alpha must be a positive finite number.")
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a positive finite number.")
        if not isinstance(self.ridge_solver, str) or not self.ridge_solver:
            raise TypeError("ridge_solver must be a non-empty string.")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer.")

    def _make_transformer(self):
        if self.transformer is None:
            return InterpRocketTransform(random_state=int(self.random_state))
        transformer = clone(self.transformer)
        parameters = transformer.get_params(deep=False)
        if (
            "random_state" in parameters
            and parameters["random_state"] is None
        ):
            transformer.set_params(random_state=int(self.random_state))
        return transformer

    def _make_selector(self):
        if self.selector is None:
            return ResampledShrinkageSelector(
                consensus_threshold=float(self.consensus_threshold),
                random_state=int(self.random_state),
            )
        selector = clone(self.selector)
        parameters = selector.get_params(deep=False)
        if "consensus_threshold" not in parameters:
            raise TypeError(
                "selector must expose a consensus_threshold parameter."
            )
        updates = {
            "consensus_threshold": float(self.consensus_threshold)
        }
        if "random_state" in parameters and parameters["random_state"] is None:
            updates["random_state"] = int(self.random_state)
        selector.set_params(**updates)
        return selector

    @staticmethod
    def _fit_selector(selector, X, y, groups):
        if groups is None:
            selector.fit(X, y)
            return selector
        if not has_fit_parameter(selector, "groups"):
            raise TypeError(
                "The supplied selector does not accept groups in fit()."
            )
        selector.fit(X, y, groups=groups)
        return selector

    def fit(self, X, y, groups=None):
        """Fit the complete fixed-parameter model on one training dataset."""
        self._reset()
        self._validate_parameters()
        X = check_array(
            X,
            dtype=np.float32,
            ensure_2d=True,
            allow_nd=False,
            accept_sparse=False,
        )
        y = column_or_1d(y)
        check_consistent_length(X, y)
        if groups is not None:
            groups = column_or_1d(groups)
            check_consistent_length(X, groups)
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("At least two classes are required.")

        transformer = self._make_transformer()
        transformer.fit(X, y)
        transformed = transformer.transform(X)

        selector = self._make_selector()
        self._fit_selector(selector, transformed, y, groups)
        selected_indices = _selector_indices(
            selector,
            float(self.consensus_threshold),
            n_features=transformed.shape[1],
        )

        scaler = StandardScaler()
        selected_scaled = scaler.fit_transform(
            transformed[:, selected_indices]
        )
        classifier = RidgeClassifier(
            alpha=float(self.alpha),
            class_weight=self.class_weight,
            solver=self.ridge_solver,
        )
        classifier.fit(selected_scaled, y)

        metadata = []
        if hasattr(transformer, "decode_feature_index"):
            probabilities = getattr(
                selector,
                "selection_probabilities_",
                np.full(transformed.shape[1], np.nan),
            )
            scores = getattr(
                selector,
                "full_scores_",
                np.full(transformed.shape[1], np.nan),
            )
            for feature_index in selected_indices:
                info = dict(
                    transformer.decode_feature_index(int(feature_index))
                )
                info["feature_index"] = int(feature_index)
                info["selection_probability"] = float(
                    probabilities[feature_index]
                )
                info["shrinkage_t_score"] = float(scores[feature_index])
                metadata.append(info)

        self.transformer_ = transformer
        self.selector_ = selector
        self.scaler_ = scaler
        self.classifier_ = classifier
        self.classes_ = np.asarray(classifier.classes_)
        self.n_features_in_ = int(X.shape[1])
        self.n_transform_features_ = int(transformed.shape[1])
        self.n_output_features_ = self.n_transform_features_
        self.selected_indices_ = selected_indices
        self.n_selected_features_ = int(selected_indices.size)
        self.n_selected_ = self.n_selected_features_
        self.nogueira_stability_ = float(
            getattr(selector, "nogueira_stability_", np.nan)
        )
        self.feature_metadata_ = tuple(metadata)
        return self

    def _selected_scaled(self, X):
        check_is_fitted(
            self,
            attributes=[
                "transformer_",
                "selector_",
                "scaler_",
                "classifier_",
                "selected_indices_",
            ],
        )
        transformed = self.transformer_.transform(X)
        return self.scaler_.transform(
            transformed[:, self.selected_indices_]
        )

    def predict(self, X):
        selected = self._selected_scaled(X)
        return self.classifier_.predict(selected)

    def decision_function(self, X):
        selected = self._selected_scaled(X)
        return self.classifier_.decision_function(selected)

    def transform_selected_features(self, X, *, scaled=True):
        """Return selected transformed features, optionally standardized."""
        check_is_fitted(
            self,
            attributes=["transformer_", "selected_indices_", "scaler_"],
        )
        transformed = self.transformer_.transform(X)[
            :, self.selected_indices_
        ]
        if scaled:
            return self.scaler_.transform(transformed)
        return transformed

    def get_support(self, indices=False):
        """Return the transformed columns retained by consensus selection."""
        check_is_fitted(
            self, attributes=["selected_indices_", "n_transform_features_"]
        )
        if indices:
            return self.selected_indices_.copy()
        support = np.zeros(self.n_transform_features_, dtype=bool)
        support[self.selected_indices_] = True
        return support

    def get_selected_feature_metadata(self):
        """Return independent dictionaries describing final selected columns."""
        check_is_fitted(self, attributes=["feature_metadata_"])
        return tuple(dict(item) for item in self.feature_metadata_)

    def decode_selected_features(self):
        """Alias for :meth:`get_selected_feature_metadata`."""
        return self.get_selected_feature_metadata()

    # ------------------------------------------------------------------
    # Interpretation compatibility on the full transformed feature space
    # ------------------------------------------------------------------

    def transform(self, X):
        """Return the complete unscaled I-ROCKET feature matrix."""
        check_is_fitted(self, attributes=["transformer_"])
        return self.transformer_.transform(X)

    def _transform(self, X):
        """Private compatibility alias used by retained plotting methods."""
        return self.transform(X)

    def decode_feature_index(self, feature_index):
        """Decode a full-transform feature through the fitted transformer."""
        check_is_fitted(self, attributes=["transformer_"])
        return self.transformer_.decode_feature_index(feature_index)

    def get_full_classifier_coefficients(self):
        """Map selected ridge coefficients back to all transformed columns."""
        check_is_fitted(
            self,
            attributes=[
                "classifier_",
                "selected_indices_",
                "n_transform_features_",
            ],
        )
        coefficients = np.asarray(self.classifier_.coef_)
        if coefficients.ndim == 1:
            full = np.zeros(self.n_transform_features_, dtype=coefficients.dtype)
            full[self.selected_indices_] = coefficients
        else:
            full = np.zeros(
                (coefficients.shape[0], self.n_transform_features_),
                dtype=coefficients.dtype,
            )
            full[:, self.selected_indices_] = coefficients
        return full

    def get_feature_importance(self, feature_mask=None):
        """Return normalized ridge importance mapped to the full transform."""
        coefficients = self.get_full_classifier_coefficients()
        if coefficients.ndim == 1:
            importance = np.abs(coefficients)
        else:
            importance = np.linalg.norm(coefficients, axis=0, ord=2)
        importance = np.asarray(importance, dtype=float)

        if feature_mask is not None:
            indices = _validate_feature_index_array(
                feature_mask,
                self.n_transform_features_,
                name="feature_mask",
            )
            masked = np.zeros_like(importance)
            masked[indices] = importance[indices]
            importance = masked

        maximum = float(np.max(importance)) if importance.size else 0.0
        if maximum > 0.0:
            importance = importance / maximum
        return importance

    def get_top_features(self, n=None, feature_mask=None):
        """Return decoded ridge-ranked features from the consensus set."""
        check_is_fitted(
            self,
            attributes=["selected_indices_", "n_transform_features_"],
        )
        if feature_mask is None:
            candidates = self.selected_indices_.copy()
        else:
            candidates = _validate_feature_index_array(
                feature_mask,
                self.n_transform_features_,
                name="feature_mask",
            )
        importance = self.get_feature_importance(feature_mask=candidates)
        if n is None:
            n = min(20, len(candidates))
        if isinstance(n, (bool, np.bool_)) or not isinstance(
            n, (int, np.integer)
        ):
            raise TypeError("n must be an integer or None.")
        if n < 1:
            raise ValueError("n must be positive.")
        n = min(int(n), len(candidates))
        order = np.argsort(importance[candidates], kind="stable")[::-1]
        top_indices = candidates[order[:n]]

        probabilities = np.asarray(
            getattr(
                self.selector_,
                "selection_probabilities_",
                np.full(self.n_transform_features_, np.nan),
            ),
            dtype=float,
        )
        scores = np.asarray(
            getattr(
                self.selector_,
                "full_scores_",
                np.full(self.n_transform_features_, np.nan),
            ),
            dtype=float,
        )
        results = []
        for index in top_indices:
            info = self.decode_feature_index(int(index))
            info["feature_index"] = int(index)
            info["importance"] = float(importance[index])
            info["selection_probability"] = float(probabilities[index])
            info["shrinkage_t_score"] = float(scores[index])
            results.append(info)
        return results

    def plot_top_kernels(self, X, y, **kwargs):
        """Plot trial-level activation for consensus-selected kernels.

        This delegates to :meth:`InterpRocket.plot_top_kernels` while defaulting
        the candidate universe to the fitted consensus-selected columns.
        """
        if kwargs.get("feature_mask") is None:
            kwargs["feature_mask"] = self.selected_indices_
        return InterpRocket.plot_top_kernels(self, X, y, **kwargs)

    def plot_feature_distributions(self, X, y, **kwargs):
        """Plot class-conditional values for consensus-selected features."""
        if kwargs.get("feature_mask") is None:
            kwargs["feature_mask"] = self.selected_indices_
        return InterpRocket.plot_feature_distributions(self, X, y, **kwargs)

    def plot_kernel_properties(self, **kwargs):
        """Summarize the fitted consensus-selected kernel set."""
        check_is_fitted(
            self, attributes=["transformer_", "selector_", "selected_indices_"]
        )
        kwargs.setdefault(
            "selection_probabilities",
            self.selector_.selection_probabilities_,
        )
        kwargs.setdefault("consensus_threshold", self.consensus_threshold)
        return self.transformer_.plot_kernel_properties(
            self.selected_indices_, **kwargs
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_grid(values, *, name, lower, upper=None, closed_upper=False):
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a one-dimensional sequence.") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise TypeError(f"{name} must contain numeric values, not booleans.")
    try:
        array = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only numeric values.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array <= lower):
        raise ValueError(f"Every value in {name} must be greater than {lower}.")
    if upper is not None:
        invalid = array > upper if closed_upper else array >= upper
        if np.any(invalid):
            comparator = "at most" if closed_upper else "less than"
            raise ValueError(f"Every value in {name} must be {comparator} {upper}.")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} must not contain duplicate values.")
    return array


def _validate_random_state(random_state):
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError("random_state must be an integer.")
    return int(random_state)


def _make_splitter(cv, *, groups, random_state, name):
    if isinstance(cv, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer or a CV splitter.")
    if isinstance(cv, (int, np.integer)):
        n_splits = int(cv)
        if n_splits < 2:
            raise ValueError(f"{name} must contain at least two folds.")
        if groups is None:
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        if StratifiedGroupKFold is not None:
            return StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        warnings.warn(
            "StratifiedGroupKFold is unavailable in this scikit-learn "
            "version; falling back to GroupKFold without class stratification.",
            RuntimeWarning,
            stacklevel=3,
        )
        return GroupKFold(n_splits=n_splits)
    if not hasattr(cv, "split"):
        raise TypeError(f"{name} must be an integer or expose split().")
    return cv


def _materialize_splits(
    splitter,
    X,
    y,
    groups,
    *,
    name,
    require_partition,
    require_all_classes=True,
):
    try:
        with warnings.catch_warnings():
            # Some ordinary splitters accept a groups argument only to ignore
            # it. We perform an explicit group-overlap audit below, which gives
            # a more useful error than scikit-learn's generic warning.
            warnings.filterwarnings(
                "ignore",
                message="The groups parameter is ignored by.*",
                category=UserWarning,
            )
            iterator = (
                splitter.split(X, y, groups)
                if groups is not None
                else splitter.split(X, y)
            )
            raw_splits = list(iterator)
    except Exception as exc:
        raise ValueError(f"Unable to generate {name} splits: {exc}") from exc

    if len(raw_splits) < 2:
        raise ValueError(f"{name} must generate at least two splits.")

    n_samples = X.shape[0]
    expected_classes = np.unique(y)
    test_counts = np.zeros(n_samples, dtype=np.int64)
    validated = []
    for split_index, (train_indices, test_indices) in enumerate(raw_splits):
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        if train_indices.ndim != 1 or test_indices.ndim != 1:
            raise ValueError(f"{name} split {split_index} indices must be 1D.")
        if train_indices.size == 0 or test_indices.size == 0:
            raise ValueError(f"{name} split {split_index} contains an empty fold.")
        if (
            np.any(train_indices < 0)
            or np.any(test_indices < 0)
            or np.any(train_indices >= n_samples)
            or np.any(test_indices >= n_samples)
        ):
            raise ValueError(f"{name} split {split_index} has invalid indices.")
        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError(
                f"{name} split {split_index} has overlapping train/test rows."
            )
        if np.unique(train_indices).size != train_indices.size or np.unique(
            test_indices
        ).size != test_indices.size:
            raise ValueError(
                f"{name} split {split_index} contains duplicate row indices."
            )
        if groups is not None:
            train_groups = np.unique(groups[train_indices])
            test_groups = np.unique(groups[test_indices])
            overlap = np.intersect1d(train_groups, test_groups)
            if overlap.size:
                raise ValueError(
                    f"{name} split {split_index} leaks {overlap.size} group(s) "
                    "between training and validation/test data."
                )
        if require_all_classes:
            train_classes = np.unique(y[train_indices])
            test_classes = np.unique(y[test_indices])
            if not np.array_equal(train_classes, expected_classes):
                raise ValueError(
                    f"{name} split {split_index} training data do not contain "
                    "every class."
                )
            if not np.array_equal(test_classes, expected_classes):
                raise ValueError(
                    f"{name} split {split_index} validation/test data do not "
                    "contain every class."
                )
        test_counts[test_indices] += 1
        validated.append((train_indices, test_indices))

    if require_partition and not np.all(test_counts == 1):
        missing = int(np.sum(test_counts == 0))
        repeated = int(np.sum(test_counts > 1))
        raise ValueError(
            f"{name} must partition the observations exactly once; "
            f"{missing} rows were never tested and {repeated} rows were tested "
            "more than once."
        )
    return validated, test_counts


def _fit_selector(selector, X, y, groups):
    if groups is None:
        selector.fit(X, y)
        return selector
    if not has_fit_parameter(selector, "groups"):
        raise TypeError("The supplied selector does not accept groups in fit().")
    selector.fit(X, y, groups=groups)
    return selector


def _selector_indices(selector, threshold, *, n_features):
    if not hasattr(selector, "get_support_at_threshold"):
        raise TypeError(
            "selector must implement get_support_at_threshold(threshold, "
            "indices=True)."
        )
    raw_indices = np.asarray(
        selector.get_support_at_threshold(float(threshold), indices=True)
    )
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        raise RuntimeError(
            f"The selector retained no features at threshold {threshold}."
        )
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise TypeError(
            "selector.get_support_at_threshold(..., indices=True) must "
            "return integer feature indices."
        )
    indices = raw_indices.astype(np.int64, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError("The selector returned duplicate feature indices.")
    if np.any(indices < 0) or np.any(indices >= int(n_features)):
        raise ValueError(
            "The selector returned a feature index outside the transformed "
            "feature matrix."
        )
    return indices.copy()


def _candidate_score(scorer, classifier, X, y):
    value = scorer(classifier, X, y)
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError("The inner scoring function returned a nonfinite value.")
    return value


def _classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _decision_scores_aligned(model, X, classes):
    scores = np.asarray(model.decision_function(X))
    model_classes = np.asarray(model.classes_)
    if not np.array_equal(model_classes, classes):
        raise RuntimeError(
            "A fitted outer model does not contain the complete class set."
        )
    if classes.size == 2:
        return scores.reshape(-1)
    if scores.ndim != 2 or scores.shape[1] != classes.size:
        raise RuntimeError("Unexpected multiclass decision-function shape.")
    return scores


# ---------------------------------------------------------------------------
# Inner search
# ---------------------------------------------------------------------------


def _select_candidate(candidates, selection_rule):
    """Return ``(index, one_se_cutoff)`` using a deterministic rule."""
    if selection_rule not in {"best", "one_se"}:
        raise ValueError("selection_rule must be 'best' or 'one_se'.")
    if not candidates:
        raise ValueError("At least one candidate is required.")

    means = np.asarray([candidate.mean_score for candidate in candidates])
    best_mean = float(np.max(means))
    best_score_indices = np.flatnonzero(
        np.isclose(means, best_mean, rtol=1e-12, atol=1e-12)
    )
    # Resolve exact performance ties by simplicity.
    empirical_best = min(
        best_score_indices,
        key=lambda index: (
            candidates[index].mean_selected_features,
            -candidates[index].consensus_threshold,
            -candidates[index].alpha,
        ),
    )

    if selection_rule == "best":
        return int(empirical_best), None

    cutoff = best_mean - candidates[empirical_best].standard_error
    eligible = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.mean_score >= cutoff - 1e-12
    ]
    selected = min(
        eligible,
        key=lambda index: (
            candidates[index].mean_selected_features,
            -candidates[index].consensus_threshold,
            -candidates[index].alpha,
            -candidates[index].mean_score,
        ),
    )
    return int(selected), float(cutoff)


def _run_inner_search(
    X,
    y,
    groups,
    *,
    selection_groups,
    global_indices,
    transformer,
    selector,
    inner_cv,
    consensus_thresholds,
    classifier_alphas,
    scoring,
    selection_rule,
    class_weight,
    ridge_solver,
    random_state,
):
    splitter = _make_splitter(
        inner_cv,
        groups=groups,
        random_state=random_state,
        name="inner_cv",
    )
    splits, _ = _materialize_splits(
        splitter,
        X,
        y,
        groups,
        name="inner_cv",
        require_partition=False,
    )
    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
    if not callable(scorer):
        raise TypeError("scoring must be a scorer name or callable scorer.")

    n_splits = len(splits)
    n_thresholds = consensus_thresholds.size
    n_alphas = classifier_alphas.size
    scores = np.empty((n_splits, n_thresholds, n_alphas), dtype=float)
    counts = np.empty((n_splits, n_thresholds), dtype=np.int64)
    fold_results = []

    for fold_index, (train_local, validation_local) in enumerate(splits):
        fold_seed = int(random_state + fold_index + 1)
        transformer_fold = clone(transformer)
        transformer_parameters = transformer_fold.get_params(deep=False)
        if (
            "random_state" in transformer_parameters
            and transformer_parameters["random_state"] is None
        ):
            transformer_fold.set_params(random_state=fold_seed)
        transformer_fold.fit(X[train_local], y[train_local])
        transformed_train = transformer_fold.transform(X[train_local])
        transformed_validation = transformer_fold.transform(X[validation_local])

        selector_fold = clone(selector)
        selector_parameters = selector_fold.get_params(deep=False)
        if (
            "random_state" in selector_parameters
            and selector_parameters["random_state"] is None
        ):
            selector_fold.set_params(random_state=fold_seed)
        fold_groups = (
            None
            if selection_groups is None
            else selection_groups[train_local]
        )
        _fit_selector(
            selector_fold,
            transformed_train,
            y[train_local],
            fold_groups,
        )

        for threshold_index, threshold in enumerate(consensus_thresholds):
            selected = _selector_indices(
                selector_fold, threshold, n_features=transformed_train.shape[1]
            )
            counts[fold_index, threshold_index] = selected.size
            scaler = StandardScaler()
            train_selected = scaler.fit_transform(
                transformed_train[:, selected]
            )
            validation_selected = scaler.transform(
                transformed_validation[:, selected]
            )
            for alpha_index, alpha in enumerate(classifier_alphas):
                classifier = RidgeClassifier(
                    alpha=float(alpha),
                    class_weight=class_weight,
                    solver=ridge_solver,
                )
                classifier.fit(train_selected, y[train_local])
                scores[fold_index, threshold_index, alpha_index] = (
                    _candidate_score(
                        scorer,
                        classifier,
                        validation_selected,
                        y[validation_local],
                    )
                )

        fold_results.append(
            InnerFoldResult(
                fold_index=fold_index,
                train_indices=np.asarray(
                    global_indices[train_local], dtype=np.int64
                ),
                validation_indices=np.asarray(
                    global_indices[validation_local], dtype=np.int64
                ),
                n_transform_features=int(transformed_train.shape[1]),
                selector_stability=float(
                    getattr(selector_fold, "nogueira_stability_", np.nan)
                ),
                selector_n_resamples=int(
                    getattr(
                        selector_fold,
                        "n_resamples_",
                        len(getattr(selector_fold, "cutoff_sizes_", [])),
                    )
                ),
                selector_requested_n_resamples=int(
                    getattr(
                        selector_fold,
                        "requested_n_resamples_",
                        getattr(selector_fold, "n_resamples", 0),
                    )
                ),
                selector_cutoff_sizes=np.asarray(
                    getattr(selector_fold, "cutoff_sizes_", []),
                    dtype=np.int64,
                ).copy(),
                selector_cutoff_improvements=np.asarray(
                    getattr(selector_fold, "cutoff_improvements_", []),
                    dtype=float,
                ).copy(),
                selected_counts=counts[fold_index].copy(),
            )
        )

    candidates = []
    for threshold_index, threshold in enumerate(consensus_thresholds):
        for alpha_index, alpha in enumerate(classifier_alphas):
            fold_scores = scores[:, threshold_index, alpha_index].copy()
            selected_counts = counts[:, threshold_index].copy()
            mean_score = float(np.mean(fold_scores))
            std_score = (
                float(np.std(fold_scores, ddof=1))
                if fold_scores.size > 1
                else 0.0
            )
            standard_error = std_score / np.sqrt(fold_scores.size)
            candidates.append(
                InnerCandidateResult(
                    consensus_threshold=float(threshold),
                    alpha=float(alpha),
                    fold_scores=fold_scores,
                    selected_counts=selected_counts,
                    mean_score=mean_score,
                    std_score=std_score,
                    standard_error=float(standard_error),
                    mean_selected_features=float(
                        np.mean(selected_counts)
                    ),
                )
            )

    best_index, one_se_cutoff = _select_candidate(
        candidates, selection_rule
    )
    best = candidates[best_index]
    return InnerSearchResult(
        candidates=tuple(candidates),
        fold_results=tuple(fold_results),
        consensus_thresholds=consensus_thresholds.copy(),
        classifier_alphas=classifier_alphas.copy(),
        best_candidate_index=best_index,
        best_consensus_threshold=best.consensus_threshold,
        best_alpha=best.alpha,
        best_mean_score=best.mean_score,
        best_standard_error=best.standard_error,
        one_se_cutoff=one_se_cutoff,
        selection_rule=selection_rule,
        scoring=scoring,
    )


# ---------------------------------------------------------------------------
# Public nested-CV entry point
# ---------------------------------------------------------------------------


def nested_stability_cv(
    X,
    y,
    *,
    groups=None,
    transformer=None,
    selector=None,
    outer_cv=10,
    inner_cv=3,
    consensus_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
    classifier_alphas=None,
    scoring="balanced_accuracy",
    selection_rule="one_se",
    class_weight=None,
    ridge_solver="lsqr",
    random_state=42,
    resample_groups=True,
    refit=True,
    store_outer_models=False,
    verbose=False,
    show_progress=None,
):
    """Run leakage-free nested validation for I-ROCKET feature selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_timepoints)
        Raw univariate time series.
    y : array-like of shape (n_samples,)
        Class labels.
    groups : array-like or None, default=None
        Independent experimental units. When supplied, default outer and inner
        splitters keep groups disjoint and the selector resamples whole groups.
    transformer : estimator or None, default=None
        Prototype cloned separately inside every inner and outer training fit.
        ``None`` creates ``InterpRocketTransform(random_state=random_state)``.
    selector : estimator or None, default=None
        Prototype cloned separately inside every inner and outer training fit.
        It must expose ``get_support_at_threshold``. ``None`` creates
        ``ResampledShrinkageSelector(random_state=random_state)``.
    outer_cv : int or splitter, default=10
        Outer folds used only for performance estimation. The outer test folds
        must partition every observation exactly once.
    inner_cv : int or splitter, default=3
        Inner folds used to select the consensus threshold and ridge alpha.
    consensus_thresholds : sequence of float
        Predeclared selection-probability thresholds in (0, 1].
    classifier_alphas : sequence of float or None
        Positive RidgeClassifier alpha values. ``None`` uses nine values from
        1e-4 through 1e4.
    scoring : str or callable, default='balanced_accuracy'
        Scikit-learn scorer used for inner model selection.
    selection_rule : {'one_se', 'best'}, default='one_se'
        ``one_se`` chooses the simplest candidate within one standard error of
        the best mean inner score. Simplicity is defined first by fewer selected
        features, then a higher consensus threshold, then stronger ridge
        regularization.
    class_weight : dict, 'balanced', or None, default=None
        Passed to every RidgeClassifier.
    ridge_solver : str, default='lsqr'
        Solver used for every ridge fit. ``lsqr`` avoids direct inversion of
        the highly correlated selected feature matrix.
    random_state : int, default=42
        Controls default splitters and default transform/selector objects.
    resample_groups : bool, default=True
        When ``groups`` are supplied, also preserve whole groups inside the
        resampled consensus selector. Set to ``False`` to keep group-disjoint
        outer/inner validation while estimating feature stability from
        stratified row subsamples of the training groups. This changes the
        stability estimand and should be reported explicitly.
    refit : bool, default=True
        Tune again by inner CV on all development data and fit a final model for
        interpretation. This final fit is not a performance estimate.
    store_outer_models : bool, default=False
        Retain each fitted outer model. This can use substantial memory.
    verbose : bool, default=False
        Print one concise line per outer fold and final refit.
    show_progress : bool or None, default=None
        Display a ``tqdm`` progress bar for outer-fold tuning/refitting and the
        optional final refit. ``None`` follows ``verbose``: the bar is shown
        when ``verbose=True`` and hidden otherwise. Set explicitly to ``True``
        or ``False`` to control the bar independently. If ``tqdm`` is not
        installed, validation continues without a bar and emits a warning.

    Returns
    -------
    NestedCVResult
        Outer predictions and metrics, fold-specific selection diagnostics,
        inner searches, and the optional final interpretation model.

    Notes
    -----
    Feature indices are coherent within one fitted transform only. Exact column
    indices from independently fitted outer transforms must not be pooled into a
    single Nogueira calculation. The returned fold metadata should instead be
    summarized by kernel, dilation, representation, pooling operator, and
    receptive field.
    """
    random_state = _validate_random_state(random_state)
    if not isinstance(resample_groups, (bool, np.bool_)):
        raise TypeError("resample_groups must be a boolean.")
    if not isinstance(refit, (bool, np.bool_)):
        raise TypeError("refit must be a boolean.")
    if not isinstance(store_outer_models, (bool, np.bool_)):
        raise TypeError("store_outer_models must be a boolean.")
    if not isinstance(verbose, (bool, np.bool_)):
        raise TypeError("verbose must be a boolean.")
    if show_progress is not None and not isinstance(
        show_progress, (bool, np.bool_)
    ):
        raise TypeError("show_progress must be a boolean or None.")
    progress_enabled = bool(verbose) if show_progress is None else bool(
        show_progress
    )
    if selection_rule not in {"one_se", "best"}:
        raise ValueError("selection_rule must be 'one_se' or 'best'.")
    if not isinstance(ridge_solver, str) or not ridge_solver:
        raise TypeError("ridge_solver must be a non-empty string.")

    X = check_array(
        X,
        dtype=np.float32,
        ensure_2d=True,
        allow_nd=False,
        accept_sparse=False,
    )
    y = column_or_1d(y)
    check_consistent_length(X, y)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("At least two classes are required.")
    if groups is not None:
        groups = column_or_1d(groups)
        check_consistent_length(X, groups)

    thresholds = _validate_grid(
        consensus_thresholds,
        name="consensus_thresholds",
        lower=0.0,
        upper=1.0,
        closed_upper=True,
    )
    if classifier_alphas is None:
        classifier_alphas = np.logspace(-4, 4, 9)
    alphas = _validate_grid(
        classifier_alphas,
        name="classifier_alphas",
        lower=0.0,
    )

    transformer_prototype = (
        InterpRocketTransform(random_state=random_state)
        if transformer is None
        else transformer
    )
    selector_prototype = (
        ResampledShrinkageSelector(random_state=random_state)
        if selector is None
        else selector
    )
    # Fail early on non-cloneable prototypes and missing selector API.
    clone(transformer_prototype)
    selector_check = clone(selector_prototype)
    if not hasattr(selector_check, "get_support_at_threshold"):
        raise TypeError(
            "selector must implement get_support_at_threshold()."
        )

    outer_splitter = _make_splitter(
        outer_cv,
        groups=groups,
        random_state=random_state,
        name="outer_cv",
    )
    outer_splits, outer_test_counts = _materialize_splits(
        outer_splitter,
        X,
        y,
        groups,
        name="outer_cv",
        require_partition=True,
    )

    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
    if not callable(scorer):
        raise TypeError("scoring must be a scorer name or callable scorer.")

    predictions = np.empty(y.shape, dtype=y.dtype)
    if classes.size == 2:
        decision_scores = np.empty(y.shape[0], dtype=float)
    else:
        decision_scores = np.empty((y.shape[0], classes.size), dtype=float)

    progress_bar = None
    if progress_enabled:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            warnings.warn(
                "show_progress=True requires tqdm. Continuing without a "
                "progress bar; install tqdm to enable it.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            n_progress_stages = 2 * len(outer_splits) + (2 if refit else 0)
            progress_bar = tqdm(
                total=n_progress_stages,
                desc="Nested CV",
                unit="stage",
                dynamic_ncols=True,
            )

    def _set_progress(description, **postfix):
        if progress_bar is None:
            return
        progress_bar.set_description_str(description, refresh=True)
        if postfix:
            progress_bar.set_postfix(postfix, refresh=True)

    def _advance_progress():
        if progress_bar is not None:
            progress_bar.update(1)

    def _emit(message):
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message)

    if progress_bar is not None or verbose:
        n_candidates = int(thresholds.size * alphas.size)
        _emit(
            "Starting nested CV. The first progress update occurs only after "
            "the complete first inner search. That stage fits the transform "
            "and selector in each inner fold, runs resampled feature "
            f"selection, and evaluates {n_candidates} threshold/alpha "
            "combinations. On first use, Numba compilation may add extra "
            "delay. A long initial pause is expected and does not by itself "
            "indicate a stalled run."
        )

    outer_results = []
    for outer_index, (train_indices, test_indices) in enumerate(outer_splits):
        X_train = X[train_indices]
        y_train = y[train_indices]
        groups_train = None if groups is None else groups[train_indices]

        _set_progress(
            f"Outer {outer_index + 1}/{len(outer_splits)}: inner search"
        )
        inner_search = _run_inner_search(
            X_train,
            y_train,
            groups_train,
            selection_groups=(
                groups_train if resample_groups else None
            ),
            global_indices=train_indices,
            transformer=transformer_prototype,
            selector=selector_prototype,
            inner_cv=inner_cv,
            consensus_thresholds=thresholds,
            classifier_alphas=alphas,
            scoring=scoring,
            selection_rule=selection_rule,
            class_weight=class_weight,
            ridge_solver=ridge_solver,
            random_state=random_state + 1000 * (outer_index + 1),
        )
        _advance_progress()
        _set_progress(
            f"Outer {outer_index + 1}/{len(outer_splits)}: refit/evaluate",
            threshold=f"{inner_search.best_consensus_threshold:.3f}",
            alpha=f"{inner_search.best_alpha:g}",
        )

        model = StableRocketClassifier(
            transformer=transformer_prototype,
            selector=selector_prototype,
            consensus_threshold=inner_search.best_consensus_threshold,
            alpha=inner_search.best_alpha,
            class_weight=class_weight,
            ridge_solver=ridge_solver,
            random_state=random_state + 1000 * (outer_index + 1) + 1,
        )
        model.fit(
            X_train,
            y_train,
            groups=(groups_train if resample_groups else None),
        )
        fold_predictions = model.predict(X[test_indices])
        fold_decision = _decision_scores_aligned(
            model, X[test_indices], classes
        )
        predictions[test_indices] = fold_predictions
        decision_scores[test_indices] = fold_decision
        metrics = _classification_metrics(y[test_indices], fold_predictions)
        primary_score = _candidate_score(
            scorer, model, X[test_indices], y[test_indices]
        )

        selector_fitted = model.selector_
        result = OuterFoldResult(
            fold_index=outer_index,
            train_indices=train_indices.copy(),
            test_indices=test_indices.copy(),
            inner_search=inner_search,
            best_consensus_threshold=float(
                inner_search.best_consensus_threshold
            ),
            best_alpha=float(inner_search.best_alpha),
            predictions=np.asarray(fold_predictions).copy(),
            decision_scores=np.asarray(fold_decision).copy(),
            metrics=metrics,
            primary_score=float(primary_score),
            n_transform_features=int(model.n_transform_features_),
            n_selected_features=int(model.n_selected_features_),
            nogueira_stability=float(
                getattr(selector_fitted, "nogueira_stability_", np.nan)
            ),
            selector_n_resamples=int(
                getattr(
                    selector_fitted,
                    "n_resamples_",
                    len(getattr(selector_fitted, "cutoff_sizes_", [])),
                )
            ),
            selector_requested_n_resamples=int(
                getattr(
                    selector_fitted,
                    "requested_n_resamples_",
                    getattr(selector_fitted, "n_resamples", 0),
                )
            ),
            cutoff_sizes=np.asarray(
                getattr(selector_fitted, "cutoff_sizes_", []), dtype=np.int64
            ).copy(),
            selection_probabilities=np.asarray(
                getattr(selector_fitted, "selection_probabilities_", []),
                dtype=float,
            ).copy(),
            selected_indices=model.selected_indices_.copy(),
            feature_metadata=model.get_selected_feature_metadata(),
            model=model if store_outer_models else None,
        )
        outer_results.append(result)
        _advance_progress()
        _set_progress(
            f"Completed outer {outer_index + 1}/{len(outer_splits)}",
            balanced_accuracy=f"{metrics['balanced_accuracy']:.4f}",
            features=model.n_selected_features_,
            stability=f"{result.nogueira_stability:.3f}",
        )
        if verbose:
            _emit(
                f"Outer fold {outer_index + 1}/{len(outer_splits)}: "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                f"threshold={inner_search.best_consensus_threshold:.3f}, "
                f"alpha={inner_search.best_alpha:g}, "
                f"features={model.n_selected_features_}, "
                f"stability={result.nogueira_stability:.3f}"
            )

    metric_names = tuple(outer_results[0].metrics)
    mean_metrics = {
        metric: float(
            np.mean([fold.metrics[metric] for fold in outer_results])
        )
        for metric in metric_names
    }
    std_metrics = {
        metric: float(
            np.std(
                [fold.metrics[metric] for fold in outer_results], ddof=1
            )
        )
        if len(outer_results) > 1
        else 0.0
        for metric in metric_names
    }
    pooled_metrics = _classification_metrics(y, predictions)

    outer_scores = np.asarray(
        [fold.primary_score for fold in outer_results], dtype=float
    )

    final_search = None
    final_model = None
    best_parameters = None
    feature_metadata = tuple()
    if refit:
        _set_progress("Final model: inner search")
        final_search = _run_inner_search(
            X,
            y,
            groups,
            selection_groups=(groups if resample_groups else None),
            global_indices=np.arange(X.shape[0], dtype=np.int64),
            transformer=transformer_prototype,
            selector=selector_prototype,
            inner_cv=inner_cv,
            consensus_thresholds=thresholds,
            classifier_alphas=alphas,
            scoring=scoring,
            selection_rule=selection_rule,
            class_weight=class_weight,
            ridge_solver=ridge_solver,
            random_state=random_state + 1_000_000,
        )
        _advance_progress()
        _set_progress(
            "Final model: refit",
            threshold=f"{final_search.best_consensus_threshold:.3f}",
            alpha=f"{final_search.best_alpha:g}",
        )
        final_model = StableRocketClassifier(
            transformer=transformer_prototype,
            selector=selector_prototype,
            consensus_threshold=final_search.best_consensus_threshold,
            alpha=final_search.best_alpha,
            class_weight=class_weight,
            ridge_solver=ridge_solver,
            random_state=random_state + 1_000_001,
        ).fit(
            X,
            y,
            groups=(groups if resample_groups else None),
        )
        best_parameters = {
            "consensus_threshold": float(
                final_search.best_consensus_threshold
            ),
            "alpha": float(final_search.best_alpha),
        }
        feature_metadata = final_model.get_selected_feature_metadata()
        _advance_progress()
        _set_progress(
            "Nested CV complete",
            features=final_model.n_selected_features_,
        )
        if verbose:
            _emit(
                "Final interpretation model: "
                f"threshold={final_search.best_consensus_threshold:.3f}, "
                f"alpha={final_search.best_alpha:g}, "
                f"features={final_model.n_selected_features_}, "
                "performance must be reported from the outer folds"
            )

    if progress_bar is not None:
        progress_bar.close()

    return NestedCVResult(
        outer_fold_results=tuple(outer_results),
        outer_scores=outer_scores,
        outer_predictions=predictions,
        outer_decision_scores=decision_scores,
        outer_test_counts=outer_test_counts,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        pooled_metrics=pooled_metrics,
        selected_counts=np.asarray(
            [fold.n_selected_features for fold in outer_results],
            dtype=np.int64,
        ),
        nogueira_stabilities=np.asarray(
            [fold.nogueira_stability for fold in outer_results], dtype=float
        ),
        cutoff_distributions=tuple(
            fold.cutoff_sizes.copy() for fold in outer_results
        ),
        classes=classes,
        scoring=scoring,
        resample_groups=bool(resample_groups),
        final_search=final_search,
        final_model=final_model,
        best_parameters=best_parameters,
        feature_metadata=feature_metadata,
    )


__all__ = [
    "InnerCandidateResult",
    "InnerFoldResult",
    "InnerSearchResult",
    "OuterFoldResult",
    "NestedCVResult",
    "StableRocketClassifier",
    "nested_stability_cv",
]
