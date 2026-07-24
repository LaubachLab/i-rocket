"""Scikit-learn-compatible shrinkage feature selectors."""

from __future__ import annotations

from itertools import combinations
from math import comb
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)

from .shrinkage import screen_features, shrinkage_t, shrinkage_t_ovr
from .stability import nogueira_stability
from .thresholds import segmented_cutoff

def _validate_feature_matrix(estimator, X, *, reset):
    """Validate X and maintain ``n_features_in_`` across sklearn versions."""
    try:
        from sklearn.utils.validation import validate_data
    except ImportError:  # scikit-learn < 1.6
        validated = check_array(
            X,
            dtype="numeric",
            ensure_2d=True,
            allow_nd=False,
        )
        if reset:
            estimator.n_features_in_ = int(validated.shape[1])
        elif not hasattr(estimator, "n_features_in_"):
            raise ValueError("The estimator has not been fitted.")
        elif validated.shape[1] != estimator.n_features_in_:
            raise ValueError(
                f"X has {validated.shape[1]} features; expected "
                f"{estimator.n_features_in_}."
            )
        return validated
    return validate_data(
        estimator,
        X,
        dtype="numeric",
        ensure_2d=True,
        allow_nd=False,
        reset=reset,
    )


class ShrinkageFeatureSelector(TransformerMixin, BaseEstimator):
    """Select feature-matrix columns using shrinkage scores.

    Parameters
    ----------
    top_k : int or None, default=500
        Number of features retained when ``threshold='top_k'``.
    threshold : {'top_k', 'segmented'}, default='top_k'
        ``segmented`` fits one breakpoint in the ranked absolute score curve.
    multiclass : {'auto', 'binary', 'ovr'}, default='auto'
    cutoff_min_size : int, default=5
        Minimum number of features in each segment of the breakpoint fit.
    score_power : float, default=1.0
        Power applied to absolute scores for the segmented fit. Use ``1.0`` for
        absolute shrinkage-*t* scores; ``2.0`` is available for sensitivity
        analysis.
    """

    _LEARNED_ATTRIBUTES = (
        "breakpoint_",
        "cutoff_idx_",
        "cutoff_result_",
        "n_features_in_",
        "selected_indices_",
        "scores_",
        "ranking_",
        "result_",
        "support_mask_",
    )

    def __init__(
        self,
        top_k=500,
        threshold="top_k",
        multiclass="auto",
        aggregate=None,
        cutoff_min_size=5,
        score_power=1.0,
        verbose=False,
    ):
        self.top_k = top_k
        self.threshold = threshold
        self.multiclass = multiclass
        self.aggregate = aggregate
        self.cutoff_min_size = cutoff_min_size
        self.score_power = score_power
        self.verbose = verbose

    def _reset(self):
        for attribute in self._LEARNED_ATTRIBUTES:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def _validate_parameters(self):
        if self.threshold not in {"top_k", "segmented"}:
            raise ValueError(
                "threshold must be 'top_k' or 'segmented'."
            )
        if self.multiclass not in {"auto", "binary", "ovr"}:
            raise ValueError(
                "multiclass must be 'auto', 'binary', or 'ovr'."
            )
        if not isinstance(self.verbose, (bool, np.bool_)):
            raise ValueError("verbose must be a boolean.")

    def fit(self, X, y):
        self._reset()
        self._validate_parameters()
        X = _validate_feature_matrix(self, X, reset=True)
        if y is None:
            raise ValueError("y is required for supervised feature selection.")

        result = screen_features(
            X,
            y,
            top_k=None if self.threshold != "top_k" else self.top_k,
            multiclass=self.multiclass,
            aggregate=self.aggregate,
            verbose=self.verbose,
        )
        if self.threshold == "top_k":
            selected = result.selected
        else:
            cutoff_result = segmented_cutoff(
                result.scores,
                ranked_indices=result.ranking,
                absolute=True,
                score_power=self.score_power,
                min_size=self.cutoff_min_size,
            )
            selected = cutoff_result.selected
            self.cutoff_result_ = cutoff_result
            self.cutoff_idx_ = int(cutoff_result.cutoff_idx)
            self.breakpoint_ = int(cutoff_result.breakpoint)

        self.n_features_in_ = int(X.shape[1])
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.scores_ = np.asarray(result.scores, dtype=np.float64)
        self.ranking_ = np.asarray(result.ranking, dtype=np.int64)
        self.result_ = result
        self.support_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_mask_[self.selected_indices_] = True
        return self

    def transform(self, X):
        check_is_fitted(
            self, attributes=["selected_indices_", "n_features_in_"]
        )
        X = _validate_feature_matrix(self, X, reset=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected "
                f"{self.n_features_in_}."
            )
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        check_is_fitted(self, attributes=["support_mask_"])
        if indices:
            return self.selected_indices_.copy()
        return self.support_mask_.copy()

    def get_feature_names_out(self, input_features=None):
        """Return names corresponding to the retained input columns."""
        check_is_fitted(
            self, attributes=["selected_indices_", "n_features_in_"]
        )
        if input_features is None:
            input_features = np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)],
                dtype=object,
            )
        else:
            input_features = np.asarray(input_features, dtype=object)
            if (
                input_features.ndim != 1
                or input_features.size != self.n_features_in_
            ):
                raise ValueError(
                    "input_features must contain one name per input feature."
                )
        return input_features[self.selected_indices_]


class ResampledShrinkageSelector(TransformerMixin, BaseEstimator):
    """Consensus feature selection from resampled shrinkage-*t* cutoffs.

    The selector operates on one fixed feature matrix. For every resample it
    computes shrinkage-*t* scores, ranks their absolute values, and fits one
    segmented-regression breakpoint. The resulting binary selection masks are
    aggregated into selection probabilities. Features meeting
    ``consensus_threshold`` form the final set, and the Nogueira measure
    quantifies reproducibility across the masks.

    This is *resampled consensus selection*, not the formal stability-selection
    procedure of Meinshausen and Buehlmann. No false-positive error bound is
    claimed.

    Parameters
    ----------
    n_resamples : int, default=50
        Number of repeated subsamples.
    sample_fraction : float, default=0.5
        Fraction of observations in each subsample.
    consensus_threshold : float, default=0.7
        Minimum selection probability for the consensus set.
    cutoff_min_size : int, default=5
        Minimum number of ranked features in each breakpoint segment.
    score_power : float, default=1.0
        Power applied to absolute shrinkage-*t* scores before breakpoint fitting.
    multiclass : {'auto', 'binary', 'ovr'}, default='auto'
        Binary or one-versus-rest shrinkage scoring.
    aggregate : {'max_abs', 'l2'} or None, default=None
        Multiclass aggregation. ``None`` uses ``max_abs``.
    min_features : int, default=1
        Explicit minimum size of the consensus set. If the probability
        threshold yields fewer features, the highest consensus-ranked features
        are retained and ``consensus_fallback_`` is set to ``True``.
    random_state : int, RandomState, or None, default=None
        Controls subsampling.
    store_subsample_indices : bool, default=False
        Retain the row indices used in every resample for auditing.

    Notes on grouped resampling
    ---------------------------
    When ``groups`` are supplied, ``sample_fraction`` is applied to the number
    of groups rather than rows. Only unique whole-group subsets are used. If
    fewer valid unique subsets exist than ``n_resamples``, all valid subsets
    are used, a warning is emitted, and the actual count is stored in
    ``n_resamples_``.
    verbose : bool, default=False
        Print a concise fit summary.

    Notes
    -----
    When this selector follows ``InterpRocketTransform`` in a pipeline, the
    transform is fitted once on that pipeline's training partition. Internal
    resampling then evaluates selection on a fixed transformed feature universe.
    No held-out outer or inner validation rows should enter that pipeline fit.
    """

    _LEARNED_ATTRIBUTES = (
        "classes_",
        "consensus_fallback_",
        "consensus_ranking_",
        "cutoff_improvements_",
        "cutoff_sizes_",
        "cutoff_slopes_after_",
        "cutoff_slopes_before_",
        "full_ranking_",
        "full_scores_",
        "n_features_in_",
        "n_resamples_",
        "n_samples_in_",
        "n_selected_",
        "requested_n_resamples_",
        "nogueira_result_",
        "nogueira_stability_",
        "resample_class_counts_",
        "resample_indices_",
        "resample_sizes_",
        "selected_indices_",
        "selection_matrix_",
        "selection_probabilities_",
        "support_mask_",
    )

    def __init__(
        self,
        n_resamples=50,
        sample_fraction=0.5,
        consensus_threshold=0.7,
        cutoff_min_size=5,
        score_power=1.0,
        multiclass="auto",
        aggregate=None,
        min_features=1,
        random_state=None,
        store_subsample_indices=False,
        verbose=False,
    ):
        self.n_resamples = n_resamples
        self.sample_fraction = sample_fraction
        self.consensus_threshold = consensus_threshold
        self.cutoff_min_size = cutoff_min_size
        self.score_power = score_power
        self.multiclass = multiclass
        self.aggregate = aggregate
        self.min_features = min_features
        self.random_state = random_state
        self.store_subsample_indices = store_subsample_indices
        self.verbose = verbose

    def _reset(self):
        for attribute in self._LEARNED_ATTRIBUTES:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def _validate_parameters(self):
        if isinstance(self.n_resamples, (bool, np.bool_)) or not isinstance(
            self.n_resamples, (int, np.integer)
        ):
            raise TypeError("n_resamples must be an integer.")
        if self.n_resamples < 2:
            raise ValueError("n_resamples must be at least 2.")
        if not np.isscalar(self.sample_fraction):
            raise TypeError("sample_fraction must be a number in (0, 1).")
        if not 0.0 < float(self.sample_fraction) < 1.0:
            raise ValueError("sample_fraction must be strictly between 0 and 1.")
        if not np.isscalar(self.consensus_threshold):
            raise TypeError("consensus_threshold must be in (0, 1].")
        if not 0.0 < float(self.consensus_threshold) <= 1.0:
            raise ValueError("consensus_threshold must be in (0, 1].")
        if self.multiclass not in {"auto", "binary", "ovr"}:
            raise ValueError(
                "multiclass must be 'auto', 'binary', or 'ovr'."
            )
        if self.aggregate not in {None, "max_abs", "l2"}:
            raise ValueError("aggregate must be None, 'max_abs', or 'l2'.")
        if isinstance(self.min_features, (bool, np.bool_)) or not isinstance(
            self.min_features, (int, np.integer)
        ):
            raise TypeError("min_features must be an integer.")
        if self.min_features < 1:
            raise ValueError("min_features must be at least 1.")
        if not isinstance(
            self.store_subsample_indices, (bool, np.bool_)
        ):
            raise TypeError("store_subsample_indices must be a boolean.")
        if not isinstance(self.verbose, (bool, np.bool_)):
            raise TypeError("verbose must be a boolean.")
        # Delegate exact validation of these two parameters to the cutoff
        # function, but fail early on their basic types.
        if isinstance(self.cutoff_min_size, (bool, np.bool_)) or not isinstance(
            self.cutoff_min_size, (int, np.integer)
        ):
            raise TypeError("cutoff_min_size must be an integer.")
        if self.cutoff_min_size < 2:
            raise ValueError("cutoff_min_size must be at least 2.")
        if not np.isscalar(self.score_power) or isinstance(
            self.score_power, (bool, np.bool_)
        ):
            raise TypeError("score_power must be a positive finite number.")
        if not np.isfinite(float(self.score_power)) or float(self.score_power) <= 0:
            raise ValueError("score_power must be a positive finite number.")

    def _score(self, X, y):
        n_classes = np.unique(y).size
        if self.multiclass == "binary" and n_classes != 2:
            raise ValueError(
                "multiclass='binary' requires exactly two classes."
            )
        use_ovr = self.multiclass == "ovr" or (
            self.multiclass == "auto" and n_classes > 2
        )
        if use_ovr:
            return shrinkage_t_ovr(
                X,
                y,
                aggregate=self.aggregate or "max_abs",
                verbose=False,
            )
        return shrinkage_t(X, y, verbose=False)

    @staticmethod
    def _has_valid_class_counts(y, classes):
        return all(np.sum(y == label) >= 2 for label in classes)

    def _iter_subsamples(self, X, y, groups, classes):
        random_state = check_random_state(self.random_state)
        # Draw one stable integer seed so sklearn splitters do not retain a
        # mutable RandomState object as estimator state.
        split_seed = int(random_state.randint(np.iinfo(np.int32).max))

        if groups is None:
            splitter = StratifiedShuffleSplit(
                n_splits=int(self.n_resamples),
                train_size=float(self.sample_fraction),
                random_state=split_seed,
            )
            try:
                splits = splitter.split(X, y)
                accepted = []
                for train_indices, _ in splits:
                    if not self._has_valid_class_counts(y[train_indices], classes):
                        raise ValueError(
                            "A stratified subsample contains fewer than two "
                            "observations in at least one class. Increase "
                            "sample_fraction or provide more observations."
                        )
                    accepted.append(np.asarray(train_indices, dtype=np.int64))
                return accepted
            except ValueError as exc:
                raise ValueError(
                    "Could not generate valid stratified subsamples. Each "
                    "resample must contain at least two observations per class."
                ) from exc

        unique_groups = np.unique(groups)
        n_groups = int(unique_groups.size)
        if n_groups < 2:
            raise ValueError("Group-aware resampling requires at least two groups.")
        n_train_groups = int(np.floor(float(self.sample_fraction) * n_groups))
        if n_train_groups < 1 or n_train_groups >= n_groups:
            raise ValueError(
                "sample_fraction must retain at least one group and leave at "
                "least one group out."
            )

        requested = int(self.n_resamples)
        maximum_unique = comb(n_groups, n_train_groups)
        accepted = []
        seen = set()

        def _try_group_set(group_values):
            key = tuple(sorted(group_values))
            if key in seen:
                return
            seen.add(key)
            row_indices = np.flatnonzero(np.isin(groups, group_values)).astype(
                np.int64
            )
            if self._has_valid_class_counts(y[row_indices], classes):
                accepted.append(row_indices)

        # Enumerate moderate combinatorial spaces exactly. For large group
        # universes, sample candidate group sets without replacement and stop
        # after a generous deterministic attempt budget.
        if maximum_unique <= 100_000:
            candidates = list(
                combinations(unique_groups.tolist(), n_train_groups)
            )
            random_state.shuffle(candidates)
            for candidate in candidates:
                _try_group_set(candidate)
                if len(accepted) == requested:
                    break
        else:
            max_attempts = max(requested * 100, 2_000)
            for _ in range(max_attempts):
                candidate = random_state.choice(
                    unique_groups, size=n_train_groups, replace=False
                )
                _try_group_set(candidate.tolist())
                if len(accepted) == requested:
                    break

        if len(accepted) < 2:
            raise ValueError(
                "Group-aware resampling produced fewer than two valid unique "
                "subsamples while retaining every class with at least two rows."
            )
        if len(accepted) < requested:
            warnings.warn(
                "Only "
                f"{len(accepted)} valid unique whole-group subsamples exist "
                f"for the requested {requested}. Using all valid unique "
                "subsamples and reporting the reduced count in n_resamples_.",
                RuntimeWarning,
                stacklevel=3,
            )
        return accepted

    def fit(self, X, y, groups=None):
        self._reset()
        self._validate_parameters()
        X = _validate_feature_matrix(self, X, reset=True)
        if y is None:
            raise ValueError("y is required for supervised feature selection.")
        y = column_or_1d(y)
        check_consistent_length(X, y)
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("At least two classes are required.")
        if not self._has_valid_class_counts(y, classes):
            raise ValueError("Every class must contain at least two observations.")
        if X.shape[1] < 2 * int(self.cutoff_min_size):
            raise ValueError(
                "The feature matrix must contain at least 2 * cutoff_min_size "
                "columns."
            )
        if int(self.min_features) > X.shape[1]:
            raise ValueError("min_features cannot exceed the feature count.")

        if groups is not None:
            groups = column_or_1d(groups)
            check_consistent_length(X, groups)

        subsample_indices = self._iter_subsamples(X, y, groups, classes)
        n_resamples = len(subsample_indices)
        n_features = int(X.shape[1])
        selection_matrix = np.zeros((n_resamples, n_features), dtype=bool)
        cutoff_sizes = np.empty(n_resamples, dtype=np.int64)
        cutoff_improvements = np.empty(n_resamples, dtype=np.float64)
        slopes_before = np.empty(n_resamples, dtype=np.float64)
        slopes_after = np.empty(n_resamples, dtype=np.float64)
        resample_sizes = np.empty(n_resamples, dtype=np.int64)
        class_counts = np.empty((n_resamples, classes.size), dtype=np.int64)

        for resample_index, row_indices in enumerate(subsample_indices):
            y_resampled = y[row_indices]
            score_result = self._score(X[row_indices], y_resampled)
            cutoff_result = segmented_cutoff(
                score_result.scores,
                ranked_indices=score_result.ranking,
                absolute=True,
                score_power=float(self.score_power),
                min_size=int(self.cutoff_min_size),
            )
            selection_matrix[resample_index, cutoff_result.selected] = True
            cutoff_sizes[resample_index] = cutoff_result.breakpoint
            cutoff_improvements[resample_index] = (
                cutoff_result.relative_improvement
            )
            slopes_before[resample_index] = cutoff_result.slope_before
            slopes_after[resample_index] = cutoff_result.slope_after
            resample_sizes[resample_index] = row_indices.size
            class_counts[resample_index] = [
                np.sum(y_resampled == label) for label in classes
            ]

        stability_result = nogueira_stability(selection_matrix)
        full_score_result = self._score(X, y)
        probabilities = stability_result.selection_probabilities
        feature_indices = np.arange(n_features, dtype=np.int64)
        score_strength = np.abs(full_score_result.scores)
        consensus_ranking = np.lexsort(
            (feature_indices, -score_strength, -probabilities)
        )

        probability_mask = probabilities >= float(self.consensus_threshold)
        selected = consensus_ranking[probability_mask[consensus_ranking]]
        fallback = selected.size < int(self.min_features)
        if fallback:
            selected = consensus_ranking[: int(self.min_features)]

        support_mask = np.zeros(n_features, dtype=bool)
        support_mask[selected] = True

        self.classes_ = classes
        self.n_features_in_ = n_features
        self.requested_n_resamples_ = int(self.n_resamples)
        self.n_resamples_ = int(n_resamples)
        self.n_samples_in_ = int(X.shape[0])
        self.selection_matrix_ = selection_matrix
        self.selection_probabilities_ = probabilities.copy()
        self.cutoff_sizes_ = cutoff_sizes
        self.cutoff_improvements_ = cutoff_improvements
        self.cutoff_slopes_before_ = slopes_before
        self.cutoff_slopes_after_ = slopes_after
        self.resample_sizes_ = resample_sizes
        self.resample_class_counts_ = class_counts
        self.nogueira_result_ = stability_result
        self.nogueira_stability_ = float(stability_result.stability)
        self.full_scores_ = np.asarray(
            full_score_result.scores, dtype=np.float64
        )
        self.full_ranking_ = np.asarray(
            full_score_result.ranking, dtype=np.int64
        )
        self.consensus_ranking_ = np.asarray(
            consensus_ranking, dtype=np.int64
        )
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.support_mask_ = support_mask
        self.n_selected_ = int(selected.size)
        self.consensus_fallback_ = bool(fallback)
        if self.store_subsample_indices:
            self.resample_indices_ = tuple(
                indices.copy() for indices in subsample_indices
            )

        if self.verbose:
            print(
                "ResampledShrinkageSelector: "
                f"selected {self.n_selected_}/{n_features}, "
                f"median cutoff={np.median(cutoff_sizes):.1f}, "
                f"Nogueira stability={self.nogueira_stability_:.3f}"
            )
        return self

    def transform(self, X):
        check_is_fitted(
            self, attributes=["selected_indices_", "n_features_in_"]
        )
        X = _validate_feature_matrix(self, X, reset=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected "
                f"{self.n_features_in_}."
            )
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        check_is_fitted(self, attributes=["support_mask_"])
        if indices:
            return self.selected_indices_.copy()
        return self.support_mask_.copy()

    def get_support_at_threshold(
        self,
        threshold,
        *,
        indices=False,
        min_features=None,
    ):
        """Return a consensus mask at another probability threshold.

        This method reuses the fitted selection probabilities. It is intended
        for inner model selection, where several predeclared consensus
        thresholds must be evaluated without repeating the resampling step.
        """
        check_is_fitted(
            self,
            attributes=[
                "selection_probabilities_",
                "consensus_ranking_",
                "n_features_in_",
            ],
        )
        if not np.isscalar(threshold) or not 0.0 < float(threshold) <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        if min_features is None:
            min_features = self.min_features
        if isinstance(min_features, (bool, np.bool_)) or not isinstance(
            min_features, (int, np.integer)
        ):
            raise TypeError("min_features must be an integer.")
        min_features = int(min_features)
        if min_features < 1 or min_features > self.n_features_in_:
            raise ValueError(
                "min_features must be between 1 and the fitted feature count."
            )

        ranking = self.consensus_ranking_
        selected = ranking[
            self.selection_probabilities_[ranking] >= float(threshold)
        ]
        if selected.size < min_features:
            selected = ranking[:min_features]
        if indices:
            return selected.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        """Return names corresponding to consensus-selected columns."""
        check_is_fitted(
            self, attributes=["selected_indices_", "n_features_in_"]
        )
        if input_features is None:
            input_features = np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)],
                dtype=object,
            )
        else:
            input_features = np.asarray(input_features, dtype=object)
            if (
                input_features.ndim != 1
                or input_features.size != self.n_features_in_
            ):
                raise ValueError(
                    "input_features must contain one name per input feature."
                )
        return input_features[self.selected_indices_]
