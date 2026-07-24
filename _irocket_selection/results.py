"""Result containers for I-ROCKET's internal selection methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class ScoreResult:
    """Feature scores and ranking for one scoring method."""

    scores: np.ndarray
    ranking: np.ndarray
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    """Selected feature indices plus supporting scores and diagnostics."""

    selected: np.ndarray
    ranking: np.ndarray
    scores: np.ndarray
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentedCutoffResult:
    """One-break, two-line fit to a ranked feature-score curve.

    ``breakpoint`` is the endpoint of the retained first segment and therefore
    equals the number of selected features. ``cutoff_idx`` is the zero-based
    index of the last selected feature.
    """

    selected: np.ndarray
    ranking: np.ndarray
    ranked_scores: np.ndarray
    breakpoint: int
    cutoff_idx: int
    single_sse: float
    segmented_sse: float
    relative_improvement: float
    slope_before: float
    slope_after: float
    intercept_before: float
    intercept_after: float
    score_power: float
    min_size: int

    @property
    def n_selected(self) -> int:
        """Number of features retained before the fitted breakpoint."""
        return self.breakpoint

    @property
    def tail_slope_ratio(self) -> float:
        """Absolute post-break slope divided by the pre-break slope.

        Values near zero indicate a substantially flatter tail. Infinite is
        returned when the first segment is effectively flat.
        """
        denominator = abs(self.slope_before)
        if denominator <= np.finfo(float).eps:
            return float("inf")
        return float(abs(self.slope_after) / denominator)


@dataclass(frozen=True)
class NogueiraStabilityResult:
    """Chance-corrected feature-selection stability diagnostics."""

    stability: float
    selection_probabilities: np.ndarray
    selected_counts: np.ndarray
    mean_selected: float
    n_resamples: int
    n_features: int
    random_selection_variance: float

    def as_dict(self) -> Dict[str, Any]:
        """Return a compatibility dictionary with descriptive quantities."""
        return {
            "stability": self.stability,
            "selection_probabilities": self.selection_probabilities,
            "selected_counts": self.selected_counts,
            "k_bar": self.mean_selected,
            "M": self.n_resamples,
            "d": self.n_features,
            "random_selection_variance": self.random_selection_variance,
        }
