"""Internal feature-selection implementation used by I-ROCKET."""

from .results import (
    NogueiraStabilityResult,
    ScoreResult,
    SegmentedCutoffResult,
    SelectionResult,
)
from .shrinkage import screen_features, shrinkage_t, shrinkage_t_ovr
from .sklearn import ResampledShrinkageSelector, ShrinkageFeatureSelector
from .stability import nogueira_stability
from .thresholds import segmented_cutoff

__all__ = [
    "ScoreResult",
    "SelectionResult",
    "SegmentedCutoffResult",
    "NogueiraStabilityResult",
    "shrinkage_t",
    "shrinkage_t_ovr",
    "screen_features",
    "segmented_cutoff",
    "nogueira_stability",
    "ShrinkageFeatureSelector",
    "ResampledShrinkageSelector",
]
