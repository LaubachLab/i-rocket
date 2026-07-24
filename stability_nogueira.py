"""Nogueira feature-selection stability for a fixed feature universe.

The canonical estimator is :func:`nogueira_stability`.  It accepts a binary
selection matrix whose rows are repeated selections and whose columns refer to
the same fitted I-ROCKET transform.  Exact column identities from independently
fitted transforms must not be combined in one stability calculation.

Only the point estimate and descriptive quantities are exposed.  Legacy
confidence intervals and comparison tests were removed because overlapping
resamples do not supply the independence those procedures require.

AUTHOR: Mark Laubach (American University, Department of Neuroscience)
LICENSE: BSD-3-Clause
"""

from _irocket_selection import NogueiraStabilityResult, nogueira_stability

__all__ = [
    "NogueiraStabilityResult",
    "nogueira_stability",
    "selection_stability",
]


def selection_stability(selection_matrix):
    """Return the Nogueira point estimate as a compatibility dictionary.

    New code should call :func:`nogueira_stability`, which returns a typed
    ``NogueiraStabilityResult``.  This compact wrapper preserves the established
    dictionary output without restoring the removed inferential procedures.
    """
    return nogueira_stability(selection_matrix).as_dict()
