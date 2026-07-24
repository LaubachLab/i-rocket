"""Regression tests for release defaults and long-running progress feedback."""

from __future__ import annotations

import inspect

import numpy as np

from irocket_model_selection import nested_stability_cv
from interpretability import composite_activation_trace


def test_nested_cv_standard_defaults():
    signature = inspect.signature(nested_stability_cv)
    assert signature.parameters["outer_cv"].default == 10
    assert signature.parameters["inner_cv"].default == 3
    assert signature.parameters["show_progress"].default is None


def test_composite_trace_progress_controls_are_public():
    signature = inspect.signature(composite_activation_trace)
    assert signature.parameters["show_progress"].default is None
    assert signature.parameters["progress_threshold"].default == 500


def test_progress_arguments_reject_invalid_values_before_computation():
    X = np.zeros((2, 10), dtype=np.float32)

    class Unfitted:
        n_output_features_ = 1

    try:
        composite_activation_trace(X, Unfitted(), selected_features=[0], show_progress="yes")
    except TypeError as exc:
        assert "show_progress" in str(exc)
    else:
        raise AssertionError("invalid show_progress value was accepted")

    try:
        composite_activation_trace(
            X,
            Unfitted(),
            selected_features=[0],
            progress_threshold=0,
        )
    except ValueError as exc:
        assert "progress_threshold" in str(exc)
    else:
        raise AssertionError("invalid progress_threshold was accepted")
