"""Small regression checks for user-visible progress reporting."""

from __future__ import annotations

import inspect

import interpretability
import irocket_model_selection


def test_nested_cv_source_contains_initial_delay_notice():
    source = inspect.getsource(irocket_model_selection.nested_stability_cv)
    assert "The first progress update occurs only after" in source
    assert "Numba compilation may add extra" in source
    assert "long initial pause is expected" in source
    assert "threshold/alpha" in source
    assert "combinations" in source


def test_composite_trace_source_uses_tqdm_auto():
    source = inspect.getsource(interpretability.composite_activation_trace)
    assert "from tqdm.auto import tqdm" in source
    assert "Composite activation trace" in source
    assert "unit=\"kernel group\"" in source
