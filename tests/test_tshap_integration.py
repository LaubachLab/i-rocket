import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier

from tshap_integration import (
    TSHAPResult,
    explain_with_tshap,
    make_tshap_predictor,
    plot_tshap_attribution,
)


def _fit_binary_model():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(30, 20))
    y = (X[:, 5:10].mean(axis=1) > 0).astype(int)
    return X, y, RidgeClassifier().fit(X, y)


def test_make_tshap_predictor_binary_orientation_and_shapes():
    X, _, model = _fit_binary_model()
    positive = make_tshap_predictor(model, 1, output="decision")
    negative = make_tshap_predictor(model, 0, output="decision")
    positive_values = positive(X[:4])
    negative_values = negative(X[:4, np.newaxis, :])
    assert positive_values.shape == (4,)
    assert np.allclose(positive_values, -negative_values)

    probability = make_tshap_predictor(model, 1, output="probability")
    values = probability(X[:4])
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_explain_with_tshap_uses_optional_adapter(monkeypatch):
    X, _, model = _fit_binary_model()

    class FakeExplainer:
        def __init__(self, window_length, stride, interpolation, roi):
            self.window_length = window_length
            self.stride = stride
            self.interpolation = interpolation
            self.roi = roi

        def explain(self, X_group, baselines, predictor, clf_targets=None):
            assert clf_targets is None
            payouts = np.asarray(predictor(X_group), dtype=float)
            window = np.repeat(
                payouts[:, np.newaxis, np.newaxis], X_group.shape[2], axis=2
            )
            roi = window * 0.5
            return window, roi

    fake_module = types.ModuleType("tshap")
    fake_module.TSHAPExplainer = FakeExplainer
    monkeypatch.setitem(sys.modules, "tshap", fake_module)

    result = explain_with_tshap(
        model,
        X[:4],
        X[10:13],
        targets=np.array([0, 1, 0, 1]),
        window_length=5,
        stride=2,
    )
    assert isinstance(result, TSHAPResult)
    assert result.window_attributions.shape == (4, 20)
    assert result.roi_attributions.shape == (4, 20)
    assert np.array_equal(result.targets, [0, 1, 0, 1])

    ax = plot_tshap_attribution(result, X=X[:4], sample_index=1)
    assert "target" in ax.get_title()
    plt.close(ax.figure)


def test_tshap_validation_and_missing_dependency(monkeypatch):
    X, _, model = _fit_binary_model()
    with pytest.raises(ValueError, match="one channel"):
        make_tshap_predictor(model, 1)(np.ones((2, 2, 20)))
    with pytest.raises(ValueError, match="not in model.classes"):
        make_tshap_predictor(model, 99)

    monkeypatch.delitem(sys.modules, "tshap", raising=False)
    monkeypatch.delitem(sys.modules, "tshap.tshap", raising=False)
    # Force both import paths to fail even if tshap is installed in a future
    # test environment.
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tshap" or name.startswith("tshap."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ImportError, match="optional"):
        explain_with_tshap(model, X[:1], X[2:4], window_length=5)
