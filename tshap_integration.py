"""Optional integration with the external TSHAP package.

TSHAP is a model-agnostic time-series attribution method introduced by
Le Nguyen and Ifrim (2025).  This module contains only I-ROCKET-owned adapter
and plotting code.  The external ``tshap`` package is imported only when an
explanation is requested and remains an optional dependency.

The upstream TSHAP project is GPL-3.0 licensed.  I-ROCKET does not copy or
redistribute its source code.  Install the tested PyPI release with
``pip install 'interp-rocket[tshap]'``.

AUTHOR: Mark Laubach (American University, Department of Neuroscience)
LICENSE: BSD-3-Clause
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted, column_or_1d


@dataclass(frozen=True)
class TSHAPResult:
    """Window and region-of-interest attributions returned by TSHAP."""

    window_attributions: np.ndarray
    roi_attributions: np.ndarray
    targets: np.ndarray
    classes: np.ndarray
    window_length: int
    stride: int
    interpolation: bool
    roi: bool
    output: str
    tshap_version: str


def _import_tshap_explainer():
    try:
        from tshap import TSHAPExplainer
    except (ImportError, AttributeError):
        try:
            from tshap.tshap import TSHAPExplainer
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "TSHAP support is optional. Install the tested dependency with "
                "`python -m pip install 'interp-rocket[tshap]'`."
            ) from exc
    return TSHAPExplainer


def _tshap_version() -> str:
    try:
        return importlib_metadata.version("tshap")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _as_univariate_3d(X, *, name: str) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, np.newaxis, :]
    elif values.ndim != 3:
        raise ValueError(
            f"{name} must have shape (samples, timepoints) or "
            "(samples, 1, timepoints)."
        )
    if values.shape[1] != 1:
        raise ValueError(
            f"{name} contains {values.shape[1]} channels; the current "
            "I-ROCKET transform is univariate and requires one channel."
        )
    if values.shape[0] < 1 or values.shape[2] < 2:
        raise ValueError(f"{name} must contain samples and at least two timepoints.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _model_classes(model) -> np.ndarray:
    check_is_fitted(model, attributes=["classes_"])
    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or classes.size < 2:
        raise ValueError("model.classes_ must contain at least two classes.")
    return classes


def _target_index(classes: np.ndarray, target_class) -> int:
    matches = np.flatnonzero(classes == target_class)
    if matches.size != 1:
        raise ValueError(f"target_class {target_class!r} is not in model.classes_.")
    return int(matches[0])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0
    output = np.empty_like(values, dtype=float)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


def make_tshap_predictor(
    model,
    target_class,
    *,
    output: str = "decision",
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a scalar-output predictor accepted by TSHAP.

    Parameters
    ----------
    model : fitted classifier
        Must expose ``classes_`` and either ``decision_function`` or
        ``predict_proba``.
    target_class : object
        Class whose score is explained.
    output : {'decision', 'probability'}, default='decision'
        ``decision`` uses the classifier's decision function. ``probability``
        uses ``predict_proba`` when available; otherwise it maps binary decision
        scores through a logistic function and multiclass scores through a
        softmax. The latter are probability-like calibration-free scores, not
        calibrated probabilities.

    Returns
    -------
    callable
        Function accepting either two-dimensional univariate arrays or TSHAP's
        three-dimensional ``(samples, 1, timepoints)`` arrays.
    """
    if output not in {"decision", "probability"}:
        raise ValueError("output must be 'decision' or 'probability'.")
    classes = _model_classes(model)
    target_index = _target_index(classes, target_class)
    has_decision = hasattr(model, "decision_function") and callable(
        model.decision_function
    )
    has_probability = hasattr(model, "predict_proba") and callable(
        model.predict_proba
    )
    if not has_decision and not has_probability:
        raise TypeError(
            "model must implement decision_function() or predict_proba()."
        )

    def predictor(data):
        values = _as_univariate_3d(data, name="data")[:, 0, :]
        if output == "probability" and has_probability:
            probabilities = np.asarray(model.predict_proba(values), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[1] != classes.size:
                raise ValueError(
                    "predict_proba() returned an array incompatible with "
                    "model.classes_."
                )
            return probabilities[:, target_index]

        if has_decision:
            scores = np.asarray(model.decision_function(values), dtype=float)
            if classes.size == 2:
                if scores.ndim == 2:
                    if scores.shape[1] == 1:
                        scores = scores[:, 0]
                    elif scores.shape[1] == 2:
                        scores = scores[:, 1]
                    else:
                        raise ValueError(
                            "Binary decision_function() returned an unexpected "
                            "number of columns."
                        )
                scores = scores.reshape(-1)
                # scikit-learn binary decision scores are oriented toward
                # classes_[1]. Reverse the sign when explaining classes_[0].
                target_scores = scores if target_index == 1 else -scores
                return _sigmoid(target_scores) if output == "probability" else target_scores

            if scores.ndim != 2 or scores.shape[1] != classes.size:
                raise ValueError(
                    "Multiclass decision_function() returned an array "
                    "incompatible with model.classes_."
                )
            if output == "probability":
                return _softmax(scores)[:, target_index]
            return scores[:, target_index]

        probabilities = np.asarray(model.predict_proba(values), dtype=float)
        return probabilities[:, target_index]

    return predictor


def _resolve_targets(model, X_2d: np.ndarray, targets) -> np.ndarray:
    classes = _model_classes(model)
    if targets is None:
        if not hasattr(model, "predict") or not callable(model.predict):
            raise TypeError("targets are required when model has no predict().")
        values = np.asarray(model.predict(X_2d))
    elif np.isscalar(targets) or isinstance(targets, str):
        values = np.full(X_2d.shape[0], targets, dtype=object)
    else:
        values = column_or_1d(targets)
        if values.shape[0] != X_2d.shape[0]:
            raise ValueError("targets must contain one class per explained sample.")
    for target in np.unique(values):
        _target_index(classes, target)
    return values


def explain_with_tshap(
    model,
    X,
    baselines,
    *,
    targets=None,
    window_length: int = 20,
    stride: int = 5,
    interpolation: bool = True,
    roi: bool = True,
    output: str = "decision",
) -> TSHAPResult:
    """Explain fitted classifier outputs with the external TSHAP package.

    Baselines are explicit because attribution values depend on the replacement
    distribution. They should be drawn only from the training data used to fit
    the explained model. Test observations must not be used as baselines.

    ``targets=None`` explains each model prediction. A scalar target explains
    the same class for every sample, and a one-dimensional target array permits
    sample-specific classes.
    """
    if isinstance(window_length, (bool, np.bool_)) or not isinstance(
        window_length, (int, np.integer)
    ):
        raise TypeError("window_length must be an integer.")
    if isinstance(stride, (bool, np.bool_)) or not isinstance(
        stride, (int, np.integer)
    ):
        raise TypeError("stride must be an integer.")
    window_length = int(window_length)
    stride = int(stride)
    if window_length < 1 or stride < 1:
        raise ValueError("window_length and stride must be positive.")
    if not isinstance(interpolation, (bool, np.bool_)):
        raise TypeError("interpolation must be a boolean.")
    if not isinstance(roi, (bool, np.bool_)):
        raise TypeError("roi must be a boolean.")
    if output not in {"decision", "probability"}:
        raise ValueError("output must be 'decision' or 'probability'.")

    X_3d = _as_univariate_3d(X, name="X")
    baselines_3d = _as_univariate_3d(baselines, name="baselines")
    if baselines_3d.shape[2] != X_3d.shape[2]:
        raise ValueError("X and baselines must have the same number of timepoints.")
    if window_length > X_3d.shape[2]:
        raise ValueError("window_length cannot exceed the series length.")

    classes = _model_classes(model)
    target_values = _resolve_targets(model, X_3d[:, 0, :], targets)
    TSHAPExplainer = _import_tshap_explainer()
    explainer = TSHAPExplainer(
        window_length=window_length,
        stride=stride,
        interpolation=bool(interpolation),
        roi=bool(roi),
    )

    window_values = np.zeros_like(X_3d, dtype=float)
    roi_values = np.zeros_like(X_3d, dtype=float)
    # The upstream implementation has a target-handling path that assumes the
    # supplied model object owns classes_. Grouping by target lets us pass a
    # scalar callable and avoids relying on that internal assumption.
    for target in np.unique(target_values):
        sample_mask = target_values == target
        predictor = make_tshap_predictor(model, target, output=output)
        window_group, roi_group = explainer.explain(
            X_3d[sample_mask],
            baselines_3d,
            predictor,
            clf_targets=None,
        )
        window_group = np.asarray(window_group, dtype=float)
        roi_group = np.asarray(roi_group, dtype=float)
        expected_shape = X_3d[sample_mask].shape
        if window_group.shape != expected_shape or roi_group.shape != expected_shape:
            raise ValueError(
                "TSHAP returned attribution arrays with unexpected shapes."
            )
        window_values[sample_mask] = window_group
        roi_values[sample_mask] = roi_group

    return TSHAPResult(
        window_attributions=window_values[:, 0, :],
        roi_attributions=roi_values[:, 0, :],
        targets=np.asarray(target_values).copy(),
        classes=classes.copy(),
        window_length=window_length,
        stride=stride,
        interpolation=bool(interpolation),
        roi=bool(roi),
        output=output,
        tshap_version=_tshap_version(),
    )


def plot_tshap_attribution(
    result: TSHAPResult,
    *,
    X=None,
    sample_index: int = 0,
    attribution: str = "window",
    time=None,
    ax=None,
    show_signal: bool = True,
):
    """Plot one TSHAP attribution trace, optionally with the input signal."""
    if not isinstance(result, TSHAPResult):
        raise TypeError("result must be a TSHAPResult.")
    if isinstance(sample_index, (bool, np.bool_)) or not isinstance(
        sample_index, (int, np.integer)
    ):
        raise TypeError("sample_index must be an integer.")
    sample_index = int(sample_index)
    if sample_index < 0 or sample_index >= result.window_attributions.shape[0]:
        raise IndexError("sample_index is out of range.")
    if attribution not in {"window", "roi"}:
        raise ValueError("attribution must be 'window' or 'roi'.")
    if not isinstance(show_signal, (bool, np.bool_)):
        raise TypeError("show_signal must be a boolean.")

    values = (
        result.window_attributions[sample_index]
        if attribution == "window"
        else result.roi_attributions[sample_index]
    )
    if time is None:
        time_values = np.arange(values.size)
    else:
        time_values = np.asarray(time, dtype=float)
        if time_values.ndim != 1 or time_values.size != values.size:
            raise ValueError("time must contain one value per timepoint.")
        if not np.all(np.isfinite(time_values)):
            raise ValueError("time must contain only finite values.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.axhline(0.0, color="#000000", linewidth=0.8, alpha=0.5)
    ax.plot(time_values, values, color="#D55E00", linewidth=1.8, label="TSHAP")
    ax.fill_between(
        time_values,
        0.0,
        values,
        where=values >= 0.0,
        color="#D55E00",
        alpha=0.25,
        interpolate=True,
    )
    ax.fill_between(
        time_values,
        0.0,
        values,
        where=values < 0.0,
        color="#0072B2",
        alpha=0.25,
        interpolate=True,
    )
    ax.set_ylabel(f"{attribution.upper()} attribution")
    ax.set_xlabel("Time")
    ax.set_title(
        f"TSHAP attribution for target {result.targets[sample_index]!r}"
    )

    if show_signal:
        if X is None:
            raise ValueError("X is required when show_signal=True.")
        X_3d = _as_univariate_3d(X, name="X")
        if X_3d.shape[0] <= sample_index or X_3d.shape[2] != values.size:
            raise ValueError("X is incompatible with the selected attribution.")
        signal_axis = ax.twinx()
        signal_axis.plot(
            time_values,
            X_3d[sample_index, 0],
            color="#000000",
            linewidth=1.0,
            alpha=0.55,
            label="Signal",
        )
        signal_axis.set_ylabel("Signal")
    return ax


__all__ = [
    "TSHAPResult",
    "make_tshap_predictor",
    "explain_with_tshap",
    "plot_tshap_attribution",
]
