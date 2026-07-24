"""Generic spectral helpers for I-ROCKET demonstrations.

The functions in this module operate on ordinary time-series arrays and on the
finite impulse responses implied by decoded I-ROCKET kernels.  They are not
specific to EEG, LFP, or any other experimental modality.

The main use case in the public demonstrations is FordA/FordB, where class
information is distributed in frequency rather than confined to a short event.

AUTHOR: Mark Laubach (American University, Department of Neuroscience)
LICENSE: BSD-3-Clause
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d


@dataclass(frozen=True)
class SpectrumResult:
    """Spectrum calculated along the final axis of an input array."""

    frequencies: np.ndarray
    power: np.ndarray
    method: str
    sampling_rate: float
    scaling: str


@dataclass(frozen=True)
class ClassSpectrumResult:
    """Class-conditional mean spectra and standard errors."""

    frequencies: np.ndarray
    classes: np.ndarray
    mean_power: np.ndarray
    sem_power: np.ndarray
    class_counts: np.ndarray
    method: str
    sampling_rate: float
    scaling: str


@dataclass(frozen=True)
class KernelSpectrumResult:
    """Frequency responses for decoded I-ROCKET features."""

    frequencies: np.ndarray
    feature_indices: np.ndarray
    responses: np.ndarray
    power: np.ndarray
    impulse_responses: Tuple[np.ndarray, ...]
    metadata: Tuple[Dict[str, Any], ...]
    sampling_rate: float
    normalization: str
    include_representation: bool


def _validate_sampling_rate(sampling_rate: float) -> float:
    if isinstance(sampling_rate, (bool, np.bool_)) or not np.isscalar(
        sampling_rate
    ):
        raise TypeError("sampling_rate must be a positive scalar.")
    sampling_rate = float(sampling_rate)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be finite and positive.")
    return sampling_rate


def _validate_axis(axis: int, ndim: int) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis, (int, np.integer)
    ):
        raise TypeError("axis must be an integer.")
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise np.AxisError(axis, ndim=ndim)
    return axis


def _validate_n_fft(n_fft: Optional[int], minimum: int) -> Optional[int]:
    if n_fft is None:
        return None
    if isinstance(n_fft, (bool, np.bool_)) or not isinstance(
        n_fft, (int, np.integer)
    ):
        raise TypeError("n_fft must be an integer or None.")
    n_fft = int(n_fft)
    if n_fft < minimum:
        raise ValueError(
            f"n_fft must be at least {minimum} for the supplied signal."
        )
    return n_fft


def power_spectrum(
    data,
    *,
    sampling_rate: float = 1.0,
    axis: int = -1,
    method: str = "welch",
    n_fft: Optional[int] = None,
    nperseg: Optional[int] = None,
    window: Union[str, tuple, np.ndarray] = "hann",
    detrend: Union[str, bool] = "constant",
    scaling: str = "density",
) -> SpectrumResult:
    """Calculate a one-sided periodogram or Welch spectrum.

    Parameters
    ----------
    data : array-like
        One or more real-valued signals. The time dimension is selected with
        ``axis``.
    sampling_rate : float, default=1.0
        Samples per unit time. With the default, frequency is reported in
        cycles per timepoint.
    axis : int, default=-1
        Time axis.
    method : {'welch', 'periodogram'}, default='welch'
        Spectral estimator.
    n_fft : int or None, default=None
        FFT length. It must not be shorter than the input time dimension.
    nperseg : int or None, default=None
        Segment length for Welch estimation. Ignored for ``periodogram``.
    window : str, tuple, or array-like, default='hann'
        Window passed to SciPy.
    detrend : str or bool, default='constant'
        Detrending rule passed to SciPy.
    scaling : {'density', 'spectrum'}, default='density'
        SciPy spectral scaling.

    Returns
    -------
    SpectrumResult
        Frequencies and power values. All non-time dimensions are preserved.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        raise ValueError("data must contain a time dimension.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")
    axis = _validate_axis(axis, values.ndim)
    n_timepoints = int(values.shape[axis])
    if n_timepoints < 2:
        raise ValueError("At least two timepoints are required.")
    sampling_rate = _validate_sampling_rate(sampling_rate)
    if scaling not in {"density", "spectrum"}:
        raise ValueError("scaling must be 'density' or 'spectrum'.")
    if method not in {"welch", "periodogram"}:
        raise ValueError("method must be 'welch' or 'periodogram'.")

    if method == "welch":
        if nperseg is None:
            if isinstance(window, np.ndarray):
                if window.ndim != 1:
                    raise ValueError("An array-valued window must be one-dimensional.")
                nperseg = int(window.size)
            else:
                # SciPy defaults to 256 for named windows, but emits a warning
                # when the signal is shorter. Choose the same effective value
                # directly so short benchmark signals remain quiet.
                nperseg = min(256, n_timepoints)
        elif isinstance(nperseg, (bool, np.bool_)) or not isinstance(
            nperseg, (int, np.integer)
        ):
            raise TypeError("nperseg must be an integer or None.")
        else:
            nperseg = int(nperseg)
        if nperseg < 2 or nperseg > n_timepoints:
            raise ValueError(
                "nperseg must be between 2 and the signal length."
            )
        n_fft = _validate_n_fft(n_fft, nperseg)
        frequencies, power = signal.welch(
            values,
            fs=sampling_rate,
            window=window,
            nperseg=nperseg,
            nfft=n_fft,
            detrend=detrend,
            scaling=scaling,
            axis=axis,
            return_onesided=True,
        )
    else:
        n_fft = _validate_n_fft(n_fft, n_timepoints)
        frequencies, power = signal.periodogram(
            values,
            fs=sampling_rate,
            window=window,
            nfft=n_fft,
            detrend=detrend,
            scaling=scaling,
            axis=axis,
            return_onesided=True,
        )

    return SpectrumResult(
        frequencies=np.asarray(frequencies, dtype=float),
        power=np.asarray(power, dtype=float),
        method=method,
        sampling_rate=sampling_rate,
        scaling=scaling,
    )


def class_power_spectra(
    X,
    y,
    *,
    sampling_rate: float = 1.0,
    method: str = "welch",
    n_fft: Optional[int] = None,
    nperseg: Optional[int] = None,
    window: Union[str, tuple, np.ndarray] = "hann",
    detrend: Union[str, bool] = "constant",
    scaling: str = "density",
) -> ClassSpectrumResult:
    """Summarize trial-wise spectra separately for every class.

    ``X`` must have shape ``(n_samples, n_timepoints)``. Spectra are calculated
    for individual samples before class means and standard errors are formed.
    """
    X = check_array(
        X,
        dtype=float,
        ensure_2d=True,
        allow_nd=False,
        accept_sparse=False,
    )
    y = column_or_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("At least two classes are required.")

    spectra = power_spectrum(
        X,
        sampling_rate=sampling_rate,
        axis=1,
        method=method,
        n_fft=n_fft,
        nperseg=nperseg,
        window=window,
        detrend=detrend,
        scaling=scaling,
    )
    means = []
    sems = []
    counts = []
    for label in classes:
        class_power = spectra.power[y == label]
        count = int(class_power.shape[0])
        means.append(np.mean(class_power, axis=0))
        if count > 1:
            sems.append(np.std(class_power, axis=0, ddof=1) / np.sqrt(count))
        else:
            sems.append(np.zeros(class_power.shape[1], dtype=float))
        counts.append(count)

    return ClassSpectrumResult(
        frequencies=spectra.frequencies,
        classes=classes,
        mean_power=np.asarray(means, dtype=float),
        sem_power=np.asarray(sems, dtype=float),
        class_counts=np.asarray(counts, dtype=np.int64),
        method=spectra.method,
        sampling_rate=spectra.sampling_rate,
        scaling=spectra.scaling,
    )


def _resolve_transformer(model):
    if hasattr(model, "transformer_"):
        transformer = model.transformer_
    else:
        transformer = model
    if not hasattr(transformer, "decode_feature_index"):
        raise TypeError(
            "model must be a fitted I-ROCKET transformer or classifier."
        )
    check_is_fitted(transformer, attributes=["base_kernels_", "n_output_features_"])
    return transformer


def _validate_feature_indices(feature_indices, n_features: int) -> np.ndarray:
    if feature_indices is None:
        raise ValueError("feature_indices must be supplied.")
    values = np.asarray(feature_indices)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("feature_indices must be one-dimensional.")
    if values.size == 0:
        raise ValueError("feature_indices must not be empty.")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("feature_indices must contain integers.")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= n_features):
        raise ValueError("feature_indices contains an out-of-range index.")
    if np.unique(values).size != values.size:
        raise ValueError("feature_indices must not contain duplicates.")
    return values


def _effective_impulse_response(
    metadata: Dict[str, Any], *, include_representation: bool
) -> np.ndarray:
    weights = np.asarray(metadata["kernel_weights"], dtype=float)
    dilation = int(metadata["dilation"])
    response = np.zeros(1 + dilation * (weights.size - 1), dtype=float)
    response[::dilation] = weights
    if include_representation and metadata["representation"] == "diff":
        # I-ROCKET applies np.diff before the convolutional transform. The
        # corresponding first-difference response is [-1, +1]. Reversal of a
        # real finite kernel changes phase but not the magnitude response.
        response = np.convolve(np.asarray([-1.0, 1.0]), response)
    return response


def kernel_frequency_response(
    model,
    feature_indices,
    *,
    sampling_rate: float = 1.0,
    n_fft: Optional[int] = None,
    include_representation: bool = True,
    normalization: str = "peak",
) -> KernelSpectrumResult:
    """Calculate frequency responses for decoded I-ROCKET features.

    Pooling operator and bias threshold do not change a kernel's linear
    frequency response, so multiple transformed features may share the same
    response. Their metadata remain distinct in the returned result.

    Parameters
    ----------
    model : fitted I-ROCKET transformer or classifier
        Object exposing ``decode_feature_index`` directly or through a fitted
        ``transformer_`` attribute.
    feature_indices : int or sequence of int
        Full transformed-column indices.
    sampling_rate : float, default=1.0
        Samples per unit time.
    n_fft : int or None, default=None
        FFT length. The default is the next power of two at least as long as
        the longest effective impulse response, with a minimum of 256.
    include_representation : bool, default=True
        Include the first-difference filter for features from the ``diff``
        representation.
    normalization : {'peak', 'energy', 'none'}, default='peak'
        Response normalization applied independently to each kernel.

    Returns
    -------
    KernelSpectrumResult
        Complex responses, power responses, impulse responses, and decoded
        feature metadata.
    """
    if not isinstance(include_representation, (bool, np.bool_)):
        raise TypeError("include_representation must be a boolean.")
    if normalization not in {"peak", "energy", "none"}:
        raise ValueError("normalization must be 'peak', 'energy', or 'none'.")
    sampling_rate = _validate_sampling_rate(sampling_rate)
    transformer = _resolve_transformer(model)
    indices = _validate_feature_indices(
        feature_indices, int(transformer.n_output_features_)
    )
    metadata = tuple(
        dict(transformer.decode_feature_index(int(index))) for index in indices
    )
    impulses = tuple(
        _effective_impulse_response(
            item, include_representation=bool(include_representation)
        )
        for item in metadata
    )
    longest = max(response.size for response in impulses)
    if n_fft is None:
        n_fft = max(256, 1 << int(np.ceil(np.log2(longest))))
    else:
        n_fft = _validate_n_fft(n_fft, longest)

    responses = np.asarray(
        [np.fft.rfft(response, n=n_fft) for response in impulses],
        dtype=np.complex128,
    )
    power = np.abs(responses) ** 2
    if normalization == "peak":
        denominators = np.max(power, axis=1, keepdims=True)
        denominators[denominators == 0.0] = 1.0
        power = power / denominators
        responses = responses / np.sqrt(denominators)
    elif normalization == "energy":
        denominators = np.sum(power, axis=1, keepdims=True)
        denominators[denominators == 0.0] = 1.0
        power = power / denominators
        responses = responses / np.sqrt(denominators)

    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    return KernelSpectrumResult(
        frequencies=np.asarray(frequencies, dtype=float),
        feature_indices=indices.copy(),
        responses=responses,
        power=np.asarray(power, dtype=float),
        impulse_responses=tuple(response.copy() for response in impulses),
        metadata=metadata,
        sampling_rate=sampling_rate,
        normalization=normalization,
        include_representation=bool(include_representation),
    )


def selected_kernel_spectrum(
    model,
    *,
    feature_indices=None,
    weighting: str = "coefficient",
    sampling_rate: float = 1.0,
    n_fft: Optional[int] = None,
    include_representation: bool = True,
    normalization: str = "peak",
) -> Tuple[np.ndarray, np.ndarray, KernelSpectrumResult, np.ndarray]:
    """Aggregate frequency responses across selected I-ROCKET features.

    Parameters
    ----------
    model : fitted I-ROCKET classifier or transformer
        A fitted ``StableRocketClassifier`` is required for coefficient or
        consensus-probability weighting.
    feature_indices : sequence of int or None
        Defaults to ``model.selected_indices_``.
    weighting : {'coefficient', 'selection_probability', 'combined', 'uniform'}
        Nonnegative weights used to average individual power responses.

    Returns
    -------
    frequencies : ndarray
    aggregate_power : ndarray
    details : KernelSpectrumResult
    weights : ndarray
        Normalized feature weights used in the aggregation.
    """
    transformer = _resolve_transformer(model)
    if feature_indices is None:
        if not hasattr(model, "selected_indices_"):
            raise ValueError(
                "feature_indices are required when model has no "
                "selected_indices_ attribute."
            )
        feature_indices = model.selected_indices_
    details = kernel_frequency_response(
        transformer,
        feature_indices,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        include_representation=include_representation,
        normalization=normalization,
    )
    indices = details.feature_indices

    if weighting == "uniform":
        weights = np.ones(indices.size, dtype=float)
    elif weighting in {"coefficient", "combined"}:
        if not hasattr(model, "get_full_classifier_coefficients"):
            raise TypeError(
                "coefficient weighting requires a fitted "
                "StableRocketClassifier."
            )
        coefficients = np.asarray(model.get_full_classifier_coefficients())
        if coefficients.ndim == 1:
            weights = np.abs(coefficients[indices]).astype(float)
        else:
            weights = np.linalg.norm(coefficients[:, indices], axis=0)
    else:
        weights = np.ones(indices.size, dtype=float)

    if weighting in {"selection_probability", "combined"}:
        if not hasattr(model, "selector_") or not hasattr(
            model.selector_, "selection_probabilities_"
        ):
            raise TypeError(
                "selection-probability weighting requires a fitted selector."
            )
        probabilities = np.asarray(
            model.selector_.selection_probabilities_, dtype=float
        )[indices]
        if weighting == "selection_probability":
            weights = probabilities
        else:
            weights = weights * probabilities
    elif weighting not in {"coefficient", "uniform"}:
        raise ValueError(
            "weighting must be 'coefficient', 'selection_probability', "
            "'combined', or 'uniform'."
        )

    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Feature weights must be finite and nonnegative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.full(indices.size, 1.0 / indices.size)
    else:
        weights = weights / total
    aggregate = np.sum(details.power * weights[:, np.newaxis], axis=0)
    return details.frequencies, aggregate, details, weights


def plot_class_power_spectra(
    result: ClassSpectrumResult,
    *,
    class_names: Optional[Sequence[str]] = None,
    ax=None,
    show_sem: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    log_power: bool = False,
):
    """Plot class-conditional spectra returned by :func:`class_power_spectra`."""
    if not isinstance(result, ClassSpectrumResult):
        raise TypeError("result must be a ClassSpectrumResult.")
    if not isinstance(show_sem, (bool, np.bool_)):
        raise TypeError("show_sem must be a boolean.")
    if not isinstance(log_power, (bool, np.bool_)):
        raise TypeError("log_power must be a boolean.")
    if class_names is None:
        class_names = [str(label) for label in result.classes]
    if len(class_names) != result.classes.size:
        raise ValueError("class_names must contain one name per class.")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    epsilon = np.finfo(float).tiny
    for index, name in enumerate(class_names):
        mean = result.mean_power[index]
        sem = result.sem_power[index]
        if log_power:
            center = 10.0 * np.log10(np.maximum(mean, epsilon))
            lower = 10.0 * np.log10(np.maximum(mean - sem, epsilon))
            upper = 10.0 * np.log10(np.maximum(mean + sem, epsilon))
        else:
            center = mean
            lower = np.maximum(mean - sem, 0.0)
            upper = mean + sem
        line = ax.plot(result.frequencies, center, label=str(name))[0]
        if show_sem:
            ax.fill_between(
                result.frequencies,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
            )
    ax.set_xlabel(
        "Frequency" if result.sampling_rate != 1.0 else "Frequency (cycles/timepoint)"
    )
    ax.set_ylabel("Power (dB)" if log_power else "Power")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(frameon=False)
    ax.set_title("Class-conditional power spectra")
    return ax


def plot_selected_kernel_spectrum(
    model,
    *,
    feature_indices=None,
    weighting: str = "coefficient",
    sampling_rate: float = 1.0,
    n_fft: Optional[int] = None,
    include_representation: bool = True,
    normalization: str = "peak",
    ax=None,
    show_individual: bool = False,
):
    """Plot the aggregate spectral profile of selected I-ROCKET kernels."""
    if not isinstance(show_individual, (bool, np.bool_)):
        raise TypeError("show_individual must be a boolean.")
    frequencies, aggregate, details, _ = selected_kernel_spectrum(
        model,
        feature_indices=feature_indices,
        weighting=weighting,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        include_representation=include_representation,
        normalization=normalization,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if show_individual:
        for row in details.power:
            ax.plot(frequencies, row, alpha=0.08, linewidth=0.7)
    ax.plot(frequencies, aggregate, linewidth=2.0, label="Aggregate")
    ax.set_xlabel(
        "Frequency" if sampling_rate != 1.0 else "Frequency (cycles/timepoint)"
    )
    ax.set_ylabel("Normalized power" if normalization != "none" else "Power")
    ax.set_title(f"Selected-kernel spectrum ({weighting} weighting)")
    if show_individual:
        ax.legend(frameon=False)
    return ax


__all__ = [
    "SpectrumResult",
    "ClassSpectrumResult",
    "KernelSpectrumResult",
    "power_spectrum",
    "class_power_spectra",
    "kernel_frequency_response",
    "selected_kernel_spectrum",
    "plot_class_power_spectra",
    "plot_selected_kernel_spectrum",
]
