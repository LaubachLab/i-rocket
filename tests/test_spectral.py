import types
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from interp_rocket import InterpRocketTransform
from spectral import (
    ClassSpectrumResult,
    class_power_spectra,
    kernel_frequency_response,
    plot_class_power_spectra,
    plot_selected_kernel_spectrum,
    power_spectrum,
    selected_kernel_spectrum,
)


def test_power_spectrum_recovers_sine_frequency():
    n = 256
    frequency = 0.125
    time = np.arange(n)
    values = np.sin(2 * np.pi * frequency * time)
    result = power_spectrum(
        values,
        method="periodogram",
        window="boxcar",
        detrend=False,
        scaling="spectrum",
    )
    peak = result.frequencies[np.argmax(result.power[1:]) + 1]
    assert peak == pytest.approx(frequency)



def test_welch_short_signal_uses_quiet_effective_segment_length():
    values = np.arange(32, dtype=float)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = power_spectrum(values, method="welch")
    assert not record
    assert result.frequencies.size == 17


def test_welch_nfft_may_match_segment_length_for_longer_signal():
    values = np.arange(64, dtype=float)
    result = power_spectrum(
        values, method="welch", nperseg=32, n_fft=32
    )
    assert result.frequencies.size == 17

def test_class_power_spectra_shapes_and_plot():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(12, 64))
    y = np.repeat([0, 1], 6)
    result = class_power_spectra(X, y, nperseg=32)
    assert isinstance(result, ClassSpectrumResult)
    assert result.mean_power.shape == (2, result.frequencies.size)
    assert result.sem_power.shape == result.mean_power.shape
    assert np.array_equal(result.class_counts, [6, 6])
    ax = plot_class_power_spectra(result)
    assert ax.get_title() == "Class-conditional power spectra"
    plt.close(ax.figure)


def test_kernel_frequency_response_decodes_raw_and_diff_features():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(12, 40)).astype(np.float32)
    transformer = InterpRocketTransform(
        num_features=84,
        max_dilations_per_kernel=4,
        representations="both",
        random_state=2,
    ).fit(X)
    raw_index = 0
    diff_index = int(transformer.n_features_per_rep_[0]) * 4
    result = kernel_frequency_response(
        transformer,
        [raw_index, diff_index],
        n_fft=256,
        normalization="peak",
    )
    assert result.power.shape == (2, 129)
    assert np.allclose(np.max(result.power, axis=1), 1.0)
    assert result.metadata[0]["representation"] == "raw"
    assert result.metadata[1]["representation"] == "diff"
    assert result.impulse_responses[1].size == result.metadata[1]["receptive_field"] + 1


def test_selected_kernel_spectrum_supports_all_weighting_modes():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(12, 40)).astype(np.float32)
    transformer = InterpRocketTransform(
        num_features=84,
        max_dilations_per_kernel=4,
        representations="raw",
        random_state=3,
    ).fit(X)

    class FakeModel:
        pass

    model = FakeModel()
    model.transformer_ = transformer
    model.selected_indices_ = np.array([0, 1, 2], dtype=np.int64)
    model.selector_ = types.SimpleNamespace(
        selection_probabilities_=np.linspace(
            0.4, 1.0, transformer.n_output_features_
        )
    )

    def coefficients():
        values = np.zeros(transformer.n_output_features_)
        values[model.selected_indices_] = [1.0, 2.0, 3.0]
        return values

    model.get_full_classifier_coefficients = coefficients

    for weighting in (
        "uniform",
        "coefficient",
        "selection_probability",
        "combined",
    ):
        frequencies, aggregate, details, weights = selected_kernel_spectrum(
            model,
            weighting=weighting,
            n_fft=256,
        )
        assert aggregate.shape == frequencies.shape
        assert details.power.shape[0] == 3
        assert np.sum(weights) == pytest.approx(1.0)

    ax = plot_selected_kernel_spectrum(model, n_fft=256)
    assert "Selected-kernel spectrum" in ax.get_title()
    plt.close(ax.figure)


def test_spectral_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="finite"):
        power_spectrum([1.0, np.nan])
    with pytest.raises(ValueError, match="At least two classes"):
        class_power_spectra(np.ones((3, 10)), np.zeros(3))
