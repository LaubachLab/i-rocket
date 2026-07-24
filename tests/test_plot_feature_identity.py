"""Regression tests for unambiguous feature identities in kernel plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import interpretability
from interp_rocket import (
    InterpRocket,
    InterpRocketTransform,
    _format_feature_label,
    compute_activation_map,
)


def _dataset():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(30, 24)).astype(np.float32)
    y = np.repeat([0, 1], 15)
    X[y == 1, 8:14] += 1.0
    return X, y


def _model():
    X, y = _dataset()
    model = InterpRocket(
        num_features=336,
        max_dilations_per_kernel=4,
        representations="raw",
        random_state=7,
        alpha_range=(0.1, 1.0, 10.0),
    ).fit(X, y)
    return X, y, model


def test_pooling_columns_share_the_same_prepool_activation():
    X, _, model = _model()
    decoded = [model.decode_feature_index(index) for index in range(4)]
    assert {item["pooling_op"] for item in decoded} == {
        "PPV", "MPV", "MIPV", "LSPV"
    }
    for item in decoded[1:]:
        assert item["kernel_index"] == decoded[0]["kernel_index"]
        assert item["dilation"] == decoded[0]["dilation"]
        assert item["bias"] == decoded[0]["bias"]
        assert item["padding_mode"] == decoded[0]["padding_mode"]
        assert item["representation"] == decoded[0]["representation"]

    activations = []
    for item in decoded:
        _, activation, time_index = compute_activation_map(
            X[0],
            item["kernel_index"],
            item["dilation"],
            item["bias"],
            item["padding_mode"],
            item["representation"],
        )
        activations.append((activation, time_index))

    for activation, time_index in activations[1:]:
        np.testing.assert_array_equal(activation, activations[0][0])
        np.testing.assert_array_equal(time_index, activations[0][1])


def test_same_kernel_and_pooling_with_different_biases_have_unique_labels():
    _, _, model = _model()
    first = model.decode_feature_index(0)
    second = model.decode_feature_index(4)

    assert first["kernel_index"] == second["kernel_index"]
    assert first["dilation"] == second["dilation"]
    assert first["pooling_op"] == second["pooling_op"] == "PPV"
    assert first["representation"] == second["representation"] == "raw"
    assert first["bias_rank_within_kernel"] != second["bias_rank_within_kernel"]
    assert first["bias"] != second["bias"]

    label_first = _format_feature_label(first, compact=True)
    label_second = _format_feature_label(second, compact=True)
    assert label_first != label_second
    assert "F0" in label_first and "b0" in label_first
    assert "F4" in label_second and "b1" in label_second


def test_plot_top_kernels_deduplicates_kernel_configurations_by_default():
    X, y, model = _model()
    feature_mask = np.asarray([0, 4], dtype=np.int64)

    unique = model.plot_top_kernels(
        X,
        y,
        n_kernels=2,
        n_examples=1,
        feature_mask=feature_mask,
    )
    # Two classes -> three axes per displayed row (kernel + two twin-axis
    # containers count as additional axes, so inspect the number of kernel bar
    # panels by their y-label instead of the raw figure axis count).
    kernel_panels = [axis for axis in unique.axes if axis.get_ylabel() == "Weight"]
    assert len(kernel_panels) == 1
    title = kernel_panels[0].get_title()
    assert "F" in title and "bias=" in title

    all_features = model.plot_top_kernels(
        X,
        y,
        n_kernels=2,
        n_examples=1,
        feature_mask=feature_mask,
        unique_kernels=False,
    )
    kernel_panels = [axis for axis in all_features.axes if axis.get_ylabel() == "Weight"]
    assert len(kernel_panels) == 2
    titles = [axis.get_title() for axis in kernel_panels]
    assert titles[0] != titles[1]

    plt.close(unique)
    plt.close(all_features)


def test_activation_and_kernel_pattern_labels_include_full_identity():
    X, y, model = _model()
    ordering = np.asarray([0, 4], dtype=np.int64)

    activation = interpretability.plot_activation_map(
        model,
        X,
        y,
        rank_order=ordering,
        n_show=2,
    )
    labels = [tick.get_text() for tick in activation.axes[0].get_yticklabels()]
    assert len(labels) == 2
    assert labels[0] != labels[1]
    assert "F0" in labels[0] and "b0" in labels[0] and "raw" in labels[0]
    assert "F4" in labels[1] and "b1" in labels[1] and "raw" in labels[1]

    pattern = interpretability.plot_kernel_pattern(
        model,
        rank_order=ordering,
        n_show=2,
    )
    labels = [tick.get_text() for tick in pattern.axes[-1].get_yticklabels()]
    assert labels[0] != labels[1]
    assert "F0" in labels[0] and "F4" in labels[1]

    plt.close(activation)
    plt.close(pattern)
