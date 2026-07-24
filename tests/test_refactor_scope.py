"""Contract tests for the final reduced I-ROCKET scope and plots."""

import inspect
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import feature_selection
import interp_rocket
import interpretability
import stability_nogueira
from _irocket_selection import ResampledShrinkageSelector
from interp_rocket import InterpRocketTransform
from irocket_model_selection import StableRocketClassifier


ROOT = Path(__file__).resolve().parents[1]


def _small_final_model():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(36, 24)).astype(np.float32)
    y = np.repeat([0, 1], 18)
    X[y == 1, 8:13] += 1.25
    model = StableRocketClassifier(
        transformer=InterpRocketTransform(
            num_features=84,
            max_dilations_per_kernel=4,
            representations="raw",
            random_state=7,
        ),
        selector=ResampledShrinkageSelector(
            n_resamples=4,
            sample_fraction=0.6,
            consensus_threshold=0.5,
            cutoff_min_size=2,
            min_features=2,
            random_state=11,
        ),
        consensus_threshold=0.5,
        alpha=1.0,
        random_state=13,
    ).fit(X, y)
    return X, y, model


def test_explicitly_removed_plots_are_absent():
    assert not hasattr(interp_rocket.InterpRocket, "plot_temporal_importance")
    removed = {
        "plot_feature_selection_comparison",
        "plot_confusion_conditioned_maps",
        "plot_aggregate_activation",
        "plot_multi_kernel_summary",
        "plot_receptive_field_diagram",
        "plot_stability_curve",
    }
    assert removed.isdisjoint(vars(interpretability))


def test_legacy_selection_pathways_are_not_public():
    removed = {
        "cv_feature_stability",
        "get_stable_features",
        "stability_curve",
        "cross_validate",
        "cat_score",
        "cat_score_ovr",
        "kneedle",
        "select_by_kneedle",
    }
    assert removed.isdisjoint(vars(feature_selection))
    assert removed.isdisjoint(vars(stability_nogueira))


def test_current_runtime_modules_have_no_removed_method_language():
    paths = [
        ROOT / "interp_rocket.py",
        ROOT / "feature_selection.py",
        ROOT / "interpretability.py",
        ROOT / "irocket_model_selection.py",
        ROOT / "stability_nogueira.py",
        ROOT / "_irocket_selection" / "__init__.py",
        ROOT / "_irocket_selection" / "shrinkage.py",
        ROOT / "_irocket_selection" / "stability.py",
        ROOT / "_irocket_selection" / "thresholds.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    removed_terms = (
        "R" + "FE",
        "PM" + "IFS",
        "PI" + "MP",
        "TAB" + "10",
        "cat_" + "score",
        "K" + "needle",
    )
    assert all(term not in text for term in removed_terms)


def test_accessible_palette_contract():
    assert interp_rocket.OI == [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ]
    assert interp_rocket.POOLING_COLORS == {
        "PPV": "#0072B2",
        "MPV": "#56B4E9",
        "MIPV": "#D55E00",
        "LSPV": "#E69F00",
    }


def test_requested_function_signatures_and_defaults():
    info_signature = inspect.signature(feature_selection.information_decomposition)
    assert info_signature.parameters["group_by"].default == "individual"
    similarity_signature = inspect.signature(interpretability.plot_kernel_similarity)
    assert "method" not in similarity_signature.parameters


def test_dataset_access_is_an_optional_extra():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r"\[project\.optional-dependencies\]\s*"
        r"datasets\s*=\s*\[(.*?)\]",
        metadata,
        flags=re.DOTALL,
    )
    assert match is not None
    dataset_extra = match.group(1).lower()
    assert '"aeon' in dataset_extra
    assert '"openml' in dataset_extra


def test_final_model_drives_consensus_selected_kernel_plots():
    X, y, model = _small_final_model()
    selected = set(model.selected_indices_.tolist())
    top = model.get_top_features(n=min(3, len(selected)))
    assert top
    assert {item["feature_index"] for item in top} <= selected
    assert all("selection_probability" in item for item in top)

    properties = model.plot_kernel_properties()
    assert len(properties.axes) == 6
    assert "Consensus-selected" in properties._suptitle.get_text()
    threshold_lines = properties.axes[-1].lines
    assert threshold_lines
    np.testing.assert_allclose(
        threshold_lines[0].get_xdata(),
        [model.consensus_threshold, model.consensus_threshold],
    )

    full_coefficients = model.get_full_classifier_coefficients()
    assert full_coefficients.shape[-1] == model.n_transform_features_
    outside = np.ones(model.n_transform_features_, dtype=bool)
    outside[model.selected_indices_] = False
    assert np.all(full_coefficients[..., outside] == 0)

    kernels = model.plot_top_kernels(X, y, n_kernels=1, n_examples=1)
    assert len(kernels.axes) >= 3

    stability = interpretability.plot_feature_stability(model, n_show=5)
    assert len(stability.axes) == 2
    stability_image = stability.axes[0].images[0]
    assert stability_image.norm.vmin == 0
    assert stability_image.norm.vmax == 1

    class_mean = interpretability.plot_class_mean_activation(
        model,
        X,
        y,
        rank_order=model.selected_indices_,
        feature_rank=0,
    )
    assert len(class_mean.axes) >= 4

    activation_map = interpretability.plot_activation_map(
        model, X, y, rank_order=model.selected_indices_, n_show=1
    )
    kernel_pattern = interpretability.plot_kernel_pattern(
        model, rank_order=model.selected_indices_, n_show=1
    )
    similarity_features = model.selected_indices_[:2]
    similarity, correlation = interpretability.plot_kernel_similarity(
        model, X, feature_mask=similarity_features
    )
    assert correlation.shape == (2, 2)

    profiled_indices = model.selected_indices_[: min(4, model.n_selected_features_)]
    profile = interpretability.localization_profile(
        model,
        X,
        y,
        profiled_indices,
    )
    assert np.array_equal(profile["feature_index"], profiled_indices)
    assert profile["mass_in_rf"].shape == profiled_indices.shape
    localization = interpretability.plot_localization_diagnostic(profile)
    assert len(localization.axes) == 2

    for figure in (
        properties,
        kernels,
        stability,
        class_mean,
        activation_map,
        kernel_pattern,
        similarity,
        localization,
    ):
        plt.close(figure)


def test_composite_trace_uses_retained_full_feature_mapping():
    from interpretability import composite_activation_trace

    X, _, model = _small_final_model()
    selected = model.selected_indices_[: min(4, model.selected_indices_.size)]
    trace = composite_activation_trace(X[:3], model, selected_features=selected)
    assert trace.shape == X[:3].shape
    assert np.isfinite(trace).all()


def test_patch05_model_selection_import_path_remains_available():
    from model_selection import StableRocketClassifier as CompatibilityClassifier
    from model_selection import nested_stability_cv as compatibility_nested_cv

    assert CompatibilityClassifier is StableRocketClassifier
    assert callable(compatibility_nested_cv)
