"""Installation-contract tests for the self-contained selection code."""

from pathlib import Path


def test_metadata_has_no_external_shrinkfs_dependency():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "shrinkfs @" not in pyproject
    assert "shrinkfs @" not in requirements
    assert 'packages = ["_irocket_selection"]' in pyproject


def test_internal_selection_package_imports():
    from _irocket_selection import (
        ResampledShrinkageSelector,
        ShrinkageFeatureSelector,
        nogueira_stability,
        screen_features,
        segmented_cutoff,
        shrinkage_t,
        shrinkage_t_ovr,
    )

    assert callable(shrinkage_t)
    assert callable(shrinkage_t_ovr)
    assert callable(screen_features)
    assert callable(segmented_cutoff)
    assert callable(nogueira_stability)
    assert ShrinkageFeatureSelector.__name__ == "ShrinkageFeatureSelector"
    assert ResampledShrinkageSelector.__name__ == "ResampledShrinkageSelector"


def test_nested_model_selection_module_imports():
    from irocket_model_selection import StableRocketClassifier, nested_stability_cv
    from model_selection import (
        StableRocketClassifier as CompatibleStableRocketClassifier,
        nested_stability_cv as compatible_nested_stability_cv,
    )

    assert StableRocketClassifier.__name__ == "StableRocketClassifier"
    assert callable(nested_stability_cv)
    assert CompatibleStableRocketClassifier is StableRocketClassifier
    assert compatible_nested_stability_cv is nested_stability_cv
