# Changelog

## 0.3.1 (2026-03-21)
- **plot_receptive_field_diagram**: Sorting of receptive fields switched from dilation to classifier importance.

## 0.3.0 (2026-03-21)
- **Permutation importance (PIMP)**: Added `permutation_importance_test()` and `plot_permutation_importance()` implementing the PIMP algorithm (Altmann et al., 2010) for statistically corrected feature importance with p-values. Uses RandomForestClassifier by default, which produces meaningful null distributions for ROCKET features (Ridge coefficients cause all features to appear significant).
- **Font size fixes**: Increased y-axis label font size to 8 in `plot_kernel_similarity` and `plot_multi_kernel_summary` for readability in notebooks and exported figures.
- **Subplot revision**: `plot_temporal_importance` was changed from having one subplot per class to using a single subplot for all classes.
- **PIMP demo notebook**: Added `demo_pimp.ipynb` demonstrating PIMP on the waveform-5000 dataset, including comparison with feature stability analysis.
- Remaining updates include incorporating code for the extensions into interp_rocket (planned for version 0.4.0) and the full set of demonstration and extension notebooks (planned for version 0.5.0).

## 0.2.0 (2026-03-18)
- **Feature stability selection**: Added `get_stable_features()` as a robust alternative to RFE for identifying important features across CV folds.
- **Class-mean visualization functions**: Added `plot_class_mean_activation()` (side-by-side activation and convolution output), `plot_multi_kernel_summary()` (heatmap across classes), and `plot_aggregate_activation()` (importance-weighted sum with differential).
- **Aggregate temporal occlusion**: Added `aggregate_temporal_occlusion()` for computing occlusion sensitivity over all trials grouped by class.
- **RF diagram font size**: Increased y-axis label font size from 5.5 to 8 for readability.
- **Public kneedle**: Renamed `_kneedle` to `kneedle` (public API).
- **CI fix**: Corrected `tests.yml` to use `pip install -e .` (removed nonexistent `[viz]` extra).
- **pyproject.toml**: Added `three_bumps` to py-modules.
- Demonstration notebooks for Receptive Field Mapping (`demo_RF_mapping.ipynb`) and the Breiman waveform dataset (`demo_waveform.ipynb`) and two benchmark scripts (`benchmark_waveform.py` and `benchmark_ucr.py`) were added to the `examples` directory.

## 0.1.0 (2026-03-16)
- Initial release
