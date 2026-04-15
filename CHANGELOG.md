# Changelog

## v0.6 (2026-04-15)

**LFP extensions.** A new `lfp_tools` module in `extensions/` provides functions for spectral analysis of event-aligned local field potential recordings. The module wraps tensorpac (Combrisson et al. 2020) and a Python port of EEGLAB's eegfilt.m, providing power spectral density, event-related spectral perturbation with z-score normalization, pairwise phase consistency (Vinck et al. 2010), and phase-amplitude coupling analysis. A companion demo notebook applies the full I-ROCKET workflow to LFP recordings from Amarante, Caetano, & Laubach (2017), demonstrating cross-validation, feature stability analysis, top kernel properties, temporal importance, receptive field mapping, single-kernel activation extraction, and validation of kernel selectivity through PSD, ERSP/PPC, and PAC by trial type.

**interp_rocket.** `plot_top_kernels` now defaults to `show_difference=False` and accepts a `colors` keyword argument for user-specified class colors.

**Revised demos.** `demo_waveform` was rewritten with `FIG_WIDTH=8` throughout, temporal occlusion and pre-FSA cells removed, and a "two complementary views" framing (temporal importance and receptive field mapping). `demo_gunpoint`, `demo_fordb`, and `demo_three_bumps` were streamlined to reduce complexity based on user feedback.

**Documentation.** Clarified that `max_dilations_per_kernel` is a ceiling, not a target. The actual dilation count is limited by signal length via `floor((L-1)/(k-1))`. The v0.5 reduction from 32 to 16 only affects signals long enough to support more than 16 valid dilations. Temporal occlusion is retained in the package but removed from the waveform and GunPoint demos.

## v0.5 (2026-03-27)

- All demonstration notebooks in `examples/` and `extensions/` have been finalized and posted.
- The default for `max_dilations_per_kernel` was changed from 32 to 16 in `InterpRocket`. Extensive testing across the benchmark datasets showed that 16 dilations produces sharper temporal importance profiles and more interpretable receptive field maps, with minimal effect on classification accuracy. The previous default of 32 often produces stable features with receptive fields spanning the entire series, which dilutes temporal localization. Users can set any value via the `max_dilations_per_kernel` keyword argument.
- The headers of `interp_rocket.py` and `interp_rocket_regressor.py` were revised to reflect all changes from v0.3 through v0.5, including updated feature selection recommendations, interpretability tool listings, and usage examples.
- `interp_rocket_regressor` was updated with all recent changes to `interp_rocket`. The regression module is provided as a demonstration of the framework's capabilities. It has received less testing than the classifier module, and the simple `RidgeCV` model may be insufficient for scientific applications where non-linear target relationships are expected.
- With this release, I-ROCKET is considered ready for practical use in research and teaching. The core classifier, feature selection methods, and visualization tools have been tested across multiple benchmark datasets and are in active use in the Laubach Lab. Bug reports and feature requests can be submitted via GitHub Issues.

## v0.4 (2026-03-25)

### Extensions (new directory: `extensions/`)

- **AMEE evaluation** (`amee_evaluation.py`): Perturbation-based evaluation framework for ranking saliency maps, implementing the approach of Nguyen et al. (2024). Includes four perturbation strategies (zero, mean, noise, inverse), random and inverse baselines, and visualization of accuracy drop curves and rankings.
- **TSHAP integration** (`tshap_integration.py`): Bridge module connecting I-ROCKET with the TSHAP package (Le Nguyen and Ifrim, 2025) for instance-level Shapley value attributions. Wraps I-ROCKET's prediction pipeline for TSHAP compatibility and provides comparison plots.
- **Channel selection** (`channel_selection.py`): Classifier-agnostic channel selection for multivariate time series based on class prototype distances, following Dhariyal et al. (2023). Includes ECP (elbow class pairwise) and top-k methods with visualization.
- **Kernel explorer** (`kernel_explorer.py`): Standalone interactive tool for exploring the 84 base kernels, dilation, bias thresholding, and pooling operators. Launches in its own matplotlib window.

### New demo notebooks

- `demo_RF_mapping.ipynb` in `examples/`: Receptive field mapping and temporal importance on the single-bump dataset, demonstrating localization of a known discriminative feature.
- `demo_tshap_amee_waveform.ipynb` in `extensions/`: AMEE and TSHAP evaluation on the waveform dataset.
- `demo_tshap_amee_bump.ipynb` in `extensions/`: AMEE and TSHAP evaluation on a synthetic single-bump dataset.
- `demo_channel_selection.ipynb` in `extensions/`: Channel selection workflow with synthetic multivariate data.

### Core changes

- **`temporal_occlusion`**: Added `sample_indices` parameter to allow analysis of specific trials (e.g., misclassified instances) instead of automatic sample selection.
- **`plot_permutation_importance`**: Fixed y-axis padding with explicit `set_ylim` to remove bar chart offset.
- **`plot_multi_kernel_summary`**: Updated colormap to match feature stability plot (`['#f0f0f0', '#1f77b4']`).

### Regression module (`interp_rocket_regressor.py`)

- Added `plot_receptive_field_diagram` method for visualizing feature receptive fields.
- Added `plot_feature_stability` compatibility for cross-validation stability results.
- Added `aggregate_temporal_occlusion` for class-level occlusion sensitivity analysis.

### Pending updates for version 0.5

- All final demonstration notebooks will be posted.
- Any issues found by testers of the functions in the Laubach Lab and elsewhere will be addressed.
- The package is currently judged as production ready and is already being used in research projects and courses.
- Release 0.5 will finalize all documentation to date on the project.

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
