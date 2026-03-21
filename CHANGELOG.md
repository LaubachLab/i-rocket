# Changelog

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
