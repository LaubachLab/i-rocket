# I-ROCKET release 260724

Python package version: `0.7.0`  
Release date: July 24, 2026

This release establishes the supported I-ROCKET pipeline for univariate time-series classification and interpretation.

## Main changes

- The MultiRocket transform preserves its requested feature budget and follows the reference dilation, bias, padding, pooling, and first-difference behavior. An issue with padding for short time-series datasets was fixed in this revision of the package.
- `InterpRocketTransform` provides a classifier-agnostic scikit-learn transformer; `InterpRocket` remains a convenience classifier.
- Shrinkage-*t* scoring is included directly in I-ROCKET. The main installation does not depend on the `shrinkfs` from the Laubach Lab, which is available for use with any type of classifer, including tabular methods, and includes methods for filtering with CAT scores.
- The supported selector uses a one-break segmented cutoff within repeated subsamples, consensus selection probabilities, and the Nogueira stability measure.
- `nested_stability_cv` fits every transform, selector, scaler, and classifier within the correct training partition. The standard trial-level configuration is 10 outer folds and 3 inner folds when sample sizes permit.
- Group-aware outer folds, inner folds, and selector resampling support participant-, session-, and recording-level validation.
- Progress reporting is available for nested validation and large composite activation-trace calculations.
- Generic spectral helpers and optional TSHAP integration are available in this release.
- Standalone kernel and pooling explorers are distributed under `tools/`.

## Migration note

Analyses produced with earlier I-ROCKET releases should be rerun with 260724. The corrected transform can produce different feature counts and feature identities, particularly for short signals.

## Examples

The `examples/` directory contains Jupyter notebooks illustrating use of the package on three benchmark datasets: the synthetic waveform dataset, a participant-aware version of the GunPoint dataset, and the FordB dataset.

## Compatibility

- Python 3.9 or later
- univariate input arrays shaped `(n_samples, n_timepoints)`
- BSD 3-Clause license
- optional TSHAP integration remains a separate GPL-3.0 dependency; no TSHAP source is copied into I-ROCKET
