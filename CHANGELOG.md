# Changelog

## 260724 — package version 0.7.0

- Updated the univariate MultiRocket transform to preserve its requested
  feature budget and align dilation allocation, bias estimation, padding,
  pooling, first-difference handling, and feature decoding with the reference
  algorithm.
- Added a classifier-agnostic transformer, internal shrinkage-*t* filtering,
  segmented cutoffs, resampled consensus selection, Nogueira stability, and
  leakage-free nested cross-validation.
- Added group-aware validation, progress reporting, focused interpretation and
  spectral tools, optional TSHAP integration, and the standalone kernel and
  pooling explorers.
- Removed obsolete experimental pathways; expanded numerical, estimator,
  leakage, plotting, packaging, and installation tests.
- Analyses produced with earlier releases should be rerun because transformed
  feature counts and identities may differ, particularly for short signals.

## Earlier releases

Version 0.6.1 and earlier document the exploratory development history in the
repository and archived releases.
