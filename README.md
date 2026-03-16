# I-ROCKET

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![tests](https://github.com/LaubachLab/i-rocket/actions/workflows/tests.yml/badge.svg)](https://github.com/LaubachLab/i-rocket/actions/workflows/tests.yml)

**Interpretable ROCKET: An Analysis Framework for Convolutional Time Series Classification**

**interp_rocket** is a standalone, single-file Python implementation of the MultiRocket algorithm (Tan et al., 2022) with a complete interpretability and analysis framework. Beyond classification, it provides kernel-level feature decoding, recursive feature elimination with principled knee detection, temporal activation and occlusion mapping, information-theoretic feature decomposition, cross-validation stability analysis, confusion-conditioned diagnostics, and receptive field visualization of the classifier's feature set. A companion module provides time series regression via the same interpretable transform. Inspired by the transparent parameter storage in ms_rocket (O'Toole, 2023), interp_rocket follows the same workflow (generate kernels, convolve the training data, extract pooled features, train a linear classifier) while exposing every intermediate parameter for inspection and analysis.

## Motivation

The ROCKET family of classifiers provides highly accurate time series classification with minimal computational cost. However, standard implementations in `sktime` and `aeon` wrap the fitted parameters and transformations inside compiled functions to maximize execution speed. While efficient, this creates an opaque model. For researchers analyzing temporal data, it is often critical to understand *why* a classification was made, and which specific temporal epochs, convolutional patterns, or summary statistics drive the separation between classes.

I-ROCKET was designed to answer these questions. It maintains algorithmic fidelity to MultiRocket but stores all kernel weights, dilations, biases, and pooling operators in documented NumPy arrays, enabling complete feature traceability from classifier decision back to temporal pattern.

## Architecture

interp_rocket is designed as a "glass box." Given any column in the classifier's coefficient vector, `decode_feature_index()` returns the exact base kernel (0–83), its dilation, bias threshold, pooling operator, and signal representation (raw or first-difference). These decoded kernels can then be reapplied to input trials to produce per-timepoint activation maps, closing the loop from classifier decision to temporal pattern.

## Algorithm

MultiRocket uses 84 deterministic convolutional kernels of length 9, inherited from MiniRocket (Dempster et al., 2021). Each kernel places a weight of +2 at 3 positions and −1 at the remaining 6, enumerating all C(9,3) possible placements. These kernels are applied at exponentially spaced dilations to both the raw signal and its first-order difference. Four pooling operators summarize each convolution output relative to a data-fitted bias threshold. The resulting feature vectors are used to train a ridge regression classifier with cross-validated regularization.

## Core Capabilities

### Classification and Feature Traceability

* **Complete Feature Decoding:** Map any feature to its generating kernel, dilation, bias, pooling operator, and signal representation.
* **Scikit-Learn Compatible:** Standard `fit()`, `transform()`, `predict()`, and `score()` interface.
* **Unbalanced Data Support:** Built-in minority class oversampling (`class_weight='balanced'`) and comprehensive scoring (macro/weighted F1, Matthews correlation coefficient, balanced accuracy, mutual information).
* **Multichannel Support:** Fit separate models per channel and concatenate features for multi-electrode, multi-sensor, or multi-region analysis.

### Regression

* **InterpRocketRegressor:** Parallel companion for continuous targets, using `RidgeCV` instead of `RidgeClassifierCV`. Same interpretable transform, same visualization tools adapted for regression (R², RMSE, Pearson r, scatter plots with target correlations).

### Interpretability and Visualization

* **Differential Activation Maps:** Per-class activation rates and max–min differences revealing where kernels discriminate, not just where they fire.
* **Temporal Importance Mapping:** Identify which time regions drive classification overall and where classes temporally diverge.
* **Class-Mean Kernel Activation:** Apply decoded kernels to class-averaged signals to visualize what each kernel detects without trial-to-trial noise.
* **Receptive Field Diagram:** Visualize the temporal footprint and scale of each feature, positioned at the location of peak discriminative activation, colored by pooling operator, overlaid on class means.
* **Temporal Occlusion Sensitivity:** Model-agnostic sliding-window perturbations measuring changes in the decision function.
* A demonstration notebook compares features from I-ROCKET with those from a classic wavelet-based method, discriminant pursuit (Buckheit and Donoho, 1995), as implemented in a Python package by the author of this package (https://github.com/LaubachLab/discriminant-pursuit; https://doi.org/10.5281/zenodo.18983376).

### Feature Selection and Diagnostics

* **Recursive Feature Elimination:** Isolate the minimal feature set for peak accuracy, with re-ranking at each step (Guyon et al., 2002; Uribarri et al., 2024). Two knee detection methods: threshold (within 1% of peak) and Kneedle algorithm (Satopaa et al., 2011).
* **Information Decomposition:** Classify each feature's contribution as redundant, synergistic, or independent using partial information analysis adapted from neural ensemble methods (Narayanan, Kimchi, & Laubach, 2005).
* **Kernel Similarity Network:** Correlation structure among features, revealing redundancy clusters and explaining why aggressive RFE preserves accuracy.
* **Confusion-Conditioned Activation Maps:** Separate temporal profiles for correct and misclassified trials, revealing where and why the model fails.
* **Cross-Validation Feature Stability:** Track which features are consistently important across CV folds versus fold-specific, assessing interpretability robustness.

### Evaluation

* **Repeated Stratified Cross-Validation:** Configurable repeats and folds, accumulating all metrics per fold with aggregated confusion matrices.
* **Single-File Portability:** Drop `interp_rocket.py` into your project directory. No package installation required.

## Installation

```bash
# Download the single file
wget https://raw.githubusercontent.com/LaubachLab/i-rocket/main/interp_rocket.py

# Or install as a package
pip install git+https://github.com/LaubachLab/i-rocket.git

# Or clone for development
git clone https://github.com/LaubachLab/i-rocket.git
pip install ./i-rocket
```

**Dependencies:** `numpy`, `numba` (>=0.50), `scikit-learn`, `matplotlib`

## Repository Structure

```text
i-rocket/
├── interp_rocket.py                        # Core classifier package (~4000 lines)
├── interp_rocket_regressor.py              # Regression companion module
├── three_bumps.py                          # Synthetic benchmark data generator
├── test_interp_rocket.py                   # Validation tests (19 tests)
├── pyproject.toml                          # Package metadata and build config
├── requirements.txt                        # Dependencies
├── LICENSE                                 # BSD-3-Clause
├── CITATION.cff                            # Citation metadata
├── CHANGELOG.md
├── .gitignore
├── README.md
└── examples/
    ├── demo_waveform.ipynb                 # ★ Start here: full tutorial with waveform-5000
    ├── demo_GunPoint.ipynb                 # Real-world motion sensor data (UCR GunPoint)
    ├── demo_FordB.ipynb                    # Real-world engine noise with frequency structure (UCR FordB)
    ├── demo_three_bumps.ipynb              # Synthetic data used for package development
    ├── demo_RF_mapping.ipynb               # Receptive field localization with single-bump data
    ├── demo_visualization.ipynb            # Comparison of temporal features from discriminant_pursuit and I-ROCKET
    ├── demo_multivariate.ipynb             # Extension to multichannel data
    ├── demo_regression.ipynb               # Time series regression (FloodModeling1 from aeon)
    ├── three_bumps.py                      # Three-bumps generator (copy for notebook use)
    ├── benchmark_waveform.py               # I-ROCKET vs aeon MultiRocket on waveform dataset
    ├── benchmark_ucr.py                    # I-ROCKET vs aeon MultiRocket across 15 UCR datasets
    └── benchmark_rfe_reproducibility.py    # RFE stability across random splits
```

## Quick Start

```python
from interp_rocket import InterpRocket

# Fit the model
model = InterpRocket(max_dilations_per_kernel=32, num_features=10000)
model.fit(X_train, y_train)

# Evaluate with multiple metrics
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"MCC: {metrics['mcc']:.4f}")

# Print a summary of the fitted model and its top features
model.summary()

# Decode a specific feature to its generating components
info = model.decode_feature_index(42)
print(f"Feature 42: kernel {info['kernel_index']}, dilation {info['dilation']}, "
      f"{info['pooling_op']} on {info['representation']} signal")

# Visualize which trial epochs drive classification
fig, importance = model.plot_temporal_importance(X_test, y_test, method='differential')

# Inspect the top kernels and their differential activation patterns
fig = model.plot_top_kernels(X_test, y_test, n_kernels=5)
```

## Regression

```python
from interp_rocket_regressor import InterpRocketRegressor

model = InterpRocketRegressor(max_dilations_per_kernel=32, num_features=10000)
model.fit(X_train, y_train)

metrics = model.evaluate(X_test, y_test)
print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

# All interpretability tools work the same way
fig, importance = model.plot_temporal_importance(X_test, y_test)
fig = model.plot_top_kernels(X_test, y_test, n_kernels=5)
```

See `examples/demo_regression.ipynb` for a complete tutorial using the FloodModeling1 dataset from aeon.

## Multichannel Data

For multi-channel recordings (e.g., multi-electrode neural data, multi-sensor motion capture), fit a separate model per channel and concatenate the features:

```python
from interp_rocket import InterpRocket
import numpy as np

# X has shape (n_samples, n_channels, n_timepoints)
models = []
for ch in range(n_channels):
    model = InterpRocket()
    model.fit(X_train[:, ch, :], y_train)
    models.append(model)

# Concatenate features across channels
train_features = np.hstack([m.transform(X_train[:, ch, :])
                            for ch, m in enumerate(models)])
test_features = np.hstack([m.transform(X_test[:, ch, :])
                           for ch, m in enumerate(models)])
```

See `examples/demo_multivariate.ipynb` for a complete tutorial including per-channel importance analysis and cross-validation.

## Recursive Feature Elimination

```python
from interp_rocket import recursive_feature_elimination, plot_elimination_curve, plot_rfe_survivors

# knee_method: 'threshold' (default, within 1% of peak), 'kneedle', or 'both'
rfe = recursive_feature_elimination(model, X_train, y_train, X_test, y_test,
                                     knee_method='both')
plot_elimination_curve(rfe)
plot_rfe_survivors(rfe, model, step=rfe['knee_idx'])

# Use RFE survivors to constrain all downstream analysis
survivors = rfe['surviving_indices'][rfe['knee_idx']]
model.plot_top_kernels(X_test, y_test, feature_mask=survivors)
model.plot_temporal_importance(X_test, y_test, feature_mask=survivors)
```

The Kneedle algorithm (Satopaa et al., 2011) finds the point of maximum curvature in the accuracy curve, providing a principled alternative to the 1% threshold rule. Use `knee_method='both'` to compare both methods on the same RFE run.

## Information Decomposition

Adapted from neural ensemble analysis (Narayanan, Kimchi, & Laubach, 2005), this decomposes each feature group's contribution into redundant, synergistic, and independent information:

```python
from interp_rocket import information_decomposition, plot_information_decomposition

info = information_decomposition(
    model, X_test, y_test,
    feature_mask=survivors,
    group_by='kernel',      # group features by base kernel
    n_shuffles=100,
)
fig = plot_information_decomposition(info)
```

Features classified as **redundant** carry information already available from other features. **Synergistic** features only contribute when combined with others. **Independent** features carry unique information. This explains the structure of RFE: survivors should be predominantly independent and possibly synergistic.

## Diagnostics

```python
from interp_rocket import (
    plot_kernel_similarity,
    plot_confusion_conditioned_maps,
    cv_feature_stability, plot_feature_stability,
    plot_receptive_field_diagram,
    temporal_occlusion, plot_occlusion,
)

# Kernel similarity: which features fire together?
fig, corr = plot_kernel_similarity(model, X_test, feature_mask=survivors)

# Confusion-conditioned maps: where does the model fail?
fig = plot_confusion_conditioned_maps(model, X_test, y_test, feature_mask=survivors)

# Feature stability: which features are robust across CV folds?
stability = cv_feature_stability(X, y, n_repeats=5, n_folds=5, n_top=50)
fig = plot_feature_stability(stability, model=model)

# Receptive field diagram: what temporal scales does the feature set cover?
fig = plot_receptive_field_diagram(model, X_test, y_test, feature_mask=survivors)

# Temporal occlusion: model-agnostic perturbation analysis
occ = temporal_occlusion(model, X_test, y_test, n_samples=6, feature_mask=survivors)
plot_occlusion(occ)
```

## Working with Unbalanced Data

```python
from interp_rocket import InterpRocket, cross_validate

model = InterpRocket(max_dilations_per_kernel=32, class_weight='balanced')
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)

results = cross_validate(X, y, n_repeats=10, n_folds=10, n_jobs=-2,
                         class_weight='balanced')
print(f"Balanced accuracy: {results['balanced_accuracy']['mean']:.4f} "
      f"+/- {results['balanced_accuracy']['std']:.4f}")
```

## Plotting and Visualization Tools

All plotting methods accept an optional `feature_mask` parameter to restrict analysis to RFE survivors or any custom subset.

| Method | What it shows |
|--------|---------------|
| `plot_top_kernels()` | Weight patterns, per-class activation rates, and differential activation |
| `plot_temporal_importance()` | Which time regions drive classification (per-class and overall) |
| `plot_feature_distributions()` | Class-conditional histograms of top feature values |
| `plot_kernel_properties()` | Dilation, receptive field, and pooling operator distributions |
| `plot_elimination_curve()` | RFE accuracy curve with threshold and/or Kneedle knee |
| `plot_rfe_survivors()` | Properties of features that survive RFE |
| `plot_occlusion()` | Temporal occlusion sensitivity per sample |
| `plot_information_decomposition()` | Redundant/synergistic/independent classification |
| `plot_kernel_similarity()` | Correlation matrix and within/between-kernel redundancy |
| `plot_confusion_conditioned_maps()` | Temporal profiles for correct vs. misclassified trials |
| `plot_feature_stability()` | Feature presence across CV folds |
| `plot_receptive_field_diagram()` | Temporal footprint of each feature by pooling operator |

## API Reference

### Classes

**`InterpRocket(max_dilations_per_kernel=32, num_features=10000, random_state=0, alpha_range=None, class_weight=None)`**

**`InterpRocketRegressor(max_dilations_per_kernel=32, num_features=10000, random_state=0, alpha_range=None)`** *(in interp_rocket_regressor.py)*

### Methods (both classes)

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Fit the transform and classifier/regressor. |
| `predict(X)` | Return predictions (class labels or continuous values). |
| `score(X, y)` | Return accuracy (classifier) or R² (regressor). |
| `evaluate(X, y)` | Return dict of metrics (6 for classifier, 6 for regressor). |
| `transform(X)` | Return raw feature matrix. |
| `decode_feature_index(i)` | Decode feature to kernel, dilation, bias, pooling op, representation. |
| `get_feature_importance(feature_mask=None)` | Per-feature importance (normalized, max=1.0). |
| `get_top_features(n=None, feature_mask=None)` | Top features, fully decoded. |
| `summary()` | Print model summary with top decoded features. |

### Module-Level Functions

| Function | Description |
|--------|-------------|
| `recursive_feature_elimination(...)` | Iterative feature elimination with dual knee detection. |
| `plot_elimination_curve(rfe_results)` | RFE curve with knee annotations. |
| `plot_rfe_survivors(rfe_results, model)` | Six-panel surviving feature analysis. |
| `cross_validate(X, y, ...)` | Repeated stratified k-fold CV with all metrics. |
| `temporal_occlusion(model, X_test, y_test)` | Sliding-window perturbation analysis. |
| `plot_occlusion(occ_results)` | Temporal occlusion sensitivity plot. |
| `information_decomposition(model, X_test, y_test)` | Partial information decomposition. |
| `plot_information_decomposition(info_results)` | Redundant/synergistic/independent scatter and summary. |
| `plot_kernel_similarity(model, X_test)` | Feature correlation matrix and distribution. |
| `plot_confusion_conditioned_maps(model, X_test, y_test)` | Activation maps split by classification outcome. |
| `cv_feature_stability(X, y)` | Feature importance consistency across CV folds. |
| `plot_feature_stability(stability_results)` | Stability heatmap and bar chart. |
| `plot_receptive_field_diagram(model)` | Temporal footprint diagram. |
| `compute_activation_map(x, kernel_index, dilation, bias)` | Single-trial kernel activation (numba-compiled). |
| `mutual_information(y_true, y_pred)` | MI between true and predicted labels. |
| `kneedle(y)` | Kneedle knee-point detection (Satopaa et al., 2011). |

## References

- Buckheit, J. & Donoho, D.L. (1995). Improved linear discrimination using time-frequency dictionaries. Proc. SPIE, 2569, 540–551.
- Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34(5), 1454–1495.
- Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost) deterministic transform for time series classification. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 248–257.
- Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1), 389–422.
- Lundy, C., & O'Toole, J. M. (2021). Random convolution kernels with multi-scale decomposition for preterm EEG inter-burst detection. *2021 29th European Signal Processing Conference (EUSIPCO)*, 1182–1186.
- Narayanan, N. S., Kimchi, E. Y., & Laubach, M. (2005). Redundancy and synergy of neuronal ensembles in motor cortex. *Journal of Neuroscience*, 25(17), 4207–4216.
- O'Toole, J. M. (2023). ms_rocket: Multi-scale ROCKET for time series classification. GitHub repository. https://github.com/otoolej/ms_rocket
- Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior. *ICDCS Workshops*.
- Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification. *Data Mining and Knowledge Discovery*, 36(5), 1623–1646.
- Uribarri, G., Barone, F., Ansuini, A., & Fransén, E. (2024). Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels. *Data Mining and Knowledge Discovery*, 38, 1–26.

## Author

Mark Laubach (American University, Department of Neuroscience). Developed with Claude (Anthropic) as AI coding assistant.

## License

BSD-3-Clause
