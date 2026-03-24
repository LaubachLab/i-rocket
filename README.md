# I-ROCKET

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![tests](https://github.com/LaubachLab/i-rocket/actions/workflows/tests.yml/badge.svg)](https://github.com/LaubachLab/i-rocket/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19055359.svg)](https://doi.org/10.5281/zenodo.19055359)

**Interpretable ROCKET: An Analysis Framework for Convolutional Time Series Classification**

I-ROCKET is a standalone Python package. It implements the MultiRocket algorithm (Tan et al., 2022). The package provides a complete interpretability and analysis framework. It includes kernel-level feature decoding, feature selection, temporal activation mapping, and temporal occlusion mapping. It also provides information-theoretic decomposition, cross-validation stability analysis, confusion-conditioned diagnostics, and receptive field visualization. A companion module handles time series regression. It uses the exact same interpretable transform. The transparent parameter storage in ms_rocket (O'Toole, 2023) inspired this design. I-ROCKET generates kernels, convolves the training data, extracts pooled features, and trains a linear classifier. It exposes every intermediate parameter for inspection.

## Motivation

The ROCKET family of classifiers provides accurate time series classification with low computational cost. Standard implementations in `sktime` and `aeon` wrap parameters inside compiled functions. This maximizes speed but creates an opaque model. Researchers often need to understand why a model makes a specific classification. They must identify the exact time regions, convolutional patterns, or summary statistics that separate classes.

I-ROCKET answers these questions. It follows the MultiRocket algorithm strictly. It stores all kernel weights, dilations, biases, and pooling operators in standard NumPy arrays. This ensures complete feature traceability. Users can map a classifier decision directly back to a specific temporal pattern in the raw data.

## Performance

I-ROCKET matches the speed of aeon's MultiRocket. Both packages use numba-compiled convolution loops. They share identical computational complexity. The interpretability tools operate post-hoc on the stored parameters. They do not increase training or inference time. Users get the speed of aeon alongside full interpretability.

## Architecture

I-ROCKET operates as a glass box. Users select any column in the coefficient vector. The `decode_feature_index()` function returns the exact base kernel, dilation, bias threshold, pooling operator, and signal representation. Users then apply these decoded kernels to input trials. This produces per-timepoint activation maps. The maps connect the classifier decision to the original temporal pattern.

## Algorithm

MultiRocket uses 84 deterministic convolutional kernels of length 9. These originate from MiniRocket (Dempster et al., 2021). Each kernel places a weight of +2 at 3 positions and -1 at the remaining 6. The algorithm applies these kernels at exponentially spaced dilations to the raw signal and its first-order difference. Four pooling operators summarize each convolution output relative to a data-fitted bias threshold. The algorithm uses the resulting feature vectors to train a ridge regression classifier.

## Core Capabilities

### Classification and Feature Traceability

* **Complete Feature Decoding:** Map any feature to its kernel parameters and signal representation. No other implementation offers this capability.
* **Scikit-Learn Compatible:** Utilize the standard `fit()`, `transform()`, `predict()`, and `score()` interface.
* **Unbalanced Data Support:** Activate built-in minority class oversampling. Access comprehensive scoring metrics.
* **Multichannel Support:** Fit separate models per channel. Concatenate features for multi-sensor analysis.

### Regression

* **InterpRocketRegressor:** Analyze continuous targets. This module uses `RidgeCV`. It shares the same interpretable transform and adapts all visualization tools for regression tasks.

### Interpretability and Visualization

* **Differential Activation Maps:** Compute per-class activation rates and max-min differences. These maps reveal where kernels discriminate.
* **Temporal Importance Mapping:** Identify the specific time regions that drive classification.
* **Temporal Occlusion Sensitivity:** Measure changes in the decision function through model-agnostic sliding-window perturbations.
* **Class-Mean Kernel Activation:** Apply decoded kernels to class-averaged signals. This visualizes kernel detection without trial-to-trial noise.
* **Receptive Field Diagram:** Visualize the time span and scale of each feature. The diagram positions features at the location of peak discriminative activation.

### Feature Selection and Diagnostics

* **Feature Stability Analysis (FSA):** Identify features that remain important across cross-validation folds.
* **Permutation Importance (PIMP):** Calculate statistically corrected feature importance with p-values.
* **Recursive Feature Elimination (RFE):** Isolate a minimal feature set for peak accuracy.
* **Information Decomposition:** Classify feature contributions as redundant, synergistic, or independent.
* **Kernel Similarity Network:** Map the correlation structure among features. This reveals redundancy clusters.
* **Confusion-Conditioned Activation Maps:** Separate temporal profiles for correct and misclassified trials.

### Evaluation

* **Repeated Stratified Cross-Validation:** Configure repeats and folds. Accumulate metrics per fold.
* **Single-File Portability:** Drop `interp_rocket.py` into any project directory. No installation is necessary.

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
|-- interp_rocket.py                        # Core classifier package (~4800 lines)
|-- interp_rocket_regressor.py              # Regression companion module
|-- three_bumps.py                          # Synthetic benchmark data generator
|-- test_interp_rocket.py                   # Validation tests (19 tests)
|-- pyproject.toml                          # Package metadata and build config
|-- requirements.txt                        # Dependencies
|-- LICENSE                                 # BSD-3-Clause
|-- CITATION.cff                            # Citation metadata
|-- CHANGELOG.md
|-- README.md
|-- examples/
|   |-- demo_waveform.ipynb                 # Full tutorial with waveform-5000
|   |-- demo_GunPoint.ipynb                 # Real-world motion sensor data (UCR GunPoint)
|   |-- demo_FordB.ipynb                    # Real-world engine noise (UCR FordB)
|   |-- demo_three_bumps.ipynb              # Synthetic data used for package development
|   |-- demo_RF_mapping.ipynb               # Receptive field localization with single-bump data
|   |-- demo_visualization.ipynb            # Comparison of temporal features from DP and I-ROCKET
|   |-- demo_multivariate.ipynb             # Extension to multichannel data
|   |-- demo_pimp.ipynb                     # Permutation importance (PIMP) on waveform
|   |-- demo_amee.ipynb                     # AMEE evaluation of saliency maps
|   |-- demo_tshap_amee.ipynb               # TSHAP + AMEE combined evaluation
|   |-- demo_channel_selection.ipynb        # Channel selection for multivariate data
|   |-- benchmark_waveform.py               # I-ROCKET vs aeon MultiRocket on waveform
|   +-- benchmark_ucr.py                    # I-ROCKET vs aeon MultiRocket across 15 UCR datasets
+-- extensions/
    |-- kernel_explorer.ipynb               # Interactive kernel/dilation/pooling explorer
    |-- amee_evaluation.py                  # AMEE explanation evaluation framework
    |-- tshap_integration.py                # TSHAP bridge for Shapley value attributions
    +-- channel_selection.py                # Channel selection for multivariate time series
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

## Alternative Classifiers

I-ROCKET uses a Ridge classifier by default because the interpretability tools depend on its linear coefficient vector. However, the ROCKET feature matrix and stability-selected features can be passed to any scikit-learn compatible classifier. This is useful when non-linear interactions across features are expected, for example in multichannel neural recordings where discriminative information may involve coordinated patterns across electrodes.

```python
import interp_rocket as IR
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Fit I-ROCKET and run stability analysis as usual
model = IR.InterpRocket(max_dilations_per_kernel=32, num_features=10000)
model.fit(X, y)

stability = IR.cv_feature_stability(X, y, n_repeats=5, n_folds=5, n_top=50)
stable_features = IR.get_stable_features(stability, threshold=0.8)

# Extract the feature matrix and restrict to stable features
train_features = model.transform(X_train)[:, stable_features]
test_features = model.transform(X_test)[:, stable_features]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# Train any sklearn classifier on the stable features
clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=42)
clf.fit(train_scaled, y_train)
y_pred = clf.predict(test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
```

The workflow separates feature extraction (ROCKET transform), feature selection (FSA), and classification into independent stages. The first two stages use I-ROCKET's full pipeline. The third stage can use any classifier without affecting the interpretability results, which are derived from the Ridge model. This separation is useful for multichannel data, where features from different channels may interact in ways that a linear classifier cannot capture.

## Feature Selection

I-ROCKET implements three feature selection methods.

### Feature Stability Analysis (recommended)

Stability-based feature selection identifies features that are consistently important across CV folds. It is insensitive to random seed and train/test split, and provides a robust alternative to recursive feature elimination ([Meinshausen & Bühlmann, 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x); [Saeys et al., 2008](https://doi.org/10.1007/978-3-540-87481-2_21)).

```python
from interp_rocket import (
    InterpRocket, cv_feature_stability, plot_feature_stability,
    get_stable_features,
)

# Fit model on all data for stability analysis
model = InterpRocket(max_dilations_per_kernel=32, num_features=10000)
model.fit(X, y)

# Run cross-validation feature stability
stability = cv_feature_stability(X, y, n_repeats=5, n_folds=5, n_top=50)
fig = plot_feature_stability(stability, model=model)

# Extract features present in >=80% of folds
stable_features = get_stable_features(stability, threshold=0.8)

# Use stable features for all downstream analysis
model.plot_top_kernels(X_test, y_test, feature_mask=stable_features)
model.plot_temporal_importance(X_test, y_test, feature_mask=stable_features)
```

### Permutation Importance (PIMP)

PIMP provides statistically corrected feature importance with p-values, adapted from Altmann et al. (2010). It compares observed importance against a null distribution built from permuted class labels. A RandomForestClassifier is used by default, which produces a well-behaved null distribution because uninformative features receive zero importance. The ROCKET transform is label-independent and computed once.

```python
from interp_rocket import permutation_importance_test, plot_permutation_importance

pimp = permutation_importance_test(model, X_train, y_train, n_permutations=100)
fig = plot_permutation_importance(pimp, model=model)

# Extract significant features
sig_features = np.where(pimp['significant_mask'])[0]
print(f"Significant features (p < 0.05): {len(sig_features)}")

# Use as feature mask for downstream analysis
model.plot_temporal_importance(X_test, y_test, feature_mask=sig_features)
```

FSA and PIMP provide complementary evidence. FSA identifies features that are consistently important across data splits. PIMP identifies features whose importance exceeds chance. Features that pass both tests form the most robust interpretable set. See `examples/demo_pimp.ipynb` for a full demonstration.

### Recursive Feature Elimination

RFE is available for datasets where the accuracy curve produces a clear knee point. Note that RFE results can vary with random seed and data split. Use `knee_method='both'` to compare threshold and Kneedle methods.

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

The Kneedle algorithm (Satopaa et al., 2011) finds the knee point in the accuracy curve, providing a principled alternative to the 1% threshold rule. Use `knee_method='both'` to compare both methods on the same RFE run.

## Information Decomposition

Adapted from neural ensemble analysis (Narayanan, Kimchi, & Laubach, 2005), this decomposes each feature group's contribution into redundant, synergistic, and independent information:

```python
from interp_rocket import information_decomposition, plot_information_decomposition

info = information_decomposition(
    model, X_test, y_test,
    feature_mask=stable_features,
    group_by='kernel',      # group features by base kernel
    n_shuffles=100,
)
fig = plot_information_decomposition(info)
```

Features classified as **redundant** carry information already available from other features. **Synergistic** features only contribute when combined with others. **Independent** features carry unique information. Features identified by FSA or PIMP tend to be predominantly independent and synergistic. This structure explains why aggressive feature reduction preserves accuracy.

## Extensions

The `extensions/` directory contains optional modules that integrate I-ROCKET with related tools and methods from the time series classification literature. These modules have no effect on the core package and are not required for basic use.

### Kernel Explorer

An interactive Jupyter notebook (`extensions/kernel_explorer.ipynb`) for understanding the building blocks of MultiRocket classifiers: the 84 fixed convolutional kernels, dilation, bias thresholding, and the four pooling operators. Uses `ipywidgets` to provide real-time controls:

- **Kernel slider** (0-83): select any of the 84 base kernels
- **Dilation slider** (1-16): stretch the kernel across different timescales
- **Signal dropdown**: Gaussian bump, two peaks, or oscillatory test signals
- **Bias slider**: manually adjust the firing threshold
- **Auto bias checkbox**: use the median convolution output (default)

Four panels update interactively: the dilated kernel weight pattern, the kernel overlaid on the signal at peak response, the full convolution output with bias threshold, and the four pooling operator values (PPV, MPV, MIPV, LSPV). Requires `ipywidgets` (`pip install ipywidgets`).

### AMEE Evaluation

`extensions/amee_evaluation.py` implements the AMEE framework (Nguyen et al., 2024) for quantitative evaluation and ranking of saliency-based explanation methods. It evaluates how informative a saliency map is by perturbing the most important time regions and measuring the resulting accuracy drop. A larger drop indicates a more informative explanation. The module provides:

- Saliency map extraction from I-ROCKET's temporal importance and occlusion tools
- Random and inverse baselines for comparison
- Four perturbation strategies (zero, mean, noise, inverse)
- Full AMEE evaluation across multiple explainers and perturbation methods
- Ranking by mean AUC of accuracy drop curves

See `examples/demo_amee.ipynb` for a demonstration on the three-bumps dataset.

### TSHAP Integration

`extensions/tshap_integration.py` bridges I-ROCKET with the TSHAP package (Le Nguyen and Ifrim, 2025) for instance-level Shapley value attributions. TSHAP provides exact SHAP values by grouping timepoints into sliding windows, keeping Shapley computation tractable for time series. The module wraps I-ROCKET's prediction pipeline into the format TSHAP expects and provides utilities for comparing TSHAP attributions with I-ROCKET's built-in interpretability tools. Requires `tshap` (`pip install tshap`).

See `examples/demo_tshap_amee.ipynb` for a combined TSHAP + AMEE evaluation comparing I-ROCKET's analytical temporal importance with TSHAP's game-theoretic attributions.

### Channel Selection for Multivariate Data

`extensions/channel_selection.py` implements classifier-agnostic channel selection for multivariate time series, following Dhariyal et al. (2023). Channels where class prototypes are well-separated carry more discriminative information and are retained; channels where classes overlap are discarded. This serves as a preprocessing step before I-ROCKET classification of multichannel data such as multi-electrode neural recordings, multi-sensor motion capture, or multi-band spectrograms.

```python
from channel_selection import select_channels, flatten_channels

# X_train has shape (n_samples, n_channels, n_timepoints)
selected, scores = select_channels(X_train, y_train, method='ecp')
X_train_flat = flatten_channels(X_train, selected)
X_test_flat = flatten_channels(X_test, selected)

# Now pass to I-ROCKET as usual
model = InterpRocket()
model.fit(X_train_flat, y_train)
```

See `examples/demo_channel_selection.ipynb` for a demonstration with synthetic multichannel data including accuracy comparisons and temporal importance mapping back to original channels.

## Diagnostics

```python
from interp_rocket import (
    plot_kernel_similarity,
    plot_confusion_conditioned_maps,
    plot_receptive_field_diagram,
    temporal_occlusion, plot_occlusion,
)

# Kernel similarity: which features fire together?
fig, corr = plot_kernel_similarity(model, X_test, feature_mask=stable_features)

# Confusion-conditioned maps: where does the model fail?
fig = plot_confusion_conditioned_maps(model, X_test, y_test, feature_mask=stable_features)

# Receptive field diagram: what temporal scales does the feature set cover?
fig = plot_receptive_field_diagram(model, X_test, y_test, feature_mask=stable_features)

# Temporal occlusion: model-agnostic perturbation analysis
occ = temporal_occlusion(model, X_test, y_test, n_samples=6, feature_mask=stable_features)
plot_occlusion(occ)
```

## Class-Mean Visualization

Apply decoded kernels to class-averaged signals to see what the classifier detects on the idealized waveform for each class:

```python
from interp_rocket import (
    plot_class_mean_activation,
    plot_multi_kernel_summary,
    plot_aggregate_activation,
)

# Side-by-side activation map and convolution output for a single feature
fig = plot_class_mean_activation(model, X_test, y_test,
                                  feature_mask=stable_features, feature_rank=0)

# Heatmap of all features firing across classes
fig = plot_multi_kernel_summary(model, X_test, y_test,
                                 feature_mask=stable_features, n_show=15)

# Aggregate importance-weighted activation with differential
fig, class_act, diff = plot_aggregate_activation(model, X_test, y_test,
                                                   feature_mask=stable_features)
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

All plotting methods accept an optional `feature_mask` parameter to restrict analysis to selected features or any custom subset.

| Method | Output |
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
| `plot_receptive_field_diagram()` | Time span of each feature by pooling operator |
| `plot_class_mean_activation()` | Activation and convolution output on class means (single feature) |
| `plot_multi_kernel_summary()` | Binary activation heatmap across all features and classes |
| `plot_aggregate_activation()` | Importance-weighted activation sum with differential |
| `plot_permutation_importance()` | PIMP bar chart with null distribution and p-value histogram |

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
| `plot_class_mean_activation(model, X, y)` | Activation and convolution on class means (single feature). |
| `plot_multi_kernel_summary(model, X, y)` | Binary activation heatmap on class means. |
| `plot_aggregate_activation(model, X, y)` | Importance-weighted activation sum with differential. |
| `aggregate_temporal_occlusion(model, X, y)` | Per-class occlusion sensitivity over all trials. |
| `get_stable_features(stability, threshold)` | Extract stable feature indices from CV stability results. |
| `permutation_importance_test(model, X, y)` | PIMP: permutation importance with p-values (Altmann et al., 2010). |
| `plot_permutation_importance(pimp_results)` | Visualize PIMP results with significance coloring. |
| `compute_activation_map(x, kernel_index, dilation, bias)` | Single-trial kernel activation (numba-compiled). |
| `mutual_information(y_true, y_pred)` | MI between true and predicted labels. |
| `kneedle(y)` | Kneedle knee-point detection (Satopaa et al., 2011). |

## References

- Altmann, A., Tolosi, L., Sander, O., & Lengauer, T. (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics*, 26(10), 1340-1347.
- Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34(5), 1454-1495.
- Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost) deterministic transform for time series classification. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 248-257.
- Dhariyal, B., Le Nguyen, T., & Ifrim, G. (2023). Scalable classifier-agnostic channel selection for multivariate time series classification. *Data Mining and Knowledge Discovery*, 37, 1010-1054.
- Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1), 389-422.
- Le Nguyen, T. & Ifrim, G. (2025). TSHAP: Fast and exact SHAP for explaining time series classification and regression. *ECML-PKDD 2025*.
- Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Lundy, C., & O'Toole, J. M. (2021). Random convolution kernels with multi-scale decomposition for preterm EEG inter-burst detection. *2021 29th European Signal Processing Conference (EUSIPCO)*, 1182-1186.
- Meinshausen, N. & Buhlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417-473.
- Narayanan, N. S., Kimchi, E. Y., & Laubach, M. (2005). Redundancy and synergy of neuronal ensembles in motor cortex. *Journal of Neuroscience*, 25(17), 4207-4216.
- Nguyen, T. T., Nguyen, T. L., & Ifrim, G. (2024). Robust explainer recommendation for time series classification. *Data Mining and Knowledge Discovery*, 38, 3372-3413.
- O'Toole, J. M. (2023). ms_rocket: Multi-scale ROCKET for time series classification. GitHub repository. https://github.com/otoolej/ms_rocket
- Saeys, Y., Abeel, T., & Van de Peer, Y. (2008). Robust feature selection using ensemble feature selection techniques. *Proc. ECML PKDD*, 313-325.
- Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior. *ICDCS Workshops*.
- Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification. *Data Mining and Knowledge Discovery*, 36(5), 1623-1646.
- Uribarri, G., Barone, F., Ansuini, A., & Fransen, E. (2024). Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels. *Data Mining and Knowledge Discovery*, 38, 3922-3947.

## Author

Mark Laubach (American University, Department of Neuroscience). Developed with Claude (Anthropic) as AI coding assistant.

## License

BSD-3-Clause
