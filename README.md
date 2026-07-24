# I-ROCKET

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![tests](https://github.com/LaubachLab/i-rocket/actions/workflows/tests.yml/badge.svg)](https://github.com/LaubachLab/i-rocket/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19055359.svg)](https://doi.org/10.5281/zenodo.19055359)

I-ROCKET is a transparent implementation of the univariate MultiRocket transform designed specifically for interpretable time-series classification. While standard ROCKET models excel at fast and accurate classification, their high-dimensional outputs often act as a black box. I-ROCKET resolves this by ensuring every transformed column can be decoded directly to its generating kernel, dilation, padding mode, bias threshold, and pooling operator.

To achieve stable, interpretable results without data leakage, the package integrates shrinkage-*t* feature selection and Nogueira stability measures within a strict nested cross-validation framework. This enables the accurate estimation of generalization performance while identifying the precise temporal features driving the model's decisions.

The complete I-ROCKET workflow consists of:

```text
MultiRocket transform (training data only)
    -> shrinkage-t ranking within repeated subsamples
    -> one-break segmented cutoff in each subsample
    -> consensus selection probabilities
    -> Nogueira feature-selection stability
    -> ridge classification
    -> nested cross-validation for performance estimation
```

The transform, selector, scaler, and classifier are fitted separately inside each training partition. Outer test observations do not contribute to kernel biases, feature scores, cutoffs, consensus probabilities, regularization selection, scaling, or classifier fitting.

For the rationale behind the pipeline—including why I-ROCKET uses shrinkage-*t* rather than CAT scores, why it filters features, and why it uses resampled consensus selection see [`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md).

## Release

**GitHub release/tag:** `260724`  
**Python package version:** `0.7.0`

Release 260724 includes an updated MultiRocket core, internal shrinkage-*t* filtering, segmented cutoff estimation, resampled consensus selection, the Nogueira stability measure, leakage-free nested validation, focused kernel-interpretation tools, generic spectral helpers, optional TSHAP integration, and two standalone conceptual explorers.

See [`RELEASE_NOTES_260724.md`](RELEASE_NOTES_260724.md) for the release summary, [`VALIDATION_260724.md`](VALIDATION_260724.md) for the validation record, and [`THIRD_PARTY.md`](THIRD_PARTY.md) for optional external integrations.

## Installation

Create a fresh environment and install from the repository:

```bash
conda create --name i-rocket python=3.12 -y
conda activate i-rocket

git clone https://github.com/LaubachLab/i-rocket.git
cd i-rocket
python -m pip install .
```

For editable installation with the test suite:

```bash
python -m pip install -e ".[test]"
```

Optional dependencies are installed separately:

```bash
python -m pip install -e ".[datasets]"   # aeon and OpenML dataset access
python -m pip install -e ".[tshap]"      # TSHAP 0.0.1
python -m pip install -e ".[notebooks]"  # JupyterLab
```

### OpenMP information message

Some systems print the following message when numerical libraries initialize:

```text
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
```

It is informational. To suppress Intel OpenMP diagnostics during a local run, set `KMP_WARNINGS=0` before importing NumPy, SciPy, scikit-learn, Numba, or I-ROCKET.

```bash
KMP_WARNINGS=0 python analysis.py
```

## Quick smoke test

This small example confirms that transformation, consensus selection, and classification work without running a full nested analysis:

```python
import numpy as np

from feature_selection import ResampledShrinkageSelector
from interp_rocket import InterpRocketTransform
from irocket_model_selection import StableRocketClassifier

rng = np.random.default_rng(42)
X = rng.normal(size=(60, 80)).astype(np.float32)
y = np.repeat([0, 1], 30)
X[y == 1, 30:40] += 0.8

model = StableRocketClassifier(
    transformer=InterpRocketTransform(
        num_features=336,
        max_dilations_per_kernel=8,
        representations="both",
        random_state=42,
    ),
    selector=ResampledShrinkageSelector(
        n_resamples=8,
        sample_fraction=0.5,
        cutoff_min_size=5,
        random_state=42,
    ),
    consensus_threshold=0.7,
    alpha=1.0,
    random_state=42,
).fit(X, y)

print(model.predict(X[:5]))
print(model.n_selected_features_)
print(model.nogueira_stability_)
```

## Main components

### `InterpRocketTransform`

`InterpRocketTransform` is a classifier-agnostic scikit-learn transformer. It fits the deterministic MultiRocket kernels, canonical dilation allocation, alternating padded and unpadded convolutions, training-derived bias thresholds, and four pooling operators.

```python
from interp_rocket import InterpRocketTransform

transformer = InterpRocketTransform(
    num_features=10_000,
    max_dilations_per_kernel=32,
    representations="both",
    random_state=42,
)

Z_train = transformer.fit_transform(X_train)
Z_test = transformer.transform(X_test)
```

`num_features` is the requested bias-feature budget per representation. MultiRocket emits four pooling values per bias. With raw and first-differenced representations, `num_features=10_000` produces 79,968 transformed columns after rounding to the 84 deterministic kernels.

```python
metadata = transformer.decode_feature_index(100)

print(metadata["kernel_weights"])
print(metadata["dilation"])
print(metadata["padding_mode"])
print(metadata["bias"])
print(metadata["pooling_op"])
print(metadata["representation"])
```

### `ResampledShrinkageSelector`

The selector operates on one fixed transformed feature universe. Within each resample it calculates shrinkage-*t* scores, ranks features by absolute score, fits a one-break segmented cutoff, and records a binary selection mask. The masks are summarized by selection probabilities and the Nogueira feature-selection stability measure.

```python
from feature_selection import ResampledShrinkageSelector

selector = ResampledShrinkageSelector(
    n_resamples=50,
    sample_fraction=0.5,
    consensus_threshold=0.7,
    cutoff_min_size=5,
    score_power=1.0,
    random_state=42,
)

Z_selected = selector.fit_transform(Z_train, y_train)

print(selector.nogueira_stability_)
print(selector.selected_indices_)
print(selector.selection_probabilities_)
print(np.median(selector.cutoff_sizes_))
```

This procedure is called **resampled consensus selection**. It is not presented as formal Meinshausen-Buehlmann stability selection, and no false-positive error bound is claimed.

### Leakage-free nested validation

Use `nested_stability_cv` to estimate generalization while selecting the consensus threshold and ridge regularization inside the inner loop. The standard trial-level configuration is 10 outer folds and 3 inner folds when class counts permit. For neuroscience data, use group-aware folds when the intended unit of generalization is a participant, session, or recording block.

```python
import numpy as np

from feature_selection import ResampledShrinkageSelector
from interp_rocket import InterpRocketTransform
from irocket_model_selection import nested_stability_cv

transformer = InterpRocketTransform(
    num_features=10_000,
    max_dilations_per_kernel=32,
    representations="both",
    random_state=42,
)

selector = ResampledShrinkageSelector(
    n_resamples=50,
    sample_fraction=0.5,
    cutoff_min_size=5,
    score_power=1.0,
    min_features=1,
    random_state=42,
)

groups = None  # or animal, participant, session, or recording IDs

result = nested_stability_cv(
    X,
    y,
    groups=groups,
    transformer=transformer,
    selector=selector,
    outer_cv=10,
    inner_cv=3,
    consensus_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
    classifier_alphas=np.logspace(-5, 5, 11),
    scoring="balanced_accuracy",
    selection_rule="one_se",
    random_state=42,
    refit=True,
    verbose=True,
    show_progress=True,
)

print(result.summary())
```

The progress bar advances after complete inner-search and refit stages rather than after every model fit. The function prints a startup notice before the first inner search because the initial stage can take several minutes on large datasets and may include Numba compilation.

The one-standard-error rule favors a simpler eligible model when inner-validation performance is statistically indistinguishable from the best mean score. Simplicity is defined by fewer selected features, then a higher consensus threshold, then stronger ridge regularization.

When `groups` are supplied, default splitters keep groups disjoint in outer and inner folds. Selector resampling also preserves complete groups unless `resample_groups=False` is explicitly requested.

### Reduced settings for development

Full nested validation with the standard transform is computationally substantial. Start notebook development with a smaller profile:

```python
transformer = InterpRocketTransform(num_features=1_008, random_state=42)
selector = ResampledShrinkageSelector(n_resamples=10, random_state=42)

result = nested_stability_cv(
    X,
    y,
    transformer=transformer,
    selector=selector,
    outer_cv=3,
    inner_cv=2,
    consensus_thresholds=(0.6, 0.8),
    classifier_alphas=(0.1, 1.0, 10.0),
    random_state=42,
)
```

Increase the transform budget, resample count, and validation folds only after the workflow succeeds on the target machine.

## Final interpretation model

When `refit=True`, nested validation returns `result.final_model`. This model is tuned and fitted on the complete development dataset only after outer-fold performance estimation is complete. It is intended for feature decoding and visualization; its training performance is not an estimate of generalization.

```python
final_model = result.final_model

selected = final_model.selected_indices_
metadata = final_model.get_selected_feature_metadata()

fig = final_model.plot_kernel_properties()
fig = final_model.plot_top_kernels(
    X,
    y,
    n_kernels=5,
    show_difference=True,
)
```

`plot_top_kernels` shows one representative feature per unique convolutional kernel configuration by default. Set `unique_kernels=False` to inspect separate bias thresholds and pooling features from the same kernel. Labels include the full feature identity so distinct activation patterns are not presented as the same feature.

## Interpretation tools

```python
from interpretability import (
    composite_activation_trace,
    localization_profile,
    plot_activation_map,
    plot_activation_trace,
    plot_class_mean_activation,
    plot_feature_stability,
    plot_kernel_pattern,
    plot_kernel_similarity,
    plot_localization_diagnostic,
)
```

| Function | Purpose |
|---|---|
| `plot_feature_stability` | Shows fixed-universe selection masks and consensus probabilities. |
| `plot_kernel_properties` | Summarizes dilation, receptive field, pooling, representation, kernel, and consensus strength of the final selected set. |
| `plot_top_kernels` | Shows kernel weights and trial-level class activation rates for several selected kernel configurations. |
| `plot_class_mean_activation` | Shows one feature's exact convolution output, bias threshold, and activation on each class-average waveform. |
| `plot_activation_map` | Displays class-mean activation and receptive-field coverage for selected kernels. |
| `plot_kernel_pattern` | Displays dilated weight patterns of selected kernels. |
| `plot_kernel_similarity` | Describes correlations among selected transformed features. |
| `composite_activation_trace` | Builds an approximate classifier-weighted pre-pooling trace from selected kernels. |
| `plot_activation_trace` | Plots class-averaged composite traces with standard-error bands. |
| `localization_profile` | Quantifies how much differential activation is concentrated near its centroid. |
| `plot_localization_diagnostic` | Evaluates localization thresholds relative to a receptive-field-width baseline. |

For large datasets, `composite_activation_trace` displays a progress bar automatically at 500 or more trials. Override this behavior with `show_progress=True`, `show_progress=False`, or a different `progress_threshold`.

`plot_top_kernels` and `plot_class_mean_activation` are complementary. The former averages activation incidence across trials and retains trial-to-trial firing information. The latter applies one feature to a class-average waveform and is best treated as a generalized illustration because averaging can hide heterogeneous trial structure.

`information_decomposition` is available as a descriptive analysis of a fixed selected set. Its auxiliary classifiers are fitted and evaluated on the same supplied data; returned mutual-information values are resubstitution diagnostics, not generalization estimates.

## Feature-type filtering

`filter_features_by_type` can identify features by pooling operator and representation:

```python
from feature_selection import filter_features_by_type

raw_ppv = filter_features_by_type(
    fitted_transformer,
    np.arange(fitted_transformer.n_output_features_),
    pooling="PPV",
    representation="raw",
)
```

This function filters an already fitted feature index set; it does not refit the selector or classifier. For a leakage-free raw-PPV analysis, the restriction must be applied inside every training fold before shrinkage scoring and classifier fitting.

A raw-signal PPV-only transform is MiniRocket-style at the transform level. A complete I-ROCKET analysis remains distinct from the standard MiniRocket classifier because it adds shrinkage-*t* filtering, resampled consensus selection, Nogueira stability, and nested model selection.

## Generic spectral helpers

The `spectral` module supports datasets whose class structure is distributed in frequency, such as the time-series benchmarks FordA and FordB and EEG or LFP datasets.

```python
from spectral import (
    class_power_spectra,
    plot_class_power_spectra,
    plot_selected_kernel_spectrum,
    selected_kernel_spectrum,
)
```

For first-difference features, the kernel response can include the frequency response of differencing. Pooling operator and bias threshold do not alter the underlying linear kernel response, so several transformed features can share a spectrum while remaining distinct classifier features.

## Optional TSHAP integration

TSHAP provides model-agnostic temporal attributions for classification and regression. It is useful as a complementary check on datasets with temporally localized discriminative structure.

```bash
python -m pip install -e ".[tshap]"
```

```python
from tshap_integration import explain_with_tshap, plot_tshap_attribution

baselines = X_train[baseline_indices]
explanation = explain_with_tshap(
    final_model,
    X_test[:10],
    baselines,
    targets=None,
    window_length=20,
    stride=5,
    output="decision",
)

plot_tshap_attribution(
    explanation,
    X=X_test[:10],
    sample_index=0,
    attribution="window",
)
```

Baseline choice changes the explanatory question and is therefore never hidden by the adapter. Choose baselines from training data and report the selection rule.

TSHAP is maintained externally and licensed under GPL-3.0. I-ROCKET contains only BSD-licensed adapter and plotting code and does not copy or redistribute TSHAP source. Version `0.0.1` is pinned because it is the API tested for release 260724.

## Conceptual tools

The repository includes two standalone Matplotlib programs under `tools/`:

- `kernel_explorer.py` explains the complete identity of a kernel feature, including dilation, representation, padding, bias, convolution, and pooling.
- `pooling_explorer.py` recreates the four pooling features illustrated in Figure 3 of the MultiRocket paper and can apply them to individual trials from user datasets.

```bash
python tools/kernel_explorer.py
python tools/pooling_explorer.py
python tools/pooling_explorer.py --self-test
```

The tools are distributed with the source repository but are not required by the core analysis pipeline.

## Examples and benchmarks

The `examples/` directory contains demonstrations of applying I-ROCKET to the synthetic waveform dataset, a participant-aware version of the GunPoint dataset, and the FordB dataset.

The `benchmarks/` directory contains:

- a Three Bumps generator with known, nonoverlapping signal locations and an estimable Bayes error;
- a fair I-ROCKET-versus-aeon comparison with matched feature budgets and downstream models;
- resampled-selection diagnostics;
- a leakage-free nested-validation benchmark.

## Input conventions

The current transform supports univariate time series:

```python
X.shape == (n_samples, n_timepoints)
y.shape == (n_samples,)
```

Inputs must be finite. At least nine timepoints are required for raw-only transforms and at least ten when first differences are used. We do not recommend applying I-ROCKET to such short data series.

## Testing

Run the release suite from the repository root:

```bash
python -m pytest -q
```

The suite covers numerical MultiRocket conformance, estimator contracts, shrinkage-*t*, segmented cutoffs, resampling, Nogueira stability, nested-CV leakage audits, feature decoding, plotting identity, spectral helpers, optional TSHAP integration, conceptual-tool parsing, and packaging metadata.

For deterministic local testing, avoid nested parallelism:

```bash
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m pytest -q
```

## Methodological references

Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery, 34*, 1454-1495.

Le Nguyen, T., & Ifrim, G. (2025). TSHAP: Fast and Exact SHAP for Explaining Time Series Classification and Regression. In *Machine Learning and Knowledge Discovery in Databases. Research Track*, 60-77. DOI: 10.1007/978-3-032-06078-5_4

Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *Journal of Machine Learning Research, 18*(174), 1-54.

Opgen-Rhein, R., & Strimmer, K. (2007). Accurate ranking of differentially expressed genes by a distribution-free shrinkage approach. *Statistical Applications in Genetics and Molecular Biology, 6*(1), Article 9.

Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification. *Data Mining and Knowledge Discovery, 36*, 1623-1646.

Additional selection-method references are provided in [`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md).

## Development notes

Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google) were used as coding and documentation assistants during development. AI assistance included code refactoring, documentation revision, test-design suggestions, and algorithmic review. All code and documentation were reviewed, edited, and approved by Mark Laubach, who takes responsibility for the final package.

## License

I-ROCKET is distributed under the BSD 3-Clause License.
