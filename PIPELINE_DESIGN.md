# I-ROCKET pipeline design

## Overview

MultiRocket creates a large, overcomplete representation of each time series. A standard univariate transform using raw and first-differenced signals produces approximately 80,000 columns. Each transformed feature is defined by a base kernel, dilation, padding mode, bias threshold, pooling operator, and signal representation.

This redundancy is useful for classification because several features can capture related aspects of the same temporal structure. It is less useful for scientific interpretation. A classifier containing tens of thousands of correlated columns does not provide a concise or reproducible account of which signal patterns distinguish the classes.

I-ROCKET separates the analysis into three tasks:

1. construct the complete MultiRocket representation;
2. identify transformed features that show reproducible class separation;
3. fit and evaluate a regularized classifier using the retained features.

The leakage-free workflow is:

```text
outer training partition
    -> fit the MultiRocket transform
    -> repeatedly subsample the transformed training data
    -> calculate shrinkage-t scores in each subsample
    -> estimate a segmented cutoff in each ranked score curve
    -> calculate feature-selection probabilities
    -> construct candidate consensus feature sets
    -> choose the consensus threshold and ridge alpha by inner validation
    -> refit the complete pipeline on the outer training partition
    -> evaluate once on the untouched outer test partition
```

After outer-fold performance estimation is complete, a separate final model is tuned and fitted on the complete dataset. That model is used for kernel decoding and visualization. Its training performance is not reported as an estimate of generalization.

## Why filter the MultiRocket feature space?

Feature filtering in I-ROCKET means screening transformed columns. It does not mean filtering or smoothing the original time series.

A ridge classifier can operate on the complete MultiRocket matrix, and the full representation often predicts well. Ridge regularization, however, does not select a concise set of features. When transformed columns are correlated, predictive weight can be distributed across a large family of coefficients. The model can therefore be accurate while remaining difficult to summarize physiologically or mechanically.

Feature filtering serves four purposes:

- it removes columns with little evidence of class separation;
- it produces a manageable set of kernels for visualization and downstream analysis;
- it allows the reproducibility of feature selection to be measured under resampling;
- it separates feature screening from the final classifier, so inclusion is not defined solely by one fitted set of ridge coefficients.

Filtering is not assumed to improve accuracy. Depending on the dataset, a filtered model may perform better, equally well, or worse than the full transform. I-ROCKET therefore selects filtering parameters inside nested cross-validation and reports predictive performance only from held-out outer folds.

The current implementation still calculates the complete MultiRocket transform before applying a feature mask. Filtering reduces the statistical and interpretive complexity of the downstream model; it does not eliminate the initial convolution cost.

## Why use the shrinkage-*t* statistic?

The shrinkage-*t* statistic of Opgen-Rhein and Strimmer (2007) measures the standardized difference between class means while stabilizing noisy feature-specific variance estimates by shrinking them toward a common target.

This is useful for MultiRocket for several reasons.

First, the pooling operators have different numerical scales and units. PPV is a proportion, MPV depends on convolution magnitude, MIPV is a temporal position, and LSPV is a run-length statistic. A standardized class contrast allows these heterogeneous feature types to be ranked on a common scale.

Second, variance shrinkage is appropriate when there are many more transformed features than independent observations. Individual variance estimates can be noisy in small datasets even when the number of trials appears large.

Third, shrinkage-*t* is a filter statistic. Its ranking is independent of the ridge classifier fitted later. A high absolute score has a direct meaning: the decoded feature shows a comparatively large and stable difference between classes.

For binary problems, I-ROCKET ranks features by the absolute shrinkage-*t* score. For multiclass problems, it calculates one-versus-rest scores and combines them using a documented aggregation rule.

## Why shrinkage-*t* rather than CAT scores?

The correlation-adjusted *t*-score, or CAT score, of Zuber and Strimmer (2009) is a valid alternative. CAT adjusts the vector of *t*-scores for correlations among candidate features and is useful when conditional rather than marginal importance is the scientific target.

CAT is not the default I-ROCKET selector for three reasons.

### Scale and computation

A complete MultiRocket transform contains nearly 80,000 columns. A dense correlation matrix at that size requires roughly 50 GB in double precision memory before decomposition. Low-rank and shrinkage-based computational strategies are possible, but repeating correlation estimation within every subsample and inner training fold would add substantial cost and complexity.

### Statistical estimation

CAT requires estimation of a large dependence structure. In the small-sample, high-dimensional settings common in scientific data, that estimate may depend strongly on the particular trials, sessions, or participants included in the training sample.

### Interpretive target

A shrinkage-*t* score describes the marginal standardized class separation of one decoded MultiRocket feature. A CAT score for the same feature depends on the complete candidate set and its estimated correlation structure. The latter is appropriate for conditional importance, but it is less direct when the goal is to visualize and explain a specific kernel.

I-ROCKET uses shrinkage-*t* to find features with reproducible marginal separation and leaves joint weighting of the retained set to the regularized classifier. Correlated substitutes are handled empirically through resampling and selection probabilities.

This choice does not imply that CAT is generally inferior. CAT may be useful after substantial prescreening, when the remaining feature set is small enough and the sample size is sufficient to estimate its correlation structure. It is not part of the default pipeline.

## How is the feature-count cutoff chosen?

Within each resample, I-ROCKET sorts features by the absolute magnitude of their shrinkage-*t* scores. It then fits a one-break segmented regression to the ranked score curve.

The breakpoint estimates the transition between:

- a relatively steep region containing stronger class contrasts; and
- a shallower tail containing weaker contrasts.

This avoids an arbitrary fixed top-*k* rule. The breakpoint is recalculated in every resample, so the number of selected features is allowed to vary with the data.

A forced one-break model always returns a breakpoint. It is not a significance test and does not prove that the scores come from two distinct populations. I-ROCKET therefore retains diagnostics such as improvement over a single-line fit and the slopes before and after the breakpoint. The usefulness of the resulting consensus set is evaluated by inner validation rather than assumed from the breakpoint alone.

## Why use resampled consensus selection?

A feature list obtained from one training sample can be sensitive to which observations happen to be included. This is especially important for small datasets and overcomplete transforms containing many correlated alternatives.

I-ROCKET repeatedly applies shrinkage-*t* scoring and segmented cutoff selection to stratified subsamples of one fixed transformed training matrix. The resulting binary masks provide a selection probability for every feature.

Features that repeatedly survive perturbations of the training data receive high selection probabilities. Features that appear only in particular resamples receive lower probabilities. A consensus threshold converts these probabilities into a candidate feature set.

The threshold is not chosen using test data. It is selected inside the inner cross-validation loop together with the ridge regularization parameter. The default one-standard-error rule favors a simpler, more strongly regularized model when its inner-validation performance is statistically indistinguishable from the best mean score.

When group identifiers are supplied, complete participants, sessions, or recording blocks are preserved during resampling. The grouping variable should represent the scientific unit to which the model is expected to generalize.

## Why report Nogueira stability?

Selection probability describes each feature separately. The Nogueira feature-selection stability measure summarizes reproducibility of the selection procedure as a whole.

Raw overlap can be misleading. Two sets overlap substantially by chance if each contains most of the available features, while two small sets may have limited raw overlap even when their agreement is meaningful. The Nogueira measure adjusts observed selection variability for the selection frequencies and feature-set size.

A value of 1 represents identical selections across resamples. Values near zero indicate reproducibility near the chance reference. Negative finite-sample values can occur when selections are less consistent than that reference.

The Nogueira value is reported as a diagnostic; it is not mixed with accuracy in an arbitrary weighted objective. Inner validation selects the consensus threshold according to predictive performance, while the stability value reports how reproducibly the underlying selector behaves.

I-ROCKET calls the procedure **resampled consensus selection**. It uses repeated subsampling and selection frequencies, but it is not presented as formal Meinshausen-Bühlmann stability selection. The package does not impose all assumptions required for classical false-selection error bounds and does not claim those guarantees.

All masks used to calculate one Nogueira value refer to the same fitted transform. Exact column identities are not pooled across independently fitted outer folds because each transform estimates its own bias thresholds.

## Why not Detach-ROCKET or recursive feature elimination?

Detach-ROCKET introduces Sequential Feature Detachment, a backward-selection method designed for ROCKET models. At each step it fits a ridge classifier, ranks active features by absolute coefficient magnitude, removes a proportion of the lowest-ranked columns, and refits. Recursive feature elimination follows the same broad wrapper principle.

These methods are useful when the main objective is to compress a particular predictive model. They are not the default I-ROCKET strategy because the scientific objective is different.

- Detachment and RFE define importance through a particular fitted classifier. Shrinkage-*t* provides a classifier-agnostic measure of class separation.
- Recursive elimination is path dependent. Once a feature is removed, it cannot re-enter later.
- With correlated feature families, ridge coefficients can divide evidence among several columns. An early pruning step can remove members of a relevant family because each has a modest individual coefficient.
- Wrapper methods require many repeated classifier fits. Embedding them inside resampling and nested cross-validation multiplies that cost.
- A compact model obtained from one pruning path does not establish that the same columns would be selected after a modest change in the observations.

Detach-ROCKET is therefore a complementary model-compression approach. I-ROCKET prioritizes a direct class-contrast statistic, reproducibility under data perturbation, and explicit mapping of retained features back to their generating kernels.

The architecture does not preclude comparisons with Sequential Feature Detachment, RFE, CAT, or other selectors. Any alternative must be fitted and tuned entirely inside the appropriate training partitions.

## Validation design for neuroscience datasets

The outer test unit should match the intended scientific claim.

- For prediction of another independent trial under the same recording context, use stratified 10-fold outer and 3-fold inner validation when class counts permit.
- For robustness to another recording block or session, use block- or session-grouped outer folds.
- For generalization to an unseen animal or participant, use leave-one-animal-out or leave-one-participant-out outer validation.

Trial-wise leave-one-out validation is not the default. It is computationally expensive, produces singleton test folds, and can leave session or subject fingerprints in both training and testing data. Group-aware validation is generally more informative for EEG and LFP studies.

## Limitations

The default selector is marginal. It may miss features that are weak individually but informative only through joint interactions.

Highly correlated features may substitute for one another across resamples, lowering exact feature-level stability even when the broader kernel family is reproducible. Kernel-configuration summaries should therefore be considered alongside exact transformed-column identities.

High Nogueira stability indicates reproducible selection. It does not establish causal, physiological, or mechanistic importance.

Neither filtering nor nested cross-validation eliminates distribution shift. Performance on an independent acquisition condition, session, participant, or archive test set should be reported whenever such data are available.

## References

Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning, 46*, 389-422. DOI: 10.1023/A:1012487302797

Meinshausen, N., & Buehlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B, 72*, 417-473. DOI: 10.1111/j.1467-9868.2010.00740.x

Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *Journal of Machine Learning Research, 18*(174), 1-54.

Opgen-Rhein, R., & Strimmer, K. (2007). Accurate ranking of differentially expressed genes by a distribution-free shrinkage approach. *Statistical Applications in Genetics and Molecular Biology, 6*(1), Article 9. DOI: 10.2202/1544-6115.1252

Uribarri, G., Barone, F., Ansuini, A., & Fransen, E. (2024). Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels. *Data Mining and Knowledge Discovery, 38*, 3922-3947. DOI: 10.1007/s10618-024-01062-7

Zuber, V., & Strimmer, K. (2009). Gene ranking and biomarker discovery under correlation. *Bioinformatics, 25*(20), 2700-2707. DOI: 10.1093/bioinformatics/btp460
