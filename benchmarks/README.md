# I-ROCKET benchmarks

The benchmark scripts serve distinct validation purposes. They are not part of
the installed package.

## Three-bumps dataset

`datasets/three_bumps.py` generates a three-class problem with one Gaussian bump
at a known, non-overlapping temporal location for each class. It supports
predictive validation against an estimable Bayes error and direct validation of
temporal localization.

## Fair transform comparison

`benchmark_three_bumps_fair.py` compares the I-ROCKET transform with
aeon's public `MultiRocket` transformer using:

- the same transformed feature count;
- the same dilation limit and random seed;
- one thread for each implementation;
- the same train/test split;
- the same scaler, ridge classifier, and regularization grid.

Run from the repository root after installing the optional dataset tools:

```bash
python -m pip install ".[datasets]"
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/benchmark_three_bumps_fair.py
```

The script performs a warm benchmark. Cold import and JIT-compilation costs
should be measured in separate processes for a formal performance comparison.

## Waveform, GunPoint, and FordB reference comparison

`benchmark_aeon_reference.py` extends the equal-budget comparison to the three
public demonstration datasets. It checks the actual transformed matrix shapes
before fitting either classifier and uses the same scaler and ridge-alpha grid
for both implementations.

Start with the reduced default budget:

```bash
python -m pip install -e ".[datasets]"
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python benchmarks/benchmark_aeon_reference.py
```

For a full-budget run and machine-readable output:

```bash
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python benchmarks/benchmark_aeon_reference.py \
        --features 10000 --repeats 3 \
        --output-json benchmark_reference_full.json
```

Waveform is split once with a fixed stratified seed. GunPoint and FordB retain
the archive train/test partitions. The script measures warm execution after a
small compilation pass; formal cold-start timing requires separate processes.

## Resampled-selection diagnostic

`benchmark_three_bumps_selection.py` fits one repaired transform, performs
repeated shrinkage-*t* selection with the segmented cutoff, and reports the
cutoff-size distribution, segmented-fit improvement, Nogueira stability, and
held-out results for all predeclared consensus thresholds.

This is a development diagnostic. Because every threshold is displayed on the
same held-out partition, it must not be used to select the final threshold.

```bash
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/benchmark_three_bumps_selection.py
```

## Leakage-free nested validation

`benchmark_three_bumps_nested.py` runs the complete validation path:

- outer folds estimate generalization;
- inner folds choose the consensus threshold and ridge alpha;
- every transform and selector is fitted on training rows only;
- the one-standard-error rule favors a simpler eligible model;
- a final full-development model is fitted only for interpretation.

```bash
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/benchmark_three_bumps_nested.py
```

The final model's training performance is intentionally not reported. The
outer-fold results are the generalization estimate.
