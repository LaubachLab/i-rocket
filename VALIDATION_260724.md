# Validation record for release 260724

Python package version: `0.7.0`

The release tree was checked using the following procedures.

## Automated source tests

The complete test suite was run with Numba JIT disabled and numerical thread
counts restricted to one:

```text
193 passed, 1 skipped
```

The skipped test is the optional direct aeon parity check because aeon was not
installed in the validation environment. The independent readable
MultiRocket-reference tests passed.

## Compiled smoke test

With Numba compilation enabled, a smoke analysis completed:

```text
transform -> resampled consensus selection -> ridge classification
-> selected-kernel plotting -> composite activation trace
-> spectral analysis -> nested cross-validation -> final refit
```

## Conceptual tools

```text
pooling_explorer.py --self-test: passed
kernel_explorer.py --help: passed from the repository root
```

## Distribution checks

- A wheel named `interp_rocket-0.7.0-py3-none-any.whl` was built.
- The wheel was installed into an isolated target without the source directory
  on `sys.path`.
- Transformation, selection, classification, spectral analysis, nested
  validation, and final refitting succeeded from the installed wheel.
- The wheel contains no bytecode or cache files.
- The source archive contains an empty `examples/` directory as requested.

The demonstration notebooks are intentionally not part of release 260724 and
were not included in these checks.
