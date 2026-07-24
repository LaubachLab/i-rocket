"""Equal-budget comparison of I-ROCKET and aeon MultiRocket on three bumps.

This is a development benchmark, not a unit test. It deliberately uses the
same downstream scaler and RidgeClassifierCV for both transforms so the timing
and prediction comparison isolate the transformed features as much as possible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path(__file__).resolve().parent / "datasets"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATASETS))

from interp_rocket import (  # noqa: E402
    _fit_biases,
    _fit_dilations,
    _quantiles,
    _transform,
)
from three_bumps import generate_three_bumps  # noqa: E402

try:  # noqa: E402
    from aeon.transformations.collection.convolution_based import MultiRocket
except ImportError as exc:  # pragma: no cover - benchmark-only dependency
    raise SystemExit(
        "This benchmark requires aeon. Install it with `python -m pip install aeon`."
    ) from exc


@dataclass(frozen=True)
class IRParameters:
    raw: tuple[np.ndarray, np.ndarray, np.ndarray]
    diff: tuple[np.ndarray, np.ndarray, np.ndarray]


def fit_ir_parameters(
    X: np.ndarray,
    n_features: int,
    max_dilations: int,
    seed: int,
) -> IRParameters:
    """Fit raw and first-difference parameters without fitting a classifier."""
    raw_d, raw_c = _fit_dilations(X.shape[1], n_features, max_dilations)
    raw_b = _fit_biases(
        X,
        raw_d,
        raw_c,
        _quantiles(84 * int(raw_c.sum())),
        seed,
    )

    X_diff = np.diff(X, axis=1).astype(np.float32)
    diff_d, diff_c = _fit_dilations(
        X_diff.shape[1], n_features, max_dilations
    )
    diff_b = _fit_biases(
        X_diff,
        diff_d,
        diff_c,
        _quantiles(84 * int(diff_c.sum())),
        seed,
    )
    return IRParameters(
        raw=(raw_d, raw_c, raw_b),
        diff=(diff_d, diff_c, diff_b),
    )


def transform_ir(X: np.ndarray, parameters: IRParameters) -> np.ndarray:
    """Transform raw and first-difference representations."""
    X_diff = np.diff(X, axis=1).astype(np.float32)
    return np.concatenate(
        [
            _transform(X, *parameters.raw),
            _transform(
                X_diff,
                *parameters.diff,
                is_first_difference=True,
            ),
        ],
        axis=1,
    )


def fit_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
) -> np.ndarray:
    """Apply the same downstream model to either transformed matrix."""
    scaler = StandardScaler(with_mean=False)
    Z_train_scaled = scaler.fit_transform(Z_train)
    Z_test_scaled = scaler.transform(Z_test)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(Z_train_scaled, y_train)
    return classifier.predict(Z_test_scaled)


def warm_jit() -> None:
    """Compile both Numba implementations outside the timed region."""
    warm = np.random.default_rng(0).normal(size=(12, 20)).astype(np.float32)
    parameters = fit_ir_parameters(warm, 84, 1, 0)
    transform_ir(warm[:2], parameters)

    aeon = MultiRocket(
        n_kernels=84,
        max_dilations_per_kernel=1,
        n_jobs=1,
        random_state=0,
    )
    aeon.fit_transform(warm[:, np.newaxis, :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--timepoints", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1.5)
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--max-dilations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    warm_jit()
    X, y = generate_three_bumps(
        n_samples=args.samples,
        n_timepoints=args.timepoints,
        noise_std=args.noise,
        random_state=args.seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=args.seed,
    )

    start = perf_counter()
    ir_parameters = fit_ir_parameters(
        X_train, args.features, args.max_dilations, args.seed
    )
    ir_fit_time = perf_counter() - start
    start = perf_counter()
    Z_ir_train = transform_ir(X_train, ir_parameters)
    ir_train_time = perf_counter() - start
    start = perf_counter()
    Z_ir_test = transform_ir(X_test, ir_parameters)
    ir_test_time = perf_counter() - start

    aeon = MultiRocket(
        n_kernels=args.features,
        max_dilations_per_kernel=args.max_dilations,
        n_jobs=1,
        random_state=args.seed,
    )
    start = perf_counter()
    aeon.fit(X_train[:, np.newaxis, :])
    aeon_fit_time = perf_counter() - start
    start = perf_counter()
    Z_aeon_train = aeon.transform(X_train[:, np.newaxis, :])
    aeon_train_time = perf_counter() - start
    start = perf_counter()
    Z_aeon_test = aeon.transform(X_test[:, np.newaxis, :])
    aeon_test_time = perf_counter() - start

    start = perf_counter()
    prediction_ir = fit_predict(Z_ir_train, y_train, Z_ir_test)
    ir_classifier_time = perf_counter() - start
    start = perf_counter()
    prediction_aeon = fit_predict(Z_aeon_train, y_train, Z_aeon_test)
    aeon_classifier_time = perf_counter() - start

    import aeon
    import numba
    import sklearn

    print("Configuration")
    print(
        f"samples={args.samples}, timepoints={args.timepoints}, "
        f"noise={args.noise}, requested_features={args.features}, "
        f"max_dilations={args.max_dilations}, seed={args.seed}"
    )
    print(
        f"versions: numpy={np.__version__}, numba={numba.__version__}, "
        f"scikit-learn={sklearn.__version__}, aeon={aeon.__version__}"
    )
    print(
        "threads: "
        f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS', 'unset')}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')}"
    )
    print(f"Training feature shapes: IR={Z_ir_train.shape}, aeon={Z_aeon_train.shape}")
    print("\nWarm timings (seconds)")
    print("component                 I-ROCKET       aeon")
    print(f"fit transform parameters  {ir_fit_time:10.4f}  {aeon_fit_time:10.4f}")
    print(f"transform training data   {ir_train_time:10.4f}  {aeon_train_time:10.4f}")
    print(f"transform test data       {ir_test_time:10.4f}  {aeon_test_time:10.4f}")
    print(f"scale + ridge              {ir_classifier_time:10.4f}  {aeon_classifier_time:10.4f}")
    ir_total = ir_fit_time + ir_train_time + ir_test_time + ir_classifier_time
    aeon_total = (
        aeon_fit_time + aeon_train_time + aeon_test_time + aeon_classifier_time
    )
    print(f"total measured             {ir_total:10.4f}  {aeon_total:10.4f}")

    print("\nPredictive comparison")
    print(
        "I-ROCKET: "
        f"accuracy={accuracy_score(y_test, prediction_ir):.4f}, "
        f"balanced={balanced_accuracy_score(y_test, prediction_ir):.4f}"
    )
    print(
        "aeon:     "
        f"accuracy={accuracy_score(y_test, prediction_aeon):.4f}, "
        f"balanced={balanced_accuracy_score(y_test, prediction_aeon):.4f}"
    )
    print(
        "Prediction agreement: "
        f"{np.mean(prediction_ir == prediction_aeon):.4f}"
    )


if __name__ == "__main__":
    main()
