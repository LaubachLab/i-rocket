"""Fair I-ROCKET versus aeon MultiRocket benchmark on reference datasets.

Both transforms receive the same requested feature budget, dilation limit,
random seed, thread count, train/test partition, scaler, and ridge-alpha grid.
The actual transformed shapes are checked before classifier fitting.

The default feature budget is reduced for a practical smoke benchmark. Use
``--features 10000`` for the full configuration after the smoke run succeeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Optional, Sequence, Tuple

# Set conservative defaults before importing NumPy, Numba, or scikit-learn.
for variable in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interp_rocket import InterpRocketTransform  # noqa: E402

try:  # benchmark-only optional dependency
    import aeon
    from aeon.datasets import load_classification
    from aeon.transformations.collection.convolution_based import MultiRocket
except ImportError:  # pragma: no cover
    aeon = None
    load_classification = None
    MultiRocket = None


def _require_aeon() -> None:
    if aeon is None or load_classification is None or MultiRocket is None:
        raise SystemExit(
            "This benchmark requires aeon. Install dataset extras with "
            "`python -m pip install -e '.[datasets]'`."
        )


REFERENCE_DATASETS = ("Waveform", "GunPoint", "FordB")
DEFAULT_ALPHAS = tuple(float(value) for value in np.logspace(-3, 3, 10))


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    n_train: int
    n_test: int
    n_timepoints: int
    n_classes: int
    requested_features: int
    output_features: int
    repeat: int
    irocket_fit_seconds: float
    irocket_train_transform_seconds: float
    irocket_test_transform_seconds: float
    irocket_classifier_seconds: float
    aeon_fit_seconds: float
    aeon_train_transform_seconds: float
    aeon_test_transform_seconds: float
    aeon_classifier_seconds: float
    irocket_accuracy: float
    irocket_balanced_accuracy: float
    aeon_accuracy: float
    aeon_balanced_accuracy: float
    prediction_agreement: float

    @property
    def irocket_total_seconds(self) -> float:
        return (
            self.irocket_fit_seconds
            + self.irocket_train_transform_seconds
            + self.irocket_test_transform_seconds
            + self.irocket_classifier_seconds
        )

    @property
    def aeon_total_seconds(self) -> float:
        return (
            self.aeon_fit_seconds
            + self.aeon_train_transform_seconds
            + self.aeon_test_transform_seconds
            + self.aeon_classifier_seconds
        )


def _as_2d_univariate(X, *, name: str) -> np.ndarray:
    values = np.asarray(X)
    if values.ndim == 3:
        if values.shape[1] != 1:
            raise ValueError(f"{name} is multivariate; I-ROCKET is univariate.")
        values = values[:, 0, :]
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D or univariate 3D array.")
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values.")
    return np.ascontiguousarray(values)


def _stratified_limit(
    X: np.ndarray,
    y: np.ndarray,
    maximum: Optional[int],
    *,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if maximum is None or maximum >= y.shape[0]:
        return X, y
    if maximum < np.unique(y).size * 2:
        raise ValueError("A sample cap must retain at least two rows per class.")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=int(maximum),
        random_state=random_state,
    )
    indices, _ = next(splitter.split(X, y))
    return X[indices], y[indices]


def load_dataset(
    name: str,
    *,
    random_state: int,
    max_train: Optional[int],
    max_test: Optional[int],
) -> DatasetSplit:
    """Load one reference dataset without combining its test partition."""
    normalized = name.strip().lower()
    if normalized == "waveform":
        dataset = fetch_openml(
            name="waveform-5000",
            version=1,
            as_frame=False,
        )
        X = np.asarray(dataset.data[:, :21], dtype=np.float32)
        y = np.asarray(dataset.target)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=random_state,
        )
        display_name = "Waveform"
    elif normalized in {"gunpoint", "fordb"}:
        display_name = "GunPoint" if normalized == "gunpoint" else "FordB"
        X_train_3d, y_train = load_classification(display_name, split="train")
        X_test_3d, y_test = load_classification(display_name, split="test")
        X_train = _as_2d_univariate(X_train_3d, name="X_train")
        X_test = _as_2d_univariate(X_test_3d, name="X_test")
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
    else:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from {REFERENCE_DATASETS}."
        )

    X_train = _as_2d_univariate(X_train, name="X_train")
    X_test = _as_2d_univariate(X_test, name="X_test")
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    X_train, y_train = _stratified_limit(
        X_train,
        y_train,
        max_train,
        random_state=random_state,
    )
    X_test, y_test = _stratified_limit(
        X_test,
        y_test,
        max_test,
        random_state=random_state + 1,
    )
    return DatasetSplit(
        name=display_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def _fit_classifier(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    *,
    alphas: Sequence[float],
) -> Tuple[np.ndarray, float]:
    start = perf_counter()
    scaler = StandardScaler(with_mean=False)
    Z_train_scaled = scaler.fit_transform(Z_train)
    Z_test_scaled = scaler.transform(Z_test)
    classifier = RidgeClassifierCV(alphas=np.asarray(alphas, dtype=float))
    classifier.fit(Z_train_scaled, y_train)
    prediction = classifier.predict(Z_test_scaled)
    return prediction, perf_counter() - start


def warm_implementations(seed: int) -> None:
    """Compile both implementations outside the measured dataset runs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(12, 20)).astype(np.float32)
    irocket = InterpRocketTransform(
        num_features=84,
        max_dilations_per_kernel=1,
        representations="both",
        random_state=seed,
    )
    irocket.fit_transform(X)
    reference = MultiRocket(
        n_kernels=84,
        max_dilations_per_kernel=1,
        n_jobs=1,
        random_state=seed,
    )
    reference.fit_transform(X[:, np.newaxis, :])


def run_once(
    split: DatasetSplit,
    *,
    requested_features: int,
    max_dilations: int,
    alphas: Sequence[float],
    random_state: int,
    repeat: int,
) -> BenchmarkResult:
    transformer = InterpRocketTransform(
        num_features=requested_features,
        max_dilations_per_kernel=max_dilations,
        representations="both",
        random_state=random_state,
    )
    start = perf_counter()
    transformer.fit(split.X_train)
    irocket_fit = perf_counter() - start
    start = perf_counter()
    Z_ir_train = transformer.transform(split.X_train)
    irocket_train_transform = perf_counter() - start
    start = perf_counter()
    Z_ir_test = transformer.transform(split.X_test)
    irocket_test_transform = perf_counter() - start

    reference = MultiRocket(
        n_kernels=requested_features,
        max_dilations_per_kernel=max_dilations,
        n_jobs=1,
        random_state=random_state,
    )
    X_train_3d = split.X_train[:, np.newaxis, :]
    X_test_3d = split.X_test[:, np.newaxis, :]
    start = perf_counter()
    reference.fit(X_train_3d)
    aeon_fit = perf_counter() - start
    start = perf_counter()
    Z_aeon_train = np.asarray(reference.transform(X_train_3d))
    aeon_train_transform = perf_counter() - start
    start = perf_counter()
    Z_aeon_test = np.asarray(reference.transform(X_test_3d))
    aeon_test_transform = perf_counter() - start

    if Z_ir_train.shape != Z_aeon_train.shape:
        raise RuntimeError(
            "Transform feature counts differ: "
            f"I-ROCKET={Z_ir_train.shape}, aeon={Z_aeon_train.shape}."
        )
    if Z_ir_test.shape != Z_aeon_test.shape:
        raise RuntimeError("Train shapes matched but test transform shapes differ.")

    prediction_ir, irocket_classifier = _fit_classifier(
        Z_ir_train,
        split.y_train,
        Z_ir_test,
        alphas=alphas,
    )
    prediction_aeon, aeon_classifier = _fit_classifier(
        Z_aeon_train,
        split.y_train,
        Z_aeon_test,
        alphas=alphas,
    )

    classes = np.unique(split.y_train)
    return BenchmarkResult(
        dataset=split.name,
        n_train=int(split.y_train.size),
        n_test=int(split.y_test.size),
        n_timepoints=int(split.X_train.shape[1]),
        n_classes=int(classes.size),
        requested_features=int(requested_features),
        output_features=int(Z_ir_train.shape[1]),
        repeat=int(repeat),
        irocket_fit_seconds=float(irocket_fit),
        irocket_train_transform_seconds=float(irocket_train_transform),
        irocket_test_transform_seconds=float(irocket_test_transform),
        irocket_classifier_seconds=float(irocket_classifier),
        aeon_fit_seconds=float(aeon_fit),
        aeon_train_transform_seconds=float(aeon_train_transform),
        aeon_test_transform_seconds=float(aeon_test_transform),
        aeon_classifier_seconds=float(aeon_classifier),
        irocket_accuracy=float(accuracy_score(split.y_test, prediction_ir)),
        irocket_balanced_accuracy=float(
            balanced_accuracy_score(split.y_test, prediction_ir)
        ),
        aeon_accuracy=float(accuracy_score(split.y_test, prediction_aeon)),
        aeon_balanced_accuracy=float(
            balanced_accuracy_score(split.y_test, prediction_aeon)
        ),
        prediction_agreement=float(np.mean(prediction_ir == prediction_aeon)),
    )


def print_result(result: BenchmarkResult) -> None:
    print(
        f"\n{result.dataset}: train={result.n_train}, test={result.n_test}, "
        f"timepoints={result.n_timepoints}, classes={result.n_classes}"
    )
    print(
        f"feature budget={result.requested_features}; "
        f"actual columns={result.output_features}; repeat={result.repeat}"
    )
    print("component                     I-ROCKET       aeon")
    print(
        f"fit transform parameters      {result.irocket_fit_seconds:10.4f}  "
        f"{result.aeon_fit_seconds:10.4f}"
    )
    print(
        f"transform training data       "
        f"{result.irocket_train_transform_seconds:10.4f}  "
        f"{result.aeon_train_transform_seconds:10.4f}"
    )
    print(
        f"transform test data           "
        f"{result.irocket_test_transform_seconds:10.4f}  "
        f"{result.aeon_test_transform_seconds:10.4f}"
    )
    print(
        f"scale + ridge                 "
        f"{result.irocket_classifier_seconds:10.4f}  "
        f"{result.aeon_classifier_seconds:10.4f}"
    )
    print(
        f"total measured                {result.irocket_total_seconds:10.4f}  "
        f"{result.aeon_total_seconds:10.4f}"
    )
    print(
        "balanced accuracy: "
        f"I-ROCKET={result.irocket_balanced_accuracy:.4f}, "
        f"aeon={result.aeon_balanced_accuracy:.4f}; "
        f"prediction agreement={result.prediction_agreement:.4f}"
    )


def _json_ready(result: BenchmarkResult) -> dict:
    values = asdict(result)
    values["irocket_total_seconds"] = result.irocket_total_seconds
    values["aeon_total_seconds"] = result.aeon_total_seconds
    return values


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(REFERENCE_DATASETS),
        help="Reference datasets: Waveform, GunPoint, FordB.",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=1_008,
        help="Requested MultiRocket bias-feature budget per representation.",
    )
    parser.add_argument("--max-dilations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional stratified cap for installation smoke runs.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional stratified cap for installation smoke runs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional destination for machine-readable results.",
    )
    args = parser.parse_args(argv)
    if args.features < 84:
        parser.error("--features must be at least 84.")
    if args.max_dilations < 1 or args.repeats < 1:
        parser.error("--max-dilations and --repeats must be positive.")

    _require_aeon()
    warm_implementations(args.seed)
    print("Fair I-ROCKET versus aeon MultiRocket reference benchmark")
    print(f"aeon={aeon.__version__}; numpy={np.__version__}")
    print(
        "threads: "
        + ", ".join(
            f"{name}={os.environ.get(name, 'unset')}"
            for name in (
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        )
    )
    print(f"ridge alphas={DEFAULT_ALPHAS}")

    results = []
    for dataset_name in args.datasets:
        split = load_dataset(
            dataset_name,
            random_state=args.seed,
            max_train=args.max_train,
            max_test=args.max_test,
        )
        for repeat in range(1, args.repeats + 1):
            result = run_once(
                split,
                requested_features=args.features,
                max_dilations=args.max_dilations,
                alphas=DEFAULT_ALPHAS,
                random_state=args.seed,
                repeat=repeat,
            )
            results.append(result)
            print_result(result)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "configuration": {
                "datasets": list(args.datasets),
                "features": args.features,
                "max_dilations": args.max_dilations,
                "repeats": args.repeats,
                "seed": args.seed,
                "max_train": args.max_train,
                "max_test": args.max_test,
                "ridge_alphas": list(DEFAULT_ALPHAS),
                "aeon_version": aeon.__version__,
                "numpy_version": np.__version__,
            },
            "results": [_json_ready(result) for result in results],
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
