"""Development benchmark for resampled shrinkage selection on three bumps.

This script is not a nested-CV performance estimate. It uses one held-out split
and reports every predeclared consensus threshold rather than choosing a
threshold from the test set. Its purpose is to diagnose the selector mechanics
and expected sparse/localized-dataset behavior independently of the nested
model-selection layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path(__file__).resolve().parent / "datasets"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATASETS))

from _irocket_selection import ResampledShrinkageSelector  # noqa: E402
from interp_rocket import InterpRocketTransform  # noqa: E402
from three_bumps import generate_three_bumps  # noqa: E402


def score_subset(Z_train, y_train, Z_test, y_test, indices):
    scaler = StandardScaler(with_mean=False)
    train = scaler.fit_transform(Z_train[:, indices])
    test = scaler.transform(Z_test[:, indices])
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(train, y_train)
    prediction = classifier.predict(test)
    return float(balanced_accuracy_score(y_test, prediction)), classifier.alpha_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--timepoints", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1.5)
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--max-dilations", type=int, default=16)
    parser.add_argument("--resamples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

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
    transform = InterpRocketTransform(
        num_features=args.features,
        max_dilations_per_kernel=args.max_dilations,
        representations="both",
        random_state=args.seed,
    ).fit(X_train)
    Z_train = transform.transform(X_train)
    Z_test = transform.transform(X_test)
    transform_seconds = perf_counter() - start

    start = perf_counter()
    selector = ResampledShrinkageSelector(
        n_resamples=args.resamples,
        sample_fraction=0.5,
        consensus_threshold=0.5,
        cutoff_min_size=5,
        score_power=1.0,
        random_state=args.seed,
    ).fit(Z_train, y_train)
    selection_seconds = perf_counter() - start

    all_indices = np.arange(Z_train.shape[1])
    full_accuracy, full_alpha = score_subset(
        Z_train, y_train, Z_test, y_test, all_indices
    )

    print("Configuration")
    print(
        f"samples={args.samples}, timepoints={args.timepoints}, noise={args.noise}, "
        f"requested_features={args.features}, resamples={args.resamples}, "
        f"seed={args.seed}"
    )
    print(f"transformed shape: train={Z_train.shape}, test={Z_test.shape}")
    print(f"transform + apply: {transform_seconds:.4f} seconds")
    print(f"resampled selection: {selection_seconds:.4f} seconds")
    print(
        "cutoff sizes: "
        f"min={selector.cutoff_sizes_.min()}, "
        f"median={np.median(selector.cutoff_sizes_):.1f}, "
        f"max={selector.cutoff_sizes_.max()}"
    )
    print(
        "breakpoint fit improvement: "
        f"median={np.median(selector.cutoff_improvements_):.4f}, "
        f"range=[{selector.cutoff_improvements_.min():.4f}, "
        f"{selector.cutoff_improvements_.max():.4f}]"
    )
    print(f"Nogueira stability: {selector.nogueira_stability_:.4f}")

    print("\nConsensus threshold results")
    print("threshold  features  balanced_accuracy  ridge_alpha")
    print(f"full       {len(all_indices):8d}  {full_accuracy:17.4f}  {full_alpha:g}")
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ranking = selector.consensus_ranking_
        mask = selector.selection_probabilities_[ranking] >= threshold
        indices = ranking[mask]
        accuracy, alpha = score_subset(
            Z_train, y_train, Z_test, y_test, indices
        )
        print(f"{threshold:0.1f}       {len(indices):8d}  {accuracy:17.4f}  {alpha:g}")


if __name__ == "__main__":
    main()
