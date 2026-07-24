"""Leakage-free nested-CV diagnostic on the three-bumps benchmark.

This script estimates generalization through outer cross-validation, chooses the
consensus threshold and ridge alpha inside inner cross-validation, and then fits
one final full-development model for interpretation.  The final model's training
performance is deliberately not reported as evidence of generalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path(__file__).resolve().parent / "datasets"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATASETS))

from _irocket_selection import ResampledShrinkageSelector  # noqa: E402
from interp_rocket import InterpRocketTransform  # noqa: E402
from irocket_model_selection import nested_stability_cv  # noqa: E402
from three_bumps import estimate_bayes_error, generate_three_bumps  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--timepoints", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1.5)
    parser.add_argument("--features", type=int, default=1000)
    parser.add_argument("--max-dilations", type=int, default=16)
    parser.add_argument("--resamples", type=int, default=10)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bayes-samples", type=int, default=100000)
    args = parser.parse_args()

    X, y = generate_three_bumps(
        n_samples=args.samples,
        n_timepoints=args.timepoints,
        noise_std=args.noise,
        random_state=args.seed,
    )
    bayes_error, bayes_accuracy = estimate_bayes_error(
        n_timepoints=args.timepoints,
        noise_std=args.noise,
        n_monte_carlo=args.bayes_samples,
        random_state=args.seed,
    )

    transformer = InterpRocketTransform(
        num_features=args.features,
        max_dilations_per_kernel=args.max_dilations,
        representations="both",
        random_state=args.seed,
    )
    selector = ResampledShrinkageSelector(
        n_resamples=args.resamples,
        sample_fraction=0.5,
        consensus_threshold=0.5,
        cutoff_min_size=5,
        score_power=1.0,
        min_features=2,
        random_state=None,
    )

    alpha_grid = np.logspace(-4, 4, 9)

    start = perf_counter()
    result = nested_stability_cv(
        X,
        y,
        transformer=transformer,
        selector=selector,
        outer_cv=args.outer_folds,
        inner_cv=args.inner_folds,
        consensus_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
        classifier_alphas=alpha_grid,
        scoring="balanced_accuracy",
        selection_rule="one_se",
        ridge_solver="lsqr",
        random_state=args.seed,
        refit=True,
    )
    elapsed = perf_counter() - start

    print("Configuration")
    print(
        f"samples={args.samples}, timepoints={args.timepoints}, "
        f"noise={args.noise}, requested_features={args.features}, "
        f"resamples={args.resamples}, outer={args.outer_folds}, "
        f"inner={args.inner_folds}, seed={args.seed}"
    )
    print(f"ridge_solver=lsqr, alpha_grid={alpha_grid.tolist()}")
    print(
        f"Bayes estimate: error={bayes_error:.4f}, "
        f"accuracy={bayes_accuracy:.4f} "
        f"({args.bayes_samples} Monte Carlo samples)"
    )
    print(f"Nested analysis time: {elapsed:.4f} seconds")
    print(
        "Actual transformed columns per outer model: "
        f"{[fold.n_transform_features for fold in result.outer_fold_results]}"
    )

    print("\nOuter-fold results")
    print("fold  bal_acc  threshold  alpha  features  stability  median_cutoff")
    for fold in result.outer_fold_results:
        print(
            f"{fold.fold_index + 1:4d}  "
            f"{fold.metrics['balanced_accuracy']:7.4f}  "
            f"{fold.best_consensus_threshold:9.2f}  "
            f"{fold.best_alpha:5g}  "
            f"{fold.n_selected_features:8d}  "
            f"{fold.nogueira_stability:9.4f}  "
            f"{np.median(fold.cutoff_sizes):13.1f}"
        )

    print("\nGeneralization summary")
    print(
        "balanced accuracy: "
        f"mean={result.mean_metrics['balanced_accuracy']:.4f}, "
        f"std={result.std_metrics['balanced_accuracy']:.4f}, "
        f"pooled={result.pooled_metrics['balanced_accuracy']:.4f}"
    )
    print(
        "selected features: "
        f"mean={np.mean(result.selected_counts):.1f}, "
        f"range=[{result.selected_counts.min()}, "
        f"{result.selected_counts.max()}]"
    )
    print(
        "Nogueira stability: "
        f"mean={np.mean(result.nogueira_stabilities):.4f}, "
        f"range=[{result.nogueira_stabilities.min():.4f}, "
        f"{result.nogueira_stabilities.max():.4f}]"
    )

    print("\nFinal interpretation model")
    print(
        f"threshold={result.best_parameters['consensus_threshold']:.2f}, "
        f"alpha={result.best_parameters['alpha']:g}, "
        f"features={result.final_model.n_selected_features_}, "
        f"stability={result.final_model.nogueira_stability_:.4f}"
    )
    print(
        "Final-model performance is not reported; use the outer-fold results "
        "as the generalization estimate."
    )


if __name__ == "__main__":
    main()
