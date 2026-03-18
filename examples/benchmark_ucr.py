"""
benchmark_ucr.py — Compare I-ROCKET vs aeon MultiRocket across UCR datasets

Runs both classifiers on the default train/test split for each dataset,
using identical parameters. Reports accuracy and timing.

Requirements:
    pip install scikit-learn aeon numpy numba

Usage:
    python benchmark_ucr.py
"""

import os
os.environ["OMP_DISPLAY_ENV"] = "FALSE"
os.environ["KMP_WARNINGS"] = "0"

# Look in the parent directory for interp_rocket
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import io
import contextlib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import balanced_accuracy_score
from aeon.datasets import load_classification
from aeon.classification.convolution_based import MultiRocketClassifier
from interp_rocket import InterpRocket


# ── Dataset selection ───────────────────────────────────────────────────────
# Selected to span a range of series lengths, number of classes, and dataset
# sizes. Includes datasets commonly used in ROCKET/MultiRocket benchmarks
# (Dempster et al., 2020; Tan et al., 2022; Uribarri et al., 2024).

DATASETS = [
    # Short series
    "ItalyPowerDemand",      # 2 classes, 24 timepoints, 1096 instances
    "GunPoint",              # 2 classes, 150 timepoints, 200 instances
    "ECG200",                # 2 classes, 96 timepoints, 200 instances
    "CBF",                   # 3 classes, 128 timepoints, 930 instances
    "SyntheticControl",      # 6 classes, 60 timepoints, 600 instances
    # Medium series
    "FordA",                 # 2 classes, 500 timepoints, 4921 instances
    "FordB",                 # 2 classes, 500 timepoints, 4446 instances
    "Wafer",                 # 2 classes, 152 timepoints, 7164 instances
    "SwedishLeaf",           # 15 classes, 128 timepoints, 1125 instances
    "FaceAll",               # 14 classes, 131 timepoints, 2250 instances
    # Longer / larger
    "ElectricDevices",       # 7 classes, 96 timepoints, 16637 instances
    "StarLightCurves",       # 3 classes, 1024 timepoints, 9236 instances
    "Crop",                  # 24 classes, 46 timepoints, 24000 instances
    "NonInvasiveFetalECGThorax1",  # 42 classes, 750 timepoints, 3765 instances
    "PhalangesOutlinesCorrect",    # 2 classes, 80 timepoints, 2658 instances
]


# ── Benchmark functions ─────────────────────────────────────────────────────

def run_interp_rocket(X_train, y_train, X_test, y_test):
    model = InterpRocket(
        max_dilations_per_kernel=32,
        num_features=10000,
        random_state=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    return metrics['accuracy'], metrics['balanced_accuracy']


def run_aeon_multirocket(X_train_3d, y_train, X_test_3d, y_test):
    clf = MultiRocketClassifier(
        max_dilations_per_kernel=32,
        n_kernels=10000,
        random_state=0,
    )
    clf.fit(X_train_3d, y_train)
    y_pred = clf.predict(X_test_3d)
    acc = float(np.mean(y_pred == y_test))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    return acc, bal_acc


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("Benchmark: I-ROCKET vs aeon MultiRocket across UCR datasets")
    print("=" * 90)

    header = (f"{'Dataset':<22s}  {'N':>5s}  {'T':>5s}  {'C':>3s}  "
              f"{'IR Acc':>7s}  {'aeon Acc':>8s}  {'Diff':>7s}  "
              f"{'IR (s)':>7s}  {'aeon (s)':>8s}  {'Ratio':>6s}")
    print(f"\n{header}")
    print("-" * len(header))

    results = []

    for name in DATASETS:
        try:
            # Load with default train/test split
            X_train_3d, y_train = load_classification(name, split="train")
            X_test_3d, y_test = load_classification(name, split="test")

            # I-ROCKET expects 2D float32
            X_train = X_train_3d.squeeze().astype(np.float32)
            X_test = X_test_3d.squeeze().astype(np.float32)
            y_train_int = y_train.astype(int) if y_train.dtype.kind in ('U', 'S', 'O') else y_train
            y_test_int = y_test.astype(int) if y_test.dtype.kind in ('U', 'S', 'O') else y_test

            n_total = len(y_train_int) + len(y_test_int)
            n_timepoints = X_train.shape[1]
            n_classes = len(np.unique(y_train_int))

            # Run I-ROCKET
            t0 = time.time()
            ir_acc, ir_bal = run_interp_rocket(X_train, y_train_int, X_test, y_test_int)
            ir_time = time.time() - t0

            # Run aeon (expects 3D and original labels)
            t0 = time.time()
            aeon_acc, aeon_bal = run_aeon_multirocket(X_train_3d, y_train, X_test_3d, y_test)
            aeon_time = time.time() - t0

            diff = ir_acc - aeon_acc
            ratio = aeon_time / ir_time if ir_time > 0 else float('inf')

            print(f"{name:<22s}  {n_total:>5d}  {n_timepoints:>5d}  {n_classes:>3d}  "
                  f"{ir_acc:>7.4f}  {aeon_acc:>8.4f}  {diff:>+7.4f}  "
                  f"{ir_time:>7.1f}  {aeon_time:>8.1f}  {ratio:>5.1f}x")

            results.append({
                'name': name, 'n': n_total, 'T': n_timepoints, 'C': n_classes,
                'ir_acc': ir_acc, 'aeon_acc': aeon_acc, 'diff': diff,
                'ir_time': ir_time, 'aeon_time': aeon_time, 'ratio': ratio,
            })

        except Exception as e:
            print(f"{name:<22s}  FAILED: {e}")

    # ── Summary ─────────────────────────────────────────────────────────
    if results:
        diffs = [r['diff'] for r in results]
        ratios = [r['ratio'] for r in results]
        ir_accs = [r['ir_acc'] for r in results]
        aeon_accs = [r['aeon_acc'] for r in results]

        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)
        print(f"\nDatasets evaluated: {len(results)}")
        print(f"I-ROCKET mean accuracy:  {np.mean(ir_accs):.4f}")
        print(f"aeon mean accuracy:      {np.mean(aeon_accs):.4f}")
        print(f"Mean accuracy difference: {np.mean(diffs):+.4f} "
              f"(range: {min(diffs):+.4f} to {max(diffs):+.4f})")
        print(f"I-ROCKET wins/ties/losses: "
              f"{sum(1 for d in diffs if d > 0.001)}/"
              f"{sum(1 for d in diffs if abs(d) <= 0.001)}/"
              f"{sum(1 for d in diffs if d < -0.001)}")
        print(f"Mean speed ratio: {np.mean(ratios):.1f}x "
              f"(range: {min(ratios):.1f}x to {max(ratios):.1f}x)")

        # Paired test — accuracy
        from scipy.stats import wilcoxon
        try:
            stat, pval = wilcoxon(ir_accs, aeon_accs)
            print(f"\nWilcoxon signed-rank test (accuracy):")
            print(f"  W={stat:.1f}, p={pval:.4f}")
            if pval > 0.05:
                print("  -> No significant difference in accuracy (p > 0.05)")
            else:
                print(f"  -> Significant difference (p = {pval:.4f})")
        except Exception as e:
            print(f"  Accuracy test skipped: {e}")

        # Paired test — speed
        ir_times_arr = np.array([r['ir_time'] for r in results])
        aeon_times_arr = np.array([r['aeon_time'] for r in results])
        try:
            stat_spd, pval_spd = wilcoxon(ir_times_arr, aeon_times_arr)
            print(f"\nWilcoxon signed-rank test (execution time):")
            print(f"  I-ROCKET mean: {ir_times_arr.mean():.2f}s +/- {ir_times_arr.std():.2f}s")
            print(f"  aeon mean:     {aeon_times_arr.mean():.2f}s +/- {aeon_times_arr.std():.2f}s")
            print(f"  W={stat_spd:.1f}, p={pval_spd:.6f}")
            if pval_spd > 0.05:
                print("  -> No significant difference in speed (p > 0.05)")
            else:
                print(f"  -> Significant difference in speed (p = {pval_spd:.6f})")
        except Exception as e:
            print(f"  Speed test skipped: {e}")


if __name__ == "__main__":
    main()
