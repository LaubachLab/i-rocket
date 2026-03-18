"""
benchmark_waveform.py — Compare I-ROCKET vs aeon MultiRocket

Runs 25 rounds of train/test splits on the Waveform-5000 dataset (OpenML)
using the same random splits for both classifiers:
    1. InterpRocket (interp_rocket.py)
    2. aeon MultiRocketClassifier

Both wrap MultiRocket + StandardScaler + RidgeClassifierCV internally.

Reports per-round accuracy and summary statistics.

Requirements:
    pip install scikit-learn aeon numpy numba matplotlib scipy

Usage:
    python benchmark_classifiers.py
"""

# Suppress OpenMP diagnostic messages that numba triggers on startup.
# These are informational only and do not affect results.
import os
os.environ["OMP_DISPLAY_ENV"] = "FALSE"
os.environ["KMP_WARNINGS"] = "0"

# Look in the parent directory for interp_rocket
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── Load Waveform-5000 Data ────────────────────────────────────────────────

def load_waveform_data():
    """
    Load the Waveform-5000 dataset from OpenML via scikit-learn.

    The dataset has 40 attributes: 21 signal timepoints + 19 noise features.
    We use only the 21 signal attributes for time series classification.

    Returns (X, y) as 2D float32 arrays.
    """
    from sklearn.datasets import fetch_openml

    waveform = fetch_openml(name='waveform-5000', version=1, as_frame=False)
    X = waveform.data[:, :21].astype(np.float32)  # 21 signal timepoints
    y = waveform.target.astype(int)

    return X, y


# ── Benchmark Functions ─────────────────────────────────────────────────────

def run_interp_rocket(X_train, y_train, X_test, y_test):
    """Fit and score InterpRocket."""
    import io, contextlib
    from interp_rocket import InterpRocket

    model = InterpRocket(
        max_dilations_per_kernel=32,
        num_features=10000,
        random_state=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    return metrics['accuracy'], metrics['balanced_accuracy']


def run_aeon_multirocket(X_train, y_train, X_test, y_test):
    """
    Fit and score aeon's MultiRocketClassifier.

    aeon's MultiRocketClassifier wraps MultiRocket + StandardScaler +
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).
    Input must be 3D: (n_instances, n_channels, n_timepoints).
    """
    from aeon.classification.convolution_based import MultiRocketClassifier

    # aeon expects 3D input
    X_train_3d = X_train[:, np.newaxis, :]
    X_test_3d = X_test[:, np.newaxis, :]

    clf = MultiRocketClassifier(
        n_kernels=10000,
        max_dilations_per_kernel=32,
        random_state=0,
    )
    clf.fit(X_train_3d, y_train)

    y_pred = clf.predict(X_test_3d)
    acc = float(np.mean(y_pred == y_test))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    return acc, bal_acc


# ── Main Benchmark ──────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Benchmark: I-ROCKET vs aeon MultiRocket")
    print("Dataset: Waveform-5000 (OpenML, 21 signal timepoints, 3 classes)")
    print("=" * 72)

    print("\nLoading Waveform-5000 data from OpenML...")
    X, y = load_waveform_data()
    print(f"Data: {X.shape[0]} instances x {X.shape[1]} timepoints")
    print(f"Classes: {np.unique(y)}, counts: {np.bincount(y)}")

    n_rounds = 25
    splitter = StratifiedShuffleSplit(
        n_splits=n_rounds, test_size=0.3, random_state=42
    )

    classifiers = {
        "interp_rocket": run_interp_rocket,
        "aeon":          run_aeon_multirocket,
    }

    results = {name: {"acc": [], "bal_acc": [], "time": []}
               for name in classifiers}

    print(f"\nRunning {n_rounds} rounds with identical train/test splits...\n")

    header = (f"{'Round':>5s}  "
              + "  ".join(f"{'[' + name + ']':>20s}" for name in classifiers))
    print(header)
    print("-" * len(header))

    for round_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        row = f"{round_idx+1:>5d}  "
        for name, func in classifiers.items():
            try:
                t0 = time.time()
                acc, bal_acc = func(X_train, y_train, X_test, y_test)
                elapsed = time.time() - t0
                results[name]["acc"].append(acc)
                results[name]["bal_acc"].append(bal_acc)
                results[name]["time"].append(elapsed)
                row += f"  {acc:.4f} ({elapsed:5.1f}s)      "
            except Exception as e:
                row += f"  FAILED: {str(e)[:15]}  "
                results[name]["acc"].append(np.nan)
                results[name]["bal_acc"].append(np.nan)
                results[name]["time"].append(np.nan)

        print(row)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n{'Classifier':>20s}  {'Accuracy':>18s}  {'Bal. Accuracy':>18s}  {'Time/round':>12s}")
    print(f"{'':>20s}  {'(mean +/- std)':>18s}  {'(mean +/- std)':>18s}  {'(mean)':>12s}")
    print("-" * 72)

    for name in classifiers:
        accs = np.array(results[name]["acc"])
        bal_accs = np.array(results[name]["bal_acc"])
        times = np.array(results[name]["time"])
        valid = ~np.isnan(accs)

        if valid.any():
            print(f"{name:>20s}  "
                  f"{accs[valid].mean():.4f} +/- {accs[valid].std():.4f}  "
                  f"{bal_accs[valid].mean():.4f} +/- {bal_accs[valid].std():.4f}  "
                  f"{times[valid].mean():>10.1f}s")
        else:
            print(f"{name:>20s}  {'FAILED':>18s}  {'FAILED':>18s}  {'N/A':>12s}")

    # ── Pairwise comparison ─────────────────────────────────────────────
    print("\nPairwise accuracy differences (interp_rocket - aeon):")
    ir_accs = np.array(results["interp_rocket"]["acc"])
    aeon_accs = np.array(results["aeon"]["acc"])
    valid = ~(np.isnan(ir_accs) | np.isnan(aeon_accs))
    if valid.any():
        diffs = ir_accs[valid] - aeon_accs[valid]
        print(f"  Mean: {diffs.mean():+.4f} +/- {diffs.std():.4f}  "
              f"(range: {diffs.min():+.4f} to {diffs.max():+.4f})")

    # ── Timing comparison ───────────────────────────────────────────────
    ir_times = np.array(results["interp_rocket"]["time"])
    aeon_times = np.array(results["aeon"]["time"])
    valid_t = ~(np.isnan(ir_times) | np.isnan(aeon_times))
    if valid_t.any():
        ratio = aeon_times[valid_t].mean() / ir_times[valid_t].mean()
        print(f"\nSpeed: interp_rocket is {ratio:.1f}x faster than aeon MultiRocket")

    # ── Statistical tests ───────────────────────────────────────────────
    from scipy.stats import wilcoxon

    print("\n" + "=" * 72)
    print("STATISTICAL TESTS (Wilcoxon signed-rank, paired)")
    print("=" * 72)

    # Accuracy test
    valid_acc = ~(np.isnan(ir_accs) | np.isnan(aeon_accs))
    if valid_acc.sum() >= 6:
        diffs_acc = ir_accs[valid_acc] - aeon_accs[valid_acc]
        try:
            stat_acc, p_acc = wilcoxon(ir_accs[valid_acc], aeon_accs[valid_acc])
            print(f"\nAccuracy:")
            print(f"  I-ROCKET mean: {ir_accs[valid_acc].mean():.4f} +/- {ir_accs[valid_acc].std():.4f}")
            print(f"  aeon mean:     {aeon_accs[valid_acc].mean():.4f} +/- {aeon_accs[valid_acc].std():.4f}")
            print(f"  Mean diff:     {diffs_acc.mean():+.4f}")
            print(f"  Wilcoxon W={stat_acc:.1f}, p={p_acc:.4f}")
            if p_acc > 0.05:
                print(f"  -> No significant difference in accuracy (p = {p_acc:.4f})")
            else:
                print(f"  -> Significant difference in accuracy (p = {p_acc:.4f})")
        except ValueError as e:
            print(f"\n  Accuracy test skipped: {e}")

    # Speed test
    valid_spd = ~(np.isnan(ir_times) | np.isnan(aeon_times))
    if valid_spd.sum() >= 6:
        try:
            stat_spd, p_spd = wilcoxon(ir_times[valid_spd], aeon_times[valid_spd])
            print(f"\nExecution time:")
            print(f"  I-ROCKET mean: {ir_times[valid_spd].mean():.2f}s +/- {ir_times[valid_spd].std():.2f}s")
            print(f"  aeon mean:     {aeon_times[valid_spd].mean():.2f}s +/- {aeon_times[valid_spd].std():.2f}s")
            print(f"  Speed ratio:   {aeon_times[valid_spd].mean() / ir_times[valid_spd].mean():.1f}x")
            print(f"  Wilcoxon W={stat_spd:.1f}, p={p_spd:.6f}")
            if p_spd > 0.05:
                print(f"  -> No significant difference in speed (p = {p_spd:.6f})")
            else:
                print(f"  -> Significant difference in speed (p = {p_spd:.6f})")
        except ValueError as e:
            print(f"\n  Speed test skipped: {e}")

    # ── Notes ───────────────────────────────────────────────────────────
    print("\nNotes:")
    print("  Data source: OpenML waveform-5000 (21 signal timepoints, "
          "19 noise features excluded)")
    print("  RidgeClassifierCV alpha ranges:")
    print("    interp_rocket: alphas=np.logspace(-10, 10, 20)")
    print("    aeon:          alphas=np.logspace(-3, 3, 10)")


if __name__ == "__main__":
    main()
