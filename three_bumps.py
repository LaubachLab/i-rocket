"""
three_bumps.py — Parametric benchmark dataset for time series classification

Generates a 3-class "three bumps" dataset where each class is defined by
a Gaussian bump at a distinct temporal position, embedded in Gaussian noise.
The difficulty (Bayes error rate) is controlled by the signal-to-noise ratio.

Unlike the Breiman waveform data (where classes are random mixtures of basis
functions), the three bumps design places each class's discriminative signal
at a known, non-overlapping time region. This makes it ideal for validating
temporal interpretability methods: a correct temporal importance map must
peak at the bump location for each class.

The Bayes-optimal classifier for this problem is the maximum-likelihood
Gaussian classifier, and the Bayes error can be estimated by Monte Carlo
simulation using the built-in estimate_bayes_error() function.

Reference
---------
This dataset was designed for validation of I-ROCKET (interp_rocket),
a reimplementation of MultiRocket with kernel-level interpretability.

Usage
-----
    from three_bumps import generate_three_bumps, estimate_bayes_error

    # Easy problem
    X, y = generate_three_bumps(noise_std=1.1)

    # Hard problem
    X, y = generate_three_bumps(noise_std=1.5)

    # Custom: unequal class amplitudes, wider bumps
    X, y = generate_three_bumps(amplitudes=[1.0, 0.7, 0.4], sigma=8)

    # Find noise level for a target Bayes error
    noise = find_noise_for_bayes_error(target_error=0.15)
    X, y = generate_three_bumps(noise_std=noise)

    # Estimate Bayes error for any parameter combination
    err, acc = estimate_bayes_error(noise_std=1.5)
"""

import numpy as np


def generate_three_bumps(
    n_samples=5000,
    n_timepoints=100,
    centers=(25, 50, 75),
    amplitudes=None,
    sigma=5.0,
    noise_std=1.0,
    random_state=42,
):
    """
    Generate a 3-class time series dataset with Gaussian bumps.

    Each class has a Gaussian bump centered at a different temporal position.
    The bump amplitude, width, and background noise level control the
    difficulty of the classification problem.

    Parameters
    ----------
    n_samples : int, default=5000
        Total number of instances (split roughly equally across classes).
    n_timepoints : int, default=100
        Length of each time series.
    centers : tuple of float, default=(25, 50, 75)
        Temporal center of the bump for each class. Length determines the
        number of classes (default 3).
    amplitudes : array-like or None, default=None
        Peak amplitude of the bump for each class. If None, all classes
        have amplitude 1.0. If a scalar, all classes share that amplitude.
        If array-like, must have one entry per class.
    sigma : float, default=5.0
        Standard deviation (width) of the Gaussian bump in timepoints.
        Larger values produce broader, more overlapping bumps.
    noise_std : float, default=1.0
        Standard deviation of the additive Gaussian noise. This is the
        primary difficulty control.
    random_state : int or None, default=42
        Seed for reproducibility.

    Returns
    -------
    X : ndarray, shape (n_samples, n_timepoints), dtype float32
        Time series data.
    y : ndarray, shape (n_samples,), dtype int
        Class labels (0, 1, 2, ...).
    """
    rng = np.random.default_rng(random_state)
    n_classes = len(centers)

    # Handle amplitudes
    if amplitudes is None:
        amplitudes = np.ones(n_classes)
    elif np.isscalar(amplitudes):
        amplitudes = np.full(n_classes, float(amplitudes))
    else:
        amplitudes = np.asarray(amplitudes, dtype=float)
        if len(amplitudes) != n_classes:
            raise ValueError(
                f"amplitudes has {len(amplitudes)} entries but "
                f"centers has {n_classes} classes."
            )

    t = np.arange(n_timepoints, dtype=np.float64)

    # Build mean signal for each class
    means = np.zeros((n_classes, n_timepoints))
    for k in range(n_classes):
        means[k] = amplitudes[k] * np.exp(
            -0.5 * ((t - centers[k]) / sigma) ** 2
        )

    # Generate data
    y = rng.integers(0, n_classes, size=n_samples)
    X = np.zeros((n_samples, n_timepoints), dtype=np.float32)
    for i in range(n_samples):
        X[i] = means[y[i]] + noise_std * rng.standard_normal(n_timepoints)

    return X, y


def estimate_bayes_error(
    n_timepoints=100,
    centers=(25, 50, 75),
    amplitudes=None,
    sigma=5.0,
    noise_std=1.0,
    n_monte_carlo=500000,
    random_state=42,
):
    """
    Estimate the Bayes error rate for the three bumps problem.

    Uses Monte Carlo simulation of the Bayes-optimal (maximum likelihood)
    classifier. With equal priors and known Gaussian noise, this is the
    minimum achievable error rate for any classifier.

    Parameters
    ----------
    n_timepoints, centers, amplitudes, sigma, noise_std :
        Same as generate_three_bumps().
    n_monte_carlo : int, default=500000
        Number of Monte Carlo samples. Standard error of the Bayes error
        estimate is approximately sqrt(err * (1-err) / n_monte_carlo).
    random_state : int or None, default=42
        Seed for reproducibility.

    Returns
    -------
    bayes_error : float
        Estimated Bayes error rate.
    bayes_accuracy : float
        Estimated Bayes accuracy (1 - bayes_error).
    """
    rng = np.random.default_rng(random_state)
    n_classes = len(centers)

    if amplitudes is None:
        amplitudes = np.ones(n_classes)
    elif np.isscalar(amplitudes):
        amplitudes = np.full(n_classes, float(amplitudes))
    else:
        amplitudes = np.asarray(amplitudes, dtype=float)

    t = np.arange(n_timepoints, dtype=np.float64)
    means = np.zeros((n_classes, n_timepoints))
    for k in range(n_classes):
        means[k] = amplitudes[k] * np.exp(
            -0.5 * ((t - centers[k]) / sigma) ** 2
        )

    # Precompute squared norms for log-likelihood
    # log P(x|class k) = -0.5 * ||x - mu_k||^2 / noise_std^2 + const
    # The constant and leading terms cancel in argmax, so we need:
    #   argmax_k [ x . mu_k - 0.5 * ||mu_k||^2 ] (sufficient statistic)
    mean_sq_norms = 0.5 * np.sum(means ** 2, axis=1)  # shape (n_classes,)

    # Generate all samples at once for speed
    true_classes = rng.integers(0, n_classes, size=n_monte_carlo)
    noise = noise_std * rng.standard_normal(
        (n_monte_carlo, n_timepoints)
    )

    n_correct = 0
    for i in range(n_monte_carlo):
        x = means[true_classes[i]] + noise[i]
        # Sufficient statistic: x . mu_k - 0.5 * ||mu_k||^2
        scores = x @ means.T - mean_sq_norms
        if np.argmax(scores) == true_classes[i]:
            n_correct += 1

    bayes_accuracy = n_correct / n_monte_carlo
    bayes_error = 1.0 - bayes_accuracy

    return bayes_error, bayes_accuracy


def find_noise_for_bayes_error(
    target_error,
    n_timepoints=100,
    centers=(25, 50, 75),
    amplitudes=None,
    sigma=5.0,
    n_monte_carlo=300000,
    tol=0.005,
    random_state=42,
    verbose=True,
):
    """
    Find the noise_std that produces a target Bayes error rate.

    Uses binary search over noise_std with Monte Carlo estimation at
    each step.

    Parameters
    ----------
    target_error : float
        Desired Bayes error rate (e.g., 0.14 for ~86% accuracy).
    n_timepoints, centers, amplitudes, sigma :
        Same as generate_three_bumps().
    n_monte_carlo : int, default=300000
        Monte Carlo samples per evaluation.
    tol : float, default=0.005
        Acceptable tolerance on the Bayes error estimate.
    random_state : int or None, default=42
        Seed for reproducibility.
    verbose : bool, default=True
        Print the search progress.

    Returns
    -------
    noise_std : float
        The noise standard deviation that achieves the target error.
    actual_error : float
        The Monte Carlo estimate of the Bayes error at the returned noise_std.
    actual_accuracy : float
        1 - actual_error.
    """
    lo, hi = 0.01, 20.0

    if verbose:
        print(
            f"Searching for noise_std with Bayes error ≈ {target_error:.3f} "
            f"(accuracy ≈ {1 - target_error:.3f})..."
        )

    best_noise = None
    best_err = None

    for iteration in range(25):
        mid = (lo + hi) / 2
        err, acc = estimate_bayes_error(
            n_timepoints=n_timepoints,
            centers=centers,
            amplitudes=amplitudes,
            sigma=sigma,
            noise_std=mid,
            n_monte_carlo=n_monte_carlo,
            random_state=random_state,
        )

        if verbose:
            print(
                f"  iter {iteration + 1:2d}: noise_std={mid:.4f}, "
                f"Bayes error={err:.4f}, accuracy={acc:.4f}"
            )

        best_noise = mid
        best_err = err

        if abs(err - target_error) < tol:
            break

        if err < target_error:
            lo = mid  # need more noise
        else:
            hi = mid  # need less noise

    if verbose:
        print(
            f"\n  Result: noise_std={best_noise:.4f} → "
            f"Bayes error={best_err:.4f}, "
            f"accuracy={1 - best_err:.4f}"
        )

    return best_noise, best_err, 1.0 - best_err


def print_calibration_table(
    amplitudes=None,
    sigma=5.0,
    centers=(25, 50, 75),
    n_timepoints=100,
    n_monte_carlo=300000,
):
    """
    Print a calibration table of noise_std vs. Bayes error.

    Useful for choosing parameters for benchmark experiments.
    """
    print(f"\nThree Bumps Calibration Table")
    print(f"  centers={centers}, sigma={sigma}, amplitudes={amplitudes}")
    print(f"  n_timepoints={n_timepoints}")
    print(f"\n{'noise_std':>10s}  {'Bayes Err':>10s}  {'Bayes Acc':>10s}  {'Difficulty':>12s}")
    print("-" * 48)

    noise_values = [0.5, 0.8, 1.0, 1.1, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]

    for noise in noise_values:
        err, acc = estimate_bayes_error(
            n_timepoints=n_timepoints,
            centers=centers,
            amplitudes=amplitudes,
            sigma=sigma,
            noise_std=noise,
            n_monte_carlo=n_monte_carlo,
        )

        if err < 0.02:
            difficulty = "trivial"
        elif err < 0.08:
            difficulty = "easy"
        elif err < 0.15:
            difficulty = "moderate"
        elif err < 0.25:
            difficulty = "hard"
        elif err < 0.33:
            difficulty = "very hard"
        else:
            difficulty = "near chance"

        print(f"{noise:>10.2f}  {err:>10.4f}  {acc:>10.4f}  {difficulty:>12s}")


# ── Main: print calibration tables if run as script ─────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Three Bumps Dataset — Calibration Tables")
    print("=" * 60)

    # Default parameters
    print_calibration_table()

    # Wider bumps (more overlap → harder)
    print_calibration_table(sigma=8.0)

    # Unequal amplitudes
    print_calibration_table(amplitudes=[1.0, 0.7, 0.4])

    # Example: find noise for specific Bayes error
    print("\n" + "=" * 60)
    print("Example: finding noise_std for ~14% Bayes error")
    print("(matching Breiman waveform difficulty)")
    print("=" * 60)
    noise, err, acc = find_noise_for_bayes_error(0.14)
    print(f"\nTo generate data at this difficulty:")
    print(f"  X, y = generate_three_bumps(noise_std={noise:.3f})")
