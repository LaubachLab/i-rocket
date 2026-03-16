"""
test_interp_rocket.py — Validation tests for interp_rocket

This file serves two purposes:
    1. Code validation: Verifies that all core functions, visualizations,
       and analysis pipelines produce correct output.
    2. Installation testing: After downloading interp_rocket.py into a new
       environment, run `python test_interp_rocket.py` to confirm that all
       dependencies are installed and the module works end-to-end.

Tests use the three bumps synthetic dataset (included in three_bumps.py)
so no external data downloads are required.

Usage:
    python test_interp_rocket.py              # run all tests
    python -m pytest test_interp_rocket.py    # run via pytest (optional)
    python -m pytest test_interp_rocket.py -v # verbose output
"""

# Suppress OpenMP diagnostic messages that numba triggers on startup.
# These are informational only and do not affect results.
import os
os.environ["OMP_DISPLAY_ENV"] = "FALSE"
os.environ["KMP_WARNINGS"] = "0"

import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_test_data(n_train=300, n_test=100):
    """Generate a small three bumps dataset for fast testing."""
    from three_bumps import generate_three_bumps
    X_train, y_train = generate_three_bumps(
        n_samples=n_train, noise_std=1.0, random_state=42
    )
    X_test, y_test = generate_three_bumps(
        n_samples=n_test, noise_std=1.0, random_state=99
    )
    return X_train, y_train, X_test, y_test


def make_fitted_model(X_train, y_train):
    """Fit an InterpRocket model with small kernel count for speed."""
    import io, contextlib
    from interp_rocket import InterpRocket
    model = InterpRocket(
        max_dilations_per_kernel=32,
        num_features=1000,  # small for speed
        random_state=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_imports():
    """Verify all public functions can be imported."""
    from interp_rocket import (
        InterpRocket,
        recursive_feature_elimination,
        plot_elimination_curve,
        plot_rfe_survivors,
        cross_validate,
        temporal_occlusion,
        plot_occlusion,
        mutual_information,
    )
    from three_bumps import (
        generate_three_bumps,
        estimate_bayes_error,
        find_noise_for_bayes_error,
    )
    print("  PASS: all imports successful")


def test_data_generation():
    """Verify three_bumps generates correct shapes and classes."""
    from three_bumps import generate_three_bumps
    X, y = generate_three_bumps(n_samples=300, n_timepoints=100)
    assert X.shape == (300, 100), f"Expected (300, 100), got {X.shape}"
    assert y.shape == (300,), f"Expected (300,), got {y.shape}"
    assert set(np.unique(y)) == {0, 1, 2}, f"Expected 3 classes, got {np.unique(y)}"
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
    print("  PASS: data generation")


def test_bayes_error_estimation():
    """Verify Bayes error estimator returns reasonable values."""
    from three_bumps import estimate_bayes_error
    # At noise_std=1.0, Bayes error should be low (~3%)
    err, acc = estimate_bayes_error(noise_std=1.0, n_monte_carlo=50000)
    assert 0.0 < err < 0.10, f"Bayes error {err:.4f} out of expected range"
    assert acc > 0.90, f"Bayes accuracy {acc:.4f} too low"
    # At noise_std=3.0, should be near chance (~37%)
    err2, acc2 = estimate_bayes_error(noise_std=3.0, n_monte_carlo=50000)
    assert err2 > 0.25, f"High-noise Bayes error {err2:.4f} too low"
    print("  PASS: Bayes error estimation")


def test_fit_and_predict():
    """Verify model fits, predicts, and achieves reasonable accuracy."""
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    # Check fitted attributes
    assert model.classifier_ is not None, "Classifier not fitted"
    assert model.scaler_ is not None, "Scaler not fitted"
    n_features = model.transform(X_train[:1]).shape[1]
    assert n_features > 0, "No features generated"

    # Predict
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape, "Prediction shape mismatch"
    assert set(np.unique(y_pred)).issubset({0, 1, 2}), "Invalid class labels"

    # Accuracy should be well above chance (33%) on easy data
    acc = np.mean(y_pred == y_test)
    assert acc > 0.70, f"Accuracy {acc:.4f} too low for noise_std=1.0"
    print(f"  PASS: fit and predict (accuracy={acc:.4f})")


def test_score():
    """Verify score() returns a scalar and evaluate() returns all metrics."""
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    # score() should return a scalar float (sklearn convention)
    acc = model.score(X_test, y_test)
    assert isinstance(acc, float), f"score() should return float, got {type(acc)}"
    assert 0.0 <= acc <= 1.0, f"score() out of range: {acc}"

    # evaluate() should return a dict with all metrics
    metrics = model.evaluate(X_test, y_test)
    expected_keys = {
        'accuracy', 'balanced_accuracy', 'f1_macro',
        'f1_weighted', 'mcc', 'mutual_info'
    }
    assert set(metrics.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(metrics.keys())}"
    )
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} is not float: {type(val)}"
        assert not np.isnan(val), f"{key} is NaN"

    # score() and evaluate()['accuracy'] should agree
    assert abs(acc - metrics['accuracy']) < 1e-10, "score() and evaluate() disagree"
    print(f"  PASS: score/evaluate (acc={acc:.4f}, "
          f"mcc={metrics['mcc']:.4f})")


def test_transform():
    """Verify transform returns correct shape."""
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)
    features = model.transform(X_test)
    assert features.shape[0] == X_test.shape[0], "Wrong number of instances"
    assert features.shape[1] > 0, "No features generated"
    assert not np.any(np.isnan(features)), "NaN in feature matrix"
    print(f"  PASS: transform ({features.shape[1]} features)")


def test_decode_feature_index():
    """Verify feature decoding returns all expected fields."""
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    info = model.decode_feature_index(0)
    expected_fields = {
        'kernel_index', 'dilation', 'bias', 'pooling_op',
        'representation', 'receptive_field'
    }
    assert expected_fields.issubset(set(info.keys())), (
        f"Missing fields: {expected_fields - set(info.keys())}"
    )
    assert 0 <= info['kernel_index'] <= 83, "Kernel index out of range"
    assert info['dilation'] >= 1, "Invalid dilation"
    assert info['pooling_op'] in ('PPV', 'MPV', 'MIPV', 'LSPV'), (
        f"Unknown pooling op: {info['pooling_op']}"
    )
    assert info['representation'] in ('raw', 'diff'), (
        f"Unknown representation: {info['representation']}"
    )
    print(f"  PASS: decode_feature_index (K{info['kernel_index']}, "
          f"d={info['dilation']}, {info['pooling_op']}, {info['representation']})")


def test_feature_importance():
    """Verify feature importance is computed and shaped correctly."""
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)
    importance = model.get_feature_importance()
    n_features = model.transform(X_train[:1]).shape[1]
    assert importance.shape == (n_features,), "Wrong importance shape"
    assert np.all(importance >= 0), "Negative importance values"
    assert np.any(importance > 0), "All importances are zero"
    print(f"  PASS: feature importance (max={importance.max():.4f})")


def test_get_top_features():
    """Verify top features are decoded and sorted."""
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)
    top = model.get_top_features(n=10)
    assert len(top) == 10, f"Expected 10 features, got {len(top)}"
    # Check sorted by importance descending
    imps = [f['importance'] for f in top]
    assert imps == sorted(imps, reverse=True), "Not sorted by importance"
    print(f"  PASS: get_top_features (top importance={imps[0]:.4f})")


def test_class_weight_balanced():
    """Verify class_weight='balanced' runs without error."""
    import io, contextlib
    from interp_rocket import InterpRocket
    X_train, y_train, X_test, y_test = make_test_data()
    model = InterpRocket(num_features=500, class_weight='balanced', random_state=0)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    assert metrics['accuracy'] > 0.5, "Balanced model accuracy too low"
    print(f"  PASS: class_weight='balanced' (acc={metrics['accuracy']:.4f})")


def test_recursive_feature_elimination():
    """Verify RFE runs and returns expected structure."""
    import io, contextlib
    from interp_rocket import recursive_feature_elimination
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    with contextlib.redirect_stdout(io.StringIO()):
        rfe = recursive_feature_elimination(
            model, X_train, y_train, X_test, y_test,
            drop_percentage=0.10,  # larger steps for speed
            total_number_steps=50,
            verbose=False,
        )

    expected_keys = {
        'fractions', 'n_features', 'train_accuracies', 'test_accuracies',
        'surviving_indices', 'full_feature_ranking',
        'knee_idx', 'knee_fraction', 'knee_n_features',
        'knee_accuracy', 'peak_accuracy', 'peak_idx'
    }
    assert expected_keys.issubset(set(rfe.keys())), (
        f"Missing keys: {expected_keys - set(rfe.keys())}"
    )
    assert len(rfe['fractions']) == len(rfe['test_accuracies']), "Length mismatch"
    assert rfe['fractions'][0] == 1.0, "Should start at 100%"
    assert rfe['knee_n_features'] > 0, "Knee has zero features"
    assert rfe['knee_n_features'] <= rfe['n_features'][0], "Knee exceeds total"
    print(f"  PASS: recursive_feature_elimination "
          f"(knee={rfe['knee_n_features']} features, "
          f"acc={rfe['knee_accuracy']:.4f})")


def test_plot_elimination_curve():
    """Verify plot_elimination_curve runs without error."""
    import io, contextlib
    import matplotlib
    matplotlib.use('Agg')
    from interp_rocket import recursive_feature_elimination, plot_elimination_curve
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    with contextlib.redirect_stdout(io.StringIO()):
        rfe = recursive_feature_elimination(
            model, X_train, y_train, X_test, y_test,
            drop_percentage=0.10, total_number_steps=50, verbose=False,
        )
    fig = plot_elimination_curve(rfe)
    assert fig is not None, "No figure returned"
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  PASS: plot_elimination_curve")


def test_plot_rfe_survivors():
    """Verify plot_rfe_survivors runs at the knee index."""
    import io, contextlib
    import matplotlib
    matplotlib.use('Agg')
    from interp_rocket import recursive_feature_elimination, plot_rfe_survivors
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    with contextlib.redirect_stdout(io.StringIO()):
        rfe = recursive_feature_elimination(
            model, X_train, y_train, X_test, y_test,
            drop_percentage=0.10, total_number_steps=50, verbose=False,
        )
    fig = plot_rfe_survivors(rfe, model, step=rfe['knee_idx'])
    assert fig is not None, "No figure returned"
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  PASS: plot_rfe_survivors")


def test_visualization_methods():
    """Verify all model visualization methods run without error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    # plot_top_kernels
    fig = model.plot_top_kernels(X_test, y_test, n_kernels=2)
    assert fig is not None
    plt.close(fig)

    # plot_temporal_importance
    fig, importance = model.plot_temporal_importance(
        X_test, y_test, n_top=10, method='differential'
    )
    assert fig is not None
    assert importance is not None
    plt.close(fig)

    # plot_feature_distributions
    fig = model.plot_feature_distributions(X_test, y_test, n_top=4)
    assert fig is not None
    plt.close(fig)

    # plot_kernel_properties
    fig = model.plot_kernel_properties(n_top=20)
    assert fig is not None
    plt.close(fig)

    print("  PASS: all visualization methods")


def test_temporal_occlusion():
    """Verify temporal occlusion runs and returns expected structure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from interp_rocket import temporal_occlusion, plot_occlusion
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    occ = temporal_occlusion(model, X_test, y_test, n_samples=3)
    assert 'sensitivities' in occ, "Missing sensitivities key"
    assert 'sample_indices' in occ, "Missing sample_indices key"

    fig = plot_occlusion(occ)
    assert fig is not None
    plt.close(fig)
    print("  PASS: temporal occlusion")


def test_cross_validate():
    """Verify cross_validate runs and returns all metrics."""
    import io, contextlib
    from interp_rocket import cross_validate
    X_train, y_train, _, _ = make_test_data(n_train=200, n_test=50)

    with contextlib.redirect_stdout(io.StringIO()):
        results = cross_validate(
            X_train, y_train,
            n_repeats=2, n_folds=3,
            num_features=500,
            verbose=False,
        )

    assert 'accuracy' in results, "Missing accuracy in CV results"
    assert 'mean' in results['accuracy'], "Missing mean in accuracy"
    assert 'std' in results['accuracy'], "Missing std in accuracy"
    assert results['accuracy']['mean'] > 0.5, "CV accuracy too low"
    print(f"  PASS: cross_validate "
          f"(acc={results['accuracy']['mean']:.4f} "
          f"± {results['accuracy']['std']:.4f})")


def test_mutual_information():
    """Verify mutual information computation."""
    from interp_rocket import mutual_information
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    mi_perfect = mutual_information(y_true=y_true, y_pred=y_pred)
    assert mi_perfect > 0, "MI should be positive for perfect prediction"

    y_random = np.array([1, 2, 0, 2, 1, 0])
    mi_random = mutual_information(y_true=y_true, y_pred=y_random)
    assert mi_perfect > mi_random, "Perfect MI should exceed random MI"
    print(f"  PASS: mutual_information (perfect={mi_perfect:.4f}, "
          f"random={mi_random:.4f})")


def test_summary():
    """Verify summary() prints without error."""
    import io, contextlib
    X_train, y_train, _, _ = make_test_data()
    model = make_fitted_model(X_train, y_train)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        model.summary()
    text = output.getvalue()
    assert len(text) > 100, "Summary output too short"
    assert "features" in text.lower(), "Summary missing feature info"
    print("  PASS: summary()")


def test_feature_mask():
    """Verify feature_mask parameter works in visualization methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    X_train, y_train, X_test, y_test = make_test_data()
    model = make_fitted_model(X_train, y_train)

    # Use top 50 features as mask
    top = model.get_top_features(n=50)
    mask = np.array([f['feature_index'] for f in top])

    fig, importance = model.plot_temporal_importance(
        X_test, y_test, n_top=10, feature_mask=mask
    )
    assert fig is not None
    plt.close(fig)

    fig = model.plot_top_kernels(X_test, y_test, n_kernels=2, feature_mask=mask)
    assert fig is not None
    plt.close(fig)
    print("  PASS: feature_mask parameter")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_imports,
        test_data_generation,
        test_bayes_error_estimation,
        test_fit_and_predict,
        test_score,
        test_transform,
        test_decode_feature_index,
        test_feature_importance,
        test_get_top_features,
        test_class_weight_balanced,
        test_mutual_information,
        test_summary,
        test_visualization_methods,
        test_temporal_occlusion,
        test_recursive_feature_elimination,
        test_plot_elimination_curve,
        test_plot_rfe_survivors,
        test_feature_mask,
        test_cross_validate,
    ]

    print("=" * 60)
    print("interp_rocket validation tests")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for test_func in tests:
        name = test_func.__name__
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAIL: {name} — {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    else:
        print("\nAll tests passed!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
