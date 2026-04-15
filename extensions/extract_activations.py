"""
extract_activations.py - Extract pre-pooling activation time series from I-ROCKET.

Uses model.decode_feature_index() to get kernel weights, dilation, bias,
and representation type for each selected feature, then performs the
dilated convolution retaining the full temporal activation.

Output is an activation_dict suitable for activation_tfa.activation_tfa().

Usage:
    from extract_activations import get_activation_dict, get_kernel_activation

    # Default: one entry per unique kernel-dilation-representation
    activation_dict, kernel_meta = get_activation_dict(
        model, X, srate=1000.0,
        selected_features=stable_features
    )

    # Access K17 at dilation 10, raw representation
    act_k17 = activation_dict[('raw', 17, 10)]

    # Quick access to a single kernel without feature selection
    act, meta = get_kernel_activation(model, X, srate=1000.0,
                                      kernel_index=17, dilation=10)

    # Optional: group by frequency (averages across kernels)
    activation_dict, kernel_meta = get_activation_dict(
        model, X, srate=1000.0,
        selected_features=stable_features,
        group_by='frequency'
    )
"""

import numpy as np


def _dilated_convolution(X_in, weights, dilation, bias, n_timepoints, n_trials):
    """Vectorized dilated convolution across all trials and channels.

    Computes sum(x[t + j*d] * w[j]) + bias for each timepoint t.

    Parameters
    ----------
    X_in : ndarray, shape (n_trials, n_channels, n_timepoints_in)
        Input data. For diff representations, n_timepoints_in may be
        n_timepoints - 1. The output is zero-padded and centered to
        align with the original n_timepoints axis. Note that diff
        introduces a 0.5-sample shift (diff[t] corresponds to the
        midpoint between x[t] and x[t+1]); at typical sampling rates
        (>= 500 Hz) this is negligible.
    weights : ndarray, shape (k_len,)
    dilation : int
    bias : float
        Additive bias. Use 0.0 for raw kernel-signal similarity (default
        in get_activation_dict). Use a feature-specific bias to reproduce
        the thresholded activation that drives a specific pooling operation.
    n_timepoints : int
        Output length (original input length before diff).
    n_trials : int

    Returns
    -------
    act : ndarray, shape (n_timepoints, n_trials)
    """
    k_len = len(weights)
    d = int(dilation)
    rf = 1 + (k_len - 1) * d
    pad = rf // 2
    n_channels = X_in.shape[1]

    act = np.zeros((n_timepoints, n_trials))

    for ch in range(n_channels):
        x_all = X_in[:, ch, :]  # (n_trials, in_timepoints)

        x_padded = np.pad(x_all, ((0, 0), (pad, pad)),
                          mode='constant', constant_values=0)

        conv_len = x_padded.shape[1] - rf + 1

        # Gather indices for dilated convolution
        t_idx = np.arange(conv_len)[:, np.newaxis]    # (conv_len, 1)
        j_idx = np.arange(k_len)[np.newaxis, :] * d   # (1, k_len)
        gather_idx = t_idx + j_idx                     # (conv_len, k_len)

        # Gather samples and dot with kernel weights
        gathered = x_padded[:, gather_idx]
        conv_out = gathered @ weights + bias  # (n_trials, conv_len)

        # Align to n_timepoints
        if conv_len >= n_timepoints:
            start = (conv_len - n_timepoints) // 2
            act += conv_out[:, start:start + n_timepoints].T
        else:
            start = (n_timepoints - conv_len) // 2
            act[start:start + conv_len, :] += conv_out.T

    return act


def get_activation_dict(model, X, srate, selected_features,
                        group_by='kernel', freq_precision=1,
                        verbose=True):
    """Extract pre-pooling activation time series from a fitted I-ROCKET.

    Performs dilated convolution using kernel parameters retrieved via
    model.decode_feature_index(), returning full temporal activations
    instead of pooled scalars.

    Parameters
    ----------
    model : InterpROCKET
        A fitted I-ROCKET model with decode_feature_index() method.
    X : ndarray, shape (n_trials, 1, n_timepoints) or (n_trials, n_timepoints)
        Input data. If 2D, a channel dimension is added.
    srate : float
        Sampling rate in Hz.
    selected_features : list of int
        Feature indices (from FSA, RFE, get_top_features, etc.).
    group_by : str
        'kernel' - one entry per unique kernel-dilation-representation
                   combination. Keys are tuples: (rep, kernel_idx, dilation).
                   Default.
        'frequency' - average activations across kernels at the same
                      frequency. Keys are floats (Hz).
    freq_precision : int
        Decimal places for rounding frequency when grouping. Default 1.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    activation_dict : dict
        If group_by='kernel':
            Keys are (rep, kernel_idx, dilation) tuples.
            Values are (n_timepoints, n_trials) arrays.
        If group_by='frequency':
            Keys are frequencies (Hz).
            Values are (n_timepoints, n_trials) arrays (averaged across
            kernels at that frequency).

        Note: output shape is (n_timepoints, n_trials), matching the
        convention expected by hilbert_tfa (frames, trials). To index
        by trial as with the original X, transpose: act.T gives
        (n_trials, n_timepoints).
    kernel_meta : list of dict
        Per-unique-kernel metadata from decode_feature_index, plus
        computed 'frequency' and 'label' fields. Order matches the
        iteration order of unique configurations.
    """
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    n_trials, n_channels, n_timepoints = X.shape

    if n_channels > 1:
        import warnings
        warnings.warn(
            f"X has {n_channels} channels. Activations are summed across "
            f"channels. This method is designed for univariate time series "
            f"(n_channels=1). For multi-channel data, extract activations "
            f"per channel separately.",
            UserWarning
        )

    # --- Decode selected features to unique kernel configurations ---
    unique_configs = {}
    for fi in selected_features:
        decoded = model.decode_feature_index(fi)
        config_key = (
            decoded['representation'],
            int(decoded['kernel_index']),
            int(decoded['dilation'])
        )
        if config_key not in unique_configs:
            unique_configs[config_key] = decoded

    if verbose:
        n_raw = sum(1 for k in unique_configs if k[0] == 'raw')
        n_diff = sum(1 for k in unique_configs if k[0] == 'diff')
        print(f"Selected features map to {len(unique_configs)} unique "
              f"kernel configurations ({n_raw} raw, {n_diff} diff)")

    # --- Precompute diff(X) if needed ---
    has_diff = any(k[0] == 'diff' for k in unique_configs)
    X_diff = np.diff(X, axis=2) if has_diff else None

    # --- Convolve each unique configuration ---
    kernel_meta = []
    kernel_activations = {}

    for ci, (config_key, decoded) in enumerate(unique_configs.items()):
        rep, k_idx, dilation = config_key
        weights = np.array(decoded['kernel_weights'], dtype=np.float64)
        k_len = len(weights)
        d = int(dilation)
        rf = int(decoded['receptive_field'])

        # Use bias=0 for raw activation (kernel-signal similarity).
        # The feature-specific bias is a threshold for pooling operations
        # (PPV, LSPV, etc.) and adds an arbitrary DC offset that varies
        # across features from the same kernel. Store it in metadata for
        # users who need pooling-specific thresholding.
        bias = 0.0
        feature_bias = float(decoded['bias'])

        freq = srate / ((k_len - 1) * d) if (k_len - 1) * d > 0 else srate / 2
        freq = round(freq, freq_precision)

        label = f"K{k_idx}d{d}"
        if rep == 'diff':
            label += '_diff'

        X_in = X if rep == 'raw' else X_diff

        act = _dilated_convolution(X_in, weights, d, bias,
                                   n_timepoints, n_trials)

        meta = {
            'representation': rep,
            'kernel_index': int(k_idx),
            'dilation': d,
            'frequency': freq,
            'kernel_length': k_len,
            'receptive_field': rf,
            'bias': bias,
            'feature_bias_example': feature_bias,
            'label': label,
            'config_key': config_key,
        }
        kernel_meta.append(meta)
        kernel_activations[config_key] = act

        if verbose and (ci + 1) % 20 == 0:
            print(f"  {ci + 1}/{len(unique_configs)} done")

    if verbose:
        print(f"  Done. {len(unique_configs)} configurations extracted.")

    # --- Return based on group_by ---
    if group_by == 'kernel':
        if verbose:
            print(f"  Returning {len(kernel_activations)} kernel activations:")
            for meta in sorted(kernel_meta, key=lambda m: m['frequency']):
                print(f"    {meta['label']:15s}  {meta['frequency']:6.1f} Hz  "
                      f"RF={meta['receptive_field']}")

        return kernel_activations, kernel_meta

    elif group_by == 'frequency':
        from collections import defaultdict
        freq_groups = defaultdict(list)
        freq_labels = defaultdict(list)
        for meta in kernel_meta:
            freq_groups[meta['frequency']].append(
                kernel_activations[meta['config_key']]
            )
            freq_labels[meta['frequency']].append(meta['label'])

        activation_dict = {}
        for freq in sorted(freq_groups.keys()):
            activation_dict[freq] = np.mean(freq_groups[freq], axis=0)

        if verbose:
            print(f"  Grouped into {len(activation_dict)} frequency bins:")
            for freq in sorted(activation_dict.keys()):
                n_k = len(freq_groups[freq])
                labels = freq_labels[freq]
                print(f"    {freq:.1f} Hz: {n_k} kernels "
                      f"({', '.join(labels)})")

        return activation_dict, kernel_meta

    else:
        raise ValueError(f"group_by must be 'kernel' or 'frequency', "
                         f"got '{group_by}'")


def get_kernel_activation(model, X, srate, kernel_index, dilation,
                          representation='raw', bias=0.0, verbose=True):
    """Extract activation for a single specific kernel.

    Convenience function when you know exactly which kernel you want,
    without needing feature selection results.

    Parameters
    ----------
    model : InterpROCKET
        A fitted I-ROCKET model.
    X : ndarray, shape (n_trials, 1, n_timepoints) or (n_trials, n_timepoints)
        Input data.
    srate : float
        Sampling rate in Hz.
    kernel_index : int
        Kernel index (e.g., 17 for K17).
    dilation : int
        Dilation value (e.g., 10).
    representation : str
        'raw' or 'diff'. Default 'raw'.
    bias : float
        Additive bias for the convolution. Default 0.0 (raw kernel-signal
        similarity). Set to a feature-specific bias from
        model.decode_feature_index() to reproduce pooling thresholds.
    verbose : bool
        Print info. Default True.

    Returns
    -------
    activation : ndarray, shape (n_timepoints, n_trials)
        Pre-pooling activation time series. This shape matches the
        convention expected by hilbert_tfa (frames, trials). To index
        by trial as with the original X, transpose: activation.T gives
        (n_trials, n_timepoints).
    meta : dict
        Kernel metadata including frequency, receptive field, label, etc.
    """
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    n_trials, n_channels, n_timepoints = X.shape

    if n_channels > 1:
        import warnings
        warnings.warn(
            f"X has {n_channels} channels. Activations are summed across "
            f"channels. Pass single-channel data for correct results.",
            UserWarning
        )

    weights = model.base_kernels_[kernel_index].astype(np.float64)
    k_len = len(weights)
    d = int(dilation)
    rf = 1 + (k_len - 1) * d
    freq = srate / ((k_len - 1) * d) if (k_len - 1) * d > 0 else srate / 2

    # Select input
    X_in = np.diff(X, axis=2) if representation == 'diff' else X

    act = _dilated_convolution(X_in, weights, d, bias, n_timepoints, n_trials)

    label = f"K{kernel_index}d{d}"
    if representation == 'diff':
        label += '_diff'

    meta = {
        'representation': representation,
        'kernel_index': kernel_index,
        'dilation': d,
        'frequency': round(freq, 1),
        'kernel_length': k_len,
        'receptive_field': rf,
        'bias': bias,
        'label': label,
        'config_key': (representation, kernel_index, d),
    }

    if verbose:
        print(f"{label}: {freq:.1f} Hz, RF={rf}, bias={bias:.4f}")

    return act, meta
