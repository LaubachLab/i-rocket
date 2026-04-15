"""
lfp_tools_emd.py - EMD-based functions for LFP analysis.

Provides data-driven spectral analysis based on Empirical Mode
Decomposition (EMD). 

EMD decomposes signals into intrinsic mode functions (IMFs) without
imposing frequency boundaries. Each IMF captures a single oscillatory
mode whose frequency and amplitude can vary freely over time. This
avoids the resolution tradeoffs of windowed Fourier and wavelet methods
and the spectral leakage of bandpass filtering.

Depends on the emd package (Quinn et al., 2021, JOSS). We recommend
the mask sift for neural data because it reduces mode mixing. Install:
    pip install emd

This set of functions will be expanded in a forthcoming release to include
additional tools for complimentary non-Fourier analyses of LFP and EEG data.

License: BSD 2-Clause

References
----------
Huang, N. E., et al. (1998). The empirical mode decomposition and the
    Hilbert spectrum for nonlinear and non-stationary time series
    analysis. Proceedings of the Royal Society A, 454, 903-995.

Quinn, A. J., Lopes-dos-Santos, V., Dupret, D., Nobre, A. C., &
    Woolrich, M. W. (2021). EMD: Empirical Mode Decomposition and
    Hilbert-Huang Spectral Analyses in Python. Journal of Open Source
    Software, 6(59), 2977.

Cole, S. R., & Voytek, B. (2017). Brain oscillations and the importance
    of waveform shape. Trends in Cognitive Sciences, 21(2), 137-149.

Kopell, N., Whittington, M. A., & Kramer, M. A. (2011). Neuronal
    assembly dynamics in the beta1 frequency range permits short-term
    memory. PNAS, 108(9), 3779-3784.

Fabus, M. S., Woolrich, M. W., Sherwood, W., & Quinn, A. J. (2021).
    Automatic decomposition of electrophysiological data into distinct
    non-sinusoidal oscillatory modes. Journal of Neurophysiology,
    126(5), 1670-1684.
"""

import numpy as np

try:
    import emd
    _HAS_EMD = True
except ImportError:
    _HAS_EMD = False

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _check_emd():
    """Raise ImportError if emd is not installed."""
    if not _HAS_EMD:
        raise ImportError(
            "The emd package is required. Install with: pip install emd"
        )


def _pbar(iterable, total=None, desc='', verbose=True):
    """Progress bar wrapper."""
    if not verbose:
        return iterable
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc, leave=True)
    return iterable


def decompose_trial(signal, srate, max_imfs=8, method='mask_sift',
                    **sift_kwargs):
    """Run EMD on a single trial and return IMFs with metadata."""
    _check_emd()

    if method == 'iterated_mask_sift':
        imfs = emd.sift.iterated_mask_sift(signal, max_imfs=max_imfs,
                                            **sift_kwargs)
    else:
        imfs = emd.sift.mask_sift(signal, max_imfs=max_imfs,
                                   **sift_kwargs)

    IP, IF, IA = emd.spectra.frequency_transform(imfs, srate, 'hilbert')

    freqs = np.array([np.nanmedian(IF[:, i]) for i in range(imfs.shape[1])])

    return {
        'imfs': imfs,
        'IP': IP,
        'IF': IF,
        'IA': IA,
        'freqs': freqs,
        'n_imfs': imfs.shape[1],
    }


def marginal_spectrum(X, srate, freq_edges=None, max_imfs=8,
                      method='mask_sift', n_trials=None, verbose=True,
                      **sift_kwargs):
    """Compute the EMD marginal spectrum for a set of trials."""
    _check_emd()

    if freq_edges is None:
        freq_edges = np.linspace(0, 100, 101)
    freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2
    n_freqs = len(freq_centers)

    if n_trials is None:
        n_trials = X.shape[0]

    spectrum = np.zeros(n_freqs)

    for tr in _pbar(range(n_trials), desc='Marginal spectrum',
                    verbose=verbose):
        result = decompose_trial(X[tr], srate, max_imfs=max_imfs,
                                  method=method, **sift_kwargs)
        for imf_i in range(result['n_imfs']):
            for t in range(result['IF'].shape[0]):
                f_bin = np.searchsorted(freq_edges, result['IF'][t, imf_i]) - 1
                if 0 <= f_bin < n_freqs:
                    spectrum[f_bin] += result['IA'][t, imf_i]

    spectrum /= n_trials

    return freq_centers, spectrum


def matched_marginal_spectrum(hi_reps, lo_reps, srate, freq_edges=None,
                               max_imfs=8, method='mask_sift',
                               n_reps=None, verbose=True, **sift_kwargs):
    """Compute matched-trial EMD marginal spectra for two conditions."""
    if freq_edges is None:
        freq_edges = np.linspace(0, 100, 101)
    if n_reps is None:
        n_reps = len(hi_reps)

    freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2
    n_freqs = len(freq_centers)

    spec_hi_all = np.zeros((n_reps, n_freqs))
    spec_lo_all = np.zeros((n_reps, n_freqs))

    for rep in range(n_reps):
        if verbose:
            print(f"  Rep {rep + 1}/{n_reps}")
        _, spec_hi_all[rep] = marginal_spectrum(
            hi_reps[rep], srate, freq_edges=freq_edges,
            max_imfs=max_imfs, method=method, verbose=False,
            **sift_kwargs
        )
        _, spec_lo_all[rep] = marginal_spectrum(
            lo_reps[rep], srate, freq_edges=freq_edges,
            max_imfs=max_imfs, method=method, verbose=False,
            **sift_kwargs
        )

    spec_hi = spec_hi_all.mean(axis=0)
    spec_lo = spec_lo_all.mean(axis=0)

    return freq_centers, spec_hi, spec_lo, spec_hi_all, spec_lo_all
