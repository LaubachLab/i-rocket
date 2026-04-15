"""
lfp_tools.py - Companion functions for LFP analysis with I-ROCKET.

Includes trial matching, time-frequency analysis (Hilbert TFA),
trial-matched PSD and PAC wrappers for tensorpac, and plotting utilities.

The hilbert_tfa code is a Python port of the HilbertEEG toolbox by
Kimberly Stachenfeld (2012), developed as a senior thesis project in
the Laubach Lab at American University.

A Python port of EEGLAB's eegfilt.m is included (Delorme and Makeig, 2004).

Code for Pairwise Phase Consistency (PPC) is based on Vinck et al. (2010).

This module depends on functions from the `tensorpac` package. We recommend
installing the package from this fork: https://github.com/LaubachLab/tensorpac
It applied three compatibility patches to utils.py for Python 3.12 / NumPy
>=1.22 / Matplotlib >=3.8. Install the package as follows:
pip install git+https://github.com/LaubachLab/tensorpac.git

License: BSD 2-Clause

References
----------
Combrisson, E., Nest, T., Brovelli, A., Ince, R. A., Soto, J. L., Guillot,
    A., & Jerbi, K. (2020). Tensorpac: An open-source Python toolbox for
    tensor-based phase-amplitude coupling measurement in electrophysiological
    brain signals. PLoS computational biology, 16(10), e1008302.

Delorme, A., Makeig, S. (2004) EEGLAB: an open source toolbox for analysis
    of single-trial EEG dynamics including independent component analysis.
    Journal of Neuroscience Methods, 134(1), 9-21.

Stachenfeld, K. (2012). HilbertEEG: Analyze EEG or LFP data via Hilbert
    Transform. Laubach Lab, American University.

Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., &
    Pennartz, C.M.A. (2010). The pairwise phase consistency: A bias-free
    measure of rhythmic neuronal synchronization. NeuroImage, 51(1),
    112-122.
"""

import numpy as np
from scipy.signal import hilbert, firwin, filtfilt, firls, firwin, filtfilt, lfilter
from scipy.stats import norm

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def _pbar(iterable, total=None, desc='', verbose=True):
    """Progress bar wrapper. Uses tqdm if available, else a simple fallback."""
    if not verbose:
        return iterable
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc, leave=True)
    else:
        if total is None:
            iterable = list(iterable)
            total = len(iterable)
        class _SimplePbar:
            def __init__(self, it, total, desc):
                self._it = iter(it)
                self._total = total
                self._desc = desc
                self._n = 0
            def __iter__(self):
                return self
            def __next__(self):
                val = next(self._it)
                self._n += 1
                pct = self._n * 100 // self._total
                filled = pct * 30 // 100
                bar = '#' * filled + '-' * (30 - filled)
                print(f'\r{self._desc}: {pct:3d}%|{bar}| {self._n}/{self._total}',
                      end='', flush=True)
                if self._n == self._total:
                    print()
                return val
        return _SimplePbar(iterable, total, desc)


# ---------------------------------------------------------------------------
# eegfilt (Python port of EEGLab's eegfilt function)
# ---------------------------------------------------------------------------

def eegfilt(data, srate, locutoff=0.0, hicutoff=0.0, epochframes=0,
            filtorder=0, revfilt=False, firtype='fir1', causal=False,
            verbose=True):
    """(High|low|band)-pass filter data using FIR filtering.

    Attempt to match the behavior of EEGLAB's eegfilt.m. By default uses
    fir1 (window method) filter design with zero-phase (filtfilt) application.

    Parameters
    ----------
    data : ndarray, shape (n_samples,) or (n_channels, n_samples)
        Data to filter. If 1D, treated as a single channel.
    srate : float
        Sampling rate in Hz.
    locutoff : float
        Low-edge frequency of the pass band in Hz. Set to 0 for lowpass only.
    hicutoff : float
        High-edge frequency of the pass band in Hz. Set to 0 for highpass only.
    epochframes : int
        Frames per epoch. Each epoch is filtered separately. Default 0 means
        the entire signal is one epoch.
    filtorder : int
        Filter order (length in points). Default 0 uses the EEGLAB heuristic:
        3 * fix(srate / locutoff) for highpass/bandpass, or
        3 * fix(srate / hicutoff) for lowpass, with a minimum of 15.
    revfilt : bool
        If True, reverse the filter (bandpass becomes notch, etc.). Only
        supported with firtype='firls'. Default False.
    firtype : str
        Filter design method: 'fir1' (window method, recommended) or 'firls'
        (least-squares). Default is 'fir1'.
    causal : bool
        If True, use causal (one-pass) filtering. Default False (zero-phase).
    verbose : bool
        Print filter information. Default True.

    Returns
    -------
    smoothdata : ndarray, same shape as data
        Filtered data.
    filtwts : ndarray
        FIR filter coefficients.
    """
    # Handle 1D input
    squeeze_output = False
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze_output = True

    chans, frames = data.shape
    nyq = srate * 0.5

    minfac = 3
    min_filtorder = 15
    trans = 0.15  # fractional width of transition zones

    # --- Input validation ---
    if locutoff > 0 and hicutoff > 0 and locutoff > hicutoff:
        raise ValueError("locutoff > hicutoff")
    if locutoff < 0 or hicutoff < 0:
        raise ValueError("locutoff and hicutoff must be >= 0")
    if locutoff > nyq:
        raise ValueError("Low cutoff frequency cannot be > srate/2")
    if hicutoff > nyq:
        raise ValueError("High cutoff frequency cannot be > srate/2")
    if locutoff == 0 and hicutoff == 0:
        raise ValueError("You must provide a non-zero low or high cutoff frequency")

    # --- Determine filter order ---
    if filtorder == 0:
        if locutoff > 0:
            filtorder = minfac * int(srate / locutoff)
        elif hicutoff > 0:
            filtorder = minfac * int(srate / hicutoff)
        filtorder = max(filtorder, min_filtorder)

    # --- Epoch handling ---
    if epochframes == 0:
        epochframes = frames
    epochs = frames // epochframes
    if epochs * epochframes != frames:
        raise ValueError("epochframes does not evenly divide the number of frames")
    if filtorder * 3 > epochframes:
        raise ValueError(
            f"Filter order ({filtorder}) is too large for epoch length "
            f"({epochframes}). epochframes must be >= 3 * filtorder."
        )

    # --- Design filter ---
    if locutoff > 0 and hicutoff > 0:
        # Bandpass (or notch if revfilt)
        if (1 + trans) * hicutoff / nyq > 1:
            raise ValueError("High cutoff frequency too close to Nyquist frequency")

        filter_desc = "notch" if revfilt else "bandpass"
        if verbose:
            print(f"eegfilt() - performing {filtorder}-point {filter_desc} filtering.")

        if firtype == 'firls':
            f = [0,
                 (1 - trans) * locutoff / nyq,
                 locutoff / nyq,
                 hicutoff / nyq,
                 (1 + trans) * hicutoff / nyq,
                 1.0]
            m = [0, 0, 1, 1, 0, 0]
            if revfilt:
                m = [1 - x for x in m]
            if verbose:
                lo_trans = (f[2] - f[1]) * nyq
                hi_trans = (f[4] - f[3]) * nyq
                print(f"  Low transition band width: {lo_trans:.1f} Hz; "
                      f"high transition band width: {hi_trans:.1f} Hz")
            filtwts = firls(filtorder, f, m)
        else:
            if revfilt:
                raise ValueError("Cannot reverse filter using 'fir1' option")
            # firwin bandpass needs odd number of taps (even filtorder)
            fo = filtorder if filtorder % 2 == 0 else filtorder + 1
            filtwts = firwin(fo + 1, [locutoff, hicutoff],
                             pass_zero=False, fs=srate)

    elif locutoff > 0:
        # Highpass
        if verbose:
            print(f"eegfilt() - performing {filtorder}-point highpass filtering.")

        if firtype == 'firls':
            f = [0,
                 locutoff * (1 - trans) / nyq,
                 locutoff / nyq,
                 1.0]
            m = [0, 0, 1, 1]
            if revfilt:
                m = [1 - x for x in m]
            if verbose:
                hp_trans = (f[2] - f[1]) * nyq
                print(f"  Highpass transition band width: {hp_trans:.1f} Hz")
            filtwts = firls(filtorder, f, m)
        else:
            if revfilt:
                raise ValueError("Cannot reverse filter using 'fir1' option")
            # firwin highpass needs odd number of taps (even filtorder)
            fo = filtorder if filtorder % 2 == 0 else filtorder + 1
            filtwts = firwin(fo + 1, locutoff, pass_zero=False, fs=srate)

    else:
        # Lowpass
        if verbose:
            print(f"eegfilt() - performing {filtorder}-point lowpass filtering.")

        if firtype == 'firls':
            f = [0,
                 hicutoff / nyq,
                 hicutoff * (1 + trans) / nyq,
                 1.0]
            m = [1, 1, 0, 0]
            if revfilt:
                m = [1 - x for x in m]
            if verbose:
                lp_trans = (f[2] - f[1]) * nyq
                print(f"  Lowpass transition band width: {lp_trans:.1f} Hz")
            filtwts = firls(filtorder, f, m)
        else:
            if revfilt:
                raise ValueError("Cannot reverse filter using 'fir1' option")
            filtwts = firwin(filtorder + 1, hicutoff, pass_zero=True, fs=srate)

    # --- Apply filter ---
    smoothdata = np.zeros_like(data, dtype=float)
    for e in range(epochs):
        start = e * epochframes
        stop = (e + 1) * epochframes
        for c in range(chans):
            segment = data[c, start:stop].astype(float)
            if causal:
                smoothdata[c, start:stop] = lfilter(filtwts, 1.0, segment)
            else:
                smoothdata[c, start:stop] = filtfilt(filtwts, 1.0, segment)

    if squeeze_output:
        smoothdata = smoothdata[0]

    return smoothdata, filtwts

# ---------------------------------------------------------------------------
# Trial matching
# ---------------------------------------------------------------------------

def match_trials(x_hi, x_lo, n_reps=20, frac=0.8, random_state=42):
    """Subsample both conditions to matched size.

    Both conditions are subsampled on every repetition. The matched
    trial count is frac * min(n_hi, n_lo). This means the smaller
    condition is also subsampled, so variance estimates reflect genuine
    resampling variability.

    Parameters
    ----------
    x_hi, x_lo : ndarray, shape (n_trials, n_times)
        Epoched LFP for each condition.
    n_reps : int
        Number of subsampling repetitions.
    frac : float
        Fraction of the smaller condition to sample (default 0.8).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    hi_reps, lo_reps : list of ndarray
        Each list has `n_reps` arrays of shape (n_match, n_times).
    n_match : int
        Number of matched trials per rep.
    """
    rng = np.random.default_rng(random_state)
    n_hi, n_lo = x_hi.shape[0], x_lo.shape[0]
    n_match = int(min(n_hi, n_lo) * frac)
    hi_reps, lo_reps = [], []
    for _ in range(n_reps):
        idx_hi = rng.choice(n_hi, size=n_match, replace=False)
        idx_lo = rng.choice(n_lo, size=n_match, replace=False)
        hi_reps.append(x_hi[idx_hi])
        lo_reps.append(x_lo[idx_lo])
    return hi_reps, lo_reps, n_match


# ---------------------------------------------------------------------------
# Trial-matched PSD
# ---------------------------------------------------------------------------

def matched_psd(hi_reps, lo_reps, srate, n_reps=None):
    """Compute PSD for each matched-trial repetition.

    Parameters
    ----------
    hi_reps, lo_reps : list of ndarray
        Output from match_trials. Each element has shape (n_match, n_times).
    srate : float
        Sampling rate in Hz.
    n_reps : int or None
        Number of reps to use. Default None uses all.

    Returns
    -------
    psd_hi_list, psd_lo_list : list of tensorpac PSD objects
        One PSD object per repetition.
    """
    from tensorpac.utils import PSD

    if n_reps is None:
        n_reps = len(hi_reps)

    psd_hi_list = []
    psd_lo_list = []
    for i in range(n_reps):
        psd_hi_list.append(PSD(hi_reps[i], srate))
        psd_lo_list.append(PSD(lo_reps[i], srate))

    return psd_hi_list, psd_lo_list


# ---------------------------------------------------------------------------
# Trial-matched PAC
# ---------------------------------------------------------------------------

def matched_pac(hi_reps, lo_reps, srate, f_pha, f_amp, idpac=(2, 0, 0),
                n_reps=None, verbose=True):
    """Compute PAC comodulograms for each matched-trial repetition.

    Parameters
    ----------
    hi_reps, lo_reps : list of ndarray
        Output from match_trials.
    srate : float
        Sampling rate in Hz.
    f_pha : array-like
        Phase frequencies for Pac object.
    f_amp : array-like
        Amplitude frequencies for Pac object.
    idpac : tuple
        PAC method identifier for tensorpac. Default (2, 0, 0) is
        modulation index (Tort et al., 2010).
    n_reps : int or None
        Number of reps to use. Default None uses all.
    verbose : bool
        Print progress.

    Returns
    -------
    pac_hi : ndarray, shape (n_reps, n_amp, n_pha)
        PAC comodulograms for high-value condition.
    pac_lo : ndarray, shape (n_reps, n_amp, n_pha)
        PAC comodulograms for low-value condition.
    p : tensorpac Pac object
        The Pac object (for plotting with p.comodulogram).
    """
    from tensorpac import Pac

    if n_reps is None:
        n_reps = len(hi_reps)

    p = Pac(idpac=idpac, f_pha=f_pha, f_amp=f_amp)

    pac_hi_list = []
    pac_lo_list = []

    iterator = range(n_reps)
    if verbose and _HAS_TQDM:
        iterator = _tqdm(iterator, desc='Matched PAC')

    for i in iterator:
        pac_hi_list.append(
            p.filterfit(srate, hi_reps[i], n_perm=0).mean(-1)
        )
        pac_lo_list.append(
            p.filterfit(srate, lo_reps[i], n_perm=0).mean(-1)
        )

    pac_hi = np.stack(pac_hi_list)
    pac_lo = np.stack(pac_lo_list)

    return pac_hi, pac_lo, p


# ---------------------------------------------------------------------------
# Trial-matched TFA
# ---------------------------------------------------------------------------

def matched_tfa(hi_reps, lo_reps, srate, freq_range, freq_int,
                bootstrap=False, n_reps=None, verbose=True, **kwargs):
    """Run hilbert_tfa on each matched-trial repetition.

    Parameters
    ----------
    hi_reps, lo_reps : list of ndarray
        Output from match_trials. Each element has shape (n_match, n_times).
    srate : float
        Sampling rate in Hz.
    freq_range : tuple of (float, float)
        (min_freq, max_freq) in Hz.
    freq_int : float
        Width of each frequency bin in Hz.
    bootstrap : bool
        Run bootstrap significance analysis. Default False.
    n_reps : int or None
        Number of reps to use. Default None uses all.
    verbose : bool
        Print progress.
    **kwargs
        Additional keyword arguments passed to hilbert_tfa.

    Returns
    -------
    tfa_hi_list, tfa_lo_list : list of dict
        hilbert_tfa results for each repetition.
    """
    if n_reps is None:
        n_reps = len(hi_reps)

    tfa_hi_list = []
    tfa_lo_list = []

    for i in range(n_reps):
        if verbose:
            print(f"\n--- Matched TFA rep {i+1}/{n_reps} ---")
        tfa_hi_list.append(
            hilbert_tfa(hi_reps[i].T, srate, freq_range, freq_int,
                        bootstrap=bootstrap, verbose=verbose, **kwargs)
        )
        tfa_lo_list.append(
            hilbert_tfa(lo_reps[i].T, srate, freq_range, freq_int,
                        bootstrap=bootstrap, verbose=verbose, **kwargs)
        )

    return tfa_hi_list, tfa_lo_list


# ---------------------------------------------------------------------------
# Hilbert TFA
# ---------------------------------------------------------------------------

def hilbert_tfa(data, srate, freq_range, freq_int, bootstrap=True,
                naccu=200, n_timebins=None, timebin_size=None,
                edge_frac=8, verbose=True):
    """Hilbert transform-based time-frequency analysis.

    Bandpass filters data at successive frequency intervals, applies the
    Hilbert transform, then computes ERSP, ITC, PPC, and bootstrap
    significance thresholds.

    Parameters
    ----------
    data : ndarray
        Input data. Accepted shapes:
        - (n_channels, frames, n_trials)
        - (frames, n_trials) -- treated as single channel
    srate : float
        Sampling rate in Hz.
    freq_range : tuple of (float, float)
        (min_freq, max_freq) in Hz.
    freq_int : float
        Width of each frequency bin in Hz.
    bootstrap : bool
        Run bootstrap significance analysis. Default True.
    naccu : int
        Number of bootstrap accumulations. Default 200.
    edge_frac : int
        Denominator for edge trimming: edge_frames = min(max_filtorder,
        frames // edge_frac). Default 8. Set to 4 for aggressive trimming
        or higher values (e.g. 12) to preserve more of the time series.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    results : dict with keys:
        'ITC'          : ndarray (n_channels, n_freq, frames) -- complex ITC
        'PPC'          : ndarray (n_channels, n_freq, frames) -- PPC values
        'ERSP'         : ndarray (n_channels, n_freq, frames) -- mean power
        'freqs'        : ndarray (n_freq + 1,) -- frequency bin edges
        'freq_centers' : ndarray (n_freq,) -- frequency bin centers
        'nchan'        : int
        'frames'       : int
        'ntrial'       : int
        'srate'        : float
        'ersp_thresh'  : ndarray or None -- ERSP bootstrap threshold
        'itc_thresh'   : ndarray or None -- ITC bootstrap threshold
        'edge_frames'  : int -- filter edge artifact width in frames
    """
    # --- Handle input dimensions ---
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    if data.ndim != 3:
        raise ValueError("Data must be 2D (frames, trials) or "
                         "3D (channels, frames, trials)")

    nchan, frames, ntrial = data.shape
    data = data.astype(float)

    # Frequency bins
    freqs = np.arange(freq_range[0], freq_range[1] + freq_int * 0.5, freq_int)
    nfreq = len(freqs) - 1

    if verbose:
        print(f"HilbertTFA: {nchan} ch, {frames} frames, {ntrial} trials, "
              f"{freqs[0]}-{freqs[-1]} Hz ({nfreq} bins)")

    # --- Bandpass filter + Hilbert transform ---
    all_hf = np.zeros((nchan, nfreq, frames, ntrial), dtype=complex)

    tf_steps = [(ch, f) for ch in range(nchan) for f in range(nfreq)]

    for ch, f in _pbar(tf_steps, desc='TF analysis', verbose=verbose):
        locutoff = freqs[f]
        hicutoff = freqs[f + 1]

        filtorder = min(frames // 3, 3 * int(srate / locutoff))
        max_for_filtfilt = (frames // 3) - 2
        filtorder = min(filtorder, max_for_filtfilt)
        if filtorder % 2 != 0:
            filtorder -= 1
        filtorder = max(filtorder, 16)

        filtwts = firwin(filtorder + 1, [locutoff, hicutoff],
                         pass_zero=False, fs=srate)

        filtered = filtfilt(filtwts, 1.0, data[ch, :, :], axis=0)
        all_hf[ch, f, :, :] = hilbert(filtered, axis=0)

    # --- ERSP: mean power across trials ---
    ersp = np.mean(np.abs(all_hf) ** 2, axis=3)

    # --- ITC: mean of unit phasors across trials ---
    unit_phasors = all_hf / np.abs(all_hf)
    unit_phasors = np.nan_to_num(unit_phasors, nan=0.0)
    itc = np.mean(unit_phasors, axis=3)

    # --- PPC: Pairwise Phase Consistency (Vinck et al. 2010) ---
    itc_mag2 = np.abs(itc) ** 2
    ppc = (ntrial * itc_mag2 - 1) / max(ntrial - 1, 1)

    # --- Bootstrap ---
    ersp_thresh = None
    itc_thresh = None

    if bootstrap:
        ersp_thresh, itc_thresh = _bootstrap_analysis(
            all_hf, ersp, itc, naccu, nchan, nfreq, ntrial, frames,
            verbose=verbose
        )

    freq_centers = (freqs[:-1] + freqs[1:]) / 2.0

    # Edge trim: longest filter order used, capped by edge_frac
    max_filtorder = 3 * int(srate / freq_range[0])
    edge_frames = min(max_filtorder, frames // edge_frac)

    results = {
        'ITC': itc,
        'PPC': ppc,
        'ERSP': ersp,
        'freqs': freqs,
        'freq_centers': freq_centers,
        'nchan': nchan,
        'frames': frames,
        'ntrial': ntrial,
        'srate': srate,
        'ersp_thresh': ersp_thresh,
        'itc_thresh': itc_thresh,
        'edge_frames': edge_frames,
    }

    if verbose:
        print("\nDone. ERSP, ITC, and PPC computed.")

    return results


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_analysis(hf_data, ersp_data, itc_data, naccu, nchan, nfreq,
                        ntrial, frames, verbose=True):
    """Bootstrap resampling for ERSP and ITC/PPC significance thresholds.

    Returns
    -------
    ersp_thresh : ndarray (nchan, nfreq)
        95th percentile of bootstrap ERSP distribution per frequency.
    itc_thresh : ndarray (nchan, nfreq)
        95th percentile of bootstrap ITC distribution per frequency.
    """
    all_amps = np.abs(hf_data)
    all_power = all_amps ** 2
    all_unit_real = np.zeros_like(all_amps)
    all_unit_imag = np.zeros_like(all_amps)
    nonzero = all_amps > 0
    all_unit_real[nonzero] = hf_data.real[nonzero] / all_amps[nonzero]
    all_unit_imag[nonzero] = hf_data.imag[nonzero] / all_amps[nonzero]

    ersp_thresh = np.zeros((nchan, nfreq))
    itc_thresh = np.zeros((nchan, nfreq))

    if _HAS_NUMBA:
        if verbose:
            print("  (using numba-accelerated bootstrap)")
        _run_bootstrap = _bootstrap_numba
    else:
        _run_bootstrap = _bootstrap_numpy

    boot_steps = [(ch, f) for ch in range(nchan) for f in range(nfreq)]

    for ch, f in _pbar(boot_steps, desc='Bootstrap', verbose=verbose):
        unit_real_cf = np.ascontiguousarray(all_unit_real[ch, f])
        unit_imag_cf = np.ascontiguousarray(all_unit_imag[ch, f])
        power_cf = np.ascontiguousarray(all_power[ch, f])

        temp_itc, temp_ersp = _run_bootstrap(
            unit_real_cf, unit_imag_cf, power_cf, naccu, frames, ntrial
        )

        itc_thresh[ch, f] = np.percentile(temp_itc, 95)

        ersp_db_boot = 10.0 * np.log10(temp_ersp + 1e-20)
        ersp_db_mean = ersp_db_boot.mean(axis=1, keepdims=True)
        ersp_db_dev = np.abs(ersp_db_boot - ersp_db_mean)
        ersp_thresh[ch, f] = np.percentile(ersp_db_dev, 95)

    return ersp_thresh, itc_thresh


def _bootstrap_numpy(unit_real, unit_imag, power, naccu, frames, ntrial):
    """Numpy-vectorized bootstrap accumulation (fallback)."""
    temp_itc = np.zeros((frames, naccu))
    temp_ersp = np.zeros((frames, naccu))

    for n in range(naccu):
        t = np.random.randint(0, ntrial, size=ntrial)
        re = unit_real[:, t]
        im = unit_imag[:, t]
        mean_re = np.mean(re, axis=1)
        mean_im = np.mean(im, axis=1)
        temp_itc[:, n] = np.sqrt(mean_re ** 2 + mean_im ** 2)
        temp_ersp[:, n] = np.mean(power[:, t], axis=1)

    return temp_itc, temp_ersp


if _HAS_NUMBA:
    @njit(fastmath=True, parallel=True, cache=True)
    def _bootstrap_numba(unit_real, unit_imag, power, naccu, frames, ntrial):
        """Numba-JIT bootstrap accumulation with parallel accumulations."""
        temp_itc = np.zeros((frames, naccu))
        temp_ersp = np.zeros((frames, naccu))

        for n in prange(naccu):
            t_idx = np.random.randint(0, ntrial, ntrial)
            for fr in range(frames):
                sum_re = 0.0
                sum_im = 0.0
                sum_pow = 0.0
                for ti in range(ntrial):
                    tr = t_idx[ti]
                    sum_re += unit_real[fr, tr]
                    sum_im += unit_imag[fr, tr]
                    sum_pow += power[fr, tr]
                inv_n = 1.0 / ntrial
                mean_re = sum_re * inv_n
                mean_im = sum_im * inv_n
                temp_itc[fr, n] = np.sqrt(mean_re * mean_re + mean_im * mean_im)
                temp_ersp[fr, n] = sum_pow * inv_n

        return temp_itc, temp_ersp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tfa(results, times=None, channel=0, alpha=0.05,
             baseline='zscore', show=('ersp', 'ppc'), trim_edges=True,
             interpolation='bilinear', ersp_cmap='RdBu_r', itc_cmap='Reds',
             ppc_cmap='Reds', figsize=None,
             ersp_vlim=None, itc_vmax=0.75, ppc_vmax=0.15):
    """Plot ERSP, PPC, and/or ITC from hilbert_tfa results.

    Parameters
    ----------
    results : dict
        Output from hilbert_tfa().
    times : ndarray or None
        Time vector in ms. Default creates one centered at 0.
    channel : int
        Channel index to plot. Default 0.
    alpha : float
        Significance level for bootstrap masking. Default 0.05.
    baseline : tuple of (float, float), 'zscore', or None
        Baseline correction for ERSP:
        - tuple (start_ms, end_ms): dB change relative to mean power in
          the specified time window (per frequency).
        - 'zscore': Global z-score across all frequencies and times.
          Preserves cross-frequency amplitude relationships.
        - None: dB relative to temporal mean at each frequency.
        Default 'zscore'.
    show : tuple of str
        Panels to display: any combination of 'ersp', 'ppc', 'itc'.
        Default ('ersp', 'ppc').
    trim_edges : bool
        Trim filter edge artifacts. Default True.
    interpolation : str or None
        Interpolation method for imshow. Default 'bilinear'.
    ersp_cmap : str or colormap
        Colormap for ERSP. Default 'RdBu_r'.
    itc_cmap : str or colormap
        Colormap for ITC. Default 'Reds'.
    ppc_cmap : str or colormap
        Colormap for PPC. Default 'Reds'.
    figsize : tuple or None
        Figure size. Default auto-scaled to number of panels.
    ersp_vlim : tuple of (float, float) or None
        (vmin, vmax) for ERSP colorbar. Default None (auto).
    itc_vmax : float
        Maximum for ITC colorbar. Default 0.75.
    ppc_vmax : float
        Maximum for PPC colorbar. Default 0.15.

    Returns
    -------
    fig : matplotlib Figure
    plot_data : dict
        Data arrays for each panel.
    """
    import matplotlib.pyplot as plt

    itc = results['ITC']
    ppc = results['PPC']
    ersp = results['ERSP']
    freq_centers = results['freq_centers']
    frames = results['frames']

    if times is None:
        times = np.arange(frames) - frames // 2

    # --- Resolve colormaps ---
    ersp_cmap = _resolve_cmap(ersp_cmap)
    itc_cmap = _resolve_cmap(itc_cmap)
    ppc_cmap = _resolve_cmap(ppc_cmap)

    # --- Edge trimming ---
    edge = results.get('edge_frames', 0) if trim_edges else 0
    t_slice = slice(edge, frames - edge) if edge > 0 else slice(None)
    times_trimmed = times[t_slice] if edge > 0 else times

    # --- Prepare ERSP data ---
    ersp_ch = ersp[channel].copy()
    ersp_db = 10.0 * np.log10(ersp_ch + 1e-20)

    if baseline == 'zscore':
        global_mean = np.mean(ersp_db)
        global_std = np.std(ersp_db)
        if global_std > 0:
            ersp_db = (ersp_db - global_mean) / global_std
        else:
            ersp_db = ersp_db - global_mean
    elif baseline is not None:
        bl_start, bl_end = baseline
        bl_mask = (times >= bl_start) & (times <= bl_end)
        if bl_mask.sum() > 0:
            bl_mean = ersp_db[:, bl_mask].mean(axis=1, keepdims=True)
            ersp_db = ersp_db - bl_mean
    else:
        temporal_mean = ersp_db.mean(axis=1, keepdims=True)
        ersp_db = ersp_db - temporal_mean

    ersp_db_plot = ersp_db[:, t_slice]

    # --- Prepare ITC data ---
    itc_amp = np.abs(itc[channel]).copy()
    if results['itc_thresh'] is not None:
        thresh = results['itc_thresh'][channel]
        mask = itc_amp < thresh[:, np.newaxis]
        itc_amp[mask] = 0.0
    itc_amp_plot = itc_amp[:, t_slice]

    # --- Prepare PPC data ---
    ppc_vals = ppc[channel].copy()
    ppc_vals = np.clip(ppc_vals, 0, None)
    if results['itc_thresh'] is not None:
        n = results['ntrial']
        itc_t = results['itc_thresh'][channel]
        ppc_thresh = (n * itc_t ** 2 - 1) / max(n - 1, 1)
        mask = ppc_vals < ppc_thresh[:, np.newaxis]
        ppc_vals[mask] = 0.0
    ppc_plot = ppc_vals[:, t_slice]

    # --- Build plot_data return dict ---
    plot_data = {
        'ersp_db': ersp_db,
        'itc_amp': itc_amp,
        'ppc': ppc_vals,
        'times': times,
        'times_trimmed': times_trimmed,
        'freq_centers': freq_centers,
    }

    # --- Determine panels ---
    if isinstance(show, str):
        show = (show,)
    panels = [s for s in show if s in ('ersp', 'ppc', 'itc')]

    n_plots = len(panels)
    if n_plots == 0:
        return None, plot_data

    if figsize is None:
        figsize = (12, 3.5 * n_plots)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
    axes = axes[:, 0]

    plot_idx = 0

    for panel in panels:
        ax = axes[plot_idx]

        if panel == 'ersp':
            if ersp_vlim is not None:
                e_vmin, e_vmax = ersp_vlim
            elif baseline == 'zscore':
                e_vmin, e_vmax = -2, 2
            else:
                e_vmax = np.percentile(np.abs(ersp_db_plot), 97)
                e_vmin = -e_vmax
            if baseline == 'zscore':
                ersp_title = 'ERSP (z-score)'
                cbar_label = 'z'
            elif baseline is not None:
                ersp_title = f'ERSP (dB re baseline {baseline[0]}-{baseline[1]} ms)'
                cbar_label = 'dB'
            else:
                ersp_title = 'ERSP (dB re temporal mean)'
                cbar_label = 'dB'
            _plot_tf_panel(ax, fig, times_trimmed, freq_centers,
                           ersp_db_plot, ersp_cmap,
                           vmin=e_vmin, vmax=e_vmax,
                           ylabel='Frequency (Hz)', title=ersp_title,
                           cbar_label=cbar_label, interpolation=interpolation)

        elif panel == 'ppc':
            ppc_title = 'Pairwise Phase Consistency'
            if results['itc_thresh'] is not None:
                ppc_title += ' (bootstrap masked)'
            _plot_tf_panel(ax, fig, times_trimmed, freq_centers,
                           ppc_plot, ppc_cmap,
                           vmin=0, vmax=ppc_vmax,
                           ylabel='Frequency (Hz)',
                           title=ppc_title,
                           cbar_label='PPC', interpolation=interpolation)

        elif panel == 'itc':
            itc_title = 'Inter-Trial Coherence'
            if results['itc_thresh'] is not None:
                itc_title += ' (bootstrap masked)'
            _plot_tf_panel(ax, fig, times_trimmed, freq_centers,
                           itc_amp_plot, itc_cmap,
                           vmin=0, vmax=itc_vmax,
                           ylabel='Frequency (Hz)',
                           title=itc_title,
                           cbar_label='ITC', interpolation=interpolation)

        plot_idx += 1

    xlim = (times_trimmed[0], times_trimmed[-1])
    for ax in axes:
        ax.set_xlim(xlim)

    axes[-1].set_xlabel('Time (ms)')
    fig.tight_layout()
    return fig, plot_data


def plot_tfa_compare(results_list, labels=None, times=None, channel=0,
                     alpha=0.05, baseline='zscore', show=('ersp', 'ppc'),
                     trim_edges=True, interpolation='bilinear',
                     ersp_cmap='RdBu_r', ppc_cmap='Reds', itc_cmap='Reds',
                     figsize=None, ersp_vlim=None, itc_vmax=0.75,
                     ppc_vmax=0.15, share_clim=True):
    """Plot TFA results for multiple conditions side by side.

    Arranges panels as rows (one per metric) and columns (one per
    condition). Supports two to four conditions. Color limits are
    shared across conditions by default for direct visual comparison.

    Parameters
    ----------
    results_list : list of dict
        Each element is a hilbert_tfa results dict. Length 2-4.
    labels : list of str or None
        Column titles for each condition. Default uses
        'Condition 1', 'Condition 2', etc.
    times : ndarray or None
        Time vector in ms. Default creates one centered at 0.
    channel : int
        Channel index to plot. Default 0.
    alpha : float
        Significance level for bootstrap masking. Default 0.05.
    baseline : tuple of (float, float), 'zscore', or None
        Baseline correction for ERSP. Default 'zscore'.
    show : tuple of str
        Panels to display per condition: any combination of
        'ersp', 'ppc', 'itc'. Default ('ersp', 'ppc').
    trim_edges : bool
        Trim filter edge artifacts. Default True.
    interpolation : str or None
        Interpolation method for imshow. Default 'bilinear'.
    ersp_cmap : str or colormap
        Colormap for ERSP. Default 'RdBu_r'.
    ppc_cmap : str or colormap
        Colormap for PPC. Default 'Reds'.
    itc_cmap : str or colormap
        Colormap for ITC. Default 'Reds'.
    figsize : tuple or None
        Figure size. Default auto-scaled.
    ersp_vlim : tuple of (float, float) or None
        Shared (vmin, vmax) for ERSP. Default None (auto).
    itc_vmax : float
        Shared maximum for ITC colorbar. Default 0.75.
    ppc_vmax : float
        Shared maximum for PPC colorbar. Default 0.15.
    share_clim : bool
        Share color limits across conditions. Default True.
        When True, ERSP and PPC limits are computed from the
        global extremes across all conditions.

    Returns
    -------
    fig : matplotlib Figure
    plot_data_list : list of dict
        Data arrays for each condition.
    """
    import matplotlib.pyplot as plt

    n_cond = len(results_list)
    if n_cond < 2 or n_cond > 4:
        raise ValueError("results_list must contain 2-4 conditions.")

    if labels is None:
        labels = [f'Condition {i+1}' for i in range(n_cond)]

    if isinstance(show, str):
        show = (show,)
    panels = [s for s in show if s in ('ersp', 'ppc', 'itc')]
    n_rows = len(panels)
    if n_rows == 0:
        raise ValueError("No valid panels specified.")

    # --- Resolve colormaps ---
    ersp_cmap = _resolve_cmap(ersp_cmap)
    ppc_cmap = _resolve_cmap(ppc_cmap)
    itc_cmap = _resolve_cmap(itc_cmap)

    # --- Precompute data for all conditions ---
    all_data = []
    for results in results_list:
        frames = results['frames']
        freq_centers = results['freq_centers']

        if times is None:
            t = np.arange(frames) - frames // 2
        else:
            t = times

        edge = results.get('edge_frames', 0) if trim_edges else 0
        t_slice = slice(edge, frames - edge) if edge > 0 else slice(None)
        times_trimmed = t[t_slice] if edge > 0 else t

        # ERSP
        ersp_ch = results['ERSP'][channel].copy()
        ersp_db = 10.0 * np.log10(ersp_ch + 1e-20)

        if baseline == 'zscore':
            global_mean = np.mean(ersp_db)
            global_std = np.std(ersp_db)
            if global_std > 0:
                ersp_db = (ersp_db - global_mean) / global_std
            else:
                ersp_db = ersp_db - global_mean
        elif baseline is not None:
            bl_start, bl_end = baseline
            bl_mask = (t >= bl_start) & (t <= bl_end)
            if bl_mask.sum() > 0:
                bl_mean = ersp_db[:, bl_mask].mean(axis=1, keepdims=True)
                ersp_db = ersp_db - bl_mean
        else:
            temporal_mean = ersp_db.mean(axis=1, keepdims=True)
            ersp_db = ersp_db - temporal_mean

        ersp_db_plot = ersp_db[:, t_slice]

        # ITC
        itc_amp = np.abs(results['ITC'][channel]).copy()
        if results['itc_thresh'] is not None:
            thresh = results['itc_thresh'][channel]
            mask = itc_amp < thresh[:, np.newaxis]
            itc_amp[mask] = 0.0
        itc_amp_plot = itc_amp[:, t_slice]

        # PPC
        ppc_vals = results['PPC'][channel].copy()
        ppc_vals = np.clip(ppc_vals, 0, None)
        if results['itc_thresh'] is not None:
            n = results['ntrial']
            itc_t = results['itc_thresh'][channel]
            ppc_thresh = (n * itc_t ** 2 - 1) / max(n - 1, 1)
            mask = ppc_vals < ppc_thresh[:, np.newaxis]
            ppc_vals[mask] = 0.0
        ppc_plot = ppc_vals[:, t_slice]

        all_data.append({
            'ersp_db': ersp_db,
            'ersp_db_plot': ersp_db_plot,
            'itc_amp': itc_amp,
            'itc_amp_plot': itc_amp_plot,
            'ppc': ppc_vals,
            'ppc_plot': ppc_plot,
            'times': t,
            'times_trimmed': times_trimmed,
            'freq_centers': freq_centers,
        })

    # --- Compute shared color limits ---
    if share_clim and ersp_vlim is None:
        if baseline == 'zscore':
            e_vmin, e_vmax = -2, 2
        else:
            all_ersp = np.concatenate(
                [d['ersp_db_plot'].ravel() for d in all_data]
            )
            e_vmax = np.percentile(np.abs(all_ersp), 97)
            e_vmin = -e_vmax
        ersp_vlim = (e_vmin, e_vmax)

    if share_clim:
        global_ppc_max = max(d['ppc_plot'].max() for d in all_data)
        if ppc_vmax is None or global_ppc_max > ppc_vmax:
            ppc_vmax = min(global_ppc_max, 0.5)

        global_itc_max = max(d['itc_amp_plot'].max() for d in all_data)
        if itc_vmax is None or global_itc_max > itc_vmax:
            itc_vmax = min(global_itc_max, 1.0)

    # --- Build figure ---
    if figsize is None:
        figsize = (5 * n_cond, 3.5 * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_cond, figsize=figsize, squeeze=False
    )

    for col, d in enumerate(all_data):
        row = 0
        times_trimmed = d['times_trimmed']
        freq_centers = d['freq_centers']

        for panel in panels:
            ax = axes[row, col]

            if panel == 'ersp':
                e_vmin, e_vmax = ersp_vlim
                if baseline == 'zscore':
                    cbar_label = 'z'
                elif baseline is not None:
                    cbar_label = 'dB'
                else:
                    cbar_label = 'dB'
                _plot_tf_panel(
                    ax, fig, times_trimmed, freq_centers,
                    d['ersp_db_plot'], ersp_cmap,
                    vmin=e_vmin, vmax=e_vmax,
                    ylabel='Frequency (Hz)' if col == 0 else '',
                    title=f'{labels[col]} - ERSP' if row == 0 else 'ERSP',
                    cbar_label=cbar_label,
                    interpolation=interpolation,
                )

            elif panel == 'ppc':
                _plot_tf_panel(
                    ax, fig, times_trimmed, freq_centers,
                    d['ppc_plot'], ppc_cmap,
                    vmin=0, vmax=ppc_vmax,
                    ylabel='Frequency (Hz)' if col == 0 else '',
                    title=f'{labels[col]} - PPC' if row == 0 else 'PPC',
                    cbar_label='PPC',
                    interpolation=interpolation,
                )

            elif panel == 'itc':
                _plot_tf_panel(
                    ax, fig, times_trimmed, freq_centers,
                    d['itc_amp_plot'], itc_cmap,
                    vmin=0, vmax=itc_vmax,
                    ylabel='Frequency (Hz)' if col == 0 else '',
                    title=f'{labels[col]} - ITC' if row == 0 else 'ITC',
                    cbar_label='ITC',
                    interpolation=interpolation,
                )

            # Remove y-axis labels for non-leftmost columns
            if col > 0:
                ax.set_yticklabels([])

            row += 1

    # Set x-axis label on bottom row only
    for col in range(n_cond):
        axes[-1, col].set_xlabel('Time (ms)')

    # Shared x-limits across all panels
    xlim = (all_data[0]['times_trimmed'][0],
            all_data[0]['times_trimmed'][-1])
    for ax in axes.flat:
        ax.set_xlim(xlim)

    fig.tight_layout()
    return fig, all_data


def _plot_tf_panel(ax, fig, times, freqs, data, cmap,
                   vmin=None, vmax=None, ylabel='', title='',
                   cbar_label='', interpolation=None):
    """Render a single time-frequency panel."""
    if interpolation is not None:
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        im = ax.imshow(data, aspect='auto', origin='lower',
                       extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       interpolation=interpolation)
    else:
        im = ax.pcolormesh(times, freqs, data, shading='auto',
                           cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)


def _resolve_cmap(name):
    """Resolve colormap name to a matplotlib colormap object."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib

    if not isinstance(name, str):
        return name

    if name == 'seaborn_diverging':
        try:
            import seaborn as sns
            return sns.diverging_palette(150, 250, as_cmap=True)
        except ImportError:
            colors = ['#33a02c', '#b2df8a', '#f7f7f7', '#a6cee3', '#1f78b4']
            return LinearSegmentedColormap.from_list('green_blue_div', colors)

    if name == 'seaborn_blue':
        try:
            import seaborn as sns
            return sns.diverging_palette(250, 250, as_cmap=True)
        except ImportError:
            colors = ['#f7f7f7', '#a6cee3', '#1f78b4', '#08306b']
            return LinearSegmentedColormap.from_list('blue_seq', colors)

    if name == 'RdBu_r_positive':
        base = matplotlib.colormaps['RdBu_r']
        colors = base(np.linspace(0.0, 0.5, 256))
        return LinearSegmentedColormap.from_list('RdBu_r_pos', colors)

    return matplotlib.colormaps[name]
