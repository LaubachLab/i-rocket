#!/usr/bin/env python
"""
I-ROCKET Kernel Explorer (standalone)

An interactive tool for understanding MultiRocket kernels, dilation,
bias thresholding, and the four pooling operators.

Launches in its own matplotlib window. No Jupyter or ipywidgets required.

Usage:
    python kernel_explorer_standalone.py

Controls (bottom of window):
    Kernel:   slider (0-83)
    Dilation: radio buttons (1, 2, 4, 8, 16)
    Signal:   radio buttons (Gaussian bump, Two peaks, Oscillatory)
    Bias:     slider (-5 to 5), or check Auto for median
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons


# ============================================================================
# BASE KERNELS AND SIGNALS
# ============================================================================

def generate_base_kernels():
    """Generate all 84 MiniRocket base kernels."""
    kernels = np.full((84, 9), -1.0, dtype=np.float32)
    indices = np.zeros((84, 3), dtype=np.int32)
    count = 0
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                kernels[count, i] = 2.0
                kernels[count, j] = 2.0
                kernels[count, k] = 2.0
                indices[count] = [i, j, k]
                count += 1
    return kernels, indices


BASE_KERNELS, BASE_INDICES = generate_base_kernels()


def make_sample_signals(n_timepoints=128):
    """Generate three sample signals for exploration."""
    t = np.arange(n_timepoints, dtype=np.float32)
    center = n_timepoints // 2

    bump = 3.0 * np.exp(-0.5 * ((t - center) / 8.0) ** 2)
    bump += np.random.default_rng(0).normal(0, 0.3, n_timepoints).astype(np.float32)

    peak1 = 2.0 * np.exp(-0.5 * ((t - center * 0.6) / 6.0) ** 2)
    peak2 = 2.5 * np.exp(-0.5 * ((t - center * 1.3) / 7.0) ** 2)
    two_peak = peak1 + peak2
    two_peak += np.random.default_rng(1).normal(0, 0.3, n_timepoints).astype(np.float32)

    freq = 2 * np.pi * 5 / n_timepoints
    osc = 2.0 * np.sin(freq * t) * np.exp(-0.5 * ((t - center) / 30.0) ** 2)
    osc += np.random.default_rng(2).normal(0, 0.3, n_timepoints).astype(np.float32)

    return {
        "Gaussian bump": bump,
        "Two peaks": two_peak,
        "Oscillatory": osc,
    }


SIGNALS = make_sample_signals()
SIGNAL_NAMES = list(SIGNALS.keys())


# ============================================================================
# CONVOLUTION AND POOLING
# ============================================================================

def convolve_kernel(x, kernel_index, dilation):
    """Compute dilated convolution of kernel on signal x."""
    input_length = len(x)
    padding = ((9 - 1) * dilation) // 2
    i0, i1, i2 = BASE_INDICES[kernel_index]

    n_conv = input_length + 2 * padding - (9 - 1) * dilation
    if n_conv < 1:
        n_conv = 1

    conv_output = np.zeros(n_conv, dtype=np.float32)
    time_indices = np.zeros(n_conv, dtype=np.float32)

    for t in range(n_conv):
        center = t - padding + 4 * dilation
        time_indices[t] = center

        total = 0.0
        for pos in range(9):
            input_idx = t + pos * dilation - padding
            if 0 <= input_idx < input_length:
                total += x[input_idx]

        sum_at_indices = 0.0
        for idx_val in (i0, i1, i2):
            input_idx = t + idx_val * dilation - padding
            if 0 <= input_idx < input_length:
                sum_at_indices += x[input_idx]

        conv_output[t] = -total + 3.0 * sum_at_indices

    bias = float(np.median(conv_output))
    activation = (conv_output > bias).astype(np.float32)
    return conv_output, activation, time_indices, bias


def compute_pooling(conv_output, bias):
    """Compute the four MultiRocket pooling operators."""
    above = conv_output > bias
    n = len(conv_output)

    ppv = float(np.mean(above)) if n > 0 else 0.0
    positive_vals = conv_output[above]
    mpv = float(np.mean(positive_vals)) if len(positive_vals) > 0 else 0.0
    positive_indices = np.where(above)[0]
    mipv = float(np.mean(positive_indices)) / max(n - 1, 1) if len(positive_indices) > 0 else 0.0

    lspv = 0
    current_run = 0
    for val in above:
        if val:
            current_run += 1
            lspv = max(lspv, current_run)
        else:
            current_run = 0
    lspv = float(lspv) / max(n, 1)

    return {"PPV": ppv, "MPV": mpv, "MIPV": mipv, "LSPV": lspv}


# ============================================================================
# STATE
# ============================================================================

state = {
    'kernel': 0,
    'dilation': 1,
    'signal': SIGNAL_NAMES[0],
    'auto_bias': True,
    'manual_bias': 0.0,
}

_updating = False  # guard against recursive callbacks

# ============================================================================
# FIGURE LAYOUT
# ============================================================================

fig = plt.figure(figsize=(12, 9))
fig.canvas.manager.set_window_title('I-ROCKET Kernel Explorer')

# Plot area: top portion
gs_plots = fig.add_gridspec(
    3, 2, height_ratios=[1, 1.2, 0.8],
    hspace=0.35, wspace=0.3,
    top=0.93, bottom=0.28, left=0.08, right=0.95,
)

ax1 = fig.add_subplot(gs_plots[0, 0])
ax2 = fig.add_subplot(gs_plots[0, 1])
ax3 = fig.add_subplot(gs_plots[1, :])
ax4 = fig.add_subplot(gs_plots[2, :])

# Track the twin axis for panel 3 so we can remove it on redraw
_twin_ax = [None]


def draw_plots():
    """Redraw all four panels with current state."""
    ki = state['kernel']
    dil = state['dilation']
    sig_name = state['signal']
    x = SIGNALS[sig_name]
    n_timepoints = len(x)

    conv_out, _, time_idx, auto_b = convolve_kernel(x, ki, dil)

    if state['auto_bias']:
        bias = auto_b
        # Update slider without triggering callback
        global _updating
        _updating = True
        bias_slider.set_val(round(bias, 1))
        _updating = False
    else:
        bias = state['manual_bias']

    pooling = compute_pooling(conv_out, bias)
    rf = 1 + 8 * dil
    pos_idx = BASE_INDICES[ki]
    weights = BASE_KERNELS[ki]
    best_t = int(np.argmax(np.abs(conv_out)))
    best_center = int(round(time_idx[best_t]))
    t_axis = np.arange(n_timepoints)

    # --- Panel 1: Dilated kernel weight pattern ---
    ax1.clear()
    dilated_positions = np.arange(9) * dil
    bar_colors = ["#1f77b4" if i in pos_idx else "#c7c7c7" for i in range(9)]
    ax1.bar(dilated_positions, weights, width=max(0.8, dil * 0.6),
            color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(0, color="#7f7f7f", linewidth=0.5)
    ax1.set_xlabel("Position (dilated)")
    ax1.set_ylabel("Weight")
    ax1.set_title(f"Kernel {ki}: positions {pos_idx.tolist()}, "
                   f"dilation={dil}, RF={rf}")
    ax1.grid(True, alpha=0.15)

    # --- Panel 2: Kernel overlaid on signal ---
    ax2.clear()
    ax2.plot(t_axis, x, color="#7f7f7f", linewidth=1, label="Signal")
    rf_start = max(0, best_center - rf // 2)
    rf_end = min(n_timepoints, rf_start + rf)
    ax2.axvspan(rf_start, rf_end, alpha=0.12, color="#1f77b4",
                label=f"RF span ({rf} pts)")
    for i in range(9):
        signal_pos = best_center - 4 * dil + i * dil
        if 0 <= signal_pos < n_timepoints:
            mc = "#1f77b4" if i in pos_idx else "#c7c7c7"
            ms = 10 if i in pos_idx else 6
            ax2.plot(signal_pos, x[signal_pos], "o", color=mc,
                     markersize=ms, zorder=5)
    ax2.set_xlabel("Timepoint")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Kernel on signal at peak response (t={best_center})")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.15)

    # --- Panel 3: Convolution output ---
    ax3.clear()
    if _twin_ax[0] is not None:
        _twin_ax[0].remove()
    ax3_sig = ax3.twinx()
    _twin_ax[0] = ax3_sig
    ax3_sig.plot(t_axis, x, color="#c7c7c7", linewidth=1.5)
    ax3_sig.set_ylabel("Signal", color="#c7c7c7", fontsize=9)
    ax3_sig.tick_params(axis="y", labelcolor="#c7c7c7")

    ax3.plot(time_idx, conv_out, color="#1f77b4", linewidth=1.5,
             label="Conv output")
    ax3.axhline(bias, color="#d62728", linestyle="--", linewidth=1,
                label=f"Bias = {bias:.2f}")
    ax3.fill_between(time_idx, bias, conv_out, where=conv_out > bias,
                      color="#1f77b4", alpha=0.2, label="Above bias (fires)")
    ax3.plot(time_idx[best_t], conv_out[best_t], "v", color="#d62728",
             markersize=8, zorder=5, label=f"Peak (t={best_center})")
    ax3.set_xlabel("Timepoint")
    ax3.set_ylabel("Convolution output")
    ax3.legend(fontsize=8, loc="upper right", ncol=2)
    ax3.grid(True, alpha=0.15)
    ax3.set_xlim(0, n_timepoints)

    # --- Panel 4: Pooling operators ---
    ax4.clear()
    pool_names = ["PPV", "MPV", "MIPV", "LSPV"]
    pool_values = [pooling[k] for k in pool_names]
    pool_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    descriptions = [
        f"Proportion above bias ({pooling['PPV']:.1%})",
        f"Mean value above bias ({pooling['MPV']:.3f})",
        f"Mean position above bias ({pooling['MIPV']:.3f})",
        f"Longest consecutive run ({pooling['LSPV']:.3f})",
    ]
    bars = ax4.barh(pool_names, pool_values, color=pool_colors,
                     edgecolor="white", height=0.6, alpha=0.8)
    for i, (bar, desc) in enumerate(zip(bars, descriptions)):
        x_pos = max(bar.get_width() + 0.02, 0.02)
        ax4.text(x_pos, i, desc, va="center", fontsize=9, color="#2c2c2c")
    ax4.set_xlabel("Value")
    ax4.set_xlim(0, max(max(pool_values) * 1.5, 0.5))
    ax4.grid(True, alpha=0.15, axis="x")
    ax4.set_ylim(-0.5, len(pool_names) - 0.5)

    fig.canvas.draw_idle()


# ============================================================================
# CONTROLS
# ============================================================================

# Kernel slider
ax_kernel = fig.add_axes([0.08, 0.19, 0.35, 0.025])
kernel_slider = Slider(ax_kernel, 'Kernel', 0, 83, valinit=0, valstep=1,
                        valfmt='%d')

# Bias slider
ax_bias = fig.add_axes([0.08, 0.14, 0.35, 0.025])
bias_slider = Slider(ax_bias, 'Bias', -5.0, 5.0, valinit=0.0, valstep=0.1)

# Dilation radio buttons
ax_dil = fig.add_axes([0.52, 0.04, 0.10, 0.19])
dil_radio = RadioButtons(ax_dil, ['1', '2', '4', '8', '16'], active=0)
ax_dil.set_title('Dilation', fontsize=10, pad=4)

# Signal radio buttons
ax_sig = fig.add_axes([0.65, 0.04, 0.15, 0.19])
sig_radio = RadioButtons(ax_sig, SIGNAL_NAMES, active=0)
ax_sig.set_title('Signal', fontsize=10, pad=4)

# Auto bias checkbox
ax_auto = fig.add_axes([0.84, 0.10, 0.14, 0.07])
auto_check = CheckButtons(ax_auto, ['Auto bias'], [True])


# ============================================================================
# CALLBACKS
# ============================================================================

def on_kernel_change(val):
    global _updating
    if _updating:
        return
    _updating = True
    state['kernel'] = int(val)
    draw_plots()
    _updating = False


def on_bias_change(val):
    global _updating
    if _updating:
        return
    if not state['auto_bias']:
        _updating = True
        state['manual_bias'] = val
        draw_plots()
        _updating = False


def on_dilation_change(label):
    global _updating
    if _updating:
        return
    _updating = True
    state['dilation'] = int(label)
    draw_plots()
    _updating = False


def on_signal_change(label):
    global _updating
    if _updating:
        return
    _updating = True
    state['signal'] = label
    draw_plots()
    _updating = False


def on_auto_change(label):
    global _updating
    if _updating:
        return
    _updating = True
    state['auto_bias'] = not state['auto_bias']
    draw_plots()
    _updating = False


kernel_slider.on_changed(on_kernel_change)
bias_slider.on_changed(on_bias_change)
dil_radio.on_clicked(on_dilation_change)
sig_radio.on_clicked(on_signal_change)
auto_check.on_clicked(on_auto_change)


# ============================================================================
# LAUNCH
# ============================================================================

draw_plots()
plt.show()
