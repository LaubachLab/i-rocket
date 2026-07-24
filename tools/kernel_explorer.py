#!/usr/bin/env python
"""I-ROCKET Kernel Explorer for the repaired I-ROCKET transform.

An interactive, standalone Matplotlib tool for understanding the complete
identity of an I-ROCKET/MultiRocket feature:

    representation + base kernel + dilation + padding + bias + pooling

Convolution and activation are delegated to :func:`interp_rocket.compute_activation_map`.

The explorer can be used in two modes.

Custom exploration
------------------
Choose a kernel, dilation, representation, padding mode, bias, and pooling
operator interactively.  In quantile-bias mode, the threshold is estimated
from the selected signal's *same-padded* convolution, matching the region used
when MultiRocket fits its biases.  The fitted package obtains each bias from
a quantile of one randomly selected training series for each kernel--dilation pair.

Fitted-feature exploration
--------------------------
Load a fitted ``InterpRocketTransform``, ``InterpRocket``,
``StableRocketClassifier``, or nested-CV result saved with ``joblib``.  Enter a
full transformed feature index to load its exact representation, kernel,
dilation, padding, fitted bias, bias rank, and pooling operator.

Usage
-----
Run with the built-in signals::

    python kernel_explorer.py

Use signals stored one per row in a NumPy or CSV file::

    python kernel_explorer.py --data class_means.npy --labels low high

Inspect a fitted feature::

    python kernel_explorer.py \
        --model final_model.joblib \
        --feature-index 340 \
        --data class_means.npy \
        --labels class_0 class_1

Notes
-----
* It requires the current ``interp_rocket`` module to be importable.
* A saved model is optional.  ``joblib`` is needed only when ``--model`` is
  used; it is already installed with scikit-learn.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Slider, TextBox
import numpy as np

try:
    from interp_rocket import OI, POOLING_COLORS, compute_activation_map
except ImportError as exc:  # pragma: no cover - exercised by users directly
    raise ImportError(
        "kernel_explorer.py requires the current I-ROCKET source or an "
        "installed I-ROCKET package. Run it from the repository root or "
        "install the package first."
    ) from exc


POOLING_NAMES = ("PPV", "MPV", "MIPV", "LSPV")
POOLING_DESCRIPTIONS = {
    "PPV": "Proportion of Positive Values",
    "MPV": "Mean of Positive Values",
    "MIPV": "Mean of Indices of Positive Values",
    "LSPV": "Longest Stretch of Positive Values",
}


# ---------------------------------------------------------------------------
# Kernels and sample signals
# ---------------------------------------------------------------------------


def generate_base_kernels() -> Tuple[np.ndarray, np.ndarray]:
    """Return the 84 deterministic length-nine MiniRocket kernels."""
    kernels = np.full((84, 9), -1.0, dtype=np.float32)
    indices = np.empty((84, 3), dtype=np.int32)
    count = 0
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                kernels[count, (i, j, k)] = 2.0
                indices[count] = (i, j, k)
                count += 1
    return kernels, indices


BASE_KERNELS, BASE_INDICES = generate_base_kernels()


def make_sample_signals(n_timepoints: int = 128) -> Dict[str, np.ndarray]:
    """Generate three deterministic example signals of a requested length."""
    if not isinstance(n_timepoints, (int, np.integer)) or n_timepoints < 9:
        raise ValueError("n_timepoints must be an integer of at least 9")

    t = np.arange(n_timepoints, dtype=np.float32)
    center = (n_timepoints - 1) / 2.0
    width = max(1.5, n_timepoints / 16.0)

    bump = 3.0 * np.exp(-0.5 * ((t - center) / width) ** 2)
    bump += np.random.default_rng(0).normal(0, 0.3, n_timepoints).astype(
        np.float32
    )

    peak1 = 2.0 * np.exp(
        -0.5 * ((t - 0.30 * (n_timepoints - 1)) / max(1.5, width * 0.75)) ** 2
    )
    peak2 = 2.5 * np.exp(
        -0.5 * ((t - 0.68 * (n_timepoints - 1)) / max(1.5, width * 0.90)) ** 2
    )
    two_peak = peak1 + peak2
    two_peak += np.random.default_rng(1).normal(0, 0.3, n_timepoints).astype(
        np.float32
    )

    cycles = 5.0
    phase = 2.0 * np.pi * cycles * t / max(n_timepoints, 1)
    envelope = np.exp(
        -0.5 * ((t - center) / max(2.0, n_timepoints / 4.0)) ** 2
    )
    oscillatory = 2.0 * np.sin(phase) * envelope
    oscillatory += np.random.default_rng(2).normal(
        0, 0.3, n_timepoints
    ).astype(np.float32)

    return {
        "Gaussian bump": np.asarray(bump, dtype=np.float32),
        "Two peaks": np.asarray(two_peak, dtype=np.float32),
        "Oscillatory": np.asarray(oscillatory, dtype=np.float32),
    }


def load_user_signals(
    path: Union[str, Path],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load one signal per row from a ``.npy`` or ``.csv`` file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".csv":
        array = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(
            f"Unsupported data file {path}. Use a .npy or .csv file."
        )

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(
            f"Loaded data must be 1D or 2D; got shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Loaded data contain NaN or infinite values.")
    if array.shape[1] < 9:
        raise ValueError("Each signal must contain at least nine timepoints.")

    labels = tuple(labels or ())
    output: Dict[str, np.ndarray] = {}
    for row_index, row in enumerate(array):
        name = (
            str(labels[row_index])
            if row_index < len(labels)
            else f"Data {row_index + 1}"
        )
        if name in output:
            raise ValueError(f"Duplicate signal label: {name!r}")
        output[name] = np.asarray(row, dtype=np.float32)
    return output


# ---------------------------------------------------------------------------
# Exact feature calculations
# ---------------------------------------------------------------------------


def analysis_signal(raw_signal: np.ndarray, representation: str) -> np.ndarray:
    """Return the raw or first-differenced signal used by the transform."""
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    if raw_signal.ndim != 1:
        raise ValueError("Each signal must be one-dimensional.")
    if representation == "raw":
        return raw_signal
    if representation == "diff":
        if raw_signal.size < 10:
            raise ValueError(
                "The differenced representation requires at least ten raw "
                "timepoints so the resulting signal has length nine."
            )
        return np.diff(raw_signal).astype(np.float32)
    raise ValueError("representation must be 'raw' or 'diff'")


def kernel_offsets(dilation: int, representation: str) -> np.ndarray:
    """Return the exact nine input offsets used by the repaired transform."""
    if not isinstance(dilation, (int, np.integer)) or int(dilation) < 1:
        raise ValueError("dilation must be a positive integer")
    offsets = (np.arange(9, dtype=np.int32) - 4) * int(dilation)
    # This reproduces the established asymmetric first-difference alignment
    # used in the reference MultiRocket implementation and I-ROCKET.
    if representation == "diff":
        offsets[:4] += 1
    elif representation != "raw":
        raise ValueError("representation must be 'raw' or 'diff'")
    return offsets


def compute_pooling_exact(
    convolution: np.ndarray,
    bias: float,
) -> Dict[str, float]:
    """Calculate the four pooling values exactly as I-ROCKET does.

    ``convolution`` must already contain the exact same- or valid-padding
    region returned by :func:`interp_rocket.compute_activation_map`.
    """
    convolution = np.asarray(convolution, dtype=np.float32)
    if convolution.ndim != 1 or convolution.size == 0:
        raise ValueError("convolution must be a nonempty one-dimensional array")
    if not np.isfinite(convolution).all() or not np.isfinite(bias):
        raise ValueError("convolution and bias must be finite")

    ppv_count = 0
    last_value = 0
    maximum_stretch = 0.0
    mean_index = 0
    mean_value = 0.0

    for local_index, value in enumerate(convolution):
        if value > bias:
            ppv_count += 1
            mean_index += local_index
            # Deliberately matches the original/aeon MultiRocket operation.
            mean_value += float(value) + float(bias)
        elif value < bias:
            stretch = local_index - last_value
            if stretch > maximum_stretch:
                maximum_stretch = float(stretch)
            last_value = local_index

    stretch = convolution.size - 1 - last_value
    if stretch > maximum_stretch:
        maximum_stretch = float(stretch)

    return {
        "PPV": float(ppv_count / convolution.size),
        "MPV": float(mean_value / ppv_count) if ppv_count else 0.0,
        "MIPV": float(mean_index / ppv_count) if ppv_count else -1.0,
        "LSPV": float(maximum_stretch),
    }


def quantile_bias(
    signal: np.ndarray,
    kernel_index: int,
    dilation: int,
    quantile: float,
    representation: str,
) -> float:
    """Estimate a teaching bias from the same-padded convolution of a signal."""
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    convolution, _, _ = compute_activation_map(
        signal,
        int(kernel_index),
        int(dilation),
        0.0,
        padding="same",
        representation=representation,
    )
    return float(np.quantile(convolution, float(quantile)))


def format_feature_identity(info: Mapping[str, Any]) -> str:
    """Return an unambiguous compact label for one transformed feature."""
    parts = []
    if "feature_index" in info:
        parts.append(f"F{int(info['feature_index'])}")
    parts.extend(
        [
            f"K{int(info['kernel_index'])}",
            f"d={int(info['dilation'])}",
        ]
    )
    if info.get("bias_rank_within_kernel") is not None:
        parts.append(f"b{int(info['bias_rank_within_kernel'])}")
    parts.extend(
        [
            str(info.get("pooling_op", "PPV")),
            str(info.get("representation", "raw")),
            str(info.get("padding_mode", "same")),
        ]
    )
    if "bias" in info:
        parts.append(f"bias={float(info['bias']):.5g}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Optional fitted-model support
# ---------------------------------------------------------------------------


def load_serialized_model(path: Union[str, Path]) -> Any:
    """Load a fitted estimator or nested-CV result with joblib."""
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading a model requires joblib, which is installed with "
            "scikit-learn."
        ) from exc
    return joblib.load(Path(path))


def resolve_feature_decoder(model: Any) -> Any:
    """Find an object exposing ``decode_feature_index`` in a saved result."""
    stack = [model]
    seen = set()
    while stack:
        candidate = stack.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        if callable(getattr(candidate, "decode_feature_index", None)):
            return candidate

        for attribute in (
            "final_model",
            "final_model_",
            "best_estimator_",
            "estimator_",
            "transformer_",
        ):
            if hasattr(candidate, attribute):
                stack.append(getattr(candidate, attribute))

        named_steps = getattr(candidate, "named_steps", None)
        if named_steps is not None:
            stack.extend(reversed(tuple(named_steps.values())))

    raise TypeError(
        "The loaded object does not contain a fitted I-ROCKET transformer "
        "with decode_feature_index()."
    )


def infer_training_length(decoder: Any, default: int = 128) -> int:
    """Infer the raw training-series length from a fitted decoder."""
    for candidate in (
        decoder,
        getattr(decoder, "transformer_", None),
    ):
        if candidate is None:
            continue
        for attribute in ("n_timepoints_in_", "n_features_in_"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, (int, np.integer)) and int(value) >= 9:
                return int(value)
    return int(default)


# ---------------------------------------------------------------------------
# Interactive application
# ---------------------------------------------------------------------------


@dataclass
class ExplorerState:
    kernel: int = 0
    dilation: int = 1
    signal_name: str = ""
    representation: str = "raw"
    padding: str = "same"
    pooling: str = "PPV"
    use_quantile_bias: bool = True
    quantile: float = 0.5
    manual_bias: float = 0.0
    loaded_feature: Optional[Dict[str, Any]] = None


class KernelExplorer:
    """Interactive Matplotlib application for inspecting I-ROCKET features."""

    def __init__(
        self,
        signals: Mapping[str, np.ndarray],
        *,
        decoder: Any = None,
        initial_feature_index: Optional[int] = None,
    ) -> None:
        if not signals:
            raise ValueError("At least one signal is required.")
        self.signals = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in signals.items()
        }
        for name, signal in self.signals.items():
            if signal.ndim != 1 or signal.size < 9:
                raise ValueError(
                    f"Signal {name!r} must be one-dimensional with at least "
                    "nine timepoints."
                )
            if not np.isfinite(signal).all():
                raise ValueError(f"Signal {name!r} contains non-finite values.")

        self.signal_names = tuple(self.signals)
        self.decoder = decoder
        self.state = ExplorerState(signal_name=self.signal_names[0])
        self._updating = False
        self._conv_twin = None
        self._status_message = ""

        if initial_feature_index is not None:
            if self.decoder is None:
                raise ValueError(
                    "initial_feature_index requires a fitted model/transformer"
                )
            self._set_state_from_feature(int(initial_feature_index))

        self._build_figure()
        self._connect_callbacks()
        self.draw()

    # --------------------------- state helpers ---------------------------

    def _set_state_from_feature(self, feature_index: int) -> None:
        info = dict(self.decoder.decode_feature_index(int(feature_index)))
        self.state.kernel = int(info["kernel_index"])
        self.state.dilation = int(info["dilation"])
        self.state.representation = str(info["representation"])
        self.state.padding = str(info["padding_mode"])
        self.state.pooling = str(info["pooling_op"])
        self.state.manual_bias = float(info["bias"])
        self.state.use_quantile_bias = False
        self.state.loaded_feature = info
        self._status_message = f"Loaded {format_feature_identity(info)}"

    def _mark_custom(self) -> None:
        was_loaded = self.state.loaded_feature is not None
        self.state.loaded_feature = None
        if was_loaded and getattr(self, "feature_box", None) is not None:
            self._updating = True
            try:
                self.feature_box.set_val("")
            finally:
                self._updating = False

    def _current_raw_signal(self) -> np.ndarray:
        return self.signals[self.state.signal_name]

    def _current_analysis_signal(self) -> np.ndarray:
        return analysis_signal(
            self._current_raw_signal(), self.state.representation
        )

    def _recommended_max_dilation(self) -> int:
        signal = self._current_analysis_signal()
        return max(1, int((signal.size - 1) // 8))

    def _sync_dilation_slider(self) -> None:
        recommended = self._recommended_max_dilation()
        visible_max = max(recommended, int(self.state.dilation))
        self.dilation_slider.valmax = visible_max
        self.dilation_slider.ax.set_xlim(1, visible_max)

    # ------------------------------ layout ------------------------------

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 10))
        try:
            self.fig.canvas.manager.set_window_title(
                "I-ROCKET Kernel Explorer"
            )
        except AttributeError:  # noninteractive backends
            pass

        grid = self.fig.add_gridspec(
            4,
            2,
            height_ratios=(1.0, 1.25, 0.42, 0.88),
            hspace=0.62,
            wspace=0.28,
            top=0.91,
            bottom=0.34,
            left=0.07,
            right=0.97,
        )
        self.ax_kernel = self.fig.add_subplot(grid[0, 0])
        self.ax_signal = self.fig.add_subplot(grid[0, 1])
        self.ax_convolution = self.fig.add_subplot(grid[1, :])
        self.ax_activation = self.fig.add_subplot(grid[2, :])
        self.ax_pooling = self.fig.add_subplot(grid[3, :])

        # Continuous controls on the left.
        ax_kernel_slider = self.fig.add_axes([0.07, 0.275, 0.34, 0.022])
        self.kernel_slider = Slider(
            ax_kernel_slider,
            "Kernel",
            0,
            83,
            valinit=self.state.kernel,
            valstep=1,
            valfmt="%d",
        )

        max_dilation = max(
            self._recommended_max_dilation(), int(self.state.dilation)
        )
        ax_dilation_slider = self.fig.add_axes([0.07, 0.235, 0.34, 0.022])
        self.dilation_slider = Slider(
            ax_dilation_slider,
            "Dilation",
            1,
            max_dilation,
            valinit=self.state.dilation,
            valstep=1,
            valfmt="%d",
        )

        ax_quantile_slider = self.fig.add_axes([0.07, 0.195, 0.34, 0.022])
        self.quantile_slider = Slider(
            ax_quantile_slider,
            "Bias quantile",
            0.01,
            0.99,
            valinit=self.state.quantile,
            valstep=0.01,
            valfmt="%.2f",
        )

        ax_bias_box = self.fig.add_axes([0.12, 0.142, 0.20, 0.035])
        self.bias_box = TextBox(
            ax_bias_box,
            "Bias ",
            initial=f"{self.state.manual_bias:.6g}",
        )

        ax_quantile_check = self.fig.add_axes([0.33, 0.128, 0.12, 0.065])
        self.quantile_check = CheckButtons(
            ax_quantile_check,
            ["Use quantile"],
            [self.state.use_quantile_bias],
        )

        # Categorical controls.
        ax_representation = self.fig.add_axes([0.47, 0.185, 0.09, 0.115])
        self.representation_radio = RadioButtons(
            ax_representation,
            ("raw", "diff"),
            active=(0 if self.state.representation == "raw" else 1),
        )
        ax_representation.set_title("Representation", fontsize=9, pad=3)

        ax_padding = self.fig.add_axes([0.58, 0.185, 0.09, 0.115])
        self.padding_radio = RadioButtons(
            ax_padding,
            ("same", "valid"),
            active=(0 if self.state.padding == "same" else 1),
        )
        ax_padding.set_title("Padding", fontsize=9, pad=3)

        ax_pooling = self.fig.add_axes([0.69, 0.145, 0.09, 0.155])
        self.pooling_radio = RadioButtons(
            ax_pooling,
            POOLING_NAMES,
            active=POOLING_NAMES.index(self.state.pooling),
        )
        ax_pooling.set_title("Feature type", fontsize=9, pad=3)

        signal_height = min(0.245, max(0.13, 0.043 * len(self.signal_names)))
        ax_signal = self.fig.add_axes([0.80, 0.055, 0.17, signal_height])
        self.signal_radio = RadioButtons(
            ax_signal,
            self.signal_names,
            active=self.signal_names.index(self.state.signal_name),
        )
        ax_signal.set_title("Signal", fontsize=9, pad=3)

        self.feature_box = None
        if self.decoder is not None:
            initial = ""
            if self.state.loaded_feature is not None:
                initial = str(self.state.loaded_feature["feature_index"])
            ax_feature_box = self.fig.add_axes([0.12, 0.088, 0.20, 0.035])
            self.feature_box = TextBox(
                ax_feature_box,
                "Feature ",
                initial=initial,
            )
            self.fig.text(
                0.33,
                0.101,
                "press Enter to decode",
                ha="left",
                va="center",
                fontsize=8,
            )

        self.status_text = self.fig.text(
            0.07,
            0.025,
            "",
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
        )

        for radio in (
            self.representation_radio,
            self.padding_radio,
            self.pooling_radio,
            self.signal_radio,
        ):
            for label in radio.labels:
                label.set_fontsize(8.5)

    def _connect_callbacks(self) -> None:
        self.kernel_slider.on_changed(self._on_kernel)
        self.dilation_slider.on_changed(self._on_dilation)
        self.quantile_slider.on_changed(self._on_quantile)
        self.bias_box.on_submit(self._on_bias)
        self.quantile_check.on_clicked(self._on_quantile_mode)
        self.representation_radio.on_clicked(self._on_representation)
        self.padding_radio.on_clicked(self._on_padding)
        self.pooling_radio.on_clicked(self._on_pooling)
        self.signal_radio.on_clicked(self._on_signal)
        if self.feature_box is not None:
            self.feature_box.on_submit(self._on_feature)

    # ----------------------------- callbacks ----------------------------

    def _on_kernel(self, value: float) -> None:
        if self._updating:
            return
        self.state.kernel = int(value)
        self._mark_custom()
        self.draw()

    def _on_dilation(self, value: float) -> None:
        if self._updating:
            return
        self.state.dilation = int(value)
        self._mark_custom()
        self.draw()

    def _on_quantile(self, value: float) -> None:
        if self._updating:
            return
        self.state.quantile = float(value)
        if self.state.use_quantile_bias:
            self._mark_custom()
            self.draw()

    def _on_bias(self, text: str) -> None:
        if self._updating:
            return
        try:
            value = float(text)
        except ValueError:
            self._status_message = f"Invalid bias: {text!r}"
            self.draw()
            return
        if not np.isfinite(value):
            self._status_message = "Bias must be finite."
            self.draw()
            return
        self.state.manual_bias = value
        if not self.state.use_quantile_bias:
            self._mark_custom()
            self.draw()

    def _on_quantile_mode(self, _label: str) -> None:
        if self._updating:
            return
        self.state.use_quantile_bias = bool(
            self.quantile_check.get_status()[0]
        )
        self._mark_custom()
        self.draw()

    def _on_representation(self, label: str) -> None:
        if self._updating:
            return
        self.state.representation = str(label)
        self._mark_custom()
        try:
            self._sync_dilation_slider()
        except ValueError as exc:
            self._status_message = str(exc)
        self.draw()

    def _on_padding(self, label: str) -> None:
        if self._updating:
            return
        self.state.padding = str(label)
        self._mark_custom()
        self.draw()

    def _on_pooling(self, label: str) -> None:
        if self._updating:
            return
        self.state.pooling = str(label)
        self._mark_custom()
        self.draw()

    def _on_signal(self, label: str) -> None:
        if self._updating:
            return
        self.state.signal_name = str(label)
        try:
            self._sync_dilation_slider()
        except ValueError as exc:
            self._status_message = str(exc)
        self.draw()

    def _on_feature(self, text: str) -> None:
        if self._updating:
            return
        try:
            feature_index = int(text.strip())
            self._set_state_from_feature(feature_index)
            self._sync_widgets_from_state()
            self.draw()
        except Exception as exc:  # show decoding errors in the window
            self._status_message = f"Feature load failed: {exc}"
            self.draw()

    def _sync_widgets_from_state(self) -> None:
        self._updating = True
        try:
            self.kernel_slider.set_val(self.state.kernel)
            self._sync_dilation_slider()
            self.dilation_slider.set_val(self.state.dilation)
            self.representation_radio.set_active(
                0 if self.state.representation == "raw" else 1
            )
            self.padding_radio.set_active(
                0 if self.state.padding == "same" else 1
            )
            self.pooling_radio.set_active(
                POOLING_NAMES.index(self.state.pooling)
            )
            current_quantile_status = bool(
                self.quantile_check.get_status()[0]
            )
            if current_quantile_status != self.state.use_quantile_bias:
                self.quantile_check.set_active(0)
            self.bias_box.set_val(f"{self.state.manual_bias:.8g}")
            if self.feature_box is not None and self.state.loaded_feature:
                self.feature_box.set_val(
                    str(self.state.loaded_feature["feature_index"])
                )
        finally:
            self._updating = False

    # ------------------------------- draw -------------------------------

    def _clear_with_error(self, message: str) -> None:
        for axis in (
            self.ax_kernel,
            self.ax_signal,
            self.ax_convolution,
            self.ax_activation,
            self.ax_pooling,
        ):
            axis.clear()
            axis.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=axis.transAxes,
                wrap=True,
            )
            axis.set_axis_off()
        if self._conv_twin is not None:
            self._conv_twin.remove()
            self._conv_twin = None
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

    def draw(self) -> None:
        """Redraw the feature using the current interactive state."""
        try:
            raw_signal = self._current_raw_signal()
            signal = self._current_analysis_signal()
            kernel_index = int(self.state.kernel)
            dilation = int(self.state.dilation)
            representation = self.state.representation
            padding = self.state.padding

            recommended_max = max(1, int((signal.size - 1) // 8))
            if self.state.use_quantile_bias:
                bias = quantile_bias(
                    signal,
                    kernel_index,
                    dilation,
                    self.state.quantile,
                    representation,
                )
                self.state.manual_bias = bias
                self._updating = True
                try:
                    self.bias_box.set_val(f"{bias:.8g}")
                finally:
                    self._updating = False
            else:
                bias = float(self.state.manual_bias)

            convolution, activation, time_indices = compute_activation_map(
                signal,
                kernel_index,
                dilation,
                bias,
                padding=padding,
                representation=representation,
            )
            pooling = compute_pooling_exact(convolution, bias)
            offsets = kernel_offsets(dilation, representation)
            weights = BASE_KERNELS[kernel_index]
            positive_positions = BASE_INDICES[kernel_index]

            peak_local = int(np.argmax(convolution))
            peak_center = int(round(float(time_indices[peak_local])))
            footprint = peak_center + offsets
            receptive_field = int(1 + 8 * dilation)
        except Exception as exc:
            self._clear_with_error(str(exc))
            return

        feature_info = {
            "kernel_index": kernel_index,
            "dilation": dilation,
            "pooling_op": self.state.pooling,
            "representation": representation,
            "padding_mode": padding,
            "bias": bias,
        }
        if self.state.loaded_feature is not None:
            feature_info.update(self.state.loaded_feature)
        identity = format_feature_identity(feature_info)
        self.fig.suptitle(
            f"I-ROCKET Kernel Explorer — {identity}",
            fontsize=14,
            y=0.965,
        )

        # Panel 1: exact kernel offsets and weights.
        self.ax_kernel.clear()
        colors = [OI[0] if value > 0 else "#B3B3B3" for value in weights]
        tap_indices = np.arange(9)
        self.ax_kernel.bar(
            tap_indices,
            weights,
            width=0.78,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
        )
        self.ax_kernel.axhline(0.0, color="#666666", linewidth=0.7)
        self.ax_kernel.set_xticks(
            tap_indices,
            labels=[f"{int(offset):+d}" for offset in offsets],
        )
        self.ax_kernel.set_xlabel(
            "Input offset for each of the nine kernel taps"
        )
        self.ax_kernel.set_ylabel("Kernel weight")
        self.ax_kernel.set_title(
            f"Base kernel K{kernel_index}: +2 at "
            f"{positive_positions.tolist()} | RF={receptive_field}"
        )
        self.ax_kernel.grid(True, axis="y", alpha=0.16)

        # Panel 2: selected representation with the exact footprint at peak.
        self.ax_signal.clear()
        sample_axis = np.arange(signal.size)
        self.ax_signal.plot(
            sample_axis,
            signal,
            color="#666666",
            linewidth=1.2,
            label=(
                "raw signal"
                if representation == "raw"
                else "first difference: raw[j+1] - raw[j]"
            ),
        )
        span_start = max(-0.5, float(np.min(footprint)) - 0.5)
        span_stop = min(signal.size - 0.5, float(np.max(footprint)) + 0.5)
        self.ax_signal.axvspan(
            span_start,
            span_stop,
            color=OI[5],
            alpha=0.18,
            label="kernel footprint at maximum response",
        )
        padded_taps = 0
        for position, weight in zip(footprint, weights):
            if 0 <= position < signal.size:
                marker_color = OI[0] if weight > 0 else "#666666"
                marker_size = 8 if weight > 0 else 7
                marker = "o" if weight > 0 else "x"
                self.ax_signal.plot(
                    position,
                    signal[position],
                    marker=marker,
                    linestyle="none",
                    color=marker_color,
                    markersize=marker_size,
                    markeredgewidth=1.5,
                    zorder=5 if weight > 0 else 6,
                )
            else:
                padded_taps += 1
        self.ax_signal.set_xlabel(
            "Raw sample index"
            if representation == "raw"
            else "Difference-signal index"
        )
        self.ax_signal.set_ylabel("Amplitude")
        title = f"Kernel footprint centered at index {peak_center}"
        if padded_taps:
            title += f" ({padded_taps} zero-padded tap(s))"
        self.ax_signal.set_title(title)
        self.ax_signal.legend(fontsize=7.5, loc="best")
        self.ax_signal.grid(True, alpha=0.16)

        # Panel 3: exact convolution region and thresholded response.
        self.ax_convolution.clear()
        if self._conv_twin is not None:
            self._conv_twin.remove()
        self._conv_twin = self.ax_convolution.twinx()
        self._conv_twin.plot(
            sample_axis,
            signal,
            color="#B3B3B3",
            linewidth=1.2,
            alpha=0.7,
        )
        self._conv_twin.set_ylabel("Input representation", color="#888888")
        self._conv_twin.tick_params(axis="y", colors="#888888", labelsize=8)

        self.ax_convolution.plot(
            time_indices,
            convolution,
            color=OI[0],
            linewidth=1.5,
            label="convolution output",
        )
        self.ax_convolution.axhline(
            bias,
            color=OI[3],
            linestyle="--",
            linewidth=1.2,
            label=f"bias = {bias:.5g}",
        )
        self.ax_convolution.fill_between(
            time_indices,
            bias,
            convolution,
            where=convolution > bias,
            color=POOLING_COLORS[self.state.pooling],
            alpha=0.22,
            interpolate=True,
            label="above bias",
        )
        self.ax_convolution.plot(
            time_indices[peak_local],
            convolution[peak_local],
            "v",
            color=OI[3],
            markersize=8,
            zorder=6,
            label=f"maximum at {peak_center}",
        )
        self.ax_convolution.set_xlim(-0.5, signal.size - 0.5)
        self.ax_convolution.set_xlabel("")
        self.ax_convolution.set_ylabel("Convolution output")
        self.ax_convolution.set_title(
            f"{padding.capitalize()} pooling region | "
            f"{len(convolution)} evaluated positions"
        )
        self.ax_convolution.legend(fontsize=8, loc="upper right", ncol=2)
        self.ax_convolution.grid(True, alpha=0.16)

        # Panel 4: binary activation used by every pooling statistic.
        self.ax_activation.clear()
        self.ax_activation.step(
            time_indices,
            activation,
            where="mid",
            color=POOLING_COLORS[self.state.pooling],
            linewidth=1.5,
        )
        self.ax_activation.fill_between(
            time_indices,
            0.0,
            activation,
            step="mid",
            color=POOLING_COLORS[self.state.pooling],
            alpha=0.32,
        )
        self.ax_activation.set_xlim(-0.5, signal.size - 0.5)
        self.ax_activation.set_ylim(-0.08, 1.12)
        self.ax_activation.set_yticks((0, 1), labels=("off", "fires"))
        self.ax_activation.set_xlabel("Representation-signal index", labelpad=2)
        self.ax_activation.set_title(
            f"Binary activation: {int(np.sum(activation))} of "
            f"{len(activation)} positions above bias"
        )
        self.ax_activation.grid(True, axis="x", alpha=0.14)

        # Panel 5: heterogeneous pooling values as a table, not one shared bar.
        self.ax_pooling.clear()
        self.ax_pooling.set_axis_off()
        table_rows = []
        for name in POOLING_NAMES:
            value = pooling[name]
            formatted = (
                f"{value:.5f}"
                if name in ("PPV", "MPV")
                else f"{value:.3f}"
            )
            table_rows.append((name, formatted, POOLING_DESCRIPTIONS[name]))
        table = self.ax_pooling.table(
            cellText=table_rows,
            colLabels=("Operator", "Value", "Meaning"),
            cellLoc="left",
            colLoc="left",
            colWidths=(0.12, 0.15, 0.68),
            bbox=(0.02, 0.02, 0.96, 0.92),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for column in range(3):
            table[(0, column)].set_text_props(weight="bold")
            table[(0, column)].set_facecolor("#E8E8E8")
        for row_index, name in enumerate(POOLING_NAMES, start=1):
            table[(row_index, 0)].set_facecolor(POOLING_COLORS[name])
            table[(row_index, 0)].set_text_props(
                color="white" if name != "LSPV" else "black",
                weight="bold",
            )
            if name == self.state.pooling:
                for column in (1, 2):
                    table[(row_index, column)].set_facecolor("#FFF4CC")
                    table[(row_index, column)].set_text_props(weight="bold")
        self.ax_pooling.set_title(
            "All four features share this convolution and bias; the selected "
            "feature type is highlighted",
            fontsize=10,
            pad=2,
        )

        status_parts = []
        if self.state.loaded_feature is not None:
            status_parts.append(
                f"MODEL FEATURE: {format_feature_identity(self.state.loaded_feature)}"
            )
        elif self.state.use_quantile_bias:
            status_parts.append(
                f"CUSTOM: q={self.state.quantile:.2f} bias from this signal's "
                "same-padded convolution"
            )
        else:
            status_parts.append(f"CUSTOM: manual bias={bias:.6g}")

        if dilation > recommended_max:
            status_parts.append(
                f"warning: d={dilation} exceeds the largest dilation normally "
                f"fitted for a length-{signal.size} representation "
                f"(d_max={recommended_max})"
            )
        if self._status_message and not self.state.loaded_feature:
            status_parts.append(self._status_message)
        self.status_text.set_text(" | ".join(status_parts))
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Open the Matplotlib window and block until it is closed."""
        plt.show()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore exact I-ROCKET kernel configurations and transformed "
            "features in an interactive Matplotlib window."
        )
    )
    parser.add_argument(
        "--data",
        default=None,
        help=".npy or .csv file with one signal per row.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Display labels for rows loaded with --data.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "joblib file containing a fitted I-ROCKET transformer/model or "
            "a nested-CV result with final_model."
        ),
    )
    parser.add_argument(
        "--feature-index",
        type=int,
        default=None,
        help="Full transformed feature index to decode at startup.",
    )
    parser.add_argument(
        "--n-timepoints",
        type=int,
        default=128,
        help=(
            "Length of built-in signals when no fitted model supplies the "
            "training length (default: 128)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    decoder = None
    n_timepoints = int(args.n_timepoints)
    if args.model is not None:
        loaded = load_serialized_model(args.model)
        decoder = resolve_feature_decoder(loaded)
        n_timepoints = infer_training_length(decoder, default=n_timepoints)
    elif args.feature_index is not None:
        raise ValueError("--feature-index requires --model")

    signals = make_sample_signals(n_timepoints)
    if args.data is not None:
        user_signals = load_user_signals(args.data, args.labels)
        signals.update(user_signals)
        print(
            f"Loaded {len(user_signals)} signal(s) from {args.data}: "
            f"{', '.join(user_signals)}"
        )

    explorer = KernelExplorer(
        signals,
        decoder=decoder,
        initial_feature_index=args.feature_index,
    )
    explorer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
