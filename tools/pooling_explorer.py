#!/usr/bin/env python
"""Interactive MultiRocket pooling-operator explorer.

This standalone Matplotlib program recreates the logic of Figure 3 in:

    Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022).
    MultiRocket: multiple pooling operators and transformations for fast and
    effective time series classification. Data Mining and Knowledge Discovery,
    36, 1623-1646. https://doi.org/10.1007/s10618-022-00844-1

The display explains the four pooling features calculated from one convolution
output:

* PPV  -- proportion of positive values
* MPV  -- mean of positive values
* MIPV -- mean index of positive values
* LSPV -- longest stretch of positive values

The built-in example reproduces the counts and summary values shown in Figure 3:
60 positive values in an output of length 100, split into runs of 25 and 35.

The tool can also display any single trial from an in-memory array or a .npy,
.npz, .csv, .tsv, or whitespace-delimited text file. In signal mode, a selected
I-ROCKET kernel is applied to each trial. A fitted I-ROCKET model can be loaded
with joblib so that an actual transformed feature supplies the kernel, dilation,
padding, representation, and fitted bias.

Paper and implementation views
------------------------------
Figure 3 and the equations in the paper explain the four features on a
zero-thresholded convolution output Z. For an actual fitted feature, this
explorer displays

    Z = convolution - fitted_bias

so Z > 0 is exactly equivalent to convolution > fitted_bias.

There is a small but real discrepancy between the Figure 3/equation
interpretation and the published source implementation. The original
MultiRocket code, aeon, and the repaired I-ROCKET transform match one another,
but the source implementation accumulates MPV and counts LSPV differently in
some boundary and nonzero-bias cases. The optional "Compare exact
implementation" checkbox shows the stored implementation values beside the
Figure 3 values. The Figure 3 definition remains the default teaching view.

Command-line examples
---------------------
Built-in Figure 3 example::

    python pooling_explorer.py

Browse raw trials stored one per row::

    python pooling_explorer.py --data X_test.npy --labels-path y_test.npy

Use an NPZ archive::

    python pooling_explorer.py \
        --data dataset.npz --x-key X_test --y-key y_test --trial 12

Apply a manually specified kernel to each trial::

    python pooling_explorer.py \
        --data X_test.npy --kernel-index 27 --dilation 6 \
        --representation raw --padding valid --bias 0.25

Inspect a fitted I-ROCKET feature::

    python pooling_explorer.py \
        --data X_test.npy --labels-path y_test.npy \
        --model final_model.joblib --feature-index 340

Treat rows as precomputed convolution outputs::

    python pooling_explorer.py \
        --data convolution_outputs.npy --input-kind convolution --bias 0

Python API
----------
::

    from pooling_explorer import launch_pooling_explorer

    app = launch_pooling_explorer(
        X_test,
        y_test,
        model=result.final_model,
        feature_index=340,
        trial=12,
    )

    # The returned object exposes the Matplotlib figure.
    app.fig.savefig("pooling_feature_340_trial_12.png", dpi=200)

Notes
-----
* I-ROCKET is required only when raw signals are convolved or a fitted model is
  loaded. The built-in Figure 3 example and direct-convolution mode require only
  NumPy and Matplotlib.
* MultiRocket is univariate. Three-dimensional arrays are accepted only by
  selecting one channel.
* MIPV is reported in zero-based local convolution indices, as in the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import CheckButtons, Slider, TextBox
import numpy as np


# Okabe-Ito colors. The package values are used when I-ROCKET is importable.
_FALLBACK_OI = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
_FALLBACK_POOLING_COLORS = {
    "PPV": "#0072B2",
    "MPV": "#56B4E9",
    "MIPV": "#D55E00",
    "LSPV": "#E69F00",
}

try:  # Lazy use is still checked before signal convolution.
    from interp_rocket import (  # type: ignore
        OI as _PACKAGE_OI,
        POOLING_COLORS as _PACKAGE_POOLING_COLORS,
        compute_activation_map as _compute_activation_map,
    )
except ImportError:  # Built-in and direct-convolution modes remain available.
    _PACKAGE_OI = None
    _PACKAGE_POOLING_COLORS = None
    _compute_activation_map = None

OI = tuple(_PACKAGE_OI) if _PACKAGE_OI is not None else _FALLBACK_OI
POOLING_COLORS = (
    dict(_PACKAGE_POOLING_COLORS)
    if _PACKAGE_POOLING_COLORS is not None
    else dict(_FALLBACK_POOLING_COLORS)
)
POOLING_NAMES = ("PPV", "MPV", "MIPV", "LSPV")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositiveRun:
    """One contiguous run of positive values in the paper-centered output."""

    start: int
    stop: int  # inclusive
    count: int
    value_sum: float
    index_sum: int


@dataclass
class FeatureSpec:
    """Kernel parameters needed to generate one convolution output."""

    kernel_index: int = 0
    dilation: int = 1
    bias: float = 0.0
    padding: str = "same"
    representation: str = "raw"
    feature_index: Optional[int] = None
    pooling_op: Optional[str] = None
    bias_rank: Optional[int] = None

    @classmethod
    def from_decoded_feature(cls, info: Mapping[str, Any]) -> "FeatureSpec":
        """Construct a specification from ``decode_feature_index`` output."""
        return cls(
            kernel_index=int(info["kernel_index"]),
            dilation=int(info["dilation"]),
            bias=float(info["bias"]),
            padding=str(info.get("padding_mode", info.get("padding", "same"))),
            representation=str(info.get("representation", "raw")),
            feature_index=(
                int(info["feature_index"])
                if info.get("feature_index") is not None
                else None
            ),
            pooling_op=(
                str(info["pooling_op"])
                if info.get("pooling_op") is not None
                else None
            ),
            bias_rank=(
                int(info["bias_rank_within_kernel"])
                if info.get("bias_rank_within_kernel") is not None
                else None
            ),
        )

    def validate(self) -> None:
        """Validate the specification before convolution."""
        if isinstance(self.kernel_index, bool) or not isinstance(
            self.kernel_index, (int, np.integer)
        ):
            raise TypeError("kernel_index must be an integer")
        if not 0 <= int(self.kernel_index) < 84:
            raise ValueError("kernel_index must be between 0 and 83")
        if isinstance(self.dilation, bool) or not isinstance(
            self.dilation, (int, np.integer)
        ):
            raise TypeError("dilation must be an integer")
        if int(self.dilation) < 1:
            raise ValueError("dilation must be positive")
        if self.padding not in {"same", "valid"}:
            raise ValueError("padding must be 'same' or 'valid'")
        if self.representation not in {"raw", "diff"}:
            raise ValueError("representation must be 'raw' or 'diff'")
        if not np.isfinite(float(self.bias)):
            raise ValueError("bias must be finite")

    def identity(self) -> str:
        """Return a compact, unambiguous feature description."""
        parts = []
        if self.feature_index is not None:
            parts.append(f"F{self.feature_index}")
        parts.extend(
            [
                f"K{self.kernel_index}",
                f"d={self.dilation}",
            ]
        )
        if self.bias_rank is not None:
            parts.append(f"b{self.bias_rank}")
        if self.pooling_op:
            parts.append(self.pooling_op)
        parts.extend(
            [
                self.representation,
                self.padding,
                f"bias={self.bias:.5g}",
            ]
        )
        return " ".join(parts)


@dataclass(frozen=True)
class PreparedTrial:
    """All arrays needed to render one trial."""

    raw_trial: np.ndarray
    analysis_signal: np.ndarray
    convolution: np.ndarray
    centered_output: np.ndarray
    source_indices: np.ndarray
    trial_index: int
    trial_label: Optional[Any]


# ---------------------------------------------------------------------------
# Figure 3 and pooling calculations
# ---------------------------------------------------------------------------


def make_figure3_example() -> np.ndarray:
    """Return a deterministic length-100 output matching Figure 3 summaries.

    Positive values occur at indices 1--25 and 60--94. Their sums are exactly
    13.37 and 12.95, respectively. Therefore:

    * PPV  = 60 / 100 = 0.60
    * MPV  = (13.37 + 12.95) / 60 = 0.438666...
    * MIPV = (325 + 2695) / 60 = 50.333...
    * LSPV = 35
    """
    output = np.zeros(100, dtype=np.float64)
    output[0] = -0.08

    first = np.sin(np.pi * np.arange(1, 26) / 26.0)
    output[1:26] = first * (13.37 / first.sum())

    middle = np.sin(np.pi * np.arange(1, 35) / 35.0)
    output[26:60] = -0.90 * middle

    second = np.sin(np.pi * np.arange(1, 36) / 36.0)
    output[60:95] = second * (12.95 / second.sum())

    tail = np.sin(np.pi * np.arange(1, 6) / 6.0)
    output[95:100] = -0.25 * tail
    return output.astype(np.float32)


def find_positive_runs(z: np.ndarray) -> Tuple[PositiveRun, ...]:
    """Return contiguous runs where ``z > 0`` using paper definitions."""
    z = _validate_vector(z, name="z")
    mask = z > 0
    runs = []
    start: Optional[int] = None

    for index, is_positive in enumerate(mask):
        if is_positive and start is None:
            start = index
        is_last = index == z.size - 1
        if start is not None and ((not is_positive) or is_last):
            stop = index if (is_positive and is_last) else index - 1
            indices = np.arange(start, stop + 1, dtype=np.int64)
            runs.append(
                PositiveRun(
                    start=start,
                    stop=stop,
                    count=stop - start + 1,
                    value_sum=float(np.sum(z[start : stop + 1], dtype=np.float64)),
                    index_sum=int(indices.sum()),
                )
            )
            start = None

    return tuple(runs)


def paper_pooling(z: np.ndarray) -> Dict[str, float]:
    """Calculate PPV, MPV, MIPV, and LSPV as illustrated in Figure 3.

    Parameters
    ----------
    z : ndarray, shape (n_values,)
        A zero-thresholded convolution output. For a fitted feature, use
        ``z = convolution - bias``.
    """
    z = _validate_vector(z, name="z")
    positive = z > 0
    positive_indices = np.flatnonzero(positive)
    count = int(positive_indices.size)
    runs = find_positive_runs(z)

    return {
        "PPV": float(count / z.size),
        "MPV": float(np.mean(z[positive], dtype=np.float64)) if count else 0.0,
        "MIPV": float(np.mean(positive_indices, dtype=np.float64)) if count else -1.0,
        "LSPV": float(max((run.count for run in runs), default=0)),
    }


def reference_implementation_pooling(
    convolution: np.ndarray,
    bias: float,
) -> Dict[str, float]:
    """Reproduce the original/aeon/I-ROCKET implementation calculations.

    This deliberately preserves the implemented MPV accumulation and LSPV
    boundary-distance calculation. It is provided for comparison with the
    paper definitions, not as a replacement for them in the teaching display.
    """
    convolution = _validate_vector(convolution, name="convolution")
    bias = float(bias)
    if not np.isfinite(bias):
        raise ValueError("bias must be finite")

    count = 0
    last_nonpositive = 0
    maximum_stretch = 0.0
    index_sum = 0
    value_sum = 0.0

    for local_index, value in enumerate(convolution):
        if value > bias:
            count += 1
            index_sum += local_index
            # Exact operation in the original MultiRocket and aeon code.
            value_sum += float(value) + bias
        elif value < bias:
            stretch = local_index - last_nonpositive
            if stretch > maximum_stretch:
                maximum_stretch = float(stretch)
            last_nonpositive = local_index

    final_stretch = convolution.size - 1 - last_nonpositive
    if final_stretch > maximum_stretch:
        maximum_stretch = float(final_stretch)

    return {
        "PPV": float(count / convolution.size),
        "MPV": float(value_sum / count) if count else 0.0,
        "MIPV": float(index_sum / count) if count else -1.0,
        "LSPV": float(maximum_stretch),
    }


def _validate_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


# ---------------------------------------------------------------------------
# Dataset and model loading
# ---------------------------------------------------------------------------


def _load_numeric_file(
    path: Union[str, Path],
    *,
    key: Optional[str] = None,
) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is not None:
                if key not in archive:
                    raise KeyError(
                        f"{key!r} was not found in {path}; available keys: "
                        f"{sorted(archive.files)}"
                    )
                return np.asarray(archive[key])
            if len(archive.files) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays {sorted(archive.files)}; "
                    "specify a key"
                )
            return np.asarray(archive[archive.files[0]])
    if suffix == ".csv":
        return np.asarray(np.loadtxt(path, delimiter=","))
    if suffix == ".tsv":
        return np.asarray(np.loadtxt(path, delimiter="\t"))
    if suffix in {".txt", ".dat"}:
        return np.asarray(np.loadtxt(path))
    raise ValueError(
        f"Unsupported file type {suffix!r}. Use .npy, .npz, .csv, .tsv, "
        "or .txt."
    )


def _normalise_trials(
    array: np.ndarray,
    *,
    channel: int = 0,
    channel_axis: int = 1,
) -> np.ndarray:
    """Return a finite float32 matrix shaped (n_trials, n_timepoints)."""
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    elif array.ndim == 3:
        if channel_axis not in {1, 2}:
            raise ValueError("channel_axis must be 1 or 2 for 3D input")
        n_channels = array.shape[channel_axis]
        if not 0 <= int(channel) < n_channels:
            raise IndexError(
                f"channel={channel} is outside the available range "
                f"0..{n_channels - 1}"
            )
        array = np.take(array, int(channel), axis=channel_axis)
    if array.ndim != 2:
        raise ValueError(
            "Data must be shaped (time,), (trials, time), or a 3D univariate/"
            "multichannel array with one selected channel. "
            f"Received shape {array.shape}."
        )
    try:
        array = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Data must be numeric") from exc
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("Data must contain at least one nonempty trial")
    if not np.isfinite(array).all():
        raise ValueError("Data contain NaN or infinite values")
    return array


def _load_labels_file(path: Union[str, Path]) -> np.ndarray:
    """Load numeric or text labels from NumPy and delimited text files."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays {sorted(archive.files)}; "
                    "store labels in one array or use the dataset NPZ y-key."
                )
            return np.asarray(archive[archive.files[0]])
    delimiters = {".csv": ",", ".tsv": "\t", ".txt": None, ".dat": None}
    if suffix in delimiters:
        return np.asarray(np.loadtxt(path, delimiter=delimiters[suffix], dtype=str))
    raise ValueError(
        f"Unsupported label-file type {suffix!r}. Use .npy, .npz, .csv, "
        ".tsv, or .txt."
    )


def load_dataset(
    path: Union[str, Path],
    *,
    labels_path: Optional[Union[str, Path]] = None,
    x_key: str = "X",
    y_key: str = "y",
    channel: int = 0,
    channel_axis: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load trials and optional labels from common array/text formats."""
    path = Path(path)
    labels: Optional[np.ndarray] = None

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if x_key not in archive:
                candidate_keys = [key for key in archive.files if key != y_key]
                if len(candidate_keys) == 1:
                    selected_x_key = candidate_keys[0]
                else:
                    raise KeyError(
                        f"{x_key!r} was not found in {path}; available keys: "
                        f"{sorted(archive.files)}"
                    )
            else:
                selected_x_key = x_key
            X_raw = np.asarray(archive[selected_x_key])
            if y_key in archive:
                labels = np.asarray(archive[y_key])
    else:
        X_raw = _load_numeric_file(path)

    X = _normalise_trials(
        X_raw,
        channel=channel,
        channel_axis=channel_axis,
    )

    if labels_path is not None:
        labels = _load_labels_file(labels_path)

    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if labels.size != X.shape[0]:
            raise ValueError(
                f"Labels contain {labels.size} values but X has "
                f"{X.shape[0]} trials"
            )
    return X, labels


def load_serialized_model(path: Union[str, Path]) -> Any:
    """Load a fitted estimator or nested-CV result with joblib."""
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - sklearn normally provides it
        raise ImportError("Loading a model requires joblib") from exc
    return joblib.load(Path(path))


def resolve_feature_decoder(model: Any) -> Any:
    """Find an object exposing ``decode_feature_index`` in a saved result."""
    queue = [model]
    visited = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))

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
                queue.append(getattr(candidate, attribute))

        named_steps = getattr(candidate, "named_steps", None)
        if named_steps is not None:
            queue.extend(reversed(tuple(named_steps.values())))

    raise TypeError(
        "The loaded object does not contain a fitted I-ROCKET transformer "
        "with decode_feature_index()."
    )


# ---------------------------------------------------------------------------
# Trial preparation
# ---------------------------------------------------------------------------


def prepare_trial(
    X: np.ndarray,
    trial_index: int,
    *,
    labels: Optional[np.ndarray],
    input_kind: str,
    feature: FeatureSpec,
) -> PreparedTrial:
    """Prepare one direct convolution or one kernel-transformed signal."""
    if not 0 <= int(trial_index) < X.shape[0]:
        raise IndexError(
            f"trial_index={trial_index} is outside 0..{X.shape[0] - 1}"
        )
    raw_trial = np.asarray(X[int(trial_index)], dtype=np.float32)
    label = None if labels is None else labels[int(trial_index)]

    if input_kind == "convolution":
        convolution = raw_trial.copy()
        analysis = raw_trial.copy()
        source_indices = np.arange(raw_trial.size, dtype=np.float32)
    elif input_kind == "signal":
        if _compute_activation_map is None:
            raise ImportError(
                "Signal mode requires the current I-ROCKET package because "
                "it delegates convolution to interp_rocket.compute_activation_map."
            )
        feature.validate()
        if feature.representation == "raw":
            analysis = raw_trial
        else:
            if raw_trial.size < 2:
                raise ValueError("First differencing requires at least two timepoints")
            analysis = np.diff(raw_trial).astype(np.float32)

        convolution, _, source_indices = _compute_activation_map(
            analysis,
            int(feature.kernel_index),
            int(feature.dilation),
            float(feature.bias),
            padding=feature.padding,
            representation=feature.representation,
        )
        convolution = np.asarray(convolution, dtype=np.float32)
        source_indices = np.asarray(source_indices, dtype=np.float32)
    else:
        raise ValueError("input_kind must be 'signal' or 'convolution'")

    centered = convolution - np.float32(feature.bias)
    return PreparedTrial(
        raw_trial=raw_trial,
        analysis_signal=np.asarray(analysis, dtype=np.float32),
        convolution=convolution,
        centered_output=np.asarray(centered, dtype=np.float32),
        source_indices=source_indices,
        trial_index=int(trial_index),
        trial_label=label,
    )


# ---------------------------------------------------------------------------
# Interactive application
# ---------------------------------------------------------------------------


class PoolingExplorer:
    """Interactive Figure-3-style explorer for MultiRocket pooling features."""

    def __init__(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        *,
        input_kind: str = "signal",
        feature: Optional[FeatureSpec] = None,
        decoder: Any = None,
        trial: int = 0,
        compare_implementation: bool = False,
        max_run_annotations: int = 6,
        title: Optional[str] = None,
    ) -> None:
        self.X = _normalise_trials(X)
        self.labels = None if labels is None else np.asarray(labels).reshape(-1)
        if self.labels is not None and self.labels.size != self.X.shape[0]:
            raise ValueError("labels must contain one value per trial")
        if input_kind not in {"signal", "convolution"}:
            raise ValueError("input_kind must be 'signal' or 'convolution'")
        if isinstance(max_run_annotations, bool) or not isinstance(
            max_run_annotations, (int, np.integer)
        ):
            raise TypeError("max_run_annotations must be an integer")
        if int(max_run_annotations) < 1:
            raise ValueError("max_run_annotations must be positive")

        self.input_kind = input_kind
        self.feature = feature if feature is not None else FeatureSpec()
        self.feature.validate()
        self.decoder = decoder
        self.trial_index = int(trial)
        if not 0 <= self.trial_index < self.X.shape[0]:
            raise IndexError("Initial trial is outside the dataset")
        self.compare_implementation = bool(compare_implementation)
        self.max_run_annotations = int(max_run_annotations)
        self.custom_title = title
        self.status_message = ""
        self._updating = False

        self._build_figure()
        self._connect_callbacks()
        self.draw()

    # ---------------------------- figure layout ----------------------------

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 9.5))
        try:
            self.fig.canvas.manager.set_window_title(
                "MultiRocket Pooling Explorer"
            )
        except AttributeError:  # noninteractive backend
            pass

        grid = self.fig.add_gridspec(
            3,
            4,
            height_ratios=(0.78, 2.35, 1.02),
            hspace=0.50,
            wspace=0.18,
            top=0.88,
            bottom=0.20,
            left=0.07,
            right=0.98,
        )
        self.ax_input = self.fig.add_subplot(grid[0, :])
        self.ax_output = self.fig.add_subplot(grid[1, :])
        self.feature_axes = [self.fig.add_subplot(grid[2, i]) for i in range(4)]

        self.trial_slider: Optional[Slider] = None
        if self.X.shape[0] > 1:
            slider_ax = self.fig.add_axes([0.10, 0.105, 0.57, 0.028])
            self.trial_slider = Slider(
                slider_ax,
                "Trial",
                0,
                self.X.shape[0] - 1,
                valinit=self.trial_index,
                valstep=1,
                valfmt="%d",
            )
        else:
            self.fig.text(0.10, 0.115, "Trial 0 (single row)", fontsize=9)

        check_ax = self.fig.add_axes([0.73, 0.075, 0.23, 0.075])
        self.compare_check = CheckButtons(
            check_ax,
            ["Compare exact implementation"],
            [self.compare_implementation],
        )
        if hasattr(self.compare_check, "set_frame_props"):
            self.compare_check.set_frame_props(
                {"edgecolor": POOLING_COLORS["PPV"]}
            )
        elif hasattr(self.compare_check, "rectangles"):
            # Matplotlib < 3.7 compatibility.
            for rectangle in self.compare_check.rectangles:
                rectangle.set_edgecolor(POOLING_COLORS["PPV"])

        self.feature_box: Optional[TextBox] = None
        if self.decoder is not None:
            feature_ax = self.fig.add_axes([0.73, 0.145, 0.12, 0.035])
            initial = (
                ""
                if self.feature.feature_index is None
                else str(self.feature.feature_index)
            )
            self.feature_box = TextBox(
                feature_ax,
                "Feature index ",
                initial=initial,
            )
            self.fig.text(
                0.86,
                0.153,
                "Enter a full transformed-column index",
                fontsize=8,
                color="0.35",
            )

        self.status_text = self.fig.text(
            0.07,
            0.035,
            "",
            fontsize=9,
            color="0.25",
            ha="left",
            va="bottom",
        )
        self.help_text = self.fig.text(
            0.98,
            0.035,
            "Left/right: change trial    S: save PNG",
            fontsize=8,
            color="0.45",
            ha="right",
            va="bottom",
        )

    def _connect_callbacks(self) -> None:
        if self.trial_slider is not None:
            self.trial_slider.on_changed(self._on_trial_change)
        self.compare_check.on_clicked(self._on_compare_change)
        if self.feature_box is not None:
            self.feature_box.on_submit(self._on_feature_submit)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    # ------------------------------ callbacks ------------------------------

    def _on_trial_change(self, value: float) -> None:
        if self._updating:
            return
        self.trial_index = int(value)
        self.status_message = ""
        self.draw()

    def _on_compare_change(self, _label: str) -> None:
        self.compare_implementation = not self.compare_implementation
        self.draw()

    def _on_feature_submit(self, text: str) -> None:
        if self._updating or self.decoder is None:
            return
        try:
            feature_index = int(text.strip())
            info = dict(self.decoder.decode_feature_index(feature_index))
            if info.get("feature_index") is None:
                info["feature_index"] = feature_index
            self.feature = FeatureSpec.from_decoded_feature(info)
            self.feature.validate()
            self.input_kind = "signal"
            self.status_message = f"Loaded {self.feature.identity()}"
        except Exception as exc:  # interactive feedback instead of traceback
            self.status_message = f"Could not load feature: {exc}"
        self.draw()

    def _on_key_press(self, event: Any) -> None:
        if event.key in {"left", "down"}:
            self.set_trial(max(0, self.trial_index - 1))
        elif event.key in {"right", "up"}:
            self.set_trial(min(self.X.shape[0] - 1, self.trial_index + 1))
        elif event.key and event.key.lower() == "s":
            filename = (
                f"pooling_trial_{self.trial_index}"
                + (
                    ""
                    if self.feature.feature_index is None
                    else f"_feature_{self.feature.feature_index}"
                )
                + ".png"
            )
            self.fig.savefig(filename, dpi=200, bbox_inches="tight")
            self.status_message = f"Saved {filename}"
            self.status_text.set_text(self.status_message)
            self.fig.canvas.draw_idle()

    def set_trial(self, trial_index: int) -> None:
        """Set the displayed trial programmatically."""
        trial_index = int(trial_index)
        if not 0 <= trial_index < self.X.shape[0]:
            raise IndexError("trial_index is outside the dataset")
        self.trial_index = trial_index
        if self.trial_slider is not None:
            self._updating = True
            try:
                self.trial_slider.set_val(trial_index)
            finally:
                self._updating = False
        self.draw()

    # ------------------------------ drawing ------------------------------

    def draw(self) -> None:
        """Redraw all panels for the current trial and feature."""
        try:
            prepared = prepare_trial(
                self.X,
                self.trial_index,
                labels=self.labels,
                input_kind=self.input_kind,
                feature=self.feature,
            )
            paper_values = paper_pooling(prepared.centered_output)
            implementation_values = reference_implementation_pooling(
                prepared.convolution,
                self.feature.bias,
            )
            runs = find_positive_runs(prepared.centered_output)
            self._draw_input(prepared)
            self._draw_output(prepared, runs, paper_values)
            self._draw_feature_boxes(
                prepared,
                runs,
                paper_values,
                implementation_values,
            )
            self._draw_titles(prepared, runs)
            self.status_text.set_text(self.status_message)
            self.status_text.set_color("0.25")
        except Exception as exc:
            self.status_message = str(exc)
            self.status_text.set_text(f"Error: {exc}")
            self.status_text.set_color(POOLING_COLORS["MIPV"])
        self.fig.canvas.draw_idle()

    def _draw_input(self, prepared: PreparedTrial) -> None:
        ax = self.ax_input
        ax.clear()
        x = np.arange(prepared.raw_trial.size)
        ax.plot(x, prepared.raw_trial, color=OI[7], linewidth=1.3)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.set_xlim(0, max(prepared.raw_trial.size - 1, 1))
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.16)

        if self.input_kind == "convolution":
            title = "Input row (treated directly as convolution output C)"
        elif self.feature.representation == "diff":
            title = "Raw trial (kernel is applied to its first difference)"
        else:
            title = "Raw trial"
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Timepoint")

    def _draw_output(
        self,
        prepared: PreparedTrial,
        runs: Tuple[PositiveRun, ...],
        paper_values: Mapping[str, float],
    ) -> None:
        ax = self.ax_output
        ax.clear()
        z = prepared.centered_output
        x = np.arange(z.size)
        positive = z > 0

        ax.plot(x, z, color=OI[7], linewidth=1.6, zorder=3)
        ax.axhline(0.0, color="0.15", linestyle="--", linewidth=1.0, zorder=2)
        ax.fill_between(
            x,
            0.0,
            z,
            where=positive,
            interpolate=True,
            color=POOLING_COLORS["PPV"],
            alpha=0.48,
            label="Z > 0 (convolution > bias)",
            zorder=1,
        )

        longest = max(runs, key=lambda run: run.count, default=None)
        if longest is not None:
            ax.axvspan(
                longest.start,
                longest.stop,
                color=POOLING_COLORS["LSPV"],
                alpha=0.13,
                label="Longest positive stretch",
                zorder=0,
            )

        mipv = float(paper_values["MIPV"])
        if mipv >= 0:
            ax.axvline(
                mipv,
                color=POOLING_COLORS["MIPV"],
                linestyle=":",
                linewidth=1.7,
                label=f"MIPV = {mipv:.2f}",
                zorder=4,
            )

        selected = self._runs_to_annotate(runs)
        run_numbers = {id(run): index + 1 for index, run in enumerate(runs)}
        y_span = max(float(np.ptp(z)), 1e-6)
        for run in selected:
            run_number = run_numbers[id(run)]
            midpoint = 0.5 * (run.start + run.stop)
            run_max = float(np.max(z[run.start : run.stop + 1]))
            ax.text(
                midpoint,
                run_max + 0.035 * y_span,
                f"a{run_number}={run.value_sum:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.15",
            )
            transform = ax.get_xaxis_transform()
            bracket_y = 0.07
            ax.plot(
                [run.start, run.stop],
                [bracket_y, bracket_y],
                transform=transform,
                color="0.25",
                linewidth=1.0,
                clip_on=False,
            )
            ax.plot(
                [run.start, run.start],
                [bracket_y - 0.012, bracket_y + 0.012],
                transform=transform,
                color="0.25",
                linewidth=1.0,
                clip_on=False,
            )
            ax.plot(
                [run.stop, run.stop],
                [bracket_y - 0.012, bracket_y + 0.012],
                transform=transform,
                color="0.25",
                linewidth=1.0,
                clip_on=False,
            )
            ax.text(
                midpoint,
                bracket_y + 0.013,
                f"p{run_number}={run.count}",
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.20",
            )
            ax.text(
                midpoint,
                0.012,
                f"i{run_number}={run.index_sum}",
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="0.35",
            )

        ax.set_xlim(0, max(z.size - 1, 1))
        y_min = float(np.min(z))
        y_max = float(np.max(z))
        margin = max(0.12 * max(y_max - y_min, 1e-6), 0.08)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel("Local convolution index (zero-based; used by MIPV)")
        ax.set_ylabel("Z = convolution - bias")
        ax.grid(True, alpha=0.14)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

        if len(runs) > len(selected):
            ax.text(
                0.01,
                0.98,
                f"{len(runs)} positive runs; labeling the "
                f"{len(selected)} longest",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="0.35",
            )

    def _runs_to_annotate(
        self,
        runs: Tuple[PositiveRun, ...],
    ) -> Tuple[PositiveRun, ...]:
        if len(runs) <= self.max_run_annotations:
            return runs
        selected = sorted(
            runs,
            key=lambda run: (run.count, run.value_sum),
            reverse=True,
        )[: self.max_run_annotations]
        return tuple(sorted(selected, key=lambda run: run.start))

    def _draw_feature_boxes(
        self,
        prepared: PreparedTrial,
        runs: Tuple[PositiveRun, ...],
        paper_values: Mapping[str, float],
        implementation_values: Mapping[str, float],
    ) -> None:
        z = prepared.centered_output
        positive = z > 0
        positive_indices = np.flatnonzero(positive)
        count = int(positive_indices.size)
        total_value = float(np.sum(z[positive], dtype=np.float64))
        total_index = int(positive_indices.sum()) if count else 0
        longest = max((run.count for run in runs), default=0)

        formulas = {
            "PPV": f"m / n = {count} / {z.size}",
            "MPV": f"A / m = {total_value:.4g} / {count}" if count else "no positive values",
            "MIPV": f"I / m = {total_index} / {count}" if count else "no positive values",
            "LSPV": f"max(p_r) = {longest}",
        }
        descriptions = {
            "PPV": "How often the kernel response exceeds its bias",
            "MPV": "Mean magnitude of the positive, bias-centered response",
            "MIPV": "Average location of the positive response",
            "LSPV": "Duration of the longest contiguous positive response",
        }

        for ax, name in zip(self.feature_axes, POOLING_NAMES):
            self._draw_feature_box(
                ax,
                name=name,
                paper_value=float(paper_values[name]),
                formula=formulas[name],
                description=descriptions[name],
                implementation_value=(
                    float(implementation_values[name])
                    if self.compare_implementation
                    else None
                ),
            )

    def _draw_feature_box(
        self,
        ax: Any,
        *,
        name: str,
        paper_value: float,
        formula: str,
        description: str,
        implementation_value: Optional[float],
    ) -> None:
        ax.clear()
        ax.set_axis_off()
        color = POOLING_COLORS[name]
        patch = FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.90,
            boxstyle="round,pad=0.025,rounding_size=0.04",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
        )
        ax.add_patch(patch)
        ax.text(
            0.07,
            0.82,
            name,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
        ax.text(
            0.93,
            0.82,
            f"{paper_value:.5g}",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color="0.10",
            ha="right",
            va="center",
        )
        ax.text(
            0.07,
            0.58,
            formula,
            transform=ax.transAxes,
            fontsize=9.2,
            color="0.20",
            ha="left",
            va="center",
        )
        ax.text(
            0.07,
            0.34,
            description,
            transform=ax.transAxes,
            fontsize=8.2,
            color="0.35",
            ha="left",
            va="center",
            wrap=True,
        )
        if implementation_value is not None:
            difference = implementation_value - paper_value
            ax.text(
                0.07,
                0.14,
                f"Exact implementation: {implementation_value:.5g} "
                f"(difference {difference:+.3g})",
                transform=ax.transAxes,
                fontsize=7.8,
                color=(
                    POOLING_COLORS["MIPV"]
                    if not np.isclose(difference, 0.0)
                    else "0.40"
                ),
                ha="left",
                va="center",
            )

    def _draw_titles(
        self,
        prepared: PreparedTrial,
        runs: Tuple[PositiveRun, ...],
    ) -> None:
        if self.custom_title:
            main_title = self.custom_title
        elif self.input_kind == "convolution" and self.X.shape[0] == 1:
            main_title = "MultiRocket pooling operators: Tan's Figure 3 reproduction"
        else:
            main_title = "MultiRocket pooling operators on a single trial"

        label_text = (
            ""
            if prepared.trial_label is None
            else f" | label={prepared.trial_label}"
        )
        if self.input_kind == "signal":
            identity = self.feature.identity()
        else:
            identity = f"direct convolution output | threshold={self.feature.bias:.5g}"

        self.fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.965)
        self.fig.text(
            0.5,
            0.925,
            f"Trial {prepared.trial_index}/{self.X.shape[0] - 1}{label_text} | "
            f"{identity} | positive runs={len(runs)}",
            ha="center",
            va="center",
            fontsize=9.5,
            color="0.28",
        )


# ---------------------------------------------------------------------------
# Public launcher and CLI
# ---------------------------------------------------------------------------


def launch_pooling_explorer(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    *,
    model: Any = None,
    feature_index: Optional[int] = None,
    input_kind: str = "signal",
    kernel_index: int = 0,
    dilation: int = 1,
    bias: float = 0.0,
    padding: str = "same",
    representation: str = "raw",
    trial: int = 0,
    compare_implementation: bool = False,
    max_run_annotations: int = 6,
    title: Optional[str] = None,
    show: bool = True,
) -> PoolingExplorer:
    """Launch the explorer from NumPy arrays or use the built-in example.

    When ``X`` is omitted, the exact Figure-3-style built-in output is shown in
    direct-convolution mode. When ``model`` is provided, ``feature_index`` is
    required and its decoded parameters replace the manual kernel arguments.
    """
    decoder = None

    if X is None:
        X = make_figure3_example()[np.newaxis, :]
        y = None
        input_kind = "convolution"
        bias = 0.0
        if title is None:
            title = "MultiRocket pooling operators: Tan's Figure 3 reproduction"
    else:
        X = _normalise_trials(X)

    if model is not None:
        if feature_index is None:
            raise ValueError("feature_index is required when model is supplied")
        decoder = resolve_feature_decoder(model)
        info = dict(decoder.decode_feature_index(int(feature_index)))
        if info.get("feature_index") is None:
            info["feature_index"] = int(feature_index)
        feature = FeatureSpec.from_decoded_feature(info)
        input_kind = "signal"
    else:
        feature = FeatureSpec(
            kernel_index=int(kernel_index),
            dilation=int(dilation),
            bias=float(bias),
            padding=padding,
            representation=representation,
        )

    app = PoolingExplorer(
        X,
        y,
        input_kind=input_kind,
        feature=feature,
        decoder=decoder,
        trial=trial,
        compare_implementation=compare_implementation,
        max_run_annotations=max_run_annotations,
        title=title,
    )
    if show:
        plt.show()
    return app


def run_self_test() -> None:
    """Run lightweight numerical checks without opening a window."""
    z = make_figure3_example()
    runs = find_positive_runs(z)
    values = paper_pooling(z)

    assert len(runs) == 2
    assert (runs[0].start, runs[0].stop, runs[0].count) == (1, 25, 25)
    assert (runs[1].start, runs[1].stop, runs[1].count) == (60, 94, 35)
    assert np.isclose(runs[0].value_sum, 13.37, atol=1e-5)
    assert np.isclose(runs[1].value_sum, 12.95, atol=1e-5)
    assert runs[0].index_sum == 325
    assert runs[1].index_sum == 2695
    assert np.isclose(values["PPV"], 0.60)
    assert np.isclose(values["MPV"], 26.32 / 60.0, atol=1e-7)
    assert np.isclose(values["MIPV"], 3020 / 60.0, atol=1e-7)
    assert values["LSPV"] == 35.0

    reference = reference_implementation_pooling(z, 0.0)
    assert np.isclose(reference["PPV"], values["PPV"])
    assert np.isclose(reference["MPV"], values["MPV"], atol=1e-7)
    assert np.isclose(reference["MIPV"], values["MIPV"], atol=1e-7)
    # The comparison is intentional: the reference code counts boundary
    # distances rather than the paper's exact number of positive samples.
    assert reference["LSPV"] == 36.0

    print("pooling_explorer self-test passed")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Figure-3-style explanation of MultiRocket PPV, MPV, "
            "MIPV, and LSPV features."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=None,
        help=".npy, .npz, .csv, .tsv, or .txt file containing trials",
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help=(
            "Optional .npy, .npz, .csv, .tsv, or .txt file containing "
            "one numeric or text label per trial"
        ),
    )
    parser.add_argument("--x-key", default="X", help="Trial-array key in an NPZ file")
    parser.add_argument("--y-key", default="y", help="Label-array key in an NPZ file")
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel selected from a 3D array",
    )
    parser.add_argument(
        "--channel-axis",
        type=int,
        choices=(1, 2),
        default=1,
        help="Channel axis for a 3D array",
    )
    parser.add_argument("--trial", type=int, default=0, help="Initial trial index")
    parser.add_argument(
        "--input-kind",
        choices=("signal", "convolution"),
        default="signal",
        help="Whether data rows are raw signals or precomputed convolution outputs",
    )
    parser.add_argument("--model", default=None, help="Joblib file containing a fitted model")
    parser.add_argument(
        "--feature-index",
        type=int,
        default=None,
        help="Full transformed-column index decoded from the fitted model",
    )
    parser.add_argument("--kernel-index", type=int, default=0)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--padding", choices=("same", "valid"), default="same")
    parser.add_argument(
        "--representation",
        choices=("raw", "diff"),
        default="raw",
    )
    parser.add_argument(
        "--compare-implementation",
        action="store_true",
        help="Initially show exact original/aeon/I-ROCKET feature values",
    )
    parser.add_argument(
        "--max-run-labels",
        type=int,
        default=6,
        help="Maximum number of positive runs annotated on the curve",
    )
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--save", default=None, help="Save the initial figure to this path")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the Matplotlib window",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run numerical checks and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    if args.data is None:
        if args.model is not None:
            raise ValueError("--model requires --data")
        app = launch_pooling_explorer(
            trial=0,
            compare_implementation=args.compare_implementation,
            max_run_annotations=args.max_run_labels,
            title=args.title,
            show=False,
        )
    else:
        X, y = load_dataset(
            args.data,
            labels_path=args.labels_path,
            x_key=args.x_key,
            y_key=args.y_key,
            channel=args.channel,
            channel_axis=args.channel_axis,
        )
        model = None if args.model is None else load_serialized_model(args.model)
        app = launch_pooling_explorer(
            X,
            y,
            model=model,
            feature_index=args.feature_index,
            input_kind=args.input_kind,
            kernel_index=args.kernel_index,
            dilation=args.dilation,
            bias=args.bias,
            padding=args.padding,
            representation=args.representation,
            trial=args.trial,
            compare_implementation=args.compare_implementation,
            max_run_annotations=args.max_run_labels,
            title=args.title,
            show=False,
        )

    if args.save:
        app.fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved {args.save}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
