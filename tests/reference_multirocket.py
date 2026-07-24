"""Independent, readable reference calculations for transform tests.

These functions intentionally favor clarity over speed. They encode the
univariate MultiRocket behavior used by the original implementation and aeon,
including alternating same/valid pooling regions and the established
first-difference alignment.
"""

from itertools import combinations

import numpy as np


INDICES = np.asarray(list(combinations(range(9), 3)), dtype=np.int32)
KERNELS = np.full((84, 9), -1.0, dtype=np.float32)
for _kernel_index, _positive_indices in enumerate(INDICES):
    KERNELS[_kernel_index, _positive_indices] = 2.0


def quantiles(n):
    phi = (np.sqrt(5.0) + 1.0) / 2.0
    return np.asarray([((i + 1) * phi) % 1.0 for i in range(n)], dtype=np.float32)


def convolution(x, kernel_index, dilation, *, is_first_difference=False):
    """Direct zero-padded convolution in the established alignment."""
    x = np.asarray(x, dtype=np.float32)
    dilation = int(dilation)

    if is_first_difference:
        offsets = np.asarray(
            [
                -4 * dilation + 1,
                -3 * dilation + 1,
                -2 * dilation + 1,
                -dilation + 1,
                0,
                dilation,
                2 * dilation,
                3 * dilation,
                4 * dilation,
            ],
            dtype=np.int32,
        )
    else:
        offsets = np.arange(-4, 5, dtype=np.int32) * dilation

    output = np.zeros(x.size, dtype=np.float32)
    kernel = KERNELS[int(kernel_index)]
    for output_index in range(x.size):
        value = np.float32(0.0)
        for weight, offset in zip(kernel, offsets):
            input_index = output_index + int(offset)
            if 0 <= input_index < x.size:
                value = np.float32(value + np.float32(weight * x[input_index]))
        output[output_index] = value
    return output


def fit_biases(X, dilations, features_per_dilation, assigned_quantiles, seed):
    """Reference one-example-per-kernel/dilation bias fitting."""
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.RandomState(seed)
    n_biases = 84 * int(np.sum(features_per_dilation))
    output = np.zeros(n_biases, dtype=np.float32)

    feature_start = 0
    for dilation_index, dilation in enumerate(dilations):
        n_this = int(features_per_dilation[dilation_index])
        for kernel_index in range(84):
            feature_stop = feature_start + n_this
            example_index = rng.randint(X.shape[0])
            C = convolution(X[example_index], kernel_index, int(dilation))
            output[feature_start:feature_stop] = np.quantile(
                C,
                assigned_quantiles[feature_start:feature_stop],
            ).astype(np.float32)
            feature_start = feature_stop
    return output


def pool(C, bias, start, stop):
    """Reference PPV, MPV, MIPV, LSPV calculation."""
    ppv = 0
    last_val = 0
    max_stretch = 0.0
    mean_index = 0
    mean = 0.0

    n_values = int(stop - start)
    for local_index in range(n_values):
        value = float(C[start + local_index])
        if value > bias:
            ppv += 1
            mean_index += local_index
            mean += value + bias
        elif value < bias:
            stretch = local_index - last_val
            max_stretch = max(max_stretch, stretch)
            last_val = local_index

    max_stretch = max(max_stretch, n_values - 1 - last_val)
    return np.asarray(
        [
            ppv / n_values,
            mean / ppv if ppv else 0.0,
            mean_index / ppv if ppv else -1.0,
            max_stretch,
        ],
        dtype=np.float32,
    )


def transform(
    X,
    dilations,
    features_per_dilation,
    biases,
    *,
    is_first_difference=False,
):
    """Reference transform in I-ROCKET's contiguous pooling-column order."""
    X = np.asarray(X, dtype=np.float32)
    n_biases = 84 * int(np.sum(features_per_dilation))
    output = np.zeros((X.shape[0], 4 * n_biases), dtype=np.float32)

    for instance_index, x in enumerate(X):
        feature_start = 0
        for dilation_index, dilation in enumerate(dilations):
            dilation = int(dilation)
            padding = 4 * dilation
            n_this = int(features_per_dilation[dilation_index])
            for kernel_index in range(84):
                C = convolution(
                    x,
                    kernel_index,
                    dilation,
                    is_first_difference=is_first_difference,
                )
                uses_same_padding = ((dilation_index + kernel_index) % 2) == 0
                if uses_same_padding:
                    start, stop = 0, C.size
                else:
                    start, stop = padding, C.size - padding

                for feature_offset in range(n_this):
                    feature_index = feature_start + feature_offset
                    output[
                        instance_index,
                        4 * feature_index : 4 * feature_index + 4,
                    ] = pool(C, float(biases[feature_index]), start, stop)
                feature_start += n_this
    return output


def aeon_to_contiguous(features, n_raw_biases, n_diff_biases):
    """Convert aeon's pool-major columns to I-ROCKET's bias-major columns."""
    features = np.asarray(features, dtype=np.float32)
    blocks = []
    start = 0
    for n_biases in (int(n_raw_biases), int(n_diff_biases)):
        if n_biases == 0:
            continue
        ppv = features[:, start : start + n_biases]
        lspv = features[:, start + n_biases : start + 2 * n_biases]
        mpv = features[:, start + 2 * n_biases : start + 3 * n_biases]
        mipv = features[:, start + 3 * n_biases : start + 4 * n_biases]
        block = np.empty((features.shape[0], 4 * n_biases), dtype=np.float32)
        block[:, 0::4] = ppv
        block[:, 1::4] = mpv
        block[:, 2::4] = mipv
        block[:, 3::4] = lspv
        blocks.append(block)
        start += 4 * n_biases
    return np.concatenate(blocks, axis=1)
