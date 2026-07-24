# Conceptual tools

These standalone Matplotlib programs are distributed with the source repository and are not required by the core I-ROCKET pipeline.

## Kernel explorer

`kernel_explorer.py` explains how a complete MultiRocket feature is constructed from a representation, base kernel, dilation, padding mode, fitted bias, and pooling operator.

```bash
python tools/kernel_explorer.py
```

It can load user signals and fitted I-ROCKET models. Run `python tools/kernel_explorer.py --help` for options.

## Pooling explorer

`pooling_explorer.py` recreates the PPV, MPV, MIPV, and LSPV features illustrated in Figure 3 of the MultiRocket paper and can apply them to individual trials.

```bash
python tools/pooling_explorer.py
python tools/pooling_explorer.py --self-test
```

Run `python tools/pooling_explorer.py --help` for dataset, trial, model, and feature-selection options.
