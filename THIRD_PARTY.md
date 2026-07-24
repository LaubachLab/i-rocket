# Optional third-party integrations

## TSHAP

I-ROCKET's TSHAP support is optional. Release 260724 was tested against:

- package: `tshap`
- version: `0.0.1`
- PyPI wheel: `tshap-0.0.1-py3-none-any.whl`
- wheel SHA-256: `257c879882e61e63ae3ce1e4841b998e855fc95e899426446fadc39a33678b09`
- upstream repository: `mlgig/tshap`
- upstream license: GPL-3.0

I-ROCKET does not copy or redistribute TSHAP source. `tshap_integration.py` contains only BSD-licensed adapter and plotting code written for I-ROCKET. Installing the optional extra installs the external package separately:

```bash
python -m pip install -e ".[tshap]"
```

The package version and wheel hash above record the executable dependency tested for this release.

## aeon and OpenML

The `datasets` extra installs aeon and the OpenML Python client for public dataset access and optional reference comparisons. Neither is required by the core I-ROCKET pipeline.
