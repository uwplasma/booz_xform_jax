# booz_xform_jax

Install from PyPI:

```bash
pip install booz_xform_jax
```

`booz_xform_jax` is a JAX-native implementation of the Boozer coordinate
transformation for VMEC equilibria. It reads VMEC `wout` data, computes Boozer
Fourier spectra, writes `boozmn` NetCDF files, exposes a differentiable Python
API, and provides a legacy-compatible command line interface matching
`xbooz_xform` workflows.

## Quickstart

Install directly from PyPI:

```bash
pip install booz_xform_jax
```

Install from a clone in editable mode when you want to modify the code:

```bash
git clone https://github.com/uwplasma/booz_xform_jax
cd booz_xform_jax
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package installs these entry points:

```bash
booz_xform_jax
xbooz_xform
xbooz_xform_jax
```

CLI usage with a standard STELLOPT-style input file:

```bash
booz_xform_jax in_booz.mycase F
```

or, from a source checkout, using a bundled regression case:

```bash
booz_xform_jax tests/test_files/booz_in.circular_tokamak F
```

Python API usage:

```python
from booz_xform_jax import Booz_xform

bx = Booz_xform()
bx.read_wout("wout_mycase.nc", flux=True)
bx.register_surfaces([0.25, 0.5, 0.75])
bx.run()
bx.write_boozmn("boozmn_mycase.nc")
```

The runtime dependencies installed from PyPI already include `jax`, `jaxlib`,
`netCDF4`, `scipy`, `numpy`, `matplotlib`, and `plotly`, so the CLI and the
NetCDF readers/writers work after a normal `pip install`.

## Documentation

The full documentation now lives in the `docs/` tree:

- [Documentation index](docs/index.rst)
- [Quickstart](docs/quickstart.rst)
- [Theory and derivations](docs/theory.rst)
- [Inputs and outputs](docs/inputs_outputs.rst)
- [Numerics and performance](docs/numerics.rst)
- [Examples](docs/examples.rst)
- [STELLOPT compatibility notes](docs/stellopt_compatibility.rst)
- [API and source reference](docs/api.rst)
- [Citations](docs/citations.rst)

## Performance Snapshot

The plots below compare `booz_xform_jax` and the reference `xbooz_xform` on a
set of bundled VMEC cases.

<p align="center">
  <img src="docs/comparison_runtime.png" width="860" />
</p>
<p align="center">
  <img src="docs/comparison_memory.png" width="560" />
</p>

| Case | booz_xform_jax | xbooz_xform | Speedup |
|---|---|---|---|
| Tokamak (small, ns=16) | 0.007 s | 0.001 s | 0.2x |
| li383 stellarator (ns=48) | 0.032 s | 0.428 s | 14x |
| LSP stellarator (ns=99) | 0.040 s | 1.710 s | 42x |
| HSX (large, ns=300) | 0.627 s | 89.9 s | 143x |

Reproduce the README comparison assets with:

```bash
python tools/readme_compare.py
```

## Project Scope

- The legacy BOOZ_XFORM input format, surface-selection conventions, and
  `boozmn` output structure are supported so existing workflows transfer
  cleanly.
- The numerical core is written in JAX and keeps the transform
  differentiable.
- The package includes both the object-oriented `Booz_xform` interface and a
  lower-level functional API in `booz_xform_jax.jax_api`.
- More technical background, equations, derivations, numerics, and source
  references have been moved from the README into the documentation.

## Citation

If you use this package, cite the original Boozer-coordinate and BOOZ_XFORM
literature listed in [docs/citations.rst](docs/citations.rst), together with
this repository.

## License

MIT. See [LICENSE](LICENSE).
