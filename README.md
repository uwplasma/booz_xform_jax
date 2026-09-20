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

## Measured Comparison

`tools/readme_compare.py` runs `booz_xform_jax` and the reference `xbooz_xform`
on the same VMEC cases, through the same legacy `booz_in` input, and records
what both actually do. Everything in this section comes from one run of that
script; nothing here is estimated.

### Agreement

The two codes produce the same Boozer spectra to machine precision. Relative
L2 differences over all modes and all transformed surfaces:

| Case | ns | `bmnc_b` | `iota_b` | `B_00` |
|---|---|---|---|---|
| circular tokamak | 17 | 2.8e-15 | 0 | 3.3e-16 |
| up/down asymmetric tokamak | 17 | 3.9e-15 | 0 | 4.0e-16 |
| li383 1.4m | 49 | 6.1e-15 | 0 | 2.7e-15 |
| LandremanSenguptaPlunk s5.3 | 51 | 4.8e-15 | 0 | 2.8e-15 |

### Runtime and memory

<p align="center">
  <img src="docs/comparison_runtime.png" width="860" />
</p>
<p align="center">
  <img src="docs/comparison_memory.png" width="860" />
</p>

Total wall-clock time and peak resident set size of each command-line program,
best of seven runs, on one Apple-silicon laptop (Darwin arm64, CPU only):

| Case | ns | `xbooz_xform` | `booz_xform_jax` | Ratio | Peak RSS ref | Peak RSS jax |
|---|---|---|---|---|---|---|
| circular tokamak | 17 | 0.02 s | 2.64 s | 0.01x | 11 MiB | 282 MiB |
| up/down asymmetric tokamak | 17 | 0.02 s | 2.76 s | 0.01x | 11 MiB | 283 MiB |
| li383 1.4m | 49 | 0.78 s | 2.69 s | 0.29x | 61 MiB | 488 MiB |
| LandremanSenguptaPlunk s5.3 | 51 | 1.90 s | 3.20 s | 0.59x | 73 MiB | 602 MiB |

**On these cases `booz_xform_jax` is slower than the compiled reference, and
uses substantially more memory.** Of its runtime, 1.14 s is a fixed cost for
starting Python and importing JAX, paid on every invocation; the rest is
dominated by JAX tracing and XLA compilation, which these problem sizes are far
too small to amortise. Subtracting the import cost entirely, it is still slower
on all four cases.

These are the only equilibria bundled with the repository, and they are small:
`ns` between 17 and 51, three or four transformed surfaces each. No large case
ships here, so no large-case number is published. Laptop timings vary by a few
tens of percent between runs, so read the table as orders of magnitude rather
than precise ratios.

What the JAX implementation buys is not raw CPU speed on small equilibria. It
is a transform that is differentiable end to end (`jax.grad`, `jax.jvp`,
`jax.jacfwd` through `booz_xform_jax.jax_api`), that composes with `jax.jit`
and `jax.vmap`, and that runs unchanged on a GPU. If you need the fastest
single CPU transform of a small equilibrium, use the original code.

Reproduce every number and both figures above with:

```bash
BOOZ_XFORM_REFERENCE_BIN=/path/to/xbooz_xform python tools/readme_compare.py
```

The script writes `docs/comparison_runtime.png`, `docs/comparison_memory.png`
and `README_assets/readme_compare_metrics.json`, which records the platform,
the repeat count and every measurement behind the tables above. The reference
binary is also found automatically if `xbooz_xform` is on `PATH`.

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
