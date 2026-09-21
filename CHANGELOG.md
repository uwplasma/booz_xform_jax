# Changelog

Releases before 0.4.0 are described in their GitHub release notes.

## 0.4.0 - 2026-09-20

The functional JAX API in `booz_xform_jax.jax_api` is byte-for-byte identical
to 0.3.0: `BoozerConfig`, `BoozerPlan` and `prepare_booz_xform_plan` (added in
0.2.0), the asymmetric (`lasym`) path, the guarded zero-mode denominators and
the opt-in magnetic-only projection all behave exactly as before. The changes
are in the `Booz_xform` class, the documentation and the tests.

### Changed

- Loading an equilibrium with `read_wout`, `read_wout_data` or
  `init_from_vmec` no longer fills `compute_surfs` with every half-grid
  surface. It stays `None`, which already meant "all surfaces", and `run()`
  and `run_jax()` expand it when they run. Code that read `compute_surfs`
  right after loading and expected a list now gets `None`. (#10)

### Fixed

- `register_surfaces()` now selects the surfaces it is given. It adds to the
  current selection, and because loading had already filled that selection
  with every surface, calling it did nothing. Starting from the default, the
  first call now narrows the transform to exactly the registered surfaces. It
  raises `RuntimeError` when called before any equilibrium is loaded, and
  accepts NumPy integer scalars and 0-d arrays. (#10, #3)
- Changing `mboz`, `nboz`, `nfp` or `asym` between two runs on the same
  `Booz_xform` object now rebuilds the cached Boozer mode lists and grids, in
  both `run()` and `run_jax()`. Before, they were built once and the second
  run reused them at the old resolution. (#10)
- `booz_xform_jax.__version__`, and the documentation version built from it,
  said `0.1.0` in every release through 0.3.0. It now matches the package
  version, and a test keeps the two equal.

### Documentation

- The README speedup table (14x, 42x and 143x over `xbooz_xform`) could not
  be reproduced and has been replaced with measurements from
  `tools/readme_compare.py`. On the four bundled equilibria (ns 17 to 51) the
  two codes agree to 2.8e-15 to 6.1e-15 relative L2 in `bmnc_b`. End to end,
  `booz_xform_jax` is slower than the compiled STELLOPT `xbooz_xform` (2.6 to
  3.2 s against 0.02 to 1.9 s, of which 1.14 s is Python and JAX import) and
  peaks at 8 to 25 times the memory. (#11)
- The STELLOPT compatibility page now warns that the Fortran STELLOPT
  `xbooz_xform` and the C++ `booz_xform` read the surface numbers in a
  `booz_in` file differently: as `jlist` entries in the first, and as 0-based
  `compute_surfs` two lower in the second. `booz_xform_jax` follows STELLOPT.
  (#11)

### Tests

- CLI parity is now checked in CI. Each bundled `booz_in.*` file goes through
  the command line, and the `boozmn` file it writes is compared at `1e-12`
  against the reference output shipped by the original `booz_xform` project.
  Before, every parity test needed a reference binary and skipped in CI. (#13)
- The CLI parity suite no longer names files by machine path. It finds the
  reference binary through `BOOZ_XFORM_REFERENCE_BIN` or `PATH`, and it will
  not use this package's own `xbooz_xform` alias or a C++ `booz_xform` as the
  reference. (#12)
- `run_jax(jit=True)` is now tested on a stellarator-symmetric equilibrium,
  and again after a retrace with a different surface count. (#14)
