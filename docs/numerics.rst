Numerics And Performance
========================

Numerical Strategy
------------------

The implementation keeps the BOOZ_XFORM algorithm structure but replaces most
inner loops with vectorized JAX operations:

- trigonometric tables are precomputed on VMEC and Boozer grids,
- Fourier synthesis uses ``jax.numpy.einsum`` contractions,
- the expensive per-mode integrals are evaluated in batched array kernels,
- the high-level functional API keeps surface-major arrays so users can apply
  ``jax.jit`` or larger end-to-end transforms.

Precision
---------

The package enables JAX 64-bit mode during import so the default numerical
behavior matches the double-precision assumptions of the original BOOZ_XFORM
codes and the bundled regression tests.

Grid Construction
-----------------

For a requested transform resolution ``(mboz, nboz)``, the code constructs
BOOZ_XFORM-style angular grids:

- ``ntheta_full = 2 * (2 * mboz + 1)``,
- ``nzeta_full = 2 * (2 * nboz + 1)`` when ``nboz > 0``,
- a reduced poloidal half-grid is used for symmetric cases,
- the toroidal angle spacing includes the VMEC field periodicity ``nfp``.

These conventions are implemented in
:func:`booz_xform_jax.jax_api.prepare_booz_xform_constants`.

Separable Fourier Projection
----------------------------

The projection onto Boozer modes evaluates phases at the mapped angles
``(theta_B, zeta_B)``. Because each phase factors into a ``theta_B``-only and
a ``zeta_B``-only table, the code contracts the zeta factor into the fields
first and never materializes a ``(points x mn_boz)`` phase tensor; every
intermediate stays at ``(points x (nboz + 1) x fields)``. This is an exact
reorganization of the dense point-by-mode contraction, and the test suite
checks values and JVP/VJP derivatives against a dense reference to
double-rounding tolerances on bundled symmetric and asymmetric cases.

A uniform-grid FFT cannot replace this projection: the mapped angles vary
from grid point to grid point (by several grid spacings across a surface),
so the quadrature nodes are not equispaced in the Boozer angles. The input
synthesis stage runs on the uniform VMEC-angle tensor grid, where an FFT
would be exact, but that stage is bounded by the (small, fixed) VMEC
spectrum and is not a bottleneck.

JIT Compilation And Differentiation
-----------------------------------

The high-level :meth:`booz_xform_jax.Booz_xform.run` path is intentionally
close to the original object-oriented interface. The lower-level
:func:`booz_xform_jax.jax_api.booz_xform_jax_impl` path is designed for:

- explicit ``jax.jit`` compilation,
- integration into larger differentiable pipelines,
- surface-major batched transforms,
- reuse of precomputed static constants and grids.

The differentiable path uses finite, masked denominators when forming the
auxiliary ``w`` spectrum from covariant field components. This preserves the
BOOZ_XFORM ``m=n=0`` convention while keeping gradients finite through
``bsubumnc`` and related spectra.

This split keeps the legacy interface readable while preserving a fully JAX
native path for advanced workflows.

Surface Resolution Guidance
---------------------------

As in legacy BOOZ_XFORM, under-resolving the transform is a common source of
aliasing and poor recovery of high-order Boozer harmonics. In practice:

- stellarators generally need higher ``mboz`` and ``nboz`` than the VMEC
  equilibrium itself,
- the STELLOPT documentation recommends roughly 5-6x more poloidal harmonics
  and 2-3x more toroidal harmonics than the underlying VMEC resolution,
- outer surfaces and strongly shaped configurations often require the highest
  transform resolution.

Plotting And Diagnostics
------------------------

The :mod:`booz_xform_jax.plots` module provides quick-look tools:

- ``surfplot``: evaluate :math:`|B|` on a Boozer-angle grid,
- ``symplot``: visualize symmetry diagnostics,
- ``modeplot``: track dominant Fourier modes,
- ``wireplot``: 3D-style geometric views.

These are useful for verifying convergence with respect to both surface choice
and Boozer harmonic truncation.

Platform Considerations
-----------------------

The package itself is pure Python. Practical portability is therefore mostly
determined by wheel availability for the dependencies:

- JAX and ``jaxlib`` determine the available accelerator backends,
- ``netCDF4`` provides the preferred reader and writer for NetCDF files,
- ``scipy`` is retained as a fallback NetCDF3 reader/writer path.

The project CI validates Linux on Python 3.10, 3.11, and 3.12. See the release
notes or CI badges for the exact tested combinations once publishing is set up.
