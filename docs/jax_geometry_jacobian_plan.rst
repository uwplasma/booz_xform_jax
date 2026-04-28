JAX Geometry And Jacobian Plan
==============================

Purpose
-------

This page records the implementation plan for making ``booz_xform_jax`` a
stable differentiable geometry provider for downstream codes such as FAX, while
remaining useful for any code that needs Boozer harmonics, Boozer-coordinate
geometry, or Jacobian sensitivities.

The immediate downstream contract is a pure-JAX transform that returns
surface-major arrays with shape ``(ns_selected, mnboz)`` for Boozer spectra and
keeps BOOZ_XFORM-compatible variable names. The key Jacobian output is:

- ``gmnc_b``: cosine harmonics of the Boozer Jacobian-related quantity.
- ``gmn_b``: compatibility alias matching the BOOZ_XFORM/netCDF variable name.

Physics Contract
----------------

The Boozer-coordinate covariant representation uses flux functions
:math:`I(s)` and :math:`G(s)`. Following the BOOZ_XFORM derivation used by the
HiddenSymmetries documentation, the transform forms

.. math::

   G_I = G + \iota I,
   \qquad
   \sqrt{g}_B = \frac{G_I}{|B|^2}.

The JAX kernel reconstructs :math:`|B|`, the VMEC-to-Boozer angle shift, and
the coordinate-transformation factor on an angular grid, then projects the
weighted fields onto Boozer Fourier modes. For the Jacobian harmonics, the
implemented projection is the same one already used by the reference Python
path:

.. math::

   g_{mn} =
   C_{mn}\sum_{\theta,\zeta}
   \sqrt{g}_B(\theta,\zeta)\,
   J_{\mathrm{VMEC}\rightarrow B}(\theta,\zeta)\,
   \cos(m\theta_B - n\zeta_B),

where :math:`C_{mn}` is the BOOZ_XFORM quadrature normalization and
:math:`J_{\mathrm{VMEC}\rightarrow B}` is the angular coordinate factor used
for all Boozer-space projections in this package.

Milestones
----------

1. Public Jacobian harmonics
   Expose the already-computed ``gmnc_b`` array from the JAX API and provide
   ``gmn_b`` as a BOOZ_XFORM-compatible alias. This is the first hard contract
   needed by FAX continuum and mode-structure operators.

2. Stable differentiable API
   Keep ``booz_xform_jax_impl`` as the low-level primitive for composed JAX
   programs. Geometry constants and mode lists remain static inputs, while VMEC
   spectra, current profiles, and rotational transform arrays remain
   differentiable numerical inputs.

3. Jacobian access patterns
   Support direct scalar objectives with ``jax.grad`` and
   ``jax.value_and_grad``. For large geometry-to-physics maps, prefer
   matrix-free products through ``jax.jvp``, ``jax.vjp``, and ``jax.linearize``
   rather than materializing dense Jacobians. Use ``jax.jacfwd`` or
   ``jax.jacrev`` only when the output/input aspect ratio makes dense
   Jacobians reasonable.

4. Performance modes
   Preserve the default vectorized Fourier projection for speed and the
   ``BOOZ_XFORM_JAX_FOURIER_MODE=streamed`` path for lower memory. Both paths
   must produce the same Jacobian harmonics within regression tolerances.

5. Validation gates
   Every geometry output used by downstream codes should have:

   - parity against ``Booz_xform.run`` on bundled VMEC cases,
   - vectorized-versus-streamed parity,
   - JIT-versus-non-JIT parity where practical,
   - finite-gradient tests through representative scalar objectives,
   - NetCDF name compatibility checks for ``gmn_b``.

6. Future asymmetric-output expansion
   The JAX path currently propagates asymmetric VMEC inputs through the
   symmetric-output spectra used by the present tests. A later milestone should
   expose full asymmetric output arrays such as ``bmns_b``, ``rmns_b``,
   ``zmnc_b``, ``pmnc_b``, and ``gmns_b`` when ``lasym`` is true.

Downstream FAX Contract
-----------------------

FAX and other spectral MHD tools should consume the JAX output dictionary
without relying on object attributes. The minimum stable keys are:

- ``ixm_b`` and ``ixn_b`` for Boozer mode numbers,
- ``iota_b``, ``buco_b``, and ``bvco_b`` for flux functions,
- ``bmnc_b`` for :math:`|B|`,
- ``rmnc_b`` and ``zmns_b`` for geometry,
- ``pmns_b`` for the legacy stored toroidal-angle shift,
- ``gmnc_b`` or ``gmn_b`` for Jacobian harmonics,
- ``jlist`` for selected 1-based full-grid surface indices.

The shape convention is intentionally surface-major in JAX outputs. Writers and
legacy BOOZ_XFORM files may transpose this layout for file compatibility.

Implementation Notes
--------------------

- Keep numerical work in ``jax.numpy`` and ``jax.lax`` primitives so ``jit``,
  ``grad``, ``jvp``, ``vjp``, and batching remain valid.
- Mark only small configuration objects as static in ``jax.jit``. JAX
  recompiles for new static values, so static arguments should be mode lists,
  grid sizes, and constants rather than frequently changing spectra.
- Treat dense Jacobian formation as a diagnostic path, not the default for
  production optimization. FAX objectives should generally use scalar losses
  with reverse-mode gradients or matrix-free JVP/VJP products.
- Keep tests physics based: compare spectral coefficients and transformation
  identities, not only array shapes.

References
----------

- `STELLOPT BOOZ_XFORM documentation
  <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM.html>`_ records
  the BOOZ_XFORM output variables, including ``gmn_b``.
- `HiddenSymmetries BOOZ_XFORM theory notes
  <https://hiddensymmetries.github.io/booz_xform/theory.html>`_ describe the
  Boozer-coordinate equations and Jacobian relation used here.
- `JAX automatic differentiation API
  <https://docs.jax.dev/en/latest/jax.html#automatic-differentiation>`_
  documents ``grad``, ``jacfwd``, ``jacrev``, ``jvp``, ``vjp``, and
  ``linearize``.
- `JAX forward- and reverse-mode autodiff guide
  <https://docs.jax.dev/en/latest/jacobian-vector-products.html>`_ explains
  the computational tradeoffs between JVPs, VJPs, and dense Jacobians.
- `JAX JIT compilation guide
  <https://docs.jax.dev/en/latest/jit-compilation.html#marking-arguments-as-static>`_
  documents when static arguments are appropriate and why they affect
  recompilation.
