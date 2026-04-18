STELLOPT Compatibility
======================

This page consolidates the operational details from the original STELLOPT
BOOZ_XFORM documentation and maps them onto ``booz_xform_jax``.

Reference Material
------------------

- `STELLOPT BOOZ_XFORM page
  <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM.html>`_
- `STELLOPT suite compilation page
  <https://princetonuniversity.github.io/STELLOPT/STELLOPT%20Compilation.html>`_
- `Boozer transformation tutorial for an NCSX-like configuration
  <https://princetonuniversity.github.io/STELLOPT/Boozer%20Transformation%20for%20NCSX-Like%20Configuration.html>`_
- `Transformation from VMEC to Boozer Coordinates
  <https://princetonuniversity.github.io/STELLOPT/docs/Transformation%20from%20VMEC%20to%20Boozer%20Coordinates.pdf>`_
- `VMECwiki
  <https://princetonuniversity.github.io/STELLOPT/VMEC.html>`_

Theory Conventions Used In STELLOPT
-----------------------------------

The STELLOPT BOOZ_XFORM documentation emphasizes the following points, all of
which are relevant to ``booz_xform_jax`` as well:

- the goal is to transform a VMEC equilibrium into Boozer straight-field-line
  coordinates,
- the output is represented as Fourier coefficients of transformed quantities,
- the Boozer representation often needs substantially more Fourier modes than
  the original VMEC representation to retain accuracy,
- practical resolution guidance is roughly 5-6x more poloidal harmonics and
  2-3x more toroidal harmonics than in the VMEC equilibrium.

The same documentation records the ``boozmn`` conventions for ``buco``,
``bvco``, ``pmns``, and ``pmnc`` that ``booz_xform_jax`` preserves when reading
and writing output files.

Legacy Input File Format
------------------------

The original BOOZ_XFORM executable accepts an input file containing:

1. the number of poloidal and toroidal Boozer harmonics,
2. a VMEC extension or ``wout`` suffix,
3. a list of full-grid surfaces to transform.

Example::

  72 15
  'test'
  25 50 75

The STELLOPT guidance suggests surfaces around one-quarter, one-half, and
three-quarters of the plasma radius, provided they align with VMEC grid
indices. ``booz_xform_jax`` accepts the same input layout.

Execution
---------

The legacy executable is invoked as::

  xbooz_xform in_booz.test

and an optional ``T`` or ``F`` flag controls screen output. ``booz_xform_jax``
implements the same calling convention:

.. code-block:: bash

   booz_xform_jax in_booz.test F

or equivalently::

  xbooz_xform in_booz.test F

When the input file omits the surface list, the historical behavior is to
transform all non-axis surfaces. ``booz_xform_jax`` preserves that default.

Output Format
-------------

The STELLOPT page documents two generations of output:

- an older custom binary format used in BOOZ_XFORM v1,
- a newer NetCDF-based format in v2.

``booz_xform_jax`` targets the NetCDF-style ``boozmn`` layout. It writes files
named ``boozmn_<suffix>.nc`` and includes the standard arrays:

- mode lists ``ixm_b`` and ``ixn_b``,
- selected-surface list ``jlist``,
- radial profiles such as ``iota_b``, ``buco_b``, and ``bvco_b``,
- spectral arrays including ``bmnc_b``, ``rmnc_b``, ``zmns_b``, ``pmns_b``,
  and ``gmn_b`` plus asymmetric counterparts when ``lasym`` is true.

The package exposes the file I/O through
:meth:`booz_xform_jax.Booz_xform.write_boozmn` and
:meth:`booz_xform_jax.Booz_xform.read_boozmn`.

Visualization Notes
-------------------

The STELLOPT BOOZ_XFORM page notes that the trigonometric argument is of the
form :math:`m u - n v`. It also stresses that the toroidal angle should be
adjusted using the transformed ``pmns`` quantity before reconstructing real
space in Boozer coordinates. In the sign convention used here:

.. math::

   p = \zeta_{vmec} - \zeta_{Boozer} = -\nu.

That sign is preserved in ``booz_xform_jax`` output, so downstream tools that
consume ``pmns`` or ``pmnc`` see the legacy-compatible convention.

Compilation And Packaging
-------------------------

The original BOOZ_XFORM code is built as part of the STELLOPT suite, whose
compilation instructions are documented on the STELLOPT compilation page. By
contrast, ``booz_xform_jax`` is distributed as a normal Python package and can
be installed with:

.. code-block:: bash

   pip install booz_xform_jax

Editable source installs are also supported with ``pip install -e .``.

Why This Matters
----------------

This compatibility layer means an existing STELLOPT/VMEC workflow can often be
ported with very little change:

- keep the same ``in_booz.*`` files,
- keep the same expectations for ``boozmn`` variable names,
- switch the executable to ``booz_xform_jax`` or ``xbooz_xform``,
- use the JAX API only when differentiability or tight Python integration is
  needed.
