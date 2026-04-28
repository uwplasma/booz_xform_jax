Quickstart
==========

Install directly from PyPI::

  pip install booz_xform_jax

For development, editable installs remain the recommended path::

  git clone https://github.com/uwplasma/booz_xform_jax
  cd booz_xform_jax
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e .

The package depends on ``jax``, ``jaxlib``, ``numpy``, ``netCDF4``,
``scipy``, ``matplotlib``, and ``plotly``. A normal ``pip install`` pulls
those in automatically, so the CLI and the NetCDF readers are ready after
installation.

Command-Line Workflow
---------------------

``booz_xform_jax`` understands the classic STELLOPT BOOZ_XFORM input-file
format::

  mboz nboz
  'mycase'
  10 20 30

Place this file next to a matching ``wout_mycase.nc`` file and run::

  booz_xform_jax in_booz.mycase F

Equivalent installed entry points are:

- ``booz_xform_jax``
- ``xbooz_xform``
- ``xbooz_xform_jax``
- ``python -m booz_xform_jax``

The optional ``F`` flag suppresses screen output, matching the legacy driver.
The output is written as ``boozmn_mycase.nc`` in the current working directory.

If you are working from a source checkout, a bundled regression case can be run
immediately::

  booz_xform_jax tests/test_files/booz_in.circular_tokamak F

That command reads ``tests/test_files/wout_circular_tokamak.nc`` and produces
``boozmn_circular_tokamak.nc``.

Object-Oriented Python API
--------------------------

The high-level API mirrors the original BOOZ_XFORM class model:

.. code-block:: python

   from booz_xform_jax import Booz_xform

   bx = Booz_xform()
   bx.read_wout("wout_li383_1.4m.nc", flux=True)
   bx.mboz = 24
   bx.nboz = 12
   bx.register_surfaces([0.25, 0.5, 0.75])
   bx.run()
   bx.write_boozmn("boozmn_li383_1.4m.nc")

Key points:

- ``read_wout(..., flux=True)`` also loads optional radial profiles such as
  ``phip``, ``chi``, ``pres``, and ``phi`` when present.
- ``register_surfaces`` accepts either half-grid indices or normalized toroidal
  flux values in ``[0, 1]``.
- ``run()`` stores the Boozer spectra on the object.
- ``write_boozmn()`` emits a NetCDF file compatible with the modern
  ``boozmn`` format.

Functional JAX API
------------------

For end-to-end JAX pipelines, use :mod:`booz_xform_jax.jax_api`:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from booz_xform_jax import Booz_xform
   from booz_xform_jax.jax_api import (
       booz_xform_jax_impl,
       prepare_booz_xform_constants,
   )

   bx = Booz_xform()
   bx.read_wout("wout_li383_1.4m.nc")
   bx.mboz = 8
   bx.nboz = 8
   bx.compute_surfs = [0, 5, 10]

   constants, grids = prepare_booz_xform_constants(
       nfp=bx.nfp,
       mboz=bx.mboz,
       nboz=bx.nboz,
       asym=bool(bx.asym),
       xm=bx.xm,
       xn=bx.xn,
       xm_nyq=bx.xm_nyq,
       xn_nyq=bx.xn_nyq,
   )

   out = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))(
       rmnc=jnp.asarray(bx.rmnc).T,
       zmns=jnp.asarray(bx.zmns).T,
       lmns=jnp.asarray(bx.lmns).T,
       bmnc=jnp.asarray(bx.bmnc).T,
       bsubumnc=jnp.asarray(bx.bsubumnc).T,
       bsubvmnc=jnp.asarray(bx.bsubvmnc).T,
       iota=jnp.asarray(bx.iota),
       xm=jnp.asarray(bx.xm),
       xn=jnp.asarray(bx.xn),
       xm_nyq=jnp.asarray(bx.xm_nyq),
       xn_nyq=jnp.asarray(bx.xn_nyq),
       constants=constants,
       grids=grids,
       surface_indices=jnp.asarray(bx.compute_surfs),
   )

The returned dictionary includes Boozer spectra such as ``bmnc_b``,
``rmnc_b``, ``zmns_b``, ``pmns_b``, and the Jacobian harmonics ``gmnc_b``.
For compatibility with BOOZ_XFORM ``boozmn`` files, ``gmn_b`` is also provided
as an alias of ``gmnc_b``.

Examples
--------

The repository ships several example scripts:

- ``examples/example_li383_basic.py``: minimal end-to-end transform and plots.
- ``examples/example_li383_jax_api_fast.py``: low-level JAX API with JIT.
- ``examples/example_li383_autodiff_opt.py``: differentiable optimization toy
  example.
- ``examples/example_vmec_jax_pipeline.py``: in-memory VMEC-to-Boozer pipeline.
- ``examples/example_li383_wireframe.py``: wireframe-style visualization.
- ``examples/example_li383_resolution_scan.py``: transform-resolution studies.
