Examples
========

Bundled Example Scripts
-----------------------

The repository includes several ready-to-run examples.

``examples/example_li383_basic.py``
  Minimal end-to-end example: reads a bundled VMEC file, runs the transform on
  one surface, and generates a surfplot plus a dominant-mode plot.

  Run it with::

    python examples/example_li383_basic.py --no-show

``examples/example_li383_jax_api_fast.py``
  Demonstrates the lower-level JAX API with precomputed constants and explicit
  ``jax.jit`` compilation.

  Run it with::

    python examples/example_li383_jax_api_fast.py

``examples/example_li383_autodiff_opt.py``
  Shows how Boozer diagnostics can be embedded in a differentiable optimization
  loop.

``examples/example_li383_jacobian_sensitivity.py``
  Demonstrates matrix-free sensitivities of ``gmnc_b`` Jacobian harmonics using
  ``jax.value_and_grad``, ``jax.jvp``, and ``jax.vjp``.

``examples/example_vmec_jax_pipeline.py``
  Demonstrates an in-memory pipeline from ``vmec_jax`` output to
  ``booz_xform_jax`` without writing intermediate files.

``examples/example_li383_wireframe.py``
  Produces a wireframe-style visualization from a transformed configuration.

``examples/example_li383_resolution_scan.py``
  Explores how the output changes as ``mboz`` and ``nboz`` are varied.

CLI Example
-----------

For a legacy-style CLI run, create an input file like:

.. code-block:: text

   16 16
   'circular_tokamak'
   2 8 14

and run::

  booz_xform_jax in_booz.circular_tokamak F

If ``wout_circular_tokamak.nc`` is in the same directory, the command writes
``boozmn_circular_tokamak.nc``.

Python Example
--------------

.. code-block:: python

   from booz_xform_jax import Booz_xform

   bx = Booz_xform()
   bx.read_wout("wout_li383_1.4m.nc", flux=True)
   bx.mboz = 16
   bx.nboz = 16
   bx.compute_surfs = [0, 10, 20]
   bx.run()

   print("Boozer I:", bx.Boozer_I)
   print("First surface |B| coefficients:", bx.bmnc_b[:, 0][:10])

Using The Plot Helpers
----------------------

.. code-block:: python

   from booz_xform_jax import Booz_xform, surfplot, modeplot

   bx = Booz_xform()
   bx.read_wout("wout_li383_1.4m.nc", flux=True)
   bx.register_surfaces(0.5)
   bx.run()

   surfplot(bx, js=0, show=True)
   modeplot(bx, nmodes=12, show=True)
