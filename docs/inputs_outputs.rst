Inputs And Outputs
==================

Accepted Inputs
---------------

``booz_xform_jax`` can be driven in three main ways.

1. VMEC ``wout`` files
~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`booz_xform_jax.Booz_xform.read_wout` to load a VMEC NetCDF file:

.. code-block:: python

   from booz_xform_jax import Booz_xform

   bx = Booz_xform()
   bx.read_wout("wout_li383_1.4m.nc", flux=True)

The reader loads:

- mode metadata: ``xm``, ``xn``, ``xm_nyq``, ``xn_nyq``,
- geometry coefficients: ``rmnc``, ``rmns``, ``zmnc``, ``zmns``,
- straight-field-line shift coefficients: ``lmnc``, ``lmns``,
- magnetic-field coefficients: ``bmnc``, ``bmns``, ``bsubumnc``,
  ``bsubumns``, ``bsubvmnc``, ``bsubvmns``,
- rotational transform and optional flux-profile quantities.

``read_wout`` prefers ``netCDF4`` when available and falls back to
``scipy.io.netcdf`` for NetCDF3-style reading. This is why both ``netCDF4`` and
``scipy`` are installed as normal dependencies.

2. In-memory VMEC-like objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`booz_xform_jax.Booz_xform.read_wout_data` when the equilibrium is
already in memory, for example from another Python package.

3. Raw coefficient arrays
~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`booz_xform_jax.Booz_xform.init_from_vmec` to initialize directly
from arrays. The accepted signatures mirror the legacy C++ interface. Mandatory
arguments are:

- ``iotas``,
- ``rmnc``, ``rmns``, ``zmnc``, ``zmns``,
- ``lmnc``, ``lmns``,
- ``bmnc``, ``bmns``,
- ``bsubumnc``, ``bsubumns``,
- ``bsubvmnc``, ``bsubvmns``.

Optional trailing arrays are ``phip``, ``chi``, ``pres``, and ``phi``.

Array Layout And Surface Handling
---------------------------------

The code accepts either ``(ns, mn)`` or ``(mn, ns)`` array layouts where
possible and canonicalizes them internally. The magnetic axis is dropped, and
the internal half grid therefore has ``ns_in = ns - 1`` surfaces.

Surface selection uses ``compute_surfs``:

- the CLI reads full-grid indices from the ``in_booz`` file and maps them to
  half-grid indices,
- the Python API accepts either explicit half-grid indices or normalized flux
  values through :meth:`register_surfaces`,
- if no surface list is provided in the CLI input file, all non-axis surfaces
  are transformed.

CLI Input File Format
---------------------

The legacy-compatible input file has the same structure described in STELLOPT:

.. code-block:: text

   72 15
   'test'
   25 50 75

The first line sets ``mboz`` and ``nboz``. The second line is either a VMEC
extension such as ``test`` or a direct ``wout`` filename. The optional third
line lists full-grid radial indices to transform.

Generated Outputs
-----------------

After :meth:`booz_xform_jax.Booz_xform.run`, the instance stores:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``bmnc_b``, ``bmns_b``
     - Boozer Fourier coefficients of :math:`|B|`.
   * - ``rmnc_b``, ``rmns_b``
     - Boozer Fourier coefficients of :math:`R`.
   * - ``zmns_b``, ``zmnc_b``
     - Boozer Fourier coefficients of :math:`Z`.
   * - ``numns_b``, ``numnc_b``
     - Boozer toroidal angle-shift coefficients for :math:`\nu`.
   * - ``gmnc_b``, ``gmns_b``
     - Jacobian-related Boozer harmonics.
   * - ``Boozer_I``, ``Boozer_G``
     - Current-like flux functions on the selected surfaces.
   * - ``s_b``
     - The normalized toroidal-flux locations of the transformed surfaces.

The NetCDF writer :meth:`write_boozmn` serializes these arrays into a
``boozmn_*.nc`` file.

``boozmn`` Variable Mapping
---------------------------

The output file follows the modern BOOZ_XFORM NetCDF convention. Important
variables include:

.. list-table::
   :header-rows: 1

   * - Variable
     - Notes
   * - ``jlist``
     - Full-grid surface indices written in the legacy 1-based convention
       ``compute_surfs + 2``.
   * - ``ixm_b``, ``ixn_b``
     - Boozer mode lists.
   * - ``iota_b``
     - Rotational-transform profile on the full radial grid.
   * - ``buco_b``, ``bvco_b``
     - Boozer current profiles corresponding to the legacy BOOZ_XFORM output.
   * - ``bmnc_b``, ``rmnc_b``, ``zmns_b``, ``gmn_b``
     - Core symmetric spectra.
   * - ``pmns_b``
     - Legacy storage for :math:`p = -\nu`, so ``pmns_b = -numns_b``.
   * - ``bmns_b``, ``rmns_b``, ``zmnc_b``, ``gmns_b``, ``pmnc_b``
     - Additional asymmetric spectra when ``lasym`` is true.

The JAX-native output dictionary follows the same names where possible. It
returns surface-major arrays, includes the true ``numns_b`` and ``numnc_b``
shift coefficients, and also includes the BOOZ_XFORM storage variables
``pmns_b = -numns_b`` and ``pmnc_b = -numnc_b``.

Compatibility Notes
-------------------

- Older STELLOPT BOOZ_XFORM versions wrote a custom binary format; modern
  versions use NetCDF.
- ``booz_xform_jax`` reads and writes the NetCDF-style ``boozmn`` format.
- The output sign convention for ``pmns`` and ``pmnc`` intentionally matches
  legacy BOOZ_XFORM rather than storing :math:`\nu` directly.
