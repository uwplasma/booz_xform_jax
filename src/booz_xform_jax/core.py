"""
Core classes for the JAX implementation of ``booz_xform``.

This module defines the :class:`Booz_xform` class, which is the primary
interface for converting Fourier data from a VMEC equilibrium
(spectral representation in VMEC coordinates) to a spectral
representation in Boozer coordinates.

Pedagogical overview
====================

**What problem are we solving?**

Given a VMEC MHD equilibrium we know (on a set of half-grid radial
surfaces) its Fourier representation in angles (θ, ζ):

  * Geometry: R(θ, ζ, s), Z(θ, ζ, s) and the poloidal angle shift
    λ(θ, ζ, s),
  * Magnetic‐field strength and covariant components:
    |B|(θ, ζ, s), B_θ(θ, ζ, s), B_ζ(θ, ζ, s).

The goal of BOOZ_XFORM is to construct a spectral representation of
the same equilibrium but in **Boozer angles** (θ_B, ζ_B), where the
magnetic field lines are straight and the contravariant components of
B take a particularly simple form. The result is stored as
Fourier coefficients B_{m,n}(s), R_{m,n}(s), Z_{m,n}(s), ν_{m,n}(s),
and Jacobian harmonics on a chosen subset of radial surfaces.

**High-level algorithm (per radial surface)**

For each selected radial surface, the core algorithm follows the
original C++ / Fortran implementation closely:

 1. Build a tensor-product grid in VMEC angles (θ, ζ) and flatten it
    to a vector of length N = N_θ × N_ζ.

 2. Using the *non-Nyquist* VMEC spectrum, synthesise:
        - R(θ, ζ), Z(θ, ζ), λ(θ, ζ),
        - ∂λ/∂θ, ∂λ/∂ζ.

 3. Using the *Nyquist* VMEC spectrum, construct:
        - an auxiliary function w(θ, ζ),
        - its derivatives ∂w/∂θ, ∂w/∂ζ,
        - |B|(θ, ζ).

 4. From the Nyquist spectra of B_θ and B_ζ, recover the Boozer
    profiles I(s) and G(s) and the auxiliary Nyquist spectrum of w.

 5. Compute the “field-line label” ν(θ, ζ) from equation (10) of the
    BOOZ_XFORM theory, and then the Boozer angles:
        θ_B = θ + λ + ι ν,
        ζ_B = ζ + ν,
    where ι is the rotational transform on this surface.

 6. From the derivatives of w and λ, construct ∂ν/∂θ and ∂ν/∂ζ and
    hence the factor dB/d(vmec) appearing in the Fourier integrals.

 7. On the (θ_B, ζ_B) grid, precompute trigonometric tables and
    perform the 2D Fourier integrals that define the Boozer
    coefficients B_{m,n}, R_{m,n}, Z_{m,n}, ν_{m,n} and the
    Boozer-Jacobian harmonics.

This module provides a **vectorised, JAX-based implementation** of the
above steps. The main performance principles are:

  * Precompute trigonometric tables on the (θ, ζ) grid once per run.
  * Hoist all per-mode cos/sin combinations that do not depend on the
    surface index out of the radial loop.
  * Replace explicit Python loops over Fourier modes by
    `jax.numpy.einsum` and broadcasting.
  * Keep the outer loop over radial surfaces in Python (a typical
    equilibrium has tens of surfaces, whereas the number of grid
    points N can be in the thousands, so most work is still inside
    JAX kernels).

Public API
==========

The external API mirrors the original BOOZ_XFORM library:

  * Create an instance of :class:`Booz_xform`.
  * Call :meth:`read_wout` or :meth:`init_from_vmec` to populate
    VMEC data.
  * Optionally call :meth:`register_surfaces` to select a subset of
    radial surfaces (by index or by normalised toroidal flux s).
  * Call :meth:`run()` to perform the Boozer transform. The resulting
    Boozer spectra and profiles are stored on the instance
    (``bmnc_b``, ``rmnc_b``, ``zmns_b``, ``numns_b``, ``gmnc_b``,
    etc., plus Boozer I/G and the chosen radial grid ``s_b``).
  * Use :meth:`write_boozmn` / :meth:`read_boozmn` and plotting helpers
    (defined in other modules) as in the original code.

This file is deliberately **pedagogical**: in addition to the
performance-oriented vectorisation, it includes detailed comments
explaining each mathematical step and its relationship to the
published BOOZ_XFORM theory and to the original implementation.
"""

from __future__ import annotations

import math
import os as _os
import numpy as _np
from functools import partial
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

try:
    import jax
    import jax.numpy as jnp

    # The original BOOZ_XFORM (and VMEC) use double precision
    # throughout. We enable 64-bit mode globally so that JAX matches
    # the reference implementation and regression tests can compare
    # against double-precision reference outputs.
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", True)
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The booz_xform_jax package requires JAX. Please install jax and "
        "jaxlib before using this module."
    ) from e

from .vmec import init_from_vmec, read_wout, read_wout_data
from .io_utils import write_boozmn, read_boozmn
from .jax_api import booz_xform_jax_impl, prepare_booz_xform_constants
from .trig import _init_trig, _init_trig_np, _init_trig_np_T


# -----------------------------------------------------------------------------
# Trigonometric table helper
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Main Booz_xform class
# -----------------------------------------------------------------------------


@dataclass
class Booz_xform:
    """
    Class implementing the Boozer coordinate transformation using JAX.

    Instances of :class:`Booz_xform` encapsulate all data required to
    convert the spectral representation of a VMEC equilibrium (in
    VMEC angles) to a spectral representation in Boozer coordinates.

    Typical usage
    -------------
    >>> bx = Booz_xform()
    >>> bx.read_wout("wout_mycase.nc", flux=True)   # or init_from_vmec(...)
    >>> bx.register_surfaces([0.2, 0.5, 0.8])       # select surfaces in s-space
    >>> bx.run()
    >>> bx.write_boozmn("boozmn_mycase.nc")

    After :meth:`run` completes, the Boozer spectra are stored in
    attributes like ``bmnc_b``, ``rmnc_b``, ``zmns_b``, etc., and the
    Boozer I/G profiles and radial grid on ``Boozer_I``, ``Boozer_G``,
    and ``s_b``.

    Attributes
    ----------
    nfp : int
        Field periodicity (number of field periods) of the equilibrium.

    asym : bool
        Whether the VMEC equilibrium is non-stellarator-symmetric.
        If ``False``, only the symmetric Fourier coefficients are used
        (cosine/sine combinations that respect stellarator symmetry).
        If ``True``, additional “ns” arrays are populated and used.

    verbose : int or bool
        Controls diagnostic printing during :meth:`run`. Historically
        this was an integer (0, 1, 2, …). In this implementation any
        truthy value enables basic per-surface diagnostics; setting
        ``verbose > 1`` prints additional information.

    mpol, ntor : int
        Maximum poloidal and toroidal mode numbers in the non-Nyquist
        VMEC spectrum, read from the wout file.

    mnmax : int
        Total number of *non-Nyquist* VMEC Fourier modes. For
        symmetric equilibria this is typically ``mpol * (2*ntor + 1)``.

    xm, xn : ndarray of int, shape (mnmax,)
        Mode list for the non-Nyquist VMEC spectrum: poloidal and
        toroidal mode numbers (with xn stored as :math:`n n_{fp}` to
        match VMEC conventions).

    xm_nyq, xn_nyq : ndarray of int
        Mode list for the Nyquist spectrum used to reconstruct w and
        |B|. Sizes and ranges mirror those in the original BOOZ_XFORM.

    mpol_nyq, ntor_nyq, mnmax_nyq : int
        Nyquist resolutions and total number of Nyquist modes in the
        VMEC input, read from the wout file.

    s_in : ndarray, shape (ns_in,)
        Radial coordinate values on the VMEC half grid (excluding the
        magnetic axis). This is stored as a NumPy array (host side)
        so that we can use standard Python indexing and
        ``numpy.argmin`` when mapping floating-point s values to
        nearest indices.

    iota : jax.numpy.ndarray, shape (ns_in,)
        Rotational transform on the VMEC half grid.

    rmnc, rmns, zmnc, zmns, lmnc, lmns : jax.numpy.ndarray
        Non-Nyquist VMEC Fourier coefficients on the half grid,
        with dimensions ``(mnmax, ns_in)``. Asymmetric quantities
        are set to ``None`` when ``asym`` is ``False``.

    bmnc, bmns, bsubumnc, bsubumns, bsubvmnc, bsubvmns : jax.numpy.ndarray
        Nyquist VMEC Fourier coefficients on the half grid, with
        dimensions ``(mnmax_nyq, ns_in)``. Asymmetric quantities are
        set to ``None`` when ``asym`` is ``False``. These are used to
        reconstruct |B| and the covariant components B_θ, B_ζ.

    Boozer_I_all, Boozer_G_all : ndarray, shape (ns_in,)
        Boozer I(s) and G(s) profiles on the full half grid. These
        correspond to the m=0, n=0 components of ``bsubumnc`` and
        ``bsubvmnc`` and are stored as NumPy arrays.

    phip, chi, pres, phi : jax.numpy.ndarray, shape (ns_in,), optional
        Optional radial profiles read from the VMEC file when the
        ``flux`` flag is passed to :meth:`read_wout`. They are not
        used directly in the Boozer transform but are convenient to
        have available for post-processing.

    aspect : float
        Aspect ratio of the equilibrium (copied from VMEC).

    toroidal_flux : float
        Total toroidal flux of the equilibrium (copied from VMEC).

    compute_surfs : list[int] or None
        Indices of the half-grid surfaces on which to compute the
        Boozer transform. Indices run from 0 to ``ns_in-1``.
        ``None`` (default) means “all surfaces”.

    s_b : ndarray, shape (ns_b,)
        Radial coordinate values on the subset of surfaces selected
        by ``compute_surfs``. Populated by :meth:`run` and
        :meth:`read_boozmn`.

    ns_in : int
        Number of half-grid surfaces (excluding the axis) in the VMEC
        input.

    ns_b : int
        Number of surfaces selected for the Boozer transform
        (i.e. ``len(compute_surfs)``).

    Boozer_I, Boozer_G : ndarray, shape (ns_b,)
        Boozer I and G profiles restricted to the selected surfaces.

    mboz, nboz : int
        Maximum poloidal and toroidal mode numbers in the *Boozer*
        spectrum. If not explicitly set by the user, these default to
        ``mpol`` and ``ntor`` respectively (mirroring the original
        BOOZ_XFORM behaviour).

    mnboz : int
        Total number of Boozer harmonics retained. The enumeration
        follows the original code:

          * m runs from 0, 1, …, mboz-1
          * for m = 0, n runs 0, 1, …, nboz
          * for m > 0, n runs -nboz, …, -1, 0, 1, …, nboz

        The toroidal index is stored as ``xn_b = n * nfp``.

    xm_b, xn_b : ndarray of int, shape (mnboz,)
        Boozer mode list as described above.

    bmnc_b, bmns_b, rmnc_b, rmns_b, zmnc_b, zmns_b,
    numnc_b, numns_b, gmnc_b, gmns_b : ndarray
        Boozer Fourier coefficients on the selected surfaces. Each has
        shape ``(mnboz, ns_b)``. Asymmetric arrays are ``None`` when
        ``asym`` is ``False``. The “c” suffix denotes cosine-like
        coefficients and the “s” suffix sine-like coefficients,
        following the usual VMEC/BOOZ_XFORM conventions.

    _prepared : bool
        Internal flag indicating whether the angular grids and related
        bookkeeping (θ, ζ, grid sizes) have been initialised.
    """

    # VMEC parameters read from the wout file
    nfp: int = 1
    asym: bool = False
    # Verbosity as described in the docstring
    verbose: int | bool = 1
    mpol: int = 0
    ntor: int = 0
    mnmax: int = 0
    xm: Optional[_np.ndarray] = None
    xn: Optional[_np.ndarray] = None
    xm_nyq: Optional[_np.ndarray] = None
    xn_nyq: Optional[_np.ndarray] = None
    mpol_nyq: Optional[int] = None
    ntor_nyq: Optional[int] = None
    mnmax_nyq: Optional[int] = None

    # Input arrays on the VMEC half grid (radial index runs over ns_in)
    s_in: Optional[_np.ndarray] = None
    iota: Optional[jnp.ndarray] = None
    rmnc: Optional[jnp.ndarray] = None
    rmns: Optional[jnp.ndarray] = None
    zmnc: Optional[jnp.ndarray] = None
    zmns: Optional[jnp.ndarray] = None
    lmnc: Optional[jnp.ndarray] = None
    lmns: Optional[jnp.ndarray] = None
    bmnc: Optional[jnp.ndarray] = None
    bmns: Optional[jnp.ndarray] = None
    bsubumnc: Optional[jnp.ndarray] = None
    bsubumns: Optional[jnp.ndarray] = None
    bsubvmnc: Optional[jnp.ndarray] = None
    bsubvmns: Optional[jnp.ndarray] = None
    Boozer_I_all: Optional[_np.ndarray] = None
    Boozer_G_all: Optional[_np.ndarray] = None
    phip: Optional[jnp.ndarray] = None
    chi: Optional[jnp.ndarray] = None
    pres: Optional[jnp.ndarray] = None
    phi: Optional[jnp.ndarray] = None
    aspect: float = 0.0
    toroidal_flux: float = 0.0

    # Derived quantities set by init_from_vmec or read_boozmn
    compute_surfs: Optional[List[int]] = field(default=None)
    s_b: Optional[_np.ndarray] = None
    ns_in: Optional[int] = None
    ns_b: Optional[int] = None
    Boozer_I: Optional[_np.ndarray] = None
    Boozer_G: Optional[_np.ndarray] = None
    mboz: Optional[int] = None
    nboz: Optional[int] = None
    mnboz: Optional[int] = None
    xm_b: Optional[_np.ndarray] = None
    xn_b: Optional[_np.ndarray] = None
    bmnc_b: Optional[_np.ndarray] = None
    bmns_b: Optional[_np.ndarray] = None
    rmnc_b: Optional[_np.ndarray] = None
    rmns_b: Optional[_np.ndarray] = None
    zmnc_b: Optional[_np.ndarray] = None
    zmns_b: Optional[_np.ndarray] = None
    numnc_b: Optional[_np.ndarray] = None
    numns_b: Optional[_np.ndarray] = None
    gmnc_b: Optional[_np.ndarray] = None
    gmns_b: Optional[_np.ndarray] = None

    # Bookkeeping
    _prepared: bool = False  # whether mode lists and grids have been prepared

    # ------------------------------------------------------------------
    # Delegated methods from external modules
    # ------------------------------------------------------------------

    def init_from_vmec(self, *args, s_in: Optional[_np.ndarray] = None) -> None:
        """
        Load Fourier data from VMEC into this instance.

        This method simply delegates to
        :func:`booz_xform_jax.vmec.init_from_vmec`. See that function
        for the full list of arguments and options.

        Parameters
        ----------
        *args :
            Passed directly to :func:`init_from_vmec`.
        s_in :
            Optional replacement radial grid of normalised toroidal
            flux. If provided, its first element should correspond to
            the axis; this element will be discarded so that
            ``s_in[0]`` on the instance is the first half-grid surface
            away from the axis.
        """
        init_from_vmec(self, *args, s_in=s_in)

    def read_wout(self, filename: str, flux: bool = False) -> None:
        """
        Read a VMEC ``wout`` file and populate the internal arrays.

        This is a thin wrapper around
        :func:`booz_xform_jax.vmec.read_wout`. In addition to the
        core Fourier coefficients needed for the Boozer transform,
        optional flux profile arrays can be loaded when ``flux=True``.

        Parameters
        ----------
        filename :
            Path to the VMEC wout file.
        flux :
            If ``True``, also read radial profile arrays (φ', χ, p, …).
        """
        read_wout(self, filename, flux)

    def read_wout_data(self, wout, flux: bool = False) -> None:
        """
        Populate the instance from an in-memory VMEC wout object.

        This is a thin wrapper around
        :func:`booz_xform_jax.vmec.read_wout_data`.

        Parameters
        ----------
        wout :
            A VMEC wout-like object (e.g. ``vmec_jax.WoutData``).
        flux :
            If ``True``, also read radial profile arrays (φ', χ, p, …) when available.
        """
        read_wout_data(self, wout, flux)

    def write_boozmn(self, filename: str) -> None:
        """
        Write the computed Boozer spectra to a NetCDF file.

        This delegates to :func:`booz_xform_jax.io_utils.write_boozmn`.
        The file format (NetCDF3 vs NetCDF4) depends on the availability
        of the ``netCDF4`` package and mirrors the behaviour of the
        original BOOZ_XFORM code.
        """
        write_boozmn(self, filename)

    def read_boozmn(self, filename: str) -> None:
        """
        Read Boozer spectra from an existing ``boozmn`` file.

        This delegates to :func:`booz_xform_jax.io_utils.read_boozmn`
        and populates the current instance with the data from that file,
        including mode definitions, radial profiles, and Boozer spectra.
        """
        read_boozmn(self, filename)

    # ------------------------------------------------------------------
    # Internal helper routines for preparing mode lists and grids
    # ------------------------------------------------------------------

    def _prepare_mode_lists(self) -> None:
        """
        Construct lists of Boozer mode indices based on ``mboz`` and ``nboz``.

        The enumeration mirrors the original C++ implementation:

          * m runs from 0, 1, ..., ``mboz - 1``.
          * For m == 0, n runs 0, 1, ..., nboz (only non-negative
            toroidal indices).
          * For m > 0, n runs -nboz, ..., -1, 0, 1, ..., nboz.

        The toroidal indices are stored as ``xn_b = n * nfp`` to match
        VMEC conventions (i.e. actual Fourier angle is ``xn_b * ζ``).

        The resulting arrays are stored on ``self.xm_b`` and
        ``self.xn_b``, and the total number of modes on ``self.mnboz``.
        """
        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before preparing mode lists")

        m_list: List[int] = []
        n_list: List[int] = []

        for m in range(self.mboz):
            if m == 0:
                # m = 0 → keep only non-negative n
                for n in range(0, self.nboz + 1):
                    m_list.append(m)
                    n_list.append(n * self.nfp)
            else:
                # m > 0 → keep full range of n
                for n in range(-self.nboz, self.nboz + 1):
                    m_list.append(m)
                    n_list.append(n * self.nfp)

        self.xm_b = _np.asarray(m_list, dtype=int)
        self.xn_b = _np.asarray(n_list, dtype=int)
        self.mnboz = len(self.xm_b)

    def _setup_grids(self) -> None:
        """
        Set up the (theta, zeta) grid and basic bookkeeping.

        This routine constructs a tensor-product grid in VMEC angles,
        following the grid-sizing logic from the original BOOZ_XFORM
        code. The grid is slightly larger than the nominal Boozer
        resolution to comfortably resolve products of harmonics.

        For symmetric equilibria (``asym == False``) we exploit
        stellarator symmetry to restrict θ to [0, π] plus the end
        points. In that case:

          * ``ntheta_full = 2 * (2*mboz + 1)``
          * we use only the first ``nu2_b = ntheta_full//2 + 1`` rows
            in θ, i.e. 0 ≤ θ ≤ π, and apply special 1/2 weights to the
            θ=0 and θ=π rows in the Fourier integrals.

        For asymmetric equilibria (``asym == True``) we use the full
        range θ ∈ [0, 2π); then ``nu3_b = ntheta_full``.

        The flattened grids are stored on ``self._theta_grid`` and
        ``self._zeta_grid``, and grid sizes on:

          * ``self._ntheta``  – total θ points in the full grid.
          * ``self._nzeta``   – total ζ points.
          * ``self._n_theta_zeta`` – product grid size.
          * ``self._nu2_b``   – number of θ rows used in the
            symmetric case.
        """
        if self._prepared:
            return

        if self.mboz is None or self.nboz is None:
            raise RuntimeError("mboz and nboz must be set before setting up grids")

        # Nominal angular resolutions (full θ range)
        ntheta_full = 2 * (2 * self.mboz + 1)
        nzeta_full = 2 * (2 * self.nboz + 1) if self.nboz > 0 else 1
        nu2_b = ntheta_full // 2 + 1  # number of θ rows in [0, π]

        if self.asym:
            # Asymmetric case: keep all θ rows in [0, 2π)
            nu3_b = ntheta_full
        else:
            # Symmetric case: exploit θ → 2π - θ symmetry, keep [0, π]
            nu3_b = nu2_b

        d_theta = (2.0 * jnp.pi) / ntheta_full
        d_zeta = (2.0 * jnp.pi) / (self.nfp * nzeta_full)

        theta_vals = jnp.arange(nu3_b) * d_theta
        zeta_vals = jnp.arange(nzeta_full) * d_zeta

        # Build flattened tensor-product grid:
        #
        #   θ_j = θ_i    for i fixed, repeated over all ζ
        #   ζ_j = ζ_k    tiled over θ rows
        #
        self._theta_grid = jnp.repeat(theta_vals, nzeta_full)
        self._zeta_grid = jnp.tile(zeta_vals, nu3_b)

        self._ntheta = int(ntheta_full)
        self._nzeta = int(nzeta_full)
        self._n_theta_zeta = int(nu3_b * nzeta_full)
        self._nu2_b = nu2_b
        self._prepared = True

    # ------------------------------------------------------------------
    # Main transform
    # ------------------------------------------------------------------

    def run(self, jit: bool = False) -> None:
        """
        Perform the Boozer coordinate transformation on selected surfaces.

        Parameters
        ----------
        jit : bool, optional
            Placeholder flag (currently unused). The transform is
            implemented entirely in terms of JAX array operations
            (``jax.numpy`` and ``einsum``). To avoid large compile
            times on CPU, we do **not** wrap the entire :meth:`run` in
            a single :func:`jax.jit` by default. Small helpers such as
            :func:`_init_trig` *are* jitted.

            Advanced users who want full JIT compilation can wrap
            :meth:`run` externally, but should be aware that this may
            lead to long compilation times for large Boozer resolutions.

        Notes
        -----
        The implementation follows the algorithm outlined in the
        module docstring and in the BOOZ_XFORM documentation. The main
        difference from a direct translation of the Fortran/C++ code is
        that all loops over Fourier modes are vectorised. Only the
        loop over radial surfaces remains as a Python loop.
        """
        _verbose = bool(self.verbose)

        if _verbose:
            pass  # Header printed after grid setup

        # Basic sanity checks: VMEC data must be initialised.
        if self.rmnc is None or self.bmnc is None:
            raise RuntimeError("VMEC data must be initialised before running the transform")
        if self.ns_in is None:
            raise RuntimeError("ns_in must be set; did init_from_vmec run correctly?")

        ns_in = int(self.ns_in)
        if ns_in <= 0:
            raise RuntimeError("ns_in must be positive; did init_from_vmec run correctly?")

        # ------------------------------------------------------------------
        # Surface selection
        # ------------------------------------------------------------------
        # Default: compute on all surfaces.
        if self.compute_surfs is None:
            self.compute_surfs = list(range(ns_in))
        else:
            for idx in self.compute_surfs:
                if idx < 0 or idx >= ns_in:
                    raise ValueError(
                        f"compute_surfs has an entry {idx} outside [0, {ns_in - 1}]"
                    )

        # ------------------------------------------------------------------
        # Boozer mode lists and grids
        # ------------------------------------------------------------------
        # Default Boozer resolution: match VMEC angular resolution.
        if self.mboz is None:
            if self.mpol is None:
                raise RuntimeError("mboz is not set and mpol is not available")
            self.mboz = int(self.mpol)

        if self.nboz is None:
            if self.ntor is None:
                raise RuntimeError("nboz is not set and ntor is not available")
            self.nboz = int(self.ntor)

        if self.mnboz is None or self.xm_b is None or self.xn_b is None:
            self._prepare_mode_lists()

        self._setup_grids()

        if _verbose:
            print(
                f"  0 <= mboz <= {int(self.mboz) - 1:4d}"
                f"   {-int(self.nboz):4d} <= nboz <= {int(self.nboz):4d}"
            )
            print(f"  nu_boz = {self._ntheta:5d} nv_boz = {self._nzeta:5d}")
            print()
            print(
                "             OUTBOARD (u=0)"
                "              JS          INBOARD (u=pi)"
            )
            print("-" * 77)
            print(
                "  v     |B|vmec    |B|booz    Error"
                "             |B|vmec    |B|booz    Error"
            )
            print()

        n_theta_zeta = self._n_theta_zeta
        theta_grid = self._theta_grid
        zeta_grid = self._zeta_grid

        # ------------------------------------------------------------------
        # Precompute trig tables for VMEC spectra (non-Nyquist and Nyquist)
        # and hoist all per-mode trig combinations out of the surface loop.
        # ------------------------------------------------------------------
        xm_non_np = _np.asarray(self.xm, dtype=int)
        xn_non_np = _np.asarray(self.xn, dtype=int)
        xm_nyq_np = _np.asarray(self.xm_nyq, dtype=int)
        xn_nyq_np = _np.asarray(self.xn_nyq, dtype=int)

        # Non-Nyquist (geometry, λ):
        mmax_non = int(_np.max(_np.abs(xm_non_np)))
        nmax_non = int(_np.max(_np.abs(xn_non_np // self.nfp)))
        cosm, sinm, cosn, sinn = _init_trig(
            theta_grid, zeta_grid, mmax_non, nmax_non, self.nfp
        )

        # Nyquist (w, |B|):
        mmax_nyq = int(_np.max(_np.abs(xm_nyq_np)))
        nmax_nyq = int(_np.max(_np.abs(xn_nyq_np // self.nfp)))
        cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = _init_trig(
            theta_grid, zeta_grid, mmax_nyq, nmax_nyq, self.nfp
        )

        # Convert mode index lists to JAX arrays once (reused per surface).
        xm_non = jnp.asarray(xm_non_np, dtype=jnp.int32)
        xn_non = jnp.asarray(xn_non_np, dtype=jnp.int32)
        xm_nyq = jnp.asarray(xm_nyq_np, dtype=jnp.int32)
        xn_nyq = jnp.asarray(xn_nyq_np, dtype=jnp.int32)

        xm_b_j = jnp.asarray(self.xm_b, dtype=jnp.int32)
        xn_b_j = jnp.asarray(self.xn_b, dtype=jnp.int32)

        # Index of (m=0, n=0) Nyquist mode → Boozer I, G.
        idx00_candidates = _np.where((xm_nyq_np == 0) & (xn_nyq_np == 0))[0]
        if len(idx00_candidates) == 0:
            raise RuntimeError("Could not find (m=0,n=0) Nyquist mode in xm_nyq/xn_nyq")
        idx00 = int(idx00_candidates[0])

        # -------------------------
        # Hoisted non-Nyquist trig combinations
        # -------------------------
        # Shapes:
        #   cosm_m_non, sinm_m_non : (N, mnmax_non)
        #   cosn_n_non, sinn_n_non : (N, mnmax_non)
        cosm_m_non = cosm[:, xm_non_np]
        sinm_m_non = sinm[:, xm_non_np]

        abs_n_non = jnp.abs(xn_non // self.nfp)
        abs_n_non_idx = _np.asarray(abs_n_non, dtype=int)
        cosn_n_non = cosn[:, abs_n_non_idx]
        sinn_n_non = sinn[:, abs_n_non_idx]

        sign_non = jnp.where(xn_non < 0, -1.0, 1.0)[None, :]

        # tcos_non / tsin_non: trigonometric factors multiplying
        # Fourier coefficients for rmnc, zmns, lmns, etc.
        tcos_non = cosm_m_non * cosn_n_non + sinm_m_non * sinn_n_non * sign_non
        tsin_non = sinm_m_non * cosn_n_non - cosm_m_non * sinn_n_non * sign_non

        m_non_f = xm_non.astype(jnp.float64)
        n_non_f = xn_non.astype(jnp.float64)

        # -------------------------
        # Hoisted Nyquist trig combinations
        # -------------------------
        cosm_m_nyq = cosm_nyq[:, xm_nyq_np]
        sinm_m_nyq = sinm_nyq[:, xm_nyq_np]

        abs_n_nyq = jnp.abs(xn_nyq // self.nfp)
        abs_n_nyq_idx = _np.asarray(abs_n_nyq, dtype=int)
        cosn_n_nyq = cosn_nyq[:, abs_n_nyq_idx]
        sinn_n_nyq = sinn_nyq[:, abs_n_nyq_idx]

        sign_nyq = jnp.where(xn_nyq < 0, -1.0, 1.0)[None, :]

        tcos_nyq = cosm_m_nyq * cosn_n_nyq + sinm_m_nyq * sinn_n_nyq * sign_nyq
        tsin_nyq = sinm_m_nyq * cosn_n_nyq - cosm_m_nyq * sinn_n_nyq * sign_nyq

        m_nyq_f = xm_nyq.astype(jnp.float64)
        n_nyq_f = xn_nyq.astype(jnp.float64)

        # ------------------------------------------------------------------
        # Convert all hoisted JAX arrays to NumPy once.
        # This eliminates every JAX dispatch and device→host sync from the
        # per-surface loop — replacing jnp.einsum with numpy matmul (@).
        # ------------------------------------------------------------------
        tcos_non_np    = _np.asarray(tcos_non)       # (N, mnmax_non)
        tsin_non_np    = _np.asarray(tsin_non)
        tcos_nyq_np    = _np.asarray(tcos_nyq)       # (N, mnmax_nyq)
        tsin_nyq_np    = _np.asarray(tsin_nyq)
        m_non_f_np     = _np.asarray(m_non_f)        # (mnmax_non,)
        n_non_f_np     = _np.asarray(n_non_f)
        m_nyq_f_np     = _np.asarray(m_nyq_f)        # (mnmax_nyq,)
        n_nyq_f_np     = _np.asarray(n_nyq_f)
        theta_grid_np  = _np.asarray(theta_grid)     # (N,)
        zeta_grid_np   = _np.asarray(zeta_grid)

        # VMEC coefficient arrays (per-surface slices become plain numpy views)
        rmnc_arr      = _np.asarray(self.rmnc)        # (mnmax_non, ns_in)
        zmns_arr      = _np.asarray(self.zmns)
        lmns_arr      = _np.asarray(self.lmns)
        bmnc_arr      = _np.asarray(self.bmnc)        # (mnmax_nyq, ns_in)
        bsubumnc_arr  = _np.asarray(self.bsubumnc)
        bsubvmnc_arr  = _np.asarray(self.bsubvmnc)
        iota_arr      = _np.asarray(self.iota)        # (ns_in,)

        if self.asym:
            rmns_arr     = _np.asarray(self.rmns)    if self.rmns     is not None else None
            zmnc_arr     = _np.asarray(self.zmnc)    if self.zmnc     is not None else None
            lmnc_arr     = _np.asarray(self.lmnc)    if self.lmnc     is not None else None
            bmns_arr     = _np.asarray(self.bmns)    if self.bmns     is not None else None
            bsubumns_arr = _np.asarray(self.bsubumns) if self.bsubumns is not None else None
            bsubvmns_arr = _np.asarray(self.bsubvmns) if self.bsubvmns is not None else None
        else:
            rmns_arr = zmnc_arr = lmnc_arr = bmns_arr = None
            bsubumns_arr = bsubvmns_arr = None

        # Hoist wmns boolean masks and safe-divisors out of the surface loop.
        m_nonzero_np       = m_nyq_f_np != 0.0
        n_nonzero_only_np  = ~m_nonzero_np & (n_nyq_f_np != 0.0)
        m_nyq_f_safe       = _np.where(m_nonzero_np,      m_nyq_f_np, 1.0)
        n_nyq_f_safe       = _np.where(n_nonzero_only_np, n_nyq_f_np, 1.0)

        # ------------------------------------------------------------------
        # Output arrays (NumPy, host side)
        # ------------------------------------------------------------------
        ns_b = len(self.compute_surfs)
        self.ns_b = ns_b
        mnboz = int(self.mnboz)

        bmnc_b = _np.zeros((mnboz, ns_b), dtype=float)
        rmnc_b = _np.zeros((mnboz, ns_b), dtype=float)
        zmns_b = _np.zeros((mnboz, ns_b), dtype=float)
        numns_b = _np.zeros((mnboz, ns_b), dtype=float)
        gmnc_b = _np.zeros((mnboz, ns_b), dtype=float)

        if self.asym:
            bmns_b = _np.zeros((mnboz, ns_b), dtype=float)
            rmns_b = _np.zeros((mnboz, ns_b), dtype=float)
            zmnc_b = _np.zeros((mnboz, ns_b), dtype=float)
            numnc_b = _np.zeros((mnboz, ns_b), dtype=float)
            gmns_b = _np.zeros((mnboz, ns_b), dtype=float)
        else:
            bmns_b = rmns_b = zmnc_b = numnc_b = gmns_b = None

        # Batch-extract Boozer I and G for all selected surfaces at once
        # (avoids one device→host transfer per surface inside the loop).
        _surfs_np = _np.asarray(self.compute_surfs, dtype=int)
        Boozer_I = _np.asarray(self.bsubumnc[idx00, _surfs_np], dtype=float)
        Boozer_G = _np.asarray(self.bsubvmnc[idx00, _surfs_np], dtype=float)

        # ------------------------------------------------------------------
        # Batch VMEC synthesis — compute all surface fields in one DGEMM.
        # tcos_non_np is (N, mnmax_non); rmnc_arr[:, surfs] is (mnmax_non, ns_b)
        # Result: (N, ns_b) per field.  Keeps the trig matrix in L3/L4 cache
        # rather than re-reading it for each of the ns_b surfaces.
        # ------------------------------------------------------------------
        _lmns_s    = lmns_arr[:, _surfs_np]                          # (mnmax_non, ns_b)
        _lmns_m_s  = _lmns_s * m_non_f_np[:, None]                   # pre-scaled
        _lmns_n_s  = _lmns_s * n_non_f_np[:, None]
        _r_all       = tcos_non_np @ rmnc_arr[:, _surfs_np]           # (N, ns_b)
        _z_all       = tsin_non_np @ zmns_arr[:, _surfs_np]
        _lam_all     = tsin_non_np @ _lmns_s
        _dlam_dth_all = tcos_non_np @ _lmns_m_s
        _dlam_dze_all = -(tcos_non_np @ _lmns_n_s)

        # wmns for all surfaces at once: (mnmax_nyq, ns_b)
        _bsubumnc_s = bsubumnc_arr[:, _surfs_np]                      # (mnmax_nyq, ns_b)
        _bsubvmnc_s = bsubvmnc_arr[:, _surfs_np]
        _wmns_all   = _np.where(m_nonzero_np[:, None],
                                _bsubumnc_s / m_nyq_f_safe[:, None],
                                _np.where(n_nonzero_only_np[:, None],
                                          -_bsubvmnc_s / n_nyq_f_safe[:, None], 0.0))
        _wmns_m_s   = _wmns_all * m_nyq_f_np[:, None]
        _wmns_n_s   = _wmns_all * n_nyq_f_np[:, None]
        _w_all       = tsin_nyq_np @ _wmns_all                        # (N, ns_b)
        _dw_dth_all  = tcos_nyq_np @ _wmns_m_s
        _dw_dze_all  = -(tcos_nyq_np @ _wmns_n_s)
        _bmod_all    = tcos_nyq_np @ bmnc_arr[:, _surfs_np]           # (N, ns_b)

        if self.asym:
            if lmnc_arr is not None:
                _lmnc_s   = lmnc_arr[:, _surfs_np]
                _lmnc_m_s = _lmnc_s * m_non_f_np[:, None]
                _lmnc_n_s = _lmnc_s * n_non_f_np[:, None]
                _r_all       = _r_all       + tsin_non_np @ rmns_arr[:, _surfs_np]
                _z_all       = _z_all       + tcos_non_np @ zmnc_arr[:, _surfs_np]
                _lam_all     = _lam_all     + tcos_non_np @ _lmnc_s
                _dlam_dth_all = _dlam_dth_all - tsin_non_np @ _lmnc_m_s
                _dlam_dze_all = _dlam_dze_all + tsin_non_np @ _lmnc_n_s
            if bsubumns_arr is not None:
                _bsubumns_s = bsubumns_arr[:, _surfs_np]
                _bsubvmns_s = bsubvmns_arr[:, _surfs_np]
                _wmnc_all = _np.where(m_nonzero_np[:, None],
                                      -_bsubumns_s / m_nyq_f_safe[:, None],
                                      _np.where(n_nonzero_only_np[:, None],
                                                _bsubvmns_s / n_nyq_f_safe[:, None], 0.0))
                _wmnc_m_s = _wmnc_all * m_nyq_f_np[:, None]
                _wmnc_n_s = _wmnc_all * n_nyq_f_np[:, None]
                _w_all      = _w_all      + tcos_nyq_np @ _wmnc_all
                _dw_dth_all = _dw_dth_all - tsin_nyq_np @ _wmnc_m_s
                _dw_dze_all = _dw_dze_all + tsin_nyq_np @ _wmnc_n_s
                _bmod_all   = _bmod_all   + tsin_nyq_np @ bmns_arr[:, _surfs_np]

        # Pre-allocate reusable scratch buffers for the double-spectral step.
        # This eliminates ~10 small allocations per surface.
        _N  = int(theta_grid_np.shape[0])
        _nb = int(self.nboz) + 1
        _mb = int(self.mboz) + 1
        _fcn_buf = _np.empty((_N, _nb), dtype=float)
        _fsn_buf = _np.empty((_N, _nb), dtype=float)
        _Xc_buf  = _np.empty((_mb, _nb), dtype=float)
        _Xs_buf  = _np.empty((_mb, _nb), dtype=float)
        _Ysc_buf = _np.empty((_mb, _nb), dtype=float)
        _Ycs_buf = _np.empty((_mb, _nb), dtype=float)

        # ------------------------------------------------------------------
        # Hoist Boozer-mode index arrays out of the surface loop.
        # These depend only on xm_b / xn_b which are constant across surfaces.
        # Computing them inside the loop triggers repeated device→host syncs.
        # ------------------------------------------------------------------
        _m_b_np_idx   = _np.asarray(xm_b_j, dtype=int)         # (mnboz,)
        _abs_n_b_np   = _np.asarray(jnp.abs(xn_b_j // self.nfp), dtype=int)  # (mnboz,)
        _sign_b_hoisted = jnp.where(xn_b_j < 0, -1.0, 1.0)[None, :]  # (1, mnboz)

        # Fourier normalisation factor (constant: depends only on grid sizes)
        _fourier_factor0 = (
            2.0 / (self._ntheta * self._nzeta) if self.asym
            else 2.0 / ((self._nu2_b - 1) * self._nzeta)
        )
        _fourier_factor = jnp.ones((mnboz,), dtype=jnp.float64) * _fourier_factor0
        _fourier_factor = _fourier_factor.at[0].set(_fourier_factor0 * 0.5)

        # ------------------------------------------------------------------
        # Chunk size for memory-bounded Fourier integrals.
        # Each chunk allocates 2×(N×L)×8 bytes for tcos_c / tsin_c.
        # Default cap: 200 MB; override via BOOZ_XFORM_JAX_CHUNK_BYTES env var.
        # ------------------------------------------------------------------
        # NumPy copies of constant mode index/weight arrays
        _m_b_chunk  = _np.asarray(_m_b_np_idx)        # (mnboz,) int
        _n_b_chunk  = _np.asarray(_abs_n_b_np)         # (mnboz,) int
        _ff_chunk   = _np.asarray(_fourier_factor)     # (mnboz,) float
        _sgn_chunk  = _np.asarray(_sign_b_hoisted[0])  # (mnboz,) float

        # NumPy copies for the verbose modbooz reconstruction
        if _verbose:
            _xm_b_np_f = _np.asarray(xm_b_j, dtype=float)
            _xn_b_np_f = _np.asarray(xn_b_j, dtype=float)

        # Convenience indices for symmetric θ integration (θ=0 and θ=π rows).
        # NumPy integer arrays — used for in-place *= 0.5 on numpy trig tables.
        idx_theta0  = _np.arange(0, self._nzeta)
        idx_thetapi = _np.arange(
            (self._nu2_b - 1) * self._nzeta, self._nu2_b * self._nzeta
        )

        # Fixed-point indices for Fortran-style accuracy check
        # (u=0,v=0), (u=pi,v=0), (u=0,v=pi), (u=pi,v=pi)
        nv2_b_idx = self._nzeta // 2  # 0-based index for v=pi
        idx_00 = 0
        idx_pi0 = (self._nu2_b - 1) * self._nzeta
        idx_0pi = nv2_b_idx
        idx_pipi = (self._nu2_b - 1) * self._nzeta + nv2_b_idx

        # ------------------------------------------------------------------
        # Loop over surfaces js_b (Python loop; heavy math is vectorised)
        # ------------------------------------------------------------------
        for js_b, js in enumerate(self.compute_surfs):
            if isinstance(self.verbose, int) and self.verbose > 1:
                print(f"[booz_xform_jax] Solving surface js_b={js_b}, js={js}")

            # ------------------------------------------------------------------
            # 2) Boozer I and G (already batch-extracted before the loop)
            # ------------------------------------------------------------------
            Boozer_I_js = Boozer_I[js_b]
            Boozer_G_js = Boozer_G[js_b]

            # ------------------------------------------------------------------
            # 3) R, Z, λ and derivatives — sliced from pre-batched arrays
            # ------------------------------------------------------------------
            r        = _r_all[:, js_b]
            z        = _z_all[:, js_b]
            lam      = _lam_all[:, js_b]
            dlam_dth = _dlam_dth_all[:, js_b]
            dlam_dze = _dlam_dze_all[:, js_b]

            # ------------------------------------------------------------------
            # 4) w, ∂w/∂θ, ∂w/∂ζ and |B| — sliced from pre-batched arrays
            # ------------------------------------------------------------------
            w      = _w_all[:, js_b]
            dw_dth = _dw_dth_all[:, js_b]
            dw_dze = _dw_dze_all[:, js_b]
            bmod   = _bmod_all[:, js_b]

            # ------------------------------------------------------------------
            # 5) ν, Boozer angles, their derivatives, J_B, and dB/d(vmec)
            # ------------------------------------------------------------------
            this_iota  = float(iota_arr[js])
            GI         = Boozer_G_js + this_iota * Boozer_I_js
            one_over_GI = 1.0 / GI

            # ν from eq (10): ν = (w - I λ) / (G + ι I)
            nu = one_over_GI * (w - Boozer_I_js * lam)

            # Boozer angles from eq (3):
            #   θ_B = θ + λ + ι ν
            #   ζ_B = ζ + ν
            theta_B = theta_grid_np + lam + this_iota * nu
            zeta_B  = zeta_grid_np  + nu

            # Derivatives of ν:
            dnu_dze = one_over_GI * (dw_dze - Boozer_I_js * dlam_dze)
            dnu_dth = one_over_GI * (dw_dth - Boozer_I_js * dlam_dth)

            # Eq (12): dB/d(vmec) factor
            dB_dvmec = (1.0 + dlam_dth) * (1.0 + dnu_dze) + \
                       (this_iota - dlam_dze) * dnu_dth

            # Store VMEC-space |B| at 4 fixed points for accuracy check later
            if _verbose:
                bmodv = (
                    float(bmod[idx_00]),   # (u=0, v=0)
                    float(bmod[idx_pi0]),  # (u=pi, v=0)
                    float(bmod[idx_0pi]),  # (u=0, v=pi)
                    float(bmod[idx_pipi]), # (u=pi, v=pi)
                )
                u_b = (
                    float(theta_B[idx_00]),  float(theta_B[idx_pi0]),
                    float(theta_B[idx_0pi]), float(theta_B[idx_pipi]),
                )
                v_b = (
                    float(zeta_B[idx_00]),  float(zeta_B[idx_pi0]),
                    float(zeta_B[idx_0pi]), float(zeta_B[idx_pipi]),
                )

            # ------------------------------------------------------------------
            # 6) Boozer trig tables on (theta_B, zeta_B) — pure NumPy
            # ------------------------------------------------------------------
            cosm_b, sinm_b, cosn_b, sinn_b = _init_trig_np(
                theta_B, zeta_B, int(self.mboz), int(self.nboz), self.nfp
            )

            # Boozer Jacobian:  J_B = (G + ι I) / |B|² = GI / |B|²
            boozer_jac = GI / (bmod * bmod)

            # ------------------------------------------------------------------
            # 7) Final Fourier integrals — double-spectral decomposition
            # ------------------------------------------------------------------
            # Separability of the Boozer trig factor:
            #   tcos[i, j] = cosm[i,m_j]*cosn[i,n_j] + sinm[i,m_j]*sinn[i,n_j]*sgn_j
            #   tsin[i, j] = sinm[i,m_j]*cosn[i,n_j] - cosm[i,m_j]*sinn[i,n_j]*sgn_j
            #
            # The Fourier integral factors as two tiny DGEMM calls per field:
            #   X_c[m, n] = cosm.T @ (field * cosn)   shape (mboz+1, nboz+1)
            #   X_s[m, n] = sinm.T @ (field * sinn)
            #   Y_sc[m,n] = sinm.T @ (field * cosn)
            #   Y_cs[m,n] = cosm.T @ (field * sinn)
            #
            # Then scatter: cos_out[j] = ff_j*(X_c[m_j,n_j] + sgn_j*X_s[m_j,n_j])
            #                sin_out[j] = ff_j*(Y_sc[m_j,n_j] - sgn_j*Y_cs[m_j,n_j])
            #
            # Peak memory: O(N*(mboz+1)) not O(N*mnboz); no chunk loop needed.

            # Apply symmetric half-weight to a dB copy (not to trig tables)
            _dB = dB_dvmec.copy() if not self.asym else dB_dvmec
            if not self.asym:
                _dB[idx_theta0]  *= 0.5
                _dB[idx_thetapi] *= 0.5

            # Reusable transpose views (no copy — BLAS handles Fortran order)
            _cmT = cosm_b.T   # (mboz+1, N)
            _smT = sinm_b.T   # (mboz+1, N)

            _cos_out = _np.empty((5 if self.asym else 3, mnboz), dtype=float)
            _sin_out = _np.empty((5 if self.asym else 2, mnboz), dtype=float)

            if self.asym:
                # All 5 fields contribute to both cosine and sine output
                _field_list = [bmod, r, z, nu, boozer_jac]
                for k, fk in enumerate(_field_list):
                    _fkdB = fk * _dB                    # (N,) weighted field
                    _np.multiply(_fkdB[:, None], cosn_b, out=_fcn_buf)  # (N, nboz+1)
                    _np.multiply(_fkdB[:, None], sinn_b, out=_fsn_buf)  # (N, nboz+1)
                    _np.dot(_cmT, _fcn_buf, out=_Xc_buf)                # (mboz+1, nboz+1)
                    _np.dot(_smT, _fsn_buf, out=_Xs_buf)
                    _np.dot(_smT, _fcn_buf, out=_Ysc_buf)
                    _np.dot(_cmT, _fsn_buf, out=_Ycs_buf)
                    _cos_out[k] = _ff_chunk * (
                        _Xc_buf[_m_b_chunk, _n_b_chunk] + _sgn_chunk * _Xs_buf[_m_b_chunk, _n_b_chunk]
                    )
                    _sin_out[k] = _ff_chunk * (
                        _Ysc_buf[_m_b_chunk, _n_b_chunk] - _sgn_chunk * _Ycs_buf[_m_b_chunk, _n_b_chunk]
                    )
            else:
                # Cosine-output fields: bmod, r, jac
                for k, fk in enumerate([bmod, r, boozer_jac]):
                    _fkdB = fk * _dB
                    _np.multiply(_fkdB[:, None], cosn_b, out=_fcn_buf)
                    _np.multiply(_fkdB[:, None], sinn_b, out=_fsn_buf)
                    _np.dot(_cmT, _fcn_buf, out=_Xc_buf)
                    _np.dot(_smT, _fsn_buf, out=_Xs_buf)
                    _cos_out[k] = _ff_chunk * (
                        _Xc_buf[_m_b_chunk, _n_b_chunk] + _sgn_chunk * _Xs_buf[_m_b_chunk, _n_b_chunk]
                    )
                # Sine-output fields: z, nu
                for k, fk in enumerate([z, nu]):
                    _fkdB = fk * _dB
                    _np.multiply(_fkdB[:, None], cosn_b, out=_fcn_buf)
                    _np.multiply(_fkdB[:, None], sinn_b, out=_fsn_buf)
                    _np.dot(_smT, _fcn_buf, out=_Ysc_buf)
                    _np.dot(_cmT, _fsn_buf, out=_Ycs_buf)
                    _sin_out[k] = _ff_chunk * (
                        _Ysc_buf[_m_b_chunk, _n_b_chunk] - _sgn_chunk * _Ycs_buf[_m_b_chunk, _n_b_chunk]
                    )

            # Write to NumPy output buffers (no .asarray needed — already host)
            bmnc_b[:, js_b] = _cos_out[0]
            rmnc_b[:, js_b] = _cos_out[1]
            if self.asym:
                zmnc_b[:, js_b]  = _cos_out[2]
                numnc_b[:, js_b] = _cos_out[3]
                gmnc_b[:, js_b]  = _cos_out[4]
                bmns_b[:, js_b]  = _sin_out[0]
                rmns_b[:, js_b]  = _sin_out[1]
                zmns_b[:, js_b]  = _sin_out[2]
                numns_b[:, js_b] = _sin_out[3]
                gmns_b[:, js_b]  = _sin_out[4]
            else:
                gmnc_b[:, js_b]  = _cos_out[2]
                zmns_b[:, js_b]  = _sin_out[0]
                numns_b[:, js_b] = _sin_out[1]

            # Fortran-style accuracy check: reconstruct |B| at 4 fixed
            # Boozer-angle points and compare with VMEC real-space |B|.
            if _verbose:
                # jrad is the 1-based full-grid index (Fortran convention)
                jrad = js + 2

                # Vectorised modbooz: u_b/v_b are (4,), _xm_b_np_f/_xn_b_np_f (mnboz,)
                # -> angles (4, mnboz), then sum over modes
                u_b_arr = _np.array(u_b)    # (4,)
                v_b_arr = _np.array(v_b)    # (4,)
                sgn_arr = _np.where(_xn_b_np_f >= 0, 1.0, -1.0)  # (mnboz,)
                n_abs_arr = _np.abs(_xn_b_np_f) / self.nfp        # (mnboz,)

                cosm_4 = _np.cos(_xm_b_np_f[None, :] * u_b_arr[:, None])            # (4, mnboz)
                sinm_4 = _np.sin(_xm_b_np_f[None, :] * u_b_arr[:, None])
                cosn_4 = _np.cos(n_abs_arr[None, :] * v_b_arr[:, None] * self.nfp)
                sinn_4 = _np.sin(n_abs_arr[None, :] * v_b_arr[:, None] * self.nfp)

                cost_4 = cosm_4 * cosn_4 + sinm_4 * sinn_4 * sgn_arr[None, :]       # (4, mnboz)
                bmodb_arr = cost_4 @ bmnc_b[:, js_b]                                  # already numpy

                if self.asym:
                    sint_4 = sinm_4 * cosn_4 - cosm_4 * sinn_4 * sgn_arr[None, :]
                    bmodb_arr = bmodb_arr + sint_4 @ bmns_b[:, js_b]

                bmodv_arr = _np.array(bmodv)
                err_arr = _np.abs(bmodb_arr - bmodv_arr) / _np.maximum(
                    _np.abs(bmodb_arr), _np.maximum(_np.abs(bmodv_arr), 1e-30)
                )
                bmodb = bmodb_arr.tolist()
                err   = err_arr.tolist()

                print(
                    f"  0  {bmodv[0]:11.3E}{bmodb[0]:11.3E}{err[0]:11.3E}"
                    f"{jrad:5d}  {bmodv[1]:11.3E}{bmodb[1]:11.3E}{err[1]:11.3E}"
                )
                print(
                    f" pi  {bmodv[2]:11.3E}{bmodb[2]:11.3E}{err[2]:11.3E}"
                    f"       {bmodv[3]:11.3E}{bmodb[3]:11.3E}{err[3]:11.3E}"
                )

        # ------------------------------------------------------------------
        # Store results on the instance
        # ------------------------------------------------------------------
        self.bmnc_b = bmnc_b
        self.rmnc_b = rmnc_b
        self.zmns_b = zmns_b
        self.numns_b = numns_b
        self.gmnc_b = gmnc_b
        if self.asym:
            self.bmns_b = bmns_b
            self.rmns_b = rmns_b
            self.zmnc_b = zmnc_b
            self.numnc_b = numnc_b
            self.gmns_b = gmns_b

        self.Boozer_I = Boozer_I
        self.Boozer_G = Boozer_G
        self.s_b = _np.asarray(self.s_in)[self.compute_surfs]

    def run_jax(self, *, jit: bool = True) -> dict:
        """Run a JAX-native Boozer transform (no Python surface loop).

        This method returns a mapping compatible with boozmn field names.
        The mapping includes ``gmnc_b`` and its BOOZ_XFORM-compatible ``gmn_b``
        alias for Boozer Jacobian harmonics, plus asymmetric spectra when
        ``asym`` is true.
        It is intended for end-to-end JIT/differentiable workflows and
        does not populate the instance attributes (unlike `run`).
        """
        if self.rmnc is None or self.bmnc is None:
            raise RuntimeError("VMEC data must be initialised before running the transform")
        if self.ns_in is None:
            raise RuntimeError("ns_in must be set; did init_from_vmec run correctly?")

        # Default surfaces: all half-grid surfaces.
        if self.compute_surfs is None:
            compute_surfs = list(range(int(self.ns_in)))
        else:
            compute_surfs = list(self.compute_surfs)

        # Default Boozer resolution: match VMEC angular resolution.
        if self.mboz is None:
            if self.mpol is None:
                raise RuntimeError("mboz is not set and mpol is not available")
            self.mboz = int(self.mpol)
        if self.nboz is None:
            if self.ntor is None:
                raise RuntimeError("nboz is not set and ntor is not available")
            self.nboz = int(self.ntor)

        if self.mnboz is None or self.xm_b is None or self.xn_b is None:
            self._prepare_mode_lists()

        constants, grids = prepare_booz_xform_constants(
            nfp=int(self.nfp),
            mboz=int(self.mboz),
            nboz=int(self.nboz),
            asym=bool(self.asym),
            xm=self.xm,
            xn=self.xn,
            xm_nyq=self.xm_nyq,
            xn_nyq=self.xn_nyq,
        )

        # Ensure surface dimension is first (ns, mn)
        rmnc = jnp.asarray(_np.asarray(self.rmnc)).T
        zmns = jnp.asarray(_np.asarray(self.zmns)).T
        lmns = jnp.asarray(_np.asarray(self.lmns)).T
        bmnc = jnp.asarray(_np.asarray(self.bmnc)).T
        bsubumnc = jnp.asarray(_np.asarray(self.bsubumnc)).T
        bsubvmnc = jnp.asarray(_np.asarray(self.bsubvmnc)).T
        iota = jnp.asarray(_np.asarray(self.iota))

        rmns = jnp.asarray(_np.asarray(self.rmns)).T if self.asym and self.rmns is not None else None
        zmnc = jnp.asarray(_np.asarray(self.zmnc)).T if self.asym and self.zmnc is not None else None
        lmnc = jnp.asarray(_np.asarray(self.lmnc)).T if self.asym and self.lmnc is not None else None
        bmns = jnp.asarray(_np.asarray(self.bmns)).T if self.asym and self.bmns is not None else None
        bsubumns = (
            jnp.asarray(_np.asarray(self.bsubumns)).T if self.asym and self.bsubumns is not None else None
        )
        bsubvmns = (
            jnp.asarray(_np.asarray(self.bsubvmns)).T if self.asym and self.bsubvmns is not None else None
        )

        surface_indices = jnp.asarray(compute_surfs, dtype=jnp.int32)

        booz_fn = booz_xform_jax_impl
        if jit:
            booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))

        return booz_fn(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc,
            bsubumnc=bsubumnc,
            bsubvmnc=bsubvmnc,
            iota=iota,
            xm=jnp.asarray(self.xm, dtype=jnp.int32),
            xn=jnp.asarray(self.xn, dtype=jnp.int32),
            xm_nyq=jnp.asarray(self.xm_nyq, dtype=jnp.int32),
            xn_nyq=jnp.asarray(self.xn_nyq, dtype=jnp.int32),
            constants=constants,
            grids=grids,
            rmns=rmns,
            zmnc=zmnc,
            lmnc=lmnc,
            bmns=bmns,
            bsubumns=bsubumns,
            bsubvmns=bsubvmns,
            surface_indices=surface_indices,
        )

    # ------------------------------------------------------------------
    # Surface registration (unchanged API)
    # ------------------------------------------------------------------

    def register_surfaces(self, s: Iterable[int | float] | int | float) -> None:
        """
        Register one or more surfaces on which to compute the transform.

        This method mirrors the original C++ ``register`` routine. It
        accepts either integer half-grid indices or floating-point
        radial coordinate values in normalised toroidal flux space.

        Parameters
        ----------
        s : int, float, or iterable of these
            Surfaces to register:

              * If an integer, it is interpreted as an index on the
                VMEC half grid (0 ≤ index < ns_in).
              * If a float, it should lie in [0, 1] and is interpreted
                as a normalised toroidal flux value. We then choose
                the nearest index based on ``self.s_in``.

        Notes
        -----
        * Any new surfaces are **appended** to the existing
          :attr:`compute_surfs` list (duplicates are removed).
        * Surfaces outside the valid index range produce a
          :class:`ValueError`.
        * The method does not perform the transform; you must call
          :meth:`run` afterwards.
        """
        # Normalise input to a list
        if isinstance(s, (int, float)):
            ss = [s]
        else:
            ss = list(s)

        if self.compute_surfs is None:
            current = set()
        else:
            current = set(self.compute_surfs)

        for val in ss:
            if isinstance(val, int):
                # Integer: treated as direct index
                idx = val
            else:
                # Float: map to nearest index based on s_in
                sval = float(val)
                if sval < 0.0 or sval > 1.0:
                    raise ValueError("Normalized toroidal flux values must lie in [0,1]")
                idx = int(_np.argmin(_np.abs(self.s_in - sval)))  # type: ignore[arg-type]

            if idx < 0 or idx >= int(self.ns_in):
                raise ValueError(
                    f"Surface index {idx} is outside the range [0, {int(self.ns_in) - 1}]"
                )
            current.add(idx)

        self.compute_surfs = sorted(current)
        # Respect the verbose flag: only print when truthy
        if bool(self.verbose):
            print(f"[booz_xform_jax] Registered surfaces: {self.compute_surfs}")
        return None
