"""Pure JAX API for end-to-end Boozer transforms.

This module provides a JIT-friendly, functional interface that avoids
Python loops over surfaces and keeps all arrays in JAX. It is intended
for end-to-end differentiation with vmec_jax -> booz_xform_jax -> neo_jax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from .trig import _init_trig


@dataclass(frozen=True)
class BoozXformConstants:
    """Static constants for the JAX Boozer transform."""

    nfp: int
    mboz: int
    nboz: int
    asym: bool
    ntheta: int
    nzeta: int
    nu2_b: int
    mmax_non: int
    nmax_non: int
    mmax_nyq: int
    nmax_nyq: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoozXformGrids:
    """Grid arrays for the JAX Boozer transform."""

    theta_grid: jnp.ndarray
    zeta_grid: jnp.ndarray
    xm_b: jnp.ndarray
    xn_b: jnp.ndarray

    def tree_flatten(self):
        children = (self.theta_grid, self.zeta_grid, self.xm_b, self.xn_b)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        theta_grid, zeta_grid, xm_b, xn_b = children
        return cls(theta_grid=theta_grid, zeta_grid=zeta_grid, xm_b=xm_b, xn_b=xn_b)


def _prepare_mode_lists(mboz: int, nboz: int, nfp: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare Boozer mode indices following the C++/Fortran convention."""
    m_list: list[int] = []
    n_list: list[int] = []
    for m in range(mboz):
        if m == 0:
            for n in range(0, nboz + 1):
                m_list.append(m)
                n_list.append(n * nfp)
        else:
            for n in range(-nboz, nboz + 1):
                m_list.append(m)
                n_list.append(n * nfp)
    return np.asarray(m_list, dtype=int), np.asarray(n_list, dtype=int)


def _prepare_grids(mboz: int, nboz: int, nfp: int, asym: bool) -> Tuple[int, int, int, jnp.ndarray, jnp.ndarray]:
    """Prepare flattened (theta, zeta) grids following BOOZ_XFORM conventions."""
    ntheta_full = 2 * (2 * mboz + 1)
    nzeta_full = 2 * (2 * nboz + 1) if nboz > 0 else 1
    nu2_b = ntheta_full // 2 + 1
    nu3_b = ntheta_full if asym else nu2_b

    d_theta = (2.0 * jnp.pi) / ntheta_full
    d_zeta = (2.0 * jnp.pi) / (nfp * nzeta_full)

    theta_vals = jnp.arange(nu3_b) * d_theta
    zeta_vals = jnp.arange(nzeta_full) * d_zeta

    theta_grid = jnp.repeat(theta_vals, nzeta_full)
    zeta_grid = jnp.tile(zeta_vals, nu3_b)

    return int(ntheta_full), int(nzeta_full), int(nu2_b), theta_grid, zeta_grid


def prepare_booz_xform_constants(
    *,
    nfp: int,
    mboz: int,
    nboz: int,
    asym: bool,
    xm: Sequence[int],
    xn: Sequence[int],
    xm_nyq: Sequence[int],
    xn_nyq: Sequence[int],
) -> tuple[BoozXformConstants, BoozXformGrids]:
    """Compute static constants for the JAX Boozer transform.

    This helper runs on the host and can be used before JIT compilation.
    """
    xm_np = np.asarray(xm, dtype=int)
    xn_np = np.asarray(xn, dtype=int)
    xm_nyq_np = np.asarray(xm_nyq, dtype=int)
    xn_nyq_np = np.asarray(xn_nyq, dtype=int)

    mmax_non = int(np.max(np.abs(xm_np)))
    nmax_non = int(np.max(np.abs(xn_np // nfp)))
    mmax_nyq = int(np.max(np.abs(xm_nyq_np)))
    nmax_nyq = int(np.max(np.abs(xn_nyq_np // nfp)))

    ntheta, nzeta, nu2_b, theta_grid, zeta_grid = _prepare_grids(mboz, nboz, nfp, asym)
    xm_b, xn_b = _prepare_mode_lists(mboz, nboz, nfp)

    constants = BoozXformConstants(
        nfp=nfp,
        mboz=mboz,
        nboz=nboz,
        asym=asym,
        ntheta=ntheta,
        nzeta=nzeta,
        nu2_b=nu2_b,
        mmax_non=mmax_non,
        nmax_non=nmax_non,
        mmax_nyq=mmax_nyq,
        nmax_nyq=nmax_nyq,
    )

    grids = BoozXformGrids(
        theta_grid=theta_grid,
        zeta_grid=zeta_grid,
        xm_b=jnp.asarray(xm_b, dtype=jnp.int32),
        xn_b=jnp.asarray(xn_b, dtype=jnp.int32),
    )

    return constants, grids


def _surface_transform(
    rmnc: jnp.ndarray,
    zmns: jnp.ndarray,
    lmns: jnp.ndarray,
    bmnc: jnp.ndarray,
    bsubumnc: jnp.ndarray,
    bsubvmnc: jnp.ndarray,
    iota: jnp.ndarray,
    *,
    constants: BoozXformConstants,
    grids: BoozXformGrids,
    tcos_non: jnp.ndarray,
    tsin_non: jnp.ndarray,
    tcos_nyq: jnp.ndarray,
    tsin_nyq: jnp.ndarray,
    m_non_f: jnp.ndarray,
    n_non_f: jnp.ndarray,
    m_nyq_f: jnp.ndarray,
    n_nyq_f: jnp.ndarray,
    idx_theta0: jnp.ndarray,
    idx_thetapi: jnp.ndarray,
    m_b: jnp.ndarray,
    abs_n_b: jnp.ndarray,
    sign_b: jnp.ndarray,
    bmns: Optional[jnp.ndarray] = None,
    bsubumns: Optional[jnp.ndarray] = None,
    bsubvmns: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Boozer spectra for a single surface."""
    nfp = constants.nfp
    theta_grid = grids.theta_grid
    zeta_grid = grids.zeta_grid

    # Boozer I/G from m=n=0 Nyquist mode
    idx00 = jnp.where((m_nyq_f == 0) & (n_nyq_f == 0), size=1)[0][0]
    Boozer_I = bsubumnc[idx00]
    Boozer_G = bsubvmnc[idx00]

    # w spectrum from B_theta and B_zeta
    m_nonzero = m_nyq_f != 0.0
    n_nonzero_only = jnp.logical_and(~m_nonzero, n_nyq_f != 0.0)
    wmns = jnp.where(
        m_nonzero,
        bsubumnc / m_nyq_f,
        jnp.where(n_nonzero_only, -bsubvmnc / n_nyq_f, 0.0),
    )
    if constants.asym and bsubumns is not None and bsubvmns is not None:
        wmnc = jnp.where(
            m_nonzero,
            -bsubumns / m_nyq_f,
            jnp.where(n_nonzero_only, bsubvmns / n_nyq_f, 0.0),
        )
    else:
        wmnc = None

    # Non-Nyquist R, Z, lambda and derivatives
    r = jnp.einsum("ij,j->i", tcos_non, rmnc)
    z = jnp.einsum("ij,j->i", tsin_non, zmns)
    lam = jnp.einsum("ij,j->i", tsin_non, lmns)
    dlam_dth = jnp.einsum("ij,j->i", tcos_non, lmns * m_non_f)
    dlam_dze = -jnp.einsum("ij,j->i", tcos_non, lmns * n_non_f)

    # Nyquist w, derivatives, and |B|
    w = jnp.einsum("ij,j->i", tsin_nyq, wmns)
    dw_dth = jnp.einsum("ij,j->i", tcos_nyq, wmns * m_nyq_f)
    dw_dze = -jnp.einsum("ij,j->i", tcos_nyq, wmns * n_nyq_f)
    bmod = jnp.einsum("ij,j->i", tcos_nyq, bmnc)

    if constants.asym and wmnc is not None and bmns is not None:
        w = w + jnp.einsum("ij,j->i", tcos_nyq, wmnc)
        dw_dth = dw_dth - jnp.einsum("ij,j->i", tsin_nyq, wmnc * m_nyq_f)
        dw_dze = dw_dze + jnp.einsum("ij,j->i", tsin_nyq, wmnc * n_nyq_f)
        bmod = bmod + jnp.einsum("ij,j->i", tsin_nyq, bmns)

    # Boozer angles and derivatives
    GI = Boozer_G + iota * Boozer_I
    one_over_GI = 1.0 / GI
    nu = one_over_GI * (w - Boozer_I * lam)
    theta_B = theta_grid + lam + iota * nu
    zeta_B = zeta_grid + nu
    dnu_dze = one_over_GI * (dw_dze - Boozer_I * dlam_dze)
    dnu_dth = one_over_GI * (dw_dth - Boozer_I * dlam_dth)
    dB_dvmec = (1.0 + dlam_dth) * (1.0 + dnu_dze) + (iota - dlam_dze) * dnu_dth

    # Boozer trig tables on (theta_B, zeta_B)
    cosm_b, sinm_b, cosn_b, sinn_b = _init_trig(
        theta_B, zeta_B, constants.mboz, constants.nboz, nfp
    )

    if not constants.asym:
        cosm_b = cosm_b.at[idx_theta0, :].set(cosm_b[idx_theta0, :] * 0.5)
        cosm_b = cosm_b.at[idx_thetapi, :].set(cosm_b[idx_thetapi, :] * 0.5)
        sinm_b = sinm_b.at[idx_theta0, :].set(sinm_b[idx_theta0, :] * 0.5)
        sinm_b = sinm_b.at[idx_thetapi, :].set(sinm_b[idx_thetapi, :] * 0.5)

    boozer_jac = GI / (bmod * bmod)

    cosm_b_m = jnp.take(cosm_b, m_b, axis=1)
    sinm_b_m = jnp.take(sinm_b, m_b, axis=1)
    cosn_b_n = jnp.take(cosn_b, abs_n_b, axis=1)
    sinn_b_n = jnp.take(sinn_b, abs_n_b, axis=1)

    tcos_modes = cosm_b_m * cosn_b_n + sinm_b_m * sinn_b_n * sign_b
    tsin_modes = sinm_b_m * cosn_b_n - cosm_b_m * sinn_b_n * sign_b

    if constants.asym:
        fourier_factor0 = 2.0 / (constants.ntheta * constants.nzeta)
    else:
        fourier_factor0 = 2.0 / ((constants.nu2_b - 1) * constants.nzeta)

    fourier_factor = jnp.ones((m_b.shape[0],), dtype=jnp.float64) * fourier_factor0
    fourier_factor = fourier_factor.at[0].set(fourier_factor0 * 0.5)

    weight = dB_dvmec[:, None] * fourier_factor[None, :]
    tcos_w = tcos_modes * weight
    tsin_w = tsin_modes * weight

    bmnc_b = jnp.einsum("ij,i->j", tcos_w, bmod)
    rmnc_b = jnp.einsum("ij,i->j", tcos_w, r)
    zmns_b = jnp.einsum("ij,i->j", tsin_w, z)
    numns_b = jnp.einsum("ij,i->j", tsin_w, nu)
    gmnc_b = jnp.einsum("ij,i->j", tcos_w, boozer_jac)

    return bmnc_b, rmnc_b, zmns_b, numns_b, gmnc_b, Boozer_I, Boozer_G


def booz_xform_jax_impl(
    rmnc: jnp.ndarray,
    zmns: jnp.ndarray,
    lmns: jnp.ndarray,
    bmnc: jnp.ndarray,
    bsubumnc: jnp.ndarray,
    bsubvmnc: jnp.ndarray,
    iota: jnp.ndarray,
    *,
    xm: jnp.ndarray,
    xn: jnp.ndarray,
    xm_nyq: jnp.ndarray,
    xn_nyq: jnp.ndarray,
    constants: BoozXformConstants,
    grids: BoozXformGrids,
    bmns: Optional[jnp.ndarray] = None,
    bsubumns: Optional[jnp.ndarray] = None,
    bsubvmns: Optional[jnp.ndarray] = None,
    surface_indices: Optional[jnp.ndarray] = None,
) -> dict:
    """JAX-native Boozer transform over all (or selected) surfaces.

    All inputs must be JAX arrays with surface dimension first, i.e. shape
    (ns, mn_non) for non-Nyquist arrays and (ns, mn_nyq) for Nyquist arrays.
    """
    if surface_indices is not None:
        rmnc = jnp.take(rmnc, surface_indices, axis=0)
        zmns = jnp.take(zmns, surface_indices, axis=0)
        lmns = jnp.take(lmns, surface_indices, axis=0)
        bmnc = jnp.take(bmnc, surface_indices, axis=0)
        bsubumnc = jnp.take(bsubumnc, surface_indices, axis=0)
        bsubvmnc = jnp.take(bsubvmnc, surface_indices, axis=0)
        iota = jnp.take(iota, surface_indices, axis=0)
        if bmns is not None:
            bmns = jnp.take(bmns, surface_indices, axis=0)
        if bsubumns is not None:
            bsubumns = jnp.take(bsubumns, surface_indices, axis=0)
        if bsubvmns is not None:
            bsubvmns = jnp.take(bsubvmns, surface_indices, axis=0)

    xm_non_j = jnp.asarray(xm, dtype=jnp.int32)
    xn_non_j = jnp.asarray(xn, dtype=jnp.int32)
    xm_nyq_j = jnp.asarray(xm_nyq, dtype=jnp.int32)
    xn_nyq_j = jnp.asarray(xn_nyq, dtype=jnp.int32)

    # Precompute trig tables and mode combinations once for all surfaces.
    cosm, sinm, cosn, sinn = _init_trig(
        grids.theta_grid, grids.zeta_grid, constants.mmax_non, constants.nmax_non, constants.nfp
    )
    cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = _init_trig(
        grids.theta_grid, grids.zeta_grid, constants.mmax_nyq, constants.nmax_nyq, constants.nfp
    )

    cosm_m_non = jnp.take(cosm, xm_non_j, axis=1)
    sinm_m_non = jnp.take(sinm, xm_non_j, axis=1)
    abs_n_non = jnp.abs(xn_non_j // constants.nfp)
    cosn_n_non = jnp.take(cosn, abs_n_non, axis=1)
    sinn_n_non = jnp.take(sinn, abs_n_non, axis=1)
    sign_non = jnp.where(xn_non_j < 0, -1.0, 1.0)[None, :]
    tcos_non = cosm_m_non * cosn_n_non + sinm_m_non * sinn_n_non * sign_non
    tsin_non = sinm_m_non * cosn_n_non - cosm_m_non * sinn_n_non * sign_non
    m_non_f = xm_non_j.astype(jnp.float64)
    n_non_f = xn_non_j.astype(jnp.float64)

    cosm_m_nyq = jnp.take(cosm_nyq, xm_nyq_j, axis=1)
    sinm_m_nyq = jnp.take(sinm_nyq, xm_nyq_j, axis=1)
    abs_n_nyq = jnp.abs(xn_nyq_j // constants.nfp)
    cosn_n_nyq = jnp.take(cosn_nyq, abs_n_nyq, axis=1)
    sinn_n_nyq = jnp.take(sinn_nyq, abs_n_nyq, axis=1)
    sign_nyq = jnp.where(xn_nyq_j < 0, -1.0, 1.0)[None, :]
    tcos_nyq = cosm_m_nyq * cosn_n_nyq + sinm_m_nyq * sinn_n_nyq * sign_nyq
    tsin_nyq = sinm_m_nyq * cosn_n_nyq - cosm_m_nyq * sinn_n_nyq * sign_nyq
    m_nyq_f = xm_nyq_j.astype(jnp.float64)
    n_nyq_f = xn_nyq_j.astype(jnp.float64)

    idx_theta0 = jnp.arange(0, constants.nzeta)
    idx_thetapi = jnp.arange(
        (constants.nu2_b - 1) * constants.nzeta, constants.nu2_b * constants.nzeta
    )

    m_b = grids.xm_b
    abs_n_b = jnp.abs(grids.xn_b // constants.nfp)
    sign_b = jnp.where(grids.xn_b < 0, -1.0, 1.0)[None, :]

    def _surf(
        _rmnc, _zmns, _lmns, _bmnc, _bsubumnc, _bsubvmnc, _iota, _bmns, _bsubumns, _bsubvmns
    ):
        return _surface_transform(
            _rmnc,
            _zmns,
            _lmns,
            _bmnc,
            _bsubumnc,
            _bsubvmnc,
            _iota,
            constants=constants,
            grids=grids,
            tcos_non=tcos_non,
            tsin_non=tsin_non,
            tcos_nyq=tcos_nyq,
            tsin_nyq=tsin_nyq,
            m_non_f=m_non_f,
            n_non_f=n_non_f,
            m_nyq_f=m_nyq_f,
            n_nyq_f=n_nyq_f,
            idx_theta0=idx_theta0,
            idx_thetapi=idx_thetapi,
            m_b=m_b,
            abs_n_b=abs_n_b,
            sign_b=sign_b,
            bmns=_bmns,
            bsubumns=_bsubumns,
            bsubvmns=_bsubvmns,
        )

    vmap_fn = jax.vmap(_surf)

    bmns_in = bmns if bmns is not None else jnp.zeros_like(bmnc)
    bsubumns_in = bsubumns if bsubumns is not None else jnp.zeros_like(bsubumnc)
    bsubvmns_in = bsubvmns if bsubvmns is not None else jnp.zeros_like(bsubvmnc)

    bmnc_b, rmnc_b, zmns_b, numns_b, gmnc_b, Boozer_I, Boozer_G = vmap_fn(
        rmnc,
        zmns,
        lmns,
        bmnc,
        bsubumnc,
        bsubvmnc,
        iota,
        bmns_in,
        bsubumns_in,
        bsubvmns_in,
    )

    ns_b = bmnc_b.shape[0]
    jlist = jnp.arange(1, ns_b + 1)

    return {
        "nfp_b": jnp.asarray(constants.nfp),
        "ixm_b": jnp.asarray(grids.xm_b),
        "ixn_b": jnp.asarray(grids.xn_b),
        "iota_b": iota,
        "buco_b": Boozer_I,
        "bvco_b": Boozer_G,
        "rmnc_b": rmnc_b,
        "zmns_b": zmns_b,
        "pmns_b": -numns_b,
        "bmnc_b": bmnc_b,
        "jlist": jlist,
    }


def booz_xform_jax(
    *,
    rmnc: jnp.ndarray,
    zmns: jnp.ndarray,
    lmns: jnp.ndarray,
    bmnc: jnp.ndarray,
    bsubumnc: jnp.ndarray,
    bsubvmnc: jnp.ndarray,
    iota: jnp.ndarray,
    xm: Sequence[int],
    xn: Sequence[int],
    xm_nyq: Sequence[int],
    xn_nyq: Sequence[int],
    nfp: int,
    mboz: int,
    nboz: int,
    asym: bool = False,
    bmns: Optional[jnp.ndarray] = None,
    bsubumns: Optional[jnp.ndarray] = None,
    bsubvmns: Optional[jnp.ndarray] = None,
    surface_indices: Optional[Sequence[int]] = None,
) -> dict:
    """Host-side convenience wrapper for :func:`booz_xform_jax_impl`.

    This wrapper computes static constants on the host (NumPy) and
    returns a JAX output dictionary. For full JIT, call
    :func:`booz_xform_jax_impl` directly with precomputed constants.
    """
    constants, grids = prepare_booz_xform_constants(
        nfp=nfp,
        mboz=mboz,
        nboz=nboz,
        asym=asym,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
    )

    surf_idx = None
    if surface_indices is not None:
        surf_idx = jnp.asarray(surface_indices, dtype=jnp.int32)

    return booz_xform_jax_impl(
        rmnc=jnp.asarray(rmnc),
        zmns=jnp.asarray(zmns),
        lmns=jnp.asarray(lmns),
        bmnc=jnp.asarray(bmnc),
        bsubumnc=jnp.asarray(bsubumnc),
        bsubvmnc=jnp.asarray(bsubvmnc),
        iota=jnp.asarray(iota),
        xm=jnp.asarray(xm, dtype=jnp.int32),
        xn=jnp.asarray(xn, dtype=jnp.int32),
        xm_nyq=jnp.asarray(xm_nyq, dtype=jnp.int32),
        xn_nyq=jnp.asarray(xn_nyq, dtype=jnp.int32),
        constants=constants,
        grids=grids,
        bmns=jnp.asarray(bmns) if bmns is not None else None,
        bsubumns=jnp.asarray(bsubumns) if bsubumns is not None else None,
        bsubvmns=jnp.asarray(bsubvmns) if bsubvmns is not None else None,
        surface_indices=surf_idx,
    )
