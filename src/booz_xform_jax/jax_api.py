"""Pure JAX API for end-to-end Boozer transforms.

This module provides a JIT-friendly, functional interface that avoids
Python loops over surfaces and keeps all arrays in JAX. It is intended
for end-to-end differentiation with vmec_jax -> booz_xform_jax -> neo_jax.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple
import os

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


_LOGGER = logging.getLogger("booz_xform_jax")

_FOURIER_MODES = ("vectorized", "streamed")


@dataclass(frozen=True)
class BoozerConfig:
    """Explicit execution configuration for the JAX Boozer transform.

    ``surface_chunk`` bounds how many surfaces one compiled contraction
    batches: the dominant phase tensors scale linearly in the batch and the
    committed baseline shows superlinear wall-time growth once they spill
    the cache hierarchy. ``"auto"`` batches everything unless
    ``memory_budget_bytes`` is set, in which case the chunk is the largest
    count whose modeled buffers fit the budget. ``fourier_mode`` selects the
    vectorized or streamed contraction; ``trig_f32`` stores the trig tables
    in single precision. The environment variables the kernel used to read
    (``BOOZ_XFORM_JAX_FOURIER_MODE``, ``BOOZ_XFORM_JAX_TRIG_F32``) supply
    defaults through :meth:`from_env` only when no config is passed.
    """

    mboz: int
    nboz: int
    surface_chunk: int | str = "auto"
    memory_budget_bytes: int | None = None
    fourier_mode: str = "vectorized"
    trig_f32: bool = False

    def __post_init__(self):
        if self.fourier_mode not in _FOURIER_MODES:
            raise ValueError(
                f"fourier_mode must be one of {_FOURIER_MODES}, "
                f"got {self.fourier_mode!r}")
        if isinstance(self.surface_chunk, str):
            if self.surface_chunk != "auto":
                raise ValueError(
                    f"surface_chunk must be a positive int or 'auto', "
                    f"got {self.surface_chunk!r}")
        elif int(self.surface_chunk) < 1:
            raise ValueError("surface_chunk must be >= 1")
        if (self.memory_budget_bytes is not None
                and int(self.memory_budget_bytes) < 1):
            raise ValueError("memory_budget_bytes must be positive")

    @classmethod
    def from_env(cls, *, mboz: int, nboz: int) -> "BoozerConfig":
        """Defaults from the legacy environment variables (CLI compatibility)."""
        fourier_mode = os.getenv(
            "BOOZ_XFORM_JAX_FOURIER_MODE", "vectorized").strip().lower()
        trig_f32 = os.getenv(
            "BOOZ_XFORM_JAX_TRIG_F32", "0").strip().lower() in {
                "1", "true", "yes", "on"}
        return cls(mboz=int(mboz), nboz=int(nboz),
                   fourier_mode=fourier_mode, trig_f32=trig_f32)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoozerPlan:
    """Reusable static plan: mode tables, angle grids, and trig tables.

    Built once per (config, mode lists, ``nfp``, ``asym``) and reused across
    surfaces, equilibrium iterates, and optimization steps — the tables are
    pure functions of the static resolution, and hoisting them out of the
    per-call trace keeps them out of every compiled program. Physical
    coefficients never live in the plan.
    """

    config: BoozerConfig
    constants: BoozXformConstants
    grids: BoozXformGrids
    tables: dict

    def tree_flatten(self):
        names = tuple(sorted(self.tables))
        children = (self.grids,) + tuple(self.tables[k] for k in names)
        return children, (self.config, self.constants, names)

    @classmethod
    def tree_unflatten(cls, aux, children):
        config, constants, names = aux
        grids = children[0]
        tables = dict(zip(names, children[1:]))
        return cls(config=config, constants=constants, grids=grids,
                   tables=tables)


def _build_tables(constants: BoozXformConstants, grids: BoozXformGrids,
                  xm, xn, xm_nyq, xn_nyq, *, trig_f32: bool) -> dict:
    """The per-resolution trig/mode tables the surface transform contracts."""
    xm_non_j = jnp.asarray(xm, dtype=jnp.int32)
    xn_non_j = jnp.asarray(xn, dtype=jnp.int32)
    xm_nyq_j = jnp.asarray(xm_nyq, dtype=jnp.int32)
    xn_nyq_j = jnp.asarray(xn_nyq, dtype=jnp.int32)

    cosm, sinm, cosn, sinn = _init_trig(
        grids.theta_grid, grids.zeta_grid, constants.mmax_non,
        constants.nmax_non, constants.nfp)
    cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = _init_trig(
        grids.theta_grid, grids.zeta_grid, constants.mmax_nyq,
        constants.nmax_nyq, constants.nfp)
    if trig_f32:
        cosm, sinm, cosn, sinn = (
            t.astype(jnp.float32) for t in (cosm, sinm, cosn, sinn))
        cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq = (
            t.astype(jnp.float32)
            for t in (cosm_nyq, sinm_nyq, cosn_nyq, sinn_nyq))

    cosm_m_non = jnp.take(cosm, xm_non_j, axis=1)
    sinm_m_non = jnp.take(sinm, xm_non_j, axis=1)
    abs_n_non = jnp.abs(xn_non_j // constants.nfp)
    cosn_n_non = jnp.take(cosn, abs_n_non, axis=1)
    sinn_n_non = jnp.take(sinn, abs_n_non, axis=1)
    sign_non = jnp.where(xn_non_j < 0, -1.0, 1.0)[None, :]

    cosm_m_nyq = jnp.take(cosm_nyq, xm_nyq_j, axis=1)
    sinm_m_nyq = jnp.take(sinm_nyq, xm_nyq_j, axis=1)
    abs_n_nyq = jnp.abs(xn_nyq_j // constants.nfp)
    cosn_n_nyq = jnp.take(cosn_nyq, abs_n_nyq, axis=1)
    sinn_n_nyq = jnp.take(sinn_nyq, abs_n_nyq, axis=1)
    sign_nyq = jnp.where(xn_nyq_j < 0, -1.0, 1.0)[None, :]

    return {
        "tcos_non": cosm_m_non * cosn_n_non + sinm_m_non * sinn_n_non * sign_non,
        "tsin_non": sinm_m_non * cosn_n_non - cosm_m_non * sinn_n_non * sign_non,
        "tcos_nyq": cosm_m_nyq * cosn_n_nyq + sinm_m_nyq * sinn_n_nyq * sign_nyq,
        "tsin_nyq": sinm_m_nyq * cosn_n_nyq - cosm_m_nyq * sinn_n_nyq * sign_nyq,
        "m_non_f": xm_non_j.astype(jnp.float64),
        "n_non_f": xn_non_j.astype(jnp.float64),
        "m_nyq_f": xm_nyq_j.astype(jnp.float64),
        "n_nyq_f": xn_nyq_j.astype(jnp.float64),
        "idx_theta0": jnp.arange(0, constants.nzeta),
        "idx_thetapi": jnp.arange(
            (constants.nu2_b - 1) * constants.nzeta,
            constants.nu2_b * constants.nzeta),
        "m_b": grids.xm_b,
        "abs_n_b": jnp.abs(grids.xn_b // constants.nfp),
        "sign_b": jnp.where(grids.xn_b < 0, -1.0, 1.0)[None, :],
    }


#: Bounded plan cache. Plans hold only per-resolution tables (a few MB at
#: high mboz); eviction only costs a rebuild, never correctness.
_PLAN_CACHE: "collections.OrderedDict[Any, BoozerPlan]" = collections.OrderedDict()
_PLAN_CACHE_MAX = 8


def prepare_booz_xform_plan(
    *,
    nfp: int,
    asym: bool,
    xm: Sequence[int],
    xn: Sequence[int],
    xm_nyq: Sequence[int],
    xn_nyq: Sequence[int],
    config: BoozerConfig,
) -> BoozerPlan:
    """Build (or fetch from a bounded cache) the reusable transform plan."""
    import numpy as np

    key = (
        config,
        int(nfp),
        bool(asym),
        np.asarray(xm, dtype=np.int32).tobytes(),
        np.asarray(xn, dtype=np.int32).tobytes(),
        np.asarray(xm_nyq, dtype=np.int32).tobytes(),
        np.asarray(xn_nyq, dtype=np.int32).tobytes(),
    )
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        _PLAN_CACHE.move_to_end(key)
        return hit
    constants, grids = prepare_booz_xform_constants(
        nfp=int(nfp), mboz=config.mboz, nboz=config.nboz, asym=bool(asym),
        xm=xm, xn=xn, xm_nyq=xm_nyq, xn_nyq=xn_nyq)
    tables = _build_tables(constants, grids, xm, xn, xm_nyq, xn_nyq,
                           trig_f32=config.trig_f32)
    plan = BoozerPlan(config=config, constants=constants, grids=grids,
                      tables=tables)
    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > _PLAN_CACHE_MAX:
        _PLAN_CACHE.popitem(last=False)
    return plan


def _resolve_surface_chunk(config: BoozerConfig,
                           constants: BoozXformConstants,
                           n_surfaces: int, mn_non: int, mn_nyq: int,
                           mn_boz: int) -> int:
    """Deterministic chunk choice; logs what it picked and why.

    The buffer model counts the dominant per-surface tensors — the input
    phase projections (``points x mn_non`` and ``points x mn_nyq``) and the
    output synthesis (``points x mn_boz``) — with a factor-four headroom for
    the intermediates the contraction materializes alongside them. Host RSS
    is never consulted (it says nothing about accelerator capacity).
    """
    if isinstance(config.surface_chunk, int):
        chunk = min(int(config.surface_chunk), n_surfaces)
    elif config.memory_budget_bytes is None:
        chunk = n_surfaces
    else:
        points = constants.ntheta * constants.nzeta
        per_surface = 4 * 8 * points * (mn_non + mn_nyq + mn_boz)
        chunk = max(1, min(n_surfaces,
                           int(config.memory_budget_bytes) // per_surface))
    _LOGGER.info(
        "surface_chunk=%d for %d surfaces (policy=%r, budget=%r)",
        chunk, n_surfaces, config.surface_chunk, config.memory_budget_bytes)
    return chunk


def _prepare_mode_lists(mboz: int, nboz: int, nfp: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    return jnp.asarray(m_list, dtype=jnp.int32), jnp.asarray(n_list, dtype=jnp.int32)


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
    xm_arr = jnp.asarray(xm, dtype=jnp.int32)
    xn_arr = jnp.asarray(xn, dtype=jnp.int32)
    xm_nyq_arr = jnp.asarray(xm_nyq, dtype=jnp.int32)
    xn_nyq_arr = jnp.asarray(xn_nyq, dtype=jnp.int32)

    mmax_non = int(jnp.max(jnp.abs(xm_arr)))
    nmax_non = int(jnp.max(jnp.abs(xn_arr // nfp)))
    mmax_nyq = int(jnp.max(jnp.abs(xm_nyq_arr)))
    nmax_nyq = int(jnp.max(jnp.abs(xn_nyq_arr // nfp)))

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


def prepare_booz_xform_constants_from_inputs(
    *,
    inputs,
    mboz: int,
    nboz: int,
    asym: bool,
) -> tuple[BoozXformConstants, BoozXformGrids]:
    """Convenience wrapper using a VMEC -> Boozer input bundle."""
    return prepare_booz_xform_constants(
        nfp=int(jnp.asarray(inputs.nfp)),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(asym),
        xm=jnp.asarray(inputs.xm),
        xn=jnp.asarray(inputs.xn),
        xm_nyq=jnp.asarray(inputs.xm_nyq),
        xn_nyq=jnp.asarray(inputs.xn_nyq),
    )


def _surface_transform(
    rmnc: jnp.ndarray,
    rmns: jnp.ndarray,
    zmnc: jnp.ndarray,
    zmns: jnp.ndarray,
    lmnc: jnp.ndarray,
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
    fourier_mode: str = "vectorized",
    trig_f32: bool = False,
) -> Tuple[jnp.ndarray, ...]:
    """Compute Boozer spectra for a single surface."""
    nfp = constants.nfp
    theta_grid = grids.theta_grid
    zeta_grid = grids.zeta_grid

    # Boozer I/G from m=n=0 Nyquist mode
    idx00 = jnp.where((m_nyq_f == 0) & (n_nyq_f == 0), size=1)[0][0]
    Boozer_I = bsubumnc[idx00]
    Boozer_G = bsubvmnc[idx00]

    # w spectrum from B_theta and B_zeta. Safe denominators avoid inf/NaN
    # tangents at m=n=0 when this kernel is differentiated.
    m_nonzero = m_nyq_f != 0.0
    n_nonzero_only = jnp.logical_and(~m_nonzero, n_nyq_f != 0.0)
    m_nyq_safe = jnp.where(m_nonzero, m_nyq_f, 1.0)
    n_nyq_safe = jnp.where(n_nonzero_only, n_nyq_f, 1.0)
    wmns = jnp.where(
        m_nonzero,
        bsubumnc / m_nyq_safe,
        jnp.where(n_nonzero_only, -bsubvmnc / n_nyq_safe, 0.0),
    )
    if constants.asym and bsubumns is not None and bsubvmns is not None:
        wmnc = jnp.where(
            m_nonzero,
            -bsubumns / m_nyq_safe,
            jnp.where(n_nonzero_only, bsubvmns / n_nyq_safe, 0.0),
        )
    else:
        wmnc = None

    # Non-Nyquist R, Z, lambda and derivatives
    r = jnp.einsum("ij,j->i", tcos_non, rmnc)
    z = jnp.einsum("ij,j->i", tsin_non, zmns)
    lam = jnp.einsum("ij,j->i", tsin_non, lmns)
    dlam_dth = jnp.einsum("ij,j->i", tcos_non, lmns * m_non_f)
    dlam_dze = -jnp.einsum("ij,j->i", tcos_non, lmns * n_non_f)

    if constants.asym:
        r = r + jnp.einsum("ij,j->i", tsin_non, rmns)
        z = z + jnp.einsum("ij,j->i", tcos_non, zmnc)
        lam = lam + jnp.einsum("ij,j->i", tcos_non, lmnc)
        dlam_dth = dlam_dth - jnp.einsum("ij,j->i", tsin_non, lmnc * m_non_f)
        dlam_dze = dlam_dze + jnp.einsum("ij,j->i", tsin_non, lmnc * n_non_f)

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
    if trig_f32:
        cosm_b = cosm_b.astype(jnp.float32)
        sinm_b = sinm_b.astype(jnp.float32)
        cosn_b = cosn_b.astype(jnp.float32)
        sinn_b = sinn_b.astype(jnp.float32)

    if not constants.asym:
        cosm_b = cosm_b.at[idx_theta0, :].set(cosm_b[idx_theta0, :] * 0.5)
        cosm_b = cosm_b.at[idx_thetapi, :].set(cosm_b[idx_thetapi, :] * 0.5)
        sinm_b = sinm_b.at[idx_theta0, :].set(sinm_b[idx_theta0, :] * 0.5)
        sinm_b = sinm_b.at[idx_thetapi, :].set(sinm_b[idx_thetapi, :] * 0.5)

    boozer_jac = GI / (bmod * bmod)

    if constants.asym:
        fourier_factor0 = 2.0 / (constants.ntheta * constants.nzeta)
    else:
        fourier_factor0 = 2.0 / ((constants.nu2_b - 1) * constants.nzeta)

    fourier_factor = jnp.ones((m_b.shape[0],), dtype=jnp.float64) * fourier_factor0
    fourier_factor = fourier_factor.at[0].set(fourier_factor0 * 0.5)

    if fourier_mode == "streamed":
        base_b = bmod * dB_dvmec
        base_r = r * dB_dvmec
        base_z = z * dB_dvmec
        base_nu = nu * dB_dvmec
        base_g = boozer_jac * dB_dvmec

        m_b_f = m_b
        abs_n_b_f = abs_n_b
        sign_b_f = jnp.reshape(sign_b, (-1,))

        def init_out():
            zeros = jnp.zeros((m_b_f.shape[0],), dtype=base_b.dtype)
            return zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros

        def body(k, state):
            (
                bmnc_b,
                bmns_b,
                rmnc_b,
                rmns_b,
                zmnc_b,
                zmns_b,
                numnc_b,
                numns_b,
                gmnc_b,
                gmns_b,
            ) = state
            m_idx = m_b_f[k]
            n_idx = abs_n_b_f[k]
            sign = sign_b_f[k]

            cosm = jax.lax.dynamic_index_in_dim(cosm_b, m_idx, axis=1, keepdims=False)
            sinm = jax.lax.dynamic_index_in_dim(sinm_b, m_idx, axis=1, keepdims=False)
            cosn = jax.lax.dynamic_index_in_dim(cosn_b, n_idx, axis=1, keepdims=False)
            sinn = jax.lax.dynamic_index_in_dim(sinn_b, n_idx, axis=1, keepdims=False)

            tcos = cosm * cosn + sinm * sinn * sign
            tsin = sinm * cosn - cosm * sinn * sign
            ff = fourier_factor[k]

            bmnc_b = bmnc_b.at[k].set(ff * jnp.sum(tcos * base_b))
            rmnc_b = rmnc_b.at[k].set(ff * jnp.sum(tcos * base_r))
            zmns_b = zmns_b.at[k].set(ff * jnp.sum(tsin * base_z))
            numns_b = numns_b.at[k].set(ff * jnp.sum(tsin * base_nu))
            gmnc_b = gmnc_b.at[k].set(ff * jnp.sum(tcos * base_g))
            if constants.asym:
                bmns_b = bmns_b.at[k].set(ff * jnp.sum(tsin * base_b))
                rmns_b = rmns_b.at[k].set(ff * jnp.sum(tsin * base_r))
                zmnc_b = zmnc_b.at[k].set(ff * jnp.sum(tcos * base_z))
                numnc_b = numnc_b.at[k].set(ff * jnp.sum(tcos * base_nu))
                gmns_b = gmns_b.at[k].set(ff * jnp.sum(tsin * base_g))
            return (
                bmnc_b,
                bmns_b,
                rmnc_b,
                rmns_b,
                zmnc_b,
                zmns_b,
                numnc_b,
                numns_b,
                gmnc_b,
                gmns_b,
            )

        (
            bmnc_b,
            bmns_b,
            rmnc_b,
            rmns_b,
            zmnc_b,
            zmns_b,
            numnc_b,
            numns_b,
            gmnc_b,
            gmns_b,
        ) = jax.lax.fori_loop(
            0, m_b_f.shape[0], body, init_out()
        )
    else:
        cosm_b_m = jnp.take(cosm_b, m_b, axis=1)
        sinm_b_m = jnp.take(sinm_b, m_b, axis=1)
        cosn_b_n = jnp.take(cosn_b, abs_n_b, axis=1)
        sinn_b_n = jnp.take(sinn_b, abs_n_b, axis=1)

        tcos_modes = cosm_b_m * cosn_b_n + sinm_b_m * sinn_b_n * sign_b
        tsin_modes = sinm_b_m * cosn_b_n - cosm_b_m * sinn_b_n * sign_b

        base_b = bmod * dB_dvmec
        base_r = r * dB_dvmec
        base_z = z * dB_dvmec
        base_nu = nu * dB_dvmec
        base_g = boozer_jac * dB_dvmec

        def project_cos(field: jnp.ndarray) -> jnp.ndarray:
            return fourier_factor * jnp.einsum("ij,i->j", tcos_modes, field)

        def project_sin(field: jnp.ndarray) -> jnp.ndarray:
            return fourier_factor * jnp.einsum("ij,i->j", tsin_modes, field)

        bmnc_b = project_cos(base_b)
        rmnc_b = project_cos(base_r)
        zmns_b = project_sin(base_z)
        numns_b = project_sin(base_nu)
        gmnc_b = project_cos(base_g)
        zeros = jnp.zeros_like(bmnc_b)
        if constants.asym:
            bmns_b = project_sin(base_b)
            rmns_b = project_sin(base_r)
            zmnc_b = project_cos(base_z)
            numnc_b = project_cos(base_nu)
            gmns_b = project_sin(base_g)
        else:
            bmns_b = zeros
            rmns_b = zeros
            zmnc_b = zeros
            numnc_b = zeros
            gmns_b = zeros

    return (
        bmnc_b,
        bmns_b,
        rmnc_b,
        rmns_b,
        zmnc_b,
        zmns_b,
        numnc_b,
        numns_b,
        gmnc_b,
        gmns_b,
        Boozer_I,
        Boozer_G,
    )


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
    rmns: Optional[jnp.ndarray] = None,
    zmnc: Optional[jnp.ndarray] = None,
    lmnc: Optional[jnp.ndarray] = None,
    bmns: Optional[jnp.ndarray] = None,
    bsubumns: Optional[jnp.ndarray] = None,
    bsubvmns: Optional[jnp.ndarray] = None,
    surface_indices: Optional[jnp.ndarray] = None,
    config: Optional[BoozerConfig] = None,
    plan: Optional[BoozerPlan] = None,
) -> dict:
    """JAX-native Boozer transform over all (or selected) surfaces.

    All inputs must be JAX arrays with surface dimension first, i.e. shape
    (ns, mn_non) for non-Nyquist arrays and (ns, mn_nyq) for Nyquist arrays.

    ``plan`` (from :func:`prepare_booz_xform_plan`) supplies the prebuilt
    per-resolution tables and its own config; ``config`` alone controls
    execution with tables built inline. With neither, the legacy environment
    variables act as the config defaults.
    """
    ns_b_full = int(rmnc.shape[0])
    if surface_indices is not None:
        rmnc = jnp.take(rmnc, surface_indices, axis=0)
        zmns = jnp.take(zmns, surface_indices, axis=0)
        lmns = jnp.take(lmns, surface_indices, axis=0)
        bmnc = jnp.take(bmnc, surface_indices, axis=0)
        bsubumnc = jnp.take(bsubumnc, surface_indices, axis=0)
        bsubvmnc = jnp.take(bsubvmnc, surface_indices, axis=0)
        iota = jnp.take(iota, surface_indices, axis=0)
        if rmns is not None:
            rmns = jnp.take(rmns, surface_indices, axis=0)
        if zmnc is not None:
            zmnc = jnp.take(zmnc, surface_indices, axis=0)
        if lmnc is not None:
            lmnc = jnp.take(lmnc, surface_indices, axis=0)
        if bmns is not None:
            bmns = jnp.take(bmns, surface_indices, axis=0)
        if bsubumns is not None:
            bsubumns = jnp.take(bsubumns, surface_indices, axis=0)
        if bsubvmns is not None:
            bsubvmns = jnp.take(bsubvmns, surface_indices, axis=0)

    if plan is not None:
        cfg = plan.config
        constants, grids = plan.constants, plan.grids
        tables = plan.tables
    else:
        cfg = config if config is not None else BoozerConfig.from_env(
            mboz=constants.mboz, nboz=constants.nboz)
        tables = _build_tables(constants, grids, xm, xn, xm_nyq, xn_nyq,
                               trig_f32=cfg.trig_f32)
    fourier_mode, trig_f32 = cfg.fourier_mode, cfg.trig_f32
    tcos_non, tsin_non = tables["tcos_non"], tables["tsin_non"]
    tcos_nyq, tsin_nyq = tables["tcos_nyq"], tables["tsin_nyq"]
    m_non_f, n_non_f = tables["m_non_f"], tables["n_non_f"]
    m_nyq_f, n_nyq_f = tables["m_nyq_f"], tables["n_nyq_f"]
    idx_theta0, idx_thetapi = tables["idx_theta0"], tables["idx_thetapi"]
    m_b, abs_n_b, sign_b = tables["m_b"], tables["abs_n_b"], tables["sign_b"]

    def _surf(
        _rmnc,
        _rmns,
        _zmnc,
        _zmns,
        _lmnc,
        _lmns,
        _bmnc,
        _bsubumnc,
        _bsubvmnc,
        _iota,
        _bmns,
        _bsubumns,
        _bsubvmns,
    ):
        return _surface_transform(
            _rmnc,
            _rmns,
            _zmnc,
            _zmns,
            _lmnc,
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
            fourier_mode=fourier_mode,
            trig_f32=trig_f32,
        )

    vmap_fn = jax.vmap(_surf)

    rmns_in = rmns if rmns is not None else jnp.zeros_like(rmnc)
    zmnc_in = zmnc if zmnc is not None else jnp.zeros_like(zmns)
    lmnc_in = lmnc if lmnc is not None else jnp.zeros_like(lmns)
    bmns_in = bmns if bmns is not None else jnp.zeros_like(bmnc)
    bsubumns_in = bsubumns if bsubumns is not None else jnp.zeros_like(bsubumnc)
    bsubvmns_in = bsubvmns if bsubvmns is not None else jnp.zeros_like(bsubvmnc)

    inputs = (
        rmnc, rmns_in, zmnc_in, zmns, lmnc_in, lmns, bmnc,
        bsubumnc, bsubvmnc, iota, bmns_in, bsubumns_in, bsubvmns_in,
    )
    n_surf = int(rmnc.shape[0])
    chunk = _resolve_surface_chunk(
        cfg, constants, n_surf, mn_non=int(m_non_f.shape[0]),
        mn_nyq=int(m_nyq_f.shape[0]), mn_boz=int(m_b.shape[0]))
    if chunk >= n_surf:
        outputs = vmap_fn(*inputs)
    else:
        # lax.map over fixed-size surface blocks: one compiled body, buffers
        # bounded by the chunk instead of the full selection. The tail pad
        # replicates the last surface (well-conditioned, unlike zeros, whose
        # 1/B would poison reverse-mode through the padded rows) and is
        # sliced away, value and cotangent alike, by the crop below.
        n_chunks = -(-n_surf // chunk)
        pad = n_chunks * chunk - n_surf

        def _blocked(a):
            padded = jnp.pad(
                a, ((0, pad),) + ((0, 0),) * (a.ndim - 1), mode="edge")
            return padded.reshape((n_chunks, chunk) + a.shape[1:])

        stacked = jax.lax.map(
            lambda block: vmap_fn(*block), tuple(_blocked(a) for a in inputs))
        outputs = tuple(
            o.reshape((n_chunks * chunk,) + o.shape[2:])[:n_surf]
            for o in stacked)
    (
        bmnc_b,
        bmns_b,
        rmnc_b,
        rmns_b,
        zmnc_b,
        zmns_b,
        numnc_b,
        numns_b,
        gmnc_b,
        gmns_b,
        Boozer_I,
        Boozer_G,
    ) = outputs

    ns_b = bmnc_b.shape[0]
    if surface_indices is None:
        jlist = jnp.arange(2, ns_b + 2)
    else:
        jlist = surface_indices + 2

    return {
        "nfp_b": jnp.asarray(constants.nfp),
        "ns_b": jnp.asarray(ns_b_full),
        "ixm_b": jnp.asarray(grids.xm_b),
        "ixn_b": jnp.asarray(grids.xn_b),
        "iota_b": iota,
        "buco_b": Boozer_I,
        "bvco_b": Boozer_G,
        "rmnc_b": rmnc_b,
        "rmns_b": rmns_b,
        "zmnc_b": zmnc_b,
        "zmns_b": zmns_b,
        "numnc_b": numnc_b,
        "numns_b": numns_b,
        "pmnc_b": -numnc_b,
        "pmns_b": -numns_b,
        "bmnc_b": bmnc_b,
        "bmns_b": bmns_b,
        "gmnc_b": gmnc_b,
        "gmns_b": gmns_b,
        # BOOZ_XFORM/netCDF-compatible spelling for the Jacobian harmonics.
        "gmn_b": gmnc_b,
        "jlist": jlist,
    }


def booz_xform_from_inputs(
    *,
    inputs,
    constants: BoozXformConstants,
    grids: BoozXformGrids,
    surface_indices: Optional[jnp.ndarray] = None,
    jit: bool = True,
) -> dict:
    """Run the JAX Boozer transform using a VMEC -> Boozer input bundle."""
    booz_fn = booz_xform_jax_impl
    if jit:
        booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))
    return booz_fn(
        rmnc=inputs.rmnc,
        zmns=inputs.zmns,
        lmns=inputs.lmns,
        bmnc=inputs.bmnc,
        bsubumnc=inputs.bsubumnc,
        bsubvmnc=inputs.bsubvmnc,
        iota=inputs.iota,
        xm=inputs.xm,
        xn=inputs.xn,
        xm_nyq=inputs.xm_nyq,
        xn_nyq=inputs.xn_nyq,
        constants=constants,
        grids=grids,
        rmns=getattr(inputs, "rmns", None),
        zmnc=getattr(inputs, "zmnc", None),
        lmnc=getattr(inputs, "lmnc", None),
        bmns=getattr(inputs, "bmns", None),
        bsubumns=getattr(inputs, "bsubumns", None),
        bsubvmns=getattr(inputs, "bsubvmns", None),
        surface_indices=surface_indices,
    )


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
    rmns: Optional[jnp.ndarray] = None,
    zmnc: Optional[jnp.ndarray] = None,
    lmnc: Optional[jnp.ndarray] = None,
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
        rmns=jnp.asarray(rmns) if rmns is not None else None,
        zmnc=jnp.asarray(zmnc) if zmnc is not None else None,
        lmnc=jnp.asarray(lmnc) if lmnc is not None else None,
        bmns=jnp.asarray(bmns) if bmns is not None else None,
        bsubumns=jnp.asarray(bsubumns) if bsubumns is not None else None,
        bsubvmns=jnp.asarray(bsubvmns) if bsubvmns is not None else None,
        surface_indices=surf_idx,
    )
