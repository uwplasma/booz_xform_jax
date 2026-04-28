"""Jacobian-harmonic sensitivity demo using the JAX-native transform.

This example shows three matrix-free autodiff access patterns that downstream
codes can use without forming a dense geometry Jacobian:

- scalar objective gradients with ``jax.value_and_grad``,
- Jacobian-vector products with ``jax.jvp``,
- vector-Jacobian products with ``jax.vjp``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from booz_xform_jax import Booz_xform
from booz_xform_jax.jax_api import booz_xform_jax_impl, prepare_booz_xform_constants


def _default_wout_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "tests" / "test_files" / "wout_li383_1.4m.nc"


def main() -> None:
    bx = Booz_xform()
    bx.read_wout(str(_default_wout_path()))

    bx.mboz = 6
    bx.nboz = 6
    bx.compute_surfs = [0]

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

    rmnc = jnp.asarray(bx.rmnc).T
    zmns = jnp.asarray(bx.zmns).T
    lmns = jnp.asarray(bx.lmns).T
    bmnc = jnp.asarray(bx.bmnc).T
    bsubumnc = jnp.asarray(bx.bsubumnc).T
    bsubvmnc = jnp.asarray(bx.bsubvmnc).T
    iota = jnp.asarray(bx.iota)
    xm = jnp.asarray(bx.xm)
    xn = jnp.asarray(bx.xn)
    xm_nyq = jnp.asarray(bx.xm_nyq)
    xn_nyq = jnp.asarray(bx.xn_nyq)
    surfaces = jnp.asarray(bx.compute_surfs)

    booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))

    def run_with_bmnc(bmnc_in: jnp.ndarray) -> dict:
        return booz_fn(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc_in,
            bsubumnc=bsubumnc,
            bsubvmnc=bsubvmnc,
            iota=iota,
            xm=xm,
            xn=xn,
            xm_nyq=xm_nyq,
            xn_nyq=xn_nyq,
            constants=constants,
            grids=grids,
            surface_indices=surfaces,
        )

    def jacobian_energy(bmnc_in: jnp.ndarray) -> jnp.ndarray:
        gmnc_b = run_with_bmnc(bmnc_in)["gmnc_b"][0]
        return jnp.sum(gmnc_b[1:] ** 2)

    value, grad_bmnc = jax.value_and_grad(jacobian_energy)(bmnc)

    def gmnc_flat(bmnc_in: jnp.ndarray) -> jnp.ndarray:
        return run_with_bmnc(bmnc_in)["gmnc_b"].ravel()

    tangent = jnp.zeros_like(bmnc).at[:, 1].set(1.0e-3)
    gmnc_value, gmnc_jvp = jax.jvp(gmnc_flat, (bmnc,), (tangent,))

    _, pullback = jax.vjp(gmnc_flat, bmnc)
    mode_seed = jnp.zeros_like(gmnc_value).at[1].set(1.0)
    (adjoint_bmnc,) = pullback(mode_seed)

    print("gmnc_b shape:", run_with_bmnc(bmnc)["gmnc_b"].shape)
    print("Jacobian-harmonic energy:", float(value))
    print("Gradient norm wrt VMEC bmnc:", float(jnp.linalg.norm(grad_bmnc)))
    print("JVP norm for a small bmnc perturbation:", float(jnp.linalg.norm(gmnc_jvp)))
    print("VJP norm for one selected gmnc_b harmonic:", float(jnp.linalg.norm(adjoint_bmnc)))


if __name__ == "__main__":
    main()
