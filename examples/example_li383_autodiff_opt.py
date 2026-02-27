"""Autodiff optimization demo using the JAX-native Boozer transform.

This toy example scales the lambda spectrum (lmns) to reduce the L2 norm of
non-axisymmetric |B| harmonics on a single surface.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from booz_xform_jax import Booz_xform
from booz_xform_jax.jax_api import prepare_booz_xform_constants, booz_xform_jax_impl


def main() -> None:
    bx = Booz_xform()
    bx.read_wout("tests/test_files/wout_li383_1.4m.nc")

    bx.mboz = 8
    bx.nboz = 8
    bx.compute_surfs = [0]

    constants = prepare_booz_xform_constants(
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

    booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))

    def objective(scale: jnp.ndarray) -> jnp.ndarray:
        out = booz_fn(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns * scale,
            bmnc=bmnc,
            bsubumnc=bsubumnc,
            bsubvmnc=bsubvmnc,
            iota=iota,
            xm=jnp.asarray(bx.xm),
            xn=jnp.asarray(bx.xn),
            xm_nyq=jnp.asarray(bx.xm_nyq),
            xn_nyq=jnp.asarray(bx.xn_nyq),
            constants=constants,
            surface_indices=jnp.asarray(bx.compute_surfs),
        )
        bmnc_b = out["bmnc_b"][0]
        # Exclude the (m=0,n=0) mode from the objective.
        return jnp.sum(bmnc_b[1:] ** 2)

    scale = jnp.array(1.0)
    opt_lr = 0.1
    for step in range(5):
        loss, grad = jax.value_and_grad(objective)(scale)
        scale = scale - opt_lr * grad
        print(f"step={step} scale={float(scale):.6f} loss={float(loss):.6e}")

    print("final scale:", float(scale))


if __name__ == "__main__":
    main()
