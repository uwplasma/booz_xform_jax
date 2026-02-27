"""Fast JAX-native Boozer transform example (no Python surface loop)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from booz_xform_jax import Booz_xform
from booz_xform_jax.jax_api import prepare_booz_xform_constants, booz_xform_jax_impl


def main() -> None:
    bx = Booz_xform()
    bx.read_wout("tests/test_files/wout_li383_1.4m.nc")

    # Smaller resolution for speed
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

    # JAX arrays (surface dimension first)
    rmnc = jnp.asarray(bx.rmnc).T
    zmns = jnp.asarray(bx.zmns).T
    lmns = jnp.asarray(bx.lmns).T
    bmnc = jnp.asarray(bx.bmnc).T
    bsubumnc = jnp.asarray(bx.bsubumnc).T
    bsubvmnc = jnp.asarray(bx.bsubvmnc).T
    iota = jnp.asarray(bx.iota)

    booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))
    out = booz_fn(
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        iota=iota,
        xm=jnp.asarray(bx.xm),
        xn=jnp.asarray(bx.xn),
        xm_nyq=jnp.asarray(bx.xm_nyq),
        xn_nyq=jnp.asarray(bx.xn_nyq),
        constants=constants,
        grids=grids,
        surface_indices=jnp.asarray(bx.compute_surfs),
    )

    print("Computed bmnc_b shape:", out["bmnc_b"].shape)
    print("First surface |B| coefficients (first 5):", out["bmnc_b"][0, :5])


if __name__ == "__main__":
    main()
