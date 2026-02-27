"""Profile the JAX-native Boozer transform with TensorBoard traces."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from booz_xform_jax import Booz_xform
from booz_xform_jax.jax_api import prepare_booz_xform_constants, booz_xform_jax_impl


def main() -> None:
    out_dir = Path("profiles/booz_xform_jax_trace")
    out_dir.mkdir(parents=True, exist_ok=True)

    bx = Booz_xform()
    bx.read_wout("tests/test_files/wout_li383_1.4m.nc")
    bx.mboz = 8
    bx.nboz = 8
    bx.compute_surfs = [0, 5, 10]

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

    # Warmup compile.
    booz_fn(
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
        surface_indices=jnp.asarray(bx.compute_surfs),
    )

    jax.profiler.start_trace(str(out_dir))
    booz_fn(
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
        surface_indices=jnp.asarray(bx.compute_surfs),
    )
    jax.profiler.stop_trace()

    print(f"Wrote trace to {out_dir}")


if __name__ == "__main__":
    main()
