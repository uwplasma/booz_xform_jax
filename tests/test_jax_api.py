import os

import numpy as np

import jax.numpy as jnp

from booz_xform_jax import Booz_xform
from booz_xform_jax.jax_api import booz_xform_jax


TEST_DIR = os.path.join(os.path.dirname(__file__), "test_files")


def test_jax_api_matches_reference_small():
    """Smoke test: JAX API matches Booz_xform.run() on a small surface set."""
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_li383_1.4m.nc"))

    # Reduce resolution for faster test.
    b.mboz = 4
    b.nboz = 4
    b.compute_surfs = [0]
    b.run()

    # Prepare inputs with surface dimension first.
    rmnc = jnp.asarray(np.asarray(b.rmnc).T)
    zmns = jnp.asarray(np.asarray(b.zmns).T)
    lmns = jnp.asarray(np.asarray(b.lmns).T)
    bmnc = jnp.asarray(np.asarray(b.bmnc).T)
    bsubumnc = jnp.asarray(np.asarray(b.bsubumnc).T)
    bsubvmnc = jnp.asarray(np.asarray(b.bsubvmnc).T)
    iota = jnp.asarray(np.asarray(b.iota))

    out = booz_xform_jax(
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        iota=iota,
        xm=b.xm,
        xn=b.xn,
        xm_nyq=b.xm_nyq,
        xn_nyq=b.xn_nyq,
        nfp=b.nfp,
        mboz=b.mboz,
        nboz=b.nboz,
        asym=bool(b.asym),
        surface_indices=[0],
    )

    assert np.allclose(np.asarray(out["jlist"]), np.array([1]))
    assert int(np.asarray(out["ns_b"])) == rmnc.shape[0]

    # Compare a few spectral coefficients.
    np.testing.assert_allclose(
        np.asarray(out["bmnc_b"])[0],
        np.asarray(b.bmnc_b)[:, 0],
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out["rmnc_b"])[0],
        np.asarray(b.rmnc_b)[:, 0],
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out["zmns_b"])[0],
        np.asarray(b.zmns_b)[:, 0],
        rtol=5e-6,
        atol=1e-8,
    )


def test_run_jax_matches_run():
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_li383_1.4m.nc"))
    b.mboz = 4
    b.nboz = 4
    b.compute_surfs = [0]
    b.run()

    out = b.run_jax(jit=False)

    np.testing.assert_allclose(
        np.asarray(out["bmnc_b"])[0],
        np.asarray(b.bmnc_b)[:, 0],
        rtol=5e-6,
        atol=1e-8,
    )
