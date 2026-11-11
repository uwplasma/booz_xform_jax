"""Fast GPU checks, selected with ``pytest -m gpu``.

They also run on CPU, where the device comparison is trivial.
"""
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from booz_xform_jax import Booz_xform
from booz_xform_jax.jax_api import booz_xform_jax

pytestmark = pytest.mark.gpu

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_files")


def _inputs(wout, surface=-1, res=4):
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, wout))
    b.mboz = b.nboz = res
    surface %= b.ns_in
    b.compute_surfs = [surface]
    b.run()
    kw = dict(
        rmnc=jnp.asarray(np.asarray(b.rmnc).T),
        zmns=jnp.asarray(np.asarray(b.zmns).T),
        lmns=jnp.asarray(np.asarray(b.lmns).T),
        bmnc=jnp.asarray(np.asarray(b.bmnc).T),
        bsubumnc=jnp.asarray(np.asarray(b.bsubumnc).T),
        bsubvmnc=jnp.asarray(np.asarray(b.bsubvmnc).T),
        iota=jnp.asarray(np.asarray(b.iota)),
        xm=b.xm, xn=b.xn, xm_nyq=b.xm_nyq, xn_nyq=b.xn_nyq,
        nfp=b.nfp, mboz=b.mboz, nboz=b.nboz, asym=bool(b.asym),
        surface_indices=[surface],
    )
    return b, kw


def test_default_device_runs_transform():
    """The default backend executes a Boozer transform and returns finite output."""
    _, kw = _inputs("wout_li383_1.4m.nc")
    out = booz_xform_jax(**kw)
    assert np.all(np.isfinite(np.asarray(out["bmnc_b"])))
    assert out["bmnc_b"].devices() <= set(jax.devices())


def test_matches_reference_li383():
    """Default-device result matches the reference implementation."""
    b, kw = _inputs("wout_li383_1.4m.nc")
    out = booz_xform_jax(**kw)
    for name in ("bmnc_b", "rmnc_b", "zmns_b", "gmnc_b"):
        np.testing.assert_allclose(
            np.asarray(out[name])[0], np.asarray(getattr(b, name))[:, 0],
            rtol=5e-6, atol=1e-8,
        )


def test_matches_cpu_device():
    """Default-device result agrees with the same transform pinned to the CPU."""
    _, kw = _inputs("wout_li383_1.4m.nc")
    out = booz_xform_jax(**kw)
    with jax.default_device(jax.devices("cpu")[0]):
        ref = booz_xform_jax(**kw)
    for name in ("bmnc_b", "rmnc_b", "zmns_b", "gmnc_b"):
        np.testing.assert_allclose(
            np.asarray(out[name]), np.asarray(ref[name]), rtol=1e-8, atol=1e-10
        )


def test_tokamak_matches_reference():
    """A second equilibrium (circular tokamak) matches the reference."""
    b, kw = _inputs("wout_circular_tokamak.nc", surface=0, res=3)
    out = booz_xform_jax(**kw)
    np.testing.assert_allclose(
        np.asarray(out["bmnc_b"])[0], np.asarray(b.bmnc_b)[:, 0],
        rtol=5e-6, atol=1e-8,
    )
