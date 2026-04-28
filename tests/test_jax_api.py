import os

import numpy as np

import jax
import jax.numpy as jnp
from netCDF4 import Dataset

from booz_xform_jax import Booz_xform
from booz_xform_jax.jax_api import booz_xform_jax


TEST_DIR = os.path.join(os.path.dirname(__file__), "test_files")


def _surface_first(b: Booz_xform, name: str) -> jnp.ndarray:
    return jnp.asarray(np.asarray(getattr(b, name)).T)


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

    assert np.allclose(np.asarray(out["jlist"]), np.array([2]))
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
    np.testing.assert_allclose(
        np.asarray(out["gmnc_b"])[0],
        np.asarray(b.gmnc_b)[:, 0],
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out["gmn_b"]),
        np.asarray(out["gmnc_b"]),
        rtol=0.0,
        atol=0.0,
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
    np.testing.assert_allclose(
        np.asarray(out["gmnc_b"])[0],
        np.asarray(b.gmnc_b)[:, 0],
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out["gmn_b"]),
        np.asarray(out["gmnc_b"]),
        rtol=0.0,
        atol=0.0,
    )


def test_streamed_fourier_matches_vectorized():
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_li383_1.4m.nc"))
    b.mboz = 4
    b.nboz = 4
    b.compute_surfs = [0]
    b.run()

    rmnc = jnp.asarray(np.asarray(b.rmnc).T)
    zmns = jnp.asarray(np.asarray(b.zmns).T)
    lmns = jnp.asarray(np.asarray(b.lmns).T)
    bmnc = jnp.asarray(np.asarray(b.bmnc).T)
    bsubumnc = jnp.asarray(np.asarray(b.bsubumnc).T)
    bsubvmnc = jnp.asarray(np.asarray(b.bsubvmnc).T)
    iota = jnp.asarray(np.asarray(b.iota))

    # Vectorized (default)
    if "BOOZ_XFORM_JAX_FOURIER_MODE" in os.environ:
        os.environ.pop("BOOZ_XFORM_JAX_FOURIER_MODE")
    out_vec = booz_xform_jax(
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

    # Streamed
    os.environ["BOOZ_XFORM_JAX_FOURIER_MODE"] = "streamed"
    out_stream = booz_xform_jax(
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
    os.environ.pop("BOOZ_XFORM_JAX_FOURIER_MODE", None)

    np.testing.assert_allclose(
        np.asarray(out_vec["bmnc_b"]),
        np.asarray(out_stream["bmnc_b"]),
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out_vec["rmnc_b"]),
        np.asarray(out_stream["rmnc_b"]),
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out_vec["zmns_b"]),
        np.asarray(out_stream["zmns_b"]),
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out_vec["gmnc_b"]),
        np.asarray(out_stream["gmnc_b"]),
        rtol=5e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(out_stream["gmn_b"]),
        np.asarray(out_stream["gmnc_b"]),
        rtol=0.0,
        atol=0.0,
    )


def test_jacobian_harmonics_are_differentiable_wrt_bmod_spectrum():
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_li383_1.4m.nc"))
    b.mboz = 4
    b.nboz = 4
    b.compute_surfs = [0]
    b.run()

    rmnc = jnp.asarray(np.asarray(b.rmnc).T)
    zmns = jnp.asarray(np.asarray(b.zmns).T)
    lmns = jnp.asarray(np.asarray(b.lmns).T)
    bmnc = jnp.asarray(np.asarray(b.bmnc).T)
    bsubumnc = jnp.asarray(np.asarray(b.bsubumnc).T)
    bsubvmnc = jnp.asarray(np.asarray(b.bsubvmnc).T)
    iota = jnp.asarray(np.asarray(b.iota))

    def jacobian_energy(bmnc_in):
        out = booz_xform_jax(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc_in,
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
        return jnp.sum(out["gmnc_b"] ** 2)

    grad_bmnc = jax.grad(jacobian_energy)(bmnc)

    assert grad_bmnc.shape == bmnc.shape
    assert np.all(np.isfinite(np.asarray(grad_bmnc)))
    assert float(jnp.linalg.norm(grad_bmnc)) > 0.0


def test_jax_api_asymmetric_outputs_match_boozxform_reference_file():
    """JAX asymmetric spectra match the bundled BOOZ_XFORM reference artifact."""
    ref_path = os.path.join(TEST_DIR, "boozmn_up_down_asymmetric_tokamak.nc")
    wout_path = os.path.join(TEST_DIR, "wout_up_down_asymmetric_tokamak.nc")

    with Dataset(ref_path) as ref:
        mboz = int(ref.variables["mboz_b"][()])
        nboz = int(ref.variables["nboz_b"][()])
        surfaces = [int(j) - 2 for j in ref.variables["jlist"][:]]
        reference = {
            name: np.asarray(ref.variables[name][:])
            for name in [
                "bmnc_b",
                "bmns_b",
                "rmnc_b",
                "rmns_b",
                "zmnc_b",
                "zmns_b",
                "pmnc_b",
                "pmns_b",
                "gmn_b",
                "gmns_b",
            ]
        }

    b = Booz_xform()
    b.read_wout(wout_path)
    b.mboz = mboz
    b.nboz = nboz

    out = booz_xform_jax(
        rmnc=_surface_first(b, "rmnc"),
        rmns=_surface_first(b, "rmns"),
        zmnc=_surface_first(b, "zmnc"),
        zmns=_surface_first(b, "zmns"),
        lmnc=_surface_first(b, "lmnc"),
        lmns=_surface_first(b, "lmns"),
        bmnc=_surface_first(b, "bmnc"),
        bmns=_surface_first(b, "bmns"),
        bsubumnc=_surface_first(b, "bsubumnc"),
        bsubumns=_surface_first(b, "bsubumns"),
        bsubvmnc=_surface_first(b, "bsubvmnc"),
        bsubvmns=_surface_first(b, "bsubvmns"),
        iota=jnp.asarray(np.asarray(b.iota)),
        xm=b.xm,
        xn=b.xn,
        xm_nyq=b.xm_nyq,
        xn_nyq=b.xn_nyq,
        nfp=b.nfp,
        mboz=b.mboz,
        nboz=b.nboz,
        asym=bool(b.asym),
        surface_indices=surfaces,
    )

    np.testing.assert_array_equal(np.asarray(out["jlist"]), np.asarray(surfaces) + 2)
    for name, expected in reference.items():
        np.testing.assert_allclose(
            np.asarray(out[name]),
            expected,
            rtol=5e-10,
            atol=5e-11,
        )
    np.testing.assert_allclose(np.asarray(out["gmnc_b"]), reference["gmn_b"], rtol=5e-10, atol=5e-11)
    np.testing.assert_allclose(np.asarray(out["pmnc_b"]), -np.asarray(out["numnc_b"]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(out["pmns_b"]), -np.asarray(out["numns_b"]), rtol=0.0, atol=0.0)


def test_run_jax_asymmetric_matches_reference_run_and_jit():
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_up_down_asymmetric_tokamak.nc"))
    b.mboz = 8
    b.nboz = 0
    b.compute_surfs = [0, 3]
    b.run()

    out = b.run_jax(jit=False)
    out_jit = b.run_jax(jit=True)

    for name in [
        "bmnc_b",
        "bmns_b",
        "rmnc_b",
        "rmns_b",
        "zmnc_b",
        "zmns_b",
        "numnc_b",
        "numns_b",
        "gmnc_b",
        "gmns_b",
    ]:
        expected = np.asarray(getattr(b, name)).T
        np.testing.assert_allclose(np.asarray(out[name]), expected, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(np.asarray(out_jit[name]), np.asarray(out[name]), rtol=5e-10, atol=5e-12)

    np.testing.assert_allclose(np.asarray(out["pmnc_b"]), -np.asarray(out["numnc_b"]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(out["gmn_b"]), np.asarray(out["gmnc_b"]), rtol=0.0, atol=0.0)


def test_asymmetric_streamed_fourier_matches_vectorized_for_all_spectra():
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_up_down_asymmetric_tokamak.nc"))
    b.mboz = 8
    b.nboz = 0
    surfaces = [0]

    kwargs = dict(
        rmnc=_surface_first(b, "rmnc"),
        rmns=_surface_first(b, "rmns"),
        zmnc=_surface_first(b, "zmnc"),
        zmns=_surface_first(b, "zmns"),
        lmnc=_surface_first(b, "lmnc"),
        lmns=_surface_first(b, "lmns"),
        bmnc=_surface_first(b, "bmnc"),
        bmns=_surface_first(b, "bmns"),
        bsubumnc=_surface_first(b, "bsubumnc"),
        bsubumns=_surface_first(b, "bsubumns"),
        bsubvmnc=_surface_first(b, "bsubvmnc"),
        bsubvmns=_surface_first(b, "bsubvmns"),
        iota=jnp.asarray(np.asarray(b.iota)),
        xm=b.xm,
        xn=b.xn,
        xm_nyq=b.xm_nyq,
        xn_nyq=b.xn_nyq,
        nfp=b.nfp,
        mboz=b.mboz,
        nboz=b.nboz,
        asym=bool(b.asym),
        surface_indices=surfaces,
    )

    os.environ.pop("BOOZ_XFORM_JAX_FOURIER_MODE", None)
    out_vec = booz_xform_jax(**kwargs)

    os.environ["BOOZ_XFORM_JAX_FOURIER_MODE"] = "streamed"
    try:
        out_stream = booz_xform_jax(**kwargs)
    finally:
        os.environ.pop("BOOZ_XFORM_JAX_FOURIER_MODE", None)

    for name in [
        "bmnc_b",
        "bmns_b",
        "rmnc_b",
        "rmns_b",
        "zmnc_b",
        "zmns_b",
        "numnc_b",
        "numns_b",
        "gmnc_b",
        "gmns_b",
    ]:
        np.testing.assert_allclose(np.asarray(out_vec[name]), np.asarray(out_stream[name]), rtol=5e-6, atol=1e-8)


def test_asymmetric_jacobian_sine_harmonics_are_differentiable_wrt_bmod_spectrum():
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_up_down_asymmetric_tokamak.nc"))
    b.mboz = 8
    b.nboz = 0
    surfaces = [0]

    rmnc = _surface_first(b, "rmnc")
    rmns = _surface_first(b, "rmns")
    zmnc = _surface_first(b, "zmnc")
    zmns = _surface_first(b, "zmns")
    lmnc = _surface_first(b, "lmnc")
    lmns = _surface_first(b, "lmns")
    bmnc = _surface_first(b, "bmnc")
    bmns = _surface_first(b, "bmns")
    bsubumnc = _surface_first(b, "bsubumnc")
    bsubumns = _surface_first(b, "bsubumns")
    bsubvmnc = _surface_first(b, "bsubvmnc")
    bsubvmns = _surface_first(b, "bsubvmns")
    iota = jnp.asarray(np.asarray(b.iota))

    def gmns_energy(bmns_in):
        out = booz_xform_jax(
            rmnc=rmnc,
            rmns=rmns,
            zmnc=zmnc,
            zmns=zmns,
            lmnc=lmnc,
            lmns=lmns,
            bmnc=bmnc,
            bmns=bmns_in,
            bsubumnc=bsubumnc,
            bsubumns=bsubumns,
            bsubvmnc=bsubvmnc,
            bsubvmns=bsubvmns,
            iota=iota,
            xm=b.xm,
            xn=b.xn,
            xm_nyq=b.xm_nyq,
            xn_nyq=b.xn_nyq,
            nfp=b.nfp,
            mboz=b.mboz,
            nboz=b.nboz,
            asym=bool(b.asym),
            surface_indices=surfaces,
        )
        return jnp.sum(out["gmns_b"] ** 2)

    grad_bmns = jax.grad(gmns_energy)(bmns)

    assert grad_bmns.shape == bmns.shape
    assert np.all(np.isfinite(np.asarray(grad_bmns)))
    assert float(jnp.linalg.norm(grad_bmns)) > 0.0


def test_jacobian_harmonics_have_finite_gradients_wrt_covariant_field():
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, "wout_li383_1.4m.nc"))
    b.mboz = 4
    b.nboz = 4
    surfaces = [0]

    rmnc = _surface_first(b, "rmnc")
    zmns = _surface_first(b, "zmns")
    lmns = _surface_first(b, "lmns")
    bmnc = _surface_first(b, "bmnc")
    bsubumnc = _surface_first(b, "bsubumnc")
    bsubvmnc = _surface_first(b, "bsubvmnc")
    iota = jnp.asarray(np.asarray(b.iota))

    def jacobian_energy(bsubumnc_in):
        out = booz_xform_jax(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc,
            bsubumnc=bsubumnc_in,
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
            surface_indices=surfaces,
        )
        return jnp.sum(out["gmnc_b"] ** 2)

    grad_bsubumnc = jax.grad(jacobian_energy)(bsubumnc)

    assert grad_bsubumnc.shape == bsubumnc.shape
    assert np.all(np.isfinite(np.asarray(grad_bsubumnc)))
    assert float(jnp.linalg.norm(grad_bsubumnc)) > 0.0
