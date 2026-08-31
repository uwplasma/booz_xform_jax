import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from netCDF4 import Dataset

from booz_xform_jax import Booz_xform, BoozerConfig, prepare_booz_xform_plan
from booz_xform_jax.jax_api import booz_xform_jax, booz_xform_jax_impl
from booz_xform_jax.trig import _init_trig


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


def test_jax_api_covariant_field_gradients_are_finite():
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

    def objective(bsubu, bsubv):
        out = booz_xform_jax(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc,
            bsubumnc=bsubu,
            bsubvmnc=bsubv,
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
        return (
            jnp.mean(out["bmnc_b"])
            + jnp.mean(out["rmnc_b"])
            + jnp.mean(out["pmns_b"])
            + jnp.mean(out["bvco_b"])
        )

    grad_u, grad_v = jax.grad(objective, argnums=(0, 1))(bsubumnc, bsubvmnc)

    assert np.all(np.isfinite(np.asarray(grad_u)))
    assert np.all(np.isfinite(np.asarray(grad_v)))


def _dense_reference(plan, arrays, indices):
    """Point-by-mode projection oracle for the separable contraction.

    Reimplements the original dense (points x mn_boz) phase-tensor
    projection so tests can verify the separable reorganization in
    ``jax_api._surface_transform`` against it, for values and derivatives.
    """
    c, g, t = plan.constants, plan.grids, plan.tables

    def one_surface(rmnc, rmns, zmnc, zmns, lmnc, lmns, bmnc,
                    bsubumnc, bsubvmnc, iota, bmns, bsubumns, bsubvmns):
        m_nyq, n_nyq = t["m_nyq_f"], t["n_nyq_f"]
        idx00 = jnp.where((m_nyq == 0) & (n_nyq == 0), size=1)[0][0]
        I_b, G_b = bsubumnc[idx00], bsubvmnc[idx00]
        m_nonzero = m_nyq != 0.0
        n_only = jnp.logical_and(~m_nonzero, n_nyq != 0.0)
        m_safe = jnp.where(m_nonzero, m_nyq, 1.0)
        n_safe = jnp.where(n_only, n_nyq, 1.0)
        wmns = jnp.where(m_nonzero, bsubumnc / m_safe,
                         jnp.where(n_only, -bsubvmnc / n_safe, 0.0))
        r = t["tcos_non"] @ rmnc
        z = t["tsin_non"] @ zmns
        lam = t["tsin_non"] @ lmns
        dlam_dth = t["tcos_non"] @ (lmns * t["m_non_f"])
        dlam_dze = -(t["tcos_non"] @ (lmns * t["n_non_f"]))
        w = t["tsin_nyq"] @ wmns
        dw_dth = t["tcos_nyq"] @ (wmns * m_nyq)
        dw_dze = -(t["tcos_nyq"] @ (wmns * n_nyq))
        bmod = t["tcos_nyq"] @ bmnc
        if c.asym:
            wmnc = jnp.where(m_nonzero, -bsubumns / m_safe,
                             jnp.where(n_only, bsubvmns / n_safe, 0.0))
            r = r + t["tsin_non"] @ rmns
            z = z + t["tcos_non"] @ zmnc
            lam = lam + t["tcos_non"] @ lmnc
            dlam_dth = dlam_dth - t["tsin_non"] @ (lmnc * t["m_non_f"])
            dlam_dze = dlam_dze + t["tsin_non"] @ (lmnc * t["n_non_f"])
            w = w + t["tcos_nyq"] @ wmnc
            dw_dth = dw_dth - t["tsin_nyq"] @ (wmnc * m_nyq)
            dw_dze = dw_dze + t["tsin_nyq"] @ (wmnc * n_nyq)
            bmod = bmod + t["tsin_nyq"] @ bmns
        GI = G_b + iota * I_b
        nu = (w - I_b * lam) / GI
        theta_b = g.theta_grid + lam + iota * nu
        zeta_b = g.zeta_grid + nu
        dnu_dze = (dw_dze - I_b * dlam_dze) / GI
        dnu_dth = (dw_dth - I_b * dlam_dth) / GI
        jac_v = ((1.0 + dlam_dth) * (1.0 + dnu_dze)
                 + (iota - dlam_dze) * dnu_dth)
        cosm_b, sinm_b, cosn_b, sinn_b = _init_trig(
            theta_b, zeta_b, c.mboz, c.nboz, c.nfp)
        if not c.asym:
            for idx in (t["idx_theta0"], t["idx_thetapi"]):
                cosm_b = cosm_b.at[idx, :].multiply(0.5)
                sinm_b = sinm_b.at[idx, :].multiply(0.5)
        cm = jnp.take(cosm_b, t["m_b"], axis=1)
        sm = jnp.take(sinm_b, t["m_b"], axis=1)
        cn = jnp.take(cosn_b, t["abs_n_b"], axis=1)
        sn = jnp.take(sinn_b, t["abs_n_b"], axis=1)
        tcos = cm * cn + sm * sn * t["sign_b"]
        tsin = sm * cn - cm * sn * t["sign_b"]
        ff0 = (2.0 / (c.ntheta * c.nzeta) if c.asym
               else 2.0 / ((c.nu2_b - 1) * c.nzeta))
        ff = jnp.full((t["m_b"].shape[0],), ff0).at[0].set(0.5 * ff0)
        base = {"b": bmod * jac_v, "r": r * jac_v, "z": z * jac_v,
                "nu": nu * jac_v, "g": (GI / (bmod * bmod)) * jac_v}
        pc = {k: ff * (tcos.T @ v) for k, v in base.items()}
        ps = {k: ff * (tsin.T @ v) for k, v in base.items()}
        zero = jnp.zeros_like(pc["b"])
        if not c.asym:
            ps = {"z": ps["z"], "nu": ps["nu"]}
            pc = {"b": pc["b"], "r": pc["r"], "g": pc["g"]}
            return (pc["b"], zero, pc["r"], zero, zero, ps["z"],
                    zero, ps["nu"], pc["g"], zero)
        return (pc["b"], ps["b"], pc["r"], ps["r"], pc["z"], ps["z"],
                pc["nu"], ps["nu"], pc["g"], ps["g"])

    zeros_non = jnp.zeros_like(arrays["rmnc"])
    zeros_nyq = jnp.zeros_like(arrays["bmnc"])
    stacked = tuple(
        jnp.take(arrays.get(name, default), indices, axis=0)
        for name, default in (
            ("rmnc", None), ("rmns", zeros_non), ("zmnc", zeros_non),
            ("zmns", None), ("lmnc", zeros_non), ("lmns", None),
            ("bmnc", None), ("bsubumnc", None), ("bsubvmnc", None),
            ("iota", None), ("bmns", zeros_nyq), ("bsubumns", zeros_nyq),
            ("bsubvmns", zeros_nyq)))
    outs = jax.vmap(one_surface)(*stacked)
    keys = ("bmnc_b", "bmns_b", "rmnc_b", "rmns_b", "zmnc_b", "zmns_b",
            "numnc_b", "numns_b", "gmnc_b", "gmns_b")
    return dict(zip(keys, outs))


def _plan_case(filename, mboz):
    b = Booz_xform()
    b.read_wout(os.path.join(TEST_DIR, filename))
    config = BoozerConfig(mboz=mboz, nboz=mboz)
    plan = prepare_booz_xform_plan(
        nfp=b.nfp, asym=bool(b.asym), xm=b.xm, xn=b.xn,
        xm_nyq=b.xm_nyq, xn_nyq=b.xn_nyq, config=config)
    arrays = dict(
        rmnc=_surface_first(b, "rmnc"), zmns=_surface_first(b, "zmns"),
        lmns=_surface_first(b, "lmns"), bmnc=_surface_first(b, "bmnc"),
        bsubumnc=_surface_first(b, "bsubumnc"),
        bsubvmnc=_surface_first(b, "bsubvmnc"),
        iota=jnp.asarray(np.asarray(b.iota)))
    if bool(b.asym):
        arrays.update(
            rmns=_surface_first(b, "rmns"), zmnc=_surface_first(b, "zmnc"),
            lmnc=_surface_first(b, "lmnc"), bmns=_surface_first(b, "bmns"),
            bsubumns=_surface_first(b, "bsubumns"),
            bsubvmns=_surface_first(b, "bsubvmns"))
    static = dict(
        xm=jnp.asarray(b.xm), xn=jnp.asarray(b.xn),
        xm_nyq=jnp.asarray(b.xm_nyq), xn_nyq=jnp.asarray(b.xn_nyq),
        constants=plan.constants, grids=plan.grids)
    return plan, arrays, static


@pytest.mark.parametrize("filename,mboz", [
    ("wout_li383_1.4m.nc", 8),
    ("wout_up_down_asymmetric_tokamak.nc", 6),
])
def test_separable_projection_matches_dense_reference(filename, mboz):
    """The separable contraction is a reorganization of the dense one, so
    values and derivatives must agree to double-rounding tolerances."""
    plan, arrays, static = _plan_case(filename, mboz)
    ns = int(arrays["rmnc"].shape[0])
    indices = jnp.arange(0, min(ns, 9))

    out = booz_xform_jax_impl(**arrays, **static,
                              surface_indices=indices, plan=plan)
    ref = _dense_reference(plan, arrays, indices)
    for key, expected in ref.items():
        np.testing.assert_allclose(
            np.asarray(out[key]), np.asarray(expected),
            rtol=1e-10, atol=1e-13, err_msg=key)

    def kernel_fn(bmnc):
        return booz_xform_jax_impl(
            **{**arrays, "bmnc": bmnc}, **static,
            surface_indices=indices, plan=plan)["bmnc_b"]

    def reference_fn(bmnc):
        return _dense_reference(plan, {**arrays, "bmnc": bmnc},
                                indices)["bmnc_b"]

    tangent = jnp.ones_like(arrays["bmnc"])
    jvp_out = jax.jvp(kernel_fn, (arrays["bmnc"],), (tangent,))[1]
    jvp_ref = jax.jvp(reference_fn, (arrays["bmnc"],), (tangent,))[1]
    np.testing.assert_allclose(np.asarray(jvp_out), np.asarray(jvp_ref),
                               rtol=1e-10, atol=1e-13)
    val, pull = jax.vjp(kernel_fn, arrays["bmnc"])
    vjp_out = pull(jnp.ones_like(val))[0]
    val, pull = jax.vjp(reference_fn, arrays["bmnc"])
    vjp_ref = pull(jnp.ones_like(val))[0]
    np.testing.assert_allclose(np.asarray(vjp_out), np.asarray(vjp_ref),
                               rtol=1e-10, atol=1e-13)


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
