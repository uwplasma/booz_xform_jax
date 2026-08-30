"""BoozerConfig / BoozerPlan / chunked-execution contracts."""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from booz_xform_jax import (  # noqa: E402  (x64 must be set before import)
    Booz_xform,
    BoozerConfig,
    prepare_booz_xform_plan,
)
from booz_xform_jax import jax_api  # noqa: E402
from booz_xform_jax.jax_api import booz_xform_jax_impl  # noqa: E402

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_files")


def _case(filename, mboz):
    bx = Booz_xform()
    bx.read_wout(os.path.join(TEST_DIR, filename))
    config = BoozerConfig(mboz=mboz, nboz=mboz)
    plan = prepare_booz_xform_plan(
        nfp=bx.nfp, asym=bool(bx.asym), xm=bx.xm, xn=bx.xn,
        xm_nyq=bx.xm_nyq, xn_nyq=bx.xn_nyq, config=config)
    arrays = dict(
        rmnc=jnp.asarray(bx.rmnc).T, zmns=jnp.asarray(bx.zmns).T,
        lmns=jnp.asarray(bx.lmns).T, bmnc=jnp.asarray(bx.bmnc).T,
        bsubumnc=jnp.asarray(bx.bsubumnc).T,
        bsubvmnc=jnp.asarray(bx.bsubvmnc).T, iota=jnp.asarray(bx.iota))
    if bool(bx.asym):
        arrays.update(
            rmns=jnp.asarray(bx.rmns).T, zmnc=jnp.asarray(bx.zmnc).T,
            lmnc=jnp.asarray(bx.lmnc).T, bmns=jnp.asarray(bx.bmns).T,
            bsubumns=jnp.asarray(bx.bsubumns).T,
            bsubvmns=jnp.asarray(bx.bsubvmns).T)
    static = dict(
        xm=jnp.asarray(bx.xm), xn=jnp.asarray(bx.xn),
        xm_nyq=jnp.asarray(bx.xm_nyq), xn_nyq=jnp.asarray(bx.xn_nyq),
        constants=plan.constants, grids=plan.grids)
    return bx, config, plan, arrays, static


def test_config_validates_inputs():
    with pytest.raises(ValueError, match="fourier_mode"):
        BoozerConfig(mboz=4, nboz=4, fourier_mode="fft")
    with pytest.raises(ValueError, match="surface_chunk"):
        BoozerConfig(mboz=4, nboz=4, surface_chunk="half")
    with pytest.raises(ValueError, match="surface_chunk"):
        BoozerConfig(mboz=4, nboz=4, surface_chunk=0)
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        BoozerConfig(mboz=4, nboz=4, memory_budget_bytes=0)


def test_from_env_reads_the_legacy_variables(monkeypatch):
    monkeypatch.setenv("BOOZ_XFORM_JAX_FOURIER_MODE", "streamed")
    monkeypatch.setenv("BOOZ_XFORM_JAX_TRIG_F32", "1")
    config = BoozerConfig.from_env(mboz=6, nboz=6)
    assert config.fourier_mode == "streamed"
    assert config.trig_f32 is True
    monkeypatch.delenv("BOOZ_XFORM_JAX_FOURIER_MODE")
    monkeypatch.delenv("BOOZ_XFORM_JAX_TRIG_F32")
    assert BoozerConfig.from_env(mboz=6, nboz=6) == BoozerConfig(
        mboz=6, nboz=6)


def test_plan_cache_hits_on_equal_content_and_stays_bounded():
    bx = Booz_xform()
    bx.read_wout(os.path.join(TEST_DIR, "wout_li383_1.4m.nc"))
    kwargs = dict(nfp=bx.nfp, asym=bool(bx.asym), xm=bx.xm, xn=bx.xn,
                  xm_nyq=bx.xm_nyq, xn_nyq=bx.xn_nyq)
    config = BoozerConfig(mboz=4, nboz=4)
    assert prepare_booz_xform_plan(**kwargs, config=config) is \
        prepare_booz_xform_plan(**kwargs, config=config)
    for extra in range(jax_api._PLAN_CACHE_MAX + 2):
        prepare_booz_xform_plan(
            **kwargs, config=BoozerConfig(mboz=3 + extra, nboz=3))
    assert len(jax_api._PLAN_CACHE) <= jax_api._PLAN_CACHE_MAX


@pytest.mark.parametrize("filename,mboz", [
    ("wout_li383_1.4m.nc", 8),
    ("wout_up_down_asymmetric_tokamak.nc", 6),
])
def test_plan_is_bit_identical_and_chunked_is_rounding_close(filename, mboz):
    """The plan path reproduces the legacy env path bit-for-bit; chunked
    execution reorders XLA fusion (one big vmap versus lax.map blocks), so
    its contract is agreement to double rounding, not bit identity."""
    _bx, config, plan, arrays, static = _case(filename, mboz)
    ns = int(arrays["rmnc"].shape[0])
    indices = jnp.arange(0, min(ns, 12))
    legacy = booz_xform_jax_impl(**arrays, **static, surface_indices=indices)
    planned = booz_xform_jax_impl(**arrays, **static,
                                  surface_indices=indices, plan=plan)
    chunk_config = BoozerConfig(mboz=mboz, nboz=mboz, surface_chunk=5)
    chunked = booz_xform_jax_impl(**arrays, **static,
                                  surface_indices=indices,
                                  config=chunk_config)
    for key in legacy:
        assert np.array_equal(np.asarray(legacy[key]),
                              np.asarray(planned[key])), key
        np.testing.assert_allclose(
            np.asarray(legacy[key]), np.asarray(chunked[key]),
            rtol=1.0e-10, atol=1.0e-13, err_msg=key)


def test_chunked_execution_is_surface_order_invariant():
    _bx, config, plan, arrays, static = _case("wout_li383_1.4m.nc", 6)
    order = jnp.asarray([7, 2, 11, 0, 5, 9, 1])
    chunk_config = BoozerConfig(mboz=6, nboz=6, surface_chunk=3)
    shuffled = booz_xform_jax_impl(
        **arrays, **static, surface_indices=order, config=chunk_config)
    straight = booz_xform_jax_impl(
        **arrays, **static, surface_indices=jnp.sort(order),
        config=chunk_config)
    inverse = jnp.argsort(jnp.argsort(order))
    np.testing.assert_allclose(
        np.asarray(shuffled["bmnc_b"]),
        np.asarray(straight["bmnc_b"])[np.asarray(inverse)],
        rtol=1.0e-10, atol=1.0e-13)


def test_derivatives_agree_through_the_chunked_path():
    _bx, config, plan, arrays, static = _case("wout_li383_1.4m.nc", 6)
    indices = jnp.arange(0, 11)  # deliberately not a multiple of the chunk
    chunk_config = BoozerConfig(mboz=6, nboz=6, surface_chunk=4)

    def value(bmnc, cfg):
        return booz_xform_jax_impl(
            **{**arrays, "bmnc": bmnc}, **static,
            surface_indices=indices, config=cfg)["bmnc_b"]

    tangent = jnp.ones_like(arrays["bmnc"])
    jvp_full = jax.jvp(lambda b: value(b, config), (arrays["bmnc"],),
                       (tangent,))[1]
    jvp_chunk = jax.jvp(lambda b: value(b, chunk_config), (arrays["bmnc"],),
                        (tangent,))[1]
    np.testing.assert_allclose(np.asarray(jvp_full), np.asarray(jvp_chunk),
                               rtol=1.0e-10, atol=1.0e-13)
    cotangent = jnp.ones((11, int(plan.grids.xm_b.size)))
    vjp_full = jax.vjp(lambda b: value(b, config), arrays["bmnc"])[1](
        cotangent)[0]
    vjp_chunk = jax.vjp(lambda b: value(b, chunk_config), arrays["bmnc"])[1](
        cotangent)[0]
    np.testing.assert_allclose(np.asarray(vjp_full), np.asarray(vjp_chunk),
                               rtol=1.0e-10, atol=1.0e-13)


def test_auto_chunk_policy_is_deterministic_and_budgeted():
    _bx, config, plan, arrays, static = _case("wout_li383_1.4m.nc", 6)
    constants = plan.constants
    mn_non = int(plan.tables["m_non_f"].shape[0])
    mn_nyq = int(plan.tables["m_nyq_f"].shape[0])
    mn_boz = int(plan.grids.xm_b.size)
    per_surface = 4 * 8 * constants.ntheta * constants.nzeta * (
        mn_non + mn_nyq + mn_boz)
    budgeted = BoozerConfig(mboz=6, nboz=6,
                            memory_budget_bytes=3 * per_surface)
    chunk = jax_api._resolve_surface_chunk(
        budgeted, constants, 12, mn_non=mn_non, mn_nyq=mn_nyq, mn_boz=mn_boz)
    assert chunk == 3
    # No budget: auto batches everything (the legacy behaviour).
    assert jax_api._resolve_surface_chunk(
        config, constants, 12, mn_non=mn_non, mn_nyq=mn_nyq,
        mn_boz=mn_boz) == 12
    # A budget below one surface still makes progress.
    tiny = BoozerConfig(mboz=6, nboz=6, memory_budget_bytes=1)
    assert jax_api._resolve_surface_chunk(
        tiny, constants, 12, mn_non=mn_non, mn_nyq=mn_nyq,
        mn_boz=mn_boz) == 1
