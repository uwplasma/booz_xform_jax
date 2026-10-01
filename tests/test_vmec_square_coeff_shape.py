from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from booz_xform_jax import Booz_xform


def _write_square_wout(path, ns=5, mnmax=5, mnmax_nyq=7):
    netCDF4 = pytest.importorskip("netCDF4")

    with netCDF4.Dataset(str(path), "w") as ds:  # type: ignore[attr-defined]
        ds.createDimension("radius", ns)
        ds.createDimension("mn_mode", mnmax)
        ds.createDimension("mn_mode_nyq", mnmax_nyq)

        def scalar(name, value, dtype="i4"):
            var = ds.createVariable(name, dtype)
            var[...] = value

        scalar("nfp", 1)
        scalar("mpol", 4)
        scalar("ntor", 0)
        scalar("mnmax", mnmax)
        scalar("mnmax_nyq", mnmax_nyq)
        scalar("ns", ns)
        scalar("aspect", 5.0, dtype="f8")

        ds.createVariable("xm", "i4", ("mn_mode",))[:] = np.arange(mnmax)
        ds.createVariable("xn", "i4", ("mn_mode",))[:] = np.zeros(mnmax, dtype=int)
        ds.createVariable("xm_nyq", "i4", ("mn_mode_nyq",))[:] = np.arange(mnmax_nyq)
        ds.createVariable("xn_nyq", "i4", ("mn_mode_nyq",))[:] = np.zeros(mnmax_nyq, dtype=int)
        ds.createVariable("iotas", "f8", ("radius",))[:] = np.linspace(0.1, 0.5, ns)

        radius = np.arange(ns, dtype=float)[:, None]
        modes = np.arange(mnmax, dtype=float)[None, :]
        nonnyq = 10.0 * radius + modes

        for name, values in {
            "rmnc": nonnyq,
            "zmns": 100.0 + nonnyq,
            "lmns": 0.01 * nonnyq,
        }.items():
            ds.createVariable(name, "f8", ("radius", "mn_mode"))[:] = values

        nyq = np.ones((ns, mnmax_nyq))
        for name, values in {
            "bmnc": nyq,
            "bsubumnc": 0.1 * nyq,
            "bsubvmnc": 0.2 * nyq,
        }.items():
            ds.createVariable(name, "f8", ("radius", "mn_mode_nyq"))[:] = values


def test_read_wout_uses_dimension_names_when_ns_equals_mnmax(tmp_path):
    wout_path = tmp_path / "wout_square_coeffs.nc"
    _write_square_wout(wout_path)

    b = Booz_xform(verbose=0)
    b.read_wout(str(wout_path))

    assert b.mnmax == 5
    assert b.rmnc.shape == (5, 4)

    rmnc_input = np.asarray([[10.0 * radius + mode for mode in range(5)] for radius in range(5)])
    expected_m0 = 0.5 * (rmnc_input[:-1, 0] + rmnc_input[1:, 0])
    np.testing.assert_allclose(np.asarray(b.rmnc[0, :]), expected_m0)


@pytest.mark.parametrize("shape", [(5, 5, 7), (9, 5, 9)])
def test_typed_vmex_wout_preserves_square_coefficients(tmp_path, shape):
    vmex = pytest.importorskip("vmex")
    path = tmp_path / "wout_square.nc"
    _write_square_wout(path, *shape)
    wout = vmex.read_wout(path)
    expected, actual = Booz_xform(verbose=0), Booz_xform(verbose=0)
    expected.read_wout(str(path))
    actual.read_wout_data(wout)
    for name in ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    # Reusing a file reader must not attach its old layout to a generic object.
    with pytest.raises(ValueError, match="ambiguous"):
        expected.read_wout_data(SimpleNamespace(**vars(wout)))
