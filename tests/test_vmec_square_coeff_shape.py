from __future__ import annotations

import numpy as np
import pytest

from booz_xform_jax import Booz_xform


def _write_square_wout(path):
    netCDF4 = pytest.importorskip("netCDF4")

    ns = 5
    mnmax = ns
    mnmax_nyq = 7

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

