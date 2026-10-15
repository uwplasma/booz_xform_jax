"""Stateless boozmn reader: layout and round trip through write_boozmn."""

import os

import numpy as np
import pytest

from booz_xform_jax import Booz_xform, read_boozmn

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_files")


@pytest.mark.parametrize("name", ["li383_1.4m", "up_down_asymmetric_tokamak"])
def test_read_boozmn_dict_round_trip(tmp_path, name):
    path = os.path.join(TEST_DIR, f"boozmn_{name}.nc")
    d = read_boozmn(path)
    mn = d["ixm_b"].size
    assert d["bmnc_b"].shape == (d["jlist"].size, mn)
    assert d["iota_b"].shape == (d["ns_b"],)
    assert (d["ixm_b"][0], d["ixn_b"][0]) == (0, 0) and d["bmnc_b"][0, 0] > 0
    assert ("bmns_b" in d) == d["lasym"]

    b = Booz_xform()
    b.verbose = 0
    b.read_boozmn(path)
    np.testing.assert_array_equal(b.bmnc_b, d["bmnc_b"].T)
    out = tmp_path / "boozmn_out.nc"
    b.write_boozmn(str(out))
    d2 = read_boozmn(out)
    for key in ("lasym", "nfp_b", "ns_b", "mboz_b", "nboz_b", "ixm_b", "ixn_b",
                "jlist", "iota_b", "buco_b", "bvco_b", "bmnc_b", "rmnc_b",
                "zmns_b", "pmns_b", "gmn_b") + (("bmns_b", "rmns_b", "zmnc_b",
                                                  "pmnc_b", "gmns_b") if d["lasym"] else ()):
        np.testing.assert_allclose(d2[key], d[key], rtol=1e-13, atol=0, err_msg=key)
