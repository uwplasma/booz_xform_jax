from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

from booz_xform_jax import Booz_xform


def _wout_like_from_netcdf(path: Path):
    netCDF4 = pytest.importorskip("netCDF4")
    ds = netCDF4.Dataset(str(path), "r")  # type: ignore

    lasym_name = "lasym__logical__"
    if lasym_name in ds.variables:
        lasym = bool(ds.variables[lasym_name][...].item())
    else:
        lasym = bool(getattr(ds, lasym_name, False))

    rmnc = np.asarray(ds.variables["rmnc"][:])
    zeros = np.zeros_like(rmnc)

    def _read(name: str) -> np.ndarray:
        return np.asarray(ds.variables[name][:]) if name in ds.variables else zeros

    wout = types.SimpleNamespace(
        lasym=lasym,
        nfp=int(ds.variables["nfp"][...].item()),
        mpol=int(ds.variables["mpol"][...].item()),
        ntor=int(ds.variables["ntor"][...].item()),
        xm=np.asarray(ds.variables["xm"][:], dtype=int),
        xn=np.asarray(ds.variables["xn"][:], dtype=int),
        xm_nyq=np.asarray(ds.variables["xm_nyq"][:], dtype=int),
        xn_nyq=np.asarray(ds.variables["xn_nyq"][:], dtype=int),
        ns=int(ds.variables["ns"][...].item()),
        rmnc=rmnc,
        rmns=_read("rmns") if lasym else zeros,
        zmnc=_read("zmnc") if lasym else zeros,
        zmns=_read("zmns"),
        lmnc=_read("lmnc") if lasym else zeros,
        lmns=_read("lmns"),
        bmnc=_read("bmnc"),
        bmns=_read("bmns") if lasym else zeros,
        bsubumnc=_read("bsubumnc"),
        bsubumns=_read("bsubumns") if lasym else zeros,
        bsubvmnc=_read("bsubvmnc"),
        bsubvmns=_read("bsubvmns") if lasym else zeros,
        iotas=np.asarray(ds.variables["iotas"][:]),
    )
    ds.close()
    return wout


def test_read_wout_data_matches_read_wout():
    root = Path(__file__).resolve().parent
    wout_path = root / "test_files" / "wout_li383_1.4m.nc"
    wout_like = _wout_like_from_netcdf(wout_path)

    b_ref = Booz_xform()
    b_ref.read_wout(str(wout_path))

    b_obj = Booz_xform()
    b_obj.read_wout_data(wout_like)

    np.testing.assert_allclose(np.asarray(b_ref.rmnc), np.asarray(b_obj.rmnc))
    np.testing.assert_allclose(np.asarray(b_ref.zmns), np.asarray(b_obj.zmns))
    np.testing.assert_allclose(np.asarray(b_ref.bmnc), np.asarray(b_obj.bmnc))
    np.testing.assert_allclose(np.asarray(b_ref.iota), np.asarray(b_obj.iota))


@pytest.mark.parametrize(
    ("wout_name", "mboz", "nboz", "surfaces"),
    [
        ("wout_circular_tokamak.nc", 6, 0, (0.25, 0.50)),
        ("wout_up_down_asymmetric_tokamak.nc", 8, 0, (0.25, 0.50)),
    ],
)
def test_read_wout_data_run_jax_matches_file_backed_transform(wout_name, mboz, nboz, surfaces):
    root = Path(__file__).resolve().parent
    wout_path = root / "test_files" / wout_name
    if not wout_path.exists():
        pytest.skip(f"Missing reference wout file: {wout_path}")
    wout_like = _wout_like_from_netcdf(wout_path)

    b_ref = Booz_xform()
    b_ref.read_wout(str(wout_path))
    b_ref.mboz = mboz
    b_ref.nboz = nboz
    b_ref.compute_surfs = []
    b_ref.register_surfaces(surfaces)
    ref = b_ref.run_jax(jit=False)

    b_obj = Booz_xform()
    b_obj.read_wout_data(wout_like)
    b_obj.mboz = mboz
    b_obj.nboz = nboz
    b_obj.compute_surfs = []
    b_obj.register_surfaces(surfaces)
    actual = b_obj.run_jax(jit=False)

    assert tuple(actual) == tuple(ref)
    for name in ("bmnc_b", "bmns_b", "rmnc_b", "rmns_b", "zmns_b", "zmnc_b", "pmns_b", "pmnc_b", "gmnc_b", "gmns_b"):
        if name in ref:
            np.testing.assert_allclose(np.asarray(actual[name]), np.asarray(ref[name]), rtol=2.0e-12, atol=2.0e-13)
    np.testing.assert_array_equal(np.asarray(actual["jlist"]), np.asarray(ref["jlist"]))
