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
