"""CLI tests for the legacy-compatible ``xbooz_xform`` wrapper."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests" / "test_files"


def _run_cli(tmp_path: Path, input_name: str, *, screen_flag: str = "F") -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-m", "booz_xform_jax", input_name, screen_flag],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_help() -> None:
    env = dict(os.environ)
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath
    proc = subprocess.run(
        [sys.executable, "-m", "booz_xform_jax", "-h"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert "xbooz_xform <infile> (T or F)" in proc.stdout


def test_cli_li383_matches_reference(tmp_path: Path) -> None:
    shutil.copy(TEST_DIR / "booz_in.li383_1.4m", tmp_path / "booz_in.li383_1.4m")
    shutil.copy(TEST_DIR / "wout_li383_1.4m.nc", tmp_path / "wout_li383_1.4m.nc")

    proc = _run_cli(tmp_path, "booz_in.li383_1.4m", screen_flag="F")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert proc.stdout.strip() == ""

    out_file = tmp_path / "boozmn_li383_1.4m.nc"
    assert out_file.exists()

    with Dataset(out_file) as out_ds, Dataset(TEST_DIR / "boozmn_li383_1.4m.nc") as ref_ds:
        assert int(out_ds.variables["mboz_b"][...].item()) == int(ref_ds.variables["mboz_b"][...].item())
        assert int(out_ds.variables["nboz_b"][...].item()) == int(ref_ds.variables["nboz_b"][...].item())
        np.testing.assert_array_equal(out_ds.variables["jlist"][:], ref_ds.variables["jlist"][:])
        for name in ["bmnc_b", "rmnc_b", "zmns_b", "pmns_b", "gmn_b"]:
            np.testing.assert_allclose(out_ds.variables[name][:], ref_ds.variables[name][:], rtol=1e-12, atol=1e-12)


def test_cli_missing_jlist_defaults_to_all_surfaces(tmp_path: Path) -> None:
    shutil.copy(TEST_DIR / "booz_in.circular_tokamak", tmp_path / "booz_in.circular_tokamak")
    shutil.copy(TEST_DIR / "wout_circular_tokamak.nc", tmp_path / "wout_circular_tokamak.nc")

    proc = _run_cli(tmp_path, "booz_in.circular_tokamak", screen_flag="F")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "No jlist data was found in Boozer input file." in proc.stdout

    out_file = tmp_path / "boozmn_circular_tokamak.nc"
    assert out_file.exists()

    with Dataset(out_file) as out_ds, Dataset(TEST_DIR / "boozmn_circular_tokamak.nc") as ref_ds:
        np.testing.assert_array_equal(out_ds.variables["jlist"][:], ref_ds.variables["jlist"][:])
        np.testing.assert_allclose(out_ds.variables["bmnc_b"][:], ref_ds.variables["bmnc_b"][:], rtol=1e-12, atol=1e-12)

