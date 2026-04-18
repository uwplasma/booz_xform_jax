"""Reference-free smoke tests for installed booz_xform_jax CLIs."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

from netCDF4 import Dataset
import pytest


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests" / "test_files"


def _materialize_case(tmp_path: Path) -> None:
    shutil.copy(TEST_DIR / "booz_in.circular_tokamak", tmp_path / "booz_in.circular_tokamak")
    shutil.copy(TEST_DIR / "wout_circular_tokamak.nc", tmp_path / "wout_circular_tokamak.nc")


def test_console_script_smoke(tmp_path: Path) -> None:
    executable = shutil.which("booz_xform_jax")
    if executable is None:
        pytest.skip("booz_xform_jax console script is not on PATH.")

    _materialize_case(tmp_path)
    proc = subprocess.run(
        [executable, "booz_in.circular_tokamak", "F"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    output = tmp_path / "boozmn_circular_tokamak.nc"
    assert output.exists()
    with Dataset(output) as ds:
        assert "bmnc_b" in ds.variables
        assert "jlist" in ds.variables
        assert int(ds.variables["nfp_b"][...].item()) == 1


def test_module_entrypoint_smoke(tmp_path: Path) -> None:
    _materialize_case(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "booz_xform_jax", "booz_in.circular_tokamak", "F"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (tmp_path / "boozmn_circular_tokamak.nc").exists()
