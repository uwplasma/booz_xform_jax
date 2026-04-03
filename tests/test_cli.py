"""CLI parity tests against the legacy STELLOPT ``xbooz_xform`` executable."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests" / "test_files"
REFERENCE_BIN = Path(
    os.environ.get("BOOZ_XFORM_REFERENCE_BIN", str(Path.home() / "bin" / "xbooz_xform"))
).expanduser()


def _pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath
    return env


def _run_jax_cli(tmp_path: Path, input_name: str, *, screen_flag: str = "F") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "booz_xform_jax", input_name, screen_flag],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=_pythonpath_env(),
    )


def _run_reference_cli(tmp_path: Path, input_name: str, *, screen_flag: str = "F") -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [str(REFERENCE_BIN), input_name, screen_flag],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return proc
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if "Usage:  xbooz_xform <inputfile>" not in (stdout + stderr):
        return proc
    return subprocess.run(
        [str(REFERENCE_BIN), input_name],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )


def _compare_boozmn_files(reference_file: Path, jax_file: Path) -> None:
    with Dataset(reference_file) as ref_ds, Dataset(jax_file) as jax_ds:
        for name in ["mboz_b", "nboz_b", "jlist"]:
            np.testing.assert_array_equal(np.asarray(ref_ds.variables[name][:]), np.asarray(jax_ds.variables[name][:]))

        for name in ["bmnc_b", "rmnc_b", "zmns_b", "pmns_b", "gmn_b"]:
            np.testing.assert_allclose(
                np.asarray(ref_ds.variables[name][:]),
                np.asarray(jax_ds.variables[name][:]),
                rtol=1e-12,
                atol=1e-12,
            )

        if bool(ref_ds.variables["lasym__logical__"][...].item()):
            for name in ["bmns_b", "rmns_b", "zmnc_b", "pmnc_b", "gmns_b"]:
                np.testing.assert_allclose(
                    np.asarray(ref_ds.variables[name][:]),
                    np.asarray(jax_ds.variables[name][:]),
                    rtol=1e-12,
                    atol=1e-12,
                )


def _materialize_case(
    tmp_path: Path,
    *,
    input_name: str,
    input_source: Path | None = None,
    input_contents: str | None = None,
    wout_source: Path,
) -> None:
    if input_source is not None:
        shutil.copy(input_source, tmp_path / input_name)
    elif input_contents is not None:
        (tmp_path / input_name).write_text(input_contents, encoding="utf-8")
    else:
        raise ValueError("Either input_source or input_contents must be provided.")

    if not wout_source.exists():
        pytest.skip(f"Missing reference wout file: {wout_source}")
    shutil.copy(wout_source, tmp_path / wout_source.name)


def _assert_cli_parity(tmp_path: Path, *, input_name: str, output_name: str, expect_missing_jlist: bool = False) -> None:
    ref_proc = _run_reference_cli(tmp_path, input_name, screen_flag="F")
    if ref_proc.returncode != 0:
        pytest.skip(f"Reference xbooz_xform is not usable for this case:\n{ref_proc.stderr or ref_proc.stdout}")

    ref_output = tmp_path / output_name
    if not ref_output.exists():
        pytest.skip("Reference xbooz_xform did not produce a boozmn file for this case.")
    ref_copy = tmp_path / f"reference_{output_name}"
    ref_output.rename(ref_copy)

    jax_proc = _run_jax_cli(tmp_path, input_name, screen_flag="F")
    assert jax_proc.returncode == 0, jax_proc.stderr or jax_proc.stdout

    jax_output = tmp_path / output_name
    assert jax_output.exists()

    if expect_missing_jlist:
        assert "No jlist data was found in Boozer input file." in ref_proc.stdout
        assert "No jlist data was found in Boozer input file." in jax_proc.stdout
    else:
        assert jax_proc.stdout.strip() == ""

    _compare_boozmn_files(ref_copy, jax_output)


def test_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "booz_xform_jax", "-h"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_pythonpath_env(),
    )
    assert proc.returncode == 0
    assert "<infile> (T or F)" in proc.stdout


pytestmark = pytest.mark.skipif(
    not REFERENCE_BIN.exists(),
    reason=f"Reference xbooz_xform binary not found at {REFERENCE_BIN}",
)


@pytest.mark.parametrize(
    ("input_name", "wout_name", "output_name", "expect_missing_jlist"),
    [
        ("booz_in.li383_1.4m", "wout_li383_1.4m.nc", "boozmn_li383_1.4m.nc", False),
        (
            "booz_in.LandremanSenguptaPlunk_section5p3",
            "wout_LandremanSenguptaPlunk_section5p3.nc",
            "boozmn_LandremanSenguptaPlunk_section5p3.nc",
            False,
        ),
        (
            "booz_in.up_down_asymmetric_tokamak",
            "wout_up_down_asymmetric_tokamak.nc",
            "boozmn_up_down_asymmetric_tokamak.nc",
            True,
        ),
    ],
)
def test_cli_matches_reference_for_bundled_cases(
    tmp_path: Path, input_name: str, wout_name: str, output_name: str, expect_missing_jlist: bool
) -> None:
    _materialize_case(
        tmp_path,
        input_name=input_name,
        input_source=TEST_DIR / input_name,
        wout_source=TEST_DIR / wout_name,
    )
    _assert_cli_parity(
        tmp_path,
        input_name=input_name,
        output_name=output_name,
        expect_missing_jlist=expect_missing_jlist,
    )


def test_cli_missing_jlist_defaults_to_all_surfaces(tmp_path: Path) -> None:
    _materialize_case(
        tmp_path,
        input_name="booz_in.circular_tokamak",
        input_source=TEST_DIR / "booz_in.circular_tokamak",
        wout_source=TEST_DIR / "wout_circular_tokamak.nc",
    )
    _assert_cli_parity(
        tmp_path,
        input_name="booz_in.circular_tokamak",
        output_name="boozmn_circular_tokamak.nc",
        expect_missing_jlist=True,
    )


@pytest.mark.parametrize(
    ("input_name", "input_contents", "wout_source", "output_name"),
    [
        (
            "in_booz.n3are_lowres",
            "16 16\nn3are_R7.75B5.7_lowres\n2 10 20\n",
            Path("/Users/rogerio/local/simsopt/tests/test_files/wout_n3are_R7.75B5.7_lowres.nc"),
            "boozmn_n3are_R7.75B5.7_lowres.nc",
        ),
        (
            "in_booz.qa_lowres",
            "16 16\n'LandremanPaul2021_QA_lowres'\n2 10 20\n",
            Path("/Users/rogerio/local/simsopt/tests/test_files/wout_LandremanPaul2021_QA_lowres.nc"),
            "boozmn_LandremanPaul2021_QA_lowres.nc",
        ),
    ],
)
def test_cli_matches_reference_for_external_generated_inputs(
    tmp_path: Path, input_name: str, input_contents: str, wout_source: Path, output_name: str
) -> None:
    _materialize_case(
        tmp_path,
        input_name=input_name,
        input_contents=input_contents,
        wout_source=wout_source,
    )
    _assert_cli_parity(tmp_path, input_name=input_name, output_name=output_name)
