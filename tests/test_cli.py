"""CLI parity tests against the legacy STELLOPT ``xbooz_xform`` executable.

The reference here is specifically the **Fortran STELLOPT** ``xbooz_xform``.
The C++ ``booz_xform`` is not a drop-in substitute: it reads the surface
numbers in a ``booz_in`` file as 0-based ``compute_surfs`` while STELLOPT, and
``booz_xform_jax``, read them as ``jlist`` entries, which are the same surfaces
plus two. Both programs accept the same file without complaint and transform
different surfaces, so these tests probe the binary's convention and refuse to
run against the wrong one rather than reporting a spurious difference.

Point ``BOOZ_XFORM_REFERENCE_BIN`` at a STELLOPT build to run this suite. The
external-equilibrium cases additionally need ``BOOZ_XFORM_EXTRA_WOUT_DIR`` to
name a directory holding the wout files they use.
"""

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

REFERENCE_BIN_ENV = "BOOZ_XFORM_REFERENCE_BIN"
EXTRA_WOUT_DIR_ENV = "BOOZ_XFORM_EXTRA_WOUT_DIR"


def _discover_reference_bin() -> Path | None:
    """Find the reference executable without assuming any machine layout."""
    from_env = os.environ.get(REFERENCE_BIN_ENV)
    if from_env:
        return Path(from_env).expanduser()
    on_path = shutil.which("xbooz_xform")
    return Path(on_path) if on_path else None


def _extra_wout(name: str) -> Path | None:
    """Locate an equilibrium that is not bundled with this repository."""
    directory = os.environ.get(EXTRA_WOUT_DIR_ENV)
    return Path(directory).expanduser() / name if directory else None


REFERENCE_BIN = _discover_reference_bin()


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
    # The screen-output flag is a STELLOPT extension. The C++ booz_xform takes
    # the input file alone and answers a second argument by printing its usage
    # text and exiting *successfully*, so a returncode check alone never
    # noticed and the binary was never actually driven. Retry whenever the
    # output looks like usage, whatever the exit status.
    if "Usage:" not in ((proc.stdout or "") + (proc.stderr or "")):
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
        pytest.skip(
            f"Missing wout file {wout_source.name} in {wout_source.parent} "
            f"(see {EXTRA_WOUT_DIR_ENV})"
        )
    shutil.copy(wout_source, tmp_path / wout_source.name)


def _assert_cli_parity(tmp_path: Path, *, input_name: str, output_name: str, expect_missing_jlist: bool = False) -> None:
    ref_proc = _run_reference_cli(tmp_path, input_name, screen_flag="F")
    combined = (ref_proc.stdout or "") + (ref_proc.stderr or "")
    if ref_proc.returncode != 0 and "Usage:" in combined:
        pytest.skip(f"Reference xbooz_xform rejected this input file:\n{combined}")
    # Anything else is the reference failing on input it accepted, which is a
    # result worth seeing rather than a reason to declare the test skipped.
    assert ref_proc.returncode == 0, combined

    ref_output = tmp_path / output_name
    assert ref_output.exists(), (
        f"Reference xbooz_xform exited cleanly but wrote no {output_name}:\n{combined}"
    )
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


def _reference_unavailable_reason() -> str | None:
    """Explain precisely why the reference cannot be used, or return None."""
    if REFERENCE_BIN is None:
        return (
            f"No reference xbooz_xform found: set {REFERENCE_BIN_ENV} to a STELLOPT "
            "build, or put xbooz_xform on PATH."
        )
    if not REFERENCE_BIN.exists():
        return f"Reference xbooz_xform does not exist: {REFERENCE_BIN}"
    if _reference_is_this_package():
        return (
            f"The xbooz_xform at {REFERENCE_BIN} is booz_xform_jax's own console "
            "script: this package installs xbooz_xform as an alias of its CLI, so "
            "anything found on PATH after a plain install is this package itself, "
            "and comparing against it would compare the package with itself. Set "
            f"{REFERENCE_BIN_ENV} to a STELLOPT build."
        )
    dialect, output = _reference_jlist_offset()
    if dialect is None:
        detail = " ".join(output.split())[:200]
        return (
            f"Reference xbooz_xform at {REFERENCE_BIN} produced no boozmn output for "
            f"the probe case, so its behaviour cannot be established: {detail!r}"
        )
    if dialect != 0:
        return (
            f"Reference xbooz_xform at {REFERENCE_BIN} reads booz_in surface "
            f"numbers as jlist minus {dialect}, which is the C++ booz_xform "
            f"convention, not STELLOPT's. Set {REFERENCE_BIN_ENV} to a STELLOPT build "
            "to run the CLI parity suite."
        )
    return None


def _reference_is_this_package() -> bool:
    """True when the discovered binary is this package's own CLI.

    ``pyproject.toml`` installs ``xbooz_xform`` as an alias of
    ``booz_xform_jax.cli:main``, so after a plain ``pip install`` the name this
    suite looks for on PATH resolves to the code under test. Comparing against
    it would pass unconditionally and prove nothing.
    """
    proc = subprocess.run(
        [str(REFERENCE_BIN), "-h"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "booz_xform_jax" in ((proc.stdout or "") + (proc.stderr or ""))


def _reference_jlist_offset() -> tuple[int | None, str]:
    """Probe the reference for how it numbers surfaces in a ``booz_in`` file.

    Returns ``jlist - <input number>`` together with whatever the binary
    printed: 0 for the STELLOPT convention that ``booz_xform_jax`` implements,
    2 for the C++ ``booz_xform`` convention, and ``None`` when the binary
    produced nothing to measure.
    """
    import tempfile

    probe_values = [4, 8]
    with tempfile.TemporaryDirectory() as raw:
        probe_dir = Path(raw)
        shutil.copy(TEST_DIR / "wout_circular_tokamak.nc", probe_dir / "wout_circular_tokamak.nc")
        (probe_dir / "booz_in.circular_tokamak").write_text(
            "8 0\ncircular_tokamak\n" + " ".join(str(v) for v in probe_values) + "\n",
            encoding="utf-8",
        )
        proc = _run_reference_cli(probe_dir, "booz_in.circular_tokamak", screen_flag="F")
        output = (proc.stdout or "") + (proc.stderr or "")
        out = probe_dir / "boozmn_circular_tokamak.nc"
        if not out.exists():
            return None, output
        with Dataset(out) as ds:
            jlist = np.asarray(ds.variables["jlist"][:], dtype=int)
        if jlist.size != len(probe_values):
            return None, output
        return int(jlist[0]) - probe_values[0], output


_REFERENCE_SKIP_REASON = _reference_unavailable_reason()

# Applied per test rather than as a module-level ``pytestmark``: the tests that
# exercise only this package's own CLI do not need a reference binary and must
# keep running when none is available.
requires_reference = pytest.mark.skipif(
    _REFERENCE_SKIP_REASON is not None,
    reason=_REFERENCE_SKIP_REASON or "",
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
@requires_reference
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


@requires_reference
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
    ("input_name", "input_contents", "wout_name", "output_name"),
    [
        (
            "in_booz.n3are_lowres",
            "16 16\nn3are_R7.75B5.7_lowres\n2 10 20\n",
            "wout_n3are_R7.75B5.7_lowres.nc",
            "boozmn_n3are_R7.75B5.7_lowres.nc",
        ),
        (
            "in_booz.qa_lowres",
            "16 16\n'LandremanPaul2021_QA_lowres'\n2 10 20\n",
            "wout_LandremanPaul2021_QA_lowres.nc",
            "boozmn_LandremanPaul2021_QA_lowres.nc",
        ),
    ],
)
@requires_reference
def test_cli_matches_reference_for_external_generated_inputs(
    tmp_path: Path, input_name: str, input_contents: str, wout_name: str, output_name: str
) -> None:
    wout_source = _extra_wout(wout_name)
    if wout_source is None:
        pytest.skip(
            f"{wout_name} is not bundled with this repository; set "
            f"{EXTRA_WOUT_DIR_ENV} to a directory that contains it to run this case."
        )
    _materialize_case(
        tmp_path,
        input_name=input_name,
        input_contents=input_contents,
        wout_source=wout_source,
    )
    _assert_cli_parity(tmp_path, input_name=input_name, output_name=output_name)
