"""Generate README comparison figures for booz_xform_jax vs original xbooz_xform.

The numbers published in the README come from this script. Every row is an
end-to-end measurement of the two command-line programs on the same legacy
``booz_in`` input: total wall-clock time and peak resident set size of the
process, taken as the best of several repeats.

Total wall-clock time is the number a user actually waits for, and for
``booz_xform_jax`` it includes a fixed Python and JAX import cost that the
compiled reference does not pay. That cost is measured separately and
reported alongside the table so the two contributions can be told apart; it
is not subtracted from any published figure.

The reference binary is located from ``--reference-bin``, the
``BOOZ_XFORM_REFERENCE_BIN`` environment variable, or ``xbooz_xform`` on
``PATH``. Two further cases are included when ``--vmec-jax-root`` or the
``VMEC_JAX_ROOT`` environment variable points at a tree that contains them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

REFERENCE_BIN_ENV = "BOOZ_XFORM_REFERENCE_BIN"
VMEC_JAX_ROOT_ENV = "VMEC_JAX_ROOT"


def _default_reference_bin() -> Path | None:
    """Locate the reference ``xbooz_xform`` without assuming any machine layout."""
    from_env = os.environ.get(REFERENCE_BIN_ENV)
    if from_env:
        return Path(from_env)
    on_path = shutil.which("xbooz_xform")
    return Path(on_path) if on_path else None


def _default_vmec_jax_root() -> Path | None:
    from_env = os.environ.get(VMEC_JAX_ROOT_ENV)
    return Path(from_env) if from_env else None

_C_REF = "#1f77b4"
_C_JAX = "#ff7f0e"
_DISPLAY = {
    "circular_tokamak": "circular tokamak",
    "LandremanSenguptaPlunk_section5p3": "LandremanSenguptaPlunk s5.3",
    "up_down_asymmetric_tokamak": "up/down asymmetric tokamak",
    "li383_1.4m": "li383 1.4m",
    "ITERModel": "ITERModel",
    "LandremanPaul2021_QA_lowres": "LandremanPaul2021 QA lowres",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    wout_path: Path
    mboz: int = 32
    nboz: int = 32


_MEASURE_LAUNCHER = textwrap.dedent(
    """
    import json, resource, subprocess, sys, time

    argv, cwd, env_overrides = json.loads(sys.argv[1])
    import os
    env = dict(os.environ)
    env.update(env_overrides)
    t0 = time.perf_counter()
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is bytes on macOS and kibibytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    sys.stdout.write(
        json.dumps(
            {
                "returncode": proc.returncode,
                "wall_s": wall,
                "peak_rss_bytes": int(usage.ru_maxrss) * scale,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
    )
    """
)


@dataclass(frozen=True)
class Measurement:
    """Best-of-N wall time and peak RSS for one command."""

    wall_s: float
    peak_rss_bytes: int
    stdout: str
    stderr: str


def _measure(
    argv: list[str],
    *,
    cwd: Path,
    python: Path,
    repeats: int,
    env_overrides: dict[str, str] | None = None,
) -> Measurement:
    """Run ``argv`` ``repeats`` times and keep the fastest run.

    Each repeat is launched from a freshly started helper process so that the
    helper's ``RUSAGE_CHILDREN`` high-water mark describes that run alone.
    """
    payload = json.dumps([argv, str(cwd), dict(env_overrides or {})])
    best: Measurement | None = None
    for _ in range(max(1, repeats)):
        proc = subprocess.run(
            [str(python), "-c", _MEASURE_LAUNCHER, payload],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(proc.stdout)
        if int(result["returncode"]) != 0:
            raise RuntimeError(
                f"{argv[0]} failed:\n{result['stderr'] or result['stdout']}"
            )
        candidate = Measurement(
            wall_s=float(result["wall_s"]),
            peak_rss_bytes=int(result["peak_rss_bytes"]),
            stdout=str(result["stdout"]),
            stderr=str(result["stderr"]),
        )
        if best is None or candidate.wall_s < best.wall_s:
            best = candidate
    assert best is not None
    return best


def _measure_import_overhead(*, python: Path, repeats: int) -> Measurement:
    """Cost of starting Python and importing the package, transform excluded."""
    return _measure(
        [str(python), "-c", "import booz_xform_jax"],
        cwd=ROOT,
        python=python,
        repeats=repeats,
        env_overrides=_pythonpath_env(),
    )


def _pythonpath_env() -> dict[str, str]:
    """Environment overrides that put this checkout's ``src`` first."""
    pythonpath = str(ROOT / "src")
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath = pythonpath + os.pathsep + existing
    return {"PYTHONPATH": pythonpath}


def _wout_ns(wout_path: Path) -> int:
    with netCDF4.Dataset(str(wout_path)) as ds:
        if "ns" in ds.variables:
            return int(np.asarray(ds.variables["ns"][...]).item())
        if "radius" in ds.dimensions:
            return int(ds.dimensions["radius"].size)
    raise ValueError(f"Could not infer ns from {wout_path}")


def _wout_ntor(wout_path: Path) -> int:
    with netCDF4.Dataset(str(wout_path)) as ds:
        if "ntor" in ds.variables:
            return int(np.asarray(ds.variables["ntor"][...]).item())
    raise ValueError(f"Could not infer ntor from {wout_path}")


def _effective_nboz(case: Case) -> int:
    """Clamp the requested nboz to what the equilibrium can carry.

    Asking for toroidal Boozer modes an axisymmetric equilibrium does not have
    makes the reference binary index past the end of its mode arrays and abort,
    so the two programs never get compared on those cases at all.
    """
    if _wout_ntor(case.wout_path) == 0:
        return 0
    return int(case.nboz)


def _default_surface_jlist(ns: int) -> list[int]:
    """Surfaces to transform, as ``jlist`` entries (compute_surfs + 2)."""
    candidates = [2, 10, 20, max(2, min(ns - 2, 40))]
    return sorted({v for v in candidates if 1 < v <= ns - 1})


@dataclass(frozen=True)
class ReferenceDialect:
    """How a particular reference binary wants to be driven.

    The C++ ``booz_xform`` and the Fortran STELLOPT ``booz_xform`` disagree
    about the legacy ``booz_in`` file: the former reads the surface numbers as
    0-based half-grid indices, the latter as ``jlist`` entries, which are the
    same surfaces plus two. ``booz_xform_jax`` follows STELLOPT. Feeding the
    same file to both therefore transforms *different surfaces*, and any
    spectrum comparison built on top of that is meaningless. Probe for the
    convention instead of assuming one.
    """

    extra_argv: tuple[str, ...]
    jlist_offset: int


def _probe_reference(ref_bin: Path, *, python: Path, workdir: Path) -> ReferenceDialect:
    probe_wout = ROOT / "tests" / "test_files" / "wout_circular_tokamak.nc"
    probe_values = [4, 8]
    for extra_argv in ((), ("F",)):
        case_dir = workdir / "_probe"
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(probe_wout, case_dir / "wout_circular_tokamak.nc")
        input_file = case_dir / "booz_in.circular_tokamak"
        input_file.write_text(
            "8 0\ncircular_tokamak\n" + " ".join(str(v) for v in probe_values) + "\n",
            encoding="utf-8",
        )
        try:
            _measure(
                [str(ref_bin), input_file.name, *extra_argv],
                cwd=case_dir,
                python=python,
                repeats=1,
            )
        except RuntimeError:
            continue
        out = case_dir / "boozmn_circular_tokamak.nc"
        if not out.exists():
            continue
        with netCDF4.Dataset(str(out)) as ds:
            jlist = np.asarray(ds.variables["jlist"][:], dtype=int)
        if jlist.size != len(probe_values):
            continue
        offset = int(jlist[0]) - probe_values[0]
        return ReferenceDialect(extra_argv=tuple(extra_argv), jlist_offset=offset)
    raise RuntimeError(
        f"Could not work out how to drive the reference binary {ref_bin}: it "
        "produced no boozmn output for either supported argument form."
    )


def _materialize_case(
    case: Case, *, workdir: Path, dialect: ReferenceDialect
) -> tuple[Path, Path, Path]:
    """Lay out one case with an input file per surface-numbering convention."""
    case_dir = workdir / case.case_id
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    ext = case.case_id
    shutil.copy2(case.wout_path, case_dir / f"wout_{ext}.nc")

    header = f"{int(case.mboz)} {_effective_nboz(case)}\n{ext}\n"
    jlist = _default_surface_jlist(_wout_ns(case.wout_path))

    in_booz_jax = case_dir / f"booz_in.{ext}"
    in_booz_jax.write_text(
        header + " ".join(str(v) for v in jlist) + "\n", encoding="utf-8"
    )

    if dialect.jlist_offset == 0:
        in_booz_ref = in_booz_jax
    else:
        in_booz_ref = case_dir / f"booz_in_reference.{ext}"
        in_booz_ref.write_text(
            header + " ".join(str(v - dialect.jlist_offset) for v in jlist) + "\n",
            encoding="utf-8",
        )
    return case_dir, in_booz_jax, in_booz_ref


def _run_reference(
    case: Case, *, case_dir: Path, input_file: Path, ref_bin: Path, python: Path,
    repeats: int, dialect: ReferenceDialect,
) -> tuple[Path, Measurement]:
    measurement = _measure(
        [str(ref_bin), input_file.name, *dialect.extra_argv],
        cwd=case_dir,
        python=python,
        repeats=repeats,
    )
    out = case_dir / f"boozmn_{case.case_id}.nc"
    if not out.exists():
        raise FileNotFoundError(f"Missing reference output {out}")
    ref_copy = case_dir / f"boozmn_{case.case_id}_reference.nc"
    shutil.copy2(out, ref_copy)
    return ref_copy, measurement


def _run_jax(
    case: Case, *, case_dir: Path, input_file: Path, python: Path, repeats: int
) -> tuple[Path, Measurement]:
    measurement = _measure(
        [str(python), "-m", "booz_xform_jax", input_file.name, "F"],
        cwd=case_dir,
        python=python,
        repeats=repeats,
        env_overrides=_pythonpath_env(),
    )
    out = case_dir / f"boozmn_{case.case_id}.nc"
    if not out.exists():
        raise FileNotFoundError(f"Missing JAX output {out}")
    jax_copy = case_dir / f"boozmn_{case.case_id}_jax.nc"
    shutil.copy2(out, jax_copy)
    return jax_copy, measurement


def _read_boozmn(path: Path) -> dict[str, np.ndarray | int | float | bool]:
    with netCDF4.Dataset(str(path)) as ds:
        jlist = np.asarray(ds.variables["jlist"][:], dtype=int)
        bmnc = np.asarray(ds.variables["bmnc_b"][:])
        if bmnc.ndim == 2 and bmnc.shape[0] == jlist.size:
            bmnc = bmnc.T
        out: dict[str, np.ndarray | int | float | bool] = {
            "lasym": bool(np.asarray(ds.variables["lasym__logical__"][...]).item()) if "lasym__logical__" in ds.variables else False,
            "nfp": int(np.asarray(ds.variables["nfp_b"][...]).item()) if "nfp_b" in ds.variables else 1,
            "ns_b": int(jlist.size),
            "bmnc_b": bmnc,
            "ixm_b": np.asarray(ds.variables["ixm_b"][:], dtype=int),
            "ixn_b": np.asarray(ds.variables["ixn_b"][:], dtype=int),
            "iota_b": np.asarray(ds.variables["iota_b"][:]),
            "jlist": jlist,
        }
        if "bmns_b" in ds.variables:
            bmns = np.asarray(ds.variables["bmns_b"][:])
            if bmns.ndim == 2 and bmns.shape[0] == jlist.size:
                bmns = bmns.T
            out["bmns_b"] = bmns
    return out


def _surface_modb(data: dict[str, np.ndarray | int | float | bool], *, js: int, ntheta: int = 100, nphi: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi / int(data["nfp"]), nphi)
    phi_g, theta_g = np.meshgrid(phi, theta)
    modb = np.zeros_like(theta_g)
    bmnc = np.asarray(data["bmnc_b"])
    bmns = np.asarray(data.get("bmns_b", np.zeros_like(bmnc)))
    for jmn, (m, n) in enumerate(zip(np.asarray(data["ixm_b"]), np.asarray(data["ixn_b"]), strict=False)):
        angle = int(m) * theta_g - int(n) * phi_g
        modb += bmnc[jmn, js] * np.cos(angle) + bmns[jmn, js] * np.sin(angle)
    return theta_g, phi_g, modb


def _normalized_radius(data: dict[str, np.ndarray | int | float | bool]) -> np.ndarray:
    iota = np.asarray(data["iota_b"], dtype=float)
    if iota.size <= 1:
        return np.zeros((0,), dtype=float)
    idx = np.arange(1, iota.size, dtype=float)
    denom = max(float(iota.size - 1), 1.0)
    return np.sqrt(np.maximum(idx / denom, 0.0))


def _selected_surface_radius(data: dict[str, np.ndarray | int | float | bool]) -> np.ndarray:
    jlist = np.asarray(data["jlist"], dtype=float)
    if jlist.size == 0:
        return np.zeros((0,), dtype=float)
    denom = max(float(np.asarray(data["iota_b"]).size - 1), 1.0)
    return np.sqrt(np.maximum((jlist - 1.0) / denom, 0.0))


def _assert_comparable(case: Case, ref_data, jax_data) -> None:
    """Refuse to publish a comparison of two different calculations."""
    for key in ("jlist", "ixm_b", "ixn_b"):
        ref = np.asarray(ref_data[key], dtype=int)
        jax = np.asarray(jax_data[key], dtype=int)
        if ref.shape != jax.shape or not np.array_equal(ref, jax):
            raise RuntimeError(
                f"{case.case_id}: the reference and booz_xform_jax did not compute "
                f"the same {key} ({ref.tolist()} vs {jax.tolist()}), so their "
                "spectra are not comparable."
            )


def _profile_metrics(ref_data, jax_data) -> dict[str, float]:
    bmnc_ref = np.asarray(ref_data["bmnc_b"])
    bmnc_jax = np.asarray(jax_data["bmnc_b"])
    iota_ref = np.asarray(ref_data["iota_b"])[1:]
    iota_jax = np.asarray(jax_data["iota_b"])[1:]
    b00_ref = bmnc_ref[0]
    b00_jax = bmnc_jax[0]
    return {
        "iota_rel_l2": float(np.linalg.norm(iota_jax - iota_ref) / max(np.linalg.norm(iota_ref), 1e-30)),
        "b00_rel_l2": float(np.linalg.norm(b00_jax - b00_ref) / max(np.linalg.norm(b00_ref), 1e-30)),
        "bmnc_rel_l2": float(np.linalg.norm(bmnc_jax - bmnc_ref) / max(np.linalg.norm(bmnc_ref), 1e-30)),
    }


def _plot_profiles(*, ref_data, jax_data, title: str, outpath: Path) -> None:
    rho_ref = _normalized_radius(ref_data)
    rho_jax = _normalized_radius(jax_data)
    rho_sel_ref = _selected_surface_radius(ref_data)
    rho_sel_jax = _selected_surface_radius(jax_data)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    axes[0].plot(rho_ref, np.asarray(ref_data["iota_b"])[1:], color=_C_REF, lw=2.2, label="xbooz_xform")
    axes[0].plot(rho_jax, np.asarray(jax_data["iota_b"])[1:], color=_C_JAX, lw=2.0, ls="--", label="booz_xform_jax")
    axes[0].set_xlabel(r"$\sqrt{s}$")
    axes[0].set_ylabel(r"$\iota_B$")
    axes[0].set_title(f"{_DISPLAY.get(title, title)}: iota_b")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(rho_sel_ref, np.asarray(ref_data["bmnc_b"])[0], color=_C_REF, lw=2.2, label="xbooz_xform")
    axes[1].plot(rho_sel_jax, np.asarray(jax_data["bmnc_b"])[0], color=_C_JAX, lw=2.0, ls="--", label="booz_xform_jax")
    axes[1].set_xlabel(r"$\sqrt{s}$")
    axes[1].set_ylabel(r"$B_{00}$")
    axes[1].set_title(f"{_DISPLAY.get(title, title)}: Boozer B00")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_surface_compare(*, ref_data, jax_data, title: str, outpath: Path) -> None:
    js = int(ref_data["ns_b"]) - 1
    theta_ref, phi_ref, modb_ref = _surface_modb(ref_data, js=js)
    _, _, modb_jax = _surface_modb(jax_data, js=min(js, int(jax_data["ns_b"]) - 1))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    vmin = min(float(np.min(modb_ref)), float(np.min(modb_jax)))
    vmax = max(float(np.max(modb_ref)), float(np.max(modb_jax)))
    display = _DISPLAY.get(title, title)
    for ax, data, name in zip(
        axes[:2],
        [modb_ref, modb_jax],
        ["xbooz_xform", "booz_xform_jax"],
        strict=False,
    ):
        im = ax.contourf(phi_ref, theta_ref, data, levels=24, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"{display}: {name}")
        ax.set_xlabel(r"$\varphi_B$")
        ax.set_ylabel(r"$\theta_B$")
    diff = modb_jax - modb_ref
    im = axes[2].contourf(phi_ref, theta_ref, diff, levels=24, cmap="coolwarm")
    axes[2].set_title(f"{display}: JAX - ref")
    axes[2].set_xlabel(r"$\varphi_B$")
    axes[2].set_ylabel(r"$\theta_B$")
    fig.colorbar(im, ax=axes, shrink=0.92)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _paired_bars(
    rows: list[dict[str, object]],
    *,
    ref_key: str,
    jax_key: str,
    scale: float,
    xlabel: str,
    title: str,
    subtitle: str,
    outpath: Path,
) -> None:
    ordered = sorted(
        rows, key=lambda row: float(row[jax_key]) / max(float(row[ref_key]), 1e-12)
    )
    labels = [_DISPLAY.get(str(row["case_id"]), str(row["case_id"])) for row in ordered]
    ref = np.asarray([float(row[ref_key]) / scale for row in ordered], dtype=float)
    jax = np.asarray([float(row[jax_key]) / scale for row in ordered], dtype=float)
    y = np.arange(len(labels), dtype=float)
    height = 0.34
    fig, ax = plt.subplots(figsize=(11.5, max(4.6, 0.62 * len(labels) + 2.2)))
    ax.barh(y - height / 2.0, ref, height=height, color=_C_REF, label="xbooz_xform")
    ax.barh(y + height / 2.0, jax, height=height, color=_C_JAX, label="booz_xform_jax")
    for value, position in zip(np.concatenate([ref, jax]),
                              np.concatenate([y - height / 2.0, y + height / 2.0]),
                              strict=False):
        ax.annotate(f"{value:.3g}", (value, position), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    # Headroom so the value labels do not run off the right edge.
    ax.set_xlim(right=float(max(ref.max(), jax.max())) * 3.0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=26, fontsize=13)
    ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="0.35")
    ax.grid(axis="x", alpha=0.2, which="both")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_runtimes(rows: list[dict[str, object]], *, outpath: Path, subtitle: str) -> None:
    _paired_bars(
        rows,
        ref_key="reference_wall_s",
        jax_key="jax_wall_s",
        scale=1.0,
        xlabel="total wall-clock time (seconds, log scale; lower is better)",
        title="Boozer transform: end-to-end runtime",
        subtitle=subtitle,
        outpath=outpath,
    )


def _plot_memory(rows: list[dict[str, object]], *, outpath: Path, subtitle: str) -> None:
    _paired_bars(
        rows,
        ref_key="reference_peak_rss_bytes",
        jax_key="jax_peak_rss_bytes",
        scale=1024.0 * 1024.0,
        xlabel="peak resident set size (MiB, log scale; lower is better)",
        title="Boozer transform: peak process memory",
        subtitle=subtitle,
        outpath=outpath,
    )


def _default_cases(vmec_jax_root: Path | None) -> list[Case]:
    cases = [
        Case("circular_tokamak", ROOT / "tests" / "test_files" / "wout_circular_tokamak.nc", mboz=24, nboz=24),
        Case("LandremanSenguptaPlunk_section5p3", ROOT / "tests" / "test_files" / "wout_LandremanSenguptaPlunk_section5p3.nc", mboz=24, nboz=24),
        Case("up_down_asymmetric_tokamak", ROOT / "tests" / "test_files" / "wout_up_down_asymmetric_tokamak.nc", mboz=24, nboz=24),
        Case("li383_1.4m", ROOT / "tests" / "test_files" / "wout_li383_1.4m.nc", mboz=24, nboz=24),
    ]
    if vmec_jax_root is None:
        return cases
    base = vmec_jax_root / "outputs" / "readme_fsq_trace_single_grid_work"
    iter_wout = base / "ITERModel" / "wout_ITERModel_VMEC2000.nc"
    qa_wout = base / "LandremanPaul2021_QA_lowres" / "wout_LandremanPaul2021_QA_lowres_VMEC2000.nc"
    if iter_wout.exists():
        cases.append(Case("ITERModel", iter_wout, mboz=16, nboz=16))
    if qa_wout.exists():
        cases.append(Case("LandremanPaul2021_QA_lowres", qa_wout, mboz=16, nboz=16))
    return cases


def _markdown_table(rows: list[dict[str, object]]) -> str:
    """Render the rows exactly as the README publishes them."""
    lines = [
        "| Case | ns | xbooz_xform | booz_xform_jax | Speedup | Peak RSS ref | Peak RSS jax |",
        "|---|---|---|---|---|---|---|",
    ]
    mib = 1024.0 * 1024.0
    for row in rows:
        ref = float(row["reference_wall_s"])
        jax = float(row["jax_wall_s"])
        lines.append(
            "| {case} | {ns} | {ref:.2f} s | {jax:.2f} s | {speedup:.2f}x | {rss_ref:.0f} MiB | {rss_jax:.0f} MiB |".format(
                case=_DISPLAY.get(str(row["case_id"]), str(row["case_id"])),
                ns=int(row["ns"]),
                ref=ref,
                jax=jax,
                speedup=ref / max(jax, 1e-12),
                rss_ref=float(row["reference_peak_rss_bytes"]) / mib,
                rss_jax=float(row["jax_peak_rss_bytes"]) / mib,
            )
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference-bin",
        type=Path,
        default=_default_reference_bin(),
        help=(
            "Reference xbooz_xform binary. Defaults to the "
            f"{REFERENCE_BIN_ENV} environment variable, then to xbooz_xform on PATH."
        ),
    )
    p.add_argument(
        "--vmec-jax-root",
        type=Path,
        default=_default_vmec_jax_root(),
        help=(
            "Optional tree holding the two extra VMEC cases. Defaults to the "
            f"{VMEC_JAX_ROOT_ENV} environment variable; the cases are skipped when unset."
        ),
    )
    p.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Interpreter used to run booz_xform_jax (defaults to the current one).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repeats per program; the fastest run is published.",
    )
    p.add_argument("--workdir", type=Path, default=ROOT / "outputs" / "readme_compare")
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs",
        help="Where the README figures are written (the README points at docs/).",
    )
    p.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "README_assets",
        help="Where the metrics JSON and the optional profile figures are written.",
    )
    args = p.parse_args()

    if args.reference_bin is None:
        raise SystemExit(
            "No reference xbooz_xform found. Pass --reference-bin, set "
            f"{REFERENCE_BIN_ENV}, or put xbooz_xform on PATH."
        )
    ref_bin = args.reference_bin.expanduser().resolve()
    if not ref_bin.exists():
        raise FileNotFoundError(f"Reference xbooz_xform not found: {ref_bin}")
    # Deliberately not resolve()d: a virtualenv interpreter is a symlink to the
    # base installation, and following it loses the environment's packages.
    python = args.python.expanduser()
    workdir = args.workdir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    asset_dir = args.asset_dir.expanduser().resolve()
    vmec_jax_root = args.vmec_jax_root.expanduser().resolve() if args.vmec_jax_root else None
    for directory in (workdir, outdir, asset_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dialect = _probe_reference(ref_bin, python=python, workdir=workdir)
    print(
        f"Reference dialect: argv {list(dialect.extra_argv)!r}, "
        f"booz_in surface numbers are jlist minus {dialect.jlist_offset}"
    )

    overhead = _measure_import_overhead(python=python, repeats=args.repeats)
    print(f"booz_xform_jax import overhead: {overhead.wall_s:.2f} s")

    rows: list[dict[str, object]] = []
    selected_profiles: list[tuple[str, dict[str, object], dict[str, object]]] = []

    for case in _default_cases(vmec_jax_root):
        case_dir, input_jax, input_ref = _materialize_case(
            case, workdir=workdir, dialect=dialect
        )
        ref_path, ref_measurement = _run_reference(
            case, case_dir=case_dir, input_file=input_ref, ref_bin=ref_bin,
            python=python, repeats=args.repeats, dialect=dialect,
        )
        jax_path, jax_measurement = _run_jax(
            case, case_dir=case_dir, input_file=input_jax,
            python=python, repeats=args.repeats,
        )
        ref_data = _read_boozmn(ref_path)
        jax_data = _read_boozmn(jax_path)
        _assert_comparable(case, ref_data, jax_data)
        metrics = _profile_metrics(ref_data, jax_data)
        rows.append(
            {
                "case_id": case.case_id,
                "ns": _wout_ns(case.wout_path),
                "mboz": int(case.mboz),
                "nboz": _effective_nboz(case),
                "reference_wall_s": ref_measurement.wall_s,
                "jax_wall_s": jax_measurement.wall_s,
                "reference_peak_rss_bytes": ref_measurement.peak_rss_bytes,
                "jax_peak_rss_bytes": jax_measurement.peak_rss_bytes,
                **metrics,
            }
        )
        print(
            f"{case.case_id}: reference {ref_measurement.wall_s:.2f} s, "
            f"booz_xform_jax {jax_measurement.wall_s:.2f} s, "
            f"bmnc relative L2 {metrics['bmnc_rel_l2']:.2e}"
        )
        if case.case_id in {"ITERModel", "LandremanPaul2021_QA_lowres"}:
            selected_profiles.append((case.case_id, ref_data, jax_data))

    platform = f"{os.uname().sysname} {os.uname().machine}"
    _plot_runtimes(
        rows,
        outpath=outdir / "comparison_runtime.png",
        subtitle=(
            f"best of {args.repeats} runs on {platform}; booz_xform_jax includes "
            f"{overhead.wall_s:.1f} s of fixed Python and JAX import cost on every invocation"
        ),
    )
    _plot_memory(
        rows,
        outpath=outdir / "comparison_memory.png",
        subtitle=(
            f"best of {args.repeats} runs on {platform}; peak resident set size of the "
            "whole process, including the JAX runtime"
        ),
    )
    for case_id, ref_data, jax_data in selected_profiles:
        short = "iter" if case_id == "ITERModel" else "qa"
        _plot_profiles(ref_data=ref_data, jax_data=jax_data, title=case_id, outpath=asset_dir / f"{short}_profiles_compare.png")
        _plot_surface_compare(ref_data=ref_data, jax_data=jax_data, title=case_id, outpath=asset_dir / f"{short}_surface_compare.png")

    payload = {
        "platform": f"{os.uname().sysname} {os.uname().release} {os.uname().machine}",
        "repeats": int(args.repeats),
        # Name only: an absolute path would leak the layout of whichever
        # machine happened to produce the published numbers.
        "reference_bin": ref_bin.name,
        "jax_import_overhead_s": overhead.wall_s,
        "note": (
            "Wall-clock times are whole-process measurements of each command-line "
            "program, best of the recorded repeats. booz_xform_jax pays the import "
            "overhead above on every invocation; it is not subtracted."
        ),
        "cases": rows,
    }
    (asset_dir / "readme_compare_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print()
    print(_markdown_table(rows))
    print()
    print(f"Wrote {outdir / 'comparison_runtime.png'}")
    print(f"Wrote {outdir / 'comparison_memory.png'}")
    for case_id, *_ in selected_profiles:
        short = "iter" if case_id == "ITERModel" else "qa"
        print(f"Wrote {asset_dir / f'{short}_profiles_compare.png'}")
        print(f"Wrote {asset_dir / f'{short}_surface_compare.png'}")
    print(f"Wrote {asset_dir / 'readme_compare_metrics.json'}")


if __name__ == "__main__":
    main()
