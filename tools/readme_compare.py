"""Generate README comparison figures for booz_xform_jax vs original xbooz_xform."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_BIN = Path("/Users/rogeriojorge/local/booz_xform/xbooz_xform")
DEFAULT_VMEC_JAX_ROOT = Path("/Users/rogeriojorge/local/test/vmec_jax")

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


def _pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath
    return env


def _wout_ns(wout_path: Path) -> int:
    with netCDF4.Dataset(str(wout_path)) as ds:
        if "ns" in ds.variables:
            return int(np.asarray(ds.variables["ns"][...]).item())
        if "radius" in ds.dimensions:
            return int(ds.dimensions["radius"].size)
    raise ValueError(f"Could not infer ns from {wout_path}")


def _default_surface_line(ns: int) -> str:
    candidates = [2, 10, 20, max(2, min(ns - 2, 40))]
    vals = sorted({v for v in candidates if 1 < v <= ns - 1})
    return " ".join(str(v) for v in vals)


def _materialize_case(case: Case, *, workdir: Path) -> tuple[Path, Path]:
    case_dir = workdir / case.case_id
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    ext = case.case_id
    wout_name = f"wout_{ext}.nc"
    shutil.copy2(case.wout_path, case_dir / wout_name)
    in_booz = case_dir / f"booz_in.{ext}"
    in_booz.write_text(
        f"{int(case.mboz)} {int(case.nboz)}\n{ext}\n{_default_surface_line(_wout_ns(case.wout_path))}\n",
        encoding="utf-8",
    )
    return case_dir, in_booz


def _run_reference(case: Case, *, case_dir: Path, input_file: Path, ref_bin: Path) -> tuple[Path, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(ref_bin), input_file.name],
        cwd=str(case_dir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    runtime = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"xbooz_xform failed for {case.case_id}:\n{proc.stderr or proc.stdout}")
    out = case_dir / f"boozmn_{case.case_id}.nc"
    if not out.exists():
        raise FileNotFoundError(f"Missing reference output {out}")
    ref_copy = case_dir / f"boozmn_{case.case_id}_reference.nc"
    shutil.copy2(out, ref_copy)
    return ref_copy, runtime


def _run_jax(case: Case, *, case_dir: Path, input_file: Path) -> tuple[Path, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["python", "-m", "booz_xform_jax.cli", input_file.name, "F"],
        cwd=str(case_dir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_pythonpath_env(),
    )
    runtime = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"booz_xform_jax failed for {case.case_id}:\n{proc.stderr or proc.stdout}")
    out = case_dir / f"boozmn_{case.case_id}.nc"
    if not out.exists():
        raise FileNotFoundError(f"Missing JAX output {out}")
    jax_copy = case_dir / f"boozmn_{case.case_id}_jax.nc"
    shutil.copy2(out, jax_copy)
    return jax_copy, runtime


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


def _plot_runtimes(rows: list[dict[str, object]], *, outpath: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["jax_runtime_s"]) / max(float(row["reference_runtime_s"]), 1e-12))
    labels = [_DISPLAY.get(str(row["case_id"]), str(row["case_id"])) for row in ordered]
    ref = np.asarray([float(row["reference_runtime_s"]) for row in ordered], dtype=float)
    jax = np.asarray([float(row["jax_runtime_s"]) for row in ordered], dtype=float)
    y = np.arange(len(labels), dtype=float)
    height = 0.34
    fig, ax = plt.subplots(figsize=(11.5, max(4.2, 0.5 * len(labels) + 1.0)))
    ax.barh(y - height / 2.0, ref, height=height, color=_C_REF, label="xbooz_xform")
    ax.barh(y + height / 2.0, jax, height=height, color=_C_JAX, label="booz_xform_jax")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("runtime (seconds, log scale)")
    ax.set_title("Boozer transform runtime")
    ax.grid(axis="x", alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _default_cases(vmec_jax_root: Path) -> list[Case]:
    cases = [
        Case("circular_tokamak", ROOT / "tests" / "test_files" / "wout_circular_tokamak.nc", mboz=24, nboz=24),
        Case("LandremanSenguptaPlunk_section5p3", ROOT / "tests" / "test_files" / "wout_LandremanSenguptaPlunk_section5p3.nc", mboz=24, nboz=24),
        Case("up_down_asymmetric_tokamak", ROOT / "tests" / "test_files" / "wout_up_down_asymmetric_tokamak.nc", mboz=24, nboz=24),
        Case("li383_1.4m", ROOT / "tests" / "test_files" / "wout_li383_1.4m.nc", mboz=24, nboz=24),
    ]
    iter_wout = vmec_jax_root / "outputs" / "readme_fsq_trace_single_grid_work" / "ITERModel" / "wout_ITERModel_VMEC2000.nc"
    qa_wout = vmec_jax_root / "outputs" / "readme_fsq_trace_single_grid_work" / "LandremanPaul2021_QA_lowres" / "wout_LandremanPaul2021_QA_lowres_VMEC2000.nc"
    if iter_wout.exists():
        cases.append(Case("ITERModel", iter_wout, mboz=16, nboz=16))
    if qa_wout.exists():
        cases.append(Case("LandremanPaul2021_QA_lowres", qa_wout, mboz=16, nboz=16))
    return cases


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-bin", type=Path, default=DEFAULT_REF_BIN)
    p.add_argument("--vmec-jax-root", type=Path, default=DEFAULT_VMEC_JAX_ROOT)
    p.add_argument("--workdir", type=Path, default=ROOT / "outputs" / "readme_compare")
    p.add_argument("--outdir", type=Path, default=ROOT / "README_assets")
    args = p.parse_args()

    ref_bin = args.reference_bin.expanduser().resolve()
    if not ref_bin.exists():
        raise FileNotFoundError(f"Reference xbooz_xform not found: {ref_bin}")
    workdir = args.workdir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    selected_profiles: list[tuple[str, dict[str, object], dict[str, object]]] = []

    for case in _default_cases(args.vmec_jax_root.expanduser().resolve()):
        case_dir, input_file = _materialize_case(case, workdir=workdir)
        ref_path, ref_runtime = _run_reference(case, case_dir=case_dir, input_file=input_file, ref_bin=ref_bin)
        jax_path, jax_runtime = _run_jax(case, case_dir=case_dir, input_file=input_file)
        ref_data = _read_boozmn(ref_path)
        jax_data = _read_boozmn(jax_path)
        metrics = _profile_metrics(ref_data, jax_data)
        rows.append(
            {
                "case_id": case.case_id,
                "reference_runtime_s": ref_runtime,
                "jax_runtime_s": jax_runtime,
                **metrics,
            }
        )
        if case.case_id in {"ITERModel", "LandremanPaul2021_QA_lowres"}:
            selected_profiles.append((case.case_id, ref_data, jax_data))

    _plot_runtimes(rows, outpath=outdir / "readme_runtime_compare.png")
    for case_id, ref_data, jax_data in selected_profiles:
        short = "iter" if case_id == "ITERModel" else "qa"
        _plot_profiles(ref_data=ref_data, jax_data=jax_data, title=case_id, outpath=outdir / f"{short}_profiles_compare.png")
        _plot_surface_compare(ref_data=ref_data, jax_data=jax_data, title=case_id, outpath=outdir / f"{short}_surface_compare.png")

    (outdir / "readme_compare_metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {outdir / 'readme_runtime_compare.png'}")
    for case_id, *_ in selected_profiles:
        short = "iter" if case_id == "ITERModel" else "qa"
        print(f"Wrote {outdir / f'{short}_profiles_compare.png'}")
        print(f"Wrote {outdir / f'{short}_surface_compare.png'}")
    print(f"Wrote {outdir / 'readme_compare_metrics.json'}")


if __name__ == "__main__":
    main()
