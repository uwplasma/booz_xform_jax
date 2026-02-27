#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_vmec_jax_pipeline.py
============================

End-to-end example:

  1) run a lightweight VMEC solve with vmec_jax,
  2) build an in-memory wout object,
  3) run booz_xform_jax with the JAX-native backend,
  4) plot a few |B| Fourier modes vs radius.

This example is intentionally small (max_iter=1) to keep runtime modest.
For higher-fidelity results, increase max_iter and resolution.
"""

from __future__ import annotations

import argparse

import numpy as np

from booz_xform_jax import Booz_xform


def _parse_surfaces(text: str) -> list[float]:
    return [float(val.strip()) for val in text.split(",") if val.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vmec_jax -> booz_xform_jax pipeline and plot Boozer modes."
    )
    parser.add_argument(
        "--case",
        type=str,
        default="circular_tokamak",
        help="VMEC example case name (vmec_jax/examples/data/input.<case>).",
    )
    parser.add_argument(
        "--surfaces",
        type=str,
        default="0.25,0.5,0.75",
        help="Comma-separated s values in [0,1] to transform (default: 0.25,0.5,0.75).",
    )
    parser.add_argument("--mboz", type=int, default=8, help="Max poloidal mode for Boozer.")
    parser.add_argument("--nboz", type=int, default=8, help="Max toroidal mode for Boozer.")
    parser.add_argument("--ntheta", type=int, default=16, help="VMEC theta grid size.")
    parser.add_argument("--nzeta", type=int, default=1, help="VMEC zeta grid size.")
    parser.add_argument("--max-iter", type=int, default=1, help="VMEC solver iterations.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot.")
    args = parser.parse_args()

    try:
        import vmec_jax as vj
        from vmec_jax.driver import example_paths, run_fixed_boundary, wout_from_fixed_boundary_run
        from vmec_jax.vmec_tomnsp import vmec_angle_grid
    except ImportError as exc:
        raise ImportError(
            "vmec_jax is required for this example. Install vmec_jax or add it to PYTHONPATH."
        ) from exc

    input_path, _ = example_paths(args.case)
    if not input_path.exists():
        raise FileNotFoundError(f"VMEC input not found: {input_path}")

    cfg, _ = vj.load_input(str(input_path))
    grid = vmec_angle_grid(
        ntheta=int(args.ntheta),
        nzeta=int(args.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )

    run = run_fixed_boundary(
        input_path,
        max_iter=int(args.max_iter),
        use_initial_guess=True,
        vmec_project=False,
        verbose=True,
        grid=grid,
    )

    wout = wout_from_fixed_boundary_run(run, include_fsq=False, fast_bcovar=True)

    bx = Booz_xform()
    bx.read_wout_data(wout)
    bx.mboz = int(args.mboz)
    bx.nboz = int(args.nboz)
    bx.register_surfaces(_parse_surfaces(args.surfaces))

    out = bx.run_jax()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting in this example.") from exc

    bmnc_b = np.asarray(out["bmnc_b"])
    ixm = np.asarray(out["ixm_b"])
    ixn = np.asarray(out["ixn_b"])
    s_b = np.asarray(bx.s_in)[bx.compute_surfs]

    # Plot the largest modes by amplitude at the outermost surface.
    sort_idx = np.argsort(-np.abs(bmnc_b[-1]))
    nmodes = min(6, bmnc_b.shape[1])
    fig, ax = plt.subplots()
    for idx in sort_idx[:nmodes]:
        ax.plot(
            s_b,
            np.abs(bmnc_b[:, idx]),
            label=f"m={int(ixm[idx])}, n={int(ixn[idx])}",
        )
    ax.set_xlabel("s (toroidal flux)")
    ax.set_ylabel("|B| Fourier amplitude")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Boozer |B| harmonics from vmec_jax -> booz_xform_jax")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
