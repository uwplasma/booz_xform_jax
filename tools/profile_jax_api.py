"""Matrix profiler for the JAX Boozer transform.

Measures ``booz_xform_jax_impl`` across a baseline matrix: symmetric and
LASYM equilibria, 1/5/20/all surfaces, three Boozer resolutions, and
value / JVP / VJP, separating first-call (trace + compile + execute) from
warm execution and recording compile counts, peak host RSS, and an analytic
buffer model for the dominant phase tensors.  One JSON record per cell; no
number here is edited by hand.

Usage::

    python tools/profile_jax_api.py --out profiles/matrix.json
    python tools/profile_jax_api.py --resolutions 8 16 --surfaces 1 5
    python tools/profile_jax_api.py --resolutions 8 --surfaces 5 \\
        --cases li383_sym --trace-dir profiles/trace

``--trace-dir`` additionally captures a TensorBoard trace of one warm value
call per profiled cell; narrow the matrix when tracing.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "li383_sym": ROOT / "tests" / "test_files" / "wout_li383_1.4m.nc",
    "updown_lasym": ROOT / "tests" / "test_files"
    / "wout_up_down_asymmetric_tokamak.nc",
}


class _CompileCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.compiles = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Compiling "):
            self.compiles += 1


def _install_counter() -> _CompileCounter:
    jax.config.update("jax_log_compiles", True)
    counter = _CompileCounter()
    logger = logging.getLogger("jax")
    logger.addHandler(counter)
    logger.setLevel(logging.WARNING)
    return counter


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) * (1024 if sys.platform.startswith("linux") else 1)


def _load_case(path: Path):
    from booz_xform_jax import Booz_xform

    bx = Booz_xform()
    bx.read_wout(str(path))
    return bx


def _inputs(bx, mboz: int, nboz: int):
    from booz_xform_jax.jax_api import prepare_booz_xform_constants

    constants, grids = prepare_booz_xform_constants(
        nfp=bx.nfp, mboz=mboz, nboz=nboz, asym=bool(bx.asym),
        xm=bx.xm, xn=bx.xn, xm_nyq=bx.xm_nyq, xn_nyq=bx.xn_nyq,
    )
    arrays = dict(
        rmnc=jnp.asarray(bx.rmnc).T, zmns=jnp.asarray(bx.zmns).T,
        lmns=jnp.asarray(bx.lmns).T, bmnc=jnp.asarray(bx.bmnc).T,
        bsubumnc=jnp.asarray(bx.bsubumnc).T,
        bsubvmnc=jnp.asarray(bx.bsubvmnc).T,
        iota=jnp.asarray(bx.iota),
    )
    if bool(bx.asym):
        arrays.update(
            rmns=jnp.asarray(bx.rmns).T, zmnc=jnp.asarray(bx.zmnc).T,
            lmnc=jnp.asarray(bx.lmnc).T, bmns=jnp.asarray(bx.bmns).T,
            bsubumns=jnp.asarray(bx.bsubumns).T,
            bsubvmns=jnp.asarray(bx.bsubvmns).T,
        )
    static = dict(
        xm=jnp.asarray(bx.xm), xn=jnp.asarray(bx.xn),
        xm_nyq=jnp.asarray(bx.xm_nyq), xn_nyq=jnp.asarray(bx.xn_nyq),
        constants=constants, grids=grids,
    )
    return arrays, static


def _buffer_model(bx, mboz: int, nboz: int, n_surfaces: int) -> dict:
    """Analytic sizes of the dominant tensors, in bytes at float64.

    ``input_phase_tables`` are the shared per-resolution synthesis tables
    (cos and sin, non-Nyquist and Nyquist); ``projection_buffers`` mirrors
    the per-surface model in ``jax_api._resolve_surface_chunk`` (Boozer
    trig tables plus separable intermediates), without its headroom factor.
    """
    mn_non = int(np.asarray(bx.xm).size)
    mn_nyq = int(np.asarray(bx.xm_nyq).size)
    mn_boz = int((2 * nboz + 1) * mboz - nboz)
    # jax_api._prepare_grids: ntheta = 2*(2*mboz+1), nzeta = 2*(2*nboz+1)
    ntheta = 2 * (2 * mboz + 1)
    nzeta = 2 * (2 * nboz + 1)
    points = ntheta * nzeta
    itemsize = 8
    return {
        "mn_non": mn_non, "mn_nyq": mn_nyq, "mn_boz": mn_boz,
        "ntheta": ntheta, "nzeta": nzeta,
        "input_phase_tables": 2 * points * (mn_non + mn_nyq) * itemsize,
        "projection_buffers": n_surfaces * points * (
            2 * (mboz + nboz + 2) + 10 * (nboz + 1)) * itemsize,
        "returned_spectra": n_surfaces * mn_boz * itemsize,
    }


def _time_call(fn, *args) -> tuple[float, object]:
    started = time.perf_counter()
    value = jax.block_until_ready(fn(*args))
    return time.perf_counter() - started, value


def profile_cell(bx, case: str, mboz: int, nboz: int,
                 n_surfaces: int | None, counter: _CompileCounter,
                 trace_dir: Path | None = None) -> dict:
    from booz_xform_jax.jax_api import booz_xform_jax_impl

    arrays, static = _inputs(bx, mboz, nboz)
    ns = int(arrays["rmnc"].shape[0])
    count = ns if n_surfaces is None else min(n_surfaces, ns)
    indices = jnp.linspace(0, ns - 1, count).astype(int)

    def value_fn(bmnc):
        out = booz_xform_jax_impl(
            **{**arrays, "bmnc": bmnc}, **static, surface_indices=indices)
        return out["bmnc_b"]

    jitted = jax.jit(value_fn)
    record: dict = {
        "case": case, "mboz": mboz, "nboz": nboz,
        "surfaces": count, "ns": ns, "asym": bool(bx.asym),
        "buffers": _buffer_model(bx, mboz, nboz, count),
        "timing_s": {}, "compiles": {},
    }

    counter.compiles = 0
    record["timing_s"]["value_first"], out = _time_call(jitted, arrays["bmnc"])
    record["compiles"]["value_first"] = counter.compiles
    record["spectrum_shape"] = list(np.shape(out))
    warm = [
        _time_call(jitted, arrays["bmnc"])[0] for _ in range(3)
    ]
    record["timing_s"]["value_warm"] = sorted(warm)[1]

    if trace_dir is not None:
        cell_dir = trace_dir / f"{case}_m{mboz}_s{record['surfaces']}"
        jax.profiler.start_trace(str(cell_dir))
        jax.block_until_ready(jitted(arrays["bmnc"]))
        jax.profiler.stop_trace()

    tangent = jnp.ones_like(arrays["bmnc"])

    def jvp_fn(bmnc):
        return jax.jvp(value_fn, (bmnc,), (tangent,))[1]

    jvp_jitted = jax.jit(jvp_fn)
    counter.compiles = 0
    record["timing_s"]["jvp_first"], _ = _time_call(jvp_jitted, arrays["bmnc"])
    record["compiles"]["jvp_first"] = counter.compiles
    record["timing_s"]["jvp_warm"] = _time_call(jvp_jitted, arrays["bmnc"])[0]

    def vjp_fn(bmnc):
        value, pullback = jax.vjp(value_fn, bmnc)
        return pullback(jnp.ones_like(value))[0]

    vjp_jitted = jax.jit(vjp_fn)
    counter.compiles = 0
    record["timing_s"]["vjp_first"], _ = _time_call(vjp_jitted, arrays["bmnc"])
    record["compiles"]["vjp_first"] = counter.compiles
    record["timing_s"]["vjp_warm"] = _time_call(vjp_jitted, arrays["bmnc"])[0]
    # ru_maxrss is the process-lifetime high-water mark, so this value is
    # monotone across records; per-cell memory attribution comes from the
    # analytic ``buffers`` model, not from this field.
    record["memory_bytes"] = {"peak_host_rss": _peak_rss_bytes()}
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=[8, 16, 32])
    parser.add_argument("--surfaces", nargs="+", default=["1", "5", "20", "all"])
    parser.add_argument("--cases", nargs="+", default=list(CASES),
                        choices=list(CASES))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    counter = _install_counter()
    records = []
    for case in args.cases:
        bx = _load_case(CASES[case])
        seen: set[tuple] = set()
        for m in args.resolutions:
            for spec in args.surfaces:
                count = None if spec == "all" else int(spec)
                ns = int(np.asarray(bx.rmnc).shape[1])
                cell = (m, ns if count is None else min(count, ns))
                if cell in seen:
                    # "20" and "all" collapse to the same count on small ns;
                    # a duplicated cell would double-count the matrix.
                    continue
                seen.add(cell)
                record = profile_cell(bx, case, m, m, count, counter,
                                      trace_dir=args.trace_dir)
                records.append(record)
                print(f"[{case} mboz={m} surf={record['surfaces']}] "
                      f"{ {k: round(v, 4) for k, v in record['timing_s'].items()} }",
                      file=sys.stderr)

    def _git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "unknown"

    payload = {
        "schema": 1,
        "repo": "uwplasma/booz_xform_jax",
        "commit": _git("rev-parse", "HEAD"),
        # Untracked files are excluded: the baseline JSON itself is
        # written into the tree, which would mark every run dirty.
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "jax": {"jax": jax.__version__, "backend": jax.default_backend(),
                "x64": bool(jax.config.jax_enable_x64)},
        "records": records,
    }
    text = json.dumps(payload, indent=1, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
