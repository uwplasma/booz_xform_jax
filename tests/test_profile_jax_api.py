"""Tests for the matrix profiler in ``tools/profile_jax_api.py``."""

import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "tools" / "profile_jax_api.py"
_SPEC = importlib.util.spec_from_file_location("profile_jax_api", _PATH)
profile_jax_api = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile_jax_api
_SPEC.loader.exec_module(profile_jax_api)


def test_tiny_matrix_produces_schema_complete_records(tmp_path):
    out = tmp_path / "matrix.json"
    code = profile_jax_api.main([
        "--resolutions", "4", "--surfaces", "1", "--cases", "li383_sym",
        "--out", str(out),
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == 1
    assert payload["repo"] == "uwplasma/booz_xform_jax"
    (record,) = payload["records"]
    assert record["case"] == "li383_sym"
    assert record["surfaces"] == 1
    assert not record["asym"]
    # value, JVP, and VJP each compile exactly one program on first call.
    assert record["compiles"] == {
        "value_first": 1, "jvp_first": 1, "vjp_first": 1,
    }
    for key in ("value", "jvp", "vjp"):
        assert record["timing_s"][f"{key}_first"] > 0.0
        assert record["timing_s"][f"{key}_warm"] > 0.0
    # Analytic buffer model matches jax_api grid formulas at mboz=nboz=4.
    buffers = record["buffers"]
    assert buffers["mn_boz"] == (2 * 4 + 1) * 4 - 4
    assert buffers["ntheta"] == 2 * (2 * 4 + 1)
    assert buffers["nzeta"] == 2 * (2 * 4 + 1)
    assert buffers["input_phase_tensor"] == (
        buffers["ntheta"] * buffers["nzeta"] * buffers["mn_nyq"] * 8
    )
    assert record["spectrum_shape"] == [1, buffers["mn_boz"]]
    assert record["memory_bytes"]["peak_host_rss"] > 0


def test_surface_specs_collapsing_to_same_count_are_deduplicated(tmp_path):
    # updown_lasym has ns=16, so "20" and "all" both resolve to 16 surfaces;
    # the matrix must record that cell once, not twice.
    out = tmp_path / "matrix.json"
    code = profile_jax_api.main([
        "--resolutions", "4", "--surfaces", "20", "all",
        "--cases", "updown_lasym", "--out", str(out),
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    (record,) = payload["records"]
    assert record["surfaces"] == record["ns"] == 16
    assert record["asym"]
