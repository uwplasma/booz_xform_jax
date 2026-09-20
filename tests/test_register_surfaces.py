#!/usr/bin/env python3

"""Regression tests for surface selection (issue #3).

``register_surfaces`` adds to the current surface selection. Reading a
VMEC file used to pre-fill that selection with *every* half-grid
surface, so any registration made afterwards was a silent no-op: the
transform still ran, and was still written, on all of the input
surfaces. These tests pin the selection down in both the in-memory
result and the written boozmn file.
"""

import os

import netCDF4
import numpy as np
import pytest

from booz_xform_jax import Booz_xform

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_files')
WOUT = os.path.join(TEST_DIR, 'wout_li383_1.4m.nc')


def _loaded() -> Booz_xform:
    b = Booz_xform()
    b.verbose = 0
    b.mboz = 10
    b.nboz = 5
    b.read_wout(WOUT, flux=True)
    return b


def test_read_wout_leaves_the_surface_selection_at_its_default() -> None:
    """``compute_surfs is None`` is the documented "all surfaces" default."""
    b = _loaded()
    assert b.compute_surfs is None
    assert int(b.ns_in) > 1


def test_register_surfaces_narrows_run_and_written_file(tmp_path) -> None:
    """The sequence reported in issue #3 must select exactly one surface."""
    b = _loaded()
    ns_in = int(b.ns_in)
    assert ns_in > 10, "fixture must have more surfaces than we select"

    b.register_surfaces(10)
    assert b.compute_surfs == [10]

    b.run()

    # In-memory result.
    assert b.ns_b == 1
    assert np.asarray(b.bmnc_b).shape[1] == 1
    assert np.asarray(b.s_b).shape == (1,)
    np.testing.assert_allclose(b.s_b, np.asarray(b.s_in)[[10]])

    # Written file.
    out = tmp_path / "boozmn_register_surfaces.nc"
    b.write_boozmn(str(out))
    with netCDF4.Dataset(str(out)) as ds:
        assert len(ds.dimensions['comput_surfs']) == 1
        assert len(ds.dimensions['pack_rad']) == 1
        np.testing.assert_array_equal(np.asarray(ds.variables['jlist'][:]), [12])
        assert np.asarray(ds.variables['bmnc_b'][:]).shape[0] == 1


def test_register_surfaces_appends_to_an_existing_selection() -> None:
    """Later registrations add to, and do not replace, earlier ones."""
    b = _loaded()
    b.register_surfaces(10)
    b.register_surfaces([3, 10, 7])
    assert b.compute_surfs == [3, 7, 10]


def test_register_surfaces_accepts_numpy_integers() -> None:
    """NumPy integers are indices, not normalised flux values."""
    b = _loaded()
    b.register_surfaces(np.int64(10))
    b.register_surfaces(np.arange(2, 4))
    assert b.compute_surfs == [2, 3, 10]


def test_register_surfaces_accepts_normalised_flux_values() -> None:
    """Floats select the nearest half-grid surface, as documented."""
    b = _loaded()
    targets = np.linspace(0.0, 1.0, 4)
    b.register_surfaces(targets)
    expected = sorted({int(np.argmin(np.abs(np.asarray(b.s_in) - s))) for s in targets})
    assert b.compute_surfs == expected
    assert len(b.compute_surfs) < int(b.ns_in)


def test_register_surfaces_before_reading_vmec_data_raises() -> None:
    """The wrong call order fails loudly instead of misbehaving later."""
    b = Booz_xform()
    with pytest.raises(RuntimeError, match="VMEC data"):
        b.register_surfaces(10)


def test_register_surfaces_rejects_out_of_range_indices() -> None:
    b = _loaded()
    with pytest.raises(ValueError):
        b.register_surfaces(int(b.ns_in))


def test_default_selection_still_covers_every_surface(tmp_path) -> None:
    """Without any registration the transform still runs on all surfaces."""
    b = _loaded()
    b.mboz = 4
    b.nboz = 2
    b.run()
    ns_in = int(b.ns_in)
    assert b.compute_surfs == list(range(ns_in))
    assert b.ns_b == ns_in

    out = tmp_path / "boozmn_all_surfaces.nc"
    b.write_boozmn(str(out))
    with netCDF4.Dataset(str(out)) as ds:
        assert len(ds.dimensions['comput_surfs']) == ns_in


def test_changing_boozer_resolution_between_runs_is_honoured() -> None:
    """A second run must not reuse mode lists and grids built for an old mboz."""
    b = _loaded()
    b.compute_surfs = [5]

    b.mboz, b.nboz = 6, 3
    b.run()
    first_mnboz = int(b.mnboz)
    assert np.asarray(b.bmnc_b).shape[0] == first_mnboz

    b.mboz, b.nboz = 12, 6
    b.run()
    second_mnboz = int(b.mnboz)
    assert second_mnboz == (6 + 1) + (12 - 1) * (2 * 6 + 1)
    assert second_mnboz != first_mnboz
    assert np.asarray(b.bmnc_b).shape[0] == second_mnboz


def test_changing_boozer_resolution_between_run_jax_calls_is_honoured() -> None:
    b = _loaded()
    b.compute_surfs = [5]

    b.mboz, b.nboz = 6, 3
    first = b.run_jax(jit=False)

    b.mboz, b.nboz = 12, 6
    second = b.run_jax(jit=False)

    assert np.asarray(first["bmnc_b"]).shape != np.asarray(second["bmnc_b"]).shape
    assert int(np.asarray(b.xm_b).max()) == 11
