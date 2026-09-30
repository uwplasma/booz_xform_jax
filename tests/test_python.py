#!/usr/bin/env python3

"""Basic API tests for the JAX implementation of Booz_xform.

This test file mirrors ``tests/test_python.py`` from the original
``booz_xform`` repository.  It verifies that simple attribute
assignments on the Booz_xform object behave as expected.  The test
does not perform any numerical transformation.
"""

import os
import numpy as np
import unittest

from booz_xform_jax import Booz_xform

# Alias for backwards compatibility with the original test code
Booz_xform = Booz_xform

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_files')


class MainTest(unittest.TestCase):
    def test_compute_surfs_edit(self) -> None:
        """Ensure that the compute_surfs property can be set and retrieved."""
        b = Booz_xform()
        b.read_wout(os.path.join(TEST_DIR, 'wout_li383_1.4m.nc'))
        # assign two surfaces and check they are stored unchanged
        b.compute_surfs = [10, 15]
        np.testing.assert_allclose(b.compute_surfs, [10, 15])


if __name__ == '__main__':
    unittest.main()


def test_host_transform_uses_numpy(monkeypatch):
    """Host execution must not compile JAX kernels after initialization."""
    import booz_xform_jax.core as core

    b = Booz_xform(mboz=6, nboz=2, verbose=0)
    b.read_wout(os.path.join(TEST_DIR, 'wout_li383_1.4m.nc'))
    b.compute_surfs = [10]
    monkeypatch.setattr(core, 'jnp', None)
    b.run()
    assert isinstance(b._theta_grid, np.ndarray)
    assert np.isfinite(b.bmnc_b).all()
