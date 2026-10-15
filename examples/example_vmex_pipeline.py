"""In-memory VMEX -> booz_xform_jax pipeline.

Solve a small fixed-boundary equilibrium with VMEX (``pip install vmex``),
hand the in-memory wout to ``Booz_xform`` without writing a file, run the
JAX-native transform, and plot the largest Boozer |B| harmonics versus s.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import vmex

from booz_xform_jax import Booz_xform

INPUT = Path(__file__).resolve().parents[1] / "tests" / "test_files" / "input.circular_tokamak"
SURFACES = [0.25, 0.5, 0.75]
MBOZ, NBOZ = 8, 0

inp = vmex.VmecInput.from_file(INPUT)
result = vmex.solve(inp, verbose=True)
wout = vmex.wout_from_result(inp, result)

bx = Booz_xform()
bx.read_wout_data(wout)
bx.mboz, bx.nboz = MBOZ, NBOZ
bx.register_surfaces(SURFACES)
out = bx.run_jax()

bmnc_b = np.asarray(out["bmnc_b"])
ixm, ixn = np.asarray(out["ixm_b"]), np.asarray(out["ixn_b"])
s_b = np.asarray(bx.s_in)[bx.compute_surfs]
for idx in np.argsort(-np.abs(bmnc_b[-1]))[:6]:
    print(f"m={ixm[idx]:2d} n={ixn[idx]:3d}  |bmnc_b| = {np.abs(bmnc_b[:, idx])}")
    plt.plot(s_b, np.abs(bmnc_b[:, idx]), "o-", label=f"m={ixm[idx]}, n={ixn[idx]}")
plt.xlabel("s (normalized toroidal flux)")
plt.ylabel("|B| Boozer harmonic amplitude")
plt.yscale("log")
plt.legend()
plt.title("Boozer |B| harmonics, VMEX -> booz_xform_jax")
plt.show()
