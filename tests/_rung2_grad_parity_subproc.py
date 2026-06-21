"""Rung-2 gate 2A: implicit-AD gradient parity, single-device vs GSPMD-sharded.

Invoked under XLA_FLAGS=--xla_force_host_platform_device_count=N (fake CPU
devices). Computes value_and_grad of the dense CTM-AD energy single vs sharded on
a WELL-CONDITIONED state and asserts the gradients agree to a tight threshold
(sharding is exact up to FP reassociation, which stays at machine precision on a
well-conditioned fixed point). Exits 0 on parity, nonzero otherwise.

Usage: _rung2_grad_parity_subproc.py [D] [chi] [thresh]
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _rung2_grad_probe import implicit_energy_and_grad  # noqa: E402

from tenax.algorithms.ctm_sharding import build_ctm_mesh  # noqa: E402


def main() -> int:
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    chi = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    kw = dict(D=D, chi=chi, seed=0, well_conditioned=True, max_iter=50)
    e1, g1 = implicit_energy_and_grad(device_mesh=None, **kw)
    mesh = build_ctm_mesh()  # fake devices from XLA_FLAGS
    e4, g4 = implicit_energy_and_grad(device_mesh=mesh, **kw)
    de = abs(e1 - e4)
    dg = float(np.max(np.abs(g1 - g4)))
    print(
        f"devices={jax.device_count()} D={D} chi={chi} thresh={thresh:.0e} "
        f"|dE|={de:.2e} grad_max|delta|={dg:.2e} gmax={np.max(np.abs(g1)):.3e}"
    )
    return 0 if (de < thresh and dg < thresh) else 1


if __name__ == "__main__":
    sys.exit(main())
