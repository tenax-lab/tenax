"""Run a small dense CTM single-device vs sharded and assert energy parity.

Invoked by tests/test_ctm_sharding.py under
XLA_FLAGS=--xla_force_host_platform_device_count=2. Exits 0 on parity (<1e-8).
"""

import os
import sys

# Pin to the CPU platform BEFORE importing jax so the fake-device count from
# ``--xla_force_host_platform_device_count`` applies (it only affects CPU).  On
# a GPU host without this, JAX picks the single real GPU and the 2-way mesh
# all-gather deadlocks.  Honour an inherited override; default to cpu.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ctm_sharding import build_ctm_mesh
from tenax.algorithms.ipeps import _heisenberg_dense_probe_energy  # see Step 4


def main() -> int:
    chi, D = 8, 2
    e_single = _heisenberg_dense_probe_energy(D=D, chi=chi, device_mesh=None, seed=0)
    mesh = build_ctm_mesh()  # fake devices from XLA_FLAGS
    e_sharded = _heisenberg_dense_probe_energy(D=D, chi=chi, device_mesh=mesh, seed=0)
    err = abs(float(e_single) - float(e_sharded))
    print(
        f"devices={jax.device_count()} e_single={e_single:.10f} "
        f"e_sharded={e_sharded:.10f} |delta|={err:.2e}"
    )
    return 0 if err < 1e-8 else 1


if __name__ == "__main__":
    sys.exit(main())
