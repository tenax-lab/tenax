"""HOTRG free energy under a GSPMD device mesh must equal the single-device
result. Invoked by tests/test_hotrg_sharding.py under
XLA_FLAGS=--xla_force_host_platform_device_count=N. Exits 0 on parity.

Usage: _hotrg_sharding_parity_subproc.py [beta] [chi] [steps] [thresh]
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ctm_sharding import build_ctm_mesh
from tenax.algorithms.hotrg import HOTRGConfig, hotrg
from tenax.algorithms.trg import compute_ising_tensor


def main() -> int:
    beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    chi = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    thresh = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-8

    T = compute_ising_tensor(beta)
    f_single = float(hotrg(T, HOTRGConfig(max_bond_dim=chi, num_steps=steps)))
    mesh = build_ctm_mesh()  # fake devices from XLA_FLAGS
    f_shard = float(
        hotrg(T, HOTRGConfig(max_bond_dim=chi, num_steps=steps, device_mesh=mesh))
    )
    err = abs(f_single - f_shard)
    print(
        f"devices={jax.device_count()} beta={beta} chi={chi} steps={steps} "
        f"f_single={f_single:.10f} f_shard={f_shard:.10f} |delta|={err:.2e}"
    )
    return 0 if err < thresh else 1


if __name__ == "__main__":
    sys.exit(main())
