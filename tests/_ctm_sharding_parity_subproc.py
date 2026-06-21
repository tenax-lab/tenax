"""Run a small dense CTM single-device vs sharded and assert energy parity.

Invoked by tests/test_ctm_sharding.py under
XLA_FLAGS=--xla_force_host_platform_device_count=N. Exits 0 on parity.

Usage: _ctm_sharding_parity_subproc.py [D] [chi] [well_conditioned] [thresh]
  well_conditioned: 0 (random, default) or 1 (near-product, low-κ fixed point)
  thresh:           parity threshold (default 1e-8)
The caller picks (D, chi) so that D² divides the device count cleanly. Sharding
reassociates the D²-axis contraction sum, so on a random (ill-conditioned) state
at larger χ the sharded vs single energies differ by O(eps·κ) (~1e-4); a
well-conditioned state keeps that at machine precision — pass well_conditioned=1
with a tight thresh to exercise the physically-relevant regime.
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

from _ctm_sharding_probe import heisenberg_dense_probe_energy

from tenax.algorithms.ctm_sharding import build_ctm_mesh


def main() -> int:
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    chi = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    well_conditioned = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    thresh = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-8
    kw = dict(D=D, chi=chi, seed=0, well_conditioned=well_conditioned)
    e_single = heisenberg_dense_probe_energy(device_mesh=None, **kw)
    mesh = build_ctm_mesh()  # fake devices from XLA_FLAGS
    e_sharded = heisenberg_dense_probe_energy(device_mesh=mesh, **kw)
    err = abs(float(e_single) - float(e_sharded))
    print(
        f"devices={jax.device_count()} D={D} chi={chi} "
        f"well_cond={int(well_conditioned)} thresh={thresh:.0e} "
        f"e_single={e_single:.10f} e_sharded={e_sharded:.10f} |delta|={err:.2e}"
    )
    return 0 if err < thresh else 1


if __name__ == "__main__":
    sys.exit(main())
