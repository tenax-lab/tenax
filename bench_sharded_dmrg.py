"""Benchmark: single-GPU JIT vs multi-GPU sharded DMRG."""

import time
import warnings

import jax

from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.algorithms.dmrg import DMRGConfig, dmrg
from tenax.core.mps import FiniteMPS

warnings.filterwarnings("ignore")

print(f"JAX backend: {jax.default_backend()}")
devices = jax.devices()
for d in devices:
    print(f"  {d}")
print()


def _build_dense_heisenberg(L):
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=False)


results = []

for L in [20, 40]:
    for chi in [1000, 1500, 2000]:
        num_sweeps = 2
        mpo = _build_dense_heisenberg(L)

        for mode in ["jit", "sharded"]:
            # Warmup run (JIT compilation)
            mps_w = FiniteMPS.random(L, d=2, chi=chi, key=jax.random.PRNGKey(0))
            _ = dmrg(
                mpo,
                mps_w,
                DMRGConfig(max_bond_dim=chi, num_sweeps=1, accelerator=mode),
            )

            # Timed run
            mps = FiniteMPS.random(L, d=2, chi=chi, key=jax.random.PRNGKey(42))
            config = DMRGConfig(
                max_bond_dim=chi,
                num_sweeps=num_sweeps,
                lanczos_max_iter=20,
                convergence_tol=0.0,
                accelerator=mode,
            )
            t0 = time.perf_counter()
            result = dmrg(mpo, mps, config)
            t_elapsed = time.perf_counter() - t0

            results.append((L, chi, mode, t_elapsed, result.energy))
            print(
                f"L={L:3d}  chi={chi:5d}  {mode:>8s}  "
                f"time={t_elapsed:8.2f}s  E={result.energy:.6f}"
            )
        print()

print("--- Summary ---")
print(f"{'L':>4s}  {'chi':>5s}  {'Mode':>8s}  {'Time':>8s}")
for L, chi, mode, t, _ in results:
    print(f"{L:4d}  {chi:5d}  {mode:>8s}  {t:8.2f}s")
