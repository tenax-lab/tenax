#!/usr/bin/env python
"""A/B benchmark: GMRES backward with vs without diagonal scaling preconditioner.

Matches YASTN benchmark parameters: D=2, chi=16, Heisenberg.
Reports per-step wall time (forward+backward) for both configurations.

Usage:
    JAX_PLATFORM_NAME=cpu python benchmarks/bench_gmres_precond.py
    JAX_PLATFORM_NAME=cpu python benchmarks/bench_gmres_precond.py --steps 30
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad

jax.config.update("jax_enable_x64", True)


def _heisenberg_gate(dtype=jnp.float64):
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    return H.reshape(2, 2, 2, 2)


def run(precond: bool, num_steps: int = 20, log_interval: int = 5):
    gate = _heisenberg_gate()
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=16, max_iter=50, conv_tol=1e-8, gmres_precondition=precond),
        gs_learning_rate=1e-3,
        gs_num_steps=num_steps,
        gs_verbose=True,
        gs_log_interval=log_interval,
    )
    t0 = time.perf_counter()
    A_opt, env, E = optimize_gs_ad(gate, None, config)
    dt = time.perf_counter() - t0
    return E, dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    N = args.steps

    print("=" * 60)
    print("GMRES Diagonal Preconditioner A/B Benchmark")
    print(f"Heisenberg 2D, D=2, chi=16, {N} AD steps")
    print("=" * 60)

    # Warm-up JIT
    print("\nWarm-up (2 steps)...")
    run(precond=True, num_steps=2, log_interval=1)

    print("\n--- Preconditioner ON (diagonal scaling) ---")
    E_on, dt_on = run(precond=True, num_steps=N)
    print(f"  Energy: {E_on:.8f}  Time: {dt_on:.1f}s  ({dt_on / N:.2f}s/step)")

    print("\n--- Preconditioner OFF ---")
    E_off, dt_off = run(precond=False, num_steps=N)
    print(f"  Energy: {E_off:.8f}  Time: {dt_off:.1f}s  ({dt_off / N:.2f}s/step)")

    print("\n" + "=" * 60)
    speedup = dt_off / dt_on if dt_on > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x  (ON={dt_on:.1f}s vs OFF={dt_off:.1f}s)")
    print(f"Energy:  ON={E_on:.8f}  OFF={E_off:.8f}")
    print("Ref (QMC): -0.66944")
    print("=" * 60)
