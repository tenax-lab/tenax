"""Benchmark: compilation time for Python-loop CTM AD."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def bench_compile_implicit(D=2, chi=8):
    """Benchmark compile + run time for ctm_energy_implicit value_and_grad."""
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (D, D, D, D, 2), dtype=jnp.float64)
    data = data / jnp.linalg.norm(data)
    A = _wrap_as_dense_tensor(data)
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS

    def loss(p):
        A_local = _wrap_as_dense_tensor(p)
        site_tensors = {(0, 0): A_local}
        return ctm_energy_implicit(
            site_tensors,
            neighbors,
            gate,
            chi=chi,
            max_iter=30,
            conv_tol=1e-8,
            gmres_tol=1e-6,
        )

    # Compile + run
    t0 = time.perf_counter()
    e, g = jax.value_and_grad(loss)(A.todense())
    jax.block_until_ready(g)
    t_compile = time.perf_counter() - t0

    # Second call (no recompile)
    t0 = time.perf_counter()
    e2, g2 = jax.value_and_grad(loss)(A.todense() * 1.01)
    jax.block_until_ready(g2)
    t_run = time.perf_counter() - t0

    print(f"D={D} chi={chi}")
    print(f"  First call (compile+run): {t_compile:.1f}s")
    print(f"  Second call (run only):   {t_run:.1f}s")
    print(f"  Energy: {float(e):.6f}")
    return t_compile, t_run


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    print("Python-loop CTM AD compilation benchmark")
    print("=" * 50)
    bench_compile_implicit(D=2, chi=4)
    print()
    bench_compile_implicit(D=2, chi=8)
