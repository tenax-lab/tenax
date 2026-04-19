"""Integration tests for Python-loop CTM AD: end-to-end optimization."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _make_random_A(D=2, d=2, key=None):
    """Create a random iPEPS site tensor as DenseTensor."""
    if key is None:
        key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (D, D, D, D, d), dtype=jnp.float64)
    data = data / jnp.linalg.norm(data)
    return _wrap_as_dense_tensor(data)


@pytest.mark.slow
def test_1site_heisenberg_implicit_optimization():
    """Full 1-site Heisenberg optimization via gradient descent with GMRES backward.

    Verifies:
    - Energy decreases over steps
    - Final energy is variational (above exact -0.6694)
    - Final energy is better than random init
    """
    A = _make_random_A(D=2, d=2, key=jax.random.PRNGKey(42))
    gate = heisenberg_gate()
    chi = 4
    neighbors = SINGLE_SITE_NEIGHBORS

    energies = []
    params = A.todense()

    for step in range(20):

        def loss(p):
            A_local = _wrap_as_dense_tensor(p)
            site_tensors = {(0, 0): A_local}
            return ctm_energy_implicit(
                site_tensors,
                neighbors,
                gate,
                chi=chi,
                max_iter=60,
                conv_tol=1e-8,
                gmres_tol=1e-6,
                gmres_maxiter=100,
            )

        e, g = jax.value_and_grad(loss)(params)
        energies.append(float(e))
        params = params - 0.01 * g
        params = params / jnp.linalg.norm(params)

    # Energy should decrease overall
    assert energies[-1] < energies[0], (
        f"Energy didn't decrease: {energies[0]:.4f} -> {energies[-1]:.4f}"
    )
    # Energy should be variational (above exact -0.6694)
    assert energies[-1] > -0.70, f"Energy {energies[-1]:.4f} below physical!"
    # Energy should improve from random init
    assert energies[-1] < -0.10, f"Energy {energies[-1]:.4f} still near random init"


@pytest.mark.slow
def test_no_recompilation_across_steps():
    """Verify that the JIT cache is reused across optimization steps.

    With the Python-loop CTM design, JIT compilation happens at the individual
    step level (not per-call). Once the JIT-compiled steps are warm, subsequent
    calls with different data should run at similar speed. This test verifies
    that multiple gradient evaluations complete successfully and that the second
    call does not take significantly *longer* (which would indicate spurious
    recompilation).
    """
    A = _make_random_A(D=2, d=2, key=jax.random.PRNGKey(0))
    gate = heisenberg_gate()
    chi = 4
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
            gmres_tol=1e-4,
            gmres_maxiter=50,
        )

    # First call (may include JIT compilation of individual steps)
    t0 = time.perf_counter()
    e1, g1 = jax.value_and_grad(loss)(A.todense())
    jax.block_until_ready(g1)
    t_first = time.perf_counter() - t0

    # Second call with different data (should reuse JIT cache)
    t0 = time.perf_counter()
    e2, g2 = jax.value_and_grad(loss)(A.todense() * 1.01)
    jax.block_until_ready(g2)
    t_second = time.perf_counter() - t0

    # The second call should NOT be significantly slower than the first.
    # A 3x slowdown would indicate spurious recompilation.
    assert t_second < t_first * 3.0, (
        f"Second call ({t_second:.1f}s) much slower than first ({t_first:.1f}s), "
        f"suggesting recompilation"
    )
    # Both calls should produce finite results
    assert jnp.isfinite(e1) and jnp.isfinite(e2)
