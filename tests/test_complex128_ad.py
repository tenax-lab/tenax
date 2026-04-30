"""Tests for complex128 iPEPS AD optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _make_complex_A(D=2, d=2, key=None):
    """Create a random complex128 iPEPS site tensor."""
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    real = jax.random.normal(k1, (D, D, D, D, d), dtype=jnp.float64)
    imag = jax.random.normal(k2, (D, D, D, D, d), dtype=jnp.float64)
    data = real + 1j * imag
    data = data / jnp.linalg.norm(data)
    return _wrap_as_dense_tensor(data)


def test_lbfgs_two_loop_complex():
    """L-BFGS two-loop recursion with complex vectors."""
    from tenax.algorithms._metric_precond import lbfgs_two_loop

    s = jnp.array([1.0 + 0.5j, 0.3 - 0.2j])
    y = jnp.array([0.5 + 0.1j, 0.2 + 0.3j])
    sy = float(jnp.real(jnp.vdot(s, y)))
    assert sy > 0
    rho = 1.0 / sy
    history = [(s, y, rho)]

    grad = jnp.array([1.0 + 0.0j, 0.0 + 1.0j])
    result = lbfgs_two_loop(grad, history, lambda v: v)

    assert result.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(result))


def test_c4v_sublattice_rotation_complex():
    """Verify C4v sublattice rotation preserves complex128 dtype."""
    from tenax.algorithms.ipeps import build_c4v_basis, c4v_tensor_from_coeffs

    D, d = 2, 2
    basis = jnp.array(build_c4v_basis(D, d))
    n_basis = basis.shape[0]
    coeffs = jnp.zeros(n_basis, dtype=jnp.complex128)
    coeffs = coeffs.at[0].set(1.0 + 0.5j)
    if n_basis > 1:
        coeffs = coeffs.at[1].set(0.3 - 0.2j)
    A_data = c4v_tensor_from_coeffs(coeffs, basis, (D, D, D, D, d))
    U_sub = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=jnp.complex128)
    B_data = jnp.einsum("luRDs,sS->luRDS", A_data, U_sub)
    assert B_data.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(B_data))


@pytest.mark.slow
def test_complex128_1site_gradient():
    """Verify gradient computation works for complex128 1-site.

    Uses a small complex perturbation on a well-conditioned real tensor
    to ensure the CTM environment is contractive (rho(J^T) < 1).
    """
    # Start from a real tensor that gives contractive CTM, add small imag part
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    real_part = jax.random.normal(k1, (2, 2, 2, 2, 2), dtype=jnp.float64)
    imag_part = 0.1 * jax.random.normal(k2, (2, 2, 2, 2, 2), dtype=jnp.float64)
    data = real_part + 1j * imag_part
    data = data / jnp.linalg.norm(data)
    A = _wrap_as_dense_tensor(data)
    gate = heisenberg_gate()

    def loss(p):
        A_local = _wrap_as_dense_tensor(p)
        return ctm_energy_implicit(
            {(0, 0): A_local},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=8,
            max_iter=80,
            conv_tol=1e-9,
        )

    params = A.todense()
    e, g = jax.value_and_grad(loss)(params)

    assert jnp.isfinite(e), f"Energy is not finite: {e}"
    assert g.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.slow
def test_complex128_2site_gradient():
    """Verify gradient computation works for complex128 2-site.

    Uses small complex perturbation on real tensors for well-conditioned CTM.
    """
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(10), 4)
    A_data = jax.random.normal(k1, (2, 2, 2, 2, 2)) + 0.1j * jax.random.normal(
        k2, (2, 2, 2, 2, 2)
    )
    B_data = jax.random.normal(k3, (2, 2, 2, 2, 2)) + 0.1j * jax.random.normal(
        k4, (2, 2, 2, 2, 2)
    )
    A_data = A_data / jnp.linalg.norm(A_data)
    B_data = B_data / jnp.linalg.norm(B_data)
    A = _wrap_as_dense_tensor(A_data)
    B = _wrap_as_dense_tensor(B_data)
    gate = heisenberg_gate()
    d_phys = 2

    def energy_fn(site_tensors, envs, gate_):
        return compute_energy_ctm_tensor_2site(
            site_tensors[(0, 0)],
            site_tensors[(1, 0)],
            envs[(0, 0)],
            envs[(1, 0)],
            gate_,
            d_phys,
        )

    def loss(p_a, p_b):
        st = {
            (0, 0): _wrap_as_dense_tensor(p_a),
            (1, 0): _wrap_as_dense_tensor(p_b),
        }
        return ctm_energy_implicit(
            st,
            CHECKERBOARD_NEIGHBORS,
            gate,
            chi=8,
            max_iter=80,
            conv_tol=1e-9,
            energy_fn=energy_fn,
        )

    e, (gA, gB) = jax.value_and_grad(loss, argnums=(0, 1))(A.todense(), B.todense())
    assert jnp.isfinite(e)
    assert gA.dtype == jnp.complex128
    assert gB.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(gA))
    assert jnp.all(jnp.isfinite(gB))


@pytest.mark.slow
def test_complex128_1site_optimization():
    """End-to-end 1-site complex128 optimization: energy decreases."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    data = jax.random.normal(k1, (2, 2, 2, 2, 2)) + 0.1j * jax.random.normal(
        k2, (2, 2, 2, 2, 2)
    )
    data = data / jnp.linalg.norm(data)
    gate = heisenberg_gate()

    energies = []
    params = data

    for step in range(15):

        def loss(p):
            A_local = _wrap_as_dense_tensor(p)
            return ctm_energy_implicit(
                {(0, 0): A_local},
                SINGLE_SITE_NEIGHBORS,
                gate,
                chi=8,
                max_iter=80,
                conv_tol=1e-9,
            )

        e, g = jax.value_and_grad(loss)(params)
        energies.append(float(jnp.real(e)))
        params = params - 0.01 * g
        params = params / jnp.linalg.norm(params)

    assert energies[-1] < energies[0], (
        f"Energy didn't decrease: {energies[0]:.4f} -> {energies[-1]:.4f}"
    )
    assert energies[-1] > -0.70, f"Non-variational: {energies[-1]:.4f}"
