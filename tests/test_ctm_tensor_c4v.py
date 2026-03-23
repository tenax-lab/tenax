"""Tests for C4v-symmetric CTM with Tensor protocol."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_dense():
    """Near-product-state DenseTensor iPEPS site tensor, D=2, d=2, trivial U1.

    A near-product-state tensor has a unique CTM fixed point that is
    naturally C4v-symmetric, so both general and C4v CTMs converge to
    the same environment.
    """
    D, d = 2, 2
    rng = np.random.RandomState(42)
    data = 0.01 * jnp.array(rng.standard_normal((D, D, D, D, d)))
    data = data.at[0, 0, 0, 0, 0].set(1.0)
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, indices)


@pytest.fixture
def heisenberg_gate():
    """Heisenberg 2-site Hamiltonian gate as dense array."""
    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #


class TestC4vCTM:
    def test_returns_ctm_tensor_env(self, small_peps_dense):
        """ctm_tensor_c4v returns a CTMTensorEnv."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_dense, chi=4, max_iter=5)
        assert isinstance(env, CTMTensorEnv)

    def test_all_tensors_finite(self, small_peps_dense):
        """All environment tensors are finite after 20 sweeps."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_dense, chi=8, max_iter=20)
        for field in env:
            dense = field.todense()
            assert jnp.all(jnp.isfinite(dense)), (
                f"Non-finite values in tensor with labels {field.labels()}"
            )

    def test_energy_matches_general_ctm(self, small_peps_dense, heisenberg_gate):
        """C4v energy matches general CTM energy (atol=1e-4)."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        chi = 8

        # General CTM
        env_gen = ctm_tensor(small_peps_dense, chi=chi, max_iter=60, conv_tol=1e-10)
        E_gen = float(
            compute_energy_ctm_tensor(small_peps_dense, env_gen, heisenberg_gate, d=2)
        )

        # C4v CTM
        env_c4v = ctm_tensor_c4v(small_peps_dense, chi=chi, max_iter=60, conv_tol=1e-10)
        E_c4v = float(
            compute_energy_ctm_tensor(small_peps_dense, env_c4v, heisenberg_gate, d=2)
        )

        np.testing.assert_allclose(E_c4v, E_gen, atol=1e-4)


# ------------------------------------------------------------------ #
# U(1) SymmetricTensor tests                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_u1():
    """Random U(1) SymmetricTensor iPEPS with D=2, d=2."""
    key = jax.random.PRNGKey(42)
    sym = U1Symmetry()
    vc = np.array([-1, 1], dtype=np.int32)
    pc = np.array([-1, 1], dtype=np.int32)
    indices = (
        TensorIndex(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, vc.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, vc.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )
    return SymmetricTensor.random_normal(indices, key)


@pytest.fixture
def small_peps_fermionic():
    """Random FermionParity SymmetricTensor iPEPS with D=2, d=2."""
    key = jax.random.PRNGKey(7)
    sym = FermionParity()
    vc = np.array([0, 1], dtype=np.int32)
    pc = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, vc.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, vc.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )
    return SymmetricTensor.random_normal(indices, key)


class TestC4vCTMSymmetric:
    def test_u1_converges(self, small_peps_u1):
        """C4v CTM converges with U(1) SymmetricTensor."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_u1, chi=6, max_iter=30, conv_tol=1e-8)
        assert isinstance(env, CTMTensorEnv)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_u1_energy_matches_dense(self, small_peps_u1, heisenberg_gate):
        """U(1) C4v CTM energy matches DenseTensor path."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        chi = 8
        A_dense = DenseTensor(small_peps_u1.todense(), small_peps_u1.indices)

        env_sym = ctm_tensor_c4v(small_peps_u1, chi=chi, max_iter=50, conv_tol=1e-10)
        E_sym = float(
            compute_energy_ctm_tensor(small_peps_u1, env_sym, heisenberg_gate, d=2)
        )

        env_dense = ctm_tensor_c4v(A_dense, chi=chi, max_iter=50, conv_tol=1e-10)
        E_dense = float(
            compute_energy_ctm_tensor(A_dense, env_dense, heisenberg_gate, d=2)
        )

        np.testing.assert_allclose(E_sym, E_dense, atol=1e-4)


class TestC4vCTMFermionic:
    def test_fermionic_converges(self, small_peps_fermionic):
        """C4v CTM converges with FermionParity SymmetricTensor."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_fermionic, chi=4, max_iter=30, conv_tol=1e-8)
        assert isinstance(env, CTMTensorEnv)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_fermionic_energy_finite(self, small_peps_fermionic, heisenberg_gate):
        """FermionParity C4v CTM produces a finite energy.

        C4v CTM internally densifies fermionic tensors (Koszul signs from
        the C4v flow-flip expansion cause cancellation for SymmetricTensor).
        Energy must be computed with the matching DenseTensor A.
        """
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        chi = 8
        A_dense = DenseTensor(
            small_peps_fermionic.todense(), small_peps_fermionic.indices
        )
        env = ctm_tensor_c4v(A_dense, chi=chi, max_iter=80, conv_tol=1e-10)
        E = float(compute_energy_ctm_tensor(A_dense, env, heisenberg_gate, d=2))
        assert jnp.isfinite(E), f"Energy not finite: {E}"

    def test_fermionic_many_sweeps_stable(self, small_peps_fermionic):
        """FermionParity C4v CTM runs 50 sweeps without crashing."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_fermionic, chi=4, max_iter=50, conv_tol=1e-14)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))
