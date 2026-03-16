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
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

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
