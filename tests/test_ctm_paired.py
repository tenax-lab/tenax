"""Tests for paired horizontal/vertical CTM moves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_tensor_sweep,
    _ctm_tensor_sweep_paired,
    _renormalize_tensor_env,
    ctm_tensor,
)
from tenax.algorithms._ctm_tensor_energy import (
    _rdm2x1_tensor,
    compute_energy_ctm_tensor,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_dense():
    """Near-product-state DenseTensor iPEPS site tensor, D=2, d=2."""
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


class TestPairedSweepDense:
    def test_paired_sweep_dense_matches_standard(
        self, small_peps_dense, heisenberg_gate
    ):
        """Energy from paired sweep matches 4-move sweep for DenseTensor."""
        chi = 8
        n_sweeps = 40

        A = small_peps_dense
        a = _build_double_layer_tensor(A)

        # Standard 4-move sweep
        env_std = initialize_ctm_tensor_env(A, chi)
        for _ in range(n_sweeps):
            env_std = _ctm_tensor_sweep(env_std, a, chi, renormalize=True)

        # Paired 2-move sweep
        env_paired = initialize_ctm_tensor_env(A, chi)
        for _ in range(n_sweeps):
            env_paired = _ctm_tensor_sweep_paired(env_paired, a, chi, renormalize=True)

        E_std = float(compute_energy_ctm_tensor(A, env_std, heisenberg_gate, d=2))
        E_paired = float(compute_energy_ctm_tensor(A, env_paired, heisenberg_gate, d=2))

        np.testing.assert_allclose(E_paired, E_std, atol=1e-4)


class TestPairedSweepFermionic:
    def test_paired_sweep_fermionic_stable(self, small_peps_fermionic):
        """FermionParity SymmetricTensor runs 15 sweeps without error."""
        chi = 4
        A = small_peps_fermionic
        a = _build_double_layer_tensor(A)

        env = initialize_ctm_tensor_env(A, chi)
        for _ in range(15):
            env = _ctm_tensor_sweep_paired(env, a, chi, renormalize=True)

        # Check all tensors are finite
        for field in env:
            dense = field.todense()
            assert jnp.all(jnp.isfinite(dense)), (
                f"Non-finite values in tensor with labels {field.labels()}"
            )

    def test_paired_sweep_fermionic_energy_converged(self, heisenberg_gate):
        """Fermionic paired CTM converges to a valid RDM.

        Uses a near-product-state tensor (dominated by vacuum) so the CTM
        fixed point is well-conditioned.  Validates convergence and RDM
        positivity instead of comparing against the dense (bosonic) path,
        which lacks Koszul signs and computes a physically different quantity.
        """
        chi = 8
        sym = FermionParity()
        D, d = 2, 2
        vc = np.array([0, 1], dtype=np.int32)
        pc = np.array([0, 1], dtype=np.int32)
        data = jnp.zeros((D, D, D, D, d))
        data = data.at[0, 0, 0, 0, 0].set(1.0)
        data = data + 0.05 * jax.random.normal(jax.random.PRNGKey(7), (D, D, D, D, d))
        data = data / jnp.linalg.norm(data)
        indices = (
            TensorIndex(sym, vc.copy(), FlowDirection.OUT, label="u"),
            TensorIndex(sym, vc.copy(), FlowDirection.IN, label="d"),
            TensorIndex(sym, vc.copy(), FlowDirection.OUT, label="l"),
            TensorIndex(sym, vc.copy(), FlowDirection.IN, label="r"),
            TensorIndex(sym, pc.copy(), FlowDirection.IN, label="phys"),
        )
        A = SymmetricTensor.from_dense(data, indices, tol=float("inf"))

        env = ctm_tensor(A, chi=chi, max_iter=60, conv_tol=1e-10)
        E = float(compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2))

        assert jnp.isfinite(E), f"Energy not finite: {E}"

        rdm = _rdm2x1_tensor(A, env)
        rdm_mat = rdm.reshape(d * d, d * d)
        eigvals = np.linalg.eigvalsh(np.array(rdm_mat))
        assert eigvals[0] > -1e-4, f"RDM not PSD: min eigenvalue = {eigvals[0]}"
        np.testing.assert_allclose(
            float(jnp.trace(rdm_mat).real),
            1.0,
            atol=1e-8,
            err_msg="RDM trace != 1",
        )


class TestPairedSweepU1:
    def test_paired_sweep_u1_stable(self, small_peps_u1):
        """U(1) SymmetricTensor runs 15 sweeps without error."""
        chi = 6
        A = small_peps_u1
        a = _build_double_layer_tensor(A)

        env = initialize_ctm_tensor_env(A, chi)
        for _ in range(15):
            env = _ctm_tensor_sweep_paired(env, a, chi, renormalize=True)

        # Check all tensors are finite
        for field in env:
            dense = field.todense()
            assert jnp.all(jnp.isfinite(dense)), (
                f"Non-finite values in tensor with labels {field.labels()}"
            )
