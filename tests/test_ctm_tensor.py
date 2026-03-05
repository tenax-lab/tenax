"""Tests for standard CTM with Tensor protocol (polymorphic dense/symmetric)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor import (
    CTMTensorEnv,
    _build_double_layer_open_tensor,
    _build_double_layer_tensor,
    _ctm_tensor_sweep,
    compute_energy_ctm_tensor,
    ctm_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps import (
    CTMConfig,
    _build_double_layer,
    compute_energy_ctm,
    ctm,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_dense():
    """Random DenseTensor iPEPS site tensor, D=2, d=2."""
    key = jax.random.PRNGKey(42)
    D, d = 2, 2
    data = jax.random.normal(key, (D, D, D, D, d))
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
def small_peps_symmetric():
    """Random SymmetricTensor iPEPS with trivial U(1) charges."""
    key = jax.random.PRNGKey(99)
    D, d = 2, 2
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
    data = jax.random.normal(key, (D, D, D, D, d))
    return SymmetricTensor.from_dense(data, indices)


@pytest.fixture
def fpeps_tensor():
    """Random SymmetricTensor iPEPS with FermionParity."""
    key = jax.random.PRNGKey(7)
    sym = FermionParity()
    virt_charges = np.array([0, 1], dtype=np.int32)
    phys_charges = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex(sym, virt_charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex(sym, virt_charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
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
# Double-layer tests                                                   #
# ------------------------------------------------------------------ #


class TestDoubleLayer:
    def test_double_layer_tensor_matches_dense(self, small_peps_dense):
        """DenseTensor double-layer matches raw einsum."""
        A = small_peps_dense
        a_tensor = _build_double_layer_tensor(A)
        a_dense = a_tensor.todense()

        # Reference: raw dense path
        A_raw = A.todense()
        a_ref = _build_double_layer(A_raw)
        D = A_raw.shape[0]
        a_ref = a_ref.reshape(D**2, D**2, D**2, D**2)

        np.testing.assert_allclose(a_dense, a_ref, atol=1e-12)

    def test_double_layer_tensor_labels(self, small_peps_dense):
        a = _build_double_layer_tensor(small_peps_dense)
        assert a.labels() == ("u2", "d2", "l2", "r2")
        assert a.ndim == 4

    def test_double_layer_tensor_symmetric(self, small_peps_symmetric):
        """SymmetricTensor double-layer todense matches DenseTensor result."""
        A_sym = small_peps_symmetric
        a_sym = _build_double_layer_tensor(A_sym)

        A_dense = DenseTensor(A_sym.todense(), A_sym.indices)
        a_dense = _build_double_layer_tensor(A_dense)

        np.testing.assert_allclose(a_sym.todense(), a_dense.todense(), atol=1e-12)

    def test_double_layer_open_tensor(self, small_peps_dense):
        """Open double-layer has correct shape and labels."""
        a_open = _build_double_layer_open_tensor(small_peps_dense)
        assert a_open.ndim == 6
        labels = a_open.labels()
        assert "phys" in labels
        assert "phys_bra" in labels
        dense = a_open.todense()
        # Check: tracing phys,phys_bra gives double-layer
        a_closed = jnp.einsum("...ss->...", dense)
        a_ref = _build_double_layer_tensor(small_peps_dense).todense()
        np.testing.assert_allclose(a_closed, a_ref, atol=1e-12)


# ------------------------------------------------------------------ #
# Initialization tests                                                 #
# ------------------------------------------------------------------ #


class TestInitialization:
    def test_init_dense_shapes(self, small_peps_dense):
        chi = 4
        env = initialize_ctm_tensor_env(small_peps_dense, chi)
        assert isinstance(env, CTMTensorEnv)
        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert C.todense().shape == (chi, chi)
        D2 = small_peps_dense.indices[0].dim ** 2
        for T in [env.T1, env.T2, env.T3, env.T4]:
            assert T.todense().shape == (chi, D2, chi)

    def test_init_dense_labels(self, small_peps_dense):
        env = initialize_ctm_tensor_env(small_peps_dense, chi=4)
        assert env.C1.labels() == ("c1_d", "c1_r")
        assert env.T1.labels() == ("t1_l", "u2", "t1_r")
        assert env.T4.labels() == ("t4_d", "l2", "t4_u")

    def test_init_dense_finite(self, small_peps_dense):
        env = initialize_ctm_tensor_env(small_peps_dense, chi=4)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_init_symmetric(self, small_peps_symmetric):
        chi = 4
        env = initialize_ctm_tensor_env(small_peps_symmetric, chi)
        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert isinstance(C, SymmetricTensor)
            assert C.todense().shape == (chi, chi)
        for T in [env.T1, env.T2, env.T3, env.T4]:
            assert isinstance(T, SymmetricTensor)

    def test_init_fpeps(self, fpeps_tensor):
        """FermionParity iPEPS produces valid symmetric env."""
        chi = 4
        env = initialize_ctm_tensor_env(fpeps_tensor, chi)
        for field in env:
            assert isinstance(field, SymmetricTensor)
            assert jnp.all(jnp.isfinite(field.todense()))


# ------------------------------------------------------------------ #
# Sweep tests                                                          #
# ------------------------------------------------------------------ #


class TestSweep:
    def test_one_sweep_dense_finite(self, small_peps_dense):
        """One CTM sweep produces finite tensors."""
        chi = 4
        a = _build_double_layer_tensor(small_peps_dense)
        env = initialize_ctm_tensor_env(small_peps_dense, chi)
        env = _ctm_tensor_sweep(env, a, chi, renormalize=True)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense())), (
                f"Non-finite values in {field.labels()}"
            )

    def test_one_sweep_symmetric_finite(self, small_peps_symmetric):
        chi = 4
        a = _build_double_layer_tensor(small_peps_symmetric)
        env = initialize_ctm_tensor_env(small_peps_symmetric, chi)
        env = _ctm_tensor_sweep(env, a, chi, renormalize=True)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_one_sweep_fpeps_finite(self, fpeps_tensor):
        chi = 4
        a = _build_double_layer_tensor(fpeps_tensor)
        env = initialize_ctm_tensor_env(fpeps_tensor, chi)
        env = _ctm_tensor_sweep(env, a, chi, renormalize=True)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))


# ------------------------------------------------------------------ #
# Convergence + energy tests                                           #
# ------------------------------------------------------------------ #


class TestConvergence:
    def test_ctm_tensor_dense_converges(self, small_peps_dense):
        """DenseTensor CTM converges (finite env after max_iter)."""
        env = ctm_tensor(small_peps_dense, chi=4, max_iter=20, conv_tol=1e-6)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_ctm_tensor_dense_matches_ctm(self, small_peps_dense, heisenberg_gate):
        """DenseTensor CTM energy matches legacy dense CTM energy."""
        chi = 6
        A_raw = small_peps_dense.todense()

        # Legacy dense CTM
        cfg = CTMConfig(chi=chi, max_iter=50, conv_tol=1e-8)
        env_legacy = ctm(A_raw, cfg)
        E_legacy = float(compute_energy_ctm(A_raw, env_legacy, heisenberg_gate, d=2))

        # Tensor-protocol CTM
        env_tensor = ctm_tensor(small_peps_dense, chi=chi, max_iter=50, conv_tol=1e-8)
        E_tensor = float(
            compute_energy_ctm_tensor(
                small_peps_dense, env_tensor, heisenberg_gate, d=2
            )
        )

        np.testing.assert_allclose(E_tensor, E_legacy, atol=1e-4)

    def test_ctm_tensor_symmetric_converges(self, small_peps_symmetric):
        """SymmetricTensor CTM converges (trivial charges)."""
        env = ctm_tensor(small_peps_symmetric, chi=4, max_iter=20, conv_tol=1e-6)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_energy_symmetric_matches_dense(
        self, small_peps_symmetric, heisenberg_gate
    ):
        """SymmetricTensor CTM energy matches DenseTensor CTM energy."""
        chi = 6
        A_dense = DenseTensor(
            small_peps_symmetric.todense(), small_peps_symmetric.indices
        )

        env_dense = ctm_tensor(A_dense, chi=chi, max_iter=40, conv_tol=1e-8)
        E_dense = float(
            compute_energy_ctm_tensor(A_dense, env_dense, heisenberg_gate, d=2)
        )

        env_sym = ctm_tensor(small_peps_symmetric, chi=chi, max_iter=40, conv_tol=1e-8)
        E_sym = float(
            compute_energy_ctm_tensor(
                small_peps_symmetric, env_sym, heisenberg_gate, d=2
            )
        )

        np.testing.assert_allclose(E_sym, E_dense, atol=1e-4)


# ------------------------------------------------------------------ #
# Fermionic integration test                                           #
# ------------------------------------------------------------------ #


class TestFermionicIntegration:
    def test_fermionic_ctm_no_densify(self, fpeps_tensor):
        """Fermionic CTM works with SymmetricTensor (no todense)."""
        env = ctm_tensor(fpeps_tensor, chi=4, max_iter=15, conv_tol=1e-5)
        assert isinstance(env, CTMTensorEnv)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_fpeps_uses_ctm_tensor(self):
        """fpeps() with SymmetricTensor now returns CTMTensorEnv."""
        from tenax.algorithms.fermionic_ipeps import (
            FPEPSConfig,
            fpeps,
            spinless_fermion_gate,
        )

        config = FPEPSConfig(
            D=2,
            t=1.0,
            V=0.0,
            dt=0.05,
            num_imaginary_steps=5,
            ctm_chi=4,
            ctm_max_iter=10,
            ctm_conv_tol=1e-4,
        )
        H = spinless_fermion_gate(config)
        energy, A_opt, env = fpeps(H, config)
        assert isinstance(env, CTMTensorEnv)
        assert np.isfinite(energy)
