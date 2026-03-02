"""Tests for fermionic iPEPS (fPEPS) algorithms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _initialize_fpeps,
    _trotter_gate,
    spinless_fermion_gate,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def fp():
    return FermionParity()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_config():
    return FPEPSConfig()


# ------------------------------------------------------------------ #
# Task 1: spinless_fermion_gate                                        #
# ------------------------------------------------------------------ #


class TestSpinlessFermionGate:
    """Tests for the 2-site Hamiltonian gate H = -t(c†c + h.c.) + V(nn)."""

    def test_returns_symmetric_tensor(self, default_config):
        H = spinless_fermion_gate(default_config)
        assert isinstance(H, SymmetricTensor)

    def test_shape_is_2x2x2x2(self, default_config):
        H = spinless_fermion_gate(default_config)
        dense = H.todense()
        assert dense.shape == (2, 2, 2, 2)

    def test_labels(self, default_config):
        H = spinless_fermion_gate(default_config)
        assert H.labels() == ("si", "sj", "si_out", "sj_out")

    def test_flows(self, default_config):
        H = spinless_fermion_gate(default_config)
        flows = [idx.flow for idx in H.indices]
        assert flows == [
            FlowDirection.IN,
            FlowDirection.IN,
            FlowDirection.OUT,
            FlowDirection.OUT,
        ]

    def test_hermitian(self, default_config):
        """H reshaped as 4x4 matrix should be Hermitian."""
        H = spinless_fermion_gate(default_config)
        dense = H.todense().reshape(4, 4)
        np.testing.assert_allclose(dense, dense.T.conj(), atol=1e-14)

    def test_free_fermion_spectrum(self):
        """For V=0 (free fermion), eigenvalues of -t(c†c + h.c.) are known."""
        cfg = FPEPSConfig(t=1.0, V=0.0)
        H = spinless_fermion_gate(cfg)
        dense = H.todense().reshape(4, 4)
        eigvals = np.sort(np.linalg.eigvalsh(np.array(dense)))
        # Basis: |00>, |01>, |10>, |11>
        # H = -t(c1†c2 + c2†c1)
        # In the 1-particle sector: eigenvalues -t, +t
        # 0-particle and 2-particle sectors: eigenvalue 0
        expected = np.array([-1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(eigvals, expected, atol=1e-14)

    def test_interaction_diagonal(self):
        """V*n_i*n_j should only contribute to the |11> state."""
        cfg = FPEPSConfig(t=0.0, V=2.5)
        H = spinless_fermion_gate(cfg)
        dense = H.todense().reshape(4, 4)
        # Only (|11>-><11>) = V
        expected = np.diag([0.0, 0.0, 0.0, 2.5])
        np.testing.assert_allclose(dense, expected, atol=1e-14)


# ------------------------------------------------------------------ #
# Task 2: _trotter_gate                                                #
# ------------------------------------------------------------------ #


class TestTrotterGate:
    """Tests for the Trotter gate exp(-dt * H)."""

    def test_returns_symmetric_tensor(self, default_config):
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, default_config.dt)
        assert isinstance(G, SymmetricTensor)

    def test_shape_preserved(self, default_config):
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, default_config.dt)
        assert G.todense().shape == (2, 2, 2, 2)

    def test_dt_zero_is_identity(self, default_config):
        """exp(0 * H) = identity operator reshaped to (2,2,2,2)."""
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, dt=0.0)
        dense = G.todense().reshape(4, 4)
        np.testing.assert_allclose(dense, np.eye(4), atol=1e-12)

    def test_unitarity(self, default_config):
        """exp(-dt*H) reshaped to 4x4 should be unitary for real dt (Hermitian H)."""
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, default_config.dt)
        dense = np.array(G.todense().reshape(4, 4))
        product = dense @ dense.T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-12)


# ------------------------------------------------------------------ #
# Task 3: _initialize_fpeps                                            #
# ------------------------------------------------------------------ #


class TestFPEPSInit:
    """Tests for fPEPS site tensor initialization."""

    def test_returns_symmetric_tensor(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        assert isinstance(A, SymmetricTensor)

    def test_ndim_is_5(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        assert A.ndim == 5

    def test_labels(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        assert A.labels() == ("u", "d", "l", "r", "phys")

    def test_physical_dim_is_2(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        phys_idx = A.indices[4]
        assert phys_idx.dim == 2

    def test_virtual_dim_matches_D(self, rng):
        for D in [2, 3, 4]:
            cfg = FPEPSConfig(D=D)
            A = _initialize_fpeps(cfg, rng)
            for i in range(4):  # u, d, l, r
                assert A.indices[i].dim == D

    def test_flows(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        flows = [idx.flow for idx in A.indices]
        assert flows == [
            FlowDirection.OUT,  # u
            FlowDirection.IN,  # d
            FlowDirection.OUT,  # l
            FlowDirection.IN,  # r
            FlowDirection.IN,  # phys
        ]

    def test_different_keys_give_different_tensors(self, default_config):
        A1 = _initialize_fpeps(default_config, jax.random.PRNGKey(0))
        A2 = _initialize_fpeps(default_config, jax.random.PRNGKey(1))
        d1 = A1.todense()
        d2 = A2.todense()
        assert not jnp.allclose(d1, d2)
