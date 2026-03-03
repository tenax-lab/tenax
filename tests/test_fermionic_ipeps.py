"""Tests for fermionic iPEPS (fPEPS) algorithms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _fpeps_simple_update,
    _fpeps_simple_update_horizontal,
    _fpeps_simple_update_vertical,
    _initialize_fpeps,
    _trotter_gate,
    fpeps,
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

    def test_symmetric_positive_definite(self, default_config):
        """exp(-dt*H) for real dt and Hermitian H is symmetric positive-definite."""
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, default_config.dt)
        dense = np.array(G.todense().reshape(4, 4))
        # Symmetric
        np.testing.assert_allclose(dense, dense.T, atol=1e-14)
        # Positive-definite: all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(dense)
        assert np.all(eigvals > 0)


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


# ------------------------------------------------------------------ #
# Task 4: _fpeps_simple_update_horizontal                              #
# ------------------------------------------------------------------ #


class TestFPEPSSimpleUpdateBond:
    """Tests for horizontal simple update on the fPEPS bond."""

    @pytest.fixture
    def setup(self):
        """Prepare A, gate, and lambdas for horizontal update."""
        cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.01)
        key = jax.random.PRNGKey(7)
        A = _initialize_fpeps(cfg, key)
        H = spinless_fermion_gate(cfg)
        gate = _trotter_gate(H, cfg.dt)
        D = cfg.D
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)
        return A, gate, lam_h, lam_v, D

    def test_returns_tensor_and_lambda(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_horizontal(A, gate, lam_h, lam_v, D)
        assert isinstance(A_new, SymmetricTensor)
        assert isinstance(lam_new, jax.Array)

    def test_output_shape(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_horizontal(A, gate, lam_h, lam_v, D)
        assert A_new.todense().shape == (D, D, D, D, 2)
        assert lam_new.shape == (D,)

    def test_output_labels(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_horizontal(A, gate, lam_h, lam_v, D)
        assert A_new.labels() == ("u", "d", "l", "r", "phys")

    def test_output_finite(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_horizontal(A, gate, lam_h, lam_v, D)
        assert jnp.all(jnp.isfinite(A_new.todense()))
        assert jnp.all(jnp.isfinite(lam_new))

    def test_lambda_positive(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_horizontal(A, gate, lam_h, lam_v, D)
        assert jnp.all(lam_new >= 0)


# ------------------------------------------------------------------ #
# Task 5: _fpeps_simple_update_vertical                                #
# ------------------------------------------------------------------ #


class TestFPEPSSimpleUpdateVertical:
    """Tests for vertical simple update on the fPEPS bond."""

    @pytest.fixture
    def setup(self):
        """Prepare A, gate, and lambdas for vertical update."""
        cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.01)
        key = jax.random.PRNGKey(7)
        A = _initialize_fpeps(cfg, key)
        H = spinless_fermion_gate(cfg)
        gate = _trotter_gate(H, cfg.dt)
        D = cfg.D
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)
        return A, gate, lam_h, lam_v, D

    def test_returns_tensor_and_lambda(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_vertical(A, gate, lam_h, lam_v, D)
        assert isinstance(A_new, SymmetricTensor)
        assert isinstance(lam_new, jax.Array)

    def test_output_shape(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_vertical(A, gate, lam_h, lam_v, D)
        assert A_new.todense().shape == (D, D, D, D, 2)
        assert lam_new.shape == (D,)

    def test_output_finite(self, setup):
        A, gate, lam_h, lam_v, D = setup
        A_new, lam_new = _fpeps_simple_update_vertical(A, gate, lam_h, lam_v, D)
        assert jnp.all(jnp.isfinite(A_new.todense()))
        assert jnp.all(jnp.isfinite(lam_new))


# ------------------------------------------------------------------ #
# Task 6: _fpeps_simple_update                                         #
# ------------------------------------------------------------------ #


class TestFPEPSSimpleUpdate:
    """Tests for the full simple update loop."""

    def test_simple_update_runs(self):
        """5 steps should complete without error."""
        cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.01)
        key = jax.random.PRNGKey(0)
        A = _initialize_fpeps(cfg, key)
        H = spinless_fermion_gate(cfg)
        A_opt, lam_h, lam_v = _fpeps_simple_update(
            A, H, max_D=cfg.D, dt=cfg.dt, steps=5
        )
        assert isinstance(A_opt, SymmetricTensor)
        assert jnp.all(jnp.isfinite(A_opt.todense()))

    def test_simple_update_changes_tensor(self):
        """20 steps of imaginary time evolution should change A."""
        cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.01)
        key = jax.random.PRNGKey(0)
        A = _initialize_fpeps(cfg, key)
        A_before = A.todense()
        H = spinless_fermion_gate(cfg)
        A_opt, lam_h, lam_v = _fpeps_simple_update(
            A, H, max_D=cfg.D, dt=cfg.dt, steps=20
        )
        A_after = A_opt.todense()
        assert not jnp.allclose(A_before, A_after)


# ------------------------------------------------------------------ #
# Task 7: fpeps (entry point)                                          #
# ------------------------------------------------------------------ #


class TestFPEPS:
    """Tests for the fpeps entry point with CTM evaluation."""

    def test_fpeps_runs(self):
        """fpeps should return a finite energy."""
        cfg = FPEPSConfig(
            D=2,
            t=1.0,
            V=0.0,
            dt=0.01,
            num_imaginary_steps=5,
            ctm_chi=4,
            ctm_max_iter=10,
            ctm_conv_tol=1e-4,
        )
        H = spinless_fermion_gate(cfg)
        key = jax.random.PRNGKey(99)
        energy, A_opt, env = fpeps(H, cfg, key=key)
        assert jnp.isfinite(energy)

    def test_fpeps_returns_symmetric_tensor(self):
        """A_opt should be a SymmetricTensor."""
        cfg = FPEPSConfig(
            D=2,
            t=1.0,
            V=0.0,
            dt=0.01,
            num_imaginary_steps=5,
            ctm_chi=4,
            ctm_max_iter=10,
            ctm_conv_tol=1e-4,
        )
        H = spinless_fermion_gate(cfg)
        key = jax.random.PRNGKey(99)
        energy, A_opt, env = fpeps(H, cfg, key=key)
        assert isinstance(A_opt, SymmetricTensor)


# ------------------------------------------------------------------ #
# Task 8: Exports                                                      #
# ------------------------------------------------------------------ #


def test_fpeps_importable_from_tenax():
    """Public API should be importable from top-level tenax package."""
    from tenax import FPEPSConfig, fpeps, spinless_fermion_gate

    assert FPEPSConfig is not None
    assert fpeps is not None
    assert spinless_fermion_gate is not None
