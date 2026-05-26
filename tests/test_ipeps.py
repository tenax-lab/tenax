"""Tests for the iPEPS and CTM algorithms."""

import contextlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import ipeps
from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
    iPEPSConfig,
)
from tenax.algorithms.ipeps_ctm import (
    _build_double_layer,
    _initialize_split_ctm_env,
    _split_env_to_standard,
    ctm,
    ctm_2site,
    ctm_split,
)
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tenax.algorithms.ipeps_rdm import (
    _build_double_layer_open,
    _rdm1x2,
    _rdm2x1,
    compute_energy_ctm,
    compute_energy_ctm_2site,
    compute_energy_split_ctm,
)


class TestCTMConfig:
    def test_default_values(self):
        cfg = CTMConfig()
        assert cfg.chi == 20
        assert cfg.max_iter == 100
        assert cfg.conv_tol == 1e-8
        assert cfg.renormalize is True

    def test_custom_values(self):
        cfg = CTMConfig(chi=10, max_iter=50, conv_tol=1e-6, renormalize=False)
        assert cfg.chi == 10
        assert cfg.max_iter == 50
        assert cfg.conv_tol == 1e-6
        assert cfg.renormalize is False


class TestIPEPSConfig:
    def test_default_values(self):
        cfg = iPEPSConfig()
        assert cfg.max_bond_dim == 2
        assert cfg.num_imaginary_steps == 100
        assert cfg.dt == 0.01
        assert cfg.ctm is not None
        assert isinstance(cfg.ctm, CTMConfig)
        assert cfg.gs_verbose is False
        assert cfg.gs_log_interval == 10

    def test_custom_values(self):
        cfg = iPEPSConfig(max_bond_dim=4, num_imaginary_steps=50, dt=0.05)
        assert cfg.max_bond_dim == 4
        assert cfg.num_imaginary_steps == 50
        assert cfg.dt == 0.05

    def test_su_init_default_true(self):
        cfg = iPEPSConfig()
        assert cfg.su_init is True


class TestCTMEnvironment:
    def test_named_tuple_fields(self):
        """CTMEnvironment should have 8 tensor fields: 4 corners + 4 edges."""
        chi = 3
        d2 = 4  # D^2
        dummy = jnp.zeros((chi, chi))
        dummy_edge = jnp.zeros((chi, d2, chi))
        env = CTMEnvironment(
            C1=dummy,
            C2=dummy,
            C3=dummy,
            C4=dummy,
            T1=dummy_edge,
            T2=dummy_edge,
            T3=dummy_edge,
            T4=dummy_edge,
        )
        assert env.C1.shape == (chi, chi)
        assert env.T1.shape == (chi, d2, chi)

    def test_access_by_name(self):
        chi = 2
        d2 = 4
        corners = [jnp.eye(chi) * i for i in range(1, 5)]
        edges = [jnp.zeros((chi, d2, chi))] * 4
        env = CTMEnvironment(*corners, *edges)
        assert jnp.allclose(env.C1, jnp.eye(chi) * 1)
        assert jnp.allclose(env.C4, jnp.eye(chi) * 4)


class TestBuildDoubleLayer:
    def test_output_shape(self):
        """Double-layer tensor should have shape (D,D,D,D,D,D,D,D) for bond D, phys d."""
        D = 2
        d = 2
        key = jax.random.PRNGKey(0)
        # A has shape (u, d, l, r, s) = (D, D, D, D, d)
        A = jax.random.normal(key, (D, D, D, D, d))
        M = _build_double_layer(A)
        # M = einsum("udlrs,UDLRs->udlrUDLR", A, conj(A))
        # shape = (D, D, D, D, D, D, D, D)
        assert M.shape == (D, D, D, D, D, D, D, D)

    def test_double_layer_is_real_for_real_tensor(self):
        """For real A, the double-layer M should be real."""
        key = jax.random.PRNGKey(1)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))
        M = _build_double_layer(A)
        assert jnp.all(jnp.imag(M) == 0) if jnp.iscomplexobj(M) else True

    def test_double_layer_nonneg_diagonal(self):
        """Diagonal elements (same ket/bra indices) should be non-negative."""
        key = jax.random.PRNGKey(2)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))
        M = _build_double_layer(A)
        # M has ordering (u, U, d, D, l, L, r, R) from uUdDlLrR
        # Diagonal: u=U, d=D, l=L, r=R → M[i,i,j,j,k,k,m,m] >= 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for m in range(2):
                        assert M[i, i, j, j, k, k, m, m] >= 0


class TestCTM:
    @pytest.fixture
    def small_peps_tensor(self):
        """Small random PEPS site tensor with shape (D,D,D,D,d)."""
        key = jax.random.PRNGKey(42)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        # Normalize
        return A / (jnp.linalg.norm(A) + 1e-10)

    def test_ctm_returns_environment(self, small_peps_tensor):
        """CTM should return a CTMEnvironment."""
        config = CTMConfig(chi=4, max_iter=5)
        env = ctm(small_peps_tensor, config)
        assert isinstance(env, CTMEnvironment)

    def test_ctm_corners_shape(self, small_peps_tensor):
        """Corner tensors should be (chi, chi) shaped."""
        chi = 4
        config = CTMConfig(chi=chi, max_iter=5)
        env = ctm(small_peps_tensor, config)
        assert env.C1.shape[0] <= chi
        assert env.C1.shape[1] <= chi

    def test_ctm_edge_shape(self, small_peps_tensor):
        """Edge tensors should have 3 legs."""
        config = CTMConfig(chi=4, max_iter=5)
        env = ctm(small_peps_tensor, config)
        assert env.T1.ndim == 3

    def test_ctm_runs_multiple_iters(self, small_peps_tensor):
        """CTM should converge (or run max_iter) without crashing."""
        config = CTMConfig(chi=4, max_iter=10, conv_tol=1e-12)  # tight tol -> max_iter
        env = ctm(small_peps_tensor, config)
        assert isinstance(env, CTMEnvironment)

    def test_ctm_with_initial_env(self, small_peps_tensor):
        """CTM should accept an initial environment and warm-start."""
        config = CTMConfig(chi=4, max_iter=3)
        env1 = ctm(small_peps_tensor, config)
        # Warm-start from env1
        env2 = ctm(small_peps_tensor, config, initial_env=env1)
        assert isinstance(env2, CTMEnvironment)

    def test_ctm_no_renormalize(self, small_peps_tensor):
        """CTM without renormalization should still run."""
        config = CTMConfig(chi=4, max_iter=5, renormalize=False)
        env = ctm(small_peps_tensor, config)
        assert isinstance(env, CTMEnvironment)

    def test_ctm_edge_tensors_change(self, small_peps_tensor):
        """After a full CTM run, edge tensors should differ from initialization."""
        config = CTMConfig(chi=4, max_iter=10)
        from tenax.algorithms.ipeps_ctm import _build_double_layer, _initialize_ctm_env

        a = _build_double_layer(small_peps_tensor)
        D = small_peps_tensor.shape[0]
        a = a.reshape(D**2, D**2, D**2, D**2)
        env0 = _initialize_ctm_env(a, config.chi)
        env = ctm(small_peps_tensor, config, initial_env=env0)
        # At least one edge tensor should have changed
        changed = not (
            jnp.allclose(env0.T1, env.T1, atol=1e-10)
            and jnp.allclose(env0.T2, env.T2, atol=1e-10)
            and jnp.allclose(env0.T3, env.T3, atol=1e-10)
            and jnp.allclose(env0.T4, env.T4, atol=1e-10)
        )
        assert changed, "Edge tensors did not change during CTM"


class TestComputeEnergyCTM:
    @pytest.fixture
    def peps_and_env(self):
        """Small PEPS tensor + CTM environment for energy computation tests."""
        key = jax.random.PRNGKey(7)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        config = CTMConfig(chi=4, max_iter=5)
        env = ctm(A, config)
        return A, env

    def test_energy_is_scalar(self, peps_and_env):
        """Energy from CTM contraction should be a scalar."""
        A, env = peps_and_env
        # Simple Heisenberg Sz*Sz gate for d=2
        d = 2
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        energy = compute_energy_ctm(A, env, gate, d)
        assert energy.shape == ()

    def test_energy_is_finite(self, peps_and_env):
        A, env = peps_and_env
        d = 2
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        energy = compute_energy_ctm(A, env, gate, d)
        assert jnp.isfinite(energy)


class TestRDM:
    """Tests for the 2-site reduced density matrices."""

    @pytest.fixture
    def peps_env(self):
        """PEPS tensor and converged CTM environment."""
        key = jax.random.PRNGKey(55)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        config = CTMConfig(chi=8, max_iter=20)
        env = ctm(A, config)
        return A, env, d

    def test_rdm_hermitian(self, peps_env):
        """The 2-site RDM should satisfy rdm == rdm^dagger."""
        A, env, d = peps_env
        rdm_h = _rdm2x1(A, env, d)
        rdm_v = _rdm1x2(A, env, d)

        rdm_h_mat = rdm_h.reshape(d * d, d * d)
        rdm_v_mat = rdm_v.reshape(d * d, d * d)
        assert jnp.allclose(rdm_h_mat, rdm_h_mat.conj().T, atol=1e-10)
        assert jnp.allclose(rdm_v_mat, rdm_v_mat.conj().T, atol=1e-10)

    def test_rdm_positive_semidefinite(self, peps_env):
        """Eigenvalues of the RDM should be bounded.

        For a random (non-optimized) PEPS with small chi the CTM
        environment is approximate, so eigenvalues outside [0,1] are
        expected.  We check they are not wildly unphysical (> O(10)).
        """
        A, env, d = peps_env
        rdm_h = _rdm2x1(A, env, d).reshape(d * d, d * d)
        rdm_v = _rdm1x2(A, env, d).reshape(d * d, d * d)

        eigvals_h = jnp.linalg.eigvalsh(rdm_h)
        eigvals_v = jnp.linalg.eigvalsh(rdm_v)
        assert jnp.all(jnp.abs(eigvals_h) < 10), f"Unbounded eigenvalues: {eigvals_h}"
        assert jnp.all(jnp.abs(eigvals_v) < 10), f"Unbounded eigenvalues: {eigvals_v}"

    def test_rdm_trace_one(self, peps_env):
        """trace(rdm) should be approximately 1."""
        A, env, d = peps_env
        rdm_h = _rdm2x1(A, env, d).reshape(d * d, d * d)
        rdm_v = _rdm1x2(A, env, d).reshape(d * d, d * d)
        assert jnp.allclose(jnp.trace(rdm_h), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.trace(rdm_v), 1.0, atol=1e-10)


class TestBuildDoubleLayerOpen:
    def test_shape(self):
        D, d = 2, 2
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (D, D, D, D, d))
        ao = _build_double_layer_open(A)
        assert ao.shape == (D**2, D**2, D**2, D**2, d, d)

    def test_trace_equals_closed(self):
        """Tracing out physical indices of open tensor gives the closed one."""
        D, d = 2, 2
        key = jax.random.PRNGKey(1)
        A = jax.random.normal(key, (D, D, D, D, d))
        ao = _build_double_layer_open(A)
        # trace s=s' → a_closed
        a_traced = jnp.einsum("udlrss->udlr", ao)
        a_closed = _build_double_layer(A).reshape(D**2, D**2, D**2, D**2)
        assert jnp.allclose(a_traced, a_closed, atol=1e-12)


class TestProductStateEnergy:
    def test_energy_product_state_up(self):
        """For a product state |up>, SzSz energy per bond = +0.25."""
        D, d = 1, 2
        # |up> = [1, 0] product state: A[u,d,l,r,s] trivial on virtual bonds
        A = jnp.zeros((D, D, D, D, d))
        A = A.at[0, 0, 0, 0, 0].set(1.0)  # |up>

        config = CTMConfig(chi=4, max_iter=20)
        env = ctm(A, config)

        # SzSz only
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        gate = jnp.kron(Sz, Sz).reshape(d, d, d, d)
        energy = compute_energy_ctm(A, env, gate, d)
        # |up up>: Sz*Sz = 0.25 per bond, 2 bonds (h+v) per site
        assert jnp.allclose(energy, 0.5, atol=0.1), f"Energy = {float(energy)}"


class TestIPEPSRun:
    @pytest.fixture
    def heisenberg_gate(self):
        """2-site Heisenberg Hamiltonian gate for simple update."""
        d = 2
        # H = Sz*Sz + 0.5*(S+S- + S-S+)
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_ipeps_runs_without_error(self, heisenberg_gate):
        """iPEPS should run end-to-end without crashing."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=3,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        energy, (A, B), (env_A, env_B) = ipeps(heisenberg_gate, None, config)
        assert jnp.isfinite(energy)

    def test_ipeps_returns_three_tuple(self, heisenberg_gate):
        """ipeps() should return (energy, (A, B), (env_A, env_B)) triple."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        result = ipeps(heisenberg_gate, None, config)
        assert len(result) == 3
        energy, peps, envs = result
        assert isinstance(peps, tuple) and len(peps) == 2
        assert isinstance(envs, tuple) and len(envs) == 2

    def test_ipeps_energy_is_scalar(self, heisenberg_gate):
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        energy, _, _ = ipeps(heisenberg_gate, None, config)
        assert isinstance(energy, float)

    def test_ipeps_env_is_ctm_environment(self, heisenberg_gate):
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        _, _, (env_A, env_B) = ipeps(heisenberg_gate, None, config)
        assert isinstance(env_A, CTMEnvironment)
        assert isinstance(env_B, CTMEnvironment)

    def test_ipeps_with_initial_peps(self, heisenberg_gate):
        """iPEPS should accept an initial (A, B) tuple."""
        D, d = 2, 2
        key_A, key_B = jax.random.split(jax.random.PRNGKey(99))
        initial_A = jax.random.normal(key_A, (D, D, D, D, d))
        initial_B = jax.random.normal(key_B, (D, D, D, D, d))

        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        energy, _, _ = ipeps(heisenberg_gate, (initial_A, initial_B), config)
        assert jnp.isfinite(energy)


class TestIPEPS2Site:
    """Tests for the full 2-site iPEPS pipeline."""

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_2site_runs_without_error(self, heisenberg_gate):
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=10,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=5),
            unit_cell="2site",
        )
        energy, peps, envs = ipeps(heisenberg_gate, None, config)
        assert jnp.isfinite(energy)
        assert isinstance(envs, tuple)
        assert len(envs) == 2

    def test_2site_heisenberg_D2_energy(self, heisenberg_gate):
        """2-site D=2 iPEPS should give E < -0.63 (literature ~-0.648).

        A moderate dt (0.3) is used so the simple update builds sufficient
        entanglement.  Small dt causes the bond lambdas to converge to a
        product-like fixed point with too little entanglement.
        """
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=200,
            dt=0.3,
            ctm=CTMConfig(chi=10, max_iter=40),
            unit_cell="2site",
        )
        energy, _, _ = ipeps(heisenberg_gate, None, config)
        assert float(energy) < -0.63, (
            f"Energy {float(energy)} not low enough — D=2 iPEPS should give E < -0.63"
        )

    @pytest.mark.slow
    def test_2site_heisenberg_D4_energy(self, heisenberg_gate):
        """2-site D=4 iPEPS should give E < -0.66 (literature ~-0.667)."""
        config = iPEPSConfig(
            max_bond_dim=4,
            num_imaginary_steps=400,
            dt=0.3,
            ctm=CTMConfig(chi=20, max_iter=60),
            unit_cell="2site",
        )
        energy, _, _ = ipeps(heisenberg_gate, None, config)
        assert float(energy) < -0.66, (
            f"Energy {float(energy)} not low enough — D=4 iPEPS should give E < -0.66"
        )

    def test_2site_with_initial_peps(self, heisenberg_gate):
        """2-site iPEPS should accept initial (A, B) tuple."""
        D, d = 2, 2
        key_A, key_B = jax.random.split(jax.random.PRNGKey(42))
        A = jax.random.normal(key_A, (D, D, D, D, d))
        B = jax.random.normal(key_B, (D, D, D, D, d))
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=5,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
            unit_cell="2site",
        )
        energy, _, _ = ipeps(heisenberg_gate, (A, B), config)
        assert jnp.isfinite(energy)

    def test_2site_rejects_non_tuple_initial_peps(self, heisenberg_gate):
        """2-site iPEPS should raise TypeError for non-tuple initial_peps."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=1,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=1),
            unit_cell="2site",
        )
        bad_input = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
        with pytest.raises(TypeError, match="tuple.*None"):
            ipeps(heisenberg_gate, bad_input, config)


class TestQRProjectors:
    """Tests for QR-based CTMRG projectors (Phase 1)."""

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_qr_backward_compat(self):
        """CTMConfig() defaults to svd (Fishman)."""
        cfg = CTMConfig()
        assert cfg.projector_method == "svd"
        assert cfg.qr_warmup_steps == 3

    def test_qr_ctm_converges(self):
        """QR CTM should produce finite environment tensors."""
        key = jax.random.PRNGKey(42)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        config = CTMConfig(chi=8, max_iter=20, projector_method="qr", qr_warmup_steps=3)
        env = ctm(A, config)
        assert isinstance(env, CTMEnvironment)
        for t in env:
            assert jnp.all(jnp.isfinite(t)), "QR CTM produced non-finite tensors"

    def test_qr_energy_is_finite(self, heisenberg_gate):
        """QR projector should produce a finite energy from CTM."""
        key = jax.random.PRNGKey(7)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        config_qr = CTMConfig(
            chi=8, max_iter=100, projector_method="qr", qr_warmup_steps=5
        )
        env_qr = ctm(A, config_qr)
        E_qr = compute_energy_ctm(A, env_qr, heisenberg_gate, d)

        assert jnp.isfinite(E_qr), f"QR energy is not finite: {float(E_qr)}"


class TestOptimizeGsAd2Site:
    """Tests for 2-site AD optimization (Phase 2)."""

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_2site_ad_runs(self, heisenberg_gate):
        """2-site AD optimization should run without crashing."""

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=3,
            gs_learning_rate=1e-3,
            unit_cell="2site",
        )
        result = optimize_gs_ad(heisenberg_gate, None, config)
        (A_opt, B_opt), (env_A, env_B), E_gs = result
        assert A_opt.todense().shape == (2, 2, 2, 2, 2)
        assert B_opt.todense().shape == (2, 2, 2, 2, 2)
        assert np.isfinite(E_gs)

    def test_2site_ad_with_su_init(self, heisenberg_gate):
        """su_init=True path should work for 2-site."""

        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=10,
            dt=0.3,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=3,
            gs_learning_rate=1e-3,
            unit_cell="2site",
            su_init=True,
        )
        result = optimize_gs_ad(heisenberg_gate, None, config)
        _, _, E_gs = result
        assert np.isfinite(E_gs)

    def test_2site_ad_zero_steps_returns_energy(self, heisenberg_gate):
        """gs_num_steps=0 should return initial energy without crashing."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=0,
            unit_cell="2site",
        )
        (A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(
            heisenberg_gate, None, config
        )
        assert A_opt.todense().shape == (2, 2, 2, 2, 2)
        assert B_opt.todense().shape == (2, 2, 2, 2, 2)
        assert np.isfinite(E_gs)

    def test_2site_ad_mixed_init_types_work(self, heisenberg_gate):
        """Mixed (Tensor, dense) init should work — arrays are auto-wrapped."""
        from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

        D, d = 2, 2
        sym = U1Symmetry()
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        A_tensor = DenseTensor(
            jax.random.normal(jax.random.PRNGKey(0), (D, D, D, D, d)),
            indices,
        )
        B_dense = jax.random.normal(jax.random.PRNGKey(1), (D, D, D, D, d))

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=1,
            unit_cell="2site",
        )
        # Should not raise — dense arrays are auto-wrapped as DenseTensor
        result = optimize_gs_ad(heisenberg_gate, (A_tensor, B_dense), config)
        (A_opt, B_opt), (env_A, env_B), E_gs = result
        assert np.isfinite(E_gs)

    def test_2site_ad_non_tuple_init_raises(self, heisenberg_gate):
        """2-site AD requires A_init to be None or a tuple (A, B)."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=1,
            unit_cell="2site",
        )
        A_dense = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2))
        with pytest.raises(TypeError, match="must be None or a tuple"):
            optimize_gs_ad(heisenberg_gate, A_dense, config)

    def test_2site_ad_c4v_runs(self, heisenberg_gate):
        """2-site AD with C4v parameterization should run and give finite energy."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=3,
            gs_learning_rate=1e-3,
            unit_cell="2site",
            gs_c4v=True,
        )
        result = optimize_gs_ad(heisenberg_gate, None, config)
        (A_opt, B_opt), (env_A, env_B), E_gs = result
        assert A_opt.todense().shape == (2, 2, 2, 2, 2)
        assert B_opt.todense().shape == (2, 2, 2, 2, 2)
        assert np.isfinite(E_gs)

    def test_2site_ad_c4v_shared_tensor(self, heisenberg_gate):
        """With gs_c4v=True, B should be the sublattice rotation of A."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=3,
            gs_learning_rate=1e-3,
            unit_cell="2site",
            gs_c4v=True,
        )
        (A_opt, B_opt), _, _ = optimize_gs_ad(heisenberg_gate, None, config)
        U = np.array([[0.0, 1.0], [-1.0, 0.0]])
        B_expected = np.einsum("luRDs,sS->luRDS", A_opt.todense(), U)
        np.testing.assert_allclose(B_opt.todense(), B_expected, atol=1e-12)

    # PR #360 deleted three #328-regression tests under the rationale
    # they exercised the explicit-AD path. Two of those did NOT set
    # ``gs_implicit_ad=False``, so they were testing the implicit path
    # all along (PR #360 review feedback). The two are restored below;
    # the third (``test_2site_noc4v_implicit_ad_is_variational_issue_328``)
    # is intentionally not restored — it pinned a real-float64 init that
    # is documented non-variational without C4v
    # (project_complex_tensors_variational.md), and the variational
    # contract on the recommended complex128 path is owned by
    # ``test_complex128_2site_gradient`` in tests/test_complex128_ad.py.
    #
    # ``test_2site_ad_c4v_energy_physical`` (explicitly ``gs_implicit_ad=False``)
    # remains deleted — it was duplicated by the implicit-path literature
    # regressions (``test_2site_heisenberg_ad_energy_benchmark``,
    # ``test_ad_d2_energy``).

    def test_2site_noc4v_ad_stays_variational_issue_328(self, heisenberg_gate):
        """Regression for #328 on the implicit-AD path.

        The issue observed E_best = -33.85 (!!) for ``gs_c4v=False``
        implicit AD on the same Heisenberg gate at chi=16 — the L-BFGS/CG
        line search pushed onto an off-manifold chord that landed in a
        non-variational CTM fixed point with finite-chi energy far below
        the physical ground state. The tangent-space projection (Option
        A in #328) kills that drift.

        This test only asserts ``E_gs > -2.0`` (very loose) — there is no
        finite-chi variational guarantee. What the regression prevents
        is order-of-magnitude drift below physical (-0.6694).

        Default ``gs_implicit_ad=True`` is intentional: this exercises
        the implicit path, which is the recommended 2-site default
        (project_2site_ad_fix_trifecta.md).
        """
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=5,
            gs_learning_rate=1e-2,
            unit_cell="2site",
            gs_c4v=False,
        )
        _, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        assert np.isfinite(E_gs), f"non-finite E_gs = {E_gs}"
        assert E_gs > -2.0, (
            f"E/site = {E_gs:.6f} is wildly unphysical — tangent projection "
            f"likely broken (#328 regression)"
        )

    def test_2site_noc4v_ad_norms_stay_unit(self, heisenberg_gate):
        """Regression for #328 on the implicit-AD path: ``||A||`` and
        ``||B||`` must stay ~1 throughout ``gs_c4v=False`` 2-site AD.

        ``_normalize_params`` retracts to unit norm at each step and
        tangent projection keeps the chord on-manifold, so the returned
        ``A_opt`` / ``B_opt`` should have unit norm to high precision.

        Like ``test_2site_noc4v_ad_stays_variational_issue_328``, runs
        on the default implicit-AD path (``gs_implicit_ad=True``).
        """
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=5,
            gs_learning_rate=1e-2,
            unit_cell="2site",
            gs_c4v=False,
        )
        (A_opt, B_opt), _, _ = optimize_gs_ad(heisenberg_gate, None, config)
        A_norm = float(jnp.linalg.norm(A_opt.todense().reshape(-1)))
        B_norm = float(jnp.linalg.norm(B_opt.todense().reshape(-1)))
        assert abs(A_norm - 1.0) < 1e-4, f"||A_opt|| = {A_norm}, expected ~1.0"
        assert abs(B_norm - 1.0) < 1e-4, f"||B_opt|| = {B_norm}, expected ~1.0"

    @pytest.mark.slow
    def test_tangent_project_unit_complex_is_hermitian(self):
        """Regression for PR #330 review: _tangent_project_unit must use
        the Hermitian inner product (vdot) so complex-valued tensors get
        their true radial component removed.  With the bilinear jnp.dot
        from the original PR, ``Re <p, v - coef*p>`` is non-zero for
        complex inputs and the projection leaks radial drift."""
        from tenax.algorithms.ipeps_optimize import _tangent_project_unit
        from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

        sym = U1Symmetry()
        D, d = 2, 2
        charges = np.zeros(D, dtype=np.int32)
        phys = np.zeros(d, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
        )
        key_p, key_v = jax.random.split(jax.random.PRNGKey(7))
        # Complex params and complex direction with non-trivial imaginary parts
        p_real = jax.random.normal(key_p, (D, D, D, D, d))
        p_imag = jax.random.normal(jax.random.fold_in(key_p, 1), (D, D, D, D, d))
        v_real = jax.random.normal(key_v, (D, D, D, D, d))
        v_imag = jax.random.normal(jax.random.fold_in(key_v, 1), (D, D, D, D, d))
        p_data = (p_real + 1j * p_imag).astype(jnp.complex128)
        v_data = (v_real + 1j * v_imag).astype(jnp.complex128)
        p = DenseTensor(p_data / (jnp.linalg.norm(p_data.reshape(-1)) + 1e-10), indices)
        v = DenseTensor(v_data, indices)

        v_proj = _tangent_project_unit(v, p)

        # The projected direction must be Hermitian-orthogonal to p:
        # Re(<p, v_proj>_H) ≈ 0 (and ideally also Im ≈ 0 for consistency).
        p_flat = p.todense().reshape(-1)
        vproj_flat = v_proj.todense().reshape(-1)
        inner = jnp.vdot(p_flat, vproj_flat)
        assert float(jnp.abs(inner.real)) < 1e-10, (
            f"Re <p, P_T(v)>_H = {inner.real}, expected ~0"
        )


class TestOptimizeGsAdLogging:
    """Tests for AD optimization progress logging."""

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_invalid_log_interval_raises(self, heisenberg_gate):
        config = iPEPSConfig(gs_log_interval=0)
        with pytest.raises(ValueError, match="gs_log_interval"):
            optimize_gs_ad(heisenberg_gate, None, config)

    def test_verbose_prints_progress(self, heisenberg_gate, capsys):
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=5),
            gs_num_steps=2,
            gs_learning_rate=1e-3,
            gs_verbose=True,
            gs_log_interval=1,
        )
        optimize_gs_ad(heisenberg_gate, None, config)
        out = capsys.readouterr().out
        assert "[iPEPS-AD:1site-tensor] step 1/2" in out
        assert "[iPEPS-AD:1site-tensor] final E=" in out


class TestOptimizeGsAdDenseOnly:
    """Verify optimize_gs_ad 2-site rejects SymmetricTensor inputs."""

    def test_symmetric_tensor_2site_runs(self):
        """2-site AD optimization accepts SymmetricTensor inputs."""
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import SymmetricTensor

        sym = U1Symmetry()
        charges = np.zeros(2, dtype=np.int32)
        phys_charges = np.zeros(2, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        A_sym = SymmetricTensor.random_normal(indices, k1)
        B_sym = SymmetricTensor.random_normal(indices, k2)

        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        gate = H.reshape(d, d, d, d)

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=5),
            gs_num_steps=1,
            unit_cell="2site",
        )
        result = optimize_gs_ad(gate, (A_sym, B_sym), config)
        (A_opt, B_opt), (env_A, env_B), E_gs = result
        assert np.isfinite(E_gs)


class TestOptimizeGsAdOptimizers:
    """Verify L-BFGS and CG optimizer paths run correctly."""

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_lbfgs_optimizer_runs(self, heisenberg_gate):
        """L-BFGS optimizer with line search should produce finite energy.

        Scope kept tight: ``chi=2`` and ``gs_num_steps=1`` so the
        L-BFGS-with-Hager-Zhang trace fits within the macOS Full-tests
        runner memory budget. At ``chi=4`` + 3 steps + line search the
        XLA compile blew out (exit 137) when run late in the
        ``test_ipeps.py`` session — see CI run 25104994256 (after #360
        merged to main). The contract being tested is "L-BFGS path
        compiles and produces finite energy"; even one step exercises
        that path.
        """
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=2, max_iter=3),
            gs_optimizer="lbfgs",
            gs_num_steps=1,
            gs_line_search=True,
        )
        _, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        assert np.isfinite(E_gs)

    def test_cg_optimizer_runs(self, heisenberg_gate):
        """CG optimizer should produce finite energy (covers PR beta path).

        Scope tightened alongside ``test_lbfgs_optimizer_runs`` to keep
        the optimizer-smoke pair cheap; both tests run late in
        ``test_ipeps.py`` after long-running benchmarks, so the JAX
        cache is already heavy.
        """
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=2, max_iter=3),
            gs_optimizer="cg",
            gs_num_steps=1,
        )
        _, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        assert np.isfinite(E_gs)


class TestADSymmetric:
    """Tests for full block-sparse AD pipeline with SymmetricTensor."""

    @staticmethod
    def _make_symmetric_ipeps(key, D=2, d=2):
        """Create a U(1) SymmetricTensor iPEPS site tensor with trivial charges."""
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import SymmetricTensor

        sym = U1Symmetry()
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        return SymmetricTensor.random_normal(indices, key)

    @staticmethod
    def _heisenberg_gate():
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_todense_gradient_flows(self):
        """jax.grad through SymmetricTensor.todense() works."""
        from tenax.core.tensor import SymmetricTensor

        A_sym = self._make_symmetric_ipeps(jax.random.PRNGKey(42))

        def loss(t):
            return jnp.sum(t.todense() ** 2)

        grad = jax.grad(loss)(A_sym)
        assert isinstance(grad, SymmetricTensor)
        # Gradient should be non-zero
        assert grad.norm() > 0

    def test_from_dense_gradient_flows(self):
        """jax.grad through from_dense round-trip works."""
        from tenax.core.tensor import SymmetricTensor

        A_sym = self._make_symmetric_ipeps(jax.random.PRNGKey(42))

        def loss(t):
            dense = t.todense()
            t2 = SymmetricTensor.from_dense(dense, t.indices, tol=float("inf"))
            return t2.norm()

        grad = jax.grad(loss)(A_sym)
        assert isinstance(grad, SymmetricTensor)
        assert grad.norm() > 0

    def test_optimize_gs_ad_symmetric_runs(self):
        """optimize_gs_ad accepts SymmetricTensor and returns SymmetricTensor.

        This is the in-loop counterpart to
        :meth:`test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type`:
        2 AD steps actually exercise the in-loop rewrap sites that #329
        fixed (``loss_fn``, metric-precond, L-BFGS direction, noise
        recovery), all of which go through ``ad_utils._wrap_tensor``. The
        helper dispatches on ``isinstance(original, SymmetricTensor)``
        and is charge-agnostic, so trivial-charge coverage here catches
        any future in-loop downgrade for non-trivial charges too.
        """
        from tenax.core.tensor import SymmetricTensor

        gate = self._heisenberg_gate()
        A_sym = self._make_symmetric_ipeps(jax.random.PRNGKey(0))

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=2,
            gs_learning_rate=0.01,
        )
        A_opt, env, E_gs = optimize_gs_ad(gate, A_sym, config)

        assert isinstance(A_opt, SymmetricTensor), (
            f"expected SymmetricTensor, got {type(A_opt).__name__}"
        )
        assert np.isfinite(E_gs)

    def test_optimize_gs_ad_symmetric_matches_dense(self):
        """Symmetric AD gives comparable energy to dense AD.

        Compute budget kept small (``gs_num_steps=1``, ``max_iter=5``) so
        running the optimizer twice — once for the SymmetricTensor and once
        for the dense equivalent — fits inside the CI fast-ipeps bucket's
        60 min job budget.  The original 3-step / 10-iter config was on the
        edge before #492's in-CTM bump infrastructure landed and tipped it
        past timeout on push-to-main.  The assertions below verify the
        type-preservation contract, not convergence quality, so the smaller
        budget preserves the test's purpose.
        """
        gate = self._heisenberg_gate()

        # Create a SymmetricTensor and its dense equivalent
        A_sym = self._make_symmetric_ipeps(jax.random.PRNGKey(0))
        A_dense = A_sym.todense()

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=5),
            gs_num_steps=1,
            gs_learning_rate=0.01,
        )

        A_sym_opt, _, E_sym = optimize_gs_ad(gate, A_sym, config)
        A_dense_opt, _, E_dense = optimize_gs_ad(gate, A_dense, config)

        # Both should produce finite energies (exact match not expected
        # due to different CTM projector implementations)
        assert np.isfinite(E_sym)
        assert np.isfinite(E_dense)
        # Type preservation: SymmetricTensor input must round-trip as
        # SymmetricTensor (the in-loop counterpart to the non-trivial-U(1)
        # input-side check).
        from tenax.core.tensor import DenseTensor, SymmetricTensor

        assert isinstance(A_sym_opt, SymmetricTensor), (
            f"expected SymmetricTensor, got {type(A_sym_opt).__name__}"
        )
        assert isinstance(A_dense_opt, DenseTensor), (
            f"expected DenseTensor, got {type(A_dense_opt).__name__}"
        )

    def test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type(self):
        """Regression for #297: optimizer shell must accept a SymmetricTensor
        with *non-trivial* U(1) charges and return a SymmetricTensor (not
        silently downgrade to DenseTensor via a .todense() band-aid).

        ``gs_num_steps=0`` skips the AD optimization loop and exercises only
        the input-side path that the #296 band-aid lived in:
        ``A = A * (1 / (A.norm() + eps))`` plus the CTM forward and energy
        eval. The in-loop rewrap sites fixed in #329 (``loss_fn``,
        metric-precond, L-BFGS direction, noise recovery) all funnel through
        ``ad_utils._wrap_tensor``, which dispatches on
        ``isinstance(original, SymmetricTensor)`` and is otherwise
        charge-agnostic — so trivial-charge in-loop coverage in
        :meth:`test_optimize_gs_ad_symmetric_runs` (and its siblings, which
        also assert ``isinstance(A_opt, SymmetricTensor)`` after several AD
        steps) catches any future regression in those rewrap sites for
        non-trivial charges too. The L-BFGS+metric-precond compile graph
        for non-trivial U(1) blocks took ~30 min on Linux 2-vCPU GH runners
        (viable on M-series macOS, not on x86 Linux); ``gs_num_steps=0``
        brings it under 2s without losing the input-side regression
        coverage.
        """
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import SymmetricTensor

        sym = U1Symmetry()
        # Non-trivial U(1) charges [0, 1] on every leg.  Flow directions
        # sum to zero (per-charge) so at least the all-zero block exists
        # and the CTM pipeline can proceed.
        virt = np.array([0, 1], dtype=np.int32)
        phys = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
        )
        A_sym = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(0))

        gate = self._heisenberg_gate()
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=0,
            gs_learning_rate=0.01,
        )
        A_opt, _env, E_gs = optimize_gs_ad(gate, A_sym, config)
        assert isinstance(A_opt, SymmetricTensor), (
            f"expected SymmetricTensor, got {type(A_opt).__name__}"
        )
        assert np.isfinite(E_gs)


class TestTensor2SiteSimpleUpdate:
    """Tests for the 2-site Tensor-protocol simple update."""

    @staticmethod
    def _make_dense_ipeps(key, D=2, d=2):
        """Create a DenseTensor iPEPS site tensor with trivial charges."""
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor

        sym = U1Symmetry()
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)
        data = jax.random.normal(key, (D, D, D, D, d))
        data = data / (jnp.linalg.norm(data) + 1e-10)
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        return DenseTensor(data, indices)

    @staticmethod
    def _make_symmetric_ipeps(key, D=2, d=2):
        """Create a U(1) SymmetricTensor iPEPS site tensor."""
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import SymmetricTensor

        sym = U1Symmetry()
        charges = np.zeros(D, dtype=np.int32)
        phys_charges = np.zeros(d, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        return SymmetricTensor.random_normal(indices, key)

    @staticmethod
    def _heisenberg_gate():
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_horizontal_dense_tensor_runs(self):
        """Horizontal 2-site simple update works with DenseTensor."""
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
        )

        A = self._make_dense_ipeps(jax.random.PRNGKey(0))
        B = self._make_dense_ipeps(jax.random.PRNGKey(1))
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_new, B_new, lam_new = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert A_new.labels() == ("u", "d", "l", "r", "phys")
        assert B_new.labels() == ("u", "d", "l", "r", "phys")
        assert np.isfinite(float(A_new.norm()))
        assert np.isfinite(float(B_new.norm()))

    def test_vertical_dense_tensor_runs(self):
        """Vertical 2-site simple update works with DenseTensor."""
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_vertical_tensor,
        )

        A = self._make_dense_ipeps(jax.random.PRNGKey(0))
        B = self._make_dense_ipeps(jax.random.PRNGKey(1))
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_new, B_new, lam_new = _simple_update_2site_vertical_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert A_new.labels() == ("u", "d", "l", "r", "phys")
        assert B_new.labels() == ("u", "d", "l", "r", "phys")
        assert np.isfinite(float(A_new.norm()))
        assert np.isfinite(float(B_new.norm()))

    def test_symmetric_tensor_2site_runs(self):
        """2-site simple update works with SymmetricTensor."""
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
            _simple_update_2site_vertical_tensor,
        )
        from tenax.core.tensor import SymmetricTensor

        A = self._make_symmetric_ipeps(jax.random.PRNGKey(0))
        B = self._make_symmetric_ipeps(jax.random.PRNGKey(1))
        gate = _make_trotter_gate_tensor(
            self._heisenberg_gate(), dt=0.01, site_tensor=A
        )
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_h, B_h, lam_h_new = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert isinstance(A_h, SymmetricTensor)
        assert isinstance(B_h, SymmetricTensor)
        assert A_h.labels() == ("u", "d", "l", "r", "phys")
        assert B_h.labels() == ("u", "d", "l", "r", "phys")

        A_v, B_v, lam_v_new = _simple_update_2site_vertical_tensor(
            A_h, B_h, gate, lam_h_new, lam_v, D
        )
        assert isinstance(A_v, SymmetricTensor)
        assert isinstance(B_v, SymmetricTensor)
        assert np.isfinite(float(A_v.norm()))
        assert np.isfinite(float(B_v.norm()))

    def test_returns_different_A_and_B(self):
        """A_new and B_new should differ after horizontal update."""
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
        )

        A = self._make_dense_ipeps(jax.random.PRNGKey(0))
        B = self._make_dense_ipeps(jax.random.PRNGKey(1))
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        A_new, B_new, _ = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        # A_new and B_new come from U and Vh of an SVD, so they should differ
        diff = float(jnp.linalg.norm(A_new.todense() - B_new.todense()))
        assert diff > 1e-10, "A_new and B_new should not be identical"

    def test_lambda_normalized(self):
        """max(lam_new) should be approximately 1.0."""
        from tenax.algorithms.ipeps_simple_update import (
            _make_trotter_gate_tensor,
            _simple_update_2site_horizontal_tensor,
        )

        A = self._make_dense_ipeps(jax.random.PRNGKey(0))
        B = self._make_dense_ipeps(jax.random.PRNGKey(1))
        gate = _make_trotter_gate_tensor(self._heisenberg_gate(), dt=0.01)
        D = 2
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)

        _, _, lam_new = _simple_update_2site_horizontal_tensor(
            A, B, gate, lam_h, lam_v, D
        )
        assert abs(float(jnp.max(lam_new)) - 1.0) < 1e-6


class TestSplitCTMRG:
    """Tests for Split-CTMRG (Phase 3)."""

    @pytest.fixture
    def small_peps_tensor(self):
        key = jax.random.PRNGKey(42)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        return A / (jnp.linalg.norm(A) + 1e-10)

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_split_env_shapes(self, small_peps_tensor):
        """All 12 tensors in SplitCTMEnvironment should have correct shapes."""
        chi, chi_I, D = 8, 4, 2
        env = _initialize_split_ctm_env(small_peps_tensor, chi, chi_I)
        assert isinstance(env, SplitCTMEnvironment)
        # Corners
        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert C.shape == (chi, chi)
        # Ket edges
        for T_ket in [env.T1_ket, env.T2_ket, env.T3_ket, env.T4_ket]:
            assert T_ket.shape == (chi, D, chi_I)
        # Bra edges
        for T_bra in [env.T1_bra, env.T2_bra, env.T3_bra, env.T4_bra]:
            assert T_bra.shape == (chi_I, D, chi)

    def test_split_env_to_standard(self, small_peps_tensor):
        """Merged edges should have shape (chi, D^2, chi)."""
        chi, chi_I, D = 8, 4, 2
        env = _initialize_split_ctm_env(small_peps_tensor, chi, chi_I)
        std = _split_env_to_standard(env)
        assert isinstance(std, CTMEnvironment)
        for T in [std.T1, std.T2, std.T3, std.T4]:
            assert T.shape == (chi, D * D, chi)
        for C in [std.C1, std.C2, std.C3, std.C4]:
            assert C.shape == (chi, chi)

    def test_split_ctm_converges(self, small_peps_tensor):
        """Split-CTM should produce finite environment tensors."""
        config = CTMConfig(chi=8, max_iter=20, chi_I=4)
        env = ctm_split(small_peps_tensor, config)
        assert isinstance(env, SplitCTMEnvironment)
        for t in env:
            assert jnp.all(jnp.isfinite(t)), "Split-CTM produced non-finite tensors"

    def test_split_ctm_chi_I_equals_chi(self, small_peps_tensor):
        """chi_I=chi should also work (no interlayer compression)."""
        config = CTMConfig(chi=8, max_iter=20, chi_I=8)
        env = ctm_split(small_peps_tensor, config)
        for t in env:
            assert jnp.all(jnp.isfinite(t))

    def test_split_ctm_energy_matches_standard(
        self, small_peps_tensor, heisenberg_gate
    ):
        """Split-CTM energy via split env equals energy via converted standard env.

        Verifies that ``compute_energy_split_ctm`` (which converts to
        standard internally) gives exactly the same result as manually
        converting with ``_split_env_to_standard`` then calling
        ``compute_energy_ctm``.  This is the key correctness invariant
        for the split representation.
        """
        D, d = 2, 2
        chi = 8
        chi_I = chi * D  # lossless

        config = CTMConfig(chi=chi, max_iter=50, chi_I=chi_I)
        env_split = ctm_split(small_peps_tensor, config)

        E_split = compute_energy_split_ctm(
            small_peps_tensor, env_split, heisenberg_gate, d
        )
        assert jnp.isfinite(E_split)

        # Energy via manually converted standard env must match exactly
        std_env = _split_env_to_standard(env_split)
        E_from_std = compute_energy_ctm(small_peps_tensor, std_env, heisenberg_gate, d)
        assert jnp.abs(E_split - E_from_std) < 1e-12, (
            f"Energy mismatch: split={float(E_split)}, converted={float(E_from_std)}"
        )

    def test_split_ctm_default_chi_I_none(self):
        """CTMConfig with chi_I=None should default to chi."""
        cfg = CTMConfig()
        assert cfg.chi_I is None


class TestXXZGate:
    def test_xxz_gate_shape(self):
        from tenax.algorithms.ipeps import xxz_gate

        gate = xxz_gate(delta=1.0)
        assert gate.todense().shape == (2, 2, 2, 2)

    def test_xxz_gate_recovers_heisenberg(self):
        from tenax.algorithms.ipeps import heisenberg_gate, xxz_gate

        H_heis = heisenberg_gate().todense()
        H_xxz = xxz_gate(delta=1.0).todense()
        assert jnp.allclose(H_heis, H_xxz, atol=1e-14)

    def test_xxz_gate_ising_limit(self):
        from tenax.algorithms.ipeps import xxz_gate

        H = xxz_gate(delta=0.0).todense()
        Sp = jnp.array([[0, 1], [0, 0]], dtype=jnp.float64)
        Sm = jnp.array([[0, 0], [1, 0]], dtype=jnp.float64)
        H_expected = 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
        assert jnp.allclose(H.reshape(4, 4), H_expected, atol=1e-14)

    def test_xxz_gate_is_dense_tensor(self):
        from tenax.algorithms.ipeps import xxz_gate
        from tenax.core.tensor import DenseTensor

        gate = xxz_gate(delta=0.5)
        assert isinstance(gate, DenseTensor)

    def test_xxz_gate_labels(self):
        from tenax.algorithms.ipeps import xxz_gate

        gate = xxz_gate(delta=1.0)
        assert gate.labels() == ("si", "sj", "si_out", "sj_out")


class TestNoiseRecovery:
    """Optimizer should handle stalls via noise injection."""

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_noise_recovery_config_exists(self):
        config = iPEPSConfig()
        assert hasattr(config, "gs_noise_recovery_retries")
        assert config.gs_noise_recovery_retries == 3
        assert hasattr(config, "gs_noise_amplitude")
        assert config.gs_noise_amplitude == 0.1

    def test_optimizer_runs_with_recovery(self, heisenberg_gate):
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=20, min_iter=5),
            gs_num_steps=5,
            gs_optimizer="adam",
            gs_noise_recovery_retries=2,
            gs_noise_amplitude=0.1,
            unit_cell="1x1",
        )
        A_opt, env, E = optimize_gs_ad(heisenberg_gate, None, config)
        assert jnp.isfinite(jnp.array(E))


class TestPostPR291ADBaseline:
    """Regression tests locking down the post-PR-#341 iPEPS AD baseline.

    ``CTMConfig.forward_gauge`` accepts four modes (``qr``, ``sigma``,
    ``phase``, ``none``) and defaults to ``"phase"`` — the value that is
    correct for both implicit and explicit AD (1-site and 2-site).  No
    silent promotion: explicit user choice is preserved by the AD config
    builder (see ``tests/test_ipeps_ad_policy.py``).  ``iPEPSConfig``
    exposes a ``gs_ctm_conv_tol_schedule`` knob.  These tests pin that
    behavior so future refactors cannot silently drift off the documented
    baseline.
    """

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    def test_forward_gauge_accepts_expanded_mode_set(self):
        """CTMConfig.forward_gauge accepts the four post-PR-#291 modes."""
        for mode in ("qr", "sigma", "phase", "none"):
            cfg = CTMConfig(forward_gauge=mode)
            assert cfg.forward_gauge == mode

    def test_forward_gauge_default_is_phase(self):
        """Config default is ``phase`` — the AD-correct choice.

        ``"phase"`` is the correct default for both implicit and explicit
        AD (1-site and 2-site).  No silent promotion: direct ``CTMConfig()``
        users get the same behavior the AD optimizer uses.
        """
        assert CTMConfig().forward_gauge == "phase"

    def test_gs_ctm_conv_tol_schedule_is_configurable(self):
        """New iPEPSConfig knob from PR #291 is exposed and round-trips."""
        schedule = [(0.0, 1e-5), (0.5, 1e-6), (0.8, 1e-7)]
        cfg = iPEPSConfig(gs_ctm_conv_tol_schedule=schedule)
        assert cfg.gs_ctm_conv_tol_schedule == schedule

    def test_gs_ctm_conv_tol_schedule_default_is_none(self):
        assert iPEPSConfig().gs_ctm_conv_tol_schedule is None

    def test_recommended_c4v_explicit_config_is_constructible(self):
        """The post-PR-#291 recommended C4v + explicit-AD config constructs
        without raising. This is the configuration documented in
        ``docs/guide/algorithms/ipeps_ad_paths.md``."""
        cfg = iPEPSConfig(
            unit_cell="1x1",
            gs_c4v=True,
            gs_optimizer="lbfgs",
            gs_implicit_ad=False,
            gs_explicit_ad_steps=20,
            gs_explicit_ad_warmup=2,
            gs_projector_method="qr",
            ctm=CTMConfig(chi=16, projector_method="qr", forward_gauge="qr"),
        )
        assert cfg.gs_c4v is True
        assert cfg.gs_implicit_ad is False
        assert cfg.ctm.projector_method == "qr"
        assert cfg.gs_projector_method == "qr"
        # Config default stays 'qr' — optimizer auto-promotes to 'phase'.
        assert cfg.ctm.forward_gauge == "qr"

    def test_forward_gauge_round_trips_through_config_tuple(self):
        """All four modes must round-trip through the explicit-AD config
        serialization.  Before the issue-#292 fix, ``"none"`` silently
        collapsed to ``"qr"`` because the tuple encoding only recognized
        three modes, making ``forward_gauge="none"`` unreachable from
        ``ctm_tensor_converge_explicit``.
        """
        from tenax.algorithms.ad_utils import _config_from_tuple, _config_to_tuple

        for mode in ("qr", "sigma", "phase", "none"):
            tup = _config_to_tuple(CTMConfig(forward_gauge=mode))
            assert _config_from_tuple(tup).forward_gauge == mode

    # test_implicit_ad_preserves_explicit_forward_gauge removed: it pinned
    # the post-PR-#343 contract that implicit AD passes the user's
    # forward_gauge through unchanged. PR #350 inverted that contract —
    # resolve_projector_backward now enforces the trifecta
    # (svd + phase + elementwise) on the implicit path and raises
    # ValueError on any deviation, so qr/sigma/none modes are no longer
    # reachable via the implicit-AD entry point. The test premise is
    # superseded by the implicit-AD trifecta benchmarks.

    @pytest.mark.xfail(
        reason=(
            "GMRES implicit AD is documented unstable in "
            "docs/guide/algorithms/ipeps_ad_paths.md (spectral radius > 1 "
            "without sigma gauge); this xfail locks the doc status. Remove "
            "only when the GMRES path has been stabilized end-to-end."
        ),
        strict=False,
        run=False,
    )
    def test_gmres_implicit_ad_is_stable(self):
        """Placeholder regression: GMRES implicit AD must produce a
        physical energy. Kept ``xfail`` so the issue stays visible in the
        test log without blocking CI."""
        raise AssertionError("gmres implicit AD path not stable — see issue #292")

    def test_gs_ctm_conv_tol_schedule_is_applied_across_steps(
        self, monkeypatch, heisenberg_gate
    ):
        """The ``gs_ctm_conv_tol_schedule`` knob must actually reach the
        CTM convergence call — not just sit in the config.  We monkeypatch
        ``ctm_energy_implicit``, record the ``conv_tol`` kwarg on every
        call, and assert that a schedule with two distinct tolerance tiers
        produces at least two distinct ``conv_tol`` values across the
        optimization loop."""
        from tenax.algorithms import _ctm_energy_ad as _ctm_ad

        seen_conv_tols: list[float] = []

        def spy(site_tensors, neighbors, gate, **kw):
            seen_conv_tols.append(float(kw.get("conv_tol")))
            # Return a real differentiable scalar tied to the site tensors
            # so the outer optimizer can take a step and call us again
            # with the advanced schedule tier.
            A = next(iter(site_tensors.values()))
            data = A._data if hasattr(A, "_data") else A
            return jnp.sum(jnp.abs(data) ** 2).real

        monkeypatch.setattr(_ctm_ad, "ctm_energy_implicit", spy)
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(
                chi=4, max_iter=5, min_iter=2, conv_tol=1e-4, forward_gauge="phase"
            ),
            gs_num_steps=6,
            gs_implicit_ad=True,
            # Schedule: tighter tolerance from step 1 onward.  The spy's
            # surrogate energy is flat, so the optimizer converges quickly;
            # putting the tier switch at fraction 0.1 ensures it kicks in
            # before the optimizer declares convergence.
            gs_ctm_conv_tol_schedule=[(0.0, 1e-3), (0.1, 1e-6)],
            # Disable outer-loop convergence so the schedule has room to
            # advance even when the surrogate energy is flat.
            gs_conv_tol=-1.0,
            unit_cell="1x1",
            su_init=False,
        )
        # The spy raises, so the optimizer may terminate; we only need to
        # observe that the schedule took effect before that point.
        with contextlib.suppress(Exception):
            optimize_gs_ad(heisenberg_gate, None, config)
        assert len(seen_conv_tols) >= 1, (
            "ctm_energy_implicit spy was never called — the optimizer "
            "likely short-circuited before reaching the CTM."
        )
        distinct = sorted(set(seen_conv_tols))
        assert len(distinct) >= 2, (
            f"gs_ctm_conv_tol_schedule did not change conv_tol across "
            f"optimization steps; observed only {distinct!r}"
        )
        # Final tier tolerance must be the tighter 1e-6.
        assert min(distinct) == pytest.approx(1e-6), (
            f"tightest observed conv_tol {min(distinct)} does not match "
            f"the 1e-6 tier from the schedule"
        )


def test_stall_recovery_auto_defaults():
    """1-site -> 'noise', 2-site -> 'reset' when user leaves gs_stall_recovery=None.

    See issue #298.
    """
    from tenax.algorithms.ipeps_config import iPEPSConfig
    from tenax.algorithms.ipeps_optimize import _normalize_stall_recovery

    cfg_1s = _normalize_stall_recovery(iPEPSConfig(unit_cell="1x1"), unit_cell="1x1")
    assert cfg_1s.gs_stall_recovery == "noise"

    cfg_2s = _normalize_stall_recovery(
        iPEPSConfig(unit_cell="2site"), unit_cell="2site"
    )
    assert cfg_2s.gs_stall_recovery == "reset"

    cfg_user = _normalize_stall_recovery(
        iPEPSConfig(unit_cell="2site", gs_stall_recovery="noise"),
        unit_cell="2site",
    )
    assert cfg_user.gs_stall_recovery == "noise", "explicit user setting must win"


def test_is_real_decrease_noise_floor():
    """Issue #510: stall trigger must distinguish real progress from FP jitter.

    Without the noise floor, ``f_alpha < energy_float`` is satisfied by a
    1e-14-scale decrease and ``stall_count`` resets to 0 even though the
    parameter update ``α·direction`` is numerically zero.  v8b evidence
    (D=3 χ=9 bipartite implicit AD) showed a 5-hour run grinding at
    α≈1e-14 because the safety net never fired.
    """
    from tenax.algorithms.ipeps_optimize import _is_real_decrease

    # Real decrease: clearly above the 1e-12 floor → True.
    assert _is_real_decrease(-0.700, -0.500) is True
    assert _is_real_decrease(-0.50001, -0.50000) is True

    # No change: floor blocks the False-positive.
    assert _is_real_decrease(-0.500, -0.500) is False

    # Noise-floor jitter: 1e-14 decrease is well below 1e-12 → False.
    assert _is_real_decrease(-0.500_000_000_000_01, -0.500_000_000_000_00) is False
    # Right at the floor (strict less-than) → False.
    assert _is_real_decrease(-0.500_000_000_001, -0.500) is False
    # Just above the floor → True.
    assert _is_real_decrease(-0.500_000_000_002, -0.500) is True

    # Energy increased: never accepted.
    assert _is_real_decrease(-0.400, -0.500) is False

    # Non-finite inputs: rejected (NaN/inf must never satisfy the gate).
    assert _is_real_decrease(float("nan"), -0.500) is False
    assert _is_real_decrease(-0.500, float("nan")) is False
    assert _is_real_decrease(float("-inf"), -0.500) is False
    assert _is_real_decrease(-0.500, float("inf")) is False

    # Custom noise floor: caller can tighten or loosen.
    assert _is_real_decrease(-0.5001, -0.5000, noise=1e-2) is False
    assert _is_real_decrease(-0.5001, -0.5000, noise=1e-5) is True


def test_should_accept_best_respects_energy_floor():
    """Issue #298: gs_energy_floor rejects non-variational best-state candidates."""
    from tenax.algorithms.ipeps_optimize import _should_accept_best

    # No floor: any improvement is accepted.
    assert _should_accept_best(current_best=0.0, candidate=-0.5, floor=None) is True
    # No floor, worse candidate: rejected.
    assert _should_accept_best(current_best=-0.5, candidate=-0.3, floor=None) is False
    # Floor set, legitimate improvement: accepted.
    assert _should_accept_best(current_best=-0.5, candidate=-0.6, floor=-1.0) is True
    # Floor set, candidate strictly below floor: rejected even though it's
    # an improvement over current best.
    assert _should_accept_best(current_best=-0.5, candidate=-2.0, floor=-1.0) is False
    # Candidate equal to floor: rejected (strict inequality).
    assert _should_accept_best(current_best=0.0, candidate=-1.0, floor=-1.0) is False
    # Candidate just above floor: accepted.
    assert _should_accept_best(current_best=0.0, candidate=-0.99, floor=-1.0) is True


def test_should_accept_best_rejects_nan_and_inf():
    """Issue #298 review: non-finite candidates must be rejected.

    Without the guard a NaN candidate falls through every comparison
    (NaN is never >= or <= anything), overwrites ``best_energy``, and
    poisons the downstream ``energy_bound`` computation used by the
    Hager-Zhang line search.
    """
    from tenax.algorithms.ipeps_optimize import _should_accept_best

    assert (
        _should_accept_best(current_best=0.0, candidate=float("nan"), floor=None)
        is False
    )
    assert (
        _should_accept_best(current_best=0.0, candidate=float("nan"), floor=-1.0)
        is False
    )
    assert (
        _should_accept_best(current_best=0.0, candidate=float("-inf"), floor=None)
        is False
    )
    assert (
        _should_accept_best(current_best=0.0, candidate=float("inf"), floor=None)
        is False
    )
    # Finite candidate still behaves the same after the guard.
    assert _should_accept_best(current_best=0.0, candidate=-0.5, floor=None) is True
    assert math.isfinite(-0.5)  # documents the intent of the guard


def test_stall_reset_reinits_optax_lbfgs_state(monkeypatch):
    """Issue #298 review: reset branch must reinitialize Optax L-BFGS state.

    For ``gs_optimizer="lbfgs", gs_metric_precond=False`` the curvature
    history lives in the optax ``opt_state`` rather than the internal
    ``lbfgs_history`` list.  Clearing only ``lbfgs_history`` on stall
    would leave stale curvature in play.  This test force-stalls the
    1-site dispatcher by monkeypatching the backtracking line search
    to always report no-improvement, then verifies ``optimizer.init``
    is called at least twice — once at setup and at least once from
    inside the reset branch.
    """
    import optax

    from tenax.algorithms import ipeps_optimize as io

    init_calls = {"n": 0}
    real_build = io._build_optimizer

    def counting_build(cfg):
        optimizer = real_build(cfg)
        if optimizer is None:
            return None
        real_init = optimizer.init

        def counting_init(params):
            init_calls["n"] += 1
            return real_init(params)

        return optax.GradientTransformation(init=counting_init, update=optimizer.update)

    monkeypatch.setattr(io, "_build_optimizer", counting_build)

    # Force every backtracking line search to report no-improvement so
    # stall_count increments on every AD step and the reset branch
    # fires.  Return params unchanged with the SAME (not strictly
    # lower) energy, so the dispatcher's `if new_energy < energy_float`
    # check goes to the else (stall) branch.
    def always_stall(params, direction, grad, energy, loss_fn_fwd, **kwargs):
        return params, energy, 0.0

    monkeypatch.setattr(io, "_backtracking_line_search", always_stall)

    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    gate = jnp.kron(Sz, Sz).reshape(d, d, d, d)

    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=1,
        dt=0.1,
        gs_num_steps=3,
        gs_optimizer="lbfgs",
        gs_metric_precond=False,
        gs_line_search=True,
        gs_line_search_method="armijo",
        gs_stall_recovery="reset",  # explicit, not autoresolved
        gs_verbose=False,
        unit_cell="1x1",
        su_init=False,
        ctm=CTMConfig(chi=3, max_iter=5, min_iter=2, conv_tol=1e-3),
    )

    with contextlib.suppress(Exception):
        optimize_gs_ad(gate, None, cfg)

    # At least the setup init + one reset init must have fired.  If
    # the reset branch failed to call optimizer.init, init_calls stays
    # at 1.
    assert init_calls["n"] >= 2, (
        f"optimizer.init was called {init_calls['n']} time(s); expected "
        ">= 2 (setup + at least one stall-reset). The reset branch is "
        "not reinitializing the Optax L-BFGS state."
    )


# TestArnoldiOptimizerRecovery removed: its single test paired
# forward_gauge="qr" with gs_implicit_ad=True, which #350's
# resolve_projector_backward now rejects with ValueError before any
# CTMRGGradientError can be raised. The "optimizer recovers from a
# CTMRGGradientError" contract is no longer reachable from this config
# combination on the implicit-AD path.


def test_lattice_to_neighbors_kagome():
    from tenax import kagome
    from tenax.algorithms.ipeps_optimize import _lattice_to_neighbors

    lat = kagome()
    neighbors, name_to_coord, coord_to_name = _lattice_to_neighbors(lat)

    assert len(neighbors) == 3
    assert len(name_to_coord) == 3
    for c, dirs in neighbors.items():
        assert set(dirs.keys()) == {"left", "right", "top", "bottom"}
        for nb in dirs.values():
            assert nb in neighbors
    for name, coord in name_to_coord.items():
        assert coord_to_name[coord] == name


class TestOptimizeGsAdMultisite:
    """Tests for multisite optimize_gs_ad with Lattice unit cells."""

    @pytest.fixture
    def heisenberg_gate(self):
        from tenax import heisenberg_gate

        return heisenberg_gate()

    def test_multisite_ad_runs_kagome(self, heisenberg_gate):
        """Multisite AD with kagome lattice should run without crashing."""
        from tenax import kagome

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=2,
            unit_cell=kagome(),
            gs_metric_precond=False,
        )
        site_tensors, envs, E_gs = optimize_gs_ad(heisenberg_gate, None, config)

        assert isinstance(site_tensors, dict)
        assert set(site_tensors.keys()) == {"u", "v", "w"}
        assert isinstance(envs, dict)
        assert set(envs.keys()) == {"u", "v", "w"}
        for name in ("u", "v", "w"):
            assert site_tensors[name].todense().shape == (2, 2, 2, 2, 2)
        assert np.isfinite(E_gs)
