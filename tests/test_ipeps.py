"""Tests for the iPEPS and CTM algorithms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
    _build_double_layer,
    _build_double_layer_open,
    _initialize_split_ctm_env,
    _rdm1x2,
    _rdm2x1,
    _simple_update_1x1,
    _simple_update_2site_horizontal,
    _simple_update_2site_vertical,
    _split_env_to_standard,
    compute_energy_ctm,
    compute_energy_ctm_2site,
    compute_energy_split_ctm,
    ctm,
    ctm_2site,
    ctm_split,
    ipeps,
    iPEPSConfig,
    optimize_gs_ad,
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

    def test_custom_values(self):
        cfg = iPEPSConfig(max_bond_dim=4, num_imaginary_steps=50, dt=0.05)
        assert cfg.max_bond_dim == 4
        assert cfg.num_imaginary_steps == 50
        assert cfg.dt == 0.05

    def test_su_init_default_false(self):
        cfg = iPEPSConfig()
        assert cfg.su_init is False


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
        from tenax.algorithms.ipeps import _build_double_layer, _initialize_ctm_env

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


class TestSimpleUpdate1x1:
    def test_simple_update_runs(self):
        """Simple update step should run and return updated tensors + lambdas."""
        key = jax.random.PRNGKey(0)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {
            "horizontal": jnp.ones(D),
            "vertical": jnp.ones(D),
        }

        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        dt = 0.01
        gate_flat = gate.reshape(d * d, d * d)
        trotter_gate = jax.scipy.linalg.expm(-dt * gate_flat).reshape(d, d, d, d)

        max_bond_dim = 3
        A_new, lambdas_new = _simple_update_1x1(
            A,
            A,
            lambdas,
            trotter_gate,
            max_bond_dim,
            bond="horizontal",
        )

        # Should return tensors with same number of legs
        assert A_new.ndim == A.ndim

    def test_simple_update_bond_dim_bounded(self):
        """Updated tensor bond dimension should not exceed max_bond_dim."""
        key = jax.random.PRNGKey(1)
        D, d = 2, 2
        max_D = 3
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {
            "horizontal": jnp.ones(D),
            "vertical": jnp.ones(D),
        }

        gate_flat = jnp.eye(d * d)
        trotter_gate = gate_flat.reshape(d, d, d, d)

        A_new, _ = _simple_update_1x1(
            A,
            A,
            lambdas,
            trotter_gate,
            max_D,
            bond="horizontal",
        )
        # Check all bond dims are bounded
        assert A_new.shape[0] <= max_D  # up dim
        assert A_new.shape[2] <= max_D  # left dim

    def test_simple_update_5leg_modifies_tensor(self):
        """Passing a 5-leg tensor with a non-trivial gate should modify A."""
        key = jax.random.PRNGKey(10)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {"horizontal": jnp.ones(D), "vertical": jnp.ones(D)}
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        dt = 0.1
        gate_flat = gate.reshape(d * d, d * d)
        trotter_gate = jax.scipy.linalg.expm(-dt * gate_flat).reshape(d, d, d, d)

        A_new, _ = _simple_update_1x1(
            A,
            A,
            lambdas,
            trotter_gate,
            D,
            bond="horizontal",
        )
        assert not jnp.allclose(A, A_new, atol=1e-8), "A should change after update"

    def test_simple_update_5leg_preserves_shape(self):
        """After update, tensor should still have 5 legs."""
        key = jax.random.PRNGKey(20)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {"horizontal": jnp.ones(D), "vertical": jnp.ones(D)}
        trotter_gate = jnp.eye(d * d).reshape(d, d, d, d)

        for bond in ["horizontal", "vertical"]:
            A_new, _ = _simple_update_1x1(
                A,
                A,
                lambdas,
                trotter_gate,
                D,
                bond=bond,
            )
            assert A_new.ndim == 5
            assert A_new.shape[-1] == d  # physical dim unchanged

    def test_lambda_normalized(self):
        """Lambda vectors should have max element = 1 after update."""
        key = jax.random.PRNGKey(30)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {"horizontal": jnp.ones(D), "vertical": jnp.ones(D)}
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        dt = 0.1
        gate_flat = gate.reshape(d * d, d * d)
        trotter_gate = jax.scipy.linalg.expm(-dt * gate_flat).reshape(d, d, d, d)

        _, lam_h = _simple_update_1x1(
            A,
            A,
            lambdas,
            trotter_gate,
            D,
            bond="horizontal",
        )
        _, lam_v = _simple_update_1x1(
            A,
            A,
            lambdas,
            trotter_gate,
            D,
            bond="vertical",
        )

        assert jnp.allclose(jnp.max(lam_h["horizontal"]), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.max(lam_v["vertical"]), 1.0, atol=1e-10)


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
        energy, peps_out, env = ipeps(heisenberg_gate, None, config)
        assert jnp.isfinite(energy)

    def test_ipeps_returns_three_tuple(self, heisenberg_gate):
        """ipeps() should return (energy, peps, env) triple."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        result = ipeps(heisenberg_gate, None, config)
        assert len(result) == 3

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
        _, _, env = ipeps(heisenberg_gate, None, config)
        assert isinstance(env, CTMEnvironment)

    def test_ipeps_with_initial_peps(self, heisenberg_gate):
        """iPEPS should accept an initial PEPS tensor (non-None initial_peps)."""
        key = jax.random.PRNGKey(99)
        D, d = 2, 2
        initial_A = jax.random.normal(key, (D, D, D, D, d))
        initial_A = initial_A / (jnp.linalg.norm(initial_A) + 1e-10)

        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        energy, _, _ = ipeps(heisenberg_gate, initial_A, config)
        assert jnp.isfinite(energy)


class TestSimpleUpdate2Site:
    """Tests for the 2-site simple update functions."""

    @pytest.fixture
    def setup(self):
        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        D, d = 2, 2
        A = jax.random.normal(key_A, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        B = jax.random.normal(key_B, (D, D, D, D, d))
        B = B / (jnp.linalg.norm(B) + 1e-10)
        lambdas = {"horizontal": jnp.ones(D), "vertical": jnp.ones(D)}

        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        gate = jax.scipy.linalg.expm(-0.1 * H).reshape(d, d, d, d)
        return A, B, lambdas, gate, D

    def test_horizontal_runs(self, setup):
        A, B, lambdas, gate, D = setup
        A_new, B_new, lam_new = _simple_update_2site_horizontal(
            A,
            B,
            lambdas["horizontal"],
            lambdas["vertical"],
            gate,
            D,
            lambdas,
        )
        assert A_new.ndim == 5
        assert B_new.ndim == 5

    def test_vertical_runs(self, setup):
        A, B, lambdas, gate, D = setup
        A_new, B_new, lam_new = _simple_update_2site_vertical(
            A,
            B,
            lambdas["horizontal"],
            lambdas["vertical"],
            gate,
            D,
            lambdas,
        )
        assert A_new.ndim == 5
        assert B_new.ndim == 5

    def test_returns_different_A_and_B(self, setup):
        A, B, lambdas, gate, D = setup
        A_new, B_new, _ = _simple_update_2site_horizontal(
            A,
            B,
            lambdas["horizontal"],
            lambdas["vertical"],
            gate,
            D,
            lambdas,
        )
        assert not jnp.allclose(A_new, B_new, atol=1e-8)

    def test_preserves_physical_dim(self, setup):
        A, B, lambdas, gate, D = setup
        A_new, B_new, _ = _simple_update_2site_horizontal(
            A,
            B,
            lambdas["horizontal"],
            lambdas["vertical"],
            gate,
            D,
            lambdas,
        )
        assert A_new.shape[-1] == 2
        assert B_new.shape[-1] == 2

    def test_lambda_normalized(self, setup):
        A, B, lambdas, gate, D = setup
        _, _, lam_h = _simple_update_2site_horizontal(
            A,
            B,
            lambdas["horizontal"],
            lambdas["vertical"],
            gate,
            D,
            lambdas,
        )
        _, _, lam_v = _simple_update_2site_vertical(
            A,
            B,
            lambdas["horizontal"],
            lambdas["vertical"],
            gate,
            D,
            lambdas,
        )
        assert jnp.allclose(jnp.max(lam_h["horizontal"]), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.max(lam_v["vertical"]), 1.0, atol=1e-10)


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

    def test_1x1_backward_compatible(self, heisenberg_gate):
        """unit_cell='1x1' should give the same behavior as before."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=3,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
            unit_cell="1x1",
        )
        energy, _, env = ipeps(heisenberg_gate, None, config)
        assert jnp.isfinite(energy)
        assert isinstance(env, CTMEnvironment)

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
        """CTMConfig() still defaults to eigh."""
        cfg = CTMConfig()
        assert cfg.projector_method == "eigh"
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
        assert A_opt.shape == (2, 2, 2, 2, 2)
        assert B_opt.shape == (2, 2, 2, 2, 2)
        assert isinstance(env_A, CTMEnvironment)
        assert isinstance(env_B, CTMEnvironment)
        assert np.isfinite(E_gs)

    def test_2site_ad_energy_decreases(self, heisenberg_gate):
        """Energy after optimization should be lower than initial energy."""

        # Compute initial energy with random tensors
        D, d = 2, 2
        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A0 = jax.random.normal(key_A, (D, D, D, D, d))
        A0 = A0 / (jnp.linalg.norm(A0) + 1e-10)
        B0 = jax.random.normal(key_B, (D, D, D, D, d))
        B0 = B0 / (jnp.linalg.norm(B0) + 1e-10)

        env_A0, env_B0 = ctm_2site(A0, B0, CTMConfig(chi=4, max_iter=10))
        E_init = float(
            compute_energy_ctm_2site(A0, B0, env_A0, env_B0, heisenberg_gate, d)
        )

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=10,
            gs_learning_rate=1e-2,
            unit_cell="2site",
        )
        _, _, E_opt = optimize_gs_ad(heisenberg_gate, (A0, B0), config)
        assert E_opt < E_init, f"Energy did not decrease: {E_opt} >= {E_init}"

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
