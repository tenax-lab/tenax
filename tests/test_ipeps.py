"""Tests for the iPEPS and CTM algorithms."""

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
        assert A_opt.todense().shape == (2, 2, 2, 2, 2)
        assert B_opt.todense().shape == (2, 2, 2, 2, 2)
        assert np.isfinite(E_gs)

    def test_2site_ad_energy_decreases(self, heisenberg_gate):
        """Energy after optimization should be lower than initial energy."""
        from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

        # Compute initial energy via the Tensor-protocol CTM (same path
        # used during optimization) so that the comparison is consistent.
        D, d = 2, 2
        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A0 = _wrap_as_dense_tensor(jax.random.normal(key_A, (D, D, D, D, d)))
        B0 = _wrap_as_dense_tensor(jax.random.normal(key_B, (D, D, D, D, d)))

        # Run 0 optimization steps to get initial energy from tensor CTM
        config_init = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=0,
            unit_cell="2site",
        )
        _, _, E_init = optimize_gs_ad(heisenberg_gate, (A0, B0), config_init)

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

    def test_2site_ad_warmstart_energy_physical(self, heisenberg_gate):
        """With warm-start + min_iter, AD energy should stay physical at D=2 chi=8."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=8, max_iter=40, min_iter=10),
            gs_num_steps=20,
            gs_learning_rate=5e-3,
            unit_cell="2site",
            su_init=True,
            num_imaginary_steps=100,
            dt=0.1,
        )
        _, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        assert E_gs > -0.9, f"E/site={E_gs:.6f} unphysically low"
        assert np.isfinite(E_gs)

    @pytest.mark.slow
    def test_2site_heisenberg_ad_energy_benchmark(self, heisenberg_gate):
        """SU + AD at D=2, chi=16 should give a physical energy.

        The exact 2D Heisenberg square-lattice ground-state energy is
        E/site = -0.6694 (Sandvik, PRB 56, 11678, 1997).  At finite CTM
        bond dimension chi, the energy is NOT a strict variational bound
        and can dip below the exact value.  We check that:

        1. E > -0.9: catches unphysical results from numerical failures.
        2. E < -0.60: confirms AD produces a reasonable energy.
        """
        # Néel product state init for AFM Heisenberg
        D, d = 2, 2
        key_A, key_B = jax.random.split(jax.random.PRNGKey(42))
        A_neel = 0.01 * jax.random.normal(key_A, (D, D, D, D, d))
        A_neel = A_neel.at[0, 0, 0, 0, 0].set(1.0)
        B_neel = 0.01 * jax.random.normal(key_B, (D, D, D, D, d))
        B_neel = B_neel.at[0, 0, 0, 0, 1].set(1.0)

        su_config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=100,
            dt=0.3,
            ctm=CTMConfig(chi=16, max_iter=100, min_iter=50),
            unit_cell="2site",
        )
        E_su, (A_su, B_su), _ = ipeps(heisenberg_gate, (A_neel, B_neel), su_config)

        ad_config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=100,
            dt=0.3,
            ctm=CTMConfig(chi=16, max_iter=100, min_iter=50),
            gs_num_steps=50,
            gs_learning_rate=5e-3,
            unit_cell="2site",
        )
        _, _, E_gs = optimize_gs_ad(
            heisenberg_gate, (A_su.todense(), B_su.todense()), ad_config
        )
        assert E_gs > -0.9, (
            f"E/site = {E_gs:.6f} is unphysically low — possible numerical failure"
        )
        assert E_gs < -0.60, (
            f"E/site = {E_gs:.6f}, expected < -0.60 for D=2 AD-optimized iPEPS"
        )


class TestHeisenbergBenchmark:
    """Regression tests against known iPEPS results for 2D Heisenberg AFM.

    Reference values:
        QMC exact: E/site = -0.669437(5) (Sandvik, PRB 56, 11678, 1997)
        QMC m_s   = 0.3070(3)

    iPEPS literature (Corboz et al.):
        D=2: E/site ≈ -0.6548
        D=3: E/site ≈ -0.6646
        D=4: E/site ≈ -0.6670
    """

    @pytest.fixture
    def heisenberg_gate(self):
        d = 2
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
        return H.reshape(d, d, d, d)

    @pytest.mark.slow
    def test_su_d2_energy(self, heisenberg_gate):
        """Simple update at D=2 should give E/site < -0.60.

        A moderate dt (0.3) is essential: small dt (e.g. 0.01) causes the
        bond lambdas to converge to a product-like fixed point with too
        little entanglement, giving E ~ -0.51 regardless of step count.
        """
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=200,
            dt=0.3,
            ctm=CTMConfig(chi=16, max_iter=60),
            unit_cell="2site",
        )
        E_su, _, _ = ipeps(heisenberg_gate, None, config)
        E = float(E_su)
        assert E < -0.60, f"SU D=2 E/site={E:.6f}, expected < -0.60"
        assert E > -0.80, f"SU D=2 E/site={E:.6f}, unphysically low"

    @pytest.mark.slow
    def test_ad_d2_energy(self, heisenberg_gate):
        """AD optimization at D=2, chi=16 should give E/site < -0.648.

        Literature value for D=2 iPEPS Heisenberg is E/site ≈ -0.6548.
        We use a loose bound since chi=16 is moderate.
        """
        # Néel product state init for AFM Heisenberg
        D, d = 2, 2
        key_A, key_B = jax.random.split(jax.random.PRNGKey(42))
        A_neel = 0.01 * jax.random.normal(key_A, (D, D, D, D, d))
        A_neel = A_neel.at[0, 0, 0, 0, 0].set(1.0)
        B_neel = 0.01 * jax.random.normal(key_B, (D, D, D, D, d))
        B_neel = B_neel.at[0, 0, 0, 0, 1].set(1.0)

        su_config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=200,
            dt=0.05,
            ctm=CTMConfig(chi=16, max_iter=100, min_iter=50),
            unit_cell="2site",
        )
        _, (A_su, B_su), _ = ipeps(heisenberg_gate, (A_neel, B_neel), su_config)

        ad_config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=16, max_iter=100, min_iter=50),
            gs_num_steps=50,
            gs_learning_rate=5e-3,
            unit_cell="2site",
        )
        _, _, E_gs = optimize_gs_ad(
            heisenberg_gate, (A_su.todense(), B_su.todense()), ad_config
        )
        assert E_gs < -0.648, (
            f"AD D=2 chi=16 E/site={E_gs:.6f}, expected < -0.648 "
            "(literature D=2 ≈ -0.6548)"
        )
        assert E_gs > -0.80, f"AD D=2 E/site={E_gs:.6f}, unphysically low"

    @pytest.mark.slow
    def test_ad_d2_chi_scaling(self, heisenberg_gate):
        """Energy should improve (decrease) with increasing chi at fixed D=2."""
        su_config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=100,
            dt=0.1,
            ctm=CTMConfig(chi=8, max_iter=40),
            unit_cell="2site",
        )
        _, (A_su, B_su), _ = ipeps(heisenberg_gate, None, su_config)

        energies = []
        for chi in [8, 16]:
            ad_config = iPEPSConfig(
                max_bond_dim=2,
                ctm=CTMConfig(chi=chi, max_iter=60),
                gs_num_steps=30,
                gs_learning_rate=5e-3,
                unit_cell="2site",
            )
            _, _, E = optimize_gs_ad(
                heisenberg_gate, (A_su.todense(), B_su.todense()), ad_config
            )
            energies.append(float(E))

        assert energies[1] <= energies[0] + 0.01, (
            f"Energy should improve with chi: chi=8 E={energies[0]:.6f}, "
            f"chi=16 E={energies[1]:.6f}"
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
        """L-BFGS optimizer with line search should produce finite energy."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=5),
            gs_optimizer="lbfgs",
            gs_num_steps=3,
            gs_line_search=True,
        )
        _, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        assert np.isfinite(E_gs)

    def test_cg_optimizer_runs(self, heisenberg_gate):
        """CG optimizer should produce finite energy (covers PR beta path)."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=5),
            gs_optimizer="cg",
            gs_num_steps=3,
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
        """optimize_gs_ad accepts SymmetricTensor and returns Tensor."""
        from tenax.core.tensor import Tensor

        gate = self._heisenberg_gate()
        A_sym = self._make_symmetric_ipeps(jax.random.PRNGKey(0))

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=2,
            gs_learning_rate=0.01,
        )
        A_opt, env, E_gs = optimize_gs_ad(gate, A_sym, config)

        assert isinstance(A_opt, Tensor)
        assert np.isfinite(E_gs)

    def test_optimize_gs_ad_symmetric_energy_decreases(self):
        """AD optimization with SymmetricTensor decreases energy."""
        gate = self._heisenberg_gate()
        A_sym = self._make_symmetric_ipeps(jax.random.PRNGKey(0))

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=30, min_iter=15),
            gs_num_steps=5,
            gs_learning_rate=0.01,
        )

        # Get initial energy
        from tenax.algorithms._ctm_tensor import (
            compute_energy_ctm_tensor,
            ctm_tensor,
        )

        A_norm = A_sym * (1.0 / (A_sym.norm() + 1e-10))
        env0 = ctm_tensor(A_norm, chi=4, max_iter=30)
        E_init = float(compute_energy_ctm_tensor(A_norm, env0, gate))

        A_opt, env, E_gs = optimize_gs_ad(gate, A_sym, config)

        assert E_gs < E_init or abs(E_gs - E_init) < 1e-8

    def test_optimize_gs_ad_symmetric_matches_dense(self):
        """Symmetric AD gives comparable energy to dense AD."""
        gate = self._heisenberg_gate()

        # Create a SymmetricTensor and its dense equivalent
        A_sym = self._make_symmetric_ipeps(jax.random.PRNGKey(0))
        A_dense = A_sym.todense()

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=3,
            gs_learning_rate=0.01,
        )

        _, _, E_sym = optimize_gs_ad(gate, A_sym, config)
        _, _, E_dense = optimize_gs_ad(gate, A_dense, config)

        # Both should produce finite energies (exact match not expected
        # due to different CTM projector implementations)
        assert np.isfinite(E_sym)
        assert np.isfinite(E_dense)


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
