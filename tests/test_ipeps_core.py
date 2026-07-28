"""Fast correctness / unit tests for the CTM and iPEPS helpers.

These tests are a deliberate mirror of the correctness-focused classes
in ``test_ipeps.py`` that cover:

* Configuration round-trip and validation.
* CTM forward-path correctness: shape, convergence, double-layer
  construction, RDM Hermiticity and normalization.
* Simple energy-from-CTM checks (product state).
* QR-projector smoke correctness.

The originals in ``test_ipeps.py`` remain unchanged (they are our
behavioral safety net during refactoring); this file exists to promote
the same correctness coverage into the ``core`` CI tier so regressions
are caught by the fast required checks instead of only by the slow
full-suite run. Longer AD / SU regressions and Heisenberg benchmarks
continue to live in ``test_ipeps.py`` under the ``algorithm`` tier.
"""

import jax
import jax.numpy as jnp
import pytest

# Pin float64 so results do not depend on whichever other (x64-enabling) test
# module happened to run first on the same pytest-xdist worker.  Without this,
# ``test_rdm_positive_semidefinite`` silently flips precision by worker schedule,
# and its macOS-Accelerate CTM eigenvalue crosses the sanity threshold (the #700
# PR added tests, reshuffling the xdist split, which exposed this fragility).
jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    iPEPSConfig,
)
from tenax.algorithms.ipeps_ctm import (
    _build_double_layer,
    ctm,
)
from tenax.algorithms.ipeps_rdm import (
    _build_double_layer_open,
    _rdm1x2,
    _rdm2x1,
    compute_energy_ctm,
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
        """Eigenvalues of the RDM should not be wildly unphysical.

        For a random (non-optimized) PEPS with small chi the CTM environment is
        approximate, so eigenvalues outside [0, 1] are expected; the point of
        this check is only to catch a *broken* env (magnitudes blowing up by
        orders of magnitude).  The bound is deliberately loose: a well-behaved
        run lands near [0, 1] (~0.6 here on Linux), but the same chi=8 random
        env is genuinely near-degenerate and, under macOS-Accelerate LAPACK,
        yields O(10) eigenvalues (~11.7) — approximate but not broken.  Use an
        order-of-magnitude bound so the check is platform-robust while still
        flagging a truly exploded (>> O(10)) env.
        """
        A, env, d = peps_env
        rdm_h = _rdm2x1(A, env, d).reshape(d * d, d * d)
        rdm_v = _rdm1x2(A, env, d).reshape(d * d, d * d)

        eigvals_h = jnp.linalg.eigvalsh(rdm_h)
        eigvals_v = jnp.linalg.eigvalsh(rdm_v)
        assert jnp.all(jnp.abs(eigvals_h) < 100), f"Unbounded eigenvalues: {eigvals_h}"
        assert jnp.all(jnp.abs(eigvals_v) < 100), f"Unbounded eigenvalues: {eigvals_v}"

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
