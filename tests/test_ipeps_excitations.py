"""Tests for iPEPS excitation calculations."""

import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    iPEPSConfig,
)
from tenax.algorithms.ipeps_ctm import ctm
from tenax.algorithms.ipeps_excitations import (
    ExcitationConfig,
    ExcitationResult,
    _build_double_layer_BB_open,
    _build_H_and_N,
    _build_mixed_double_layer,
    _build_mixed_double_layer_open,
    _compute_excitation_energy,
    _compute_norm,
    _project_out_ground_state,
    _rdm1x2_mixed,
    _rdm2x1_mixed,
    _solve_excitations,
    _transition_trace,
    compute_excitations,
    make_momentum_path,
)
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tenax.algorithms.ipeps_rdm import _build_double_layer_open, compute_energy_ctm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def heisenberg_gate():
    """2-site Heisenberg Hamiltonian gate."""
    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


@pytest.fixture
def small_peps_and_env():
    """Small random PEPS tensor with converged CTM environment."""
    key = jax.random.PRNGKey(42)
    D, d = 2, 2
    A = jax.random.normal(key, (D, D, D, D, d))
    A = A / (jnp.linalg.norm(A) + 1e-10)
    config = CTMConfig(chi=8, max_iter=40)
    env = ctm(A, config)
    return A, env, d


# ---------------------------------------------------------------------------
# optimize_gs_ad tests
# ---------------------------------------------------------------------------


class TestOptimizeGsAd:
    def test_runs_without_error(self, heisenberg_gate):
        """AD optimization should run without crashing."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=5),
            gs_num_steps=3,
            gs_learning_rate=1e-2,
        )
        A_opt, env, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        assert A_opt.todense().shape == (2, 2, 2, 2, 2)
        assert np.isfinite(E_gs)

    def test_heisenberg_negative_energy(self, heisenberg_gate):
        """Heisenberg D=2 should give E < 0 after some optimization steps."""
        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=20,
            gs_learning_rate=1e-2,
        )
        _, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        # Loose check — with small D and few steps, energy may not be very negative
        assert E_gs < 1.0, f"Energy should be negative-ish, got {E_gs}"

    def test_su_init_runs_without_error(self, heisenberg_gate):
        """optimize_gs_ad with su_init=True should produce a valid tensor."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=10,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=3,
            gs_learning_rate=1e-2,
            su_init=True,
        )
        A_opt, env, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
        assert A_opt.todense().shape == (2, 2, 2, 2, 2)
        assert jnp.all(jnp.isfinite(A_opt.todense()))
        assert np.isfinite(E_gs)

    def test_su_init_ignored_when_A_init_provided(self, heisenberg_gate):
        """When A_init is provided, su_init=True should be ignored."""
        key = jax.random.PRNGKey(42)
        D, d = 2, 2
        A_init = jax.random.normal(key, (D, D, D, D, d))
        A_init = A_init / (jnp.linalg.norm(A_init) + 1e-10)

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=4, max_iter=10),
            gs_num_steps=3,
            gs_learning_rate=1e-2,
            su_init=True,
        )
        A_opt, env, E_gs = optimize_gs_ad(heisenberg_gate, A_init, config)
        assert A_opt.todense().shape == (D, D, D, D, d)
        assert jnp.all(jnp.isfinite(A_opt.todense()))
        assert np.isfinite(E_gs)


# ---------------------------------------------------------------------------
# Mixed double-layer tests
# ---------------------------------------------------------------------------


class TestMixedDoubleLayer:
    def test_shape_closed(self, small_peps_and_env):
        """Mixed double-layer (closed) should be (D^2, D^2, D^2, D^2)."""
        A, _, d = small_peps_and_env
        D = A.shape[0]
        B = jax.random.normal(jax.random.PRNGKey(1), A.shape)
        dl = _build_mixed_double_layer(A, B, "ket")
        assert dl.shape == (D**2, D**2, D**2, D**2)

    def test_shape_open(self, small_peps_and_env):
        """Mixed double-layer (open) should be (D^2, D^2, D^2, D^2, d, d)."""
        A, _, d = small_peps_and_env
        D = A.shape[0]
        B = jax.random.normal(jax.random.PRNGKey(2), A.shape)
        dl = _build_mixed_double_layer_open(A, B, "ket")
        assert dl.shape == (D**2, D**2, D**2, D**2, d, d)

    def test_reduces_to_standard_when_B_equals_A(self, small_peps_and_env):
        """When B=A, mixed double-layer should equal standard double-layer."""
        A, _, d = small_peps_and_env
        dl_mixed = _build_mixed_double_layer_open(A, A, "ket")
        dl_standard = _build_double_layer_open(A)
        assert jnp.allclose(dl_mixed, dl_standard, atol=1e-12)

    def test_trace_closed_matches(self, small_peps_and_env):
        """Tracing physical indices of open mixed tensor gives closed one."""
        A, _, d = small_peps_and_env
        B = jax.random.normal(jax.random.PRNGKey(5), A.shape)
        dl_open = _build_mixed_double_layer_open(A, B, "ket")
        dl_closed = _build_mixed_double_layer(A, B, "ket")
        # Trace over physical indices (s == t)
        dl_traced = jnp.einsum("udlrss->udlr", dl_open)
        assert jnp.allclose(dl_traced, dl_closed, atol=1e-12)

    def test_BB_open_shape(self, small_peps_and_env):
        """BB double-layer should have correct shape."""
        A, _, d = small_peps_and_env
        D = A.shape[0]
        B = jax.random.normal(jax.random.PRNGKey(3), A.shape)
        dl = _build_double_layer_BB_open(B)
        assert dl.shape == (D**2, D**2, D**2, D**2, d, d)


# ---------------------------------------------------------------------------
# H_eff and N matrix tests
# ---------------------------------------------------------------------------


class TestBuildHAndN:
    def test_shapes(self, small_peps_and_env, heisenberg_gate):
        """H_eff and N should be square matrices of size D^4*d."""
        A, env, d = small_peps_and_env
        D = A.shape[0]
        basis_size = D**4 * d
        k = jnp.array([np.pi / 2, 0.0])
        E_gs = float(compute_energy_ctm(A, env, heisenberg_gate, d))

        config = ExcitationConfig(num_excitations=2)
        H_eff, N_mat = _build_H_and_N(A, env, k, heisenberg_gate, E_gs, d, config)

        assert H_eff.shape == (basis_size, basis_size)
        assert N_mat.shape == (basis_size, basis_size)

    def test_N_matrix_approximately_hermitian(
        self, small_peps_and_env, heisenberg_gate
    ):
        """Norm matrix should be approximately Hermitian.

        With finite chi and a random (not optimized) tensor, the asymmetry
        can be nontrivial, so we use a relative tolerance.
        """
        A, env, d = small_peps_and_env
        k = jnp.array([0.0, 0.0])
        E_gs = float(compute_energy_ctm(A, env, heisenberg_gate, d))
        config = ExcitationConfig()

        _, N_mat = _build_H_and_N(A, env, k, heisenberg_gate, E_gs, d, config)

        N_sym = 0.5 * (N_mat + N_mat.conj().T)
        asymmetry = np.max(np.abs(N_mat - N_sym))
        scale = np.max(np.abs(N_mat)) + 1e-15
        relative_asymmetry = asymmetry / scale
        assert relative_asymmetry < 1.0, (
            f"N relative asymmetry too large: {relative_asymmetry}"
        )

    def test_N_matrix_has_positive_eigenvalues(
        self, small_peps_and_env, heisenberg_gate
    ):
        """Symmetrized N should have some positive eigenvalues.

        With a random (non-optimized) tensor and small chi, the norm
        matrix may not be positive semi-definite. We just verify it has
        at least some positive eigenvalues, confirming the matrix is
        nontrivial.
        """
        A, env, d = small_peps_and_env
        k = jnp.array([0.0, 0.0])
        E_gs = float(compute_energy_ctm(A, env, heisenberg_gate, d))
        config = ExcitationConfig()

        _, N_mat = _build_H_and_N(A, env, k, heisenberg_gate, E_gs, d, config)

        N_sym = 0.5 * (N_mat + N_mat.conj().T)
        eigvals = np.linalg.eigvalsh(N_sym)
        assert np.any(np.abs(eigvals) > 1e-10), (
            "N matrix is trivially zero — expected nontrivial entries"
        )


# ---------------------------------------------------------------------------
# Norm and energy functional tests
# ---------------------------------------------------------------------------


class TestNormFunctional:
    def test_norm_nonnegative(self, small_peps_and_env):
        """Norm should be real-valued and finite.

        Formal positivity of <Phi_k(B)|Phi_k(B)> requires an exact CTM fixed
        point.  With finite chi and a random (non-optimized) A, truncation
        error can drive the computed norm slightly negative — the same
        regime caveat that test_N_matrix_has_positive_eigenvalues calls out
        for the N matrix.  This test verifies only the computational
        contract (real-valued and finite), since the positivity bound was
        seed/BLAS-dependent and failed on macOS Accelerate (#529).
        """
        A, env, d = small_peps_and_env
        B = jax.random.normal(jax.random.PRNGKey(10), A.shape)
        k = jnp.array([0.0, 0.0])
        norm = _compute_norm(A, B, env, k, d)
        assert jnp.isfinite(norm), f"Norm should be finite, got {norm}"
        # Norm is bilinear in B and B*, so the imaginary part must vanish
        # up to floating-point noise relative to the real part.
        imag_part = float(jnp.imag(norm))
        real_part = float(jnp.real(norm))
        assert abs(imag_part) < 1e-8 * (abs(real_part) + 1.0), (
            f"Norm should be real, got imag={imag_part}, real={real_part}"
        )

    def test_norm_zero_for_zero_B(self, small_peps_and_env):
        """Norm should be zero (or near zero) when B=0."""
        A, env, d = small_peps_and_env
        B = jnp.zeros_like(A)
        k = jnp.array([0.0, 0.0])
        norm = _compute_norm(A, B, env, k, d)
        assert abs(float(norm)) < 1e-10


# ---------------------------------------------------------------------------
# Generalized eigenvalue solver tests
# ---------------------------------------------------------------------------


class TestSolveExcitations:
    def test_positive_definite_case(self):
        """For known positive-definite H and N, eigenvalues should be correct."""
        N = np.eye(4)
        H = np.diag([1.0, 2.0, 3.0, 4.0])
        eigvals = _solve_excitations(H, N, num_excitations=3)
        assert len(eigvals) == 3
        np.testing.assert_allclose(eigvals, [1.0, 2.0, 3.0], atol=1e-10)

    def test_with_null_space(self):
        """Should handle N with null space correctly."""
        # N with one zero eigenvalue
        N = np.diag([1.0, 1.0, 1.0, 0.0])
        H = np.diag([1.0, 2.0, 3.0, 0.0])
        eigvals = _solve_excitations(H, N, num_excitations=2, null_tol=1e-3)
        assert len(eigvals) == 2
        np.testing.assert_allclose(eigvals, [1.0, 2.0], atol=1e-10)

    def test_returns_sorted(self):
        """Eigenvalues should be returned in ascending order."""
        N = np.eye(5)
        H = np.diag([5.0, 1.0, 3.0, 2.0, 4.0])
        eigvals = _solve_excitations(H, N, num_excitations=3)
        assert np.all(np.diff(eigvals) >= -1e-10)

    def test_output_shape_always_num_excitations(self):
        """Output length must always equal num_excitations, even when the
        safe subspace is smaller."""
        # Only 2 non-null modes, but request 5 excitations
        N = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
        H = np.diag([3.0, 7.0, 0.0, 0.0, 0.0])
        eigvals = _solve_excitations(H, N, num_excitations=5, null_tol=1e-3)
        assert len(eigvals) == 5
        # Padded entries should be zero
        np.testing.assert_allclose(eigvals[2:], 0.0)

    def test_output_finite(self):
        """All returned eigenvalues must be finite."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((6, 6))
        N = A.T @ A + 0.1 * np.eye(6)  # positive definite
        H = rng.standard_normal((6, 6))
        H = 0.5 * (H + H.T)
        eigvals = _solve_excitations(H, N, num_excitations=4)
        assert np.all(np.isfinite(eigvals))

    def test_physical_branch_nonnegative(self):
        """For H = E*N (zero excitation gap), eigenvalues should be ~0."""
        N = np.diag([1.0, 0.5, 0.25])
        # H_eff = H - E_gs*N; if physical H = E_gs*N then H_eff = 0
        H_eff = np.zeros((3, 3))
        eigvals = _solve_excitations(H_eff, N, num_excitations=3)
        np.testing.assert_allclose(eigvals, 0.0, atol=1e-12)

    def test_zero_N_returns_zeros(self):
        """When N is all zeros, should return zeros without error."""
        N = np.zeros((4, 4))
        H = np.eye(4)
        eigvals = _solve_excitations(H, N, num_excitations=3)
        assert len(eigvals) == 3
        np.testing.assert_allclose(eigvals, 0.0)


# ---------------------------------------------------------------------------
# Excitation energy tests
# ---------------------------------------------------------------------------


class TestExcitationEnergies:
    def test_positive_at_nonzero_k(self, small_peps_and_env, heisenberg_gate):
        """At non-zero momentum, excitation energies should be positive
        for a gapped model (approximate test)."""
        A, env, d = small_peps_and_env
        E_gs = float(compute_energy_ctm(A, env, heisenberg_gate, d))

        config = ExcitationConfig(num_excitations=1, null_space_tol=1e-2)
        momenta = [(np.pi, 0.0)]
        result = compute_excitations(A, env, heisenberg_gate, E_gs, momenta, config)

        assert isinstance(result, ExcitationResult)
        assert result.energies.shape == (1, 1)
        # With a random A tensor, the spectrum is unpredictable,
        # so we just check finiteness
        assert np.all(np.isfinite(result.energies))


# ---------------------------------------------------------------------------
# Tensor-protocol input acceptance (issue #636)
# ---------------------------------------------------------------------------


def _wrap_array_as_dense_tensor(arr):
    """Wrap a raw array as a DenseTensor with trivial (all-zero U(1)) charges.

    Flows/labels are irrelevant for the excitation path — it only calls
    ``.todense()`` — so trivial indices suffice to exercise the conversion.
    """
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    sym = U1Symmetry()
    arr = jnp.asarray(arr)
    indices = tuple(
        TensorIndex.from_charges(
            sym, np.zeros(arr.shape[i], dtype=np.int32), FlowDirection.OUT
        )
        for i in range(arr.ndim)
    )
    return DenseTensor(arr, indices)


class TestTensorInputAcceptance:
    """compute_excitations must accept the Tensor-protocol outputs of
    ``optimize_gs_ad`` (DenseTensor site tensor + Tensor-based env), not only
    raw ``jax.Array``/``CTMEnvironment`` inputs.  Regression for issue #636.
    """

    def test_dense_tensor_inputs_match_raw_arrays(
        self, small_peps_and_env, heisenberg_gate
    ):
        """Wrapping A, the gate, and every env field as DenseTensor must yield
        the same excitation spectrum as the raw-array call."""
        A, env, d = small_peps_and_env
        E_gs = float(compute_energy_ctm(A, env, heisenberg_gate, d))
        config = ExcitationConfig(num_excitations=2, null_space_tol=1e-2)
        momenta = [(0.0, 0.0), (np.pi, 0.0)]

        ref = compute_excitations(A, env, heisenberg_gate, E_gs, momenta, config)

        # Mimic optimize_gs_ad's return types: DenseTensor A + gate, and a
        # CTM environment whose 8 fields are DenseTensors.
        A_t = _wrap_array_as_dense_tensor(A)
        gate_t = _wrap_array_as_dense_tensor(heisenberg_gate)
        env_t = CTMEnvironment(*(_wrap_array_as_dense_tensor(f) for f in env))

        res = compute_excitations(A_t, env_t, gate_t, E_gs, momenta, config)

        assert isinstance(res, ExcitationResult)
        np.testing.assert_allclose(res.energies, ref.energies, atol=1e-10, rtol=0)

    def test_mixed_dense_tensor_and_raw_inputs(
        self, small_peps_and_env, heisenberg_gate
    ):
        """A DenseTensor A with a raw-array env (and vice versa) must work —
        the normalization is per-argument."""
        A, env, d = small_peps_and_env
        E_gs = float(compute_energy_ctm(A, env, heisenberg_gate, d))
        config = ExcitationConfig(num_excitations=1, null_space_tol=1e-2)
        momenta = [(np.pi, 0.0)]

        ref = compute_excitations(A, env, heisenberg_gate, E_gs, momenta, config)
        res = compute_excitations(
            _wrap_array_as_dense_tensor(A), env, heisenberg_gate, E_gs, momenta, config
        )
        np.testing.assert_allclose(res.energies, ref.energies, atol=1e-10, rtol=0)

    def test_split_env_rejected_with_clear_error(
        self, small_peps_and_env, heisenberg_gate
    ):
        """A non-8-tensor (e.g. split) environment must raise a clear error
        rather than fail deep in the contraction."""
        from tenax.algorithms.ipeps_excitations import _as_dense_env

        A, env, d = small_peps_and_env
        twelve_field_env = tuple(env) + tuple(env[:4])  # 12 fields, like split CTM
        with pytest.raises(ValueError, match="8-tensor CTM environment"):
            _as_dense_env(twelve_field_env)

    def test_symmetric_tensor_rejected(self, small_peps_and_env, heisenberg_gate):
        """SymmetricTensor inputs must raise (dense-only path) rather than
        silently densify a block-sparse tensor — project rule against
        ``todense()`` on the symmetric path."""
        from tenax.algorithms.ipeps_excitations import _as_dense_array
        from tenax.core.index import FlowDirection, TensorIndex
        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import SymmetricTensor

        sym = U1Symmetry()
        idx = TensorIndex.from_charges(
            sym, np.array([0, 0], dtype=np.int32), FlowDirection.OUT
        )
        sym_t = SymmetricTensor.from_dense(jnp.eye(2), (idx, idx.dual()))
        with pytest.raises(NotImplementedError, match="SymmetricTensor"):
            _as_dense_array(sym_t)


# ---------------------------------------------------------------------------
# Momentum path tests
# ---------------------------------------------------------------------------


class TestMomentumPath:
    def test_brillouin_covers_high_symmetry_points(self):
        """Path should include points near Gamma, X, and M."""
        path = make_momentum_path("brillouin", num_points=30)
        assert len(path) == 30

        kx_vals = [p[0] for p in path]
        ky_vals = [p[1] for p in path]

        # Gamma (0,0) should be the first point
        assert abs(kx_vals[0]) < 1e-10
        assert abs(ky_vals[0]) < 1e-10

        # Should contain points near X(pi, 0) and M(pi, pi)
        has_near_X = any(abs(kx - np.pi) < 0.5 and abs(ky) < 0.5 for kx, ky in path)
        has_near_M = any(
            abs(kx - np.pi) < 0.5 and abs(ky - np.pi) < 0.5 for kx, ky in path
        )
        assert has_near_X, "Path should include points near X(pi, 0)"
        assert has_near_M, "Path should include points near M(pi, pi)"

    def test_diagonal_path(self):
        """Diagonal path from Gamma to M."""
        path = make_momentum_path("diagonal", num_points=10)
        assert len(path) == 10
        # First point: Gamma
        assert abs(path[0][0]) < 1e-10
        assert abs(path[0][1]) < 1e-10
        # Last point: M(pi, pi)
        assert abs(path[-1][0] - np.pi) < 1e-10
        assert abs(path[-1][1] - np.pi) < 1e-10

    def test_invalid_path_type_raises(self):
        """Unknown path type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown path_type"):
            make_momentum_path("invalid_type")


# ---------------------------------------------------------------------------
# Opt-in benchmark: RUN_EXCITATION_BENCH=1 for local/nightly runs
# ---------------------------------------------------------------------------

_RUN_BENCH = os.environ.get("RUN_EXCITATION_BENCH", "") == "1"
_bench_reason = "Set RUN_EXCITATION_BENCH=1 to run (GEV is ill-conditioned at D=2)"

logger = logging.getLogger(__name__)


@pytest.mark.slow
class TestExcitationBenchmark:
    @pytest.mark.skipif(not _RUN_BENCH, reason=_bench_reason)
    def test_heisenberg_excitation_dispersion(self, heisenberg_gate):
        """Verify excitation spectrum for 2D Heisenberg AFM (D=2, chi=16).

        Checks that the excitation spectrum is physically reasonable:
        finite excitation energies and positive gaps at zone-boundary
        momenta. Opt-in via ``RUN_EXCITATION_BENCH=1`` because the GEV
        is ill-conditioned at D=2 and results are BLAS-dependent.
        """
        D, d = 2, 2

        best_A, best_env, best_E = None, None, 0.0
        for seed in [42, 0, 7]:
            key = jax.random.PRNGKey(seed)
            A_init = jax.random.normal(key, (D, D, D, D, d))
            A_init = A_init / (jnp.linalg.norm(A_init) + 1e-10)

            config = iPEPSConfig(
                max_bond_dim=D,
                ctm=CTMConfig(chi=16, max_iter=60),
                gs_num_steps=100,
                gs_learning_rate=1e-3,
            )
            A_opt, env, E_gs = optimize_gs_ad(heisenberg_gate, A_init, config)
            if best_A is None or E_gs < best_E:
                best_A, best_env, best_E = A_opt, env, E_gs
            if best_E < -0.5:
                break

        assert best_E < -0.3, f"Ground state energy should be negative, got {best_E}"

        # Compute excitations with diagnostic logging
        momenta = [(0.0, 0.0), (np.pi, 0.0), (np.pi, np.pi)]
        exc_config = ExcitationConfig(num_excitations=2)

        for i, (kx, ky) in enumerate(momenta):
            k = jnp.array([kx, ky])
            H_eff, N_mat = _build_H_and_N(
                best_A, best_env, k, heisenberg_gate, best_E, d, exc_config
            )
            # Diagnostic: log cond(N) and null-space filtering
            N_sym = 0.5 * (np.array(N_mat) + np.array(N_mat).conj().T)
            eigvals_N = np.linalg.eigvalsh(N_sym)
            max_eig = eigvals_N[-1]
            kept = np.sum(eigvals_N > exc_config.null_space_tol * max_eig)
            cond = max_eig / max(eigvals_N[eigvals_N > 0].min(), 1e-30)
            logger.info(
                "k=(%.3f,%.3f) cond(N)=%.2e  kept=%d/%d  eigN_range=[%.2e, %.2e]",
                kx,
                ky,
                cond,
                kept,
                len(eigvals_N),
                eigvals_N[0],
                max_eig,
            )

        result = compute_excitations(
            best_A, best_env, heisenberg_gate, best_E, momenta, exc_config
        )

        assert result.energies.shape == (3, 2)
        assert np.all(np.isfinite(result.energies)), (
            f"Non-finite excitation energies: {result.energies}"
        )

        E_X = result.energies[1, 0]
        E_M = result.energies[2, 0]
        logger.info("Excitation energies:\n%s", result.energies)

        assert E_X > 0.1, (
            f"Excitation at X should be positive, got {E_X}; "
            f"all energies: {result.energies}"
        )
        assert E_M > 0.1, (
            f"Excitation at M should be positive, got {E_M}; "
            f"all energies: {result.energies}"
        )


# ---------------------------------------------------------------------------
# Exact-environment oracles for the #954/#955/#956 defect cluster
# ---------------------------------------------------------------------------
#
# A D=chi=1 environment of ones is the EXACT environment of any product
# state, so every expectation below has a closed form and every assertion is
# tight (atol ~1e-12) -- no CTM convergence caveats anywhere.  These three
# classes carry ``core``: the whole cluster shipped because nothing exact
# guarded this module (the pre-existing tests assert shapes and finiteness on
# a random unconverged fixture), and the oracles cost milliseconds.


def _exact_product_env() -> CTMEnvironment:
    """The exact CTM environment of a product state at D = chi = 1."""
    return CTMEnvironment(*([jnp.ones((1, 1))] * 4 + [jnp.ones((1, 1, 1))] * 4))


def _site(vec) -> jnp.ndarray:
    """A (1, 1, 1, 1, d) product-state site tensor from a length-d vector."""
    v = jnp.asarray(vec)
    assert v.ndim == 1, "product-state helper takes a bare state vector"
    return v.reshape(1, 1, 1, 1, -1)


_SX = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
_SZ = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])


class TestMixedRdmAxisOrder955:
    """#955: mixed RDMs must use the grouped (ket1, ket2, bra1, bra2) layout.

    The defect reshaped the interleaved (ket1, bra1, ket2, bra2) axes
    straight to a matrix, so on the exact product state |1, 0> the "density
    matrix" came out with trace zero and off-diagonal entries at 5e14 after
    normalisation amplified them.
    """

    pytestmark = pytest.mark.core

    def test_distinct_product_state_horizontal(self):
        env = _exact_product_env()
        A, B = _site([1.0, 0.0]), _site([0.0, 1.0])
        rho = np.asarray(_rdm2x1_mixed(A, B, env, 2, ("B", "B"), ("A", "A"))).reshape(
            4, 4
        )
        np.testing.assert_allclose(rho, np.diag([0.0, 0.0, 1.0, 0.0]), atol=1e-12)

    def test_distinct_product_state_vertical(self):
        env = _exact_product_env()
        A, B = _site([1.0, 0.0]), _site([0.0, 1.0])
        rho = np.asarray(_rdm1x2_mixed(A, B, env, 2, ("B", "B"), ("A", "A"))).reshape(
            4, 4
        )
        np.testing.assert_allclose(rho, np.diag([0.0, 0.0, 1.0, 0.0]), atol=1e-12)

    def test_substitution_at_the_other_site(self):
        env = _exact_product_env()
        A, B = _site([1.0, 0.0]), _site([0.0, 1.0])
        rho = np.asarray(_rdm2x1_mixed(A, B, env, 2, ("A", "A"), ("B", "B"))).reshape(
            4, 4
        )
        np.testing.assert_allclose(rho, np.diag([0.0, 1.0, 0.0, 0.0]), atol=1e-12)

    def test_complex_product_state_keeps_conjugation_direction(self):
        """rho = |b><b| x |0><0| -- the off-diagonal must be b0 * conj(b1)."""
        env = _exact_product_env()
        b = np.array([1.0, 1.0j]) / np.sqrt(2.0)
        A, B = _site([1.0, 0.0]), _site(b)
        rho = np.asarray(_rdm2x1_mixed(A, B, env, 2, ("B", "B"), ("A", "A"))).reshape(
            4, 4
        )
        expected = np.kron(np.outer(b, b.conj()), np.diag([1.0, 0.0]))
        np.testing.assert_allclose(rho, expected, atol=1e-12)

    def test_all_A_substitution_reduces_to_the_standard_rdm(self):
        """With B never substituted, the mixed helper is the ordinary RDM."""
        from tenax.algorithms.ipeps_rdm import _rdm2x1

        env = _exact_product_env()
        v = np.array([0.6, 0.8])
        A = _site(v)
        rho_mixed = np.asarray(
            _rdm2x1_mixed(A, A, env, 2, ("A", "A"), ("A", "A"))
        ).reshape(4, 4)
        rho_std = np.asarray(_rdm2x1(A, env, 2)).reshape(4, 4)
        np.testing.assert_allclose(rho_mixed, rho_std, atol=1e-12)
        np.testing.assert_allclose(np.trace(rho_mixed), 1.0, atol=1e-12)

    def test_mixed_rdm_trace_is_one_on_a_converged_environment(
        self, small_peps_and_env
    ):
        """The AA normaliser must be the trace of the same-geometry AA network.

        On a nontrivial converged environment (where that scalar is far from
        1) an all-A mixed RDM still has unit trace -- this is what pins the
        divisor to the pure-AA contraction rather than to nothing.
        """
        A, env, d = small_peps_and_env
        for helper in (_rdm2x1_mixed, _rdm1x2_mixed):
            rho = helper(A, A, env, d, ("A", "A"), ("A", "A"))
            tr = complex(_transition_trace(rho))
            np.testing.assert_allclose(tr, 1.0, atol=1e-10)


class TestQuadraticScaling954:
    """#954: the functionals must be quadratic forms in B.

    The defect normalised every B-dependent contraction by its own trace, so
    scaling B changed neither the norm nor the energy and the generalized
    eigenproblem assembled from their gradients was not the advertised
    excitation problem.
    """

    pytestmark = pytest.mark.core

    _K = jnp.array([0.4, 0.7])
    _H_DIAG = jnp.diag(jnp.array([0.0, 1.0, 1.0, 2.0])).reshape(2, 2, 2, 2)

    def _fixture(self):
        env = _exact_product_env()
        A = _site(jnp.array([1.0, 2.0]) / jnp.sqrt(5.0))
        B = _site([3.0, 4.0])
        return A, B, env

    def test_norm_scales_quadratically(self):
        A, B, env = self._fixture()
        base = float(_compute_norm(A, B, env, self._K, 2))
        assert base > 1.0  # anti-vacuous: the ratio below must divide something
        doubled = float(_compute_norm(A, 2.0 * B, env, self._K, 2))
        np.testing.assert_allclose(doubled, 4.0 * base, rtol=1e-12)

    def test_energy_scales_quadratically(self):
        A, B, env = self._fixture()
        base = float(
            _compute_excitation_energy(A, B, env, self._K, self._H_DIAG, 0.0, 2)
        )
        assert abs(base) > 1.0
        doubled = float(
            _compute_excitation_energy(A, 2.0 * B, env, self._K, self._H_DIAG, 0.0, 2)
        )
        np.testing.assert_allclose(doubled, 4.0 * base, rtol=1e-12)

    def test_complex_multiplier_scales_by_modulus_squared(self):
        A, B, env = self._fixture()
        base = float(_compute_norm(A, B, env, self._K, 2))
        scaled = float(
            _compute_norm(A, 2.0j * B.astype(jnp.complex128), env, self._K, 2)
        )
        np.testing.assert_allclose(scaled, 4.0 * base, rtol=1e-12)
        phased = float(
            _compute_norm(A, jnp.exp(0.3j) * B.astype(jnp.complex128), env, self._K, 2)
        )
        np.testing.assert_allclose(phased, base, rtol=1e-12)

    def test_mixed_rdm_itself_scales_quadratically(self):
        A, B, env = self._fixture()
        rho = np.asarray(_rdm2x1_mixed(A, B, env, 2, ("B", "B"), ("A", "A")))
        rho2 = np.asarray(_rdm2x1_mixed(A, 2.0 * B, env, 2, ("B", "B"), ("A", "A")))
        np.testing.assert_allclose(rho2, 4.0 * rho, rtol=1e-12)

    def test_scaling_holds_on_a_converged_environment(self, small_peps_and_env):
        """The quadratic property is not an artifact of the exact D=1 env."""
        A, env, d = small_peps_and_env
        B = jax.random.normal(jax.random.PRNGKey(11), A.shape)
        base = float(_compute_norm(A, B, env, self._K, d))
        assert abs(base) > 1e-6  # regime: nonzero base, the ratio is meaningful
        doubled = float(_compute_norm(A, 2.0 * B, env, self._K, d))
        np.testing.assert_allclose(doubled, 4.0 * base, rtol=1e-8)

    def test_functionals_are_invariant_under_an_environment_gauge_scalar(self):
        """Multiplying one corner by 2e^{0.7i} must change nothing.

        The raw transition networks are defined only up to the environment's
        arbitrary complex scalar (see ``_normalise_rdm``).  Dividing by the
        B-independent AA contraction cancels it because numerator and
        denominator share the SAME environment tensors -- this is the half of
        the old per-trace normalisation that had to survive #954.
        """
        A, B, env = self._fixture()
        fields = list(env)
        fields[0] = 2.0 * jnp.exp(0.7j) * fields[0].astype(jnp.complex128)
        env_scaled = CTMEnvironment(*fields)

        n1 = float(_compute_norm(A, B, env, self._K, 2))
        n2 = float(_compute_norm(A, B, env_scaled, self._K, 2))
        np.testing.assert_allclose(n2, n1, rtol=1e-12)

        e1 = float(_compute_excitation_energy(A, B, env, self._K, self._H_DIAG, 0.2, 2))
        e2 = float(
            _compute_excitation_energy(A, B, env_scaled, self._K, self._H_DIAG, 0.2, 2)
        )
        np.testing.assert_allclose(e2, e1, rtol=1e-12)


class TestEnergyFunctionalOracle:
    """Closed-form energy of the excitation ansatz on an exact product state.

    With A = |0>, B = |1>, the transition matrix element structure separates
    cleanly: SzSz sees only the on-site terms (-1/4 per bond, four bond
    terms), SxSx sees only the off-site hopping terms (1/4 each, carrying the
    momentum phases), and the identity in the E_gs shift sees exactly the
    four on-site terms.  Any axis-layout mistake in the RDMs, the gate, or
    the shift identity (#955) moves these numbers.
    """

    pytestmark = pytest.mark.core

    def setup_method(self):
        self.env = _exact_product_env()
        self.A, self.B = _site([1.0, 0.0]), _site([0.0, 1.0])
        self.Hxx = jnp.kron(_SX, _SX).reshape(2, 2, 2, 2)
        self.Hzz = jnp.kron(_SZ, _SZ).reshape(2, 2, 2, 2)

    def test_diagonal_gate_gives_the_onsite_sum(self):
        for k in (jnp.array([0.0, 0.0]), jnp.array([0.4, 0.7])):
            e = float(
                _compute_excitation_energy(
                    self.A, self.B, self.env, k, self.Hzz, 0.0, 2
                )
            )
            np.testing.assert_allclose(e, -1.0, atol=1e-12)

    def test_hopping_gate_gives_the_momentum_dispersion(self):
        for kx, ky in ((0.0, 0.0), (np.pi, 0.0), (0.4, 0.7)):
            k = jnp.array([kx, ky])
            e = float(
                _compute_excitation_energy(
                    self.A, self.B, self.env, k, self.Hxx, 0.0, 2
                )
            )
            np.testing.assert_allclose(e, 0.5 * (np.cos(kx) + np.cos(ky)), atol=1e-12)

    def test_ground_state_shift_subtracts_from_onsite_terms_only(self):
        k = jnp.array([0.4, 0.7])
        e0 = float(
            _compute_excitation_energy(self.A, self.B, self.env, k, self.Hxx, 0.0, 2)
        )
        e = float(
            _compute_excitation_energy(self.A, self.B, self.env, k, self.Hxx, 0.3, 2)
        )
        # -(E_gs / 2) per bond term; the identity sees the 4 on-site terms
        # (each 1) and none of the off-site ones (orthogonal transition).
        np.testing.assert_allclose(e, e0 - 2.0 * 0.3, atol=1e-12)

    def test_norm_counts_the_onsite_overlap_once(self):
        """<Phi|Phi> per site is exactly 1 on this fixture: <B|B> = 1 on-site
        and every off-site overlap vanishes (<A|B> = 0).

        Pre-fix the four windows each contributed the same on-site overlap
        and were summed, giving 4.0 — which silently scaled every
        generalized eigenvalue by 1/4 (review P1 on #961: the energy
        numerator was exact, so the pinned oracles above never noticed).
        """
        for k in (jnp.array([0.0, 0.0]), jnp.array([0.4, 0.7])):
            n = float(_compute_norm(self.A, self.B, self.env, k, 2))
            np.testing.assert_allclose(n, 1.0, atol=1e-12)

    def test_generalized_eigenvalue_matches_the_exact_dispersion(self):
        """End to end through H/N assembly: the Rayleigh quotient on the
        B = |1> direction equals the exact single-flip dispersion

            w(k) = -2 Jz + (Jx / 2)(cos kx + cos ky),   E_gs = Jz / 2.

        This is the ratio the P1 norm fix restores; it is exact on the
        product fixture, so any window double-counting (in H or N), a wrong
        E_gs bookkeeping, or an axis-layout slip moves it.
        """
        Jz, Jx = 0.7, 1.0
        gate = Jz * self.Hzz + Jx * self.Hxx
        for kx, ky in ((0.0, 0.0), (np.pi, 0.0), (0.4, 0.7)):
            H, N = _build_H_and_N(
                self.A,
                self.env,
                jnp.array([kx, ky]),
                gate,
                Jz / 2.0,
                2,
                ExcitationConfig(num_excitations=1),
            )
            quot = float((H[1, 1] / N[1, 1]).real)
            true = -2.0 * Jz + 0.5 * Jx * (np.cos(kx) + np.cos(ky))
            np.testing.assert_allclose(quot, true, atol=1e-12)

    def test_ground_state_direction_is_projected_to_an_exact_null(self):
        """B = A is exactly null at k != 0, but the NN-truncated norm
        misrepresents it as 1 + 2cos kx + 2cos ky — down to -3 at M (#961
        review round 2).  The projector must zero that direction in both
        matrices, restoring a PSD pencil, while the physical orthogonal
        mode keeps its exact dispersion through the solver.
        """
        k = jnp.array([np.pi, np.pi])
        # Regime pin: the raw truncated form really is -3 along B = A at M.
        np.testing.assert_allclose(
            float(_compute_norm(self.A, self.A, self.env, k, 2)), -3.0, atol=1e-12
        )
        Jz, Jx = 0.7, 1.0
        gate = Jz * self.Hzz + Jx * self.Hxx
        H, N = _build_H_and_N(
            self.A, self.env, k, gate, Jz / 2.0, 2, ExcitationConfig(num_excitations=2)
        )
        Hp, Np = _project_out_ground_state(H, N, self.A)
        eigs = np.linalg.eigvalsh(0.5 * (Np + Np.conj().T))
        assert eigs.min() > -1e-12, f"pencil still indefinite after projection: {eigs}"
        true = -2.0 * Jz + 0.5 * Jx * (np.cos(np.pi) + np.cos(np.pi))
        np.testing.assert_allclose(_solve_excitations(Hp, Np, 1)[0], true, atol=1e-12)

    def test_contaminated_momentum_returns_the_exact_dispersion_end_to_end(self):
        """The unprojected A-direction is a spurious ZERO level, not a
        shifted one: the E_gs identity annihilates every pure-GS window, so
        H[A-dir, A-dir] == 0 identically while its truncated norm at
        k = (1.8, 1.8) is small but positive (1 + 4cos 1.8 ~ 0.09) and
        survives the solver's null filter.  On a positive dispersion — the
        physical situation for a gapped system — that fake omega = 0 level
        undercuts the true mode and compute_excitations reports a gapless
        spectrum.  With the projection the direction is exactly removed and
        the exact dispersion comes back (review round 2 on #961).
        """
        Jz, Jx = -0.7, 1.0  # sign chosen so the true dispersion is positive
        gate = Jz * self.Hzz + Jx * self.Hxx
        result = compute_excitations(
            self.A,
            self.env,
            gate,
            Jz / 2.0,
            [(1.8, 1.8)],
            ExcitationConfig(num_excitations=1),
        )
        true = -2.0 * Jz + 0.5 * Jx * (np.cos(1.8) + np.cos(1.8))
        assert true > 1.0  # regime pin: the spurious 0 must sit BELOW the mode
        np.testing.assert_allclose(result.energies[0][0], true, atol=1e-12)

    def test_complex_hermitian_gate_contracts_against_bra_axes(self):
        """Sx (x) Sy is Hermitian but transposes to its negative, so it
        separates Tr(rho H) from Tr(rho H^T): the hopping dispersion is

            E(k) = -(1/2)(sin kx + sin ky),

        and the pre-fix elementwise contraction returned its negation
        (review P2 on #961 — every other oracle gate here is real
        symmetric, which is exactly the class that hides the transpose).
        """
        sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
        gate = jnp.einsum("ab,cd->acbd", _SX, sy)
        for kx, ky in ((0.5, 0.0), (0.0, 0.5), (0.4, 0.7)):
            e = float(
                _compute_excitation_energy(
                    self.A, self.B, self.env, jnp.array([kx, ky]), gate, 0.0, 2
                )
            )
            np.testing.assert_allclose(e, -0.5 * (np.sin(kx) + np.sin(ky)), atol=1e-12)


class TestComplexAssembly956:
    """#956: the H/N assembly must keep imaginary matrix elements.

    The real-only basis made ``jax.grad`` return float cotangents, so H and N
    lost every imaginary entry: an exact Hermitian quadratic form with
    spectrum [0, 2] assembled to the identity's spectrum [1, 1].
    """

    pytestmark = pytest.mark.core

    def test_exact_hermitian_quadratic_form_is_reconstructed(self):
        from unittest.mock import patch

        import tenax.algorithms.ipeps_excitations as ex

        env = _exact_product_env()
        A = _site([1.0, 0.0])
        M = jnp.array([[1.0, -1.0j], [1.0j, 1.0]])

        def energy(A, B, *args):
            return jnp.real(jnp.vdot(B.ravel(), M @ B.ravel()))

        def norm(A, B, *args):
            return jnp.real(jnp.vdot(B, B))

        with (
            patch.object(ex, "_compute_excitation_energy", energy),
            patch.object(ex, "_compute_norm", norm),
        ):
            H, N = ex._build_H_and_N(
                A,
                env,
                jnp.zeros(2),
                jnp.zeros((2, 2, 2, 2)),
                0.0,
                2,
                ExcitationConfig(num_excitations=2),
            )
        # Elementwise, not just spectrally: a conjugation flip in the
        # assembly produces M.T, whose eigenvalues are also [0, 2] but whose
        # off-diagonal entries are wrong.
        np.testing.assert_allclose(H, np.asarray(M), atol=1e-12)
        np.testing.assert_allclose(N, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(
            _solve_excitations(H, N, 2), np.linalg.eigvalsh(np.asarray(M)), atol=1e-12
        )

    def test_real_functionals_assemble_hermitian_matrices_on_the_exact_env(self):
        """End to end: on an exact environment the forms are exactly Hermitian
        and N is positive semidefinite -- no symmetrisation needed to see it."""
        env = _exact_product_env()
        A = _site([1.0, 0.0])
        gate = jnp.kron(_SX, _SX).reshape(2, 2, 2, 2)
        k = jnp.array([0.4, 0.7])

        H, N = _build_H_and_N(A, env, k, gate, 0.0, 2, ExcitationConfig())
        np.testing.assert_allclose(H, H.conj().T, atol=1e-10)
        np.testing.assert_allclose(N, N.conj().T, atol=1e-10)
        eigs = np.linalg.eigvalsh(0.5 * (N + N.conj().T))
        assert eigs.min() > -1e-10, f"N not PSD on an exact environment: {eigs}"
        assert np.abs(H).max() > 1e-6 and np.abs(N).max() > 1e-6  # anti-vacuous
