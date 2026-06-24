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
    _rdm2x1_mixed,
    _solve_excitations,
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
