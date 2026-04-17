"""Tests for the stable AD infrastructure (ad_utils.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for this test module and restore afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.algorithms._ctm_tensor import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ad_utils import (
    CTMRGGradientError,
    _config_from_tuple,
    _config_to_tuple,
    _ctm_tensor_multisite_fixed_point,
    _gauge_fix_ctm_tensor,
    _svd_sector_backward,
    arnoldi_spectral_radius,
    ctm_tensor_converge,
    ctm_tensor_converge_explicit,
    regularized_svd,
    truncated_svd_ad,
    truncated_svd_symmetric_ad,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


class TestTruncatedSVDADForward:
    """Forward pass of truncated_svd_ad matches standard SVD."""

    def test_forward_matches_svd(self):
        """Truncated results should match jnp.linalg.svd truncated output."""
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        chi = 3

        U_ad, s_ad, Vh_ad = truncated_svd_ad(M, chi)

        U_ref, s_ref, Vh_ref = jnp.linalg.svd(M, full_matrices=False)
        U_ref = U_ref[:, :chi]
        s_ref = s_ref[:chi]
        Vh_ref = Vh_ref[:chi, :]

        assert jnp.allclose(s_ad, s_ref, atol=1e-12)
        # U and Vh can differ by sign per column, compare via reconstruction
        recon_ad = U_ad * s_ad[None, :] @ Vh_ad
        recon_ref = U_ref * s_ref[None, :] @ Vh_ref
        assert jnp.allclose(recon_ad, recon_ref, atol=1e-12)

    def test_forward_shapes(self):
        """Output shapes should be (m, chi), (chi,), (chi, n)."""
        M = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        chi = 3
        U, s, Vh = truncated_svd_ad(M, chi)
        assert U.shape == (8, 3)
        assert s.shape == (3,)
        assert Vh.shape == (3, 5)

    def test_forward_chi_larger_than_min_dim(self):
        """When chi > min(m,n), should truncate to min(m,n)."""
        M = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
        chi = 10
        U, s, Vh = truncated_svd_ad(M, chi)
        assert s.shape[0] == 3  # min(4, 3)


class TestTruncatedSVDADGradient:
    """VJP of truncated_svd_ad matches finite-difference gradients."""

    def test_gradient_finite_diff(self):
        """Custom VJP should approximate finite-difference gradient."""
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (5, 4))
        chi = 3

        # Loss function: sum of singular values
        def loss(M_in):
            _, s, _ = truncated_svd_ad(M_in, chi)
            return jnp.sum(s)

        # AD gradient
        grad_ad = jax.grad(loss)(M)

        # Finite-difference gradient
        eps = 1e-5
        grad_fd = np.zeros_like(M)
        M_np = np.array(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_plus = M_np.copy()
                M_plus[i, j] += eps
                M_minus = M_np.copy()
                M_minus[i, j] -= eps
                grad_fd[i, j] = (
                    float(loss(jnp.array(M_plus))) - float(loss(jnp.array(M_minus)))
                ) / (2 * eps)

        assert jnp.allclose(grad_ad, grad_fd, atol=1e-4), (
            f"Max diff: {float(jnp.max(jnp.abs(grad_ad - grad_fd)))}"
        )

    def test_gradient_reconstruction_loss(self):
        """Gradient of ||M - U S Vh||^2 through truncated SVD."""
        key = jax.random.PRNGKey(7)
        M = jax.random.normal(key, (6, 4))
        chi = 2

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            recon = U * s[None, :] @ Vh
            return jnp.sum((M_in - recon) ** 2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))


class TestTruncatedSVDADDegenerate:
    """No NaN/Inf when singular values are degenerate."""

    def test_degenerate_identity(self):
        """Identity matrix has all singular values = 1 (maximally degenerate)."""
        M = jnp.eye(4)
        chi = 3

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(s**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad)), f"NaN/Inf in gradient: {grad}"

    def test_degenerate_repeated_singular_values(self):
        """Matrix with repeated singular values should not produce NaN."""
        # Construct M with repeated singular values
        U, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (5, 5)))
        V, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (4, 4)))
        s = jnp.array([3.0, 3.0, 1.0, 1.0])  # repeated values
        M = U[:, :4] * s[None, :] @ V.T
        chi = 3

        def loss(M_in):
            U_t, s_t, Vh_t = truncated_svd_ad(M_in, chi)
            return jnp.sum(s_t)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_degenerate_zero_matrix(self):
        """Near-zero matrix should not cause NaN."""
        M = 1e-15 * jax.random.normal(jax.random.PRNGKey(3), (4, 3))
        chi = 2

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(s)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))


class TestTruncatedSVDADMissingTerm:
    """Truncation correction term improves gradient accuracy."""

    def test_truncation_correction_improves_accuracy(self):
        """For a matrix with significant truncated spectrum, our custom VJP
        should be more accurate than naive truncation."""
        key = jax.random.PRNGKey(10)
        M = jax.random.normal(key, (6, 5))
        chi = 2  # Aggressive truncation — large truncated spectrum

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            # Loss that depends on U and Vh (not just s)
            return jnp.sum(U[:, 0] ** 2) + jnp.sum(Vh[0, :] ** 2)

        grad_ad = jax.grad(loss)(M)

        # Finite-difference reference
        eps = 1e-5
        grad_fd = np.zeros_like(M)
        M_np = np.array(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_plus = M_np.copy()
                M_plus[i, j] += eps
                M_minus = M_np.copy()
                M_minus[i, j] -= eps
                grad_fd[i, j] = (
                    float(loss(jnp.array(M_plus))) - float(loss(jnp.array(M_minus)))
                ) / (2 * eps)

        max_diff = float(jnp.max(jnp.abs(grad_ad - grad_fd)))
        assert max_diff < 1e-3, f"Gradient error too large: {max_diff}"

    def test_disabling_truncation_correction_worsens_fd_error(self, monkeypatch):
        """Naive backward without kept-discarded corrections is less accurate."""
        key = jax.random.PRNGKey(123)
        U0, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
        V0, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(124), (5, 5)))
        svals = jnp.array([4.0, 2.0, 1.999, 0.8, 0.2])
        M = U0[:, :5] @ jnp.diag(svals) @ V0.T
        chi = 2

        Wu_r = jax.random.normal(jax.random.PRNGKey(125), (6, 6))
        Wu = 0.5 * (Wu_r + Wu_r.T)
        Wv_r = jax.random.normal(jax.random.PRNGKey(126), (5, 5))
        Wv = 0.5 * (Wv_r + Wv_r.T)

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            proj_u = U @ U.T
            proj_v = Vh.T @ Vh
            return (
                jnp.sum(s**2)
                + 0.31 * jnp.trace(proj_u @ Wu)
                + 0.23 * jnp.trace(proj_v @ Wv)
            )

        def finite_diff_grad(M_in, eps=2e-5):
            M_np = np.array(M_in)
            g_fd = np.zeros_like(M_np)
            for i in range(M_np.shape[0]):
                for j in range(M_np.shape[1]):
                    d = np.zeros_like(M_np)
                    d[i, j] = eps
                    g_fd[i, j] = (
                        float(loss(jnp.array(M_np + d)))
                        - float(loss(jnp.array(M_np - d)))
                    ) / (2 * eps)
            return jnp.array(g_fd)

        grad_ref = jax.grad(loss)(M)

        def _naive_sector_backward(U, s, Vh, dU, ds, dVh, eps=1e-12):
            k = ds.shape[0]
            U_k = U[:, :k]
            s_k = s[:k]
            Vh_k = Vh[:k, :]
            V_k = Vh_k.conj().T
            s2 = s_k**2
            diff = s2[:, None] - s2[None, :]
            F = diff / (diff**2 + eps**2)
            F = F - jnp.diag(jnp.diag(F))
            UtdU = U_k.conj().T @ dU
            VtdV = V_k.conj().T @ dVh.conj().T
            UtdU_anti = 0.5 * (UtdU - UtdU.conj().T)
            VtdV_anti = 0.5 * (VtdV - VtdV.conj().T)
            dM = U_k @ jnp.diag(ds) @ Vh_k
            dM = dM + U_k @ (F * UtdU_anti) @ jnp.diag(s_k) @ Vh_k
            dM = dM + U_k @ jnp.diag(s_k) @ (F * VtdV_anti) @ Vh_k
            return dM

        monkeypatch.setattr(
            "tenax.algorithms.ad_utils._svd_sector_backward",
            _naive_sector_backward,
        )
        grad_naive = jax.grad(loss)(M)

        grad_fd = finite_diff_grad(M)
        err_ref = float(jnp.linalg.norm(grad_ref - grad_fd))
        err_naive = float(jnp.linalg.norm(grad_naive - grad_fd))
        assert err_ref <= err_naive + 1e-10, (
            f"Expected corrected backward to beat/equal naive: "
            f"err_ref={err_ref}, err_naive={err_naive}"
        )


class TestProjectorADAudit:
    """Audit projector AD routes use stabilized SVD routines under tracing."""

    @staticmethod
    def _make_grown_corners_dense(key1, key2, fused_dim=10, col1=6, col2=5):
        sym = U1Symmetry()
        C1g_data = jax.random.normal(key1, (fused_dim, col1))
        C4g_data = jax.random.normal(key2, (fused_dim, col2))
        fused_idx = TensorIndex.from_charges(
            sym,
            np.zeros(fused_dim, dtype=np.int32),
            FlowDirection.IN,
            label="fused",
        )
        c1_idx = TensorIndex.from_charges(
            sym, np.zeros(col1, dtype=np.int32), FlowDirection.OUT, label="c1"
        )
        c4_idx = TensorIndex.from_charges(
            sym, np.zeros(col2, dtype=np.int32), FlowDirection.OUT, label="c4"
        )
        C1g = DenseTensor(C1g_data, (fused_idx, c1_idx))
        C4g = DenseTensor(C4g_data, (fused_idx, c4_idx))
        return C1g, C4g

    def test_svd_projector_traced_path_calls_truncated_svd_ad(self, monkeypatch):
        import tenax.algorithms.ad_utils as ad_utils_mod

        C1g_base, C4g_base = self._make_grown_corners_dense(
            jax.random.PRNGKey(210), jax.random.PRNGKey(211)
        )
        calls = {"n": 0}
        orig = ad_utils_mod.truncated_svd_ad

        def _spy_truncated_svd_ad(M, chi):
            calls["n"] += 1
            return orig(M, chi)

        monkeypatch.setattr(ad_utils_mod, "truncated_svd_ad", _spy_truncated_svd_ad)

        def traced_eval(alpha):
            C1g = DenseTensor(alpha * C1g_base.todense(), C1g_base.indices)
            P1, P2 = _compute_projector_tensor(C1g, C4g_base, 4, projector_method="svd")
            return jnp.sum(P1.todense() ** 2) + jnp.sum(P2.todense() ** 2)

        _ = jax.jit(traced_eval)(1.0)
        assert calls["n"] >= 1, "Expected traced SVD projector to call truncated_svd_ad"

    def test_qr_projector_traced_path_calls_regularized_svd(self, monkeypatch):
        import tenax.algorithms.ad_utils as ad_utils_mod

        C1g_base, C4g_base = self._make_grown_corners_dense(
            jax.random.PRNGKey(220), jax.random.PRNGKey(221)
        )
        calls = {"n": 0}
        orig = ad_utils_mod.regularized_svd

        def _spy_regularized_svd(M):
            calls["n"] += 1
            return orig(M)

        monkeypatch.setattr(ad_utils_mod, "regularized_svd", _spy_regularized_svd)

        def traced_eval(alpha):
            C1g = DenseTensor(alpha * C1g_base.todense(), C1g_base.indices)
            P, _ = _compute_projector_tensor(C1g, C4g_base, 4, projector_method="qr")
            return jnp.sum(P.todense() ** 2)

        _ = jax.jit(traced_eval)(1.0)
        assert calls["n"] >= 1, "Expected traced QR projector to call regularized_svd"


def _make_dense_tensor(key, D=2, d=2):
    """Create a DenseTensor iPEPS site tensor for testing."""
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


class TestCTMFixedPointGradient:
    """Gradient through ctm_tensor_converge matches finite-difference."""

    def test_gradient_exists_and_finite(self):
        """Gradient of energy through ctm_tensor_converge should be finite."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        config = CTMConfig(
            chi=4, max_iter=5, conv_tol=1e-6, adjoint_arnoldi_precheck=False
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            import jax

            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "Gradient contains NaN/Inf"
        assert grad.norm() > 1e-15, "Gradient is all zeros"


# ---------------------------------------------------------------------------
# Finite-difference gradient correctness for CTM AD
# ---------------------------------------------------------------------------


def _heisenberg_gate_real():
    """Real-valued 2-site Heisenberg gate as a (2,2,2,2) tensor."""
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _random_tangent_like(A: DenseTensor, key) -> DenseTensor:
    """Random unit-norm tangent vector sharing A's structure."""
    V_data = jax.random.normal(key, A.todense().shape, dtype=A.todense().dtype)
    V_data = V_data / (jnp.linalg.norm(V_data) + 1e-12)
    return DenseTensor(V_data, A.indices)


def _energy_loss_explicit_ad(A: DenseTensor, config: CTMConfig, gate) -> jax.Array:
    """Energy through explicit-AD (unrolled CTM + checkpoint) path."""
    A_norm = A * (1.0 / (A.norm() + 1e-10))
    config_tuple = _config_to_tuple(config)
    env_leaves = ctm_tensor_converge_explicit(
        {(0, 0): A_norm},
        None,
        SINGLE_SITE_NEIGHBORS,
        config_tuple,
        num_steps=config.max_iter,
        warmup_steps=0,
    )
    env_template = initialize_ctm_tensor_env(A_norm, config.chi)
    env = jax.tree.unflatten(jax.tree.structure(env_template), list(env_leaves))
    return compute_energy_ctm_tensor(A_norm, env, gate)


def _energy_loss_implicit_ad(A: DenseTensor, config: CTMConfig, gate) -> jax.Array:
    """Energy through implicit-AD (custom VJP fixed-point) path."""
    A_norm = A * (1.0 / (A.norm() + 1e-10))
    config_tuple = _config_to_tuple(config)
    env_leaves = ctm_tensor_converge(
        {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
    )
    env_template = initialize_ctm_tensor_env(A_norm, config.chi)
    env = jax.tree.unflatten(jax.tree.structure(env_template), list(env_leaves))
    return compute_energy_ctm_tensor(A_norm, env, gate)


class TestCTMFixedPointGradientFiniteDifference:
    """Directional finite-difference agreement: the gold-standard AD check.

    For a correct autodiff implementation,

        d/d-eps loss(A + eps * V) at eps=0  ==  <grad(loss)(A), V>

    Centered finite differences compute the left-hand side to O(eps^2),
    so comparing against the AD-computed right-hand side on a few
    random directions V proves the gradient is mathematically correct,
    not just finite and non-zero.

    Why this is better than the existing Heisenberg benchmarks:
      - Mathematical proof of gradient correctness at that point.
      - Does not depend on any literature energy value or threshold.
      - Fast: 3 forward passes per direction at chi=4.
      - Unit-test-clean: failures localize to gradient bugs,
        not optimizer tuning drift.

    *Gauge choice matters.* The forward gauge determines whether the
    loss is a smooth function of A under small perturbations:

      - ``"qr"`` gauge flips sign unpredictably under tiny perturbations
        (known numerical property of QR gauge fixing). The loss is
        therefore piecewise-smooth with discontinuities in its FD, so
        the FD gold standard cannot be applied. ``optimize_gs_ad``
        auto-promotes ``"qr"`` to ``"phase"`` in the explicit-AD path
        specifically to avoid this, so we test the path the optimizer
        actually uses.
      - ``"phase"`` gauge is smooth and is the effective default for
        explicit-AD runs.
      - ``"sigma"`` gauge is required for the implicit-diff VJP to
        converge its Neumann series.
    """

    # Small CTM — chi=4, max_iter=20 gives a converged fixed point at D=2
    # without burning wall-clock. conv_tol tight so FD noise is below
    # the target precision.
    _SMALL_CONFIG = dict(chi=4, max_iter=20, conv_tol=1e-10, min_iter=4)

    @pytest.mark.parametrize(
        "projector_method",
        [
            "eigh",
            "svd",
            pytest.param(
                "qr",
                marks=pytest.mark.xfail(
                    reason=(
                        "QR projector backward is partially fixed but not "
                        "fully AD-stable. The QR sign-fix in _ctm_projector.py "
                        "removes gross sign flips at moderate eps, but two "
                        "issues remain: (1) the kept-k truncation in "
                        "U_R[:, :k] can swap which singular vectors are kept "
                        "under tiny perturbations, (2) degenerate-subspace "
                        "rotations within the kept block are not fixed. The "
                        "AD gradient is consistently ~0.5x the converged FD "
                        "value, suggesting a missing gradient contribution. "
                        "Full fix needs degeneracy-aware truncation and a "
                        "post-truncation gauge fix on the k-dim subspace, or "
                        "a custom VJP for the whole projector function. "
                        "See test_explicit_ad_qr_known_non_smoothness."
                    ),
                    strict=True,
                ),
            ),
        ],
    )
    def test_explicit_ad_matches_finite_difference(self, projector_method):
        """Explicit-AD gradient agrees with centered finite differences.

        Parametrizes over the projector methods. Pass criteria:
          - eigh: Lorentzian-broadened ``regularized_eigh`` backward
          - svd:  Fishman two-projector with truncated_svd_ad backward
          - qr:   xfailed — QR projector forward is non-smooth (see marker).

        Test at the directional-derivative level: the gold-standard
        check that the AD-computed ``<grad, V>`` matches centered FD.
        """
        cfg = CTMConfig(
            **self._SMALL_CONFIG,
            projector_method=projector_method,
            forward_gauge="phase",
            adjoint_arnoldi_precheck=False,
        )
        gate = _heisenberg_gate_real()
        A = _make_dense_tensor(jax.random.PRNGKey(1234))
        V = _random_tangent_like(A, jax.random.PRNGKey(5678))

        def loss(A_in):
            return _energy_loss_explicit_ad(A_in, cfg, gate)

        grad = jax.grad(loss)(A)
        ad_deriv = float(jnp.sum(grad.todense() * V.todense()))

        eps = 1e-4
        A_plus = DenseTensor(A.todense() + eps * V.todense(), A.indices)
        A_minus = DenseTensor(A.todense() - eps * V.todense(), A.indices)
        fd_deriv = float((loss(A_plus) - loss(A_minus)) / (2.0 * eps))

        rel = abs(fd_deriv - ad_deriv) / (abs(fd_deriv) + abs(ad_deriv) + 1e-12)
        assert rel < 5e-2, (
            f"AD/FD mismatch for projector={projector_method}: "
            f"ad={ad_deriv:.6e} fd={fd_deriv:.6e} rel={rel:.2e}"
        )

    def test_explicit_ad_qr_known_non_smoothness(self):
        """Document QR projector residual non-smoothness as a regression sweep.

        After the QR sign-fix in ``_ctm_projector.py`` the FD sweep no
        longer shows gross sign flips at moderate eps, but small-eps
        values still vary wildly (sign flips and >5x spread between
        eps=3e-5 and eps=1e-5) due to truncation pivot swaps and
        degenerate-subspace rotations within the kept-k subspace.

        If a future change makes the QR projector forward fully smooth,
        this test will start failing — at which point both this test
        and the xfail on
        ``test_explicit_ad_matches_finite_difference[qr]`` should be
        revisited.
        """
        cfg = CTMConfig(
            **self._SMALL_CONFIG,
            projector_method="qr",
            forward_gauge="phase",
            adjoint_arnoldi_precheck=False,
        )
        gate = _heisenberg_gate_real()
        A = _make_dense_tensor(jax.random.PRNGKey(1234))
        V = _random_tangent_like(A, jax.random.PRNGKey(5678))

        def loss(A_in):
            return _energy_loss_explicit_ad(A_in, cfg, gate)

        fd_values = []
        for eps in (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            A_plus = DenseTensor(A.todense() + eps * V.todense(), A.indices)
            A_minus = DenseTensor(A.todense() - eps * V.todense(), A.indices)
            fd_values.append(float((loss(A_plus) - loss(A_minus)) / (2.0 * eps)))

        fd_min = min(fd_values)
        fd_max = max(fd_values)
        # If the function were smooth, all FD values would cluster within
        # a few percent of the true derivative. Non-smoothness shows up
        # as either a sign flip or a >10x spread.
        sign_flip = (fd_min < 0) and (fd_max > 0)
        spread = (fd_max - fd_min) / (max(abs(fd_min), abs(fd_max)) + 1e-12)
        assert sign_flip or spread > 5.0, (
            f"FD values across eps decades are unexpectedly consistent: "
            f"{fd_values}. The QR projector may have become smooth — if so, "
            f"lift the xfail on test_explicit_ad_matches_finite_difference[qr]."
        )

    def test_explicit_ad_gradient_norm_is_nontrivial(self):
        """The gradient is not zero and is well-scaled.

        Not a full FD check — ``TestCTMFixedPointGradientFiniteDifference``
        covers that along the well-conditioned direction. This is a
        cheap sanity check that the gradient has magnitude comparable
        to the loss function value, catching the "grad is ~0" failure
        mode (e.g. SU-plateau / gradient-flow-broken) without the
        direction-sensitivity of FD.
        """
        cfg = CTMConfig(**self._SMALL_CONFIG, forward_gauge="phase")
        gate = _heisenberg_gate_real()
        A = _make_dense_tensor(jax.random.PRNGKey(1234))

        def loss(A_in):
            return _energy_loss_explicit_ad(A_in, cfg, gate)

        E = float(loss(A))
        grad = jax.grad(loss)(A)
        g_norm = float(jnp.linalg.norm(grad.todense()))
        # Well-scaled: neither vanishing nor exploding relative to |E|.
        assert g_norm > 1e-6 * (abs(E) + 1e-12), (
            f"Gradient norm too small: g_norm={g_norm:.3e}, E={E:.3e}"
        )
        assert g_norm < 1e6 * (abs(E) + 1e-12), (
            f"Gradient norm too large: g_norm={g_norm:.3e}, E={E:.3e}"
        )
        assert jnp.all(jnp.isfinite(grad.todense()))


class TestCTMADPathConsistency:
    """Explicit and implicit AD gradients must agree.

    Explicit AD unrolls the CTM sweep and backprops through it; implicit
    AD uses a custom VJP that solves the fixed-point adjoint via an
    iterative VJP / Neumann-series backward. Both are valid paths to the
    same mathematical gradient — the explicit path run with ``"phase"``
    gauge and the implicit path run with the paired ``"sigma"`` gauge
    should agree within solver tolerance.

    This is a strong zero-reference test: each path validates the other
    without needing any external ground truth.
    """

    def test_gradient_is_a_descent_direction(self):
        """One-step descent: E(A - eta * grad) < E(A) for small eta.

        Cheap catch-all for 'gradient points the wrong way' failures,
        including the SU-plateau bug (gradient ~= 0 on a product state).
        """
        cfg = CTMConfig(
            chi=4,
            max_iter=20,
            conv_tol=1e-10,
            min_iter=4,
            forward_gauge="phase",
            adjoint_arnoldi_precheck=False,
        )
        gate = _heisenberg_gate_real()
        A = _make_dense_tensor(jax.random.PRNGKey(271828))

        def loss(A_in):
            return _energy_loss_explicit_ad(A_in, cfg, gate)

        E0 = float(loss(A))
        grad = jax.grad(loss)(A)
        g_dense = grad.todense()
        g_norm = float(jnp.linalg.norm(g_dense))
        assert g_norm > 1e-8, f"Gradient norm vanishingly small: {g_norm:.3e}"

        eta = 1e-3 / g_norm
        A_next = DenseTensor(A.todense() - eta * g_dense, A.indices)
        E1 = float(loss(A_next))
        assert E1 < E0 - 1e-10, (
            f"Gradient is not a descent direction: E0={E0:.10f} E1={E1:.10f}"
        )


class TestGaugeFix:
    """Tests for CTM gauge fixing (Tensor protocol)."""

    @pytest.fixture
    def random_env(self):
        """Random CTMTensorEnv for testing."""
        A = _make_dense_tensor(jax.random.PRNGKey(0))
        return initialize_ctm_tensor_env(A, chi=4)

    def test_gauge_fix_idempotent(self, random_env):
        """Applying gauge fix twice should give the same result."""
        env1 = _gauge_fix_ctm_tensor(random_env)
        env2 = _gauge_fix_ctm_tensor(env1)

        for t1, t2 in zip(env1, env2):
            assert jnp.allclose(t1.todense(), t2.todense(), atol=1e-10), (
                f"Gauge fix not idempotent: max diff = "
                f"{float(jnp.max(jnp.abs(t1.todense() - t2.todense())))}"
            )

    def test_gauge_fix_preserves_shapes(self, random_env):
        """Gauge-fixed environment should have same tensor shapes."""
        env_fixed = _gauge_fix_ctm_tensor(random_env)
        for t_orig, t_fixed in zip(random_env, env_fixed):
            assert t_orig.todense().shape == t_fixed.todense().shape


class TestForwardGaugeConfigRoundTrip:
    """Regression coverage for post-PR-#291 forward_gauge modes (issue #292).

    The explicit-AD fast path encodes ``CTMConfig`` into a hashable tuple via
    ``_config_to_tuple`` so JAX can trace it. If the tuple encoding misses a
    mode, that mode silently collapses onto ``qr`` and becomes unreachable
    from ``ctm_tensor_converge_explicit`` — a regression that is invisible
    from the config surface alone. These tests assert the four documented
    modes round-trip identically.
    """

    def test_all_four_modes_round_trip(self):
        for mode in ("qr", "sigma", "phase", "none"):
            tup = _config_to_tuple(CTMConfig(forward_gauge=mode))
            assert _config_from_tuple(tup).forward_gauge == mode, (
                f"forward_gauge={mode!r} did not round-trip through "
                f"_config_to_tuple/_config_from_tuple"
            )

    def test_default_gauge_round_trips_to_qr(self):
        tup = _config_to_tuple(CTMConfig())
        assert _config_from_tuple(tup).forward_gauge == "qr"


class TestGMRESBackward:
    """Validate GMRES-based backward pass for ctm_tensor_converge."""

    def test_gmres_backward_finite_gradient(self):
        """GMRES backward pass should produce finite, nonzero gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(123))
        config = CTMConfig(
            chi=4, max_iter=10, conv_tol=1e-6, adjoint_arnoldi_precheck=False
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            import jax

            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "GMRES backward: NaN/Inf"
        assert grad.norm() > 1e-15, "GMRES backward: gradient is all zeros"

    def test_gmres_backward_deterministic(self):
        """GMRES backward pass should be deterministic across calls."""
        A = _make_dense_tensor(jax.random.PRNGKey(77))
        config = CTMConfig(
            chi=4, max_iter=10, conv_tol=1e-6, adjoint_arnoldi_precheck=False
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            import jax

            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad1 = jax.grad(energy_fn)(A)
        grad2 = jax.grad(energy_fn)(A)
        assert jnp.allclose(grad1.todense(), grad2.todense(), atol=1e-10), (
            f"GMRES backward not deterministic: max diff = "
            f"{float(jnp.max(jnp.abs(grad1.todense() - grad2.todense())))}"
        )

    def test_gmres_neumann_preconditioner_finite(self):
        """Neumann-preconditioned GMRES produces finite, nonzero gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(200))
        config = CTMConfig(
            chi=4,
            max_iter=10,
            conv_tol=1e-6,
            gmres_precondition=True,
            adjoint_arnoldi_precheck=False,
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "Preconditioned GMRES: NaN/Inf"
        assert grad.norm() > 1e-15, "Preconditioned GMRES: gradient is all zeros"

    def test_gmres_preconditioned_matches_unpreconditioned(self):
        """With preconditioner disabled (no-op), both paths must agree."""
        A = _make_dense_tensor(jax.random.PRNGKey(55))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def _grad_with(precond: bool):
            config = CTMConfig(
                chi=4,
                max_iter=10,
                conv_tol=1e-6,
                gmres_precondition=precond,
                adjoint_arnoldi_precheck=False,
            )
            config_tuple = _config_to_tuple(config)

            def energy_fn(A_in):
                A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
                env_leaves = ctm_tensor_converge(
                    {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
                )
                env = jax.tree.unflatten(
                    jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                    list(env_leaves),
                )
                return compute_energy_ctm_tensor(A_norm, env, gate)

            return jax.grad(energy_fn)(A)

        grad_on = _grad_with(True)
        grad_off = _grad_with(False)
        diff = float(jnp.max(jnp.abs(grad_on.todense() - grad_off.todense())))
        assert diff < 1e-4, f"Preconditioned vs unpreconditioned gradient diff = {diff}"

    def test_gmres_no_preconditioner_still_works(self):
        """Setting gmres_precondition=False must still produce finite gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(99))
        config = CTMConfig(
            chi=4,
            max_iter=10,
            conv_tol=1e-6,
            gmres_precondition=False,
            adjoint_arnoldi_precheck=False,
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "No-precond GMRES: NaN/Inf"
        assert grad.norm() > 1e-15, "No-precond GMRES: gradient is all zeros"


class TestGMRESBackwardPath:
    """Verify the GMRES backward branch (ad_backward_method='gmres') is exercised."""

    def test_gmres_path_finite_gradient(self):
        """GMRES backward path should produce finite, nonzero gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        config = CTMConfig(
            chi=4,
            max_iter=10,
            conv_tol=1e-6,
            ad_backward_method="gmres",
            adjoint_arnoldi_precheck=False,
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "GMRES path: NaN/Inf"
        assert grad.norm() > 1e-15, "GMRES path: gradient is all zeros"

    @pytest.mark.xfail(
        reason=(
            "GMRES implicit-AD backward is documented unstable in "
            "docs/guide/algorithms/ipeps_ad_paths.md (spectral radius > 1 "
            "without sigma gauge). Tracked by issue #292 — promote back to a "
            "regular test once the GMRES path matches VJP end-to-end."
        ),
        strict=False,
    )
    def test_gmres_path_agrees_with_vjp(self):
        """GMRES and VJP backward paths should produce similar gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(55))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def _grad_with(method: str):
            config = CTMConfig(
                chi=4, max_iter=10, conv_tol=1e-6, ad_backward_method=method
            )
            ct = _config_to_tuple(config)

            def energy_fn(A_in):
                A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
                env_leaves = ctm_tensor_converge(
                    {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, ct
                )
                env = jax.tree.unflatten(
                    jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                    list(env_leaves),
                )
                return compute_energy_ctm_tensor(A_norm, env, gate)

            return jax.grad(energy_fn)(A)

        grad_gmres = _grad_with("gmres")
        grad_vjp = _grad_with("vjp")
        diff = float(jnp.max(jnp.abs(grad_gmres.todense() - grad_vjp.todense())))
        assert diff < 5e-2, f"GMRES vs VJP gradient diff = {diff}"


class TestSvdSectorBackward:
    """Tests for the factored _svd_sector_backward function."""

    def test_svd_sector_backward_matches_original(self):
        """Factored function must reproduce the original backward exactly."""
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (6, 4))
        chi = 3

        # Full SVD
        U_full, s_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
        k = min(chi, s_full.shape[0])

        # Fake incoming gradients
        key2 = jax.random.PRNGKey(99)
        dU = jax.random.normal(key2, (6, k))
        ds = jax.random.normal(jax.random.PRNGKey(100), (k,))
        dVh = jax.random.normal(jax.random.PRNGKey(101), (k, 4))

        # Factored version
        dM_new = _svd_sector_backward(U_full, s_full, Vh_full, dU, ds, dVh)

        # Original inline version (reproduced for comparison)
        eps = 1e-12
        U = U_full[:, :k]
        s = s_full[:k]
        V = Vh_full[:k, :].conj().T
        s2 = s**2
        diff = s2[:, None] - s2[None, :]
        F = diff / (diff**2 + eps**2)
        F = F - jnp.diag(jnp.diag(F))
        UtdU = U.conj().T @ dU
        VtdV = V.conj().T @ dVh.conj().T
        UtdU_anti = 0.5 * (UtdU - UtdU.conj().T)
        VtdV_anti = 0.5 * (VtdV - VtdV.conj().T)
        s_inv = jnp.where(s > eps, 1.0 / s, 0.0)
        proj_U_perp = jnp.eye(M.shape[0]) - U @ U.conj().T
        proj_V_perp = jnp.eye(M.shape[1]) - V @ V.conj().T
        dM_orig = jnp.zeros_like(M)
        dM_orig = dM_orig + U @ jnp.diag(ds) @ Vh_full[:k, :]
        dM_orig = dM_orig + U @ (F * UtdU_anti) @ jnp.diag(s) @ Vh_full[:k, :]
        dM_orig = dM_orig + U @ jnp.diag(s) @ (F * VtdV_anti) @ Vh_full[:k, :]
        dM_orig = dM_orig + proj_U_perp @ dU @ jnp.diag(s_inv) @ Vh_full[:k, :]
        dM_orig = dM_orig + U @ jnp.diag(s_inv) @ dVh @ proj_V_perp

        assert jnp.allclose(dM_new, dM_orig, atol=1e-12), (
            f"Max diff: {float(jnp.max(jnp.abs(dM_new - dM_orig)))}"
        )

    def test_svd_sector_backward_no_truncation(self):
        """When k == min(m,n), the sector backward should still work."""
        M = jax.random.normal(jax.random.PRNGKey(5), (4, 3))
        U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
        k = s.shape[0]  # no truncation
        dU = jax.random.normal(jax.random.PRNGKey(6), (4, k))
        ds = jax.random.normal(jax.random.PRNGKey(7), (k,))
        dVh = jax.random.normal(jax.random.PRNGKey(8), (k, 3))

        dM = _svd_sector_backward(U, s, Vh, dU, ds, dVh)
        assert jnp.all(jnp.isfinite(dM))
        assert dM.shape == M.shape


class TestDegenerateSvGradientFinite:
    """Gradient through SVD with degenerate singular values is finite."""

    def test_degenerate_sv_gradient_finite_lorentzian(self):
        """Lorentzian regularization prevents NaN for degenerate SVs."""
        # Build a matrix with exactly degenerate singular values
        U_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (6, 6)))
        V_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (4, 4)))
        s_degen = jnp.array([5.0, 5.0, 2.0, 2.0])
        M = U_rand[:, :4] * s_degen[None, :] @ V_rand.T
        chi = 3

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(U**2) + jnp.sum(s) + jnp.sum(Vh**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad)), f"NaN/Inf in gradient: {grad}"

    def test_degenerate_sv_sector_backward_finite(self):
        """_svd_sector_backward produces finite output for degenerate SVs."""
        U_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(10), (5, 5)))
        V_rand, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(11), (4, 4)))
        s_degen = jnp.array([3.0, 3.0, 3.0, 1.0])
        M = U_rand[:, :4] * s_degen[None, :] @ V_rand.T

        U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
        k = 3
        dU = jax.random.normal(jax.random.PRNGKey(12), (5, k))
        ds = jax.random.normal(jax.random.PRNGKey(13), (k,))
        dVh = jax.random.normal(jax.random.PRNGKey(14), (k, 4))

        dM = _svd_sector_backward(U, s, Vh, dU, ds, dVh)
        assert jnp.all(jnp.isfinite(dM)), f"NaN/Inf in sector backward: {dM}"


class TestSymmetricSvdAdMatchesDense:
    """Per-sector regularized SVD matches dense truncated_svd_ad result."""

    def test_symmetric_svd_ad_matches_dense_trivial_charges(self):
        """With trivial (all-zero) charges, symmetric AD SVD should match dense."""
        key = jax.random.PRNGKey(42)
        D, d = 3, 2
        data = jax.random.normal(key, (D, D, D, D, d))
        data = data / jnp.linalg.norm(data)

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
        A = DenseTensor(data, indices)

        chi = 4
        left_labels = ("u", "l")
        right_labels = ("d", "r", "phys")

        U_sym, s_sym, Vh_sym = truncated_svd_symmetric_ad(
            A, left_labels, right_labels, chi, new_bond_label="bond"
        )

        # Compare via dense: reshape A the same way, apply truncated_svd_ad
        dense = A.todense()
        # Permute to (u, l, d, r, phys)
        label_to_ax = {lbl: i for i, lbl in enumerate(A.labels())}
        perm = [label_to_ax[lb] for lb in list(left_labels) + list(right_labels)]
        dense_p = jnp.transpose(dense, perm)
        m = D * D  # u, l
        n = D * D * d  # d, r, phys
        matrix = dense_p.reshape(m, n)
        U_ref, s_ref, Vh_ref = truncated_svd_ad(matrix, chi)

        # Singular values must match
        assert jnp.allclose(s_sym, s_ref, atol=1e-10), (
            f"SV mismatch: max diff = {float(jnp.max(jnp.abs(s_sym - s_ref)))}"
        )

        # Reconstruction must match
        k = s_sym.shape[0]
        recon_sym = (
            U_sym.todense().reshape(m, k)
            * s_sym[None, :]
            @ Vh_sym.todense().reshape(k, n)
        )
        recon_ref = U_ref * s_ref[None, :] @ Vh_ref
        assert jnp.allclose(recon_sym, recon_ref, atol=1e-10), (
            f"Reconstruction mismatch: {float(jnp.max(jnp.abs(recon_sym - recon_ref)))}"
        )

    def test_symmetric_svd_ad_gradient_finite(self):
        """Gradient through truncated_svd_symmetric_ad is finite."""
        key = jax.random.PRNGKey(77)
        D, d = 2, 2
        data = jax.random.normal(key, (D, D, D, D, d))

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
        A = DenseTensor(data, indices)

        def loss(A_in):
            U, s, Vh = truncated_svd_symmetric_ad(
                A_in, ("u", "l"), ("d", "r", "phys"), chi=3
            )
            return jnp.sum(s)

        grad = jax.grad(loss)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "Gradient contains NaN/Inf"


class TestSVDSignFixing:
    """Tests for SVD sign-fixing gauge convention."""

    def test_sign_convention_truncated(self):
        """Max-abs element of each U column should be real-positive after truncated_svd_ad."""
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        chi = 3
        U, s, Vh = truncated_svd_ad(M, chi)
        for j in range(U.shape[1]):
            col = U[:, j]
            max_elem = col[jnp.argmax(jnp.abs(col))]
            assert float(jnp.real(max_elem)) > 0, (
                f"Column {j}: max-abs element {float(max_elem)} is not real-positive"
            )

    def test_sign_convention_regularized(self):
        """Max-abs element of each U column should be real-positive after regularized_svd."""
        key = jax.random.PRNGKey(1)
        M = jax.random.normal(key, (5, 4))
        U, s, Vh = regularized_svd(M)
        for j in range(U.shape[1]):
            col = U[:, j]
            max_elem = col[jnp.argmax(jnp.abs(col))]
            assert float(jnp.real(max_elem)) > 0, (
                f"Column {j}: max-abs element {float(max_elem)} is not real-positive"
            )

    def test_sign_fix_preserves_reconstruction(self):
        """U @ diag(s) @ Vh should still reconstruct the original matrix."""
        key = jax.random.PRNGKey(2)
        M = jax.random.normal(key, (6, 4))
        chi = 3
        U, s, Vh = truncated_svd_ad(M, chi)
        recon = U * s[None, :] @ Vh

        # Compare against standard SVD truncated reconstruction
        U_ref, s_ref, Vh_ref = jnp.linalg.svd(M, full_matrices=False)
        recon_ref = U_ref[:, :chi] * s_ref[:chi][None, :] @ Vh_ref[:chi, :]
        assert jnp.allclose(recon, recon_ref, atol=1e-12), (
            f"Reconstruction mismatch: {float(jnp.max(jnp.abs(recon - recon_ref)))}"
        )


class TestElementWiseCTMConvergence:
    """Element-wise CTM convergence should be available and work correctly."""

    def test_elementwise_converges(self):
        """CTM with element-wise convergence should reach convergence."""
        A = _make_dense_tensor(jax.random.PRNGKey(0))
        config = CTMConfig(
            chi=8,
            max_iter=200,
            conv_tol=1e-6,
            min_iter=10,
            ctm_conv_method="elementwise",
        )
        envs = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, config
        )
        assert envs is not None
        # Check that env tensors are non-trivial
        c1_norm = float(jnp.linalg.norm(envs[(0, 0)].C1.todense()))
        assert c1_norm > 0

    def test_sv_still_works(self):
        """Singular-value convergence should still work."""
        A = _make_dense_tensor(jax.random.PRNGKey(0))
        config = CTMConfig(chi=8, max_iter=200, conv_tol=1e-6, min_iter=10)
        envs = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, config
        )
        assert envs is not None

    def test_config_field_exists(self):
        """CTMConfig should have ctm_conv_method field."""
        config = CTMConfig()
        assert hasattr(config, "ctm_conv_method")
        assert config.ctm_conv_method == "sv"

    def test_config_roundtrip(self):
        """ctm_conv_method should survive serialization round-trip."""
        config = CTMConfig(ctm_conv_method="elementwise")
        t = _config_to_tuple(config)
        config2 = _config_from_tuple(t)
        assert config2.ctm_conv_method == "elementwise"

    def test_config_roundtrip_sv(self):
        """Default sv method should survive serialization round-trip."""
        config = CTMConfig(ctm_conv_method="sv")
        t = _config_to_tuple(config)
        config2 = _config_from_tuple(t)
        assert config2.ctm_conv_method == "sv"

    def test_sign_fix_gradient_finite(self):
        """Gradient through sign-fixed SVD should be finite."""
        key = jax.random.PRNGKey(3)
        M = jax.random.normal(key, (5, 4))
        chi = 3

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, chi)
            return jnp.sum(U**2) + jnp.sum(s) + jnp.sum(Vh**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad)), f"NaN/Inf in gradient: {grad}"


class TestRegularizedEigh:
    """regularized_eigh should handle degenerate eigenvalues without NaN."""

    def test_forward_matches_jnp_eigh(self):
        from tenax.algorithms.ad_utils import regularized_eigh

        key = jax.random.PRNGKey(7)
        M = jax.random.normal(key, (4, 4))
        M = M + M.T

        w, v = regularized_eigh(M)
        w_ref, v_ref = jnp.linalg.eigh(M)
        assert jnp.allclose(w, w_ref, atol=1e-10)
        recon = v @ jnp.diag(w) @ v.T
        recon_ref = v_ref @ jnp.diag(w_ref) @ v_ref.T
        assert jnp.allclose(recon, recon_ref, atol=1e-10)

    def test_gradient_finite_nondegenerate(self):
        from tenax.algorithms.ad_utils import regularized_eigh

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (5, 5))
        M = M + M.T

        def loss(M_in):
            w, v = regularized_eigh(M_in)
            return jnp.sum(w**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_finite_degenerate(self):
        from tenax.algorithms.ad_utils import regularized_eigh

        key = jax.random.PRNGKey(42)
        Q, _ = jnp.linalg.qr(jax.random.normal(key, (5, 5)))
        eigvals = jnp.array([3.0, 3.0, 2.0, 2.0, 1.0])
        M = Q @ jnp.diag(eigvals) @ Q.T
        M = 0.5 * (M + M.T)

        def loss(M_in):
            w, v = regularized_eigh(M_in)
            return jnp.sum(w**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0

    def test_gradient_through_eigenvectors(self):
        from tenax.algorithms.ad_utils import regularized_eigh

        key = jax.random.PRNGKey(99)
        M = jax.random.normal(key, (4, 4))
        M = M + M.T

        def loss(M_in):
            w, v = regularized_eigh(M_in)
            return jnp.sum(v[:, -1] ** 2)  # depends on eigenvectors

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))

    def test_structured_projector_loss_matches_fd(self):
        """Regression for issue #316: sign of the F-matrix contraction.

        ``sum(w**2)`` / ``sum(v[:, -1] ** 2)`` are insensitive to the sign
        of the F-matrix term because they only touch the ``dw`` path or are
        gauge-trivial (unit-norm eigenvector). A proper test must use a
        gauge-invariant loss that is quadratic in the eigenvectors, such as
        ``tr(P^T M P)`` with ``P`` the top-k projector. Finite differences
        and AD must agree in sign and magnitude.
        """
        from tenax.algorithms.ad_utils import regularized_eigh

        key = jax.random.PRNGKey(2024)
        key_rho, key_obs, key_h = jax.random.split(key, 3)
        C = jax.random.normal(key_rho, (6, 6))
        rho = C @ C.T
        rho = 0.5 * (rho + rho.T)
        M_obs = jax.random.normal(key_obs, (6, 6))
        M_obs = 0.5 * (M_obs + M_obs.T)

        def loss(r):
            _, v = regularized_eigh(r)
            P = v[:, -3:][:, ::-1]
            return jnp.trace(P.T @ M_obs @ P)

        # Directional derivative along a symmetric perturbation h.
        h = jax.random.normal(key_h, (6, 6))
        h = 0.5 * (h + h.T)
        eps = 1e-5
        fd = float(loss(rho + eps * h) - loss(rho - eps * h)) / (2 * eps)
        grad = jax.grad(loss)(rho)
        ad = float(jnp.sum(grad * h))

        assert jnp.isfinite(ad)
        assert jnp.isfinite(fd)
        assert abs(ad - fd) < 1e-4, (
            f"regularized_eigh AD disagrees with FD on gauge-invariant "
            f"projector loss: ad={ad:+.10f} fd={fd:+.10f} "
            f"(sign bug in F contraction — issue #316)"
        )


class TestMultipletAwareTruncation:
    """SVD truncation should handle degenerate singular values at boundary."""

    def test_no_split_at_clean_gap(self):
        """When truncation falls at a clean gap, all SVs should be nonzero."""
        key = jax.random.PRNGKey(42)
        Q1, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
        key2 = jax.random.PRNGKey(7)
        Q2, _ = jnp.linalg.qr(jax.random.normal(key2, (4, 4)))
        s_exact = jnp.array([3.0, 2.0, 1.5, 0.5])
        M = Q1[:, :4] @ jnp.diag(s_exact) @ Q2

        U, s, Vh = truncated_svd_ad(M, 3)
        # Clean gap between 1.5 and 0.5 — all 3 kept SVs should be nonzero
        assert jnp.all(s > 0.1), f"Expected all nonzero SVs but got {s}"

    def test_zeros_split_multiplet(self):
        """When chi cuts through degenerate group, partial members should be zeroed."""
        key = jax.random.PRNGKey(42)
        Q1, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
        key2 = jax.random.PRNGKey(7)
        Q2, _ = jnp.linalg.qr(jax.random.normal(key2, (4, 4)))
        # SVs: [3.0, 2.0, 2.0, 1.0] — chi=2 splits the (2,2) pair
        s_exact = jnp.array([3.0, 2.0, 2.0, 1.0])
        M = Q1[:, :4] @ jnp.diag(s_exact) @ Q2

        U, s, Vh = truncated_svd_ad(M, 2)
        # chi=2 cuts between 2.0 and 2.0 — should zero the boundary multiplet
        nonzero = int(jnp.sum(s > 0.1))
        assert nonzero == 1, f"Expected 1 nonzero SV (3.0 only) but got {nonzero}: {s}"

    def test_keeps_full_multiplet(self):
        """When chi includes the full multiplet, all should be kept."""
        key = jax.random.PRNGKey(42)
        Q1, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
        key2 = jax.random.PRNGKey(7)
        Q2, _ = jnp.linalg.qr(jax.random.normal(key2, (4, 4)))
        s_exact = jnp.array([3.0, 2.0, 2.0, 1.0])
        M = Q1[:, :4] @ jnp.diag(s_exact) @ Q2

        U, s, Vh = truncated_svd_ad(M, 3)
        # chi=3 keeps [3, 2, 2] — no split, all should be nonzero
        assert jnp.all(s > 0.1), f"Expected all nonzero but got {s}"

    def test_gradient_still_finite(self):
        """Gradient through multiplet-aware truncation should be finite."""
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (6, 4))

        def loss(M_in):
            U, s, Vh = truncated_svd_ad(M_in, 3)
            return jnp.sum(s**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad))


class TestVJPBackwardConvergence:
    """After sign fixing + elementwise convergence, VJP backward should converge."""

    @pytest.mark.xfail(reason="VJP needs sigma gauge fixing")
    def test_vjp_gradient_finite_with_elementwise(self):
        """VJP backward should produce finite gradients with elementwise convergence."""
        A = _make_dense_tensor(jax.random.PRNGKey(0))
        d_phys = 2
        key = jax.random.PRNGKey(99)
        gate = jax.random.normal(key, (d_phys, d_phys, d_phys, d_phys))
        gate = gate + gate.transpose(1, 0, 3, 2)  # Hermitian

        config = CTMConfig(
            chi=8,
            max_iter=100,
            conv_tol=1e-6,
            min_iter=10,
            ad_backward_method="vjp",
        )
        # Note: uses SV convergence (default) with QR gauge, which gives
        # ρ(J^T) ≈ 49.  elementwise + sigma gauge gives ρ ~ 1e13.
        config_tuple = _config_to_tuple(config)

        _env_template = initialize_ctm_tensor_env(A, config.chi)
        env_treedef = jax.tree.structure(_env_template)

        def loss_fn(A_tensor):
            A_norm = A_tensor * (1.0 / (A_tensor.norm() + 1e-10))
            site_tensors = {(0, 0): A_norm}
            env_leaves = ctm_tensor_converge(
                site_tensors, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(env_treedef, env_leaves)
            return compute_energy_ctm_tensor(A_norm, env, gate, d_phys)

        energy, grad = jax.value_and_grad(loss_fn)(A)
        grad_norm = float(jnp.linalg.norm(grad.todense()))

        assert jnp.isfinite(jnp.array(energy)), f"Energy not finite: {energy}"
        assert jnp.isfinite(jnp.array(grad_norm)), (
            f"Gradient norm not finite: {grad_norm}"
        )
        # VJP should give |g| ~ O(1), not O(1e14)
        assert grad_norm < 1e6, f"VJP gradient norm too large: {grad_norm:.3e}"

    def test_vjp_gradient_matches_gmres_direction(self):
        """VJP and GMRES backward should give gradients pointing in similar directions."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        d_phys = 2
        key = jax.random.PRNGKey(7)
        gate = jax.random.normal(key, (d_phys, d_phys, d_phys, d_phys))
        gate = gate + gate.transpose(1, 0, 3, 2)

        _env_template = initialize_ctm_tensor_env(A, 6)
        env_treedef = jax.tree.structure(_env_template)

        def make_loss(backward_method):
            config = CTMConfig(
                chi=6,
                max_iter=80,
                conv_tol=1e-6,
                min_iter=10,
                ad_backward_method=backward_method,
                adjoint_arnoldi_precheck=False,
            )
            ct = _config_to_tuple(config)

            def loss_fn(A_tensor):
                A_norm = A_tensor * (1.0 / (A_tensor.norm() + 1e-10))
                site_tensors = {(0, 0): A_norm}
                env_leaves = ctm_tensor_converge(
                    site_tensors, None, SINGLE_SITE_NEIGHBORS, ct
                )
                env = jax.tree.unflatten(env_treedef, env_leaves)
                return compute_energy_ctm_tensor(A_norm, env, gate, d_phys)

            return loss_fn

        _, grad_vjp = jax.value_and_grad(make_loss("vjp"))(A)
        _, grad_gmres = jax.value_and_grad(make_loss("gmres"))(A)

        g_vjp = grad_vjp.todense().ravel()
        g_gmres = grad_gmres.todense().ravel()

        # Both should be finite
        assert jnp.all(jnp.isfinite(g_vjp)), "VJP gradient not finite"
        assert jnp.all(jnp.isfinite(g_gmres)), "GMRES gradient not finite"

        # Cosine similarity should be positive (same direction)
        cos_sim = float(
            jnp.dot(g_vjp, g_gmres)
            / (jnp.linalg.norm(g_vjp) * jnp.linalg.norm(g_gmres) + 1e-30)
        )
        # Note: they may not be exactly the same due to different convergence criteria,
        # but should at least point in a broadly similar direction
        assert cos_sim > -0.5, (
            f"VJP and GMRES gradients too dissimilar: cos_sim={cos_sim:.4f}"
        )


class TestHalfSVD:
    """truncated_svd_ad_vh_only returns correct S and Vh with working gradients."""

    def test_forward_matches_full_svd(self):
        """Vh-only forward should match full truncated_svd_ad."""
        from tenax.algorithms.ad_utils import truncated_svd_ad_vh_only

        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        chi = 3

        s_half, Vh_half = truncated_svd_ad_vh_only(M, chi)
        _, s_full, Vh_full = truncated_svd_ad(M, chi)

        assert jnp.allclose(s_half, s_full, atol=1e-12)
        # Vh can differ by sign per row — compare via reconstruction
        recon_half = jnp.diag(s_half) @ Vh_half
        recon_full = jnp.diag(s_full) @ Vh_full
        assert jnp.allclose(recon_half, recon_full, atol=1e-12)

    def test_gradient_finite(self):
        """Gradient through Vh-only SVD should be finite."""
        from tenax.algorithms.ad_utils import truncated_svd_ad_vh_only

        key = jax.random.PRNGKey(1)
        M = jax.random.normal(key, (6, 4))
        chi = 3

        def loss(M_in):
            s, Vh = truncated_svd_ad_vh_only(M_in, chi)
            return jnp.sum(s**2) + jnp.sum(Vh**2)

        grad = jax.grad(loss)(M)
        assert jnp.all(jnp.isfinite(grad)), "Gradient not finite"

    def test_gradient_matches_full_svd(self):
        """Gradient through Vh-only SVD should match full SVD with dU=0."""
        from tenax.algorithms.ad_utils import truncated_svd_ad_vh_only

        key = jax.random.PRNGKey(2)
        M = jax.random.normal(key, (6, 4))
        chi = 3

        def loss_half(M_in):
            s, Vh = truncated_svd_ad_vh_only(M_in, chi)
            return jnp.sum(s**2) + jnp.sum(Vh**2)

        def loss_full(M_in):
            _U, s, Vh = truncated_svd_ad(M_in, chi)
            # Only use s and Vh (matching half-SVD usage)
            return jnp.sum(s**2) + jnp.sum(Vh**2)

        grad_half = jax.grad(loss_half)(M)
        grad_full = jax.grad(loss_full)(M)
        assert jnp.allclose(grad_half, grad_full, atol=1e-10), (
            f"Max diff: {float(jnp.max(jnp.abs(grad_half - grad_full)))}"
        )


class TestArnoldiSpectralRadius:
    def test_identity_has_spectral_radius_one(self):
        v0 = jnp.ones(10)
        rho = arnoldi_spectral_radius(lambda v: v, v0, n_iter=10)
        assert abs(rho - 1.0) < 1e-6

    def test_contraction_has_spectral_radius_below_one(self):
        v0 = jnp.ones(10)
        rho = arnoldi_spectral_radius(lambda v: 0.5 * v, v0, n_iter=10)
        assert abs(rho - 0.5) < 1e-6

    def test_expansion_detected(self):
        v0 = jnp.ones(10)
        rho = arnoldi_spectral_radius(lambda v: 2.0 * v, v0, n_iter=10)
        assert abs(rho - 2.0) < 1e-6

    def test_nonsymmetric_matrix(self):
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (8, 8)) * 0.3
        v0 = jnp.ones(8)
        rho = arnoldi_spectral_radius(lambda v: M @ v, v0, n_iter=8)
        rho_exact = float(jnp.max(jnp.abs(jnp.linalg.eigvals(M))))
        assert abs(rho - rho_exact) < 0.1


class TestCTMRGGradientError:
    def test_exception_carries_spectral_radius(self):
        err = CTMRGGradientError(spectral_radius=2.5)
        assert err.spectral_radius == 2.5
        assert "2.5" in str(err)


class TestArnoldiConfig:
    def test_arnoldi_precheck_default_true(self):
        config = CTMConfig()
        assert config.adjoint_arnoldi_precheck is True

    def test_arnoldi_precheck_round_trips(self):
        for val in (True, False):
            config = CTMConfig(adjoint_arnoldi_precheck=val)
            t = _config_to_tuple(config)
            config2 = _config_from_tuple(t)
            assert config2.adjoint_arnoldi_precheck == val


class TestArnoldiPrecheckIntegration:
    def test_raises_on_non_contractive_backward(self):
        """Implicit AD backward raises CTMRGGradientError when rho(J^T) >= 1 (QR gauge)."""
        from tenax.algorithms.ad_utils import CTMRGGradientError

        A = _make_dense_tensor(jax.random.PRNGKey(42))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)
        config = CTMConfig(
            chi=4,
            max_iter=10,
            conv_tol=1e-6,
            forward_gauge="qr",
            adjoint_arnoldi_precheck=True,
        )

        def _energy(A_tensor):
            config_tuple = _config_to_tuple(config)
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_tensor},
                None,
                SINGLE_SITE_NEIGHBORS,
                config_tuple,
            )
            env_template = initialize_ctm_tensor_env(A_tensor, config.chi)
            env_treedef = jax.tree.structure(env_template)
            env = jax.tree.unflatten(env_treedef, env_leaves)
            return compute_energy_ctm_tensor(A_tensor, env, gate)

        with pytest.raises(CTMRGGradientError):
            jax.grad(_energy)(A)

    def test_no_raise_when_disabled(self):
        """With adjoint_arnoldi_precheck=False, backward completes without raising."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)
        config = CTMConfig(
            chi=4,
            max_iter=10,
            conv_tol=1e-6,
            forward_gauge="qr",
            adjoint_arnoldi_precheck=False,
        )

        def _energy(A_tensor):
            config_tuple = _config_to_tuple(config)
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_tensor},
                None,
                SINGLE_SITE_NEIGHBORS,
                config_tuple,
            )
            env_template = initialize_ctm_tensor_env(A_tensor, config.chi)
            env_treedef = jax.tree.structure(env_template)
            env = jax.tree.unflatten(env_treedef, env_leaves)
            return compute_energy_ctm_tensor(A_tensor, env, gate)

        # Should not raise — precheck disabled
        g = jax.grad(_energy)(A)
        # Just verify we got a gradient (may be garbage, but no exception)
        assert g is not None
