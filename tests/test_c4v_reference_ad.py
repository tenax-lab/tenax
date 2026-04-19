"""Tests for the dense C4v reference-mode path."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_c4v import _c4v_to_full_env
from tenax.algorithms._ctm_tensor_c4v_reference_ad import (
    _solve_linear_adjoint,
    _truncated_eigh_lorentzian_backward,
    ctm_tensor_c4v_reference_backward,
    ctm_tensor_c4v_reference_converge_reduced,
    ctm_tensor_c4v_reference_fixed_point,
    truncated_eigh_regularized,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor, optimize_gs_ad
from tenax.core.tensor import DenseTensor


@pytest.fixture
def heisenberg_gate():
    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


def test_c4v_reference_fixed_point_returns_meta():
    A = _wrap_as_dense_tensor(jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2)))
    cfg = CTMConfig(
        chi=4,
        max_iter=8,
        min_iter=2,
        conv_tol=1e-8,
        projector_method="eigh",
        ctm_ad_mode="c4v_reference",
    )
    env, meta = ctm_tensor_c4v_reference_fixed_point(A, cfg)

    assert np.isfinite(env.C1.todense()).all()
    assert np.isfinite(env.T1.todense()).all()
    assert set(meta) >= {"iters", "residual", "converged"}
    assert 1 <= meta["iters"] <= cfg.max_iter
    assert isinstance(meta["converged"], bool)
    assert np.isfinite(meta["residual"]) or np.isinf(meta["residual"])


def test_ctmconfig_reference_mode_defaults():
    cfg = CTMConfig()
    assert cfg.ctm_ad_mode is None
    assert cfg.adjoint_solver == "bicgstab"
    assert cfg.adjoint_maxiter == 50
    assert cfg.adjoint_tol == 1e-8
    assert cfg.adjoint_degen_tol == 1e-10
    assert cfg.adjoint_diag_shift == 1e-12
    # Non-zero Tikhonov floor — keeps the adjoint Krylov solve from stalling
    # when (I - J^T) is near-singular (see adjoint_tikhonov docstring).
    assert cfg.adjoint_tikhonov == 1e-6
    assert cfg.adjoint_tikhonov > 0.0


def test_invalid_ctm_ad_mode_raises():
    with pytest.raises(ValueError, match="ctm_ad_mode"):
        CTMConfig(ctm_ad_mode="bad_mode")


def test_invalid_adjoint_solver_raises():
    with pytest.raises(ValueError, match="adjoint_solver"):
        CTMConfig(adjoint_solver="bad_solver")


def test_reference_mode_dispatch_is_strictly_gated(heisenberg_gate, monkeypatch):
    """Reference path dispatch should trigger only for the strict mode gate."""
    import tenax.algorithms.ipeps_optimize as _opt

    class _ReferenceCalled(Exception):
        pass

    class _DefaultCalled(Exception):
        pass

    def _reference_spy(*_args, **_kwargs):
        raise _ReferenceCalled

    def _default_spy(*_args, **_kwargs):
        raise _DefaultCalled

    monkeypatch.setattr(_opt, "_optimize_gs_ad_tensor_reference_c4v", _reference_spy)
    monkeypatch.setattr(_opt, "_optimize_gs_ad_tensor", _default_spy)

    A0 = jax.random.normal(jax.random.PRNGKey(123), (2, 2, 2, 2, 2))

    cfg_reference = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=3, ctm_ad_mode="c4v_reference"),
        gs_num_steps=0,
        gs_implicit_ad=True,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
    )
    with pytest.raises(_ReferenceCalled):
        optimize_gs_ad(heisenberg_gate, A0, cfg_reference)

    cfg_not_reference = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=3, ctm_ad_mode="c4v_reference"),
        gs_num_steps=0,
        gs_implicit_ad=True,
        gs_c4v=False,
        unit_cell="1x1",
        su_init=False,
    )
    with pytest.raises(_DefaultCalled):
        optimize_gs_ad(heisenberg_gate, A0, cfg_not_reference)


def test_optimize_gs_ad_reference_mode_zero_steps_runs(heisenberg_gate):
    A0 = jax.random.normal(jax.random.PRNGKey(1), (2, 2, 2, 2, 2))
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=10, min_iter=2, ctm_ad_mode="c4v_reference"),
        gs_num_steps=0,
        gs_implicit_ad=True,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
    )
    A_opt, env, E = optimize_gs_ad(heisenberg_gate, A0, config)

    assert isinstance(A_opt, DenseTensor)
    assert np.isfinite(env.C1.todense()).all()
    assert np.isfinite(E)


def test_optimize_gs_ad_reference_mode_nonzero_steps_runs(heisenberg_gate):
    A0 = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 2, 2, 2))
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=10, min_iter=2, ctm_ad_mode="c4v_reference"),
        gs_num_steps=1,
        gs_implicit_ad=True,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
    )
    A_opt, env, E = optimize_gs_ad(heisenberg_gate, A0, config)
    assert isinstance(A_opt, DenseTensor)
    assert np.isfinite(env.C1.todense()).all()
    assert np.isfinite(E)


def test_truncated_eigh_regularized_gradient_finite_degenerate():
    key = jax.random.PRNGKey(77)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
    w = jnp.array([3.0, 3.0, 2.0, 2.0, 1.0, 0.5])
    M = Q @ jnp.diag(w) @ Q.T
    M = 0.5 * (M + M.T)

    def loss(M_in):
        vals, vecs = truncated_eigh_regularized(M_in, 3)
        return jnp.sum(vals**2) + jnp.sum(vecs[:, 0] ** 2)

    grad = jax.grad(loss)(M)
    assert jnp.all(jnp.isfinite(grad))


def _finite_difference_grad(loss_fn, M, eps=1e-5):
    M_np = np.array(M)
    grad = np.zeros_like(M_np)
    for i, j in itertools.product(range(M_np.shape[0]), range(M_np.shape[1])):
        d = np.zeros_like(M_np)
        d[i, j] = eps
        plus = M_np + d
        minus = M_np - d
        grad[i, j] = (
            float(loss_fn(jnp.array(plus))) - float(loss_fn(jnp.array(minus)))
        ) / (2 * eps)
    return grad


def test_truncated_eigh_regularized_finite_difference_agreement():
    key = jax.random.PRNGKey(1234)
    M0 = jax.random.normal(key, (5, 5))
    M0 = 0.5 * (M0 + M0.T)
    key_w = jax.random.PRNGKey(1235)
    W = jax.random.normal(key_w, (5, 5))
    W = 0.5 * (W + W.T)

    def loss(M_in):
        vals, vecs = truncated_eigh_regularized(M_in, 3)
        P = vecs @ vecs.T
        return jnp.sum(vals**2) + 0.2 * jnp.trace(P @ W)

    g_ad = jax.grad(loss)(M0)
    g_fd = _finite_difference_grad(loss, M0)
    max_diff = float(jnp.max(jnp.abs(g_ad - g_fd)))
    assert max_diff < 1.5e-1, (
        f"Regularized truncated eigh FD mismatch too large: {max_diff}"
    )


def test_regularized_truncation_correction_improves_fd_error():
    key = jax.random.PRNGKey(999)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
    # Near-degenerate boundary between kept/discarded sectors for k=3.
    w_spec = jnp.array([4.0, 3.0, 2.0000, 1.9999, 1.0, 0.5])
    M = Q @ jnp.diag(w_spec) @ Q.T
    M = 0.5 * (M + M.T)
    key_w = jax.random.PRNGKey(1000)
    W = jax.random.normal(key_w, (6, 6))
    W = 0.5 * (W + W.T)
    w, v = jnp.linalg.eigh(M)
    k = 3

    def loss(M_in):
        vals, vecs = truncated_eigh_regularized(M_in, k)
        P = vecs @ vecs.T
        return jnp.sum(vals**2) + 0.3 * jnp.trace(P @ W)

    g_fd = _finite_difference_grad(loss, M)
    vals, vecs = truncated_eigh_regularized(M, k)
    dvals = 2.0 * vals
    dvecs = 0.6 * (W @ vecs)
    g_ref = _truncated_eigh_lorentzian_backward(w, v, dvals, dvecs, k=k)

    # Naive backward: keep-kept only (omit kept-discarded correction)
    V_k = v[:, :k]
    w_k = w[:k]
    A = jnp.conj(V_k).T @ dvecs
    A_anti = 0.5 * (A - jnp.conj(A).T)
    gap = w_k[:, None] - w_k[None, :]
    F = gap / (gap**2 + 1e-12**2)
    F = F - jnp.diag(jnp.diag(F))
    g_naive = (
        V_k @ jnp.diag(dvals) @ jnp.conj(V_k).T + V_k @ (F * A_anti) @ jnp.conj(V_k).T
    )
    g_naive = 0.5 * (g_naive + jnp.conj(g_naive).T)

    err_ref = float(jnp.linalg.norm(g_ref - g_fd))
    err_naive = float(jnp.linalg.norm(g_naive - g_fd))
    assert err_ref <= err_naive + 1e-8, (
        f"Expected corrected regularized backward to beat/equal naive: "
        f"err_ref={err_ref}, err_naive={err_naive}"
    )


def test_reference_mode_backward_gradient_is_finite_and_deterministic(heisenberg_gate):
    cfg = CTMConfig(
        chi=4,
        max_iter=8,
        min_iter=2,
        conv_tol=1e-8,
        projector_method="eigh",
        ctm_ad_mode="c4v_reference",
    )
    A0 = jax.random.normal(jax.random.PRNGKey(321), (2, 2, 2, 2, 2))

    def loss(A_data):
        A = _wrap_as_dense_tensor(A_data)
        A = A * (1.0 / (A.norm() + 1e-10))
        C, T = ctm_tensor_c4v_reference_converge_reduced(A, cfg)
        env = _c4v_to_full_env(C, T)
        return compute_energy_ctm_tensor(A, env, heisenberg_gate, 2)

    g1 = jax.grad(loss)(A0)
    g2 = jax.grad(loss)(A0)
    assert jnp.all(jnp.isfinite(g1))
    assert jnp.all(jnp.isfinite(g2))
    assert jnp.allclose(g1, g2, rtol=1e-10, atol=1e-10)


def test_reference_mode_krylov_backward_residual_below_threshold():
    cfg = CTMConfig(
        chi=4,
        max_iter=8,
        min_iter=2,
        conv_tol=1e-8,
        projector_method="eigh",
        ctm_ad_mode="c4v_reference",
        adjoint_solver="bicgstab",
        adjoint_maxiter=30,
        adjoint_tol=1e-8,
    )
    A = _wrap_as_dense_tensor(
        jax.random.normal(jax.random.PRNGKey(404), (2, 2, 2, 2, 2))
    )
    A = A * (1.0 / (A.norm() + 1e-10))
    C, T = ctm_tensor_c4v_reference_converge_reduced(A, cfg)
    g_c = C * 0.1
    g_t = T * 0.1
    dA, meta = ctm_tensor_c4v_reference_backward(A, C, T, g_c, g_t, cfg)
    assert isinstance(dA, DenseTensor)
    assert jnp.all(jnp.isfinite(dA.todense()))
    assert meta["residual"] <= max(cfg.adjoint_tol * 10.0, 1e-12)


def test_reference_mode_krylov_fallback_to_gmres(monkeypatch):
    import tenax.algorithms._ctm_tensor_c4v_reference_ad as reference_mod

    cfg = CTMConfig(
        ctm_ad_mode="c4v_reference",
        adjoint_solver="bicgstab",
        adjoint_maxiter=20,
        adjoint_tol=1e-8,
    )

    def _apply(v):
        return v

    rhs = (jnp.array([1.0, -2.0, 3.0]),)

    def _bad_bicg(*_args, **_kwargs):
        bad = (jnp.array([100.0, 100.0, 100.0]),)
        return bad, None

    monkeypatch.setattr(reference_mod, "_krylov_bicgstab", _bad_bicg)
    _lam, meta = _solve_linear_adjoint(_apply, rhs, cfg)
    assert meta["used_fallback"] is True
    assert meta["solver_used"] == "gmres"
    assert meta["residual"] <= max(cfg.adjoint_tol * 10.0, 1e-12)


def test_complex_adjoint_gradient_is_hermitian():
    key_r = jax.random.PRNGKey(2001)
    key_i = jax.random.PRNGKey(2002)
    X = jax.random.normal(key_r, (5, 5)) + 1j * jax.random.normal(key_i, (5, 5))
    M = 0.5 * (X + jnp.conj(X).T)
    key_wr = jax.random.PRNGKey(2003)
    key_wi = jax.random.PRNGKey(2004)
    W = jax.random.normal(key_wr, (5, 5)) + 1j * jax.random.normal(key_wi, (5, 5))
    W = 0.5 * (W + jnp.conj(W).T)

    def loss(M_in):
        vals, vecs = truncated_eigh_regularized(M_in, 3)
        P = vecs @ jnp.conj(vecs).T
        return jnp.real(jnp.sum(vals**2) + 0.15 * jnp.trace(P @ W))

    grad = jax.grad(loss)(M)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.max(jnp.abs(grad - jnp.conj(grad).T)) < 1e-10


def test_complex_adjoint_backward_output_is_hermitian():
    key = jax.random.PRNGKey(2010)
    X = jax.random.normal(key, (4, 4)) + 1j * jax.random.normal(
        jax.random.PRNGKey(2011), (4, 4)
    )
    M = 0.5 * (X + jnp.conj(X).T)
    w, v = jnp.linalg.eigh(M)
    k = 2
    dw = jax.random.normal(jax.random.PRNGKey(2012), (k,))
    dV = jax.random.normal(jax.random.PRNGKey(2013), (4, k)) + 1j * jax.random.normal(
        jax.random.PRNGKey(2014), (4, k)
    )
    dM = _truncated_eigh_lorentzian_backward(w, v, dw, dV, k=k)
    assert jnp.all(jnp.isfinite(dM))
    assert jnp.max(jnp.abs(dM - jnp.conj(dM).T)) < 1e-10


# ---------------------------------------------------------------------------
# Tikhonov damping on the linear adjoint system
# ---------------------------------------------------------------------------


def _reference_backward_on_random_A(cfg: CTMConfig, seed: int = 512):
    """Run one reference-mode adjoint backward on a normalized random A."""
    A = _wrap_as_dense_tensor(
        jax.random.normal(jax.random.PRNGKey(seed), (2, 2, 2, 2, 2))
    )
    A = A * (1.0 / (A.norm() + 1e-10))
    C, T = ctm_tensor_c4v_reference_converge_reduced(A, cfg)
    g_c = C * 0.1
    g_t = T * 0.1
    dA, meta = ctm_tensor_c4v_reference_backward(A, C, T, g_c, g_t, cfg)
    return dA, meta


def _make_reference_cfg(**overrides) -> CTMConfig:
    base = dict(
        chi=4,
        max_iter=8,
        min_iter=2,
        conv_tol=1e-8,
        projector_method="eigh",
        ctm_ad_mode="c4v_reference",
        adjoint_solver="bicgstab",
        adjoint_maxiter=30,
        adjoint_tol=1e-8,
    )
    base.update(overrides)
    return CTMConfig(**base)


def test_adjoint_tikhonov_is_surfaced_in_meta():
    """Backward meta should report the tau value actually used."""
    cfg = _make_reference_cfg(adjoint_tikhonov=2.5e-4)
    _dA, meta = _reference_backward_on_random_A(cfg)
    assert meta["tikhonov"] == pytest.approx(2.5e-4)


def test_adjoint_tikhonov_zero_meta_reflects_zero():
    cfg = _make_reference_cfg(adjoint_tikhonov=0.0)
    _dA, meta = _reference_backward_on_random_A(cfg)
    assert meta["tikhonov"] == 0.0


def test_adjoint_tikhonov_changes_gradient():
    """Non-zero tau must visibly change ``dA`` compared to tau=0.

    This is the wiring check: if the Tikhonov branch were ever
    accidentally dropped, the gradient would be identical across any
    choice of tau and this test would fail.
    """
    cfg_zero = _make_reference_cfg(adjoint_tikhonov=0.0)
    cfg_damped = _make_reference_cfg(adjoint_tikhonov=1e-2)

    dA_zero, _ = _reference_backward_on_random_A(cfg_zero)
    dA_damped, _ = _reference_backward_on_random_A(cfg_damped)

    assert jnp.all(jnp.isfinite(dA_zero.todense()))
    assert jnp.all(jnp.isfinite(dA_damped.todense()))

    diff_norm = float(jnp.linalg.norm(dA_zero.todense() - dA_damped.todense()))
    zero_norm = float(jnp.linalg.norm(dA_zero.todense()))
    # 1e-2 is a large tau — the gradient should differ well above
    # floating-point noise but not be degenerate (still similar order).
    assert diff_norm > 1e-6 * max(zero_norm, 1.0), (
        f"Gradients look identical across tau=0 and tau=1e-2 "
        f"(diff_norm={diff_norm:.3e}); the Tikhonov branch may have "
        f"regressed."
    )


def test_adjoint_tikhonov_damped_system_converges_to_biased_solution():
    """Hand-crafted near-singular ``(I - J^T)``: verify Tikhonov math.

    Bypasses the full CTM backward and calls ``_solve_linear_adjoint``
    directly with a diagonal matvec whose spectrum has one ~zero
    eigenvalue. Wrapping the matvec with ``+ tau * I`` makes the system
    well-conditioned; the resulting solution is the Tikhonov-biased
    ``lam_i = rhs_i / (spec_i + tau)``.

    This documents the mathematical principle the integration tests
    above depend on.
    """
    spec = jnp.array([1e-10, 0.5, 1.0, 1.5, 2.0])
    rhs = (jnp.ones(5),)

    def apply_raw(v):
        return (v[0] * spec,)

    tau = 1e-3

    def apply_damped(v):
        base = apply_raw(v)
        return (base[0] + tau * v[0],)

    cfg_damped = _make_reference_cfg(
        adjoint_solver="bicgstab",
        adjoint_maxiter=20,
        adjoint_tol=1e-10,
        adjoint_tikhonov=tau,
    )
    lam, meta = _solve_linear_adjoint(apply_damped, rhs, cfg_damped)
    assert meta["converged"] is True
    # Tikhonov-biased solution: lam_i = rhs_i / (spec_i + tau).
    expected = rhs[0] / (spec + tau)
    assert jnp.allclose(lam[0], expected, atol=1e-6)
    # Sanity check the shift: for the near-null mode, damped solution is
    # ~1/tau, not ~1/spec, because tau dominates.
    assert jnp.abs(lam[0][0] - 1.0 / tau) < 1e-3


# ---------------------------------------------------------------------------
# SU-init plateau regression
# ---------------------------------------------------------------------------


@pytest.mark.algorithm
def test_random_init_escapes_su_plateau_with_tikhonov(heisenberg_gate):
    """Reference-mode C4v with random init + Tikhonov must break past E=-0.5.

    Regression for the bug that motivated ``adjoint_tikhonov``. The
    sublattice-rotated Heisenberg Hamiltonian has the |↑↑⟩ product
    state as a gradient-zero saddle at exactly E = -0.5 per site.
    Simple-update init converges to that saddle, so L-BFGS cannot
    escape; random init + small Tikhonov damping must.
    """
    from tenax import sublattice_rotate_gate

    gate = sublattice_rotate_gate(heisenberg_gate)
    # Fixed seed so the first-step adjoint system is deterministic.
    A_init = jax.random.normal(jax.random.PRNGKey(2024), (2, 2, 2, 2, 2))
    cfg = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=8,
            max_iter=100,
            min_iter=2,
            ctm_ad_mode="c4v_reference",
            adjoint_solver="bicgstab",
            adjoint_maxiter=200,
            # Loose tolerances — this test is about *escape*, not about
            # gradient accuracy. A larger Tikhonov shift keeps the adjoint
            # solve stable across the first few steps where the outer
            # optimizer is still far from any fixed point.
            adjoint_tol=1e-3,
            adjoint_tikhonov=1e-2,
        ),
        gs_optimizer="lbfgs",
        gs_num_steps=3,
        gs_implicit_ad=True,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
    )
    _A_opt, _env, E_final = optimize_gs_ad(gate, A_init, cfg)
    assert np.isfinite(E_final)
    # Classical |↑↑⟩ gives E=-0.5 exactly; any genuine progress off the
    # plateau must land strictly below.
    assert float(E_final) < -0.5 - 1e-3, (
        f"E_final={float(E_final):.6f} is not below the product-state "
        f"plateau at E=-0.5 — random init may be failing to break the "
        f"saddle, or the Tikhonov-damped adjoint may have regressed."
    )
