"""Tests for the dense C4v paper-mode forward path (Appendix C-F roadmap)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_c4v import _c4v_to_full_env
from tenax.algorithms._ctm_tensor_c4v_paper_ad import (
    _appendix_c_truncated_eigh_backward,
    _solve_linear_adjoint_paper,
    ctm_tensor_c4v_paper_backward,
    ctm_tensor_c4v_paper_converge_reduced,
    ctm_tensor_c4v_paper_fixed_point,
    truncated_eigh_appendix_c,
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


def test_c4v_paper_fixed_point_returns_meta():
    A = _wrap_as_dense_tensor(jax.random.normal(jax.random.PRNGKey(0), (2, 2, 2, 2, 2)))
    cfg = CTMConfig(
        chi=4,
        max_iter=8,
        min_iter=2,
        conv_tol=1e-8,
        projector_method="eigh",
        paper_ctm_ad="c4v_appendix_cf",
    )
    env, meta = ctm_tensor_c4v_paper_fixed_point(A, cfg)

    assert np.isfinite(env.C1.todense()).all()
    assert np.isfinite(env.T1.todense()).all()
    assert set(meta) >= {"iters", "residual", "converged"}
    assert 1 <= meta["iters"] <= cfg.max_iter
    assert isinstance(meta["converged"], bool)
    assert np.isfinite(meta["residual"]) or np.isinf(meta["residual"])


def test_ctmconfig_paper_mode_defaults():
    cfg = CTMConfig()
    assert cfg.paper_ctm_ad is None
    assert cfg.paper_krylov_solver == "bicgstab"
    assert cfg.paper_krylov_maxiter == 50
    assert cfg.paper_krylov_tol == 1e-8
    assert cfg.paper_degen_tol == 1e-10
    assert cfg.paper_diag_shift == 1e-12


def test_invalid_paper_ctm_ad_raises():
    with pytest.raises(ValueError, match="paper_ctm_ad"):
        CTMConfig(paper_ctm_ad="bad_mode")


def test_invalid_paper_krylov_solver_raises():
    with pytest.raises(ValueError, match="paper_krylov_solver"):
        CTMConfig(paper_krylov_solver="bad_solver")


def test_paper_mode_dispatch_is_strictly_gated(heisenberg_gate, monkeypatch):
    """Paper path dispatch should trigger only for the strict mode gate."""
    import tenax.algorithms.ipeps_optimize as _opt

    class _PaperCalled(Exception):
        pass

    class _DefaultCalled(Exception):
        pass

    def _paper_spy(*_args, **_kwargs):
        raise _PaperCalled

    def _default_spy(*_args, **_kwargs):
        raise _DefaultCalled

    monkeypatch.setattr(_opt, "_optimize_gs_ad_tensor_paper_c4v", _paper_spy)
    monkeypatch.setattr(_opt, "_optimize_gs_ad_tensor", _default_spy)

    A0 = jax.random.normal(jax.random.PRNGKey(123), (2, 2, 2, 2, 2))

    cfg_paper = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=3, paper_ctm_ad="c4v_appendix_cf"),
        gs_num_steps=0,
        gs_explicit_ad=False,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
    )
    with pytest.raises(_PaperCalled):
        optimize_gs_ad(heisenberg_gate, A0, cfg_paper)

    cfg_not_paper = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=3, paper_ctm_ad="c4v_appendix_cf"),
        gs_num_steps=0,
        gs_explicit_ad=False,
        gs_c4v=False,
        unit_cell="1x1",
        su_init=False,
    )
    with pytest.raises(_DefaultCalled):
        optimize_gs_ad(heisenberg_gate, A0, cfg_not_paper)


def test_optimize_gs_ad_paper_mode_zero_steps_runs(heisenberg_gate):
    A0 = jax.random.normal(jax.random.PRNGKey(1), (2, 2, 2, 2, 2))
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=10, min_iter=2, paper_ctm_ad="c4v_appendix_cf"),
        gs_num_steps=0,
        gs_explicit_ad=False,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
    )
    A_opt, env, E = optimize_gs_ad(heisenberg_gate, A0, config)

    assert isinstance(A_opt, DenseTensor)
    assert np.isfinite(env.C1.todense()).all()
    assert np.isfinite(E)


def test_optimize_gs_ad_paper_mode_nonzero_steps_runs(heisenberg_gate):
    A0 = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 2, 2, 2))
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=10, min_iter=2, paper_ctm_ad="c4v_appendix_cf"),
        gs_num_steps=1,
        gs_explicit_ad=False,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
    )
    A_opt, env, E = optimize_gs_ad(heisenberg_gate, A0, config)
    assert isinstance(A_opt, DenseTensor)
    assert np.isfinite(env.C1.todense()).all()
    assert np.isfinite(E)


def test_truncated_eigh_appendix_c_gradient_finite_degenerate():
    key = jax.random.PRNGKey(77)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (6, 6)))
    w = jnp.array([3.0, 3.0, 2.0, 2.0, 1.0, 0.5])
    M = Q @ jnp.diag(w) @ Q.T
    M = 0.5 * (M + M.T)

    def loss(M_in):
        vals, vecs = truncated_eigh_appendix_c(M_in, 3)
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


def test_truncated_eigh_appendix_c_finite_difference_agreement():
    key = jax.random.PRNGKey(1234)
    M0 = jax.random.normal(key, (5, 5))
    M0 = 0.5 * (M0 + M0.T)
    key_w = jax.random.PRNGKey(1235)
    W = jax.random.normal(key_w, (5, 5))
    W = 0.5 * (W + W.T)

    def loss(M_in):
        vals, vecs = truncated_eigh_appendix_c(M_in, 3)
        P = vecs @ vecs.T
        return jnp.sum(vals**2) + 0.2 * jnp.trace(P @ W)

    g_ad = jax.grad(loss)(M0)
    g_fd = _finite_difference_grad(loss, M0)
    max_diff = float(jnp.max(jnp.abs(g_ad - g_fd)))
    assert max_diff < 1.5e-1, (
        f"Appendix-C truncated eigh FD mismatch too large: {max_diff}"
    )


def test_appendix_c_truncation_correction_improves_fd_error():
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
        vals, vecs = truncated_eigh_appendix_c(M_in, k)
        P = vecs @ vecs.T
        return jnp.sum(vals**2) + 0.3 * jnp.trace(P @ W)

    g_fd = _finite_difference_grad(loss, M)
    vals, vecs = truncated_eigh_appendix_c(M, k)
    dvals = 2.0 * vals
    dvecs = 0.6 * (W @ vecs)
    g_ref = _appendix_c_truncated_eigh_backward(w, v, dvals, dvecs, k=k)

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
        f"Expected corrected Appendix-C backward to beat/equal naive: "
        f"err_ref={err_ref}, err_naive={err_naive}"
    )


def test_paper_mode_backward_gradient_is_finite_and_deterministic(heisenberg_gate):
    cfg = CTMConfig(
        chi=4,
        max_iter=8,
        min_iter=2,
        conv_tol=1e-8,
        projector_method="eigh",
        paper_ctm_ad="c4v_appendix_cf",
    )
    A0 = jax.random.normal(jax.random.PRNGKey(321), (2, 2, 2, 2, 2))

    def loss(A_data):
        A = _wrap_as_dense_tensor(A_data)
        A = A * (1.0 / (A.norm() + 1e-10))
        C, T = ctm_tensor_c4v_paper_converge_reduced(A, cfg)
        env = _c4v_to_full_env(C, T)
        return compute_energy_ctm_tensor(A, env, heisenberg_gate, 2)

    g1 = jax.grad(loss)(A0)
    g2 = jax.grad(loss)(A0)
    assert jnp.all(jnp.isfinite(g1))
    assert jnp.all(jnp.isfinite(g2))
    assert jnp.allclose(g1, g2, rtol=1e-10, atol=1e-10)


def test_paper_mode_krylov_backward_residual_below_threshold():
    cfg = CTMConfig(
        chi=4,
        max_iter=8,
        min_iter=2,
        conv_tol=1e-8,
        projector_method="eigh",
        paper_ctm_ad="c4v_appendix_cf",
        paper_krylov_solver="bicgstab",
        paper_krylov_maxiter=30,
        paper_krylov_tol=1e-8,
    )
    A = _wrap_as_dense_tensor(
        jax.random.normal(jax.random.PRNGKey(404), (2, 2, 2, 2, 2))
    )
    A = A * (1.0 / (A.norm() + 1e-10))
    C, T = ctm_tensor_c4v_paper_converge_reduced(A, cfg)
    g_c = C * 0.1
    g_t = T * 0.1
    dA, meta = ctm_tensor_c4v_paper_backward(A, C, T, g_c, g_t, cfg)
    assert isinstance(dA, DenseTensor)
    assert jnp.all(jnp.isfinite(dA.todense()))
    assert meta["residual"] <= max(cfg.paper_krylov_tol * 10.0, 1e-12)


def test_paper_mode_krylov_fallback_to_gmres(monkeypatch):
    import tenax.algorithms._ctm_tensor_c4v_paper_ad as paper

    cfg = CTMConfig(
        paper_ctm_ad="c4v_appendix_cf",
        paper_krylov_solver="bicgstab",
        paper_krylov_maxiter=20,
        paper_krylov_tol=1e-8,
    )

    def _apply(v):
        return v

    rhs = (jnp.array([1.0, -2.0, 3.0]),)

    def _bad_bicg(*_args, **_kwargs):
        bad = (jnp.array([100.0, 100.0, 100.0]),)
        return bad, None

    monkeypatch.setattr(paper, "_krylov_bicgstab", _bad_bicg)
    _lam, meta = _solve_linear_adjoint_paper(_apply, rhs, cfg)
    assert meta["used_fallback"] is True
    assert meta["solver_used"] == "gmres"
    assert meta["residual"] <= max(cfg.paper_krylov_tol * 10.0, 1e-12)
