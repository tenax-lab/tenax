"""Tests for the dense C4v paper-mode forward path (Appendix C-F roadmap)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_c4v_paper_ad import ctm_tensor_c4v_paper_fixed_point
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


def test_optimize_gs_ad_paper_mode_nonzero_steps_not_implemented(heisenberg_gate):
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
    with pytest.raises(NotImplementedError, match="gs_num_steps>0"):
        optimize_gs_ad(heisenberg_gate, A0, config)
