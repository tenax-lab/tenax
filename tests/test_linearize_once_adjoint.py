"""Linearise once, jit only the transpose, for the implicit-AD adjoint.

The default backward (``adjoint_chain_rule="joint"``) builds the gauge-fixed
sweep's pullback once (``_jit_linearize_joint``) and hands it to the Neumann
loop (``_jit_neumann_bwd``) and the joint GMRES solve
(``_jit_gmres_joint_solve``), which then compile only the transpose.  The
oracle is ``"separate"``, whose programs are unchanged.  Spec:
docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md
"""

from __future__ import annotations

import collections
import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _block_sparse_jit
from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)

# ----------------------------------------------------------- _neumann_loop


def _contractive(rho, n=6, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    M *= rho / np.max(np.abs(np.linalg.eigvals(M)))
    b = rng.standard_normal(n)
    return M, b


def _loop(M, b, init=None, use_init=False, tol=1e-13, maxiter=500):
    J = jnp.asarray(M)
    init = jnp.zeros(len(b)) if init is None else jnp.asarray(init)
    return cea._neumann_loop(
        lambda v: (J.T @ v[0],),
        (jnp.asarray(b),),
        (init,),
        jnp.array(use_init),
        tol=tol,
        maxiter=maxiter,
    )


def test_the_neumann_loop_solves_a_contractive_system():
    M, b = _contractive(0.5)
    diverged, converged, n_iter, lam, abs_resid, b_norm = _loop(M, b)
    assert bool(converged) and not bool(diverged)
    assert int(n_iter) > 5  # regime: it iterated
    np.testing.assert_allclose(
        np.asarray(lam[0]), np.linalg.solve(np.eye(6) - M.T, b), rtol=1e-10
    )
    np.testing.assert_allclose(float(b_norm), np.linalg.norm(b), rtol=1e-14)
    assert float(abs_resid) < 1e-12


def test_the_neumann_loop_flags_divergence():
    M, b = _contractive(1.5)
    diverged, converged, n_iter, *_ = _loop(M, b)
    assert bool(diverged) and not bool(converged)
    assert int(n_iter) <= 10


def test_the_warm_start_is_used_only_when_asked():
    M, b = _contractive(0.5)
    exact = np.linalg.solve(np.eye(6) - M.T, b)
    n_warm = int(_loop(M, b, init=exact, use_init=True)[2])
    n_cold = int(_loop(M, b, init=exact, use_init=False)[2])
    assert n_warm == 1
    assert n_cold > 5


# --------------------------------------------------------- _fusion_variant

needs_no_fusion = pytest.mark.skipif(
    jax.default_backend() != "cpu"
    or not _block_sparse_jit.JIT_TAKES_COMPILER_OPTIONS
    or not _block_sparse_jit.no_fusion_compiles(),
    reason="the no-fusion variant is CPU-only and needs compiler_options",
)


@pytest.fixture
def any_size(monkeypatch):
    """Drop the size threshold, so a small U(1) state dispatches as a large one."""
    monkeypatch.setattr(_block_sparse_jit, "NO_FUSION_MIN_SITE_BLOCKS", 1)


@needs_no_fusion
@pytest.mark.parametrize("by_bond_dim", [True, False])
def test_a_large_symmetric_site_picks_the_no_fusion_variant(any_size, by_bond_dim):
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    prog = _block_sparse_jit.block_sparse_jit(lambda x: x)
    got = cea._fusion_variant(prog, (A, B), by_bond_dim=by_bond_dim)
    assert got is prog.block_sparse


@needs_no_fusion
def test_dense_sites_pick_the_default_variant(any_size):
    prog = _block_sparse_jit.block_sparse_jit(lambda x: x)
    got = cea._fusion_variant(
        prog, (_wrap_as_dense_tensor(jnp.ones((2, 2, 2, 2, 2))),), by_bond_dim=True
    )
    assert got is prog.dense
