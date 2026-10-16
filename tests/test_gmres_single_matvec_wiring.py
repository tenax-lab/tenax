"""The eager-GMRES adjoint compiles the J^T matvec once (single-matvec GMRES).

Before: the GMRES program inlined three matvec call sites and the standalone
``_jit_apply_Jt`` compiled a fourth copy for the seed choice and residual
check; the fixed-point path also compiled a standalone ``_jit_dE_denv`` for
its cold seed and again for its fallback.  These tests count XLA backend
compiles by function name, so they hold whatever the solver's internals are.
"""

from __future__ import annotations

import collections

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

_COMPILED: list[str] = []


def _bare(name: str) -> str:
    """``jit(f)`` -> ``f``: XLA reports jitted functions wrapped in ``jit(...)``."""
    return name[4:-1] if name.startswith("jit(") and name.endswith(")") else name


jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _COMPILED.append(_bare(kw.get("fun_name", "?")))
        if event == "/jax/core/compile/backend_compile_duration"
        else None
    )
)

_GATE = heisenberg_gate()
#: Same configuration as tests/test_shared_chain_rule_bwd.py: the fused loop
#: solves seed 0 and gives up on seed 2 (asserted per call below).
_FUSED_SEED, _FALLBACK_SEED = 0, 2


def _energy(A_arr, adjoint_method, gmres_tol=1e-6):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(A_arr)},
            SINGLE_SITE_NEIGHBORS,
            _GATE,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
            adjoint_method=adjoint_method,
            gmres_tol=gmres_tol,
        )
    )


def _site(seed):
    rng = np.random.default_rng(seed)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    return A0 / jnp.linalg.norm(A0)


@pytest.fixture
def _fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


@pytest.mark.usefixtures("_fresh_vjp_cache")
def test_fixed_point_path_compiles_no_standalone_matvec_or_rhs():
    grad = jax.grad(lambda a: _energy(a, "fixed_point"))
    mark = len(_COMPILED)
    grad(_site(_FUSED_SEED))
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("converged") and not diag.get("diverged"), "premise: fused path"
    grad(_site(_FALLBACK_SEED))
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("diverged") or not diag.get("converged"), "premise: fallback ran"
    compiled = set(_COMPILED[mark:])
    assert "_jit_gmres_one_matvec_solve" in compiled
    assert "_jit_apply_Jt" not in compiled, sorted(compiled)
    assert "_jit_dE_denv" not in compiled, sorted(compiled)


@pytest.mark.usefixtures("_fresh_vjp_cache")
def test_gmres_branch_compiles_no_standalone_matvec():
    mark = len(_COMPILED)
    jax.grad(lambda a: _energy(a, "gmres"))(_site(_FUSED_SEED))
    compiled = set(_COMPILED[mark:])
    assert "_jit_gmres_one_matvec_solve" in compiled
    assert "_jit_apply_Jt" not in compiled, sorted(compiled)


@pytest.mark.usefixtures("_fresh_vjp_cache")
def test_gmres_branch_candidate_structure_is_fixed():
    """Cold (no warm start) and warm backwards share one compiled solve: a
    missing warm start must be passed as zeros, not None, or the pytree
    structure of ``candidates`` changes and the solve recompiles."""
    grad = jax.grad(lambda a: _energy(a, "gmres"))
    grad(_site(_FUSED_SEED))  # cold
    mark = len(_COMPILED)
    grad(_site(_FUSED_SEED) * 1.001)  # warm, same shapes
    assert "_jit_gmres_one_matvec_solve" not in _COMPILED[mark:]


def test_fallback_gradient_matches_gmres_branch():
    a = _site(_FALLBACK_SEED)
    g_fb = jax.grad(lambda x: _energy(x, "fixed_point", gmres_tol=1e-11))(a)
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("diverged") or not diag.get("converged"), "premise: fallback ran"
    g_gm = jax.grad(lambda x: _energy(x, "gmres", gmres_tol=1e-11))(a)
    np.testing.assert_allclose(
        np.asarray(g_fb), np.asarray(g_gm), rtol=1e-7, atol=1e-10
    )


def test_fused_gradient_matches_gmres_branch():
    a = _site(_FUSED_SEED)
    g_fp = jax.grad(lambda x: _energy(x, "fixed_point", gmres_tol=1e-11))(a)
    g_gm = jax.grad(lambda x: _energy(x, "gmres", gmres_tol=1e-11))(a)
    np.testing.assert_allclose(
        np.asarray(g_fp), np.asarray(g_gm), rtol=1e-7, atol=1e-10
    )
