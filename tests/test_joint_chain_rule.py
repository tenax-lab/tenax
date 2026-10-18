"""Opt-in joint-VJP chain rule (``CTMConfig.adjoint_chain_rule="joint"``).

The implicit-AD backward's chain rule needs ``J_params^T λ``, the params-VJP
of the CTM sweep, which ``_jit_chain_rule`` compiles as its own program -- the
largest single compile at D=3 (2,604 s on GPU).  Under ``"joint"`` the adjoint
solve's matvec is one VJP over ``(params, env)`` and yields that term itself.
These tests pin gradient parity with ``"separate"`` on every backward path and
that the separate chain rule is never compiled.  Spec:
docs/superpowers/specs/2026-10-18-joint-vjp-chain-rule-design.md
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
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig

_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)

#: One gate object: the gate fingerprint is part of the dispatch cache key.
_GATE = heisenberg_gate()

#: Same fixture as tests/test_shared_chain_rule_bwd.py: at chi=4 the fused
#: loop solves seed 0 and its divergence guard fires on seed 2.  Asserted per
#: call, not assumed.
_FUSED_SEED, _FALLBACK_SEED = 0, 2


@pytest.fixture
def fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


def _site(seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((2, 2, 2, 2, 2))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        A0 = A0 + 0.1j * rng.standard_normal((2, 2, 2, 2, 2))
    A0 = jnp.asarray(A0, dtype=dtype)
    return A0 / jnp.linalg.norm(A0)


def _energy(arrs, mode, *, method="fixed_point", tol=1e-11, neighbors=None, **kw):
    if not isinstance(arrs, (tuple, list)):
        arrs = (arrs,)
    neighbors = neighbors or SINGLE_SITE_NEIGHBORS
    coords = sorted(neighbors)
    return jnp.real(
        ctm_energy_implicit(
            {c: _wrap_as_dense_tensor(a) for c, a in zip(coords, arrs)},
            neighbors,
            _GATE,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
            adjoint_method=method,
            gmres_tol=tol,
            adjoint_chain_rule=mode,
            **kw,
        )
    )


def _grad(arrs, mode, **kw):
    return jax.grad(lambda a: _energy(a, mode, **kw))(arrs)


# ----------------------------------------------------------------- config


def test_default_is_separate():
    assert CTMConfig().adjoint_chain_rule == "separate"


def test_invalid_value_raises():
    with pytest.raises(ValueError, match="adjoint_chain_rule"):
        CTMConfig(adjoint_chain_rule="fused")


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_modes_get_separate_cache_entries():
    A0 = _site(_FUSED_SEED)
    _energy(A0, "separate")
    _energy(A0, "joint")
    keys = list(cea._VJP_CACHE)
    assert len(keys) == 2
    assert {"separate", "joint"} <= {p for k in keys for p in k if isinstance(p, str)}


# ----------------------------------------------------------- GMRES method


def test_joint_gmres_matches_separate():
    A0 = _site(_FUSED_SEED)
    g_sep = _grad(A0, "separate", method="gmres")
    g_joint = _grad(A0, "joint", method="gmres")
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-12, atol=1e-14
    )


def test_joint_gmres_matches_separate_complex():
    A0 = _site(_FUSED_SEED, dtype=jnp.complex128)
    g_sep = _grad(A0, "separate", method="gmres")
    g_joint = _grad(A0, "joint", method="gmres")
    assert g_joint.dtype == jnp.complex128
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-12, atol=1e-14
    )


def test_joint_gmres_matches_separate_flow():
    A0 = _site(_FUSED_SEED)
    kw = dict(method="gmres", projector_backward="flow", forward_gauge="bond_phase")
    g_sep = _grad(A0, "separate", **kw)
    used_sep = get_last_implicit_ad_diagnostics().get("projector_backward_used")
    g_joint = _grad(A0, "joint", **kw)
    used_joint = get_last_implicit_ad_diagnostics().get("projector_backward_used")
    assert used_sep == used_joint, (used_sep, used_joint)
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_joint_gmres_never_traces_the_separate_chain_rule():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "joint", method="gmres")
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" not in traced, sorted(traced)
    assert {"_jit_gmres_joint_solve", "_jit_direct"} <= traced, sorted(traced)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_separate_gmres_never_traces_the_joint_programs():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "separate", method="gmres")
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" in traced
    assert not traced & {"_jit_gmres_joint_solve", "_jit_direct", "_jit_apply_joint"}


# ------------------------------------------------------- fused fixed point


def _fused_premise():
    d = get_last_implicit_ad_diagnostics()
    assert d.get("converged") and not d.get("diverged"), d
    return d


def test_joint_fused_matches_separate():
    """lambda_{K-1} vs lambda_K: within the solve tolerance."""
    A0 = _site(_FUSED_SEED)
    g_sep = _grad(A0, "separate")
    _fused_premise()
    g_joint = _grad(A0, "joint")
    _fused_premise()
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-8, atol=1e-9
    )


def test_joint_fused_at_the_default_tolerance():
    A0 = _site(_FUSED_SEED)
    g_sep = np.asarray(_grad(A0, "separate", tol=1e-6))
    g_joint = np.asarray(_grad(A0, "joint", tol=1e-6))
    assert np.max(np.abs(g_joint - g_sep)) <= 1e-4 * np.max(np.abs(g_sep))


def test_joint_fallback_matches_separate():
    A0 = _site(_FALLBACK_SEED)
    g_sep = _grad(A0, "separate", tol=1e-6)
    d = get_last_implicit_ad_diagnostics()
    assert d.get("diverged") or not d.get("converged"), "premise: fallback ran"
    g_joint = _grad(A0, "joint", tol=1e-6)
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


def test_joint_fused_matches_separate_two_site():
    arrs = (_site(0), _site(1))
    kw = dict(neighbors=CHECKERBOARD_NEIGHBORS)
    g_sep = _grad(arrs, "separate", **kw)
    g_joint = _grad(arrs, "joint", **kw)
    for a, b in zip(g_joint, g_sep):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-7, atol=1e-9)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_joint_fused_warm_start_is_used():
    A0 = _site(_FUSED_SEED)
    _grad(A0, "joint")
    n_cold = _fused_premise()["n_iter"]
    _grad(A0, "joint")
    d = _fused_premise()
    assert not d.get("warm_start_invalidated")
    assert d["n_iter"] < n_cold, (d["n_iter"], n_cold)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_joint_fused_and_fallback_never_trace_the_separate_chain_rule():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "joint")
    _fused_premise()
    _grad(_site(_FALLBACK_SEED), "joint", tol=1e-6)
    d = get_last_implicit_ad_diagnostics()
    assert d.get("diverged") or not d.get("converged"), "premise: fallback ran"
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" not in traced, sorted(traced)
    assert "_jit_direct" not in traced, "the fallback must reuse the fused direct"
    assert "_jit_gmres_joint_solve" in traced


@pytest.mark.slow
def test_joint_fused_matches_separate_u1():
    """Block-sparse params: the cotangent is a SymmetricTensor pytree carried
    through the while loop.  Trivial-charge U(1) fixture of
    tests/test_block_sparse_ctm_ad.py (one block, full block-sparse path)."""
    from tests.test_block_sparse_ctm_ad import _make_u1_symmetric_ipeps

    A_sym = _make_u1_symmetric_ipeps()

    def grad(mode):
        return jax.grad(
            lambda A: jnp.real(
                ctm_energy_implicit(
                    {(0, 0): A},
                    SINGLE_SITE_NEIGHBORS,
                    _GATE,
                    chi=4,
                    max_iter=30,
                    conv_tol=1e-8,
                    gmres_tol=1e-11,
                    adjoint_chain_rule=mode,
                )
            )
        )(A_sym)

    g_sep, g_joint = grad("separate"), grad("joint")
    assert type(g_joint) is type(g_sep)
    np.testing.assert_allclose(
        np.asarray(g_joint._data), np.asarray(g_sep._data), rtol=1e-8, atol=1e-9
    )
