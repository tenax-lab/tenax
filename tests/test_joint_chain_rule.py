"""Joint-VJP chain rule (``CTMConfig.adjoint_chain_rule="joint"``, the default via ``"auto"``).

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


def test_default_is_auto():
    assert CTMConfig().adjoint_chain_rule == "auto"


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_auto_runs_joint_and_shares_its_programs():
    """``"auto"`` resolves to ``"joint"`` before the dispatch key: one cache
    entry for both, no ``_jit_chain_rule``, and the joint gradient."""
    A0 = _site(_FUSED_SEED)
    mark = len(_TRACED)
    g_auto = _grad(A0, "auto")
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" not in traced, sorted(traced)
    assert {"_jit_joint_adjoint_solve", "_jit_direct"} <= traced, sorted(traced)
    mark = len(_TRACED)
    g_joint = _grad(A0, "joint")
    assert len(cea._VJP_CACHE) == 1, list(cea._VJP_CACHE)
    assert not set(_TRACED[mark:]) & {"_jit_joint_adjoint_solve", "_jit_direct"}
    # Not bit-equal: the shared entry warm-starts the second solve from the
    # first one's lambda (#501), so they agree to the solve tolerance.
    np.testing.assert_allclose(
        np.asarray(g_auto), np.asarray(g_joint), rtol=1e-10, atol=1e-12
    )


def test_ctm_energy_implicit_defaults_to_auto():
    import inspect

    sig = inspect.signature(ctm_energy_implicit)
    assert sig.parameters["adjoint_chain_rule"].default == "auto"


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
    # assert_allclose passes on two NaN gradients (equal_nan); this test also
    # stands in for the deleted test_complex128_ad 1-site finiteness check.
    assert np.all(np.isfinite(np.asarray(g_joint)))
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
    assert used_joint == "flow", used_joint
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_joint_gmres_never_traces_the_separate_chain_rule():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "joint", method="gmres")
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" not in traced, sorted(traced)
    assert {"_jit_joint_adjoint_solve", "_jit_direct"} <= traced, sorted(traced)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_separate_gmres_never_traces_the_joint_programs():
    mark = len(_TRACED)
    _grad(_site(_FUSED_SEED), "separate", method="gmres")
    traced = set(_TRACED[mark:])
    assert "_jit_chain_rule" in traced
    assert not traced & {"_jit_joint_adjoint_solve", "_jit_direct", "_jit_apply_joint"}


# ------------------------------------------------------- fused fixed point


def _fused_premise():
    d = get_last_implicit_ad_diagnostics()
    assert d.get("converged") and not d.get("diverged"), d
    return d


def test_joint_fused_matches_separate():
    """The joint GMRES polish takes the fused loop's lambda as its seed, so the
    gradient is the separate one to round-off."""
    A0 = _site(_FUSED_SEED)
    g_sep = _grad(A0, "separate")
    _fused_premise()
    g_joint = _grad(A0, "joint")
    _fused_premise()
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


def test_joint_fused_at_the_default_tolerance():
    A0 = _site(_FUSED_SEED)
    g_sep = np.asarray(_grad(A0, "separate", tol=1e-6))
    _fused_premise()
    g_joint = np.asarray(_grad(A0, "joint", tol=1e-6))
    _fused_premise()
    assert np.max(np.abs(g_joint - g_sep)) <= 1e-4 * np.max(np.abs(g_sep))


def _fallback_premise():
    d = get_last_implicit_ad_diagnostics()
    assert d.get("diverged") or not d.get("converged"), "premise: fallback ran"
    # A stationary forward, or the gradient is not reproducible across
    # machines: on the seed-2 fixture (stationarity residual 0.795) Ubuntu CI
    # gave gradients ~3e-3 off this host's in BOTH modes, and 3e-2 apart.
    assert d.get("forward_converged"), d.get("forward_stationarity_residual")
    assert d["forward_stationarity_residual"] < 1e-8, d


def test_joint_fallback_matches_separate():
    """The GMRES fallback on a converged forward: the fused loop is capped at
    3 iterations (``gmres_maxiter`` caps it; GMRES gets 3 restart cycles)."""
    A0 = _site(_FUSED_SEED)
    g_sep = _grad(A0, "separate", tol=1e-6, gmres_maxiter=3)
    _fallback_premise()
    g_joint = _grad(A0, "joint", tol=1e-6, gmres_maxiter=3)
    _fallback_premise()
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


def test_joint_fused_matches_separate_two_site():
    arrs = (_site(0), _site(1))
    kw = dict(neighbors=CHECKERBOARD_NEIGHBORS)
    g_sep = _grad(arrs, "separate", **kw)
    _fused_premise()
    g_joint = _grad(arrs, "joint", **kw)
    _fused_premise()
    for a, b in zip(g_joint, g_sep):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


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
    assert {"_jit_joint_adjoint_solve", "_jit_direct"} <= traced, sorted(traced)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["gmres", "fixed_point"])
def test_joint_matches_separate_u1_multiblock(method):
    """Multi-block U(1) site, built as benchmarks/bench_chain_rule_compile.py
    ``_site(2, "u1", 0)`` does (copied: ``benchmarks`` is not a package).

    The block-sparse cotangent pytree carried through the GMRES loop is
    exercised here on a real multi-block site; the trivial-charge (one block)
    U(1) twin this file used to carry was a strict subset of it and was
    removed in the slow-bucket prune (#1149)."""
    from tenax.algorithms.ipeps import (
        heisenberg_gate_u1sz,
        heisenberg_u1sz_init_pair,
    )
    from tenax.core.tensor import SymmetricTensor

    A, _ = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
    nrm = jnp.sqrt(sum(jnp.sum(b**2) for b in A.blocks.values()))
    site = SymmetricTensor({k: b / nrm for k, b in A.blocks.items()}, A.indices)
    assert len(site.blocks) > 1, "premise: multi-block site"
    gate = heisenberg_gate_u1sz()

    def grad(mode):
        return jax.grad(
            lambda A: jnp.real(
                ctm_energy_implicit(
                    {(0, 0): A},
                    SINGLE_SITE_NEIGHBORS,
                    gate,
                    chi=4,
                    max_iter=60,
                    min_iter=8,
                    conv_tol=1e-10,
                    adjoint_method=method,
                    gmres_tol=1e-11,
                    adjoint_chain_rule=mode,
                )
            )
        )(site)

    g_sep = grad("separate")
    if method == "fixed_point":
        _fused_premise()
    g_joint = grad("joint")
    if method == "fixed_point":
        _fused_premise()
    assert type(g_joint) is type(g_sep)
    assert np.all(np.isfinite(np.asarray(g_joint._data)))  # equal_nan below
    np.testing.assert_allclose(
        np.asarray(g_joint._data),
        np.asarray(g_sep._data),
        rtol=1e-10,
        atol=1e-12,
    )


# ----------------------------------------------------------------- PESS


class _Captured(Exception):
    pass


@pytest.mark.parametrize("builder", ["supersite", "exact", "3site"])
def test_pess_loss_builders_forward_the_chain_rule_setting(builder, monkeypatch):
    """Codex review of #1123: the PESS builders take a ``CTMConfig`` but did not
    pass ``adjoint_chain_rule`` on, so ``"joint"`` was silently ignored."""
    from tenax.algorithms import pess_optimize as po
    from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
    from tenax.algorithms.pess import (
        IPESSState,
        kagome_xxz_pess_cg_gates,
        kagome_xxz_pess_cg_gates_exact,
    )

    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Captured

    monkeypatch.setattr(po, "ctm_energy_implicit", spy)
    cfg = CTMConfig(
        chi=4,
        projector_method="svd",
        forward_gauge="phase",
        chi_ramp=None,
        adjoint_chain_rule="joint",
    )
    if builder == "supersite":
        loss = po.build_pess_loss(kagome_xxz_pess_cg_gates(delta=1.0, d=3), cfg)
        state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    elif builder == "exact":
        loss = po.build_pess_loss_exact(kagome_xxz_pess_cg_gates_exact(1.0, 2), cfg)
        state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    else:
        loss = po.build_pess_loss_3site_multisite(
            kagome_3site_bond_gates(delta=1.0, d=3), cfg
        )
        state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    with pytest.raises(_Captured):
        loss(state)
    assert seen.get("adjoint_chain_rule") == "joint", sorted(seen)


# ------------------------------------------------------------- split CTM


@pytest.mark.parametrize("recipe", ["1x1", "2x2"])
def test_split_ctm_refuses_the_joint_chain_rule(recipe):
    """Codex review of #1123: the split backward never reads the setting, so
    "joint" there would be silently ignored."""
    from tenax.algorithms.ipeps_ad_policy import validate_split_ctm_config

    with pytest.raises(NotImplementedError, match="adjoint_chain_rule"):
        validate_split_ctm_config(CTMConfig(adjoint_chain_rule="joint"), recipe)
    validate_split_ctm_config(CTMConfig(), recipe)  # the default "auto" passes
    validate_split_ctm_config(CTMConfig(adjoint_chain_rule="separate"), recipe)
