"""Smoke tests for the kagome iPESS AD loss closures.

Covers both the CG-iPEPS supersite path
(:func:`tenax.algorithms.pess_optimize.build_pess_loss`) and the 3-site
multisite path
(:func:`tenax.algorithms.pess_optimize.build_pess_loss_3site_multisite`).
The two paths target the same physics but live on different variational
manifolds (multisite includes ``T_d`` as a real parameter, supersite
freezes it bit-exact).
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from unittest.mock import patch as _mock_patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import IPESSState, kagome_xxz_pess_cg_gates
from tenax.algorithms.pess_optimize import (
    _backtracking_line_search,
    _initial_alpha,
    build_pess_loss,
    build_pess_loss_3site_multisite,
    optimize_pess_3site_multisite_ad,
    optimize_pess_ad,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_test_config(chi: int) -> CTMConfig:
    """Small CTM config for smoke tests: tight iteration budget, SVD
    projectors (the standard square CTM AD default)."""
    return CTMConfig(
        chi=chi,
        max_iter=20,
        min_iter=4,
        conv_tol=1e-6,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-4,
        gmres_maxiter=50,
        gmres_restart=20,
        chi_ramp=None,
    )


def test_build_pess_loss_3site_multisite_rejects_non_svd_projector():
    bad_config = _dc_replace(_make_test_config(chi=4), projector_method="eigh")
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    with pytest.raises(ValueError, match="projector_method"):
        build_pess_loss_3site_multisite(bond_gates, bad_config)


def test_build_pess_loss_3site_multisite_rejects_non_phase_gauge():
    bad_config = _dc_replace(_make_test_config(chi=4), forward_gauge="sigma")
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    with pytest.raises(ValueError, match="forward_gauge"):
        build_pess_loss_3site_multisite(bond_gates, bad_config)


def test_pess_loss_returns_finite_real_scalar():
    """The loss closure returns a finite real scalar for a random PESS state."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(cg_gates, config)
    e0 = loss_fn(state)

    assert jnp.shape(e0) == ()
    assert jnp.isrealobj(e0) or jnp.allclose(e0.imag, 0.0, atol=1e-10)
    assert jnp.isfinite(e0)


def test_pess_loss_is_differentiable():
    """``jax.grad`` of the loss returns finite gradients on the iPESS primitives.

    The CG-iPEPS path optimizes ``(R_a, R_b, R_c, T_u, lambdas)``; ``T_d``
    is held frozen (its effect is absorbed via the down-bond ``sqrt(λ)``
    gauges).
    """
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(cg_gates, config)
    e0 = loss_fn(state)
    g = jax.grad(loss_fn)(state)

    assert jnp.isfinite(e0)
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u):
        assert jnp.all(jnp.isfinite(arr)), (
            "non-finite values in PESS gradient — implicit-AD CTM backward "
            "produced NaN/Inf on a smoke-test state."
        )
        # IPESSState.random returns complex128 tensors — gradients should
        # match dtype to keep the variational regime intact.
        assert arr.dtype == jnp.complex128
    for lam_g in g.lambdas:
        assert jnp.all(jnp.isfinite(lam_g))


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_pess_loss_runs_at_multiple_anisotropies(delta: float):
    """Smoke check: closure runs for a few values of XXZ anisotropy."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(1))
    cg_gates = kagome_xxz_pess_cg_gates(delta=delta, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss(cg_gates, config)
    e = loss_fn(state)
    assert jnp.isfinite(e)


def test_optimize_pess_ad_decreases_energy():
    """L-BFGS lowers the triangle energy below the random-init baseline."""
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(2))
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    e0 = float(build_pess_loss(cg_gates, config)(state0))
    state_opt, e_opt = optimize_pess_ad(
        state0, cg_gates, config, max_iter=5, verbose=False
    )

    assert jnp.isfinite(e_opt)
    assert e_opt < e0, f"L-BFGS did not decrease energy: e0={e0}, e_opt={e_opt}"


def test_optimize_pess_ad_preserves_shapes_and_T_d():
    """Shapes/dtype preserved; T_d frozen unchanged (it's not a CG-path AD variable)."""
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(3))
    cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    state_opt, _ = optimize_pess_ad(state0, cg_gates, config, max_iter=2)

    assert state_opt.R_a.shape == state0.R_a.shape
    assert state_opt.R_b.shape == state0.R_b.shape
    assert state_opt.R_c.shape == state0.R_c.shape
    assert state_opt.T_u.shape == state0.T_u.shape
    assert state_opt.T_d.shape == state0.T_d.shape
    assert state_opt.R_a.dtype == jnp.complex128
    assert state_opt.T_u.dtype == jnp.complex128
    # T_d is not optimized in the CG path — it's preserved bit-exact.
    assert jnp.array_equal(state_opt.T_d, state0.T_d)


# ---------------------------------------------------------------------------
# 3-site multisite AD path smoke tests
# ---------------------------------------------------------------------------


def test_pess_loss_3site_multisite_returns_finite_real_scalar():
    """Multisite loss closure returns a finite real scalar on a random state."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss_3site_multisite(bond_gates, config)
    e0 = loss_fn(state)

    assert jnp.shape(e0) == ()
    assert jnp.isrealobj(e0) or jnp.allclose(e0.imag, 0.0, atol=1e-10)
    assert jnp.isfinite(e0)


def test_pess_loss_3site_multisite_is_differentiable_through_T_d():
    """``jax.grad`` of the multisite loss flows through every iPESS primitive
    *including* ``T_d`` — this is the defining contrast with the supersite
    path, which freezes ``T_d`` bit-exact."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss_3site_multisite(bond_gates, config)
    e0 = loss_fn(state)
    g = jax.grad(loss_fn)(state)

    assert jnp.isfinite(e0)
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u, g.T_d):
        assert jnp.all(jnp.isfinite(arr)), (
            "non-finite values in multisite gradient — implicit-AD CTM "
            "backward produced NaN/Inf on a smoke-test state."
        )
        assert arr.dtype == jnp.complex128
    for lam_g in g.lambdas:
        assert jnp.all(jnp.isfinite(lam_g))
    # Non-trivial T_d gradient: the multisite encoding actually uses T_d, so
    # ||g.T_d|| > 0 unless the state happens to land at a critical point.
    assert float(jnp.linalg.norm(g.T_d)) > 0.0, (
        "T_d gradient is identically zero — the multisite loss is not "
        "differentiating through T_d (regression vs supersite path)."
    )


def test_optimize_pess_3site_multisite_ad_decreases_energy():
    """L-BFGS lowers the multisite energy below the random-init baseline."""
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(2))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    e0 = float(build_pess_loss_3site_multisite(bond_gates, config)(state0))
    state_opt, e_opt = optimize_pess_3site_multisite_ad(
        state0, bond_gates, config, max_iter=5, verbose=False
    )

    assert jnp.isfinite(e_opt)
    assert e_opt < e0, (
        f"multisite L-BFGS did not decrease energy: e0={e0}, e_opt={e_opt}"
    )


def test_optimize_pess_3site_multisite_ad_preserves_shapes_and_optimises_T_d():
    """Shapes/dtype preserved; ``T_d`` is updated by the multisite optimiser
    (unlike the CG path)."""
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(3))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=8)

    state_opt, _ = optimize_pess_3site_multisite_ad(
        state0, bond_gates, config, max_iter=2
    )

    assert state_opt.R_a.shape == state0.R_a.shape
    assert state_opt.R_b.shape == state0.R_b.shape
    assert state_opt.R_c.shape == state0.R_c.shape
    assert state_opt.T_u.shape == state0.T_u.shape
    assert state_opt.T_d.shape == state0.T_d.shape
    assert state_opt.T_d.dtype == jnp.complex128
    # T_d *is* optimized in the multisite path — must differ from input.
    assert not jnp.array_equal(state_opt.T_d, state0.T_d), (
        "T_d unchanged after L-BFGS in the multisite path — the optimiser "
        "is silently freezing T_d (regression: it should be a live param)."
    )


# ---------------------------------------------------------------------------
# Line search initial-step heuristic (issue #401)
# ---------------------------------------------------------------------------


def test_initial_alpha_positive_when_energy_zero():
    """``_initial_alpha`` must return a positive step when ``|energy| → 0``.

    Regression for issue #401: the Nocedal & Wright ``f_low = 0`` heuristic
    ``alpha0 = |E| / |slope|`` collapses to zero whenever the user supplies
    a constant-shifted Hamiltonian that drives ``f(x0) → 0``, even when the
    gradient is nonzero.  ``alpha == 0`` then propagates through Armijo /
    Hager-Zhang as a no-op step and the optimizer stalls without descending.
    """
    alpha = _initial_alpha(energy=0.0, slope=1.0)
    assert alpha > 0.0


def test_initial_alpha_zero_when_slope_zero():
    """At a true stationary point (slope == 0) the initial step must stay 0.

    Regression for the PR-#404 review: blanket flooring on ``|energy|`` would
    return a positive step even when the descent direction is identically
    zero.  ``trial = params + alpha * 0 == params`` then satisfies the
    Armijo condition trivially (``f_trial == energy``) so the line search
    accepts the no-op, ``alpha == 0`` no longer signals "no improvement",
    and ``optimize_pess_*`` burns ``max_iter`` CTM passes without making
    progress.  Keep the documented ``alpha == 0`` break-signal contract.
    """
    assert _initial_alpha(energy=0.0, slope=0.0) == 0.0
    # Energy nonzero but slope zero is the same regime — no descent.
    assert _initial_alpha(energy=0.0, slope=1e-40) == 0.0


def test_initial_alpha_scales_with_slope_in_low_energy_regime():
    """``_initial_alpha`` must keep slope-scaling when energy is tiny.

    Regression for the second PR-#404 review: a slope-independent floor
    (e.g. ``alpha0 = 1e-3`` whenever ``|E| < 1e-12``) is far too coarse
    for stiff or poorly scaled low-energy objectives.  At
    ``|E| = 5e-13`` and ``|slope| = 1e-3`` the optimal step is
    ``|E|/|slope| = 5e-10``; Armijo at most reaches ``1e-3 · 0.5⁷ ≈
    7.8e-6`` from a ``1e-3`` start, missing it by orders of magnitude.

    Floor the *numerator* ``|E|`` at ``1e-12`` so the heuristic stays
    ``|E_floor|/|slope|`` and tracks the slope: at ``|slope| = 1e-3``
    we get ``alpha0 = 1e-9`` (close to the optimum), at
    ``|slope| = 1e-6`` we get ``alpha0 = 1e-6`` (capped well below 1).
    """
    # Stiff low-energy regime: original |E|/|slope| was 5e-10, floor only
    # bumps it to 1e-9 (still within an Armijo bisection of optimal).
    alpha = _initial_alpha(energy=5e-13, slope=-1e-3)
    assert 1e-10 < alpha < 1e-2, (
        f"alpha0 should track |E_floor|/|slope| ~ 1e-9, got {alpha}"
    )
    # Doubling |slope| should roughly halve alpha0 — confirms the heuristic
    # is still slope-scaled rather than a fixed step.
    alpha_2x = _initial_alpha(energy=5e-13, slope=-2e-3)
    assert alpha_2x == pytest.approx(alpha / 2.0, rel=1e-9)
    # Normal regime (|E| > 1e-12) is unchanged by the floor.
    assert _initial_alpha(energy=0.1, slope=-100.0) == pytest.approx(1e-3, rel=1e-9)


def test_backtracking_line_search_progresses_with_zero_energy():
    """Backtracking line search descends on a constant-shifted quadratic.

    Regression for issue #401: with ``f(x0) = 0`` and nonzero gradient, the
    descent direction is well-defined but ``_initial_alpha`` previously
    returned 0, so the search returned ``best_alpha = 0`` and the optimizer
    made no progress.
    """
    # Loss f(x) = 0.5 * sum(x**2) - 0.5 * sum(x0**2) — constant-shifted
    # quadratic.  At x = x0 the loss is exactly 0 but gradient is x0 ≠ 0,
    # so any positive step in the steepest-descent direction reduces f.
    x0 = {"w": jnp.array([0.5, -0.25, 0.1])}
    shift = 0.5 * float(jnp.sum(x0["w"] ** 2))

    def loss_fn(params):
        return 0.5 * jnp.sum(params["w"] ** 2) - shift

    energy = float(loss_fn(x0))
    assert abs(energy) < 1e-12

    grad = {"w": x0["w"]}
    direction = jax.tree.map(lambda g: -g, grad)

    new_params, new_energy, alpha = _backtracking_line_search(
        x0, direction, grad, energy, loss_fn
    )

    assert alpha > 0.0, "line search returned zero step on a descending direction"
    assert new_energy < energy, (
        f"line search did not decrease energy: {energy} -> {new_energy}"
    )
    assert jnp.isfinite(new_energy)
    assert jnp.all(jnp.isfinite(new_params["w"]))


# ---------------------------------------------------------------------------
# Tensor-valued bond gates (issue #402)
# ---------------------------------------------------------------------------


def _wrap_gate_as_dense_tensor(H: jnp.ndarray) -> DenseTensor:
    """Wrap a (d,d,d,d) ndarray gate as a DenseTensor with trivial U(1)
    charges, mirroring the convention used by ``ipeps.heisenberg_gate`` /
    ``ipeps.xxz_gate``.

    ``compute_energy_pess_3site_multisite`` accepts ``Tensor`` valued
    gates and demotes them via ``.todense().reshape(d,d,d,d)``; this
    helper exists so we can construct that branch's input from the test
    side without depending on a public wrapper function.
    """
    d = int(H.shape[0])
    sym = U1Symmetry()
    charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )
    return DenseTensor(jnp.asarray(H), indices)


def test_pess_loss_3site_multisite_accepts_tensor_bond_gates():
    """Closure builder accepts DenseTensor-valued gates (issue #402).

    The downstream energy fn ``compute_energy_pess_3site_multisite``
    already advertises ``jax.Array | Tensor`` and routes Tensor through
    ``.todense().reshape(d,d,d,d)``.  ``build_pess_loss_3site_multisite``
    must therefore not reject Tensor inputs at the entry-point
    ``int(... .shape[0])`` access — that broke symmetry-aware /
    Tensor-protocol gates before they ever reached the supported code
    path.  Test feeds a ``DenseTensor``-wrapped gate dict and asserts the
    loss closure builds and returns a finite real scalar.
    """
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    array_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    bond_gates = {k: _wrap_gate_as_dense_tensor(g) for k, g in array_gates.items()}
    assert all(isinstance(g, DenseTensor) for g in bond_gates.values())
    config = _make_test_config(chi=8)

    loss_fn = build_pess_loss_3site_multisite(bond_gates, config)
    e0 = loss_fn(state)

    assert jnp.shape(e0) == ()
    assert jnp.isfinite(e0)
    assert jnp.isrealobj(e0) or jnp.allclose(e0.imag, 0.0, atol=1e-10)


def test_optimize_pess_3site_multisite_ad_accepts_tensor_bond_gates():
    """``optimize_pess_3site_multisite_ad`` must work with Tensor gates.

    Regression for issue #402 follow-up (PR #405 review): the loss closure
    promotes Tensor gates locally, but the optimizer's warm-start helper
    ``_build_site_tensors_for_warm_start`` separately did
    ``next(iter(bond_gates.values())).shape[0]`` on the un-promoted dict.
    The first ``loss(params)`` call succeeds, then the immediate
    ``_update_env_cache(params)`` raises ``AttributeError`` and the
    advertised Tensor-valued multisite optimisation path stays unusable.

    One L-BFGS step is enough to drive both the loss path and
    ``_update_env_cache``; we don't assert on energy descent (covered by
    ``test_optimize_pess_3site_multisite_ad_decreases_energy`` for arrays).
    """
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    array_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    bond_gates = {k: _wrap_gate_as_dense_tensor(g) for k, g in array_gates.items()}
    config = _make_test_config(chi=8)

    state_opt, e_opt = optimize_pess_3site_multisite_ad(
        state0, bond_gates, config, max_iter=1, verbose=False
    )

    assert jnp.isfinite(e_opt)
    assert state_opt.R_a.shape == state0.R_a.shape
    assert state_opt.T_d.shape == state0.T_d.shape


def test_build_pess_loss_3site_multisite_passes_env_init_from_cache():
    """When ``env_cache`` is provided, the loss reads ``envs`` from it and
    forwards them as ``env_init=`` to the CTM AD entry point."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=4)
    SENTINEL = object()
    env_cache = {"envs": SENTINEL}

    seen: dict = {}

    def fake_ctm_energy_implicit(*args, **kwargs):
        seen.update(kwargs)
        return jnp.array(0.0)

    loss_fn = build_pess_loss_3site_multisite(bond_gates, config, env_cache=env_cache)
    with _mock_patch(
        "tenax.algorithms.pess_optimize.ctm_energy_implicit",
        fake_ctm_energy_implicit,
    ):
        loss_fn(state)

    assert seen.get("env_init") is SENTINEL


def test_build_pess_loss_3site_multisite_default_env_cache_is_none():
    """No env_cache → env_init=None on every call (existing behavior)."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=4)

    seen: dict = {}

    def fake_ctm_energy_implicit(*args, **kwargs):
        seen.update(kwargs)
        return jnp.array(0.0)

    loss_fn = build_pess_loss_3site_multisite(bond_gates, config)
    with _mock_patch(
        "tenax.algorithms.pess_optimize.ctm_energy_implicit",
        fake_ctm_energy_implicit,
    ):
        loss_fn(state)

    assert seen.get("env_init") is None


def test_build_pess_loss_3site_multisite_forwards_plateau_patience():
    """PESS implicit-AD loss must forward ``CTMConfig.plateau_patience``.

    Codex review on PR #439: prior to the fix, the PESS supersite and
    multisite loss functions called ``ctm_energy_implicit`` without
    passing ``config.plateau_patience``, so PESS gradient evaluations
    silently disabled the user-configured plateau stop-loss while
    warm-starts through ``ctm_converge_kwargs`` did honor it.  This
    asserts the kwarg now reaches ``ctm_energy_implicit``.
    """
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=4)
    config.plateau_patience = 7

    seen: dict = {}

    def fake_ctm_energy_implicit(*args, **kwargs):
        seen.update(kwargs)
        return jnp.array(0.0)

    loss_fn = build_pess_loss_3site_multisite(bond_gates, config)
    with _mock_patch(
        "tenax.algorithms.pess_optimize.ctm_energy_implicit",
        fake_ctm_energy_implicit,
    ):
        loss_fn(state)

    assert seen.get("plateau_patience") == 7


def test_optimize_pess_3site_multisite_ad_warm_starts_envs():
    """After the first L-BFGS step, `_update_env_cache` populates the cache,
    so the SECOND gradient evaluation receives a non-None env_init.

    We monkey-patch ``ctm_energy_implicit`` to capture the ``env_init``
    kwarg on every call and assert it transitions from None → not-None.
    """
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    config = _make_test_config(chi=4)

    captured_env_inits: list = []

    # Wrap the real implicit-AD entry so the optimizer can still make progress
    # but we observe the env_init kwarg on every call.
    from tenax.algorithms._ctm_energy_ad import (
        ctm_energy_implicit as real_ctm_energy_implicit,
    )

    def spy(*args, **kwargs):
        captured_env_inits.append(kwargs.get("env_init"))
        return real_ctm_energy_implicit(*args, **kwargs)

    with _mock_patch("tenax.algorithms.pess_optimize.ctm_energy_implicit", spy):
        optimize_pess_3site_multisite_ad(
            state,
            bond_gates,
            config,
            max_iter=2,
            verbose=False,
        )

    assert len(captured_env_inits) >= 2, "loss must be called at least twice"
    assert captured_env_inits[0] is None, "first call should be cold-start"
    assert any(env is not None for env in captured_env_inits[1:]), (
        "subsequent calls should warm-start from env_cache populated by "
        "_update_env_cache"
    )


def test_build_pess_loss_3site_multisite_forwards_projector_backward():
    """`config.projector_backward` must be threaded through to ctm_energy_implicit."""
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    bond_gates = kagome_3site_bond_gates(delta=1.0, d=3)
    base_config = _make_test_config(chi=4)
    config = _dc_replace(base_config, projector_backward="standard")

    seen: dict = {}

    def fake_ctm_energy_implicit(*args, **kwargs):
        seen.update(kwargs)
        return jnp.array(0.0)

    loss_fn = build_pess_loss_3site_multisite(bond_gates, config)
    with _mock_patch(
        "tenax.algorithms.pess_optimize.ctm_energy_implicit",
        fake_ctm_energy_implicit,
    ):
        loss_fn(state)

    assert seen.get("projector_backward") == "standard"
