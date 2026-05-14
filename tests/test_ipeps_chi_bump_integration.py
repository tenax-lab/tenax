"""End-to-end integration test for variPEPS §2.8.2 auto-χ_E bump.

Lives in its own file (mapped ``"algorithm"`` in ``conftest.py``) because
``optimize_gs_ad`` is JAX-JIT-heavy and the test takes ~3 minutes; it
should not slow down ``pytest -m core`` (the CI required-check set).
"""

from __future__ import annotations

import math

import jax

from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad


def test_optimize_gs_ad_auto_bump_raises_chi_under_pressure():
    """Heisenberg D=2 at chi=2: auto-bump must raise chi during optimization.

    Asserts:
        - Returned env's corner χ axis is strictly larger than the initial chi=2.
        - Final energy is finite and a Python float.

    Why this works: chi=2 with D=2 (-> chi_eff = D²×chi = 8 needed for full
    information). The CTM SVD projector's cross-product M = C1g^H @ C4g is
    chi×chi and cannot measure discarded modes.  To get a meaningful eps_T > 0,
    _update_env_cache runs one non-JIT eigh sweep (rho = C1g @ C1g^H + C4g @
    C4g^H is chi*D² × chi*D² = 8×8) and discards 6 eigenmodes -> eps_T >> 1e-5,
    triggering the bump on the first L-BFGS step.
    """
    gate = heisenberg_gate()
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    A_init = jax.random.normal(k1, (2, 2, 2, 2, 2)) + 1j * jax.random.normal(
        k2, (2, 2, 2, 2, 2)
    )

    cfg = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=2,
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-5,
            chi_auto_bump_step=2,
            chi_max=8,
            max_iter=10,
            min_iter=2,
            conv_tol=1e-3,
        ),
        gs_num_steps=3,
        gs_implicit_ad=True,
        gs_verbose=False,
        su_init=False,
    )

    A_opt, env, e_opt = optimize_gs_ad(gate, A_init, cfg)

    # env is a CTMTensorEnv (NamedTuple of DenseTensor); corners are (chi, chi).
    final_chi = env.C1._data.shape[0]

    assert final_chi > 2, f"auto-bump never fired (final_chi={final_chi})"
    assert final_chi <= 8, f"chi exceeded chi_max=8 (final_chi={final_chi})"
    assert math.isfinite(float(e_opt)), f"final energy is not finite: {e_opt}"


def test_optimize_gs_ad_auto_bump_fires_after_line_search():
    """Regression for issue #419.

    The auto-χ_E bump must fire *between* L-BFGS iterations, never
    *within* one. The Wolfe line search compares ``f(0) = energy_val``
    and ``f'(0) = slope`` (both at the OLD χ from ``value_and_grad``)
    with ``f(α) = _phi(α)`` (closure that reads ``ctm_cfg`` lazily).
    If the bump runs between ``value_and_grad`` and the line search,
    the closures evaluate at the NEW χ and the three quantities come
    from two different objectives — a step can be rejected purely
    because of the χ discontinuity, or accepted spuriously when the
    NEW χ's energy is artificially lower.

    Spy on ``_maybe_bump_chi`` and ``hager_zhang_line_search`` and
    assert the line search runs first on the first iteration. Before
    the fix the order was ``[bump, line_search, ...]``; after the
    fix it is ``[line_search, bump, ...]``.
    """
    from tenax.algorithms import _line_search as ls_mod
    from tenax.algorithms import ipeps_optimize as opt_mod

    call_order: list[str] = []
    real_bump = opt_mod._maybe_bump_chi
    real_ls = ls_mod.hager_zhang_line_search

    def spy_bump(*args, **kwargs):
        call_order.append("bump")
        return real_bump(*args, **kwargs)

    def spy_ls(*args, **kwargs):
        call_order.append("line_search")
        return real_ls(*args, **kwargs)

    opt_mod._maybe_bump_chi = spy_bump
    ls_mod.hager_zhang_line_search = spy_ls
    try:
        gate = heisenberg_gate()
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        A_init = jax.random.normal(k1, (2, 2, 2, 2, 2)) + 1j * jax.random.normal(
            k2, (2, 2, 2, 2, 2)
        )

        cfg = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(
                chi=2,
                chi_auto_bump=True,
                chi_auto_bump_eps=1e-5,
                chi_auto_bump_step=2,
                chi_max=8,
                max_iter=10,
                min_iter=2,
                conv_tol=1e-3,
            ),
            gs_num_steps=2,
            gs_implicit_ad=True,
            gs_verbose=False,
            su_init=False,
        )
        optimize_gs_ad(gate, A_init, cfg)
    finally:
        opt_mod._maybe_bump_chi = real_bump
        ls_mod.hager_zhang_line_search = real_ls

    assert call_order, "neither _maybe_bump_chi nor hager_zhang_line_search ran"
    assert "bump" in call_order, "bump never fired — test premise invalid"
    assert "line_search" in call_order, (
        "line search never fired — Hager-Zhang path not exercised; test premise invalid"
    )
    assert call_order[0] == "line_search", (
        f"_maybe_bump_chi fired before the first hager_zhang_line_search call "
        f"in iteration 1 — the Wolfe line-search invariant requires both to run "
        f"at the same χ. Call order observed: {call_order}. (Issue #419.)"
    )


def test_optimize_gs_ad_auto_bump_fires_on_convergence_break():
    """The end-of-iter bump must also fire on the convergence-break path.

    Without this, an iteration that exits via ``delta_energy < gs_conv_tol``
    would skip the bump even when ``_update_env_cache`` just measured a
    large ``max_truncation_error``. A run that energy-stalls at too-small
    χ would return its final env at the old χ instead of re-evaluating
    at the requested auto-bump χ — codex P2 review comment on PR #432
    that was lost in #432's squash and is re-included here.

    Setup: ``gs_conv_tol`` is set very loose (100.0) so iteration 1
    breaks via the conv check. Iteration 0 runs the full body and
    bumps via the end-of-iter path. The conv-break in iteration 1
    must also bump for ``final_chi > chi_after_iter_0`` to hold.
    Without the fix the conv-break path skipped the bump, so
    ``final_chi`` stayed at the post-iter-0 value
    (chi=4 with ``chi_auto_bump_step=2``, ``chi_init=2``).
    """
    gate = heisenberg_gate()
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    A_init = jax.random.normal(k1, (2, 2, 2, 2, 2)) + 1j * jax.random.normal(
        k2, (2, 2, 2, 2, 2)
    )

    cfg = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=2,
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-5,
            chi_auto_bump_step=2,
            chi_max=8,
            max_iter=10,
            min_iter=2,
            conv_tol=1e-3,
        ),
        gs_num_steps=5,
        gs_conv_tol=100.0,  # force iter-1 conv break
        gs_implicit_ad=True,
        gs_verbose=False,
        su_init=False,
    )

    _, env, _ = optimize_gs_ad(gate, A_init, cfg)
    final_chi = env.C1._data.shape[0]

    # Iter 0 bumps chi 2→4 on the normal-flow path. Iter 1 trips the
    # conv check and must ALSO bump (chi 4→6) for this assertion to
    # hold. Without the fix the conv-break path skipped the bump
    # → final_chi=4.
    assert final_chi >= 6, (
        f"conv-break path skipped the auto-χ bump (final_chi={final_chi}); "
        "expected the bump to also fire when delta_energy < gs_conv_tol "
        "(PR #432 codex review, lost in #432's squash and re-included)."
    )


def test_optimize_gs_ad_auto_bump_runs_on_2site(monkeypatch):
    """2-site path: ``chi_auto_bump=True`` runs end-to-end and the
    reactive bump fires when ε_T > threshold (#472).

    Asserts:
        - ``optimize_gs_ad`` returns the expected 2-site tuple shape
          with no ``NotImplementedError`` (the guard dropped in #472).
        - When a non-zero ε_T is injected into the env cache between
          steps, the reactive ``_maybe_bump_chi`` call lands and the
          final env's chi exceeds the initial chi=2.

    Why the injection: the 2-site forward CTM uses the 2x2 plaquette
    recipe, whose ε_T is currently a 0.0 placeholder (the plaquette
    projector doesn't yet track it).  This test pins the *wiring* —
    that ``_maybe_bump_chi`` is invoked with the cache's stored ε_T —
    without depending on the 2x2 ε_T-tracking follow-up.
    """
    from tenax.algorithms import ipeps_optimize as opt_mod

    real_bump = opt_mod._maybe_bump_chi
    bump_calls: list[float] = []

    def spy_bump(ctm_cfg, env_cache, last_eps_t, *args, **kwargs):
        bump_calls.append(float(last_eps_t))
        # Override the cache's 0.0 placeholder so the reactive trigger
        # actually fires on the 2-site path.
        return real_bump(ctm_cfg, env_cache, 1e-3, *args, **kwargs)

    monkeypatch.setattr(opt_mod, "_maybe_bump_chi", spy_bump)

    gate = heisenberg_gate()
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    A_init = jax.random.normal(k1, (2, 2, 2, 2, 2)) + 1j * jax.random.normal(
        k2, (2, 2, 2, 2, 2)
    )
    B_init = jax.random.normal(k3, (2, 2, 2, 2, 2)) + 1j * jax.random.normal(
        k4, (2, 2, 2, 2, 2)
    )

    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=2,
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-5,
            chi_auto_bump_step=2,
            chi_max=8,
            max_iter=10,
            min_iter=2,
            conv_tol=1e-3,
        ),
        gs_optimizer="lbfgs",
        gs_num_steps=3,
        gs_verbose=False,
        su_init=False,
    )

    (A_opt, B_opt), envs, e_opt = optimize_gs_ad(gate, (A_init, B_init), cfg)

    final_chi = envs[0].C1._data.shape[0]

    assert bump_calls, "_maybe_bump_chi never ran on the 2-site path"
    assert final_chi > 2, (
        f"reactive bump never raised chi on 2-site (final_chi={final_chi}); "
        f"the wiring added in #472 did not invoke ``_maybe_bump_chi`` in a "
        f"position where its return value updates ``ctm_cfg_2s``."
    )
    assert final_chi <= 8, f"chi exceeded chi_max=8 (final_chi={final_chi})"
    assert math.isfinite(float(e_opt)), f"final energy is not finite: {e_opt}"
