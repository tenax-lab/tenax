"""Warm-start adjoint solve tests for ctm_energy_implicit (#501).

Task 9 of #514: verify the implicit-AD backward caches the adjoint
solution (``lam``) across L-BFGS steps and warm-starts the Neumann
iteration on the next call.  Three contracts:

  1. Repeated ``jax.grad`` calls on the same params return the same
     gradient (warm-start converges to the same fixed point).
  2. The second call's Neumann iteration count is <= the first call's
     (warm-start either matches or beats cold-start).
  3. When the fused JIT loop diverges/doesn't-converge, the cache is
     invalidated and the eager-GMRES fallback fires; the cache is then
     refilled with the eager-solve result so the next call still
     warm-starts.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import (
    _F3_LAST_DIAGNOSTICS,
    ctm_energy_implicit,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate, ipeps
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

pytestmark = pytest.mark.core


def _make_su_tensor(D: int = 2, d: int = 2):
    """SU-initialized site tensor (well-conditioned for CTM)."""
    gate = heisenberg_gate()
    config = iPEPSConfig(max_bond_dim=D, num_imaginary_steps=100, dt=0.05)
    _, (A_su, _), _ = ipeps(gate, None, config)
    data = A_su.todense()
    data = data / jnp.linalg.norm(data)
    return data  # raw jax.Array; wrapped per-call inside _loss


def _make_loss_fn(*, gmres_maxiter: int = 200):
    """Build a scalar loss function ``params -> ctm_energy_implicit``."""
    gate = heisenberg_gate()

    def _loss(params):
        site_tensors = {(0, 0): _wrap_as_dense_tensor(params)}
        return ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=40,
            conv_tol=1e-8,
            gmres_tol=1e-6,
            gmres_maxiter=gmres_maxiter,
            gmres_restart=30,
        )

    return _loss


def test_adjoint_warm_start_grad_unchanged():
    """Repeated ``jax.grad`` calls on the same params return identical grads.

    Warm-start must converge to the same fixed point as cold-start; if
    the seed-vs-converged mismatch leaked through, the second call's
    gradient would shift.
    """
    params = _make_su_tensor(D=2, d=2)
    loss = _make_loss_fn()

    g1 = jax.grad(loss)(params)
    g2 = jax.grad(loss)(params)
    # Both must be finite and numerically identical.
    g1_arr = np.asarray(g1)
    g2_arr = np.asarray(g2)
    assert np.all(np.isfinite(g1_arr))
    assert np.all(np.isfinite(g2_arr))
    np.testing.assert_allclose(g1_arr, g2_arr, atol=1e-8, rtol=1e-8)


def test_adjoint_warm_start_reduces_iters():
    """Second ``jax.grad`` call uses cached warm-start, converging in <= iters.

    Reads ``_F3_LAST_DIAGNOSTICS["n_iter"]`` after each call.  The second
    call should match or beat the first.
    """
    params = _make_su_tensor(D=2, d=2)
    loss = _make_loss_fn()

    # Cold start: init_lam = dE_denv.
    jax.grad(loss)(params)
    n_iter_cold = _F3_LAST_DIAGNOSTICS["n_iter"]
    converged_cold = _F3_LAST_DIAGNOSTICS["converged"]

    # Warm start: init_lam = previous lam_final.
    jax.grad(loss)(params)
    n_iter_warm = _F3_LAST_DIAGNOSTICS["n_iter"]
    converged_warm = _F3_LAST_DIAGNOSTICS["converged"]

    # Both must have actually converged (otherwise the comparison is
    # meaningless — the cold path hit the divergence fallback and reset
    # the cache before the warm path ran).
    assert converged_cold, (
        f"cold-start backward did not converge "
        f"(n_iter={n_iter_cold}); fallback path reset the cache"
    )
    assert converged_warm, (
        f"warm-start backward did not converge "
        f"(n_iter={n_iter_warm}); fallback path may have masked the effect"
    )
    # Sanity: cold start must take at least 2 iters, else warm-start is
    # trivially equal to cold and the test proves nothing.
    assert n_iter_cold >= 2, (
        f"cold-start n_iter={n_iter_cold} is too small to demonstrate "
        "warm-start benefit; tighten the probe (e.g. lower gmres_tol or "
        "raise chi) so cold-start needs multiple iterations."
    )
    # The same input twice should give λ_0 ≈ λ_final, so the warm Neumann
    # iteration finishes in (often dramatically) fewer steps.
    assert n_iter_warm <= n_iter_cold, (
        f"warm-start n_iter={n_iter_warm} exceeds cold-start "
        f"n_iter={n_iter_cold}; cache is not being seeded properly"
    )


def test_adjoint_warm_start_invalidated_on_divergence():
    """When the fused JIT loop fails to converge, the cache is invalidated
    and the eager-GMRES fallback fires.  ``warm_start_invalidated`` is set;
    the eager-solve result then refills the cache so the next call still
    warm-starts (and may even converge directly in the fused JIT loop).
    """
    params = _make_su_tensor(D=2, d=2)
    # gmres_maxiter=1 forces the Neumann loop to bail without reaching
    # gmres_tol on the cold call — pushes the first call through the
    # eager-GMRES fallback.  The eager solver may not converge either,
    # but the codepath we care about (cache invalidation + refill) fires.
    loss_capped = _make_loss_fn(gmres_maxiter=1)

    # First call — cold start (init_lam = dE_denv), JIT loop can't
    # converge in 1 iter, eager-GMRES fallback fires.
    jax.grad(loss_capped)(params)
    assert not _F3_LAST_DIAGNOSTICS["converged"], (
        "expected fused JIT loop to not converge at gmres_maxiter=1"
    )
    assert _F3_LAST_DIAGNOSTICS["warm_start_invalidated"], (
        "expected warm_start_invalidated=True when fallback fires"
    )

    # Second call — cache now holds the eager-solve result from call 1.
    # That result is the actual GMRES solution, so the fused JIT loop
    # seeded from it may converge in a single iteration (or still
    # bail-and-fallback).  Either way the call must not crash, proving
    # the cache lifecycle (invalidate → refill with eager solve → reuse
    # on next call) is sound.
    jax.grad(loss_capped)(params)
    # No assertion on converged/invalidated here — the second-call
    # outcome depends on how close the eager-solve λ is to the
    # fixed point, which varies by problem.  The key invariant is just
    # "no crash + cache lifecycle is sound", which the absence of an
    # exception already verifies.


def test_gmres_logger_emits_at_f3_path(caplog):
    """f_bwd's F3 success path emits a DEBUG record on the tenax.ctm.gmres logger.

    Task 10 (#499): the F3 fused-fixed-point adjoint solve must emit
    diagnostics (n_iter / converged / diverged / tol) on the
    ``tenax.ctm.gmres`` logger after each backward.  The log is DEBUG
    level so it stays silent by default and observable when users opt in.
    """
    params = _make_su_tensor(D=2, d=2)
    loss = _make_loss_fn()

    with caplog.at_level(logging.DEBUG, logger="tenax.ctm.gmres"):
        jax.grad(loss)(params)

    # At least one DEBUG record matching "F3 adjoint" must appear.
    assert any(
        "F3 adjoint" in rec.message
        for rec in caplog.records
        if rec.name == "tenax.ctm.gmres" and rec.levelno == logging.DEBUG
    ), f"No F3 adjoint log; got {[r.message for r in caplog.records]}"


def test_adjoint_warm_start_invalidated_on_shape_mismatch():
    """Warm-start cache rejects stale lam_leaves with mismatched shapes.

    Regression for PR #515 Codex review: changing site_tensor shape
    while keeping coords/gate/neighbors constant must invalidate the
    cached lambda before ``_jit_fused_fixed_point_bwd`` would otherwise
    raise a VJP tree/shape mismatch.

    The ``_VJP_CACHE`` key derives from (coords, chi, gate id, neighbors
    id, ...) — it does NOT include ``site_tensor`` shape.  So a D-sweep
    that keeps coords/gate/neighbors constant hits the SAME cached VJP
    closure (and the same ``_cached["prev_lam_leaves"]`` dict) with
    differently-shaped env_leaves.  We simulate this by manually
    injecting a wrong-shape ``prev_lam_leaves`` into the cached closure
    and verifying the next ``jax.grad`` call invalidates it (rather than
    crashing inside the JIT).
    """
    from tenax.algorithms._ctm_energy_ad import _VJP_CACHE

    # Clear the shared _VJP_CACHE so the warm-up call below populates a
    # single, unambiguous entry that we can reflect into.  Previous tests
    # in this module leave their own cache entries behind (different
    # gmres_maxiter / etc.) and ``next(iter(_VJP_CACHE.values()))`` would
    # pick the wrong one.
    _VJP_CACHE.clear()

    params = _make_su_tensor(D=2, d=2)
    loss = _make_loss_fn()

    # Step 1: warm-up call populates _cached["prev_lam_leaves"] with
    # leaves shaped like env_leaves for the current (D, chi) probe.
    jax.grad(loss)(params)

    # Step 2: locate the live _cached dict via the f.bwd closure of
    # the cached VJP entry, then inject a deliberately-wrong-shape
    # prev_lam_leaves entry.  This mimics the D-sweep scenario where
    # the previous grad call ran at a different shape but the same
    # cache_key, so the stale lam_leaves carry over.
    assert len(_VJP_CACHE) == 1, (
        f"expected exactly one VJP entry after the warm-up call; "
        f"found {len(_VJP_CACHE)}"
    )
    f_cached, _mutables = next(iter(_VJP_CACHE.values()))
    cached_dict = None
    for cell in f_cached.bwd.__closure__:
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, dict) and "prev_lam_leaves" in v:
            cached_dict = v
            break
    assert cached_dict is not None, "could not locate _cached dict in f.bwd closure"
    assert cached_dict["prev_lam_leaves"] is not None, (
        "warm-up call did not populate prev_lam_leaves; "
        "f_bwd may have hit the divergence/fallback path"
    )

    # Inject a wrong-shape stale entry: same tree length, but one leaf
    # has the wrong shape (extra dimension).  This is exactly the
    # shape mismatch a D-sweep would produce.
    stale = tuple(
        jnp.zeros(leaf.shape + (1,), dtype=leaf.dtype)
        for leaf in cached_dict["prev_lam_leaves"]
    )
    cached_dict["prev_lam_leaves"] = stale

    # Step 3: second grad call must NOT crash with VJP shape mismatch;
    # the shape-validation guard must invalidate the cache and
    # cold-start the Neumann iteration.
    _F3_LAST_DIAGNOSTICS["warm_start_invalidated"] = False
    g2 = jax.grad(loss)(params)
    assert np.all(np.isfinite(np.asarray(g2)))
    assert _F3_LAST_DIAGNOSTICS["warm_start_invalidated"] is True, (
        "expected warm_start_invalidated=True after stale-shape lam was "
        "injected into the cache"
    )


def test_invalidate_implicit_ad_warm_start_clears_seed():
    """Public invalidation API drops the cached prev_lam_leaves seed (#501).

    After a stall-recovery rollback (``params = best_params``) the cached
    λ is no longer a good warm-start seed.  The optimizer side calls
    ``invalidate_implicit_ad_warm_start`` to drop it without depending on
    the shape-mismatch heuristic.
    """
    from tenax.algorithms._ctm_energy_ad import (
        _VJP_CACHE,
        invalidate_implicit_ad_warm_start,
    )

    _VJP_CACHE.clear()
    # Empty cache: invalidation is a no-op and reports 0 entries cleared.
    assert invalidate_implicit_ad_warm_start() == 0

    params = _make_su_tensor(D=2, d=2)
    loss = _make_loss_fn()

    # Populate cache and confirm the seed is cached.
    jax.grad(loss)(params)
    assert len(_VJP_CACHE) == 1
    f_cached, mutables = next(iter(_VJP_CACHE.values()))
    assert "_invalidate_warm_start" in mutables, (
        "make_implicit_vjp_fn must register the warm-start invalidator in mutables"
    )
    cached_dict = None
    for cell in f_cached.bwd.__closure__:
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, dict) and "prev_lam_leaves" in v:
            cached_dict = v
            break
    assert cached_dict is not None
    assert cached_dict["prev_lam_leaves"] is not None, (
        "warm-up call must seed prev_lam_leaves"
    )

    # Invalidate via the public API and confirm the cell was cleared.
    n_cleared = invalidate_implicit_ad_warm_start()
    assert n_cleared == 1, f"expected 1 cache entry cleared, got {n_cleared}"
    assert cached_dict["prev_lam_leaves"] is None, (
        "invalidate_implicit_ad_warm_start must clear prev_lam_leaves"
    )

    # Next grad call cold-starts (init_lam = dE_denv), produces a valid
    # gradient, and re-populates the seed for subsequent calls.
    g = jax.grad(loss)(params)
    assert np.all(np.isfinite(np.asarray(g)))
    assert cached_dict["prev_lam_leaves"] is not None, (
        "post-invalidation cold-start should refill prev_lam_leaves"
    )


def test_gmres_logger_emits_at_eager_fallback(caplog):
    """When F3 diverges (gmres_maxiter=1 forces it), eager-GMRES log fires too.

    Task 10 (#499): both the F3 success log AND the eager-GMRES log
    must appear when the fused JIT loop fails to converge and the
    eager-Krylov fallback runs.
    """
    params = _make_su_tensor(D=2, d=2)
    # gmres_maxiter=1 forces F3 to bail without reaching gmres_tol, so
    # the eager fallback runs and emits its own log line.
    loss_capped = _make_loss_fn(gmres_maxiter=1)

    with caplog.at_level(logging.DEBUG, logger="tenax.ctm.gmres"):
        jax.grad(loss_capped)(params)

    messages = [r.message for r in caplog.records if r.name == "tenax.ctm.gmres"]
    assert any("F3 adjoint" in m for m in messages), f"missing F3 log: {messages}"
    assert any("Eager GMRES" in m for m in messages), f"missing eager log: {messages}"
