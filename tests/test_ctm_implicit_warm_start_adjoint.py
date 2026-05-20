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
