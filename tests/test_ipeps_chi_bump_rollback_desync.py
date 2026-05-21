"""Regression test for env_cache + stall-rollback desync (#518).

When ``ctmrg_heuristic_increase_chi=True`` (in-CTM χ-bump enabled) is
combined with ``gs_stall_recovery="reset"``, the optimizer's rollback
to ``best_params`` also restored ``_env_cache`` from a ``best_env_cache``
snapshot that may have been taken at a pre-bump (smaller) χ.  The next
CTM forward then fed a smaller-χ env into the post-bump ``ctm_cfg.chi``
and crashed inside the projector contraction with::

    ValueError: Size of label 'd' for operand 1 (N) does not match
    previous terms (M).

The fix (Option A from the issue) is to clear ``_env_cache`` on
rollback rather than restore it from the chi-stale snapshot; the next
CTM call starts cold at the current ``ctm_cfg.chi``.  The test
reproduces by letting line-search succeed once (so ``best_env_cache``
gets snapshotted), then forcing failures (which trigger the rollback).
``chi_auto_bump_eps=1e-30`` guarantees the reactive bump fires
unconditionally after each successful step, so the snapshot is
guaranteed to be at a pre-bump (now-stale) χ when the rollback runs.

Marker: ``core``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._line_search as _ls_mod
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def _heisenberg_gate(d):
    """S=1/2 two-site Heisenberg gate ``(d, d, d, d)``."""
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    h = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real
    return h


def _random_2site_init(d, D, seed):
    """Random complex (A, B) tuple matching the 2-site AD init convention."""
    rng = np.random.default_rng(seed)
    A = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    B = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    return (A / jnp.linalg.norm(A), B / jnp.linalg.norm(B))


@pytest.mark.core
def test_rollback_after_chi_bump_does_not_crash_2site(monkeypatch):
    """#518: rollback must not feed a stale-χ env into a post-bump ctm_cfg.

    Sequence with the bug present:

    1. Step 1 line search succeeds (mock) → ``best_env_cache`` snapshot
       at the initial χ.
    2. End-of-step reactive bump (``chi_auto_bump_eps=1e-30``) raises
       ``ctm_cfg.chi`` by ``chi_auto_bump_step``.  ``_env_cache`` is
       padded to the new χ but ``best_env_cache`` is NOT.
    3. Step 2 line search fails (mock) → reset block fires.
       Pre-fix: ``_env_cache.update(best_env_cache)`` writes
       pre-bump-χ envs into the live cache.
    4. Step 3 value_and_grad warm-starts CTM from ``_env_cache`` at the
       stale χ against ``ctm_cfg.chi`` at the post-bump χ → contraction
       crashes with ``Size of label 'd' ... does not match``.

    Fix: rollback clears ``_env_cache`` instead of restoring stale
    state; the next CTM call cold-starts at the post-bump χ.

    The test asserts the run completes without raising.
    """
    calls = {"n": 0}

    def _succeed_once_then_fail(_phi, _dphi, phi0, _slope, **_kwargs):
        """First call: report improvement so step is accepted as best.

        Subsequent calls: report ``f_alpha == phi0`` so the optimizer
        increments ``stall_count`` and enters the reset block on the
        very next iteration.
        """
        calls["n"] += 1
        if calls["n"] == 1:
            # f_alpha strictly below phi0 triggers ``stall_count = 0``
            # and the line-search "success" branch (params updated).
            return 1e-3, phi0 - 1e-3, True
        return 0.0, phi0, False

    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _succeed_once_then_fail)

    d = 2
    gate = _heisenberg_gate(d)
    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4,
            chi_max=8,
            chi_auto_bump=True,
            # 1e-30 is well below any realistic eps_T → bump fires every
            # step.  This guarantees a chi-stale ``best_env_cache``.
            chi_auto_bump_eps=1e-30,
            chi_auto_bump_step=2,
        ),
        gs_num_steps=6,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=2,
        gs_line_search_method="hager_zhang",
        su_init=False,
        # ``grad_norm`` keeps the dE underflow from short-circuiting the
        # outer loop before the rollback fires.
        gs_conv_criterion="grad_norm",
    )
    A_init = _random_2site_init(d, D=2, seed=0)

    # The non-C4v 2-site path emits a UserWarning about variational
    # caveats; not relevant to this regression.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Pre-fix: this raises ValueError("Size of label 'd' ...")
        # inside the post-rollback CTM step.  Post-fix: runs to
        # completion (or to the gs_stall_recovery_retries cap).
        _opt.optimize_gs_ad(gate, A_init, cfg)

    # Sanity probe: the monkeypatch must reach both the success branch
    # (call #1) and the failure branch (call >=2) for the regression
    # path to be exercised.  Without these calls the bug scenario
    # never sets up.
    assert calls["n"] >= 2, (
        f"line-search mock invoked {calls['n']} times; expected >=2 so "
        f"both the snapshot-then-stale-bump sequence and the rollback "
        f"fire.  Monkeypatch target may be stale."
    )
