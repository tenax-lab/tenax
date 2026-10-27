"""Regression test for env_cache + stall-rollback desync (#518).

When an outer χ bump (``chi_auto_bump`` / ``gs_chi_schedule_steps``) is
combined with a rollback to ``best_params``, the optimizer used to restore
``_env_cache`` from a ``best_env_cache`` snapshot taken at the pre-bump
(smaller) χ.  On the 2x2 forward of the time the next CTM call then crashed
inside the projector contraction with::

    ValueError: Size of label 'd' for operand 1 (N) does not match
    previous terms (M).

The fix (#519) clears ``_env_cache`` on rollback instead.

What this pins now (#1150).  Re-introducing the pre-#519 restore no longer
crashes: the forward accepts a smaller-χ seed and grows it to ``ctm_cfg.chi``,
so a stale snapshot is just a worse warm start, and the fix and the bug are
indistinguishable from outside.  The behaviour that still matters is that a
rollback which restores a pre-bump snapshot completes and leaves the run at
the bumped χ.  The original test reached its rollbacks through line-search
stalls, after the step's own evaluation had already re-snapshotted the best
env at the bumped χ (#1095), so it never restored a stale snapshot at all.
This version reaches the regime through the gradient-spike rollback, which
runs before the re-snapshot, and asserts that it did.

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
def test_a_rollback_right_after_a_chi_bump_completes_at_the_bumped_chi(monkeypatch):
    """#518/#1150: a rollback that restores a pre-bump snapshot still completes.

    1. Step 1 is accepted: ``best_env_cache`` is snapshotted at chi=4.
    2. The end-of-step reactive bump (``chi_auto_bump_eps=1e-30``) raises
       ``ctm_cfg.chi`` to 6.
    3. Step 2's gradient norm is reported as a spike, so the guard rolls
       back to ``best_params`` *before* that step re-snapshots the best env:
       the snapshot it could restore is still at chi=4.
    4. The run must carry on and finish at the bumped chi.
    """

    def _always_succeed(_phi, _dphi, phi0, _slope, **_kwargs):
        return 1e-3, phi0 - 1e-3, True

    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_succeed)

    real_norm = _opt._grad_l2_norm
    norms = []

    def _spike_after_the_first(grads):
        v = float(real_norm(grads))
        norms.append(v)
        return v if len(norms) == 1 else 1e6 * (1.0 + v)

    monkeypatch.setattr(_opt, "_grad_l2_norm", _spike_after_the_first)

    real_bump = _opt._apply_chi_bump
    events = []

    def _record_bump(ctm_cfg, env_cache, chi_new, **kw):
        events.append(("bump", ctm_cfg.chi, chi_new, len(norms)))
        return real_bump(ctm_cfg, env_cache, chi_new, **kw)

    monkeypatch.setattr(_opt, "_apply_chi_bump", _record_bump)

    d = 2
    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4,
            chi_max=8,
            chi_auto_bump=True,
            # 1e-30 is below any realistic eps_T: the bump fires every step.
            chi_auto_bump_eps=1e-30,
            chi_auto_bump_step=2,
        ),
        gs_num_steps=4,
        gs_line_search_method="hager_zhang",
        gs_grad_spike_ratio=2.0,
        gs_grad_spike_window=1,
        su_init=False,
        # ``grad_norm`` keeps the dE underflow from ending the loop early.
        gs_conv_criterion="grad_norm",
    )
    with warnings.catch_warnings():
        # chi_auto_bump's deprecation and the non-C4v 2-site caveat.
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        _, envs, _ = _opt.optimize_gs_ad(
            _heisenberg_gate(d), _random_2site_init(d, D=2, seed=0), cfg
        )

    # Regime: a bump from 4 fired after step 1's gradient and before step
    # 2's, so step 2's spike rollback faced a chi=4 snapshot at chi=6.  If a
    # change moves the bump or the guard, this says so instead of passing
    # vacuously (#1150).
    assert len(norms) >= 2, norms
    assert ("bump", 4, 6, 1) in events, events
    # Behaviour: the run finished at the last bumped chi.
    env_a, _ = envs
    assert env_a.C1.todense().shape[0] == events[-1][2], events
