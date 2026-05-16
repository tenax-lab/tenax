"""Behavioural cap test for ``gs_stall_recovery='reset'`` (issue #454).

Forces every line search to fail; verifies the optimizer exits within
``gs_stall_recovery_retries`` instead of burning the full
``gs_num_steps`` budget.  This replaces a prior stdout-parsing version
(deleted in PR #488 along with similarly fragile checks) — codex P2 on
PR #488 flagged that deletion as a coverage gap: the cap mechanism is
real production behaviour and the line-search-failure path needs
required-CI coverage that doesn't depend on log-string layout.

The test asserts on observable mock state (the patched line search's
invocation count) rather than on captured stdout.  If the cap is
removed at the 2-site or C4v reset site, the mock would be called
``gs_num_steps`` times; with the cap intact, the loop breaks after
``gs_stall_recovery_retries + 1`` consecutive failures.
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


def _random_1site_init(d, D, seed):
    """Random complex single-tensor init."""
    rng = np.random.default_rng(seed)
    A = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    return A / jnp.linalg.norm(A)


@pytest.mark.core
def test_reset_loop_exits_within_retry_cap_2site(monkeypatch):
    """Force every line search to fail; assert the 2-site optimizer exits
    within ``gs_stall_recovery_retries`` rather than running for the full
    ``gs_num_steps`` budget.

    The 2-site optimizer (``_optimize_gs_ad_tensor_2site``) imports
    ``hager_zhang_line_search`` from ``tenax.algorithms._line_search``
    inside its loop body, so we must monkeypatch the *source* module
    rather than the optimizer module.
    """
    calls = {"n": 0}

    def _always_fail(_phi, _dphi, phi0, _slope, **_kwargs):
        # ``f_alpha == phi0`` (no improvement) → triggers stall_count += 1
        # in the 2-site optimizer.
        calls["n"] += 1
        return 0.0, phi0, False

    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_fail)

    d = 2
    gate = _heisenberg_gate(d)
    gs_num_steps = 20
    gs_stall_recovery_retries = 3
    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=gs_num_steps,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=gs_stall_recovery_retries,
        su_init=False,
        # ``grad_norm`` keeps the dE underflow that naturally follows a forced-
        # fail line search from short-circuiting the outer loop before the cap
        # fires.  The forced-fail monkeypatch keeps params fixed, so the
        # grad-norm stays well above ``gs_grad_norm_tol`` throughout.
        gs_conv_criterion="grad_norm",
    )
    A_init = _random_2site_init(d, D=2, seed=0)

    # The non-C4v 2-site path emits a UserWarning about variational/perf
    # caveats; not relevant to the cap mechanism.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _opt.optimize_gs_ad(gate, A_init, cfg)

    # Sanity probe: catches the silent-no-op case where a future refactor
    # moves the line-search import out of the loop body and the patch never
    # reaches the production call site.
    assert calls["n"] > 0, (
        "patched line search was never called; monkeypatch target stale"
    )
    # Behavioural cap: without the cap, the optimizer would call line search
    # once per step → ``calls['n'] == gs_num_steps``.  With the cap, the outer
    # loop breaks after ``gs_stall_recovery_retries + 1`` consecutive
    # failures, so the count is bounded strictly below ``gs_num_steps``.
    assert calls["n"] < gs_num_steps, (
        f"expected the optimizer to exit early via the retry cap "
        f"(< {gs_num_steps} line-search calls); got {calls['n']}.  "
        f"The gs_stall_recovery_retries cap is bypassed at the 2-site reset "
        f"site (#454)."
    )


@pytest.mark.core
def test_reset_loop_exits_within_retry_cap_c4v(monkeypatch):
    """Same as the 2-site test, but for the 1-site C4v path."""
    calls = {"n": 0}

    def _always_fail(_phi, _dphi, phi0, _slope, **_kwargs):
        calls["n"] += 1
        return 0.0, phi0, False

    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_fail)

    d = 2
    gate = _heisenberg_gate(d)
    gs_num_steps = 20
    gs_stall_recovery_retries = 3
    cfg = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=gs_num_steps,
        gs_stall_recovery="reset",  # explicit; default for 1x1 is ``"noise"``
        gs_stall_recovery_retries=gs_stall_recovery_retries,
        su_init=False,
        gs_c4v=True,
        gs_conv_criterion="grad_norm",
    )
    A_init = _random_1site_init(d, D=2, seed=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _opt.optimize_gs_ad(gate, A_init, cfg)

    assert calls["n"] > 0, (
        "patched line search was never called; monkeypatch target stale"
    )
    assert calls["n"] < gs_num_steps, (
        f"expected the C4v optimizer to exit early via the retry cap "
        f"(< {gs_num_steps} line-search calls); got {calls['n']}.  "
        f"The gs_stall_recovery_retries cap is bypassed at the C4v reset "
        f"site (#454)."
    )
