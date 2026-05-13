"""Cap + rollback for gs_stall_recovery='reset' (issue #454).

The optimizer must exit cleanly after gs_stall_recovery_retries
consecutive resets and return best_params.

This is a *failing* test committed as the red half of TDD: Task 3 of
the #454 plan. Task 4 implements the cap+rollback at the 2-site reset
site and turns this green.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._line_search as _ls_mod
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.core
def test_reset_loop_exits_after_retry_cap(monkeypatch, capsys):
    """Force every line search to fail; assert the loop exits at retries+1 resets.

    The 2-site optimizer (``_optimize_gs_ad_tensor_2site``) imports
    ``hager_zhang_line_search`` from ``tenax.algorithms._line_search``
    inside its loop body, so we must monkeypatch the *source* module
    rather than the optimizer module.
    """

    calls = {"n": 0}

    def _always_fail(_phi, _dphi, phi0, _slope, **_kwargs):
        # f_alpha == phi0 means "no improvement" -> triggers stall_count += 1
        # in the 2-site optimizer.
        calls["n"] += 1
        return 0.0, phi0, False

    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _always_fail)

    d = 2
    gate = _heisenberg_gate(d)
    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        # Keep step budget tight: once the cap lands (Task 4), the loop exits
        # after retries+1 stalls. Pre-cap the loop runs all gs_num_steps, so
        # this is also the upper bound on red-test runtime.
        gs_num_steps=10,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=3,
        gs_verbose=True,
        su_init=False,
        # Use grad_norm convergence so the dE underflow that naturally
        # follows a forced-fail line search doesn't short-circuit the
        # outer loop before the cap can fire. The forced-fail
        # monkeypatch keeps params fixed, so ||grad|| stays well above
        # gs_grad_norm_tol throughout.
        gs_conv_criterion="grad_norm",
    )
    A_init = _random_2site_init(d, D=2, seed=0)

    # The non-C4v 2-site path emits a UserWarning about variational/perf
    # caveats (see ipeps_optimize.py around L1716-1724); not relevant here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _opt.optimize_gs_ad(gate, A_init, cfg)

    out = capsys.readouterr().out
    # Sanity probe: catch the silent-no-op case where a future refactor
    # moves the line-search import out of the loop body and the patch
    # never reaches the production call site.
    assert calls["n"] > 0, (
        "patched line search was never called; monkeypatch target stale"
    )
    assert "stall budget exhausted" in out, (
        f"missing exhaustion log; last 2000 chars of stdout:\n{out[-2000:]}"
    )
    stall_lines = [
        ln for ln in out.splitlines() if "stall #" in ln and "reset L-BFGS" in ln
    ]
    assert len(stall_lines) == cfg.gs_stall_recovery_retries, (
        f"expected {cfg.gs_stall_recovery_retries} reset events, "
        f"got {len(stall_lines)}: {stall_lines}"
    )


def _heisenberg_gate(d):
    """S=1/2 two-site Heisenberg gate (d,d,d,d)."""
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
    """Random complex (A, B) tuple matching the 2-site AD init convention.

    The 2-site implicit-AD path is complex internally; a real init would
    be promoted on entry.
    """
    rng = np.random.default_rng(seed)
    A = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    B = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    A = A / jnp.linalg.norm(A)
    B = B / jnp.linalg.norm(B)
    return (A, B)
