"""#1095: ``best_energy`` must track a certified re-measurement of ``best_params``.

``best_energy`` used to be monotone: it only moved when a strictly lower
candidate arrived.  A single in-loop forward that read low (e.g. an
under-converged CTM environment) therefore stuck as the best for the rest of
the run, even after a rollback re-solved ``best_params`` and measured it
honestly higher -- ``_should_accept_best`` rejected the honest value because it
was not a decrease, and every later candidate was ranked against the fake one.

The fix: when the energy being scored was measured at ``best_params`` itself
(``params is best_params``, which is what every rollback produces) AND that
forward is certified (site-1 verdict, else the env refresh is not positively
unconverged), the new finite value replaces ``best_energy`` even if it is
higher, and the 1-site / 2-site paths force a best-checkpoint write so the
on-disk best agrees.  An uncertified high re-measure leaves the best alone.

Each integration test injects one in-loop energy ``_SHIFT`` too low, forces a
rollback right after it became best, and checks what the re-solve at the
rolled-back params does.  Before the fix the re-measured value (-0.48504 on
the 2-site fixture) was rejected and best stayed at the fake -0.49004.
The ``_should_accept_best`` / ``_remeasure_certified`` contracts are unit
tested in the core-gated ``test_ctm_unconverged_policy.py``.
"""

from __future__ import annotations

import sys
import types
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tenax.algorithms._checkpoint as _ckpt
import tenax.algorithms._ctm_python_loop as _cpl
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms._ad_primitives import CTMRGGradientError
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.core.lattice import checkerboard

_SHIFT = 5e-3


def _gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    g = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    )
    return g.real


def _rand(seed, D=2, d=2):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    return jnp.asarray(a / np.linalg.norm(a))


def _cfg(unit_cell, steps, policy="raise", **kw):
    return iPEPSConfig(
        unit_cell=unit_cell,
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4, max_iter=300, min_iter=1, conv_tol=1e-10, on_unconverged=policy
        ),
        gs_num_steps=steps,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=2,
        su_init=False,
        gs_conv_criterion="grad_norm",
        gs_conv_tol=0.0,
        return_history=True,
        **kw,
    )


def _record_accept_calls(monkeypatch):
    """Record (candidate, current_best, accepted, params_is_best) per call."""
    rec = []
    real = _opt._should_accept_best

    def wrapped(**kw):
        f = sys._getframe(1).f_locals
        ok = real(**kw)
        rec.append(
            (kw["candidate"], kw["current_best"], ok, f["params"] is f["best_params"])
        )
        return ok

    monkeypatch.setattr(_opt, "_should_accept_best", wrapped)
    return rec


def _record_best_saves(monkeypatch):
    saves = []
    real = _ckpt.save_checkpoint

    def wrapped(state, path, *a, is_best=False, **kw):
        if is_best:
            saves.append((state["step"], state["best_energy"]))
        return real(state, path, *a, is_best=is_best, **kw)

    monkeypatch.setattr(_ckpt, "save_checkpoint", wrapped)
    return saves


def _install_fake_low_then_spike(monkeypatch):
    """2nd forward reads _SHIFT low (becomes best); 3rd gradient spikes."""
    n = {"vg": 0, "spiked": False}
    real_vg = _opt._site1_value_and_grad
    real_gn = _opt._grad_l2_norm

    def vg(*a, **k):
        n["vg"] += 1
        e, g = real_vg(*a, **k)
        return (e - _SHIFT if n["vg"] == 2 else e), g

    def gn(g):
        v = real_gn(g)
        if n["vg"] == 3 and not n["spiked"]:
            n["spiked"] = True
            return 1e6
        return v

    monkeypatch.setattr(_opt, "_site1_value_and_grad", vg)
    monkeypatch.setattr(_opt, "_grad_l2_norm", gn)
    return n


def _find_rising_remeasure(rec):
    # A stalled line search can also leave params is best_params with an
    # unchanged energy; the rollback re-measure is the one that moved.
    remeasures = [
        i
        for i, r in enumerate(rec)
        if r[3] and np.isfinite(r[1]) and r[0] > r[1] + 0.5 * _SHIFT
    ]
    assert remeasures, f"fixture never re-measured best_params higher: {rec}"
    i = remeasures[0]
    assert i + 1 < len(rec), rec
    return i


def _assert_remeasure_overwrites(rec):
    i = _find_rising_remeasure(rec)
    cand = rec[i][0]
    # The next scoring call must rank against the honest re-measurement.
    assert rec[i + 1][1] == cand, (
        f"best_energy kept the fake {rec[i + 1][1]:.10f} instead of the "
        f"re-measured {cand:.10f}"
    )
    return i, cand


# --------------------------------------------------------- integration level


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_spike_rollback_remeasure_overwrites_fake_best(
    monkeypatch, tmp_path, unit_cell
):
    _install_fake_low_then_spike(monkeypatch)
    rec = _record_accept_calls(monkeypatch)
    saves = _record_best_saves(monkeypatch)
    # 1x1 starts from the built-in random init: the dense _rand(0) start
    # stalls its first line search and is ~2x slower to optimise.
    A0 = (_rand(0), _rand(1)) if unit_cell == "2site" else None
    cfg = _cfg(
        unit_cell,
        steps=5,
        gs_grad_spike_ratio=2.0,
        gs_checkpoint_path=str(tmp_path),
        gs_c4v=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _opt.optimize_gs_ad(_gate(), A0, cfg)
    _i, cand = _assert_remeasure_overwrites(rec)
    # The on-disk best checkpoint must follow the re-measurement too, even
    # though it went up (the old write trigger fired only on a decrease).
    assert any(abs(e - cand) == 0.0 for _s, e in saves), (cand, saves)


def _run_multisite(monkeypatch, *, unconverged_remeasure=False):
    """Multisite has no spike guard; a CTMRGGradientError reset rolls back.

    ``value_and_grad`` call 2 reads ``_SHIFT`` low, call 3 raises (reset to
    best_params), call 4 is the re-measure.  ``unconverged_remeasure`` marks
    the env refresh that follows call 4 unconverged.
    """
    rec = _record_accept_calls(monkeypatch)
    n = {"k": 0, "forced": False}
    real_vag = jax.value_and_grad

    def vag(fn, *a, **k):
        inner = real_vag(fn, *a, **k)

        def call(*x, **y):
            n["k"] += 1
            if n["k"] == 3:
                raise CTMRGGradientError(spectral_radius=1.5)
            e, g = inner(*x, **y)
            return (e - _SHIFT if n["k"] == 2 else e), g

        return call

    # Patch value_and_grad only as ipeps_optimize sees it.
    proxy = types.SimpleNamespace(
        **{k: getattr(jax, k) for k in dir(jax) if not k.startswith("__")}
    )
    proxy.value_and_grad = vag
    monkeypatch.setattr(_opt, "jax", proxy)
    if unconverged_remeasure:
        real_loop = _cpl.python_loop_ctm_converge

        def loop(*a, **k):
            envs, info = real_loop(*a, **k)
            if n["k"] == 4 and not n["forced"]:
                n["forced"] = True  # the refresh right after the re-measure
                info = info._replace(converged=False)
            return envs, info

        monkeypatch.setattr(_cpl, "python_loop_ctm_converge", loop)
    cfg = _cfg(
        checkerboard(), steps=6, policy="warn" if unconverged_remeasure else "raise"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _opt.optimize_gs_ad(_gate(), None, cfg)
    if unconverged_remeasure:
        assert n["forced"], "the re-measure refresh was never reached"
    return rec


def test_multisite_reset_remeasure_overwrites_fake_best(monkeypatch):
    _assert_remeasure_overwrites(_run_multisite(monkeypatch))


def _assert_uncertified_remeasure_kept_best(rec):
    i = _find_rising_remeasure(rec)
    cand, cur, ok, _ = rec[i]
    assert not ok, f"uncertified re-measure {cand:.10f} raised the best {cur:.10f}"
    assert rec[i + 1][1] == cur, (rec[i + 1][1], cur)


def test_uncertified_rising_remeasure_keeps_best_2site(monkeypatch):
    """on_unconverged="warn": an unconverged high re-measure must not raise best.

    Raising the bar on an uncertified (typically cold) re-measure would let a
    worse state become best and lose the original best_params.  Site-1
    verdict path: the re-measure forward is reported unconverged.
    """
    n = _install_fake_low_then_spike(monkeypatch)
    rec = _record_accept_calls(monkeypatch)
    real_info = _opt._site1_forward_info

    def info(*a, **k):
        ok = real_info(*a, **k)
        return False if n["vg"] == 4 else ok

    monkeypatch.setattr(_opt, "_site1_forward_info", info)
    cfg = _cfg("2site", steps=5, policy="warn", gs_grad_spike_ratio=2.0, gs_c4v=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _opt.optimize_gs_ad(_gate(), (_rand(0), _rand(1)), cfg)
    _assert_uncertified_remeasure_kept_best(rec)


def test_uncertified_rising_remeasure_keeps_best_multisite(monkeypatch):
    """Refresh-verdict path (no site-1 check): an unconverged refresh refuses."""
    _assert_uncertified_remeasure_kept_best(
        _run_multisite(monkeypatch, unconverged_remeasure=True)
    )
