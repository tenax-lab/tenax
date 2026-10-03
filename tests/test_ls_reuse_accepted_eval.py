"""The HZ ``dφ`` probe's evaluation at the accepted α is reused, not repeated.

Hager-Zhang accepts an α right after ``dφ(α)`` ran a full
``value_and_grad(loss_fn)`` there, and the next step's top-of-step
``value_and_grad`` used to recompute the same (energy, gradient) at the same
parameters — a whole CTM forward plus implicit-AD backward thrown away per
step.  ``_AcceptedProbeEval`` carries the probe's result into the next step.

These tests pin the MECHANISM, not convergence:

* real HZ on all three optimizers (1-site, 2-site, multisite): fresh
  top-of-step evaluations drop by exactly the number of reuses, and the
  energy trajectory matches a run with reuse disabled;
* a scripted HZ (one φ and one dφ probe per step, accept/reject on demand)
  on the 2-site optimizer: a reuse happens only after an accepted α whose
  dφ probe ran at exactly that α, and every event that changes the loss or
  the parameters in between — stall reset, noise recovery, χ bump, CTM
  schedule change — forces a fresh evaluation.  The explicit-AD loss
  depends on its warm-start seed, so it is never reused.

Evaluations are counted by spying ``jax.value_and_grad`` on functions named
``loss_fn`` (the optimizers' differentiated loss), so the count is the real
number of forward+backward passes, independent of the carry's bookkeeping.
"""

from __future__ import annotations

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

import tenax.algorithms._line_search as _ls_mod  # noqa: E402
import tenax.algorithms.ipeps_optimize as _opt  # noqa: E402
from tenax import CTMConfig, heisenberg_gate, iPEPSConfig, optimize_gs_ad  # noqa: E402
from tenax.algorithms.ipeps import ipeps  # noqa: E402

# Reuse vs no-reuse trajectories differ only in which CTM warm start the
# evaluation at the accepted point came from (the φ probe's env at the same α
# vs the pre-line-search env); both are the implicit-AD fixed point, so they
# agree to the CTM convergence tolerance.  Measured max |ΔE| over the
# trajectories below: <=4e-10 on CPU (phase gauge); 1e-7 leaves backend room.
_TRAJ_TOL = 1e-7


class _Counts:
    """Spy state for one optimizer run."""

    def __init__(self):
        self.loss_evals = 0  # value_and_grad(loss_fn) invocations
        self.probes = 0  # of which were HZ dφ probes
        self.takes = []  # per top-of-step: True if reused


@pytest.fixture
def spy(monkeypatch):
    """Install the counting spies; ``spy(reuse=...)`` returns fresh counts."""
    state = {"counts": None, "reuse": True}
    orig_vag = jax.value_and_grad

    def _vag(fn, *a, **kw):
        g = orig_vag(fn, *a, **kw)
        if getattr(fn, "__name__", "") != "loss_fn":
            return g

        def _counted(*args, **kwargs):
            state["counts"].loss_evals += 1
            return g(*args, **kwargs)

        return _counted

    monkeypatch.setattr(jax, "value_and_grad", _vag)

    base = _opt._AcceptedProbeEval

    class _Spied(base):
        def __init__(self, enabled):
            # reuse=False is the pristine pre-fix path: no probe recorded,
            # accept recomputes params, every top-of-step evaluates.
            super().__init__(enabled and state["reuse"])

        def probe(self, *a, **kw):
            state["counts"].probes += 1
            return super().probe(*a, **kw)

        def take(self, *a, **kw):
            out = super().take(*a, **kw)
            state["counts"].takes.append(out is not None)
            return out

    monkeypatch.setattr(_opt, "_AcceptedProbeEval", _Spied)

    def _new(*, reuse=True):
        state["counts"] = _Counts()
        state["reuse"] = reuse
        return state["counts"]

    return _new


@pytest.fixture(scope="module")
def su_2site():
    """Short 2-site simple update: a CTM-friendly D=2 start (random tensors
    can make the CTM oscillate)."""
    su_cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=200,
        dt=0.05,
        unit_cell="2site",
        ctm=CTMConfig(chi=4),
    )
    _, (A, B), _ = ipeps(heisenberg_gate(), None, su_cfg, compute_energy=False)
    return A, B


def _ad_config(unit_cell, *, num_steps, **over):
    ctm_kw = dict(chi=6, max_iter=60, conv_tol=1e-10, forward_gauge="phase")
    ctm_kw.update(over.pop("ctm", {}))
    kw = dict(
        max_bond_dim=2,
        unit_cell=unit_cell,
        ctm=CTMConfig(**ctm_kw),
        gs_implicit_ad=True,
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_num_steps=num_steps,
        gs_conv_tol=1e-15,  # run the full step budget
        gs_grad_norm_tol=1e-15,
        su_init=False,
        return_history=True,
    )
    kw.update(over)
    return iPEPSConfig(**kw)


def _run(gate, A_init, cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return optimize_gs_ad(gate, A_init, cfg)


def _fresh_tops(c: _Counts) -> int:
    return c.loss_evals - c.probes


# ---------------------------------------------------------------------------
# Real Hager-Zhang: one representative per optimizer variant
# ---------------------------------------------------------------------------


def _variant(name, su_2site):
    from tenax.core.lattice import checkerboard

    A, B = su_2site
    gate = heisenberg_gate()
    if name == "2site":
        return gate, (A, B), _ad_config("2site", num_steps=4, gs_c4v=False)
    if name == "multisite":
        return gate, {"a": A, "b": B}, _ad_config(checkerboard(), num_steps=4)
    # 1-site: C4v-symmetrised A on the sublattice-rotated gate (the
    # production 1x1 Heisenberg setup; see test_optimizer_run_independence_973).
    from tenax import sublattice_rotate_gate

    return (
        sublattice_rotate_gate(gate),
        A,
        _ad_config("1x1", num_steps=4, gs_c4v=True, gs_stall_recovery="reset"),
    )


@pytest.mark.parametrize("variant", ["1site", "2site", "multisite"])
def test_real_hz_reuses_accepted_probe_and_matches_no_reuse(spy, su_2site, variant):
    gate, A_init, cfg = _variant(variant, su_2site)

    off = spy(reuse=False)
    out_off = _run(gate, A_init, cfg)
    on = spy(reuse=True)
    out_on = _run(gate, A_init, cfg)

    # Regime: the pre-fix path evaluates fresh at every step, and the line
    # search really ran dφ probes (otherwise there is nothing to reuse).
    assert not any(off.takes)
    assert _fresh_tops(off) == len(off.takes) == cfg.gs_num_steps
    assert on.probes > 0

    hits = sum(on.takes)
    assert hits >= 1, f"no accepted-probe reuse on {variant}: takes={on.takes}"
    # The spy counts real value_and_grad(loss_fn) calls: every reuse is one
    # forward+backward that did not run.
    assert _fresh_tops(on) == len(on.takes) - hits
    assert not on.takes[0]  # nothing to carry into the first step

    E_off, E_on = out_off[3]["energies"], out_on[3]["energies"]
    assert len(E_off) == len(E_on) == cfg.gs_num_steps
    diff = max(abs(a - b) for a, b in zip(E_off, E_on))
    assert diff < _TRAJ_TOL, f"{variant}: reuse moved the trajectory by {diff:.2e}"
    assert abs(float(out_off[2]) - float(out_on[2])) < _TRAJ_TOL


# ---------------------------------------------------------------------------
# Scripted Hager-Zhang (2-site): accept / reject on demand
# ---------------------------------------------------------------------------


def _scripted_hz(monkeypatch, plan, *, absolute=False):
    """Replace HZ with ``plan[i]`` for the i-th line search.

    Each entry is ``(probe_alpha_factor, return_alpha_factor, accept)``:
    φ and dφ are evaluated at ``probe * alpha_init``, the returned α is
    ``ret * alpha_init`` (φ is evaluated there first if it differs), and
    ``accept`` decides the reported φ: below ``phi0`` (taken) or ``phi0``
    (no decrease, so the step counts as a stall).
    ``probe=None`` skips dφ entirely.  ``absolute=True`` takes the factors
    as α itself (the same α in every search, as when ``alpha_init`` caps at
    1.0 in production).
    """
    calls = {"n": 0}

    def _hz(phi, dphi, phi0, _slope, *, alpha_init, **_kw):
        probe, ret, accept = plan[min(calls["n"], len(plan) - 1)]
        calls["n"] += 1
        if absolute:
            alpha_init = 1.0
        if probe is not None:
            phi(probe * alpha_init)
            dphi(probe * alpha_init)
        if probe != ret:
            phi(ret * alpha_init)
        # The verdict is scripted, not measured: an accepted step reports a
        # decrease so the optimizer takes it even where φ happens to rise.
        return ret * alpha_init, (phi0 - 1e-3 if accept else phi0), True

    monkeypatch.setattr(_ls_mod, "hager_zhang_line_search", _hz)
    return calls


_ACCEPT = (0.5, 0.5, True)
_REJECT = (0.5, 0.5, False)


def _scripted_2site(su_2site, num_steps, **over):
    A, B = su_2site
    over.setdefault("gs_stall_recovery_retries", 50)
    cfg = _ad_config("2site", num_steps=num_steps, gs_c4v=False, **over)
    return heisenberg_gate(), (A, B), cfg


def test_scripted_accept_reuses_every_following_step(spy, su_2site, monkeypatch):
    _scripted_hz(monkeypatch, [_ACCEPT])
    c = spy()
    _run(*_scripted_2site(su_2site, 4))
    assert c.takes == [False, True, True, True]
    # One dφ per step plus only the FIRST top-of-step evaluation: 2 -> 1 per
    # step here (the real-HZ case with two dφ probes goes 3 -> 2).
    assert c.probes == 4 and _fresh_tops(c) == 1


@pytest.mark.parametrize("recovery", ["reset", "noise"])
def test_stall_recovery_forces_fresh_evaluation(spy, su_2site, monkeypatch, recovery):
    # accept, reject (stall -> rollback / noise kick), accept, accept
    _scripted_hz(monkeypatch, [_ACCEPT, _REJECT, _ACCEPT, _ACCEPT])
    c = spy()
    _run(*_scripted_2site(su_2site, 5, gs_stall_recovery=recovery))
    # step 3 follows the stalled line search: its params are best_params
    # (reset) or a noise-kicked point, never the rejected probe's trial.
    assert c.takes == [False, True, False, True, True]
    assert _fresh_tops(c) == 2


def test_dphi_at_a_different_alpha_is_not_reused(spy, su_2site, monkeypatch):
    # dφ at 0.5·α0, but HZ returns 0.25·α0 (φ only there): the probe is the
    # wrong point and must not be carried.
    _scripted_hz(monkeypatch, [(0.5, 0.25, True)])
    c = spy()
    _run(*_scripted_2site(su_2site, 3))
    assert c.takes == [False, False, False]
    assert _fresh_tops(c) == 3


def test_probe_from_an_earlier_line_search_is_not_reused(spy, su_2site, monkeypatch):
    # LS 1 rejects after dφ at α=0.01; LS 2 accepts the same α=0.01 having
    # run φ only.  The LS-1 probe matches α exactly but belongs to another
    # search (at the pre-noise params), so it must not be carried.
    _scripted_hz(
        monkeypatch,
        [(0.01, 0.01, False), (None, 0.01, True), (None, 0.01, True)],
        absolute=True,
    )
    c = spy()
    _run(*_scripted_2site(su_2site, 3, gs_stall_recovery="noise"))
    assert c.takes == [False, False, False]


def test_chi_bump_forces_fresh_evaluation(spy, su_2site, monkeypatch):
    _scripted_hz(monkeypatch, [_ACCEPT])
    c = spy()
    # eps=0 makes the reactive bump fire at the end of step 1 (chi 4 -> 6),
    # then chi_max stops it.
    gate, A_init, cfg = _scripted_2site(
        su_2site,
        4,
        ctm=dict(
            chi=4,
            chi_auto_bump=True,
            chi_auto_bump_eps=0.0,
            chi_auto_bump_step=2,
            chi_max=6,
        ),
    )
    _run(gate, A_init, cfg)
    assert c.takes == [False, False, True, True]


def test_ctm_schedule_change_forces_fresh_evaluation(spy, su_2site, monkeypatch):
    _scripted_hz(monkeypatch, [_ACCEPT])
    c = spy()
    # conv_tol tightens at step index 2 of 4 (fraction 0.5).
    gate, A_init, cfg = _scripted_2site(
        su_2site, 4, gs_ctm_conv_tol_schedule=[(0.0, 1e-8), (0.5, 1e-10)]
    )
    _run(gate, A_init, cfg)
    assert c.takes == [False, True, False, True]


def test_explicit_ad_is_never_reused(spy, su_2site, monkeypatch):
    # The explicit loss runs a fixed number of sweeps from the cached env, so
    # its value depends on the seed: the probe's (seeded at the trial point)
    # is not what the top-of-step call would compute.
    _scripted_hz(monkeypatch, [_ACCEPT])
    c = spy()
    gate, A_init, cfg = _scripted_2site(
        su_2site,
        3,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
    )
    _run(gate, A_init, cfg)
    assert c.probes == 3
    assert c.takes == [False, False, False]
    assert _fresh_tops(c) == 3


def test_reused_step_time_includes_the_probe_evaluation(spy, su_2site, monkeypatch):
    """``history['step_times']`` keeps meaning "gradient-evaluation wall at
    this step's params": a reused step is charged the probe's own time."""
    _scripted_hz(monkeypatch, [_ACCEPT])
    spy()
    orig_probe = _opt._AcceptedProbeEval.probe
    import time as _time

    def _slow_loss_probe(self, loss_fn, alpha, trial, cfg):
        def _slow(p):
            _time.sleep(0.2)
            return loss_fn(p)

        _slow.__name__ = "loss_fn"
        return orig_probe(self, _slow, alpha, trial, cfg)

    monkeypatch.setattr(_opt._AcceptedProbeEval, "probe", _slow_loss_probe)
    out = _run(*_scripted_2site(su_2site, 3))
    # steps 2 and 3 are reused; their step_times carry the >=0.2 s probe.
    assert len(out[3]["step_times"]) == 2
    assert min(out[3]["step_times"]) >= 0.2


def test_carry_requires_identical_params_and_cfg():
    """Unit check of the match rule (identity, not value)."""
    ev = _opt._AcceptedProbeEval(enabled=True)
    p0 = jnp.ones(3)
    d = jnp.ones(3)
    cfg = CTMConfig(chi=4)
    ev.start()
    trial = _opt._normalize_params(_opt._tree_add(p0, _opt._tree_scale(d, 0.5)))

    def loss_fn(x):
        return jnp.sum(x**2)

    ev.probe(loss_fn, 0.5, trial, cfg)
    new = ev.accept(0.5, p0, d)
    assert new is trial
    # equal-valued copy of params, or an equal-valued replacement cfg: no reuse
    assert ev.take(jnp.array(new), cfg) is None
    ev.probe(loss_fn, 0.5, trial, cfg)
    ev.accept(0.5, p0, d)
    assert ev.take(new, CTMConfig(chi=4)) is None
    ev.probe(loss_fn, 0.5, trial, cfg)
    ev.accept(0.5, p0, d)
    E, g, dt = ev.take(new, cfg)
    assert float(E) == float(jnp.sum(trial**2)) and dt >= 0.0
    assert ev.take(new, cfg) is None  # a carry is used at most once


def test_probe_timer_waits_for_the_evaluation(monkeypatch):
    """On async backends value_and_grad returns before the work is done; the
    recorded probe time must cover a sync on the probe's own outputs."""
    synced = []
    real_block = jax.block_until_ready

    def _spy_block(x):
        synced.append(x)
        return real_block(x)

    monkeypatch.setattr(_opt.jax, "block_until_ready", _spy_block)
    ev = _opt._AcceptedProbeEval(enabled=True)
    trial = jnp.arange(3.0)
    ev.start()
    grads = ev.probe(lambda x: jnp.sum(x**2), 0.5, trial, CTMConfig(chi=4))
    assert len(synced) == 1
    energy, synced_grads = synced[0]
    assert float(energy) == float(jnp.sum(trial**2))
    assert synced_grads is grads
