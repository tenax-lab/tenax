"""#1111: a solved flowing-projector adjoint on a non-stationary forward can
give a gradient orders of magnitude off finite differences (1497x on the
#1028 fixture, 86x at D=3 chi=12).  The backward now checks it with one extra
forward energy evaluation along the gradient's own descent direction:
r = dE_measured / dE_predicted must lie in [0.5, 2].  Measured: garbage flow
r <= 0.002, good flow 0.97-1.24, frozen 1.00-1.54."""

from __future__ import annotations

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import tenax.algorithms._ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import heisenberg_gate, heisenberg_u1sz_init_pair
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _heis():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = Sp.T
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


_H1 = _heis()
_G2 = heisenberg_gate().todense()


def _pair_params(D=2, seed=0):
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))
    return (jnp.asarray(A.todense()), jnp.asarray(B.todense()))


def _pair_energy(p, *, chi=8, max_iter=20, pb="flow", gmres_restart=30):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(p[0]), (1, 0): _wrap_as_dense_tensor(p[1])},
            CHECKERBOARD_NEIGHBORS,
            _G2,
            chi=chi,
            max_iter=max_iter,
            projector_method="svd",
            forward_gauge="bond_phase",
            conv_method="elementwise",
            projector_backward=pb,
            gmres_restart=gmres_restart,
        )
    )


def _single_params(seed=2026):
    A = jax.random.normal(jax.random.PRNGKey(seed), (2, 2, 2, 2, 2))
    return (A / jnp.linalg.norm(A),)


def _single_energy(p, *, chi=4, max_iter=20, conv_tol=1e-10, pb="flow"):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(p[0])},
            SINGLE_SITE_NEIGHBORS,
            _H1,
            chi=chi,
            max_iter=max_iter,
            conv_tol=conv_tol,
            forward_gauge="bond_phase",
            projector_backward=pb,
        )
    )


def _grad(energy, params, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = jax.grad(lambda p: energy(p, **kw))(params)
    return g, dict(get_last_implicit_ad_diagnostics())


def _all_latches():
    out = {}
    for key, entry in cea._VJP_CACHE.items():
        if not isinstance(entry, tuple):
            continue
        _f, mutables = entry
        for rkey, (token, state, _objs) in mutables.get("_runs", {}).items():
            out[(id(key), token)] = {
                k: v for k, v in state.items() if k.endswith("_warned")
            }
    return out


def test_slope_check_rejects_the_1028_flow_gradient():
    # The rejection depends on the dense wrapping of the #1028 pair: with raw
    # SymmetricTensor params the same fixture keeps flow (r = 0.746, FD errors
    # 35%/37% vs frozen's 13%/4%) -- the check catches gross errors only.
    #
    # Restart 60, not the default 30: the forward is not converged, so its
    # state depends on round-off, and on some states (the macos-26-arm64
    # runner's, |J^T| ~ 9e3) GMRES(30) stagnates by chance -- a 1e-13 change
    # of the right-hand side flips it between 9e-7 and 6e-4, in either solver
    # (#1153, diagnosed on #1156).  The flow solve then fails before the
    # slope check runs.  Restart 45 and 60 did not stall in 12 tries each.
    p = _pair_params()
    g, d = _grad(_pair_energy, p, gmres_restart=60)
    assert d["forward_stationarity_residual"] > d["forward_stationarity_threshold"], (
        "premise"
    )
    assert d["projector_backward_used"] == "frozen_fallback", d
    assert "slope check" in d["flow_fallback_reason"], d["flow_fallback_reason"]
    assert d["flow_slope_ratio"] < 0.5, d["flow_slope_ratio"]
    g_frozen, _ = _grad(_pair_energy, p, pb="auto", gmres_restart=60)
    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(g_frozen)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


def test_slope_check_keeps_a_good_flow_gradient():
    """The chi=4 limit cycle: non-stationary (0.15) but flow is the better
    gradient (FD error 1.4% / 0.5% against frozen's 6% / 19%), r = 1.002."""
    _, d = _grad(_single_energy, _single_params())
    assert d["forward_stationarity_residual"] > d["forward_stationarity_threshold"], (
        "premise"
    )
    assert d["projector_backward_used"] == "flow", d
    assert 0.5 <= d["flow_slope_ratio"] <= 2.0, d["flow_slope_ratio"]


def test_stationary_forward_skips_the_check(monkeypatch):
    # A checked call first, so a stale flow_slope_ratio would be visible.
    _grad(_single_energy, _single_params())
    assert "flow_slope_ratio" in get_last_implicit_ad_diagnostics()

    def must_not_run(*a, **k):
        raise AssertionError("the slope check ran on a stationary forward")

    monkeypatch.setattr(cea, "_slope_probe_energy", must_not_run)
    _, d = _grad(_single_energy, _single_params(), chi=8, max_iter=400, conv_tol=1e-11)
    assert d["forward_stationarity_residual"] <= d["forward_stationarity_threshold"], (
        "premise"
    )
    assert d["projector_backward_used"] == "flow"
    assert "flow_slope_ratio" not in d


def test_forward_diagnostics_survive_the_check():
    p = _single_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _single_energy(p)  # forward only
    before = dict(get_last_implicit_ad_diagnostics())
    _, after = _grad(_single_energy, p)
    assert "flow_slope_ratio" in after, "premise: the check ran"
    for k in (
        "forward_stationarity_residual",
        "forward_iterations",
        "forward_converged",
    ):
        assert after[k] == before[k], (k, before[k], after[k])


def test_inconclusive_check_falls_back_to_frozen(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe forward failed")

    monkeypatch.setattr(cea, "_slope_probe_energy", boom)
    _, d = _grad(_single_energy, _single_params())
    assert d["projector_backward_used"] == "frozen_fallback", d
    assert "could not evaluate" in d["flow_fallback_reason"], d["flow_fallback_reason"]
    assert "probe forward failed" in d["flow_fallback_reason"]
    assert "flow_slope_ratio" not in d


def test_the_check_spends_no_latch_and_emits_no_warning(monkeypatch):
    """Discriminating: the forward and warm-up already trip the once-per-run
    latches, so reset them between the forward and the pullback and require
    that the probe leaves them reset."""
    real = cea._slope_probe_energy

    def noisy(*a, **k):
        warnings.warn("probe-forward warning", RuntimeWarning)
        return real(*a, **k)

    monkeypatch.setattr(cea, "_slope_probe_energy", noisy)
    p = _single_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pullback = jax.vjp(lambda q: _single_energy(q), p)
    n_runs = 0
    for entry in cea._VJP_CACHE.values():
        if not isinstance(entry, tuple):
            continue
        for _tok, state, _objs in entry[1].get("_runs", {}).values():
            n_runs += 1
            for k in list(state):
                if k.endswith("_warned"):
                    state[k] = False
    assert n_runs, "premise: a run exists"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pullback(jnp.ones(()))
    assert "flow_slope_ratio" in get_last_implicit_ad_diagnostics(), (
        "premise: the check ran"
    )
    assert not any("probe-forward warning" in str(w.message) for w in caught)
    for k, v in _all_latches().items():
        assert not any(v.values()), (k, v)


def test_pullback_twice_gives_the_same_verdict():
    p = _pair_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pullback = jax.vjp(lambda q: _pair_energy(q), p)
        g1 = pullback(jnp.ones(()))
        d1 = dict(get_last_implicit_ad_diagnostics())
        g2 = pullback(jnp.ones(()))
        d2 = dict(get_last_implicit_ad_diagnostics())
    assert (
        d1["projector_backward_used"]
        == d2["projector_backward_used"]
        == "frozen_fallback"
    )
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


def test_optimize_gs_ad_step_on_the_1028_fixture(monkeypatch):
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    real_probe = cea._slope_probe_energy
    probe_energies = []

    def spy(*a, **k):
        e, chi = real_probe(*a, **k)
        probe_energies.append(e)
        return e, chi

    monkeypatch.setattr(cea, "_slope_probe_energy", spy)
    A, B = _pair_params()
    config = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        gs_num_steps=1,
        ctm=CTMConfig(
            chi=8,
            max_iter=20,
            projector_method="svd",
            ctm_conv_method="elementwise",
            forward_gauge="bond_phase",
            projector_backward="flow",
            on_unconverged="warn",  # the fixture's 20-sweep forward is non-stationary
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimize_gs_ad(heisenberg_gate(), (A, B), config)
    flow_warnings = [
        w for w in caught if "flowing-projector backward" in str(w.message)
    ]
    assert len(flow_warnings) <= 1
    _a_opt, _envs, energy = result  # 2-site: ((A, B), (env_A, env_B), E_gs)
    assert np.isfinite(float(energy)), energy
    # Whether the check runs here depends on whether the forward the optimizer
    # differentiates is stationary, and on this fixture that is borderline:
    # measured stationary on one CPU (12 sweeps, converged) and non-stationary
    # on CI.  So assert only what holds either way: any probe forward returned
    # a finite energy, and if the check ran, the last backward reports a
    # verdict consistent with it.  The check's accept/reject logic is pinned
    # by the direct-gradient tests above.
    assert all(np.isfinite(e) for e in probe_energies), probe_energies
    d = get_last_implicit_ad_diagnostics()
    if "flow_slope_ratio" in d:
        r = d["flow_slope_ratio"]
        if 0.5 <= r <= 2.0:
            assert d["projector_backward_used"] == "flow", d
        else:
            assert d["projector_backward_used"] == "frozen_fallback", d
            assert "slope check" in d["flow_fallback_reason"], d


def test_slope_check_keeps_flow_for_complex_params():
    """JAX's grad of a real function of complex z is the conjugate of the
    steepest-ascent direction, so the probe must step along conj(G); without
    that r was -0.04..-0.12 and flow was always discarded.

    The implicit-AD VJP cache and warm start are not keyed on dtype (issue
    #1113 fixes the underlying bug), so the warm start is invalidated in a
    finally block and later real-dtype tests cannot pick up a complex seed.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(2026))
    shape = (2, 2, 2, 2, 2)
    A = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
    params = (A / jnp.linalg.norm(A),)
    try:
        _, d = _grad(_single_energy, params)
    finally:
        cea.invalidate_implicit_ad_warm_start()
    assert d["forward_stationarity_residual"] > d["forward_stationarity_threshold"], (
        "premise"
    )
    assert d["projector_backward_used"] == "flow", d
    assert 0.5 <= d["flow_slope_ratio"] <= 2.0, d["flow_slope_ratio"]


def _single_energy_with(p, *, gate=_H1, env_init=None):
    """``_single_energy`` with the per-call inputs exposed: same config, so the
    same implicit-AD cache entry, whatever ``gate`` object or ``env_init``."""
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(p[0])},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=20,
            conv_tol=1e-10,
            forward_gauge="bond_phase",
            projector_backward="flow",
            env_init=env_init,
        )
    )


def _foreign_env():
    """A converged chi=4 env of other params (seed 3).  As the probe's start
    it moves the limit-cycle fixture's r from 1.002 to 0.276."""
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge

    (other,) = _single_params(seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        envs, _info = python_loop_ctm_converge(
            {(0, 0): _wrap_as_dense_tensor(other)},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=20,
            renormalize=True,
        )
    return envs


def test_late_pullback_uses_its_own_forward_inputs():
    """Codex on #1114: a pullback called after another call on the same cache
    entry probed with that call's ``env_init``.  Measured without the fix:
    flow r = 1.0022 immediately, but frozen_fallback r = 0.276 late."""
    p = _single_params()
    envs = _foreign_env()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, now = jax.vjp(_single_energy_with, p)
        g_now = now(jnp.ones(()))
        d_now = dict(get_last_implicit_ad_diagnostics())
        _, late = jax.vjp(_single_energy_with, p)
        n_entries = len(cea._VJP_CACHE)
        _single_energy_with(p, env_init=envs)  # same entry, other env_init
        assert len(cea._VJP_CACHE) == n_entries, "premise: same cache entry"
        g_late = late(jnp.ones(()))
        d_late = dict(get_last_implicit_ad_diagnostics())
    assert d_now["projector_backward_used"] == "flow", d_now
    assert d_late["projector_backward_used"] == d_now["projector_backward_used"], d_late
    np.testing.assert_allclose(
        d_late["flow_slope_ratio"], d_now["flow_slope_ratio"], rtol=1e-10
    )
    for a, b in zip(jax.tree.leaves(g_late), jax.tree.leaves(g_now)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


def test_late_pullback_spends_no_latch_of_the_current_run():
    """Codex on #1114: the probe's forward recorded its stationarity latch
    against the run current at pullback time, which the probe did not
    restore.  A gate copy (equal value, new object) selects a new run on the
    same cache entry."""
    p = _single_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pullback = jax.vjp(_single_energy_with, p)
        n_entries = len(cea._VJP_CACHE)
        n_runs = len(_all_latches())
        _single_energy_with(p, gate=jnp.array(_H1))  # run B
    assert len(cea._VJP_CACHE) == n_entries, "premise: same cache entry"
    assert len(_all_latches()) == n_runs + 1, "premise: a second run"
    for entry in cea._VJP_CACHE.values():
        if isinstance(entry, tuple):
            for _tok, state, _objs in entry[1].get("_runs", {}).values():
                for k in list(state):
                    if k.endswith("_warned"):
                        state[k] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pullback(jnp.ones(()))
    d = get_last_implicit_ad_diagnostics()
    assert "flow_slope_ratio" in d, "premise: the check ran"
    assert d["projector_backward_used"] == "flow", d
    for k, v in _all_latches().items():
        assert not any(v.values()), (k, v)


def test_evicted_forward_inputs_make_the_check_inconclusive(monkeypatch):
    monkeypatch.setattr(cea, "_PROBE_INPUTS_PER_ENTRY", 1)
    p = _single_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, first = jax.vjp(_single_energy_with, p)
        _, _second = jax.vjp(_single_energy_with, p)  # evicts first's inputs
        first(jnp.ones(()))
    d = get_last_implicit_ad_diagnostics()
    assert d["projector_backward_used"] == "frozen_fallback", d
    assert "evicted" in d["flow_fallback_reason"], d["flow_fallback_reason"]
    assert "flow_slope_ratio" not in d
