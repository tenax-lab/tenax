"""CTM hold test (#1035): successive-sweep agreement does not certify an attractor.

The eager 2-site CTM certified a SADDLE S of the fermionic t-V D=3 chi=12 step
(successive sweeps agree to 1e-10, yet a displacement grows x1.041/sweep and
the loop, given ~650 more sweeps, lands on a stable fixed point B 1.4e-2
away).  These tests pin the mechanism with maps whose stability is known by
construction -- not by running a real CTM to convergence -- plus one real
dense CTM that must still pass.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_tensor_convergence as conv
from tenax.algorithms._ctm_hold import (
    env_invariant_distance,
    env_spectral_invariants,
    hold_test,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_multisite,
)
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# hold_test on linear maps of known spectrum                                   #
# --------------------------------------------------------------------------- #


class _Vec(NamedTuple):
    x: jax.Array


def _vec_env(v):
    return {(0, 0): _Vec(jnp.asarray(v, dtype=jnp.float64))}


def _identity_invariants(envs):
    return {"x": np.asarray(envs[(0, 0)].x)}


def _linear_step(J, xstar):
    J = jnp.asarray(J)

    def step(envs):
        x = envs[(0, 0)].x
        return _vec_env(xstar + J @ (x - xstar))

    return step


XSTAR = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])


def test_a_saddle_fails_the_hold():
    """Five stable directions at 0.87 and one unstable at 1.041 -- the #1035
    measured rates.  The unstable direction is 1/6 of a random perturbation."""
    J = np.diag([0.87, 0.87, 0.87, 0.87, 0.87, 1.041])
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR),
        sweeps=40,
        invariants=_identity_invariants,
    )
    assert not held.passed
    assert held.rate > 1.0
    # A saddle never passes a re-test either: it runs the full 3*K extension,
    # two CTM steps per sweep.
    assert held.sweeps == 2 * 3 * 40


def test_an_attractor_with_a_transient_passes_where_d_k_below_d_0_would_not():
    """Non-normal but stable (eigenvalues 0.95, 0.9): the displacement first
    grows ~20x -- the attractor B of #1035 grows ~25x -- then decays.  The
    fitted tail rate is < 1 and passes; the naive ``d_K < d_0`` would still
    read "unstable" at K=40, which is why the hold does not use it."""
    J = np.eye(6) * 0.9
    J[0, 0] = 0.95
    J[0, 1:] = 3.0  # strong non-normal coupling into the slow direction
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR),
        sweeps=40,
        invariants=_identity_invariants,
    )
    assert held.passed
    assert held.rate < 1.0
    d = np.asarray(held.distances)
    assert d.max() > 10 * d[0]  # regime: the transient is real
    assert d[40] > d[0]  # regime: d_K < d_0 would have failed this attractor


def test_a_claimed_point_off_by_its_residual_still_holds():
    """The loop certifies a point only to ``conv_tol``: here the claimed
    point sits 1e-7 from the true attractor, so the perturbed trajectory's
    distance to the CLAIMED point plateaus at 1e-7.  The verdict compares
    against the co-evolved unperturbed trajectory, which has no such floor,
    and recovers the true rate.
    (The discriminating case -- a plateau that fits to a rate >= 1 -- is the
    mocked loop below, whose attractor the distance rule rejected once.)"""
    J = np.eye(6) * 0.9
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR + 1e-7),
        sweeps=40,
        perturbation=1e-6,
        invariants=_identity_invariants,
    )
    assert held.passed
    assert abs(held.rate - 0.9) < 1e-6
    # Regime: the perturbed trajectory ends 1e-7 from the claimed point --
    # the plateau a distance-to-claimed-point statistic would sit on.
    assert float(jnp.max(jnp.abs(held.envs[(0, 0)].x - (XSTAR + 1e-7)))) > 0.5e-7


def test_a_transient_that_outlasts_the_first_window_is_not_a_saddle():
    """A Jordan block at 0.975: ``d_k ~ k 0.975^k`` peaks at k~40, so the fit
    over [20, 40] still grows (0.975 * 2**(1/20) = 1.009).  Measured on the
    real #1035 attractor too (1.019 at K=40, 0.959 at K=60).  The re-test on
    the next window passes it; with no extension it would be rejected."""
    J = np.eye(6) * 0.5
    J[0, 0] = J[1, 1] = 0.975
    J[0, 1] = 5.0
    kw = dict(sweeps=40, invariants=_identity_invariants)
    rejected = hold_test(_linear_step(J, XSTAR), _vec_env(XSTAR), max_sweeps=40, **kw)
    assert not rejected.passed  # regime: the first window alone gets it wrong
    held = hold_test(_linear_step(J, XSTAR), _vec_env(XSTAR), **kw)
    assert held.passed
    assert 2 * 40 < held.sweeps < 2 * 3 * 40


def test_a_fast_attractor_passes_early():
    J = np.eye(6) * 0.1
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR),
        sweeps=40,
        invariants=_identity_invariants,
    )
    assert held.passed
    assert held.sweeps < 10


def test_the_default_metric_is_blind_to_a_period_two_sign_cycle():
    """An attractor whose step flips the sign of alternate rows every sweep --
    the gauge-covariant env's +-1 period-2 cycle.  Element-wise, the
    trajectory never approaches the claimed point; in the spectral invariants
    it contracts, and the hold passes."""
    rng = np.random.default_rng(0)
    Cstar = jnp.asarray(rng.standard_normal((5, 5)))
    sigma = jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0])[:, None]

    def step(envs):
        # Stateless period-2 gauge cycle: find which of {C*, sigma C*} the
        # input sits near (g), contract the displacement, and emit it next to
        # the OTHER one.  The sign pattern never settles; the spectrum does.
        C = envs[(0, 0)].x
        g = (
            1.0
            if jnp.sum((C - Cstar) ** 2) <= jnp.sum((C - sigma * Cstar) ** 2)
            else sigma
        )
        delta = C - g * Cstar
        return {(0, 0): _Vec(sigma * g * Cstar + 0.5 * sigma * delta)}

    start = {(0, 0): _Vec(Cstar)}
    # Regime: element-wise, one sweep moves the claimed point by O(1) -- the
    # element-wise metric would never see this attractor converge.
    assert float(jnp.max(jnp.abs(step(start)[(0, 0)].x - Cstar))) > 0.1
    held = hold_test(step, start, sweeps=20)
    assert held.passed


def test_the_invariant_distance_is_gauge_blind_and_continuous():
    rng = np.random.default_rng(1)
    C = jnp.asarray(rng.standard_normal((4, 4)))
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    a = env_spectral_invariants({(0, 0): _Vec(C)})
    b = env_spectral_invariants({(0, 0): _Vec(jnp.asarray(Q) @ C[:, ::-1])})
    assert env_invariant_distance(a, b) < 1e-12
    c = env_spectral_invariants({(0, 0): _Vec(C + 1e-7 * jnp.eye(4))})
    assert 0 < env_invariant_distance(a, c) < 1e-5
    nan = env_spectral_invariants({(0, 0): _Vec(C * jnp.nan)})
    assert math.isinf(env_invariant_distance(a, nan))


# --------------------------------------------------------------------------- #
# The eager loop: a mocked sweep with a saddle and an attractor                #
# --------------------------------------------------------------------------- #


def _site(seed=0, D=2, d=2):
    from tests._split_ctm_oracle import make_site

    return make_site(D, d, seed)


CHI = 4


def _saddle_sweep(env0, a, contract=0.5):
    """A fake sweep with a known fixed point ``ref`` and known stability.

    ``ref`` is the loop's initial environment plus a random full-rank corner
    offset (the initial corner is rank 1, which the criterion is blind to).
    Every direction contracts onto ``ref`` at ``contract``, except one scalar
    ``s`` along a corner direction ``u``: ``s <- s + a s (1 - s^2 / s_B^2)``.
    For ``a > 0`` s=0 is a saddle (rate 1 + a) and s=+-s_B are attractors
    (rate 1 - 2a); for ``a < 0`` s=0 is the attractor.  ``u`` is orthogonal
    to the initial displacement, so the loop converges exactly onto s=0.
    """
    rng = np.random.default_rng(3)
    ref, u = {}, {}
    for c, e in env0.items():
        C1 = e.C1.todense()
        R = rng.standard_normal(C1.shape)
        ref[c] = e._replace(C1=type(e.C1)(C1 + jnp.asarray(R), e.C1.indices))
        v = rng.standard_normal(R.shape)
        v -= np.vdot(R, v) / np.vdot(R, R) * R
        u[c] = jnp.asarray(v / np.linalg.norm(v))
    s_B = 0.1 * float(jnp.linalg.norm(ref[(0, 0)].C1.todense()))
    ref_leaves = {c: jax.tree.leaves(e) for c, e in ref.items()}

    def fake(envs, *args, **kwargs):
        out = {}
        for c, e in envs.items():
            leaves, tdef = jax.tree.flatten(e)
            new = []
            for i, (x, r) in enumerate(zip(leaves, ref_leaves[c])):
                dx = x - r
                if i == 0:  # C1
                    s = jnp.vdot(u[c], dx).real
                    perp = dx - s * u[c]
                    s = s + a * s * (1 - s**2 / s_B**2)
                    new.append(r + contract * perp + s * u[c])
                else:
                    new.append(r + contract * dx)
            out[c] = jax.tree.unflatten(tdef, new)
        return out, None, None

    def s_of(envs):
        d = envs[(0, 0)].C1.todense() - ref[(0, 0)].C1.todense()
        return float(jnp.vdot(u[(0, 0)], d).real)

    return fake, s_B, s_of


def _run_mocked(monkeypatch, a, verdicts=None, **kw):
    A = _site()
    sites = {(0, 0): A, (1, 0): A}
    env0 = {c: initialize_ctm_tensor_env(t, CHI) for c, t in sites.items()}
    fake, s_B, s_of = _saddle_sweep(env0, a)
    monkeypatch.setattr(conv, "_ctm_tensor_sweep_multisite", fake)
    if verdicts is not None:
        real = conv.hold_test

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            verdicts.append(out.passed)
            return out

        monkeypatch.setattr(conv, "hold_test", spy)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        envs = _ctm_tensor_multisite(sites, CHECKERBOARD_NEIGHBORS, CHI, **kw)
    not_conv = [w for w in rec if "did not converge" in str(w.message)]
    blind = [w for w in rec if "could not be certified" in str(w.message)]
    assert not blind  # regime: the criterion can see these corners
    return envs, s_B, s_of, not_conv


def test_regime_without_the_hold_the_loop_certifies_the_saddle(monkeypatch):
    """The defect this file exists for: the loop starts ON the saddle, so
    successive sweeps agree exactly and the successive criterion certifies it."""
    envs, s_B, s_of, not_conv = _run_mocked(
        monkeypatch, 0.1, max_iter=2000, hold_sweeps=0
    )
    assert abs(s_of(envs)) < 1e-12
    assert not not_conv


def test_the_loop_walks_off_a_saddle_to_the_attractor(monkeypatch):
    verdicts = []
    envs, s_B, s_of, not_conv = _run_mocked(monkeypatch, 0.1, verdicts, max_iter=2000)
    assert abs(abs(s_of(envs)) - s_B) < 1e-6 * s_B
    assert not not_conv
    # One hold rejects the saddle, the next accepts the attractor -- the
    # attractor is not rejected for sitting conv_tol off its own fixed point
    # (a rate fitted to the distance-to-claimed-point did: [False, False, True]).
    assert verdicts == [False, True]


def test_a_saddle_at_max_iter_is_reported_not_converged(monkeypatch):
    # Criterion passes at ~sweep 22; the hold, capped by the budget at
    # (150 - 22) // 2 sweeps, rejects the saddle and exhausts max_iter.
    envs, s_B, s_of, not_conv = _run_mocked(monkeypatch, 0.1, max_iter=150)
    assert not_conv
    assert "saddle" in str(not_conv[0].message)


def test_an_attractor_is_returned_as_is(monkeypatch):
    envs, s_B, s_of, not_conv = _run_mocked(monkeypatch, -0.1, max_iter=2000)
    assert abs(s_of(envs)) < 1e-12  # the claimed point, not the perturbed one
    assert not not_conv


def test_no_budget_left_for_the_hold_is_not_converged(monkeypatch):
    """The criterion passes at sweep ~22; 22 + 2*40 > max_iter=45, so nothing
    certified the point -- fail closed rather than skip the hold."""
    _, _, _, not_conv = _run_mocked(monkeypatch, -0.1, max_iter=45)
    assert not_conv
    assert "uncertified" in str(not_conv[0].message)


def test_hold_sweeps_below_four_is_refused():
    A = _site()
    with pytest.raises(ValueError, match="hold_sweeps"):
        _ctm_tensor_multisite(
            {(0, 0): A, (1, 0): A}, CHECKERBOARD_NEIGHBORS, CHI, hold_sweeps=2
        )


# --------------------------------------------------------------------------- #
# A real, converged dense environment must pass                                #
# --------------------------------------------------------------------------- #


def test_a_converged_dense_d2_environment_holds():
    """Dense D=2 Heisenberg SU state, chi=12: the hold passes early (the
    displacement collapses ~1e-3 in a few sweeps) and hands back exactly the
    point the hold-free loop returns."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.core.tensor import DenseTensor
    from tests.test_ctm_chi_truncation_policy_922 import _su_pair

    A, B = _su_pair(D=2)
    A = DenseTensor(A.todense(), A.indices)
    B = DenseTensor(B.todense(), B.indices)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        held = ctm_tensor_2site(A, B, 12, max_iter=200, conv_tol=1e-10)
    plain = ctm_tensor_2site(A, B, 12, max_iter=200, conv_tol=1e-10, hold_sweeps=0)
    for x, y in zip(jax.tree.leaves(held), jax.tree.leaves(plain)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
