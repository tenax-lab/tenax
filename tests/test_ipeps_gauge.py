"""The absorbed-form convention: a site tensor IS the wavefunction.

The Vidal warning in bp_gauge_checkerboard is about passing ones() alongside
BARE Gamma, which drops lambda.  Absorbing sqrt(lambda) into both ends first
and then passing ones() is the same state written differently --
A sqrt(lam) . sqrt(lam) B == A lam B -- and that is what this file proves.

Runs on the symmetric path too: a flow mistake there collapses charge sectors
rather than raising, and the dense path cannot see it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_gauge_helpers import (  # tests/ is on sys.path
    _INDEPENDENT_BOND_OF,
    _PAIRS,
    _torus_2x2,
    assert_leg_split,
)

from tenax.algorithms.ipeps_bp_gauge import BondWeights, bp_gauge_checkerboard
from tenax.algorithms.ipeps_gauge import absorb_weights, gauge_fix, torus_2x2_graded

ABSORB_TOL = 1e-13


def _unit(x):
    return x / np.linalg.norm(x)


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
@pytest.mark.parametrize("D", [2, 3])
def test_absorb_weights_splits_each_bond_onto_both_ends_not_just_one(kind, D):
    """The torus-based tests below cannot see an asymmetric split.

    ``_torus_2x2`` is a closed loop: a diagonal weight factors arbitrarily
    between the two ends of a bond without changing the total the torus sums
    over, so a broken ``absorb_weights`` that dumped the *whole* weight onto
    site A's legs and left B's legs at 1 -- a very plausible "just multiply
    lambda into one Gamma" mistake -- would reproduce the identical torus
    value in ``test_absorbed_form_is_the_same_state``, and would very likely
    still reach the same BP fixed point in
    ``test_gauge_fix_matches_the_vidal_route``, since BP-gauging is
    insensitive to which valid gauge of the same physical state it starts
    from. Only a check on each site's tensor in isolation, against an
    independently-derived bond map, can catch a wrong 50/50 split.
    """
    A, B = _PAIRS[kind](D=D)
    w = BondWeights(
        h_AB=jnp.array([1.0, 0.4, 0.1][:D]),
        h_BA=jnp.array([1.0, 0.6, 0.2][:D]),
        v_AB=jnp.array([1.0, 0.3, 0.05][:D]),
        v_BA=jnp.array([1.0, 0.7, 0.3][:D]),
    )
    Aa, Ba = absorb_weights(A, B, w)

    for site, before, after in (("A", A, Aa), ("B", B, Ba)):
        scale_of_leg = {
            leg: np.sqrt(np.asarray(getattr(w, _INDEPENDENT_BOND_OF[(site, leg)])))
            for leg in ("u", "d", "l", "r")
        }
        assert_leg_split(
            site, before, after, scale_of_leg, ABSORB_TOL, msg=f"{kind} D={D} "
        )


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
@pytest.mark.parametrize("D", [2, 3])
def test_absorbed_form_is_the_same_state(kind, D):
    """Absorbing sqrt(lambda) into both ends does not move the physical state."""
    A, B = _PAIRS[kind](D=D)
    w = BondWeights(
        h_AB=jnp.array([1.0, 0.4, 0.1][:D]),
        h_BA=jnp.array([1.0, 0.6, 0.2][:D]),
        v_AB=jnp.array([1.0, 0.3, 0.05][:D]),
        v_BA=jnp.array([1.0, 0.7, 0.3][:D]),
    )
    Aa, Ba = absorb_weights(A, B, w)
    ones = BondWeights.ones(D, D)

    before = _unit(_torus_2x2(A, B, w))
    after = _unit(_torus_2x2(Aa, Ba, ones))
    rel = float(np.linalg.norm(after - before))
    assert rel < ABSORB_TOL, (
        f"{kind} D={D}: absorbing the weights moved the state by {rel:.3e}; "
        f"the absorbed form must be the SAME wavefunction, not an approximation"
    )


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
@pytest.mark.parametrize("D", [2, 3])
def test_gauge_fix_matches_the_vidal_route(kind, D):
    """gauge_fix on absorbed input reaches the Vidal call's state AND fixed point.

    If this fails, gauge_fix is gauging a different state than the caller
    handed it, and every acceptance criterion above it is meaningless.
    """
    A, B = _PAIRS[kind](D=D)
    w = BondWeights(
        h_AB=jnp.array([1.0, 0.4, 0.1][:D]),
        h_BA=jnp.array([1.0, 0.6, 0.2][:D]),
        v_AB=jnp.array([1.0, 0.3, 0.05][:D]),
        v_BA=jnp.array([1.0, 0.7, 0.3][:D]),
    )
    A1, B1, w1, i1 = bp_gauge_checkerboard(A, B, w, tol=1e-12, max_iter=400)
    Aa, Ba = absorb_weights(A, B, w)
    A2, B2, w2, i2 = gauge_fix(Aa, Ba, tol=1e-12, max_iter=400)

    assert i1.converged and i2.converged

    state = float(
        np.linalg.norm(_unit(_torus_2x2(A2, B2, w2)) - _unit(_torus_2x2(A1, B1, w1)))
    )
    assert state < 1e-12, f"{kind} D={D}: routes reached different states ({state:.3e})"

    for f in BondWeights._fields:
        a, b = getattr(w1, f), getattr(w2, f)
        a, b = a / jnp.max(a), b / jnp.max(b)
        d = float(jnp.max(jnp.abs(a - b)))
        assert d < 1e-10, f"{kind} D={D} bond {f}: different fixed point ({d:.3e})"


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_graded_probe_matches_einsum_probe_on_bosonic(kind):
    """Validate the graded probe where the einsum probe is still correct.

    Without this, a fermionic gauge test is comparing two unknowns.
    """
    A, B = _PAIRS[kind](D=3)
    w = BondWeights.ones(3, 3)
    ref = _unit(np.asarray(_torus_2x2(A, B, w)).ravel())
    got = _unit(np.asarray(torus_2x2_graded(A, B, w).todense()).ravel())
    rel = float(np.linalg.norm(got - ref))
    assert rel < 1e-13, (
        f"{kind}: graded probe disagrees with the einsum probe by {rel:.3e} on a "
        f"BOSONIC pair, where both are valid.  Fix the probe before trusting it "
        f"on a fermionic one."
    )


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_graded_probe_places_each_weight_on_the_right_leg(kind):
    """The same agreement with a *distinct* weight on every bond.

    ``BondWeights.ones`` above makes all four weights identical, so it cannot
    see a probe that hangs ``h_AB`` where ``v_AB`` belongs, or that dresses
    ``u``/``l`` instead of ``d``/``r`` -- every mis-placement multiplies the
    same ones.  Four different, non-degenerate vectors make each of those
    mistakes change the answer.  The weights below deliberately have no
    repeated entry within or across bonds, so even a swap of two legs of the
    same bond type shows up.
    """
    A, B = _PAIRS[kind](D=3)
    w = BondWeights(
        h_AB=jnp.array([1.0, 0.4, 0.1]),
        h_BA=jnp.array([0.9, 0.6, 0.2]),
        v_AB=jnp.array([0.8, 0.3, 0.05]),
        v_BA=jnp.array([0.7, 0.5, 0.3]),
    )
    ref = _unit(np.asarray(_torus_2x2(A, B, w)).ravel())
    got = _unit(np.asarray(torus_2x2_graded(A, B, w).todense()).ravel())
    rel = float(np.linalg.norm(got - ref))
    assert rel < 1e-13, (
        f"{kind}: graded probe disagrees with the einsum probe by {rel:.3e} once "
        f"the four bonds carry different weights -- the wiring or the weight "
        f"placement is wrong, not the gauge"
    )


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_graded_probe_sees_the_known_bosonic_gauge_invariance(kind):
    """The instrument reproduces the known answer on a known-exact gauge.

    ``bp_gauge_checkerboard`` is already verified exact to ~1e-15 on both
    bosonic pair types, so a probe that cannot see that is broken.  This is
    the calibration, not the experiment: the fermionic case -- where the
    einsum probe is *invalid* and this one is the only witness -- is
    deliberately left to the next task, so the instrument is fixed before the
    measurement it is built for.
    """
    A, B = _PAIRS[kind](D=3)
    ones = BondWeights.ones(3, 3)
    A1, B1, w1, info = bp_gauge_checkerboard(A, B, ones, tol=1e-12, max_iter=400)
    assert info.converged

    before = _unit(np.asarray(torus_2x2_graded(A, B, ones).todense()).ravel())
    after = _unit(np.asarray(torus_2x2_graded(A1, B1, w1).todense()).ravel())
    rel = float(np.linalg.norm(after - before))
    assert rel < 1e-12, (
        f"{kind}: the graded probe reports a {rel:.3e} state change across a "
        f"gauge already known exact to ~1e-15; the probe is what is wrong"
    )
