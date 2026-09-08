"""The BP solve must not certify a bond weight that underflowed to zero.

``_is_representable`` is scale-invariant by design, so it reads
``max(lambda) = 1`` and passes even when a bond's smallest weight has reached
exactly 0.  It catches a *total* collapse and cannot see a *partial* one.

That gap produced a confidently converged non-gauge.  On a D=3 U(1)-Sz pair the
smallest weight on each bond decayed geometrically and hit 0.0 at sweep 109;
``_sqrt_pinv`` then had no direction left to invert, so the transformation
stopped being a gauge, and by sweep 114 the solve reported
``residual = 1.19e-16`` and ``converged`` on a state that had moved by 3.0e-01
— with the health gate returning ``True`` at every sweep.  #870 is the same
failure with the sign flipped: growth to ``inf`` there, decay to zero here, and
in both the residual certifies the corpse.

**A weight reaching zero is not by itself the failure**, and an earlier version
of this check assumed it was.  Counting rank rejected
``test_su_step_survives_a_bond_direction_the_state_does_not_use`` on its first
sweep: a direction the state genuinely does not use dies from a *full-sized*
weight in one step, and that is a real solve.  The two are separated by where
the weight came from, measured on both sides — relative ``1.0`` legitimately,
``1.1e-08`` at worst fatally — which is what :data:`_UNDERFLOW_EPS` sits
between.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms.ipeps_bp_gauge import (
    _UNDERFLOW_EPS,
    _a_weight_underflowed,
    _is_representable,
)
from tenax.algorithms.ipeps_simple_update import BondWeights


def _w(**kw):
    base = dict(
        h_AB=jnp.array([1.0, 0.4, 0.1]),
        h_BA=jnp.array([1.0, 0.6, 0.2]),
        v_AB=jnp.array([1.0, 0.3, 0.05]),
        v_BA=jnp.array([1.0, 0.7, 0.3]),
    )
    base.update(kw)
    return BondWeights(**base)


def test_a_collapsing_weight_that_reaches_zero_is_refused():
    """The measured failure: 3.3e-10 -> 0.0 after a geometric decay."""
    before = _w(h_AB=jnp.array([1.0, 0.4, 3.3e-10]))
    after = _w(h_AB=jnp.array([1.0, 0.4, 0.0]))

    assert bool(_a_weight_underflowed(after, before))


def test_a_direction_the_state_does_not_use_is_accepted():
    """The measured legitimate case: a full-sized weight dies in one sweep.

    This is what ``test_su_step_survives_a_bond_direction_the_state_does_not_use``
    builds, and rejecting it stops that solve at sweep 0.
    """
    before = _w(h_AB=jnp.array([1.0, 1.0, 1.0]))
    after = _w(h_AB=jnp.array([1.0, 0.7766, 0.0]))

    assert not bool(_a_weight_underflowed(after, before))


@pytest.mark.parametrize("came_from", [1.0, 0.5, 1e-2, 1e-4])
def test_a_weight_above_the_threshold_may_die(came_from):
    before = _w(h_AB=jnp.array([1.0, 0.4, came_from]))
    after = _w(h_AB=jnp.array([1.0, 0.4, 0.0]))

    assert not bool(_a_weight_underflowed(after, before))


@pytest.mark.parametrize("came_from", [1e-7, 1e-8, 1e-10, 1e-30])
def test_a_weight_below_the_threshold_may_not(came_from):
    before = _w(h_AB=jnp.array([1.0, 0.4, came_from]))
    after = _w(h_AB=jnp.array([1.0, 0.4, 0.0]))

    assert bool(_a_weight_underflowed(after, before))


def test_the_threshold_sits_between_the_two_measurements():
    """Both sides are measured, so the constant is auditable rather than tuned.

    Legitimate deaths came from relative 1.0; fatal ones from 1.1e-08 at the
    very worst.  If either bound moves, this says so rather than letting the
    constant drift out from under the evidence.
    """
    assert 1.1e-08 < _UNDERFLOW_EPS < 1.0
    assert _UNDERFLOW_EPS / 1.1e-08 > 10, "less than an order of margin above fatal"
    assert 1.0 / _UNDERFLOW_EPS > 10, "less than an order of margin below legitimate"


def test_a_weight_that_merely_shrinks_is_accepted():
    """Only reaching zero is the failure; small is not."""
    before = _w(h_AB=jnp.array([1.0, 0.4, 1e-8]))
    after = _w(h_AB=jnp.array([1.0, 0.4, 1e-30]))

    assert not bool(_a_weight_underflowed(after, before))


def test_a_weight_that_was_already_zero_is_not_a_new_failure():
    """Otherwise every low-entanglement fixture would be refused forever."""
    w = _w(h_AB=jnp.array([1.0, 0.4, 0.0]))
    assert not bool(_a_weight_underflowed(w, w))


def test_a_length_change_is_accepted_rather_than_raising():
    """A charge sector emptying shortens the vector (#904/#906).

    The two then cannot be aligned entry by entry, so the sweep is accepted.
    Documented as a hole rather than hidden: the failure this exists for keeps
    every bond at its width throughout.
    """
    before = _w(h_AB=jnp.array([1.0, 0.4, 1e-10]))
    after = _w(h_AB=jnp.array([1.0, 0.4]))

    assert not bool(_a_weight_underflowed(after, before))


def test_the_check_is_traceable():
    """Both drivers use it, and the traced one from inside a while_loop."""
    before = _w(h_AB=jnp.array([1.0, 0.4, 1e-10]))
    after = _w(h_AB=jnp.array([1.0, 0.4, 0.0]))

    out = jax.jit(_a_weight_underflowed)(after, before)

    assert out.shape == () and out.dtype == jnp.bool_
    assert bool(out)


def _names(code):
    """Every global name reachable from ``code``, nested closures included.

    ``_bp_solve`` calls the check inside its ``body`` closure, so a flat read of
    ``co_names`` misses it -- which the first version of this test did, and
    reported the wiring as absent when it was there.
    """
    import types

    out = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            out |= _names(const)
    return out


def test_it_is_wired_into_both_drivers():
    """A gate nothing calls is not a gate."""
    import tenax.algorithms.ipeps_bp_gauge as bp

    for driver in (bp._bp_solve_eager, bp._bp_solve):
        assert "_a_weight_underflowed" in _names(driver.__code__), (
            f"{driver.__name__} does not consult the underflow check"
        )


def test_is_representable_still_catches_total_collapse():
    """The new check adds to the old one rather than replacing it."""
    zero = _w(
        h_AB=jnp.zeros(3), h_BA=jnp.zeros(3), v_AB=jnp.zeros(3), v_BA=jnp.zeros(3)
    )
    assert not bool(_is_representable({}, zero))
