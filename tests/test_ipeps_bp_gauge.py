"""The BP gauge is a gauge: it must move the weights and not the state.

Every way this can go wrong leaves a plausible-looking Schmidt spectrum behind,
so a test that only inspects the returned weights proves nothing.  Two mistakes
made while writing it, both of which produced perfectly reasonable spectra:

* dropping ``lambda_old`` from the re-gauging SVD -- moved the energy by 1.2e-01;
* contracting the gauge matrix into ``Gamma`` with matching flows, which on a
  ``SymmetricTensor`` silently collapses charge sectors instead of raising --
  broke the gauge by 2.7e-01 while ``DenseTensor`` stayed exact at 5e-16.

Only the invariance check below catches either, so it is the centre of this
file, and it runs on both tensor types because the second failure is invisible
on dense input.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_gauge_helpers import (  # tests/ is on sys.path
    _PAIRS,
    _dense_pair,
    _symmetric_pair,
    _torus_2x2,
    retraced,  # noqa: F401  -- a fixture; pytest reads it off this namespace
)

from tenax.algorithms.ipeps import (
    _make_trotter_gate_tensor,
    _wrap_as_dense_tensor,
    heisenberg_gate,
    sublattice_rotate_gate,
)
from tenax.algorithms.ipeps_bp_gauge import (
    BondWeights,
    _gauge_bond,
    _is_representable,
    _message,
    bp_gauge_checkerboard,
)
from tenax.algorithms.ipeps_simple_update import (
    _simple_update_checkerboard_sweep,
)
from tenax.contraction.contractor import contract
from tenax.core._tensor_utils import scale_bond_axis

D = 3
GAUGE_TOL = 1e-13


def _simple_update(A, B, *, phases, rotate):
    """Run ``phases`` phases of the shipped checkerboard sweep; return the pair
    and the weights it stored.

    ``rotate`` picks the sublattice-rotated gate, which is the physical choice
    on a dense pair but does **not** conserve Sz, so it cannot be cast to a
    U(1)-Sz ``SymmetricTensor`` at all.
    """
    gate = heisenberg_gate()
    gate_t = _make_trotter_gate_tensor(
        sublattice_rotate_gate(gate) if rotate else gate, 0.05, site_tensor=A
    )
    # The sweep returns the four bonds itself now (#851); with the default
    # shared spectra ``h_AB is h_BA`` and ``v_AB is v_BA``, which is what this
    # used to rebuild by hand.
    return _simple_update_checkerboard_sweep(A, B, gate_t, D, phases)


def _direction(t):
    """``t`` rescaled to unit max-abs, for comparing states up to normalisation.

    Max-abs rather than the 2-norm because the 2-norm squares first: an
    SU-evolved state can have bond spectra spanning ~29 orders, the torus
    multiplies eight of them, and the resulting ~1e-172 tensor squares to
    1e-344 -- which underflows f64 to exactly zero and turns the comparison
    into ``nan``.
    """
    m = float(np.max(np.abs(t)))
    return t / m if m > 0.0 else t


def _two_site(gam_L, gam_R, leg_L, leg_R, lam):
    """``gam_L -- lam -- gam_R`` across one bond, every other leg left free.

    Gauge-sensitive by construction: a gauge that does not cancel between the
    two ends shows up here, and nothing else in the pair changes.
    """
    left = scale_bond_axis(gam_L, leg_L, lam).relabel(leg_L, "__shared")
    right = gam_R.relabels(
        {lab: f"{lab}_R" for lab in gam_R.labels() if lab != leg_R}
    ).relabel(leg_R, "__shared")
    return contract(left, right)


@pytest.mark.parametrize("kind", list(_PAIRS))
def test_the_bond_gauge_leaves_the_physical_state_untouched(kind):
    """The whole construction rests on this, so it is checked exactly."""
    A, B = _PAIRS[kind]()
    lam = jnp.array([1.0, 0.4, 0.1])
    weights = BondWeights(h_AB=lam, h_BA=lam, v_AB=lam, v_BA=lam)

    before = _two_site(A, B, "r", "l", lam)
    A2, B2, lam_new = _gauge_bond(
        A,
        B,
        "r",
        "l",
        _message(A, "A", "r", weights),
        _message(B, "B", "l", weights),
        lam,
    )
    after = _two_site(A2, B2, "r", "l", lam_new)

    # The returned weight is renormalised to max 1 (the simple-update
    # convention), which rescales what the pair represents by a scalar -- so the
    # state is preserved up to normalisation, and only the direction is
    # meaningful.  Comparing without this reports the scale factor (~3.1e+01
    # here) and hides whether the gauge itself is right.
    a = np.asarray(before.todense())
    b = np.asarray(after.todense())
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    rel = float(np.linalg.norm(b - a))
    assert rel < GAUGE_TOL, (
        f"{kind}: re-gauging changed the physical state by {rel:.3e}; a gauge "
        f"transformation must leave Gamma_L lambda Gamma_R exactly invariant"
    )
    # A gauge that silently collapsed sectors would also flatten the spectrum,
    # so pin that the weights are a real, non-degenerate spectrum.
    assert float(jnp.min(lam_new)) > 0.0
    assert float(jnp.max(lam_new)) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", list(_PAIRS))
def test_the_whole_solve_preserves_a_state_with_nontrivial_weights(kind):
    """The invariance guarantee, end to end and on all four bonds at once.

    ``_gauge_bond`` is checked above with a nontrivial ``lambda``, but only one
    bond in isolation; the solve itself was only checked for structure, on
    unweighted bonds.  That gap is what let ``weights`` default to one: with
    ``lambda = 1`` every way of mishandling the incoming weights preserves the
    state anyway, so nothing here could see it (#870).
    """
    A, B = _PAIRS[kind]()
    w = BondWeights(
        h_AB=jnp.array([1.0, 0.4, 0.1]),
        h_BA=jnp.array([1.0, 0.6, 0.2]),
        v_AB=jnp.array([1.0, 0.3, 0.05]),
        v_BA=jnp.array([1.0, 0.7, 0.3]),
    )

    before = _torus_2x2(A, B, w)
    A2, B2, w2, info = bp_gauge_checkerboard(A, B, w, max_iter=400, tol=1e-13)
    assert info.converged, f"{kind}: BP did not converge ({info.residual:.2e})"
    after = _torus_2x2(A2, B2, w2)

    # Gamma and lambda are both renormalised each sweep, so only the direction
    # of the torus tensor is meaningful -- as in the single-bond check above.
    before = before / np.linalg.norm(before)
    after = after / np.linalg.norm(after)
    rel = float(np.linalg.norm(after - before))
    assert rel < GAUGE_TOL, (
        f"{kind}: the solve changed the physical state by {rel:.3e}; every step "
        f"is meant to be a gauge transformation of Gamma_A lambda Gamma_B"
    )


@pytest.mark.parametrize("kind", list(_PAIRS))
def test_the_solve_converges_and_hands_back_the_same_tensor_structure(kind):
    """Labels, axis order and flows must survive, or callers break silently."""
    A, B = _PAIRS[kind]()
    A2, B2, weights, info = bp_gauge_checkerboard(
        A, B, BondWeights.ones(D, D), max_iter=400, tol=1e-13
    )

    assert info.converged, f"{kind}: BP did not converge (residual {info.residual:.2e})"
    for original, gauged, tag in ((A, A2, "A"), (B, B2, "B")):
        assert gauged.labels() == original.labels(), f"{kind}/{tag}: axis order changed"
        assert [int(i.flow) for i in gauged.indices] == [
            int(i.flow) for i in original.indices
        ], f"{kind}/{tag}: flows changed"
        assert type(gauged) is type(original)
        # ``> 0``, not merely finite: an all-zero return is finite, and is what
        # an overflowed iterate normalises to, so a finiteness-only assertion
        # passes on the one output that is definitely wrong (#870).
        assert float(gauged.norm()) > 0.0, f"{kind}/{tag}: gauged to zero"
    for name in weights._fields:
        w = np.asarray(getattr(weights, name))
        assert np.all(np.isfinite(w)) and np.all(w >= 0.0)
        assert w.max() > 0.0, f"{kind}: {name} came back all zero"


def test_a_dense_pair_whose_flows_are_not_this_modules_keeps_its_own():
    """The structure check above, on a pair that does *not* start canonical.

    ``_gauge_bond`` stamps the flow its SVD produced on every virtual leg --
    ``IN`` on the left/upper end of each bond, ``OUT`` on the right/lower one --
    so the returned flows used to be that convention whatever the caller's were.
    Every fixture the test above runs on already uses it, so nothing could see
    the difference.

    ``_simple_update_checkerboard_sweep``'s output does not: it comes back with
    all five legs inverted relative to ``_wrap_as_dense_tensor`` (measured:
    ``u`` OUT->IN, ``d`` IN->OUT, ``l`` OUT->IN, ``r`` IN->OUT, ``phys``
    IN->OUT).  That is the pair this module exists to re-gauge, so "flows must
    survive" was false for the only caller that matters.

    Dense only.  On a ``SymmetricTensor`` the charges are load-bearing and
    rewriting the metadata would be #834's silent mis-pairing, so that path
    keeps the stamped flows it has always returned.
    """
    A, B, stored = _simple_update(*_dense_pair(), phases=40, rotate=True)
    flows = [int(i.flow) for i in A.indices]
    canonical = [int(i.flow) for i in _dense_pair()[0].indices]
    assert flows != canonical, (
        "the simple update's output now uses this module's flow convention, so "
        "this test no longer probes anything -- find another non-canonical pair "
        "or delete it, do not weaken it"
    )

    A2, B2, weights, info = bp_gauge_checkerboard(A, B, stored, max_iter=400, tol=1e-13)
    assert info.converged, f"BP did not converge ({info.residual:.2e})"

    for original, gauged, tag in ((A, A2, "A"), (B, B2, "B")):
        assert gauged.labels() == original.labels(), f"{tag}: axis order changed"
        assert [int(i.flow) for i in gauged.indices] == [
            int(i.flow) for i in original.indices
        ], f"{tag}: flows changed"

    # And restoring the metadata did not buy that by moving the state.
    rel = float(
        np.max(
            np.abs(
                _direction(_torus_2x2(A2, B2, weights))
                - _direction(_torus_2x2(A, B, stored))
            )
        )
    )
    assert rel < GAUGE_TOL, f"the solve moved the state by {rel:.3e}"


@pytest.mark.parametrize("phases", [1, 4, 8])
def test_an_su_evolved_symmetric_state_does_not_walk_out_of_f64(phases):
    """The iterate must stay representable, and a corpse must not be certified.

    Measured before the per-bond rescale: at ``phases=1`` this returned
    ``|A| = 0`` with all four weights zero and
    ``BPGaugeInfo(iterations=84, residual=0.0, converged=True)``.  ``X^-1`` is
    a pseudo-inverse square root, so a decaying spectrum inflates ``||Gamma||``
    without bound; rescaling once per sweep let four bonds compound and the
    iterate reached ``1e112`` by sweep 59, then ``inf``, then zero.

    The residual test cannot catch this on its own -- it fell monotonically to
    9.8e-12 on the way down, and once both weight sets are zero,
    ``norm(0 - 0) / max(0, 1e-300)`` is 0, which passes any tolerance.  So the
    assertions below are on the *state*, not on the report.

    The invariance check holds at full tolerance despite the spectra spanning
    ~29 orders here (measured: 4.2e-15, 1.2e-16, 4.4e-16 for 1, 4 and 8
    phases), so it is not weakened for these states -- only the *comparison*
    changes, to max-abs, since the torus underflows the 2-norm.
    """
    A, B, stored = _simple_update(*_symmetric_pair(), phases=phases, rotate=False)
    before = _torus_2x2(A, B, stored)

    A2, B2, weights, info = bp_gauge_checkerboard(A, B, stored, max_iter=400, tol=1e-13)

    assert float(A2.norm()) > 0.0 and float(B2.norm()) > 0.0, (
        f"{phases} phase(s): the solve returned a zero tensor and reported {info}"
    )
    for name in weights._fields:
        assert float(jnp.max(getattr(weights, name))) > 0.0, f"{name} all zero"
    assert info.converged, f"{phases} phase(s): {info}"

    # And it is still a gauge -- the guard must not be buying health by
    # quietly changing the state.
    rel = float(
        np.max(np.abs(_direction(_torus_2x2(A2, B2, weights)) - _direction(before)))
    )
    assert rel < GAUGE_TOL, f"{phases} phase(s): state moved by {rel:.3e}"


#: How far past its fixed point the #870 guard below has to run.  The state
#: converges at sweep ~60 and the hoisted-rescale overflow does not arrive until
#: sweep ~116, so anything that stops at the fixed point cannot see it.  150
#: clears the blow-up with margin without doubling the test's cost.
_HOIST_GUARD_SWEEPS = 150


def test_rescaling_once_per_sweep_instead_of_once_per_bond_still_overflows():
    """The #870 guard, run *past* the fixed point where the defect actually is.

    ``_sweep`` rescales after every **bond**, 8x per sweep, and the comment
    there says a cleanup that hoists it to once per sweep reintroduces the
    overflow.  That claim was untested: hoisting the two ``_rescale`` calls out
    of the ``for bond ...`` loop turns this whole file green, including
    ``test_an_su_evolved_symmetric_state_does_not_walk_out_of_f64``, which is
    the test nominally guarding it.

    The mechanism is not stale, the *guard* was.  Instrumented on this fixture:
    hoisted, ``peak ||Gamma||`` reaches ``inf`` and the first unhealthy sweep is
    116; per-bond, ``||Gamma||`` sits at exactly 1.000 for the whole run.  The
    existing test misses it because the solve **converges at sweep ~60** and
    returns 56 sweeps before the blow-up: it asks for the fixed point, and the
    defect lives past it.

    So this one deliberately does not converge -- ``tol=0.0`` forces all
    ``_HOIST_GUARD_SWEEPS`` sweeps -- and asserts every one of them was healthy.
    ``iterations`` is the discriminator rather than the returned norm, because
    the rollback hands back the *last good* iterate either way, so the state
    alone looks fine in both arms; only the sweep count says whether the solve
    walked out of f64 on the way.
    """
    A, B, stored = _simple_update(*_symmetric_pair(), phases=4, rotate=False)

    _, _, _, info = bp_gauge_checkerboard(
        A, B, stored, max_iter=_HOIST_GUARD_SWEEPS, tol=0.0
    )

    assert info.iterations == _HOIST_GUARD_SWEEPS, (
        f"only {info.iterations} of {_HOIST_GUARD_SWEEPS} sweeps stayed inside "
        f"f64 (residual {info.residual}).  The per-bond rescale in ``_sweep`` "
        f"bounds the compounding of ``X_inv``; if it has been hoisted to once "
        f"per sweep, this is #870 and the fix is to put it back, not to lower "
        f"the sweep count until the test passes"
    )
    assert info.residual != float("inf"), "a sweep was rejected as unrepresentable"


def test_a_state_whose_messages_underflow_is_rejected_by_the_real_predicate():
    """The health guard, reached through the solve rather than injected.

    Every other rollback test in this tree patches ``_sweep_is_healthy``, so
    nothing pins that the solve consults :func:`_is_representable` at all -- a
    driver that dropped the call entirely would keep them all green.

    The reachable death is **underflow**, not overflow.  ``_PINV_CUTOFF`` is a
    *relative* cutoff (``s > 1e-12 * max(s)``), so ``s_inv`` is capped at
    ``1e12 / max(s)`` and cannot run away to ``inf`` however fast the message
    spectrum decays.  What it does not stop is the whole message underflowing:
    here ``Gamma``'s O(1) weight sits at virtual index 1 on all four legs and
    ``lambda[1] = 1e-170``, so ``_message`` -- which multiplies three incoming
    weights in and then squares -- lands below 1e-308 and rounds to exactly
    zero.  The pseudo-inverse of a zero message is zero, ``Gamma`` becomes
    exactly zero, and ``_rescale`` leaves it there.

    Zero is the *absorbing fixed point*: ``norm(0 - 0) / max(0, 1e-300)`` is 0,
    which passes any tolerance.  This is what the ``n > 0`` clause exists for,
    and without it the solve would report a confident ``converged=True`` on a
    corpse (#870).  The input is a legal one -- it passes ``_validate_weights``
    -- so this is a state a caller could hand over, not a mutation of internals.
    """
    D_ = 2
    arr = np.full((D_, D_, D_, D_, 2), 1e-300)
    arr[1, 1, 1, 1, :] = 1.0  # the O(1) weight, on virtual index 1 of every leg
    A = _wrap_as_dense_tensor(jnp.asarray(arr))
    B = _wrap_as_dense_tensor(jnp.asarray(arr))
    lam = jnp.array([1.0, 1e-170])
    w = BondWeights(h_AB=lam, h_BA=lam, v_AB=lam, v_BA=lam)

    A2, B2, _, info = bp_gauge_checkerboard(A, B, w, max_iter=50, tol=1e-12)

    assert not info.converged, (
        "a state whose messages underflow to zero was certified converged; "
        "zero is an absorbing fixed point and reports residual 0"
    )
    assert info.iterations == 0, f"reported sweeps that were not healthy: {info}"
    assert info.residual == float("inf")
    # The rollback hands back a usable state -- here the rescaled input.
    assert float(A2.norm()) > 0.0 and float(B2.norm()) > 0.0


@pytest.mark.parametrize("kind", list(_PAIRS))
@pytest.mark.parametrize("target", ["gamma", "lambda"])
def test_an_overall_scale_on_the_input_changes_nothing(kind, target):
    """An unobservable prefactor must not decide whether the solve works.

    ``_message`` squares ``Gamma`` *and* multiplies three incoming weights into
    it, so an overall factor on either one under/overflowed the very first
    message and returned ``iterations=0, residual=inf`` on a state that solves
    in ~38 sweeps at unit scale.  Both are unobservable: a factor on ``Gamma``
    or on ``lambda`` rescales the whole state, and the output weights are
    max-normalised regardless.

    Rescaling ``Gamma`` by the Frobenius norm does not fix its half --
    ``||Gamma||`` squares before it sums, so it is itself 0 at 1e-200 and
    ``inf`` at 1e200, and the rescale silently does nothing.  Hence max-abs.

    Compared after a *fixed* number of sweeps rather than at the fixed point,
    so the whole trajectory is checked and not just where it lands.
    """
    A, B = _PAIRS[kind]()
    lam = jnp.array([1.0, 0.4, 0.1])
    base = BondWeights(h_AB=lam, h_BA=lam, v_AB=lam, v_BA=lam)
    sweeps = 15

    def run(scale):
        if target == "gamma":
            args = (A * scale, B * scale, base)
        else:
            args = (A, B, BondWeights(*(w * scale for w in base)))
        _, _, w, info = bp_gauge_checkerboard(*args, max_iter=sweeps, tol=0.0)
        return np.concatenate([np.asarray(x) for x in w]), info

    ref, ref_info = run(1.0)
    assert ref_info.iterations == sweeps
    assert np.max(ref) > 0.0, "the unit-scale reference is itself degenerate"

    for scale in (1e-200, 1e200):
        got, info = run(scale)
        assert info.iterations == sweeps, f"{kind}/{target} at {scale:.0e}: {info}"
        d = float(np.max(np.abs(got - ref)))
        assert d < 1e-13, (
            f"{kind}: scaling {target} by {scale:.0e} moved the weights {d:.3e}"
        )


def test_the_health_predicate_rejects_every_way_an_iterate_dies():
    """Unit-test on ``_is_representable``, because nothing else reaches it.

    After the per-bond rescale no state in this file overflows any more, so the
    guard is unreachable from the solve -- which means without this test it is
    unverified code that could be deleted with every test still green.  It is
    kept rather than deleted because the rescale bounds the *compounding*, not
    the per-bond factor: ``X^-1`` is still an unbounded pseudo-inverse.
    """
    A, _ = _dense_pair()
    healthy = {"h_AB": jnp.ones(D)}

    def ok(gam, weights):
        """The predicate as a Python ``bool``, and pinned to 0-d.

        It returns a **0-d jnp array** now, because the traced driver calls it
        inside a ``lax.while_loop``.  Truthiness of a 0-d array is well defined,
        so the assertions below would still read correctly -- but only as long
        as it stays 0-d, and a predicate that started returning one entry per
        bond would make ``assert not ...`` raise rather than fail.  Pin the
        shape here so that change is caught at its source.
        """
        got = _is_representable(gam, weights)
        assert got.shape == (), f"health predicate returned shape {got.shape}"
        return bool(got)

    assert ok({"A": A}, healthy)
    # Representable, not unit-scaled: an unobservable prefactor is not a defect,
    # and a norm-based predicate would reject both of these -- ||Gamma|| is 0 at
    # 1e-200 and inf at 1e200 because it squares before it sums.
    assert ok({"A": A * 1e-200}, healthy), "small scale rejected"
    assert ok({"A": A * 1e200}, healthy), "large scale rejected"
    assert not ok({"A": A * 0.0}, healthy), "zero Gamma accepted"
    assert not ok({"A": A * jnp.inf}, healthy), "inf Gamma accepted"
    assert not ok({"A": A * jnp.nan}, healthy), "nan Gamma accepted"
    assert not ok({"A": A}, {"h_AB": jnp.zeros(D)}), "zero lambda"
    assert not ok({"A": A}, {"h_AB": jnp.array([1.0, jnp.nan, 0.1])}), (
        "nan lambda accepted"
    )
    # The four bonds arrive as a ``BondWeights`` from ``_sweep`` and as a plain
    # dict from here, so the predicate has to read both.
    assert ok({"A": A}, BondWeights.ones(D, D))
    assert not ok({"A": A}, BondWeights.ones(D, D)._replace(v_BA=jnp.zeros(D)))


# ``usefixtures`` rather than an argument: ``retraced`` is imported into this
# module's namespace so pytest can find it, and naming it in the signature too
# would shadow that import (ruff F811).  It is used for its side effect only.
@pytest.mark.usefixtures("retraced")
@pytest.mark.parametrize("kind", list(_PAIRS))
def test_an_unrepresentable_sweep_rolls_back_and_is_not_certified(monkeypatch, kind):
    """The solve must honour the predicate: return the last good gauge, say so.

    Driven by a mocked health signal rather than by finding an input that still
    overflows -- what is being tested is the loop's *reaction*, and a real
    overflow would only make the test hostage to whichever state still triggers
    it.

    Injected at ``_sweep_is_healthy``, keyed on the **sweep index**, rather than
    by counting calls.  The dense pair now runs inside a ``lax.while_loop``,
    whose body is traced exactly once, so a host-side counter would fire once
    and never reach the third sweep -- it would silently test nothing.  The
    index is what both drivers have (the Python loop variable; the carry's
    ``done`` slot), so one injection covers the traced and the eager path, and
    running this on both pair kinds is what pins that they react identically.
    """
    import tenax.algorithms.ipeps_bp_gauge as bp

    A, B = _PAIRS[kind]()
    w0 = BondWeights.ones(D, D)
    before = _torus_2x2(A, B, w0)

    monkeypatch.setattr(bp, "_sweep_is_healthy", lambda gam, weights, sweep: sweep < 2)
    A2, B2, weights, info = bp.bp_gauge_checkerboard(A, B, w0, max_iter=50, tol=0.0)

    assert not info.converged, "an unrepresentable iterate was certified"
    assert info.residual == float("inf")
    assert info.iterations == 2, "reported sweeps it did not complete"

    # The rollback must hand back a usable state, not the corpse -- and the two
    # completed sweeps are exact gauges, so it is still the same state.
    assert float(A2.norm()) > 0.0 and float(B2.norm()) > 0.0
    rel = float(
        np.max(np.abs(_direction(_torus_2x2(A2, B2, weights)) - _direction(before)))
    )
    assert rel < GAUGE_TOL, f"rollback returned a different state ({rel:.3e})"


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(jnp.zeros(D), id="all-zero"),
        pytest.param(jnp.array([1.0, jnp.nan, 0.1]), id="non-finite"),
    ],
)
def test_a_weight_vector_that_is_not_a_state_is_rejected(bad):
    """Zero weights are the absorbing fixed point, so they cannot be an input.

    ``residual`` divides by ``norm(lambda_old)``, guarded at ``1e-300``: given
    all-zero input weights the ratio is ``0/1e-300 = 0``, so the very first
    sweep would report a perfect solve.
    """
    A, B = _dense_pair()
    w = BondWeights.ones(D, D)._replace(h_BA=bad)
    with pytest.raises(ValueError, match="usable bond weight"):
        bp_gauge_checkerboard(A, B, w)


def test_bp_resolves_the_two_horizontal_bonds_separately():
    """#851's premise, measured: away from the fixed point h_AB != h_BA.

    The shipped simple update stores one spectrum for both, so it cannot
    represent this at all.
    """
    A, B = _symmetric_pair()
    _, _, weights, info = bp_gauge_checkerboard(
        A, B, BondWeights.ones(D, D), max_iter=400, tol=1e-13
    )
    assert info.converged
    h_AB = np.asarray(weights.h_AB)
    h_BA = np.asarray(weights.h_BA)
    rel = float(np.linalg.norm(h_AB - h_BA) / np.linalg.norm(h_AB))
    assert rel > 1e-2, (
        f"the two horizontal bonds came back equal to {rel:.2e}; on this input "
        f"they are inequivalent and BP should resolve them (#851)"
    )


def test_the_weights_simple_update_stores_are_not_bp_self_consistent():
    """Why this module exists: the stored weights have drifted from the spectra.

    A non-unitary gate on a neighbouring bond changes this bond's Schmidt
    values, and simple update never recomputes them -- the defect TeNPy's
    ``update_bond_imag`` and YASTN's ``EnvBP.post_truncation_`` are built to
    avoid (#869).  If this assertion ever fails, the module has lost its
    motivation and should be reconsidered, not "fixed".
    """
    A, B, stored = _simple_update(*_dense_pair(), phases=400, rotate=True)
    lam_h = stored.h_AB
    _, _, weights, info = bp_gauge_checkerboard(A, B, stored, max_iter=400, tol=1e-13)
    assert info.converged, f"BP did not converge (residual {info.residual:.2e})"

    drift = float(
        np.linalg.norm(np.asarray(weights.h_AB) - np.asarray(lam_h))
        / np.linalg.norm(np.asarray(lam_h))
    )
    assert drift > 1e-2, (
        f"the stored spectrum and the BP-consistent one agree to {drift:.2e}; "
        f"simple update's weights were expected to have drifted (#869)"
    )
