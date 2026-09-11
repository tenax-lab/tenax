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

import tenax.algorithms.ipeps_bp_gauge as bp_mod
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
from tenax.algorithms.ipeps_gauge import gauge_fix
from tenax.algorithms.ipeps_simple_update import (
    _simple_update_checkerboard_sweep,
)
from tenax.contraction.contractor import contract
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.index import FlowDirection

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


def _square_pair_at(dtype, D: int = 2, seed: int = 0):
    """A dense pair at an explicit precision, built without going through
    ``_wrap_as_dense_tensor``, which fixes the dtype to the caller's array."""
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor, TensorIndex

    sym = U1Symmetry()

    def leg(n, label, flow):
        return TensorIndex.from_charges(
            sym, np.zeros(n, dtype=np.int32), flow, label=label
        )

    def site(key):
        arr = jnp.asarray(
            np.random.default_rng(key).normal(size=(D, D, D, D, 2)), dtype=dtype
        )
        return DenseTensor(
            arr,
            (
                leg(D, "u", FlowDirection.OUT),
                leg(D, "d", FlowDirection.IN),
                leg(D, "l", FlowDirection.OUT),
                leg(D, "r", FlowDirection.IN),
                leg(2, "phys", FlowDirection.IN),
            ),
        )

    return site(seed), site(seed + 1)


@pytest.mark.parametrize(
    ("dtype", "real"),
    [
        (jnp.float32, jnp.float32),
        (jnp.float64, jnp.float64),
        (jnp.complex64, jnp.float32),
        (jnp.complex128, jnp.float64),
    ],
    ids=["f32", "f64", "c64", "c128"],
)
def test_the_traced_loop_takes_the_precisions_the_eager_loop_takes(dtype, real):
    """A carry may not mix precisions, and the pair's dtype decides which.

    tenax enables x64 globally, so ``BondWeights.ones`` and ``gauge_fix``'s
    identity weights are ``float64``.  Against a ``float32`` pair the first
    sweep promoted the candidate tensors to ``float64`` while the carry's tensor
    slot stayed ``float32``, and ``lax.while_loop`` rejected the body with
    "carry input and carry output must have equal types" -- so **every** dense
    ``float32`` pair failed.

    The eager driver rebinds Python names each sweep and never noticed, which is
    what makes this a regression introduced with the traced loop rather than a
    standing limitation, and why both drivers are exercised here: the reference
    must not accept a mix the traced path rejects.

    Weights are singular values, so a complex pair wants the *real* dtype behind
    it -- ``float32`` for ``complex64``.  Watched failing on ``f32``/``c64``
    before the cast in ``_prepare``; ``f64``/``c128`` pass either way and are
    the controls that keep this honest.
    """
    A, B = _square_pair_at(dtype)

    A_g, B_g, w, info = gauge_fix(A, B)
    assert info.converged, f"gauge_fix did not converge at {np.dtype(dtype).name}"
    for name, t in (("A", A_g), ("B", B_g)):
        assert t.todense().dtype == dtype, (
            f"{name} came back as {t.todense().dtype}, not {np.dtype(dtype).name}; "
            f"the gauge silently changed the caller's precision"
        )
    for bond in ("h_AB", "h_BA", "v_AB", "v_BA"):
        got = getattr(w, bond).dtype
        assert got == real, (
            f"weight {bond} came back as {got}, expected {np.dtype(real).name} -- "
            f"weights are singular values and belong at the pair's real precision"
        )

    # Both drivers, same input: the eager one is the reference the traced one is
    # checked against, so a precision it accepts must not fail the other.
    for traced in (True, False):
        monkey = bp_mod._use_traced_loop
        bp_mod._use_traced_loop = lambda *a, **k: traced
        try:
            out = bp_gauge_checkerboard(
                A, B, BondWeights.ones(2, 2), max_iter=20, tol=1e-10
            )
        finally:
            bp_mod._use_traced_loop = monkey
        assert out[0].todense().dtype == dtype, (
            f"the {'traced' if traced else 'eager'} driver returned "
            f"{out[0].todense().dtype} for a {np.dtype(dtype).name} pair"
        )


@pytest.mark.parametrize(
    ("dtype_A", "dtype_B", "common"),
    [
        (jnp.float32, jnp.float64, jnp.float64),
        (jnp.float64, jnp.float32, jnp.float64),
        (jnp.complex64, jnp.float32, jnp.complex64),
    ],
    ids=["f32+f64", "f64+f32", "c64+f32"],
)
def test_a_mixed_precision_pair_is_brought_to_one_dtype(dtype_A, dtype_B, common):
    """Matching the weights to the pair is not enough when the pair disagrees.

    ``A`` at ``float32`` beside ``B`` at ``float64`` promotes ``A``'s candidate
    inside the sweep while its ``lax.while_loop`` carry slot stays ``float32``,
    which is the same carry rejection as a uniformly-``float32`` pair, one level
    in.  Fixing only the weights left this reachable, so it gets its own cells.

    The common dtype is the promoted one, so the cast only ever widens and the
    result is exact.
    """
    A, _ = _square_pair_at(dtype_A, seed=0)
    _, B = _square_pair_at(dtype_B, seed=4)

    A_g, B_g, w, info = gauge_fix(A, B)

    assert info.converged, (
        f"gauge_fix did not converge on a {np.dtype(dtype_A).name}/"
        f"{np.dtype(dtype_B).name} pair: {info}"
    )
    for name, t in (("A", A_g), ("B", B_g)):
        assert t.dtype == common, (
            f"{name} came back as {t.dtype}, not the promoted {np.dtype(common).name}"
        )
    real = jnp.finfo(common).dtype
    assert w.h_AB.dtype == real, (
        f"weights came back as {w.h_AB.dtype}, expected {np.dtype(real).name}"
    )


def test_prepare_reads_dtypes_without_densifying():
    """``_prepare`` must not call ``todense()`` to learn a dtype.

    ``Tensor.dtype`` is right there, and densifying costs a full ``D**4 * d``
    array per site -- on the ``SymmetricTensor`` path that discards block
    sparsity before the solve has started, which is exactly what CLAUDE.md's
    ``todense()`` rule exists to prevent.  An earlier revision of the
    carry-dtype fix did precisely this, so the rule gets a test rather than a
    comment.

    Fails by raising out of the patched ``todense``, not by asserting after the
    fact, so it cannot pass because a later line happened not to look.
    """
    A, B = _symmetric_pair(D=3)
    calls: list[str] = []

    real_todense = type(A).todense

    def spy(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return real_todense(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(type(A), "todense", spy)
        bp_mod._prepare(A, B, BondWeights.ones(3, 3))

    assert not calls, (
        f"_prepare densified {len(calls)} tensor(s) ({calls}); Tensor.dtype "
        f"gives the dtype without materialising D**4 * d entries per site"
    )


# --------------------------------------------------------------------- #
# The traced symmetric driver (#882 Phase 3)                             #
# --------------------------------------------------------------------- #


def _nontrivial_weights(D: int = 3) -> BondWeights:
    return BondWeights(
        h_AB=jnp.array([1.0, 0.4, 0.1][:D]),
        h_BA=jnp.array([1.0, 0.6, 0.2][:D]),
        v_AB=jnp.array([1.0, 0.3, 0.05][:D]),
        v_BA=jnp.array([1.0, 0.7, 0.3][:D]),
    )


def test_the_traced_and_eager_drivers_agree_on_a_symmetric_pair(monkeypatch):
    """The wiring's whole contract: tracing must not move the answer.

    Same input through both drivers.  The verdict, the sweep count, the
    spectrum and the physical state all have to match -- the sweep body is
    shared verbatim and, since the sector-mode SVD, so is every decomposition
    code path, which is what makes exact sweep-count agreement a fair
    assertion rather than an aspiration.  Weights are compared sorted: both
    drivers emit the sector-grouped layout today, but the contract is the
    spectrum, not the layout.
    """
    A, B = _symmetric_pair()
    w = _nontrivial_weights()
    before = _direction(_torus_2x2(A, B, w))

    results = {}
    for traced in (True, False):
        monkeypatch.setattr(bp_mod, "_use_traced_loop", lambda *a, **k: traced)
        results[traced] = bp_gauge_checkerboard(A, B, w, max_iter=400, tol=1e-13)

    (A_t, B_t, w_t, info_t), (A_e, B_e, w_e, info_e) = results[True], results[False]
    assert info_t.converged == info_e.converged
    assert info_t.iterations == info_e.iterations, (
        f"the drivers took different trajectories: traced {info_t}, eager {info_e}"
    )
    for bond in w._fields:
        st = np.sort(np.asarray(getattr(w_t, bond)))
        se = np.sort(np.asarray(getattr(w_e, bond)))
        assert st == pytest.approx(se, abs=1e-12), f"{bond} spectrum differs"
    for tag, X, wx in (("traced", (A_t, B_t), w_t), ("eager", (A_e, B_e), w_e)):
        drift = float(np.max(np.abs(_direction(_torus_2x2(*X, wx)) - before)))
        assert drift < GAUGE_TOL, f"{tag} driver moved the state by {drift:.3e}"


def test_the_canonical_relayout_is_the_same_state():
    """``_canonical_symmetric_layout`` is a relabel, not a transformation.

    Checked on both caller conventions it has to absorb: the helpers' pair
    (this module's flows already, charges merely unsorted) and a
    simple-update-evolved pair (flows inverted, so every leg is dual-relabeled
    -- the sweep itself emits ``[-2, -1, 2]``/IN for a ``[-2, 1, 2]``/OUT
    caller, and the relayout has to land on the same reading).  Also
    idempotent: canonical input comes back metadata-identical.
    """
    su_A, su_B, su_w = _simple_update(*_symmetric_pair(), phases=4, rotate=False)
    cases = {
        "module-flow": (*_symmetric_pair(), _nontrivial_weights()),
        "su-evolved": (su_A, su_B, su_w),
    }
    for tag, (A, B, w) in cases.items():
        gam, wp = bp_mod._prepare(A, B, w)
        before = _direction(_torus_2x2(gam["A"], gam["B"], wp))

        canon, wc = bp_mod._canonical_symmetric_layout(gam, wp)
        drift = float(
            np.max(np.abs(_direction(_torus_2x2(canon["A"], canon["B"], wc)) - before))
        )
        assert drift < 1e-14, f"{tag}: the relayout moved the state by {drift:.3e}"

        again, wc2 = bp_mod._canonical_symmetric_layout(canon, wc)
        for s in ("A", "B"):
            assert again[s].indices == canon[s].indices, f"{tag}: not idempotent"
        for bond in wc._fields:
            assert np.array_equal(
                np.asarray(getattr(wc2, bond)), np.asarray(getattr(wc, bond))
            ), f"{tag}: weights moved on the second pass"

        # A wrong-but-self-consistent relabel would survive both checks
        # above -- the torus is invariant under any consistent relabel, and
        # a wrong reading can be idempotent.  What it cannot survive is the
        # sweep: the carry needs the relayout to be the sweep's own fixed
        # point, so assert that directly.
        swept, _ = bp_mod._sweep(dict(canon), wc)
        for s in ("A", "B"):
            out = bp_mod._restore_caller_structure(swept[s], canon[s])
            assert out.indices == canon[s].indices, (
                f"{tag}/{s}: one sweep left the canonical layout -- the "
                f"relayout is not the sweep's fixed point, so the traced "
                f"carry could not hold it"
            )


@pytest.mark.usefixtures("retraced")
def test_the_flow_inverted_su_pair_stays_on_the_traced_path(monkeypatch):
    """The dual-relabel path -- the headline case -- must not fall back.

    A simple-update-evolved pair carries the opposite flow convention, so
    every virtual leg takes the full canonicalization: dual relabel, sort,
    weight permutation.  If any of that ever raises
    ``_StructureNotTraceable``, the solve would still be *correct* through
    the eager fallback -- and silently 10^4 times slower, which is the
    regression this cell exists to catch.  ``retraced`` keeps the assertion
    honest: served from a warm cache, a refusal would never fire either.
    """
    A, B, stored = _simple_update(*_symmetric_pair(), phases=4, rotate=False)
    before = _direction(_torus_2x2(A, B, stored))

    def no_fallback(*a, **k):
        raise AssertionError("the SU-evolved pair fell back to the eager loop")

    monkeypatch.setattr(bp_mod, "_bp_solve_eager", no_fallback)
    A2, B2, w2, info = bp_gauge_checkerboard(A, B, stored, max_iter=400, tol=1e-13)
    assert info.converged, f"the traced solve did not converge: {info}"
    drift = float(np.max(np.abs(_direction(_torus_2x2(A2, B2, w2)) - before)))
    assert drift < GAUGE_TOL, f"the solve moved the state by {drift:.3e}"


@pytest.mark.usefixtures("retraced")
def test_a_slot_no_block_occupies_is_dropped_and_stays_on_the_traced_path(
    monkeypatch,
):
    """The #906 pair class must not cost the traced driver.

    A ``D >= 3`` simple-update evolution leaves legs counting a charge no
    block occupies, and the sweep's SVD -- which sees occupied sectors only
    -- shrinks the bond on contact.  A static carry cannot follow a shrink,
    so the canonical relayout drops the dead slot up front instead: no block
    references it, so nothing moves, and the pair this matters for most (the
    D=3 acceptance fixture) stays on the compiled path rather than falling
    back to the eager loop it just escaped.

    Built synthetically: charge 0's blocks on ``v_BA``'s two ends (``B.d``,
    ``A.u``) are zeroed -- value-identical to deleting them -- and then
    structurally deleted.
    """
    from tenax.core.tensor import SymmetricTensor

    A, B = _symmetric_pair()

    def kill(t, leg):
        ax = t.labels().index(leg)
        blocks = {k: b for k, b in t.blocks.items() if k[ax] != 0}
        return SymmetricTensor._from_blocks_unchecked(blocks, t.indices)

    A_dead, B_dead = kill(A, "u"), kill(B, "d")
    w = _nontrivial_weights()
    before = _direction(_torus_2x2(A_dead, B_dead, w))

    gam, wp = bp_mod._prepare(A_dead, B_dead, w)
    canon, wc = bp_mod._canonical_symmetric_layout(gam, wp)
    dead_leg = canon["A"].indices[canon["A"].labels().index("u")]
    assert 0 not in dead_leg.charges.tolist(), (
        f"the dead slot survived the relayout: {dead_leg.charges.tolist()}"
    )
    assert len(np.asarray(wc.v_BA)) == len(dead_leg.charges), (
        "the weight vector did not shrink with its leg"
    )
    drift = float(
        np.max(np.abs(_direction(_torus_2x2(canon["A"], canon["B"], wc)) - before))
    )
    assert drift < 1e-14, f"dropping the dead slot moved the state by {drift:.3e}"

    # ... and the whole solve takes the traced driver, not the fallback.
    def no_fallback(*a, **k):
        raise AssertionError("the dead-slot pair fell back to the eager loop")

    monkeypatch.setattr(bp_mod, "_bp_solve_eager", no_fallback)
    A2, B2, w2, info = bp_gauge_checkerboard(A_dead, B_dead, w, max_iter=400, tol=1e-13)
    assert info.converged, f"the traced solve did not converge: {info}"
    drift = float(np.max(np.abs(_direction(_torus_2x2(A2, B2, w2)) - before)))
    assert drift < GAUGE_TOL, f"the solve moved the state by {drift:.3e}"


@pytest.mark.usefixtures("retraced")
def test_a_zero_sweep_solve_returns_the_callers_own_slot_structure(monkeypatch):
    """The carry's relabel must not leak on a path where no sweep ever ran.

    The traced carry canonicalizes its input -- sorted charges, module
    flows, dead slots dropped.  After one accepted sweep that structure is
    what the sweep itself would have stamped, eager or traced; after ZERO
    accepted sweeps the relabel would be the only change, and it is
    caller-visible.  Watched failing on the D=4 seed-2 SU trajectory: its
    gauge rejects the first sweep, the dropped dead slot left a 3-slot bond
    on a 4-slot pair, and ``_su_evolve``'s ``max_D`` uniformity check raised
    where the eager driver's identical rejection sails through with a
    warning.

    Forced here by patching the health gate shut (hence ``retraced``), on a
    pair carrying a dead slot so the drop would be visible if it leaked.
    """
    from tenax.core.tensor import SymmetricTensor

    def kill(t, leg):
        ax = t.labels().index(leg)
        blocks = {k: b for k, b in t.blocks.items() if k[ax] != 0}
        return SymmetricTensor._from_blocks_unchecked(blocks, t.indices)

    A, B = _symmetric_pair()
    A, B = kill(A, "u"), kill(B, "d")
    w = _nontrivial_weights()

    monkeypatch.setattr(bp_mod, "_sweep_is_healthy", lambda *a, **k: jnp.asarray(False))

    A2, B2, w2, info = bp_gauge_checkerboard(A, B, w, max_iter=8, tol=1e-13)
    assert info.iterations == 0 and not info.converged
    for tag, before, after in (("A", A, A2), ("B", B, B2)):
        for leg in "udlr":
            n_in = len(before.indices[before.labels().index(leg)].charges)
            n_out = len(after.indices[after.labels().index(leg)].charges)
            assert n_out == n_in, (
                f"{tag}.{leg}: a zero-sweep solve changed the leg from "
                f"{n_in} to {n_out} slots -- the carry's relabel leaked"
            )
    for bond in w._fields:
        assert len(np.asarray(getattr(w2, bond))) == len(
            np.asarray(getattr(w, bond))
        ), f"{bond}: the weight vector changed length on a zero-sweep solve"

    # ... and through gauge_fix, whose traced route absorbs before returning.
    A3, B3, _w3, info3 = gauge_fix(A, B, max_iter=8, tol=1e-13)
    assert info3.iterations == 0 and not info3.converged
    for tag, before, after in (("A", A, A3), ("B", B, B3)):
        for leg in "udlr":
            n_in = len(before.indices[before.labels().index(leg)].charges)
            n_out = len(after.indices[after.labels().index(leg)].charges)
            assert n_out == n_in, f"gauge_fix {tag}.{leg}: {n_in} -> {n_out} slots"


@pytest.mark.usefixtures("retraced")
def test_a_pair_the_carry_cannot_hold_falls_back_to_the_eager_loop(monkeypatch):
    """The traced driver's refusal is a dispatch, not a failure.

    ``_StructureNotTraceable`` is raised at trace time; both entry points
    must catch it and hand the pair to the eager loop, which represents the
    same physics.  The trigger is injected rather than constructed -- a pair
    that genuinely defeats the canonicalization also defeats the eager
    positional-weight convention, so no honest fixture reaches the fallback
    today; the fallback exists for the structures we have not met yet.

    Two things keep this cell honest, both watched being necessary:
    ``retraced``, because the injection fires at *trace* time and an earlier
    cell has already compiled this exact key -- served from that cache, the
    refusal never runs and every assertion here passes off the traced
    result; and the eager spy, which turns "the fallback ran" from an
    assumption into an assertion.
    """
    eager_runs = []
    real_eager = bp_mod._bp_solve_eager

    def spy(*args, **kwargs):
        eager_runs.append(1)
        return real_eager(*args, **kwargs)

    def refuse(gam, weights):
        raise bp_mod._StructureNotTraceable("injected: carry cannot hold this pair")

    monkeypatch.setattr(bp_mod, "_bp_solve_eager", spy)
    monkeypatch.setattr(bp_mod, "_canonical_symmetric_layout", refuse)

    A, B = _symmetric_pair()
    w = _nontrivial_weights()
    before = _direction(_torus_2x2(A, B, w))

    A2, B2, w2, info = bp_gauge_checkerboard(A, B, w, max_iter=400, tol=1e-13)
    assert eager_runs, (
        "the eager fallback never ran -- the traced call was served from a "
        "stale jit cache and this cell asserted nothing about the fallback"
    )
    assert info.converged, f"the fallback did not converge: {info}"
    drift = float(np.max(np.abs(_direction(_torus_2x2(A2, B2, w2)) - before)))
    assert drift < GAUGE_TOL, f"the fallback moved the state by {drift:.3e}"

    # ... and gauge_fix's own traced route falls back the same way.  The
    # helpers' pair doubles as an absorbed-form pair (its implicit weights
    # are ones), which is the form gauge_fix takes.
    runs_before_gauge_fix = len(eager_runs)
    A3, B3, w3, info3 = gauge_fix(A, B, max_iter=400, tol=1e-10)
    assert len(eager_runs) > runs_before_gauge_fix, (
        "gauge_fix's fallback never reached the eager driver"
    )
    assert info3.converged, f"gauge_fix's fallback did not converge: {info3}"


# --------------------------------------------------------------------- #
# A bond weight vector that changes length mid-solve (#904)              #
# --------------------------------------------------------------------- #


def test_the_residual_survives_a_bond_whose_sector_count_changes():
    """A weight vector carries one entry per **non-empty block**, not per slot.

    So it changes length whenever a charge sector dies or revives, and
    ``_residual`` is the one place two sweeps' vectors meet.  Before #904 it
    subtracted them directly and died four frames down::

        TypeError: sub got incompatible shapes for broadcasting: (2,), (3,)

    **The state is not broken when this happens**, which is what makes padding
    the right answer rather than a papering-over.  ``_gauge_bond`` takes
    ``lam_new`` from a block-sparse ``svd`` and hands back tensors whose bond
    leg *is* that same new bond, so the leg and the weights shrink together.
    Measured on a U(1)-Sz ``D=3`` pair, sweep 56: ``v_BA`` went 3 -> 2 and
    ``Gamma_A.u``/``Gamma_B.d`` both came back at dim 2 on charges
    ``[-1, 1]`` -- and ``v_BA`` is exactly ``A.u <-> B.d``.

    It is also **transient**: the sector revived before the solve finished, and
    all four bonds ended at length 3.  The old code raised on a shape change it
    would have recovered from, which is the same behaviour
    ``_ctm_tensor_convergence._ctm_sv_diff`` already pads for (#670).

    This is **not** the #834 hazard.  Nothing indexes ``new`` against ``old``;
    each weight vector is only ever used with the leg it came back with, and
    this comparison is a convergence *indicator* -- a vector that changed shape
    must read as "still moving", which zero-padding gives it.
    """
    short = BondWeights(
        h_AB=jnp.asarray([1.0, 0.5]),
        h_BA=jnp.asarray([1.0, 0.5, 0.25]),
        v_AB=jnp.asarray([1.0, 0.5, 0.25]),
        v_BA=jnp.asarray([1.0, 0.5, 0.25]),
    )
    long = BondWeights(
        h_AB=jnp.asarray([1.0, 0.5, 0.25]),
        h_BA=jnp.asarray([1.0, 0.5, 0.25]),
        v_AB=jnp.asarray([1.0, 0.5, 0.25]),
        v_BA=jnp.asarray([1.0, 0.5, 0.25]),
    )

    # Both directions: a sector dying (long -> short) and reviving (short -> long).
    for tag, (new, old) in (("died", (short, long)), ("revived", (long, short))):
        r = float(bp_mod._residual(new, old))
        assert np.isfinite(r), f"{tag}: residual is {r}, not finite"
        assert r > 0.0, (
            f"{tag}: residual is exactly {r}, so a bond that changed shape "
            f"reads as converged -- the padding must make the dropped entry "
            f"register as a difference, not cancel it"
        )

    # And the unchanged case is untouched: identical weights read exactly zero,
    # so the padding cannot be manufacturing a difference where there is none.
    assert float(bp_mod._residual(long, long)) == 0.0, (
        "identical weights no longer read as a zero residual; the padding is "
        "perturbing the comparison it is supposed to leave alone"
    )
