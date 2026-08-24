"""The CTM convergence criterion must not certify what it cannot see (#898).

`_ctm_sv_diff` compares the corner spectrum **normalised by its sum**. That
normalisation is deliberate — under `renormalize=True` the absolute corner
scale is meaningless, so a scale-free comparison is the only one that carries
across sweeps. But it has a blind spot that is structural rather than
numerical: a rank-1 spectrum normalises to `[1, 0, ..., 0]` *whatever* the
environment is doing, so two completely different environments compare equal
to within one ulp.

The loop that reads it then exits on sweep 2 or 3 and reports success. Measured
before the fix: the returned energy was bit-identical at `max_iter` 60/120/300/
400/800, at `conv_tol` 1e-9/1e-12/1e-14 and at `chi` 8/12/24/48 — **8.8e-3
above** the fixed point the same loop reaches by sweep 41. There was no knob a
caller could turn to notice.

These tests are written against that mechanism, not against an energy. A frozen
number from a collapsed environment is a snapshot of a sweep index (#898), so
asserting one would re-freeze the bug.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_sv_diff,
    _spectrum_can_show_change,
)

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------- #
# The blind spot itself                                                  #
# --------------------------------------------------------------------- #


#: ``ctm_tensor``/``ctm_tensor_2site``'s default. The operative question is not
#: "is the diff zero" but "does the loop certify", and this is where that is
#: decided.
_DEFAULT_CONV_TOL = 1e-8


def _raw_sum_normalised_diff(a, b):
    """What `_ctm_sv_diff` computed before #898 — the blind comparison itself.

    Recomputed here rather than imported, so these tests pin the *mechanism*
    the guard protects against even after the guard makes it unreachable.
    """
    n = max(a.shape[0], b.shape[0])
    a = jnp.pad(a, (0, n - a.shape[0]))
    b = jnp.pad(b, (0, n - b.shape[0]))
    a = a / (jnp.sum(a) + 1e-15)
    b = b / (jnp.sum(b) + 1e-15)
    return float(jnp.max(jnp.abs(a - b)))


@pytest.mark.parametrize("s1,s2", [(1.0, 1e10), (1e-3, 1e7), (1e5, 1e5), (1e-4, 1.0)])
def test_two_rank_one_corners_now_report_inf_instead_of_agreement(s1, s2):
    """The fix, and the defect it replaces, in one place.

    `inf` is not a sentinel: on a rank-1 corner the true difference between the
    two environments is genuinely unbounded, because the sum-normalised
    spectrum carries no information about them. Every one of the nine loops
    that reads this already compares against `conv_tol`, so all nine fail
    closed with no change at the call site.
    """
    a = jnp.asarray([s1, 0.0, 0.0, 0.0])
    b = jnp.asarray([s2, 0.0, 0.0, 0.0])

    # The fix.
    assert float(_ctm_sv_diff(a, b)) == float("inf")

    # The defect it protects against: the raw comparison certifies, at every
    # magnitude, however far apart the two corners really are. The residue it
    # does leave is the `+1e-15` guard (about `1e-15 / min(scale)`), which
    # tracks absolute scale — the very thing `renormalize=True` exists to make
    # irrelevant.
    assert _raw_sum_normalised_diff(a, b) < _DEFAULT_CONV_TOL

    # Healthy corners at the same magnitudes are unaffected by the guard and
    # separated immediately, so the blindness belongs to the rank, not the scale.
    c = jnp.asarray([s1, 0.5 * s1, 0.1 * s1, 0.01 * s1])
    d = jnp.asarray([s2, 0.9 * s2, 0.8 * s2, 0.7 * s2])
    assert float(_ctm_sv_diff(c, d)) > 1e-2
    assert _spectrum_can_show_change(a) is False
    assert _spectrum_can_show_change(c) is True


def test_a_healthy_comparison_is_left_exactly_as_it_was():
    """The guard must be inert wherever the criterion could already see.

    Otherwise it would be trading one wrong answer for another, and every
    converged CTM in the suite would shift.
    """
    a = jnp.asarray([1.0, 0.4, 0.2, 0.05])
    b = jnp.asarray([1.0, 0.4000001, 0.2, 0.05])
    assert float(_ctm_sv_diff(a, b)) == pytest.approx(
        _raw_sum_normalised_diff(a, b), rel=0, abs=0
    )


def test_a_corner_smaller_than_the_epsilon_guard_was_already_failing_CLOSED():
    """The guard's other regime, pinned so nobody "tidies" it into blindness.

    Once `sum(sv)` drops below the `1e-15` in the denominator the raw
    comparison returns ~1.0 — the loop keeps sweeping, which is the *safe*
    direction and the opposite of the #898 failure. The rank guard now covers
    it too, so the loop is protected from both sides.
    """
    tiny = jnp.asarray([1e-30, 0.0, 0.0, 0.0])
    normal = jnp.asarray([1.0, 0.0, 0.0, 0.0])

    assert _raw_sum_normalised_diff(normal, tiny) > 0.5
    assert float(_ctm_sv_diff(normal, tiny)) == float("inf")
    assert _spectrum_can_show_change(tiny) is False


def test_the_criterion_survives_tracing():
    """One of the nine loops runs inside ``jax.lax.while_loop`` (#898).

    The first version of this guard branched on the predicate with a Python
    ``if``, which raises ``TracerBoolConversionError`` the moment the spectrum
    is a tracer -- and the eager tests all passed, because none of them traces.
    Both arms of the ``jnp.where`` are cheap, so there is no reason to branch.
    """

    def body(carry):
        sv, i = carry
        _ = _ctm_sv_diff(sv, sv * 1.01)
        return (sv, i + 1)

    for sv in ([1.0, 0.5, 0.2], [1.0, 0.0, 0.0]):
        jax.lax.while_loop(lambda c: c[1] < 2, body, (jnp.asarray(sv), 0))

    # And jit gives the same answers as eager, in both regimes.
    healthy = (jnp.asarray([1.0, 0.5, 0.2]), jnp.asarray([1.0, 0.51, 0.2]))
    blind = (jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([9.0, 0.0, 0.0]))
    assert float(jax.jit(_ctm_sv_diff)(*healthy)) == float(_ctm_sv_diff(*healthy))
    assert float(jax.jit(_ctm_sv_diff)(*blind)) == float("inf")
    assert float(_ctm_sv_diff(*blind)) == float("inf")


@pytest.mark.parametrize(
    "sv,expected,why",
    [
        ([1.0, 0.5, 0.25, 0.1], True, "healthy full-rank corner"),
        ([1.0, 1e-14, 0.0, 0.0], False, "rank 1: second value is numerical dust"),
        ([1.0, 0.0, 0.0, 0.0], False, "rank 1 exactly"),
        ([1.0], False, "a single value cannot show a shape change"),
        ([0.0, 0.0], False, "zero corner (#666)"),
        ([1.0, 1e-9, 0.0], True, "1e-9 is above the 1e-10 relative cutoff"),
    ],
)
def test_the_predicate_matches_the_collapse_detector(sv, expected, why):
    """`_spectrum_can_show_change` is deliberately `env_is_collapsed`'s twin.

    A corner the collapse detector calls dead is exactly one the convergence
    detector cannot see. If these two predicates ever drift apart, one of them
    is lying about the same environment.
    """
    assert _spectrum_can_show_change(jnp.asarray(sv)) is expected, why


def test_the_predicate_refuses_a_non_finite_corner():
    """NaN must not read as "fine to certify" — `nan > x` is False."""
    assert _spectrum_can_show_change(jnp.asarray([jnp.nan, 1.0])) is False
    assert _spectrum_can_show_change(jnp.asarray([])) is False


def test_the_predicate_is_scale_free():
    """It is a rank test, not a magnitude test."""
    base = np.array([1.0, 0.3, 0.01])
    for scale in (1e-30, 1.0, 1e30):
        assert _spectrum_can_show_change(jnp.asarray(scale * base)) is True
    for scale in (1e-30, 1.0, 1e30):
        assert (
            _spectrum_can_show_change(jnp.asarray(scale * np.array([1.0, 0.0])))
            is False
        )


# --------------------------------------------------------------------- #
# The loop must act on it                                                #
# --------------------------------------------------------------------- #


def _collapsing_pair():
    """A 2-site fixture whose 2x2 CTM corner collapses to rank 1.

    Imported from the direction-dependent-bonds test rather than rebuilt, so
    this file cannot drift into exercising a different state than the one #898
    was measured on.
    """
    from test_ctm_direction_dependent_bonds import _su_direction_dependent_pair

    return _su_direction_dependent_pair()


@pytest.mark.slow
def test_a_collapsed_corner_is_not_certified_and_the_budget_is_respected():
    """The behavioural claim: the answer must depend on `max_iter` again.

    Before the fix the returned environment was bit-identical at every budget,
    which is the observable signature of a criterion that exits on a quantity
    it cannot read. After it, more sweeps must do more work — that is the whole
    contract of `max_iter`.
    """
    import tenax
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site

    A, B = _collapsing_pair()
    gate = tenax.heisenberg_gate()

    energies = {}
    for max_iter in (4, 12, 30):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eA, eB = ctm_tensor_2site(
                A, B, chi=12, max_iter=max_iter, conv_tol=1e-9, recipe="2x2"
            )
        energies[max_iter] = float(
            compute_energy_ctm_tensor_2site(A, B, eA, eB, gate, 2)
        )

    print(f"#898 budget response: {energies}")
    # Strictly monotone in the budget: each extra sweep is still buying
    # something, which is exactly what the early exit was throwing away.
    assert energies[4] != energies[12], energies
    assert energies[12] != energies[30], energies
    # And it is descending toward the true fixed point, not wandering.
    assert energies[30] < energies[12] < energies[4], energies


@pytest.mark.slow
def test_a_collapsed_corner_says_so_out_loud():
    """Silence was the actual damage: the caller had no way to know.

    The environment is still returned — the sweeps ran and, on this fixture,
    kept improving — but it must not be mistaken for a fixed point.
    """
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site

    A, B = _collapsing_pair()

    with pytest.warns(RuntimeWarning, match="could not be certified") as rec:
        ctm_tensor_2site(A, B, chi=12, max_iter=6, conv_tol=1e-9, recipe="2x2")

    # The strong diagnosis, and it must be reserved for this case: the corner
    # is *still* rank-1, so "mean-field, will not respond to chi" is a claim
    # about the environment being returned rather than about the comparison.
    # Its twin
    # (``test_a_corner_blind_on_the_penultimate_sweep_still_warns``) asserts
    # the same phrase is absent when the corner recovered -- together they pin
    # that the message says only what that run established (#903 review, P2
    # round 3).
    text = str(rec[0].message)
    assert "mean-field" in text, text


@pytest.mark.slow
def test_a_healthy_state_still_converges_early_and_silently():
    """The over-rejection check, and the one that would sink the fix.

    A physical state has full-rank corners, so the criterion is meaningful
    there and must still be believed: early exit preserved, no warning, and
    critically **the same answer as before the guard** — `_ctm_sv_diff` is
    inert wherever it could already see.
    """
    from tenax.algorithms._ctm_diagnostics import ctm_corner_rank
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tests._su_fixtures import physical_su_d2

    A = physical_su_d2()

    with warnings.catch_warnings():
        # Any warning at all fails this test: silence is part of the contract.
        warnings.simplefilter("error", RuntimeWarning)
        eA, eB = ctm_tensor_2site(A, A, chi=8, max_iter=60, conv_tol=1e-8, recipe="2x2")

    rank_a, rank_b = ctm_corner_rank(eA), ctm_corner_rank(eB)
    print(f"#898 healthy-state corner ranks: {rank_a}, {rank_b} (chi=8)")
    assert rank_a > 1, rank_a
    assert rank_b > 1, rank_b


# --------------------------------------------------------------------- #
# The path a user actually reaches                                       #
# --------------------------------------------------------------------- #


def test_the_public_ipeps_loop_uses_the_guarded_criterion():
    """The guard is worth nothing if ``ipeps()`` runs a different comparison.

    It did.  ``ipeps_ctm_convergence`` carried its **own** ``_ctm_sv_diff`` --
    the same sum-normalised comparison without the rank test -- and that is the
    copy the public path ran: ``ipeps.py`` imports ``ctm_2site`` from there and
    calls it, so every ``ipeps()`` call kept comparing ``[1, 0, ..., 0]``
    against ``[1, 0, ..., 0]`` and exiting early, while the tests above passed
    against the guarded module.  A fix that leaves the production path broken
    is worse than no fix, because the closed issue stops anyone looking.

    Asserted as **object identity**, not by re-testing the behaviour: the
    behaviour is already covered above, and what regresses is someone
    re-introducing a local definition -- which identity catches and a
    value assertion would not, since a fresh unguarded copy returns the same
    number as the guarded one on every input except the collapsed ones.

    The three loops in that module (``ctm``, ``ctm_2site``, ``ctm_split``) all
    call the name bound at module scope, so binding it once is what routes all
    three.  ``ctm_2site`` is the one ``ipeps()`` reaches.
    """
    from tenax.algorithms import ipeps_ctm_convergence

    assert ipeps_ctm_convergence._ctm_sv_diff is _ctm_sv_diff, (
        "ipeps_ctm_convergence._ctm_sv_diff is not the guarded implementation "
        "from _ctm_tensor_convergence -- the public ipeps() path compares "
        "corner spectra with an unguarded copy again (#898), so a rank-1 "
        "corner will certify as converged there no matter what this file's "
        "other tests say"
    )


def test_the_public_loop_is_the_one_ipeps_calls():
    """Pin the routing the test above depends on.

    ``test_the_public_ipeps_loop_uses_the_guarded_criterion`` is only
    load-bearing while ``ipeps()`` actually reaches this module.  If the import
    moves, that test keeps passing and stops meaning anything -- so the routing
    is asserted rather than assumed.
    """
    import inspect

    # The MODULE, not the re-exported ``ipeps`` function of the same name --
    # ``from tenax.algorithms import ipeps`` binds the callable, and asking it
    # for ``ctm_2site`` raises AttributeError rather than failing the claim.
    import tenax.algorithms.ipeps as ipeps_module
    from tenax.algorithms import ipeps_ctm_convergence

    assert ipeps_module.ctm_2site is ipeps_ctm_convergence.ctm_2site, (
        "ipeps no longer calls ipeps_ctm_convergence.ctm_2site, so the guard "
        "assertion in the test above is no longer about the public path -- "
        "re-point it at whatever ipeps() reaches now"
    )
    source = inspect.getsource(ipeps_ctm_convergence)
    assert "def _ctm_sv_diff" not in source, (
        "ipeps_ctm_convergence defines _ctm_sv_diff again; it must import the "
        "guarded one from _ctm_tensor_convergence (#898)"
    )


# --------------------------------------------------------------------- #
# The warning must describe the sweep the loop exited on                 #
# --------------------------------------------------------------------- #


@pytest.mark.core
def test_a_corner_that_recovers_is_not_reported_as_blind(monkeypatch):
    """A corner blind early and healthy later must not warn (#903 review, P2).

    ``blind_coords`` was accumulated with ``.add`` and never cleared, so one
    transiently rank-deficient sweep marked a coordinate for the rest of the
    run.  The loop could then converge and exit early while the post-loop
    warning still fired -- asserting *in its own text* that "the full max_iter
    budget was run instead of exiting early", which had not happened.  That is
    a false statement to the caller, and under ``-W error::RuntimeWarning`` it
    turns a successful run into a failure.

    The blindness is injected rather than fished out of a physical fixture on
    purpose.  What is under test is the bookkeeping -- *does the reported set
    describe the sweep the loop exited on* -- and a fixture that happened to
    recover would be testing the fixture's luck as much as the code.  One
    sweep reports blind, every later sweep reports the truth.

    Note the two are not independent: a genuinely blind corner forces
    ``_ctm_sv_diff`` to ``inf``, so ``converged`` cannot be true while any
    corner is blind.  The last sweep's set is therefore non-empty exactly when
    the budget ran out because of blindness -- which is what the warning says.
    """
    from tenax.algorithms import _ctm_tensor_convergence as conv
    from tests._su_fixtures import physical_su_d2

    A = physical_su_d2()
    real = conv._spectrum_can_show_change
    sweeps = {"n": 0}

    def blind_on_the_first_sweep_only(sv, *args, **kwargs):
        sweeps["n"] += 1
        if sweeps["n"] <= 1:
            return False
        return real(sv, *args, **kwargs)

    monkeypatch.setattr(
        conv, "_spectrum_can_show_change", blind_on_the_first_sweep_only
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        conv.ctm_tensor_2site(A, A, chi=8, max_iter=60, conv_tol=1e-8, recipe="2x2")

    assert sweeps["n"] > 1, (
        f"the patch was consulted {sweeps['n']} time(s), so the loop never got "
        f"past the sweep this test marks blind -- it would pass with the bug "
        f"present.  This test is only meaningful if a later, healthy sweep ran."
    )


@pytest.mark.core
def test_a_corner_blind_on_the_penultimate_sweep_still_warns(monkeypatch):
    """Budget exhausted *because* of blindness must warn, even when the final
    sweep's own spectrum is healthy (#903 review, P2 round 2).

    ``_ctm_sv_diff`` reads **both** spectra.  A corner that is blind on sweep
    ``N-1`` and healthy on sweep ``N`` still forces ``inf`` on sweep ``N``'s
    comparison, so the loop spends its whole budget without certifying -- and a
    ``blind_coords`` built from the *current* spectrum alone is empty in exactly
    that case.  The warning is suppressed precisely where it is load-bearing.

    This is the mirror image of
    ``test_a_corner_that_recovers_is_not_reported_as_blind``: that one pins
    "recovered, converged, stay silent", this one pins "recovered, still ran
    out, say so".  A set keyed on the current spectrum passes the first and
    fails this one; a set keyed on the whole comparison passes both.

    **Both predicates are faked together because production couples them.**
    ``_spectrum_can_show_change`` is the eager per-spectrum form of the test
    ``_ctm_sv_diff`` applies to each of its two arguments, so a fake that blinds
    one without the other builds a state the real code cannot reach -- and the
    test would then be measuring the fake.  Here one spectrum object is marked,
    and every question either function is asked about *that object* answers
    "uninformative", exactly as it would if the corner really were rank-1.
    """
    from tenax.algorithms import _ctm_tensor_convergence as conv
    from tests._su_fixtures import physical_su_d2

    A = physical_su_d2()
    real_can_show = conv._spectrum_can_show_change
    real_diff = conv._ctm_sv_diff
    marked: dict[str, object] = {}

    def _is_marked(sv):
        return "sv" in marked and sv is marked["sv"]

    def fake_can_show(sv, *args, **kwargs):
        if "sv" not in marked:  # the first spectrum the loop ever sees
            marked["sv"] = sv
            return False
        return False if _is_marked(sv) else real_can_show(sv, *args, **kwargs)

    def fake_diff(a, b, *args, **kwargs):
        if _is_marked(a) or _is_marked(b):
            return jnp.asarray(jnp.inf)
        return real_diff(a, b, *args, **kwargs)

    monkeypatch.setattr(conv, "_spectrum_can_show_change", fake_can_show)
    monkeypatch.setattr(conv, "_ctm_sv_diff", fake_diff)

    # Exactly two sweeps: sweep 1's spectrum is the marked (blind) one, sweep 2
    # is healthy but must still compare against it.
    with pytest.warns(RuntimeWarning, match="could not be certified") as rec:
        conv.ctm_tensor_2site(A, A, chi=8, max_iter=2, conv_tol=1e-8, recipe="2x2")

    # It must warn -- and it must not overclaim.  The returned corner is rank
    # > 1, so this environment is *uncertified*, not established to be bad.
    # Calling it a mean-field number that will not respond to chi would invite
    # callers to discard a healthy environment, which is the opposite of what
    # the warning is for (#903 review, P2 round 3).
    text = str(rec[0].message)
    assert "mean-field" not in text, (
        f"the recovered case must not borrow the collapsed case's diagnosis; "
        f"the corner returned here has rank > 1: {text}"
    )
    assert "recovered" in text, text
