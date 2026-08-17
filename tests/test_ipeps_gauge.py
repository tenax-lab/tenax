"""The absorbed-form convention: a site tensor IS the wavefunction.

The Vidal warning in bp_gauge_checkerboard is about passing ones() alongside
BARE Gamma, which drops lambda.  Absorbing sqrt(lambda) into both ends first
and then passing ones() is the same state written differently --
A sqrt(lam) . sqrt(lam) B == A lam B -- and that is what this file proves.

The convention is a contract on gauge_fix's *output* as much as its input: the
pair it returns must already be the state, with the weights carried alongside
as a diagnostic.  Reading a returned pair with its own weights and reading it
with ones() cannot both be right, and
test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one is the test that exists
to separate them.  It is not the only one that would notice, and an earlier
version of this docstring said it was: that claim was measured and is FALSE.
Reverting gauge_fix to return Vidal also fires
test_gauge_fix_matches_the_vidal_route, which reads the gauge_fix route with
ones() and gates at 1e-12 -- see that test's docstring for the per-cell
numbers.  The distinction matters because "everything else is blind" reads as a
licence to put w2 back into test_gauge_fix_matches_the_vidal_route's state
comparison, and it is not one: that comparison reading ones() is precisely what
makes the second test see the output form at all.  What was true is the
*history* -- before 4d2bc00 every test here passed the weights back in, which
is how the output form stayed wrong for four commits.

Runs on the symmetric path too: a flow mistake there collapses charge sectors
rather than raising, and the dense path cannot see it.

The last two tests in the file are a different kind of claim from the rest.
Every other test here -- and every acceptance criterion in Phase 1 -- is a
*self-consistency* check: the state did not move, the two routes agree, the
witness separates.  ``test_bp_weights_are_the_chains_schmidt_values`` and its
block-sparse twin ``test_bp_weights_are_the_symmetric_chains_schmidt_values``
compare BP's output against an answer known a priori, because a chain is a tree
and BP is exact on trees.  That is the only anchor geometry available (#882
§6.3) -- the loopy square lattice admits no valid reference spectrum -- so if
either fails, nothing measured downstream of it means anything.  Both are
@pytest.mark.core and both assert their reference against a spectrum derived
outside tenax (_CHAIN_TRUTH / _SYM_CHAIN_TRUTH); without that assertion the
live comparison is tenax against tenax and a common-mode defect in the shared
contract/svd/eigh machinery passes both sides.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_gauge_helpers import (  # tests/ is on sys.path
    _INDEPENDENT_BOND_OF,
    _PAIRS,
    _WITNESS_PAIRS,
    _chain_middle_spectra,
    _chain_pair,
    _chain_pair_as_peps,
    _sym_chain_middle_spectra,
    _sym_chain_pair,
    _sym_chain_pair_as_peps,
    _torus_2x2,
    _udlrp,
    assert_leg_split,
)

from tenax.algorithms.ipeps_bp_gauge import BondWeights, bp_gauge_checkerboard
from tenax.algorithms.ipeps_gauge import (
    absorb_weights,
    ctm_rdm2x1_planar,
    gauge_fix,
    torus_2x2_sign_free,
)
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.index import FlowDirection

ABSORB_TOL = 1e-13

#: The ``(kind, D)`` grid for the two tests that run a **full BP solve** per
#: cell, with the expensive cell marked ``slow``.
#:
#: ``conftest.py`` maps this file to ``algorithm``, and
#: ``pytest_collection_modifyitems`` withholds a file marker only when that
#: marker is ``core``; an ``algorithm`` file's tests therefore keep *both*
#: markers.  That is what makes this work: the two non-core buckets select
#: ``not core and not slow``, so the ``slow`` half deselects these cells from
#: ``fast-other``, while the ``slow`` bucket (``-m slow``, run on push to main
#: and behind the ``run-full-tests`` label) picks them up.  Nothing is lost.
#:
#: Measured here, ``--no-cov``, load1 ~6 of 128: ``symmetric D=3`` costs
#: **219.2 s** in ``matches_the_vidal_route`` and **158.3 s** in
#: ``returns_an_absorbed_pair``.  That second figure is 2x what it was before
#: the ``gauged`` assertion, which runs a second block-sparse BP solve; the
#: solve is eager and host-bound (#566/#618) and is essentially the whole cost.
#: Together with the fermionic witness that is ~9 min taken out of
#: ``fast-other``.  The ``symmetric D=2`` cells stay -- 63.1 s and 49.8 s --
#: because they are affordable and they keep the block-sparse path covered on
#: every PR.  The ids are pinned to what the two stacked ``parametrize``
#: decorators used to produce, so ``-k`` selections and the review transcript
#: still name the same cells.
#:
#: The ``core`` chain anchors below are deliberately **not** here: they are the
#: only externally certified claims in the rewrite and must stay in the required
#: gate.
_KIND_D_SOLVE_CASES = [
    pytest.param("dense", 2, id="2-dense"),
    pytest.param("dense", 3, id="3-dense"),
    pytest.param("symmetric", 2, id="2-symmetric"),
    pytest.param("symmetric", 3, id="3-symmetric", marks=pytest.mark.slow),
]


def _unit(x):
    return x / np.linalg.norm(x)


def _torus_rel(x, y):
    """Distance between two torus readings **as states**: normalise, then compare.

    Used by three call sites in two tests: one in the Vidal-route test and two
    in the absorbed-form guard.  It is **not** "the convention every
    state-equality assertion here uses", as this docstring once said -- four
    other tests compare torus readings inline with ``_unit(...)`` and a plain
    norm, which is this minus the ``todense()`` handling.  That is not an
    oversight worth consolidating: those four compare two *arrays* and never a
    ``SymmetricTensor``.

    The normalisation is the whole content.  ``gauge_fix`` rescales each site
    tensor by max-abs and max-normalises each weight vector -- both deliberate,
    both unobservable -- so an overall factor is *expected* and is not a state
    difference.  Skipping it does not merely inflate the number, it makes it
    **meaningless in either direction**, because the torus inherits whatever
    scale the two site tensors happen to carry.  Measured on readings that
    agree to ~1e-15 once normalised:

    * absorbed-form call sites (:func:`torus_2x2_sign_free`): 3.2e-02
      (dense D=2), 1.1e+00 (dense D=3), 1.2e+01 (symmetric D=2), 2.2e+01
      (symmetric D=3) -- inflated, which is how this defect was first
      mis-measured;
    * the Vidal-route call site (``_torus_2x2`` against a ``Gamma`` pair):
      1.2e-19, 3.4e-16, 1.8e-92, 1.5e-93 -- *deflated*, and a raw threshold
      there would pass anything at all.

    **No sign freedom, deliberately.**  An earlier version took
    ``min(|a-b|, |a+b|)`` on the grounds that the torus is "degree 4 in every
    site tensor".  It is not: ``_TORUS_COPIES`` puts ``A`` at A00/A11 and ``B``
    at B10/B01, so the network is degree **2** in each, ``A -> -A`` gives
    ``(+1)``, and there is no overall sign to accommodate.  Measured, the
    ``+`` branch was dead: ``|a+b|`` scored 2.000 in all four cells of both
    tests (1.813-1.998 for the ``as_vidal`` reading), so ``min`` always took
    ``|a-b|`` and the branch only stood ready to *weaken* a future comparison
    for a reason that does not exist.  Per-column ``+-1`` freedom from the
    eigh/SVD inside ``_sqrt_pinv`` is real, but it cancels here -- every bond
    is closed, carrying the sign on one end and its inverse on the other --
    which is exactly why :func:`_abs_gap` needs an ``abs()`` and this does not.

    Accepts a ``Tensor`` or an array; the torus is a ``(d,d,d,d)`` object, so
    densifying it is cheap even on the symmetric path.
    """
    a = _unit(np.asarray(x.todense() if hasattr(x, "todense") else x).ravel())
    b = _unit(np.asarray(y.todense() if hasattr(y, "todense") else y).ravel())
    return float(np.linalg.norm(a - b))


def _abs_gap(x, y):
    """Elementwise ``max| |x| - |y| |`` on two site tensors, in ``(u,d,l,r,phys)``.

    A comparison of the tensors themselves rather than of a contraction of
    them, which is the point: every closed-network probe in this file is gauge
    invariant by design and therefore cannot witness that a gauge was applied.

    The ``abs()`` is not slack.  The eigh/SVD inside ``_sqrt_pinv`` fixes each
    output column only up to a sign, so two runs of the same solve can differ
    by a diagonal ``+-1`` on a virtual leg; that sign cancels in every closed
    contraction and is unobservable, but a bare elementwise comparison would
    see it.  ``test_ipeps_gauge_perf.py``'s module docstring documents the same
    freedom, measured: ``|A_eager - A_traced| = 2.000e+00`` while
    ``||A_eager| - |A_traced|| = 1.0e-13``.  What ``abs()`` does *not* hide is
    a different gauge, which changes magnitudes.
    """
    return float(np.max(np.abs(np.abs(_udlrp(x)) - np.abs(_udlrp(y)))))


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


@pytest.mark.parametrize("kind,D", _KIND_D_SOLVE_CASES)
def test_gauge_fix_matches_the_vidal_route(kind, D):
    """gauge_fix on absorbed input reaches the Vidal call's state AND fixed point.

    If this fails, gauge_fix is gauging a different state than the caller
    handed it, and every acceptance criterion above it is meaningless.

    The two routes are read with *different* weights, which is the point: the
    ``bp_gauge_checkerboard`` route returns Vidal and is read with its ``w1``,
    while ``gauge_fix`` returns absorbed and is read with ``ones``.  Both name
    the same state.  This assertion previously read the ``gauge_fix`` route as
    Vidal too, and passed -- because ``gauge_fix`` used to return Vidal.  It
    was testing agreement between the routes, which is real, and not the
    returned form, which nothing tested; see
    ``test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one``.

    **Since that fix, this test does see the output form**, and the ``ones``
    on the ``gauge_fix`` side is load-bearing rather than cosmetic.  Measured:
    reverting ``gauge_fix`` to return the Vidal pair moves the state
    comparison to **1.398e+00** (dense D=2), **9.938e-01** (dense D=3),
    **7.102e-01** (symmetric D=2) and **9.457e-01** (symmetric D=3), against
    the 1e-12 gate below.  So do not "simplify" that ``ones`` back to ``w2``:
    it would restore the blindness the module docstring describes, and this
    test would go back to checking route agreement only.  What this still
    cannot see is an identity ``gauge_fix`` -- both sides move together --
    which is what the ``gauged`` assertion in the companion test is for.
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

    ones = BondWeights.ones(D, D)
    state = _torus_rel(_torus_2x2(A2, B2, ones), _torus_2x2(A1, B1, w1))
    assert state < 1e-12, f"{kind} D={D}: routes reached different states ({state:.3e})"

    for f in BondWeights._fields:
        a, b = getattr(w1, f), getattr(w2, f)
        a, b = a / jnp.max(a), b / jnp.max(b)
        d = float(jnp.max(jnp.abs(a - b)))
        assert d < 1e-10, f"{kind} D={D} bond {f}: different fixed point ({d:.3e})"


@pytest.mark.parametrize("kind,D", _KIND_D_SOLVE_CASES)
def test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one(kind, D):
    """The returned pair IS the state, and it is the *gauged* one.  Three ways.

    This is the claim the whole absorbed-form convention rests on and it was
    asserted nowhere: before ``4d2bc00`` every other test in this file passed
    ``gauge_fix``'s weights back into the torus, so all of them exercised the
    *Vidal* reading and none of them could tell the two apart.  Phase 2's
    ``_su_step`` holds no spectrum -- ``_SUState`` has nowhere to put one
    (#882 §3) -- so if the returned pair is not already the state, the step
    silently evolves a different one and reports a plausible, wrong energy.
    That is #667 and #865 verbatim.

    ``as_vidal`` is what makes the first two a *discriminating* pair rather
    than a guard that merely accepts an absorbed pair: feeding the weights back
    in double-counts them, so the Vidal reading has to break.  A test that only
    checked ``dropped`` would still pass if ``weights`` silently degenerated to
    all-ones, which is the trivial way to satisfy it.

    Measured on this branch, before and after ``4d2bc00`` (``gauge_fix``
    output, torus read the two ways, normalised; "sign-free" in
    :func:`torus_2x2_sign_free`'s sense, i.e. no Koszul signs, not a free
    overall sign -- there is none, see :func:`_torus_rel`)::

        pair          before: +ones    +w2  |  after: +ones      +w2
        dense D=2         1.252e+00  4e-15  |     3.7e-15  3.983e-01
        dense D=3         8.948e-01  2e-14  |     1.7e-14  2.776e-01
        symmetric D=2     6.051e-01  8e-16  |     8.2e-16  8.999e-02
        symmetric D=3     2.491e-01  6e-15  |     5.7e-15  8.444e-01

    Before, the two readings were swapped: the pair alone was off by 0.25 to
    1.25 and only the Vidal reading was exact.  The thresholds below sit
    between the two columns with room to spare -- the tightest discriminating
    cell is symmetric D=2 at 9.0e-02, 9x above the 1e-2 gate.

    **``gauged`` is the third assertion and it is a different kind of claim.**
    ``dropped`` and ``as_vidal`` are both readings of a *closed torus*, which is
    exactly gauge invariant -- that is what it is for -- and so is every other
    state-equality probe on this branch (``_torus_2x2``,
    :func:`torus_2x2_sign_free`, :func:`ctm_rdm2x1_planar`).  A probe that
    cannot see a gauge cannot see the *absence* of one either.  Mutate
    ``gauge_fix`` to ``return A, B, weights, info`` -- the caller's own
    un-gauged pair, with the correctly solved weights -- and, without
    ``gauged``, the entire branch stays green: ``dropped`` is then *exactly*
    zero (its reference is built from the same input pair), ``as_vidal`` still
    fires because the weights are right, and both ``core`` chain anchors
    discard the returned pair.  Measured under that mutation, ``dropped`` is
    ``0.000e+00`` in **all four** cells and ``as_vidal`` is 6.08e-01 / 6.88e-01
    / 4.83e-01 / 2.89e-01 -- every one of them passing.

    That matters because Phase 2 truncates using the returned weights **in the
    returned pair's basis**.  An implementation that gauges correctly and hands
    back the wrong representative truncates in the wrong basis -- a plausible,
    wrong energy, the #667/#865 shape -- with all of Phase 1 green.

    So ``gauged`` compares the returned pair *elementwise* against the same
    state solved through the other entry point, ``bp_gauge_checkerboard`` +
    :func:`absorb_weights`, which is the composition ``gauge_fix`` promises to
    be.  It is **not** a traced-vs-eager cross-check (``bp_gauge_checkerboard``
    dispatches on tensor type exactly as ``gauge_fix`` does, so the dense arm
    is traced on both sides; ``test_ipeps_gauge_perf.py`` owns that comparison);
    it is a check that ``gauge_fix`` applied *a* gauge at all, and the right
    one.  Measured, shipped against identity-mutant::

        dense D=2      0.00e+00   vs   2.545e-01
        dense D=3      0.00e+00   vs   5.955e-01
        symmetric D=2  0.00e+00   vs   2.155e+00
        symmetric D=3  0.00e+00   vs   2.255e+00

    Bit-identical when correct, on both tensor types; the 1e-11 gate below has
    ten orders of margin against the smallest mutant value (2.5e-01, dense
    D=2).  The gate is 1e-11 rather than 0.0 only so that a future jaxlib whose
    two entry points reassociate differently does not fail for a reason that is
    not a defect; nothing measured needs the slack.

    Two more principled-looking discriminators were tried and are refuted by
    measurement -- do not reach for them.  *Idempotence* (re-gauge the output,
    expect ~1 sweep): the pair is not idempotent, re-gauging takes 45 sweeps at
    D=2 and 31 at D=3 against 44/32 for the raw pair, so a sweep-count guard is
    dead on arrival (the *weights* are idempotent to 4.6e-13; the pair is not).
    *Message off-diagonality*: separates only 3.15e-01 from 4.93e-01 on dense
    D=2 and is exactly 0.00e+00 on both arms of the symmetric path.
    """
    A, B = _PAIRS[kind](D=D)
    ones = BondWeights.ones(D, D)
    ref = torus_2x2_sign_free(A, B, ones)

    A2, B2, w2, info = gauge_fix(A, B, tol=1e-12, max_iter=400)
    assert info.converged

    dropped = _torus_rel(torus_2x2_sign_free(A2, B2, ones), ref)
    assert dropped < 1e-11, (
        f"{kind} D={D}: dropping gauge_fix's weights moved the state by "
        f"{dropped:.3e}.  The returned pair must BE the state -- the weights "
        f"are a diagnostic, and Phase 2 has nowhere to store them."
    )

    as_vidal = _torus_rel(torus_2x2_sign_free(A2, B2, w2), ref)
    assert as_vidal > 1e-2, (
        f"{kind} D={D}: reading gauge_fix's output as Vidal moved the state by "
        f"only {as_vidal:.3e}, so this guard cannot tell an absorbed pair from "
        f"a Vidal one.  Either the weights came back trivial or the pair did."
    )

    # The one assertion here that is NOT gauge invariant.  See the docstring:
    # without it nothing on this branch distinguishes gauge_fix from an
    # identity function.
    ref_A, ref_B = absorb_weights(
        *bp_gauge_checkerboard(A, B, ones, tol=1e-12, max_iter=400)[:3]
    )
    gauged = max(_abs_gap(A2, ref_A), _abs_gap(B2, ref_B))
    assert gauged < 1e-11, (
        f"{kind} D={D}: gauge_fix's pair differs elementwise (up to per-column "
        f"sign) from bp_gauge_checkerboard + absorb_weights on the same input "
        f"by {gauged:.3e}.  gauge_fix is documented as exactly that "
        f"composition, and this is the only assertion on the branch that is "
        f"not gauge invariant -- an identity gauge_fix scores 2.5e-01 to "
        f"2.3e+00 here and passes everything else."
    )


#: Four bond weights with **twelve distinct entries**, so that permuting any
#: two of them -- across bonds or within one -- changes the torus value.
#: ``BondWeights.ones`` cannot do that: every mis-placement then multiplies the
#: same ones and the probe agrees for the wrong reason.
_DISTINCT_W = BondWeights(
    h_AB=jnp.array([1.0, 0.4, 0.1]),
    h_BA=jnp.array([0.9, 0.6, 0.2]),
    v_AB=jnp.array([0.8, 0.35, 0.05]),
    v_BA=jnp.array([0.7, 0.5, 0.3]),
)


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_sign_free_probe_matches_einsum_probe_on_bosonic(kind):
    """Validate the ``contract``-routed torus against the ``np.einsum`` one.

    Both are sign-free, so this is a *wiring* check, not a sign check: it
    compares an edge map transcribed into ``_TORUS_EDGE_OF`` against the one
    ``_torus_2x2`` hard-codes positionally.  A self-consistent but wrong torus
    -- every label used twice, every pairing flow-legal, nothing raised -- is
    the failure this catches.
    """
    A, B = _PAIRS[kind](D=3)
    w = BondWeights.ones(3, 3)
    ref = _unit(np.asarray(_torus_2x2(A, B, w)).ravel())
    got = _unit(np.asarray(torus_2x2_sign_free(A, B, w).todense()).ravel())
    rel = float(np.linalg.norm(got - ref))
    assert rel < 1e-13, (
        f"{kind}: the two torus probes disagree by {rel:.3e} on a BOSONIC pair, "
        f"where both are valid.  The wiring is wrong, not the gauge."
    )


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_sign_free_probe_places_each_weight_on_the_right_leg(kind):
    """The same agreement with a *distinct* weight on every bond.

    ``BondWeights.ones`` above makes all four weights identical, so it cannot
    see a probe that hangs ``h_AB`` where ``v_AB`` belongs.  Measured, that
    horizontal/vertical swap moves this comparison by 3.6e-01 (dense) and
    2.2e-01 (symmetric), and ``ones`` scores it 0.

    What this does *not* claim to catch is dressing ``u``/``l`` instead of
    ``d``/``r``.  With each leg carrying its own bond's weight that is a
    mathematical no-op -- lambda is diagonal on the shared bond, so hanging it
    on either end of the same edge contracts to the same thing (measured:
    2.6e-15 / 1.6e-16, i.e. baseline).  Only carrying the *wrong bond's*
    weight is a real mutation.
    """
    A, B = _PAIRS[kind](D=3)
    ref = _unit(np.asarray(_torus_2x2(A, B, _DISTINCT_W)).ravel())
    got = _unit(np.asarray(torus_2x2_sign_free(A, B, _DISTINCT_W).todense()).ravel())
    rel = float(np.linalg.norm(got - ref))
    assert rel < 1e-13, (
        f"{kind}: the two torus probes disagree by {rel:.3e} once the four bonds "
        f"carry different weights -- the wiring or the weight placement is "
        f"wrong, not the gauge"
    )


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_sign_free_probe_sees_the_known_bosonic_gauge_invariance(kind):
    """The instrument reproduces the known answer on a known-exact gauge.

    ``bp_gauge_checkerboard`` is already verified exact to ~1e-15 on both
    bosonic pair types, so a probe that cannot see that is broken.  Bosonic
    only, deliberately: on a fermionic pair this probe carries no sign
    information at all (see :func:`torus_2x2_sign_free`), and the witness for
    that case is :func:`ctm_rdm2x1_planar` below.
    """
    A, B = _PAIRS[kind](D=3)
    ones = BondWeights.ones(3, 3)
    A1, B1, w1, info = bp_gauge_checkerboard(A, B, ones, tol=1e-12, max_iter=400)
    assert info.converged

    before = _unit(np.asarray(torus_2x2_sign_free(A, B, ones).todense()).ravel())
    after = _unit(np.asarray(torus_2x2_sign_free(A1, B1, w1).todense()).ravel())
    rel = float(np.linalg.norm(after - before))
    assert rel < 1e-12, (
        f"{kind}: the sign-free probe reports a {rel:.3e} state change across a "
        f"gauge already known exact to ~1e-15; the probe is what is wrong"
    )


# --- the planar (fermion-safe) witness ------------------------------------

#: A diagonal gauge on the ``h_AB`` bond: ``g`` on ``A.r``, ``1/g`` on ``B.l``.
#: Exact for *any* statistics -- it rescales the basis of one shared index and
#: the contraction sums ``g_k * (1/g_k)`` over it -- with no transpose, no
#: fuse and no decomposition, so no sign convention can enter.  That is what
#: makes it a usable calibration input for the graded case: unlike BP, the
#: answer is known a priori.
#:
#: Diagonal is also all that D=2 can express on the fermionic arm: its virtual
#: parity sectors are ``{0: 1, 1: 1}``, so every parity-preserving matrix is
#: 1x1 per sector (see ``_fermionic_pair``).  The tests below are claims about
#: the *witness*, not about non-diagonal gauges; a witness calibrated only on
#: diagonal gauges needs D >= 3 to say more (#882 §5.2).
_GAUGE_G = jnp.array([1.7, 0.55])

_WITNESS_CHI = 8
_WITNESS_KW = {"chi": _WITNESS_CHI, "max_iter": 40, "conv_tol": 1e-12}


def _rdm_rel(x, y):
    return float(np.linalg.norm(np.asarray(x) - np.asarray(y)) / np.linalg.norm(y))


#: ``_WITNESS_PAIRS`` again, with the fermionic arm marked ``slow``.  It runs
#: **three** block-sparse CTM convergences at chi=8 and costs 167 s against the
#: dense arm's 18 -- a fermionic CTM in the ``fast-other`` bucket -- and the
#: fermionic line is deferred to #882 §5.2 anyway, so it belongs in the bucket
#: that runs on push to main rather than on every PR.  Derived from ``_WITNESS_PAIRS`` rather than
#: written out, so adding an arm there cannot silently skip this decision.  The
#: ids stay ``[dense]`` / ``[fermionic]``.
_WITNESS_CASES = [
    pytest.param(k, marks=[pytest.mark.slow] if k == "fermionic" else [])
    for k in _WITNESS_PAIRS
]


@pytest.mark.parametrize("kind", _WITNESS_CASES)
def test_planar_witness_separates_a_real_gauge_from_a_mispaired_one(kind):
    """The CTM/RDM witness must move on a broken gauge and not on a real one.

    Both halves matter and neither is enough alone.  A witness that is
    invariant under *everything* -- a collapsed environment, a constant --
    passes the control trivially and would certify any gauge as exact; that
    is the "assertion that cannot fail" failure mode.  So the mutation is
    measured in the same test, on the same state, and the assertion is on the
    **separation** between them.

    The mutation is a mispaired gauge: ``g`` on ``A.r`` but ``1/g`` on
    ``B.u`` instead of ``B.l``, so the two halves sit on different bonds and
    nothing cancels.  It is the same magnitude of perturbation as the
    control, which is the point -- the witness must distinguish them by
    *structure*, not by size.

    Note the control does **not** go to machine precision, and must not be
    asserted to.  CTM truncates the environment to ``chi`` states and a gauge
    changes the virtual basis that truncation is taken in, so at finite
    ``chi`` the two runs keep slightly different subspaces.

    What that floor does with ``chi`` differs by arm, and only the dense one
    behaves like truncation (all numbers from ``task-4-report.md``; none
    re-measured here):

    - dense, D=2: 5.5e-04 at chi=4 down to 2.5e-09 at chi=32 -- falls steeply.
    - fermionic, D=2: 3.13e-03 (chi=6), 1.06e-03 (chi=8), 1.24e-03 (chi=12) --
      a plateau around 1e-3.

    Flat in ``chi`` is this project's own defect signature, so the fermionic
    floor is **not** established as truncation; it is unexplained, and the
    question is deferred to #882 §5.2.  A gauge-strength scan was previously
    offered here as a second, independent scaling that settled it.  It is not
    independent: as the gauge tends to the identity the gauged run and the
    reference run become the *same computation*, so the displacement must fall
    whatever the cause.  Both observations are descriptive only.

    None of that weakens what this test asserts, which is the **separation**
    between control and mutation, and holds on both arms (measured at the
    shipped chi=8: 5800x dense, 882x fermionic, against a 100x gate).  It does
    mean the ``ctl < 1e-2`` bound is a bound on a *plateau* on the fermionic
    arm rather than on something known to vanish, so it must not be tightened
    without re-measuring that arm.
    """
    A, B = _WITNESS_PAIRS[kind](D=2)
    base = ctm_rdm2x1_planar(A, B, **_WITNESS_KW)

    A_g = scale_bond_axis(A, "r", _GAUGE_G)
    real = ctm_rdm2x1_planar(
        A_g, scale_bond_axis(B, "l", 1.0 / _GAUGE_G), **_WITNESS_KW
    )
    broken = ctm_rdm2x1_planar(
        A_g, scale_bond_axis(B, "u", 1.0 / _GAUGE_G), **_WITNESS_KW
    )

    ctl = _rdm_rel(real, base)
    mut = _rdm_rel(broken, base)

    assert ctl < 1e-2, (
        f"{kind}: an exact diagonal gauge moved the witness by {ctl:.3e}.  That "
        f"is far above this witness's finite-chi noise floor on either arm and "
        f"means it is not measuring a gauge-invariant quantity."
    )
    assert mut > 0.1, (
        f"{kind}: a mispaired gauge -- which changes the physical state -- moved "
        f"the witness by only {mut:.3e}.  A witness that cannot see this cannot "
        f"certify anything."
    )
    assert mut > 100 * ctl, (
        f"{kind}: the witness separates a real gauge ({ctl:.3e}) from a broken "
        f"one ({mut:.3e}) by only {mut / max(ctl, 1e-300):.0f}x.  Below 100x the "
        f"truncation floor is too close to the signal to trust a verdict."
    )


def test_planar_witness_floor_shrinks_with_chi():
    """Pin the *trend*, so a future defect cannot hide inside the floor.

    The control residual above is a finite-``chi`` truncation artefact **on
    the dense arm**, and the way to keep that claim honest is to require it to
    behave like one.  A genuine contraction defect -- a dropped sign, a
    mispaired leg -- would sit at a fixed size no matter how much environment
    it is given, so "shrinks with chi" is what distinguishes the two.
    Measured here: 718x across chi 6 -> 16 (``task-4-report.md``).

    Dense only, and **not** merely because it is the cheapest arm: the
    fermionic arm is not expected to pass this.  Its floor at D=2 is 3.13e-03
    (chi=6), 1.06e-03 (chi=8), 1.24e-03 (chi=12) -- about 2.5x across a 2x
    change in ``chi``, against the 10x required below (``task-4-report.md``).

    That "would therefore fail" is an **inference, not a measurement**, and the
    gap is worth naming: the cited fermionic data stops at chi=12, while the
    assertion below compares chi=6 against chi=**16**.  Nobody has run the
    fermionic arm at chi=16, so what is established is a ~2.5x drop over 6->12
    where 10x over 6->16 is required -- strong, but not the same statement.
    Running it would cost a fermionic CTM at chi=16 (~2-3 min each), which this
    round does not spend; the question is deferred to #882 §5.2.  So read this
    test as validating the *dense* witness only -- it says nothing about the
    fermionic one, which the docstring on :func:`ctm_rdm2x1_planar` describes
    instead.
    """
    A, B = _WITNESS_PAIRS["dense"](D=2)
    A_g = scale_bond_axis(A, "r", _GAUGE_G)
    B_g = scale_bond_axis(B, "l", 1.0 / _GAUGE_G)

    def floor(chi):
        kw = {"chi": chi, "max_iter": 40, "conv_tol": 1e-12}
        return _rdm_rel(
            ctm_rdm2x1_planar(A_g, B_g, **kw), ctm_rdm2x1_planar(A, B, **kw)
        )

    lo, hi = floor(6), floor(16)
    assert hi < lo / 10.0, (
        f"the gauge-invariance floor went {lo:.3e} (chi=6) -> {hi:.3e} (chi=16), "
        f"less than the 10x drop expected of a truncation effect.  A floor that "
        f"is flat in chi is a defect, not truncation, and it would mask exactly "
        f"the violation this witness exists to detect."
    )


# --- the ground-truth anchor: BP on a chain -------------------------------

#: The two chain lengths the reference is measured at, and cross-checked
#: between.  20/40 -- where the task brief starts -- is nowhere near enough: on
#: this draw the middle bond still moves 6.0e-05 between them, seven orders
#: above the tolerance BP is then judged at.  Measured drift, largest of the two
#: middle bonds: 20->40 6.0e-05, 40->60 2.6e-09, 60->80 1.2e-13, 80->100
#: 3.1e-16.  60/80 would already clear the 1e-12 gate, but only by 8x, which
#: would leave the reference's own uncertainty within one order of the answer it
#: is certifying; 80/100 sits at the f64 floor instead.
_CHAIN_L_LO, _CHAIN_L_HI = 80, 100

#: Both the reference's self-agreement in ``L`` and BP's agreement with it, on
#: both arms of the anchor.  Measured, dense: reference 3.1e-16, BP 1.8e-15
#: (``h_AB``) and 3.7e-16 (``h_BA``).  Block-sparse: reference 4.1e-15, BP
#: 7.1e-16 and 1.0e-15.  One number rather than two because the claim is the
#: same one -- BP is exact on a tree -- and both arms sit at the f64 floor;
#: what differs between them is the ``L`` needed to get there, which is why
#: that is the constant each arm has its own copy of.
_ANCHOR_TOL = 1e-12

#: ``gauge_fix``'s default ``tol=1e-6`` is a *weight* tolerance, and the weights
#: are what this test reads: at the default it converges in 14 sweeps with
#: ``h_AB`` 2.2e-07 off the exact spectrum, which is BP stopping early, not BP
#: being wrong.  At 1e-14 it takes 33 sweeps and lands at 1.8e-15.
_ANCHOR_BP_KW = {"tol": 1e-14, "max_iter": 500}


def _as_spectrum(x):
    """A bond spectrum in the convention this comparison uses.

    Sorted descending, then normalised to unit 2-norm.  Neither half is
    cosmetic.  ``compute_singular_values`` normalises each bond to
    ``sum(sv**2) == 1`` while ``bp_gauge_checkerboard`` max-normalises
    (``lam / max(lam)``), so the two sides arrive on different scales and one
    of the two conventions has to be imposed; unit 2-norm is the physical one
    and is what the reference already uses.  The sort is because a bond weight
    is a *multiset* -- the block-sparse path orders it by charge sector rather
    than by size -- so its order carries nothing to preserve.

    This is deliberately not ``_torus_rel``: that normalises a whole *tensor*
    to unit 2-norm after flattening it, which for a bond spectrum would be the
    same arithmetic but for the wrong reason -- and it does not sort, which a
    multiset comparison needs.
    """
    v = np.sort(np.asarray(x))[::-1]
    return v / np.linalg.norm(v)


#: The infinite chain's two horizontal-bond spectra for the ``_chain_pair``
#: draw, **derived outside tenax**: rebuilt from the 2-site transfer matrix's
#: left and right fixed points in Python ``decimal``, importing nothing from
#: tenax but the two ``float64`` site tensors.  Truncated to 32 digits.
#:
#: This is the half of the anchor that makes it an anchor.  Everything else in
#: the test compares tenax against tenax -- ``FiniteMPS.compute_singular_values``
#: for the reference, ``contract``/``eigh``/``svd`` for the subject -- and a
#: common-mode defect in the shared machinery (#834's class) passes both sides.
#: Asserting the *reference* against these numbers puts the external
#: certification inside the required gate instead of in a review transcript.
#:
#: Measured on this branch: the tenax reference at ``L=100`` agrees with these
#: to **8.3e-17** (``h_AB``) and **2.8e-17** (``h_BA``), four orders under
#: ``_ANCHOR_TOL``.
_CHAIN_TRUTH = {
    "h_AB": np.array(
        [
            0.96732762280898837899360109180522,
            0.23086689230373387723744625441654,
            0.10416414807156935624729313135820,
            0.01129506287064498789186875888256,
        ]
    ),
    "h_BA": np.array(
        [
            0.92503143833487241612154711250545,
            0.37454306530169057931648786267052,
            0.05593437791937427037005353223542,
            0.03009444622024623806971961761028,
        ]
    ),
}


def _assert_reference_matches_the_external_truth(refs, truth, what):
    """The tenax-derived reference must reproduce the ``decimal`` derivation.

    Both chain anchors call this.  It is what converts "certified outside
    tenax" from a docstring claim into a gate: without it the only live
    comparison is BP against ``FiniteMPS.compute_singular_values``, which
    shares ``tenax.linalg.svd``/``eigh`` and ``contract`` with the code under
    test, so a charge-bookkeeping defect common to both sides passes.
    """
    for bond, ref in refs.items():
        err = float(np.max(np.abs(ref - truth[bond])))
        assert err < _ANCHOR_TOL, (
            f"the {what} reference's {bond} is {err:.3e} away from the "
            f"externally derived spectrum (Python ``decimal`` on the transfer "
            f"matrix's fixed points, importing nothing from tenax).  This is "
            f"the only comparison here that is not tenax-against-tenax, so a "
            f"failure means either the draw/builder moved -- in which case "
            f"re-derive the constant, do not widen this -- or tenax and the "
            f"external derivation genuinely disagree: got "
            f"{np.array2string(ref, precision=17)} want "
            f"{np.array2string(truth[bond], precision=17)}"
        )


@pytest.mark.core
def test_bp_weights_are_the_chains_schmidt_values():
    """BP's fixed-point weights ARE the Schmidt values, on the one case we know.

    Marked ``core`` deliberately, and it is the only test in this file that is.
    ``conftest.py`` maps this file to ``algorithm``, and the required CI checks
    run ``pytest -m core`` -- so without this marker the one test in the whole
    rewrite with an externally known answer would land only in the non-required
    ``fast-other`` bucket, and a Phase 2/3 change that broke the anchor would
    leave every required check green.  The two markers coexist: the collection
    hook only *adds* the file's bucket marker, ``-m core`` is a positive
    selector, and the non-core buckets filter ``not core and not slow``, so
    this moves the test into the required gate rather than duplicating it.
    It costs 6.3 s.

    BP is exact on a tree, so on a 1D chain its converged bond weights are the
    exact Schmidt spectrum -- not an approximation to it.  §6.3 of the spec
    establishes that the loopy square lattice admits no valid reference
    spectrum, so this is the only place in the whole rewrite where the correct
    answer is known independently of the code being tested.

    Subject and reference are the *same state* by construction: one pair of
    random MPS tensors ``a``, ``b`` is repeated into a long finite MPS for the
    reference and embedded as a PEPS pair with dimension-1 vertical legs for
    the subject (see ``_ipeps_gauge_helpers``).  The plan's original version
    compared ``gauge_fix`` against a *different* finite random MPS's middle
    bond; those two spectra have no reason to agree and the only ways to make
    that pass are to loosen the tolerance or to compare almost nothing.

    Five claims, in order:

    1. The reference has converged in ``L``.  "Far from the boundaries" is an
       approximation and this task's whole value is that its reference is
       exact, so it is measured rather than assumed.
    1b. The reference reproduces the **externally derived** spectrum
       (:data:`_CHAIN_TRUTH`, and see *Certified outside tenax* below).  This
       is the only comparison in the file with a side that is not tenax.
    2. BP reproduces it on **both** inequivalent horizontal bonds.  ``h_AB``
       and ``h_BA`` carry genuinely different spectra here (they differ by
       1.44e-01), and checking only one -- as the plan did -- leaves the other
       unpinned.
    3. The *crossed* pairing fails.  Comparing ``h_AB`` against the wrong-parity
       reference bond is a silent mistake, so the parity claim is asserted, not
       just reasoned about in a comment.
    4. The vertical bonds, which sit on dimension-1 legs, come back with shape
       ``(1,)``.  Free, and it catches a ``gauge_fix`` that has transposed its
       bond bookkeeping -- a defect class this project has hit twice (#834,
       #602).  The companion "and the value is exactly 1.0" check this once
       carried **cannot fail** and has been removed: ``lam / max(lam)`` is
       ``x/x`` on a length-1 vector, and the non-finite and all-zero escapes
       are already caught by ``assert info.converged`` above.  The symmetric
       twin says the same thing in its own claim 4.

    Measured on this branch (seed 10, ``d=2``, ``chi=4``, ``L=100`` reference)::

        h_AB vs h_AB-parity reference   1.776e-15
        h_BA vs h_BA-parity reference   3.712e-16
        h_AB vs h_BA-parity reference   1.437e-01   <- crossed, must fail
        h_BA vs h_AB-parity reference   1.437e-01   <- crossed, must fail

    and perturbing a single reference singular value by 1e-6 moves the
    comparison to between 2.2e-07 and 1.0e-06, i.e. five to six orders above
    the gate, so the tolerance is not doing the work.

    **Certified outside tenax, and now asserted.**  Everything else here runs
    through tenax -- ``FiniteMPS.compute_singular_values`` for the reference,
    ``contract`` / ``eigh`` / ``svd`` for the subject -- so agreement between
    them is not by itself independent: a charge-bookkeeping defect in the
    shared machinery (#834's class) is common-mode and passes both sides.  The
    infinite chain's spectra were therefore rebuilt from the transfer matrix's
    left and right fixed points in Python ``decimal``, importing nothing from
    tenax but the two site tensors.  Those 32-digit values are
    :data:`_CHAIN_TRUTH`, and claim 1b asserts the tenax reference against
    them, so the external certification lives in the required gate rather than
    in a review transcript.  Measured agreement: **8.3e-17** (``h_AB``) and
    **2.8e-17** (``h_BA``), against ``_ANCHOR_TOL`` = 1e-12.

    If claim 1b ever fails, it is what tells a regression from a
    re-derivation: a changed seed or a changed builder moves those numbers, a
    broken ``gauge_fix`` does not (it is asserted before ``gauge_fix`` runs).
    """
    a, b, vl, vr = _chain_pair()

    lo = _chain_middle_spectra(a, b, vl, vr, _CHAIN_L_LO)
    hi = _chain_middle_spectra(a, b, vl, vr, _CHAIN_L_HI)
    drift = max(float(np.max(np.abs(x - y))) for x, y in zip(lo, hi, strict=True))
    assert drift < _ANCHOR_TOL, (
        f"the reference chain's middle-bond spectrum still moves {drift:.3e} "
        f"between L={_CHAIN_L_LO} and L={_CHAIN_L_HI}, so it is not yet the "
        f"infinite chain's and cannot certify anything at {_ANCHOR_TOL:.0e}.  "
        f"Raise L -- do not loosen the tolerance below, which is the number "
        f"this whole test exists to defend."
    )

    ref_AB, ref_BA = (_as_spectrum(s) for s in hi)
    # Claim 1b, and it runs before the solve so a moved reference is reported as
    # a moved reference rather than as a BP defect.
    _assert_reference_matches_the_external_truth(
        {"h_AB": ref_AB, "h_BA": ref_BA}, _CHAIN_TRUTH, "dense chain"
    )

    A, B = _chain_pair_as_peps(a, b)
    _, _, w, info = gauge_fix(A, B, **_ANCHOR_BP_KW)
    assert info.converged, (
        f"BP did not converge on a chain, where it is exact: {info.iterations} "
        f"sweeps, residual {info.residual:.3e}"
    )

    got_AB, got_BA = _as_spectrum(w.h_AB), _as_spectrum(w.h_BA)
    for bond, got, want in (("h_AB", got_AB, ref_AB), ("h_BA", got_BA, ref_BA)):
        err = float(np.max(np.abs(got - want)))
        assert err < _ANCHOR_TOL, (
            f"BP's {bond} is {err:.3e} away from the chain's exact Schmidt "
            f"spectrum (BP converged in {info.iterations} sweeps to residual "
            f"{info.residual:.3e}).  BP is exact on a tree, so this is not a "
            f"tolerance to widen: got {np.array2string(got, precision=12)} "
            f"want {np.array2string(want, precision=12)}"
        )

    crossed = min(
        float(np.max(np.abs(got_AB - ref_BA))),
        float(np.max(np.abs(got_BA - ref_AB))),
    )
    assert crossed > 1e-3, (
        f"swapping the two reference bonds moves the comparison by only "
        f"{crossed:.3e}, so the assertions above cannot tell the h_AB parity "
        f"from the h_BA one and pass for the wrong reason.  This draw's two "
        f"bonds have gone (nearly) degenerate -- redraw _CHAIN_SEED."
    )

    # Shape only.  "and the value is exactly 1.0" cannot fail -- ``lam/max(lam)``
    # is x/x on a length-1 vector -- and the symmetric twin already documents
    # that in prose rather than asserting it.
    for bond in ("v_AB", "v_BA"):
        v = np.asarray(getattr(w, bond))
        assert v.shape == (1,), (
            f"{bond} sits on a dimension-1 leg but came back with shape "
            f"{v.shape}; gauge_fix has its bonds crossed"
        )


# --- the same anchor, on the block-sparse path ----------------------------

#: The two chain lengths the *symmetric* reference is measured at.  Measured
#: here, not inherited: a different state converges in ``L`` at a different
#: rate, and this one is much faster than the dense arm's.  Drift of the larger
#: of the two middle bonds: 28->36 6.4e-08, 36->44 2.6e-10, 44->52 1.0e-12,
#: 52->60 4.1e-15.  44/52 would *fail* the 1e-12 gate below on its own
#: (1.04e-12), which is the reason the pair is not 44/52; 52/60 sits at the f64
#: floor with three orders to spare.
_SYM_CHAIN_L_LO, _SYM_CHAIN_L_HI = 52, 60

#: What each horizontal bond's charge content must still be after gauging,
#: as ``(sectors, multiplicities)``.  Written out here from the definition
#: rather than read off the input pair, for the reason ``_INDEPENDENT_BOND_OF``
#: gives: an expectation copied from the object under test echoes its mistakes.
#: ``h_AB`` is ``a.r``/``b.l``, whose MPS space is ``_SYM_VIRT_AB =
#: [-1, -1, 0, 1]``; the flow inversion negates it to ``[1, 1, 0, -1]``, i.e.
#: sectors ``[-1, 0, 1]`` with multiplicities ``[1, 1, 2]``.  ``h_BA`` is
#: ``_SYM_VIRT_BA = [-1, 0, 1, 1]`` negated to ``[1, 0, -1, -1]``, i.e.
#: ``[2, 1, 1]``.  The two differ, which is what lets this double as a second,
#: structural read of the ``h_AB``/``h_BA`` parity the spectra are compared at.
_SYM_BOND_SECTORS = {
    "h_AB": ([-1, 0, 1], [1, 1, 2]),
    "h_BA": ([-1, 0, 1], [2, 1, 1]),
}

#: The flows the shipped iPEPS convention puts on the two horizontal legs
#: (``ipeps._wrap_as_dense_tensor``, ``ipeps.heisenberg_u1sz_init_pair``,
#: ``fermionic_ipeps._build_initial_fpeps_tensor``).  An MPS site is left IN /
#: right OUT, so reaching this *is* the flow inversion.
_SYM_PEPS_FLOWS = {"l": FlowDirection.OUT, "r": FlowDirection.IN}

#: The block-sparse arm's external truth, the twin of :data:`_CHAIN_TRUTH` and
#: carrying the same load: it is the only side of any comparison in this file
#: that is not tenax.  Rebuilt from the 2-site transfer matrix's left and right
#: fixed points in Python ``decimal`` for the seed-30 draw, truncated to 32
#: digits.  Two independent derivations agree on every digit shown -- this one
#: and the reviewer's, same method, separate implementations.
#:
#: It matters more here than on the dense arm.  The dense reference and the
#: dense subject are both all-zero-charge, so a charge-bookkeeping defect could
#: not reach either; here both sides are block-sparse and share the charge
#: machinery that #834/#602/#865 all lived in, which is exactly the common mode
#: a tenax-vs-tenax comparison cannot see.
#:
#: Measured on this branch: the tenax reference at ``L=60`` agrees with these to
#: **1.1e-16** on both bonds, four orders under ``_ANCHOR_TOL``.
_SYM_CHAIN_TRUTH = {
    "h_AB": np.array(
        [
            0.99427402003591040195305550186645,
            0.08453854492747805197392288081348,
            0.06518798423820836634457954834226,
            0.00478896796125356249540515639037,
        ]
    ),
    "h_BA": np.array(
        [
            0.75797768426499114645172401877520,
            0.64403699904936766172236098989718,
            0.10331921524594954594674073374516,
            0.00336359520860306414572422435044,
        ]
    ),
}


def _sectors_of(idx):
    """``(sectors, multiplicities)`` of a ``TensorIndex``, as plain ints."""
    return ([int(q) for q in idx.sectors], [int(m) for m in idx.multiplicities])


@pytest.mark.core
def test_bp_weights_are_the_symmetric_chains_schmidt_values():
    """The chain anchor again, with every leg of the subject charged.

    ``test_bp_weights_are_the_chains_schmidt_values`` is the same claim on the
    dense path, and it is structurally blind to the defect class this project
    keeps hitting: all of its charges are zero, so it cannot see block-sparse
    ``contract()`` pairing legs by charge **value** where dense pairs by
    **position** (#834), an enumeration-order versus charge-grouped bond
    (#602), or per-sector keep counts making an SVD discard the *largest*
    singular value (#865).  The third of those was a symmetric simple-update
    collapse, and "dense was unaffected" turned out to be a confound.  Landing
    the symmetric simple update on top of a gauge with no ground truth on the
    block-sparse path repeats that setup exactly.

    This is not a comparison against the dense arm and must not become one:
    symmetric and dense are *expected* to differ exactly even for the same
    physics (#602), so the reference is built independently on the block-sparse
    path -- one symmetric 2-site cell, repeated into the long MPS *and* embedded
    as the PEPS pair.

    Marked ``core`` for the reason the dense anchor's docstring gives.  It
    costs 13.7 s bare and 19.5 s under the coverage instrumentation the suite
    runs with (three and two repeats, spread under 2%) -- most of it
    ``gauge_fix``'s 25 BP sweeps, which are block-sparse and host-bound, plus
    ~2.4 s for the two reference chains.  That is 2.4x the dense anchor and the
    largest single test this branch adds to the required gate; it buys the only
    ground truth the block-sparse gauge has.  There is no cheap lever: dropping
    ``tol`` to 1e-13 saves 2 sweeps of 25 and cuts the margin under the gate
    from 987x to 85x, and 1e-12 saves 4 and leaves 5x, which is unusable.

    Six claims:

    1. The reference has converged in ``L``, measured at two lengths.
    1b. The reference reproduces the **externally derived** spectrum
       (:data:`_SYM_CHAIN_TRUTH`, and see *Certified outside tenax* below).
    2. BP reproduces it on **both** inequivalent horizontal bonds.
    3. The *crossed* pairing fails, so the bond parity is pinned rather than
       assumed.
    4. The vertical bonds sit on dimension-1 legs and come back with shape
       ``(1,)``.  Those legs carry charge ``+1``, not 0, so this also says no
       charged dimension-1 sector was dropped.  (The companion "and the value
       is 1.0" check the plan once had cannot fail -- ``lam / max(lam)`` is
       exactly 1.0 on a length-1 vector -- so it is not repeated here.)
    5. The charge bookkeeping, in two halves, because one object cannot carry
       both claims.  On the **input** pair: the horizontal legs have the
       shipped iPEPS flows (``l`` OUT / ``r`` IN) over the *dualised* charge
       spaces, i.e. the flow inversion actually happened.  On the **gauged**
       pair: every horizontal bond still carries the sectors and
       multiplicities it started with, i.e. the gauge moved no states between
       charge sectors.  Neither claim exists on the dense path at all.

       The split is not tidiness.  This task's review replaced both
       ``.dual()`` calls with the bare MPS indices -- no inversion -- and the
       whole test still passed, twice: ``gauge_fix`` rebuilds each horizontal
       leg from its own ``eigh``/``svd`` bond, so the *gauged* metadata is
       identical under both conventions and is structurally blind to the one
       thing this arm exists to pin.
    6. ``info.converged``.

    Measured on this branch (seed 30, spin-1 physical, chi=4, ``L=60``
    reference; BP converged in 25 sweeps to residual 2.4e-15)::

        h_AB vs h_AB-parity reference   7.078e-16
        h_BA vs h_BA-parity reference   1.013e-15
        h_AB vs h_BA-parity reference   5.595e-01   <- crossed, must fail
        reference drift L=52 -> L=60    4.108e-15

    and perturbing any single reference singular value by 1e-6 moves the
    comparison to between 8.4e-08 and 1.0e-06, four to six orders above the
    gate.

    **Certified outside tenax, and now asserted**, the same way the dense
    anchor is and for the same reason -- both sides of every other comparison
    above are tenax code, so their agreement is not independent of tenax, and
    on *this* arm the shared machinery is the charge bookkeeping that #834,
    #602 and #865 all lived in.  The 32-digit values are
    :data:`_SYM_CHAIN_TRUTH`; claim 1b asserts the tenax reference against
    them, at a measured **1.1e-16** on both bonds.  Two independent derivations
    agree on every digit: this one and the reviewer's, same method, separate
    implementations.  BP against that truth, rather than against the finite
    chain, is 7.85e-16 (``h_AB``) and 9.42e-16 (``h_BA``).
    """
    a, b, vl, vr = _sym_chain_pair()

    lo = _sym_chain_middle_spectra(a, b, vl, vr, _SYM_CHAIN_L_LO)
    hi = _sym_chain_middle_spectra(a, b, vl, vr, _SYM_CHAIN_L_HI)
    lo_s = [_as_spectrum(s) for s in lo]
    hi_s = [_as_spectrum(s) for s in hi]
    drift = max(float(np.max(np.abs(x - y))) for x, y in zip(lo_s, hi_s, strict=True))
    assert drift < _ANCHOR_TOL, (
        f"the symmetric reference chain's middle-bond spectrum still moves "
        f"{drift:.3e} between L={_SYM_CHAIN_L_LO} and L={_SYM_CHAIN_L_HI}, so "
        f"it is not yet the infinite chain's and cannot certify anything at "
        f"{_ANCHOR_TOL:.0e}.  Raise L -- do not loosen the tolerance below, "
        f"which is the number this whole test exists to defend."
    )

    # Claim 1b, before the solve, for the reason the dense twin gives.
    _assert_reference_matches_the_external_truth(
        {"h_AB": hi_s[0], "h_BA": hi_s[1]}, _SYM_CHAIN_TRUTH, "symmetric chain"
    )

    A, B = _sym_chain_pair_as_peps(a, b)

    # Claim 5, first half, and it has to be read off the *input* pair.  The
    # review of this task showed why: replacing both ``.dual()`` calls with the
    # bare MPS indices -- no inversion at all -- leaves the *gauged* pair's
    # metadata bit-identical, because gauge_fix rebuilds each horizontal leg
    # from its own eigh/SVD bond and puts the shipped flows on it either way.
    # Every other assertion in this test passed under that mutation, twice.
    for site, t in (("A", A), ("B", B)):
        labels = t.labels()
        for leg, want_flow in _SYM_PEPS_FLOWS.items():
            idx = t.indices[labels.index(leg)]
            bond = _INDEPENDENT_BOND_OF[(site, leg)]
            assert idx.flow == want_flow, (
                f"{site}.{leg} has flow {idx.flow.name}, not {want_flow.name}: "
                f"the subject is not in the convention gauge_fix will be handed "
                f"by the simple update (l OUT / r IN).  An MPS site is left IN / "
                f"right OUT, so this is the flow inversion "
                f"_sym_chain_pair_as_peps exists to perform -- if it was "
                f"simplified away, put it back rather than relaxing this."
            )
            assert _sectors_of(idx) == _SYM_BOND_SECTORS[bond], (
                f"{site}.{leg} (bond {bond}) carries {_sectors_of(idx)}, not "
                f"{_SYM_BOND_SECTORS[bond]}.  The inversion must negate the "
                f"charges as well as flip the flow (TensorIndex.dual), with the "
                f"block keys moved to match; flipping the flow alone does not "
                f"conserve and negating alone is not the shipped convention."
            )

    A_g, B_g, w, info = gauge_fix(A, B, **_ANCHOR_BP_KW)
    assert info.converged, (
        f"BP did not converge on a symmetric chain, where it is exact: "
        f"{info.iterations} sweeps, residual {info.residual:.3e}"
    )

    # Bond parity as _middle_bond_pair derives it: site i is ``a`` for even i,
    # ``mid`` is forced even, and _sym_chain_pair_as_peps sends the MPS right
    # bond to ``r``, so the first of the two middle bonds is ``a.r <-> b.l`` ==
    # h_AB.  Claim 5 below confirms it a second way, from the charge content.
    ref_AB, ref_BA = hi_s
    got_AB, got_BA = _as_spectrum(w.h_AB), _as_spectrum(w.h_BA)
    for bond, got, want in (("h_AB", got_AB, ref_AB), ("h_BA", got_BA, ref_BA)):
        # Before subtracting.  numpy broadcasts a length-1 weight vector
        # against the length-4 reference -- a collapsed bond scored 9.95e-01
        # rather than raising, when that was reachable -- and a length-2 one
        # raises a bare ValueError with no context.  Both are lost charge
        # sectors; both should arrive as this assertion rather than as either.
        assert got.shape == want.shape == (4,), (
            f"BP's {bond} came back with shape {got.shape} against a reference "
            f"of shape {want.shape}; a bond that changed dimension is a lost "
            f"charge sector, not a numerical difference"
        )
        err = float(np.max(np.abs(got - want)))
        assert err < _ANCHOR_TOL, (
            f"BP's {bond} is {err:.3e} away from the symmetric chain's exact "
            f"Schmidt spectrum (BP converged in {info.iterations} sweeps to "
            f"residual {info.residual:.3e}).  BP is exact on a tree, so this is "
            f"not a tolerance to widen: got {np.array2string(got, precision=12)} "
            f"want {np.array2string(want, precision=12)}"
        )

    crossed = min(
        float(np.max(np.abs(got_AB - ref_BA))),
        float(np.max(np.abs(got_BA - ref_AB))),
    )
    assert crossed > 1e-3, (
        f"swapping the two reference bonds moves the comparison by only "
        f"{crossed:.3e}, so the assertions above cannot tell the h_AB parity "
        f"from the h_BA one and pass for the wrong reason.  This draw's two "
        f"bonds have gone (nearly) degenerate -- redraw _SYM_CHAIN_SEED."
    )

    for bond in ("v_AB", "v_BA"):
        v = np.asarray(getattr(w, bond))
        assert v.shape == (1,), (
            f"{bond} sits on a dimension-1 leg but came back with shape "
            f"{v.shape}; gauge_fix has its bonds crossed"
        )

    # Claim 5, second half: the *gauge* moved no states between charge sectors.
    # Blind to the flow convention -- gauge_fix rebuilds these legs from its own
    # eigh/SVD bonds, so only the input check above can see that -- and blind to
    # a mis-built subject too, which the input check now rejects first.  What is
    # left is the real claim: a gauge_fix that redistributes weight across
    # charges (#865) or pairs legs by position (#834) lands here.  Proved
    # discriminating by swapping the two expected entries, which fails.
    for site, t in (("A", A_g), ("B", B_g)):
        labels = t.labels()
        for leg in ("l", "r"):
            idx = t.indices[labels.index(leg)]
            bond = _INDEPENDENT_BOND_OF[(site, leg)]
            got_sectors = _sectors_of(idx)
            assert got_sectors == _SYM_BOND_SECTORS[bond], (
                f"gauging changed the charge content of {site}.{leg} (bond "
                f"{bond}): {got_sectors} vs {_SYM_BOND_SECTORS[bond]}.  The "
                f"gauge is supposed to be a change of basis within each sector; "
                f"a sector that lost or gained states means the SVD behind it "
                f"redistributed weight across charges (#865) or the legs were "
                f"paired by position rather than by charge value (#834)."
            )
