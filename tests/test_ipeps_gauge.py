"""The absorbed-form convention: a site tensor IS the wavefunction.

The Vidal warning in bp_gauge_checkerboard is about passing ones() alongside
BARE Gamma, which drops lambda.  Absorbing sqrt(lambda) into both ends first
and then passing ones() is the same state written differently --
A sqrt(lam) . sqrt(lam) B == A lam B -- and that is what this file proves.

The convention is a contract on gauge_fix's *output* as much as its input: the
pair it returns must already be the state, with the weights carried alongside
as a diagnostic.  Reading a returned pair with its own weights and reading it
with ones() cannot both be right, and only one test here distinguishes them --
test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one.  Every other test in
this file is blind to the difference by construction, which is how the output
form was wrong for four commits.

Runs on the symmetric path too: a flow mistake there collapses charge sectors
rather than raising, and the dense path cannot see it.

The last test in the file is a different kind of claim from the rest.  Every
other test here -- and every acceptance criterion in Phase 1 -- is a
*self-consistency* check: the state did not move, the two routes agree, the
witness separates.  ``test_bp_weights_are_the_chains_schmidt_values`` compares
BP's output against an answer known a priori, because a chain is a tree and BP
is exact on trees.  It is the only such anchor available (#882 §6.3), so if it
fails, nothing measured downstream of it means anything.
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
    _torus_2x2,
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

ABSORB_TOL = 1e-13


def _unit(x):
    return x / np.linalg.norm(x)


def _torus_rel(x, y):
    """Distance between two torus readings **as states**.

    Comparison convention, used by every state-equality assertion here: both
    sides are normalised and the overall sign is free.  ``gauge_fix`` rescales
    each site tensor by max-abs and max-normalises each weight vector -- both
    deliberate, both unobservable -- and the torus is degree 4 in every site
    tensor, so an overall factor is *expected* and is not a state difference.
    Skipping the normalisation scores a reading that is exact to 1e-15 at
    6.5e-01 instead, which is how this defect was first mis-measured.

    Accepts a ``Tensor`` or an array; the torus is a ``(d,d,d,d)`` object, so
    densifying it is cheap even on the symmetric path.
    """
    a = _unit(np.asarray(x.todense() if hasattr(x, "todense") else x).ravel())
    b = _unit(np.asarray(y.todense() if hasattr(y, "todense") else y).ravel())
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


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

    The two routes are read with *different* weights, which is the point: the
    ``bp_gauge_checkerboard`` route returns Vidal and is read with its ``w1``,
    while ``gauge_fix`` returns absorbed and is read with ``ones``.  Both name
    the same state.  This assertion previously read the ``gauge_fix`` route as
    Vidal too, and passed -- because ``gauge_fix`` used to return Vidal.  It
    was testing agreement between the routes, which is real, and not the
    returned form, which nothing tested; see
    ``test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one``.
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


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
@pytest.mark.parametrize("D", [2, 3])
def test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one(kind, D):
    """Drop the weights and the state must be unchanged.  Both directions.

    This is the claim the whole absorbed-form convention rests on and it was
    asserted nowhere: every other test in this file passes ``gauge_fix``'s
    weights back into the torus, so all of them exercise the *Vidal* reading
    and none of them can tell the two apart.  Phase 2's ``_su_step`` holds no
    spectrum -- ``_SUState`` has nowhere to put one (#882 §3) -- so if the
    returned pair is not already the state, the step silently evolves a
    different one and reports a plausible, wrong energy.  That is #667 and
    #865 verbatim.

    The second assertion is what makes this a *discriminating* guard rather
    than one that merely accepts an absorbed pair: feeding the weights back in
    double-counts them, so the Vidal reading has to break.  A test that only
    checked the first direction would still pass if ``weights`` silently
    degenerated to all-ones, which is the trivial way to satisfy it.

    Measured on this branch, before and after the fix (``gauge_fix`` output,
    torus read the two ways, normalised and sign-free)::

        pair          before: +ones    +w2  |  after: +ones      +w2
        dense D=2         1.252e+00  4e-15  |     3.7e-15  3.983e-01
        dense D=3         8.948e-01  2e-14  |     1.7e-14  2.776e-01
        symmetric D=2     6.051e-01  8e-16  |     8.2e-16  8.999e-02
        symmetric D=3     2.491e-01  6e-15  |     5.7e-15  8.444e-01

    Before, the two readings were swapped: the pair alone was off by 0.25 to
    1.25 and only the Vidal reading was exact.  The thresholds below sit
    between the two columns with room to spare -- the tightest discriminating
    cell is symmetric D=2 at 9.0e-02, 9x above the 1e-2 gate.
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


@pytest.mark.parametrize("kind", list(_WITNESS_PAIRS))
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
    fermionic arm does not pass this.  Its floor at D=2 is 3.13e-03 (chi=6),
    1.06e-03 (chi=8), 1.24e-03 (chi=12) -- about 2.5x across a 2x change in
    ``chi``, against the 10x required below (``task-4-report.md``).
    Parametrising this over ``_WITNESS_PAIRS`` would therefore fail, and it
    would be reporting something real rather than being flaky.  That is
    recorded rather than asserted because this round may not spend a fermionic
    CTM run; the question is deferred to #882 §5.2.  So read this test as
    validating the *dense* witness only -- it says nothing about the fermionic
    one, which the docstring on :func:`ctm_rdm2x1_planar` describes instead.
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

#: Both the reference's self-agreement in ``L`` and BP's agreement with it.
#: Measured: reference 3.1e-16, BP 1.8e-15 (``h_AB``) and 3.7e-16 (``h_BA``).
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

    This is deliberately not ``_torus_rel``: that compares two readings of a
    *tensor* and allows an overall sign, which is meaningless for a vector of
    singular values.
    """
    v = np.sort(np.asarray(x))[::-1]
    return v / np.linalg.norm(v)


def test_bp_weights_are_the_chains_schmidt_values():
    """BP's fixed-point weights ARE the Schmidt values, on the one case we know.

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

    Four claims, in order:

    1. The reference has converged in ``L``.  "Far from the boundaries" is an
       approximation and this task's whole value is that its reference is
       exact, so it is measured rather than assumed.
    2. BP reproduces it on **both** inequivalent horizontal bonds.  ``h_AB``
       and ``h_BA`` carry genuinely different spectra here (they differ by
       1.44e-01), and checking only one -- as the plan did -- leaves the other
       unpinned.
    3. The *crossed* pairing fails.  Comparing ``h_AB`` against the wrong-parity
       reference bond is a silent mistake, so the parity claim is asserted, not
       just reasoned about in a comment.
    4. The vertical bonds, which sit on dimension-1 legs, come back as a single
       exact 1.0.  Free, and it catches a ``gauge_fix`` that has transposed its
       bond bookkeeping -- a defect class this project has hit twice (#834,
       #602).

    Measured on this branch (seed 10, ``d=2``, ``chi=4``, ``L=100`` reference)::

        h_AB vs h_AB-parity reference   1.776e-15
        h_BA vs h_BA-parity reference   3.712e-16
        h_AB vs h_BA-parity reference   1.437e-01   <- crossed, must fail
        h_BA vs h_AB-parity reference   1.437e-01   <- crossed, must fail

    and perturbing a single reference singular value by 1e-6 moves the
    comparison to between 2.2e-07 and 1.0e-06, i.e. five to six orders above
    the gate, so the tolerance is not doing the work.
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

    A, B = _chain_pair_as_peps(a, b)
    _, _, w, info = gauge_fix(A, B, **_ANCHOR_BP_KW)
    assert info.converged, (
        f"BP did not converge on a chain, where it is exact: {info.iterations} "
        f"sweeps, residual {info.residual:.3e}"
    )

    ref_AB, ref_BA = (_as_spectrum(s) for s in hi)
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

    for bond in ("v_AB", "v_BA"):
        v = np.asarray(getattr(w, bond))
        assert v.shape == (1,), (
            f"{bond} sits on a dimension-1 leg but came back with shape "
            f"{v.shape}; gauge_fix has its bonds crossed"
        )
        assert abs(float(v[0]) - 1.0) < 1e-15, (
            f"{bond} is a single number on a dimension-1 bond and BP "
            f"max-normalises every weight vector, so it must be exactly 1.0; "
            f"got {float(v[0]):.17g}"
        )
