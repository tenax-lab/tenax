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
    _WITNESS_PAIRS,
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
    ``chi`` the two runs keep slightly different subspaces.  Measured, that
    floor falls with ``chi`` (dense, D=2: 5.5e-4 at chi=4 down to 2.5e-09 at
    chi=32) and with the strength of the gauge (fermionic, D=2, chi=8:
    1.1e-03 at ratio 3.09 down to 1.5e-05 at ratio 1.008), which is the
    signature of truncation rather than a defect -- it is flat in neither.
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
        f"is far above the finite-chi truncation floor and means the witness is "
        f"not measuring a gauge-invariant quantity."
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

    The control residual above is a finite-``chi`` truncation artefact, and
    the way to keep that claim honest is to require it to behave like one.
    A genuine contraction defect -- a dropped sign, a mispaired leg -- would
    sit at a fixed size no matter how much environment it is given, so
    "shrinks with chi" is what distinguishes the two.  Dense only: this needs
    several CTM runs and the dense path is by far the cheapest.
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
