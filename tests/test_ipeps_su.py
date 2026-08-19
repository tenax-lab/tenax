"""The simple-update engine that cannot hold a stale bond spectrum (#882 Phase 2).

``_SUState`` has two fields and no third.  That is the whole premise of the
rewrite -- the defect class behind #667, #851, #865 and #869 is a stored
spectrum invalidated by a non-unitary gate on *another* bond, and the design
deletes it by making it unrepresentable rather than by fixing it a fifth time.
A guard on the dataclass is cheap and it is the only thing that stays true as
the module grows.

``_su_step`` then needs **four** separate guards, because a step can be wrong in
four ways and no one reading sees more than one of them:

* **state** -- is it the *gated* state?  Checked exactly, by stepping with no
  truncation and comparing the 2x2 torus against the gate applied to the input
  torus.  This is also the no-op control: a step that returns its input scores
  3.1e-02 to 4.1e-02 where a correct one scores 8.0e-15 to 9.1e-15.
* **split** -- is ``sqrt(sigma)`` on *both* ends of the bond, rather than all
  of it on one?
* **truncation** -- of the singular values available, are these the largest
  ``max_D``, and are they attached to the right singular *vectors*?  Nothing
  above looks at either: an ``_su_step`` keeping the **smallest** ``max_D``
  -- retaining 0.0000 of the spectral weight -- once passed all 23 dense tests
  here with every other reading clean, and one keeping the top ``max_D``
  *values* against the bottom ``max_D`` *vectors* passed all 28 with the
  spectrum reading itself at 9.5e-16 while returning a state **orthogonal** to
  the correct one.  Spectra are blind to which subspace they describe, so this
  guard carries a second, subspace-level assertion.
* **gaugeability** -- can the next step's ``gauge_fix`` still solve on this
  output?  That is a convergence check and nothing more; see
  ``test_su_step_output_can_still_be_gauged`` for why its former state-equality
  half was removed rather than kept as decoration.

**Every bond reading below is taken in the Vidal metric** -- the pair with one
more ``sqrt(lambda)`` on the legs *outside* the bond, which is what
:func:`_vidal_pair` builds and what ``_su_step`` truncates in.  That is Task
10's reopening (``task-10-reopen-report.md``) and it is not a refactor: the
split and truncation guards used to read in *absorbed* form, where the
environment of the two-site tensor is not the identity, and in that metric they
were true of a step that truncates the wrong tensor and false of one that does
not.  They passed through four review rounds while ``_su_evolve`` scored 0 of 9
on energy.  A guard whose reference is re-derived in the same wrong metric as
the code cannot see a wrong metric; the one that could,
``test_su_step_truncates_in_the_state_s_own_basis``, compares against
Eckart-Young rather than against anything ``_su_step`` computed.

The first three are mutually blind, and for the first two that is arithmetic
rather than a gap in the probes.  A diagonal weight factors arbitrarily between
the two legs it joins without changing the contracted value, so ``sqrt(s)`` at
both ends and ``s`` at one end are the same physical *state*; every
closed-network reading in this tree, the torus included, calls them equal.  Nor
does the state guard follow from the split guard: a squared bond weight is a
different state that is still, formally, a perfectly good absorbed pair.
``test_the_bond_guards_see_different_mutations`` measures both directions on
mutations built in the test itself, so neither claim rests on argument.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The two externally-derived chain spectra come from here too.  They used to be
# imported from ``test_ipeps_gauge`` -- one test module importing another, which
# is the practice this helper exists to avoid (#882 final review, P3-15).
# Imported rather than copied either way: they are certified outside tenax
# (rebuilt from the transfer matrix's fixed points in Python ``decimal``), and a
# second copy of a certified constant is a second thing that can go stale.
from _ipeps_gauge_helpers import (  # tests/ is on sys.path
    _CHAIN_TRUTH,
    _PAIRS,
    _SYM_CHAIN_TRUTH,
    _chain_pair,
    _chain_pair_as_peps,
    _sym_chain_pair,
    _sym_chain_pair_as_peps,
    assert_leg_split,
)

import tenax.algorithms.ipeps_su as ipeps_su_module
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms.ipeps import (
    _wrap_as_dense_tensor,
    heisenberg_gate,
    sublattice_rotate_gate,
)
from tenax.algorithms.ipeps_bp_gauge import BondWeights, BPGaugeInfo
from tenax.algorithms.ipeps_gauge import gauge_fix, torus_2x2_sign_free
from tenax.algorithms.ipeps_simple_update import (
    _make_trotter_gate_tensor,
    _simple_update_checkerboard_sweep,
    _to_physical_pair,
)
from tenax.algorithms.ipeps_su import (
    _BOND_ENDS,
    _align_gate_to_ket,
    _sqrt_and_inv_sqrt,
    _su_evolve,
    _su_step,
    _SUState,
)
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.index import TensorIndex
from tenax.core.tensor import DenseTensor


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_su_state_from_pair_round_trips_the_tensors(kind):
    """``from_pair`` stores exactly what it is handed, and validates labels."""
    A, B = _PAIRS[kind](D=3)
    state = _SUState.from_pair(A, B)

    assert state.A is A
    assert state.B is B

    with pytest.raises(ValueError, match="checkerboard site tensor"):
        _SUState.from_pair(A.relabel("phys", "p"), B)


def test_su_state_has_no_lambda_fields():
    """The design's premise, as an executable assertion.

    If a lambda field ever appears here, the bug class this rewrite deletes
    (#667/#851/#865/#869) becomes representable again -- and it does not come
    back as a wrong number, it comes back as a plausible one.  #869's D=3 run
    peaks at -0.654 near 800 steps and falls to -0.477 by 4000, from the same
    code: one step count reports either verdict.

    ``hasattr`` on the *class* covers the property and method namespace as well
    as the fields, so a ``lambdas`` cached on the instance by a later task is
    caught only if it is declared; a bare ``object.__setattr__`` on a frozen
    dataclass is not something this can see.  The field check is the load-
    bearing half.
    """
    names = {f.name for f in dataclasses.fields(_SUState)}
    assert names == {"A", "B"}, f"_SUState grew fields: {names - {'A', 'B'}}"
    for bad in ("lam", "lam_h", "lam_v", "lambdas", "weights", "singular_values"):
        assert not hasattr(_SUState, bad), f"_SUState.{bad} must not exist"


@pytest.mark.parametrize("kind", ["dense", "symmetric"])
def test_su_state_max_D_reads_every_virtual_leg(kind):
    """``max_D`` is the largest *bond* dimension, and ``phys`` is not a bond.

    A pair whose four virtual legs all share ``D`` cannot discriminate: it
    passes on ``A.r.dim``, on ``min(...)``, on ``max(A.shape)``, and on
    anything else that happens to return ``D``.  Two anisotropic pairs are used
    instead, and between them every wrong reading this could plausibly be lands
    somewhere else:

    * the chain anchor -- ``D_v = 1``, ``D_h = 4`` -- kills ``min`` (1) and
      ``A.u.dim`` (1);
    * the same pair with the horizontal and vertical legs exchanged -- a pure
      relabel, and flow-preserving, since ``u``/``l`` are both OUT and
      ``d``/``r`` both IN -- kills ``A.r.dim`` (1).

    Neither chain pair excludes ``phys``, and the previous version of this note
    claimed otherwise -- it said the assertion was "on the leg *set*", which no
    line here does; ``max_D`` is compared to a number and nothing else.  It was
    measured false in #882's final review: the mutant
    ``max(idx.dim for t in (A, B) for idx in t.indices)`` -- ``phys`` included,
    contradicting :attr:`_SUState.max_D`'s own docstring -- **passes both
    parametrisations**, because every pair above has ``phys.dim == 2`` and a
    largest virtual dim of 3 or 4, so the two readings agree arithmetically.

    The third row is what closes it, and it is the only fixture in the file
    where ``phys`` is larger than every virtual leg: ``(2, 2, 2, 2, 4)``, so
    ``max_D`` must be **2** and a ``phys``-including reading returns 4.
    Measured against the same three variants: shipped 2, ``phys``-included 4,
    ``min`` 2 -- the first two now disagree, which is what the earlier fixtures
    could not make them do.  ``min`` is still killed by the chain rows (1
    against 4), so the pair of controls between them pins both directions.

    ``_SUState.max_D`` is not read anywhere in ``src/`` today, so this is a
    forward-looking guard: a spin-1 or Hubbard site at small ``D`` in Phase 4 is
    exactly the configuration where a ``phys``-including reading stops being
    arithmetically indistinguishable and starts returning the wrong bond
    dimension.
    """
    A, B = _PAIRS[kind](D=3)
    assert _SUState.from_pair(A, B).max_D == 3

    a, b, _vl, _vr = _chain_pair()
    Ac, Bc = _chain_pair_as_peps(a, b)
    swap = {"u": "l", "l": "u", "d": "r", "r": "d"}
    As, Bs = Ac.relabels(swap), Bc.relabels(swap)

    # ``phys`` above every virtual leg -- the only pair here that separates
    # "largest virtual dim" from "largest dim".  Dense on both parametrisations
    # on purpose: the property is metadata arithmetic on ``.indices``, identical
    # for both tensor types, and building a symmetric pair at ``d=4`` would add
    # a charge layout this assertion says nothing about.
    kA, kB = jax.random.split(jax.random.PRNGKey(7))
    Ap = _wrap_as_dense_tensor(jax.random.normal(kA, (2, 2, 2, 2, 4)))
    Bp = _wrap_as_dense_tensor(jax.random.normal(kB, (2, 2, 2, 2, 4)))

    for name, pair, expected, phys_dim, want in (
        ("chain", (Ac, Bc), {"u": 1, "d": 1, "l": 4, "r": 4}, 2, 4),
        ("swapped", (As, Bs), {"u": 4, "d": 4, "l": 1, "r": 1}, 2, 4),
        ("phys-dominant", (Ap, Bp), {"u": 2, "d": 2, "l": 2, "r": 2}, 4, 2),
    ):
        t = pair[0]
        dims = {
            lab: t.indices[t.labels().index(lab)].dim for lab in ("u", "d", "l", "r")
        }
        assert dims == expected, f"{name}: {dims}"
        got_phys = t.indices[t.labels().index("phys")].dim
        assert got_phys == phys_dim, f"{name}: phys is {got_phys}, not {phys_dim}"
        assert _SUState.from_pair(*pair).max_D == want, (
            f"{name}: max_D read {_SUState.from_pair(*pair).max_D}, expected "
            f"{want} -- the pair's virtual legs are {dims} and its phys leg is "
            f"{got_phys}, so a reading that included phys returns "
            f"{max(list(dims.values()) + [got_phys])}"
        )


# --- _su_step (#882 Task 10) ---------------------------------------------

D = 3

#: Full rank of the two-site ``theta`` across a bond at ``D``, ``d = 2``: three
#: virtual legs and one physical.  Stepping with this as ``max_D`` truncates
#: nothing, which is what makes
#: :func:`test_su_step_applies_the_gate_across_the_bond` an exact identity
#: rather than an identity plus a truncation error to argue about.
_FULL_RANK = D**3 * 2

_BONDS = ("h_AB", "h_BA", "v_AB", "v_BA")

#: ``(kind, bond)`` cells.  All four bonds on ``dense``; two on ``symmetric``.
#:
#: Not a coverage compromise made blind -- a symmetric ``_su_step`` at ``D=3``
#: costs **38 s**, effectively all of it the eager BP solve inside
#: ``gauge_fix`` (measured: 52 sweeps at the default ``tol``, ~0.7 s/sweep,
#: against 1.06 s for the whole dense step, which is traced).  Eight symmetric
#: cells would put this file past ten minutes on its own.
#:
#: The two chosen cells are the ones that differ structurally: ``h_AB`` has its
#: ``IN`` end on ``A`` and is horizontal, ``v_BA`` has its ``IN`` end on ``B``
#: and is vertical, so between them every row of ``_BOND_ENDS`` is exercised in
#: both of the two ways an entry can be built.  What the symmetric arm is for
#: is the charge bookkeeping -- the new bond's sectors, and whether the two ends
#: pair the way the dense path would (#834, #602) -- and that is a property of
#: the bond, not of which of the four it is.  The exhaustive four-bond sweep,
#: which is where a transposed leg map would show, runs on ``dense``.
_CASES = [("dense", b) for b in _BONDS] + [("symmetric", b) for b in ("h_AB", "v_BA")]

#: Cells for the truncation-quality guard, which needs **two** extra steps per
#: cell (a truncated one and an untruncated reference at the same ``dt``) and so
#: costs 76 s on a symmetric cell against 0.6 s on a dense one.  All four dense
#: bonds, and ``h_AB`` on symmetric -- the symmetric cell is not optional, since
#: ``base_charges`` is *ignored* on the dense path (``linalg.svd``'s own
#: docstring), so #865 has no dense arm to be caught on.
_TRUNCATION_CASES = [("dense", b) for b in _BONDS] + [("symmetric", "h_AB")]

#: Time step for the truncation guard.  At ``dt=0.05`` the truncation drops
#: 0.04% of the weight and pinning ``base_charges`` is *inert* -- measured, the
#: kept spectrum is identical to the unpinned one at ``max_D=3`` and at
#: ``max_D=2``.  At ``dt=2.0`` it drops 27% (dense) / 26% (symmetric) and the
#: pin bites: it keeps ``1, 0.95831, 0.41614`` where the largest three of the
#: untruncated spectrum are ``1, 0.95831, 0.59483``, a relative error of
#: 1.787e-01.  (Those are the figures
#: ``test_su_step_keeps_the_largest_singular_values`` records and the
#: ``865-base-charges-pinned-on-the-truncation-guard`` mutation row kills at.
#: This constant used to carry a second, stale copy -- "keeps 0.43903 where the
#: top three are 1, 0.9915, 0.46407" -- which disagreed with it; there is now
#: one set of numbers and it is the measured one.)
#: A guard for #865 has to run where #865 is visible.
_TRUNCATION_DT = 2.0

#: Which two physical legs of the 2x2 torus each bond's gate acts on, as
#: ``((si_axis, sj_axis), (si_axis, sj_axis))`` into the torus's
#: ``(A00, B10, B01, A11)`` axis order.
#:
#: Re-derived here from ``ipeps_gauge``'s own edge table rather than imported
#: from ``_BOND_ENDS``, so a transposed map in the implementation cannot echo
#: itself into the expectation.  That table reads::
#:
#:     i = A00.r-B10.l  h_AB     m = A00.d-B01.u  v_AB
#:     j = B10.r-A00.l  h_BA     n = B01.d-A00.u  v_BA
#:     k = B01.r-A11.l  h_BA     o = B10.d-A11.u  v_BA
#:     l = A11.r-B01.l  h_AB     p = A11.d-B10.u  v_AB
#:
#: Each bond appears on exactly **two** of the eight edges, and both take the
#: gate: an iPEPS step evolves every copy of the bond in the lattice, not one
#: of them.  ``si`` goes to the end whose leg is ``r`` or ``d`` -- the
#: left/upper site -- which is the convention every gate builder in this tree
#: assumes.
_TORUS_GATE_AXES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "h_AB": ((0, 1), (3, 2)),  # edges i, l
    "h_BA": ((1, 0), (2, 3)),  # edges j, k
    "v_AB": ((0, 2), (3, 1)),  # edges m, p
    "v_BA": ((2, 0), (1, 3)),  # edges n, o
}


def _ones_for(A):
    """``BondWeights.ones`` at *this* pair's four bond dimensions.

    Every weight is **1**, so this adds nothing to the network: it is the
    identity weight :func:`torus_2x2_sign_free` requires for a pair that is
    already in absorbed form.  Nothing here re-gauges, and nothing here is a
    gauge -- the previous version of this note said otherwise and #882's final
    review measured it false.

    The four lengths are read off the pair's **own four virtual legs**, which
    is the same per-leg sizing ``ipeps_gauge._identity_weights`` does since
    #887.  ``BondWeights.ones(D_h, D_v)`` cannot: it gives both horizontal
    bonds one dimension and both vertical bonds another, which is right only
    while all four agree.  A single ``_su_step`` changes exactly one bond's
    dimension and leaves the other three alone, so **every** stepped pair below
    is mid-cycle -- truncating it (``max_D=2``) and growing it
    (``max_D=_FULL_RANK``) break the two-dimension shape alike.

    Measured, dense ``D=3``, one ``h_AB`` step at ``max_D=2`` (legs
    ``{r: 2, u: 3, d: 3, l: 3}``): this helper returns lengths
    ``{h_AB: 2, h_BA: 3, v_AB: 3, v_BA: 3}``, ``torus_2x2_sign_free`` accepts
    them, and ``gauge_fix`` on that same pair converges in 20 sweeps at
    residual 5.5e-07 -- so the pair is **not** ungaugeable, and this helper's
    shape was never what stopped anything.  ``BondWeights.ones(A.r.dim,
    A.d.dim)`` returns ``{2, 2, 3, 3}`` on it and the torus raises
    ``TypeError: cannot reshape array of shape (2,) ... into shape
    [1, 1, 1, 3, 1]``.

    Watched failing rather than argued: with the two-dimension form installed
    in place of this body, ``test_su_step_applies_the_gate_across_the_bond``
    ``[dense-h_AB]`` dies at ``TypeError: cannot reshape array of shape (54,)
    ... into shape [1, 1, 1, 3, 1]`` -- 54 being the ``_FULL_RANK`` bond that
    step grows.
    """
    dim = {lab: A.indices[A.labels().index(lab)].dim for lab in ("u", "d", "l", "r")}
    return BondWeights(
        h_AB=jnp.ones(dim["r"]),
        h_BA=jnp.ones(dim["l"]),
        v_AB=jnp.ones(dim["d"]),
        v_BA=jnp.ones(dim["u"]),
    )


def _unit(x):
    x = np.asarray(x.todense() if hasattr(x, "todense") else x).ravel()
    return x / np.linalg.norm(x)


def _torus_rel(x, y):
    """Distance between two torus readings **as states**: normalise, then compare.

    The convention ``test_ipeps_gauge.py:_torus_rel`` establishes, for its
    reason: ``gauge_fix`` rescales each site tensor by max-abs and ``_su_step``
    does not renormalise at all, so an overall factor is *expected* and is not
    a state difference.  Skipping the normalisation does not merely inflate the
    number, it makes it meaningless in either direction -- the torus inherits
    whatever scale its two site tensors happen to carry.
    """
    return float(np.linalg.norm(_unit(x) - _unit(y)))


#: ``(site, leg) -> bond``, both ends of all four bonds, derived here from
#: ``_BOND_ENDS`` rather than imported from ``ipeps_su._BOND_OF_LEG``.  The
#: implementation derives its own copy the same way, and keeping this one
#: separate is what makes a hand-edited map in the module visible from here.
_BOND_OF_LEG = {
    (site, leg): bond for bond, ends in _BOND_ENDS.items() for site, leg in ends
}


def _vidal_pair(state, bond, weights, power=1.0):
    """``bond``'s two sites with one more ``sqrt(lambda)`` on their outer legs.

    **Every bond reading in this file goes through here, and that is the
    correction Task 10's reopening carries.**  ``gauge_fix`` returns the pair in
    *absorbed* form -- each bond weight split ``sqrt(lambda)`` into both of its
    ends -- so a two-site tensor built straight from it carries only
    ``lambda**1`` across the ket-bra pair on each outer leg.  The Vidal
    canonical condition wants ``lambda**2`` there, so in absorbed form the
    environment of the two-site tensor is **not** the identity: its singular
    values are not the state's Schmidt values, its singular vectors are not
    Schmidt vectors, and a bond spectrum read in that metric is a spectrum of
    the wrong operator.

    ``_su_step`` truncates in this metric (its stages 2 and 5), so the guards
    that check *what it kept* have to read in it too.  They did not, and the
    absorbed-metric readings they used were true only of the pre-fix step,
    which is the same thing as saying they were calibrated against the defect.
    Measured on the ``dt=2.0`` truncation cells, correct step, both metrics:

    ==============================  ===================  =================
    reading                         absorbed metric      Vidal metric
    ==============================  ===================  =================
    end Gram off-diagonal *        2.1e-01 to 9.7e-01   <= 2.7e-15
    ``|G_i - G_j|`` (the split)     4.1e-01 to 1.5e+00   <= 1.4e-14
    ``diag(G)`` against ``sigma``   1.9e-01 to 6.4e-01   <= 7.1e-15
    kept against top-``D``          3.8e-03 to 5.4e-02   <= 1.4e-15
    ``cos`` against ``sqrt(1-d)``   5.8e-04 to 6.6e-03   <= 1.0e-15
    ==============================  ===================  =================

    (``*`` = the four **dense** cells; the symmetric one reads exactly 0 in both
    metrics, structurally, and the split guard says why.)

    Every one of those five sits at machine precision in the Vidal metric and
    between 3.8e-03 and 1.5e+00 in the absorbed one -- twelve orders of
    magnitude apart and more -- on a step whose truncation is independently
    certified optimal by ``test_su_step_truncates_in_the_state_s_own_basis``.
    So the metric was the error, not the step.

    ``weights`` must be the ``BondWeights`` ``gauge_fix`` returns for the
    **input** pair -- the same ones the step used.  They describe the *state*,
    not the step, and a step cannot make them agree with it by being wrong:
    the untouched outer legs are what they are.

    Args:
        state:   An ``_SUState``, or any ``{"A": ..., "B": ...}`` mapping.
        bond:    Which bond's two sites to take.
        weights: ``gauge_fix``'s spectrum for the pair these legs came from.
        power:   Net power of ``lambda`` to leave on each outer leg **in the
                 ket**, counting what absorbed form already carries.  ``1.0``
                 is the Vidal metric and the only value any guard reads;
                 the knob exists so that
                 ``test_the_vidal_metric_matches_a_spectrum_derived_outside_tenax``
                 can score the two neighbouring powers against an external
                 spectrum and show that only this one lands on it.  ``0.0``
                 reproduces the absorbed reading this file used before Task
                 10's reopening.

    Returns:
        ``{"A": ..., "B": ...}`` with the six outer legs weighted.  The bond's
        own two legs are untouched -- weighting those would be #667.
    """
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]
    pair = dict(state) if isinstance(state, dict) else {"A": state.A, "B": state.B}
    for site, on_bond in ((site_i, leg_i), (site_j, leg_j)):
        for leg in ("u", "d", "l", "r"):
            if leg == on_bond:
                continue
            lam = np.asarray(getattr(weights, _BOND_OF_LEG[(site, leg)]), dtype=float)
            pair[site] = scale_bond_axis(
                pair[site], leg, jnp.asarray(lam ** (power / 2.0))
            )
    return pair


def _gram(t, leg):
    """``<t|t>`` traced over every leg but ``leg``, as a dense ``(leg, leg)`` matrix.

    This is the split guard's whole content, and it is only a statement about
    the split when ``t`` is an end **in the Vidal metric** -- i.e. what
    :func:`_vidal_pair` returns.  There the step's two outputs are
    ``F_j = U sqrt(sigma)`` and ``F_i = sqrt(sigma) Vh`` with ``U`` and ``Vh``
    the SVD's isometries, so ``F_j^dag F_j`` and ``F_i F_i^dag`` are both
    ``diag(sigma)`` -- the *same* matrix at both ends, diagonal, and equal to
    the bond's spectrum.  Put the whole weight on one end and they become
    ``diag(sigma**2)`` and the identity.  Read the same ends in absorbed form
    and ``U^dag Lambda^-1 U`` sits between them, which is neither diagonal nor
    equal at the two ends however the split was made; see :func:`_vidal_pair`
    for that measurement.

    Unlike an element-wise comparison of the tensors, this is invariant under
    the sign and basis freedom the SVD and BP's ``eigh`` each carry, and under
    the **scale** freedom as well -- which is what makes it usable at all.

    The scale is the part the earlier version of this note got wrong.  It said
    ``gauge_fix`` applied to a stepped pair comes back "1.7e-01 (dense) ...
    away from it element-wise ... because the two routines pick different bases
    on the bond".  Re-measured, ``dt=0`` gate on an already-gauged dense input,
    ``max ||x| - |y||``: the raw reading is **2.68e-02** (``A``) / **2.92e-02**
    (``B``), not 1.7e-01 -- and once each side is divided by its own max-abs it
    falls to **6.2e-07** / **4.3e-07**, which is ``gauge_fix``'s own
    ``tol=1e-6``.  So that reading is scale-dominated and the stated mechanism
    is the opposite of what it measures: for a ``dt=0`` gate on a gauged input
    the two routines pick the *same* basis to within the solve tolerance, and
    what separates them element-wise is ``gauge_fix``'s max-abs rescale.  (Dense
    only; the 5.7e-03 symmetric figure was not re-measured.)  A basis-free
    reading is still what this returns and is still the right tool -- it is also
    scale-free, which the raw element-wise one is not; the number that justified
    it was the wrong number.

    ``todense()`` here is a ``(D, D)`` bond matrix -- always small.
    """
    m = contract(t, t.bar().relabel(leg, "__bra"))
    m = m.transpose(tuple(m.labels().index(lab) for lab in (leg, "__bra")))
    return np.asarray(m.todense())


def _theta_labels(bond):
    """The rename maps and SVD leg groups for reassembling ``bond``'s ``theta``.

    Written out once so :func:`_two_site_tensor` and every reader of the tensor
    it returns agree on which leg is which without importing ``_su_step``'s own
    private names for them.
    """
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]

    def rename(leg, prefix, phys):
        out = {lg: prefix + lg for lg in ("u", "d", "l", "r") if lg != leg}
        out[leg] = "__shared"
        out["phys"] = phys
        return out

    ren_i = rename(leg_i, "__i", "__pi")
    ren_j = rename(leg_j, "__j", "__pj")
    left = [ren_j[lg] for lg in ("u", "d", "l", "r") if lg != leg_j] + ["__pj"]
    right = [ren_i[lg] for lg in ("u", "d", "l", "r") if lg != leg_i] + ["__pi"]
    return (site_i, ren_i), (site_j, ren_j), left, right


def _two_site_tensor(state, bond, weights, power=None):
    """The two sites sharing ``bond``, contracted across it, in the Vidal metric.

    This is the ``theta`` an ``_su_step`` on ``bond`` produced: ``F_j F_i``,
    which is ``U sqrt(sigma) sqrt(sigma) Vh = U diag(sigma) Vh`` however the
    ``sqrt`` was shared out between the two factors.  Two states stepped from
    the *same* input at the same ``dt`` give tensors on identical outer legs --
    the internal ``gauge_fix`` is deterministic and the truncation touches only
    the bond -- so they are directly comparable, element for element.

    ``weights`` is not optional and there is no absorbed-metric route to this:
    see :func:`_vidal_pair`.  A reading of this tensor without them is a
    reading of a different operator, and it is the one that let the truncation
    defect through four review rounds.
    """
    (site_i, ren_i), (site_j, ren_j), _left, _right = _theta_labels(bond)
    # ``power=None`` forwards nothing, so :func:`_vidal_pair`'s signature is the
    # **one** place the shipped metric is written down.  Repeating ``1.0`` here
    # would mask a changed default from every caller -- measured: with the
    # literal in place, mutating ``_vidal_pair``'s default passed the chain
    # anchor.
    pair = (
        _vidal_pair(state, bond, weights)
        if power is None
        else _vidal_pair(state, bond, weights, power)
    )
    return contract(pair[site_j].relabels(ren_j), pair[site_i].relabels(ren_i))


def _bond_spectrum(state, bond, max_D, weights, base_charges=None, power=None):
    """The Schmidt spectrum of ``state`` across ``bond``, independent of the split.

    Reassembles the two-site tensor in the Vidal metric (:func:`_vidal_pair`)
    and takes its singular values.  Because ``F_j F_i`` is the same object
    however the ``sqrt`` was shared out between the two factors, this number is
    the same for a correct split, a one-sided one, and anything in between --
    which is exactly what makes it a valid reference for the split check rather
    than a restatement of it.

    ``base_charges`` is exposed so a test can build the **pinned** truncation of
    the same ``theta`` and show that this configuration can tell the two apart
    (#865).  It is ``None`` for every reading of the step itself.
    """
    _ends_i, _ends_j, left, right = _theta_labels(bond)
    _U, sigma, _Vh, _full = truncated_svd(
        _two_site_tensor(state, bond, weights, power),
        left_labels=left,
        right_labels=right,
        new_bond_label="__reference_bond",
        max_singular_values=max_D,
        base_charges=base_charges,
    )
    return np.asarray(sigma)


def _theta_cosine(state_a, state_b, bond, weights):
    """``<theta_a, theta_b> / (||theta_a|| ||theta_b||)`` across ``bond``.

    The **subspace** reading, and the only one in this file that is not blind to
    a truncation that keeps the right singular *values* attached to the wrong
    singular *vectors*.  Every spectrum-based reading here -- the truncation
    guard's own, the Gram/split guard, :func:`_bond_spectrum` -- is invariant
    under replacing the kept subspace with any other orthonormal one of the same
    dimension, because a spectrum does not know what it is a spectrum *of*.

    For a correct truncation ``theta_kept = P theta_full`` with ``P`` the
    orthogonal projector onto the top-``max_D`` subspace, so the inner product
    is ``sum(kept sigma**2)`` and this returns exactly
    ``sqrt(1 - dropped)``.  Keeping the bottom ``max_D`` vectors instead makes
    the two tensors orthogonal and this returns 0.

    ``bar()`` supplies the conjugate with flows flipped, so every leg pairs; the
    contraction is closed and the result is a scalar.

    In the **Vidal** metric, like every other bond reading here: the projector
    ``P`` is orthogonal only there.  Measured on the same correct step, the
    absorbed-metric reading of this quantity misses ``sqrt(1 - dropped)`` by
    5.8e-04 to 6.6e-03 against a guard at 1e-11, so reading it in the wrong
    metric does not merely blunt the check -- it fails on correct code.
    """
    ta = _two_site_tensor(state_a, bond, weights)
    tb = _two_site_tensor(state_b, bond, weights)
    num = float(
        np.asarray(contract(ta, tb.bar(), output_labels=[]).todense()).reshape(())
    )
    return num / (float(ta.norm()) * float(tb.norm()))


def _apply_gate_to_torus(T, G, axes):
    """Apply ``G`` to the named physical-leg pairs of a 2x2 torus reading.

    ``T`` is ``(A00, B10, B01, A11)`` and ``G`` is ``(si, sj, si_out, sj_out)``;
    both are ``d**4 == 16`` entries, so this stays in numpy and owes nothing to
    the contraction machinery it is checking.
    """
    letters = "wxyz"
    for si_axis, sj_axis in axes:
        out = list(letters)
        p, q = letters[si_axis], letters[sj_axis]
        out[si_axis], out[sj_axis] = "P", "Q"
        T = np.einsum(f"{letters},{p}{q}PQ->{''.join(out)}", T, G)
    return T


def _asymmetric_hamiltonian():
    """Heisenberg plus a field on the **first** site only.

    Not a modelling choice -- ``0.7 * Sz (x) I`` is there to break the gate's
    symmetry under exchanging its two sites, and without it two of this file's
    guards are vacuous.

    The 2x2 torus has four sites, so ``A(0,0)`` and ``B(1,0)`` are horizontal
    neighbours **twice**: once through edge ``i`` (``h_AB``) and once through
    the wrap-around edge ``j`` (``h_BA``).  Evolving either bond therefore acts
    on the same two *sites*, and the only thing that distinguishes the two on
    the torus is which of them takes the gate's ``si`` leg.  With a plain
    Heisenberg gate -- ``G[si,sj,*] == G[sj,si,*]`` -- there is no distinction
    at all: measured, ``_su_step(..., "h_AB")`` and ``_su_step(..., "h_BA")``
    produce torus readings that agree to 1e-15, and a ``_BOND_ENDS`` with its
    two horizontal rows transposed passed every test in this file.

    ``Sz (x) I`` conserves ``Sz``, so the block-sparse arm is unaffected: no
    sector is created or destroyed and ``_make_trotter_gate_tensor`` drops
    nothing when it rebuilds the gate on the site's own physical charges.
    """
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    H = np.asarray(heisenberg_gate().todense()).reshape(4, 4)
    return (H + 0.7 * np.kron(Sz, np.eye(2))).reshape(2, 2, 2, 2)


def _gate(A, dt=0.05):
    return _make_trotter_gate_tensor(
        jnp.asarray(_asymmetric_hamiltonian()), dt, site_tensor=A
    )


@pytest.fixture(scope="module")
def su():
    """Memoised pairs and stepped states, shared by every test below.

    Module-scoped and cached because a symmetric ``_su_step`` costs 38 s: the
    four guards below would otherwise re-run the same step four times and this
    file would take a quarter of an hour to say nothing extra.  Nothing here
    mutates a state, so sharing them is safe -- ``_SUState`` is frozen and the
    tensors under it are immutable.
    """
    pairs: dict[str, tuple] = {}
    steps: dict[tuple, object] = {}
    gauges: dict[str, object] = {}

    class _Cache:
        def pair(self, kind):
            if kind not in pairs:
                pairs[kind] = _PAIRS[kind](D=D)
            return pairs[kind]

        def weights(self, kind):
            """``gauge_fix``'s spectrum for this kind's **input** pair.

            The metric every bond reading below is taken in
            (:func:`_vidal_pair`).  Memoised because a symmetric solve is 38 s
            and five cells want it; convergence is asserted here rather than in
            each of them, since a spectrum from a failed solve is not the
            state's and nothing downstream could tell.
            """
            if kind not in gauges:
                A, B = self.pair(kind)
                A_g, B_g, w, info = gauge_fix(A, B)
                assert info.converged, (
                    f"{kind}: BP did not converge on the input pair "
                    f"({info.iterations} sweeps, residual {info.residual:.3e}), "
                    f"so the metric every bond reading below is taken in is not "
                    f"the state's"
                )
                gauges[kind] = (A_g, B_g, w)
            return gauges[kind][2]

        def gauged(self, kind):
            """The gauged pair ``_su_step`` forms its ``theta`` from.

            Off the *same* solve :meth:`weights` runs -- ``gauge_fix`` returns
            the pair and the spectrum together and this used to throw the pair
            away.  Keeping it is what lets a guard build the step's own input
            ``theta`` without paying a second 38 s symmetric BP solve, and
            ``gauge_fix`` is deterministic, so it is the same pair
            ``_su_step``'s opening call produced.
            """
            self.weights(kind)
            return gauges[kind][0], gauges[kind][1]

        def step(self, kind, bond, max_D=D, dt=0.05):
            key = (kind, bond, max_D, dt)
            if key not in steps:
                A, B = self.pair(kind)
                steps[key] = _su_step(
                    _SUState.from_pair(A, B), _gate(A, dt=dt), max_D=max_D, bond=bond
                )
            return steps[key]

    return _Cache()


@pytest.mark.parametrize("kind,bond", _CASES)
def test_su_step_output_can_still_be_gauged(kind, bond, su):
    """The next step's ``gauge_fix`` converges on this step's output.

    This is what survives of the plan's §6.2a round-trip, and the shrinkage is
    measured rather than editorial.  That guard also asserted that the re-gauge
    did not move the state, and **that assertion fired on none of ten
    mutations** -- including one whose output is numerically zero
    (``keep_smallest``, 0.00% of the spectral weight kept, round-trip reading
    7.892e-15).  It could not have: ``gauge_fix`` *is* a gauge, so it preserves
    whatever state it is handed, well-formed or not.  Nor is the block-sparse
    justification it carried reachable -- a new-bond charge mis-pairing that
    keeps flow and dim cannot be built as a valid ``SymmetricTensor``
    (``__init__`` raises on the conservation check), and one that flips a flow
    is caught by ``test_su_step_returns_the_convention_it_accepts`` first.  An
    assertion nobody can make fail is decoration, so it is gone.

    **The convergence assertion has no demonstrated kill, and the trigger this
    docstring used to name for it is refuted on the exact cell it named.**  It
    claimed the assertion had "been watched to use them: with the gate's flows
    left unaligned, this fires on symmetric ``v_BA`` at 100 sweeps, residual
    1.399e-01".  Measured with ``_align_gate_to_ket`` neutered to the identity,
    on all six cells this test parametrises::

        dense     h_AB  converged=True  15 sweeps  residual 7.156e-07
        dense     h_BA  converged=True  15 sweeps  residual 8.097e-07
        dense     v_AB  converged=True  15 sweeps  residual 7.028e-07
        dense     v_BA  converged=True  15 sweeps  residual 6.830e-07
        symmetric h_AB  converged=True  53 sweeps  residual 8.486e-07
        symmetric v_BA  converged=True  67 sweeps  residual 8.769e-07

    -- the last being the named cell.  The mechanism could not have produced it
    either: within one ``_su_step`` the internal ``gauge_fix`` runs *before* the
    gate is used at all, so a change to the gate's flows cannot move that solve.

    Four further production mutations were driven against it and **none** kills
    it: stage 5 multiplying instead of dividing, ``sigma**2`` on the bond, all
    of ``sigma`` on one factor (the plan's §6.2a hole), and stage 2/5 made inert
    (#869's flat weights) each leave an output BP still solves, on all four
    dense bonds.  So what stands here is a **precondition on the next step that
    has never been observed failing**, and this docstring says that rather than
    claiming a kill it does not have.  It is kept because the property is real
    -- the gauge sets the basis the next truncation is taken in -- and because
    reading it is one solve on a pair the ``su`` cache has already built.

    The *live* half of the same condition is elsewhere and is measured: one
    ``-m slow`` pass of this file and its mutation sibling emits 3339
    ``gauge_fix did not converge`` warnings from ``ipeps_su.py``, 41 of them
    inside five cells that report green, and
    ``test_su_step_warns_when_the_gauge_did_not_converge`` guards the reporting
    path with a fabricated ``BPGaugeInfo``.

    The other two properties are pinned elsewhere and neither is claimed here:
    ``test_su_step_applies_the_gate_across_the_bond`` pins the state,
    ``test_su_step_splits_sqrt_sigma_into_both_ends`` pins the share between
    the two ends, and ``test_su_step_keeps_the_largest_singular_values`` pins
    what the truncation kept.
    """
    stepped = su.step(kind, bond)
    _A2, _B2, _w2, info = gauge_fix(stepped.A, stepped.B)
    assert info.converged, (
        f"{kind} {bond}: BP did not converge on the step output in "
        f"{info.iterations} sweeps (residual {info.residual:.3e}).  The next "
        f"step would truncate in the basis this failed solve left behind."
    )


@pytest.mark.parametrize("kind,bond", _CASES)
def test_su_step_applies_the_gate_across_the_bond(kind, bond, su):
    """The step's *state* is the gated state, exactly -- and is not the input.

    This is the guard on what ``_su_step`` computes, as opposed to what shape
    it returns.  With ``max_D`` at the full rank of ``theta`` nothing is
    truncated, so the identity is exact and there is no tolerance to argue
    about: the stepped pair's 2x2 torus must equal the input pair's torus with
    the gate applied to **both** copies of ``bond``.  Both readings are gauge
    invariant, which is what lets the step re-gauge internally without
    disturbing the comparison.

    The expectation comes from ``_TORUS_GATE_AXES``, re-derived from
    ``ipeps_gauge``'s edge table rather than from ``_BOND_ENDS``, so a
    transposed bond-to-leg map has to be wrong twice in two independently
    written places to pass.

    It is also this module's **no-op control**, and the one guard that catches
    a wrong bond.  Measured on ``dense`` at ``D=3``, ``dt=0.05``: a correct
    step scores 8.0e-15 to 9.1e-15 (8.500e-15, 8.011e-15, 8.949e-15, 9.125e-15
    on ``h_AB``, ``h_BA``, ``v_AB``, ``v_BA``; the 8.8e-15 to 9.5e-15 this used
    to quote reproduced neither endpoint), an ``_su_step`` that returns
    ``state`` untouched scores 3.1e-02 to 4.1e-02, and a ``_BOND_ENDS`` with
    two of its rows transposed -- or with the gate's ``si``/``sj`` legs
    crossed -- scores 1.9e-02.  Several of the plan's Phase 2 tests pass on a do-nothing
    implementation; under a ``dt=0`` gate at ``max_D == D`` almost anything
    does, which is why the gate here is a real one and the control is asserted
    rather than assumed.
    """
    A, B = su.pair(kind)
    gate = _gate(A)
    stepped = su.step(kind, bond, max_D=_FULL_RANK)

    T_before = np.asarray(torus_2x2_sign_free(A, B, _ones_for(A)).todense())
    T_after = torus_2x2_sign_free(stepped.A, stepped.B, _ones_for(stepped.A))
    T_expected = _apply_gate_to_torus(
        T_before, np.asarray(gate.todense()), _TORUS_GATE_AXES[bond]
    )

    control = _torus_rel(T_before, T_expected)
    assert control > 1e-3, (
        f"{kind} {bond}: the gate moves this state by only {control:.3e}, so "
        f"the assertion below could not tell a working step from a no-op"
    )
    rel = _torus_rel(T_after, T_expected)
    assert rel < 1e-11, (
        f"{kind} {bond}: the untruncated step is {rel:.3e} away from applying "
        f"the gate to that bond (a no-op would score {control:.3e}).  Either "
        f"the wrong bond was evolved, the gate's legs are crossed, or the bond "
        f"weight came out at the wrong power."
    )


@pytest.mark.parametrize("kind,bond", _CASES)
def test_su_step_splits_sqrt_sigma_into_both_ends(kind, bond, su):
    """Each end of the updated bond carries ``sqrt(sigma)`` -- the same one.

    No closed-network probe can see this, and that is arithmetic rather than a
    gap in the probes: a diagonal weight factors arbitrarily between the two
    legs it joins without changing the contracted value, so ``sqrt(s)`` at both
    ends and ``s`` at one end are the *same physical state*.  The torus reads
    them as identical and so does the round-trip guard above;
    ``test_the_bond_guards_see_different_mutations`` measures that.

    The reading that does see it is each end's bond Gram matrix (see
    :func:`_gram`): both must be diagonal and equal to each other, and the
    weight they carry must be the *right power* of the bond's own spectrum.

    **The third assertion's reference is the gated INPUT theta, not the stepped
    pair, and that is #882's final review being right about this cell.**  It
    used to compare ``diag(G_i)`` against :func:`_bond_spectrum` of the same
    stepped state, and that comparison cannot fail: given ``off < 1e-11`` at
    each end and ``gap < 1e-11`` between them, write ``G_i = G_j = diag(d)``;
    then ``F_j = W sqrt(d)`` and ``F_i = sqrt(d) V^dag``, so ``theta = F_j F_i``
    has singular values exactly ``d`` -- and ``_bond_spectrum`` is the SVD of
    that same ``theta``, rebuilt by :func:`_two_site_tensor` from the very
    tensors the Gram was taken of.  The reference was re-derived from the code
    under test, one level down.  Measured: under
    ``absorb_sqrt_singular_values`` handed ``sigma**2`` -- literally the wrong
    power the message names -- the three readings are 9.99e-16, 3.55e-15 and
    3.55e-15 and **all four dense cells pass**, while
    ``test_su_step_applies_the_gate_across_the_bond`` fails at 2.298e-01.

    ``_su_step`` did not compute the gated input ``theta``'s spectrum, so it is
    a reference the split cannot move.  Both spectra are normalised on their
    leading value, because ``_su_step`` does not renormalise and ``gauge_fix``
    rescales by max-abs -- an overall factor is expected and is not a power
    error.  Measured on all four dense bonds, as the number *this* assertion
    takes: a correct step reads 2.2e-16 to 1.8e-15; ``sigma**2`` reads
    2.211e-01, 2.500e-01, 2.465e-01, 2.385e-01 and is killed **here**, having
    passed every assertion in this cell before; dropping ``sigma`` altogether
    (``F_j, F_i = U, Vh``) reads 3.301e-01 to 4.959e-01 and is also killed
    here.  All of ``sigma`` on one factor (the §6.2a hole) reads the same
    3.301e-01 to 4.959e-01, but ``gap`` fires on it first, at 3.45 -- which is
    the row ``tests/test_ipeps_su_mutations.py`` pins, and it is unchanged.

    **Both readings are taken in the Vidal metric, and that is the correction
    Task 10's reopening carries here.** ``_su_step``'s SVD is of the two-site
    tensor with the other three bonds' weights on the six outer legs, so its
    isometries are isometries *there*; in absorbed form ``U^dag Lambda^-1 U``
    sits between the two ends and neither the diagonality nor the equality
    holds however the split was made.  The absorbed-vs-Vidal numbers are in
    :func:`_vidal_pair`'s table and are the ``dt=2.0`` truncation cells', not
    this ``dt=0.05`` cell's -- this docstring used to quote them as though they
    were, which is the mis-attribution #882's final review found.  The
    absorbed reading was true only of the pre-fix step, which truncated in that
    metric -- so it was a guard calibrated against the defect,
    and it went on passing while ``_su_evolve`` scored 0 of 9 on energy.

    **The three claims are asserted before the meta-assertion**, which is the
    ordering Task 13's fix round established on the #865 guard for the same
    reason.  ``spread`` says the cell *can* discriminate -- a ``sigma`` flat at
    1 satisfies ``diag(sigma) == diag(sigma**2)`` -- but it is not one of the
    claims, and ordered first it pre-empts them: measured, under
    ``F_j, F_i = U, Vh`` (no ``sigma`` on the bond at all) ``spread`` reads
    1.000 on all four dense cells and reports that this cell cannot tell the
    two apart, where the power assertion reports the actual defect at 3.301e-01
    to 4.959e-01.  Detection was never in doubt; attribution was.  Reordering
    weakens nothing, because ``spread`` still runs on every green pass and
    still fails loudly on a cell that has gone blind.
    """
    stepped = su.step(kind, bond)
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]
    weights = su.weights(kind)
    pair = _vidal_pair(stepped, bond, weights)
    G_i = _gram(pair[site_i], leg_i)
    G_j = _gram(pair[site_j], leg_j)

    for name, G in (("i", G_i), ("j", G_j)):
        off = float(np.max(np.abs(G - np.diag(np.diag(G)))))
        # This one has teeth on the **dense** cells only, and the asymmetry is
        # structural rather than a sampling accident: at D=3 the symmetric
        # pair's bond carries charges (0, 1, -1), one basis vector per sector,
        # so a charge-conserving Gram is 1x1-block diagonal by construction and
        # reads exactly 0.000e+00 whatever the step did.  On dense it reads
        # 9.7e-01 in the wrong metric and 9.2e-16 in the right one.  Kept
        # because the four dense cells are where it can fail, and named here so
        # nobody reads the symmetric zeros as evidence.
        assert off < 1e-11, (
            f"{kind} {bond}: end {name}'s bond Gram matrix is not diagonal "
            f"(max off-diagonal {off:.3e}) -- the SVD factor is not an "
            f"isometry in the metric it was taken in, so what sits on the bond "
            f"is not a spectrum"
        )
    gap = float(np.max(np.abs(G_i - G_j)))
    assert gap < 1e-11, (
        f"{kind} {bond}: the two ends of the bond carry different weights "
        f"(max |G_i - G_j| = {gap:.3e}).  sqrt(sigma) must go into BOTH "
        f"factors; putting it on one end leaves the same physical state, so no "
        f"closed-loop probe in this tree would notice."
    )
    # The bond's spectrum as the *input* had it: the SVD of the gated two-site
    # tensor ``_su_step`` formed, built here from ``gauge_fix``'s own output
    # rather than from anything the step returned.  Truncation is why only the
    # top ``D`` are compared; normalisation is why an overall factor is not
    # read as a power error.
    A_in, _B_in = su.pair(kind)
    A_g, B_g = su.gauged(kind)
    order, theta_in = _vidal_theta({"A": A_g, "B": B_g}, weights, bond, _gate(A_in))
    left = [f"__j{lg}" for lg in ("u", "d", "l", "r") if lg != leg_j] + ["__pj"]
    perm = [order.index(x) for x in left] + [
        order.index(x) for x in order if x not in left
    ]
    arr = theta_in.transpose(perm)
    sigma_in = np.linalg.svd(
        arr.reshape(int(np.prod(arr.shape[: len(left)])), -1), compute_uv=False
    )
    ref = np.sort(sigma_in)[::-1][:D]
    ref = ref / ref[0]
    got = np.sort(np.diag(G_i).real)[::-1]
    got = got / got[0]
    err = float(np.max(np.abs(got - ref)))
    assert err < 1e-11, (
        f"{kind} {bond}: each end carries {got} where the top {D} of the "
        f"gated input theta's own spectrum are {ref} (max diff {err:.3e}) -- "
        f"the weight is on the bond at the wrong power.  Both are normalised "
        f"on their leading value, so this is not an overall-scale reading; the "
        f"reference is the SVD of the theta _su_step formed, taken from "
        f"gauge_fix's output and not from the step's."
    )

    # --- the meta-assertion, after the claims (see the docstring) -----------
    sigma = _bond_spectrum(stepped, bond, D, weights)
    spread = float(np.max(sigma) / np.min(sigma))
    assert spread > 1.1, (
        f"{kind} {bond}: sigma is flat ({spread:.3f}), so this test cannot "
        f"distinguish sqrt(sigma) at both ends from sigma at one"
    )


@pytest.mark.parametrize("kind,bond", _CASES)
def test_su_step_returns_the_convention_it_accepts(kind, bond, su):
    """Same leg order, same labels, same flows, same dimensions in and out.

    Not tidiness.  A pair's axis order and flow convention are part of the
    traced gauge's pytree treedef, hence of its compile-cache key, so a step
    that returned a different convention from the one it accepts would make
    every second call recompile: Phase 1 measured a second ``gauge_fix``
    compile at ~285 ms against a whole-run budget of 450 ms.  A 100-step
    evolve has to compile once.

    The flow half is the load-bearing one and it is not automatic.
    ``linalg.svd`` stamps ``OUT`` on ``U``'s new leg and ``IN`` on ``Vh``'s
    whatever it was handed, and ``_make_trotter_gate_tensor`` builds
    ``si_out``/``sj_out`` ``OUT`` against a site ``phys`` of ``IN``.  Taken
    naively the bond legs and the physical leg all come back inverted;
    ``_BOND_ENDS`` and ``_align_gate_to_ket`` are what put them back, and this
    is the test that says so.

    **Charges are checked as a multiset, not element-wise, and the asymmetry is
    deliberate.**  The internal ``gauge_fix`` derives every *untouched* virtual
    leg afresh from its own ``eigh``/``svd``, so their charge layouts come back
    permuted -- measured, symmetric ``h_AB``: ``A.u`` and ``A.d``
    ``(0,1,-1) -> (-1,0,1)``, ``A.l`` and ``B.r`` ``-> (1,-1,0)``, while the
    *updated* bond ``A.r``/``B.l`` keeps ``(0,1,-1)``.  That permutation is
    benign and asserting against it would fail on correct code.  What is not
    benign is a changed multiset, which would mean a charge sector had been
    created or destroyed, so that is what is pinned.

    ``phys`` is checked **element-wise**, because it is the one leg a caller
    holds a matching object for: a gate is built once from the initial pair and
    reused for every step, and ``_align_gate_to_ket`` refuses a gate whose
    charges do not match the site's element-wise.  A permuted ``phys`` would
    therefore turn a working run into a hard failure -- or, without that check,
    into a silent mis-pairing.  Measured, it is stable at ``(1,-1)``.

    Both charge assertions have been **watched to fail**, which matters because
    they are vacuous on the dense arm (every charge there is 0) and because the
    last assertion in this file that could not fail had to be deleted:

    * *multiset* -- pin the SVD's ``base_charges`` to the trivial layout
      (``zeros(max_D)``): ``A.r`` comes back carrying ``[0, 0, 0]`` where the
      pair had ``[-1, 0, 1]``, at the same dimension and the same flow, so no
      other assertion here sees it;
    * *element-wise* ``phys`` -- have ``_align_gate_to_ket`` **dualise** the
      gate rather than flip its flows (negating the charges as well): ``phys``
      comes back ``[-1, 1]`` where it was ``[1, -1]`` -- a permutation, so the
      multiset check above is blind to it by construction.

    Each fires on both symmetric cells and neither fires anything else.
    """
    A, B = su.pair(kind)
    stepped = su.step(kind, bond)

    for name, before, after in (("A", A, stepped.A), ("B", B, stepped.B)):
        assert after.labels() == before.labels(), (
            f"{kind} {bond} {name}: leg order {after.labels()} != {before.labels()}"
        )
        for i_before, i_after in zip(before.indices, after.indices):
            # Implied by the ``labels()`` equality four lines up -- kept because
            # it is what localises a mismatch to a leg if that ever stops being
            # true, not because it can fail on its own.
            assert i_after.label == i_before.label
            assert i_after.flow == i_before.flow, (
                f"{kind} {bond} {name}: leg {i_after.label} came back "
                f"{i_after.flow.name}, was {i_before.flow.name}"
            )
            assert i_after.dim == i_before.dim, (
                f"{kind} {bond} {name}: leg {i_after.label} came back at "
                f"dim {i_after.dim}, was {i_before.dim}"
            )
            q_before = np.sort(np.asarray(i_before.charges))
            q_after = np.sort(np.asarray(i_after.charges))
            assert np.array_equal(q_before, q_after), (
                f"{kind} {bond} {name}: leg {i_after.label} came back carrying "
                f"the charge multiset {list(q_after)}, was {list(q_before)} -- "
                f"a sector was created or destroyed, not merely re-ordered"
            )
        i_before = before.indices[before.labels().index("phys")]
        i_after = after.indices[after.labels().index("phys")]
        assert np.array_equal(
            np.asarray(i_before.charges), np.asarray(i_after.charges)
        ), (
            f"{kind} {bond} {name}: phys came back as "
            f"{list(np.asarray(i_after.charges))}, was "
            f"{list(np.asarray(i_before.charges))}.  A gate is built once from "
            f"the initial pair and reused every step; a permuted phys makes it "
            f"pair the wrong basis states."
        )


def test_the_bond_guards_see_different_mutations(su):
    """Neither bond guard subsumes the other, measured rather than argued.

    Both mutations are built here from a *correct* step rather than by breaking
    ``_su_step``, which makes the test durable: it pins the discriminating
    power of the two readings against the two failure modes they exist for, and
    it goes on doing so however the implementation is rewritten.

    * ``one_sided`` moves the whole bond weight onto one end -- the "just
      multiply sigma into one factor" mistake.  It is a **gauge**, so the state
      does not move at all and the state guard is blind to it by construction.
      Only the Gram reading sees it.
    * ``squared`` puts ``sigma`` on each end instead of ``sqrt(sigma)``, so the
      bond carries ``sigma**2``.  The two ends still agree, so the Gram reading
      is blind to it; only the state guard sees it.

    **The two readings live in different metrics and that is not incidental.**
    The torus is a closed contraction of the pair as it stands, so it must be
    taken on the raw absorbed pair; the Gram is a statement about the SVD's
    isometries, which are isometries in the *Vidal* metric only, so it is taken
    on :func:`_vidal_pair`'s reweighting of the same pair.  Mixing them is not a
    style question: the split reading of the unmutated step scores 5.3e-01 in
    the absorbed metric, so the ``baseline_split`` premise below fails outright
    and no contrast can be measured at all.  The mutants themselves are built on
    the **raw** pair, so ``one_sided`` stays the gauge transformation it claims
    to be under the torus.

    Measured here (``dense``, ``h_AB``, ``D=3``, ``dt=0.05``; both guards gate
    at 1e-11):

    ============  =================  ==================
    mutation      state (torus)      split (|Gi - Gj|)
    ============  =================  ==================
    correct step  8.5e-15            2.2e-15
    one_sided     3.6e-16  *blind*   3.4e+00  fires
    squared       2.3e-01  fires     3.6e-15  *blind*
    ============  =================  ==================

    (The ``correct step`` state figure is
    ``test_su_step_applies_the_gate_across_the_bond``'s reading at full rank,
    which is the only place a *state* has a reference to be right against; the
    two mutant rows are distances from the unmutated step.)
    """
    bond = "h_AB"
    stepped = su.step("dense", bond)
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]
    weights = su.weights("dense")
    raw = {"A": stepped.A, "B": stepped.B}
    root = jnp.asarray(np.sqrt(_bond_spectrum(stepped, bond, D, weights)))

    mutants = {
        # all of sigma on end j and none on end i: a re-split, not a new state
        "one_sided": {
            site_j: scale_bond_axis(raw[site_j], leg_j, root),
            site_i: scale_bond_axis(raw[site_i], leg_i, 1.0 / root),
        },
        # sigma on each end instead of sqrt(sigma): the bond carries sigma**2
        "squared": {
            site_j: scale_bond_axis(raw[site_j], leg_j, root),
            site_i: scale_bond_axis(raw[site_i], leg_i, root),
        },
    }

    # The one-sided mutant really is the leg-wise rescale it is claimed to be,
    # so the test's premise is pinned rather than assumed.  ``assert_leg_split``
    # is the tool Phase 1 built for exactly this comparison: element-wise,
    # against a bond map written out independently.
    site = raw[site_j]
    scale = {
        lg: np.ones(site.indices[site.labels().index(lg)].dim)
        for lg in ("u", "d", "l", "r")
    }
    scale[leg_j] = np.asarray(root)
    assert_leg_split(
        site_j, raw[site_j], mutants["one_sided"][site_j], scale, 1e-12, msg="mutant "
    )

    ref = torus_2x2_sign_free(stepped.A, stepped.B, _ones_for(stepped.A))
    # The unmutated baseline both readings are compared against below.  It
    # restates the split guard's first assertion on purpose: the two mutants
    # are derived from *this* pair, so if it were already split-asymmetric the
    # "one_sided fires / squared is blind" contrast would mean nothing.
    gauged = _vidal_pair(raw, bond, weights)
    baseline_split = float(
        np.max(np.abs(_gram(gauged[site_i], leg_i) - _gram(gauged[site_j], leg_j)))
    )
    assert baseline_split < 1e-11, (
        f"the unmutated step this test mutates is already split-asymmetric "
        f"({baseline_split:.3e}); the contrast below would be meaningless"
    )

    seen = {}
    for name, m in mutants.items():
        got = torus_2x2_sign_free(m["A"], m["B"], _ones_for(m["A"]))
        mv = _vidal_pair(m, bond, weights)
        seen[name] = (
            _torus_rel(got, ref),
            float(np.max(np.abs(_gram(mv[site_i], leg_i) - _gram(mv[site_j], leg_j)))),
        )

    # The four assertions below are about mutations **this test builds**, from
    # the ``root`` map written out above -- so they are statements of torus and
    # Gram algebra, not readings of ``_su_step``.  What ties them to real code
    # is the ``baseline_split`` assertion above, which is taken on the actual
    # step's output; without it the contrast would hold on any pair at all.
    # Labelled rather than removed, on the model of the ``jnp.sqrt`` VJP canary:
    # they are what would report a ``dt`` or draw at which the two guards stop
    # being complementary.
    state_gap, split_gap = seen["one_sided"]
    assert state_gap < 1e-11, (
        f"one_sided moved the state by {state_gap:.3e}; it is a gauge and must "
        f"not, or the split guard is not testing what this test claims"
    )
    assert split_gap > 1e-3, (
        f"the Gram reading scored {split_gap:.3e} on a fully one-sided split -- "
        f"it cannot see the failure mode it exists for"
    )

    state_gap, split_gap = seen["squared"]
    assert split_gap < 1e-11, (
        f"squared scored {split_gap:.3e} on the Gram reading; both ends agree "
        f"there, so this mutation has to be invisible to it"
    )
    assert state_gap > 1e-3, (
        f"the state reading scored {state_gap:.3e} on a squared bond weight -- "
        f"it cannot see a wrong total"
    )


def test_su_step_rejects_an_unknown_bond(su):
    """A checkerboard has four inequivalent bonds and this takes one of them."""
    A, B = su.pair("dense")
    with pytest.raises(ValueError, match="bond must be one of"):
        _su_step(_SUState.from_pair(A, B), _gate(A), max_D=D, bond="h")


@pytest.mark.parametrize("kind,bond", _TRUNCATION_CASES)
def test_su_step_keeps_the_largest_singular_values(kind, bond, su):
    """The truncation keeps the top ``max_D``, not some other ``max_D``.

    Nothing else in this file looks at *which* singular values survived, and
    the hole was not theoretical: an ``_su_step`` mutated to keep the
    **smallest** ``max_D`` values -- retaining 0.0000 of the spectral weight,
    kept ``sigma = [0, 0, 0]`` -- passed all 23 dense tests with every other
    guard reading clean.  Two independent reasons, both now closed here:
    ``test_su_step_applies_the_gate_across_the_bond`` steps at ``_FULL_RANK``
    and so never truncates at all, and ``_bond_spectrum`` re-derives its
    reference from the *stepped* state, so it agrees with whatever was kept by
    construction.

    The reference has to come from a **different** step: the untruncated one at
    the same ``dt``, whose bond spectrum is the full ``sigma`` of the very
    ``theta`` the truncated step formed (both start from the same state and the
    internal ``gauge_fix`` is deterministic).  Then "kept == top ``max_D``" is
    an assertion about the truncation and not a restatement of it.

    **Both spectra are read in the Vidal metric** (:func:`_vidal_pair`), which
    Task 10's reopening corrected.  "The largest ``max_D``" is only a
    well-posed statement once the metric is named: measured on a correct step,
    reading the same two spectra in absorbed form puts the kept set 3.8e-03 to
    5.4e-02 from the top ``D`` against a 1e-11 gate, and the subspace cosine
    5.8e-04 to 6.6e-03 from ``sqrt(1 - dropped)``.  The absorbed-metric reading
    was true only of the pre-fix step, which truncated in that metric, and it
    passed throughout the four review rounds that shipped the defect.

    This is also the only executable coverage of ``base_charges=None``, brief
    constraint #3 and one of the four bugs the rewrite exists to delete (#865).
    Pinning the new bond's per-sector keep counts to the old bond's layout stops
    the SVD taking the globally largest values.  Re-measured under a faithful
    re-introduction of that pin (``test_ipeps_su_mutations.py``'s ``_mutant_865``,
    this cell, ``dt=_TRUNCATION_DT``): it keeps ``[1, 0.95831, 0.41614]`` where
    the largest three of the untruncated spectrum are ``[1, 0.95831, 0.59483]``
    -- the third one is traded for a smaller value in another sector -- a
    relative error of **1.787e-01** against the 1e-11 gate below.  It shows up
    **only** on the symmetric arm and **only** at a ``dt`` where truncation
    bites -- see ``_TRUNCATION_DT``.

    **The claim is asserted before the symmetric meta-assertion, deliberately.**
    That block builds ``sigma_pinned`` by applying *the same pin the defect
    applies*, so under the defect its reference and the step coincide and the
    two readings swap exactly: separation 6.237e-16 where the claim reads
    1.787e-01.  Ordered meta-first -- which is how this cell stood until #882
    Task 13's fix round -- the meta-assertion fires and the claim is never
    evaluated, so the one cell this file names for #865 could report that it had
    *detected* the pin and never that its own assertion watches it.  Reordering
    weakens nothing, because the meta-assertion still runs on every green pass:
    a cell that has gone blind (a ``dt`` at which pinning stops separating) still
    fails there, loudly.  The claim's own kill is pinned by the row named
    ``865-base-charges-pinned-on-the-truncation-guard`` in
    ``tests/test_ipeps_su_mutations.py``.
    """
    A, B = su.pair(kind)
    weights = su.weights(kind)
    full = su.step(kind, bond, max_D=_FULL_RANK, dt=_TRUNCATION_DT)
    kept = su.step(kind, bond, max_D=D, dt=_TRUNCATION_DT)
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]

    sigma_full = np.sort(_bond_spectrum(full, bond, _FULL_RANK, weights))[::-1]
    sigma_kept = np.sort(_bond_spectrum(kept, bond, D, weights))[::-1]
    dropped = 1.0 - float(np.sum(sigma_kept**2) / np.sum(sigma_full**2))

    # --- the precondition: this cell exercises a real truncation ------------
    #
    # Ordered *before* the two claims, unlike the #865 meta-assertion at the
    # bottom, and that is measured rather than an oversight.  Under ``sigma**2``
    # on the bond this fires on dense ``v_AB`` at ``dropped = 0.0458`` -- but
    # the two claims genuinely pass under that mutation (spectrum err 3.4e-16,
    # subspace gap 2.2e-16, all four bonds at machine precision), because both
    # are *relative* comparisons between two steps taken under the same
    # mutation.  So moving this later would change which line reports and not
    # what is reported.  It also has to be computed here: the subspace claim's
    # own reference is ``sqrt(1 - dropped)``.
    assert dropped > 0.05, (
        f"{kind} {bond}: the truncation discards only {dropped:.4f} of the "
        f"weight at dt={_TRUNCATION_DT}, so this test is not exercising a real "
        f"truncation and cannot see a wrong keep set"
    )

    # --- what was kept: the right VALUES ... --------------------------------
    for name, site, leg in (
        (site_i, kept.A if site_i == "A" else kept.B, leg_i),
        (site_j, kept.A if site_j == "A" else kept.B, leg_j),
    ):
        got = site.indices[site.labels().index(leg)].dim
        assert got == D, (
            f"{kind} {bond}: end {name}'s bond leg came back at dim {got}, "
            f"expected {D}.  (Asserted on the tensor rather than on "
            f"len(sigma_kept): _bond_spectrum caps its own SVD at D, so it "
            f"could only ever see a step that kept too FEW.)"
        )
    err = float(np.max(np.abs(sigma_kept - sigma_full[:D])) / sigma_full[0])
    assert err < 1e-11, (
        f"{kind} {bond}: the step kept {sigma_kept / sigma_full[0]} where the "
        f"largest {D} of the untruncated spectrum are "
        f"{sigma_full[:D] / sigma_full[0]} (relative error {err:.3e}).  The "
        f"truncation is not taking the globally largest singular values -- "
        f"check that base_charges is None (#865), which pins the per-sector "
        f"keep counts and stops it doing so."
    )

    # --- ... attached to the right VECTORS ----------------------------------
    # Everything above is a spectrum reading, and a spectrum is invariant under
    # swapping the kept subspace for any other orthonormal one of the same
    # dimension.  The two assertions are therefore complementary, and measured
    # to be so -- dense h_AB, dt=2.0, both gating at 1e-11:
    #
    #   mutation                     spectrum err   subspace gap
    #   correct                      4.762e-16      0.000e+00
    #   top values, bottom vectors   9.525e-16      8.532e-01   <- only this
    #   keep the smallest max_D      1.000e+00      5.603e-18   <- only above
    #
    # The middle row passes every other test in this file (28 dense cells)
    # while returning a state ORTHOGONAL to the correct one: cos = -0.000000
    # where a projection onto the top-3 subspace must give 0.853179.
    cos = _theta_cosine(kept, full, bond, weights)
    expected = float(np.sqrt(1.0 - dropped))
    gap = abs(cos - expected)
    assert gap < 1e-11, (
        f"{kind} {bond}: the truncated step overlaps the untruncated one by "
        f"{cos:.6f} where a projection onto the top-{D} subspace must give "
        f"sqrt(1 - dropped) = {expected:.6f} (gap {gap:.3e}).  The kept "
        f"singular values are attached to the wrong singular vectors: the "
        f"spectrum is right and the retained subspace is not."
    )

    # --- last: the meta-assertion that this cell can see #865 at all --------
    # It runs *after* the claim, and the order is load-bearing: this block
    # builds its reference with the same ``base_charges`` pin #865 applies, so
    # under that defect reference and step coincide (separation 6.237e-16)
    # while the claim above reads 1.787e-01 -- placed first, it fired and the
    # claim was never reached.  See the docstring.  It still runs on every
    # green pass, which is the whole of its job: a cell that has gone blind
    # fails here rather than passing quietly.
    if kind == "symmetric":
        # ``dropped`` alone does NOT establish that #865 is visible here, and
        # the gap is measured rather than imagined: at dt=0.05, max_D=2 this
        # same cell drops 0.11580 of the weight -- 2.3x the gate above -- while
        # the pinned and unpinned truncations are byte-identical (1.606e-16).
        # So build the pin from the same ``theta`` and require it to separate.
        # ``base_charges`` is a *multiset* through ``_derive_charges`` (which
        # slices, and whose keep counts are a histogram), so the input pair's
        # leg charges pin exactly as the gauged pair's permutation of them do.
        src = {"A": A, "B": B}[site_i]
        pin = np.asarray(src.indices[src.labels().index(leg_i)].charges)
        sigma_pinned = np.sort(
            _bond_spectrum(full, bond, D, weights, base_charges=pin)
        )[::-1]
        sep = float(np.max(np.abs(sigma_pinned - sigma_kept)) / sigma_full[0])
        assert sep > 1e-3, (
            f"{kind} {bond}: pinning base_charges to {list(pin)} keeps the "
            f"same spectrum as not pinning it (separation {sep:.3e}), so this "
            f"cell cannot see #865 and the claim above is not covering it. "
            f"dt={_TRUNCATION_DT} was chosen to make it visible; re-measure "
            f"before changing it."
        )
    else:
        # ``linalg.svd`` documents base_charges as "Ignored on the dense path",
        # so there is no pin to separate from here and no meta-assertion to
        # make.  Measured: separation 4.8e-16 (h_AB), i.e. exactly nothing.
        # This is why _TRUNCATION_CASES carries a symmetric cell at all.
        pass


def test_su_step_survives_a_bond_direction_the_state_does_not_use():
    """A zero bond weight round-trips through the outer-leg reweighting.

    ``_su_step`` multiplies six outer legs by ``sqrt(lambda)`` and divides the
    same six back out, and **the division is the dangerous half**: a true
    ``1/sqrt(lambda)`` is ``+inf`` at ``lambda == 0`` and the dead slice it
    multiplies is exactly ``0``, so the product is ``nan`` and the whole tensor
    goes with it.  That is #789's shape and the reason
    ``absorb_sqrt_singular_values`` carries a guard;
    :func:`~tenax.algorithms.ipeps_su._sqrt_and_inv_sqrt` is the same guard for
    the inverse.

    **Reachable, not hypothetical.**  A pair whose third virtual direction is
    scaled to exactly zero on all four legs of both sites gives a BP fixed point
    with an exact zero in every one of the four spectra -- measured
    ``h_AB = [1, 0.7766, 0]``, ``h_BA = [1, 0.4146, 0]``,
    ``v_AB = [1, 0.6708, 0]``, ``v_BA = [1, 0.9177, 0]``, from a solve that
    converged at residual 7.5e-07.  The meta-assertion below refuses to conclude
    anything if that stops being true, because on a well-conditioned spectrum
    this test cannot fail.

    The naive route is computed here as the control, and it reads ``inf`` on the
    same four vectors.  The output is then checked to be the *gated state* and
    not merely finite: a guard that only asked for ``isfinite`` would pass on an
    implementation that dropped the dead direction and returned zeros.
    """
    A, B = _PAIRS["dense"](D=D)
    dead = jnp.asarray([1.0] * (D - 1) + [0.0])
    for leg in ("u", "d", "l", "r"):
        A, B = scale_bond_axis(A, leg, dead), scale_bond_axis(B, leg, dead)

    _A_g, _B_g, w, info = gauge_fix(A, B)
    assert info.converged, (
        f"BP did not converge on the starved pair ({info.iterations} sweeps, "
        f"residual {info.residual:.3e}), so the spectrum below is not a fixed "
        f"point and the zero this test needs is not certified"
    )
    zeros = {f: int(np.sum(np.asarray(getattr(w, f)) == 0.0)) for f in w._fields}
    assert all(n > 0 for n in zeros.values()), (
        f"no bond spectrum has an exact zero ({zeros}); this pair does not "
        f"reach the division guard at all and the assertions below are vacuous"
    )
    # There was an ``assert all(not np.isfinite(1/sqrt(lam)))`` here.  It is
    # deleted rather than kept as a control: given the exact zero the assertion
    # above certifies, ``1/sqrt(0) == inf`` is IEEE arithmetic, so it could not
    # fail for any state of this module.  That is the thirteenth-assertion
    # pattern this branch keeps producing, and a review caught it.

    # The *backward* half of the same claim.  Nothing differentiates through
    # ``_su_step`` yet -- ``gauge_fix`` is a traced ``while_loop`` and reverse
    # mode does not go through one -- so the unit that carries the guard is
    # asserted directly rather than left as prose in its docstring.  The naive
    # route is the control and reads ``nan`` on the same input.
    lam = jnp.asarray([1.0, 0.5, 0.0])

    def _both(x):
        root, inv_root = _sqrt_and_inv_sqrt(x)
        return jnp.sum(root) + jnp.sum(inv_root)

    guarded = np.asarray(jax.grad(_both)(lam))
    naive_grad = np.asarray(
        jax.grad(lambda x: jnp.sum(jnp.sqrt(x) + 1 / jnp.sqrt(x)))(lam)
    )
    # A canary on JAX, **not** a guard on this module: the input is a literal
    # and nothing in ``ipeps_su`` can falsify it, so it cannot fail for any
    # change to the code under test.  It is kept because the assertion below
    # is worth nothing if ``jnp.sqrt``'s adjoint ever stops being ``inf`` at
    # zero -- at which point the double-``where`` would be dead weight and this
    # line is what would say so.
    assert not np.all(np.isfinite(naive_grad)), (
        f"jnp.sqrt's VJP at 0 is finite ({naive_grad}); JAX changed under this "
        f"test and the double-where below is no longer guarding anything"
    )
    assert np.all(np.isfinite(guarded)), (
        f"_sqrt_and_inv_sqrt's VJP at lambda=0 is {guarded} -- the double-where "
        f"is not masking the adjoint, which is #789's inf-VJP shape"
    )

    gate = _gate(A)
    stepped = _su_step(_SUState.from_pair(A, B), gate, max_D=_FULL_RANK, bond="h_AB")
    for name, t in (("A", stepped.A), ("B", stepped.B)):
        arr = np.asarray(t.todense())
        assert np.all(np.isfinite(arr)), (
            f"{name} came back with "
            f"{int(np.sum(~np.isfinite(arr)))} non-finite entries -- the "
            f"1/sqrt(lambda) on the outer legs hit a dead bond direction"
        )

    T_before = np.asarray(torus_2x2_sign_free(A, B, _ones_for(A)).todense())
    T_expected = _apply_gate_to_torus(
        T_before, np.asarray(gate.todense()), _TORUS_GATE_AXES["h_AB"]
    )
    control = _torus_rel(T_before, T_expected)
    assert control > 1e-3, (
        f"the gate moves the starved state by only {control:.3e}, so the "
        f"assertion below could not tell a working step from a no-op"
    )
    rel = _torus_rel(
        torus_2x2_sign_free(stepped.A, stepped.B, _ones_for(stepped.A)), T_expected
    )
    assert rel < 1e-11, (
        f"the untruncated step on a starved pair is {rel:.3e} from the gated "
        f"state (a no-op would score {control:.3e}) -- the dead direction was "
        f"not returned to where it started by the divide-back-out"
    )

    # --- and the case the exact zero does NOT cover -------------------------
    #
    # ``_sqrt_and_inv_sqrt`` masks at ``lam <= 1e-12 * max(lam)``, and *between*
    # zero and that floor the absorbed slice is not zero -- it carries
    # ``sqrt(lam)`` -- so the mask deletes a direction the state is still
    # (barely) using.  The round trip is exact only at exactly zero.  The worst
    # case is right at the threshold and it is bounded by the weight being
    # dropped; scanned out of band, ``eps`` 1e-3 / 1e-5 / 1e-6 / 1e-7 / 0 give
    # 9.774e-15, 1.482e-14, **1.426e-12**, 2.142e-14, 2.390e-15.  ``eps=1e-06``
    # is the peak (``lam_3/lam_1 = 5.9e-13``, just under the floor) and is the
    # cell run here, so the bound is pinned rather than only described in
    # ``_sqrt_and_inv_sqrt``'s docstring.
    A2, B2 = _PAIRS["dense"](D=D)
    nearly_dead = jnp.asarray([1.0] * (D - 1) + [1e-6])
    for leg in ("u", "d", "l", "r"):
        A2 = scale_bond_axis(A2, leg, nearly_dead)
        B2 = scale_bond_axis(B2, leg, nearly_dead)
    _a2, _b2, w2, info2 = gauge_fix(A2, B2)
    assert info2.converged, (
        f"BP did not converge on the nearly-starved pair ({info2.iterations} "
        f"sweeps, residual {info2.residual:.3e})"
    )
    lam2 = np.asarray(w2.h_AB, dtype=float)
    ratio = float(lam2[-1] / lam2[0])
    assert 0.0 < ratio < 1e-12, (
        f"the nearly-starved pair's lam_3/lam_1 is {ratio:.3e}, which is not "
        f"strictly between zero and _sqrt_and_inv_sqrt's 1e-12 relative floor "
        f"-- this cell is then either the exact-zero case already covered above "
        f"or an ordinary well-conditioned one, and it certifies neither bound"
    )
    gate2 = _gate(A2)
    stepped2 = _su_step(
        _SUState.from_pair(A2, B2), gate2, max_D=_FULL_RANK, bond="h_AB"
    )
    T_before2 = np.asarray(torus_2x2_sign_free(A2, B2, _ones_for(A2)).todense())
    T_expected2 = _apply_gate_to_torus(
        T_before2, np.asarray(gate2.todense()), _TORUS_GATE_AXES["h_AB"]
    )
    rel2 = _torus_rel(
        torus_2x2_sign_free(stepped2.A, stepped2.B, _ones_for(stepped2.A)),
        T_expected2,
    )
    assert rel2 < 1e-11, (
        f"a bond direction just under the relative floor (lam_3/lam_1 = "
        f"{ratio:.3e}) costs {rel2:.3e} through the reweighting round trip, "
        f"above the 1e-11 this file gates everything else at.  The mask deletes "
        f"it, which is intended; the claim being pinned here is that the loss "
        f"is bounded by the weight dropped."
    )


def test_su_step_warns_when_the_gauge_did_not_converge(su, monkeypatch):
    """A failed BP solve is surfaced, not swallowed.

    ``_su_step`` cannot fail on this and must not: the state is still correct,
    only the *basis the truncation is taken in* is worse than it should be.
    That is precisely the shape of defect this project keeps being bitten by --
    #870's residual fell monotonically to 9.8e-12 while certifying a state that
    had gone to zero -- so a silent degradation on every step of a 100-step
    run is not acceptable either.

    **The condition is real; the provenance this docstring used to give for it
    was not.**  It said BP hits ``max_iter=100`` at residual 1.399e-01 "with the
    gate's flows unaligned, on a symmetric ``v_BA`` step".  Measured, all six of
    ``test_su_step_output_can_still_be_gauged``'s cells converge under exactly
    that mutation (the readings are in its docstring), and within one step the
    internal ``gauge_fix`` runs before the gate is touched, so no gate change
    could have moved it.  The citation is withdrawn.

    What *is* measured is that the condition fires in real runs: one
    ``-m slow`` pass of this file and its mutation sibling emits **3339** of
    these warnings from ``ipeps_su.py``, 41 of them inside five cells that
    report green.

    The ``BPGaugeInfo`` below is therefore **fabricated**, which is the right
    construction for this test either way: the thing under test is the
    *reporting*, not BP, and the cheapest genuinely non-converging pair costs
    38 s.
    """
    A, B = su.pair("dense")
    real = ipeps_su_module.gauge_fix

    def _unconverged(*args, **kwargs):
        A2, B2, w2, _info = real(*args, **kwargs)
        # Fabricated, not measured -- see the docstring.  Only ``converged``
        # is load-bearing; the two numbers are there so the message has
        # something to format.
        return A2, B2, w2, BPGaugeInfo(100, 1.399e-01, False)

    monkeypatch.setattr(ipeps_su_module, "gauge_fix", _unconverged)
    with pytest.warns(RuntimeWarning, match=r"gauge_fix did not converge"):
        _su_step(_SUState.from_pair(A, B), _gate(A), max_D=D, bond="h_AB")


@pytest.mark.parametrize(
    "broken,match",
    [
        ("flow", "must carry the same flow"),
        ("charges", "Block matching pairs by charge value"),
    ],
)
def test_su_step_rejects_a_gate_that_does_not_match_the_site(broken, match, su):
    """A gate whose input legs do not match ``phys`` is refused, not repaired.

    ``_align_gate_to_ket`` flips flows, which fixes the one mismatch that *is*
    repairable.  The other two are not, and both mis-pair silently rather than
    raising: block matching goes by charge **value**, so a gate whose ``si``
    charges are a permutation of the site's contracts the wrong physical basis
    states together and returns a plausible number, and a gate whose ``si`` and
    ``sj`` disagree on flow cannot be aligned by any single flip.
    """
    A, _B = su.pair("symmetric")
    gate = _gate(A)
    labels = list(gate.labels())
    indices = list(gate.indices)
    if broken == "flow":
        indices[labels.index("sj")] = indices[labels.index("sj")].flip_flow()
    else:
        idx = indices[labels.index("si")]
        indices[labels.index("si")] = TensorIndex.from_charges(
            idx.symmetry,
            np.asarray(idx.charges)[::-1].copy(),
            idx.flow,
            label=idx.label,
        )
    broken_gate = DenseTensor(gate.todense(), tuple(indices))

    with pytest.raises(ValueError, match=match):
        _align_gate_to_ket(broken_gate, A)


# --- _su_evolve (#882 Task 11) -------------------------------------------
#
# Everything below runs on the **dense** arm and says so once here rather than
# four times.  A symmetric ``_su_step`` costs 38 s (see ``_CASES``), so the
# cheapest symmetric statement any of these could make -- a two-step evolve --
# is 76 s on its own, against a budget of 90 s for the whole of Task 11.  What
# a symmetric cell would add is block-sparse *algebra*, and none of it happens
# here: ``_su_evolve`` picks bond names off a tuple and calls ``_su_step``,
# whose own symmetric cells already cover the charge bookkeeping.
#
# An earlier version of this note claimed the loop owned one block-sparse
# hazard of its own -- a bond coming back below ``max_D`` because a charge
# block truncated away entirely.  **It does not, and the correction is
# measured.**  With ``base_charges=None``, ``linalg.svd`` drops sectors but
# keeps the total: on this file's own symmetric ``D=3`` ``theta`` the returned
# bond dimension is exactly ``max_D`` for ``max_D`` in 1, 2, 3, 4, 5, 8, 12, 20
# and 40, and falls short only past the structural capacity of 50
# (``54 -> 50``, ``60 -> 50``).  ``max_D == D == 3`` is nowhere near that.  So
# there is no symmetric-only behaviour here that the dense arm misses, and
# ``test_su_evolve_names_the_step_that_broke_bond_uniformity`` is testing a
# broken ``_su_step``, not a truncation outcome.

#: The cycle order asserted below.  Written out here rather than imported from
#: ``ipeps_su._SU_CYCLE`` so the assertion is a second, independent statement of
#: it instead of a tautology -- the same discipline ``_TORUS_GATE_AXES`` uses.
#: Its source is ``_simple_update_checkerboard_sweep``'s own phase table
#: (``ipeps_simple_update.py:255-260``): ``phase 0 -> h_AB``, ``1 -> v_AB``,
#: ``2 -> h_BA``, ``3 -> v_BA``.
_EXPECTED_CYCLE = ("h_AB", "v_AB", "h_BA", "v_BA")


def _torus(state):
    """The 2x2 torus of an absorbed-form pair -- the state reading used below."""
    return torus_2x2_sign_free(state.A, state.B, _ones_for(state.A))


def _shrink_leg(t, leg, keep):
    """``t`` with ``leg`` truncated to its first ``keep`` entries.

    Used to build a pair that is uniform on every bond's ``IN`` end and *not* on
    one ``OUT`` end -- the case a four-leg uniformity read cannot see.  Same
    index-rebuilding idiom as
    ``test_su_step_rejects_a_gate_that_does_not_match_the_site``.
    """
    if not isinstance(t, DenseTensor):
        # It would otherwise densify and hand back a ``DenseTensor`` still
        # carrying the input's non-trivial charges on its indices: a silent type
        # change, and one whose ``charges[:keep]`` is a *positional* truncation
        # of a block-sparse leg rather than a choice of which sector loses a
        # basis vector.  One caller today, dense; refuse the rest.
        raise TypeError(
            f"_shrink_leg is dense-only; got {type(t).__name__}.  Truncating a "
            f"SymmetricTensor's leg means choosing which sector loses a basis "
            f"vector, which this helper does not do."
        )
    axis = t.labels().index(leg)
    idx = t.indices[axis]
    indices = list(t.indices)
    indices[axis] = TensorIndex.from_charges(
        idx.symmetry, np.asarray(idx.charges)[:keep].copy(), idx.flow, label=idx.label
    )
    arr = np.take(np.asarray(t.todense()), np.arange(keep), axis=axis)
    return DenseTensor(jnp.asarray(arr), tuple(indices))


def _bond_dims(state):
    """Each bond's dimension, read through ``_BOND_ENDS``' ``IN`` end.

    Deliberately **not** ``ipeps_su._bond_dims``, even though the implementation
    has one -- same discipline as ``_EXPECTED_CYCLE`` above and
    ``_TORUS_GATE_AXES`` further up.  The only assertion that uses this
    (``test_su_evolve_rejects_a_max_D_the_pair_does_not_have``) is a statement
    about what ``_su_step`` leaves behind, i.e. about the premise the
    implementation's guard rests on; reading it through that guard's own helper
    would make it agree by construction.  The two are also shaped differently on
    purpose -- this returns four entries keyed by bond, the implementation's
    returns eight keyed by ``bond@site.leg`` -- so they cannot be swapped for
    each other by accident.
    """
    pair = {"A": state.A, "B": state.B}
    return {
        bond: pair[site].indices[pair[site].labels().index(leg)].dim
        for bond, ((site, leg), _other) in _BOND_ENDS.items()
    }


def test_su_evolve_has_no_steps_mod_4_dependence(su):
    """#851: stopping the sweep at different phases must not change the state.

    The defect this replaces stored one horizontal and one vertical spectrum for
    four inequivalent bonds, so phases 0 and 2 wrote the same slot and
    ``steps % 4`` decided which bond's gauge was stamped on the lattice.  With
    nothing stored there is nothing to stamp, and under a ``dt=0`` gate at
    ``max_D == D`` the statement is *exact* rather than approximate: each step
    is a gauge plus a lossless re-split, so the state is unmoved.  Measured,
    dense ``D=3``: ``steps`` 5, 6, 7, 8 land 1.8e-14, 2.6e-14, 1.9e-14, 2.7e-14
    from ``steps=4``, and ``steps=4`` lands 1.9e-14 from the input.

    **This test is necessary and it is not sufficient, and the second half is
    not a caveat but the reason the next two tests exist.**  An ``_su_evolve``
    that ignores ``steps`` and returns its input passes it *exactly*, at
    ``d = 0`` -- better than the correct implementation does.  And what it
    cannot see is wider than that: measured, ``for i in range(steps % 4)`` --
    the closest analogue of #851 in this architecture, and a loop that really
    does drop steps -- **also passes**, because under a ``dt=0`` gate every
    ``steps`` gives the same state.  ``test_su_evolve_actually_evolves``
    catches that one at 0.000e+00.  The last
    assertion below says so out loud by pinning the ``dt=0`` invariance it
    depends on; the guards that can tell a working loop from ``return state``
    are ``test_su_evolve_actually_evolves`` and
    ``test_su_evolve_visits_four_distinct_bonds_per_cycle``, both of which use a
    real gate.

    What it *is* sufficient for is the defect it is named after, and all three
    assertions have been watched failing (guards at 1e-12 / 1e-11 / 1e-11):

    ===================================================  ==========  =========
    mutation                                             reading     fires
    ===================================================  ==========  =========
    #851: stamp the last phase's spectrum on its twin    1.000e-01   ``d``
    #667: re-absorb ``gauge_fix``'s weights every step   1.487e-01   ``d``
    #667: re-absorb them once, after the loop            9.110e-07   ``d``
    a stray diagonal on ``phys`` after the loop          3.725e-01   ``held``
    the gate built at ``dt=0.05`` instead of ``0``       3.077e-02   ``off``
    ===================================================  ==========  =========

    The third and fourth rows are the interesting ones.  A *terminal* distortion
    is the same for every ``steps >= 1``, so one might expect ``d`` to be blind
    to it -- but each run leaves the bond in its own basis, and a distortion
    that does not commute with that basis shows up in ``d`` anyway (row 3, and
    a fixed one-sided bond weight at 1.745e-01).  Only a distortion on ``phys``,
    which *does* commute with the bond gauge, gets past ``d``; that is the case
    ``held`` exists for, and it is why ``held`` is an assertion rather than a
    comment.
    """
    A, B = su.pair("dense")
    state = _SUState.from_pair(A, B)
    identity = _gate(A, dt=0.0)

    # The premise: dt=0 really is the identity, so "the state must not move" is
    # a statement about the loop and not about the gate.  Measured **2.220e-16**
    # -- max|G - I|, the expression below.  (An earlier comment said 3.1e-16,
    # which is the Frobenius norm of the same matrix: a scratch probe used
    # ``np.linalg.norm`` where the assertion uses ``np.max(np.abs(...))``.  Same
    # class of mistake as quoting a coverage-on runtime against a --no-cov
    # budget, and worth the correction for the same reason.)
    off = float(
        np.max(np.abs(np.asarray(identity.todense()).reshape(4, 4) - np.eye(4)))
    )
    # Nothing in ``ipeps_su`` can make this fire either: ``identity`` is built
    # by ``ipeps_simple_update._make_trotter_gate_tensor`` at ``dt=0`` from a
    # zero Hamiltonian, so this is IEEE arithmetic about *that* function.  It is
    # the premise of the loop assertion below rather than a reading of the loop.
    assert off < 1e-12, f"the dt=0 gate is not the identity (max |G - I| = {off:.3e})"

    ref = _torus(_su_evolve(state, identity, max_D=D, steps=4))
    for n in (5, 6, 7, 8):
        got = _torus(_su_evolve(state, identity, max_D=D, steps=n))
        d = _torus_rel(got, ref)
        assert d < 1e-11, (
            f"steps={n} differs from steps=4 by {d:.3e} under a dt=0 gate -- "
            f"where the sweep stops is deciding the answer (#851)"
        )

    held = _torus_rel(_torus(state), ref)
    assert held < 1e-11, (
        f"a dt=0 evolve moved the state by {held:.3e}; each step should be a "
        f"gauge plus a lossless re-split at max_D == D.  (This assertion is "
        f"also what makes the blindness above explicit: it is exactly why a "
        f"do-nothing _su_evolve passes this test.)"
    )


def test_su_evolve_actually_evolves(su):
    """The no-op control: a real gate moves the state, and ``steps`` decides how far.

    The test above cannot make this statement -- under ``dt=0`` a do-nothing
    loop is *indistinguishable from a correct one*, so something has to fail on
    ``return state`` or nothing in this file does.  Every reading here is taken
    from a separate full run out of the same input, so it also pins that
    ``_su_evolve`` is a pure function of ``steps`` rather than of call order.

    Measured, dense ``D=3``, ``dt=0.05``, against guards at 1e-3:

    ========================  =========  ================
    reading                   correct    ``return state``
    ========================  =========  ================
    ``steps=4`` vs input      4.99e-02   0
    consecutive ``n``/``n-1`` 1.7-2.0e-02  0
    ========================  =========  ================

    ``steps=0`` is checked by identity rather than by distance: it must be the
    input object, which no distance reading could distinguish from an
    unnecessary gauge-and-resplit round trip.
    """
    A, B = su.pair("dense")
    gate = _gate(A)
    state = _SUState.from_pair(A, B)

    assert _su_evolve(state, gate, max_D=D, steps=0) is state, (
        "steps=0 must return the input pair itself, not a re-gauged copy of it"
    )

    T = {n: _torus(_su_evolve(state, gate, max_D=D, steps=n)) for n in range(6)}

    # Not implied by the consecutive-step assertion below, and it took a
    # mutation to establish that rather than an argument: ``range(steps % 4)``
    # -- a loop that drops whole cycles and runs only the remainder -- makes
    # ``steps=4`` run zero steps, so this fires at 0.000e+00 while every
    # consecutive pair (0,1,2,3,0,1 steps) still differs and the loop below
    # stays green.  A cycle that returns to where it started is a real shape;
    # this is the only reading that sees it.
    moved = _torus_rel(T[4], T[0])
    assert moved > 1e-3, (
        f"four steps of a dt=0.05 gate moved the state by only {moved:.3e} -- "
        f"_su_evolve is not evolving.  A loop that returns its input scores 0 "
        f"here and still passes the steps % 4 test above, exactly."
    )

    for n in range(1, 6):
        d = _torus_rel(T[n], T[n - 1])
        assert d > 1e-3, (
            f"steps={n} and steps={n - 1} give the same state ({d:.3e}), so "
            f"the loop is not running the step count it was asked for"
        )


def test_su_evolve_visits_four_distinct_bonds_per_cycle(su, monkeypatch):
    """Four consecutive steps update four *different* bonds, in the shipped order.

    A checkerboard unit cell has four inequivalent nearest-neighbour bonds, and
    an implementation that gates ``h_AB`` four times passes both tests above:
    it moves the state, ``steps`` still decides how far, and under ``dt=0`` it
    is still the identity.  What it produces is the spuriously dimerised state
    of #667 -- ``A`` only ever picks up weight on its ``r`` leg and half the
    lattice bonds get none at all.

    Two independent readings, because neither alone is enough:

    * **Which bonds** -- a recorder wrapped round ``_su_step`` (which really
      runs, so this costs nothing but the eight steps).  It is the only reading
      that can state "four *distinct*" and "period 4" directly, and it pins the
      order as well.  The expected order is written out in ``_EXPECTED_CYCLE``
      from ``_simple_update_checkerboard_sweep``'s phase table rather than
      imported from the implementation's own constant.
    * **What that buys** -- the recorder is a claim about calls, not about
      physics, so the second half evolves one bond four times by hand and
      requires the result to differ.  Measured against a four-bond cycle at
      ``dt=0.05``, dense ``D=3``: 6.44e-02 (4x ``h_AB``), 4.35e-02 (``v_AB``),
      5.34e-02 (``h_BA``), 5.97e-02 (``v_BA``), all against a guard at 1e-3.
      (Re-measured for #882's final review: the previous 5.7e-02 / 4.5e-02 /
      5.3e-02 / 6.2e-02 did not reproduce, ``h_AB`` least of all.)

    The **order** is deliberately not asserted through the state, and that is
    measured too: a cycle re-ordered to ``(h_AB, h_BA, v_AB, v_BA)`` lands
    4.93e-04 away (re-measured; 4.4e-04 was the earlier figure), a hundredth
    of the single-bond contrast and merely a different Trotter ordering.
    Coverage is physics; order is a convention, so the recorder pins it and the
    state reading does not pretend to.

    **The second half is not belt-and-braces, and it took a mutation to prove
    it.**  Every loop-level defect trips the recorder first, so the obvious
    reading is that the state contrast is decoration.  It is not: an
    ``_su_step`` that ignores the ``bond`` it is handed and always evolves
    ``h_AB`` leaves the recorder showing a perfect ``h_AB, v_AB, h_BA, v_BA``
    and is caught *only* here, at 0.000e+00 against a guard of 1e-3.  That is
    the lying-recorder case in full, and it is the one an implementation that
    stops routing through ``_su_step`` would also land in.

    ``_asymmetric_hamiltonian`` matters here for its documented reason: with a
    plain Heisenberg gate the torus cannot tell an ``h_AB`` update from an
    ``h_BA`` one at all.
    """
    A, B = su.pair("dense")
    gate = _gate(A)
    state = _SUState.from_pair(A, B)

    seen = []
    real_step = ipeps_su_module._su_step

    def _record(st, g, max_D, bond):
        seen.append(bond)
        return real_step(st, g, max_D, bond)

    monkeypatch.setattr(ipeps_su_module, "_su_step", _record)
    _su_evolve(state, gate, max_D=D, steps=8)
    monkeypatch.undo()

    # Weaker than the order assertion below and **not** redundant with it, for a
    # reason that is about the two references rather than about the sets: this
    # compares against ``_BOND_ENDS``, the module's own inventory of bonds,
    # while the next line compares against a literal written out in this file.
    # So this is the assertion that pins the ``_BOND_ENDS`` <-> ``_SU_CYCLE``
    # coupling.  Watched: add a fifth entry to ``_BOND_ENDS`` and leave the
    # cycle alone -- the recorded order is still a perfect
    # ``h_AB, v_AB, h_BA, v_BA`` twice over, the order assertion passes, and
    # only this one reports "the four bonds of a checkerboard cell are
    # ['h_AB', 'h_BA', 'h_XX', 'v_AB', 'v_BA']".  It is also the *durable* half:
    # any permutation covering all four is a valid Trotter split, so the order
    # below may legitimately be re-pointed one day, and coverage may not.
    assert set(seen[:4]) == set(_BOND_ENDS), (
        f"the first cycle updated {seen[:4]} -- the four bonds of a checkerboard "
        f"cell are {sorted(_BOND_ENDS)}, and evolving a subset leaves the state "
        f"dimerised (#667)"
    )
    assert tuple(seen) == _EXPECTED_CYCLE * 2, (
        f"the cycle ran {tuple(seen)}, expected {_EXPECTED_CYCLE * 2} -- the "
        f"phase order _simple_update_checkerboard_sweep ships, which Phase 4 "
        f"re-points ipeps() at"
    )

    cycled = _torus(_su_evolve(state, gate, max_D=D, steps=4))
    for bond in _BOND_ENDS:
        stuck = state
        for _ in range(4):
            stuck = _su_step(stuck, gate, max_D=D, bond=bond)
        d = _torus_rel(_torus(stuck), cycled)
        assert d > 1e-3, (
            f"four steps on {bond} alone give a state {d:.3e} from a full "
            f"four-bond cycle -- this reading cannot see a loop that never "
            f"leaves one bond"
        )


@pytest.mark.parametrize("max_D", [D - 1, D + 1])
def test_su_evolve_rejects_a_max_D_the_pair_does_not_have(max_D, su):
    """``max_D`` is the pair's dimension, not a truncation knob -- and it is checked.

    Before #887, ``gauge_fix`` read one ``D_h`` off ``A.r`` and one ``D_v`` off
    ``A.d`` and handed each to *both* bonds of that orientation, so a pair
    whose two horizontal bonds differed could not be gauged at all, and a step
    at any ``max_D != D`` made the *next* step die four frames down inside
    ``absorb_weights`` with ``cannot reshape array of shape (4,) into shape
    [1, 1, 3, 1, 1]``.  #887 changed ``ipeps_gauge._identity_weights`` to size
    each of the four bonds from its own leg, and that mechanism is gone:
    measured post-#887 (dense, ``D=3``, calling :func:`_su_step` directly so
    this guard is bypassed), a full four-bond cycle now completes without
    raising for both ``max_D=D-1`` (shrink) and ``max_D=D+1`` (grow).  The
    guard below is therefore **policy**, kept because nothing has measured that
    a state produced by truncating across a mixed-dimension pair agrees with a
    fresh run built at the new ``D`` -- see
    :func:`ipeps_su._require_uniform_bonds` for the measurements and #897 for
    the open question.  "Completes without raising" is not "correct."

    The last half measures what one step actually leaves behind, rather than
    assuming it: an unguarded ``_su_step`` still resizes only the bond it
    touches, so the pair it returns is genuinely non-uniform -- the premise
    the guard below refuses -- whichever direction ``max_D`` moves.

    **The message is matched on ``the input pair``, not on the shared prefix,
    and that is the whole difference between this test having teeth and not.**
    ``_su_evolve`` checks the same invariant after every step, and that check
    fires on the pair step 0 leaves behind -- so a build with the *input* check
    deleted still raises a ``ValueError`` whose text starts identically, one
    wasted step later.  Watched: matching only ``needs all four bonds at
    max_D`` passed with the input check removed.  The ``steps=0`` case below
    closes the same hole from the other side, since no post-step check runs at
    all there.
    """
    A, B = su.pair("dense")
    gate = _gate(A)
    state = _SUState.from_pair(A, B)

    with pytest.raises(ValueError, match="the input pair carries"):
        _su_evolve(state, gate, max_D=max_D, steps=4)

    with pytest.raises(ValueError, match="the input pair carries"):
        _su_evolve(state, gate, max_D=max_D, steps=0)

    stepped = _su_step(state, gate, max_D=max_D, bond="h_AB")
    dims = _bond_dims(stepped)
    assert dims == {"h_AB": max_D, "h_BA": D, "v_AB": D, "v_BA": D}, (
        f"one step at max_D={max_D} left the bonds at {dims}; the guard this "
        f"feeds is policy pending a correctness measurement (#897), not a "
        f"response to that pair being ungaugeable, so if the step no longer "
        f"produces this the guard's premise needs re-deriving, not deleting"
    )
    # Watched failing: an ``_su_step`` that drops ``max_singular_values``
    # reports ``{'h_AB': 54, 'h_BA': 3, 'v_AB': 3, 'v_BA': 3}`` here -- still
    # non-uniform, so the ValueError above still raises and only this line
    # notices that the premise moved.


@pytest.mark.parametrize(
    "steps,exc,match",
    [
        (-1, ValueError, "steps must be non-negative"),
        (4.0, TypeError, "steps must be an int"),
        (True, TypeError, "steps must be an int"),
    ],
)
def test_su_evolve_rejects_a_step_count_it_cannot_run(steps, exc, match, su):
    """Each of the three is silent or misleading if it is not caught here.

    ``-1`` is the dangerous one: ``range(-1)`` is empty, so a negative count
    would run zero steps and return a plausible state rather than complain.
    ``4.0`` raises out of ``range`` anyway, but three frames down and without
    naming the argument.  ``True`` is an ``int`` to Python and would quietly
    mean "one step"; ``steps`` counts bonds, and nobody writing ``steps=True``
    meant one bond.
    """
    A, B = su.pair("dense")
    state = _SUState.from_pair(A, B)
    with pytest.raises(exc, match=match):
        _su_evolve(state, _gate(A), max_D=D, steps=steps)


@pytest.mark.parametrize("steps", [np.int64(4), np.int32(4)])
def test_su_evolve_accepts_a_numpy_integer_step_count(steps, su):
    """``np.int64`` is a step count; the check is on ``numbers.Integral``.

    Not hypothetical plumbing.  ``iPEPSConfig.num_imaginary_steps``
    (``ipeps_config.py:524``) is an uncoerced dataclass field handed straight
    through at ``ipeps.py:451``, so anything that reaches it from numpy -- a
    schedule built with ``np.arange``, a value read back out of a checkpoint --
    arrives as ``np.int64``.  An ``isinstance(steps, int)`` check rejects that,
    and would have done so the first time Phase 4 wired ``ipeps()`` to
    ``_su_evolve``.  Watched: with the check narrowed back to ``int`` both
    parameters raise ``TypeError: steps must be an int; got int64``.

    The ``bool`` exclusion above is what stops this widening letting
    ``steps=True`` through -- ``bool`` is ``Integral`` too, so that ``and not
    isinstance(steps, bool)`` is now the *only* thing rejecting it, which is
    why the two tests sit next to each other.
    """
    A, B = su.pair("dense")
    state = _SUState.from_pair(A, B)
    out = _su_evolve(state, _gate(A, dt=0.0), max_D=D, steps=steps)
    assert out.max_D == D


def test_su_evolve_names_the_step_that_broke_bond_uniformity(su, monkeypatch):
    """A bond that shrinks mid-run is reported here, not as a reshape error later.

    **This pins an internal invariant, not a guard against an expected event,
    and the distinction is a correction.**  An earlier version of this docstring
    called the post-step check "real rather than defensive" on the grounds that
    a symmetric truncation which empties a charge block makes ``linalg.svd``
    return a bond below ``max_D``.  Measured, it does not: with
    ``base_charges=None`` the fallback allocation (``linalg.py:816-842``)
    spreads the budget over the sectors and then drains the excess *to zero*, so
    sectors are dropped and the total stays at exactly ``max_singular_values``.
    On this file's own symmetric ``D=3`` ``theta``, ``max_D`` in 1, 2, 3, 4, 5,
    8, 12, 20 and 40 all come back exact; only 54 and 60 under-deliver, at the
    pair's structural capacity of 50.  ``max_D == D`` never goes near it.

    So with a well-formed input and a correct ``_su_step`` the check cannot
    fire, and the only thing that reaches it is a broken step -- which is what
    the monkeypatch below substitutes.  It is kept because eight metadata reads
    are free, because the alternative report is a reshape ``TypeError`` several
    frames below the caller naming neither the step nor the bond, and because an
    invariant worth stating is worth stating where it is cheap.  What it is not
    is a prediction about the block-sparse path.

    Monkeypatching rather than engineering a real shrink is the same trade
    ``test_su_step_warns_when_the_gauge_did_not_converge`` makes: the thing under
    test is the *reporting*, and one symmetric step costs 38 s.

    **The ``out_end_only`` case is why the check reads all eight virtual legs
    rather than the four ``IN`` ends.** For a working ``_su_step`` one end per
    bond is enough -- both come out of one SVD -- but the only thing this check
    exists to catch is a step that did *not* work, and a step that misbehaved on
    an ``OUT`` end alone is exactly what a four-leg read waves through. Watched:
    with ``_bond_dims`` narrowed to ``ends[:1]``, this parameter stops raising
    and every other test in the file still passes. That is the whole argument
    for the extra four reads, and without this case it would have been an
    unfalsifiable one.
    """
    A, B = su.pair("dense")
    state = _SUState.from_pair(A, B)

    # Every IN end (A.r, B.r, A.d, B.d) stays at D; only A.l -- h_BA's OUT end
    # -- shrinks, so a four-leg read sees a perfectly uniform pair.
    out_end_only = _SUState.from_pair(_shrink_leg(A, "l", D - 1), B)
    both_ends = _SUState.from_pair(*_PAIRS["dense"](D=D - 1))

    # Both patterns pin the step index as well as the symptom -- "step 1
    # (v_AB)" is the step that returned the broken pair, so a check that fired
    # late (say, only after the loop) would report step 3 and not match.
    for broken, match in (
        (both_ends, r"the pair after step 1 \(v_AB\) carries"),
        (out_end_only, r"the pair after step 1 \(v_AB\).*off on \{'h_BA@A\.l': 2\}"),
    ):
        real_step = ipeps_su_module._su_step
        calls = {"n": 0}

        def _shrinks_on_the_second(st, g, max_D, bond, _broken=broken, _real=real_step):
            calls["n"] += 1
            if calls["n"] == 2:
                return _broken
            return _real(st, g, max_D, bond)

        monkeypatch.setattr(ipeps_su_module, "_su_step", _shrinks_on_the_second)
        with pytest.raises(ValueError, match=match):
            _su_evolve(state, _gate(A), max_D=D, steps=4)
        monkeypatch.undo()


# --- #882 Task 12: the acceptance sweep, seeds x D ------------------------
#
# Everything above this line asks whether ``_su_step`` and ``_su_evolve`` do
# what they say.  This section asks the only question the rewrite exists to
# answer: does the state they produce have the right *energy*?  Nothing above
# can see that -- a step can gate, split and truncate exactly as specified and
# still converge to the wrong fixed point, which is what #667, #851, #865 and
# #869 each were.
#
# **This section was red on 0 of 9 and is now green on 5 of 9, and the four
# that stay red are the finding rather than a defect in the tests.**  Task 12
# measured ``_su_evolve`` on the product state at every cell; Task 10's
# reopening put the truncation into the state's own basis
# (``ipeps_su._su_step`` stages 2 and 5) and re-ran the same grid.  Nothing else
# changed and no threshold moved.  Measured, ``JAX_PLATFORMS=cpu``, dt=0.05,
# energies at 400/800/1200/2000 steps:
#
#   D=2, all three seeds   -0.658880 at every step count       PASS  (was
#                                                              -0.5406 / 0.0 /
#                                                              -0.500000)
#   D=3 seeds 1, 2         -0.662838 -> -0.662839, pinned      PASS
#   D=3 seed 0             -0.651785 -> -0.607822, decaying    FAIL, and BP
#                          fails on 1782 of 2000 steps
#   D=4, all three seeds   -0.500000 from 800 steps on         FAIL, with BP
#                          converging on every step
#
# **A gap this section does not close, recorded rather than left to be
# rediscovered.**  ``_su_step`` warns when its internal ``gauge_fix`` fails, and
# one ``-m slow`` pass of this file and its mutation sibling emits **3339** of
# those warnings -- and they are not confined to the red cells.
#
# **Count the emissions, not the summary lines.**  pytest's warnings summary
# collapses byte-identical messages, and the twin cells here emit byte-identical
# text, so that summary shows only **1377** entries and attributes each collapsed
# group to a single node id.  Reading it as a per-cell census is what produced an
# earlier version of this note claiming a 970/388/16/3 split over four cells --
# every twin cell had silently vanished into its sibling.  Measured properly with
# ``-rw`` (per node id, three batches), the green-cell emissions are::
#
#      16  test_d2_reaches_the_heisenberg_energy_not_the_product_state[1]
#      16  test_the_energy_does_not_drift_away_with_more_steps[2-1]
#       3  test_the_energy_does_not_drift_away_with_more_steps[3-1]
#       3  test_su_evolve_reaches_the_simple_update_reference_energy[3-1]
#       3  test_d3_actually_uses_its_third_bond_direction[1]
#
# so **41** of them come from **five** cells that report green.  The remainder
# come from the red cells.  No test here asserts on, counts
# or ``filterwarnings("error")``s the warning, so the degradation mode
# ``ipeps_su.py``'s *Warns* documents -- "it quietly costs truncation quality on
# **every** step" -- is unwatched inside cells reporting green.  Closing it is a
# design decision (does a warning fail an otherwise-green acceptance cell?) and
# is deliberately not taken here; the counts are recorded so that whoever takes
# it starts from a measurement.
#
# **The two residues are out of scope and must stay visible.**  Neither is
# explained, both are reproduced exactly from ``task-12-report.md``'s prototype
# numbers, and a change that turned either green without an explanation would
# be far likelier to be a new bug than a fix -- this project has shipped that
# mistake before (#869: one step count reporting either verdict from the same
# code).  Do not tune ``steps``, ``dt`` or ``chi`` to close them.
#
# The mechanism the fix addressed is localised by
# ``test_su_step_truncates_in_the_state_s_own_basis`` below, which is the
# cheapest cell in the section: as shipped, ``_su_step`` truncated the SVD of
# the **absorbed** two-site tensor, whose environment is not the identity,
# rather than the **Vidal** one, whose is.  Its reference is not re-derived from
# ``_su_step`` -- and neither is
# ``test_the_vidal_metric_matches_a_spectrum_derived_outside_tenax``'s, which is
# more independent still: that one is not computed by tenax at all.


#: Seeds for every cell of the sweep.  Three, not one, and the reason is
#: historical rather than statistical: every earlier attempt at these numbers
#: measured one seed, and a single-seed D=3 run passes on a broken
#: implementation while a single-seed D=4 run cannot tell "always broken" from
#: "unlucky".  Measured here, the axis paid for itself twice over -- before the
#: truncation-basis fix ``_su_evolve`` failed in three *different* ways across
#: the three D=2 seeds (a drifting -0.54, a state that was exactly zero, and the
#: product state), and after it the D=3 reference is reached on seeds 1 and 2 but
#: not on seed 0.  One seed reports either verdict, in both directions.
_SEEDS = (0, 1, 2)

#: The sublattice-rotated Heisenberg Hamiltonian, which is what makes an energy
#: comparable at all here.  Under the rotation the Neel ground state becomes
#: uniform, so ``A`` and ``B`` converge to the same physical tensor and
#: ``|E(A) - E(B)|`` becomes a convergence diagnostic rather than a property of
#: the sublattice.  ``test_su_667_product_state.py`` uses the same gate for the
#: same reason.
_H_HEISENBERG_ROT = sublattice_rotate_gate(heisenberg_gate())

#: What "broken" looks like.  Not a round number pulled from the air: it is the
#: exact fixed point #667's ``lambda**1.5`` bond produced, and three of the nine
#: cells below sit on it to fifteen digits.
_PRODUCT_STATE_ENERGY = -0.5

#: Post-#667 simple-update references for this model, per bond dimension.
#: ``test_the_energy_reference_reproduces_the_known_su_number`` reproduces the
#: D=2 entry from the shipped engine, which is what keeps these expectations
#: reproduced rather than gospel assumed.  D=3 and D=4 were reproduced the same
#: way out of band (-0.662859 and -0.667071); see ``task-12-report.md`` section
#: 5 for the chi scan behind all three.
_SU_REFERENCE = {2: -0.6593, 3: -0.6632, 4: -0.6674}

#: Environment dimension per ``D``.  ``chi >> D**2`` is the requirement (C-3):
#: at D=2 and D=3 ``chi=16`` is 4x and 1.8x ``D**2``, and at D=4 ``chi=24`` is
#: 1.5x it.  The margin at D=3 and D=4 is thin, so the ratio is not the
#: evidence -- the chi-to-chi movement was measured directly on shipped-engine
#: states and is 1.4e-11 (D=2, 8->16), 1.1e-07 (D=3, 16->24) and 2.0e-06 (D=4,
#: 16->24), with nothing further from chi=32.  ``task-12-report.md`` section 5
#: carries the scan; the D=2 half of it is asserted by
#: ``test_the_energy_reference_reproduces_the_known_su_number``.
_CHI = {2: 16, 3: 16, 4: 24}

#: Imaginary time step.  The same 0.05 the shipped path uses, so a shortfall
#: here cannot be blamed on a Trotter step nothing else in this tree uses.
_SU_DT = 0.05


def _random_state(D, seed):
    """A random dense checkerboard pair at ``D``, as an ``_SUState``.

    The same builder ``ipeps()`` uses for its own default initialisation
    (``jax.random.normal`` on ``(D, D, D, D, 2)``, normalised), so the sweep
    starts where the shipped path starts and a difference in outcome is a
    difference in the *engine*.
    """
    return _SUState.from_pair(*_PAIRS["dense"](D=D, seed=seed))


def _su_heisenberg_gate(state, dt=_SU_DT):
    """``exp(-dt H_rot)`` on the pair's own physical leg."""
    return _make_trotter_gate_tensor(
        jnp.asarray(_H_HEISENBERG_ROT.todense()), dt, site_tensor=state.A
    )


def _energy_of(state, chi, recipe="2x2"):
    """Energy per site of the checkerboard pair, from a **checked** CTM.

    Three things about this reference are deliberate, and each of them is a
    trap this project has already fallen into once.

    * **The 2-site reading, not the 1-site one.**  ``test_su_667_product_state``
      measures ``E_1x1(A)`` -- the uniform lattice built from ``A`` alone -- and
      that is sound *there*, because its states come out of ``_to_physical_pair``
      with all four bonds in one consistent gauge.  ``_su_evolve``'s output is
      not in that gauge: only the last-updated bond carries the SVD's basis and
      the other three carry whatever the step's opening ``gauge_fix`` left, so
      ``A`` alone is not a valid uniform ansatz for the state.  Measured on a
      state whose 2-site energy is pinned at -0.662839 across 400, 800, 1200 and
      2000 steps, the 1x1 reading of the same state moves -0.663174, -0.360570,
      -0.360570, -0.072255.  A 1x1 reference would have reported a wildly
      unconverged energy for a state that had converged.
    * **``recipe="2x2"``, and ``rank(C1)`` asserted rather than assumed.**  The
      ``1x1`` recipe collapses the environment to rank-1 corners and returns a
      plausible, wrong number (#723/#726/#747); that defect invalidated a whole
      benchmark on this project.  Asserting the rank is two singular-value
      decompositions of a ``chi x chi`` matrix and it is the only thing that
      distinguishes a converged environment from a collapsed one.

      ``recipe`` is a parameter **only** so that
      ``test_the_energy_reference_reproduces_the_known_su_number`` can watch
      that assertion fire, by driving this function at ``"1x1"`` inside a
      ``pytest.raises``.  Until #882's final review, that control called
      ``ctm_tensor_2site`` directly and checked the corner rank itself, so the
      assertion below was never watched failing anywhere in either test file --
      an unwatched refusal is the same defect class as an unfalsifiable
      assertion.  Every measurement caller uses the default.
    * **A zero environment is reported as a zero *state*.**  If the pair itself
      has gone to zero -- which one cell of this sweep does -- ``C1`` is
      identically zero and there is no energy to report.  Saying so is the
      point: ``_normalize_tensor``-style norm checks pass on a corpse
      (``||A||`` reads 1.0 right up to the step where it is exactly 0), so the
      collapse has to surface as an energy failure or not at all.

    Args:
        state:  The pair to measure.
        chi:    Environment bond dimension; must be well above ``D**2``.
        recipe: CTM recipe.  Leave it at ``"2x2"``; see above.

    Returns:
        Energy per site as a ``float``.

    Raises:
        AssertionError: if either ``C1`` is zero or rank 1.
    """
    env_A, env_B = ctm_tensor_2site(
        state.A, state.B, chi=chi, max_iter=200, conv_tol=1e-10, recipe=recipe
    )
    for name, env in (("A", env_A), ("B", env_B)):
        sv = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
        assert sv[0] > 0, (
            f"the {name} corner C1 is identically zero, so there is no energy "
            f"to report -- the pair itself has collapsed to zero.  Note that a "
            f"norm check would not have said so: _su_step's output is rescaled "
            f"by gauge_fix on the way in to the next step, so ||A|| reads a "
            f"healthy 1.0 right up to the step where it is exactly 0 (#878)."
        )
        rank = int(np.sum(sv > sv[0] * 1e-10))
        assert rank > 1, (
            f"the {name} corner C1 has rank {rank} at chi={chi} on "
            f"recipe={recipe!r} -- a rank-1 corner is the collapsed-environment "
            f"signature (#747), and it returns a plausible wrong energy rather "
            f"than failing.  On the default recipe='2x2' that is a finding "
            f"about the state; on '1x1' it is the recipe and is what "
            f"test_the_energy_reference_reproduces_the_known_su_number's "
            f"control provokes on purpose."
        )
    return float(
        compute_energy_ctm_tensor_2site(
            state.A, state.B, env_A, env_B, _H_HEISENBERG_ROT, 2
        )
    )


def _shipped_su_run(D, seed, steps):
    """The same run on the **shipped** stored-lambda engine.

    Read-only use of ``ipeps_simple_update``: this is the reference engine the
    rewrite is measured against, and the two tests that use it are the ones
    that validate the measuring apparatus itself (against a CTM number this
    project already has) and the rewrite's premise (C-1).

    Returns:
        ``(state, lambdas)`` -- the pair in physical (absorbed) form, and the
        four spectra the engine *stored*.  Both halves are needed: C-1 is
        exactly the question of whether the second describes the first.
    """
    A0, B0 = _PAIRS["dense"](D=D, seed=seed)
    gate = _make_trotter_gate_tensor(
        jnp.asarray(_H_HEISENBERG_ROT.todense()), _SU_DT, site_tensor=A0
    )
    A, B, lambdas = _simple_update_checkerboard_sweep(A0, B0, gate, D, steps)
    return _SUState.from_pair(*_to_physical_pair(A, B, lambdas)), lambdas


def _vidal_theta(pair, weights, bond, gate=None):
    """:func:`_vidal_pair`'s reweighting, contracted across ``bond`` and densified.

    **This is the whole of what Task 12 found, written as ten lines of test
    code**, and :func:`_vidal_pair` is now the shared statement of it -- the
    reweighting used to live in both places.  ``gauge_fix`` returns the pair in
    *absorbed* form -- every bond weight split ``sqrt(lambda)`` into both of its
    ends -- so the two-site tensor ``A.B`` carries ``sqrt(lambda)`` on each of
    its six outer legs.  The Vidal-gauge canonical condition is
    ``sum Gamma (prod lambda**2) Gamma* = I`` with the **square** on the outer
    legs, and an absorbed pair supplies only ``lambda**1`` of that (``sqrt``
    from the ket and ``sqrt`` from the bra).  So the environment of the absorbed
    two-site tensor is *not* the identity, its SVD is *not* a Schmidt
    decomposition, and truncating it does not keep the largest Schmidt values of
    the state.

    One extra ``sqrt(lambda)`` on each outer leg fixes it: the ket then carries
    ``lambda``, the bra carries ``lambda``, the condition is met, and the SVD
    across the bond is the state's own Schmidt decomposition.

    That insertion is emphatically **not** "absorbing the weights again", which
    is the mistake ``_su_step``'s docstring rules out and which really would put
    ``lambda**2`` on the bond (#667's shape).  It touches the *outer* legs only,
    it leaves the bond alone, and a correct step divides it straight back out --
    measured, with no truncation the weighted and unweighted routes produce the
    same state to 1.3e-15 - 2.4e-15 on the four bonds (re-measured; the
    1.1e-15 this used to quote is below the minimum).  The two are
    different operations
    and the rewrite conflated them.

    This one densifies and returns an axis order, because the guard below wants
    a numpy SVD of a reshaped matrix; :func:`_two_site_tensor` is the same
    contraction kept as a ``Tensor``.  Both are three lines around
    :func:`_vidal_pair` and neither is worth folding into the other.
    """
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]
    weighted = _vidal_pair(pair, bond, weights)

    def rename(leg, prefix, phys):
        out = {lg: prefix + lg for lg in ("u", "d", "l", "r") if lg != leg}
        out[leg] = "__shared"
        out["phys"] = phys
        return out

    gated = gate is not None
    theta = contract(
        weighted[site_j].relabels(rename(leg_j, "__j", "sj" if gated else "__pj")),
        weighted[site_i].relabels(rename(leg_i, "__i", "si" if gated else "__pi")),
    )
    if gated:
        theta = contract(theta, _align_gate_to_ket(gate, weighted[site_i]))
        theta = theta.relabels({"si_out": "__pi", "sj_out": "__pj"})
    order = sorted(theta.labels())
    arr = theta.transpose(tuple(theta.labels().index(lab) for lab in order))
    return order, np.asarray(arr.todense())


def _truncation_error_of(theta_full, theta_kept):
    """``sqrt(1 - cos**2)`` between two two-site tensors, i.e. the relative
    error the truncation introduced, measured scale-free.

    The cosine rather than a difference of norms, because ``_su_step``
    deliberately does not renormalise and ``gauge_fix`` rescales by max-abs, so
    an overall factor between the two is expected and is not an error.  The
    cosine quotients it out exactly; normalising and subtracting does not (it
    agrees only to ``O(err**3)``, which is 4e-04 relative here -- above the
    1e-06 tolerance the guard below wants and therefore not usable).
    """
    cos = abs(np.vdot(theta_full, theta_kept)) / (
        np.linalg.norm(theta_full) * np.linalg.norm(theta_kept)
    )
    return float(np.sqrt(max(1.0 - cos**2, 0.0)))


# --- the metric itself, against numbers derived outside tenax --------------
#
# Everything above certifies ``_su_step`` *given* the Vidal metric.  Nothing
# above certifies the metric, and that gap has the same shape as the defect
# this task fixed: ``_vidal_pair`` (test) and ``_su_step``'s stage 2 (source)
# are the same prescription written twice, so a wrong prescription would score
# 1.000000 on both sides.  Two copies catch a mutation of either file -- they
# do not catch a shared conceptual error, which is exactly what the absorbed
# metric was.
#
# The anchor below has no copy of the prescription in it.  It runs on a
# **chain**, where belief propagation is exact rather than approximate, and
# compares against ``test_ipeps_gauge``'s ``_CHAIN_TRUTH`` /
# ``_SYM_CHAIN_TRUTH`` -- the infinite chain's Schmidt spectra rebuilt from the
# 2-site transfer matrix's fixed points in Python ``decimal``, importing
# nothing from tenax but two float64 arrays.  Phase 1 Task 6/6b built those and
# proved ``gauge_fix``'s weights equal them; this asks the next question, which
# is whether the *two-site tensor* the metric builds has them as its singular
# values.

#: The external truths and the pair builders, per arm.  Imported from
#: ``test_ipeps_gauge`` rather than copied: a second copy of an
#: externally-certified constant is a second thing to keep right, and the
#: import fails loudly if it moves.  (``tests/`` is on ``sys.path``, which is
#: how ``_ipeps_gauge_helpers`` is already reached.)
_CHAIN_ARMS = {
    "dense": (_chain_pair, _chain_pair_as_peps, _CHAIN_TRUTH),
    "symmetric": (_sym_chain_pair, _sym_chain_pair_as_peps, _SYM_CHAIN_TRUTH),
}

#: How the chain is laid on the lattice, and what that does to the *leg set*
#: the anchor exercises.
#:
#: **This axis exists because the first version of the anchor was blind on half
#: of it, and the blindness had the same shape as the defect the whole task is
#: about.**  ``_chain_pair_as_peps`` gives the vertical bonds dimension 1, so
#: ``gauge_fix`` returns ``v_AB = v_BA = [1.0]`` exactly -- and any power of 1.0
#: is 1.0.  Four of the six outer legs of a horizontal bond are therefore
#: *inert*: a metric error confined to ``u``/``d`` changes nothing the
#: horizontal arm can see.
#:
#: **What that error actually does, re-measured, because the first version of
#: this note got half of it wrong.**  Build it as a *shared* error -- stage 2/5
#: in ``_su_step`` and :func:`_vidal_pair` both skipping ``u`` and ``d``, so
#: reference and code move together -- and:
#:
#: * it does **not** pass the square-lattice guard.  This note used to say it
#:   did.  ``test_su_step_truncates_in_the_state_s_own_basis`` kills it on all
#:   four bonds at its reading 0, whose right-hand side is the bond's own
#:   ``gauge_fix`` weight rather than the prescription under test: off-diagonal
#:   6.971e-02 (``h_AB``), 7.623e-02 (``h_BA``), 3.160e-02 (``v_AB``) and
#:   4.980e-02 (``v_BA``) against ``_CANONICAL_TOL`` = 1e-4;
#: * it **does** pass the horizontal-only anchor, on both parities -- that half
#:   of the old claim reproduces exactly;
#: * it dies on the ``vertical`` anchor, at 2.212e-01 (``v_AB``) and 1.921e-01
#:   (``v_BA``) on reading 1.
#:
#: The *production-only* variant -- the test metric left correct, so the error
#: is not shared -- costs 1.008741x (``h_AB``), 1.006569x (``h_BA``), 1.002311x
#: (``v_AB``) and 1.002605x (``v_BA``) a step.  Three of those four sit
#: **below** the 1.0048 floor of the 1.0048-1.0234x range this note used to
#: compare them to, so "the same order as the defect this task fixed" was
#: generous; 1.008741x, the reading it quoted, is the largest of the four.
#:
#: **So the ``vertical`` axis is kept on a narrower justification than it was
#: written on: it is the only place ``u``/``d`` are certified against a
#: non-tenax number.**  The square lattice catches a leg-set error too, but
#: against tenax's own ``gauge_fix`` weights; only the chain has a spectrum
#: nothing in this tree computed.  Measured cost of the axis in the *required*
#: gate: **1.31 s** for its two ``core``-marked dense cells, against 2.26 s for
#: the two horizontal ones (which pay the chain's first gauge solve), box load
#: ~6.  Its symmetric half is ~40 s and stays in ``algorithm``.
#:
#: The chain embedding is a free parameter, and that is the whole remedy: laying
#: the same two MPS tensors along ``u``/``d`` instead makes the *vertical* bonds
#: carry the spectrum and the horizontal ones ``[1.0]``, so the live leg set is
#: exactly complementary.  Same state, same ``_CHAIN_TRUTH`` constants, and
#: nothing about the reference changes -- MPS bond ``i`` is ``a.right <-> b.left``
#: for even ``i``, which the horizontal embedding sends to ``A.r <-> B.l``
#: (``h_AB``) and the vertical one to ``A.d <-> B.u`` (``v_AB``).
#:
#: The relabel is an involution and it preserves every flow, which is why it is
#: a relabel and not a second builder: ``u`` and ``l`` are both ``OUT`` and
#: ``d`` and ``r`` are both ``IN`` in the shipped convention, so swapping the
#: two pairs leaves each index object -- flow, charges, dimension -- exactly
#: where it was and only renames the axis.  On the block-sparse arm that also
#: leaves the conservation law untouched: ``-q_u + q_d - q_l + q_r + q_phys``
#: is the same sum with ``u <-> l`` and ``d <-> r`` exchanged.
#:
#: ``(relabel, parity -> bond, the leg names that carry a live weight)``.
_CHAIN_ORIENTATIONS = {
    "horizontal": ({}, {"h_AB": "h_AB", "h_BA": "h_BA"}, {"l", "r"}),
    "vertical": (
        {"u": "l", "l": "u", "d": "r", "r": "d"},
        {"h_AB": "v_AB", "h_BA": "v_BA"},
        {"u", "d"},
    ),
}

#: The union of the two orientations' live leg sets is all four virtual legs.
#: Asserted at import rather than left to a reader: if a future edit makes one
#: orientation inert again -- which is what happened to the vertical half the
#: first time -- this is what says so, and the per-cell meta-assertion in the
#: test says which one.
assert set().union(*(live for _r, _m, live in _CHAIN_ORIENTATIONS.values())) == {
    "u",
    "d",
    "l",
    "r",
}, "the chain orientations no longer cover all four virtual legs between them"
#
# That one is table self-consistency and nothing in ``ipeps_su`` can falsify
# it -- it is an assertion about the two literals four lines above.  It is the
# per-cell reading 0 inside the test that checks the table against the weights
# ``gauge_fix`` actually returns, and that one can fire.

#: How close the metric must land on the external truth.  Not the 1e-12 the
#: Phase 1 anchor uses, and the difference is one thing: that test solves at
#: ``tol=1e-14`` while this one must use ``gauge_fix``'s **default** ``tol=1e-6``
#: -- the call ``_su_step`` makes -- so that the reference pair and the stepped
#: pair are in the *same* BP basis.  (Solving the reference at 1e-14 instead
#: puts the two thetas in different gauges and the optimality ratio below reads
#: 8.5 on correct code; that was measured, not guessed.)  At the default the
#: weights sit ~2e-7 from the fixed point.  Measured here: the Vidal metric
#: lands 2.1e-08 to 1.9e-07 from the truth, the two neighbouring powers land
#: 1.4e-02 to 2.2e-01 away -- five orders apart, so this gate has no judgement
#: in it.
_METRIC_TOL = 1e-5

#: Lower bound the two wrong powers must clear, so the cell cannot pass because
#: the anchor has gone flat.
_METRIC_CONTROL = 1e-3

#: How close the externally-anchored optimality ratio must sit to 1.  Two-sided,
#: unlike the square-lattice guard's one-sided Eckart-Young bound, and that is
#: the point: with ``optimal`` computed from the external truth rather than from
#: a tenax SVD of the same theta, a *wrong metric* makes the ratio land on the
#: wrong side as easily as the right one.  Measured on this pair, the absorbed
#: reading gives 2.9x to 10.0x and the doubled one 0.09x to 0.87x, against a
#: correct 1.000000 to 1.000002.  The 2e-6 residual is ``gauge_fix``'s default
#: tolerance, not the anchor's.
_METRIC_RATIO_TOL = 1e-4

#: How close the *square-lattice* BP gauge must sit to the Vidal canonical
#: condition under the shipped metric.  Set off the measurement at
#: ``gauge_fix``'s default ``tol=1e-6``, where the correct metric reads
#: 1.2e-07 to 6.2e-07 (dense ``D=3``, three seeds, four bonds, both readings)
#: and the wrong ones floor at 1.9e-02; two orders of headroom above the former
#: and **2.3** below the latter.  (The previous note said "five below", which
#: was wrong arithmetic on its own figures -- 4.6e-02 / 1e-4 is 2.7 orders --
#: and quoted a correct-metric band that did not reproduce.)  It tracks the solve
#: tolerance (at ``tol=1e-10`` the same readings are ~3e-11), so tightening it
#: without re-measuring would couple this guard to a knob it does not set.
_CANONICAL_TOL = 1e-4


@pytest.fixture(scope="module")
def chain_anchor():
    """Memoised ``(pair, gauged pair, weights, identity gate, truth)`` per arm.

    Keyed by ``(arm, orientation)`` and module-scoped, because a symmetric
    chain ``gauge_fix`` is ~6 s and two cells share each solve.  The gate is
    ``exp(-0 * 0) == 1``: an identity, so the two-site tensor the step forms is
    the *ungated* one, whose spectrum the external truth is a statement about.
    A real gate would move the state to one nothing outside tenax has a number
    for.

    The ``vertical`` orientation is the same pair with ``u <-> l`` and
    ``d <-> r`` relabelled -- see :data:`_CHAIN_ORIENTATIONS` for why that is a
    relabel rather than a second builder, and for the blind region it exists to
    close.
    """
    cache: dict[tuple[str, str], tuple] = {}

    def get(arm, orientation):
        key = (arm, orientation)
        if key not in cache:
            build_pair, as_peps, truth = _CHAIN_ARMS[arm]
            relabel, _parity_map, _live = _CHAIN_ORIENTATIONS[orientation]
            a, b, _vl, _vr = build_pair()
            A, B = as_peps(a, b)
            if relabel:
                A, B = A.relabels(relabel), B.relabels(relabel)
            # ``gauge_fix``'s DEFAULT tol -- the call ``_su_step`` makes.  See
            # ``_METRIC_TOL`` for why matching it is load-bearing.
            A_g, B_g, weights, info = gauge_fix(A, B)
            assert info.converged, (
                f"{arm} {orientation} chain: BP did not converge where it is "
                f"exact ({info.iterations} sweeps, residual "
                f"{info.residual:.3e}); nothing below can be concluded from a "
                f"failed solve"
            )
            d = A.indices[A.labels().index("phys")].dim
            gate = _make_trotter_gate_tensor(
                jnp.zeros((d, d, d, d)), 0.0, site_tensor=A
            )
            cache[key] = (A, B, A_g, B_g, weights, gate, truth)
        return cache[key]

    return get


# The ``dense`` half of this guard carries an explicit ``core`` mark, and that is
# the one marker decision in this file that is not about cost.
#
# ``tests/conftest.py`` maps the whole file to ``"algorithm"`` and CI's required
# checks run ``-m core``, so without this the **only externally-anchored fact in
# #882 Phase 2** would live in the non-required bucket -- the one this project's
# own memory records as chronically red on ``main``.  Phase 1 made the same call
# explicitly for its twin (``test_ipeps_gauge.py``'s
# ``test_bp_weights_are_the_chains_schmidt_values``, whose docstring argues that
# "without this marker the one test in the whole rewrite with an externally known
# answer would land only in the non-required fast-other bucket").
#
# **Dense only, and the split is measured rather than cautious.**  All eight
# cells cost 44.24 s; the four ``dense`` ones cost **4.70 s**, and between them
# they cover both orientations -- hence all four virtual legs (see
# ``_CHAIN_ORIENTATIONS``) -- and both bond parities.  The symmetric arm is ~40 s
# of the 44 and stays in ``algorithm``; what it adds is the charge bookkeeping,
# which is not what a required gate needs at that price.
#
# **The composition was checked, not assumed**, because getting it backwards
# drops the anchor from *every* bucket rather than adding it to one:
# ``pytest_collection_modifyitems`` *adds* the file's bucket marker and withholds
# it only for ``core``-mapped files carrying an explicit ``slow``, so these cells
# end up ``core`` **and** ``algorithm``.  ``-m core`` selects them; the fast
# buckets (``not core and not slow``) drop them, which is the intended trade --
# they run in a required job instead.  Verified on
# ``tests/test_ipeps_gauge.py``, an ``algorithm`` file that already does this:
# ``-m core`` collects 2 of 27 and ``-m "not core and not slow"`` collects 22.
@pytest.mark.parametrize(
    "arm", [pytest.param("dense", marks=pytest.mark.core), "symmetric"]
)
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("parity", ["h_AB", "h_BA"])
def test_the_vidal_metric_matches_a_spectrum_derived_outside_tenax(
    arm, orientation, parity, chain_anchor
):
    """The metric is right, checked against a number tenax did not compute.

    **This is the guard that closes the last self-reference in the file.**  The
    optimality guard below certifies ``_su_step``'s truncation *given* a metric;
    it builds its reference with ``_vidal_pair``, which is the same prescription
    ``_su_step``'s stage 2 applies, so a shared conceptual error in the metric
    reads 1.000000 on both sides.  That is the shape of the defect this task
    fixed, one level up.

    The reference here is ``_CHAIN_TRUTH`` / ``_SYM_CHAIN_TRUTH``: the infinite
    chain's two horizontal-bond Schmidt spectra, rebuilt from the 2-site
    transfer matrix's left and right fixed points in Python ``decimal``,
    importing nothing from tenax but two ``float64`` site tensors.  No
    prescription of any kind is involved in producing them.

    **Why a chain.**  BP is exact on a tree and only approximate on the loopy
    square lattice, so a chain is the one geometry where "the BP gauge is the
    canonical form" is a theorem rather than an ansatz -- which makes it the one
    geometry where the metric has a ground truth at all.  Phase 1 Task 6/6b
    built this anchor and proved ``gauge_fix``'s *weights* equal those numbers;
    this asks the next question, which no test asked: does the **two-site
    tensor** the metric builds have them as its singular values?  It does, and
    the two neighbouring powers do not.

    **Why two orientations.**  A chain embedded horizontally gives the vertical
    bonds dimension 1, so ``gauge_fix`` returns ``v_AB = v_BA = [1.0]`` and four
    of the six outer legs are *inert* -- any power of 1.0 is 1.0.  The first
    version of this guard had only that embedding, and a metric error confined
    to ``u``/``d`` passed it on both parities.  The ``vertical`` orientation is
    the identical state with ``u <-> l`` and ``d <-> r`` relabelled, so the live
    leg set is exactly complementary and the union is all four; measured on the
    widened guard, the shared ``u``/``d`` error dies at **2.212e-01**
    (``v_AB``) and **1.921e-01** (``v_BA``) on reading 1.

    **What the axis is not.**  It is not the only thing that catches that
    error, and this docstring used to say it was -- that the shared ``u``/``d``
    error "passed the square-lattice optimality guard" as well.  It does not:
    ``test_su_step_truncates_in_the_state_s_own_basis``' reading 0 kills it on
    all four bonds at 3.160e-02 to 7.623e-02 against a 1e-4 gate.  What is left,
    and what keeps the axis, is narrower and worth stating exactly: this is the
    only place ``u``/``d`` are certified against a number **tenax did not
    compute**.  See :data:`_CHAIN_ORIENTATIONS` for both measurements and for
    the axis's 1.31 s cost in the required gate, and the per-cell reading 0
    below, which asserts which legs this cell is actually exercising rather
    than trusting the table.

    Two readings, both against the external truth:

    * **the spectrum** -- ``svd(theta_vidal)`` must be the truth.  Measured,
      ``power=1`` (the shipped metric) lands 2.1e-08 to 1.9e-07 away;
      ``power=0`` (the absorbed reading this file used before the reopening)
      1.1e-01 to 2.2e-01; ``power=2`` 1.4e-02 to 7.6e-02.
    * **the truncation** -- ``_su_step``'s own error at ``max_D`` in 2, 3
      against ``||truth[max_D:]|| / ||truth||``, which is computed from the
      external constant alone.  ``power=1`` gives 1.000000 to 1.000002;
      ``power=0`` gives 2.9x to 10.0x and ``power=2`` gives 0.09x to 0.87x.

    The second reading is the one that reaches ``_su_step``: the ``optimal``
    half comes from outside tenax entirely, so a metric that is wrong *in both
    files at once* still cannot produce a ratio of 1.  The ratio gate is
    therefore **two-sided** here, unlike the square-lattice guard's one-sided
    Eckart-Young bound -- a wrong metric measures the error of the wrong
    operator and lands below 1 as readily as above it, which is exactly what
    ``power=2`` does.

    **What this still does not certify**, stated because the point of the guard
    is to be precise about its own reach: BP on the *square lattice* is
    approximate, so this says the prescription is the canonical-form
    prescription, not that ``gauge_fix``'s square-lattice fixed point reproduces
    the *exact* environment.  Nothing can say that -- there is no exact
    reference on a loopy lattice (#882 section 6.3).  Two things do reach the
    square lattice: ``test_su_step_truncates_in_the_state_s_own_basis``'s
    reading 0, which checks the *metric* against the canonical condition's own
    absolute right-hand side there (five orders of separation, and it is what
    catches a leg-set error on the real geometry -- it calls no ``_su_step``,
    so it reaches the square-lattice **geometry**, not the step); and the
    energy, 0 of 9 (seed, D) cells before the fix and 5 of 9 after, against
    references this project reproduced independently.
    """
    A, B, A_g, B_g, weights, gate, truth = chain_anchor(arm, orientation)
    _relabel, parity_map, live_expected = _CHAIN_ORIENTATIONS[orientation]
    bond = parity_map[parity]
    want = np.sort(np.asarray(truth[parity]))[::-1]
    want = want / np.linalg.norm(want)
    gauged = {"A": A_g, "B": B_g}
    chi = len(want)

    # --- reading 0: which of this cell's six outer legs are not inert --------
    #
    # The meta-assertion that keeps the two orientations honest.  A leg whose
    # weight is a single number (or a flat vector) contributes a global factor
    # that both the normalised spectrum and the cosine quotient out, so the
    # readings below are blind to whatever the metric does to it.  On the
    # horizontal embedding that is ``u`` and ``d``; on the vertical one ``l``
    # and ``r``.  If a future edit made *this* cell's live set shrink, the two
    # arms would stop covering the four legs between them and a metric error
    # confined to the newly-inert pair would pass everything -- which is
    # exactly what happened to the vertical half before this arm existed.
    live = set()
    for site, leg in _BOND_ENDS[bond]:
        for other_leg in ("u", "d", "l", "r"):
            if other_leg == leg:
                continue
            lam = np.asarray(getattr(weights, _BOND_OF_LEG[(site, other_leg)]))
            if lam.size > 1 and not np.allclose(lam / lam[0], 1.0):
                live.add(other_leg)
    assert live == live_expected, (
        f"{arm} {orientation} {bond}: the outer legs carrying a non-trivial "
        f"weight are {sorted(live)}, expected {sorted(live_expected)}.  A leg "
        f"whose weight is flat is inert -- every reading below quotients a "
        f"global factor out -- so this cell certifies the metric only on the "
        f"legs listed.  The two orientations are what make the union all four; "
        f"if this set shrinks they no longer do."
    )

    # --- reading 1: the spectrum of the metric's own theta -------------------
    got = {}
    for power in (None, 0.0, 2.0):
        # ``None`` = ``_vidal_pair``'s shipped default; see the same note in
        # ``test_su_step_truncates_in_the_state_s_own_basis``.
        sigma = (
            _bond_spectrum(gauged, bond, chi, weights)
            if power is None
            else _bond_spectrum(gauged, bond, chi, weights, power=power)
        )
        sigma = np.sort(np.asarray(sigma))[::-1]
        got[power] = float(np.max(np.abs(sigma / np.linalg.norm(sigma) - want)))
    for power in (0.0, 2.0):
        assert got[power] > _METRIC_CONTROL, (
            f"{arm} {orientation} {bond}: the metric with lambda**{power:.0f} on the outer "
            f"legs is {got[power]:.3e} from the chain's exact Schmidt spectrum, "
            f"inside the {_METRIC_CONTROL:.0e} this cell needs it to miss by.  "
            f"Then the assertion below cannot tell the right power from a wrong "
            f"one and passes for the wrong reason -- the anchor has gone flat, "
            f"redraw the chain seed rather than loosening anything."
        )
    assert got[None] < _METRIC_TOL, (
        f"{arm} {orientation} {bond}: the two-site tensor built in the Vidal metric has "
        f"singular values {got[None]:.3e} away from the chain's exact Schmidt "
        f"spectrum, which was derived outside tenax.  BP is exact on a tree, so "
        f"this is not a tolerance to widen: the metric _vidal_pair applies is "
        f"not the canonical-form metric.  (Reading 1 calls no _su_step at all "
        f"-- it is a reading of _vidal_pair on the gauged pair -- so it cannot "
        f"say anything directly about the source's stage 2, whatever this "
        f"message used to claim.  Reading 2 below is the half that steps; a "
        f"metric error present in BOTH files does reach here, which is the "
        f"whole point of the external truth.)  lambda**0 scores {got[0.0]:.3e} "
        f"and lambda**2 scores {got[2.0]:.3e}."
    )

    # --- reading 2: _su_step's truncation, scored from outside tenax ---------
    state = _SUState.from_pair(A, B)
    for max_D in (2, chi - 1):
        optimal = float(np.linalg.norm(want[max_D:]) / np.linalg.norm(want))
        # A statement about ``_CHAIN_TRUTH``, **not** about this module: both
        # sides are derived from the frozen external literal, so no change to
        # ``ipeps_su`` can make it fire.  Labelled rather than deleted, on the
        # model of the ``jnp.sqrt`` VJP canary above: it is what would say so if
        # a future re-draw of the chain seed made the truncation free.
        assert optimal > _METRIC_CONTROL, (
            f"{arm} {orientation} {bond}: keeping {max_D} of {chi} costs only {optimal:.3e} "
            f"on this draw, so the ratio below cannot discriminate"
        )
        stepped = _su_step(state, gate, max_D=max_D, bond=bond)
        ratios = {}
        for power in (None, 0.0, 2.0):
            args = () if power is None else (power,)
            full = _two_site_tensor(gauged, bond, weights, *args)
            kept = _two_site_tensor(stepped, bond, weights, *args)
            actual = _truncation_error_of(
                np.asarray(full.todense()), np.asarray(kept.todense())
            )
            ratios[power] = actual / optimal
        for power in (0.0, 2.0):
            assert abs(ratios[power] - 1.0) > 1e-2, (
                f"{arm} {orientation} {bond} max_D={max_D}: scoring the step in the "
                f"lambda**{power:.0f} metric still gives ratio "
                f"{ratios[power]:.6f}, so this cell cannot tell the metric it "
                f"is supposed to be certifying from a wrong one"
            )
        assert abs(ratios[None] - 1.0) < _METRIC_RATIO_TOL, (
            f"{arm} {orientation} {bond} max_D={max_D}: _su_step's truncation error is "
            f"{ratios[None]:.6f}x the best achievable, where the best is "
            f"||truth[{max_D}:]||/||truth|| = {optimal:.9f} computed from a "
            f"spectrum derived outside tenax.  The gate is two-sided on "
            f"purpose: below 1 means the error is being measured in the wrong "
            f"metric, not that the step beat Eckart-Young.  lambda**0 gives "
            f"{ratios[0.0]:.6f}x and lambda**2 gives {ratios[2.0]:.6f}x."
        )


@pytest.mark.parametrize("bond", _BONDS)
def test_su_step_truncates_in_the_state_s_own_basis(bond):
    """The kept subspace must be the best rank-``max_D`` one **for the state**.

    This is the most localised statement of what Task 12 found, and the cheapest
    -- one step, no CTM, no imaginary-time run.  Everything else in this section
    is this defect, compounded 800 times.

    Eckart-Young in the state's own metric: of all rank-``max_D`` truncations of
    the gated two-site tensor, the one that keeps the top ``max_D`` singular
    values of the *Vidal* theta has the smallest error, and every other one is
    strictly worse.  So ``error(_su_step) / error(optimal) >= 1`` always, with
    equality **iff** the step truncated in the right basis.  That makes the
    tolerance below a statement about correctness rather than about numerics.

    Measured out of band on dense ``D=3``, ``dt=0.05``, all four bonds x three
    seeds (this test runs seed 0's four bonds; the seed axis added nothing here
    and costs three gauge solves a bond, so it is not parametrised):

    ==================================  ==============================
    ``_su_step`` truncating absorbed    ratio 1.004775 to 1.023395
    ``_su_step`` truncating Vidal       ratio 1.000000 on all twelve
    ==================================  ==============================

    Both rows were re-measured for Task 10's reopening and reproduce Task 12's
    1.0048-1.0234 exactly; seed 0's four bonds, which is what this test runs,
    read 1.009201, 1.009628, 1.010690 and 1.010944 before the fix.  So the guard
    has been watched in **both** directions -- failing on all twelve cells as
    shipped, and passing at exactly 1.000000 on all twelve with the outer-leg
    weights inserted.  A guard whose passing state has never been observed is a
    guard that might not have one.  1-2% a step does not sound like much;
    compounded over 800 steps it was the difference between -0.6589 and the
    product state (see the energy guards below, and
    ``task-10-reopen-report.md`` for the re-run grid).

    **The reference does not come from ``_su_step``'s SVD, truncation or split,
    and that is why this is the guard that caught the defect.** The twelve
    guards above it compare the step against a spectrum, a subspace or a state
    rebuilt from the step's own output, so a step that corrupts the tensor it is
    truncating corrupts the reference in the same breath and they agree.  The
    reference here is Eckart-Young -- a lower bound on the truncation error that
    holds for *any* rank-``max_D`` truncation of the gated state, whoever
    computed it.

    **What it does share with the code it audits**, stated precisely because on
    this task the scope of an independence claim is the thing that matters:

    * ``gauge_fix`` -- unavoidable, and Phase 1 code rather than code under test.
    * ``_align_gate_to_ket`` and ``_BOND_ENDS``, both imported from
      ``ipeps_su``.  A gate-alignment error would move ``full`` and ``stepped``
      together and this ratio would still read 1.000000.  Covered elsewhere:
      ``test_su_step_applies_the_gate_across_the_bond`` derives its gate axes
      from ``_TORUS_GATE_AXES``, written out independently from ``ipeps_gauge``'s
      edge table, and ``test_su_evolve_visits_four_distinct_bonds_per_cycle``
      pins the bond map against a literal written out in this file.
    * **the metric** -- ``_vidal_pair`` here and stage 2 in the source are the
      same prescription written twice, so a *shared* error in it would read
      1.000000 on both sides.  Two copies in two files catch a mutation of
      either (measured: mutating the module's ``_BOND_OF_LEG`` dies at
      1.000551x, halving its leg set at 1.005972x); they do not catch a shared
      conceptual error, which is exactly what the absorbed metric was.
      ``test_the_vidal_metric_matches_a_spectrum_derived_outside_tenax`` is the
      guard that does, against transfer-matrix fixed points computed in Python
      ``decimal``.

    **Scope: dense, seed 0, four bonds.**  The seed axis was measured to add
    nothing here (all twelve pre-fix cells sit in 1.004775-1.023395) and costs
    three gauge solves a bond.  There is no symmetric cell *in this test*, and
    the block-sparse path is where this project's charge-order and pairing bugs
    live (#602, #834, #865) -- so the symmetric optimality reading lives in the
    chain anchor above, which covers both arms in 25.8 s for four cells because
    a chain's ``gauge_fix`` is 6.1 s where a symmetric ``D=3`` square-lattice
    one is 38 s.

    The reference is built here rather than imported, and deliberately does not
    reuse ``_su_step``'s own SVD: it re-derives the two-site tensor from the
    gauged pair, applies the gate itself, and takes the singular values with
    numpy.  A reference that shared the implementation's basis would agree with
    it by construction.
    """
    A, B = _PAIRS["dense"](D=D, seed=0)
    state = _SUState.from_pair(A, B)
    gate = _su_heisenberg_gate(state)

    A_g, B_g, weights, info = gauge_fix(A, B)
    assert info.converged, (
        f"the reference gauge did not converge ({info.iterations} sweeps, "
        f"residual {info.residual:.3e}), so the basis this compares against is "
        f"not the BP fixed point and neither number below means anything"
    )

    # --- reading 0: the metric makes THIS gauge canonical -------------------
    #
    # **The prescription-free half of this test, and the one that survives a
    # shared metric error.**  Everything below scores `_su_step` against a
    # reference built with `_vidal_pair`, which is the same prescription stage 2
    # applies -- so a metric error present in *both* files reads 1.000000 here
    # and is invisible.  This reading is not: the Vidal canonical condition has
    # an absolute right-hand side.  With the metric applied, each end's
    # environment traced over every other leg must be **diagonal**, and its
    # diagonal must be that bond's own weight.  Nothing in this tree computes a
    # diagonal matrix to compare against, and `lambda` comes from `gauge_fix`'s
    # weights table rather than from the prescription, so neither side can be
    # moved by getting the prescription wrong.
    #
    # What it certifies is that the prescription is the one under which the BP
    # *fixed point* is a canonical form -- BP's own fixed-point equation, on the
    # real square lattice.  It does **not** certify that the BP environment is
    # the exact one; on a loopy lattice it is not, and that is what the chain
    # anchor covers, where the two coincide by theorem.
    #
    # Measured, dense D=3, three seeds, at `gauge_fix`'s default `tol=1e-6`:
    #
    #     metric       max off-diagonal      diag against lambda
    #     lambda**1    1.2e-07 - 4.2e-07     1.4e-07 - 6.2e-07
    #     lambda**0    2.1e-02 - 1.0e-01     5.3e-02 - 2.1e-01
    #     lambda**2    1.9e-02 - 1.1e-01     5.2e-02 - 1.7e-01
    #     skip u,d     2.0e-02 - 7.6e-02     3.8e-02 - 2.4e-01
    #     skip l,r     1.1e-02 - 6.3e-02     2.1e-02 - 1.2e-01
    #
    # **Every floor in that table was quoted too high before #882's final
    # review; the ceilings were right.**  It matters for one row: `skip l,r`
    # reaches 1.1e-02 off-diagonal on some (seed, bond) against the 1e-2 the
    # control below gates at -- a 1.1x margin where the old figures implied
    # 4.6x.  Two things keep that from being a flake.  The control is an `or`,
    # and the same cell's `diag against lambda` reading floors at 2.1e-02, so
    # the pair together clears 1e-2 by 2.1x; and `skip l,r` is not one of the
    # rows this test asserts on at all -- it runs `lambda**0` and `lambda**2`,
    # whose worst floor over three seeds is 1.9e-02 / 5.2e-02, i.e. 5.2x on the
    # `or`.  On seed 0, which is the seed this cell actually runs, those two
    # rows floor at 6.4e-02 and 5.2e-02.  The table is documentation of the
    # neighbourhood; the assertion's own margin is the second pair of numbers.
    #
    # Five orders of separation between `lambda**1` and the wrong metrics, and
    # the last two rows are the point: a *leg-set* error confined to one
    # orientation is caught here, on the square lattice, where the chain
    # anchor's two embeddings each see only half the legs.  The residual at
    # `lambda**1` tracks the solve tolerance -- at `tol=1e-10` the same readings
    # are 2.3e-11 and 3.7e-11 -- so the gate is set off the default-`tol` figure
    # with two orders of headroom and is not a tolerance to tighten without
    # re-measuring.
    for power in (None, 0.0, 2.0):
        # ``None`` means "whatever ``_vidal_pair`` ships as its default", not
        # "1.0".  Passing the literal instead would make this reading survive a
        # changed default -- measured: with the argument spelled out, the
        # shared ``lambda**0`` mutation passes this guard.
        vp = (
            _vidal_pair({"A": A_g, "B": B_g}, bond, weights)
            if power is None
            else _vidal_pair({"A": A_g, "B": B_g}, bond, weights, power)
        )
        lam = np.asarray(getattr(weights, bond), dtype=float)
        worst_off, worst_lam = 0.0, 0.0
        for site, leg in _BOND_ENDS[bond]:
            G = _gram(vp[site], leg)
            worst_off = max(
                worst_off,
                float(np.max(np.abs(G - np.diag(np.diag(G)))) / np.max(np.abs(G))),
            )
            d = np.diag(G).real
            worst_lam = max(worst_lam, float(np.max(np.abs(d / d[0] - lam / lam[0]))))
        if power is None:
            assert worst_off < _CANONICAL_TOL and worst_lam < _CANONICAL_TOL, (
                f"{bond}: with the shipped metric the gauged pair's one-site "
                f"environment is {worst_off:.3e} off diagonal and its diagonal "
                f"is {worst_lam:.3e} from this bond's own weight.  The Vidal "
                f"canonical condition is what makes an SVD across the bond a "
                f"Schmidt decomposition, and its right-hand side is not "
                f"computed by the prescription being checked -- so this is the "
                f"one reading in this test that a metric error present in BOTH "
                f"_vidal_pair and _su_step's stage 2 cannot hide from."
            )
        else:
            assert worst_off > 1e-2 or worst_lam > 1e-2, (
                f"{bond}: the lambda**{power:.0f} metric leaves the environment "
                f"{worst_off:.3e} off diagonal and {worst_lam:.3e} from the "
                f"bond weight, inside the 1e-2 this control needs it to miss "
                f"by -- the reading above then cannot tell the canonical metric "
                f"from a wrong one and passes for the wrong reason"
            )

    order, full = _vidal_theta({"A": A_g, "B": B_g}, weights, bond, gate)

    (_site_i, leg_i), (_site_j, leg_j) = _BOND_ENDS[bond]
    left = [f"__j{lg}" for lg in ("u", "d", "l", "r") if lg != leg_j] + ["__pj"]
    perm = [order.index(x) for x in left] + [
        order.index(x) for x in order if x not in left
    ]
    sigma = np.linalg.svd(
        full.transpose(perm).reshape(2 * D**3, 2 * D**3), compute_uv=False
    )
    # Like the anchor's twin, this cannot fire on anything in ``ipeps_su``: it
    # is a property of the *fixture's* theta, measured at ~2.0e-02 on all four
    # bonds, and would need ``theta`` to collapse to rank <= 3 for a D=3 pair.
    # Kept and labelled rather than deleted -- it is what would report a dt or
    # a draw at which truncation stopped biting.
    optimal = float(np.linalg.norm(sigma[D:]) / np.linalg.norm(sigma))
    assert optimal > 1e-6, (
        f"the untruncated theta on {bond} is already rank {D}, so keeping "
        f"{D} of it is exact and this cell cannot discriminate between two "
        f"truncation bases -- pick a dt or a pair where truncation bites"
    )

    stepped = _su_step(state, gate, max_D=D, bond=bond)
    _order, kept = _vidal_theta({"A": stepped.A, "B": stepped.B}, weights, bond)
    actual = _truncation_error_of(full, kept)

    # One-sided, unlike the chain anchor's twin, and the asymmetry is worth
    # naming: ``_truncation_error_of`` contracts the bond away, so an
    # ``_su_step`` that ignored ``max_D`` entirely would read ``actual ~ 0`` and
    # pass here.  Eckart-Young only bounds it from below.  The anchor gates two
    # -sided because its ``optimal`` comes from outside tenax, where "below 1"
    # means the error is being measured in the wrong metric; here ``optimal``
    # is a numpy SVD of the same theta, so below-1 is unreachable rather than
    # meaningful.  A step that ignores ``max_D`` is caught by
    # ``test_su_step_keeps_the_largest_singular_values``' dimension assertion
    # and by ``_su_evolve``'s post-step bond-uniformity invariant.
    assert actual <= optimal * (1 + 1e-6), (
        f"{bond}: _su_step's truncation costs {actual:.9f} where the best "
        f"rank-{D} truncation of the same gated state costs {optimal:.9f} "
        f"({actual / optimal:.6f}x).  It truncates the SVD of the *absorbed* "
        f"two-site tensor, whose environment is not the identity, so its "
        f"singular values are not the state's Schmidt values.  Insert one "
        f"sqrt(lambda) on each of the six outer legs before the SVD (the "
        f"weights gauge_fix already returns and _su_step drops) and divide it "
        f"back out after: measured, that makes this ratio 1.000000 on all "
        f"twelve dense cells, and it is exact at full rank (1.3e-15 to "
        f"2.4e-15), so it "
        f"changes the truncation basis and nothing else."
    )


def test_the_energy_reference_reproduces_the_known_su_number():
    """``_energy_of`` is checked before anything is concluded from it (C-3).

    A CTM reference that has silently collapsed returns a plausible wrong
    number, and that has invalidated a whole benchmark on this project before.
    Three independent things are asserted here, on a state built by the
    **shipped** engine so that the apparatus is validated against physics this
    tree already knows rather than against the code under test:

    * **The number.**  ``-0.659004`` at D=2, against this project's post-#667
      reference of ``-0.6593``.
    * **It has settled in chi.**  ``chi=8`` and ``chi=16`` agree to 1.4e-11,
      and ``chi=24`` and ``chi=32`` add nothing (3.3e-16 and 0.0).  An energy
      still moving with chi is not an answer.
    * **The recipe is not the collapsing one.**  ``recipe="1x1"`` on the same
      state collapses to ``rank(C1) == 1``, and ``_energy_of`` is driven at that
      recipe here inside a ``pytest.raises`` so its rank assertion is watched
      firing.  (Out of band, the number it would have returned is ``-0.648904``
      -- wrong by 1e-02, the size of the whole D=2-to-D=4 spread.  This test
      does not compute it and does not assert it: ``_energy_of`` refuses before
      returning, which is the point.)

    **This cell validates the apparatus, not the engine, and that is why it
    survives sabotage of the module under test.**  Measured: with ``_su_step``,
    ``_su_evolve``, ``_align_gate_to_ket``, ``_sqrt_and_inv_sqrt``,
    ``_require_uniform_bonds``, ``_bond_dims`` and ``_reorder`` all replaced by
    functions that raise, in both ``ipeps_su``'s namespace and this file's, this
    cell **passes** (19.4 s).  It should: the state comes from
    ``ipeps_simple_update`` via :func:`_shipped_su_run` and the reading from
    ``_ctm_tensor_*``; the only ``ipeps_su`` code it touches is
    ``_SUState.from_pair``'s label check.  So **do not count it as coverage of
    ``ipeps_su``** -- no mutation of that module can fail it, by design.  What
    it covers is :func:`_energy_of`, which every energy guard below is scored
    with, against a CTM number this project already had.

    Also measured out of band and asserted nowhere here, so it is a record
    rather than a guard: ``chi=24`` and ``chi=32`` add 3.3e-16 and 0.0 to the
    ``chi=16`` reading, and no ``check_rdm`` non-PSD warning fires at any chi on
    any of D=2, 3 or 4.
    """
    state, _lambdas = _shipped_su_run(D=2, seed=0, steps=600)

    chi_coarse, chi_fine = 8, 16
    coarse = _energy_of(state, chi=chi_coarse)
    fine = _energy_of(state, chi=chi_fine)
    assert fine == pytest.approx(_SU_REFERENCE[2], abs=1e-3), (
        f"the D=2 reference reads {fine:.6f}, not {_SU_REFERENCE[2]} -- either "
        f"this apparatus or the shipped engine has moved, and nothing else in "
        f"this section means anything until that is resolved"
    )
    # The two chi are named from the variables rather than written into the
    # string: a message that hard-codes "chi=8" keeps saying so under a
    # mutation that changed it, which is how a guard comes to report a number
    # it did not measure.
    assert abs(fine - coarse) < 1e-6, (
        f"E(chi={chi_coarse})={coarse:.9f} and E(chi={chi_fine})={fine:.9f} "
        f"differ by {abs(fine - coarse):.2e}: the environment has not settled, "
        f"so this is a truncation error and not yet an energy"
    )

    # The collapsed-recipe control, driven **through** ``_energy_of``.  Until
    # #882's final review this called ``ctm_tensor_2site(recipe="1x1")``
    # directly and asserted ``rank(C1) == 1`` on its own -- which watched the
    # recipe, not the guard: ``_energy_of`` hard-coded ``recipe="2x2"``, so its
    # ``assert rank > 1`` had never been seen firing anywhere in either file.
    # What actually stood between this file and -0.648904 was a string literal.
    # ``_energy_of`` now takes ``recipe`` for exactly this call.
    with pytest.raises(AssertionError, match=r"corner C1 has rank 1 at chi="):
        _energy_of(state, chi=chi_fine, recipe="1x1")

    # And the corner really is rank 1 -- i.e. the refusal above is provoked by
    # the collapse and not by something else the '1x1' recipe does.
    env_A, _env_B = ctm_tensor_2site(
        state.A, state.B, chi=chi_fine, max_iter=200, conv_tol=1e-10, recipe="1x1"
    )
    sv = np.linalg.svd(np.asarray(env_A.C1.todense()), compute_uv=False)
    assert int(np.sum(sv > sv[0] * 1e-10)) == 1, (
        "recipe='1x1' no longer collapses to a rank-1 corner, so _energy_of's "
        "rank guard has lost the mutation that demonstrates it can fire "
        "(#723/#726/#747)"
    )


def test_the_shipped_engines_stored_spectra_are_closer_than_the_plan_says():
    """C-1, resolved by measurement -- and the measurement contradicts the plan.

    **This cell measures ``ipeps_simple_update._simple_update_checkerboard_sweep``
    -- the engine being *replaced* -- and nothing in ``ipeps_su``.**  Measured
    rather than reasoned: with ``_su_step``, ``_su_evolve``,
    ``_align_gate_to_ket``, ``_sqrt_and_inv_sqrt``, ``_require_uniform_bonds``,
    ``_bond_dims`` and ``_reorder`` all replaced by functions that raise, in
    both namespaces, this cell **passes** in 2.8 s.  The only ``ipeps_su`` code
    it reaches is ``_SUState.from_pair``'s label check.  So no mutation of the
    module under test can kill it, and it must not be counted as coverage of
    that module; the name says which engine it is about for the same reason.
    It is here because C-1 is a question about the *old* engine's stored
    spectra, and that is where they exist.

    **The two lower bounds are properties of the shipped engine as it stands
    today, so they are anti-correlated with a future fix to it.**  ``1e-3 <
    worst_l2`` and ``0.10 < worst_elem`` fire if the shipped engine's stored
    spectra ever *become* the BP messages -- which is exactly what the
    four-independent-lambda D=4 path already does, at 0.00%.  They are kept,
    unmoved, because without them the comparison passes on a re-point of
    ``lambdas`` to ``bp`` and the cell goes silently vacuous.  If they fire, the
    claim they record has been settled: retire the cell, do not lower the bound.
    (Latent rather than live at this cell's own configuration: flipping
    ``independent_bonds=True`` here moves neither reading -- 2.2500% / 16.3057%
    both ways.)

    The plan's Step 1 asks the new engine's returned weights to match "the
    diagnostic the engine reported".  ``_su_evolve`` reports none: ``_SUState``
    has two fields and nowhere to put a spectrum, which is the design.  So the
    only available comparison on the new engine is ``gauge_fix``'s weights
    against themselves, which is vacuous in the dangerous direction -- it passes
    on ``return state``.  ``task-12-corrections.md`` C-1 therefore re-points the
    test at the **old** engine, whose stored spectra are a real object that can
    be wrong, and quotes the plan's claim that they miss the BP messages of
    their own output "by 15-35%".

    **They do not, under the natural metric.**  Measured here at D=2 and
    reproduced across D in 2, 3, 4 and seeds 0, 1, 2 in ``task-12-report.md``:
    the shipped default sits **2.25% to 2.38%** away in L2, normalised on the
    leading value -- small, not large.  On the four-independent-lambda path at
    D=4 it is **0.00%**: the stored spectra *are* the BP messages, bit for bit,
    which refutes the premise outright for that configuration.

    Where the plan's number comes from is the *element-wise* maximum relative
    deviation, which reads **16.3%** on this same D=2 data and 37-38% at D=3 and
    D=4.  That metric is dominated by the smallest singular value and it is not
    robust: on the one cell that reaches the plan's range in L2 (D=3, four
    independent lambdas, 800 steps, 37.52%) the BP solve did **not converge**,
    and the element-wise reading of that same cell is 2023%.  ``task-12-
    corrections.md`` C-5 warns about exactly this -- "a Frobenius-vs-max-abs
    mismatch produced a wrong tolerance figure" -- and it happened again.

    Both metrics are pinned below, so the file records which number belongs to
    which reading and neither can be quoted as the other.  Watched failing: with
    the BP weights compared against themselves, both readings go to 0.0 and the
    lower bounds fire.

    This is the whole of C-1.  There is no matching number for the new engine,
    and "unrepresentable by construction" is the honest answer rather than a
    gap: ``test_su_state_has_no_lambda_fields`` is that property as an
    assertion, and it is the only form the property can take once the field is
    gone.
    """
    state, lambdas = _shipped_su_run(D=2, seed=0, steps=800)
    _A, _B, bp, info = gauge_fix(state.A, state.B, tol=1e-10)
    assert info.converged, (
        f"BP did not converge on the shipped engine's own output "
        f"({info.iterations} sweeps, residual {info.residual:.3e}); a "
        f"discrepancy measured against a failed solve is #870's shape and is "
        f"not evidence of anything"
    )

    l2, elementwise = {}, {}
    for field in BondWeights._fields:
        stored = np.asarray(getattr(lambdas, field), dtype=float)
        message = np.asarray(getattr(bp, field), dtype=float)
        stored, message = stored / stored[0], message / message[0]
        l2[field] = float(np.linalg.norm(stored - message) / np.linalg.norm(message))
        elementwise[field] = float(np.max(np.abs(stored - message) / message))

    worst_l2, worst_elem = max(l2.values()), max(elementwise.values())
    assert 1e-3 < worst_l2 < 0.05, (
        f"the shipped engine's stored spectra sit {worst_l2 * 100:.2f}% from "
        f"the BP messages of its own output in L2 (per bond: "
        f"{ {k: round(v * 100, 2) for k, v in l2.items()} }).  The plan claims "
        f"15-35%; this measurement says the premise is metric-dependent."
    )
    assert 0.10 < worst_elem < 0.50, (
        f"the element-wise maximum relative deviation is "
        f"{worst_elem * 100:.2f}% (per bond: "
        f"{ {k: round(v * 100, 2) for k, v in elementwise.items()} }) -- this "
        f"is the reading the plan's 15-35% belongs to, and pinning it here is "
        f"what stops the two numbers being quoted for each other"
    )
    # There was an ``assert worst_elem > 3.0 * worst_l2`` here, added to give
    # C-5 a reading that is not a band around a measured constant.  **It is
    # deleted rather than kept, because it cannot fail on what it claims to
    # watch**, and #882's re-review measured why: at ``D=2`` each spectrum has
    # two entries, so the ratio reduces *exactly* to ``sqrt(1 + m**2) / m > 3``
    # -- i.e. ``m < 0.354`` -- where ``m`` is the second value of the **BP
    # message** spectrum.  The stored spectra this cell exists to measure drop
    # out of it algebraically: the ratio stayed 6.212 under three different
    # perturbations of ``lambdas``.  It was unreachable as well as inert, since
    # ``0.10 < worst_elem`` two assertions above fires first in both scenarios
    # the deleted text named.
    #
    # The observation it was trying to make is true and worth keeping as an
    # observation: computing ``elementwise`` with the L2 formula collapses the
    # ratio to 1, and on this cell's own data the element-wise reading is ~7x
    # the L2 one.  What stops the two numbers being quoted for each other is
    # the pair of bands above, which are disjoint (``< 0.05`` against
    # ``> 0.10``) and each pinned to its own measured constant.


@pytest.mark.parametrize(
    "seed",
    [0] + [pytest.param(s, marks=pytest.mark.slow) for s in _SEEDS[1:]],
)
def test_d2_reaches_the_heisenberg_energy_not_the_product_state(seed):
    """#667's guard: D=2 must reach ~-0.659, not -0.5.

    -0.5 is the product state, which is what the ``lambda**1.5`` bond made the
    fixed point.  Assert on ENERGY, never on norm: ``gauge_fix`` rescales its
    input by max-abs on the way in to the next step, so ``||A||`` reads a
    healthy 1.0 right up to the step where it is exactly 0.

    **This passes on all three seeds now, and it failed on all three in three
    different ways** before ``_su_step``'s truncation basis was corrected --
    which is why the seed axis is here and not a formality:

    ======  ==========================================================
    seed 0  was -0.576225, and not settling: -0.512783 at 400 steps,
            -0.576225 at 800, back **up** to -0.540608 at 1200
    seed 1  the pair was exactly **zero** -- ``_energy_of`` reported it
            rather than returning a number, and a norm check would not
            have (#878)
    seed 2  exactly -0.500000, the product state, to fifteen digits
    ======  ==========================================================

    A one-seed run would have reported any one of those three as *the* symptom.
    The cause was one thing and it was not seed-dependent: see
    ``test_su_step_truncates_in_the_state_s_own_basis``.  With that step's
    truncation basis corrected the three seeds read **-0.658880 at 400, 800,
    1200 and 2000 steps alike**, agreeing with each other to 1e-15 and with the
    reference to 4.2e-04 -- which is a stronger statement than this test's
    ``abs=0.02`` makes, and the tolerance is deliberately left where Task 12 set
    it rather than tightened onto one measurement of one engine.
    """
    state = _random_state(2, seed)
    state = _su_evolve(state, _su_heisenberg_gate(state), 2, 800)
    E = _energy_of(state, _CHI[2])
    assert E < -0.60, (
        f"seed {seed}: E={E:.6f} -- at or above the product state "
        f"({_PRODUCT_STATE_ENERGY}), which is what #667 looked like"
    )
    assert E == pytest.approx(_SU_REFERENCE[2], abs=0.02), (
        f"seed {seed}: E={E:.6f}, reference {_SU_REFERENCE[2]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("D", [3, 4])
def test_su_evolve_reaches_the_simple_update_reference_energy(D, seed):
    """The D axis of the sweep: D=3 and D=4 against this project's references.

    Both axes are needed and neither alone discriminates.  D=3 is where #869
    stayed ambiguous for months, and D=4 is where the shipped four-lambda
    baseline diverges on every seed -- so a D=3 single-seed run can pass on a
    broken implementation and a D=4 single-seed run cannot tell "always broken"
    from "unlucky".

    ``steps`` is 1600, not the plan's 1200, and that is measured rather than
    cautious: the shipped engine needs 1600 steps at ``dt=0.05`` to reach
    -0.667071 at D=4 (against the -0.6674 reference), and reporting a D=4
    shortfall from an under-converged run is the failure mode #882 section 6.3
    warns about.  1600 is used at D=3 too so the two cells differ in one
    variable.

    **Two of the six pass now** -- D=3 seeds 1 and 2 -- and **four still fail,
    on purpose**: D=3 seed 0 and all three D=4 seeds.  Both residues are named
    in the section header, neither is explained, and neither is in scope for the
    truncation-basis fix that made the other two pass.  ``task-10-reopen-report.md``
    §"what did not close" is the standing record; do not tune ``steps``,
    ``_SU_DT`` or ``_CHI`` to close them, because a cell that goes green without
    a mechanism is more likely a new bug than a fix.

    The 1600-step energies are not quoted here, because a docstring number that
    was not taken at the step count the test runs is exactly the kind of figure
    this task exists to stop being repeated; ``task-10-reopen-report.md`` has
    the re-run grid at 400, 800, 1200 and 2000 steps, where D=3 seeds 1 and 2
    sit at -0.662839, D=3 seed 0 decays -0.651785 -> -0.607822 with BP failing
    on 1782 of 2000 steps, and every D=4 cell is at -0.500000 from 800 steps
    onward with BP converging throughout.
    """
    state = _random_state(D, seed)
    state = _su_evolve(state, _su_heisenberg_gate(state), D, 1600)
    E = _energy_of(state, _CHI[D])
    assert E < -0.60, (
        f"D={D} seed={seed}: E={E:.6f} -- at or above the product state "
        f"({_PRODUCT_STATE_ENERGY})"
    )
    assert E == pytest.approx(_SU_REFERENCE[D], abs=0.01), (
        f"D={D} seed={seed}: E={E:.6f}, reference {_SU_REFERENCE[D]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("D", [2, 3])
def test_the_energy_does_not_drift_away_with_more_steps(D, seed):
    """C-2: pin the **trend**, because one step count reports either verdict.

    Dense four-lambda simple update is non-monotonic in step count -- #869's D=3
    run peaks near -0.654 around 800 steps and falls to -0.477 by 4000, from the
    same code -- so a single-step-count energy assertion is not evidence of
    convergence even when it passes.  What convergence looks like is an energy
    that stops moving and *stays* stopped.

    Chained in 400-step blocks, which is a multiple of the four-bond cycle, so
    the bond ordering is identical to one 1200-step call and only the
    measurement points are added.  (``_su_evolve`` restarts its cycle at
    ``h_AB`` on every call, so a block that was not a multiple of 4 would
    silently change the Trotter ordering.)

    Measured, this is the guard that separated the two engines most cleanly.
    Before the truncation-basis fix, D=2 seed 0 ran -0.512783, -0.576225,
    -0.540608 -- it came back *up* by 3.6e-02 and the second assertion fired --
    and D=3 seed 1 ran -0.536106, -0.514482, -0.604504 and fired on the first.
    With ``_su_step``'s truncation basis corrected the same three points read
    -0.658880, -0.658880, -0.658880 at D=2 on every seed, flat to 1e-06, and D=3
    seeds 1 and 2 read -0.662838, -0.662839, -0.662839 against a -0.6632
    reference.  **Five of the six cells pass; D=3 seed 0 still fires**, on the
    first assertion, running -0.651785, -0.643480, -0.622322 -- uphill, and the
    only cell of the six where the internal BP stops converging.  That residue is
    out of scope and is meant to stay visible.

    **This guard is necessary and it is not sufficient, which is measured rather
    than argued, and it is emphatically not an acceptance criterion.**  On the
    pre-fix engine both ``seed 2`` cells *passed* it: at D=2 and at D=3 that seed
    settled on exactly -0.500000 and stayed there, so the trend was flat, pinned,
    and completely wrong.  A state stuck at the product state is perfectly
    settled, and this reading calls that convergence.  That is why it sits
    beside ``test_d2_reaches_the_heisenberg_energy_not_the_product_state`` and
    ``test_su_evolve_reaches_the_simple_update_reference_energy`` rather than in
    place of them -- convergence and correctness are two questions and neither
    reading answers the other.  Do not quote a pass here as evidence that the
    engine is right.
    """
    state = _random_state(D, seed)
    gate = _su_heisenberg_gate(state)
    energies, done = {}, 0
    for target in (400, 800, 1200):
        state = _su_evolve(state, gate, D, target - done)
        done = target
        energies[target] = _energy_of(state, _CHI[D])

    trail = ", ".join(f"{k}: {v:.6f}" for k, v in energies.items())
    assert energies[800] <= energies[400] + 1e-3, (
        f"D={D} seed={seed}: the energy rose between 400 and 800 steps "
        f"({trail}) -- imaginary time does not go uphill on a converging run"
    )
    assert energies[1200] <= energies[800] + 1e-3, (
        f"D={D} seed={seed}: the energy rose between 800 and 1200 steps ({trail})"
    )
    assert abs(energies[1200] - energies[800]) < 5e-3, (
        f"D={D} seed={seed}: the energy is still moving at 1200 steps "
        f"({trail}); a value that has not stopped moving is not an answer, "
        f"whatever it happens to equal"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
def test_d3_actually_uses_its_third_bond_direction(seed):
    """A nominally-D=3 state whose ``lam_3`` is 2e-6 is really D=2.

    That trap is why pre-#667 D=3 results from ``ipeps()`` were meaningless --
    the third direction was there in the shape and absent from the state.

    **A pass here is not evidence that the state is right, and that is measured
    rather than hedged.** On the pre-fix engine all three seeds cleared the rank
    reading comfortably (``lam_3/lam_1`` from 4.6e-03 to 7.5e-01) while seed 2's
    state had an energy of exactly -0.500000: *the product state* with a
    full-rank bond spectrum. A tensor ``|up> (x) M`` has a product physical state
    and an arbitrarily entangled virtual structure, so this reading is blind to
    it by construction. **This test is not an acceptance criterion** -- the
    energy guards above are. It is kept because it is cheap and because it is the
    only reading that catches #667's specific symptom, which the energy guards
    would report only as a number that is too high.

    **Seed 0 fails here now, on its precondition rather than on the rank**, and
    it is the same D=3-seed-0 residue the energy guards report: after 1200 steps
    BP no longer converges on that pair (100 sweeps, residual 7.2e-02 against
    ``tol=1e-10``), so there is no fixed-point spectrum to read and the test says
    so instead of reading one anyway. That is the assertion behaving as designed
    -- #870 is the standing reason not to trust a spectrum from a failed solve --
    and it is out of scope here; see the section header.

    The control below is what keeps it from being an assertion that cannot
    fail: a pair whose third virtual direction is scaled by 1e-06 is a genuinely
    D=2 state wearing a D=3 shape, and the same reading must reject it.
    """
    state = _random_state(3, seed)
    state = _su_evolve(state, _su_heisenberg_gate(state), 3, 1200)
    _A, _B, w, info = gauge_fix(state.A, state.B, tol=1e-10)
    assert info.converged, (
        f"seed {seed}: BP did not converge on the evolved pair "
        f"({info.iterations} sweeps, residual {info.residual:.3e}), so the "
        f"spectrum below is not the state's"
    )
    for field in BondWeights._fields:
        lam = np.asarray(getattr(w, field), dtype=float)
        assert lam[-1] / lam[0] > 1e-3, (
            f"seed {seed} bond {field}: lam_3/lam_1 = {lam[-1] / lam[0]:.2e} -- "
            f"this is a D=2 state in a D=3 shape (#667)"
        )

    # The control: a pair that really is rank-deficient must be rejected here.
    A, B = _PAIRS["dense"](D=3, seed=seed)
    starved = jnp.asarray([1.0, 1.0, 1e-6])
    for leg in ("u", "d", "l", "r"):
        A, B = scale_bond_axis(A, leg, starved), scale_bond_axis(B, leg, starved)
    _a, _b, w_starved, info_starved = gauge_fix(A, B, tol=1e-10)
    # Checked, four lines after the main path asserts the same thing "because
    # #870 is the standing reason not to trust a spectrum from a failed solve".
    # It was unpacked and discarded before #882's final review.  Latent, not
    # live: the starved solve converges on all three seeds (40/34/69 sweeps,
    # residual 6.2e-11 / 5.7e-11 / 9.9e-11 against tol=1e-10).
    assert info_starved.converged, (
        f"seed {seed}: BP did not converge on the starved control pair "
        f"({info_starved.iterations} sweeps, residual {info_starved.residual:.3e}), "
        f"so the spectrum this control rejects is not a fixed point and the "
        f"rejection is not evidence that the reading above has teeth"
    )
    ratios = [
        float(
            np.asarray(getattr(w_starved, f))[-1] / np.asarray(getattr(w_starved, f))[0]
        )
        for f in BondWeights._fields
    ]
    assert min(ratios) <= 1e-3, (
        f"a pair whose third virtual direction was scaled by 1e-06 still reads "
        f"lam_3/lam_1 = {min(ratios):.2e} -- this reading cannot see a "
        f"rank-deficient bond and the assertion above is decoration"
    )
