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
  3.1e-02 to 4.1e-02 where a correct one scores 8.8e-15 to 9.5e-15.
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

import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_gauge_helpers import (  # tests/ is on sys.path
    _PAIRS,
    _chain_pair,
    _chain_pair_as_peps,
    assert_leg_split,
)

import tenax.algorithms.ipeps_su as ipeps_su_module
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_bp_gauge import BondWeights, BPGaugeInfo
from tenax.algorithms.ipeps_gauge import gauge_fix, torus_2x2_sign_free
from tenax.algorithms.ipeps_simple_update import _make_trotter_gate_tensor
from tenax.algorithms.ipeps_su import (
    _BOND_ENDS,
    _align_gate_to_ket,
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

    ``phys`` stays out of it on both: it is 2, so a reading that included it
    would agree here, which is why the assertion is on the leg *set* rather
    than only on the number.
    """
    A, B = _PAIRS[kind](D=3)
    assert _SUState.from_pair(A, B).max_D == 3

    a, b, _vl, _vr = _chain_pair()
    Ac, Bc = _chain_pair_as_peps(a, b)
    swap = {"u": "l", "l": "u", "d": "r", "r": "d"}
    As, Bs = Ac.relabels(swap), Bc.relabels(swap)

    for name, pair, expected in (
        ("chain", (Ac, Bc), {"u": 1, "d": 1, "l": 4, "r": 4}),
        ("swapped", (As, Bs), {"u": 4, "d": 4, "l": 1, "r": 1}),
    ):
        t = pair[0]
        dims = {
            lab: t.indices[t.labels().index(lab)].dim for lab in ("u", "d", "l", "r")
        }
        assert dims == expected, f"{name}: {dims}"
        assert t.indices[t.labels().index("phys")].dim == 2
        assert _SUState.from_pair(*pair).max_D == 4, name


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
#: pin bites: it keeps 0.43903 where the top three are 1, 0.9915, 0.46407.
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

    ``BondWeights.ones(D_h, D_v)`` gives both horizontal bonds one dimension
    and both vertical bonds another, which is right only while all four agree.
    A single ``_su_step`` truncates one bond and leaves the other three alone,
    so mid-cycle they do not.  (``gauge_fix`` reads its dimensions the same way
    and therefore cannot be run on such a pair at all -- see ``_su_step``'s
    Note; that is why the untruncated cases below never re-gauge.)
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


def _gram(t, leg):
    """``<t|t>`` traced over every leg but ``leg``, as a dense ``(leg, leg)`` matrix.

    This is the split guard's whole content.  Write the step's two outputs as
    ``F_j = U sqrt(sigma)`` and ``F_i = sqrt(sigma) Vh`` with ``U`` and ``Vh``
    the SVD's isometries; then ``F_j^dag F_j`` and ``F_i F_i^dag`` are both
    ``diag(sigma)`` -- the *same* matrix at both ends, diagonal, and equal to
    the bond's spectrum.  Put the whole weight on one end and they become
    ``diag(sigma**2)`` and the identity.

    Unlike an element-wise comparison of the tensors, this is invariant under
    the sign and basis freedom the SVD and BP's ``eigh`` each carry, which is
    what makes it usable at all.  Measured, ``gauge_fix`` applied to a stepped
    pair comes back 1.7e-01 (dense) / 5.7e-03 (symmetric) away from it
    element-wise -- in ``max ||x| - |y||``, so with the sign freedom already
    quotiented out -- even for a ``dt=0`` gate on an already-gauged input,
    because the two routines pick different bases on the bond.  Both are the
    same state; only a basis-free reading can say so.  That is the measurement
    behind not comparing the stepped pair to a re-derived ``gauge_fix``
    spectrum leg by leg.

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


def _two_site_tensor(state, bond):
    """The two sites sharing ``bond``, contracted across it.

    This is the ``theta`` an ``_su_step`` on ``bond`` produced: ``F_j F_i``,
    which is ``U sqrt(sigma) sqrt(sigma) Vh = U diag(sigma) Vh`` however the
    ``sqrt`` was shared out between the two factors.  Two states stepped from
    the *same* input at the same ``dt`` give tensors on identical outer legs --
    the internal ``gauge_fix`` is deterministic and the truncation touches only
    the bond -- so they are directly comparable, element for element.
    """
    (site_i, ren_i), (site_j, ren_j), _left, _right = _theta_labels(bond)
    pair = {"A": state.A, "B": state.B}
    return contract(pair[site_j].relabels(ren_j), pair[site_i].relabels(ren_i))


def _bond_spectrum(state, bond, max_D, base_charges=None):
    """The Schmidt spectrum of ``state`` across ``bond``, independent of the split.

    Reassembles the two-site tensor and takes its singular values.  Because
    ``F_j F_i`` is the same object however the ``sqrt`` was shared out between
    the two factors, this number is the same for a correct split, a one-sided
    one, and anything in between -- which is exactly what makes it a valid
    reference for the split check rather than a restatement of it.

    ``base_charges`` is exposed so a test can build the **pinned** truncation of
    the same ``theta`` and show that this configuration can tell the two apart
    (#865).  It is ``None`` for every reading of the step itself.
    """
    _ends_i, _ends_j, left, right = _theta_labels(bond)
    _U, sigma, _Vh, _full = truncated_svd(
        _two_site_tensor(state, bond),
        left_labels=left,
        right_labels=right,
        new_bond_label="__reference_bond",
        max_singular_values=max_D,
        base_charges=base_charges,
    )
    return np.asarray(sigma)


def _theta_cosine(state_a, state_b, bond):
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
    """
    ta = _two_site_tensor(state_a, bond)
    tb = _two_site_tensor(state_b, bond)
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

    class _Cache:
        def pair(self, kind):
            if kind not in pairs:
                pairs[kind] = _PAIRS[kind](D=D)
            return pairs[kind]

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

    The convergence assertion left behind has teeth, and has been watched to
    use them: with the gate's flows left unaligned, this fires on symmetric
    ``v_BA`` at ``100 sweeps, residual 1.399e-01``.  What it guards is that the
    step's output is still something BP can solve -- the property the *next*
    step depends on, since the gauge sets the basis its truncation is taken in.
    ``_su_step`` now warns on the same condition (see its *Warns*), so this
    pins the warning's trigger as well.

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
    step scores 8.8e-15 to 9.5e-15, an ``_su_step`` that returns ``state``
    untouched scores 3.1e-02 to 4.1e-02, and a ``_BOND_ENDS`` with two of its
    rows transposed -- or with the gate's ``si``/``sj`` legs crossed -- scores
    1.9e-02.  Several of the plan's Phase 2 tests pass on a do-nothing
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
    :func:`_gram`): both must be diagonal, equal to each other, and equal to
    the bond's Schmidt spectrum, which :func:`_bond_spectrum` derives from the
    *reassembled* two-site tensor and therefore independently of how the split
    was made.

    The spectrum is checked to be spread before it is used, because a flat
    ``sigma`` at 1 satisfies ``diag(sigma) == diag(sigma**2)`` and the guard
    would then have no teeth on the very mutation it exists for.
    """
    stepped = su.step(kind, bond)
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]
    pair = {"A": stepped.A, "B": stepped.B}
    G_i = _gram(pair[site_i], leg_i)
    G_j = _gram(pair[site_j], leg_j)
    sigma = _bond_spectrum(stepped, bond, D)

    spread = float(np.max(sigma) / np.min(sigma))
    assert spread > 1.1, (
        f"{kind} {bond}: sigma is flat ({spread:.3f}), so this test cannot "
        f"distinguish sqrt(sigma) at both ends from sigma at one"
    )

    for name, G in (("i", G_i), ("j", G_j)):
        off = float(np.max(np.abs(G - np.diag(np.diag(G)))))
        assert off < 1e-11, (
            f"{kind} {bond}: end {name}'s bond Gram matrix is not diagonal "
            f"(max off-diagonal {off:.3e}) -- the SVD factor is not an "
            f"isometry, so what sits on the bond is not a spectrum"
        )
    gap = float(np.max(np.abs(G_i - G_j)))
    assert gap < 1e-11, (
        f"{kind} {bond}: the two ends of the bond carry different weights "
        f"(max |G_i - G_j| = {gap:.3e}).  sqrt(sigma) must go into BOTH "
        f"factors; putting it on one end leaves the same physical state, so no "
        f"closed-loop probe in this tree would notice."
    )
    err = float(np.max(np.abs(np.sort(np.diag(G_i)) - np.sort(sigma))))
    assert err < 1e-11, (
        f"{kind} {bond}: each end carries {np.sort(np.diag(G_i))} where the "
        f"bond's own spectrum is {np.sort(sigma)} (max diff {err:.3e}) -- the "
        f"weight is on the bond at the wrong power"
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

    Measured here (``dense``, ``h_AB``, ``D=3``, ``dt=0.05``; both guards gate
    at 1e-11):

    ============  =================  ==================
    mutation      state (torus)      split (|Gi - Gj|)
    ============  =================  ==================
    correct step  8.8e-15            4.9e-15
    one_sided     2.6e-16  *blind*   1.1e+01  fires
    squared       1.7e-01  fires     2.0e-14  *blind*
    ============  =================  ==================
    """
    bond = "h_AB"
    stepped = su.step("dense", bond)
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]
    pair = {"A": stepped.A, "B": stepped.B}
    root = jnp.asarray(np.sqrt(_bond_spectrum(stepped, bond, D)))

    mutants = {
        # all of sigma on end j and none on end i: a re-split, not a new state
        "one_sided": {
            site_j: scale_bond_axis(pair[site_j], leg_j, root),
            site_i: scale_bond_axis(pair[site_i], leg_i, 1.0 / root),
        },
        # sigma on each end instead of sqrt(sigma): the bond carries sigma**2
        "squared": {
            site_j: scale_bond_axis(pair[site_j], leg_j, root),
            site_i: scale_bond_axis(pair[site_i], leg_i, root),
        },
    }

    # The one-sided mutant really is the leg-wise rescale it is claimed to be,
    # so the test's premise is pinned rather than assumed.  ``assert_leg_split``
    # is the tool Phase 1 built for exactly this comparison: element-wise,
    # against a bond map written out independently.
    site = pair[site_j]
    scale = {
        lg: np.ones(site.indices[site.labels().index(lg)].dim)
        for lg in ("u", "d", "l", "r")
    }
    scale[leg_j] = np.asarray(root)
    assert_leg_split(
        site_j, pair[site_j], mutants["one_sided"][site_j], scale, 1e-12, msg="mutant "
    )

    ref = torus_2x2_sign_free(stepped.A, stepped.B, _ones_for(stepped.A))
    # The unmutated baseline both readings are compared against below.  It
    # restates the split guard's first assertion on purpose: the two mutants
    # are derived from *this* pair, so if it were already split-asymmetric the
    # "one_sided fires / squared is blind" contrast would mean nothing.
    baseline_split = float(
        np.max(np.abs(_gram(pair[site_i], leg_i) - _gram(pair[site_j], leg_j)))
    )
    assert baseline_split < 1e-11, (
        f"the unmutated step this test mutates is already split-asymmetric "
        f"({baseline_split:.3e}); the contrast below would be meaningless"
    )

    seen = {}
    for name, m in mutants.items():
        got = torus_2x2_sign_free(m["A"], m["B"], _ones_for(m["A"]))
        seen[name] = (
            _torus_rel(got, ref),
            float(np.max(np.abs(_gram(m[site_i], leg_i) - _gram(m[site_j], leg_j)))),
        )

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

    This is also the only executable coverage of ``base_charges=None``, brief
    constraint #3 and one of the four bugs the rewrite exists to delete (#865).
    Pinning the new bond's per-sector keep counts to the old bond's layout
    stops the SVD taking the globally largest values; here it keeps 0.43903
    where the top three are 1, 0.9915, 0.46407, a relative error of 2.504e-02
    against a guard at 1e-11.  It shows up **only** on the symmetric arm and
    **only** at a ``dt`` where truncation bites -- see ``_TRUNCATION_DT``.
    """
    A, B = su.pair(kind)
    full = su.step(kind, bond, max_D=_FULL_RANK, dt=_TRUNCATION_DT)
    kept = su.step(kind, bond, max_D=D, dt=_TRUNCATION_DT)
    (site_i, leg_i), (site_j, leg_j) = _BOND_ENDS[bond]

    sigma_full = np.sort(_bond_spectrum(full, bond, _FULL_RANK))[::-1]
    sigma_kept = np.sort(_bond_spectrum(kept, bond, D))[::-1]
    dropped = 1.0 - float(np.sum(sigma_kept**2) / np.sum(sigma_full**2))

    # --- the two meta-assertions: this cell can see both failure modes -------
    assert dropped > 0.05, (
        f"{kind} {bond}: the truncation discards only {dropped:.4f} of the "
        f"weight at dt={_TRUNCATION_DT}, so this test is not exercising a real "
        f"truncation and cannot see a wrong keep set"
    )
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
        sigma_pinned = np.sort(_bond_spectrum(full, bond, D, base_charges=pin))[::-1]
        sep = float(np.max(np.abs(sigma_pinned - sigma_kept)) / sigma_full[0])
        assert sep > 1e-3, (
            f"{kind} {bond}: pinning base_charges to {list(pin)} keeps the "
            f"same spectrum as not pinning it (separation {sep:.3e}), so this "
            f"cell cannot see #865 and the assertion below is not covering it. "
            f"dt={_TRUNCATION_DT} was chosen to make it visible; re-measure "
            f"before changing it."
        )
    else:
        # ``linalg.svd`` documents base_charges as "Ignored on the dense path",
        # so there is no pin to separate from here and no meta-assertion to
        # make.  Measured: separation 4.8e-16 (h_AB), i.e. exactly nothing.
        # This is why _TRUNCATION_CASES carries a symmetric cell at all.
        pass

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
    cos = _theta_cosine(kept, full, bond)
    expected = float(np.sqrt(1.0 - dropped))
    gap = abs(cos - expected)
    assert gap < 1e-11, (
        f"{kind} {bond}: the truncated step overlaps the untruncated one by "
        f"{cos:.6f} where a projection onto the top-{D} subspace must give "
        f"sqrt(1 - dropped) = {expected:.6f} (gap {gap:.3e}).  The kept "
        f"singular values are attached to the wrong singular vectors: the "
        f"spectrum is right and the retained subspace is not."
    )


def test_su_step_warns_when_the_gauge_did_not_converge(su, monkeypatch):
    """A failed BP solve is surfaced, not swallowed.

    ``_su_step`` cannot fail on this and must not: the state is still correct,
    only the *basis the truncation is taken in* is worse than it should be.
    That is precisely the shape of defect this project keeps being bitten by --
    #870's residual fell monotonically to 9.8e-12 while certifying a state that
    had gone to zero -- so a silent degradation on every step of a 100-step
    run is not acceptable either.

    The trigger is real: with the gate's flows unaligned, BP hits
    ``max_iter=100`` at residual 1.399e-01 on a symmetric ``v_BA`` step.  It is
    reproduced here by monkeypatching ``gauge_fix`` rather than by constructing
    such a pair, because the cheapest genuine one costs 38 s and the thing
    under test is the *reporting*, not BP.
    """
    A, B = su.pair("dense")
    real = ipeps_su_module.gauge_fix

    def _unconverged(*args, **kwargs):
        A2, B2, w2, _info = real(*args, **kwargs)
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
