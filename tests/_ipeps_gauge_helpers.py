"""Shared probes for BP-gauge tests: a small random checkerboard pair and the
exactly gauge-invariant closed 2x2 torus contraction.

Split out of ``test_ipeps_bp_gauge.py`` (#882) so ``test_ipeps_gauge.py`` can
reuse them without one test module importing another -- ``_su_fixtures.py``
and ``_split_ctm_oracle.py`` are the existing precedent for a shared,
non-``test_``-prefixed helper module living in ``tests/``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_u1sz_init_pair
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.mps import FiniteMPS
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

D = 3


def _dense_pair(D: int = D, seed: int = 0):
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A = _wrap_as_dense_tensor(jax.random.normal(kA, (D, D, D, D, 2)))
    B = _wrap_as_dense_tensor(jax.random.normal(kB, (D, D, D, D, 2)))
    return A * (1.0 / float(A.norm())), B * (1.0 / float(B.norm()))


def _symmetric_pair(D: int = D, seed: int = 0):
    return heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))


_PAIRS = {"dense": _dense_pair, "symmetric": _symmetric_pair}


def _fermionic_pair(D: int = 2, seed: int = 0):
    """A random ``FermionParity`` checkerboard pair, same legs and flows.

    **The default is D=2 while ``_dense_pair``/``_symmetric_pair`` default to
    D=3**, so pass ``D`` explicitly whenever two builders are iterated together
    -- both shipped call sites do.  Kept at 2 because every measured number in
    ``task-4-report.md`` was taken there and re-pointing it would invalidate
    them for no gain in this round.

    D=2 is not a general operating point for a *gauge-structure* question.
    ``_build_initial_fpeps_tensor`` builds every virtual leg from
    ``virt_charges = [i % 2 for i in range(D)]`` (``fermionic_ipeps.py:167``),
    so at D=2 each leg carries parity sectors ``{0: 1, 1: 1}``: every
    parity-preserving matrix on it is 1x1 per sector, hence diagonal, and a
    non-diagonal parity-preserving gauge cannot be expressed at all.  Anything
    asking whether a gauge is exact *in general* needs D >= 3 (this is why
    deferred Task 5 was re-pointed there; #882 §5.2).  The witness tests in
    ``test_ipeps_gauge.py`` use D=2 deliberately -- they gauge with an explicit
    diagonal matrix whose answer is known a priori, which is a claim about the
    *witness*, not about non-diagonal gauges.

    Deliberately **not** in ``_PAIRS``.  ``test_ipeps_bp_gauge.py``
    parametrises four tests over ``list(_PAIRS)``, all of which assert that
    ``bp_gauge_checkerboard`` is an exact gauge; adding a fermionic entry
    would silently extend those assertions to the graded case, which is
    exactly the open question (#882) that is deliberately left unanswered
    here.  Callers that want it must ask for it by name.
    """
    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _build_initial_fpeps_tensor,
    )

    cfg = FPEPSConfig(D=D)
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A = _build_initial_fpeps_tensor(cfg, kA)
    B = _build_initial_fpeps_tensor(cfg, kB)
    return A * (1.0 / float(A.norm())), B * (1.0 / float(B.norm()))


#: Pair builders the CTM witness runs on: dense plus one *block-sparse*
#: fermionic path.  The U(1)-Sz ``"symmetric"`` pair is left out on cost
#: grounds -- its CTM is the slowest of the three and it adds no coverage the
#: other two lack (dense covers the dense contraction path, fermionic covers
#: the block-sparse one and the graded one).  It was measured in scratch and
#: behaves like the other two.
#:
#: An earlier version of this comment said the fermionic entry also covers "the
#: Koszul-carrying ``fuse`` in the double layer".  It does not: ``fuse`` carries
#: no Koszul sign (``_fuse_indices_symmetric`` permutes blocks with a bare
#: ``jnp.transpose``; grep ``_koszul_sign`` -- only ``SymmetricTensor.transpose``
#: and the ``linalg`` decompositions call it).
#:
#: The two builders here have **different default D** -- 3 for ``_dense_pair``,
#: 2 for ``_fermionic_pair`` -- so iterating this dict without an explicit ``D``
#: silently compares two bond dimensions.  Both call sites in
#: ``test_ipeps_gauge.py`` pass ``D=2`` explicitly; keep it that way.
_WITNESS_PAIRS = {"dense": _dense_pair, "fermionic": _fermionic_pair}


def _udlrp(t):
    """``t`` as a plain array in ``(u, d, l, r, phys)`` order."""
    labels = t.labels()
    return np.asarray(t.todense()).transpose(
        [labels.index(lab) for lab in ("u", "d", "l", "r", "phys")]
    )


_LEG_AXIS = {"u": 0, "d": 1, "l": 2, "r": 3}

# Which bond sits on each leg of each site, independently re-derived from
# BondWeights' own docstring (ipeps_simple_update.py) --
# ``h_AB: A.r<->B.l``, ``h_BA: B.r<->A.l``, ``v_AB: A.d<->B.u``,
# ``v_BA: B.d<->A.u`` -- rather than imported from
# ``ipeps_bp_gauge._BOND_OF``.  If the implementation's map were ever wrong,
# importing its own map back into the test would echo the same mistake into
# the expectation; writing it out again from the physical definition means
# the two have to agree independently.
_INDEPENDENT_BOND_OF: dict[tuple[str, str], str] = {
    ("A", "u"): "v_BA",
    ("A", "d"): "v_AB",
    ("A", "l"): "h_BA",
    ("A", "r"): "h_AB",
    ("B", "u"): "v_AB",
    ("B", "d"): "v_BA",
    ("B", "l"): "h_AB",
    ("B", "r"): "h_BA",
}


def assert_leg_split(site, before, after, scale_of_leg, tol, msg=""):
    """Assert ``after`` is ``before`` scaled independently along each leg.

    ``_torus_2x2`` is a closed loop, so it cannot see an asymmetric split: a
    diagonal weight factors arbitrarily between the two ends of a bond
    without changing the total the torus sums over (``sqrt(lam)*sqrt(lam)``
    and ``lam*1`` agree on the shared index), so a routine that dumps the
    *whole* weight onto one site's legs and leaves the other at 1 reproduces
    the identical torus value -- and, because BP-gauging is insensitive to
    which valid gauge of the same state it starts from, would likely still
    reach the same fixed point too.  This compares dense ``(u,d,l,r,phys)``
    arrays directly instead, leg by leg, on a single tensor with no torus and
    no BP involved, so that failure mode cannot hide.

    Args:
        site:          Label used only in the assertion message.
        before, after: The tensor before and after the claimed scaling.
        scale_of_leg:  ``{"u": ..., "d": ..., "l": ..., "r": ...}``, the
                       factor expected on each leg (already at the intended
                       power, e.g. already ``sqrt(lambda)``).
        tol:           Max allowed elementwise abs difference.
        msg:           Prefix for the assertion message.
    """
    expected = _udlrp(before)
    for leg, axis in _LEG_AXIS.items():
        scale = np.asarray(scale_of_leg[leg])
        shape = [1, 1, 1, 1, 1]
        shape[axis] = scale.shape[0]
        expected = expected * scale.reshape(shape)
    got = _udlrp(after)
    d = float(np.max(np.abs(got - expected)))
    assert d < tol, (
        f"{msg}site {site}: leg-wise split check failed (max abs diff {d:.3e})"
    )


def _torus_2x2(A, B, weights):
    """The closed 2x2 checkerboard torus, physical legs open.

    Exactly gauge invariant, and the only probe here that covers all four bonds
    at once: every bond appears twice, each time with the gauge on one end and
    its inverse on the other, so a gauge that fails to cancel shows up.
    ``_two_site`` cannot do this -- it leaves three legs open per site, so a
    gauge on any of them survives uncancelled and the comparison is meaningless.

    Sites ``A(0,0) B(1,0) / B(0,1) A(1,1)``.  Each site carries the weight of
    the bond leaving its ``r`` and ``d`` legs, which places each of the four
    weights exactly twice and each edge's weight exactly once::

        i = A00.r-B10.l  h_AB     m = A00.d-B01.u  v_AB
        j = B10.r-A00.l  h_BA *   n = B01.d-A00.u  v_BA *
        k = B01.r-A11.l  h_BA     o = B10.d-A11.u  v_BA
        l = A11.r-B01.l  h_AB *   p = A11.d-B10.u  v_AB *

    (``*`` = wraps around the torus.)
    """
    a = _udlrp(A) * np.asarray(weights.v_AB)[None, :, None, None, None]
    a = a * np.asarray(weights.h_AB)[None, None, None, :, None]
    b = _udlrp(B) * np.asarray(weights.v_BA)[None, :, None, None, None]
    b = b * np.asarray(weights.h_BA)[None, None, None, :, None]
    #      u  d  l  r  phys
    # A00  n  m  j  i  w
    # B10  p  o  i  j  x
    # B01  m  n  l  k  y
    # A11  o  p  k  l  z
    return np.einsum("nmjiw,poijx,mnlky,opklz->wxyz", a, b, b, a, optimize=True)


# --- the 1D-chain anchor (#882 Task 6) ------------------------------------
#
# A chain is a tree, so BP is *exact* on it: its fixed-point bond weights are
# the Schmidt values, to machine precision.  That makes it the one geometry in
# this plan with an answer known a priori -- the loopy square lattice has no
# valid reference spectrum at all (#882 §6.3), so every other criterion is a
# self-consistency check.
#
# The subject and the reference have to be the *same* state, which is why both
# are built here from one pair of MPS site tensors ``a``, ``b``:
#
#   reference -- repeat them into a long finite MPS ``a b a b ...`` and read the
#                middle bonds off tenax's own ``compute_singular_values``;
#   subject   -- embed the same two as a checkerboard PEPS pair whose vertical
#                legs have dimension 1, and hand that to ``gauge_fix``.
#
# Comparing BP against a *different* finite chain's middle bond -- a fresh
# ``FiniteMPS.random(6, 2, 4)`` -- compares two unrelated states and can only
# be made to pass by loosening something.

#: The recorded draw.  ``d``/``chi`` are the sizing the task brief starts from;
#: only the seed was chosen, on the measurements in :func:`_chain_pair`.
_CHAIN_SEED = 10
_CHAIN_D = 2
_CHAIN_CHI = 4


def _chain_pair(seed: int = _CHAIN_SEED, d: int = _CHAIN_D, chi: int = _CHAIN_CHI):
    """Two random MPS site tensors ``(left, phys, right)``, plus boundary vectors.

    ``numpy``'s ``default_rng`` rather than ``jax.random`` -- unlike every other
    builder in this module -- deliberately: this draw is the anchor's ground
    truth, and NEP 19 guarantees the PCG64 stream is stable across numpy
    versions, which is the property a *recorded* seed needs.

    Seed 10 of the first twelve, picked on three measured properties (all
    ``d=2``, ``chi=4``; the scan is in ``task-6-report.md``):

    * The 2-site transfer matrix's subleading ratio ``|lam2/lam1|`` is 0.134,
      the smallest of the twelve.  That ratio is what sets how fast the finite
      chain's middle bond approaches the infinite chain's, hence the ``L`` the
      reference needs: 0.134 gets to the f64 floor by ``L = 80``, where seed
      0's 0.645 still drifts 6.9e-14 between ``L = 160`` and ``L = 200``.
    * Its two inequivalent bonds' spectra differ by 1.44e-01, the largest of
      the six whose spectra were computed.  The parity-swap discriminator in
      ``test_ipeps_gauge.py`` *is* that number, so a seed whose two bonds
      nearly agreed would make it vacuous.
    * Neither spectrum is degenerate (smallest within-bond gap 2.6e-02), so
      sorting is well defined and perturbing one entry is visible.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(chi, d, chi))
    b = rng.normal(size=(chi, d, chi))
    return (
        a / np.linalg.norm(a),
        b / np.linalg.norm(b),
        rng.normal(size=chi),
        rng.normal(size=chi),
    )


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    """A dense (all-charges-zero) U(1) index, the convention both builders use."""
    return TensorIndex.from_charges(
        U1Symmetry(), np.zeros(dim, dtype=np.int32), flow, label=label
    )


def _chain_site_labels(i: int) -> tuple[str, str, str]:
    """``(left, phys, right)`` leg labels of ``FiniteMPS``'s site ``i``.

    Copied from ``mps._build_random_dense_tensors``: the canonicalization
    sweeps address bonds by the names ``v{i-1}_{i}`` / ``v{i}_{i+1}``
    (``v_-1_0`` at the left edge), so these are not free.  Shared by the dense
    and the block-sparse arm of the anchor, which differ in how a site is
    *built* but not in what its legs are called.
    """
    return (
        "v_-1_0" if i == 0 else f"v{i - 1}_{i}",
        f"p{i}",
        f"v{i}_{i + 1}",
    )


def _chain_mps_site(arr: np.ndarray, i: int) -> DenseTensor:
    """``(left, phys, right)`` array as ``FiniteMPS``'s site ``i``."""
    left, phys, right = _chain_site_labels(i)
    return DenseTensor(
        jnp.asarray(arr),
        (
            _trivial_index(arr.shape[0], FlowDirection.IN, left),
            _trivial_index(arr.shape[1], FlowDirection.IN, phys),
            _trivial_index(arr.shape[2], FlowDirection.OUT, right),
        ),
    )


def _middle_bond_pair(tensors: list) -> tuple[np.ndarray, np.ndarray]:
    """The middle ``(h_AB-parity, h_BA-parity)`` bond spectra of an MPS.

    Takes the finished ``a b a b ...`` site list and returns the two adjacent
    middle bonds from :meth:`FiniteMPS.compute_singular_values`, which fills
    every bond and normalises each to ``sum(sv**2) == 1``.
    (:meth:`canonicalize` fills only the centre bond and cannot be used for
    this.)

    **Bond parity, derived from the construction, not fitted.**  Site ``i`` is
    ``a`` for even ``i`` and ``b`` for odd ``i``, and MPS site ``i``'s right
    bond contracts with site ``i+1``'s left bond.  Both ``_as_peps`` builders
    below send the MPS left bond to the PEPS ``l`` leg and the right bond to
    ``r``, so chain bond ``i`` is ``a.r <-> b.l`` for even ``i`` -- which
    ``BondWeights`` names ``h_AB`` -- and ``b.r <-> a.l`` for odd ``i``, which
    is ``h_BA``.  ``mid`` is forced even so ``(mid, mid+1)`` is exactly one
    ``(h_AB, h_BA)`` pair, both ~``L/2`` sites from either boundary.

    Getting that parity backwards is silent -- the comparison simply fails and
    looks like a BP defect -- so ``test_ipeps_gauge.py`` also asserts that the
    crossed pairing *does* fail, which pins the claim rather than assuming it.
    On the block-sparse arm it is pinned a second, structural way: the two
    bonds there carry *different charge multiplicities*, so the pair's own
    index metadata says which bond is which (see
    :func:`_sym_chain_pair_as_peps`).
    """
    sv = FiniteMPS.from_tensors(tensors).compute_singular_values().singular_values
    mid = 2 * (len(tensors) // 4)
    return np.asarray(sv[mid]), np.asarray(sv[mid + 1])


def _chain_middle_spectra(a, b, vl, vr, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Middle ``(h_AB-parity, h_BA-parity)`` bond spectra of ``a b a b ...``.

    Builds the ``L``-site finite MPS whose bulk alternates ``a``, ``b`` and
    closes the two open ends with ``vl``/``vr`` so it is a genuine state rather
    than a ``chi``-dimensional family, then reads the two middle bonds off it
    with :func:`_middle_bond_pair`, which is where the bond-parity argument
    lives.
    """
    tensors = []
    for i in range(L):
        arr = a if i % 2 == 0 else b
        if i == 0:
            arr = np.tensordot(vl, arr, axes=(0, 0))[None]
        if i == L - 1:
            arr = np.tensordot(arr, vr, axes=(2, 0))[..., None]
        tensors.append(_chain_mps_site(arr, i))
    return _middle_bond_pair(tensors)


def _chain_pair_as_peps(a, b) -> tuple[DenseTensor, DenseTensor]:
    """The same two MPS tensors as a checkerboard PEPS pair, one row of it.

    ``u``/``d`` get dimension 1, so the 2D network factorises into decoupled
    horizontal rows and each row is the chain ``a b a b ...`` verbatim -- the
    state :func:`_chain_middle_spectra` takes the reference from.  The vertical
    bond weights are then length-1 and must come back as exactly 1.0, which is
    a free check on ``gauge_fix``'s bond bookkeeping.

    Flows follow ``ipeps._wrap_as_dense_tensor`` (``l`` OUT, ``r`` IN) so that
    ``A.r`` pairs with ``B.l``; that builder itself cannot be reused because it
    assumes all four virtual legs share one dimension.
    """
    out = []
    for arr in (a, b):
        chi_l, d, chi_r = arr.shape
        out.append(
            DenseTensor(
                jnp.asarray(np.transpose(np.asarray(arr), (0, 2, 1))[None, None]),
                (
                    _trivial_index(1, FlowDirection.OUT, "u"),
                    _trivial_index(1, FlowDirection.IN, "d"),
                    _trivial_index(chi_l, FlowDirection.OUT, "l"),
                    _trivial_index(chi_r, FlowDirection.IN, "r"),
                    _trivial_index(d, FlowDirection.IN, "phys"),
                ),
            )
        )
    return out[0], out[1]


# --- the same anchor on the block-sparse path (#882 Task 6b) --------------
#
# The chain anchor above is dense: every charge in it is zero (``_trivial_index``
# builds nothing else), so it cannot see the one defect class this project keeps
# hitting.  Block-sparse ``contract()`` pairs legs by charge **value** while
# dense einsum pairs by **position** (#834); CTM env-init once emitted chi bonds
# in enumeration order against charge-grouped ones (#602); pinned per-sector keep
# counts made an SVD discard the *largest* singular value (#865).  All three are
# charge bookkeeping, all three are invisible to an all-zero-charge test, and
# #865 in particular was a symmetric simple-update collapse.  This arm repeats
# the same anchor with every leg of the subject carrying a non-trivial charge.

#: The recorded draw for the block-sparse arm.  Chosen the way ``_CHAIN_SEED``
#: was, on the same three measured properties, over 48 seeds (the scan is in
#: ``task-6b-report.md``):
#:
#: * The 2-site transfer matrix's subleading ratio (restricted to its
#:   charge-diagonal block, which is where the environment lives) is 0.063.
#:   That is what sets the ``L`` the reference needs; here the middle-bond
#:   spectrum reaches the f64 floor by ``L = 60``.
#: * Its two inequivalent bonds' spectra differ by 5.595e-01, the largest in
#:   the scan.  The parity-swap discriminator in ``test_ipeps_gauge.py`` *is*
#:   that number, so a seed whose two bonds nearly agreed would make it
#:   vacuous.  (The reviewer's own probe of this task produced effectively
#:   rank-2 bonds, which is the failure this avoids.)
#: * Neither spectrum is degenerate -- smallest within-bond gap 1.9e-02, same
#:   order as the dense arm's 2.6e-02 -- so sorting is well defined and
#:   perturbing one entry is visible.
_SYM_CHAIN_SEED = 30

#: Physical charges: **spin-1**, ``2*Sz`` over ``{+1, 0, -1}``, not the spin-1/2
#: ``[+1, -1]`` used elsewhere in this tree.  The reason is structural, not
#: physical.  With only *odd* physical charges a virtual charge's parity flips
#: at every site, so the 2-site transfer matrix splits into an even and an odd
#: invariant subspace; the finite chain then picks one of them by its boundary
#: charge while BP on the infinite chain picks whichever has the larger
#: eigenvalue, and reference and subject would be different states with no
#: warning.  A physical charge of 0 connects the two parities and the split
#: disappears.  (This is why the reviewer's ``[+1, -1]`` probe came out rank-2:
#: with virtual ``[0, 1, -1, 0]`` each bond could only hold one parity's worth
#: of sectors.)
_SYM_PHYS_CHARGES = np.array([1, 0, -1], dtype=np.int32)

#: The two virtual spaces, one per inequivalent horizontal bond: ``_SYM_VIRT_BA``
#: is ``a.l <-> b.r`` (which the PEPS calls ``h_BA``) and ``_SYM_VIRT_AB`` is
#: ``a.r <-> b.l`` (``h_AB``).  Deliberately **different multisets of the same
#: dimension**: same dimension because ``gauge_fix`` reads one ``D_h`` off
#: ``A.r`` and hands it to both horizontal bonds, different multisets because
#: that is what makes the two bonds structurally distinguishable -- after the
#: flow inversion they carry multiplicities ``[1, 1, 2]`` and ``[2, 1, 1]`` over
#: sectors ``[-1, 0, 1]``, so a transposed ``l``/``r`` is visible in the gauged
#: pair's own index metadata and not only in the numbers.
_SYM_VIRT_BA = np.array([-1, 0, 1, 1], dtype=np.int32)
_SYM_VIRT_AB = np.array([-1, -1, 0, 1], dtype=np.int32)

#: Charge of the dimension-1 MPS boundary caps.  Any sector of ``_SYM_VIRT_BA``
#: works -- it fixes the chain's total charge and washes out of the bulk -- and
#: ``+1`` is the one with multiplicity 2, so the cap contracts a genuine vector
#: rather than a scalar.
_SYM_EDGE_CHARGE = 1

#: Charge carried by all four dimension-1 vertical legs.  A *non-zero* one on
#: purpose: at 0 the vertical bonds would be exactly the all-zero case the dense
#: arm already covers, and then the ``v_AB``/``v_BA`` shape check could not see
#: a dropped charged sector.  Conservation is untouched because ``A.u`` and
#: ``A.d`` carry the same charge with opposite flows, so their contributions
#: cancel.
_SYM_VERT_CHARGE = 1


def _sym_index(charges, flow: FlowDirection, label: str) -> TensorIndex:
    """A U(1) index over an explicit charge multiset."""
    return TensorIndex.from_charges(
        U1Symmetry(), np.asarray(charges, dtype=np.int32), flow, label=label
    )


def _sym_chain_site(rng, left, right) -> SymmetricTensor:
    """One random U(1) MPS site tensor, legs ``(l IN, phys IN, r OUT)``.

    Every symmetry-allowed block is filled: for U(1) the conservation law
    ``q_l + q_phys - q_r == 0`` is written out here rather than taken from
    ``_compute_valid_blocks`` so the test does not certify the block enumerator
    with the block enumerator.

    ``numpy``'s ``default_rng`` rather than ``jax.random``, for the reason
    :func:`_chain_pair` gives: this draw is the anchor's ground truth and NEP 19
    guarantees the PCG64 stream is stable across numpy versions.
    """
    li = _sym_index(left, FlowDirection.IN, "l")
    pi = _sym_index(_SYM_PHYS_CHARGES, FlowDirection.IN, "phys")
    ri = _sym_index(right, FlowDirection.OUT, "r")
    blocks = {}
    for q_l in li.sectors:
        for q_p in pi.sectors:
            q_r = int(q_l) + int(q_p)
            if not ri.has_sector(q_r):
                continue
            shape = (
                li.multiplicity(int(q_l)),
                pi.multiplicity(int(q_p)),
                ri.multiplicity(q_r),
            )
            blocks[(int(q_l), int(q_p), q_r)] = jnp.asarray(rng.normal(size=shape))
    t = SymmetricTensor(blocks, (li, pi, ri))
    return t * (1.0 / float(t.norm()))


def _sym_chain_pair(seed: int = _SYM_CHAIN_SEED):
    """Two symmetric MPS site tensors that tile, plus boundary vectors.

    ``a`` maps ``_SYM_VIRT_BA -> _SYM_VIRT_AB`` and ``b`` maps back, so
    ``a b a b ...`` is a consistent chain: ``a.r`` meets ``b.l`` on the
    ``_SYM_VIRT_AB`` space and ``b.r`` meets ``a.l`` on ``_SYM_VIRT_BA``.

    Returns ``(a, b, vl, vr)``; the two vectors live in the ``_SYM_EDGE_CHARGE``
    sector of ``_SYM_VIRT_BA`` and cap the chain's two open ends.
    """
    rng = np.random.default_rng(seed)
    a = _sym_chain_site(rng, _SYM_VIRT_BA, _SYM_VIRT_AB)
    b = _sym_chain_site(rng, _SYM_VIRT_AB, _SYM_VIRT_BA)
    n = int(np.sum(_SYM_VIRT_BA == _SYM_EDGE_CHARGE))
    vl, vr = rng.normal(size=n), rng.normal(size=n)
    return a, b, vl / np.linalg.norm(vl), vr / np.linalg.norm(vr)


def _sym_chain_cap(t: SymmetricTensor, vec, axis: int) -> SymmetricTensor:
    """Contract ``vec`` into leg ``axis`` of ``t``, leaving a dimension-1 leg.

    The block-sparse counterpart of ``_chain_middle_spectra``'s
    ``np.tensordot(vl, arr, ...)[None]``.  A symmetric leg cannot be capped by
    an arbitrary vector: the surviving dimension-1 leg has to carry a definite
    charge, so ``vec`` covers exactly the ``_SYM_EDGE_CHARGE`` sector and every
    block outside it is dropped.  The retained blocks keep their key, since the
    capped leg still carries that same charge -- which is what keeps the
    conservation law satisfied without touching the other two legs.
    """
    q = _SYM_EDGE_CHARGE
    idx = t.indices[axis]
    blocks = {}
    for key, blk in t.blocks.items():
        if key[axis] != q:
            continue
        v = jnp.asarray(vec)
        blocks[key] = (
            jnp.tensordot(v, blk, axes=(0, 0))[None]
            if axis == 0
            else jnp.tensordot(blk, v, axes=(axis, 0))[..., None]
        )
    indices = list(t.indices)
    indices[axis] = _sym_index([q], idx.flow, idx.label)
    return SymmetricTensor(blocks, tuple(indices))


def _sym_chain_middle_spectra(a, b, vl, vr, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Middle ``(h_AB-parity, h_BA-parity)`` bond spectra of symmetric ``a b a b``.

    The block-sparse twin of :func:`_chain_middle_spectra`, sharing its bond
    labels and its middle-bond/parity logic (:func:`_chain_site_labels`,
    :func:`_middle_bond_pair`) and differing only where the two paths genuinely
    do: how a site is built and how the open ends are capped.

    ``L`` must be **even**, unlike the dense twin, and the check is worth its
    line.  Both caps take a vector of length ``mult(_SYM_VIRT_BA, +1) == 2``,
    which is right for ``a``'s left leg and for ``b``'s right leg.  For odd
    ``L`` the last site is ``a``, whose *right* leg has multiplicity 1 in that
    sector, so the cap would either silently take the wrong slice or raise a
    shape error several frames away from the cause.
    """
    if L % 2 != 0:
        raise ValueError(
            f"L must be even: the right cap assumes the last site is `b` "
            f"(right leg _SYM_VIRT_BA, multiplicity 2 in sector "
            f"{_SYM_EDGE_CHARGE}); at odd L it is `a`, whose right leg is "
            f"_SYM_VIRT_AB with multiplicity 1 there.  Got L={L}."
        )
    tensors = []
    for i in range(L):
        t = a if i % 2 == 0 else b
        if i == 0:
            t = _sym_chain_cap(t, vl, 0)
        if i == L - 1:
            t = _sym_chain_cap(t, vr, 2)
        left, phys, right = _chain_site_labels(i)
        tensors.append(t.relabels({"l": left, "phys": phys, "r": right}))
    return _middle_bond_pair(tensors)


def _sym_chain_pair_as_peps(a, b) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The same two symmetric MPS tensors as a checkerboard PEPS pair.

    ``u``/``d`` get dimension 1, so the 2D network factorises into decoupled
    horizontal rows and each row is the chain ``a b a b ...`` verbatim -- the
    state :func:`_sym_chain_middle_spectra` takes the reference from.

    **The flow inversion.**  An MPS site has left IN / right OUT; the shipped
    iPEPS convention (``ipeps._wrap_as_dense_tensor``,
    ``ipeps.heisenberg_u1sz_init_pair``, ``fermionic_ipeps``) is ``l`` OUT /
    ``r`` IN.  On the dense arm that swap is a no-op because every charge is
    zero.  Here it is not, and the way to express it that leaves the state
    alone is :meth:`TensorIndex.dual` -- flip the flow *and* negate the charges
    -- applied to both virtual legs with the block keys rewritten to match.
    ``flow_charge`` is then unchanged leg by leg, so every stored block still
    satisfies the conservation law and none has to move.

    Three variants were measured (numbers in ``task-6b-report.md``):

    * :meth:`TensorIndex.flip_flow` -- flip the flow, keep the charges -- makes
      the very first block violate conservation and ``SymmetricTensor``
      raises.  Not usable, and loudly so.
    * Dualising only *one* of the two legs still constructs, and ``gauge_fix``
      still reports ``converged=True`` (residual 0.0 after one sweep) with bond
      weights 9.95e-01 away from the truth.  That is #834's signature exactly:
      a charge-value mis-pairing that collapses silently instead of raising,
      which is what this anchor exists to catch.
    * Skipping the inversion entirely -- keeping ``l`` IN / ``r`` OUT and only
      adding the vertical legs -- gives the *same* answer to 1.9e-15.  That is
      not a licence to skip it.  It happens because BP only ever contracts
      ``r`` against ``l``, so a global flip of both is invisible, and because
      the vertical legs here are dimension 1: on a real PEPS the vertical legs
      are charged and the conservation law
      ``-q_u + q_d - q_l + q_r + q_phys = 0`` is genuinely a different equation
      under the two conventions.  The subject has to be in the convention the
      simple update will actually be handed, which is ``l`` OUT / ``r`` IN.

    Because ``dual`` negates charges, the PEPS bond spaces are the duals of the
    MPS ones: ``h_AB`` (``A.r`` / ``B.l``) ends up with multiplicities
    ``[1, 1, 2]`` over sectors ``[-1, 0, 1]`` and ``h_BA`` (``A.l`` / ``B.r``)
    with ``[2, 1, 1]``.  They are different, which is the point.

    ``test_ipeps_gauge.py`` asserts those flows and those spaces on the pair
    this function **returns**, before ``gauge_fix`` sees it.  That is
    deliberate and was learned the hard way: the *gauged* pair cannot witness
    the inversion, because ``gauge_fix`` derives each horizontal leg from its
    own ``eigh``/``svd`` bond and stamps the shipped flows on it whatever it
    was handed, so with the two ``.dual()`` calls removed the gauged metadata
    comes back bit-identical and the whole anchor still passes.  If you change
    what this function returns, that input assertion is the one that will tell
    you.
    """
    sym = U1Symmetry()

    def dual_q(q: int) -> int:
        # The block keys have to move exactly the way ``TensorIndex.dual``
        # moves the sectors, so ask the symmetry rather than writing ``-q``:
        # they agree for U(1) and would not for a group whose dual is not
        # negation.
        return int(sym.dual(np.array([q], dtype=np.int32))[0])

    out = []
    for t in (a, b):
        indices = t.indices
        blocks = {
            (
                _SYM_VERT_CHARGE,
                _SYM_VERT_CHARGE,
                dual_q(key[0]),
                dual_q(key[2]),
                key[1],
            ): jnp.transpose(blk, (0, 2, 1))[None, None]
            for key, blk in t.blocks.items()
        }
        out.append(
            SymmetricTensor(
                blocks,
                (
                    _sym_index([_SYM_VERT_CHARGE], FlowDirection.OUT, "u"),
                    _sym_index([_SYM_VERT_CHARGE], FlowDirection.IN, "d"),
                    indices[0].dual(),
                    indices[2].dual(),
                    indices[1],
                ),
            )
        )
    return out[0], out[1]
