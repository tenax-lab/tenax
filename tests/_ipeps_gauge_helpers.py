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
from tenax.core.tensor import DenseTensor

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


def _chain_mps_site(arr: np.ndarray, i: int) -> DenseTensor:
    """``(left, phys, right)`` array as ``FiniteMPS``'s site ``i``.

    Labels and flows copied from ``mps._build_random_dense_tensors``: the
    canonicalization sweeps address bonds by the names ``v{i-1}_{i}`` /
    ``v{i}_{i+1}`` (``v_-1_0`` at the left edge), so these are not free.
    """
    left = "v_-1_0" if i == 0 else f"v{i - 1}_{i}"
    return DenseTensor(
        jnp.asarray(arr),
        (
            _trivial_index(arr.shape[0], FlowDirection.IN, left),
            _trivial_index(arr.shape[1], FlowDirection.IN, f"p{i}"),
            _trivial_index(arr.shape[2], FlowDirection.OUT, f"v{i}_{i + 1}"),
        ),
    )


def _chain_middle_spectra(a, b, vl, vr, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Middle ``(h_AB-parity, h_BA-parity)`` bond spectra of ``a b a b ...``.

    Builds the ``L``-site finite MPS whose bulk alternates ``a``, ``b``, closes
    the two open ends with ``vl``/``vr`` so it is a genuine state rather than a
    ``chi``-dimensional family, and returns the two adjacent middle bonds from
    :meth:`FiniteMPS.compute_singular_values`, which fills every bond and
    normalises each to ``sum(sv**2) == 1``.  (:meth:`canonicalize` fills only
    the centre bond and cannot be used for this.)

    **Bond parity, derived from the construction, not fitted.**  Site ``i`` is
    ``a`` for even ``i`` and ``b`` for odd ``i``, and MPS site ``i``'s right
    bond contracts with site ``i+1``'s left bond.  :func:`_chain_pair_as_peps`
    sends the MPS left bond to the PEPS ``l`` leg and the right bond to ``r``,
    so chain bond ``i`` is ``a.r <-> b.l`` for even ``i`` -- which
    ``BondWeights`` names ``h_AB`` -- and ``b.r <-> a.l`` for odd ``i``, which
    is ``h_BA``.  ``mid`` is forced even so ``(mid, mid+1)`` is exactly one
    ``(h_AB, h_BA)`` pair, both ~``L/2`` sites from either boundary.

    Getting that parity backwards is silent -- the comparison simply fails and
    looks like a BP defect -- so ``test_ipeps_gauge.py`` also asserts that the
    crossed pairing *does* fail, which pins the claim rather than assuming it.
    """
    tensors = []
    for i in range(L):
        arr = a if i % 2 == 0 else b
        if i == 0:
            arr = np.tensordot(vl, arr, axes=(0, 0))[None]
        if i == L - 1:
            arr = np.tensordot(arr, vr, axes=(2, 0))[..., None]
        tensors.append(_chain_mps_site(arr, i))
    sv = FiniteMPS.from_tensors(tensors).compute_singular_values().singular_values
    mid = 2 * (L // 4)
    return np.asarray(sv[mid]), np.asarray(sv[mid + 1])


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
