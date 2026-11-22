"""The 2-sublattice C4v CTM sweep for charged C4v input (#762, #763).

A bosonic C4v ``SymmetricTensor`` ``A`` whose virtual charges are not all
self-dual describes the 2-sublattice state ``(A, negate(A))`` (see
``_ctm_c4v_charges``).  This module runs its CTM with one corner and one
edge, as ``_c4v_sweep`` does for self-dual input, and keeps the symmetry and
the lattice mirror exact.

The state, in the C1/T1 orientation of the #905 flow table:

- ``C``: the corner of an A-site, legs ``(c_a IN, c_b OUT)``.
- ``T``: the edge above an A-site (its first site is a B-site), legs
  ``(t_l IN, D2 OUT, t_r OUT)``; ``D2`` pairs with the ``u2`` face of ``a``.
- ``a``: the double layer of ``A`` (``_build_double_layer_tensor``).

One step enlarges the corner as ``_c4v_sweep`` does, truncates it with one
block projector (an SVD per mirror pair of blocks, an eigh per self-partner
block, one ranking that never splits a pair), and sets ``C_new = P^T Q P``
and ``T_new = negate(P^T (T a) P)``, which is again the edge above an
A-site.  Design: ``docs/superpowers/specs/2026-11-22-c4v-charged-u1-design.md``.
"""

from __future__ import annotations

__all__ = [
    "_c4v_staggered_projectors",
]

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_c4v_charges import _dual, _dual_charge
from tenax.algorithms._ctm_tensor_init import IN, OUT
from tenax.core.index import TensorIndex
from tenax.core.tensor import SymmetricTensor
from tenax.linalg import _dense_svd

# Values below this fraction of the largest are dropped: the cutoff of the SVD
# projector in ``_ctm_projector.py``.
_REL_CUTOFF = 1e-14


def _c4v_staggered_projectors(
    Qf: SymmetricTensor, chi: int
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The block projector of the 2-sublattice sweep: ``(P_left, P_right)``.

    ``Qf`` is the enlarged corner with the legs ``fused`` (IN, bottom face)
    and ``col`` (OUT, right face).  Both faces carry the same flow-weighted
    charges, so ``col`` holds the duals of the ``fused`` charges, every block
    has the key (q, q), and block q joins the fused states of charge q with
    those of charge ``sym.dual(q)``.  The lattice mirror maps block q to
    block ``sym.dual(q)``, transposed, with the same singular values.

    - A mirror pair (q, dual(q)), q != dual(q): one SVD, Q_q = U S V^T.  The
      projector takes the kept columns of U in sector q and the same columns
      of V in sector dual(q).
    - A self-partner block, q == dual(q) (0 for U(1); also n/2 for Z_n with
      n even): eigh of the symmetrised block, ranked by |lambda|.

    All values are ranked together; a pair value counts once in each of its
    two sectors.  The chi largest are kept, except that a cut between the
    two copies of one pair value drops both (the dimension is then chi - 1).
    Values below ``_REL_CUTOFF`` of the largest are dropped.  This is the
    ranking of the dense eigh of ``_c4v_sweep``: the dense Q has eigenvalues
    +s and -s for each pair value s.

    ``P_left`` is ``(fused OUT, chi_new IN)`` and contracts with the fused
    face; ``P_right`` is ``(col IN, chi_new_r OUT)`` and contracts with the
    col face.  They hold the same arrays, one dense P on both faces; only the
    keys and flows differ.

    Raises:
        RuntimeError: the two faces of ``Qf`` do not carry the same
            flow-weighted charges (not a 2-sublattice corner).
        ValueError: nothing survives the cutoff (a zero corner), or chi = 1
            and the largest value belongs to a mirror pair.
    """
    labels = Qf.labels()
    fpos, cpos = labels.index("fused"), labels.index("col")
    fused, col = Qf.indices[fpos], Qf.indices[cpos]
    sym = fused.symmetry
    if not np.array_equal(col.charges, _dual(sym, fused.charges)):
        raise RuntimeError(
            "charged C4v projector: the two faces of the enlarged corner do not "
            "carry the same flow-weighted charges, so the env is not a "
            "2-sublattice env"
        )
    # Block (q, q): rows are the fused states of charge q, columns the col
    # states of charge q, which are the fused states of charge dual(q).
    mats = {
        int(key[fpos]): (block if fpos < cpos else block.T)
        for key, block in Qf.blocks.items()
    }

    decomp, cands, done = {}, [], set()
    for q in sorted(mats):
        if q in done:
            continue
        qd = _dual_charge(sym, q)
        M = mats[q]
        if qd == q:
            # Self-partner block: symmetric by the mirror; rank by |lambda|.
            lam, W = jnp.linalg.eigh(0.5 * (M + M.T))
            decomp[q] = ("self", W, None)
            for n, v in enumerate(np.abs(np.asarray(lam))):
                cands.append((float(v), q, n, 0))
            done.add(q)
        else:
            # Mirror pair (q, dual(q)): one SVD.  Each value has two copies,
            # one state in sector q and one in sector dual(q).
            U, S, Vh = _dense_svd(M, full_matrices=False)
            decomp[q] = ("pair", U, Vh.T)
            for n, v in enumerate(np.asarray(S)):
                cands += [(float(v), q, n, 0), (float(v), q, n, 1)]
            done.update((q, qd))

    vmax = max((c[0] for c in cands), default=0.0)
    alive = sorted(
        (c for c in cands if c[0] > 0.0 and c[0] >= _REL_CUTOFF * vmax),
        key=lambda c: (-c[0], c[1], c[2], c[3]),
    )
    if not alive:
        raise ValueError("charged C4v projector: the enlarged corner is zero")

    copies: dict[tuple[int, int], int] = {}
    for _v, q, n, _copy in alive[:chi]:
        copies[(q, n)] = copies.get((q, n), 0) + 1
    keep: dict[int, list[int]] = {}
    for (q, n), count in copies.items():
        # A cut between the two copies of a pair value would keep one state
        # of a mirror pair without its partner: drop both.
        if decomp[q][0] == "pair" and count == 1:
            continue
        keep.setdefault(q, []).append(n)
    if not keep:
        raise ValueError(
            f"ctm_tensor_c4v: chi={chi} keeps no state of the charged C4v "
            "corner. Its largest value belongs to a mirror pair (q, dual(q)), "
            "which needs two states. Use chi >= 2."
        )

    cols = {}
    for q in sorted(keep):
        kind, U, V = decomp[q]
        ns = sorted(keep[q])
        cols[q] = U[:, ns]
        if kind == "pair":
            cols[_dual_charge(sym, q)] = V[:, ns]

    # One array per sector on both faces.  On the col face, the states of
    # fused charge x have col charge dual(x), and so does their chi_new_r leg.
    left, right, chi_charges = {}, {}, []
    for x in sorted(cols):
        xd = _dual_charge(sym, x)
        left[(x, x)] = cols[x]
        right[(xd, xd)] = cols[x]
        chi_charges += [x] * cols[x].shape[1]
    chi_charges = np.asarray(chi_charges, dtype=np.int32)
    P_left = SymmetricTensor(
        left,
        (
            fused.flip_flow(),
            TensorIndex.from_charges(sym, chi_charges, IN, label="chi_new"),
        ),
    )
    P_right = SymmetricTensor(
        right,
        (
            col.flip_flow(),
            TensorIndex.from_charges(
                sym, _dual(sym, chi_charges), OUT, label="chi_new_r"
            ),
        ),
    )
    return P_left, P_right
