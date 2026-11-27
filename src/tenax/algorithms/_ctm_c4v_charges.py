"""Charge helpers for C4v input whose virtual charges are not self-dual.

A bosonic C4v ``SymmetricTensor`` ``A`` whose four faces carry the same
flow-weighted charge at each position, and whose virtual charges are not all
self-dual, describes the 2-sublattice state ``(A, negate(A))``: ``A`` on the
sites with x + y even and ``_negate_charges(A)`` on the others (#762).
peps-torch builds its U(1) C4v state the same way
(``IPEPS_ABELIAN_C4V.get_bipartite_state``; Hasik, Poilblanc and Becca,
SciPost Phys. 10, 012 (2021)).

This is a leaf module: it imports only ``tenax.core``, so the energy routine
can use it without importing the sweep.  Every charge inversion goes through
``sym.dual`` (in ``_dual``) and every flow weighting through
``sym.flow_charge``; no code here writes ``-q`` (#734).
"""

from __future__ import annotations

__all__ = [
    "_C4V_FACES",
    "_dual",
    "_dual_charge",
    "_first_unequal_c4v_face",
    "_is_charged_c4v",
    "_negate_charges",
    "_negate_index",
    "_pairs",
]

import numpy as np

from tenax.core.index import FuseInfo, TensorIndex
from tenax.core.symmetry import BaseSymmetry
from tenax.core.tensor import SymmetricTensor, Tensor

# The four virtual faces of a site tensor; the C4v group permutes them.
_C4V_FACES = ("u", "d", "l", "r")


def _dual(sym: BaseSymmetry, charges) -> np.ndarray:
    """``sym.dual`` of a charge array, as int32: the one charge inversion here."""
    return sym.dual(np.asarray(charges, dtype=np.int32))


def _dual_charge(sym: BaseSymmetry, q: int) -> int:
    """``sym.dual`` of a single charge."""
    return int(_dual(sym, [q])[0])


def _pairs(x: TensorIndex, y: TensorIndex) -> bool:
    """True when the legs ``x`` and ``y`` contract: opposite flows, same charges."""
    return x.flow != y.flow and np.array_equal(x.charges, y.charges)


def _negate_index(sym: BaseSymmetry, idx: TensorIndex) -> TensorIndex:
    """``idx`` with every charge q replaced by ``sym.dual(q)``, at every depth.

    A fused leg keeps its ``fuse_info``, with the parents negated the same way
    and the fused flow kept, so ``split_index`` gives the negated parents.  The
    split is correct because ``_fuse_layout`` orders a fused block by the flat
    position ``i * db + j`` of its basis states, and negation moves none.
    """
    neg = TensorIndex.from_charges(
        sym, _dual(sym, idx.charges), idx.flow, label=idx.label
    )
    if idx.fuse_info is None:
        return neg
    out = TensorIndex(
        symmetry=sym,
        sectors=neg.sectors,
        multiplicities=neg.multiplicities,
        flow=neg.flow,
        label=neg.label,
        fuse_info=FuseInfo(
            parent_indices=tuple(
                _negate_index(sym, p) for p in idx.fuse_info.parent_indices
            ),
            fused_flow=idx.fuse_info.fused_flow,
        ),
    )
    object.__setattr__(out, "_charges_cache", neg.charges)
    return out


def _negate_charges(T: Tensor) -> SymmetricTensor:
    """Replace every charge q of ``T`` by ``sym.dual(q)``: index charges and block keys.

    Flows, labels and block data are kept, so ``todense()`` does not change.
    ``_negate_charges(A)`` is the tensor on the other sublattice of a charged
    C4v state.  Cost O(blocks), no ``todense()``.  A fused leg keeps its
    fusion metadata, with negated parents (``_negate_index``), so
    ``split_index`` works on it.

    Raises:
        TypeError: ``T`` is not a ``SymmetricTensor``.
        NotImplementedError: ``T`` has a fermionic symmetry.  ``ctm_tensor_c4v``
            densifies fermionic input, so no caller negates one.
    """
    if not isinstance(T, SymmetricTensor):
        raise TypeError(
            f"_negate_charges needs a SymmetricTensor, got {type(T).__name__}"
        )
    sym = T.indices[0].symmetry
    if sym.is_fermionic:
        raise NotImplementedError(
            "_negate_charges: fermionic symmetries are not supported"
        )
    indices = tuple(_negate_index(sym, idx) for idx in T.indices)
    blocks = {
        tuple(int(q) for q in _dual(sym, key)): block for key, block in T.blocks.items()
    }
    return SymmetricTensor(blocks, indices)


def _first_unequal_c4v_face(A: SymmetricTensor) -> str | None:
    """The first face whose flow-weighted charges differ from those of face ``u``.

    The faces are compared position by position, not as multisets:
    ``symmetrize_c4v`` permutes the four virtual legs, so it keeps the block
    structure only when all four faces carry the same flow-weighted charge at
    each position.  Returns ``None`` when they do.  ``A`` must have the labels
    ``u``, ``d``, ``l`` and ``r``.
    """
    sym = A.indices[0].symmetry
    faces = {idx.label: idx for idx in A.indices if idx.label in _C4V_FACES}
    ref = sym.flow_charge(faces["u"].flow, faces["u"].charges)
    for face in ("d", "l", "r"):
        fc = sym.flow_charge(faces[face].flow, faces[face].charges)
        if not np.array_equal(fc, ref):
            return face
    return None


def _is_charged_c4v(A: Tensor) -> bool:
    """True for the input that ``ctm_tensor_c4v`` sends to the 2-sublattice sweep.

    That is a bosonic ``SymmetricTensor`` with the labels u, d, l, r and phys,
    four faces with equal flow-weighted charges at each position, and at least
    one virtual charge q with ``q != sym.dual(q)`` at its position.  Self-dual
    charges (Z2, all-zero U(1), Z4 charge 2) give False, and so does a tensor
    that conserves the uniform U(1) (its d face carries the dual of the u
    face).  Reads index metadata only, so it is safe under ``jit`` and
    ``grad``.
    """
    if not isinstance(A, SymmetricTensor):
        return False
    if not {*_C4V_FACES, "phys"} <= set(A.labels()):
        return False
    sym = A.indices[0].symmetry
    if sym.is_fermionic:
        return False
    if _first_unequal_c4v_face(A) is not None:
        return False
    return any(
        not np.array_equal(_dual(sym, idx.charges), idx.charges)
        for idx in A.indices
        if idx.label in _C4V_FACES
    )
