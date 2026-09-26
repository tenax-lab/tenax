"""Reference graded (Grassmann) contraction for fermionic SymmetricTensors (#1035).

Storage order is the Grassmann generator order.  Three rules:

1. reorder with the Koszul sign (``SymmetricTensor.transpose``);
2. a contracted pair is +1 when the left operand's leg is OUT and
   ``(-1)**p`` when it is IN;
3. the bra is ``graded_bar``: ``bar()`` times ``(-1)**sum_{i<j} p_i p_j``
   (a full order reversal, kept in the original storage order).

Beside the production path; nothing calls it yet.
"""

from __future__ import annotations

import numpy as np

from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection
from tenax.core.tensor import SymmetricTensor, _reject_anyonic_twist


def _is_graded(t) -> bool:
    """True for a tensor on a fermionic (Z2-graded) symmetry."""
    return bool(t.indices) and t.indices[0].symmetry.is_fermionic


def _parities(t: SymmetricTensor, key) -> list[int]:
    sym = t.indices[0].symmetry
    return [int(sym.parity(np.array([q]))[0]) for q in key]


def _scale_blocks(t: SymmetricTensor, exponent) -> SymmetricTensor:
    """Multiply each block by ``(-1)**exponent(parities_of_its_key)``."""
    blocks = {}
    for key, block in t.blocks.items():
        blocks[key] = -block if exponent(_parities(t, key)) % 2 else block
    return SymmetricTensor._from_blocks_unchecked(blocks, t.indices)


def twist_axes(t: SymmetricTensor, axes) -> SymmetricTensor:
    """The twist by leg *position* -- the one implementation of the sign.

    :func:`twist_legs` is the label-based spelling and
    :meth:`~tenax.core.tensor.SymmetricTensor.twist` the public axis-based
    one; both land here, so the two cannot drift apart.
    """
    rank = len(t.indices)
    for ax in axes:
        if not -rank <= ax < rank:
            raise IndexError(f"twist axis {ax} out of range for a rank-{rank} tensor")
    sym = t.indices[0].symmetry if t.indices else None
    if sym is None or not axes:
        return t
    # An anyonic symmetry declares a ribbon phase that (-1)**p cannot carry,
    # so refuse rather than silently return the tensor unchanged.
    _reject_anyonic_twist(sym)
    if not _is_graded(t):
        return t
    # Repeats are kept, not de-duplicated: twisting one axis twice must be
    # the identity, which the parity sum already gives.
    norm = [ax % rank for ax in axes]
    return _scale_blocks(t, lambda p: sum(p[ax] for ax in norm))


def twist_legs(t: SymmetricTensor, labels) -> SymmetricTensor:
    """Multiply each block by ``(-1)**(sum of parities on the named legs)``.

    Every label must name a leg of ``t``: a misspelt label would otherwise
    drop its sign silently."""
    labels = set(labels)
    unknown = labels - set(t.labels())
    if unknown:
        raise ValueError(f"twist_legs: no leg labelled {sorted(unknown, key=str)}")
    axes = [i for i, lab in enumerate(t.labels()) if lab in labels]
    return twist_axes(t, axes)


def graded_bar(t: SymmetricTensor) -> SymmetricTensor:
    """Rule 3: the Grassmann conjugate, in the original storage order."""
    b = t.bar()
    if not _is_graded(t):
        return b
    n = t.ndim
    return _scale_blocks(
        b, lambda p: sum(p[i] * p[j] for i in range(n) for j in range(i + 1, n))
    )


def graded_reorder(t: SymmetricTensor, labels) -> SymmetricTensor:
    """Rule 1: reorder legs to ``labels`` with the Koszul sign."""
    current = t.labels()
    return t.transpose(tuple(current.index(lab) for lab in labels))


def graded_contract(a: SymmetricTensor, b: SymmetricTensor) -> SymmetricTensor:
    """Contract the labels ``a`` and ``b`` share, graded.  Output legs are
    ``a``'s free legs then ``b``'s, each in its own storage order."""
    if (_is_graded(a) or _is_graded(b)) and not (
        isinstance(a, SymmetricTensor) and isinstance(b, SymmetricTensor)
    ):
        raise TypeError(
            "graded_contract needs SymmetricTensor operands when either is "
            "fermionic: DenseTensor carries no parity grading"
        )
    if a.ndim and b.ndim and _is_graded(a) != _is_graded(b):
        raise TypeError(
            "graded_contract got one fermionic and one bosonic operand; a "
            "bosonic leg has no parity, so the graded sign is undefined"
        )
    b_labels = set(b.labels())
    shared = [lab for lab in a.labels() if lab in b_labels]
    free_a = [lab for lab in a.labels() if lab not in b_labels]
    free_b = [lab for lab in b.labels() if lab not in set(shared)]
    out = tuple(free_a + free_b)
    if not _is_graded(a):
        return contract(a, b, output_labels=out)
    # nest the pairs: free_a s1..sk | sk..s1 free_b -- each pair adjacent, a's leg first
    a2 = graded_reorder(a, free_a + shared)
    b2 = graded_reorder(b, list(reversed(shared)) + free_b)
    flows = dict(zip(a2.labels(), (idx.flow for idx in a2.indices)))
    a2 = twist_legs(
        a2, [lab for lab in shared if flows[lab] == FlowDirection.IN]
    )  # rule 2
    return contract(a2, b2, output_labels=out)


def graded_svd(
    t: SymmetricTensor, left_labels, right_labels, new_bond_label: str, **svd_kwargs
):
    """``tenax.linalg.svd`` under graded semantics.

    ``svd`` matricizes with a sign-free reorder, which is only correct when
    the legs are already in ``left + right`` order, so reorder them first with
    the Koszul sign.  ``U`` is returned as ``(left..., bond)`` and ``Vh`` as
    ``(bond, right...)``, so ``graded_contract(U * S, Vh)`` reconstructs
    ``graded_reorder(t, left + right)``; ``svd``'s bond orientation (``U``'s
    bond leg OUT) is exactly rule 2's +1 pairing, so no twist is needed.
    """
    from tenax.linalg import svd

    left, right = list(left_labels), list(right_labels)
    return svd(
        graded_reorder(t, left + right),
        left,
        right,
        new_bond_label=new_bond_label,
        **svd_kwargs,
    )
