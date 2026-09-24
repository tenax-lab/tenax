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
from tenax.core.tensor import SymmetricTensor


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


def twist_legs(t: SymmetricTensor, labels) -> SymmetricTensor:
    """Multiply each block by ``(-1)**(sum of parities on the named legs)``."""
    labels = set(labels)
    axes = [i for i, lab in enumerate(t.labels()) if lab in labels]
    if not axes or not _is_graded(t):
        return t
    return _scale_blocks(t, lambda p: sum(p[i] for i in axes))


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
