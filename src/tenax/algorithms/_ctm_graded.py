"""Graded-aware primitives for the Tensor CTM (#1035, design §5 step 4).

On a fermionic ``SymmetricTensor`` each function applies the graded rules of
:mod:`tenax.core._graded` (rule 1: Koszul reorder; rule 2: the pair sign;
rule 3: ``graded_bar``).  On a ``DenseTensor`` or a bosonic ``SymmetricTensor``
each is production's ``contract``, ``bar``, ``svd`` or double-layer builder,
unchanged -- so nothing bosonic moves.

Transitional: design §5 step 6 flips ``contract`` and ``bar`` themselves, and
this module goes away with it.
"""

from __future__ import annotations

from tenax.contraction.contractor import contract as _contract
from tenax.core._graded import (
    _is_graded,
    _scale_blocks,
    graded_bar,
    graded_contract,
    graded_reorder,
    graded_svd,
)
from tenax.core.tensor import SymmetricTensor

__all__ = ["bar", "contract", "dense_rdm", "is_fermionic", "svd"]


def is_fermionic(t) -> bool:
    """True for a ``SymmetricTensor`` on a fermionic (Z2-graded) symmetry."""
    return isinstance(t, SymmetricTensor) and _is_graded(t)


def contract(a, b, *, output_labels=None):
    """Pairwise contraction on shared labels: graded when either operand is
    fermionic, production's ``contract`` otherwise.

    The output leg order is the same in both cases -- ``a``'s free legs then
    ``b``'s, each in its own storage order -- so call sites that rely on it
    need no change.  An explicit ``output_labels`` is applied with a graded
    reorder on the fermionic path.
    """
    if not (is_fermionic(a) or is_fermionic(b)):
        return _contract(a, b, output_labels=output_labels)
    out = graded_contract(a, b)
    if output_labels is not None and list(output_labels) != list(out.labels()):
        out = graded_reorder(out, list(output_labels))
    return out


def bar(t):
    """``graded_bar`` (rule 3) on a fermionic tensor, ``t.bar()`` otherwise."""
    return graded_bar(t) if is_fermionic(t) else t.bar()


def svd(t, left_labels, right_labels, new_bond_label="bond", **kwargs):
    """``tenax.linalg.svd`` with a graded reorder into ``left + right`` order
    first when ``t`` is fermionic (see :func:`graded_svd`)."""
    if is_fermionic(t):
        return graded_svd(t, left_labels, right_labels, new_bond_label, **kwargs)
    from tenax.linalg import svd as _svd

    return _svd(
        t,
        left_labels=left_labels,
        right_labels=right_labels,
        new_bond_label=new_bond_label,
        **kwargs,
    )


def dense_rdm(rdm_t, bra_labels):
    """The dense reduced density matrix of a contracted RDM tensor.

    ``rdm_t`` holds the ket physical legs, then the bra legs ``bra_labels``.
    On a fermionic tensor the raw blocks are not the matrix elements
    ``<|P><p|>``: reading one off means contracting the bra legs with a
    basis operator, which costs rule 2's pair sign ``(-1)**P_i`` on each bra
    leg and the Koszul sign ``(-1)**sum_{i<j} P_i P_j`` of taking the bra
    legs in reverse (the operator's annihilation half).  Without it the
    1-site RDM reads ``diag(q0, -q1)`` and the pairing elements come out
    antisymmetric.  Checked against the finite-patch reference in
    ``tests/test_graded_ctm.py``.  Non-fermionic tensors are read as is.
    """
    if is_fermionic(rdm_t):
        axes = [rdm_t.labels().index(lab) for lab in bra_labels]
        rdm_t = _scale_blocks(
            rdm_t,
            lambda p: (
                sum(p[i] for i in axes)
                + sum(p[i] * p[j] for n, i in enumerate(axes) for j in axes[n + 1 :])
            ),
        )
    return rdm_t.todense()
