"""Graded fusion of ket/bra leg pairs and the graded double layer (#1035,
design §5 step 3).

Beside the production path: nothing in ``src/`` imports this module yet.

**The fuse rule.**  A double-layer leg fuses a ket leg ``k`` with its bra
partner ``K`` (opposite flows).  Bringing them together is a graded reorder
(rule 1), and the fused leg takes the bra leg's flow, as production's
``_build_double_layer_tensor`` does.  That is not yet enough: rule 2 twists
a contracted leg by the parity of the whole leg when it is IN on the left
operand, and the fused basis must be enumerated in the same ``(k, K)`` order
on both ends of the bond.  Which sign the unfused pair needs depends on
which operand order a later ``graded_contract`` uses: in one order the
unfused pair needs ``(-1)^{p_k}`` alone and the fused leg's own twist would
be ``(-1)^{p_k p_K}``; in the other order the unfused twist falls on the
bra leg alone, ``(-1)^{p_K}``, and the fused leg's twist would be
``(-1)^{p_k + p_K}``.  Either way the NET correction -- what turns the
fused leg's twist into the unfused pair's -- is the same: ``(-1)^{p_k}``.
Both needs are repaired by one sign on exactly one end of every bond -- the
end whose ket leg is IN:

    (-1)^{p_k p_K + p_k}

``p_k p_K`` is the Koszul swap between the ``(K, k)`` order nesting needs
and the ``(k, K)`` order the fused basis uses; ``p_k`` is that net
correction, applied once at fuse time so it holds in either later
contraction order.  The sign depends only on the parity of a contracted
pair, so either end would do; tying it to the ket leg's flow makes the rule
local.  With it, contracting two fused tensors equals contracting them
unfused, in either operand order.
"""

from __future__ import annotations

from tenax.algorithms._tensor_utils import fuse_indices, split_index
from tenax.core._graded import (
    _is_graded,
    _scale_blocks,
    graded_bar,
    graded_contract,
    graded_reorder,
)
from tenax.core.index import FlowDirection
from tenax.core.tensor import SymmetricTensor

IN = FlowDirection.IN


def _pair_sign(t: SymmetricTensor, k: int) -> SymmetricTensor:
    """``(-1)^{p_k p_K + p_k}`` for the ket leg at axis ``k`` and its bra
    partner at ``k + 1``."""
    return _scale_blocks(t, lambda p: p[k] * p[k + 1] + p[k])


def graded_fuse_pair(t: SymmetricTensor, ket, bra, fused_label) -> SymmetricTensor:
    """Fuse ket leg ``ket`` and bra leg ``bra`` into ``fused_label``, graded.

    The fused leg sits where the first of the two legs was and takes the
    bra leg's flow.  On a bosonic tensor this is plain ``fuse_indices``
    after bringing the legs together.
    """
    if not isinstance(t, SymmetricTensor):
        raise TypeError("graded_fuse_pair needs a SymmetricTensor")
    labels = list(t.labels())
    missing = [lab for lab in (ket, bra) if lab not in labels]
    if missing:
        raise ValueError(f"graded_fuse_pair: no leg labelled {missing}")
    flow_k = t.indices[labels.index(ket)].flow
    flow_b = t.indices[labels.index(bra)].flow
    if flow_k == flow_b:
        raise ValueError(
            f"graded_fuse_pair: {ket!r} and {bra!r} have the same flow; a "
            "ket/bra pair has opposite flows"
        )
    rest = [lab for lab in labels if lab not in (ket, bra)]
    if fused_label in rest:
        raise ValueError(
            f"graded_fuse_pair: fused_label {fused_label!r} is already a leg "
            "of this tensor"
        )
    i = min(labels.index(ket), labels.index(bra))
    t = graded_reorder(t, rest[:i] + [ket, bra] + rest[i:])
    if _is_graded(t) and flow_k == IN:
        t = _pair_sign(t, i)
    return fuse_indices(t, i, i + 1, fused_label, flow_b)


def graded_split_pair(t: SymmetricTensor, fused_label) -> SymmetricTensor:
    """Inverse of :func:`graded_fuse_pair` for a leg that came straight from
    it: the ket and bra legs come back, in that order, where the fused leg
    was, for a fuse-then-split round trip.

    A leg fused by production's sign-free fuse is not detectable here (its
    ``fuse_info`` looks the same) and must not be split with this function.
    """
    labels = list(t.labels())
    if fused_label not in labels:
        raise ValueError(f"graded_split_pair: no leg labelled {fused_label!r}")
    i = labels.index(fused_label)
    info = t.indices[i].fuse_info
    if info is None:
        raise ValueError(f"graded_split_pair: {fused_label!r} is not a fused leg")
    if t.indices[i].flow != info.fused_flow:
        raise ValueError(
            f"graded_split_pair: {fused_label!r}'s flow has changed since "
            "fusion (e.g. by graded_bar); the fusion-time flow can no "
            "longer be inverted correctly"
        )
    out = split_index(t, i)
    if _is_graded(out) and info.parent_indices[0].flow == IN:
        out = _pair_sign(out, i)  # the sign is its own inverse
    return out


def build_graded_double_layer(A: SymmetricTensor, *, phys_bra=None) -> SymmetricTensor:
    """Graded twin of ``_build_double_layer_tensor`` (``phys_bra=None``: the
    physical leg is contracted) and ``_build_double_layer_open_tensor``
    (``phys_bra="phys_bra"``: it stays open under that label).

    Same labels, order and flows as production -- ``(u2, d2, l2, r2)`` then
    ``phys, phys_bra`` when open -- built with ``graded_bar`` (rule 3),
    ``graded_contract`` and :func:`graded_fuse_pair`.  ``A`` has labels
    ``(u, d, l, r, phys)``.
    """
    rename = {x: x.upper() for x in "udlr"}
    if phys_bra is not None:
        rename["phys"] = phys_bra
    a = graded_contract(A, graded_bar(A).relabels(rename))
    for x in "udlr":
        a = graded_fuse_pair(a, x, x.upper(), f"{x}2")
    return a
