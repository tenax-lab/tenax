"""Double-layer construction and env initialization for honeycomb CTM."""

from __future__ import annotations

from tenax.algorithms._ctm_tensor_init import _fuse_pair_by_label
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection
from tenax.core.tensor import Tensor

__all__ = ["_double_layer_honeycomb"]

IN = FlowDirection.IN
OUT = FlowDirection.OUT


def _double_layer_honeycomb(A: Tensor) -> Tensor:
    """Build the rank-3 double-layer tensor for a honeycomb site.

    Input:  A with labels ``(e0, e1, e2, phys)``, 4 legs.
    Output: 3-leg tensor with labels ``(e0_d2, e1_d2, e2_d2)``, dimensions D².

    Mirrors ``_ctm_tensor_init._build_double_layer_tensor`` (rank-5 square
    case) but with 3 virtual legs instead of 4.
    """
    A_bra = A.bar().relabels({"e0": "E0", "e1": "E1", "e2": "E2"})
    a6 = contract(A, A_bra)
    result = _fuse_pair_by_label(a6, "e0", "E0", "e0_d2", OUT)
    result = _fuse_pair_by_label(result, "e1", "E1", "e1_d2", OUT)
    result = _fuse_pair_by_label(result, "e2", "E2", "e2_d2", OUT)
    return result
