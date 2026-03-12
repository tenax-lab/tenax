"""Standard CTM with Tensor protocol — data structures and initialization."""

from __future__ import annotations

__all__ = [
    "CTMTensorEnv",
    "IN",
    "OUT",
    "_STD_EDGE_SPECS",
    "_build_double_layer_open_tensor",
    "_build_double_layer_tensor",
    "_fuse_pair_by_label",
    "_init_symmetric_standard_corner",
    "_init_symmetric_standard_edge",
    "_make_dense_standard_edge",
    "initialize_ctm_tensor_env",
]

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_utils import (
    _CORNER_SPECS,
    _derive_charges,
    _make_dense_corner,
)
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ------------------------------------------------------------------ #
# Environment data structure                                          #
# ------------------------------------------------------------------ #


class CTMTensorEnv(NamedTuple):
    """Standard CTM environment with Tensor-protocol fields.

    Corners are 2-leg tensors ``(chi, chi)``.
    Edges are 3-leg tensors ``(chi, D², chi)`` carrying the fused double-layer.

    Corner label/flow conventions match ``_ctm_utils._CORNER_SPECS``.
    """

    C1: Tensor  # (c1_d, c1_r)    flows: (IN, OUT)
    C2: Tensor  # (c2_l, c2_d)    flows: (IN, OUT)
    C3: Tensor  # (c3_u, c3_l)    flows: (OUT, IN)
    C4: Tensor  # (c4_r, c4_u)    flows: (OUT, IN)
    T1: Tensor  # (t1_l, u2, t1_r)  flows: (IN, ?, OUT)
    T2: Tensor  # (t2_u, r2, t2_d)  flows: (OUT, ?, IN)
    T3: Tensor  # (t3_r, d2, t3_l)  flows: (OUT, ?, IN)
    T4: Tensor  # (t4_d, l2, t4_u)  flows: (IN, ?, OUT)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

IN = FlowDirection.IN
OUT = FlowDirection.OUT


def _fuse_pair_by_label(
    T: Tensor,
    label_a: Label,
    label_b: Label,
    fused_label: Label,
    fused_flow: FlowDirection,
) -> Tensor:
    """Find axes by label, then call ``fuse_indices``."""
    labels = T.labels()
    axis_a = labels.index(label_a)
    axis_b = labels.index(label_b)
    return fuse_indices(T, axis_a, axis_b, fused_label, fused_flow)


# ------------------------------------------------------------------ #
# Double-layer construction via bar()                                  #
# ------------------------------------------------------------------ #


def _build_double_layer_tensor(A: Tensor) -> Tensor:
    """Build the 4-leg double-layer tensor from iPEPS site tensor A.

    Uses ``A.bar()`` (conjugate + flip flows, no charge dual) as the bra layer.
    Contracts over the physical index, then fuses ket/bra virtual pairs.

    Input:  A with labels (u, d, l, r, phys), 5 legs.
    Output: 4-leg tensor with labels (u2, d2, l2, r2), dimensions D².
    """
    # Build bra via bar() + relabel virtual legs to uppercase
    A_bra = A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})
    # Contract over shared "phys" label → 8-leg tensor
    a8 = contract(A, A_bra)
    # Fuse pairs: (u, U) → u2, (d, D) → d2, (l, L) → l2, (r, R) → r2
    # fuse_indices puts axis_a as slow-varying (row-major)
    result = _fuse_pair_by_label(a8, "u", "U", "u2", IN)
    result = _fuse_pair_by_label(result, "d", "D", "d2", OUT)
    result = _fuse_pair_by_label(result, "l", "L", "l2", IN)
    result = _fuse_pair_by_label(result, "r", "R", "r2", OUT)
    return result


def _build_double_layer_open_tensor(A: Tensor) -> Tensor:
    """Build the double-layer tensor with physical indices left open.

    Same as ``_build_double_layer_tensor`` but the physical index is relabeled
    to ``phys_bra`` on the bra side so it stays as a free leg.

    Output: 6-leg tensor (u2, d2, l2, r2, phys, phys_bra).
    """
    A_bra = A.bar().relabels(
        {"u": "U", "d": "D", "l": "L", "r": "R", "phys": "phys_bra"}
    )
    a_open = contract(A, A_bra)
    result = _fuse_pair_by_label(a_open, "u", "U", "u2", IN)
    result = _fuse_pair_by_label(result, "d", "D", "d2", OUT)
    result = _fuse_pair_by_label(result, "l", "L", "l2", IN)
    result = _fuse_pair_by_label(result, "r", "R", "r2", OUT)
    return result


# ------------------------------------------------------------------ #
# Initialization                                                       #
# ------------------------------------------------------------------ #


# Edge specs for standard CTM: (label_chi1, label_D2, label_chi2,
#   flow_chi1, flow_D2, flow_chi2, ref_axis_chi, ref_axis_Da, ref_axis_Db)
# ref_axis_Da/Db are A's axes that get fused into the D² leg.
_STD_EDGE_SPECS = {
    # T1: top edge. chi connects to C1.c1_r and C2.c2_l.
    # D² leg is "up" direction: fuse (u, U) = A axes (0, 0).
    "T1": ("t1_l", "u2", "t1_r", IN, IN, OUT, 3, 0, 0),
    # T2: right edge. chi connects to C2.c2_d and C3.c3_u.
    "T2": ("t2_u", "r2", "t2_d", OUT, OUT, IN, 0, 3, 3),
    # T3: bottom edge. chi connects to C4.c4_r and C3.c3_l.
    "T3": ("t3_r", "d2", "t3_l", OUT, OUT, IN, 3, 1, 1),
    # T4: left edge. chi connects to C1.c1_d and C4.c4_u.
    "T4": ("t4_d", "l2", "t4_u", IN, IN, OUT, 1, 2, 2),
}


def _make_dense_standard_edge(
    chi: int,
    D2: int,
    label_chi1: Label,
    label_D2: Label,
    label_chi2: Label,
    flow_chi1: FlowDirection,
    flow_D2: FlowDirection,
    flow_chi2: FlowDirection,
    dtype,
) -> DenseTensor:
    """Create identity-like DenseTensor edge (chi, D², chi)."""
    from tenax.core.symmetry import U1Symmetry

    sym = U1Symmetry()
    T_chi = min(chi, D2)
    T = jnp.zeros((chi, D2, chi), dtype=dtype)
    for i in range(min(T_chi, chi)):
        T = T.at[i, :, i].add(jnp.ones(D2, dtype=dtype))
    return DenseTensor(
        T,
        (
            TensorIndex(
                sym, np.zeros(chi, dtype=np.int32), flow_chi1, label=label_chi1
            ),
            TensorIndex(sym, np.zeros(D2, dtype=np.int32), flow_D2, label=label_D2),
            TensorIndex(
                sym, np.zeros(chi, dtype=np.int32), flow_chi2, label=label_chi2
            ),
        ),
    )


def _init_symmetric_standard_edge(
    A: SymmetricTensor,
    chi: int,
    D: int,
    label_chi1: Label,
    label_D2: Label,
    label_chi2: Label,
    flow_chi1: FlowDirection,
    flow_D2: FlowDirection,
    flow_chi2: FlowDirection,
    ref_axis_chi: int,
    ref_axis_Da: int,
    ref_axis_Db: int,
) -> SymmetricTensor:
    """Create identity-like SymmetricTensor standard edge (chi, D², chi).

    The D² leg charges are derived by fusing A's two virtual axes (ket+bra).
    """
    from tenax.algorithms._tensor_utils import _compute_fused_charges

    sym = A.indices[0].symmetry
    D2 = D * D

    chi_charges = _derive_charges(A.indices[ref_axis_chi].charges, chi)

    # D² charges: fuse the ket virtual axis with the bar'd (flipped-flow) copy
    idx_ket = A.indices[ref_axis_Da]
    idx_bra = idx_ket.flip_flow()  # bar() flips flow
    D2_charges = _compute_fused_charges(idx_ket, idx_bra, flow_D2, sym)

    idx_chi1 = TensorIndex(sym, chi_charges.copy(), flow_chi1, label=label_chi1)
    idx_D2 = TensorIndex(sym, D2_charges, flow_D2, label=label_D2)
    idx_chi2 = TensorIndex(sym, chi_charges.copy(), flow_chi2, label=label_chi2)

    T = jnp.zeros((chi, D2, chi), dtype=A.dtype)
    T_chi = min(chi, D2)
    for i in range(min(T_chi, chi)):
        T = T.at[i, :, i].add(jnp.ones(D2, dtype=A.dtype))
    return SymmetricTensor.from_dense(T, (idx_chi1, idx_D2, idx_chi2), tol=float("inf"))


def _init_symmetric_standard_corner(
    A: SymmetricTensor,
    chi: int,
    label_a: Label,
    label_b: Label,
    flow_a: FlowDirection,
    flow_b: FlowDirection,
    ref_axis: int,
) -> SymmetricTensor:
    """Create an identity-like SymmetricTensor corner for the standard CTM.

    Unlike the split CTM corner, the standard CTM corner has chi-bonds
    whose charges are derived from D² (not D), matching the double-layer
    tensor's fused charges.
    """
    from tenax.algorithms._tensor_utils import _compute_fused_charges

    ref_idx = A.indices[ref_axis]
    sym = ref_idx.symmetry

    # The chi bonds carry D²-derived charges: fuse ref_idx with bar'd copy
    idx_bra = ref_idx.flip_flow()
    fused_charges = _compute_fused_charges(ref_idx, idx_bra, flow_a, sym)
    # fused_charges has size D²; tile to chi
    base_D2_charges = fused_charges
    if chi <= len(base_D2_charges):
        chi_charges = np.asarray(base_D2_charges[:chi], dtype=np.int32)
    else:
        reps = chi // len(base_D2_charges) + 1
        chi_charges = np.asarray(np.tile(base_D2_charges, reps)[:chi], dtype=np.int32)

    idx_a = TensorIndex(sym, chi_charges.copy(), flow_a, label=label_a)
    idx_b = TensorIndex(sym, chi_charges.copy(), flow_b, label=label_b)
    return SymmetricTensor.from_dense(
        jnp.eye(chi, dtype=A.dtype),
        (idx_a, idx_b),
    )


def initialize_ctm_tensor_env(
    A: Tensor,
    chi: int,
) -> CTMTensorEnv:
    """Initialize a CTMTensorEnv from an iPEPS site tensor.

    Args:
        A:   Site tensor with 5 legs ``(u, d, l, r, phys)``.
        chi: Environment bond dimension.

    Returns:
        Initialized CTMTensorEnv.
    """
    D = A.indices[0].dim
    D2 = D * D
    dtype = A.dtype

    if isinstance(A, SymmetricTensor):
        corners = {}
        for name, (la, lb, fa, fb, ref) in _CORNER_SPECS.items():
            corners[name] = _init_symmetric_standard_corner(A, chi, la, lb, fa, fb, ref)

        edges = {}
        for name, (
            l1,
            l2,
            l3,
            f1,
            f2,
            f3,
            ref_chi,
            ref_Da,
            ref_Db,
        ) in _STD_EDGE_SPECS.items():
            edges[name] = _init_symmetric_standard_edge(
                A, chi, D, l1, l2, l3, f1, f2, f3, ref_chi, ref_Da, ref_Db
            )
    else:
        corners = {}
        for name, (la, lb, fa, fb, _ref) in _CORNER_SPECS.items():
            corners[name] = _make_dense_corner(chi, D2, la, lb, fa, fb, dtype)

        edges = {}
        for name, (l1, l2, l3, f1, f2, f3, _rc, _rda, _rdb) in _STD_EDGE_SPECS.items():
            edges[name] = _make_dense_standard_edge(
                chi, D2, l1, l2, l3, f1, f2, f3, dtype
            )

    return CTMTensorEnv(
        C1=corners["C1"],
        C2=corners["C2"],
        C3=corners["C3"],
        C4=corners["C4"],
        T1=edges["T1"],
        T2=edges["T2"],
        T3=edges["T3"],
        T4=edges["T4"],
    )
