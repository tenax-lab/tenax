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
    "_make_rank1_dense_corner",
    "initialize_ctm_tensor_env",
]

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_utils import (
    _CORNER_SPECS,
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

    Uses ``A.bar()`` (conjugate + flip flows + Koszul twist for
    fermionic symmetries; identical to ``bar()`` for bosonic) as the bra
    layer. Contracts over the physical index, then fuses ket/bra virtual
    pairs.

    Input:  A with labels (u, d, l, r, phys), 5 legs.
    Output: 4-leg tensor with labels (u2, d2, l2, r2), dimensions D².
    """
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
#   flow_chi1, flow_D2, flow_chi2, ref_axis_chi1, ref_axis_chi2,
#   ref_axis_Da, ref_axis_Db)
# ref_axis_chi1/chi2 are A's axes used to derive the two chi legs' charges
# (matching the connecting corners' ref axes for charge compatibility).
# ref_axis_Da/Db are A's axes that get fused into the D² leg.
_STD_EDGE_SPECS = {
    # T1: top edge. chi1(t1_l) connects to C1.c1_r (C1 ref=1=d),
    #               chi2(t1_r) connects to C2.c2_l (C2 ref=0=u).
    # D² leg is "up" direction: fuse (u, U) = A axes (0, 0).
    "T1": ("t1_l", "u2", "t1_r", IN, IN, OUT, 1, 0, 0, 0),
    # T2: right edge. chi1(t2_u) connects to C2.c2_d (C2 ref=0=u),
    #                 chi2(t2_d) connects to C3.c3_u (C3 ref=1=d).
    "T2": ("t2_u", "r2", "t2_d", OUT, OUT, IN, 0, 1, 3, 3),
    # T3: bottom edge. chi1(t3_r) connects to C4.c4_u (C4 ref=0=u),  # note: C4.c4_u
    #                  chi2(t3_l) connects to C3.c3_l (C3 ref=1=d).
    "T3": ("t3_r", "d2", "t3_l", OUT, OUT, IN, 0, 1, 1, 1),
    # T4: left edge. chi1(t4_d) connects to C1.c1_d (C1 ref=1=d),
    #                chi2(t4_u) connects to C4.c4_r (C4 ref=0=u).
    "T4": ("t4_d", "l2", "t4_u", IN, IN, OUT, 1, 0, 2, 2),
}


# Backward-compatible accessor for tests/external code that destructures
# the 9-element tuple format.
def _std_edge_specs_compat() -> dict:
    """Return edge specs in the old 9-element format for backward compat."""
    return {
        name: (l1, l2, l3, f1, f2, f3, rc1, rda, rdb)
        for name, (
            l1,
            l2,
            l3,
            f1,
            f2,
            f3,
            rc1,
            _rc2,
            rda,
            rdb,
        ) in _STD_EDGE_SPECS.items()
    }


def _make_rank1_dense_corner(
    chi: int,
    label_a: Label,
    label_b: Label,
    flow_a: FlowDirection,
    flow_b: FlowDirection,
    dtype,
) -> DenseTensor:
    """Rank-1 identity-like corner for the standard CTM chi_init=1 init.

    Writes only entry ``(0, 0) = 1`` inside the chi-target-shaped buffer.
    The rest of the (chi, chi) corner stays zero until subsequent CTM
    absorptions grow chi via SVD truncation.  Mirrors variPEPS's
    ``chi_init=1`` semantics (rank-1 corner) without breaking the
    fixed-shape JIT contract.
    """
    from tenax.core.symmetry import U1Symmetry

    sym = U1Symmetry()
    C = jnp.zeros((chi, chi), dtype=dtype).at[0, 0].set(1.0)
    return DenseTensor(
        C,
        (
            TensorIndex.from_charges(
                sym, np.zeros(chi, dtype=np.int32), flow_a, label=label_a
            ),
            TensorIndex.from_charges(
                sym, np.zeros(chi, dtype=np.int32), flow_b, label=label_b
            ),
        ),
    )


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
    # Identity-like edge: T1[i, ket, bra, j] = δ_{i=j} · δ_{ket=bra}.  After
    # fusing (ket, bra) → fused_idx = ket*D + bra, only fused_idx = j*(D+1)
    # for j ∈ 0..D-1 is non-zero. The previous all-ones init (T[i, :, i] = 1
    # across the full D² axis) implements the wrong boundary (1_ket ⊗ 1_bra
    # instead of δ_{ket=bra}) and traps CTM at a degenerate fixed point.
    # variPEPS chi_init=1: write the δ_{ket=bra} pattern only on the
    # leading (i=0) chi slot; subsequent absorptions grow chi via SVD
    # truncation.  See docs/plans/2026-05-11-ctm-bug-3a-design.md.
    D = int(np.round(np.sqrt(D2)))
    assert D * D == D2, f"D² leg dim {D2} is not a perfect square"
    diag_idx = np.arange(D, dtype=np.int32) * (D + 1)
    T = jnp.zeros((chi, D2, chi), dtype=dtype)
    T = T.at[0, diag_idx, 0].set(jnp.ones(D, dtype=dtype))
    return DenseTensor(
        T,
        (
            TensorIndex.from_charges(
                sym, np.zeros(chi, dtype=np.int32), flow_chi1, label=label_chi1
            ),
            TensorIndex.from_charges(
                sym, np.zeros(D2, dtype=np.int32), flow_D2, label=label_D2
            ),
            TensorIndex.from_charges(
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
    ref_axis_chi1: int,
    ref_axis_chi2: int,
    ref_axis_Da: int,
    ref_axis_Db: int,
) -> SymmetricTensor:
    """Create identity-like SymmetricTensor standard edge (chi, D², chi).

    The D² leg charges are derived by fusing A's two virtual axes (ket+bra).
    Each chi leg's charges are derived from D²-fused charges of the
    corresponding corner's reference axis, ensuring charge compatibility.
    """
    from tenax.algorithms._tensor_utils import _compute_fused_charges

    sym = A.indices[0].symmetry
    D2 = D * D

    def _fused_chi_charges(ref_axis: int, flow: FlowDirection) -> np.ndarray:
        """Derive chi charges from D²-fused charges of the given ref axis."""
        ref_idx = A.indices[ref_axis]
        ref_bra = ref_idx.flip_flow()
        fused = _compute_fused_charges(ref_idx, ref_bra, flow, sym)
        if chi <= len(fused):
            return np.asarray(fused[:chi], dtype=np.int32)
        reps = chi // len(fused) + 1
        return np.asarray(np.tile(fused, reps)[:chi], dtype=np.int32)

    chi1_charges = _fused_chi_charges(ref_axis_chi1, flow_chi1)
    chi2_charges = _fused_chi_charges(ref_axis_chi2, flow_chi2)

    # D² charges: fuse the ket virtual axis with the bar'd (flipped-flow) copy
    idx_ket = A.indices[ref_axis_Da]
    idx_bra = idx_ket.flip_flow()  # bar() flips flow
    D2_charges = _compute_fused_charges(idx_ket, idx_bra, flow_D2, sym)

    idx_chi1 = TensorIndex.from_charges(sym, chi1_charges, flow_chi1, label=label_chi1)
    idx_D2 = TensorIndex.from_charges(sym, D2_charges, flow_D2, label=label_D2)
    idx_chi2 = TensorIndex.from_charges(sym, chi2_charges, flow_chi2, label=label_chi2)

    # variPEPS chi_init=1: write the δ_{ket=bra} pattern only on the
    # leading (i=0) chi slot; subsequent absorptions grow chi via SVD
    # truncation.  See `_make_dense_standard_edge` for the rationale (the
    # previous diag-pattern across i ∈ 0..min(chi,D)-1 traps CTM at a
    # degenerate fixed point on generic complex iPEPS).
    diag_idx = np.arange(D, dtype=np.int32) * (D + 1)
    T = jnp.zeros((chi, D2, chi), dtype=A.dtype)
    T = T.at[0, diag_idx, 0].set(jnp.ones(D, dtype=A.dtype))
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

    idx_a = TensorIndex.from_charges(sym, chi_charges.copy(), flow_a, label=label_a)
    idx_b = TensorIndex.from_charges(sym, chi_charges.copy(), flow_b, label=label_b)
    # variPEPS chi_init=1: rank-1 corner — only the leading (0, 0) entry
    # is non-zero. Subsequent absorptions grow chi via SVD truncation.
    C_dense = jnp.zeros((chi, chi), dtype=A.dtype).at[0, 0].set(1.0)
    return SymmetricTensor.from_dense(
        C_dense,
        (idx_a, idx_b),
        tol=float("inf"),
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
            ref_chi1,
            ref_chi2,
            ref_Da,
            ref_Db,
        ) in _STD_EDGE_SPECS.items():
            edges[name] = _init_symmetric_standard_edge(
                A, chi, D, l1, l2, l3, f1, f2, f3, ref_chi1, ref_chi2, ref_Da, ref_Db
            )
    else:
        corners = {}
        for name, (la, lb, fa, fb, _ref) in _CORNER_SPECS.items():
            corners[name] = _make_rank1_dense_corner(chi, la, lb, fa, fb, dtype)

        edges = {}
        for name, (
            l1,
            l2,
            l3,
            f1,
            f2,
            f3,
            _rc1,
            _rc2,
            _rda,
            _rdb,
        ) in _STD_EDGE_SPECS.items():
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
