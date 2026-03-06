"""Standard CTM using the Tensor protocol (polymorphic dense/symmetric).

Builds the full double-layer tensor via ``bar()`` + ``contract()`` + ``fuse_indices()``,
then runs the standard projector-based CTM with dense projectors and Tensor-protocol
environment tensors.

This module parallels the dense CTM in ``ipeps.py`` but uses Tensor objects
(DenseTensor or SymmetricTensor) throughout, enabling block-sparse acceleration
for symmetric iPEPS without code duplication.

Reference: Corboz et al., PRB 81, 165104 (2010)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._split_ctm_tensor import (
    _CORNER_SPECS,
    _compute_projector_dense,
    _derive_charges,
    _make_dense_corner,
    _wrap_corner_dense,
)
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ------------------------------------------------------------------ #
# Environment data structure                                          #
# ------------------------------------------------------------------ #


class CTMTensorEnv(NamedTuple):
    """Standard CTM environment with Tensor-protocol fields.

    Corners are 2-leg tensors ``(chi, chi)``.
    Edges are 3-leg tensors ``(chi, D², chi)`` carrying the fused double-layer.

    Corner label/flow conventions match ``_split_ctm_tensor._CORNER_SPECS``.
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


# ------------------------------------------------------------------ #
# Wrap dense results back as Tensor                                    #
# ------------------------------------------------------------------ #


def _wrap_standard_edge_dense(
    data: jax.Array,
    label_chi1: Label,
    label_D2: Label,
    label_chi2: Label,
    ref_indices: tuple[TensorIndex, ...],
    chi: int,
    D2: int,
    symmetric: bool = False,
    base_chi_charges: np.ndarray | None = None,
    base_D2_charges: np.ndarray | None = None,
) -> Tensor:
    """Wrap a dense (chi, D², chi) array as a Tensor with correct metadata."""
    sym = ref_indices[0].symmetry
    chi_charges = (
        _derive_charges(base_chi_charges, chi)
        if base_chi_charges is not None
        else np.zeros(chi, dtype=np.int32)
    )
    D2_charges = (
        np.asarray(base_D2_charges, dtype=np.int32)
        if base_D2_charges is not None
        else np.zeros(D2, dtype=np.int32)
    )

    idx1 = TensorIndex(sym, chi_charges.copy(), ref_indices[0].flow, label=label_chi1)
    idx2 = TensorIndex(sym, D2_charges, ref_indices[1].flow, label=label_D2)
    idx3 = TensorIndex(sym, chi_charges.copy(), ref_indices[2].flow, label=label_chi2)
    indices = (idx1, idx2, idx3)
    if symmetric:
        return SymmetricTensor.from_dense(data, indices, tol=float("inf"))
    return DenseTensor(data, indices)


# ------------------------------------------------------------------ #
# CTM moves                                                            #
# ------------------------------------------------------------------ #


def _ctm_tensor_move_left(
    env: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Left move: updates C1, T4, C4.

    Dense reference: C1g = einsum('ab,buc->auc', C1, T1)
                     C4g = einsum('gh,hdi->gdi', C4, T3)
                     T4g = einsum('alg,udlr->augdr', T4, a)
    """
    D2 = a.indices[0].dim

    # C1(c1_d=a, c1_r=b) · T1(t1_l=b, u2, t1_r=c) → contract b: c1_r ↔ t1_l
    C1_r = env.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env.T1)  # (c1_d, u2, t1_r)
    C1g = _fuse_pair_by_label(C1g, "c1_d", "u2", "fused", IN)  # (fused, t1_r)

    # C4(c4_r=g, c4_u=h) · T3(t3_r=h, d2, t3_l) → contract h: c4_u ↔ t3_r
    C4_u = env.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env.T3)  # (c4_r, d2, t3_l)
    C4g = _fuse_pair_by_label(C4g, "c4_r", "d2", "fused", IN)  # (fused, t3_l)

    # T4(t4_d=a, l2, t4_u=g) · a(u2, d2, l2, r2) → contract l2
    # Dense result (a,u,g,d,r), transpose(0,1,4,2,3)→(a,u,r,g,d)
    # reshape → (a*u, r, g*d) = fuse(t4_d,u2), r2, fuse(t4_u,d2)
    T4_with_a = contract(env.T4, a)
    T4g = _fuse_pair_by_label(T4_with_a, "t4_d", "u2", "fl", IN)
    T4g = _fuse_pair_by_label(T4g, "t4_u", "d2", "fr", IN)
    T4g_dense = T4g.todense()
    T4g_labels = T4g.labels()
    perm = [T4g_labels.index("fl"), T4g_labels.index("r2"), T4g_labels.index("fr")]
    T4g_dense = T4g_dense.transpose(perm)

    # 4. Dense projector
    C1g_dense = C1g.todense()
    C4g_dense = C4g.todense()
    P = _compute_projector_dense(C1g_dense, C4g_dense, chi)

    # 5. Apply P
    C1_new_dense = P.conj().T @ C1g_dense  # (chi, col1)
    C4_new_dense = P.conj().T @ C4g_dense  # (chi, col2)
    T4_new_dense = jnp.einsum("ia,idj,jb->adb", P, T4g_dense, P)  # (chi, D2, chi)

    # 6. Wrap back as Tensor
    _sym = isinstance(a, SymmetricTensor)
    _c1_q = a.indices[1].charges if _sym else None  # ref for C1 charges
    _c4_q = a.indices[0].charges if _sym else None  # ref for C4 charges

    C1_new = _wrap_corner_dense(
        C1_new_dense,
        "c1_d",
        "c1_r",
        env.C1.indices[0],
        env.C1.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c1_q,
    )
    C4_new = _wrap_corner_dense(
        C4_new_dense,
        "c4_r",
        "c4_u",
        env.C4.indices[0],
        env.C4.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c4_q,
    )

    _t4_chi_q = a.indices[1].charges if _sym else None
    _t4_D2_q = env.T4.indices[1].charges if _sym else None
    T4_new = _wrap_standard_edge_dense(
        T4_new_dense,
        "t4_d",
        "l2",
        "t4_u",
        env.T4.indices,
        chi,
        D2,
        symmetric=_sym,
        base_chi_charges=_t4_chi_q,
        base_D2_charges=_t4_D2_q,
    )

    return env._replace(C1=C1_new, C4=C4_new, T4=T4_new)


def _ctm_tensor_move_right(
    env: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Right move: updates C2, T2, C3.

    Dense reference: C2g = einsum('ce,buc->eub', C2, T1)
                     C3g = einsum('im,hdi->mdh', C3, T3)
                     T2g = einsum('erm,udlr->eumdl', T2, a)
    """
    D2 = a.indices[0].dim

    # C2(c2_l=c, c2_d=e) · T1(t1_l=b, u2, t1_r=c) → contract c: c2_l ↔ t1_r
    C2_l = env.C2.relabel("c2_l", "t1_r")
    C2g = contract(C2_l, env.T1)  # (c2_d, t1_l, u2)
    C2g = _fuse_pair_by_label(C2g, "c2_d", "u2", "fused", IN)  # (fused, t1_l)

    # C3(c3_u=i, c3_l=m) · T3(t3_r=h, d2, t3_l=i) → contract i: c3_u ↔ t3_l
    C3_u = env.C3.relabel("c3_u", "t3_l")
    C3g = contract(C3_u, env.T3)  # (c3_l, t3_r, d2)
    C3g = _fuse_pair_by_label(C3g, "c3_l", "d2", "fused", IN)  # (fused, t3_r)

    # T2(t2_u=e, r2, t2_d=m) · a(u2, d2, l2, r2) → contract r2
    # Dense result (e,u,m,d,l), transpose(0,1,4,2,3)→(e,u,l,m,d)
    # reshape → (e*u, l, m*d) = fuse(t2_u,u2), l2, fuse(t2_d,d2)
    T2_with_a = contract(env.T2, a)
    T2g = _fuse_pair_by_label(T2_with_a, "t2_u", "u2", "fl", IN)
    T2g = _fuse_pair_by_label(T2g, "t2_d", "d2", "fr", IN)
    T2g_dense = T2g.todense()
    T2g_labels = T2g.labels()
    perm = [T2g_labels.index("fl"), T2g_labels.index("l2"), T2g_labels.index("fr")]
    T2g_dense = T2g_dense.transpose(perm)

    C2g_dense = C2g.todense()
    C3g_dense = C3g.todense()
    P = _compute_projector_dense(C2g_dense, C3g_dense, chi)

    C2_new_dense = P.conj().T @ C2g_dense
    C3_new_dense = P.conj().T @ C3g_dense
    T2_new_dense = jnp.einsum("ia,idj,jb->adb", P, T2g_dense, P)

    _sym = isinstance(a, SymmetricTensor)
    _c2_q = a.indices[0].charges if _sym else None
    _c3_q = a.indices[1].charges if _sym else None

    C2_new = _wrap_corner_dense(
        C2_new_dense,
        "c2_l",
        "c2_d",
        env.C2.indices[0],
        env.C2.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c2_q,
    )
    C3_new = _wrap_corner_dense(
        C3_new_dense,
        "c3_u",
        "c3_l",
        env.C3.indices[0],
        env.C3.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c3_q,
    )

    _t2_chi_q = a.indices[0].charges if _sym else None
    _t2_D2_q = env.T2.indices[1].charges if _sym else None
    T2_new = _wrap_standard_edge_dense(
        T2_new_dense,
        "t2_u",
        "r2",
        "t2_d",
        env.T2.indices,
        chi,
        D2,
        symmetric=_sym,
        base_chi_charges=_t2_chi_q,
        base_D2_charges=_t2_D2_q,
    )

    return env._replace(C2=C2_new, C3=C3_new, T2=T2_new)


def _ctm_tensor_move_top(
    env: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Top move: updates C1, T1, C2.

    Dense reference: C1g = einsum('ab,alg->blg', C1, T4)
                     C2g = einsum('ce,erm->crm', C2, T2)
                     T1g = einsum('buc,udlr->bcdlr', T1, a)
    """
    D2 = a.indices[0].dim

    # C1(c1_d=a, c1_r=b) · T4(t4_d=a, l2, t4_u) → contract a: c1_d ↔ t4_d
    C1_d = env.C1.relabel("c1_d", "t4_d")
    C1g = contract(C1_d, env.T4)  # (c1_r, l2, t4_u)
    C1g = _fuse_pair_by_label(C1g, "c1_r", "l2", "fused", IN)  # (fused, t4_u)

    # C2(c2_l=c, c2_d=e) · T2(t2_u=e, r2, t2_d) → contract e: c2_d ↔ t2_u
    C2_d = env.C2.relabel("c2_d", "t2_u")
    C2g = contract(C2_d, env.T2)  # (c2_l, r2, t2_d)
    C2g = _fuse_pair_by_label(C2g, "c2_l", "r2", "fused", IN)  # (fused, t2_d)

    # T1(t1_l=b, u2, t1_r=c) · a(u2, d2, l2, r2) → contract u2
    # Dense result (b,c,d,l,r), transpose(0,3,2,1,4)→(b,l,d,c,r)
    # reshape → (b*l, d, c*r) = fuse(t1_l,l2), d2, fuse(t1_r,r2)
    T1_with_a = contract(env.T1, a)
    T1g = _fuse_pair_by_label(T1_with_a, "t1_l", "l2", "fl", IN)
    T1g = _fuse_pair_by_label(T1g, "t1_r", "r2", "fr", IN)
    T1g_dense = T1g.todense()
    T1g_labels = T1g.labels()
    perm = [T1g_labels.index("fl"), T1g_labels.index("d2"), T1g_labels.index("fr")]
    T1g_dense = T1g_dense.transpose(perm)

    C1g_dense = C1g.todense()
    C2g_dense = C2g.todense()
    P = _compute_projector_dense(C1g_dense, C2g_dense, chi)

    C1_new_dense = P.conj().T @ C1g_dense
    C2_new_dense = P.conj().T @ C2g_dense
    T1_new_dense = jnp.einsum("ia,idj,jb->adb", P, T1g_dense, P)

    _sym = isinstance(a, SymmetricTensor)
    _c1_q = a.indices[1].charges if _sym else None
    _c2_q = a.indices[0].charges if _sym else None

    C1_new = _wrap_corner_dense(
        C1_new_dense,
        "c1_d",
        "c1_r",
        env.C1.indices[0],
        env.C1.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c1_q,
    )
    C2_new = _wrap_corner_dense(
        C2_new_dense,
        "c2_l",
        "c2_d",
        env.C2.indices[0],
        env.C2.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c2_q,
    )

    _t1_chi_q = a.indices[3].charges if _sym else None
    _t1_D2_q = env.T1.indices[1].charges if _sym else None
    T1_new = _wrap_standard_edge_dense(
        T1_new_dense,
        "t1_l",
        "u2",
        "t1_r",
        env.T1.indices,
        chi,
        D2,
        symmetric=_sym,
        base_chi_charges=_t1_chi_q,
        base_D2_charges=_t1_D2_q,
    )

    return env._replace(C1=C1_new, C2=C2_new, T1=T1_new)


def _ctm_tensor_move_bottom(
    env: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Bottom move: updates C4, T3, C3.

    Dense reference: C4g = einsum('gh,alg->hal', C4, T4).transpose(0,2,1)
                     C3g = einsum('im,erm->ire', C3, T2)
                     T3g = einsum('hdi,udlr->hiulr', T3, a)
    """
    D2 = a.indices[0].dim

    # C4(c4_r=g, c4_u=h) · T4(t4_d=a, l2, t4_u=g) → contract g: c4_r ↔ t4_u
    C4_r = env.C4.relabel("c4_r", "t4_u")
    C4g = contract(C4_r, env.T4)  # (c4_u, t4_d, l2)
    # Dense: result (h,a,l) transposed to (h,l,a), reshape → fuse(c4_u, l2), t4_d
    C4g = _fuse_pair_by_label(C4g, "c4_u", "l2", "fused", IN)  # (fused, t4_d)

    # C3(c3_u=i, c3_l=m) · T2(t2_u=e, r2, t2_d=m) → contract m: c3_l ↔ t2_d
    C3_l = env.C3.relabel("c3_l", "t2_d")
    C3g = contract(C3_l, env.T2)  # (c3_u, t2_u, r2)
    C3g = _fuse_pair_by_label(C3g, "c3_u", "r2", "fused", IN)  # (fused, t2_u)

    # T3(t3_r=h, d2, t3_l=i) · a(u2, d2, l2, r2) → contract d2
    # Dense result (h,i,u,l,r), transpose(0,3,2,1,4)→(h,l,u,i,r)
    # reshape → (h*l, u, i*r) = fuse(t3_r,l2), u2, fuse(t3_l,r2)
    T3_with_a = contract(env.T3, a)
    T3g = _fuse_pair_by_label(T3_with_a, "t3_r", "l2", "fl", IN)
    T3g = _fuse_pair_by_label(T3g, "t3_l", "r2", "fr", IN)
    T3g_dense = T3g.todense()
    T3g_labels = T3g.labels()
    perm = [T3g_labels.index("fl"), T3g_labels.index("u2"), T3g_labels.index("fr")]
    T3g_dense = T3g_dense.transpose(perm)

    C4g_dense = C4g.todense()
    C3g_dense = C3g.todense()
    P = _compute_projector_dense(C4g_dense, C3g_dense, chi)

    C4_new_dense = P.conj().T @ C4g_dense
    C3_new_dense = P.conj().T @ C3g_dense
    T3_new_dense = jnp.einsum("ia,idj,jb->adb", P, T3g_dense, P)

    _sym = isinstance(a, SymmetricTensor)
    _c4_q = a.indices[0].charges if _sym else None
    _c3_q = a.indices[1].charges if _sym else None

    C4_new = _wrap_corner_dense(
        C4_new_dense,
        "c4_r",
        "c4_u",
        env.C4.indices[0],
        env.C4.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c4_q,
    )
    C3_new = _wrap_corner_dense(
        C3_new_dense,
        "c3_u",
        "c3_l",
        env.C3.indices[0],
        env.C3.indices[1],
        chi,
        symmetric=_sym,
        base_charges=_c3_q,
    )

    _t3_chi_q = a.indices[3].charges if _sym else None
    _t3_D2_q = env.T3.indices[1].charges if _sym else None
    T3_new = _wrap_standard_edge_dense(
        T3_new_dense,
        "t3_r",
        "d2",
        "t3_l",
        env.T3.indices,
        chi,
        D2,
        symmetric=_sym,
        base_chi_charges=_t3_chi_q,
        base_D2_charges=_t3_D2_q,
    )

    return env._replace(C4=C4_new, C3=C3_new, T3=T3_new)


# ------------------------------------------------------------------ #
# Sweep + renormalize                                                  #
# ------------------------------------------------------------------ #


def _normalize_tensor(T: Tensor) -> Tensor:
    """Normalize tensor by max abs value, matching dense CTM convention.

    Uses native Tensor ops (``max_abs()``, scalar ``*``) so this is
    fully differentiable for both DenseTensor and SymmetricTensor
    without any ``todense()`` round-trip.
    """
    norm = T.max_abs()
    return T * (1.0 / (norm + EPS))


def _renormalize_tensor_env(env: CTMTensorEnv) -> CTMTensorEnv:
    """Normalize all environment tensors to prevent exponential growth."""
    return CTMTensorEnv(
        C1=_normalize_tensor(env.C1),
        C2=_normalize_tensor(env.C2),
        C3=_normalize_tensor(env.C3),
        C4=_normalize_tensor(env.C4),
        T1=_normalize_tensor(env.T1),
        T2=_normalize_tensor(env.T2),
        T3=_normalize_tensor(env.T3),
        T4=_normalize_tensor(env.T4),
    )


def _ctm_tensor_sweep(
    env: CTMTensorEnv,
    a: Tensor,
    chi: int,
    renormalize: bool,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """One full CTM sweep: left, right, top, bottom + optional renormalize."""
    env = _ctm_tensor_move_left(env, a, chi, projector_method)
    env = _ctm_tensor_move_right(env, a, chi, projector_method)
    env = _ctm_tensor_move_top(env, a, chi, projector_method)
    env = _ctm_tensor_move_bottom(env, a, chi, projector_method)
    if renormalize:
        env = _renormalize_tensor_env(env)
    return env


# ------------------------------------------------------------------ #
# Main entry: convergence loop                                         #
# ------------------------------------------------------------------ #


def _ctm_sv_diff(sv_new: jax.Array, sv_old: jax.Array) -> jax.Array:
    """Compute max absolute difference between normalized singular value vectors."""
    sv1 = sv_new / (jnp.sum(sv_new) + 1e-15)
    sv2 = sv_old / (jnp.sum(sv_old) + 1e-15)
    return jnp.max(jnp.abs(sv1 - sv2))


def ctm_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Run standard CTM to convergence using the Tensor protocol.

    Builds the full double-layer tensor via ``bar()`` + ``contract()`` +
    ``fuse_indices()``, then iterates CTM moves until the corner singular
    values converge.

    Args:
        A:                 iPEPS site tensor (DenseTensor or SymmetricTensor)
                           with 5 legs ``(u, d, l, r, phys)``.
        chi:               Environment bond dimension.
        max_iter:          Maximum CTM iterations.
        conv_tol:          Convergence tolerance on corner singular values.
        renormalize:       Renormalize environment at each step.
        projector_method:  ``"eigh"`` or ``"qr"``.

    Returns:
        Converged CTMTensorEnv.
    """
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    prev_sv = None
    for _ in range(max_iter):
        env = _ctm_tensor_sweep(env, a, chi, renormalize, projector_method)

        current_sv = jnp.linalg.svd(env.C1.todense(), compute_uv=False)
        if prev_sv is not None:
            diff = _ctm_sv_diff(current_sv, prev_sv)
            if float(diff) < conv_tol:
                break
        prev_sv = current_sv

    return env


# ------------------------------------------------------------------ #
# RDMs + energy                                                        #
# ------------------------------------------------------------------ #


def _env_to_dense_standard(env: CTMTensorEnv):
    """Convert CTMTensorEnv to dense CTMEnvironment for RDM computation."""
    from tenax.algorithms.ipeps import CTMEnvironment

    return CTMEnvironment(
        C1=env.C1.todense(),
        C2=env.C2.todense(),
        C3=env.C3.todense(),
        C4=env.C4.todense(),
        T1=env.T1.todense(),
        T2=env.T2.todense(),
        T3=env.T3.todense(),
        T4=env.T4.todense(),
    )


def compute_energy_ctm_tensor(
    A: Tensor,
    env: CTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site using a standard Tensor-protocol CTM environment.

    Converts to dense CTMEnvironment and delegates to the existing
    RDM-based energy computation.

    Args:
        A:                iPEPS site tensor.
        env:              Converged CTMTensorEnv.
        hamiltonian_gate: 2-site Hamiltonian gate.
        d:                Physical dimension (inferred from A if None).

    Returns:
        Scalar energy per site.
    """
    from tenax.algorithms.ipeps import compute_energy_ctm

    A_dense = A.todense()
    if d is None:
        d = A_dense.shape[-1]

    if isinstance(hamiltonian_gate, Tensor):
        H = hamiltonian_gate.todense().reshape(d, d, d, d)
    else:
        H = hamiltonian_gate.reshape(d, d, d, d)

    std_env = _env_to_dense_standard(env)
    return compute_energy_ctm(A_dense, std_env, H, d)
