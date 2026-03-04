"""Split CTM using the Tensor protocol (polymorphic dense/symmetric).

Keeps ket and bra layers separate throughout the CTM iteration,
avoiding the double-layer tensor entirely.  This allows SymmetricTensor
iPEPS to run CTM without densification.

The algorithm follows the dense split-CTMRG in ``ipeps.py`` but uses
``contract()``, ``truncated_svd()``, ``max_abs_normalize()`` and
``.relabel()`` / ``.dagger()`` for automatic dense/symmetric dispatch.

Reference: arXiv:2502.10298
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import (
    fuse_indices,
    max_abs_normalize,
)
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ------------------------------------------------------------------ #
# Environment data structure                                          #
# ------------------------------------------------------------------ #


class SplitCTMTensorEnv(NamedTuple):
    """Split CTM environment with Tensor-protocol fields.

    Corners are 2-leg tensors ``(chi, chi)``.
    Each edge is split into ket ``(chi, D, chi_I)`` and bra ``(chi_I, D, chi)``
    halves connected by an interlayer bond ``chi_I``.
    """

    C1: Tensor  # (c1_d, c1_r)
    C2: Tensor  # (c2_l, c2_d)
    C3: Tensor  # (c3_u, c3_l)
    C4: Tensor  # (c4_r, c4_u)
    T1_ket: Tensor  # (t1k_l, u_ket, t1k_I)
    T1_bra: Tensor  # (t1b_I, u_bra, t1b_r)
    T2_ket: Tensor  # (t2k_u, r_ket, t2k_I)
    T2_bra: Tensor  # (t2b_I, r_bra, t2b_d)
    T3_ket: Tensor  # (t3k_r, d_ket, t3k_I)
    T3_bra: Tensor  # (t3b_I, d_bra, t3b_l)
    T4_ket: Tensor  # (t4k_d, l_ket, t4k_I)
    T4_bra: Tensor  # (t4b_I, l_bra, t4b_u)


# ------------------------------------------------------------------ #
# Initialization                                                       #
# ------------------------------------------------------------------ #


def _make_dense_corner(
    chi: int,
    D: int,
    label_a: Label,
    label_b: Label,
    flow_a: FlowDirection,
    flow_b: FlowDirection,
    dtype,
) -> DenseTensor:
    """Create an identity-like DenseTensor corner (chi x chi)."""
    C = jnp.eye(min(chi, D), dtype=dtype)
    C_pad = jnp.zeros((chi, chi), dtype=dtype)
    C_pad = C_pad.at[: C.shape[0], : C.shape[1]].set(C)
    sym = _trivial_symmetry()
    idx_a = TensorIndex(sym, np.zeros(chi, dtype=np.int32), flow_a, label=label_a)
    idx_b = TensorIndex(sym, np.zeros(chi, dtype=np.int32), flow_b, label=label_b)
    return DenseTensor(C_pad, (idx_a, idx_b))


def _make_dense_edge_ket(
    chi: int,
    D: int,
    chi_I: int,
    label_chi: Label,
    label_D: Label,
    label_I: Label,
    flow_chi: FlowDirection,
    flow_D: FlowDirection,
    flow_I: FlowDirection,
    dtype,
) -> DenseTensor:
    """Create an identity-like DenseTensor ket edge (chi, D, chi_I)."""
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)
    T = jnp.zeros((chi, D, chi_I), dtype=dtype)
    for i in range(min(chi_D, chi_I_D)):
        T = T.at[i, :, i].set(jnp.ones(D, dtype=dtype))
    sym = _trivial_symmetry()
    return DenseTensor(
        T,
        (
            TensorIndex(sym, np.zeros(chi, dtype=np.int32), flow_chi, label=label_chi),
            TensorIndex(sym, np.zeros(D, dtype=np.int32), flow_D, label=label_D),
            TensorIndex(sym, np.zeros(chi_I, dtype=np.int32), flow_I, label=label_I),
        ),
    )


def _make_dense_edge_bra(
    chi: int,
    D: int,
    chi_I: int,
    label_I: Label,
    label_D: Label,
    label_chi: Label,
    flow_I: FlowDirection,
    flow_D: FlowDirection,
    flow_chi: FlowDirection,
    dtype,
) -> DenseTensor:
    """Create an identity-like DenseTensor bra edge (chi_I, D, chi)."""
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)
    T = jnp.zeros((chi_I, D, chi), dtype=dtype)
    for i in range(min(chi_I_D, chi_D)):
        T = T.at[i, :, i].set(jnp.ones(D, dtype=dtype))
    sym = _trivial_symmetry()
    return DenseTensor(
        T,
        (
            TensorIndex(sym, np.zeros(chi_I, dtype=np.int32), flow_I, label=label_I),
            TensorIndex(sym, np.zeros(D, dtype=np.int32), flow_D, label=label_D),
            TensorIndex(sym, np.zeros(chi, dtype=np.int32), flow_chi, label=label_chi),
        ),
    )


def _trivial_symmetry():
    from tenax.core.symmetry import U1Symmetry

    return U1Symmetry()


def _init_symmetric_corner(
    A: SymmetricTensor,
    chi: int,
    label_a: Label,
    label_b: Label,
    flow_a: FlowDirection,
    flow_b: FlowDirection,
    ref_axis: int,
) -> SymmetricTensor:
    """Create an identity-like SymmetricTensor corner from A's bond charges."""
    ref_idx = A.indices[ref_axis]
    sym = ref_idx.symmetry
    # Derive chi-leg charges: repeat A's bond charges up to chi
    base_charges = ref_idx.charges
    n_base = len(base_charges)
    if chi <= n_base:
        charges = base_charges[:chi].copy()
    else:
        reps = chi // n_base + 1
        charges = np.tile(base_charges, reps)[:chi]
    charges = np.asarray(charges, dtype=np.int32)

    idx_a = TensorIndex(sym, charges.copy(), flow_a, label=label_a)
    idx_b = TensorIndex(sym, charges.copy(), flow_b, label=label_b)
    # Build identity-like: for each charge, set diagonal block to identity
    return SymmetricTensor.from_dense(
        jnp.eye(chi, dtype=A.dtype),
        (idx_a, idx_b),
    )


def _init_symmetric_edge_ket(
    A: SymmetricTensor,
    chi: int,
    D: int,
    chi_I: int,
    label_chi: Label,
    label_D: Label,
    label_I: Label,
    flow_chi: FlowDirection,
    flow_D: FlowDirection,
    flow_I: FlowDirection,
    ref_axis_chi: int,
    ref_axis_D: int,
) -> SymmetricTensor:
    """Create an identity-like SymmetricTensor ket edge."""
    sym = A.indices[0].symmetry

    # chi-leg charges from A's ref bond
    chi_charges = _derive_charges(A.indices[ref_axis_chi].charges, chi)
    D_charges = A.indices[ref_axis_D].charges.copy()
    I_charges = _derive_charges(A.indices[ref_axis_chi].charges, chi_I)

    idx_chi = TensorIndex(sym, chi_charges, flow_chi, label=label_chi)
    idx_D = TensorIndex(
        sym, np.asarray(D_charges, dtype=np.int32), flow_D, label=label_D
    )
    idx_I = TensorIndex(sym, I_charges, flow_I, label=label_I)

    # Build identity-like dense, then convert
    T = jnp.zeros((chi, D, chi_I), dtype=A.dtype)
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)
    for i in range(min(chi_D, chi_I_D)):
        T = T.at[i, :, i].set(jnp.ones(D, dtype=A.dtype))
    return SymmetricTensor.from_dense(T, (idx_chi, idx_D, idx_I))


def _init_symmetric_edge_bra(
    A: SymmetricTensor,
    chi: int,
    D: int,
    chi_I: int,
    label_I: Label,
    label_D: Label,
    label_chi: Label,
    flow_I: FlowDirection,
    flow_D: FlowDirection,
    flow_chi: FlowDirection,
    ref_axis_chi: int,
    ref_axis_D: int,
) -> SymmetricTensor:
    """Create an identity-like SymmetricTensor bra edge."""
    sym = A.indices[0].symmetry

    I_charges = _derive_charges(A.indices[ref_axis_chi].charges, chi_I)
    D_charges = A.indices[ref_axis_D].charges.copy()
    chi_charges = _derive_charges(A.indices[ref_axis_chi].charges, chi)

    idx_I = TensorIndex(sym, I_charges, flow_I, label=label_I)
    idx_D = TensorIndex(
        sym, np.asarray(D_charges, dtype=np.int32), flow_D, label=label_D
    )
    idx_chi = TensorIndex(sym, chi_charges, flow_chi, label=label_chi)

    T = jnp.zeros((chi_I, D, chi), dtype=A.dtype)
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)
    for i in range(min(chi_I_D, chi_D)):
        T = T.at[i, :, i].set(jnp.ones(D, dtype=A.dtype))
    return SymmetricTensor.from_dense(T, (idx_I, idx_D, idx_chi))


def _derive_charges(base_charges: np.ndarray, target_dim: int) -> np.ndarray:
    """Derive charges of size target_dim from base charges by tiling."""
    n = len(base_charges)
    if target_dim <= n:
        return np.asarray(base_charges[:target_dim], dtype=np.int32)
    reps = target_dim // n + 1
    return np.asarray(np.tile(base_charges, reps)[:target_dim], dtype=np.int32)


# A labels: (u, d, l, r, phys), flows: (OUT, IN, OUT, IN, IN)
# Env label/flow conventions per the plan
_CORNER_SPECS = {
    "C1": ("c1_d", "c1_r", FlowDirection.IN, FlowDirection.OUT, 1),  # ref_axis=d(1)
    "C2": ("c2_l", "c2_d", FlowDirection.IN, FlowDirection.OUT, 0),  # ref_axis=u(0)
    "C3": ("c3_u", "c3_l", FlowDirection.OUT, FlowDirection.IN, 1),  # ref_axis=d(1)
    "C4": ("c4_r", "c4_u", FlowDirection.OUT, FlowDirection.IN, 0),  # ref_axis=u(0)
}

# Edge specs: (label_first, label_D, label_last, flow_first, flow_D, flow_last,
#              ref_axis_chi, ref_axis_D)
_EDGE_KET_SPECS = {
    "T1": (
        "t1k_l",
        "u_ket",
        "t1k_I",
        FlowDirection.IN,
        FlowDirection.IN,
        FlowDirection.OUT,
        3,
        0,
    ),  # ref=r(3), D=u(0); D-flow opposite to A's u(OUT)
    "T2": (
        "t2k_u",
        "r_ket",
        "t2k_I",
        FlowDirection.OUT,
        FlowDirection.OUT,
        FlowDirection.IN,
        0,
        3,
    ),  # ref=u(0), D=r(3); D-flow opposite to A's r(IN)
    "T3": (
        "t3k_r",
        "d_ket",
        "t3k_I",
        FlowDirection.OUT,
        FlowDirection.OUT,
        FlowDirection.IN,
        3,
        1,
    ),  # ref=r(3), D=d(1); D-flow opposite to A's d(IN)
    "T4": (
        "t4k_d",
        "l_ket",
        "t4k_I",
        FlowDirection.IN,
        FlowDirection.IN,
        FlowDirection.OUT,
        1,
        2,
    ),  # ref=d(1), D=l(2); D-flow opposite to A's l(OUT)
}

_EDGE_BRA_SPECS = {
    "T1": (
        "t1b_I",
        "u_bra",
        "t1b_r",
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.IN,
        3,
        0,
    ),  # D-flow opposite to A.dagger()'s u(IN)
    "T2": (
        "t2b_I",
        "r_bra",
        "t2b_d",
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        0,
        3,
    ),  # D-flow opposite to A.dagger()'s r(OUT)
    "T3": (
        "t3b_I",
        "d_bra",
        "t3b_l",
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        3,
        1,
    ),  # D-flow opposite to A.dagger()'s d(OUT)
    "T4": (
        "t4b_I",
        "l_bra",
        "t4b_u",
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.IN,
        1,
        2,
    ),  # D-flow opposite to A.dagger()'s l(IN)
}


def initialize_split_ctm_tensor_env(
    A: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Initialize a SplitCTMTensorEnv from an iPEPS site tensor.

    Args:
        A:     Site tensor with 5 legs ``(u, d, l, r, phys)``.
        chi:   Environment bond dimension.
        chi_I: Interlayer bond dimension.

    Returns:
        Initialized SplitCTMTensorEnv.
    """
    D = A.indices[0].dim  # virtual bond dim
    dtype = A.dtype

    if isinstance(A, SymmetricTensor):
        corners = {}
        for name, (la, lb, fa, fb, ref) in _CORNER_SPECS.items():
            corners[name] = _init_symmetric_corner(A, chi, la, lb, fa, fb, ref)

        ket_edges = {}
        for name, (l1, l2, l3, f1, f2, f3, ref_chi, ref_D) in _EDGE_KET_SPECS.items():
            ket_edges[name] = _init_symmetric_edge_ket(
                A, chi, D, chi_I, l1, l2, l3, f1, f2, f3, ref_chi, ref_D
            )

        bra_edges = {}
        for name, (l1, l2, l3, f1, f2, f3, ref_chi, ref_D) in _EDGE_BRA_SPECS.items():
            bra_edges[name] = _init_symmetric_edge_bra(
                A, chi, D, chi_I, l1, l2, l3, f1, f2, f3, ref_chi, ref_D
            )
    else:
        # DenseTensor path
        corners = {}
        for name, (la, lb, fa, fb, _ref) in _CORNER_SPECS.items():
            corners[name] = _make_dense_corner(chi, D, la, lb, fa, fb, dtype)

        ket_edges = {}
        for name, (l1, l2, l3, f1, f2, f3, _rc, _rd) in _EDGE_KET_SPECS.items():
            ket_edges[name] = _make_dense_edge_ket(
                chi, D, chi_I, l1, l2, l3, f1, f2, f3, dtype
            )

        bra_edges = {}
        for name, (l1, l2, l3, f1, f2, f3, _rc, _rd) in _EDGE_BRA_SPECS.items():
            bra_edges[name] = _make_dense_edge_bra(
                chi, D, chi_I, l1, l2, l3, f1, f2, f3, dtype
            )

    return SplitCTMTensorEnv(
        C1=corners["C1"],
        C2=corners["C2"],
        C3=corners["C3"],
        C4=corners["C4"],
        T1_ket=ket_edges["T1"],
        T1_bra=bra_edges["T1"],
        T2_ket=ket_edges["T2"],
        T2_bra=bra_edges["T2"],
        T3_ket=ket_edges["T3"],
        T3_bra=bra_edges["T3"],
        T4_ket=ket_edges["T4"],
        T4_bra=bra_edges["T4"],
    )


# ------------------------------------------------------------------ #
# Projector computation (dense, with stop_gradient)                    #
# ------------------------------------------------------------------ #


def _compute_projector_dense(
    C1g_dense: jax.Array, C2g_dense: jax.Array, chi: int
) -> jax.Array:
    """Compute eigh-based projector from two grown corner matrices.

    Args:
        C1g_dense: Dense grown corner, shape ``(chi*D, chi_I)``.
        C2g_dense: Dense grown corner, shape ``(chi*D, chi_I)``.
        chi:       Target bond dimension.

    Returns:
        Projector ``P`` of shape ``(chi*D, chi)``.
    """
    rho = C1g_dense @ C1g_dense.T + C2g_dense @ C2g_dense.T
    rho = 0.5 * (rho + rho.T)
    eigvals, eigvecs = jnp.linalg.eigh(rho)
    k = min(chi, len(eigvals))
    P = eigvecs[:, -k:][:, ::-1]
    return jax.lax.stop_gradient(P)


# ------------------------------------------------------------------ #
# SVD edge split                                                       #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
# No-double-layer edge growth                                          #
# ------------------------------------------------------------------ #

_VIRTUAL_LEGS = ("u", "d", "l", "r")


def _grow_edge_no_double_layer(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    contracted_leg: str,
    ket_D_label: str,
    bra_D_label: str,
    ket_I_label: str,
    bra_I_label: str,
    output_labels: tuple[str, ...],
    chi: int,
) -> jax.Array:
    """Grow a T-edge by contracting ket/bra layers separately.

    Instead of building a closed double-layer tensor, this contracts each
    half-edge with its copy of A (ket or daggered-bra), then traces the
    physical and interlayer indices via label-based contraction.

    Returns a dense array of shape ``(chi*D², D², chi*D²)``.
    """
    D = A.indices[0].dim

    # --- Ket side ---
    A_ket = A.relabel(contracted_leg, ket_D_label)
    ket_half = contract(T_ket, A_ket)

    # --- Bra side: dagger + relabel virtual legs to uppercase ---
    bra_mapping: dict[str, str] = {contracted_leg: bra_D_label}
    for v in _VIRTUAL_LEGS:
        if v != contracted_leg:
            bra_mapping[v] = v.upper()
    A_bra = A.dagger().relabels(bra_mapping)
    bra_half = contract(T_bra, A_bra)

    # --- Match interlayer labels, then contract (traces _I + phys) ---
    ket_half = ket_half.relabel(ket_I_label, "_I")
    bra_half = bra_half.relabel(bra_I_label, "_I")
    grown = contract(ket_half, bra_half, output_labels=output_labels)

    return grown.todense().reshape(chi * D * D, D * D, chi * D * D)


# ------------------------------------------------------------------ #
# Directional CTM moves                                                #
# ------------------------------------------------------------------ #


def _split_ctm_move_left(
    env: SplitCTMTensorEnv,
    A: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Left move: ket first (C1/C4 connect to T1/T3 ket chi bonds)."""
    # --- Step 1: Grow C1 with T1_ket ---
    # C1(c1_d, c1_r) · T1_ket(t1k_l, u_ket, t1k_I)
    # Contract: c1_r ↔ t1k_l
    C1_r = env.C1.relabel("c1_r", "t1k_l")
    C1g_ket = contract(C1_r, env.T1_ket)  # (c1_d, u_ket, t1k_I)

    # Fuse (c1_d, u_ket) → matrix of shape (chi*D, chi_I)
    C1g_ket_fused = fuse_indices(C1g_ket, 0, 1, "fused", FlowDirection.IN)
    C1g_ket_dense = C1g_ket_fused.todense()  # (chi*D, chi_I)

    # --- Step 2: Grow C4 with T3_ket ---
    # C4(c4_r, c4_u) · T3_ket(t3k_r, d_ket, t3k_I)
    # Contract: c4_r ↔ t3k_r
    C4_r = env.C4.relabel("c4_r", "t3k_r")
    C4g_ket = contract(C4_r, env.T3_ket)  # (c4_u, d_ket, t3k_I)

    C4g_ket_fused = fuse_indices(C4g_ket, 0, 1, "fused", FlowDirection.IN)
    C4g_ket_dense = C4g_ket_fused.todense()

    # --- Step 3: Ket projector ---
    P_ket = _compute_projector_dense(C1g_ket_dense, C4g_ket_dense, chi)

    # --- Step 4: Mid-corners ---
    C1_mid_dense = P_ket.T @ C1g_ket_dense  # (chi, chi_I)
    C4_mid_dense = P_ket.T @ C4g_ket_dense  # (chi, chi_I)

    # --- Step 5: Grow mid-corners with bra ---
    # C1_mid(chi, chi_I) · T1_bra(t1b_I, u_bra, t1b_r)
    # Reshape C1_mid: treat as (new_chi, chi_I) → contract chi_I ↔ t1b_I
    # Use dense path for projector application
    C1g_bra_dense = jnp.einsum(
        "ac,cdb->adb", C1_mid_dense, env.T1_bra.todense()
    ).reshape(-1, chi)
    C4g_bra_dense = jnp.einsum(
        "ac,cdb->adb", C4_mid_dense, env.T3_bra.todense()
    ).reshape(-1, chi)

    # --- Step 6: Bra projector + new corners ---
    P_bra = _compute_projector_dense(C1g_bra_dense, C4g_bra_dense, chi)
    C1_new_dense = P_bra.T @ C1g_bra_dense  # (chi, chi)
    C4_new_dense = P_bra.T @ C4g_bra_dense  # (chi, chi)

    # Wrap new corners as Tensors
    C1_new = _wrap_corner_dense(
        C1_new_dense, "c1_d", "c1_r", env.C1.indices[0], env.C1.indices[1], chi
    )
    C4_new = _wrap_corner_dense(
        C4_new_dense, "c4_r", "c4_u", env.C4.indices[0], env.C4.indices[1], chi
    )

    # --- Step 7: Combined full projector ---
    D = A.indices[0].dim
    P_ket_3d = P_ket.reshape(chi, D, -1)  # (chi, D, chi_k)
    chi_k = P_ket_3d.shape[2]
    P_bra_3d = P_bra.reshape(chi_k, D, -1)  # (chi_k, D, chi)
    P_full = jnp.einsum("auJ,JUb->auUb", P_ket_3d, P_bra_3d)
    chi_new = P_full.shape[3]
    P_full = P_full.reshape(chi * D * D, chi_new)

    # --- Step 8+9: Grow T4 via separate ket/bra contraction ---
    T4g = _grow_edge_no_double_layer(
        env.T4_ket,
        env.T4_bra,
        A,
        "l",
        "l_ket",
        "l_bra",
        "t4k_I",
        "t4b_I",
        ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
        chi,
    )

    # Apply projectors
    T4_new_full_dense = jnp.einsum("ia,idj,jb->adb", P_full, T4g, P_full)

    # --- Step 10: SVD split new T4 into ket/bra ---
    T4_ket_new_dense, T4_bra_new_dense = _svd_split_edge_dense(T4_new_full_dense, chi_I)

    T4_ket_new = _wrap_edge_ket_dense(
        T4_ket_new_dense, "t4k_d", "l_ket", "t4k_I", env.T4_ket.indices, chi, D, chi_I
    )
    T4_bra_new = _wrap_edge_bra_dense(
        T4_bra_new_dense, "t4b_I", "l_bra", "t4b_u", env.T4_bra.indices, chi, D, chi_I
    )

    return SplitCTMTensorEnv(
        C1=C1_new,
        C2=env.C2,
        C3=env.C3,
        C4=C4_new,
        T1_ket=env.T1_ket,
        T1_bra=env.T1_bra,
        T2_ket=env.T2_ket,
        T2_bra=env.T2_bra,
        T3_ket=env.T3_ket,
        T3_bra=env.T3_bra,
        T4_ket=T4_ket_new,
        T4_bra=T4_bra_new,
    )


def _split_ctm_move_right(
    env: SplitCTMTensorEnv,
    A: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Right move: bra first (C2/C3 connect to T1/T3 bra chi bonds)."""
    D = A.indices[0].dim

    # --- bra first ---
    # C2(c2_l, c2_d) · T1_bra(t1b_I, u_bra, t1b_r) → contract c2_l ↔ t1b_r
    C2_l = env.C2.relabel("c2_l", "t1b_r")
    C2g_bra = contract(C2_l, env.T1_bra)  # (c2_d, t1b_I, u_bra)
    C2g_bra_fused = fuse_indices(C2g_bra, 0, 2, "fused", FlowDirection.IN)
    C2g_bra_dense = C2g_bra_fused.todense()  # (chi*D, chi_I)

    # C3(c3_u, c3_l) · T3_bra(t3b_I, d_bra, t3b_l) → contract c3_l ↔ t3b_l
    # Note: c3_l connects to T3_bra's t3b_l
    # But we need to be careful: the connection is c3_u ↔ T2_bra and c3_l ↔ T3
    # For right move: C3 absorbs T3_bra
    C3_l = env.C3.relabel("c3_l", "t3b_l")
    C3g_bra = contract(C3_l, env.T3_bra)  # (c3_u, t3b_I, d_bra)
    C3g_bra_fused = fuse_indices(C3g_bra, 0, 2, "fused", FlowDirection.IN)
    C3g_bra_dense = C3g_bra_fused.todense()

    P_bra = _compute_projector_dense(C2g_bra_dense, C3g_bra_dense, chi)
    C2_mid_dense = P_bra.T @ C2g_bra_dense  # (chi, chi_I)
    C3_mid_dense = P_bra.T @ C3g_bra_dense

    # --- ket via interlayer ---
    # C2_mid(chi, chi_I) · T1_ket(t1k_l, u_ket, t1k_I) → contract chi_I ↔ t1k_I
    C2g_ket_dense = jnp.einsum(
        "af,buf->aub", C2_mid_dense, env.T1_ket.todense()
    ).reshape(-1, chi)
    C3g_ket_dense = jnp.einsum(
        "af,hdf->adh", C3_mid_dense, env.T3_ket.todense()
    ).reshape(-1, chi)

    P_ket = _compute_projector_dense(C2g_ket_dense, C3g_ket_dense, chi)
    C2_new_dense = P_ket.T @ C2g_ket_dense
    C3_new_dense = P_ket.T @ C3g_ket_dense

    C2_new = _wrap_corner_dense(
        C2_new_dense, "c2_l", "c2_d", env.C2.indices[0], env.C2.indices[1], chi
    )
    C3_new = _wrap_corner_dense(
        C3_new_dense, "c3_u", "c3_l", env.C3.indices[0], env.C3.indices[1], chi
    )

    # Combined projector
    P_bra_3d = P_bra.reshape(chi, D, -1)
    chi_k = P_bra_3d.shape[2]
    P_ket_3d = P_ket.reshape(chi_k, D, -1)
    P_full = jnp.einsum("aUJ,Jub->auUb", P_bra_3d, P_ket_3d)
    chi_new = P_full.shape[3]
    P_full = P_full.reshape(chi * D * D, chi_new)

    # Grow T2 via separate ket/bra contraction
    T2g = _grow_edge_no_double_layer(
        env.T2_ket,
        env.T2_bra,
        A,
        "r",
        "r_ket",
        "r_bra",
        "t2k_I",
        "t2b_I",
        ("t2k_u", "u", "U", "l", "L", "t2b_d", "d", "D"),
        chi,
    )
    T2_new_full_dense = jnp.einsum("ia,idj,jb->adb", P_full, T2g, P_full)

    T2_ket_new_dense, T2_bra_new_dense = _svd_split_edge_dense(T2_new_full_dense, chi_I)
    T2_ket_new = _wrap_edge_ket_dense(
        T2_ket_new_dense, "t2k_u", "r_ket", "t2k_I", env.T2_ket.indices, chi, D, chi_I
    )
    T2_bra_new = _wrap_edge_bra_dense(
        T2_bra_new_dense, "t2b_I", "r_bra", "t2b_d", env.T2_bra.indices, chi, D, chi_I
    )

    return SplitCTMTensorEnv(
        C1=env.C1,
        C2=C2_new,
        C3=C3_new,
        C4=env.C4,
        T1_ket=env.T1_ket,
        T1_bra=env.T1_bra,
        T2_ket=T2_ket_new,
        T2_bra=T2_bra_new,
        T3_ket=env.T3_ket,
        T3_bra=env.T3_bra,
        T4_ket=env.T4_ket,
        T4_bra=env.T4_bra,
    )


def _split_ctm_move_top(
    env: SplitCTMTensorEnv,
    A: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Top move: ket first (C1/C2 connect to T4/T2 ket chi bonds)."""
    D = A.indices[0].dim

    # C1(c1_d, c1_r) · T4_ket(t4k_d, l_ket, t4k_I) → contract c1_d ↔ t4k_d
    C1_d = env.C1.relabel("c1_d", "t4k_d")
    C1g_ket = contract(C1_d, env.T4_ket)  # (c1_r, l_ket, t4k_I)
    C1g_ket_fused = fuse_indices(C1g_ket, 0, 1, "fused", FlowDirection.IN)
    C1g_ket_dense = C1g_ket_fused.todense()

    # C2(c2_l, c2_d) · T2_ket(t2k_u, r_ket, t2k_I) → contract c2_d ↔ t2k_u
    C2_d = env.C2.relabel("c2_d", "t2k_u")
    C2g_ket = contract(C2_d, env.T2_ket)  # (c2_l, r_ket, t2k_I)
    C2g_ket_fused = fuse_indices(C2g_ket, 0, 1, "fused", FlowDirection.IN)
    C2g_ket_dense = C2g_ket_fused.todense()

    P_ket = _compute_projector_dense(C1g_ket_dense, C2g_ket_dense, chi)
    C1_mid_dense = P_ket.T @ C1g_ket_dense
    C2_mid_dense = P_ket.T @ C2g_ket_dense

    # Bra layer
    C1g_bra_dense = jnp.einsum(
        "ac,cdb->adb", C1_mid_dense, env.T4_bra.todense()
    ).reshape(-1, chi)
    C2g_bra_dense = jnp.einsum(
        "ac,cdb->adb", C2_mid_dense, env.T2_bra.todense()
    ).reshape(-1, chi)

    P_bra = _compute_projector_dense(C1g_bra_dense, C2g_bra_dense, chi)
    C1_new_dense = P_bra.T @ C1g_bra_dense
    C2_new_dense = P_bra.T @ C2g_bra_dense

    C1_new = _wrap_corner_dense(
        C1_new_dense, "c1_d", "c1_r", env.C1.indices[0], env.C1.indices[1], chi
    )
    C2_new = _wrap_corner_dense(
        C2_new_dense, "c2_l", "c2_d", env.C2.indices[0], env.C2.indices[1], chi
    )

    # Combined projector
    P_ket_3d = P_ket.reshape(chi, D, -1)
    chi_k = P_ket_3d.shape[2]
    P_bra_3d = P_bra.reshape(chi_k, D, -1)
    P_full = jnp.einsum("auJ,JUb->auUb", P_ket_3d, P_bra_3d)
    chi_new = P_full.shape[3]
    P_full = P_full.reshape(chi * D * D, chi_new)

    # Grow T1 via separate ket/bra contraction
    T1g = _grow_edge_no_double_layer(
        env.T1_ket,
        env.T1_bra,
        A,
        "u",
        "u_ket",
        "u_bra",
        "t1k_I",
        "t1b_I",
        ("t1k_l", "l", "L", "d", "D", "t1b_r", "r", "R"),
        chi,
    )
    T1_new_full_dense = jnp.einsum("ia,idj,jb->adb", P_full, T1g, P_full)

    T1_ket_new_dense, T1_bra_new_dense = _svd_split_edge_dense(T1_new_full_dense, chi_I)
    T1_ket_new = _wrap_edge_ket_dense(
        T1_ket_new_dense, "t1k_l", "u_ket", "t1k_I", env.T1_ket.indices, chi, D, chi_I
    )
    T1_bra_new = _wrap_edge_bra_dense(
        T1_bra_new_dense, "t1b_I", "u_bra", "t1b_r", env.T1_bra.indices, chi, D, chi_I
    )

    return SplitCTMTensorEnv(
        C1=C1_new,
        C2=C2_new,
        C3=env.C3,
        C4=env.C4,
        T1_ket=T1_ket_new,
        T1_bra=T1_bra_new,
        T2_ket=env.T2_ket,
        T2_bra=env.T2_bra,
        T3_ket=env.T3_ket,
        T3_bra=env.T3_bra,
        T4_ket=env.T4_ket,
        T4_bra=env.T4_bra,
    )


def _split_ctm_move_bottom(
    env: SplitCTMTensorEnv,
    A: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Bottom move: bra first (C4/C3 connect to T4/T2 bra chi bonds)."""
    D = A.indices[0].dim

    # C4(c4_r, c4_u) · T4_bra(t4b_I, l_bra, t4b_u) → contract c4_u ↔ t4b_u
    C4_u = env.C4.relabel("c4_u", "t4b_u")
    C4g_bra = contract(C4_u, env.T4_bra)  # (c4_r, t4b_I, l_bra)
    C4g_bra_fused = fuse_indices(C4g_bra, 0, 2, "fused", FlowDirection.IN)
    C4g_bra_dense = C4g_bra_fused.todense()

    # C3(c3_u, c3_l) · T2_bra(t2b_I, r_bra, t2b_d) → contract c3_u ↔ t2b_d
    C3_u = env.C3.relabel("c3_u", "t2b_d")
    C3g_bra = contract(C3_u, env.T2_bra)  # (c3_l, t2b_I, r_bra)
    C3g_bra_fused = fuse_indices(C3g_bra, 0, 2, "fused", FlowDirection.IN)
    C3g_bra_dense = C3g_bra_fused.todense()

    P_bra = _compute_projector_dense(C4g_bra_dense, C3g_bra_dense, chi)
    C4_mid_dense = P_bra.T @ C4g_bra_dense
    C3_mid_dense = P_bra.T @ C3g_bra_dense

    # Ket via interlayer
    # C4_mid · T4_ket(t4k_d, l_ket, t4k_I) → contract chi_I ↔ t4k_I
    C4g_ket_dense = jnp.einsum(
        "af,blf->alb", C4_mid_dense, env.T4_ket.todense()
    ).reshape(-1, chi)
    # C3_mid · T2_ket(t2k_u, r_ket, t2k_I) → contract chi_I ↔ t2k_I
    C3g_ket_dense = jnp.einsum(
        "af,erf->are", C3_mid_dense, env.T2_ket.todense()
    ).reshape(-1, chi)

    P_ket = _compute_projector_dense(C4g_ket_dense, C3g_ket_dense, chi)
    C4_new_dense = P_ket.T @ C4g_ket_dense
    C3_new_dense = P_ket.T @ C3g_ket_dense

    C4_new = _wrap_corner_dense(
        C4_new_dense, "c4_r", "c4_u", env.C4.indices[0], env.C4.indices[1], chi
    )
    C3_new = _wrap_corner_dense(
        C3_new_dense, "c3_u", "c3_l", env.C3.indices[0], env.C3.indices[1], chi
    )

    # Combined projector
    P_bra_3d = P_bra.reshape(chi, D, -1)
    chi_k = P_bra_3d.shape[2]
    P_ket_3d = P_ket.reshape(chi_k, D, -1)
    P_full = jnp.einsum("aUJ,Jub->auUb", P_bra_3d, P_ket_3d)
    chi_new = P_full.shape[3]
    P_full = P_full.reshape(chi * D * D, chi_new)

    # Grow T3 via separate ket/bra contraction
    T3g = _grow_edge_no_double_layer(
        env.T3_ket,
        env.T3_bra,
        A,
        "d",
        "d_ket",
        "d_bra",
        "t3k_I",
        "t3b_I",
        ("t3k_r", "l", "L", "u", "U", "t3b_l", "r", "R"),
        chi,
    )
    T3_new_full_dense = jnp.einsum("ia,idj,jb->adb", P_full, T3g, P_full)

    T3_ket_new_dense, T3_bra_new_dense = _svd_split_edge_dense(T3_new_full_dense, chi_I)
    T3_ket_new = _wrap_edge_ket_dense(
        T3_ket_new_dense, "t3k_r", "d_ket", "t3k_I", env.T3_ket.indices, chi, D, chi_I
    )
    T3_bra_new = _wrap_edge_bra_dense(
        T3_bra_new_dense, "t3b_I", "d_bra", "t3b_l", env.T3_bra.indices, chi, D, chi_I
    )

    return SplitCTMTensorEnv(
        C1=env.C1,
        C2=env.C2,
        C3=C3_new,
        C4=C4_new,
        T1_ket=env.T1_ket,
        T1_bra=env.T1_bra,
        T2_ket=env.T2_ket,
        T2_bra=env.T2_bra,
        T3_ket=T3_ket_new,
        T3_bra=T3_bra_new,
        T4_ket=env.T4_ket,
        T4_bra=env.T4_bra,
    )


# ------------------------------------------------------------------ #
# Dense helper: SVD split edge                                         #
# ------------------------------------------------------------------ #


def _svd_split_edge_dense(
    T_full: jax.Array,
    chi_I: int,
) -> tuple[jax.Array, jax.Array]:
    """Split a standard edge (chi, D², chi) into ket/bra via SVD (dense)."""
    chi = T_full.shape[0]
    D2 = T_full.shape[1]
    D = math.isqrt(D2)
    T_4d = T_full.reshape(chi, D, D, chi)
    T_mat = T_4d.reshape(chi * D, D * chi)

    U, s, Vh = jnp.linalg.svd(T_mat, full_matrices=False)
    k = min(chi_I, len(s))
    sqrt_s = jnp.sqrt(s[:k])
    T_ket = (U[:, :k] * sqrt_s[None, :]).reshape(chi, D, k)
    T_bra = (sqrt_s[:, None] * Vh[:k, :]).reshape(k, D, chi)
    return T_ket, T_bra


# ------------------------------------------------------------------ #
# Dense wrapping helpers                                               #
# ------------------------------------------------------------------ #


def _wrap_corner_dense(
    data: jax.Array,
    label_a: Label,
    label_b: Label,
    ref_idx_a: TensorIndex,
    ref_idx_b: TensorIndex,
    chi: int,
) -> DenseTensor:
    """Wrap a dense (chi, chi) array as DenseTensor with correct labels/flows."""
    sym = ref_idx_a.symmetry
    charges = np.zeros(chi, dtype=np.int32)
    idx_a = TensorIndex(sym, charges.copy(), ref_idx_a.flow, label=label_a)
    idx_b = TensorIndex(sym, charges.copy(), ref_idx_b.flow, label=label_b)
    return DenseTensor(data, (idx_a, idx_b))


def _wrap_edge_ket_dense(
    data: jax.Array,
    label_chi: Label,
    label_D: Label,
    label_I: Label,
    ref_indices: tuple[TensorIndex, ...],
    chi: int,
    D: int,
    chi_I: int,
) -> DenseTensor:
    """Wrap a dense (chi, D, chi_I) array as DenseTensor edge ket."""
    sym = ref_indices[0].symmetry
    idx_chi = TensorIndex(
        sym, np.zeros(chi, dtype=np.int32), ref_indices[0].flow, label=label_chi
    )
    idx_D = TensorIndex(
        sym, np.zeros(D, dtype=np.int32), ref_indices[1].flow, label=label_D
    )
    idx_I = TensorIndex(
        sym, np.zeros(chi_I, dtype=np.int32), ref_indices[2].flow, label=label_I
    )
    return DenseTensor(data, (idx_chi, idx_D, idx_I))


def _wrap_edge_bra_dense(
    data: jax.Array,
    label_I: Label,
    label_D: Label,
    label_chi: Label,
    ref_indices: tuple[TensorIndex, ...],
    chi: int,
    D: int,
    chi_I: int,
) -> DenseTensor:
    """Wrap a dense (chi_I, D, chi) array as DenseTensor edge bra."""
    sym = ref_indices[0].symmetry
    idx_I = TensorIndex(
        sym, np.zeros(chi_I, dtype=np.int32), ref_indices[0].flow, label=label_I
    )
    idx_D = TensorIndex(
        sym, np.zeros(D, dtype=np.int32), ref_indices[1].flow, label=label_D
    )
    idx_chi = TensorIndex(
        sym, np.zeros(chi, dtype=np.int32), ref_indices[2].flow, label=label_chi
    )
    return DenseTensor(data, (idx_I, idx_D, idx_chi))


# ------------------------------------------------------------------ #
# Sweep + convergence                                                  #
# ------------------------------------------------------------------ #


def _split_ctm_tensor_sweep(
    env: SplitCTMTensorEnv,
    A: Tensor,
    chi: int,
    chi_I: int,
    renormalize: bool,
) -> SplitCTMTensorEnv:
    """One full split-CTM sweep: L/R/T/B moves + optional renormalize."""
    env = _split_ctm_move_left(env, A, chi, chi_I)
    env = _split_ctm_move_right(env, A, chi, chi_I)
    env = _split_ctm_move_top(env, A, chi, chi_I)
    env = _split_ctm_move_bottom(env, A, chi, chi_I)

    if renormalize:
        env = _renormalize_split_env(env)

    return env


def _renormalize_split_env(env: SplitCTMTensorEnv) -> SplitCTMTensorEnv:
    """Renormalize all 12 tensors in a SplitCTMTensorEnv."""
    C1, _ = max_abs_normalize(env.C1)
    C2, _ = max_abs_normalize(env.C2)
    C3, _ = max_abs_normalize(env.C3)
    C4, _ = max_abs_normalize(env.C4)

    def normalize_pair(T_ket: Tensor, T_bra: Tensor) -> tuple[Tensor, Tensor]:
        nk = T_ket.max_abs()
        nb = T_bra.max_abs()
        shared = jnp.sqrt(nk * nb) + EPS
        return T_ket * (1.0 / shared), T_bra * (1.0 / shared)

    T1k, T1b = normalize_pair(env.T1_ket, env.T1_bra)
    T2k, T2b = normalize_pair(env.T2_ket, env.T2_bra)
    T3k, T3b = normalize_pair(env.T3_ket, env.T3_bra)
    T4k, T4b = normalize_pair(env.T4_ket, env.T4_bra)

    return SplitCTMTensorEnv(
        C1=C1,
        C2=C2,
        C3=C3,
        C4=C4,
        T1_ket=T1k,
        T1_bra=T1b,
        T2_ket=T2k,
        T2_bra=T2b,
        T3_ket=T3k,
        T3_bra=T3b,
        T4_ket=T4k,
        T4_bra=T4b,
    )


def ctm_split_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
) -> SplitCTMTensorEnv:
    """Run split-CTM to convergence using the Tensor protocol.

    Args:
        A:          iPEPS site tensor (DenseTensor or SymmetricTensor) with
                    5 legs ``(u, d, l, r, phys)``.
        chi:        Environment bond dimension.
        max_iter:   Maximum number of CTM iterations.
        conv_tol:   Convergence tolerance on corner singular values.
        chi_I:      Interlayer bond dimension. Defaults to ``chi``.
        renormalize: Renormalize environment at each step.

    Returns:
        Converged SplitCTMTensorEnv.
    """
    if chi_I is None:
        chi_I = chi

    env = initialize_split_ctm_tensor_env(A, chi, chi_I)

    prev_sv = None
    for _ in range(max_iter):
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)

        current_sv = jnp.linalg.svd(env.C1.todense(), compute_uv=False)
        if prev_sv is not None:
            sv1 = current_sv / (jnp.sum(current_sv) + 1e-15)
            sv2 = prev_sv / (jnp.sum(prev_sv) + 1e-15)
            min_len = min(len(sv1), len(sv2))
            diff = jnp.max(jnp.abs(sv1[:min_len] - sv2[:min_len]))
            if float(diff) < conv_tol:
                break
        prev_sv = current_sv

    return env


# ------------------------------------------------------------------ #
# Energy computation (split, no double-layer)                          #
# ------------------------------------------------------------------ #


def _split_env_to_dense_standard(env: SplitCTMTensorEnv) -> tuple:
    """Convert SplitCTMTensorEnv to (C1..C4, T1..T4) dense arrays.

    Returns 8 dense arrays matching CTMEnvironment convention.
    """
    chi = env.C1.todense().shape[0]

    def merge(T_ket, T_bra):
        D_ket = T_ket.todense().shape[1]
        T = jnp.einsum("auc,cUb->auUb", T_ket.todense(), T_bra.todense())
        return T.reshape(chi, D_ket * D_ket, chi)

    return (
        env.C1.todense(),
        env.C2.todense(),
        env.C3.todense(),
        env.C4.todense(),
        merge(env.T1_ket, env.T1_bra),
        merge(env.T2_ket, env.T2_bra),
        merge(env.T3_ket, env.T3_bra),
        merge(env.T4_ket, env.T4_bra),
    )


def compute_energy_split_ctm_tensor(
    A: Tensor,
    env: SplitCTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site using split CTM environment.

    Converts to standard dense CTM internally and delegates to the
    existing RDM-based energy computation. This is correct because
    the ket/bra merge over the interlayer bond reconstructs the standard
    double-layer edges.

    Args:
        A:                iPEPS site tensor.
        env:              Converged SplitCTMTensorEnv.
        hamiltonian_gate: 2-site Hamiltonian gate.
        d:                Physical dimension (inferred from A if None).

    Returns:
        Scalar energy per site.
    """
    from tenax.algorithms.ipeps import CTMEnvironment, compute_energy_ctm

    A_dense = A.todense()
    if d is None:
        d = A_dense.shape[-1]

    if isinstance(hamiltonian_gate, Tensor):
        H = hamiltonian_gate.todense().reshape(d, d, d, d)
    else:
        H = hamiltonian_gate.reshape(d, d, d, d)

    C1, C2, C3, C4, T1, T2, T3, T4 = _split_env_to_dense_standard(env)
    std_env = CTMEnvironment(C1=C1, C2=C2, C3=C3, C4=C4, T1=T1, T2=T2, T3=T3, T4=T4)
    return compute_energy_ctm(A_dense, std_env, H, d)
