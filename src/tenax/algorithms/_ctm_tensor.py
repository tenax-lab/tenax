"""Standard CTM using the Tensor protocol (polymorphic dense/symmetric).

Builds the full double-layer tensor via ``bar()`` + ``contract()`` + ``fuse_indices()``,
then runs the standard projector-based CTM with native Tensor projectors applied
via label-based contraction.

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
    _derive_charges,
    _make_dense_corner,
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


def _eigh_projector_symmetric(
    C1g: SymmetricTensor,
    C4g: SymmetricTensor,
    chi: int,
) -> SymmetricTensor:
    r"""Block-sparse projector via per-sector density matrix eigh.

    For each charge sector *q* of the ``fused`` leg, accumulates
    :math:`\rho_q = M_1 M_1^\dagger + M_4 M_4^\dagger` where :math:`M_i`
    is the sector's dense block of C_ig, then eigendecomposes :math:`\rho_q`.
    Eigenvalues are merged across sectors and globally truncated to *chi*.

    The result is wrapped via ``from_dense`` with trivial ``chi_new`` charges
    so that downstream label-based contractions remain charge-consistent with
    the existing CTM environment structure.

    Args:
        C1g: Grown corner SymmetricTensor ``(fused, col1)``.
        C4g: Grown corner SymmetricTensor ``(fused, col2)``.
        chi: Target bond dimension.

    Returns:
        Projector ``P`` with labels ``(fused, chi_new)``, flows ``(IN, OUT)``.
    """
    fused_pos = C1g.labels().index("fused")
    col_pos = 1 - fused_pos  # the other leg
    fused_idx = C1g.indices[fused_pos]
    sym = fused_idx.symmetry
    fused_total = fused_idx.dim

    # Group blocks by fused charge for each corner
    def _group_by_fused(Cg: SymmetricTensor) -> dict[int, list[tuple[int, jax.Array]]]:
        """Map fused_charge -> list of (col_charge, block)."""
        grouped: dict[int, list[tuple[int, jax.Array]]] = {}
        for key, block in Cg.blocks.items():
            fq = int(key[fused_pos])
            cq = int(key[col_pos])
            grouped.setdefault(fq, []).append((cq, block))
        return grouped

    c1_groups = _group_by_fused(C1g)
    c4_groups = _group_by_fused(C4g)

    all_fused_charges = sorted(set(c1_groups.keys()) | set(c4_groups.keys()))

    # Per-sector eigh results: q -> (eigvecs, eigvals, fused_dim, row_offset)
    sector_results: dict[int, tuple[jax.Array, jax.Array, int, int]] = {}

    # Build a map from fused charge to its row indices in the dense fused index
    charges_arr = np.asarray(fused_idx.charges)
    charge_rows: dict[int, np.ndarray] = {}
    for fq in all_fused_charges:
        charge_rows[fq] = np.where(charges_arr == fq)[0]

    for fq in all_fused_charges:
        fused_dim = int(len(charge_rows.get(fq, [])))
        if fused_dim == 0:
            continue

        # Accumulate rho = sum of M @ M^dagger for both corners
        rho = jnp.zeros((fused_dim, fused_dim), dtype=C1g.dtype)

        for entries in [c1_groups.get(fq, []), c4_groups.get(fq, [])]:
            for _cq, block in entries:
                if fused_pos == 0:
                    M = block.reshape(fused_dim, -1)
                else:
                    M = block.reshape(-1, fused_dim).T
                rho = rho + M @ M.conj().T

        rho = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho)
        sector_results[fq] = (eigvecs, eigvals, fused_dim, charge_rows[fq])

    # Global truncation: merge eigenvalues, keep top-chi
    all_eig_pairs: list[tuple[float, int, int]] = []  # (value, fused_charge, index)
    for fq, (_, eigvals, _, _) in sector_results.items():
        for i, val in enumerate(np.array(eigvals)):
            all_eig_pairs.append((float(val), fq, i))

    # Sort descending by eigenvalue, then descending by index to match
    # the dense convention of taking eigvecs[:, -k:] (highest indices first
    # among degenerate eigenvalues).
    all_eig_pairs.sort(key=lambda x: (-x[0], -x[2]))
    n_keep = min(chi, len(all_eig_pairs))

    # Count per-sector keeps
    sector_keep: dict[int, list[int]] = {}
    for _, fq, idx in all_eig_pairs[:n_keep]:
        sector_keep.setdefault(fq, []).append(idx)

    # Assemble dense projector from per-sector eigenvectors, then wrap via
    # from_dense with trivial chi_new charges.  Only the identity-charge
    # sector survives conservation filtering, matching the dense path.
    P_dense = jnp.zeros((fused_total, n_keep), dtype=C1g.dtype)
    col = 0
    for fq in sorted(sector_keep.keys()):
        keep_indices = sorted(sector_keep[fq], reverse=True)
        n_q = len(keep_indices)

        eigvecs, _, fused_dim, row_idx = sector_results[fq]
        V_q = eigvecs[:, keep_indices]  # (fused_dim, n_q)

        P_dense = P_dense.at[row_idx, col : col + n_q].set(V_q)
        col += n_q

    P_dense = jax.lax.stop_gradient(P_dense)

    chi_new_idx = TensorIndex(
        sym, np.zeros(n_keep, dtype=np.int32), OUT, label="chi_new"
    )
    return SymmetricTensor.from_dense(
        P_dense, (fused_idx, chi_new_idx), tol=float("inf")
    )


def _compute_projector_tensor(
    C1g: Tensor,
    C4g: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> Tensor:
    r"""Compute isometric projector P as a Tensor.

    When ``projector_method == "eigh"``, forms the full density matrix
    :math:`\rho = C_{1g} C_{1g}^\dagger + C_{4g} C_{4g}^\dagger`,
    eigendecomposes, then wraps the top-k eigenvectors as a Tensor.

    When ``projector_method == "qr"``, QR-factors the concatenated corners
    ``[C1g, C4g]`` to reduce to a small ``(2*col, 2*col)`` eigenproblem,
    following the approach in arXiv:2505.00494.

    For SymmetricTensor inputs with ``"eigh"``, uses block-sparse eigh
    (per charge sector) to avoid dense round-trip.

    Args:
        C1g: Grown corner with labels ``(fused, <col1>)``.
        C4g: Grown corner with labels ``(fused, <col2>)``.
        chi: Target bond dimension.
        projector_method: ``"eigh"`` or ``"qr"``.

    Returns:
        Projector ``P`` with labels ``(fused, chi_new)``,
        flows ``(IN, OUT)``.  Wrapped in ``stop_gradient``.

    Raises:
        ValueError: If ``projector_method`` is not ``"eigh"`` or ``"qr"``.
    """
    if projector_method not in ("eigh", "qr"):
        raise ValueError(
            f"Unknown projector_method={projector_method!r}; expected 'eigh' or 'qr'."
        )

    # --- QR path: densify → QR → small eigh ---
    if projector_method == "qr":
        C1g_dense = C1g.todense()
        C4g_dense = C4g.todense()

        M = jnp.concatenate([C1g_dense, C4g_dense], axis=1)
        Q, R = jnp.linalg.qr(M)

        rho_small = R @ R.conj().T
        rho_small = 0.5 * (rho_small + rho_small.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho_small)

        k = min(chi, len(eigvals))
        V = eigvecs[:, -k:][:, ::-1]
        P_dense = Q @ V
        P_dense = jax.lax.stop_gradient(P_dense)

        fused_idx = C1g.indices[C1g.labels().index("fused")]
        chi_new_idx = TensorIndex(
            fused_idx.symmetry,
            np.zeros(k, dtype=np.int32),
            OUT,
            label="chi_new",
        )
        if isinstance(C1g, SymmetricTensor):
            return SymmetricTensor.from_dense(
                P_dense, (fused_idx, chi_new_idx), tol=float("inf")
            )
        return DenseTensor(P_dense, (fused_idx, chi_new_idx))

    # --- eigh path ---
    # Use block-sparse path for SymmetricTensor unless blocks contain
    # JAX tracers (during AD), in which case fall back to dense path
    # since eigenvalue sorting requires concrete values.
    if isinstance(C1g, SymmetricTensor) and isinstance(C4g, SymmetricTensor):
        has_tracers = any(
            isinstance(b, jax.core.Tracer)
            for b in (*C1g.blocks.values(), *C4g.blocks.values())
        )
        if not has_tracers and C1g.blocks and C4g.blocks:
            return _eigh_projector_symmetric(C1g, C4g, chi)

    C1g_dense = C1g.todense()
    C4g_dense = C4g.todense()

    rho = C1g_dense @ C1g_dense.conj().T + C4g_dense @ C4g_dense.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho)
    k = min(chi, len(eigvals))
    P_dense = eigvecs[:, -k:][:, ::-1]
    P_dense = jax.lax.stop_gradient(P_dense)

    # Wrap as Tensor with fused index from C1g and new chi_new bond
    fused_idx = C1g.indices[C1g.labels().index("fused")]
    chi_new_idx = TensorIndex(
        fused_idx.symmetry,
        np.zeros(k, dtype=np.int32),
        OUT,
        label="chi_new",
    )
    if isinstance(C1g, SymmetricTensor):
        return SymmetricTensor.from_dense(
            P_dense, (fused_idx, chi_new_idx), tol=float("inf")
        )
    return DenseTensor(P_dense, (fused_idx, chi_new_idx))


def _apply_projector_tensor(
    P: Tensor,
    C1g: Tensor,
    C4g: Tensor,
    Tg: Tensor,
    fused_l: str,
    fused_r: str,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Apply projector to grown corners and edge using Tensor contraction.

    Computes :math:`P^\dagger C_{1g}`, :math:`P^\dagger C_{4g}`,
    and the sandwich :math:`P^\dagger T_g P`.

    Uses ``P.bar()`` (= :math:`P^\dagger` for real isometries) on the left
    and ``P`` on the right of the edge sandwich.

    Args:
        P:       Projector with labels ``(fused, chi_new)``.
        C1g:     Grown corner ``(fused, col1)``.
        C4g:     Grown corner ``(fused, col2)``.
        Tg:      Grown edge ``(fused_l, D2_label, fused_r)``.
        fused_l: Label of Tg's left fused leg.
        fused_r: Label of Tg's right fused leg.

    Returns:
        ``(C1_new, C4_new, T_new)`` as Tensor objects.
    """
    P_bar = P.bar()  # (fused_OUT, chi_new_IN) — contracts on "fused"

    C1_new = contract(P_bar, C1g)  # (chi_new, col1)
    C4_new = contract(P_bar, C4g)  # (chi_new, col2)

    # Sandwich: P^bar @ Tg @ P  (left fused, then right fused)
    P_left = P_bar.relabel("fused", fused_l)
    step = contract(P_left, Tg)  # (chi_new, D2, fused_r)

    P_right = P.relabels({"fused": fused_r, "chi_new": "chi_new_r"})
    T_new = contract(step, P_right)  # (chi_new, D2, chi_new_r)

    return C1_new, C4_new, T_new


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
# CTM moves                                                            #
# ------------------------------------------------------------------ #


def _ctm_tensor_move_left(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Left move: updates C1, T4, C4.

    Corners (C1, C4) from env_self, perpendicular edges (T1, T3) from
    env_neighbor, parallel edge (T4) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C1g = einsum('ab,buc->auc', C1, T1)
                     C4g = einsum('gh,hdi->gdi', C4, T3)
                     T4g = einsum('alg,udlr->augdr', T4, a)
    """
    # C1(self) · T1(neighbor)
    C1_r = env_self.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env_neighbor.T1)  # (c1_d, u2, t1_r)
    C1g = _fuse_pair_by_label(C1g, "c1_d", "u2", "fused", IN)  # (fused, t1_r)

    # C4(self) · T3(neighbor)
    C4_u = env_self.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env_neighbor.T3)  # (c4_r, d2, t3_l)
    C4g = _fuse_pair_by_label(C4g, "c4_r", "d2", "fused", IN)  # (fused, t3_l)

    # T4(self) · a(neighbor)
    T4_with_a = contract(env_self.T4, a)
    T4g = _fuse_pair_by_label(T4_with_a, "t4_d", "u2", "fl", IN)
    T4g = _fuse_pair_by_label(T4g, "t4_u", "d2", "fr", OUT)

    # Native projector
    P = _compute_projector_tensor(C1g, C4g, chi, projector_method)
    C1_new, C4_new, T4_new = _apply_projector_tensor(P, C1g, C4g, T4g, "fl", "fr")

    # Relabel to expected output labels
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1_r": "c1_r"})
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3_l": "c4_u"})
    T4_new = T4_new.relabels({"chi_new": "t4_d", "chi_new_r": "t4_u", "r2": "l2"})

    return env_self._replace(C1=C1_new, C4=C4_new, T4=T4_new)


def _ctm_tensor_move_right(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Right move: updates C2, T2, C3.

    Corners (C2, C3) from env_self, perpendicular edges (T1, T3) from
    env_neighbor, parallel edge (T2) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C2g = einsum('ce,buc->eub', C2, T1)
                     C3g = einsum('im,hdi->mdh', C3, T3)
                     T2g = einsum('erm,udlr->eumdl', T2, a)
    """
    # C2(self) · T1(neighbor)
    C2_l = env_self.C2.relabel("c2_l", "t1_r")
    C2g = contract(C2_l, env_neighbor.T1)  # (c2_d, t1_l, u2)
    C2g = _fuse_pair_by_label(C2g, "c2_d", "u2", "fused", IN)  # (fused, t1_l)

    # C3(self) · T3(neighbor)
    C3_u = env_self.C3.relabel("c3_u", "t3_l")
    C3g = contract(C3_u, env_neighbor.T3)  # (c3_l, t3_r, d2)
    C3g = _fuse_pair_by_label(C3g, "c3_l", "d2", "fused", IN)  # (fused, t3_r)

    # T2(self) · a(neighbor)
    T2_with_a = contract(env_self.T2, a)
    T2g = _fuse_pair_by_label(T2_with_a, "t2_u", "u2", "fl", IN)
    T2g = _fuse_pair_by_label(T2g, "t2_d", "d2", "fr", OUT)

    # Native projector
    P = _compute_projector_tensor(C2g, C3g, chi, projector_method)
    C2_new, C3_new, T2_new = _apply_projector_tensor(P, C2g, C3g, T2g, "fl", "fr")

    # Relabel to expected output labels
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t1_l": "c2_d"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3_r": "c3_l"})
    T2_new = T2_new.relabels({"chi_new": "t2_u", "chi_new_r": "t2_d", "l2": "r2"})

    return env_self._replace(C2=C2_new, C3=C3_new, T2=T2_new)


def _ctm_tensor_move_top(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Top move: updates C1, T1, C2.

    Corners (C1, C2) from env_self, perpendicular edges (T4, T2) from
    env_neighbor, parallel edge (T1) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C1g = einsum('ab,alg->blg', C1, T4)
                     C2g = einsum('ce,erm->crm', C2, T2)
                     T1g = einsum('buc,udlr->bcdlr', T1, a)
    """
    # C1(self) · T4(neighbor)
    C1_d = env_self.C1.relabel("c1_d", "t4_d")
    C1g = contract(C1_d, env_neighbor.T4)  # (c1_r, l2, t4_u)
    C1g = _fuse_pair_by_label(C1g, "c1_r", "l2", "fused", IN)  # (fused, t4_u)

    # C2(self) · T2(neighbor)
    C2_d = env_self.C2.relabel("c2_d", "t2_u")
    C2g = contract(C2_d, env_neighbor.T2)  # (c2_l, r2, t2_d)
    C2g = _fuse_pair_by_label(C2g, "c2_l", "r2", "fused", IN)  # (fused, t2_d)

    # T1(self) · a(neighbor)
    T1_with_a = contract(env_self.T1, a)
    T1g = _fuse_pair_by_label(T1_with_a, "t1_l", "l2", "fl", IN)
    T1g = _fuse_pair_by_label(T1g, "t1_r", "r2", "fr", OUT)

    # Native projector
    P = _compute_projector_tensor(C1g, C2g, chi, projector_method)
    C1_new, C2_new, T1_new = _apply_projector_tensor(P, C1g, C2g, T1g, "fl", "fr")

    # Relabel to expected output labels
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t4_u": "c1_r"})
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2_d": "c2_d"})
    T1_new = T1_new.relabels({"chi_new": "t1_l", "chi_new_r": "t1_r", "d2": "u2"})

    return env_self._replace(C1=C1_new, C2=C2_new, T1=T1_new)


def _ctm_tensor_move_bottom(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
    """Bottom move: updates C4, T3, C3.

    Corners (C4, C3) from env_self, perpendicular edges (T4, T2) from
    env_neighbor, parallel edge (T3) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C4g = einsum('gh,alg->hal', C4, T4).transpose(0,2,1)
                     C3g = einsum('im,erm->ire', C3, T2)
                     T3g = einsum('hdi,udlr->hiulr', T3, a)
    """
    # C4(self) · T4(neighbor)
    C4_r = env_self.C4.relabel("c4_r", "t4_u")
    C4g = contract(C4_r, env_neighbor.T4)  # (c4_u, t4_d, l2)
    C4g = _fuse_pair_by_label(C4g, "c4_u", "l2", "fused", IN)  # (fused, t4_d)

    # C3(self) · T2(neighbor)
    C3_l = env_self.C3.relabel("c3_l", "t2_d")
    C3g = contract(C3_l, env_neighbor.T2)  # (c3_u, t2_u, r2)
    C3g = _fuse_pair_by_label(C3g, "c3_u", "r2", "fused", IN)  # (fused, t2_u)

    # T3(self) · a(neighbor)
    T3_with_a = contract(env_self.T3, a)
    T3g = _fuse_pair_by_label(T3_with_a, "t3_r", "l2", "fl", IN)
    T3g = _fuse_pair_by_label(T3g, "t3_l", "r2", "fr", OUT)

    # Native projector
    P = _compute_projector_tensor(C4g, C3g, chi, projector_method)
    C4_new, C3_new, T3_new = _apply_projector_tensor(P, C4g, C3g, T3g, "fl", "fr")

    # Relabel to expected output labels
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t4_d": "c4_u"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t2_u": "c3_l"})
    T3_new = T3_new.relabels({"chi_new": "t3_r", "chi_new_r": "t3_l", "u2": "d2"})

    return env_self._replace(C4=C4_new, C3=C3_new, T3=T3_new)


# ------------------------------------------------------------------ #
# Sweep + renormalize                                                  #
# ------------------------------------------------------------------ #


def _normalize_tensor(T: Tensor) -> Tensor:
    """Normalize tensor by max abs value, matching dense CTM convention.

    Uses ``data / (norm + EPS)`` (single division) rather than
    ``data * (1/norm)`` (reciprocal + multiplication) to avoid
    O(eps) floating-point differences vs the dense path.
    """
    data = T.todense()
    norm = jnp.max(jnp.abs(data))
    data_normed = data / (norm + EPS)
    if isinstance(T, SymmetricTensor):
        return SymmetricTensor.from_dense(data_normed, T.indices, tol=float("inf"))
    return type(T)(data_normed, T.indices)


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
    env = _ctm_tensor_move_left(env, env, a, chi, projector_method)
    env = _ctm_tensor_move_right(env, env, a, chi, projector_method)
    env = _ctm_tensor_move_top(env, env, a, chi, projector_method)
    env = _ctm_tensor_move_bottom(env, env, a, chi, projector_method)
    if renormalize:
        env = _renormalize_tensor_env(env)
    return env


# ------------------------------------------------------------------ #
# Neighbor maps for unit cell topologies                              #
# ------------------------------------------------------------------ #

Coord = tuple[int, int]

SINGLE_SITE_NEIGHBORS: dict[Coord, dict[str, Coord]] = {
    (0, 0): {"left": (0, 0), "right": (0, 0), "top": (0, 0), "bottom": (0, 0)},
}

CHECKERBOARD_NEIGHBORS: dict[Coord, dict[str, Coord]] = {
    (0, 0): {"left": (1, 0), "right": (1, 0), "top": (1, 0), "bottom": (1, 0)},
    (1, 0): {"left": (0, 0), "right": (0, 0), "top": (0, 0), "bottom": (0, 0)},
}

_DIRECTION_MOVES = [
    ("left", _ctm_tensor_move_left),
    ("right", _ctm_tensor_move_right),
    ("top", _ctm_tensor_move_top),
    ("bottom", _ctm_tensor_move_bottom),
]


def _ctm_tensor_sweep_multisite(
    envs: dict[Coord, CTMTensorEnv],
    double_layers: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    renormalize: bool,
    projector_method: str = "eigh",
) -> dict[Coord, CTMTensorEnv]:
    """One full multisite CTM sweep over all sites and directions."""
    for direction, move_fn in _DIRECTION_MOVES:
        for coord in sorted(envs.keys()):
            nb = neighbors[coord][direction]
            envs[coord] = move_fn(
                envs[coord], envs[nb], double_layers[nb], chi, projector_method
            )
    if renormalize:
        envs = {c: _renormalize_tensor_env(e) for c, e in envs.items()}
    return envs


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
    qr_warmup_steps: int = 3,
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
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.

    Returns:
        Converged CTMTensorEnv.
    """
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            env = _ctm_tensor_sweep(env, a, chi, renormalize, "eigh")
        max_iter = max_iter - warmup

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


def _ctm_tensor_multisite(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
) -> dict[Coord, CTMTensorEnv]:
    """Run multisite CTM to convergence using the Tensor protocol.

    Args:
        site_tensors: Map from coordinate to iPEPS site tensor.
        neighbors:    Map from coordinate to direction→neighbor coordinate.
        chi:          Environment bond dimension.
        max_iter:     Maximum CTM iterations.
        conv_tol:     Convergence tolerance on corner singular values.
        renormalize:  Renormalize environment at each step.
        projector_method: ``"eigh"`` or ``"qr"``.
        qr_warmup_steps:  Number of eigh warm-up sweeps before QR kicks in.

    Returns:
        Dict mapping coordinates to converged CTMTensorEnv.
    """
    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    envs = {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs = _ctm_tensor_sweep_multisite(
                envs, double_layers, neighbors, chi, renormalize, "eigh"
            )
        max_iter = max_iter - warmup

    prev_svs: dict[Coord, jax.Array] = {}
    for _ in range(max_iter):
        envs = _ctm_tensor_sweep_multisite(
            envs, double_layers, neighbors, chi, renormalize, projector_method
        )
        converged = True
        for c in sorted(envs):
            sv = jnp.linalg.svd(envs[c].C1.todense(), compute_uv=False)
            if c in prev_svs:
                if float(_ctm_sv_diff(sv, prev_svs[c])) >= conv_tol:
                    converged = False
            else:
                converged = False
            prev_svs[c] = sv
        if converged:
            break

    return envs


def ctm_tensor_2site(
    A: Tensor,
    B: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
) -> tuple[CTMTensorEnv, CTMTensorEnv]:
    """Run 2-site checkerboard CTM to convergence using the Tensor protocol.

    Args:
        A:   Site tensor for sublattice A (DenseTensor or SymmetricTensor)
             with 5 legs ``(u, d, l, r, phys)``.
        B:   Site tensor for sublattice B.
        chi: Environment bond dimension.
        max_iter:     Maximum CTM iterations.
        conv_tol:     Convergence tolerance on corner singular values.
        renormalize:  Renormalize environment at each step.
        projector_method: ``"eigh"`` or ``"qr"``.
        qr_warmup_steps:  Number of eigh warm-up sweeps before QR kicks in.

    Returns:
        ``(env_A, env_B)`` — converged CTMTensorEnv for each sublattice.
    """
    envs = _ctm_tensor_multisite(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi,
        max_iter,
        conv_tol,
        renormalize,
        projector_method,
        qr_warmup_steps,
    )
    return envs[(0, 0)], envs[(1, 0)]


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


def compute_energy_ctm_tensor_2site(
    A: Tensor,
    B: Tensor,
    env_A: CTMTensorEnv,
    env_B: CTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site for a 2-site checkerboard iPEPS.

    Converts to dense and delegates to ``compute_energy_ctm_2site``.

    Args:
        A:                Site tensor for sublattice A.
        B:                Site tensor for sublattice B.
        env_A:            Converged CTMTensorEnv for sublattice A.
        env_B:            Converged CTMTensorEnv for sublattice B.
        hamiltonian_gate: 2-site Hamiltonian gate.
        d:                Physical dimension (inferred from A if None).

    Returns:
        Scalar energy per site.
    """
    from tenax.algorithms.ipeps import compute_energy_ctm_2site

    A_d = A.todense()
    B_d = B.todense()
    if d is None:
        d = A_d.shape[-1]

    if isinstance(hamiltonian_gate, Tensor):
        H = hamiltonian_gate.todense().reshape(d, d, d, d)
    else:
        H = hamiltonian_gate.reshape(d, d, d, d)

    std_env_A = _env_to_dense_standard(env_A)
    std_env_B = _env_to_dense_standard(env_B)
    return compute_energy_ctm_2site(A_d, B_d, std_env_A, std_env_B, H, d)
