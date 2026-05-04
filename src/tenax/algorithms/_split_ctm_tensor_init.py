"""Split CTM with Tensor protocol — data structures and initialization."""

from __future__ import annotations

__all__ = [
    "SplitCTMTensorEnv",
    "_EDGE_BRA_SPECS",
    "_EDGE_KET_SPECS",
    "_canonical_I_charges",
    "_init_symmetric_corner",
    "_init_symmetric_edge_bra",
    "_init_symmetric_edge_ket",
    "_make_dense_edge_bra",
    "_make_dense_edge_ket",
    "initialize_split_ctm_tensor_env",
]

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_utils import (
    _CORNER_SPECS,
    _derive_charges,
    _make_dense_corner,
    _trivial_symmetry,
)
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
            TensorIndex.from_charges(
                sym, np.zeros(chi, dtype=np.int32), flow_chi, label=label_chi
            ),
            TensorIndex.from_charges(
                sym, np.zeros(D, dtype=np.int32), flow_D, label=label_D
            ),
            TensorIndex.from_charges(
                sym, np.zeros(chi_I, dtype=np.int32), flow_I, label=label_I
            ),
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
            TensorIndex.from_charges(
                sym, np.zeros(chi_I, dtype=np.int32), flow_I, label=label_I
            ),
            TensorIndex.from_charges(
                sym, np.zeros(D, dtype=np.int32), flow_D, label=label_D
            ),
            TensorIndex.from_charges(
                sym, np.zeros(chi, dtype=np.int32), flow_chi, label=label_chi
            ),
        ),
    )


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

    idx_a = TensorIndex.from_charges(sym, charges.copy(), flow_a, label=label_a)
    idx_b = TensorIndex.from_charges(sym, charges.copy(), flow_b, label=label_b)
    # Match dense reference: eye(min(chi, D)) padded to chi,
    # which initializes fewer diagonal entries when chi > D.
    D = A.indices[ref_axis].dim
    C = jnp.eye(min(chi, D), dtype=A.dtype)
    C_pad = jnp.zeros((chi, chi), dtype=A.dtype)
    C_pad = C_pad.at[: C.shape[0], : C.shape[1]].set(C)
    return SymmetricTensor.from_dense(C_pad, (idx_a, idx_b), tol=float("inf"))


def _conservation_holds(sym, qc, qd, qI, flow_chi, flow_D, flow_I) -> bool:
    """Check whether ``flow_chi*qc + flow_D*qd + flow_I*qI = identity`` under
    the symmetry's group operation.

    For U(1) this matches raw integer arithmetic; for Z₂/FermionParity it
    reduces mod 2 so that canonical I-charges produced by
    :func:`_canonical_I_charges` register as conservation-compatible (raw
    arithmetic would miss e.g. ``1+1-0 = 2`` even though ``2 ≡ 0 mod 2``).
    """
    qc_a = np.array([int(qc)], dtype=np.int32)
    qd_a = np.array([int(qd)], dtype=np.int32)
    qI_a = np.array([int(qI)], dtype=np.int32)
    qc_s = sym.dual(qc_a) if int(flow_chi) < 0 else qc_a
    qd_s = sym.dual(qd_a) if int(flow_D) < 0 else qd_a
    qI_s = sym.dual(qI_a) if int(flow_I) < 0 else qI_a
    total = sym.fuse(sym.fuse(qc_s, qd_s), qI_s)
    return int(total[0]) == sym.identity()


def _canonical_I_charges(
    sym,
    chi_charges: np.ndarray,
    D_charges: np.ndarray,
    flow_chi: FlowDirection,
    flow_D: FlowDirection,
    flow_I: FlowDirection,
    chi_I: int,
) -> np.ndarray:
    """Derive interlayer-bond charges via local conservation, then reduce to
    the symmetry's fundamental domain (mod-n for Zₙ, no-op for U(1)).

    Pre-canonicalisation (raw integer arithmetic) gave ``qI = qc + qd`` for the
    ket and ``qI = qd − qc`` for the bra under their respective flow specs;
    although equivalent under Z₂, the raw sequences differed, so the SVD bond
    came out with mismatched labels and ``_split_ctm_move_left`` could not
    contract them at certain chi (issue #391).

    The canonicalisation pipeline:
      1. derive raw I-charges via local conservation (matches pre-fix logic),
      2. tile to chi_I (preserves per-parity allocation across layers),
      3. fuse each raw value with the symmetry identity, which reduces it to
         the fundamental domain for Zₙ and is a no-op for U(1).

    Crucially, the *caller* must invoke this with the same flow tuple for both
    halves of a split edge (ket and bra), so the resulting charge sequence is
    identical on both sides of the SVD bond.
    """
    fc = int(flow_chi)
    fd = int(flow_D)
    fI = int(flow_I)
    needed_raw: set[int] = set()
    for qc in sorted({int(c) for c in chi_charges}):
        for qd in sorted({int(c) for c in D_charges}):
            qi_needed = -(fc * qc + fd * qd)
            if fI != 0 and qi_needed % fI == 0:
                needed_raw.add(qi_needed // fI)
    base = sorted(needed_raw) or [0]
    if chi_I <= len(base):
        raw_arr = np.array(base[:chi_I], dtype=np.int32)
    else:
        reps = chi_I // len(base) + 1
        raw_arr = np.array((base * reps)[:chi_I], dtype=np.int32)
    # Reduce each raw value to the symmetry's fundamental domain.
    identity_arr = np.full(len(raw_arr), sym.identity(), dtype=np.int32)
    return np.asarray(sym.fuse(raw_arr, identity_arr), dtype=np.int32)


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
    *,
    I_charges_arr: np.ndarray | None = None,
) -> SymmetricTensor:
    """Create an identity-like SymmetricTensor ket edge.

    Builds blocks for every valid charge sector with ones in the diagonal
    entries (chi_idx == I_idx within each sector), matching the dense
    reference ``T_ket[i, :, i] = ones(D)`` but only at charge-conserving
    positions.
    """
    sym = A.indices[0].symmetry

    # chi-leg charges from A's ref bond
    chi_charges = _derive_charges(A.indices[ref_axis_chi].charges, chi)
    D_charges = np.asarray(A.indices[ref_axis_D].charges.copy(), dtype=np.int32)

    # Canonical I-charges via the symmetry's group operation.  Caller may pass
    # the same I_charges_arr to both ket and bra inits so the SVD bond carries
    # identical raw charges on both sides under Z₂ (issue #391).  When None,
    # derive locally from the ket's flow conventions.
    if I_charges_arr is None:
        I_charges_arr = _canonical_I_charges(
            sym, chi_charges, D_charges, flow_chi, flow_D, flow_I, chi_I
        )

    idx_chi = TensorIndex.from_charges(sym, chi_charges, flow_chi, label=label_chi)
    idx_D = TensorIndex.from_charges(sym, D_charges, flow_D, label=label_D)
    idx_I = TensorIndex.from_charges(sym, I_charges_arr, flow_I, label=label_I)

    # Build conservation-compatible init matching dense reference pattern
    # T[i, :, i] = ones(D) for i in range(min(chi_D, chi_I_D)).
    # Only set entries where charge conservation holds (sym-aware: required
    # so canonical Z₂ charges in {0,1} register valid sectors that raw
    # integer arithmetic would miss).
    T = jnp.zeros((chi, D, chi_I), dtype=A.dtype)
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)
    for i in range(min(chi_D, chi_I_D)):
        for di in range(D):
            if _conservation_holds(
                sym,
                chi_charges[i],
                D_charges[di],
                I_charges_arr[i],
                flow_chi,
                flow_D,
                flow_I,
            ):
                T = T.at[i, di, i].set(1.0)
    return SymmetricTensor.from_dense(T, (idx_chi, idx_D, idx_I), tol=float("inf"))


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
    *,
    I_charges_arr: np.ndarray | None = None,
) -> SymmetricTensor:
    """Create an identity-like SymmetricTensor bra edge.

    Same conservation-aware charge derivation as the ket edge.  Caller may
    pass an explicit ``I_charges_arr`` so the bra and the matching ket share
    the same SVD-bond charges (issue #391).
    """
    sym = A.indices[0].symmetry

    D_charges = np.asarray(A.indices[ref_axis_D].charges.copy(), dtype=np.int32)
    chi_charges = _derive_charges(A.indices[ref_axis_chi].charges, chi)

    # Canonical I-charges (caller-supplied or derived from the bra's local
    # conservation rule).  Sharing the same array between ket and bra makes
    # the SVD bond carry identical raw charges on both sides under Z₂.
    if I_charges_arr is None:
        I_charges_arr = _canonical_I_charges(
            sym, chi_charges, D_charges, flow_chi, flow_D, flow_I, chi_I
        )

    idx_I = TensorIndex.from_charges(sym, I_charges_arr, flow_I, label=label_I)
    idx_D = TensorIndex.from_charges(sym, D_charges, flow_D, label=label_D)
    idx_chi = TensorIndex.from_charges(sym, chi_charges, flow_chi, label=label_chi)

    # Conservation-compatible init matching dense reference pattern
    # T_bra[i, :, i] = ones(D) for i in range(min(chi_I_D, chi_D)).
    # Only set entries where charge conservation holds (sym-aware).
    T = jnp.zeros((chi_I, D, chi), dtype=A.dtype)
    chi_D = min(chi, D)
    chi_I_D = min(chi_I, D)
    for i in range(min(chi_I_D, chi_D)):
        for di in range(D):
            if _conservation_holds(
                sym,
                chi_charges[i],
                D_charges[di],
                I_charges_arr[i],
                flow_chi,
                flow_D,
                flow_I,
            ):
                T = T.at[i, di, i].set(1.0)
    return SymmetricTensor.from_dense(T, (idx_I, idx_D, idx_chi), tol=float("inf"))


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
    ),  # D-flow opposite to A.bar()'s u(IN)
    "T2": (
        "t2b_I",
        "r_bra",
        "t2b_d",
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        0,
        3,
    ),  # D-flow opposite to A.bar()'s r(OUT)
    "T3": (
        "t3b_I",
        "d_bra",
        "t3b_l",
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        3,
        1,
    ),  # D-flow opposite to A.bar()'s d(OUT)
    "T4": (
        "t4b_I",
        "l_bra",
        "t4b_u",
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.IN,
        1,
        2,
    ),  # D-flow opposite to A.bar()'s l(IN)
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

        # Compute I-charges once per edge using the ket's flow conventions
        # and share with the bra so both halves of the SVD bond carry
        # identical raw charges (issue #391).
        sym = A.indices[0].symmetry
        shared_I_charges: dict[str, np.ndarray] = {}
        for name, (_, _, _, f1, f2, f3, ref_chi, ref_D) in _EDGE_KET_SPECS.items():
            chi_charges = _derive_charges(A.indices[ref_chi].charges, chi)
            D_charges = np.asarray(A.indices[ref_D].charges.copy(), dtype=np.int32)
            shared_I_charges[name] = _canonical_I_charges(
                sym, chi_charges, D_charges, f1, f2, f3, chi_I
            )

        ket_edges = {}
        for name, (l1, l2, l3, f1, f2, f3, ref_chi, ref_D) in _EDGE_KET_SPECS.items():
            ket_edges[name] = _init_symmetric_edge_ket(
                A,
                chi,
                D,
                chi_I,
                l1,
                l2,
                l3,
                f1,
                f2,
                f3,
                ref_chi,
                ref_D,
                I_charges_arr=shared_I_charges[name],
            )

        bra_edges = {}
        for name, (l1, l2, l3, f1, f2, f3, ref_chi, ref_D) in _EDGE_BRA_SPECS.items():
            bra_edges[name] = _init_symmetric_edge_bra(
                A,
                chi,
                D,
                chi_I,
                l1,
                l2,
                l3,
                f1,
                f2,
                f3,
                ref_chi,
                ref_D,
                I_charges_arr=shared_I_charges[name],
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
