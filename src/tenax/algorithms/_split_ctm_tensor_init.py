"""Split CTM with Tensor protocol — data structures and initialization."""

from __future__ import annotations

__all__ = [
    "SplitCTMTensorEnv",
    "_EDGE_BRA_SPECS",
    "_EDGE_KET_SPECS",
    "_make_dense_edge_bra",
    "_make_dense_edge_ket",
    "initialize_split_ctm_tensor_env",
]

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_utils import (
    _CORNER_SPECS,
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


# Edge specs: (label_first, label_D, label_last, flow_first, flow_D, flow_last,
#              ref_axis_chi, ref_axis_D)
#
# ``ref_axis_chi`` must name the SAME axis of ``A`` that the corner on the other
# end of that chi bond uses (``_CORNER_SPECS``).  Both ends are seeded by tiling
# one of A's virtual legs with :func:`_derive_charges`, so if they tile
# *different* legs the seam is contractible only while those legs happen to
# carry the same charge layout.  A uniform iPEPS satisfies that by accident; a
# state whose horizontal and vertical bonds differ does not, and the 2x2
# plaquette projector dies in ``_build_enlarged_corner`` with a shape error
# (#1024).  Such states are not exotic -- they are what simple update produces
# once the truncation may discover the bond charges instead of being pinned to
# the initial guess (#878).
#
# The fused table ``_ctm_tensor_init._STD_EDGE_SPECS`` states and satisfies the
# same rule, annotating each chi leg with the corner it meets; this table had
# drifted from it on the four horizontal seams.  Corner reference axes are
# C1=d(1), C2=u(0), C3=d(1), C4=u(0).
#
# "The same axis" means literally the same axis, not an axis carrying the same
# charges.  ``_derive_charges`` tiles the leg's charge ARRAY, so it is sensitive
# to charge ORDER, not just to the multiset: ``[0,1,0]`` tiled to chi=16 yields
# 5 odd slots, ``[1,0,0]`` yields 6.  ``u`` and ``d`` are opposite ends of one
# lattice bond and therefore carry the same multiset, but simple update can
# leave them in different ORDERS -- measured on the D=3 fermionic sweep, ``u``
# and ``d`` both report ``{even:2, odd:1}`` while their corners seed
# ``{0: 11, 1: 5}`` and ``{0: 10, 1: 6}`` respectively.  Substituting one for
# the other on the grounds that "u and d carry the same charges" is therefore
# wrong, and it is how the two vertical seams stayed broken after the
# horizontal ones were fixed.
_EDGE_KET_SPECS = {
    "T1": (
        "t1k_l",
        "u_ket",
        "t1k_I",
        FlowDirection.IN,
        FlowDirection.IN,
        FlowDirection.OUT,
        0,
        0,
    ),  # t1k_l meets C1.c1_r (C1 ref=d(1)); D=u(0), flow opposite A's u(OUT)
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
        0,
        1,
    ),  # t3k_r meets C4.c4_r (C4 ref=u(0)); D=d(1), flow opposite A's d(IN)
    "T4": (
        "t4k_d",
        "l_ket",
        "t4k_I",
        FlowDirection.IN,
        FlowDirection.IN,
        FlowDirection.OUT,
        0,
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
        0,
        0,
    ),  # t1b_r meets C2.c2_l (C2 ref=u(0)); D-flow opposite A.bar()'s u(IN)
    "T2": (
        "t2b_I",
        "r_bra",
        "t2b_d",
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        0,
        3,
    ),  # t2b_d meets C3.c3_u (C3 ref=d(1)); D-flow opposite A.bar()'s r(OUT)
    "T3": (
        "t3b_I",
        "d_bra",
        "t3b_l",
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        0,
        1,
    ),  # t3b_l meets C3.c3_l (C3 ref=d(1)); D-flow opposite A.bar()'s d(OUT)
    "T4": (
        "t4b_I",
        "l_bra",
        "t4b_u",
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.IN,
        0,
        2,
    ),  # t4b_u meets C4.c4_u (C4 ref=u(0)); D-flow opposite A.bar()'s l(IN)
}


def _refuse_symmetric_split(*site_tensors: Tensor) -> None:
    """The split CTM is dense-only: refuse ``SymmetricTensor`` input.

    Its memory win (chi^2 D^4 instead of chi^2 D^6) is a large-D effect
    (D >~ 16, #825), where symmetric iPEPS-AD in tenax already loses to dense;
    the symmetric split path had no user and carried its own defects (#907,
    #1024, #1048).  Symmetric and fermionic iPEPS use the fused CTM.
    """
    if any(isinstance(t, SymmetricTensor) for t in site_tensors):
        raise NotImplementedError(
            "split CTM (fuse_virtual_legs=False) is dense-only: SymmetricTensor "
            "site tensors are not supported; use the fused CTM "
            "(fuse_virtual_legs=True, ctm_tensor / ctm_tensor_2site), or "
            "densify a small tensor with DenseTensor(A.todense(), A.indices)"
        )


def initialize_split_ctm_tensor_env(
    A: Tensor,
    chi: int,
    chi_I: int,
) -> SplitCTMTensorEnv:
    """Initialize a SplitCTMTensorEnv from a dense iPEPS site tensor.

    Args:
        A:     Site tensor with 5 legs ``(u, d, l, r, phys)``.  A
               ``SymmetricTensor`` is refused: the split CTM is dense-only.
        chi:   Environment bond dimension.
        chi_I: Interlayer bond dimension.

    Returns:
        Initialized SplitCTMTensorEnv.
    """
    _refuse_symmetric_split(A)

    D = A.indices[0].dim  # virtual bond dim
    dtype = A.dtype

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
