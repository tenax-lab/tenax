"""Split CTM with Tensor protocol — conversion and energy computation."""

from __future__ import annotations

__all__ = [
    "_split_env_to_tensor_standard",
    "compute_energy_split_ctm_tensor",
]

import jax

from tenax.algorithms._ctm_tensor import CTMTensorEnv, compute_energy_ctm_tensor
from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection
from tenax.core.tensor import Tensor

# ------------------------------------------------------------------ #
# Energy computation (split, no double-layer)                          #
# ------------------------------------------------------------------ #


def _split_env_to_tensor_standard(env: SplitCTMTensorEnv) -> CTMTensorEnv:
    """Convert SplitCTMTensorEnv to CTMTensorEnv via Tensor contraction.

    Merges each (T_ket, T_bra) pair by contracting over the interlayer bond
    and fusing the two D-legs into a single double-layer D² leg.
    Corners pass through unchanged (same labels/flows).
    """

    def _merge_edge(
        T_ket,
        T_bra,
        ket_I,
        bra_I,
        d_ket,
        d_bra,
        fused_label,
        fused_flow,
        ket_chi,
        bra_chi,
        std_chi_l,
        std_chi_r,
    ):
        # Contract over interlayer bond by relabelling both I-labels to "_I"
        k = T_ket.relabel(ket_I, "_I")
        b = T_bra.relabel(bra_I, "_I")
        merged = contract(k, b)
        # Fuse D-ket and D-bra legs
        labels = merged.labels()
        merged = fuse_indices(
            merged,
            labels.index(d_ket),
            labels.index(d_bra),
            fused_label,
            fused_flow,
        )
        # Relabel chi legs to standard CTMTensorEnv convention
        merged = merged.relabels({ket_chi: std_chi_l, bra_chi: std_chi_r})
        return merged

    T1 = _merge_edge(
        env.T1_ket,
        env.T1_bra,
        "t1k_I",
        "t1b_I",
        "u_ket",
        "u_bra",
        "u2",
        FlowDirection.IN,
        "t1k_l",
        "t1b_r",
        "t1_l",
        "t1_r",
    )
    T2 = _merge_edge(
        env.T2_ket,
        env.T2_bra,
        "t2k_I",
        "t2b_I",
        "r_ket",
        "r_bra",
        "r2",
        FlowDirection.OUT,
        "t2k_u",
        "t2b_d",
        "t2_u",
        "t2_d",
    )
    T3 = _merge_edge(
        env.T3_ket,
        env.T3_bra,
        "t3k_I",
        "t3b_I",
        "d_ket",
        "d_bra",
        "d2",
        FlowDirection.OUT,
        "t3k_r",
        "t3b_l",
        "t3_r",
        "t3_l",
    )
    T4 = _merge_edge(
        env.T4_ket,
        env.T4_bra,
        "t4k_I",
        "t4b_I",
        "l_ket",
        "l_bra",
        "l2",
        FlowDirection.IN,
        "t4k_d",
        "t4b_u",
        "t4_d",
        "t4_u",
    )

    return CTMTensorEnv(
        C1=env.C1,
        C2=env.C2,
        C3=env.C3,
        C4=env.C4,
        T1=T1,
        T2=T2,
        T3=T3,
        T4=T4,
    )


def compute_energy_split_ctm_tensor(
    A: Tensor,
    env: SplitCTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site using split CTM environment.

    Converts to standard Tensor-protocol CTM internally and delegates to
    ``compute_energy_ctm_tensor``. The ket/bra merge over the interlayer
    bond reconstructs the standard double-layer edges.

    Args:
        A:                iPEPS site tensor.
        env:              Converged SplitCTMTensorEnv.
        hamiltonian_gate: 2-site Hamiltonian gate.
        d:                Physical dimension (inferred from A if None).

    Returns:
        Scalar energy per site.
    """
    std_env = _split_env_to_tensor_standard(env)
    return compute_energy_ctm_tensor(A, std_env, hamiltonian_gate, d)
