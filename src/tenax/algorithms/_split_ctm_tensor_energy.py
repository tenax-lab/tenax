"""Split CTM with Tensor protocol — conversion and energy computation."""

from __future__ import annotations

__all__ = [
    "_rdm_1site_split_tensor",
    "_split_env_to_tensor_standard",
    "compute_energy_split_ctm_tensor",
]

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.index import FlowDirection
from tenax.core.tensor import Tensor

# ------------------------------------------------------------------ #
# Energy computation (split, no double-layer)                          #
# ------------------------------------------------------------------ #


def _make_split_edge(
    T_ket: Tensor,
    T_bra: Tensor,
    ket_I: str,
    bra_I: str,
    ket_chi: str,
    bra_chi: str,
    out_chi_l: str,
    out_chi_r: str,
) -> Tensor:
    """Contract T_ket·T_bra over the interlayer bond; do NOT fuse the two D-legs.

    Returns a 4-leg tensor with labels (out_chi_l, <ket D-label>, <bra D-label>, out_chi_r).
    The D-leg labels are inherited from the inputs (e.g. ``u_ket``/``u_bra`` for T1).
    Leaving the D-legs unfused lets downstream RDM construction contract them
    directly against the ket/bra layers of A, avoiding the chi^2 * D^4 double-layer
    peak that the standard shim (``_split_env_to_tensor_standard``) produces.
    """
    # Use ``_I_tmp`` rather than ``_I`` to avoid label collisions when several
    # split edges built by this helper are later contracted together in the same
    # RDM network. The standard shim's ``_merge_edge`` uses ``_I`` because each
    # call there is contracted in isolation.
    k = T_ket.relabel(ket_I, "_I_tmp")
    b = T_bra.relabel(bra_I, "_I_tmp")
    merged = contract(k, b)
    return merged.relabels({ket_chi: out_chi_l, bra_chi: out_chi_r})


def _make_split_edges(env: SplitCTMTensorEnv) -> dict[str, Tensor]:
    """Build 4-leg split edges for all four boundary T's.

    Returns a dict keyed ``"T1"``, ``"T2"``, ``"T3"``, ``"T4"`` with each value
    a 4-leg ``(chi, D_ket, D_bra, chi)`` tensor. D-leg labels are
    ``{u,r,d,l}_ket`` / ``{u,r,d,l}_bra`` (inherited from inputs and globally
    unique across the four edges). Chi-leg labels follow the standard
    ``CTMTensorEnv`` convention: ``t1_l/t1_r``, ``t2_u/t2_d``, ``t3_r/t3_l``,
    ``t4_d/t4_u``.
    """
    return {
        "T1": _make_split_edge(
            env.T1_ket,
            env.T1_bra,
            ket_I="t1k_I",
            bra_I="t1b_I",
            ket_chi="t1k_l",
            bra_chi="t1b_r",
            out_chi_l="t1_l",
            out_chi_r="t1_r",
        ),
        "T2": _make_split_edge(
            env.T2_ket,
            env.T2_bra,
            ket_I="t2k_I",
            bra_I="t2b_I",
            ket_chi="t2k_u",
            bra_chi="t2b_d",
            out_chi_l="t2_u",
            out_chi_r="t2_d",
        ),
        "T3": _make_split_edge(
            env.T3_ket,
            env.T3_bra,
            ket_I="t3k_I",
            bra_I="t3b_I",
            ket_chi="t3k_r",
            bra_chi="t3b_l",
            out_chi_l="t3_r",
            out_chi_r="t3_l",
        ),
        "T4": _make_split_edge(
            env.T4_ket,
            env.T4_bra,
            ket_I="t4k_I",
            bra_I="t4b_I",
            ket_chi="t4k_d",
            bra_chi="t4b_u",
            out_chi_l="t4_d",
            out_chi_r="t4_u",
        ),
    }


def _rdm_1site_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Single-site RDM via split-aware contraction.

    Same network topology as ``_rdm_1site_tensor`` but uses 4-leg split edges
    and keeps ``A`` and ``A.bar_super()`` separate (no double-layer fusion).

    Note: 1-site has an intrinsic chi^2 * D^6 frame stage; the chi^2 * D^4 peak
    target of the design only applies to the 2x1/1x2/diagonal RDMs (Tasks 4-6).
    1-site exists for parity testing and small-D probes; energy never uses it
    for nearest-neighbour bonds.
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    A_ket = A.relabels({"u": "u_ket", "d": "d_ket", "l": "l_ket", "r": "r_ket"})
    A_bra = A.bar_super().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    # Build the boundary frame: corners + four split T's, all chi-bonds matched.
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    top_row = contract(contract(C1, T1), C2)  # (c1_d, u_ket, u_bra, c2_d)

    C4 = env.C4.relabel("c4_u", "t3_r")
    C3 = env.C3.relabel("c3_l", "t3_l")
    bot_row = contract(contract(C4, T3), C3)  # (c4_r, d_ket, d_bra, c3_u)

    T4_e = T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    T2_e = T2.relabels({"t2_u": "c2_d", "t2_d": "c3_u"})

    # Frame: top·T4·T2·bot. Pairwise so the 4-leg split edges keep their
    # D_ket/D_bra structure (no D^2 fuse anywhere).
    frame_top = contract(top_row, T4_e)
    frame_full = contract(frame_top, T2_e)
    frame_full = contract(frame_full, bot_row)
    # frame_full has 8 D-legs: u_ket, u_bra, l_ket, l_bra, r_ket, r_bra,
    # d_ket, d_bra.

    # Absorb A then A_bra. A_ket's u/l/r/d match the ket D-legs; A_bra's match bra.
    rdm_t = contract(frame_full, A_ket)  # consumes u_ket, l_ket, r_ket, d_ket
    rdm_t = contract(
        rdm_t, A_bra, output_labels=["phys", "phys_bra"]
    )  # consumes u_bra, l_bra, r_bra, d_bra

    rdm = rdm_t.todense()
    rdm = 0.5 * (rdm + rdm.conj().T)
    rdm = rdm / (jnp.trace(rdm) + EPS)
    return rdm


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
