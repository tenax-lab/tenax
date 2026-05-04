"""Split CTM with Tensor protocol — conversion and energy computation."""

from __future__ import annotations

__all__ = [
    "_rdm1x2_split_tensor",
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


def _rdm1x2_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Vertical 1x2 RDM via split-aware contraction. Bounded peak chi^2 * D^4.

    Top half contraction order (mirrored for bottom)::

        top_row     = C1 . T1_split . C2                # chi^2 * D^2
        top_T4      = top_row . T4_T_split              # chi^2 * D^4   (peak edge stage)
        top_T4_A    = top_T4 . A_ket                    # chi^2 * D^4 * d (peak overall)
        top_T4_A_T2 = top_T4_A . T2_T_split             # chi^2 * D^3 * d
        top_half    = top_T4_A_T2 . A_bra               # chi^2 * D^2 * d^2

    Combine on (t4_u<->t4_uB chi seam, t2_d<->t2_dB chi seam,
    d_ket<->u_ketB inner-D, d_bra<->u_braB inner-D).

    Returns dense RDM of shape ``(d, d, d, d)`` in
    ``(s1_ket, s2_ket, s1_bra, s2_bra)``, symmetrised and trace-normalised.
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    A_bra = A.bar_super().relabels(
        {
            "u": "u_bra",
            "d": "d_bra",
            "l": "l_bra",
            "r": "r_bra",
            "phys": "phys_bra",
        }
    )

    # ---------- Top half ----------
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    top_row = contract(contract(C1, T1), C2)  # (c1_d, u_ket, u_bra, c2_d)

    T4_T = T4.relabels({"t4_d": "c1_d"})  # (c1_d, l_ket, l_bra, t4_u)
    T2_T = T2.relabels({"t2_u": "c2_d"})  # (c2_d, r_ket, r_bra, t2_d)

    A_top = A.relabels({"u": "u_ket", "l": "l_ket", "r": "r_ket"})

    top_T4 = contract(top_row, T4_T)  # chi^2 * D^4
    top_T4_A = contract(top_T4, A_top)  # chi^2 * D^4 * d (peak)
    top_T4_A_T2 = contract(top_T4_A, T2_T)  # chi^2 * D^3 * d
    top_half = contract(top_T4_A_T2, A_bra)  # chi^2 * D^2 * d^2

    # ---------- Bottom half ----------
    # Suffix bottom-site labels with "B" so they don't collide with the top half.
    bot_row_C4 = env.C4.relabel("c4_u", "t3_r")
    bot_row_C3 = env.C3.relabel("c3_l", "t3_l")
    bot_row = contract(contract(bot_row_C4, T3), bot_row_C3)
    # bot_row: (c4_r, d_ket, d_bra, c3_u) — the d_ket/d_bra here are T3's D-legs,
    # i.e. the bottom site's *down* legs. Rename to d_ketB/d_braB so they don't
    # collide with the top-half seam labels (top A's bare ``d`` leg and A_bra's
    # ``d_bra`` leg are the seam, not the bottom site's down-legs).
    bot_row = bot_row.relabels({"d_ket": "d_ketB", "d_bra": "d_braB"})

    T4_B = T4.relabels(
        {
            "t4_u": "c4_r",  # T4's chi-up matches C4_r at the bottom seam
            "t4_d": "t4_uB",  # other chi-end becomes the bottom-half top-seam
            "l_ket": "l_ketB",
            "l_bra": "l_braB",
        }
    )
    T2_B = T2.relabels(
        {
            "t2_d": "c3_u",
            "t2_u": "t2_dB",
            "r_ket": "r_ketB",
            "r_bra": "r_braB",
        }
    )

    A_bot = A.relabels(
        {
            "u": "u_ketB",  # OPEN — becomes seam to top_half's ``d`` leg
            "l": "l_ketB",
            "r": "r_ketB",
            "d": "d_ketB",  # matches bot_row's d_ketB (T3 inner-D)
            "phys": "phys_B",
        }
    )
    A_bra_bot = A_bra.relabels(
        {
            "u_bra": "u_braB",  # OPEN — becomes seam to top_half's d_bra
            "l_bra": "l_braB",  # matches T4_B's l_braB
            "r_bra": "r_braB",  # matches T2_B's r_braB
            "d_bra": "d_braB",  # matches bot_row's d_braB (T3 inner-D)
            "phys_bra": "phys_braB",
        }
    )

    bot_T4 = contract(bot_row, T4_B)  # chi^2 * D^4
    bot_T4_A = contract(bot_T4, A_bot)  # chi^2 * D^4 * d (peak)
    bot_T4_A_T2 = contract(bot_T4_A, T2_B)  # chi^2 * D^3 * d
    bot_half = contract(bot_T4_A_T2, A_bra_bot)  # chi^2 * D^2 * d^2

    # ---------- Combine ----------
    # Open labels on top_half:  (t4_u, d, phys, t2_d, d_bra, phys_bra)
    # Open labels on bot_half:  (t4_uB, u_ketB, phys_B, t2_dB, u_braB, phys_braB)
    # Seam labels:
    #   t4_u  (top T4 chi-up)   <-> t4_uB  (bot T4 chi-up)
    #   t2_d  (top T2 chi-down) <-> t2_dB  (bot T2 chi-down)
    #   d     (top A d-leg)     <-> u_ketB (bot A u-leg)
    #   d_bra (top A_bra d-leg) <-> u_braB (bot A_bra u-leg)
    bot_half = bot_half.relabels(
        {
            "t4_uB": "t4_u",
            "t2_dB": "t2_d",
            "u_ketB": "d",
            "u_braB": "d_bra",
        }
    )

    rdm_t = contract(
        top_half,
        bot_half,
        output_labels=["phys", "phys_B", "phys_bra", "phys_braB"],
    )
    # -> (s1_ket, s2_ket, s1_bra, s2_bra)

    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)


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
