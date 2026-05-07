"""Paired horizontal/vertical CTM moves with charge-stabilized projectors.

Replaces the 4 independent CTM moves with 2 paired moves (horizontal
and vertical).  Each projector uses ``base_charges`` derived from the
double-layer tensor to force consistent per-sector allocation, preventing
the charge-distribution drift that causes block-size mismatches in
SymmetricTensor after multiple sweeps.
"""

from __future__ import annotations

__all__ = [
    "_ctm_tensor_move_horizontal",
    "_ctm_tensor_move_vertical",
]

import numpy as np

from tenax.algorithms._ctm_projector import (
    _compute_projector_tensor,
    _reembed_fused,
)
from tenax.algorithms._ctm_tensor_init import (
    IN,
    OUT,
    CTMTensorEnv,
    _fuse_pair_by_label,
)
from tenax.algorithms._ctm_tensor_moves import (
    _apply_projector_tensor,
    _flip_leg_flow,
    _normalize_tensor,
)
from tenax.contraction.contractor import contract
from tenax.core.tensor import SymmetricTensor, Tensor


def _get_base_charges(a: Tensor) -> np.ndarray | None:
    """Extract base charges from the double-layer tensor for projector stabilization.

    For SymmetricTensor with non-trivial charges, returns the charges
    of one of the D^2 legs.  For DenseTensor, returns None (no charge
    stabilization needed).
    """
    if not isinstance(a, SymmetricTensor):
        return None
    # Use the u2 leg charges as the base (any D^2 leg would do)
    u2_pos = a.labels().index("u2")
    charges = np.asarray(a.indices[u2_pos].charges, dtype=np.int32)
    # Only stabilize if charges are non-trivial
    if np.all(charges == 0):
        return None
    return charges


def _apply_projector_with_reembed(
    P_1: Tensor,
    P_2: Tensor,
    C1g: Tensor,
    C4g: Tensor,
    Tg: Tensor,
    fused_l: str,
    fused_r: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply projector pair with automatic re-embedding for mismatched fused indices.

    When the projectors have a unified fused index (from combining two corners
    with different charge distributions), re-embeds the grown corners and
    edge to match the projectors' fused dimension before applying.

    For DenseTensor inputs, delegates directly to ``_apply_projector_tensor``.
    """
    if not isinstance(P_1, SymmetricTensor):
        return _apply_projector_tensor(P_1, P_2, C1g, C4g, Tg, fused_l, fused_r)

    # Check if re-embedding is needed
    p_fused_idx = P_1.indices[P_1.labels().index("fused")]

    def _maybe_reembed(T: SymmetricTensor, fused_label: str) -> SymmetricTensor:
        fused_pos = T.labels().index(fused_label)
        current_idx = T.indices[fused_pos]
        if np.array_equal(current_idx.charges, p_fused_idx.charges):
            return T
        # Build target index with the projector's charges but matching the
        # tensor's label/flow.
        from tenax.core.index import TensorIndex

        target_idx = TensorIndex.from_charges(
            p_fused_idx.symmetry,
            p_fused_idx.charges.copy(),
            current_idx.flow,
            label="fused",
        )
        # Temporarily relabel so _reembed_fused can find "fused"
        orig_label = fused_label
        T_tmp = T.relabel(orig_label, "fused")
        T_re = _reembed_fused(T_tmp, target_idx)
        return T_re.relabel("fused", orig_label)

    if isinstance(C1g, SymmetricTensor):
        C1g = _maybe_reembed(C1g, "fused")
        C4g = _maybe_reembed(C4g, "fused")
        Tg = _maybe_reembed(Tg, fused_l)
        Tg = _maybe_reembed(Tg, fused_r)

    return _apply_projector_tensor(P_1, P_2, C1g, C4g, Tg, fused_l, fused_r)


# ------------------------------------------------------------------ #
# Paired horizontal move (left + right)                                #
# ------------------------------------------------------------------ #


def _ctm_tensor_move_horizontal(
    env_self: CTMTensorEnv,
    env_nb: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "svd",
    projector_backward: str = "auto",
) -> CTMTensorEnv:
    """Combined left+right CTM move with charge-stabilized projectors.

    Uses standard 1x1 grown corners (same as individual move_left/move_right)
    but passes ``base_charges`` to the projector computation to ensure
    consistent charge-sector allocation across all projectors.

    Args:
        env_self:  Environment for the site being updated.
        env_nb:    Environment for the neighbor site.
        a:         Double-layer tensor from the neighbor site.
        chi:       Target bond dimension.
        projector_method: ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.

    Returns:
        Updated CTMTensorEnv with all 4 corners and T4, T2 updated.
    """
    base_charges = _get_base_charges(a)

    # ---- LEFT SIDE: same as move_left but with base_charges ----

    # C1g_L = C1 · T1_nb fused
    C1_r = env_self.C1.relabel("c1_r", "t1_l")
    C1g_L = contract(C1_r, env_nb.T1)  # (c1_d, u2, t1_r)
    C1g_L = _fuse_pair_by_label(C1g_L, "c1_d", "u2", "fused", IN)  # (fused, t1_r)

    # C4g_L = C4 · T3_nb fused
    C4_u = env_self.C4.relabel("c4_u", "t3_r")
    C4g_L = contract(C4_u, env_nb.T3)  # (c4_r, d2, t3_l)
    C4g_L = _fuse_pair_by_label(C4g_L, "c4_r", "d2", "fused", IN)  # (fused, t3_l)

    # T4g = T4 · a fused
    T4_with_a = contract(env_self.T4, a)
    T4g = _fuse_pair_by_label(T4_with_a, "t4_d", "u2", "fl", IN)
    T4g = _fuse_pair_by_label(T4g, "t4_u", "d2", "fr", OUT)

    # Projector with charge stabilization
    P_L1, P_L2 = _compute_projector_tensor(
        C1g_L, C4g_L, chi, projector_method, base_charges, projector_backward
    )
    C1_new, C4_new, T4_new = _apply_projector_with_reembed(
        P_L1, P_L2, C1g_L, C4g_L, T4g, "fl", "fr"
    )

    # Relabel left outputs
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1_r": "c1_r"})
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3_l": "c4_u"})
    T4_new = T4_new.relabels({"chi_new": "t4_d", "chi_new_r": "t4_u", "r2": "l2"})
    T4_new = _flip_leg_flow(T4_new, "l2")

    # ---- RIGHT SIDE: same as move_right but with base_charges ----

    # C2g_R = C2 · T1_nb fused
    C2_l = env_self.C2.relabel("c2_l", "t1_r")
    C2g_R = contract(C2_l, env_nb.T1)  # (c2_d, t1_l, u2)
    C2g_R = _fuse_pair_by_label(C2g_R, "c2_d", "u2", "fused", IN)  # (fused, t1_l)

    # C3g_R = C3 · T3_nb fused
    C3_u = env_self.C3.relabel("c3_u", "t3_l")
    C3g_R = contract(C3_u, env_nb.T3)  # (c3_l, t3_r, d2)
    C3g_R = _fuse_pair_by_label(C3g_R, "c3_l", "d2", "fused", IN)  # (fused, t3_r)

    # T2g = T2 · a fused
    T2_with_a = contract(env_self.T2, a)
    T2g = _fuse_pair_by_label(T2_with_a, "t2_u", "u2", "fl", IN)
    T2g = _fuse_pair_by_label(T2g, "t2_d", "d2", "fr", OUT)

    # Projector with charge stabilization
    P_R1, P_R2 = _compute_projector_tensor(
        C2g_R, C3g_R, chi, projector_method, base_charges, projector_backward
    )
    C2_new, C3_new, T2_new = _apply_projector_with_reembed(
        P_R1, P_R2, C2g_R, C3g_R, T2g, "fl", "fr"
    )

    # Relabel right outputs
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t1_l": "c2_d"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3_r": "c3_l"})
    T2_new = T2_new.relabels({"chi_new": "t2_u", "chi_new_r": "t2_d", "l2": "r2"})
    T2_new = _flip_leg_flow(T2_new, "r2")

    # Per-absorption normalization (matches YASTN)
    return CTMTensorEnv(
        C1=_normalize_tensor(C1_new),
        C2=_normalize_tensor(C2_new),
        C3=_normalize_tensor(C3_new),
        C4=_normalize_tensor(C4_new),
        T1=env_self.T1,
        T2=_normalize_tensor(T2_new),
        T3=env_self.T3,
        T4=_normalize_tensor(T4_new),
    )


# ------------------------------------------------------------------ #
# Paired vertical move (top + bottom)                                  #
# ------------------------------------------------------------------ #


def _ctm_tensor_move_vertical(
    env_self: CTMTensorEnv,
    env_nb: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "svd",
    projector_backward: str = "auto",
) -> CTMTensorEnv:
    """Combined top+bottom CTM move with charge-stabilized projectors.

    Args:
        env_self:  Environment for the site being updated.
        env_nb:    Environment for the neighbor site.
        a:         Double-layer tensor from the neighbor site.
        chi:       Target bond dimension.
        projector_method: ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.

    Returns:
        Updated CTMTensorEnv with all 4 corners and T1, T3 updated.
    """
    base_charges = _get_base_charges(a)

    # ---- TOP SIDE: same as move_top but with base_charges ----

    # C1g = C1 · T4_nb fused
    C1_d = env_self.C1.relabel("c1_d", "t4_d")
    C1g = contract(C1_d, env_nb.T4)  # (c1_r, l2, t4_u)
    C1g = _fuse_pair_by_label(C1g, "c1_r", "l2", "fused", IN)  # (fused, t4_u)

    # C2g = C2 · T2_nb fused
    C2_d = env_self.C2.relabel("c2_d", "t2_u")
    C2g = contract(C2_d, env_nb.T2)  # (c2_l, r2, t2_d)
    C2g = _fuse_pair_by_label(C2g, "c2_l", "r2", "fused", IN)  # (fused, t2_d)

    # T1g = T1 · a fused
    T1_with_a = contract(env_self.T1, a)
    T1g = _fuse_pair_by_label(T1_with_a, "t1_l", "l2", "fl", IN)
    T1g = _fuse_pair_by_label(T1g, "t1_r", "r2", "fr", OUT)

    # Projector with charge stabilization
    P_T1, P_T2 = _compute_projector_tensor(
        C1g, C2g, chi, projector_method, base_charges, projector_backward
    )
    C1_new, C2_new, T1_new = _apply_projector_with_reembed(
        P_T1, P_T2, C1g, C2g, T1g, "fl", "fr"
    )

    # Relabel top outputs
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t4_u": "c1_r"})
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2_d": "c2_d"})
    T1_new = T1_new.relabels({"chi_new": "t1_l", "chi_new_r": "t1_r", "d2": "u2"})
    T1_new = _flip_leg_flow(T1_new, "u2")

    # ---- BOTTOM SIDE: same as move_bottom but with base_charges ----

    # C4g = C4 · T4_nb fused
    C4_r = env_self.C4.relabel("c4_r", "t4_u")
    C4g = contract(C4_r, env_nb.T4)  # (c4_u, t4_d, l2)
    C4g = _fuse_pair_by_label(C4g, "c4_u", "l2", "fused", IN)  # (fused, t4_d)

    # C3g = C3 · T2_nb fused
    C3_l = env_self.C3.relabel("c3_l", "t2_d")
    C3g = contract(C3_l, env_nb.T2)  # (c3_u, t2_u, r2)
    C3g = _fuse_pair_by_label(C3g, "c3_u", "r2", "fused", IN)  # (fused, t2_u)

    # T3g = T3 · a fused
    T3_with_a = contract(env_self.T3, a)
    T3g = _fuse_pair_by_label(T3_with_a, "t3_r", "l2", "fl", IN)
    T3g = _fuse_pair_by_label(T3g, "t3_l", "r2", "fr", OUT)

    # Projector with charge stabilization
    P_B1, P_B2 = _compute_projector_tensor(
        C4g, C3g, chi, projector_method, base_charges, projector_backward
    )
    C4_new, C3_new, T3_new = _apply_projector_with_reembed(
        P_B1, P_B2, C4g, C3g, T3g, "fl", "fr"
    )

    # Relabel bottom outputs
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t4_d": "c4_u"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t2_u": "c3_l"})
    T3_new = T3_new.relabels({"chi_new": "t3_r", "chi_new_r": "t3_l", "u2": "d2"})
    T3_new = _flip_leg_flow(T3_new, "d2")

    # Per-absorption normalization (matches YASTN)
    return CTMTensorEnv(
        C1=_normalize_tensor(C1_new),
        C2=_normalize_tensor(C2_new),
        C3=_normalize_tensor(C3_new),
        C4=_normalize_tensor(C4_new),
        T1=_normalize_tensor(T1_new),
        T2=env_self.T2,
        T3=_normalize_tensor(T3_new),
        T4=env_self.T4,
    )
