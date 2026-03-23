"""Standard CTM with Tensor protocol — projector application and directional moves."""

from __future__ import annotations

__all__ = [
    "_apply_projector_tensor",
    "_ctm_tensor_move_bottom",
    "_ctm_tensor_move_left",
    "_ctm_tensor_move_right",
    "_ctm_tensor_move_top",
]

from tenax.algorithms._ctm_projector import (
    _compute_projector_tensor,
)
from tenax.algorithms._ctm_tensor_init import (
    IN,
    OUT,
    CTMTensorEnv,
    _fuse_pair_by_label,
)
from tenax.contraction.contractor import contract
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def _flip_leg_flow(tensor: Tensor, label: str) -> Tensor:
    """Flip the FlowDirection of a single leg identified by label.

    For SymmetricTensor this is required after CTM moves relabel the D²
    leg (e.g. d2→u2) because the inherited flow from the double-layer
    tensor is opposite to the edge spec.  DenseTensors are returned
    unchanged since flow is cosmetic for non-fermionic tensors.
    """
    if isinstance(tensor, DenseTensor):
        return tensor  # flow doesn't affect DenseTensor contractions
    new_indices = []
    for idx in tensor.indices:
        if idx.label == label:
            new_indices.append(idx.flip_flow())
        else:
            new_indices.append(idx)
    # Use _raw to skip conservation validation: the flow flip is an
    # intentional convention change (matching the CTMTensorEnv label spec),
    # not a physics change.  The block keys remain valid for contraction.
    return SymmetricTensor._raw(
        indices=tuple(new_indices),
        data=tensor._data,
        block_keys=tensor._block_keys,
        block_shapes=tensor._block_shapes,
        block_offsets=tensor._block_offsets,
    )


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
    T4_new = _flip_leg_flow(T4_new, "l2")  # r2(OUT) -> l2 needs IN

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
    T2_new = _flip_leg_flow(T2_new, "r2")  # l2(IN) -> r2 needs OUT

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
    T1_new = _flip_leg_flow(T1_new, "u2")  # d2(OUT) -> u2 needs IN

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
    T3_new = _flip_leg_flow(T3_new, "d2")  # u2(IN) -> d2 needs OUT

    return env_self._replace(C4=C4_new, C3=C3_new, T3=T3_new)
