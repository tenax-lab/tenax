"""Standard CTM with Tensor protocol — projector application and directional moves."""

from __future__ import annotations

__all__ = [
    "_apply_projector_tensor",
    "_compute_plaquette_projector_pair",
    "_ctm_tensor_absorb_bottom_2plaq",
    "_ctm_tensor_absorb_left_2plaq",
    "_ctm_tensor_absorb_right_2plaq",
    "_ctm_tensor_absorb_top_2plaq",
    "_ctm_tensor_move_bottom",
    "_ctm_tensor_move_bottom_2x2",
    "_ctm_tensor_move_left",
    "_ctm_tensor_move_left_2x2",
    "_ctm_tensor_move_right",
    "_ctm_tensor_move_right_2x2",
    "_ctm_tensor_move_top",
    "_ctm_tensor_move_top_2x2",
]

import jax
import numpy as np

from tenax.algorithms._ctm_chunked_absorb import (
    _chunked_T_new_bottom,
    _chunked_T_new_left,
    _chunked_T_new_right,
    _chunked_T_new_top,
    _raw_in_label_order,
)
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
from tenax.algorithms._ctm_tensor_projector_2x2 import (
    _build_enlarged_corner,
    _compute_2x2_projector,
)
from tenax.algorithms._tensor_utils import split_index
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def _normalize_tensor(T: Tensor) -> Tensor:
    """Normalize tensor by max abs value (inf-norm)."""
    norm = T.max_abs()
    return T * (1.0 / (norm + EPS))


def _flow_flip_no_conj(t):
    """Flip flow direction on every leg WITHOUT conjugating the data.

    For ``SymmetricTensor``, flipping every leg's flow preserves the
    conservation law ``sum(flow_i * charge_i) = 0`` (the sign flips on
    both sides), so block keys are unchanged.  The flat buffer and
    block metadata can be reused via ``_raw``, mirroring how
    :meth:`SymmetricTensor.bar` constructs its result (only the data
    conjugation is dropped).
    """
    new_indices = tuple(idx.flip_flow() for idx in t.indices)
    if isinstance(t, DenseTensor):
        return DenseTensor(t._data, new_indices)
    return SymmetricTensor._raw(
        indices=new_indices,
        data=t._data,
        block_keys=t._block_keys,
        block_shapes=t._block_shapes,
        block_offsets=t._block_offsets,
    )


def _apply_proj_unfused(
    P: Tensor,
    env_T: Tensor,
    chi_label: str,
    d2_label: str,
    *,
    chi_new: str = "chi_new",
    env_first: bool = False,
) -> Tensor:
    """Project ``env_T`` with ``P`` by UNFUSING the projector's fused leg first.

    The 2-plaquette projector arrives with a hard-fused ``(chi_outer, fused_D2)``
    leg (label ``"fused"``).  Hard-fusing the env-``chi`` and ``D²`` legs (two
    flow-OUT constituents) under a single flow-IN signature produces fused
    charges that are the **charge-conjugate** of the env edge/corner's own
    fused leg, so the raw-charge block pairing in :func:`contract` clashes at
    D≥3 (issue #605; masked at D=2 where the Sz seam charge set is self-dual).

    The fix mirrors YASTN's CTM (`proj_corners` does ``.unfuse_legs`` before
    applying): split the projector back into its bare ``(chi_outer, fused_D2)``
    constituents and contract *those* against the env tensor's own ``chi_label``
    / ``d2_label`` legs.  The constituents are individually contractible (equal
    charges, opposite flow — they are literally the seam's two sides), so the
    manufactured conjugation never arises.  Splitting also decouples the new
    bond ``chi_new``: its flow is taken straight from the projector, so the
    renormalised corner/edge chi bonds keep the projector-pair's mutually
    opposite flows without any flow-flip.

    No flow flip is applied (#905).  ``_flow_flip_no_conj`` preserves the
    conservation law only while *every* leg is flipped together, but
    ``split_index`` restores the parent legs with the flows recorded in
    ``fuse_info`` — so the flip survived on ``chi_new`` alone and left
    ``net(P_un) = 2 * q_chi_new``.  Only the ``q_chi_new = 0`` column was
    self-consistent; :func:`contract` then correctly refused every charged
    block product (the #834 guard), which is what annihilated the charged
    environment sectors.  The projector's ``chi_outer`` / ``fused_D2`` legs
    are inherited verbatim from the enlarged corner, i.e. from the env
    tensors on the *other* side of the seam being cut, so they are already
    the exact duals of the env legs they contract against; nothing needs
    flipping.

    ``env_first`` selects the contraction operand order: ``contract(P_un, env_T)``
    by default, or ``contract(env_T, P_un)`` when ``env_first=True``.  This must
    match the operand order of the fuse-then-contract path it replaces — for
    graded (fermionic) tensors swapping the operands' free legs introduces a
    Koszul sign, so e.g. the edge's second (``fr``) projection, which the fused
    path applied as ``contract(step, P_right)``, must use ``env_first=True``.
    Matching the order also yields the canonical output leg layout directly, so
    no Koszul-bearing ``transpose`` is needed afterwards.

    The *unfusing* itself is value-preserving (fusion is invertible; the
    constituents pair the same elements the fused leg did), so on
    ``DenseTensor`` — whose contraction ignores flow entirely — this whole
    function is byte-for-byte what fuse-then-contract was, before and after
    #905 (measured: the dense D=2 energy is bit-identical at chi=8 and chi=16).
    Dropping the flip is what changes the *symmetric* result, at every D: the
    self-duality of D=2's ``{0, ±2}`` Sz seam masks a conjugated charge *list*,
    not a broken conservation law, and it was the latter that annihilated the
    charged sectors.
    """
    P_un = split_index(P, P.labels().index("fused"))
    relabel = {"chi_outer": chi_label, "fused_D2": d2_label}
    if chi_new != "chi_new":
        relabel["chi_new"] = chi_new
    P_un = P_un.relabels(relabel)
    return contract(env_T, P_un) if env_first else contract(P_un, env_T)


def _env_is_fermionic(env_src: CTMTensorEnv) -> bool:
    """True when the env tensors are graded (fermionic) ``SymmetricTensor``s.

    The unfused projector application (#605) is byte-identical to the original
    fuse-then-contract for DenseTensor and bosonic ``SymmetricTensor`` (verified
    per-absorb across all 4 directions), and it is what fixes the D≥3 U(1)-Sz
    charge-sector mismatch.  But for **fermionic** tensors the fused path carries
    Koszul signs from grouping the chi⊕D² edge legs (``fr = fuse(t4_u, d2)``
    stays a single graded leg through the ``fl`` contraction); the unfused
    sandwich contracts the bare legs and accumulates a *different* Koszul sign on
    the renormalised EDGE (corners are unaffected).  Since #605 cannot occur for
    FermionParity (Z₂ charges are self-dual, so the fused legs never come out
    asymmetric-conjugate), fermionic envs keep the proven fused path.
    """
    C1 = env_src.C1
    return isinstance(C1, SymmetricTensor) and C1.indices[0].symmetry.is_fermionic


def _phase_fix_normalize_tensor(T: Tensor) -> Tensor:
    """Frobenius-normalize + phase-fix a single C or T tensor.

    Matches variPEPS ``_post_process_CTM_tensors``: normalize by Frobenius
    norm, then fix the global U(1) phase by making the first element
    with ``|x| >= 0.1 * max(|arr|)`` real-positive.  This pins the
    sign/phase of each tensor so that consecutive CTM sweeps see a
    sign-stable fixed point, which regularizes the Jacobian ``J^T``
    in implicit AD.

    The "first above threshold" rule is computed in dense layout order
    so that the symmetric and dense paths agree element-by-element on
    the converged env tensors (preserving
    ``test_energy_symmetric_matches_dense``). For SymmetricTensor with
    non-trivial charges, the dense and buffer orders pick different
    anchors, which causes the symmetric env to drift from the dense
    one. The C and T tensors are small (``chi×chi`` corners,
    ``chi×D²×chi`` edges), so the per-sweep ``todense() + from_dense()``
    cost is negligible — see ``test_symmetric_sweep_no_todense`` for
    the architecture guard's justified-exemption note.

    The Frobenius norm is wrapped in ``stop_gradient`` so the backward
    sees only the tangential gradient. At the converged fixed point
    ``||T||=1`` already, so the radial component of the cotangent is
    discarded by the outer unit-sphere projection anyway. Without this,
    SymmetricTensor inputs with empty charge sectors (e.g. fermionic
    fpeps tensors) produce NaN gradients through the implicit-AD
    backward — the radial-gradient path interacts with the dense
    layout's out-of-block zeros to create 0/0 leaves. (#362)
    """
    import jax
    import jax.numpy as jnp

    arr = T.todense()
    norm = jnp.linalg.norm(arr)
    arr = arr / jax.lax.stop_gradient(norm + 1e-30)
    flat = arr.ravel()
    abs_flat = jnp.abs(flat)
    threshold = 0.1 * jnp.max(abs_flat)
    idx = jnp.argmax(abs_flat >= threshold)  # first element above threshold
    phase = flat[idx] / (jnp.abs(flat[idx]) + 1e-30)
    arr = arr * jnp.conj(phase)
    if isinstance(T, SymmetricTensor):
        return SymmetricTensor.from_dense(arr, T.indices, tol=float("inf"))
    return DenseTensor(arr, T.indices)


def _flip_leg_flow(tensor: Tensor, label: str) -> Tensor:
    """Flip the FlowDirection of a single leg identified by label.

    For SymmetricTensor this is required after CTM moves relabel the D²
    leg (e.g. d2→u2) because the inherited flow from the double-layer
    tensor is opposite to the edge spec.  DenseTensors are returned
    unchanged since flow is cosmetic for non-fermionic tensors.

    For non-trivial charges, this duals the leg (flips flow + negates
    charges) and remaps the corresponding block keys so that the
    conservation law ``sum(flow_i * charge_i) = 0`` is preserved.
    """
    if isinstance(tensor, DenseTensor):
        return tensor  # flow doesn't affect DenseTensor contractions

    axis = None
    new_indices = []
    for i, idx in enumerate(tensor.indices):
        if idx.label == label:
            axis = i
            new_indices.append(idx.dual())
        else:
            new_indices.append(idx)

    if axis is None:
        return SymmetricTensor(tensor.blocks, tuple(new_indices))

    # For trivial charges, block keys don't change — use _raw to skip
    # validation (the flow flip is an intentional convention change).
    old_idx = tensor.indices[axis]
    if all(c == 0 for c in old_idx.charges):
        return SymmetricTensor._raw(
            indices=tuple(new_indices),
            data=tensor._data,
            block_keys=tensor._block_keys,
            block_shapes=tensor._block_shapes,
            block_offsets=tensor._block_offsets,
        )

    # Non-trivial charges: remap block keys to use dual charges
    sym = old_idx.symmetry
    dual_charges = sym.dual(old_idx.charges)
    charge_to_dual = {int(o): int(d) for o, d in zip(old_idx.charges, dual_charges)}

    new_blocks = {}
    for key, val in tensor.blocks.items():
        new_key = list(key)
        new_key[axis] = charge_to_dual[key[axis]]
        new_blocks[tuple(new_key)] = val

    return SymmetricTensor(new_blocks, tuple(new_indices))


def _apply_projector_tensor(
    P_1: Tensor,
    P_2: Tensor,
    C1g: Tensor,
    C4g: Tensor,
    Tg: Tensor,
    fused_l: str,
    fused_r: str,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Apply projector pair to grown corners and edge.

    Uses the two-projector formulation (arXiv:2502.10298 Eq. 10):

    .. math::
        C_1' = P_1^\dagger C_{1g}, \quad
        C_4' = P_2^\dagger C_{4g}, \quad
        T'   = P_1^\dagger T_g P_2

    ``P_1`` acts on the C1g side (fused_l of the edge), ``P_2`` on the
    C4g side (fused_r).  For isometric projectors (eigh/qr), ``P_1 = P_2``
    and this reduces to the standard single-projector formulation.

    Args:
        P_1:     Projector for C1g side, labels ``(fused, chi_new)``.
        P_2:     Projector for C4g side, labels ``(fused, chi_new)``.
        C1g:     Grown corner ``(fused, col1)``.
        C4g:     Grown corner ``(fused, col2)``.
        Tg:      Grown edge ``(fused_l, D2_label, fused_r)``.
        fused_l: Label of Tg's left fused leg (C1g side).
        fused_r: Label of Tg's right fused leg (C4g side).

    Returns:
        ``(C1_new, C4_new, T_new)`` as Tensor objects.
    """
    # Projector .bar() (no Koszul): the absorb step P1_bar @ ... @ P_2 is
    # a paired contraction. Post-#555 the contractor no longer applies any
    # auto-Koszul on fermionic legs, and ``bar()`` carries no per-block phase,
    # so projector pair-cancellation works uniformly across bosonic and
    # fermionic A.
    P1_bar = P_1.bar()  # (fused_OUT, chi_new_IN) — contracts on "fused"
    P2_bar = P_2.bar()

    C1_new = contract(P1_bar, C1g)  # (chi_new, col1)
    C4_new = contract(P2_bar, C4g)  # (chi_new, col2)

    # Sandwich: P_1^bar @ Tg @ P_2  (left fused, then right fused)
    P_left = P1_bar.relabel("fused", fused_l)
    step = contract(P_left, Tg)  # (chi_new, D2, fused_r)

    P_right = P_2.relabels({"fused": fused_r, "chi_new": "chi_new_r"})
    T_new = contract(step, P_right)  # (chi_new, D2, chi_new_r)

    return C1_new, C4_new, T_new


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

    p_fused_idx = P_1.indices[P_1.labels().index("fused")]

    def _maybe_reembed(T: SymmetricTensor, fused_label: str) -> SymmetricTensor:
        fused_pos = T.labels().index(fused_label)
        current_idx = T.indices[fused_pos]
        if np.array_equal(current_idx.charges, p_fused_idx.charges):
            return T
        from tenax.core.index import TensorIndex

        target_idx = TensorIndex.from_charges(
            p_fused_idx.symmetry,
            p_fused_idx.charges.copy(),
            current_idx.flow,
            label="fused",
        )
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
# 2x2 plaquette projector + 2-plaquette absorption helpers            #
# ------------------------------------------------------------------ #
#
# These helpers replace the prior "1x1 absorption + 2x2 projector" path
# (the ``_ctm_tensor_move_*_2x2`` functions below) with the variPEPS
# style: TWO plaquettes per cell so that LEFT/RIGHT use ABOVE+CURRENT
# and TOP/BOTTOM use LEFT+CURRENT.  This is required for multisite
# unitcells (e.g. kagome 3-site) where the single-plaquette absorption
# leaks RDMs below the spin-1/2 AFH per-bond floor — see
# ``project_c3_floor_breach_smoking_gun.md``.


def _half_to_chi_new_top(P_top: Tensor) -> Tensor:
    """Convert raw P_top ``(chi_outer, fused_D2, chi_new_top)`` to ``(fused, chi_new)``.

    Fuses ``(chi_outer, fused_D2)`` into a single ``fused`` leg with flow
    IN, and renames ``chi_new_top`` to ``chi_new``.  The result has the
    layout consumed by :func:`_apply_projector_tensor`.
    """
    P = _fuse_pair_by_label(P_top, "chi_outer", "fused_D2", "fused", IN)
    return P.relabel("chi_new_top", "chi_new")


def _half_to_chi_new_bot(P_bot: Tensor) -> Tensor:
    """Convert raw P_bot ``(chi_new_bot, chi_outer, fused_D2)`` to ``(chi_new, fused)``.

    Fuses ``(chi_outer, fused_D2)`` into a single ``fused`` leg (flow IN)
    and renames ``chi_new_bot`` to ``chi_new``.
    """
    P = _fuse_pair_by_label(P_bot, "chi_outer", "fused_D2", "fused", IN)
    return P.relabel("chi_new_bot", "chi_new")


def _compute_plaquette_projector_pair(
    env_TL: CTMTensorEnv,
    env_TR: CTMTensorEnv,
    env_BL: CTMTensorEnv,
    env_BR: CTMTensorEnv,
    a_TL: Tensor,
    a_TR: Tensor,
    a_BL: Tensor,
    a_BR: Tensor,
    chi: int,
    direction: str,
    base_charges: np.ndarray | None = None,
) -> tuple[Tensor, Tensor, jax.Array, jax.Array]:
    """Compute the (P_top, P_bot, eps_T, smallest_S) projector quad for a 2x2 plaquette.

    The plaquette is anchored at TL with TR=neighbors[TL]['right'],
    BL=neighbors[TL]['bottom'], BR=neighbors[TR]['bottom'].

    For ``direction in ("left", "right")``, the cut seam is the LEFT or
    RIGHT vertical seam between two stacked-cell pairs.  For
    ``direction in ("top", "bottom")``, the cut seam is the TOP or
    BOTTOM horizontal seam.

    The returned ``(P_top, P_bot)`` are pre-fused into the
    ``(fused, chi_new)`` / ``(chi_new, fused)`` layout consumed by
    :func:`_apply_projector_tensor`, with::

        P_top labels: ("fused", "chi_new"),   flow IN on "fused"
        P_bot labels: ("chi_new", "fused"),   flow IN on "fused"

    ``eps_T`` is the variPEPS §2.8.2 truncation-error scalar driving
    end-of-outer-step ``chi_auto_bump`` (Issue #474); ``smallest_S`` is
    the variPEPS ``norm_smallest_S`` scalar driving in-CTM χ-bump
    (Issue #492); ``stop_gradient`` applied upstream to both.

    For LEFT/RIGHT direction, ``P_top`` projects the BOTTOM face of TL
    (or top face of TR for direction='right') and ``P_bot`` projects the
    TOP face of BL (or bottom face of BR).  See ``_compute_2x2_projector``.
    """
    Q_TL = _build_enlarged_corner(
        env_TL.C1, env_TL.T1, env_TL.T4, a_TL, position="top_left"
    )
    Q_TR = _build_enlarged_corner(
        env_TR.C2, env_TR.T1, env_TR.T2, a_TR, position="top_right"
    )
    Q_BL = _build_enlarged_corner(
        env_BL.C4, env_BL.T3, env_BL.T4, a_BL, position="bottom_left"
    )
    Q_BR = _build_enlarged_corner(
        env_BR.C3, env_BR.T3, env_BR.T2, a_BR, position="bottom_right"
    )
    P_top_raw, P_bot_raw, eps_T, smallest_S = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction=direction, base_charges=base_charges
    )
    return (
        _half_to_chi_new_top(P_top_raw),
        _half_to_chi_new_bot(P_bot_raw),
        eps_T,
        smallest_S,
    )


def _ctm_tensor_absorb_left_2plaq(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_above: Tensor,
    P_bot_above: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style LEFT absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_left_absorption`:

      * ``new_C1 = (C1[s_src] · T1[s_src]) projected by P_bot_above``
      * ``new_T4 = (T4[s_src] · ket[s_src]) sandwiched by P_top_above
        (top face) and P_bot_curr (bottom face)``
      * ``new_C4 = (C4[s_src] · T3[s_src]) projected by P_top_curr``

    where ``P_*_above`` are halves of the projector pair for the plaquette
    anchored at ``neighbors[s_src]["top"]`` (compressing the seam between
    that cell and ``s_src``), and ``P_*_curr`` are halves for the
    plaquette anchored at ``s_src`` (compressing the seam between
    ``s_src`` and ``neighbors[s_src]["bottom"]``).

    The new env tensors should be stored at ``s_dst = neighbors[s_src]
    ["right"]``: this is the cell whose left edge boundary is the column
    just absorbed.
    """
    # Fermionic envs keep the Koszul-correct fused path (see _env_is_fermionic).
    if _env_is_fermionic(env_src):
        return _ctm_tensor_absorb_left_2plaq_fused(
            env_src, a_src, P_top_above, P_bot_above, P_top_curr, P_bot_curr
        )
    # Grown corners / edge are kept UNFUSED; projectors are applied via
    # ``_apply_proj_unfused`` (split the projector's fused leg and contract the
    # bare chi / D² constituents) to avoid the hard-fusion charge-conjugation
    # that breaks the raw-charge block pairing at D≥3 (#605).
    # ---- C1·T1 (both at s_src) ----
    C1_r = env_src.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env_src.T1)  # (c1_d, u2, t1_r)

    # ---- C4·T3 (both at s_src) ----
    C4_u = env_src.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env_src.T3)  # (c4_r, d2, t3_l)

    # ---- T4·ket (T4 and ket both at s_src) ----
    T4_with_a = contract(env_src.T4, a_src)  # (t4_d, t4_u, u2, d2, r2)

    # ---- C1: project with P_bot_above only ----
    C1_new = _apply_proj_unfused(P_bot_above, C1g, "c1_d", "u2")  # (chi_new, t1_r)
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1_r": "c1_r"})

    # ---- C4: project with P_top_curr only ----
    C4_new = _apply_proj_unfused(P_top_curr, C4g, "c4_r", "d2")  # (chi_new, t3_l)
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3_l": "c4_u"})

    # ---- T4: sandwiched by P_top_above (top face, t4_d⊕u2) and
    #          P_bot_curr (bottom face, t4_u⊕d2) ----
    step = _apply_proj_unfused(
        P_top_above, T4_with_a, "t4_d", "u2"
    )  # (chi_new, t4_u, d2, r2)
    # ``env_first`` matches the fused path's ``contract(step, P_right)`` order
    # (Koszul-correct for fermions) and yields canonical leg order directly.
    T4_new = _apply_proj_unfused(
        P_bot_curr, step, "t4_u", "d2", chi_new="chi_new_r", env_first=True
    )  # (chi_new, r2, chi_new_r)
    T4_new = T4_new.relabels({"chi_new": "t4_d", "chi_new_r": "t4_u", "r2": "l2"})
    # 'r2'->'l2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.l2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    # ---- phase-fix + normalize (matches variPEPS) ----
    C1_new = _phase_fix_normalize_tensor(C1_new)
    C4_new = _phase_fix_normalize_tensor(C4_new)
    T4_new = _phase_fix_normalize_tensor(T4_new)
    return C1_new, T4_new, C4_new


def _ctm_tensor_absorb_right_2plaq(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_above: Tensor,
    P_bot_above: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style RIGHT absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_right_absorption`.  The
    new (C2, T2, C3) get stored at ``s_dst = neighbors[s_src]["left"]``;
    here ``s_src`` is the cell whose right column is being absorbed and
    ``s_above = neighbors[s_src]["top"]`` is the cell whose plaquette
    provides the top half of T2's projector pair.

    For RIGHT direction, the same plaquette geometry as LEFT is used;
    only the SEAM being cut shifts to the right column of the plaquette.
    Specifically, the projector pair for the "above" plaquette anchored
    at ``s_above`` has direction="right", which compresses the seam
    between TR=``neighbors[s_above]["right"]`` and BR=
    ``neighbors[s_src]["right"]``.  Since BR of above == TR of current
    in our coordinate convention, ``P_*_above`` and ``P_*_curr`` together
    project the chi+D² seam at the boundary between ``s_src`` and
    ``s_dst = neighbors[s_src]["right"]``.

    The Tenax ``_compute_2x2_projector`` for direction='right' has
    ``prime_order=first_second`` (M_prime = M1 @ M2), so the resulting
    ``P_top``/``P_bot`` naming is INVERTED relative to LEFT (where
    direction='left' uses ``prime_order=second_first``).  Specifically:

      * Tenax ``P_top`` for direction='right' acts on BR's TOP face
        (BOTTOM side of the cut seam) ≡ variPEPS ``.bottom``.
      * Tenax ``P_bot`` for direction='right' acts on TR's BOTTOM face
        (TOP side of the cut seam) ≡ variPEPS ``.top``.

    Hence variPEPS RIGHT absorption maps to:
      * ``C2`` uses ``(above).bottom`` ≡ ``P_top_above``.
      * ``T2`` top side (fl = t2_u ⊕ u2) uses ``(above).top`` ≡
        ``P_bot_above``.
      * ``T2`` bottom side (fr = t2_d ⊕ d2) uses ``(curr).bottom`` ≡
        ``P_top_curr``.
      * ``C3`` uses ``(curr).top`` ≡ ``P_bot_curr``.
    """
    if _env_is_fermionic(env_src):
        return _ctm_tensor_absorb_right_2plaq_fused(
            env_src, a_src, P_top_above, P_bot_above, P_top_curr, P_bot_curr
        )
    # Unfused projector application (#605); see ``_apply_proj_unfused``.
    # ---- C2·T1 (both at s_src) ----
    C2_l = env_src.C2.relabel("c2_l", "t1_r")
    C2g = contract(C2_l, env_src.T1)  # (c2_d, t1_l, u2)

    # ---- C3·T3 (both at s_src) ----
    C3_l = env_src.C3.relabel("c3_l", "t3_l")
    C3g = contract(C3_l, env_src.T3)  # (c3_u, t3_r, d2)

    # ---- T2·ket (both at s_src) ----
    T2_with_a = contract(env_src.T2, a_src)  # (t2_u, t2_d, u2, d2, l2)

    # ---- C2: project with P_top_above only (≡ variPEPS .bottom of above) ----
    C2_new = _apply_proj_unfused(P_top_above, C2g, "c2_d", "u2")
    C2_new = C2_new.relabels({"chi_new": "c2_d", "t1_l": "c2_l"})

    # ---- C3: project with P_bot_curr only (≡ variPEPS .top of curr) ----
    C3_new = _apply_proj_unfused(P_bot_curr, C3g, "c3_u", "d2")
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3_r": "c3_l"})

    # ---- T2: sandwiched by P_bot_above (fl, t2_u⊕u2) and P_top_curr (fr, t2_d⊕d2) ----
    step = _apply_proj_unfused(
        P_bot_above, T2_with_a, "t2_u", "u2"
    )  # (chi_new, t2_d, d2, l2)
    T2_new = _apply_proj_unfused(
        P_top_curr, step, "t2_d", "d2", chi_new="chi_new_r", env_first=True
    )
    T2_new = T2_new.relabels({"chi_new": "t2_u", "chi_new_r": "t2_d", "l2": "r2"})
    # 'l2'->'r2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.r2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    C2_new = _phase_fix_normalize_tensor(C2_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T2_new = _phase_fix_normalize_tensor(T2_new)
    return C2_new, T2_new, C3_new


def _ctm_tensor_absorb_top_2plaq(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_left: Tensor,
    P_bot_left: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style TOP absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_top_absorption`.  The
    "left plaquette" is anchored at ``neighbors[s_src]["left"]`` and the
    "current plaquette" at ``s_src``.  The new (C1, T1, C2) tensors get
    stored at ``s_dst = neighbors[s_src]["bottom"]``.

    For TOP direction, the Tenax ``_compute_2x2_projector`` uses
    ``prime_order=first_second``, so the ``P_top`` / ``P_bot`` naming
    is INVERTED relative to the variPEPS ``Top_Projectors(left, right)``
    pair.  Specifically:

      * Tenax ``P_top`` for direction='top' acts on TR's LEFT face
        (RIGHT side of cut seam) ≡ variPEPS ``.right``.
      * Tenax ``P_bot`` for direction='top' acts on TL's RIGHT face
        (LEFT side of cut seam) ≡ variPEPS ``.left``.

    Hence variPEPS TOP absorption maps to:
      * ``C1`` uses ``(left-plaq).right`` ≡ ``P_top_left``.
      * ``T1`` left side (fl = t1_l ⊕ l2) uses ``(left-plaq).left`` ≡
        ``P_bot_left``.
      * ``T1`` right side (fr = t1_r ⊕ r2) uses ``(curr).right`` ≡
        ``P_top_curr``.
      * ``C2`` uses ``(curr).left`` ≡ ``P_bot_curr``.
    """
    if _env_is_fermionic(env_src):
        return _ctm_tensor_absorb_top_2plaq_fused(
            env_src, a_src, P_top_left, P_bot_left, P_top_curr, P_bot_curr
        )
    # Unfused projector application (#605); see ``_apply_proj_unfused``.
    # ---- C1·T4 (both at s_src) ----
    C1_d = env_src.C1.relabel("c1_d", "t4_d")
    C1g = contract(C1_d, env_src.T4)  # (c1_r, l2, t4_u)

    # ---- C2·T2 (both at s_src) ----
    C2_d = env_src.C2.relabel("c2_d", "t2_u")
    C2g = contract(C2_d, env_src.T2)  # (c2_l, r2, t2_d)

    # ---- T1·ket (both at s_src) ----
    T1_with_a = contract(env_src.T1, a_src)  # (t1_l, t1_r, d2, l2, r2)

    # ---- C1: project with P_top_left (≡ variPEPS .right of left-plaq) ----
    C1_new = _apply_proj_unfused(P_top_left, C1g, "c1_r", "l2")
    C1_new = C1_new.relabels({"chi_new": "c1_r", "t4_u": "c1_d"})

    # ---- C2: project with P_bot_curr (≡ variPEPS .left of curr) ----
    C2_new = _apply_proj_unfused(P_bot_curr, C2g, "c2_l", "r2")
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2_d": "c2_d"})

    # ---- T1: sandwiched by P_bot_left (fl, t1_l⊕l2) and P_top_curr (fr, t1_r⊕r2) ----
    step = _apply_proj_unfused(
        P_bot_left, T1_with_a, "t1_l", "l2"
    )  # (chi_new, t1_r, d2, r2)
    T1_new = _apply_proj_unfused(
        P_top_curr, step, "t1_r", "r2", chi_new="chi_new_r", env_first=True
    )
    T1_new = T1_new.relabels({"chi_new": "t1_l", "chi_new_r": "t1_r", "d2": "u2"})
    # 'd2'->'u2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.u2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    C1_new = _phase_fix_normalize_tensor(C1_new)
    C2_new = _phase_fix_normalize_tensor(C2_new)
    T1_new = _phase_fix_normalize_tensor(T1_new)
    return C1_new, T1_new, C2_new


def _ctm_tensor_absorb_bottom_2plaq(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_left: Tensor,
    P_bot_left: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style BOTTOM absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_bottom_absorption`.  The
    new (C4, T3, C3) get stored at ``s_dst = neighbors[s_src]["top"]``.
    """
    if _env_is_fermionic(env_src):
        return _ctm_tensor_absorb_bottom_2plaq_fused(
            env_src, a_src, P_top_left, P_bot_left, P_top_curr, P_bot_curr
        )
    # Unfused projector application (#605); see ``_apply_proj_unfused``.
    # ---- C4·T4 (both at s_src) ----
    C4_r = env_src.C4.relabel("c4_r", "t4_u")
    C4g = contract(C4_r, env_src.T4)  # (c4_u, t4_d, l2)

    # ---- C3·T2 (both at s_src) ----
    # C3.c3_u <-> T2.t2_d  (energy/RDM convention; #670)
    C3_u = env_src.C3.relabel("c3_u", "t2_d")
    C3g = contract(C3_u, env_src.T2)  # (c3_l, t2_u, r2)  # #670

    # ---- T3·ket (both at s_src) ----
    T3_with_a = contract(env_src.T3, a_src)  # (t3_r, t3_l, u2, l2, r2)

    # ---- C4: project with P_bot_left ----
    C4_new = _apply_proj_unfused(P_bot_left, C4g, "c4_u", "l2")
    C4_new = C4_new.relabels({"chi_new": "c4_u", "t4_d": "c4_r"})

    # ---- C3: project with P_top_curr ----
    C3_new = _apply_proj_unfused(P_top_curr, C3g, "c3_l", "r2")  # #670
    C3_new = C3_new.relabels({"chi_new": "c3_l", "t2_u": "c3_u"})

    # ---- T3: sandwiched by P_top_left (fl, t3_r⊕l2) + P_bot_curr (fr, t3_l⊕r2) ----
    step = _apply_proj_unfused(
        P_top_left, T3_with_a, "t3_r", "l2"
    )  # (chi_new, t3_l, u2, r2)
    T3_new = _apply_proj_unfused(
        P_bot_curr, step, "t3_l", "r2", chi_new="chi_new_r", env_first=True
    )
    T3_new = T3_new.relabels({"chi_new": "t3_r", "chi_new_r": "t3_l", "u2": "d2"})
    # 'u2'->'d2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.d2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    C4_new = _phase_fix_normalize_tensor(C4_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T3_new = _phase_fix_normalize_tensor(T3_new)
    return C4_new, T3_new, C3_new


def _ctm_tensor_absorb_left_2plaq_fused(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_above: Tensor,
    P_bot_above: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style LEFT absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_left_absorption`:

      * ``new_C1 = (C1[s_src] · T1[s_src]) projected by P_bot_above``
      * ``new_T4 = (T4[s_src] · ket[s_src]) sandwiched by P_top_above
        (top face) and P_bot_curr (bottom face)``
      * ``new_C4 = (C4[s_src] · T3[s_src]) projected by P_top_curr``

    where ``P_*_above`` are halves of the projector pair for the plaquette
    anchored at ``neighbors[s_src]["top"]`` (compressing the seam between
    that cell and ``s_src``), and ``P_*_curr`` are halves for the
    plaquette anchored at ``s_src`` (compressing the seam between
    ``s_src`` and ``neighbors[s_src]["bottom"]``).

    The new env tensors should be stored at ``s_dst = neighbors[s_src]
    ["right"]``: this is the cell whose left edge boundary is the column
    just absorbed.
    """
    # Grown corners / edge are kept UNFUSED; projectors are applied via
    # ``_apply_proj_unfused`` (split the projector's fused leg and contract the
    # bare chi / D² constituents) to avoid the hard-fusion charge-conjugation
    # that breaks the raw-charge block pairing at D≥3 (#605).
    # ---- C1·T1 (both at s_src) ----
    C1_r = env_src.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env_src.T1)  # (c1_d, u2, t1_r)

    # ---- C4·T3 (both at s_src) ----
    C4_u = env_src.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env_src.T3)  # (c4_r, d2, t3_l)

    # ---- T4·ket (T4 and ket both at s_src) ----
    T4_with_a = contract(env_src.T4, a_src)  # (t4_d, t4_u, u2, d2, r2)

    # ---- C1: project with P_bot_above only ----
    C1_new = _apply_proj_unfused(P_bot_above, C1g, "c1_d", "u2")  # (chi_new, t1_r)
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1_r": "c1_r"})

    # ---- C4: project with P_top_curr only ----
    C4_new = _apply_proj_unfused(P_top_curr, C4g, "c4_r", "d2")  # (chi_new, t3_l)
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3_l": "c4_u"})

    # ---- T4: sandwiched by P_top_above (top face, t4_d⊕u2) and
    #          P_bot_curr (bottom face, t4_u⊕d2) ----
    step = _apply_proj_unfused(
        P_top_above, T4_with_a, "t4_d", "u2"
    )  # (chi_new, t4_u, d2, r2)
    T4_new = _apply_proj_unfused(
        P_bot_curr, step, "t4_u", "d2", chi_new="chi_new_r"
    )  # (chi_new, r2, chi_new_r)
    T4_new = T4_new.relabels({"chi_new": "t4_d", "chi_new_r": "t4_u", "r2": "l2"})
    # Unfused contraction can emit the free legs in a different order than the
    # fused path; restore the canonical edge layout so downstream positional
    # uses (e.g. phase-fix ravel order) match the dense/fused result (#605).
    T4_new = T4_new.transpose(
        tuple(T4_new.labels().index(lbl) for lbl in ("t4_d", "l2", "t4_u"))
    )
    T4_new = _flip_leg_flow(T4_new, "l2")  # r2(OUT) -> l2 needs IN

    # ---- phase-fix + normalize (matches variPEPS) ----
    C1_new = _phase_fix_normalize_tensor(C1_new)
    C4_new = _phase_fix_normalize_tensor(C4_new)
    T4_new = _phase_fix_normalize_tensor(T4_new)
    return C1_new, T4_new, C4_new


def _ctm_tensor_absorb_right_2plaq_fused(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_above: Tensor,
    P_bot_above: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style RIGHT absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_right_absorption`.  The
    new (C2, T2, C3) get stored at ``s_dst = neighbors[s_src]["left"]``;
    here ``s_src`` is the cell whose right column is being absorbed and
    ``s_above = neighbors[s_src]["top"]`` is the cell whose plaquette
    provides the top half of T2's projector pair.

    For RIGHT direction, the same plaquette geometry as LEFT is used;
    only the SEAM being cut shifts to the right column of the plaquette.
    Specifically, the projector pair for the "above" plaquette anchored
    at ``s_above`` has direction="right", which compresses the seam
    between TR=``neighbors[s_above]["right"]`` and BR=
    ``neighbors[s_src]["right"]``.  Since BR of above == TR of current
    in our coordinate convention, ``P_*_above`` and ``P_*_curr`` together
    project the chi+D² seam at the boundary between ``s_src`` and
    ``s_dst = neighbors[s_src]["right"]``.

    The Tenax ``_compute_2x2_projector`` for direction='right' has
    ``prime_order=first_second`` (M_prime = M1 @ M2), so the resulting
    ``P_top``/``P_bot`` naming is INVERTED relative to LEFT (where
    direction='left' uses ``prime_order=second_first``).  Specifically:

      * Tenax ``P_top`` for direction='right' acts on BR's TOP face
        (BOTTOM side of the cut seam) ≡ variPEPS ``.bottom``.
      * Tenax ``P_bot`` for direction='right' acts on TR's BOTTOM face
        (TOP side of the cut seam) ≡ variPEPS ``.top``.

    Hence variPEPS RIGHT absorption maps to:
      * ``C2`` uses ``(above).bottom`` ≡ ``P_top_above``.
      * ``T2`` top side (fl = t2_u ⊕ u2) uses ``(above).top`` ≡
        ``P_bot_above``.
      * ``T2`` bottom side (fr = t2_d ⊕ d2) uses ``(curr).bottom`` ≡
        ``P_top_curr``.
      * ``C3`` uses ``(curr).top`` ≡ ``P_bot_curr``.
    """
    # Unfused projector application (#605); see ``_apply_proj_unfused``.
    # ---- C2·T1 (both at s_src) ----
    C2_l = env_src.C2.relabel("c2_l", "t1_r")
    C2g = contract(C2_l, env_src.T1)  # (c2_d, t1_l, u2)

    # ---- C3·T3 (both at s_src) ----
    C3_l = env_src.C3.relabel("c3_l", "t3_l")
    C3g = contract(C3_l, env_src.T3)  # (c3_u, t3_r, d2)

    # ---- T2·ket (both at s_src) ----
    T2_with_a = contract(env_src.T2, a_src)  # (t2_u, t2_d, u2, d2, l2)

    # ---- C2: project with P_top_above only (≡ variPEPS .bottom of above) ----
    C2_new = _apply_proj_unfused(P_top_above, C2g, "c2_d", "u2")
    C2_new = C2_new.relabels({"chi_new": "c2_d", "t1_l": "c2_l"})

    # ---- C3: project with P_bot_curr only (≡ variPEPS .top of curr) ----
    C3_new = _apply_proj_unfused(P_bot_curr, C3g, "c3_u", "d2")
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3_r": "c3_l"})

    # ---- T2: sandwiched by P_bot_above (fl, t2_u⊕u2) and P_top_curr (fr, t2_d⊕d2) ----
    step = _apply_proj_unfused(
        P_bot_above, T2_with_a, "t2_u", "u2"
    )  # (chi_new, t2_d, d2, l2)
    T2_new = _apply_proj_unfused(P_top_curr, step, "t2_d", "d2", chi_new="chi_new_r")
    T2_new = T2_new.relabels({"chi_new": "t2_u", "chi_new_r": "t2_d", "l2": "r2"})
    T2_new = T2_new.transpose(
        tuple(T2_new.labels().index(lbl) for lbl in ("t2_u", "r2", "t2_d"))
    )
    T2_new = _flip_leg_flow(T2_new, "r2")  # l2(IN) -> r2 needs OUT

    C2_new = _phase_fix_normalize_tensor(C2_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T2_new = _phase_fix_normalize_tensor(T2_new)
    return C2_new, T2_new, C3_new


def _ctm_tensor_absorb_top_2plaq_fused(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_left: Tensor,
    P_bot_left: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style TOP absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_top_absorption`.  The
    "left plaquette" is anchored at ``neighbors[s_src]["left"]`` and the
    "current plaquette" at ``s_src``.  The new (C1, T1, C2) tensors get
    stored at ``s_dst = neighbors[s_src]["bottom"]``.

    For TOP direction, the Tenax ``_compute_2x2_projector`` uses
    ``prime_order=first_second``, so the ``P_top`` / ``P_bot`` naming
    is INVERTED relative to the variPEPS ``Top_Projectors(left, right)``
    pair.  Specifically:

      * Tenax ``P_top`` for direction='top' acts on TR's LEFT face
        (RIGHT side of cut seam) ≡ variPEPS ``.right``.
      * Tenax ``P_bot`` for direction='top' acts on TL's RIGHT face
        (LEFT side of cut seam) ≡ variPEPS ``.left``.

    Hence variPEPS TOP absorption maps to:
      * ``C1`` uses ``(left-plaq).right`` ≡ ``P_top_left``.
      * ``T1`` left side (fl = t1_l ⊕ l2) uses ``(left-plaq).left`` ≡
        ``P_bot_left``.
      * ``T1`` right side (fr = t1_r ⊕ r2) uses ``(curr).right`` ≡
        ``P_top_curr``.
      * ``C2`` uses ``(curr).left`` ≡ ``P_bot_curr``.
    """
    # Unfused projector application (#605); see ``_apply_proj_unfused``.
    # ---- C1·T4 (both at s_src) ----
    C1_d = env_src.C1.relabel("c1_d", "t4_d")
    C1g = contract(C1_d, env_src.T4)  # (c1_r, l2, t4_u)

    # ---- C2·T2 (both at s_src) ----
    C2_d = env_src.C2.relabel("c2_d", "t2_u")
    C2g = contract(C2_d, env_src.T2)  # (c2_l, r2, t2_d)

    # ---- T1·ket (both at s_src) ----
    T1_with_a = contract(env_src.T1, a_src)  # (t1_l, t1_r, d2, l2, r2)

    # ---- C1: project with P_top_left (≡ variPEPS .right of left-plaq) ----
    C1_new = _apply_proj_unfused(P_top_left, C1g, "c1_r", "l2")
    C1_new = C1_new.relabels({"chi_new": "c1_r", "t4_u": "c1_d"})

    # ---- C2: project with P_bot_curr (≡ variPEPS .left of curr) ----
    C2_new = _apply_proj_unfused(P_bot_curr, C2g, "c2_l", "r2")
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2_d": "c2_d"})

    # ---- T1: sandwiched by P_bot_left (fl, t1_l⊕l2) and P_top_curr (fr, t1_r⊕r2) ----
    step = _apply_proj_unfused(
        P_bot_left, T1_with_a, "t1_l", "l2"
    )  # (chi_new, t1_r, d2, r2)
    T1_new = _apply_proj_unfused(P_top_curr, step, "t1_r", "r2", chi_new="chi_new_r")
    T1_new = T1_new.relabels({"chi_new": "t1_l", "chi_new_r": "t1_r", "d2": "u2"})
    T1_new = T1_new.transpose(
        tuple(T1_new.labels().index(lbl) for lbl in ("t1_l", "u2", "t1_r"))
    )
    T1_new = _flip_leg_flow(T1_new, "u2")  # d2(OUT) -> u2 needs IN

    C1_new = _phase_fix_normalize_tensor(C1_new)
    C2_new = _phase_fix_normalize_tensor(C2_new)
    T1_new = _phase_fix_normalize_tensor(T1_new)
    return C1_new, T1_new, C2_new


def _ctm_tensor_absorb_bottom_2plaq_fused(
    env_src: CTMTensorEnv,
    a_src: Tensor,
    P_top_left: Tensor,
    P_bot_left: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """variPEPS-style BOTTOM absorption using two plaquettes' projectors.

    Mirrors :func:`varipeps.ctmrg.absorption.do_bottom_absorption`.  The
    new (C4, T3, C3) get stored at ``s_dst = neighbors[s_src]["top"]``.
    """
    # Unfused projector application (#605); see ``_apply_proj_unfused``.
    # ---- C4·T4 (both at s_src) ----
    C4_r = env_src.C4.relabel("c4_r", "t4_u")
    C4g = contract(C4_r, env_src.T4)  # (c4_u, t4_d, l2)

    # ---- C3·T2 (both at s_src) ----
    # C3.c3_u <-> T2.t2_d  (energy/RDM convention; #674, mirrors #670)
    C3_u = env_src.C3.relabel("c3_u", "t2_d")
    C3g = contract(C3_u, env_src.T2)  # (c3_l, t2_u, r2)  # #674

    # ---- T3·ket (both at s_src) ----
    T3_with_a = contract(env_src.T3, a_src)  # (t3_r, t3_l, u2, l2, r2)

    # ---- C4: project with P_bot_left ----
    C4_new = _apply_proj_unfused(P_bot_left, C4g, "c4_u", "l2")
    C4_new = C4_new.relabels({"chi_new": "c4_u", "t4_d": "c4_r"})

    # ---- C3: project with P_top_curr ----
    C3_new = _apply_proj_unfused(P_top_curr, C3g, "c3_l", "r2")  # #674
    C3_new = C3_new.relabels({"chi_new": "c3_l", "t2_u": "c3_u"})

    # ---- T3: sandwiched by P_top_left (fl, t3_r⊕l2) + P_bot_curr (fr, t3_l⊕r2) ----
    step = _apply_proj_unfused(
        P_top_left, T3_with_a, "t3_r", "l2"
    )  # (chi_new, t3_l, u2, r2)
    T3_new = _apply_proj_unfused(P_bot_curr, step, "t3_l", "r2", chi_new="chi_new_r")
    T3_new = T3_new.relabels({"chi_new": "t3_r", "chi_new_r": "t3_l", "u2": "d2"})
    T3_new = T3_new.transpose(
        tuple(T3_new.labels().index(lbl) for lbl in ("t3_r", "d2", "t3_l"))
    )
    T3_new = _flip_leg_flow(T3_new, "d2")  # u2(IN) -> d2 needs OUT

    C4_new = _phase_fix_normalize_tensor(C4_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T3_new = _phase_fix_normalize_tensor(T3_new)
    return C4_new, T3_new, C3_new


def _chunked_T_new_apply(
    edge,
    a,
    P_1,
    P_2,
    C1g,
    C4g,
    edge_label_order,
    chi_leg,
    surv_leg,
    chunked_fn,
    chunk_size,
):
    """Chunked edge absorption for a 1x1 dense move.

    Computes (C1_new, C4_new, T_new) with the expensive chi^2 * D^6 edge
    contraction chunked over the boundary-chi axis via ``chunked_fn`` (from
    _ctm_chunked_absorb), avoiding materializing the full intermediate.
    Bit-identical to the fused-index path (_apply_projector_tensor). Dense only.

    Args:
        edge:  the parallel edge DenseTensor being grown (T4/T2/T1/T3).
        a:     double-layer DenseTensor (labels u2,d2,l2,r2).
        P_1, P_2: projector pair (labels fused, chi_new).
        C1g, C4g: grown corners (P_1 side, P_2 side).
        edge_label_order: canonical edge leg order, e.g. ["t4_d","l2","t4_u"].
        chi_leg: edge's boundary-chi (chunk) leg label, e.g. "t4_d".
        surv_leg: surviving a-leg label (the new edge's D2 leg), e.g. "r2".
        chunked_fn: the matching _chunked_T_new_* raw function.
        chunk_size: lax.map batch size over the boundary-chi axis.

    Returns:
        (C1_new, C4_new, T_new) Tensors. T_new labels (chi_new, <surv_leg>, chi_new_r).
    """
    chi_dim = edge.indices[list(edge.labels()).index(chi_leg)].dim
    D2 = a.indices[list(a.labels()).index(surv_leg)].dim
    edge_raw = _raw_in_label_order(edge, edge_label_order)
    a_raw = _raw_in_label_order(a, ["u2", "d2", "l2", "r2"])
    # The raw core conjugates P1 internally (matches P_1.bar()); pass P_1 as-is.
    P1_raw = _raw_in_label_order(P_1, ["fused", "chi_new"])
    P2_raw = _raw_in_label_order(P_2, ["fused", "chi_new"])
    T_arr = chunked_fn(edge_raw, a_raw, P1_raw, P2_raw, chi_dim, D2, chunk_size)

    # Cheap corners (identical to _apply_projector_tensor)
    P1_bar = P_1.bar()
    C1_new = contract(P1_bar, C1g)
    C4_new = contract(P_2.bar(), C4g)
    chi_new_idx = P1_bar.indices[list(P1_bar.labels()).index("chi_new")]
    surv_idx = a.indices[list(a.labels()).index(surv_leg)]
    chi_new_r_idx = P_2.indices[list(P_2.labels()).index("chi_new")].relabel(
        "chi_new_r"
    )
    T_new = DenseTensor(T_arr, (chi_new_idx, surv_idx, chi_new_r_idx))
    return C1_new, C4_new, T_new


def _ctm_tensor_move_left(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
    projector_backward: str = "auto",
    chunk_size: int | None = None,
) -> tuple[CTMTensorEnv, jax.Array]:
    """Left move: updates C1, T4, C4.

    Corners (C1, C4) from env_self, perpendicular edges (T1, T3) from
    env_neighbor, parallel edge (T4) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C1g = einsum('ab,buc->auc', C1, T1)
                     C4g = einsum('gh,hdi->gdi', C4, T3)
                     T4g = einsum('alg,udlr->augdr', T4, a)

    Args:
        chunk_size: When not None and env_self.T4 is a DenseTensor, use the
            chunked raw core (_chunked_T_new_left) to compute T4_new without
            materializing the full chi^2*D^6 intermediate.  Default (None)
            uses the standard fused-index path unchanged.
    """
    # C1(self) · T1(neighbor)
    C1_r = env_self.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env_neighbor.T1)  # (c1_d, u2, t1_r)
    C1g = _fuse_pair_by_label(C1g, "c1_d", "u2", "fused", IN)  # (fused, t1_r)

    # C4(self) · T3(neighbor)
    C4_u = env_self.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env_neighbor.T3)  # (c4_r, d2, t3_l)
    C4g = _fuse_pair_by_label(C4g, "c4_r", "d2", "fused", IN)  # (fused, t3_l)

    # Native projector (needs only the grown corners, not the grown edge)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C1g, C4g, chi, projector_method, base_charges, projector_backward
    )

    if chunk_size is not None and isinstance(env_self.T4, DenseTensor):
        C1_new, C4_new, T4_new = _chunked_T_new_apply(
            env_self.T4,
            a,
            P_1,
            P_2,
            C1g,
            C4g,
            ["t4_d", "l2", "t4_u"],
            "t4_d",
            "r2",
            _chunked_T_new_left,
            chunk_size,
        )
    else:
        if chunk_size is not None:
            import warnings

            warnings.warn(
                "chunk_size is set but env is not dense (SymmetricTensor); "
                "falling back to the standard fused-index CTM path.",
                stacklevel=2,
            )
        # T4(self) · a(neighbor)
        T4_with_a = contract(env_self.T4, a)
        T4g = _fuse_pair_by_label(T4_with_a, "t4_d", "u2", "fl", IN)
        T4g = _fuse_pair_by_label(T4g, "t4_u", "d2", "fr", OUT)

        C1_new, C4_new, T4_new = _apply_projector_with_reembed(
            P_1, P_2, C1g, C4g, T4g, "fl", "fr"
        )

    # Relabel to expected output labels
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1_r": "c1_r"})
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3_l": "c4_u"})
    T4_new = T4_new.relabels({"chi_new": "t4_d", "chi_new_r": "t4_u", "r2": "l2"})
    # 'r2'->'l2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.l2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    # Per-absorption normalization + phase fix (matches variPEPS, stabilizes J^T)
    C1_new = _phase_fix_normalize_tensor(C1_new)
    C4_new = _phase_fix_normalize_tensor(C4_new)
    T4_new = _phase_fix_normalize_tensor(T4_new)
    return env_self._replace(C1=C1_new, C4=C4_new, T4=T4_new), _eps_t


def _ctm_tensor_move_right(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
    projector_backward: str = "auto",
    chunk_size: int | None = None,
) -> tuple[CTMTensorEnv, jax.Array]:
    """Right move: updates C2, T2, C3.

    Corners (C2, C3) from env_self, perpendicular edges (T1, T3) from
    env_neighbor, parallel edge (T2) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C2g = einsum('ce,buc->eub', C2, T1)
                     C3g = einsum('im,hdi->mdh', C3, T3)
                     T2g = einsum('erm,udlr->eumdl', T2, a)

    Args:
        chunk_size: When not None and env_self.T2 is a DenseTensor, use the
            chunked raw core (_chunked_T_new_right) to compute T2_new without
            materializing the full chi^2*D^6 intermediate.  Default (None)
            uses the standard fused-index path unchanged.
    """
    # C2(self) · T1(neighbor)
    C2_l = env_self.C2.relabel("c2_l", "t1_r")
    C2g = contract(C2_l, env_neighbor.T1)  # (c2_d, t1_l, u2)
    C2g = _fuse_pair_by_label(C2g, "c2_d", "u2", "fused", IN)  # (fused, t1_l)

    # C3(self) · T3(neighbor)
    C3_u = env_self.C3.relabel("c3_u", "t3_l")
    C3g = contract(C3_u, env_neighbor.T3)  # (c3_l, t3_r, d2)
    C3g = _fuse_pair_by_label(C3g, "c3_l", "d2", "fused", IN)  # (fused, t3_r)

    # Native projector (needs only the grown corners, not the grown edge)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C2g, C3g, chi, projector_method, base_charges, projector_backward
    )

    if chunk_size is not None and isinstance(env_self.T2, DenseTensor):
        C2_new, C3_new, T2_new = _chunked_T_new_apply(
            env_self.T2,
            a,
            P_1,
            P_2,
            C2g,
            C3g,
            ["t2_u", "r2", "t2_d"],
            "t2_u",
            "l2",
            _chunked_T_new_right,
            chunk_size,
        )
    else:
        if chunk_size is not None:
            import warnings

            warnings.warn(
                "chunk_size is set but env is not dense (SymmetricTensor); "
                "falling back to the standard fused-index CTM path.",
                stacklevel=2,
            )
        # T2(self) · a(neighbor)
        T2_with_a = contract(env_self.T2, a)
        T2g = _fuse_pair_by_label(T2_with_a, "t2_u", "u2", "fl", IN)
        T2g = _fuse_pair_by_label(T2g, "t2_d", "d2", "fr", OUT)

        C2_new, C3_new, T2_new = _apply_projector_with_reembed(
            P_1, P_2, C2g, C3g, T2g, "fl", "fr"
        )

    # Relabel to expected output labels
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t1_l": "c2_d"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3_r": "c3_l"})
    T2_new = T2_new.relabels({"chi_new": "t2_u", "chi_new_r": "t2_d", "l2": "r2"})
    # 'l2'->'r2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.r2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    C2_new = _phase_fix_normalize_tensor(C2_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T2_new = _phase_fix_normalize_tensor(T2_new)
    return env_self._replace(C2=C2_new, C3=C3_new, T2=T2_new), _eps_t


def _ctm_tensor_move_top(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
    projector_backward: str = "auto",
    chunk_size: int | None = None,
) -> tuple[CTMTensorEnv, jax.Array]:
    """Top move: updates C1, T1, C2.

    Corners (C1, C2) from env_self, perpendicular edges (T4, T2) from
    env_neighbor, parallel edge (T1) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C1g = einsum('ab,alg->blg', C1, T4)
                     C2g = einsum('ce,erm->crm', C2, T2)
                     T1g = einsum('buc,udlr->bcdlr', T1, a)

    Args:
        chunk_size: When not None and env_self.T1 is a DenseTensor, use the
            chunked raw core (_chunked_T_new_top) to compute T1_new without
            materializing the full chi^2*D^6 intermediate.  Default (None)
            uses the standard fused-index path unchanged.
    """
    # C1(self) · T4(neighbor)
    C1_d = env_self.C1.relabel("c1_d", "t4_d")
    C1g = contract(C1_d, env_neighbor.T4)  # (c1_r, l2, t4_u)
    C1g = _fuse_pair_by_label(C1g, "c1_r", "l2", "fused", IN)  # (fused, t4_u)

    # C2(self) · T2(neighbor)
    C2_d = env_self.C2.relabel("c2_d", "t2_u")
    C2g = contract(C2_d, env_neighbor.T2)  # (c2_l, r2, t2_d)
    C2g = _fuse_pair_by_label(C2g, "c2_l", "r2", "fused", IN)  # (fused, t2_d)

    # Native projector (needs only the grown corners, not the grown edge)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C1g, C2g, chi, projector_method, base_charges, projector_backward
    )

    if chunk_size is not None and isinstance(env_self.T1, DenseTensor):
        C1_new, C2_new, T1_new = _chunked_T_new_apply(
            env_self.T1,
            a,
            P_1,
            P_2,
            C1g,
            C2g,
            ["t1_l", "u2", "t1_r"],
            "t1_l",
            "d2",
            _chunked_T_new_top,
            chunk_size,
        )
    else:
        if chunk_size is not None:
            import warnings

            warnings.warn(
                "chunk_size is set but env is not dense (SymmetricTensor); "
                "falling back to the standard fused-index CTM path.",
                stacklevel=2,
            )
        # T1(self) · a(neighbor)
        T1_with_a = contract(env_self.T1, a)
        T1g = _fuse_pair_by_label(T1_with_a, "t1_l", "l2", "fl", IN)
        T1g = _fuse_pair_by_label(T1g, "t1_r", "r2", "fr", OUT)

        C1_new, C2_new, T1_new = _apply_projector_with_reembed(
            P_1, P_2, C1g, C2g, T1g, "fl", "fr"
        )

    # Relabel to expected output labels
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t4_u": "c1_r"})
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2_d": "c2_d"})
    T1_new = T1_new.relabels({"chi_new": "t1_l", "chi_new_r": "t1_r", "d2": "u2"})
    # 'd2'->'u2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.u2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    C1_new = _phase_fix_normalize_tensor(C1_new)
    C2_new = _phase_fix_normalize_tensor(C2_new)
    T1_new = _phase_fix_normalize_tensor(T1_new)
    return env_self._replace(C1=C1_new, C2=C2_new, T1=T1_new), _eps_t


def _ctm_tensor_move_bottom(
    env_self: CTMTensorEnv,
    env_neighbor: CTMTensorEnv,
    a: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
    projector_backward: str = "auto",
    chunk_size: int | None = None,
) -> tuple[CTMTensorEnv, jax.Array]:
    """Bottom move: updates C4, T3, C3.

    Corners (C4, C3) from env_self, perpendicular edges (T4, T2) from
    env_neighbor, parallel edge (T3) from env_self, double-layer ``a``
    from neighbor site.

    Dense reference: C4g = einsum('gh,alg->hal', C4, T4).transpose(0,2,1)
                     C3g = einsum('im,erm->ire', C3, T2)
                     T3g = einsum('hdi,udlr->hiulr', T3, a)

    Args:
        chunk_size: When not None and env_self.T3 is a DenseTensor, use the
            chunked raw core (_chunked_T_new_bottom) to compute T3_new without
            materializing the full chi^2*D^6 intermediate.  Default (None)
            uses the standard fused-index path unchanged.
    """
    # C4(self) · T4(neighbor)
    C4_r = env_self.C4.relabel("c4_r", "t4_u")
    C4g = contract(C4_r, env_neighbor.T4)  # (c4_u, t4_d, l2)
    C4g = _fuse_pair_by_label(C4g, "c4_u", "l2", "fused", IN)  # (fused, t4_d)

    # C3(self) · T2(neighbor)
    C3_l = env_self.C3.relabel("c3_l", "t2_d")
    C3g = contract(C3_l, env_neighbor.T2)  # (c3_u, t2_u, r2)
    C3g = _fuse_pair_by_label(C3g, "c3_u", "r2", "fused", IN)  # (fused, t2_u)

    # Native projector (needs only the grown corners, not the grown edge)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C4g, C3g, chi, projector_method, base_charges, projector_backward
    )

    if chunk_size is not None and isinstance(env_self.T3, DenseTensor):
        C4_new, C3_new, T3_new = _chunked_T_new_apply(
            env_self.T3,
            a,
            P_1,
            P_2,
            C4g,
            C3g,
            ["t3_r", "d2", "t3_l"],
            "t3_r",
            "u2",
            _chunked_T_new_bottom,
            chunk_size,
        )
    else:
        if chunk_size is not None:
            import warnings

            warnings.warn(
                "chunk_size is set but env is not dense (SymmetricTensor); "
                "falling back to the standard fused-index CTM path.",
                stacklevel=2,
            )
        # T3(self) · a(neighbor)
        T3_with_a = contract(env_self.T3, a)
        T3g = _fuse_pair_by_label(T3_with_a, "t3_r", "l2", "fl", IN)
        T3g = _fuse_pair_by_label(T3g, "t3_l", "r2", "fr", OUT)

        C4_new, C3_new, T3_new = _apply_projector_with_reembed(
            P_1, P_2, C4g, C3g, T3g, "fl", "fr"
        )

    # Relabel to expected output labels
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t4_d": "c4_u"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t2_u": "c3_l"})
    T3_new = T3_new.relabels({"chi_new": "t3_r", "chi_new_r": "t3_l", "u2": "d2"})
    # 'u2'->'d2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.d2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    C4_new = _phase_fix_normalize_tensor(C4_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T3_new = _phase_fix_normalize_tensor(T3_new)
    return env_self._replace(C4=C4_new, C3=C3_new, T3=T3_new), _eps_t


def _ctm_tensor_move_left_2x2(
    env_TL: CTMTensorEnv,
    env_TR: CTMTensorEnv,
    env_BL: CTMTensorEnv,
    env_BR: CTMTensorEnv,
    a_TL: Tensor,
    a_TR: Tensor,
    a_BL: Tensor,
    a_BR: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
) -> CTMTensorEnv:
    """Left CTM move using 2x2 plaquette projectors. Updates env_TL.{C1,C4,T4}.

    The plaquette is laid out as ``s_TL ── s_TR / s_BL ── s_BR`` where
    ``s_TL`` is the site whose env is being updated.  The four enlarged
    corners ``Q_TL .. Q_BR`` are built from each site's local env, then
    a Fishman cross-projector pair ``(P_top, P_bot)`` is computed for
    ``direction="left"`` and absorbed into ``env_TL`` to produce updated
    ``C1``, ``C4`` and ``T4`` tensors.

    For a uniform state (all four envs / all four sites identical), this
    reduces to the same truncation that ``_ctm_tensor_move_left`` performs
    via :func:`_compute_projector_tensor`, up to projector gauge.

    Args:
        env_TL:   Env at the top-left site of the 2x2 plaquette (the site
                  whose corners and edges are being updated).
        env_TR:   Env at ``neighbors[s_TL]["right"]``.
        env_BL:   Env at ``neighbors[s_TL]["bottom"]``.
        env_BR:   Env at ``neighbors[s_TR]["bottom"]``.
        a_TL ..a_BR:  Double-layer site tensors at the four plaquette sites.
        chi:      Target bond dimension of the new chi seam.
        projector_method:  Currently ignored — the 2x2 path uses Fishman
                  SVD via :func:`_compute_2x2_projector`.  Kept for API
                  parity with the 1x1 ``_ctm_tensor_move_left``.

    Returns:
        ``env_TL`` with ``C1``, ``C4`` and ``T4`` replaced by their
        projected versions.  Other fields are returned unchanged.
    """
    del projector_method  # 2x2 projector uses Fishman SVD unconditionally

    # ---- Step 1: build the four enlarged corners. ----
    Q_TL = _build_enlarged_corner(
        env_TL.C1, env_TL.T1, env_TL.T4, a_TL, position="top_left"
    )
    Q_TR = _build_enlarged_corner(
        env_TR.C2, env_TR.T1, env_TR.T2, a_TR, position="top_right"
    )
    Q_BL = _build_enlarged_corner(
        env_BL.C4, env_BL.T3, env_BL.T4, a_BL, position="bottom_left"
    )
    Q_BR = _build_enlarged_corner(
        env_BR.C3, env_BR.T3, env_BR.T2, a_BR, position="bottom_right"
    )

    # ---- Step 2: compute the (P_top, P_bot) projector pair. ----
    # P_top:  (chi_outer [IN], fused_D2 [IN], chi_new_top [OUT])
    # P_bot:  (chi_new_bot [IN], chi_outer [OUT], fused_D2 [OUT])
    P_top, P_bot, _, _ = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction="left", base_charges=base_charges
    )

    # ---- Step 3: build C1g, C4g, T4g from env_TL. ----
    # Identical structure to the 1x1 ``_ctm_tensor_move_left`` body, with
    # all env tensors / ``a`` taken from site_TL.
    C1_r = env_TL.C1.relabel("c1_r", "t1_l")
    C1g = contract(C1_r, env_TL.T1)  # (c1_d, u2, t1_r)
    C1g = _fuse_pair_by_label(C1g, "c1_d", "u2", "fused", IN)  # (fused, t1_r)

    C4_u = env_TL.C4.relabel("c4_u", "t3_r")
    C4g = contract(C4_u, env_TL.T3)  # (c4_r, d2, t3_l)
    C4g = _fuse_pair_by_label(C4g, "c4_r", "d2", "fused", IN)  # (fused, t3_l)

    T4_with_a = contract(env_TL.T4, a_TL)
    T4g = _fuse_pair_by_label(T4_with_a, "t4_d", "u2", "fl", IN)
    T4g = _fuse_pair_by_label(T4g, "t4_u", "d2", "fr", OUT)

    # ---- Step 4: convert (P_top, P_bot) to ``(fused, chi_new)`` projectors
    #              compatible with ``_apply_projector_tensor``. ----
    # P_top has axes (chi_outer, fused_D2, chi_new_top) with flows (IN, IN, OUT).
    # Fuse the first two into a single ``fused`` leg with flow IN so that
    # P1.bar() yields fused-flow OUT, matching C1g.fused (IN).
    P_1 = _fuse_pair_by_label(P_top, "chi_outer", "fused_D2", "fused", IN)
    P_1 = P_1.relabel("chi_new_top", "chi_new")  # (fused, chi_new)

    # P_bot has axes (chi_new_bot, chi_outer, fused_D2) with flows
    # (IN, OUT, OUT).  Fuse axes 1+2 into "fused" with flow IN; the
    # resulting tensor is (chi_new_bot, fused).  Reorder to (fused,
    # chi_new) by relabeling chi_new_bot -> chi_new -- contract() is
    # order-insensitive.
    P_2 = _fuse_pair_by_label(P_bot, "chi_outer", "fused_D2", "fused", IN)
    P_2 = P_2.relabel("chi_new_bot", "chi_new")  # (chi_new, fused)

    # ---- Step 5: absorb projectors into the env. ----
    C1_new, C4_new, T4_new = _apply_projector_with_reembed(
        P_1, P_2, C1g, C4g, T4g, "fl", "fr"
    )

    # ---- Step 6: relabel + flow-flip to env-standard labels. ----
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t1_r": "c1_r"})
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t3_l": "c4_u"})
    T4_new = T4_new.relabels({"chi_new": "t4_d", "chi_new_r": "t4_u", "r2": "l2"})
    # 'r2'->'l2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.l2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    # ---- Step 7: phase-fix + normalize (matches variPEPS / 1x1 path). ----
    C1_new = _phase_fix_normalize_tensor(C1_new)
    C4_new = _phase_fix_normalize_tensor(C4_new)
    T4_new = _phase_fix_normalize_tensor(T4_new)
    return env_TL._replace(C1=C1_new, C4=C4_new, T4=T4_new)


def _ctm_tensor_move_right_2x2(
    env_TL: CTMTensorEnv,
    env_TR: CTMTensorEnv,
    env_BL: CTMTensorEnv,
    env_BR: CTMTensorEnv,
    a_TL: Tensor,
    a_TR: Tensor,
    a_BL: Tensor,
    a_BR: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
) -> CTMTensorEnv:
    """Right CTM move using 2x2 plaquette projectors. Updates env_TL.{C2,C3,T2}.

    Mirrors :func:`_ctm_tensor_move_left_2x2` with ``direction="right"``: the
    same four enlarged corners are built from the plaquette ``s_TL ── s_TR /
    s_BL ── s_BR``, then a Fishman cross-projector pair for ``direction=
    "right"`` is computed and absorbed into ``env_TL``'s right edge to
    produce updated ``C2``, ``C3`` and ``T2`` tensors.

    Args:
        env_TL:   Env at the top-left site of the 2x2 plaquette (whose
                  corners and edges are being updated).
        env_TR:   Env at ``neighbors[s_TL]["right"]``.
        env_BL:   Env at ``neighbors[s_TL]["bottom"]``.
        env_BR:   Env at ``neighbors[s_TR]["bottom"]``.
        a_TL ..a_BR:  Double-layer site tensors at the four plaquette sites.
        chi:      Target bond dimension of the new chi seam.
        projector_method:  Currently ignored — the 2x2 path uses Fishman
                  SVD via :func:`_compute_2x2_projector`.  Kept for API
                  parity with the 1x1 ``_ctm_tensor_move_right``.

    Returns:
        ``env_TL`` with ``C2``, ``C3`` and ``T2`` replaced by their
        projected versions.
    """
    del projector_method  # 2x2 projector uses Fishman SVD unconditionally

    # ---- Step 1: build the four enlarged corners. ----
    Q_TL = _build_enlarged_corner(
        env_TL.C1, env_TL.T1, env_TL.T4, a_TL, position="top_left"
    )
    Q_TR = _build_enlarged_corner(
        env_TR.C2, env_TR.T1, env_TR.T2, a_TR, position="top_right"
    )
    Q_BL = _build_enlarged_corner(
        env_BL.C4, env_BL.T3, env_BL.T4, a_BL, position="bottom_left"
    )
    Q_BR = _build_enlarged_corner(
        env_BR.C3, env_BR.T3, env_BR.T2, a_BR, position="bottom_right"
    )

    # ---- Step 2: compute the (P_top, P_bot) projector pair. ----
    P_top, P_bot, _, _ = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction="right", base_charges=base_charges
    )

    # ---- Step 3: build C2g, C3g, T2g from env_TL. ----
    # Same C/T fusion structure as the 1x1 ``_ctm_tensor_move_right`` body.
    C2_l = env_TL.C2.relabel("c2_l", "t1_r")
    C2g = contract(C2_l, env_TL.T1)  # (c2_d, t1_l, u2)
    C2g = _fuse_pair_by_label(C2g, "c2_d", "u2", "fused", IN)  # (fused, t1_l)

    C3_u = env_TL.C3.relabel("c3_u", "t3_l")
    C3g = contract(C3_u, env_TL.T3)  # (c3_l, t3_r, d2)
    C3g = _fuse_pair_by_label(C3g, "c3_l", "d2", "fused", IN)  # (fused, t3_r)

    T2_with_a = contract(env_TL.T2, a_TL)
    T2g = _fuse_pair_by_label(T2_with_a, "t2_u", "u2", "fl", IN)
    T2g = _fuse_pair_by_label(T2g, "t2_d", "d2", "fr", OUT)

    # ---- Step 4: convert (P_top, P_bot) to ``(fused, chi_new)`` projectors. ----
    P_1 = _fuse_pair_by_label(P_top, "chi_outer", "fused_D2", "fused", IN)
    P_1 = P_1.relabel("chi_new_top", "chi_new")
    P_2 = _fuse_pair_by_label(P_bot, "chi_outer", "fused_D2", "fused", IN)
    P_2 = P_2.relabel("chi_new_bot", "chi_new")

    # ---- Step 5: absorb projectors into the env. ----
    C2_new, C3_new, T2_new = _apply_projector_with_reembed(
        P_1, P_2, C2g, C3g, T2g, "fl", "fr"
    )

    # ---- Step 6: relabel + flow-flip to env-standard labels. ----
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t1_l": "c2_d"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t3_r": "c3_l"})
    T2_new = T2_new.relabels({"chi_new": "t2_u", "chi_new_r": "t2_d", "l2": "r2"})
    # 'l2'->'r2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.r2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    # ---- Step 7: phase-fix + normalize. ----
    C2_new = _phase_fix_normalize_tensor(C2_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T2_new = _phase_fix_normalize_tensor(T2_new)
    return env_TL._replace(C2=C2_new, C3=C3_new, T2=T2_new)


def _ctm_tensor_move_top_2x2(
    env_TL: CTMTensorEnv,
    env_TR: CTMTensorEnv,
    env_BL: CTMTensorEnv,
    env_BR: CTMTensorEnv,
    a_TL: Tensor,
    a_TR: Tensor,
    a_BL: Tensor,
    a_BR: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
) -> CTMTensorEnv:
    """Top CTM move using 2x2 plaquette projectors. Updates env_TL.{C1,C2,T1}.

    Mirrors :func:`_ctm_tensor_move_left_2x2` with ``direction="top"``.

    Args:
        env_TL:   Env at the top-left site of the 2x2 plaquette (whose
                  corners and edges are being updated).
        env_TR:   Env at ``neighbors[s_TL]["right"]``.
        env_BL:   Env at ``neighbors[s_TL]["bottom"]``.
        env_BR:   Env at ``neighbors[s_TR]["bottom"]``.
        a_TL ..a_BR:  Double-layer site tensors at the four plaquette sites.
        chi:      Target bond dimension of the new chi seam.
        projector_method:  Ignored; kept for API parity.

    Returns:
        ``env_TL`` with ``C1``, ``C2`` and ``T1`` replaced.
    """
    del projector_method

    # ---- Step 1: build the four enlarged corners. ----
    Q_TL = _build_enlarged_corner(
        env_TL.C1, env_TL.T1, env_TL.T4, a_TL, position="top_left"
    )
    Q_TR = _build_enlarged_corner(
        env_TR.C2, env_TR.T1, env_TR.T2, a_TR, position="top_right"
    )
    Q_BL = _build_enlarged_corner(
        env_BL.C4, env_BL.T3, env_BL.T4, a_BL, position="bottom_left"
    )
    Q_BR = _build_enlarged_corner(
        env_BR.C3, env_BR.T3, env_BR.T2, a_BR, position="bottom_right"
    )

    # ---- Step 2: compute the (P_top, P_bot) projector pair. ----
    P_top, P_bot, _, _ = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction="top", base_charges=base_charges
    )

    # ---- Step 3: build C1g, C2g, T1g from env_TL. ----
    # Same C/T fusion structure as the 1x1 ``_ctm_tensor_move_top`` body.
    C1_d = env_TL.C1.relabel("c1_d", "t4_d")
    C1g = contract(C1_d, env_TL.T4)  # (c1_r, l2, t4_u)
    C1g = _fuse_pair_by_label(C1g, "c1_r", "l2", "fused", IN)  # (fused, t4_u)

    C2_d = env_TL.C2.relabel("c2_d", "t2_u")
    C2g = contract(C2_d, env_TL.T2)  # (c2_l, r2, t2_d)
    C2g = _fuse_pair_by_label(C2g, "c2_l", "r2", "fused", IN)  # (fused, t2_d)

    T1_with_a = contract(env_TL.T1, a_TL)
    T1g = _fuse_pair_by_label(T1_with_a, "t1_l", "l2", "fl", IN)
    T1g = _fuse_pair_by_label(T1g, "t1_r", "r2", "fr", OUT)

    # ---- Step 4: convert (P_top, P_bot) to ``(fused, chi_new)`` projectors. ----
    P_1 = _fuse_pair_by_label(P_top, "chi_outer", "fused_D2", "fused", IN)
    P_1 = P_1.relabel("chi_new_top", "chi_new")
    P_2 = _fuse_pair_by_label(P_bot, "chi_outer", "fused_D2", "fused", IN)
    P_2 = P_2.relabel("chi_new_bot", "chi_new")

    # ---- Step 5: absorb projectors into the env. ----
    C1_new, C2_new, T1_new = _apply_projector_with_reembed(
        P_1, P_2, C1g, C2g, T1g, "fl", "fr"
    )

    # ---- Step 6: relabel + flow-flip to env-standard labels. ----
    C1_new = C1_new.relabels({"chi_new": "c1_d", "t4_u": "c1_r"})
    C2_new = C2_new.relabels({"chi_new": "c2_l", "t2_d": "c2_d"})
    T1_new = T1_new.relabels({"chi_new": "t1_l", "chi_new_r": "t1_r", "d2": "u2"})
    # 'd2'->'u2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.u2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    # ---- Step 7: phase-fix + normalize. ----
    C1_new = _phase_fix_normalize_tensor(C1_new)
    C2_new = _phase_fix_normalize_tensor(C2_new)
    T1_new = _phase_fix_normalize_tensor(T1_new)
    return env_TL._replace(C1=C1_new, C2=C2_new, T1=T1_new)


def _ctm_tensor_move_bottom_2x2(
    env_TL: CTMTensorEnv,
    env_TR: CTMTensorEnv,
    env_BL: CTMTensorEnv,
    env_BR: CTMTensorEnv,
    a_TL: Tensor,
    a_TR: Tensor,
    a_BL: Tensor,
    a_BR: Tensor,
    chi: int,
    projector_method: str = "svd",
    base_charges: np.ndarray | None = None,
) -> CTMTensorEnv:
    """Bottom CTM move using 2x2 plaquette projectors. Updates env_TL.{C4,C3,T3}.

    Mirrors :func:`_ctm_tensor_move_left_2x2` with ``direction="bottom"``.

    Args:
        env_TL:   Env at the top-left site of the 2x2 plaquette (whose
                  corners and edges are being updated).
        env_TR:   Env at ``neighbors[s_TL]["right"]``.
        env_BL:   Env at ``neighbors[s_TL]["bottom"]``.
        env_BR:   Env at ``neighbors[s_TR]["bottom"]``.
        a_TL ..a_BR:  Double-layer site tensors at the four plaquette sites.
        chi:      Target bond dimension of the new chi seam.
        projector_method:  Ignored; kept for API parity.

    Returns:
        ``env_TL`` with ``C4``, ``C3`` and ``T3`` replaced.
    """
    del projector_method

    # ---- Step 1: build the four enlarged corners. ----
    Q_TL = _build_enlarged_corner(
        env_TL.C1, env_TL.T1, env_TL.T4, a_TL, position="top_left"
    )
    Q_TR = _build_enlarged_corner(
        env_TR.C2, env_TR.T1, env_TR.T2, a_TR, position="top_right"
    )
    Q_BL = _build_enlarged_corner(
        env_BL.C4, env_BL.T3, env_BL.T4, a_BL, position="bottom_left"
    )
    Q_BR = _build_enlarged_corner(
        env_BR.C3, env_BR.T3, env_BR.T2, a_BR, position="bottom_right"
    )

    # ---- Step 2: compute the (P_top, P_bot) projector pair. ----
    P_top, P_bot, _, _ = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction="bottom", base_charges=base_charges
    )

    # ---- Step 3: build C4g, C3g, T3g from env_TL. ----
    # Same C/T fusion structure as the 1x1 ``_ctm_tensor_move_bottom`` body.
    C4_r = env_TL.C4.relabel("c4_r", "t4_u")
    C4g = contract(C4_r, env_TL.T4)  # (c4_u, t4_d, l2)
    C4g = _fuse_pair_by_label(C4g, "c4_u", "l2", "fused", IN)  # (fused, t4_d)

    C3_l = env_TL.C3.relabel("c3_l", "t2_d")
    C3g = contract(C3_l, env_TL.T2)  # (c3_u, t2_u, r2)
    C3g = _fuse_pair_by_label(C3g, "c3_u", "r2", "fused", IN)  # (fused, t2_u)

    T3_with_a = contract(env_TL.T3, a_TL)
    T3g = _fuse_pair_by_label(T3_with_a, "t3_r", "l2", "fl", IN)
    T3g = _fuse_pair_by_label(T3g, "t3_l", "r2", "fr", OUT)

    # ---- Step 4: convert (P_top, P_bot) to ``(fused, chi_new)`` projectors. ----
    P_1 = _fuse_pair_by_label(P_top, "chi_outer", "fused_D2", "fused", IN)
    P_1 = P_1.relabel("chi_new_top", "chi_new")
    P_2 = _fuse_pair_by_label(P_bot, "chi_outer", "fused_D2", "fused", IN)
    P_2 = P_2.relabel("chi_new_bot", "chi_new")

    # ---- Step 5: absorb projectors into the env. ----
    C4_new, C3_new, T3_new = _apply_projector_with_reembed(
        P_1, P_2, C4g, C3g, T3g, "fl", "fr"
    )

    # ---- Step 6: relabel + flow-flip to env-standard labels. ----
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t4_d": "c4_u"})
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t2_u": "c3_l"})
    T3_new = T3_new.relabels({"chi_new": "t3_r", "chi_new_r": "t3_l", "u2": "d2"})
    # 'u2'->'d2' is a pure relabel (#905): the leg already carries the
    # same charge list and the opposite flow to the next cell's a.d2, so
    # dualing it here made that pair same-flow and deleted every
    # charged sector at the T*a contraction.

    # ---- Step 7: phase-fix + normalize. ----
    C4_new = _phase_fix_normalize_tensor(C4_new)
    C3_new = _phase_fix_normalize_tensor(C3_new)
    T3_new = _phase_fix_normalize_tensor(T3_new)
    return env_TL._replace(C4=C4_new, C3=C3_new, T3=T3_new)
