"""Honeycomb CTM directional moves and step composition.

Built sub-step by sub-step per Task 6 of the honeycomb CTM plan
(``docs/plans/2026-04-25-honeycomb-ctm-plan.md``).

Conventions (Path 1, see plan G1 synthesis 2026-04-27):

* For each sublattice ``s ∈ {A, B}`` and direction ``α ∈ {0, 1, 2}``:

      L^s_α^new = L^s_α · T_s              (own-sublattice absorption)
      R^s_α^new = R^s_α · T_(other s)      (neighbor-sublattice absorption)
      C^s_α^new_unproj = L^s_α · C^s_α · R^s_α · T_s · T_(other s)

* The corner enlargement is the natural composition
  ``L_new · C · R_new``, so the asymmetric L/R pairing produces exactly
  one ``T_s`` and one ``T_(other s)`` per corner update — matching
  Paper 2 §II.C / Fig. 10(c).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_init import _double_layer_honeycomb
from tenax.algorithms._ctm_honeycomb_projector import (
    compute_honeycomb_corner_biorthogonal_projector,
    compute_honeycomb_projector,
)
from tenax.algorithms._ctm_honeycomb_topology import Coord
from tenax.algorithms._ctm_tensor_init import _fuse_pair_by_label
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor

__all__ = [
    "_enlarged_left_column",
    "_enlarged_right_column",
    "_enlarged_corner_matrix",
    "_renormalize_honeycomb_env",
    "honeycomb_ctm_step",
    "move_direction_alpha",
]

OUT = FlowDirection.OUT


def _enlarged_left_column(L_alpha: Tensor, T_self: Tensor, *, alpha: int) -> Tensor:
    """Build ``L^s_α · T_s`` and re-fuse to rank-3 with ``chi → chi·D²``.

    Algorithm:

    1. ``contract(L_α, T_self)`` auto-contracts on the shared label
       ``e{α}_d2``. Result is rank-4 with labels
       ``(chi_in_α, chi_out_α, e_{β}_d2, e_{γ}_d2)`` where
       ``β = (α+1) % 3`` and ``γ = (α+2) % 3``.
    2. Fuse ``chi_in_α`` (slow-varying) with ``e_{γ}_d2`` (fast-varying)
       into a single grown leg of dim ``chi · D²``, retaining the label
       ``chi_in_α``. We grow ``chi_in`` (the "away from C" side) so that
       ``chi_out`` stays at ``chi`` and connects directly to ``C^s_α`` at
       the corner-build step (sub-step 3).
    3. The remaining ``e_{β}_d2`` label is renamed back to ``e{α}_d2`` so
       the result has the canonical ``L_α`` rank-3 layout
       ``(chi_in_α_grown, e{α}_d2, chi_out_α)``.

    Args:
        L_alpha: Rank-3 column tensor with labels
            ``(chi_in_{α}, e{α}_d2, chi_out_{α})`` — the layout produced
            by ``initialize_honeycomb_env``.
        T_self: Rank-3 double-layer tensor with labels
            ``(e0_d2, e1_d2, e2_d2)`` — produced by
            ``_double_layer_honeycomb``. Pass the SAME-sublattice
            double-layer (``T_A`` for ``L^A``, ``T_B`` for ``L^B``).
        alpha: Direction index ``{0, 1, 2}``.

    Returns:
        Rank-3 :class:`Tensor` with labels
        ``(chi_in_{α}, e{α}_d2, chi_out_{α})`` and dims
        ``(chi · D², D², chi)``. ``chi_in`` is the grown leg.
    """
    e_alpha = f"e{alpha}_d2"
    e_beta = f"e{(alpha + 1) % 3}_d2"
    e_gamma = f"e{(alpha + 2) % 3}_d2"
    chi_in_label = f"chi_in_{alpha}"
    chi_out_label = f"chi_out_{alpha}"
    from tenax.core.index import FlowDirection as _FD

    # Step 1: contract on the shared e{α}_d2 label.
    inter = contract(L_alpha, T_self)
    # Free labels: (chi_in_α, chi_out_α, e_β_d2, e_γ_d2)

    # Step 2: fuse chi_in_α with e_γ_d2 → chi_in grown to chi · D².
    inter = _fuse_pair_by_label(inter, chi_in_label, e_gamma, chi_in_label, _FD.IN)
    # Free labels: (chi_in_α_grown, chi_out_α, e_β_d2)

    # Step 3: rename the remaining T-leg (e_β) back to the canonical e{α}_d2.
    inter = inter.relabel(e_beta, e_alpha)

    # Step 4: reorder to canonical layout (chi_in_α, e_α_d2, chi_out_α).
    labels = inter.labels()
    perm = (
        labels.index(chi_in_label),
        labels.index(e_alpha),
        labels.index(chi_out_label),
    )
    return inter.transpose(perm)


def _enlarged_right_column(R_alpha: Tensor, T_other: Tensor, *, alpha: int) -> Tensor:
    """Build ``R^s_α · T_(other s)`` and re-fuse to rank-3 with ``chi → chi·D²``.

    Symmetric to :func:`_enlarged_left_column`, but grows on the OPPOSITE
    chi side — ``chi_out`` (the away-from-C side for R) — so that
    ``chi_in`` stays at ``chi`` and connects to ``C`` at the corner-build
    step. ``T_other`` is the bipartite NEIGHBOR's double-layer
    (``T_B`` for ``R^A``, ``T_A`` for ``R^B``). This asymmetric L/R
    pairing makes the corner enlargement ``L_new · C · R_new``
    automatically contain one ``T_self`` (from L) and one
    ``T_(other s)`` (from R) — matching Paper 2 Fig. 10(c).

    Args:
        R_alpha: Rank-3 column tensor with labels
            ``(chi_in_{α}, e{α}_d2, chi_out_{α})`` — same layout as L.
        T_other: Rank-3 double-layer tensor of the OTHER sublattice.
        alpha: Direction index ``{0, 1, 2}``.

    Returns:
        Rank-3 :class:`Tensor` with labels
        ``(chi_in_{α}, e{α}_d2, chi_out_{α})`` and dims
        ``(chi, D², chi · D²)``. ``chi_out`` is the grown leg.
    """
    e_alpha = f"e{alpha}_d2"
    e_beta = f"e{(alpha + 1) % 3}_d2"
    e_gamma = f"e{(alpha + 2) % 3}_d2"
    chi_in_label = f"chi_in_{alpha}"
    chi_out_label = f"chi_out_{alpha}"

    inter = contract(R_alpha, T_other)
    # Free labels: (chi_in_α, chi_out_α, e_β_d2, e_γ_d2)

    # Fuse chi_OUT (away-from-C side for R) with e_γ → chi_out grown to chi·D².
    inter = _fuse_pair_by_label(inter, chi_out_label, e_gamma, chi_out_label, OUT)

    # Rename e_β back to canonical e{α}_d2.
    inter = inter.relabel(e_beta, e_alpha)

    labels = inter.labels()
    perm = (
        labels.index(chi_in_label),
        labels.index(e_alpha),
        labels.index(chi_out_label),
    )
    return inter.transpose(perm)


def _enlarged_corner_matrix(
    L_alpha: Tensor,
    C_alpha: Tensor,
    R_alpha: Tensor,
    T_self: Tensor,
    T_other: Tensor,
    *,
    alpha: int,
):
    """Build the full enlarged corner ``L · C · R · T_self · T_other`` and
    reshape it to a ``(chi·D², chi·D²)`` dense matrix for biorthogonalization.

    Construction (Path 1 convention):

    1. ``L_new = _enlarged_left_column(L, T_self)`` —  shape ``(chi·D², D², chi)``,
       grown leg = ``chi_in`` (away-from-C side).
    2. ``R_new = _enlarged_right_column(R, T_other)`` — shape ``(chi, D², chi·D²)``,
       grown leg = ``chi_out`` (away-from-C side).
    3. Relabel for the 3-way contract ``L_new · C · R_new``:
         * C's ``chi_in_α`` and L_new's ``chi_out_α`` → ``_lc_bond`` (joins L↔C)
         * C's ``chi_out_α`` and R_new's ``chi_in_α`` → ``_rc_bond`` (joins R↔C)
       Both ``L_new`` and ``R_new`` keep label ``e_α_d2``, so they auto-contract
       on the shared bipartite ``T_self ↔ T_other`` bond.
    4. After the 3-tensor contraction, the only free legs are ``L_new.chi_in_α``
       (dim ``chi·D²``) and ``R_new.chi_out_α`` (dim ``chi·D²``). Densify to
       a 2D matrix.

    Returns:
        ``jax.Array`` of shape ``(chi·D², chi·D²)`` representing the enlarged
        corner reshaped as a matrix. Row index = grown ``L`` chi leg,
        column index = grown ``R`` chi leg.
    """
    L_new = _enlarged_left_column(L_alpha, T_self, alpha=alpha)
    R_new = _enlarged_right_column(R_alpha, T_other, alpha=alpha)

    chi_in_label = f"chi_in_{alpha}"
    chi_out_label = f"chi_out_{alpha}"

    # Disambiguate the two C↔column bonds via shared sentinel labels.
    L_for_corner = L_new.relabel(chi_out_label, "_lc_bond")
    R_for_corner = R_new.relabel(chi_in_label, "_rc_bond")
    C_for_corner = C_alpha.relabels(
        {chi_in_label: "_lc_bond", chi_out_label: "_rc_bond"}
    )

    # 3-tensor contract; auto-contracts on _lc_bond, _rc_bond, and e_α_d2.
    # Force output ordering to match (row, col) = (L's grown leg, R's grown leg).
    enlarged = contract(
        L_for_corner,
        C_for_corner,
        R_for_corner,
        output_labels=(chi_in_label, chi_out_label),
    )
    return enlarged.todense()


# ------------------------------------------------------------------ #
# Sub-step 5+6: projector application + full directional move          #
# ------------------------------------------------------------------ #


def _isometric_to_biorthogonal_pair(
    L_alpha: Tensor, *, method: str, chi: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert the isometric ``compute_honeycomb_projector`` output
    (rank-3 ``P``, rank-3 ``P_dag``) into the (P_L, P_R) matrix pair used
    by :func:`move_direction_alpha`.

    For the A=B opt-in path: ``P_dag · P = I_chi_new`` is the isometric
    identity, which is exactly the biorthogonality condition once we
    flatten ``(chi_in, e_α_d2)`` into a single ``chi·D²`` axis.

    Returns:
        (P_L, P_R) where P_L has shape ``(chi_new, chi·D²)`` and P_R has
        shape ``(chi·D², chi_new)``.
    """
    P, P_dag = compute_honeycomb_projector(L_alpha, method=method, chi=chi)
    # P:     (chi_in, e_α_d2, chi_new_in)   → reshape to (chi_in*D², chi_new) = P_R
    # P_dag: (chi_new_out, chi_in, e_α_d2)  → reshape to (chi_new, chi_in*D²) = P_L
    P_arr = P.todense()
    Pd_arr = P_dag.todense()
    chi_in, d2, chi_new = P_arr.shape
    P_R = P_arr.reshape(chi_in * d2, chi_new)
    P_L = Pd_arr.reshape(chi_new, chi_in * d2)
    return P_L, P_R


def _make_chi_index(chi_new: int, flow: FlowDirection, label: str) -> TensorIndex:
    """Build a trivial-charge chi index of dim ``chi_new``."""
    return TensorIndex.from_charges(
        U1Symmetry(), np.zeros(chi_new, dtype=np.int32), flow, label=label
    )


def _wrap_corner(arr: jnp.ndarray, alpha: int) -> DenseTensor:
    """Wrap a (chi, chi) jax array as a corner DenseTensor with canonical labels."""
    chi = arr.shape[0]
    indices = (
        _make_chi_index(chi, FlowDirection.IN, f"chi_in_{alpha}"),
        _make_chi_index(chi, FlowDirection.OUT, f"chi_out_{alpha}"),
    )
    return DenseTensor(arr, indices)


def _wrap_column(arr: jnp.ndarray, alpha: int, d2_index: TensorIndex) -> DenseTensor:
    """Wrap a (chi, D², chi) jax array as a column DenseTensor with canonical labels.

    ``d2_index`` is reused from the input column so the leg's symmetry /
    charge metadata stays consistent with the env's existing convention.
    """
    chi_in_dim, d2_dim, chi_out_dim = arr.shape
    indices = (
        _make_chi_index(chi_in_dim, FlowDirection.IN, f"chi_in_{alpha}"),
        d2_index,
        _make_chi_index(chi_out_dim, FlowDirection.OUT, f"chi_out_{alpha}"),
    )
    return DenseTensor(arr, indices)


def move_direction_alpha(
    envs: dict[Coord, HoneycombCTMEnv],
    sites: dict[Coord, Tensor],
    *,
    alpha: int,
    chi: int,
    projector_method: str = "biorthogonal",
    forward_gauge: str = "phase",
) -> dict[Coord, HoneycombCTMEnv]:
    """Update ``(C_α, L_α, R_α)`` for BOTH sublattices via a single
    biorthogonal projector pair derived from the joint A/B enlarged corner.

    Algorithm (Path 1 + Paper 2 §II.C / Fig. 10):

    1. Build ``T_A``, ``T_B`` double-layers from ``sites``.
    2. Build per-sublattice enlarged columns:
         L^A_unproj = L^A · T_A,    R^A_unproj = R^A · T_B,
         L^B_unproj = L^B · T_B,    R^B_unproj = R^B · T_A.
    3. Build per-sublattice enlarged corner matrices ``M_A``, ``M_B`` of
       shape ``(chi·D², chi·D²)``.
    4. Compute ``(P_L, P_R)`` from ``(M_A, M_B)`` via the Corboz
       biorthogonalization (``P_L · P_R = I_chi``). For the ``eigh`` /
       ``svd`` opt-in (A=B), build the pair from ``compute_honeycomb_projector``
       on ``L^A`` and reshape.
    5. Truncate per Paper 2:
         C^A_new = P_L · M_A · P_L^T              (chi, chi)
         C^B_new = P_R^T · M_B · P_R              (chi, chi)
         L^A_new = P_L · L^A_unproj  (axis 0)     (chi, D², chi)
         R^A_new = R^A_unproj · P_L^T (axis 2)    (chi, D², chi)
         L^B_new = P_R^T · L^B_unproj (axis 0)    (chi, D², chi)
         R^B_new = R^B_unproj · P_R   (axis 2)    (chi, D², chi)
    6. Return new envs dict.

    The ``forward_gauge`` parameter is reserved for the isometric A=B
    path's per-absorption sigma-gauge fix; in the biorthogonal default,
    biorthogonalization absorbs the gauge dof and ``forward_gauge`` is a
    no-op (memory: ``feedback_phase_gauge_default.md``).
    """
    A_site = sites[(0, 0)]
    B_site = sites[(1, 0)]
    T_A = _double_layer_honeycomb(A_site)
    T_B = _double_layer_honeycomb(B_site)

    env_A = envs[(0, 0)]
    env_B = envs[(1, 0)]
    L_A = getattr(env_A, f"L{alpha}")
    C_A = getattr(env_A, f"C{alpha}")
    R_A = getattr(env_A, f"R{alpha}")
    L_B = getattr(env_B, f"L{alpha}")
    C_B = getattr(env_B, f"C{alpha}")
    R_B = getattr(env_B, f"R{alpha}")

    # Step 2: enlarged columns (rank-3 with one chi grown to chi·D²).
    L_A_unproj = _enlarged_left_column(L_A, T_A, alpha=alpha)
    R_A_unproj = _enlarged_right_column(R_A, T_B, alpha=alpha)
    L_B_unproj = _enlarged_left_column(L_B, T_B, alpha=alpha)
    R_B_unproj = _enlarged_right_column(R_B, T_A, alpha=alpha)

    # Step 3: enlarged corner matrices (chi·D², chi·D²).
    M_A = _enlarged_corner_matrix(L_A, C_A, R_A, T_A, T_B, alpha=alpha)
    M_B = _enlarged_corner_matrix(L_B, C_B, R_B, T_B, T_A, alpha=alpha)

    # Step 4: biorthogonal (default) or isometric (eigh/svd opt-in for A=B).
    if projector_method == "biorthogonal":
        P_L, P_R = compute_honeycomb_corner_biorthogonal_projector(M_A, M_B, chi=chi)
    elif projector_method in ("eigh", "svd"):
        P_L, P_R = _isometric_to_biorthogonal_pair(
            L_A, method=projector_method, chi=chi
        )
    else:
        raise ValueError(
            f"Unknown projector_method: {projector_method!r}; expected "
            "'biorthogonal', 'eigh', or 'svd'."
        )

    # Step 5: truncate. P_L: (chi_new, chi·D²); P_R: (chi·D², chi_new).
    chi_new = P_L.shape[0]

    # Corner truncation: A uses P_L on both legs, B uses P_R^T on both.
    C_A_new = (P_L @ M_A) @ P_L.T  # (chi_new, chi_new)
    C_B_new = (P_R.T @ M_B) @ P_R  # (chi_new, chi_new)

    # Column truncation: contract the grown chi·D² axis of each.
    # einsum letters: 'A' = grown leg (chi·D²), 'k' = new chi, 'd' = D², 'c' = unchanged chi.
    # L_unproj shape: (chi·D², D², chi); apply P on axis 0.
    # R_unproj shape: (chi, D², chi·D²); apply P on axis 2.
    L_A_arr = L_A_unproj.todense()
    R_A_arr = R_A_unproj.todense()
    L_B_arr = L_B_unproj.todense()
    R_B_arr = R_B_unproj.todense()

    # Sublattice A → P_L; sublattice B → P_R (used as P_R.T on the L/grown-axis-0 side
    # to match P_R.T's (chi_new, chi·D²) shape, and as P_R on the R/axis-2 side).
    L_A_new_arr = jnp.einsum("kA,Adc->kdc", P_L, L_A_arr)
    R_A_new_arr = jnp.einsum("cdA,Ak->cdk", R_A_arr, P_L.T)
    L_B_new_arr = jnp.einsum("kA,Adc->kdc", P_R.T, L_B_arr)
    R_B_new_arr = jnp.einsum("cdA,Ak->cdk", R_B_arr, P_R)

    # Step 6: wrap back to DenseTensors with canonical labels.
    # Reuse the d2 index from the original column (it carries the right symmetry).
    d2_index_A = L_A.indices[L_A.labels().index(f"e{alpha}_d2")]
    d2_index_B = L_B.indices[L_B.labels().index(f"e{alpha}_d2")]

    new_env_A = env_A._replace(
        **{
            f"C{alpha}": _wrap_corner(C_A_new, alpha),
            f"L{alpha}": _wrap_column(L_A_new_arr, alpha, d2_index_A),
            f"R{alpha}": _wrap_column(R_A_new_arr, alpha, d2_index_A),
        }
    )
    new_env_B = env_B._replace(
        **{
            f"C{alpha}": _wrap_corner(C_B_new, alpha),
            f"L{alpha}": _wrap_column(L_B_new_arr, alpha, d2_index_B),
            f"R{alpha}": _wrap_column(R_B_new_arr, alpha, d2_index_B),
        }
    )

    new_envs = dict(envs)
    new_envs[(0, 0)] = new_env_A
    new_envs[(1, 0)] = new_env_B
    _ = chi_new, forward_gauge  # currently unused; reserved for sigma-gauge follow-up
    return new_envs


# ------------------------------------------------------------------ #
# Task 7: full step (3-direction sweep + per-step renormalization)     #
# ------------------------------------------------------------------ #


def _normalize_tensor(T: Tensor) -> Tensor:
    """Inf-norm normalize ``T`` to keep the env from blowing up over many sweeps."""
    return T * (1.0 / (T.max_abs() + EPS))


def _renormalize_honeycomb_env(env: HoneycombCTMEnv) -> HoneycombCTMEnv:
    """Normalize all 9 fields of one sublattice's env by max-abs."""
    return HoneycombCTMEnv(*[_normalize_tensor(getattr(env, f)) for f in env._fields])


def honeycomb_ctm_step(
    envs: dict[Coord, HoneycombCTMEnv],
    sites: dict[Coord, Tensor],
    *,
    chi: int,
    projector_method: str = "biorthogonal",
    forward_gauge: str = "phase",
    renormalize: bool = True,
) -> dict[Coord, HoneycombCTMEnv]:
    """One full CTM iteration: 3 directional moves + optional renorm.

    Sweeps ``α ∈ {0, 1, 2}`` in order, calling :func:`move_direction_alpha`
    for each direction; after the sweep, optionally renormalizes every env
    field by max-abs to prevent exponential growth (matches YASTN /
    variPEPS per-step normalization).
    """
    for alpha in (0, 1, 2):
        envs = move_direction_alpha(
            envs,
            sites,
            alpha=alpha,
            chi=chi,
            projector_method=projector_method,
            forward_gauge=forward_gauge,
        )
    if renormalize:
        envs = {coord: _renormalize_honeycomb_env(env) for coord, env in envs.items()}
    return envs
