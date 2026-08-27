"""Standard CTM with Tensor protocol — sweep loop, convergence, and main entry points."""

from __future__ import annotations

__all__ = [
    "CHECKERBOARD_NEIGHBORS",
    "Coord",
    "SINGLE_SITE_NEIGHBORS",
    "_DIRECTION_MOVES",
    "_DIRECTION_MOVES_2X2",
    "_corner_singular_values",
    "_ctm_sv_diff",
    "_ctm_tensor_multisite",
    "_ctm_tensor_sweep",
    "_ctm_tensor_sweep_multisite",
    "_ctm_tensor_sweep_paired",
    "_normalize_tensor",
    "_renormalize_tensor_env",
    "ctm_multisite",
    "make_neighbors",
    "ctm_tensor",
    "ctm_tensor_2site",
]

import warnings

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import (
    _compute_plaquette_projector_pair,
    _ctm_tensor_absorb_bottom_2plaq,
    _ctm_tensor_absorb_left_2plaq,
    _ctm_tensor_absorb_right_2plaq,
    _ctm_tensor_absorb_top_2plaq,
    _ctm_tensor_move_bottom,
    _ctm_tensor_move_bottom_2x2,
    _ctm_tensor_move_left,
    _ctm_tensor_move_left_2x2,
    _ctm_tensor_move_right,
    _ctm_tensor_move_right_2x2,
    _ctm_tensor_move_top,
    _ctm_tensor_move_top_2x2,
)
from tenax.algorithms._ctm_tensor_paired_moves import (
    _ctm_tensor_move_horizontal,
    _ctm_tensor_move_vertical,
)
from tenax.core import EPS
from tenax.core.lattice import Lattice
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
from tenax.linalg import _dense_svd

# ------------------------------------------------------------------ #
# Sweep + renormalize                                                  #
# ------------------------------------------------------------------ #


def _normalize_tensor(T: Tensor) -> Tensor:
    """Normalize tensor by max abs value, matching dense CTM convention.

    Uses polymorphic ``max_abs()`` and scalar division so that
    SymmetricTensor stays block-sparse without a dense round-trip.
    """
    norm = T.max_abs()
    return T * (1.0 / (norm + EPS))


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
    projector_method: str = "svd",
    projector_backward: str = "auto",
) -> tuple[CTMTensorEnv, float]:
    """One full CTM sweep: left, top, right, bottom (variPEPS order) + optional renormalize.

    For SymmetricTensor inputs, derives ``base_charges`` from the double-layer
    ``a`` and threads them into each projector via the move functions.  This
    matches the multisite path (``_ctm_tensor_sweep_multisite``) and ensures
    the ε_T measurement reflects the same per-sector allocation the auto-χ
    bump will apply when padding the env (PR #433 codex review on
    ``ipeps_optimize.py:1007``).

    Returns:
        ``(env, max_eps)`` where ``max_eps`` is the maximum per-move truncation
        error across the four directional moves in this sweep.
    """
    base_charges = _get_base_charges(a)
    env, eps_left = _ctm_tensor_move_left(
        env,
        env,
        a,
        chi,
        projector_method,
        base_charges=base_charges,
        projector_backward=projector_backward,
    )
    env, eps_top = _ctm_tensor_move_top(
        env,
        env,
        a,
        chi,
        projector_method,
        base_charges=base_charges,
        projector_backward=projector_backward,
    )
    env, eps_right = _ctm_tensor_move_right(
        env,
        env,
        a,
        chi,
        projector_method,
        base_charges=base_charges,
        projector_backward=projector_backward,
    )
    env, eps_bottom = _ctm_tensor_move_bottom(
        env,
        env,
        a,
        chi,
        projector_method,
        base_charges=base_charges,
        projector_backward=projector_backward,
    )
    if renormalize:
        env = _renormalize_tensor_env(env)
    # Moves return eps as a (traced) array so the jitted multisite path stays
    # jit-safe; this eager single-site sweep converts to a Python float here.
    max_eps = max(float(eps_left), float(eps_top), float(eps_right), float(eps_bottom))
    return env, max_eps


def _ctm_tensor_sweep_paired(
    env: CTMTensorEnv,
    a: Tensor,
    chi: int,
    renormalize: bool,
    projector_method: str = "svd",
    projector_backward: str = "auto",
) -> tuple[CTMTensorEnv, float]:
    """One full CTM sweep using paired moves: horizontal then vertical.

    Uses 2x2 enlarged corners for projector computation, ensuring
    consistent charge-sector distributions across sweeps for
    SymmetricTensor inputs.

    Returns:
        ``(env, max_eps)`` where ``max_eps`` is the maximum per-move truncation
        error across the two paired moves in this sweep.
    """
    env, eps_horiz = _ctm_tensor_move_horizontal(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env, eps_vert = _ctm_tensor_move_vertical(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    if renormalize:
        env = _renormalize_tensor_env(env)
    max_eps = max(eps_horiz, eps_vert)
    return env, max_eps


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


def make_neighbors(nx: int, ny: int) -> dict[Coord, dict[str, Coord]]:
    """Build periodic neighbor map for an nx * ny unit cell.

    Coordinates are (x, y) with periodic boundary conditions.
    """
    neighbors: dict[Coord, dict[str, Coord]] = {}
    for x in range(nx):
        for y in range(ny):
            neighbors[(x, y)] = {
                "left": ((x - 1) % nx, y),
                "right": ((x + 1) % nx, y),
                "top": (x, (y - 1) % ny),
                "bottom": (x, (y + 1) % ny),
            }
    return neighbors


_DIRECTION_MOVES = [
    ("left", _ctm_tensor_move_left),
    ("top", _ctm_tensor_move_top),
    ("right", _ctm_tensor_move_right),
    ("bottom", _ctm_tensor_move_bottom),
]


# 2x2 plaquette moves take the same plaquette layout in every direction:
# (TL=s, TR=s.right, BL=s.bottom, BR=s.right.bottom).  The "direction" is
# encoded inside the move via :func:`_compute_2x2_projector`, which selects
# which seam (left/right/top/bottom) the projector pair compresses.  This
# matches variPEPS's ``do_*_absorption`` workhorses, where each direction's
# plaquette is anchored at the cell whose env is being updated and uses the
# 4 sites at offsets {(0,0), (1,0), (0,1), (1,1)} relative to that anchor.
_DIRECTION_MOVES_2X2 = [
    ("left", _ctm_tensor_move_left_2x2),
    ("top", _ctm_tensor_move_top_2x2),
    ("right", _ctm_tensor_move_right_2x2),
    ("bottom", _ctm_tensor_move_bottom_2x2),
]


def _get_base_charges(a: Tensor):
    """Extract base charges from a double-layer tensor for projector stabilization."""
    if not isinstance(a, SymmetricTensor):
        return None
    import numpy as _np

    u2_pos = a.labels().index("u2")
    charges = _np.asarray(a.indices[u2_pos].charges, dtype=_np.int32)
    if _np.all(charges == 0):
        return None
    return charges


def _sort_coords_for_direction(coords: list[Coord], direction: str) -> list[Coord]:
    """Sort coordinates for correct cascading order in a CTM direction move.

    Following Corboz et al. PRB 84, 041108(R) (2011):
    - Left move absorbs from the left, so process columns left-to-right
      (increasing x) so updated environments cascade rightward.
    - Right move: process right-to-left (decreasing x).
    - Top move: process top-to-bottom (increasing y).
    - Bottom move: process bottom-to-top (decreasing y).

    Within each column/row, the perpendicular coordinate is sorted in
    natural order.
    """
    if direction == "left":
        return sorted(coords, key=lambda c: (c[0], c[1]))
    elif direction == "right":
        return sorted(coords, key=lambda c: (-c[0], c[1]))
    elif direction == "top":
        return sorted(coords, key=lambda c: (c[1], c[0]))
    elif direction == "bottom":
        return sorted(coords, key=lambda c: (-c[1], c[0]))
    return sorted(coords)


def _ctm_tensor_sweep_multisite(
    envs: dict[Coord, CTMTensorEnv],
    double_layers: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    renormalize: bool,
    projector_method: str = "svd",
    projector_backward: str = "auto",
    recipe: str = "2x2",
    device_mesh=None,
    chunk_size: int | None = None,
) -> tuple[dict[Coord, CTMTensorEnv], jax.Array, jax.Array]:
    """One full multisite CTM sweep over all sites and directions.

    Args:
        envs:           Per-coord environments to update.
        double_layers:  Per-coord double-layer tensors.
        neighbors:      Per-coord direction → neighbor coord map.
        chi:            Target environment bond dimension.
        renormalize:    Renormalize env tensors after each sweep.
        projector_method:  ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
                        Only consulted on the 1x1 path; the 2x2 path always
                        uses Fishman SVD via :func:`_compute_2x2_projector`.
        projector_backward:  Forwarded to the 1x1 move functions; ignored
                        on the 2x2 path.
        recipe:         ``"2x2"`` (default) — use the 2x2 plaquette projector
                        of :func:`_ctm_tensor_move_*_2x2` (matches variPEPS's
                        ``do_*_absorption`` workhorses).
                        ``"1x1"`` — use the legacy 1x1 (single-site enlarged
                        corner) projector pair.  Available for backward
                        compatibility / regression bisection.

    Returns:
        ``(envs, max_eps, max_smallest_S)`` — updated per-coord env dict,
        the max truncation error across all moves (drives end-of-outer-step
        ``chi_auto_bump``; Issue #474), and the max ``norm_smallest_S``
        across all projector SVDs (drives in-CTM χ-bump; Issue #492).
        ``max_smallest_S`` is 0.0 on the 1x1 path (not yet tracked there).
    """
    # Extract base charges from any double-layer tensor for projector stabilization
    base_charges = None
    for a in double_layers.values():
        base_charges = _get_base_charges(a)
        if base_charges is not None:
            break

    # Shallow-copy so callers that saved a reference to the input dict
    # (e.g. ``envs_old = envs`` for sigma gauge) still see the pre-sweep
    # environments.  Only the dict is copied; env objects are not cloned.
    envs = dict(envs)

    all_coords = list(envs.keys())
    max_eps = jnp.asarray(0.0)
    max_smallest_S = jnp.asarray(0.0)

    # GSPMD: when a device mesh is supplied, re-shard the absorbed double-layer
    # ``a`` onto its surviving-leg layout for the move's direction so the
    # dominant χ²·D⁶ absorption intermediate stays at ≈1/N per device.
    # ``with_sharding_constraint`` is a pure layout hint (never changes
    # numerics), so the flag-off path below is a literal no-op identical to
    # the single-device code.  Hoisted here so it is available to both the
    # 1x1 and 2x2 branches.
    if device_mesh is not None:
        from tenax.algorithms.ctm_sharding import (
            constrain_double_layer_for_move,
        )

        def _shard_a(a, direction):
            return constrain_double_layer_for_move(a, direction, device_mesh)
    else:

        def _shard_a(a, direction):
            return a

    if recipe == "1x1":
        for direction, move_fn in _DIRECTION_MOVES:
            for coord in _sort_coords_for_direction(all_coords, direction):
                nb = neighbors[coord][direction]
                envs[coord], eps_t = move_fn(
                    envs[coord],
                    envs[nb],
                    _shard_a(double_layers[nb], direction),
                    chi,
                    projector_method,
                    base_charges=base_charges,
                    projector_backward=projector_backward,
                    chunk_size=chunk_size,
                )
                max_eps = jnp.maximum(max_eps, jnp.asarray(eps_t))
    elif recipe == "2x2":
        # (The _shard_a closure is now defined above, shared with 1x1.)

        # variPEPS-style 2-plaquette absorption: for each direction, two
        # phases.
        #
        # Phase 1: compute projector pair (P_top, P_bot) per cell, anchored
        # at that cell, with direction-specific seam.  Each plaquette is
        # ``s_anchor, neighbors[s_anchor]["right"],
        # neighbors[s_anchor]["bottom"],
        # neighbors[neighbors[s_anchor]["right"]]["bottom"]``.
        #
        # Phase 2: for each cell ``s_dst`` whose env is being updated, the
        # absorption uses TWO plaquettes:
        #
        #   - LEFT:  s_src = neighbors[s_dst]["left"];
        #            "above" plaquette anchored at neighbors[s_src]["top"];
        #            "current" plaquette anchored at s_src.
        #   - RIGHT: s_src = neighbors[s_dst]["right"];
        #            "above" anchored at neighbors[s_above_TL]["top"]
        #            where the plaquette geometry covers s_src on the
        #            right column — see _ctm_tensor_absorb_right_2plaq.
        #   - TOP:   s_src = neighbors[s_dst]["top"];
        #            "left" plaquette anchored at neighbors[s_src]["left"];
        #            "current" plaquette anchored at s_src.
        #   - BOTTOM:s_src = neighbors[s_dst]["bottom"];
        #            "left" anchored at neighbors[s_src]["left"];
        #            "current" anchored at s_src.
        #
        # Each direction draws projectors from the cell whose env tensors
        # are being absorbed, so projectors and env tensors come from the
        # same snapshot of ``envs`` taken at the start of the direction's
        # phase 1.  Without this snapshot, cells processed later in the
        # direction would see partially-updated env tensors used to build
        # the plaquette (a stale-projector vs fresh-absorbed inconsistency
        # known to destabilise multisite CTM convergence).
        for direction in ("left", "top", "right", "bottom"):
            envs_old = dict(envs)
            # Phase 1: precompute projector pairs anchored at every cell.
            # ``_compute_plaquette_projector_pair`` returns
            # ``(P_top, P_bot, eps_T, smallest_S)`` (Issues #474 / #492).
            # Strip the scalars here and track running aggregates across
            # all (direction × cell) computations.  ``max_eps`` drives the
            # end-of-outer-step ``chi_auto_bump``; ``max_smallest_S``
            # drives the in-CTM χ-bump (bump when ANY direction's normalized
            # smallest kept SV exceeds the threshold — variPEPS semantics).
            projectors: dict[Coord, tuple] = {}
            for s_anchor in all_coords:
                s_TR = neighbors[s_anchor]["right"]
                s_BL = neighbors[s_anchor]["bottom"]
                s_BR = neighbors[s_TR]["bottom"]
                P_top, P_bot, eps_T_plaq, smallest_S_plaq = (
                    _compute_plaquette_projector_pair(
                        envs_old[s_anchor],
                        envs_old[s_TR],
                        envs_old[s_BL],
                        envs_old[s_BR],
                        double_layers[s_anchor],
                        double_layers[s_TR],
                        double_layers[s_BL],
                        double_layers[s_BR],
                        chi,
                        direction,
                        base_charges=base_charges,
                    )
                )
                projectors[s_anchor] = (P_top, P_bot)
                max_eps = jnp.maximum(max_eps, jnp.asarray(eps_T_plaq))
                max_smallest_S = jnp.maximum(
                    max_smallest_S, jnp.asarray(smallest_S_plaq)
                )

            # Phase 2: absorb per cell using TWO plaquettes' projectors.
            new_envs: dict[Coord, CTMTensorEnv] = {}
            for s_dst in _sort_coords_for_direction(all_coords, direction):
                if direction == "left":
                    # Source cell whose column is absorbed and projectors
                    # are sourced from.  variPEPS writes (new C1, T4, C4)
                    # of cell ``(x, y+1)`` from the absorbed column at
                    # ``(x, y)``.  In Tenax terms, env at ``s_dst`` has a
                    # left edge that bounds against ``s_src`` on its left,
                    # and the pre-projection column is at ``s_src``.
                    s_src = neighbors[s_dst]["left"]
                    s_above_anchor = neighbors[s_src]["top"]
                    P_top_above, P_bot_above = projectors[s_above_anchor]
                    P_top_curr, P_bot_curr = projectors[s_src]
                    C1_new, T4_new, C4_new = _ctm_tensor_absorb_left_2plaq(
                        envs_old[s_src],
                        _shard_a(double_layers[s_src], "left"),
                        P_top_above,
                        P_bot_above,
                        P_top_curr,
                        P_bot_curr,
                    )
                    new_envs[s_dst] = envs_old[s_dst]._replace(
                        C1=C1_new, T4=T4_new, C4=C4_new
                    )
                elif direction == "right":
                    # Mirror of LEFT: source cell is to the RIGHT of dst.
                    # Plaquettes that compress the seam between ``s_src``
                    # and ``s_dst`` are anchored at the cell whose
                    # plaquette includes both as the LEFT and RIGHT
                    # columns.  For the RIGHT direction with the same
                    # plaquette geometry as LEFT (TL=anchor,
                    # TR=anchor.right, BL=anchor.bottom, BR=TR.bottom),
                    # the plaquette anchored at ``neighbors[s_dst]["top"]``
                    # has TR = neighbors[s_dst]["top"].right = s_src.top
                    # (and we want the RIGHT-column seam of that plaquette,
                    # which compresses the seam between TR and BR =
                    # between s_src.top and s_src).  The "current"
                    # plaquette anchored at ``s_dst`` similarly compresses
                    # the seam between TR=s_src and BR=s_src.bottom.
                    s_src = neighbors[s_dst]["right"]
                    s_above_anchor = neighbors[s_dst]["top"]
                    P_top_above, P_bot_above = projectors[s_above_anchor]
                    P_top_curr, P_bot_curr = projectors[s_dst]
                    C2_new, T2_new, C3_new = _ctm_tensor_absorb_right_2plaq(
                        envs_old[s_src],
                        _shard_a(double_layers[s_src], "right"),
                        P_top_above,
                        P_bot_above,
                        P_top_curr,
                        P_bot_curr,
                    )
                    new_envs[s_dst] = envs_old[s_dst]._replace(
                        C2=C2_new, T2=T2_new, C3=C3_new
                    )
                elif direction == "top":
                    s_src = neighbors[s_dst]["top"]
                    s_left_anchor = neighbors[s_src]["left"]
                    P_top_left, P_bot_left = projectors[s_left_anchor]
                    P_top_curr, P_bot_curr = projectors[s_src]
                    C1_new, T1_new, C2_new = _ctm_tensor_absorb_top_2plaq(
                        envs_old[s_src],
                        _shard_a(double_layers[s_src], "top"),
                        P_top_left,
                        P_bot_left,
                        P_top_curr,
                        P_bot_curr,
                    )
                    new_envs[s_dst] = envs_old[s_dst]._replace(
                        C1=C1_new, T1=T1_new, C2=C2_new
                    )
                else:  # direction == "bottom"
                    # Mirror of TOP: same geometry as TOP, but the seam
                    # being compressed is the BOTTOM row of the plaquette.
                    # Plaquette anchored at neighbors[s_dst]["left"] has
                    # BL = s_dst.left.bottom and BR = s_dst.bottom = s_src;
                    # cutting the BOTTOM row gives the seam between BL and
                    # BR.  The "current" plaquette anchored at s_dst has
                    # BL = s_src and BR = s_src.right; cutting BOTTOM row
                    # gives the seam between s_src and s_src.right.
                    s_src = neighbors[s_dst]["bottom"]
                    s_left_anchor = neighbors[s_dst]["left"]
                    P_top_left, P_bot_left = projectors[s_left_anchor]
                    P_top_curr, P_bot_curr = projectors[s_dst]
                    C4_new, T3_new, C3_new = _ctm_tensor_absorb_bottom_2plaq(
                        envs_old[s_src],
                        _shard_a(double_layers[s_src], "bottom"),
                        P_top_left,
                        P_bot_left,
                        P_top_curr,
                        P_bot_curr,
                    )
                    new_envs[s_dst] = envs_old[s_dst]._replace(
                        C4=C4_new, T3=T3_new, C3=C3_new
                    )
            envs = new_envs
    else:
        raise ValueError(f"Unknown CTM recipe {recipe!r}: expected '1x1' or '2x2'.")
    if renormalize:
        envs = {c: _renormalize_tensor_env(e) for c, e in envs.items()}
    return envs, max_eps, max_smallest_S


# ------------------------------------------------------------------ #
# Main entry: convergence loop                                         #
# ------------------------------------------------------------------ #


def _ctm_sv_diff(
    sv_new: jax.Array, sv_old: jax.Array, max_rank: int | None = None
) -> jax.Array:
    """Compute max absolute difference between normalized singular value vectors.

    On direction-dependent (asymmetric-bond) states the corner block
    structure fills out empty charge sectors during CTM warmup, so the
    concatenated per-sector SV vector can have a different length from one
    iteration to the next (#670).  Zero-pad the shorter vector to the common
    length: newly-appearing singular values then register as a nonzero diff,
    which correctly reports the env as *not yet converged* while it is still
    changing shape.  Once the block structure stabilises the lengths match
    and the diff reduces to the usual element-wise comparison.

    **Returns ``inf`` when the comparison cannot mean anything (#898).**
    Normalising by the sum is what makes this comparable across sweeps under
    ``renormalize`` -- the absolute corner scale is meaningless by design --
    but it also means a **rank-1** spectrum normalises to ``[1, 0, ..., 0]``
    *whatever* the environment is doing.  Two completely different
    environments then compare equal to within an ulp, so every loop that tests
    this against ``conv_tol`` certifies a collapsed environment as converged:
    measured, the returned energy was bit-identical at ``max_iter``
    60/120/300/400/800, ``conv_tol`` 1e-9/1e-12/1e-14 and ``chi`` 8/12/24/48,
    while sitting 8.8e-3 above the fixed point the same loop reaches by sweep
    41.

    ``inf`` is the honest value rather than a sentinel: on a rank-1 corner the
    true difference between the two environments is *unbounded*, because the
    spectrum carries no information about them.  Reporting it here rather than
    at the nine separate call sites is deliberate -- every one of them already
    compares against ``conv_tol``, so they all fail closed with no change, and
    a future tenth loop inherits the guard instead of re-acquiring the bug.
    """
    # ``jnp.where``, not a Python ``if``: one of the nine loops runs inside
    # ``jax.lax.while_loop`` (``ipeps_ctm_convergence``), where the predicate
    # is a tracer and branching on it raises TracerBoolConversionError.  Both
    # arms are cheap, so evaluating them unconditionally costs nothing.
    blind = _spectrum_is_uninformative(
        sv_new, max_rank=max_rank
    ) | _spectrum_is_uninformative(sv_old, max_rank=max_rank)
    n = max(sv_new.shape[0], sv_old.shape[0])
    if n == 0:
        # Both empty: a ``SymmetricTensor`` corner with no populated blocks
        # returns an empty spectrum, and if that persists across sweeps the
        # reduction below raises on an empty array *before* the ``jnp.where``
        # can return the fail-closed value (#903 review).  Shapes are static,
        # so this branch is safe under tracing.
        return jnp.asarray(jnp.inf)
    sv_new = jnp.pad(sv_new, (0, n - sv_new.shape[0]))
    sv_old = jnp.pad(sv_old, (0, n - sv_old.shape[0]))
    sv1 = sv_new / (jnp.sum(sv_new) + 1e-15)
    sv2 = sv_old / (jnp.sum(sv_old) + 1e-15)
    diff = jnp.max(jnp.abs(sv1 - sv2))
    return jnp.where(blind, jnp.inf, diff)


#: Relative singular-value cutoff for the rank test, matching
#: ``_ctm_diagnostics.ctm_corner_rank`` so the convergence detector and the
#: collapse detector cannot disagree about the same corner.
_RANK_TOL = 1e-10


def _max_virtual_bond_dim(t) -> int:
    """Largest virtual bond dimension, over **all** virtual legs.

    Reading ``indices[0]`` alone -- which this did until #903's review caught it
    -- is wrong for anisotropic states.  A chain embedding with
    ``(u, d, l, r) = (1, 1, 4, 4)`` reports ``1`` purely because the first leg
    is trivial, the call sites then pass ``max_rank=1``, and every positive
    rank-one corner is accepted as informative while the horizontal bonds still
    carry correlations.  That silently restores the premature-convergence defect
    this whole change exists to remove -- a *wrong* bound is worse than a
    missing one, because a missing one fails closed.

    The rank-one exemption belongs only to states whose virtual dimensions are
    **all** one, so the maximum is the quantity to take: it is 1 exactly when
    every virtual leg is 1.

    Physical legs are excluded by label, so this is correct for a site tensor
    ``(u, d, l, r, phys)`` and for a double layer, which has no physical leg.
    """
    idx = getattr(t, "indices", None)
    if idx is None:
        shape = tuple(t.shape)
        dims = list(shape[:4]) if len(shape) >= 4 else list(shape)
        return max(int(d) for d in dims) if dims else 1
    labels = tuple(t.labels()) if hasattr(t, "labels") else ()
    dims = [
        int(ix.dim)
        for k, ix in enumerate(idx)
        if not (k < len(labels) and str(labels[k]).startswith("phys"))
    ]
    return max(dims) if dims else 1


def _forced_corner_rank(bond_dim: int) -> int:
    """The rank the *state* holds a corner to, as opposed to a collapse.

    **``chi`` is deliberately not an argument.**  The first version of this took
    ``min(chi, bond_dim)``, reasoning that rank 1 says nothing about collapse
    whenever rank 1 was all that was on offer.  The first half of that is true
    for ``chi == 1`` as well as ``D == 1``; the second half is not, and the
    difference is the whole point (#903 review, P1 round 2).

    At ``chi = 1`` with ``D > 1`` the corner is 1x1, so its sum-normalised
    spectrum is ``[1]`` *however the environment moves* -- which is #898's
    blindness exactly, not a fixed point.  Certifying there would report
    success on the first eligible comparison of an environment still in
    motion.  Only ``D == 1`` makes the environment exact by construction,
    because only then has the state nothing further to express.

    A small ``chi`` makes the comparison *less* able to see, which is a reason
    to stay fail-closed and never a licence to certify.  Do not reintroduce it
    here; ``test_the_rank_ceiling_comes_from_the_state_not_from_chi`` pins that.

    **This is not a tight bound on corner rank and does not claim to be.**  A
    healthy ``D=2`` corner at ``chi=12`` reaches rank 12, far above
    ``D**2 = 4``.  It only has to be tight at ``D == 1``.  For ``D >= 2`` it is
    at least 4, so a rank-1 corner is still read as collapsed and #898's guard
    is untouched -- the property this whole change puts at risk.

    Args:
        bond_dim: Double-layer bond dimension (``D**2`` for a PEPS of bond
                  dimension ``D``).

    Returns:
        ``max(1, bond_dim)``.
    """
    return max(1, int(bond_dim))


def _spectrum_is_uninformative(
    sv: jax.Array, tol: float = _RANK_TOL, max_rank: int | None = None
) -> jax.Array:
    """Whether a sum-normalised comparison of ``sv`` is forced (#898).

    True when the corner has numerical rank <= 1 -- one non-zero singular value
    normalises to ``1.0`` and the rest to ``0.0`` regardless of the environment
    -- or when the spectrum is empty or non-finite.  This is deliberately
    ``env_is_collapsed``'s predicate: a corner the collapse detector calls dead
    is exactly one the convergence detector cannot see.

    **Unless rank 1 was all the corner could carry** (#903 review, P1).  A
    ``D=1`` PEPS is an exact product state, and this project supports and tests
    one (``TestProductStateEnergy``); a ``chi=1`` environment is legitimate
    too.  Every corner there is rank 1 *at the exact fixed point*, so equating
    rank 1 with collapse made those states unable to converge at all: measured,
    ``ctm(D=1, chi=4, max_iter=20)`` returned ``converged=False`` with
    ``diff=inf`` after burning the whole budget, ``ipeps()`` warned that a
    state which had converged exactly had not, and ``_ctm_sv_diff(sv, sv)`` --
    a spectrum against *itself* -- returned ``inf``.

    Rank is therefore not the discriminator; **reachable** rank is.  Pass
    ``max_rank = min(chi, D_doublelayer)`` and a corner sitting at its ceiling
    is read as informative, because there is nothing further for it to show.
    Omitting ``max_rank`` keeps the conservative reading -- a caller that has
    not said what was reachable gets the fail-closed answer.

    Written in ``jnp`` and returning an array so it survives tracing; one of
    the nine convergence loops runs inside ``jax.lax.while_loop``.  ``max_rank``
    comes from tensor shapes, so it is static and safe to branch on in Python.
    """
    if sv.shape[0] == 0:
        return jnp.asarray(True)
    top = sv[0]
    # EVERY element, not just the leading one (#903 review, P1).  A block-sparse
    # corner can go non-finite in one block while others stay healthy, giving
    # e.g. ``[1, 0.5, nan]``.  Checking ``sv[0]`` alone calls that informative,
    # ``_ctm_sv_diff`` then returns NaN, and the Python loops compare
    # ``diff >= conv_tol`` -- which is **False** for NaN -- so the corrupted
    # environment is silently certified.  Failing closed requires the whole
    # spectrum to be finite.
    healthy = jnp.all(jnp.isfinite(sv)) & (top > 0.0)
    rank = jnp.sum(sv > tol * top)
    informative = healthy & (rank > 1)
    if max_rank is not None:
        # At the ceiling: no higher rank was available, so a low rank is not
        # evidence of collapse.  ``healthy`` still gates it -- a zero or
        # non-finite corner is not a fixed point at any ceiling.
        informative = informative | (healthy & (rank >= max_rank))
    return jnp.logical_not(informative)


def _blind_corner_message(blind: set[Coord], collapsed: set[Coord]) -> str:
    """Text for the uncertified-environment warning (#898).

    Pure, and separate from the loop, because the interesting cases are
    combinations of two coordinate sets and driving a real CTM into each of
    them is far harder than the message logic deserves.  Every combination is
    unit-tested directly.

    Two claims of very different strength are on offer here, and which one is
    licensed is **per coordinate**:

    * A corner that is *still* rank <= 1 licenses the strong statement -- that
      environment is not a converged fixed point and its energy is a mean-field
      number that will not respond to ``chi``.
    * A corner that was rank <= 1 on an earlier sweep and has since recovered
      licenses only the weak one: the comparison could not be certified, but
      the environment returned has a full-rank corner and is **not** known to
      be bad.

    Applying the strong text to a recovered coordinate invites the caller to
    discard a healthy environment, which is the opposite of this warning's
    purpose -- so a mixed sweep must name each group separately rather than
    picking one diagnosis for all of them.

    Args:
        blind:     Coordinates whose comparison had a rank <= 1 spectrum on
                   either side.  Non-empty whenever this is called.
        collapsed: The subset of ``blind`` still rank <= 1 on the final sweep.

    Returns:
        The warning text.
    """
    parts = [
        "CTM convergence could not be certified: the corner-spectrum "
        "criterion compares spectra normalised by their sum, which a rank-1 "
        "corner forces to [1, 0, ..., 0], so it is structurally blind there. "
        "The full max_iter budget was run instead of exiting early."
    ]
    if collapsed:
        where = ", ".join(str(c) for c in sorted(collapsed))
        parts.append(
            f" The corner at {where} is STILL rank <= 1: that environment is "
            "NOT a converged fixed point, and its energy is a mean-field "
            "number that will not respond to chi."
        )
    recovered = blind - collapsed
    if recovered:
        where = ", ".join(str(c) for c in sorted(recovered))
        parts.append(
            f" The corner at {where} was rank <= 1 on an earlier sweep and has "
            "since recovered to rank > 1: that environment is NOT known to be "
            "bad, only uncertified. Re-run with a larger max_iter to get a "
            "comparison the criterion can read."
        )
    parts.append(" (#898, #723/#726/#747)")
    return "".join(parts)


def _spectrum_can_show_change(
    sv: jax.Array, tol: float = _RANK_TOL, max_rank: int | None = None
) -> bool:
    """Eager ``bool`` form of :func:`_spectrum_is_uninformative`, negated.

    For the loops that want to *report* the blindness rather than merely fail
    closed on it.  ``max_rank`` must match whatever the loop passes to
    :func:`_ctm_sv_diff`, or the warning will name coordinates the criterion
    did not actually refuse.
    """
    return not bool(_spectrum_is_uninformative(sv, tol, max_rank))


def _tensor_leaf_data(leaf):
    """Return numeric buffer for a CTM pytree leaf.

    For ``DenseTensor`` and ``SymmetricTensor``, uses the internal ``_data``
    buffer to avoid dense materialization in elementwise convergence checks.
    """
    if isinstance(leaf, (DenseTensor, SymmetricTensor)):
        return leaf._data
    return leaf.todense() if hasattr(leaf, "todense") else leaf


def _max_env_leaf_diff(env_old: CTMTensorEnv, env_new: CTMTensorEnv) -> float:
    """Maximum absolute element-wise difference across environment leaves."""
    max_diff = 0.0
    for told, tnew in zip(jax.tree.leaves(env_old), jax.tree.leaves(env_new)):
        a = _tensor_leaf_data(told)
        b = _tensor_leaf_data(tnew)
        diff = float(jnp.max(jnp.abs(b - a)))
        max_diff = max(max_diff, diff)
    return max_diff


def _corner_singular_values(C):  # noqa: N802
    """Extract sorted singular values from a 2-leg corner tensor.

    For SymmetricTensor: per-sector SVD, concatenate, sort descending.
    This avoids allocating a full dense chi x chi matrix.
    For DenseTensor / dense array: standard dense SVD.
    """
    if isinstance(C, SymmetricTensor):
        svs = []
        for key in C._block_keys:
            block = C.blocks[key]
            s = _dense_svd(block, compute_uv=False)
            svs.append(s)
        if svs:
            all_svs = jnp.concatenate(svs)
            return jnp.sort(all_svs)[::-1]
        return jnp.zeros(0)
    # DenseTensor or raw array
    data = _tensor_leaf_data(C)
    return _dense_svd(data, compute_uv=False)


def ctm_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    recipe: str = "2x2",
) -> tuple[CTMTensorEnv, float]:
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
        projector_method:  ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
                           Consulted only on the ``"1x1"`` recipe; the ``"2x2"``
                           recipe always uses Fishman SVD via
                           ``_compute_2x2_projector``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.
        recipe:            ``"2x2"`` (default) — the variPEPS-style 2x2
                           plaquette projector, run on a 1-site neighbour map.
                           ``"1x1"`` — the legacy single-site corner-pair
                           projector, kept only for regression bisection.

                           **``"1x1"`` collapses the environment to rank-1
                           corners and must not be used for physics** (#723,
                           #726, #747).  Its projector comes from
                           ``M = C1g^H C4g``, which is ``chi x chi`` — the
                           ``chi * D**2`` seam is summed away, so
                           ``rank(P) <= rank(C1g)``, and the cold ``chi_init=1``
                           seed makes rank-1 an absorbing state.  The symptom is
                           an energy that is bit-identical across a 4x change in
                           chi.  Switching ``projector_method`` does not help:
                           ``eigh``/``qr`` escape the rank collapse but are
                           wildly non-convergent on the same recipe.

    Returns:
        ``(env, max_truncation_error)`` where ``env`` is the converged
        CTMTensorEnv and ``max_truncation_error`` is the maximum per-move
        truncation error ε_T from the **last** sweep before convergence
        (or the last sweep if ``max_iter`` was reached without convergence).
        This is a Python ``float`` suitable for use in the optimizer loop
        (variPEPS §2.8.2 auto-χ trigger).

        **v1 scope caveat:** ``max_truncation_error`` is meaningful only on
        the dense, non-tracer SVD path.  It is ``0.0`` when
        ``projector_method`` is ``"eigh"`` or ``"qr"``, when the input is a
        ``SymmetricTensor`` (block-sparse truncation; global ε_T extraction
        is a v2 follow-up), or when the SVD runs inside a JAX tracer (AD
        backward pass).
    """
    # Determine sweep function: use paired moves for SymmetricTensors
    # with non-trivial virtual charges (fixes charge-sector mismatch
    # from independent projectors in standard 4-move CTM).  When virtual
    # charges are asymmetric (e.g. after simple update truncation),
    # fall back to DenseTensor since the D^2 leg charges change per
    # direction.
    use_paired = False
    if isinstance(A, SymmetricTensor):
        import numpy as _np

        virtual_indices = [A.indices[i] for i in range(4)]
        has_nontrivial = any(
            not (_np.array_equal(vi.sectors, [0]) and vi.n_sectors == 1)
            for vi in virtual_indices
        )
        idx0 = virtual_indices[0]
        all_same = all(
            _np.array_equal(idx0.sectors, virtual_indices[i].sectors)
            and _np.array_equal(idx0.multiplicities, virtual_indices[i].multiplicities)
            for i in range(1, 4)
        )
        if has_nontrivial and all_same:
            use_paired = True
        elif has_nontrivial and not all_same:
            # Asymmetric virtual charges: densify for compatibility
            A = DenseTensor(A.todense(), A.indices)

    if recipe not in ("2x2", "1x1"):
        raise ValueError(f"Unknown recipe={recipe!r}; expected '2x2' or '1x1'.")
    # Validated here rather than inside the projector: the 2x2 recipe never
    # consults projector_method, so an unknown value would otherwise be
    # silently accepted on the default path.
    if projector_method not in ("eigh", "qr", "svd"):
        raise ValueError(
            f"Unknown projector_method={projector_method!r}; "
            f"expected 'eigh', 'qr', or 'svd'."
        )

    if recipe == "2x2":
        # A uniform 1-site lattice is just the multisite path with a
        # self-referential neighbour map, so the 2x2 plaquette projector
        # applies verbatim.  Wrapped to keep the single-site convergence loop,
        # normalisation and ``(env, eps)`` return contract below unchanged.
        def sweep_fn(
            env, a, chi, renormalize, projector_method, *, projector_backward="auto"
        ):
            envs, eps, _smallest_s = _ctm_tensor_sweep_multisite(
                {(0, 0): env},
                {(0, 0): a},
                SINGLE_SITE_NEIGHBORS,
                chi,
                renormalize,
                projector_method,
                projector_backward=projector_backward,
                recipe="2x2",
            )
            return envs[(0, 0)], float(eps)
    else:
        sweep_fn = _ctm_tensor_sweep_paired if use_paired else _ctm_tensor_sweep

    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            env, _ = sweep_fn(
                env, a, chi, renormalize, "eigh", projector_backward=projector_backward
            )
        max_iter = max_iter - warmup

    last_max_eps: float = 0.0
    prev_sv = None
    max_rank = _forced_corner_rank(_max_virtual_bond_dim(a))
    for _ in range(max_iter):
        env, last_max_eps = sweep_fn(
            env,
            a,
            chi,
            renormalize,
            projector_method,
            projector_backward=projector_backward,
        )

        current_sv = _corner_singular_values(env.C1)
        if prev_sv is not None:
            diff = _ctm_sv_diff(current_sv, prev_sv, max_rank=max_rank)
            if float(diff) < conv_tol:
                break
        prev_sv = current_sv

    return env, last_max_eps


def _ctm_tensor_multisite(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    recipe: str = "2x2",
) -> dict[Coord, CTMTensorEnv]:
    """Run multisite CTM to convergence using the Tensor protocol.

    Args:
        site_tensors: Map from coordinate to iPEPS site tensor.
        neighbors:    Map from coordinate to direction→neighbor coordinate.
        chi:          Environment bond dimension.
        max_iter:     Maximum CTM iterations.
        conv_tol:     Convergence tolerance on corner singular values.
        renormalize:  Renormalize environment at each step.
        projector_method: ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
        qr_warmup_steps:  Number of eigh warm-up sweeps before QR kicks in.
        recipe:       ``"2x2"`` (default) uses the variPEPS-style 2x2
                      plaquette projector at every site/direction.  ``"1x1"``
                      falls back to the legacy single-site projector pair.

    Returns:
        Dict mapping coordinates to converged CTMTensorEnv.
    """
    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    envs = {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}

    # #910 review: capture the caller's budget *here*, above the QR warm-up.
    # The warm-up rewrites ``max_iter`` to the remainder
    # (``max_iter = max_iter - warmup``), so reading it after the loop quoted a
    # number the caller never passed -- with ``qr_warmup_steps=6``, a caller
    # passing ``max_iter=40`` was told it "ran the full max_iter=34 sweeps".
    # That matters because the message's own advice is "Raise max_iter", which
    # is unactionable against a value that is not the parameter.  The warm-up
    # sweeps do run, so the caller's total is also the honest sweep count.
    budget = max_iter

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs, _, _ = _ctm_tensor_sweep_multisite(
                envs,
                double_layers,
                neighbors,
                chi,
                renormalize,
                "eigh",
                projector_backward=projector_backward,
                recipe=recipe,
            )
        max_iter = max_iter - warmup

    prev_svs: dict[Coord, jax.Array] = {}
    blind_coords: set[Coord] = set()
    # Per coordinate, not per cell (#903 review).  A cell-wide aggregate is
    # wrong in both directions: `min` lets one trivial site exempt every
    # corner (fails open), and `max` makes a legitimate D=1 coordinate blind
    # on every sweep so the loop can never certify it (fails closed, but
    # wrongly).  The reachable rank is a property of the site sitting at that
    # coordinate, so it is computed there.  Built before the loop and outside
    # every branch.
    max_ranks = {
        c: _forced_corner_rank(_max_virtual_bond_dim(dl))
        for c, dl in double_layers.items()
    }
    # #901: assigned before the loop, not inside it.  Everything below the
    # loop reads them, and a zero-iteration budget would otherwise raise
    # ``UnboundLocalError`` from the reporting path rather than return.
    converged = False
    final_diff = float("inf")
    # #910 review P2: the criterion is a property of a *pair* of spectra, so it
    # does not exist until two sweeps have been compared.  ``sweep_diff`` starts
    # at 0.0 and is only raised by an actual comparison, so on a budget with one
    # measured sweep every ``prev`` is None, nothing is computed, and assigning
    # it anyway made the warning report "final criterion 0" -- a perfectly
    # converged number -- in the same breath as saying conv_tol was not reached.
    # Reachable two ways: ``max_iter=1``, and any QR warm-up that leaves exactly
    # one measured sweep (``max_iter=7, qr_warmup_steps=6``).
    ever_measured = False
    for _ in range(max_iter):
        envs, _, _ = _ctm_tensor_sweep_multisite(
            envs,
            double_layers,
            neighbors,
            chi,
            renormalize,
            projector_method,
            projector_backward=projector_backward,
            recipe=recipe,
        )
        converged = True
        sweep_diff = 0.0
        # Rebuilt each sweep rather than accumulated, and keyed on the whole
        # comparison rather than on the current spectrum alone.
        #
        # Accumulating was wrong because a corner can be blind while the
        # environment is warming up and informative by the time the loop exits;
        # an append-only set keeps naming it, and the warning below would then
        # assert -- in its own text -- that the budget ran out when the loop had
        # in fact exited early.
        #
        # But rebuilding from ``sv`` alone is wrong in the mirror case, and it
        # fails in the direction that matters.  ``_ctm_sv_diff`` reads **both**
        # spectra: a corner blind on sweep ``N-1`` and healthy on sweep ``N``
        # still forces ``inf`` on sweep ``N``, so the loop spends its whole
        # budget uncertified -- while a set built from the current spectrum is
        # empty exactly then, and the caller is told nothing. What must be
        # recorded is whether the comparison the loop actually made was blind,
        # which is a property of the pair.
        blind_coords = set()
        collapsed_coords = set()
        for c in sorted(envs):
            sv = _corner_singular_values(envs[c].C1)
            prev = prev_svs.get(c)
            # ``_ctm_sv_diff`` returns ``inf`` when *either* spectrum has rank
            # <= 1 (#898), so the comparison below already fails closed and no
            # special case is needed for *correctness*.  What is recorded here
            # is only which coordinate's comparison went blind, so the warning
            # after the loop can name it -- the loop is the only place that
            # knows the budget ran out rather than the criterion being
            # satisfied.  Mirror the ``or`` in the criterion exactly: reading
            # one side would leave the other silently uncertified.
            _mr_c = max_ranks[c]
            sv_blind = not _spectrum_can_show_change(sv, max_rank=_mr_c)
            if sv_blind:
                # Still blind *now*: the environment being returned is the
                # collapsed one, so the strong diagnosis below applies.
                collapsed_coords.add(c)
            if sv_blind or (
                prev is not None and not _spectrum_can_show_change(prev, max_rank=_mr_c)
            ):
                blind_coords.add(c)
            if prev is not None:
                d = float(_ctm_sv_diff(sv, prev, max_rank=_mr_c))
                sweep_diff = max(sweep_diff, d)
                ever_measured = True
                if d >= conv_tol:
                    converged = False
            else:
                converged = False
            prev_svs[c] = sv
        # Only overwrite the reported criterion once a comparison has actually
        # produced one; otherwise ``final_diff`` keeps its ``inf``, which is what
        # the zero-measured-sweep case already reports.
        if ever_measured:
            final_diff = sweep_diff
        if converged:
            break

    if blind_coords:
        # Not silent, and not fatal.  The sweeps still ran, so the environment
        # returned here is the best this budget reached.  What the caller must
        # not do is read it as converged.
        warnings.warn(
            _blind_corner_message(blind_coords, collapsed_coords),
            RuntimeWarning,
            stacklevel=2,
        )
    elif not converged:
        # #901.  The blind branch above covers a criterion that could not
        # *see*; this covers one that saw fine and never settled -- full-rank
        # corners on a genuine limit cycle.  That case returned **silently**,
        # which is how a 2-site ``recipe="1x1"`` cycle of ~6e-3 amplitude sat
        # underneath a passing ``|E_qr - E_eigh| < 1e-3`` assertion for weeks:
        # nothing in the call chain said the number was not a fixed point.
        # ``ipeps()`` already warns in exactly this situation, and the wording
        # deliberately mirrors it -- same category, so a caller filtering one
        # filters both.
        criterion = (
            f"final criterion {final_diff:.3g}"
            if ever_measured
            else (
                "the criterion was never evaluated -- fewer than two corner "
                "spectra were compared, so no measurement of it exists"
            )
        )
        warnings.warn(
            f"CTM did not converge in ctm_tensor_multisite(): ran the full "
            f"max_iter={budget} sweeps at chi={chi} without reaching "
            f"conv_tol={conv_tol:g} ({criterion}). The "
            f"returned environment is not a fixed point and any observable "
            f"read from it can move with max_iter -- on a limit cycle it will "
            f"move without ever settling, so raising max_iter is not always a "
            f"fix. Raise max_iter, or switch recipe: the 1x1 recipe does not "
            f"converge on a non-C4v multisite cell (#425/#426/#901).",
            UserWarning,
            stacklevel=2,
        )

    return envs


def ctm_tensor_2site(
    A: Tensor,
    B: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    recipe: str = "2x2",
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
        projector_method: ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
        qr_warmup_steps:  Number of eigh warm-up sweeps before QR kicks in.
        recipe:       ``"2x2"`` (default) or ``"1x1"`` projector recipe;
                      see :func:`_ctm_tensor_sweep_multisite`.

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
        projector_backward=projector_backward,
        recipe=recipe,
    )
    return envs[(0, 0)], envs[(1, 0)]


def ctm_multisite(
    site_tensors: dict[str, Tensor],
    lattice: Lattice,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    recipe: str = "2x2",
) -> dict[str, CTMTensorEnv]:
    """Run multisite CTM to convergence for an arbitrary lattice.

    Translates the string-keyed ``Lattice.neighbor_map`` to the
    coordinate-based format expected by ``_ctm_tensor_multisite()``,
    then maps the results back to site names.

    Use this for unit cells with 3+ sites.  For 1- or 2-site cells,
    prefer ``ctm_tensor()`` or ``ctm_tensor_2site()`` which are
    optimized for those cases.

    Args:
        site_tensors:      ``{site_name: Tensor}`` for each site in
                           ``lattice.sites``.
        lattice:           A :class:`~tenax.core.lattice.Lattice` describing
                           the unit cell geometry.
        chi:               Environment bond dimension.
        max_iter:          Maximum CTM iterations.
        conv_tol:          Convergence tolerance on corner singular values.
        renormalize:       Renormalize environment at each step.
        projector_method:  ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.
        recipe:            ``"2x2"`` (default) — variPEPS-style 2x2 plaquette
                           projector; ``"1x1"`` — legacy single-site projector
                           pair (for backward-compat / regression bisection).

    Returns:
        ``{site_name: CTMTensorEnv}`` — converged environments.
    """
    # Validate: every lattice site must have a corresponding tensor
    missing = set(lattice.sites) - set(site_tensors.keys())
    if missing:
        raise ValueError(
            f"site_tensors is missing sites defined in lattice: {sorted(missing)}"
        )
    extra = set(site_tensors.keys()) - set(lattice.sites)
    if extra:
        raise ValueError(f"site_tensors contains sites not in lattice: {sorted(extra)}")

    # Validate lattice neighbor map topology
    required_dirs = {"left", "right", "top", "bottom"}
    map_missing = set(lattice.sites) - set(lattice.neighbor_map.keys())
    if map_missing:
        raise ValueError(
            f"lattice.neighbor_map is missing sites: {sorted(map_missing)}"
        )
    map_extra = set(lattice.neighbor_map.keys()) - set(lattice.sites)
    if map_extra:
        raise ValueError(
            f"lattice.neighbor_map contains unknown sites: {sorted(map_extra)}"
        )
    for site in lattice.sites:
        neighbors = lattice.neighbor_map[site]
        dirs = set(neighbors.keys())
        missing_dirs = required_dirs - dirs
        extra_dirs = dirs - required_dirs
        if missing_dirs or extra_dirs:
            raise ValueError(
                f"lattice.neighbor_map[{site!r}] has invalid directions: "
                f"missing={sorted(missing_dirs)}, extra={sorted(extra_dirs)}"
            )
        bad_neighbors = {
            direction: nb
            for direction, nb in neighbors.items()
            if nb not in lattice.sites
        }
        if bad_neighbors:
            raise ValueError(
                f"lattice.neighbor_map[{site!r}] has neighbors not in lattice.sites: "
                f"{bad_neighbors}"
            )

    # Map site names to coordinates: site_i -> (i, 0)
    name_to_coord: dict[str, Coord] = {
        name: (i, 0) for i, name in enumerate(lattice.sites)
    }
    coord_to_name: dict[Coord, str] = {v: k for k, v in name_to_coord.items()}

    # Translate site_tensors to coordinate keys
    coord_tensors: dict[Coord, Tensor] = {
        name_to_coord[name]: t for name, t in site_tensors.items()
    }

    # Translate neighbor_map to coordinate keys
    coord_neighbors: dict[Coord, dict[str, Coord]] = {
        name_to_coord[name]: {
            direction: name_to_coord[nb_name]
            for direction, nb_name in neighbors.items()
        }
        for name, neighbors in lattice.neighbor_map.items()
    }

    # Delegate to existing multisite CTM
    coord_envs = _ctm_tensor_multisite(
        coord_tensors,
        coord_neighbors,
        chi,
        max_iter,
        conv_tol,
        renormalize,
        projector_method,
        qr_warmup_steps,
        projector_backward=projector_backward,
        recipe=recipe,
    )

    # Map results back to site names
    return {coord_to_name[c]: env for c, env in coord_envs.items()}
