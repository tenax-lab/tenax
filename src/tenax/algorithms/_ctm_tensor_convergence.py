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

    Returns:
        ``(env, max_eps)`` where ``max_eps`` is the maximum per-move truncation
        error across the four directional moves in this sweep.
    """
    env, eps_left = _ctm_tensor_move_left(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env, eps_top = _ctm_tensor_move_top(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env, eps_right = _ctm_tensor_move_right(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env, eps_bottom = _ctm_tensor_move_bottom(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    if renormalize:
        env = _renormalize_tensor_env(env)
    max_eps = max(eps_left, eps_top, eps_right, eps_bottom)
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
) -> tuple[dict[Coord, CTMTensorEnv], jax.Array]:
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
        ``(envs, max_eps)`` — updated per-coord env dict and the max
        truncation error (JAX scalar) across all moves in this sweep.
        The 2x2 path returns ``jnp.asarray(0.0)`` as a placeholder.
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
    if recipe == "1x1":
        for direction, move_fn in _DIRECTION_MOVES:
            for coord in _sort_coords_for_direction(all_coords, direction):
                nb = neighbors[coord][direction]
                envs[coord], eps_t = move_fn(
                    envs[coord],
                    envs[nb],
                    double_layers[nb],
                    chi,
                    projector_method,
                    base_charges=base_charges,
                    projector_backward=projector_backward,
                )
                max_eps = jnp.maximum(max_eps, jnp.asarray(eps_t))
    elif recipe == "2x2":
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
            projectors: dict[Coord, tuple] = {}
            for s_anchor in all_coords:
                s_TR = neighbors[s_anchor]["right"]
                s_BL = neighbors[s_anchor]["bottom"]
                s_BR = neighbors[s_TR]["bottom"]
                projectors[s_anchor] = _compute_plaquette_projector_pair(
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
                        double_layers[s_src],
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
                        double_layers[s_src],
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
                        double_layers[s_src],
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
                        double_layers[s_src],
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
    return envs, max_eps


# ------------------------------------------------------------------ #
# Main entry: convergence loop                                         #
# ------------------------------------------------------------------ #


def _ctm_sv_diff(sv_new: jax.Array, sv_old: jax.Array) -> jax.Array:
    """Compute max absolute difference between normalized singular value vectors."""
    sv1 = sv_new / (jnp.sum(sv_new) + 1e-15)
    sv2 = sv_old / (jnp.sum(sv_old) + 1e-15)
    return jnp.max(jnp.abs(sv1 - sv2))


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
            s = jnp.linalg.svd(block, compute_uv=False)
            svs.append(s)
        if svs:
            all_svs = jnp.concatenate(svs)
            return jnp.sort(all_svs)[::-1]
        return jnp.zeros(0)
    # DenseTensor or raw array
    data = _tensor_leaf_data(C)
    return jnp.linalg.svd(data, compute_uv=False)


def ctm_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
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
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.

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
            diff = _ctm_sv_diff(current_sv, prev_sv)
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

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs, _ = _ctm_tensor_sweep_multisite(
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
    for _ in range(max_iter):
        envs, _ = _ctm_tensor_sweep_multisite(
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
        for c in sorted(envs):
            sv = _corner_singular_values(envs[c].C1)
            if c in prev_svs:
                if float(_ctm_sv_diff(sv, prev_svs[c])) >= conv_tol:
                    converged = False
            else:
                converged = False
            prev_svs[c] = sv
        if converged:
            break

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
