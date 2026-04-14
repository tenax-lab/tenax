"""Standard CTM with Tensor protocol — sweep loop, convergence, and main entry points."""

from __future__ import annotations

__all__ = [
    "CHECKERBOARD_NEIGHBORS",
    "Coord",
    "SINGLE_SITE_NEIGHBORS",
    "_DIRECTION_MOVES",
    "_ctm_sv_diff",
    "_ctm_tensor_multisite",
    "_ctm_tensor_sweep",
    "_ctm_tensor_sweep_multisite",
    "_ctm_tensor_sweep_paired",
    "_normalize_tensor",
    "_renormalize_tensor_env",
    "ctm_multisite",
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
    _ctm_tensor_move_bottom,
    _ctm_tensor_move_left,
    _ctm_tensor_move_right,
    _ctm_tensor_move_top,
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
    projector_method: str = "eigh",
    projector_backward: str = "auto",
) -> CTMTensorEnv:
    """One full CTM sweep: left, right, top, bottom + optional renormalize."""
    env = _ctm_tensor_move_left(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env = _ctm_tensor_move_right(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env = _ctm_tensor_move_top(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env = _ctm_tensor_move_bottom(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    if renormalize:
        env = _renormalize_tensor_env(env)
    return env


def _ctm_tensor_sweep_paired(
    env: CTMTensorEnv,
    a: Tensor,
    chi: int,
    renormalize: bool,
    projector_method: str = "eigh",
    projector_backward: str = "auto",
) -> CTMTensorEnv:
    """One full CTM sweep using paired moves: horizontal then vertical.

    Uses 2x2 enlarged corners for projector computation, ensuring
    consistent charge-sector distributions across sweeps for
    SymmetricTensor inputs.
    """
    env = _ctm_tensor_move_horizontal(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    env = _ctm_tensor_move_vertical(
        env, env, a, chi, projector_method, projector_backward=projector_backward
    )
    if renormalize:
        env = _renormalize_tensor_env(env)
    return env


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

_DIRECTION_MOVES = [
    ("left", _ctm_tensor_move_left),
    ("right", _ctm_tensor_move_right),
    ("top", _ctm_tensor_move_top),
    ("bottom", _ctm_tensor_move_bottom),
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
    projector_method: str = "eigh",
    projector_backward: str = "auto",
) -> dict[Coord, CTMTensorEnv]:
    """One full multisite CTM sweep over all sites and directions."""
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
    for direction, move_fn in _DIRECTION_MOVES:
        for coord in _sort_coords_for_direction(all_coords, direction):
            nb = neighbors[coord][direction]
            envs[coord] = move_fn(
                envs[coord],
                envs[nb],
                double_layers[nb],
                chi,
                projector_method,
                base_charges=base_charges,
                projector_backward=projector_backward,
            )
    if renormalize:
        envs = {c: _renormalize_tensor_env(e) for c, e in envs.items()}
    return envs


# ------------------------------------------------------------------ #
# Main entry: convergence loop                                         #
# ------------------------------------------------------------------ #


def _ctm_sv_diff(sv_new: jax.Array, sv_old: jax.Array) -> jax.Array:
    """Compute max absolute difference between normalized singular value vectors."""
    sv1 = sv_new / (jnp.sum(sv_new) + 1e-15)
    sv2 = sv_old / (jnp.sum(sv_old) + 1e-15)
    return jnp.max(jnp.abs(sv1 - sv2))


def ctm_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
) -> CTMTensorEnv:
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
        projector_method:  ``"eigh"`` or ``"qr"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.

    Returns:
        Converged CTMTensorEnv.
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
            env = sweep_fn(
                env, a, chi, renormalize, "eigh", projector_backward=projector_backward
            )
        max_iter = max_iter - warmup

    prev_sv = None
    for _ in range(max_iter):
        env = sweep_fn(
            env,
            a,
            chi,
            renormalize,
            projector_method,
            projector_backward=projector_backward,
        )

        current_sv = jnp.linalg.svd(env.C1.todense(), compute_uv=False)
        if prev_sv is not None:
            diff = _ctm_sv_diff(current_sv, prev_sv)
            if float(diff) < conv_tol:
                break
        prev_sv = current_sv

    return env


def _ctm_tensor_multisite(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
) -> dict[Coord, CTMTensorEnv]:
    """Run multisite CTM to convergence using the Tensor protocol.

    Args:
        site_tensors: Map from coordinate to iPEPS site tensor.
        neighbors:    Map from coordinate to direction→neighbor coordinate.
        chi:          Environment bond dimension.
        max_iter:     Maximum CTM iterations.
        conv_tol:     Convergence tolerance on corner singular values.
        renormalize:  Renormalize environment at each step.
        projector_method: ``"eigh"`` or ``"qr"``.
        qr_warmup_steps:  Number of eigh warm-up sweeps before QR kicks in.

    Returns:
        Dict mapping coordinates to converged CTMTensorEnv.
    """
    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    envs = {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs = _ctm_tensor_sweep_multisite(
                envs,
                double_layers,
                neighbors,
                chi,
                renormalize,
                "eigh",
                projector_backward=projector_backward,
            )
        max_iter = max_iter - warmup

    prev_svs: dict[Coord, jax.Array] = {}
    for _ in range(max_iter):
        envs = _ctm_tensor_sweep_multisite(
            envs,
            double_layers,
            neighbors,
            chi,
            renormalize,
            projector_method,
            projector_backward=projector_backward,
        )
        converged = True
        for c in sorted(envs):
            sv = jnp.linalg.svd(envs[c].C1.todense(), compute_uv=False)
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
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
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
        projector_method: ``"eigh"`` or ``"qr"``.
        qr_warmup_steps:  Number of eigh warm-up sweeps before QR kicks in.

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
    )
    return envs[(0, 0)], envs[(1, 0)]


def ctm_multisite(
    site_tensors: dict[str, Tensor],
    lattice: Lattice,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
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
        projector_method:  ``"eigh"`` or ``"qr"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.

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
    )

    # Map results back to site names
    return {coord_to_name[c]: env for c, env in coord_envs.items()}
