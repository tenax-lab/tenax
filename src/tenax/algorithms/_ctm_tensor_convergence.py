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
from tenax.core import EPS
from tenax.core.lattice import Lattice
from tenax.core.tensor import Tensor

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
) -> CTMTensorEnv:
    """One full CTM sweep: left, right, top, bottom + optional renormalize."""
    env = _ctm_tensor_move_left(env, env, a, chi, projector_method)
    env = _ctm_tensor_move_right(env, env, a, chi, projector_method)
    env = _ctm_tensor_move_top(env, env, a, chi, projector_method)
    env = _ctm_tensor_move_bottom(env, env, a, chi, projector_method)
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


def _ctm_tensor_sweep_multisite(
    envs: dict[Coord, CTMTensorEnv],
    double_layers: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    renormalize: bool,
    projector_method: str = "eigh",
) -> dict[Coord, CTMTensorEnv]:
    """One full multisite CTM sweep over all sites and directions."""
    for direction, move_fn in _DIRECTION_MOVES:
        for coord in sorted(envs.keys()):
            nb = neighbors[coord][direction]
            envs[coord] = move_fn(
                envs[coord], envs[nb], double_layers[nb], chi, projector_method
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
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    # QR warm-up: run a few eigh iterations before switching to QR
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            env = _ctm_tensor_sweep(env, a, chi, renormalize, "eigh")
        max_iter = max_iter - warmup

    prev_sv = None
    for _ in range(max_iter):
        env = _ctm_tensor_sweep(env, a, chi, renormalize, projector_method)

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
                envs, double_layers, neighbors, chi, renormalize, "eigh"
            )
        max_iter = max_iter - warmup

    prev_svs: dict[Coord, jax.Array] = {}
    for _ in range(max_iter):
        envs = _ctm_tensor_sweep_multisite(
            envs, double_layers, neighbors, chi, renormalize, projector_method
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
    )

    # Map results back to site names
    return {coord_to_name[c]: env for c, env in coord_envs.items()}
