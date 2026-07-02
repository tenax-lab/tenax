"""Split CTM with Tensor protocol — sweep loop, convergence, and main entry."""

from __future__ import annotations

__all__ = [
    "_SplitCTMInfo",
    "_initialize_split_multisite_env",
    "_renormalize_split_env",
    "_split_ctm_multisite",
    "_split_ctm_sweep_multisite",
    "_split_ctm_tensor_sweep",
    "ctm_split_tensor",
]

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import (
    Coord,
    _corner_singular_values,
    _ctm_sv_diff,
    _sort_coords_for_direction,
)
from tenax.algorithms._split_ctm_tensor_init import (
    SplitCTMTensorEnv,
    initialize_split_ctm_tensor_env,
)
from tenax.algorithms._split_ctm_tensor_moves import (
    _split_ctm_move_bottom,
    _split_ctm_move_left,
    _split_ctm_move_right,
    _split_ctm_move_top,
)
from tenax.algorithms._tensor_utils import max_abs_normalize
from tenax.core import EPS
from tenax.core.tensor import Tensor


class _SplitCTMInfo(NamedTuple):
    """Convergence info for ctm_split_tensor (mirrors the dense path's info)."""

    iterations: int
    converged: bool


# ------------------------------------------------------------------ #
# Sweep + convergence                                                  #
# ------------------------------------------------------------------ #


def _split_ctm_tensor_sweep(
    env: SplitCTMTensorEnv,
    A: Tensor,
    chi: int,
    chi_I: int,
    renormalize: bool,
) -> SplitCTMTensorEnv:
    """One full split-CTM sweep: L/R/T/B moves + optional renormalize."""
    A_bar = A.bar()
    # variPEPS sweep order (L/T/R/B), matching the fused ``_ctm_tensor_sweep``
    # so the split path tracks the same fixed point as the oracle.
    env = _split_ctm_move_left(env, A, A_bar, chi, chi_I)
    env = _split_ctm_move_top(env, A, A_bar, chi, chi_I)
    env = _split_ctm_move_right(env, A, A_bar, chi, chi_I)
    env = _split_ctm_move_bottom(env, A, A_bar, chi, chi_I)

    if renormalize:
        env = _renormalize_split_env(env)

    return env


def _renormalize_split_env(env: SplitCTMTensorEnv) -> SplitCTMTensorEnv:
    """Renormalize all 12 tensors in a SplitCTMTensorEnv."""
    C1, _ = max_abs_normalize(env.C1)
    C2, _ = max_abs_normalize(env.C2)
    C3, _ = max_abs_normalize(env.C3)
    C4, _ = max_abs_normalize(env.C4)

    def normalize_pair(T_ket: Tensor, T_bra: Tensor) -> tuple[Tensor, Tensor]:
        nk = T_ket.max_abs()
        nb = T_bra.max_abs()
        shared = jnp.sqrt(nk * nb) + EPS
        return T_ket * (1.0 / shared), T_bra * (1.0 / shared)

    T1k, T1b = normalize_pair(env.T1_ket, env.T1_bra)
    T2k, T2b = normalize_pair(env.T2_ket, env.T2_bra)
    T3k, T3b = normalize_pair(env.T3_ket, env.T3_bra)
    T4k, T4b = normalize_pair(env.T4_ket, env.T4_bra)

    return SplitCTMTensorEnv(
        C1=C1,
        C2=C2,
        C3=C3,
        C4=C4,
        T1_ket=T1k,
        T1_bra=T1b,
        T2_ket=T2k,
        T2_bra=T2b,
        T3_ket=T3k,
        T3_bra=T3b,
        T4_ket=T4k,
        T4_bra=T4b,
    )


def ctm_split_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    min_iter: int = 2,
    return_info: bool = False,
) -> SplitCTMTensorEnv | tuple[SplitCTMTensorEnv, _SplitCTMInfo]:
    """Run split-CTM to convergence using the Tensor protocol.

    Convergence is measured on the corner (``C1``) singular-value spectrum
    via the same :func:`_corner_singular_values` / :func:`_ctm_sv_diff`
    helpers the fused :func:`ctm_tensor` uses, so the split path tracks the
    same fixed point as the oracle.

    .. note::
        The corner singular-value criterion can **plateau** for degenerate
        or low-rank corners — e.g. the boundary of a 1-site uniform iPEPS is
        genuinely rank-1/2, so the *normalized* corner spectrum is constant
        from the first sweep even while the environment (and energy) are
        still relaxing toward the fixed point. The ``min_iter`` floor stops
        the loop from breaking on that initial transient, but it cannot
        detect convergence beyond it. For exact convergence studies on such
        inputs (e.g. the lossless-``chi_I`` parity oracle), pass
        ``conv_tol=0.0`` with a sufficiently large ``max_iter`` to force a
        fixed number of full sweeps instead of relying on the spectral break.

    Args:
        A:          iPEPS site tensor (DenseTensor or SymmetricTensor) with
                    5 legs ``(u, d, l, r, phys)``.
        chi:        Environment bond dimension.
        max_iter:   Maximum number of CTM iterations.
        conv_tol:   Convergence tolerance on corner singular values. Use
                    ``0.0`` to disable the early break (run all ``max_iter``
                    sweeps).
        chi_I:      Interlayer bond dimension. Defaults to ``chi``.
        renormalize: Renormalize environment at each step.
        min_iter:   Minimum number of sweeps before the ``conv_tol`` early
                    break may fire. Guards against a premature break on the
                    initial transient plateau of a degenerate corner.
                    Effectively floored at 2: the first sweep has no previous
                    spectrum to compare against, so the earliest possible break
                    is the second sweep regardless of ``min_iter``.
        return_info: If True, return ``(env, _SplitCTMInfo(iterations, converged))``
                     instead of just ``env``.

    Returns:
        Converged SplitCTMTensorEnv.
    """
    if chi_I is None:
        chi_I = chi

    env = initialize_split_ctm_tensor_env(A, chi, chi_I)

    prev_sv = None
    converged = False
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)

        current_sv = _corner_singular_values(env.C1)
        if prev_sv is not None and iteration + 1 >= min_iter:
            diff = _ctm_sv_diff(current_sv, prev_sv)
            if float(diff) < conv_tol:
                converged = True
                break
        prev_sv = current_sv

    if return_info:
        return env, _SplitCTMInfo(iterations=iterations, converged=converged)
    return env


# ------------------------------------------------------------------ #
# Multisite env init                                                   #
# ------------------------------------------------------------------ #


def _initialize_split_multisite_env(
    site_tensors: dict[Coord, Tensor],
    chi: int,
    chi_I: int,
) -> dict[Coord, SplitCTMTensorEnv]:
    """Per-coord split env init: reuse the single-site builder per site."""
    return {
        c: initialize_split_ctm_tensor_env(A, chi, chi_I)
        for c, A in site_tensors.items()
    }


# ------------------------------------------------------------------ #
# Direction-move dispatch table                                        #
# ------------------------------------------------------------------ #

_SPLIT_DIRECTION_MOVES = {
    "left": _split_ctm_move_left,
    "top": _split_ctm_move_top,
    "right": _split_ctm_move_right,
    "bottom": _split_ctm_move_bottom,
}


# ------------------------------------------------------------------ #
# Multisite sweep + driver                                             #
# ------------------------------------------------------------------ #


def _split_ctm_sweep_multisite_2x2(
    envs: dict[Coord, SplitCTMTensorEnv],
    site_tensors: dict[Coord, Tensor],
    bars: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    chi_I: int,
) -> dict[Coord, SplitCTMTensorEnv]:
    """One 2x2-recipe split sweep — twin of the fused 2x2 branch of
    :func:`_ctm_tensor_sweep_multisite`.

    Two phases per direction: Phase 1 builds the split plaquette projector
    pair anchored at every cell (from the pre-sweep ``envs_old`` snapshot);
    Phase 2 absorbs the neighbour column/row into each ``s_dst`` and replaces
    the destination env's corner + ket/bra edge fields.  Neighbour/anchor
    lookups and the cascade order mirror the fused sweep exactly.
    """
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_ctm_absorb_bottom_2plaq,
        _split_ctm_absorb_left_2plaq,
        _split_ctm_absorb_right_2plaq,
        _split_ctm_absorb_top_2plaq,
    )

    all_coords = list(envs.keys())
    for direction in ("left", "top", "right", "bottom"):
        envs_old = dict(envs)
        # Phase 1: projector pair anchored at every cell (from the snapshot).
        projectors: dict[Coord, tuple] = {}
        for s in all_coords:
            s_TR = neighbors[s]["right"]
            s_BL = neighbors[s]["bottom"]
            s_BR = neighbors[s_TR]["bottom"]
            Pt, Pb, _eps, _sS = _compute_split_plaquette_projector_pair(
                envs_old[s],
                envs_old[s_TR],
                envs_old[s_BL],
                envs_old[s_BR],
                site_tensors[s],
                bars[s],
                site_tensors[s_TR],
                bars[s_TR],
                site_tensors[s_BL],
                bars[s_BL],
                site_tensors[s_BR],
                bars[s_BR],
                chi,
                direction,
                base_charges=None,
            )
            projectors[s] = (Pt, Pb)
        # Phase 2: absorb per destination cell using two plaquettes' halves.
        new_envs: dict[Coord, SplitCTMTensorEnv] = {}
        for s_dst in _sort_coords_for_direction(all_coords, direction):
            if direction == "left":
                s_src = neighbors[s_dst]["left"]
                s_a = neighbors[s_src]["top"]
                Pta, Pba = projectors[s_a]
                Ptc, Pbc = projectors[s_src]
                C1n, T4k, T4b, C4n = _split_ctm_absorb_left_2plaq(
                    envs_old[s_src],
                    site_tensors[s_src],
                    bars[s_src],
                    Pta,
                    Pba,
                    Ptc,
                    Pbc,
                    chi_I,
                )
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C1=C1n, T4_ket=T4k, T4_bra=T4b, C4=C4n
                )
            elif direction == "right":
                s_src = neighbors[s_dst]["right"]
                s_a = neighbors[s_dst]["top"]
                Pta, Pba = projectors[s_a]
                Ptc, Pbc = projectors[s_dst]
                C2n, T2k, T2b, C3n = _split_ctm_absorb_right_2plaq(
                    envs_old[s_src],
                    site_tensors[s_src],
                    bars[s_src],
                    Pta,
                    Pba,
                    Ptc,
                    Pbc,
                    chi_I,
                )
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C2=C2n, T2_ket=T2k, T2_bra=T2b, C3=C3n
                )
            elif direction == "top":
                s_src = neighbors[s_dst]["top"]
                s_a = neighbors[s_src]["left"]
                Ptl, Pbl = projectors[s_a]
                Ptc, Pbc = projectors[s_src]
                C1n, T1k, T1b, C2n = _split_ctm_absorb_top_2plaq(
                    envs_old[s_src],
                    site_tensors[s_src],
                    bars[s_src],
                    Ptl,
                    Pbl,
                    Ptc,
                    Pbc,
                    chi_I,
                )
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C1=C1n, T1_ket=T1k, T1_bra=T1b, C2=C2n
                )
            else:  # bottom
                s_src = neighbors[s_dst]["bottom"]
                s_a = neighbors[s_dst]["left"]
                Ptl, Pbl = projectors[s_a]
                Ptc, Pbc = projectors[s_dst]
                C4n, T3k, T3b, C3n = _split_ctm_absorb_bottom_2plaq(
                    envs_old[s_src],
                    site_tensors[s_src],
                    bars[s_src],
                    Ptl,
                    Pbl,
                    Ptc,
                    Pbc,
                    chi_I,
                )
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C4=C4n, T3_ket=T3k, T3_bra=T3b, C3=C3n
                )
        envs = new_envs
    return envs


def _split_ctm_sweep_multisite(
    envs: dict[Coord, SplitCTMTensorEnv],
    site_tensors: dict[Coord, Tensor],
    bars: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    chi_I: int,
    renormalize: bool,
    recipe: str = "2x2",
) -> dict[Coord, SplitCTMTensorEnv]:
    """One full split multisite CTM sweep.

    Args:
        envs:        Per-coord split CTM environments.
        site_tensors: Per-coord site tensors.
        bars:        Per-coord conjugate (bar) tensors.
        neighbors:   Per-coord directional neighbor map.
        chi:         Corner bond dimension.
        chi_I:       Interlayer bond dimension.
        renormalize: If True, renormalize each environment after the sweep.
        recipe:      ``'1x1'`` reuses single-site directional moves;
                     ``'2x2'`` applies genuine 2×2 plaquette projectors
                     (not yet implemented — lands in Task 1.3).

    Returns:
        Updated per-coord environments.
    """
    envs = dict(envs)
    all_coords = list(envs.keys())
    if recipe == "1x1":
        for direction in ("left", "top", "right", "bottom"):
            move_fn = _SPLIT_DIRECTION_MOVES[direction]
            for coord in _sort_coords_for_direction(all_coords, direction):
                nb = neighbors[coord][direction]
                envs[coord] = move_fn(
                    envs[coord], site_tensors[nb], bars[nb], chi, chi_I
                )
    elif recipe == "2x2":
        envs = _split_ctm_sweep_multisite_2x2(
            envs, site_tensors, bars, neighbors, chi, chi_I
        )
    else:
        raise ValueError(
            f"Unknown split CTM recipe {recipe!r}: expected '1x1' or '2x2'."
        )
    if renormalize:
        envs = {c: _renormalize_split_env(e) for c, e in envs.items()}
    return envs


def _split_ctm_multisite(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    recipe: str = "2x2",
) -> dict[Coord, SplitCTMTensorEnv]:
    """Run split multisite CTM to convergence (mirrors ``_ctm_tensor_multisite``).

    Args:
        site_tensors: Per-coord iPEPS site tensors.
        neighbors:   Per-coord directional neighbor map (e.g. ``CHECKERBOARD_NEIGHBORS``).
        chi:         Corner bond dimension.
        max_iter:    Maximum number of CTM sweep iterations.
        conv_tol:    Convergence tolerance on corner singular values. Use
                     ``0.0`` to disable early stopping and run all ``max_iter``
                     sweeps.
        chi_I:       Interlayer bond dimension. Defaults to ``chi``.
        renormalize: Renormalize environment tensors after each sweep.
        recipe:      ``'1x1'`` or ``'2x2'`` (see :func:`_split_ctm_sweep_multisite`).

    Returns:
        Per-coord converged :class:`SplitCTMTensorEnv` dict.
    """
    if chi_I is None:
        chi_I = chi
    bars = {c: A.bar() for c, A in site_tensors.items()}
    envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    prev_svs: dict[Coord, jax.Array] = {}
    for _ in range(max_iter):
        envs = _split_ctm_sweep_multisite(
            envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, recipe
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
