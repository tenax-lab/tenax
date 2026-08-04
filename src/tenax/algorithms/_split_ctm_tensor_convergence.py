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
    "ctm_split_tensor_2site",
]

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
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
    recipe: str = "2x2",
) -> SplitCTMTensorEnv | tuple[SplitCTMTensorEnv, _SplitCTMInfo]:
    """Run split-CTM to convergence using the Tensor protocol.

    Convergence is measured on the corner (``C1``) singular-value spectrum
    via the same :func:`_corner_singular_values` / :func:`_ctm_sv_diff`
    helpers the fused :func:`ctm_tensor` uses, so the split path tracks the
    same fixed point as the oracle.

    .. note::
        The corner singular-value criterion can **plateau** for degenerate
        or low-rank corners, so the *normalized* corner spectrum can be
        constant from the first sweep even while the environment (and energy)
        are still relaxing toward the fixed point. The ``min_iter`` floor
        stops the loop from breaking on that initial transient, but it cannot
        detect convergence beyond it. For exact convergence studies (e.g. the
        lossless-``chi_I`` parity oracle), pass ``conv_tol=0.0`` with a
        sufficiently large ``max_iter`` to force a fixed number of full
        sweeps instead of relying on the spectral break.

        This note used to claim the boundary of a 1-site uniform iPEPS "is
        genuinely rank-1/2". That was the ``1x1`` collapse (#726/#746) being
        read as a property of the physics; on the ``2x2`` default the same
        state gives a rank-6 corner with spectrum
        ``1, 0.128, 0.127, 0.016, 2.1e-3, 2.0e-3``.

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
        recipe:      ``"2x2"`` (default) — the variPEPS-style 2x2 plaquette
                     projector, run on a 1-site neighbour map.  ``"1x1"`` —
                     the legacy single-site corner-pair moves, kept only for
                     regression bisection.

                     **``"1x1"`` collapses the environment to rank-1 corners
                     and must not be used for physics** (#723, #726, #746,
                     #747).  Its projector comes from ``M = C1g^H C4g``, which
                     is ``chi x chi`` — the ``chi * D**2`` seam is summed away,
                     so ``rank(P) <= rank(C1g)``, and the cold rank-1 seed
                     makes rank-1 an absorbing state.  The symptom is an energy
                     bit-identical across a 4x change in chi.  This is the
                     *same* projector the fused path used before #723; it is
                     shared verbatim, so this was never a split-CTM defect.

    Returns:
        Converged SplitCTMTensorEnv.
    """
    if chi_I is None:
        chi_I = chi
    if recipe not in ("2x2", "1x1"):
        raise ValueError(
            f"Unknown split CTM recipe {recipe!r}: expected '1x1' or '2x2'."
        )

    env = initialize_split_ctm_tensor_env(A, chi, chi_I)
    # A uniform 1-site lattice is just the multisite path with a
    # self-referential neighbour map, so the working 2x2 plaquette projector
    # applies verbatim.  Wrapped to keep the convergence loop, ``min_iter``
    # floor and ``(env, info)`` return contract below unchanged.
    bar = A.bar() if recipe == "2x2" else None

    prev_sv = None
    converged = False
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        if recipe == "2x2":
            env = _split_ctm_sweep_multisite(
                {(0, 0): env},
                {(0, 0): A},
                {(0, 0): bar},
                SINGLE_SITE_NEIGHBORS,
                chi,
                chi_I,
                renormalize,
                recipe="2x2",
            )[(0, 0)]
        else:
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
# Fermionic (graded) routing — #463 Phase 4                            #
# ------------------------------------------------------------------ #
#
# The split 2×2 grow/projector kernels build double-layer corners and edges
# from graded ``A.bar()`` alone; they do NOT carry the order-dependent Koszul
# signs that the *fused* 2-site path applies via its ``*_2plaq_fused`` variants
# (gated by ``_env_is_fermionic``).  A raw fermionic split sweep therefore
# diverges from the proven fused sweep — the corner GROW alone is already wrong
# (#641).  Until a bounded (χ²·D⁴) Koszul-correct split kernel exists, fermionic
# envs route through the fused sweep on the merged (χ²·D⁶) representation, then
# the output edges are re-split back into ket/bra halves for storage.  The merge
# (``_split_env_to_tensor_standard``) is a faithful inverse of the split, so this
# is exact — see ``test_split_ctm_2site_fermionic``.


def _split_env_is_fermionic(env: SplitCTMTensorEnv) -> bool:
    """True when the split env tensors are graded (fermionic) SymmetricTensors."""
    from tenax.core.tensor import SymmetricTensor

    C1 = env.C1
    return isinstance(C1, SymmetricTensor) and C1.indices[0].symmetry.is_fermionic


# Per-edge spec to re-split a fused ``CTMTensorEnv`` edge into ket/bra halves.
# Inverse of ``_split_env_to_tensor_standard._merge_edge``: the fused D² leg is
# split back into its two parents (``d_ket``/``d_bra``), then an SVD over the
# interlayer bond re-creates the ket ``(chi, D, chi_I)`` and bra ``(chi_I, D,
# chi)`` halves with the standard split labels.
_RESPLIT_SPEC = {
    "T1": dict(
        d2="u2",
        left=("t1_l", "u_ket"),
        right=("u_bra", "t1_r"),
        ket_relabels={"t1_l": "t1k_l", "_svd_bond": "t1k_I"},
        bra_relabels={"_svd_bond": "t1b_I", "t1_r": "t1b_r"},
    ),
    "T2": dict(
        d2="r2",
        left=("t2_u", "r_ket"),
        right=("r_bra", "t2_d"),
        ket_relabels={"t2_u": "t2k_u", "_svd_bond": "t2k_I"},
        bra_relabels={"_svd_bond": "t2b_I", "t2_d": "t2b_d"},
    ),
    "T3": dict(
        d2="d2",
        left=("t3_r", "d_ket"),
        right=("d_bra", "t3_l"),
        ket_relabels={"t3_r": "t3k_r", "_svd_bond": "t3k_I"},
        bra_relabels={"_svd_bond": "t3b_I", "t3_l": "t3b_l"},
    ),
    "T4": dict(
        d2="l2",
        left=("t4_d", "l_ket"),
        right=("l_bra", "t4_u"),
        ket_relabels={"t4_d": "t4k_d", "_svd_bond": "t4k_I"},
        bra_relabels={"_svd_bond": "t4b_I", "t4_u": "t4b_u"},
    ),
}


def _tensor_env_to_split_standard(
    env, site_tensor: Tensor, chi_I: int
) -> SplitCTMTensorEnv:
    """Convert a fused ``CTMTensorEnv`` back to a ``SplitCTMTensorEnv``.

    Inverse of ``_split_env_to_tensor_standard``: corners pass through
    unchanged; each edge's fused D² leg is split into its two parents and an
    SVD over the interlayer bond re-creates the ket/bra halves (truncated to
    *chi_I*).

    The fused edges produced by the fused CTM sweep carry no ``fuse_info`` on
    their D² leg (it was built by projector application, not ``fuse_indices``),
    so ``split_index`` cannot act on them directly.  Because the D² index
    depends only on the site tensor (not the env), we transplant the
    ``fuse_info``-bearing D² index from a freshly-initialised reference env of
    the same site (sectors/multiplicities/flow are identical), then split.
    """
    from tenax.algorithms._split_ctm_tensor_energy import (
        _split_env_to_tensor_standard,
    )
    from tenax.algorithms._split_ctm_tensor_moves import (
        _split_base_charges,
        _svd_split_edge_tensor,
    )
    from tenax.algorithms._tensor_utils import split_index
    from tenax.core.tensor import SymmetricTensor

    chi = env.C1.indices[0].dim
    ref = _split_env_to_tensor_standard(
        initialize_split_ctm_tensor_env(site_tensor, chi, chi_I)
    )
    base_charges = _split_base_charges(site_tensor)

    halves = {}
    for name, spec in _RESPLIT_SPEC.items():
        edge = getattr(env, name)
        ref_edge = getattr(ref, name)
        d2 = spec["d2"]
        ax = edge.labels().index(d2)
        ref_d2 = ref_edge.indices[ref_edge.labels().index(d2)]
        # Transplant the fuse_info-bearing D² index (shares block data).
        new_indices = list(edge._indices)
        new_indices[ax] = ref_d2.relabel(d2)
        edge = SymmetricTensor._raw(
            indices=tuple(new_indices),
            data=edge._data,
            block_keys=edge._block_keys,
            block_shapes=edge._block_shapes,
            block_offsets=edge._block_offsets,
        )
        edge = split_index(edge, ax)
        halves[name] = _svd_split_edge_tensor(
            edge,
            left_labels=list(spec["left"]),
            right_labels=list(spec["right"]),
            chi_I=chi_I,
            ket_relabels=spec["ket_relabels"],
            bra_relabels=spec["bra_relabels"],
            base_charges=base_charges,
        )

    return SplitCTMTensorEnv(
        C1=env.C1,
        C2=env.C2,
        C3=env.C3,
        C4=env.C4,
        T1_ket=halves["T1"][0],
        T1_bra=halves["T1"][1],
        T2_ket=halves["T2"][0],
        T2_bra=halves["T2"][1],
        T3_ket=halves["T3"][0],
        T3_bra=halves["T3"][1],
        T4_ket=halves["T4"][0],
        T4_bra=halves["T4"][1],
    )


def _split_ctm_sweep_multisite_2x2_via_fused(
    envs: dict[Coord, SplitCTMTensorEnv],
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    chi_I: int,
    renormalize: bool = True,
) -> dict[Coord, SplitCTMTensorEnv]:
    """One fermionic 2×2 split sweep via ``merge → fused sweep → resplit``.

    ``renormalize`` is honored on the merged (fused) representation so a caller
    disabling per-sweep normalization gets the same raw fixed-point map as the
    dense/bosonic split path — it is *not* hard-coded on (#690).
    """
    from tenax.algorithms._ctm_tensor_convergence import (
        _ctm_tensor_sweep_multisite,
    )
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
    from tenax.algorithms._split_ctm_tensor_energy import (
        _split_env_to_tensor_standard,
    )

    fused_envs = {c: _split_env_to_tensor_standard(e) for c, e in envs.items()}
    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    new_fused, _eps, _sS = _ctm_tensor_sweep_multisite(
        fused_envs, double_layers, neighbors, chi, renormalize, recipe="2x2"
    )
    return {
        c: _tensor_env_to_split_standard(fe, site_tensors[c], chi_I)
        for c, fe in new_fused.items()
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
    renormalize: bool = True,
) -> dict[Coord, SplitCTMTensorEnv]:
    """One 2x2-recipe split sweep — twin of the fused 2x2 branch of
    :func:`_ctm_tensor_sweep_multisite`.

    Two phases per direction: Phase 1 builds the split plaquette projector
    pair anchored at every cell (from the pre-sweep ``envs_old`` snapshot);
    Phase 2 absorbs the neighbour column/row into each ``s_dst`` and replaces
    the destination env's corner + ket/bra edge fields.  Neighbour/anchor
    lookups and the cascade order mirror the fused sweep exactly.

    Fermionic (graded) envs take the ``merge → fused sweep → resplit`` route
    (see ``_split_ctm_sweep_multisite_2x2_via_fused``), since the split
    double-layer kernels below do not yet carry the order-dependent Koszul
    signs (#463 Phase 4 / #641).

    ``renormalize`` is forwarded to the fermionic fused route so its internal
    normalization matches the caller's flag (#690).  The bosonic branch does not
    normalize here; :func:`_split_ctm_sweep_multisite` applies the post-sweep
    ``renormalize`` for it.
    """
    if _split_env_is_fermionic(next(iter(envs.values()))):
        return _split_ctm_sweep_multisite_2x2_via_fused(
            envs, site_tensors, neighbors, chi, chi_I, renormalize=renormalize
        )
    # Function-local import: _split_ctm_tensor_moves imports from this module,
    # so importing the absorb helpers at module scope would form a cycle.
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_base_charges,
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
                base_charges=_split_base_charges(site_tensors[s]),
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
        recipe:      ``'2x2'`` (default) runs the genuine joint 2-site forward
                     via :func:`_split_ctm_sweep_multisite_2x2` (2×2 plaquette
                     projectors, matching the fused sweep); ``'1x1'`` reuses the
                     single-site directional moves (bisection / uniform smoke).

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
            envs, site_tensors, bars, neighbors, chi, chi_I, renormalize=renormalize
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


def ctm_split_tensor_2site(
    A: Tensor,
    B: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    recipe: str = "2x2",
) -> tuple[SplitCTMTensorEnv, SplitCTMTensorEnv]:
    """Run 2-site checkerboard split-CTM to convergence.

    Twin of :func:`ctm_tensor_2site`: builds ``{(0, 0): A, (1, 0): B}`` with
    ``CHECKERBOARD_NEIGHBORS`` and delegates to :func:`_split_ctm_multisite`.
    Returns ``(env_A, env_B)`` genuinely coupled -- A's environment absorbs B's
    double layer and vice versa (a true bipartite checkerboard fixed point),
    unlike two independently-converged single-site envs.

    Args:
        A, B:       The two checkerboard iPEPS site tensors (5-leg
                    ``(u, d, l, r, phys)`` each).
        chi:        Environment bond dimension.
        max_iter:   Maximum CTM iterations.
        conv_tol:   Convergence tolerance on corner singular values (``0.0``
                    runs all ``max_iter`` sweeps).
        chi_I:      Interlayer bond dimension. Defaults to ``chi``.
        renormalize: Renormalize the environment each sweep.
        recipe:     ``"2x2"`` (default, the genuine joint forward) or ``"1x1"``
                    (single-site-move reuse, for bisection).

    Returns:
        ``(env_A, env_B)`` -- the converged split environments at ``(0, 0)``
        and ``(1, 0)``.
    """
    envs = _split_ctm_multisite(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        chi_I=chi_I,
        renormalize=renormalize,
        recipe=recipe,
    )
    return envs[(0, 0)], envs[(1, 0)]
