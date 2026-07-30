"""Root implicit differentiation for asymmetric CTMRG on a unit cell (#715 Phase 2).

Generalises :mod:`tenax.algorithms._ctm_root_implicit_asym` — which fixes the
unit cell at 1x1 — to an arbitrary ``nrows x ncols`` cell, following Appendix F
of Burgelman, Francuz, Brehmer, Devos, Haegeman, Verstraete and Vanhecke,
*Implicit differentiation of tensor network algorithms*, arXiv:2607.15030.

What actually changes
---------------------
The characteristic equations themselves are unchanged: Eqs. 76-80 hold per
direction exactly as in Phase 1.  What the unit cell adds is that the singular
values entering a given equation no longer all belong to the *same* coordinate.
``S``, its inverse, and the two quartic roots ``s^L = (s s†)^1/4``,
``s^R = (s† s)^1/4`` are each read from a **shifted** cell, and the shift
differs per quantity (Appendix F, Tables 1+).

That is the whole content of this module's index layer, and it is the part
worth being paranoid about: a wrong shift still produces a well-conditioned
Jacobian and a plausible-looking root, then a silently wrong gradient.  Same
failure shape as #700 / #702.  So the tables are transcribed from the authors'
reference implementation rather than re-derived, and pinned element by element
in ``tests/test_ctm_root_implicit_multisite.py``.

Indexing convention
-------------------
Everything here is **0-based**: ``k ∈ {0, 1, 2, 3}`` for the direction and
``(r, c)`` for the cell, all periodic.  The reference is Julia and therefore
1-based with ``mod1``; the translation is ``k = dir - 1`` and plain ``%``.

Directions follow the Phase 1 module: ``k`` numbers the corner ``C{k+1}`` and
the edge ``T{k+1}``, going around the ring.  A coordinate is the triple
``(k, r, c)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_root_implicit_asym import AsymEnv, rotate_a

__all__ = [
    "above",
    "above_left",
    "cell_maps_to_env",
    "enlarged_corner",
    "env_to_cell_maps",
    "left",
    "left_projector",
    "leftvec_invfroot_indices",
    "next_coordinate",
    "prev_coordinate",
    "proj_sinv_indices",
    "rightvec_invfroot_indices",
    "rotate_a_times",
]

Coord = tuple[int, int, int]


def _next(i: int, total: int) -> int:
    return (i + 1) % total


def _prev(i: int, total: int) -> int:
    return (i - 1) % total


# ---------------------------------------------------------------------------
# Walking the ring of directions
# ---------------------------------------------------------------------------


def next_coordinate(co: Coord, nrows: int, ncols: int) -> Coord:
    """The next ``(k, r, c)`` going around the environment clockwise.

    Advancing the direction also steps the cell, because direction ``k+1``'s
    corner sits one site along the edge that direction ``k`` just absorbed.
    """
    k, r, c = co
    if k == 0:
        return (1, r, _next(c, ncols))
    if k == 1:
        return (2, _next(r, nrows), c)
    if k == 2:
        return (3, r, _prev(c, ncols))
    return (0, _prev(r, nrows), c)


def prev_coordinate(co: Coord, nrows: int, ncols: int) -> Coord:
    """Inverse of :func:`next_coordinate`."""
    k, r, c = co
    if k == 0:
        return (3, _next(r, nrows), c)
    if k == 1:
        return (0, r, _prev(c, ncols))
    if k == 2:
        return (1, _prev(r, nrows), c)
    return (2, r, _next(c, ncols))


# ---------------------------------------------------------------------------
# Appendix F: which cell each singular-value factor comes from
# ---------------------------------------------------------------------------


def proj_sinv_indices(co: Coord, nrows: int, ncols: int) -> Coord:
    """Cell supplying ``S^-1`` for the projectors at ``co``.

    The direction is unchanged; the cell steps one site *outward* along that
    direction — the bond being cut lies between ``co`` and this neighbour.
    """
    k, r, c = co
    if k == 0:
        return (k, _prev(r, nrows), c)
    if k == 1:
        return (k, r, _next(c, ncols))
    if k == 2:
        return (k, _next(r, nrows), c)
    return (k, r, _prev(c, ncols))


def leftvec_invfroot_indices(co: Coord, nrows: int, ncols: int) -> Coord:
    """Cell supplying ``(s† s)^1/4`` for the ``U`` isometry at ``co``.

    Direction steps *back* by one: the quartic root rides the cut leg of the
    **previous** direction's edge, which is the leg ``U`` is contracted along.
    At a 1x1 cell this is the ``k-1`` of Phase 1's ``_covariant_pieces``.
    """
    k, r, c = co
    if k == 0:
        rc = (_next(r, nrows), _prev(c, ncols))
    elif k == 1:
        rc = (_prev(r, nrows), _prev(c, ncols))
    elif k == 2:
        rc = (_prev(r, nrows), _next(c, ncols))
    else:
        rc = (_next(r, nrows), _next(c, ncols))
    return (_prev(k, 4), *rc)


def rightvec_invfroot_indices(co: Coord, nrows: int, ncols: int) -> Coord:
    """Cell supplying ``(s s†)^1/4`` for the ``V`` isometry at ``co``.

    Direction steps *forward* by one, and the cell by **two** — the only
    two-step shift in the assignment, and the one most likely to be wrong if
    re-derived.  At a 1x1 cell this is Phase 1's ``k+1``.
    """
    k, r, c = co
    if k == 0:
        rc = (r, _next(_next(c, ncols), ncols))
    elif k == 1:
        rc = (_next(_next(r, nrows), nrows), c)
    elif k == 2:
        rc = (r, _prev(_prev(c, ncols), ncols))
    else:
        rc = (_prev(_prev(r, nrows), nrows), c)
    return (_next(k, 4), *rc)


# ---------------------------------------------------------------------------
# Positions relative to an enlarged corner
# ---------------------------------------------------------------------------


def above_left(co: Coord, nrows: int, ncols: int) -> Coord:
    """Corner diagonally up-left of the enlarged corner at ``co``."""
    k, r, c = co
    if k == 0:
        return (k, _prev(r, nrows), _prev(c, ncols))
    if k == 1:
        return (k, _prev(r, nrows), _next(c, ncols))
    if k == 2:
        return (k, _next(r, nrows), _next(c, ncols))
    return (k, _next(r, nrows), _prev(c, ncols))


def left(co: Coord, nrows: int, ncols: int) -> Coord:
    """Edge to the left of the enlarged corner at ``co``."""
    k, r, c = co
    if k == 0:
        return (_prev(k, 4), r, _prev(c, ncols))
    if k == 1:
        return (_prev(k, 4), _prev(r, nrows), c)
    if k == 2:
        return (_prev(k, 4), r, _next(c, ncols))
    return (_prev(k, 4), _next(r, nrows), c)


def above(co: Coord, nrows: int, ncols: int) -> Coord:
    """Edge above the enlarged corner at ``co``."""
    k, r, c = co
    if k == 0:
        return (k, _prev(r, nrows), c)
    if k == 1:
        return (k, r, _next(c, ncols))
    if k == 2:
        return (k, _next(r, nrows), c)
    return (k, r, _prev(c, ncols))


def left_projector(co: Coord, nrows: int, ncols: int) -> Coord:
    """Projector pair to the left of the enlarged corner at ``co``."""
    k, r, c = co
    if k == 0:
        return (k, r, _prev(c, ncols))
    if k == 1:
        return (k, _prev(r, nrows), c)
    if k == 2:
        return (k, r, _next(c, ncols))
    return (k, _next(r, nrows), c)


# ---------------------------------------------------------------------------
# Coordinate-indexed environment
# ---------------------------------------------------------------------------
#
# The environment is two dicts keyed by ``(k, r, c)`` rather than the eight
# named fields of :class:`AsymEnv`.  Dicts are JAX pytrees, so ``jax.vjp``
# traverses them unchanged, and keying by coordinate is what lets the Appendix
# F tables above be applied literally instead of being unrolled into per-field
# special cases.


def env_to_cell_maps(env: AsymEnv, r: int = 0, c: int = 0):
    """Spread one :class:`AsymEnv` over cell ``(r, c)`` of a coordinate map.

    Corner ``C{k+1}`` and edge ``T{k+1}`` land at ``(k, r, c)``.  Used to seed
    a 1x1 cell from the Phase 1 representation, and by the tests that hold the
    two implementations against each other.
    """
    corners = {(k, r, c): getattr(env, f"C{k + 1}") for k in range(4)}
    edges = {(k, r, c): getattr(env, f"T{k + 1}") for k in range(4)}
    return corners, edges


def cell_maps_to_env(corners, edges, r: int = 0, c: int = 0) -> AsymEnv:
    """Inverse of :func:`env_to_cell_maps` for a single cell."""
    return AsymEnv(
        *[corners[(k, r, c)] for k in range(4)],
        *[edges[(k, r, c)] for k in range(4)],
    )


def rotate_a_times(a: jax.Array, k: int) -> jax.Array:
    """Rotate the double-layer tensor ``k`` quarter turns counter-clockwise."""
    for _ in range(k % 4):
        a = rotate_a(a)
    return a


def enlarged_corner(corners, edges, a_by_cell, co: Coord, nrows: int, ncols: int):
    """The enlarged corner at ``co``, axes ``(chi_r, a_r, chi_d, a_d)``.

    Three environment tensors meet the local double layer:

    ===============  ===========================
    corner           ``above_left(co)``
    edge above       ``above(co)``
    edge to the left ``left(co)``
    double layer     cell ``(r, c)``, rotated ``k``
    ===============  ===========================

    which is Phase 1's ``_upper_left_quadrant`` read by coordinate instead of
    by rotating the whole environment.  At a 1x1 cell the two agree to machine
    precision — the bridge test in
    ``tests/test_ctm_root_implicit_multisite.py`` — and that equality is what
    licenses reusing the Phase 1 contraction formulae here verbatim.

    The ``(chi_d, a_d)`` pair is the bond about to be truncated; ``(chi_r,
    a_r)`` stays open.
    """
    k, r, c = co
    C = corners[above_left(co, nrows, ncols)]
    T_above = edges[above(co, nrows, ncols)]
    T_left = edges[left(co, nrows, ncols)]
    a = rotate_a_times(a_by_cell[(r, c)], k)
    return jnp.einsum("ce,efg,hic,fjik->gkhj", C, T_above, T_left, a)
