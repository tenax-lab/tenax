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

from typing import NamedTuple

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
    "remove_inverse_roots_multisite",
    "absorb_inverse_roots_multisite",
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


# ---------------------------------------------------------------------------
# Forward: simultaneous CTMRG on the unit cell
# ---------------------------------------------------------------------------
#
# Upper-half convention throughout (see the design note): the cut at ``co``
# lies between ``EC[co]`` and ``EC[next_coordinate(co)]``.  Phase 1 cuts the
# left half instead; the two are the same truncation up to a shift and a
# transpose, which is why the 1x1 gate below compares *energies* and not
# tensors.


def coordinates(nrows: int, ncols: int) -> list[Coord]:
    """Every ``(k, r, c)`` of the cell, directions slowest."""
    return [(k, r, c) for k in range(4) for r in range(nrows) for c in range(ncols)]


def _as_matrix(EC: jax.Array) -> jax.Array:
    """Enlarged corner as a matrix: rows ``(chi_d, a_d)``, cols ``(chi_r, a_r)``.

    This is the ``T`` of the design note.  ``T(X).T == X.reshape(n, n)`` is
    what makes the upper-half cut the transpose of Phase 1's left-half one.
    """
    n = EC.shape[0] * EC.shape[1]
    return jnp.transpose(EC, (2, 3, 0, 1)).reshape(n, n)


def all_enlarged_corners(corners, edges, a_by_cell, nrows: int, ncols: int):
    return {
        co: enlarged_corner(corners, edges, a_by_cell, co, nrows, ncols)
        for co in coordinates(nrows, ncols)
    }


def half_infinite_multisite(ECs, co: Coord, nrows: int, ncols: int):
    """Paper Eq. 65 on the cell: ``M = A @ B`` with the two pieces returned too.

    ``A`` is the enlarged corner at ``co``, ``B`` the one at
    ``next_coordinate(co)``; the projectors need both, not just the product.
    """
    A = _as_matrix(ECs[co])
    B = _as_matrix(ECs[next_coordinate(co, nrows, ncols)])
    return A @ B, A, B


def all_projectors_multisite(ECs, chi: int, nrows: int, ncols: int, prev=None):
    """One Fishman pair per coordinate, all from the *same* environment.

    Simultaneous, not Gauss-Seidel: a sequential sweep has a fixed point that
    does not satisfy Eqs. 76-77, because those evaluate every direction at the
    same ``y``.  Same argument as Phase 1's :func:`all_projectors`.

    Returns ``{co: (P_left, P_right, U, S, Vh)}``.  ``P_left`` is ``(n, chi)``
    and attaches to the ``co`` side of the cut; ``P_right`` is ``(chi, n)`` and
    attaches to the ``next_coordinate(co)`` side, so ``P_right @ P_left`` is
    the identity on the retained subspace by construction.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _inv_sqrt, _pin_bond_gauge

    out = {}
    for co in coordinates(nrows, ncols):
        M, A, B = half_infinite_multisite(ECs, co, nrows, ncols)
        U, s, Vh = jnp.linalg.svd(M, full_matrices=True)
        # Floor before S becomes a matrix: an early environment is rank
        # deficient and a singular S makes the matrix inverse square root NaN.
        s_k = jnp.maximum(s[:chi], 1e-12 * s[0])
        # Cast to M's dtype — S is a *variable* of the characteristic
        # equations and its cotangent must be free to leave the reals (#721).
        S_keep = jnp.diag(s_k / (jnp.linalg.norm(s_k) + 1e-300)).astype(M.dtype)
        inv_sqrt = _inv_sqrt(S_keep)
        P_left = B @ Vh[:chi].conj().T @ inv_sqrt
        P_right = inv_sqrt @ (U[:, :chi].conj().T @ A)
        U, Vh, P_left, P_right = _pin_bond_gauge(
            U, Vh, P_left, P_right, chi, None if prev is None else prev[co][0]
        )
        out[co] = (P_left, P_right, U, S_keep, Vh)
    return out


def _absorbed_edge(edges, a_by_cell, co: Coord, nrows: int, ncols: int):
    """``edge[above(co)]`` with the local double layer absorbed.

    Legs ``((chi, a), a_out, (chi, a))`` with ``chi`` slow in each fused pair,
    matching the projectors' cut-leg order.
    """
    k, r, c = co
    T = edges[above(co, nrows, ncols)]
    a = rotate_a_times(a_by_cell[(r, c)], k)
    chi, d2 = T.shape[0], a.shape[0]
    # T[l, x, r] with x contracting a.u; a[u, d, l, r] = a[x, j, i, m].
    raw = jnp.einsum("lxr,xjim->lijrm", T, a)
    return raw.reshape(chi * d2, d2, chi * d2)


def sweep_multisite(
    corners, edges, a_by_cell, chi: int, nrows: int, ncols: int, prev=None
):
    """One simultaneous sweep over the whole cell.

    Renormalisation coordinates are PEPSKit's, uniform in ``co``::

        corner[co] = P_right[prev_coordinate(co)] · EC[co]        · P_left[co]
        edge[co]   = P_right[left_projector(co)]  · (E[above] ⊗ a) · P_left[co]

    The same tables the characteristic equations will use — a sweep and an
    equation that disagree give a fixed point that is not a root, which is
    how #718 started.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _normalize

    ECs = all_enlarged_corners(corners, edges, a_by_cell, nrows, ncols)
    projs = all_projectors_multisite(ECs, chi, nrows, ncols, prev)

    new_corners, new_edges = {}, {}
    for co in coordinates(nrows, ncols):
        P_left = projs[co][0]
        P_right_prev = projs[prev_coordinate(co, nrows, ncols)][1]
        new_corners[co] = _normalize(P_right_prev @ _as_matrix(ECs[co]) @ P_left)

        P_right_lp = projs[left_projector(co, nrows, ncols)][1]
        raw = _absorbed_edge(edges, a_by_cell, co, nrows, ncols)
        new_edges[co] = _normalize(
            jnp.einsum("ai,ixj,jb->axb", P_right_lp, raw, P_left)
        )
    return new_corners, new_edges, projs


def init_cell_env(a_by_cell, A_by_cell, chi: int, nrows: int, ncols: int):
    """Seed every cell from :func:`initialize_ctm_tensor_env`.

    Each cell gets its own initial environment, built from its own site
    tensor, in this module's rotation-uniform convention.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _init_env

    corners, edges = {}, {}
    for (r, c), A in A_by_cell.items():
        env, _a = _init_env(A, chi)
        ck, ek = env_to_cell_maps(env, r, c)
        corners.update(ck)
        edges.update(ek)
    del a_by_cell, nrows, ncols
    return corners, edges


def converge_multisite(
    A_by_cell,
    chi: int,
    nrows: int,
    ncols: int,
    *,
    max_iter: int = 200,
    conv_tol: float = 1e-12,
    min_iter: int = 4,
    return_projectors: bool = False,
):
    """Sweep the cell until every corner and edge stops moving element-wise.

    Element-wise, not spectral, for the Phase 1 reason: corner *singular
    values* are invariant under independent rotations of each bond, so a
    spectral criterion calls convergence while the tensors are still moving —
    and the characteristic equations compare tensors.
    """
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

    a_by_cell = {}
    for (r, c), A in A_by_cell.items():
        a_t = _build_double_layer_tensor(A)
        labels = list(a_t.labels())
        perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
        a_by_cell[(r, c)] = jnp.asarray(a_t.transpose(perm).todense())

    corners, edges = init_cell_env(a_by_cell, A_by_cell, chi, nrows, ncols)
    prev_state = None
    prev_projs = None
    residual = float("inf")
    converged = False
    iters = 0
    for it in range(int(max_iter)):
        corners, edges, prev_projs = sweep_multisite(
            corners, edges, a_by_cell, chi, nrows, ncols, prev_projs
        )
        iters = it + 1
        state = {
            k: v / (jnp.linalg.norm(v) + 1e-300)
            for k, v in list(corners.items()) + list(edges.items())
        }
        if prev_state is not None:
            residual = float(
                max(
                    jnp.max(jnp.abs(state[k] - prev_state[k]))
                    for k in state
                    if state[k].shape == prev_state[k].shape
                )
            )
            if iters >= min_iter and residual < conv_tol:
                converged = True
                break
        prev_state = state

    meta = {"iters": iters, "residual": residual, "converged": converged}
    if return_projectors:
        return corners, edges, meta, prev_projs, a_by_cell
    return corners, edges, meta


# ---------------------------------------------------------------------------
# The y <-> x map on the cell (paper Eq. 82)
# ---------------------------------------------------------------------------
#
# The characteristic equations are written in *modified* corners and edges,
# which carry the inverse singular values explicitly on their environment legs.
# Only that form transforms covariantly under Eq. 84, and only then is holding
# ``U*`` and ``V*`` constant licensed by Eq. 88.
#
# Which coordinate's root sits on which leg follows the projectors that put it
# there.  ``corner[co] = P_R[prev_coordinate(co)] · EC · P_L[co]`` and
# ``edge[co] = P_R[left_projector(co)] · … · P_L[co]``, and each projector
# carries one ``S^-1/2``, so undoing them needs the roots at exactly those two
# coordinates.  At 1x1 both collapse to Phase 1's ``roots[k]`` on the edge and
# ``roots[k-1], roots[k]`` on the corner.


def _map_cell_roots(corners, edges, roots, nrows: int, ncols: int, *, normalize: bool):
    new_corners, new_edges = {}, {}
    for co in coordinates(nrows, ncols):
        C = roots[prev_coordinate(co, nrows, ncols)] @ corners[co] @ roots[co]
        E = jnp.einsum(
            "ai,ixj,jb->axb",
            roots[left_projector(co, nrows, ncols)],
            edges[co],
            roots[co],
        )
        if normalize:
            C = C / (jnp.linalg.norm(C) + 1e-300)
            E = E / (jnp.linalg.norm(E) + 1e-300)
        new_corners[co] = C
        new_edges[co] = E
    return new_corners, new_edges


def remove_inverse_roots_multisite(corners, edges, S_all, nrows: int, ncols: int):
    """Regular ``x`` -> modified ``(C̃, Ẽ)``: multiply by ``sqrt(S)``."""
    from tenax.algorithms._ctm_root_implicit_asym import _denman_beavers

    roots = {co: _denman_beavers(S)[0] for co, S in S_all.items()}
    return _map_cell_roots(corners, edges, roots, nrows, ncols, normalize=True)


def absorb_inverse_roots_multisite(corners_t, edges_t, S_all, nrows: int, ncols: int):
    """Modified ``(C̃, Ẽ)`` -> regular ``x``: multiply by ``sqrt(S^-1)``.

    The differentiable direction.  ``S`` enters here, so the energy — evaluated
    on the *regular* environment — depends on ``S`` through this map, and that
    is what gives ``S`` a nonzero adjoint.  Writing ``F`` in the regular
    variables instead sets it to zero, which was the #718 bug.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _inv_sqrt

    roots = {co: _inv_sqrt(S) for co, S in S_all.items()}
    return _map_cell_roots(corners_t, edges_t, roots, nrows, ncols, normalize=True)


# ---------------------------------------------------------------------------
# Characteristic equations on the unit cell (paper Eqs. 76-80, Appendix F)
# ---------------------------------------------------------------------------
#
# Structurally identical to Phase 1's
# ``asym_characteristic_residual_covariant``.  The one difference is that every
# ``s``-derived factor is read from a *shifted* coordinate rather than from the
# same direction:
#
#     S^-1 in the projectors      proj_sinv_indices(co)
#     (s† s)^1/4 on U             leftvec_invfroot_indices(co)
#     (s s†)^1/4 on V             rightvec_invfroot_indices(co)
#     iCi = s · C̃ · s             prev_coordinate(co) and co
#
# At a 1x1 cell all four collapse to Phase 1's bare direction arithmetic.
#
# Phase 1 needs ``asym_root_to_covariant_convention`` to shift and transpose
# its forward data, because its forward truncates with the left half-plane
# while §V.3 uses the upper half.  This module's forward already uses the upper
# half, so the forward's (U, S, Vh) at ``co`` *are* the §V.3 data at ``co`` and
# no relabelling step exists here at all.


class CellRoot(NamedTuple):
    """Root variables plus the constants held fixed while differentiating."""

    corners: dict  # C̃, modified
    edges: dict  # Ẽ, modified
    u: dict  # (n - chi) x chi
    s: dict  # chi x chi, a general matrix
    v: dict  # chi x (n - chi)
    U_star: dict
    U_perp: dict
    Vh_star: dict
    Vh_perp: dict
    s_star_inv: dict
    nrows: int
    ncols: int

    @property
    def y(self):
        return (self.corners, self.edges, self.u, self.s, self.v)


def _covariant_pieces_multisite(consts: CellRoot, S_all, u_all, v_all, d2: int):
    """Per-coordinate covariant building blocks (paper Eqs. 71-75).

    The quartic roots attach to the ``n = chi * d2`` *cut* leg on its ``chi``
    sub-leg — hence ``kron(root, I_d2)`` with ``chi`` the slow index — and they
    come from the shifted coordinates, not from ``co``.  Phase 1 justifies the
    placement from the Eq. 87 transformation laws; nothing about that argument
    depends on the unit cell.
    """
    from tenax.algorithms._ctm_root_implicit_asym import _quartic_root

    nrows, ncols = consts.nrows, consts.ncols
    s_all = {co: jnp.linalg.inv(S) for co, S in S_all.items()}
    eye_d2 = jnp.eye(d2, dtype=next(iter(s_all.values())).dtype)
    K_L = {
        co: jnp.kron(_quartic_root(s.conj().T @ s), eye_d2) for co, s in s_all.items()
    }
    K_R = {
        co: jnp.kron(_quartic_root(s @ s.conj().T), eye_d2) for co, s in s_all.items()
    }

    Ud, Vd, ULd, VRd = {}, {}, {}, {}
    for co in coordinates(nrows, ncols):
        chi = S_all[co].shape[0]
        kl = K_L[leftvec_invfroot_indices(co, nrows, ncols)]
        kr = K_R[rightvec_invfroot_indices(co, nrows, ncols)]
        U = consts.U_star[co] + consts.U_perp[co] @ u_all[co]  # Eq. 71
        Vh = consts.Vh_star[co] + v_all[co] @ consts.Vh_perp[co]  # Eq. 72
        Ud[co] = U[:, :chi].conj().T @ kl
        Vd[co] = kr @ Vh[:chi].conj().T
        ULd[co] = consts.U_perp[co].conj().T @ kl
        VRd[co] = kr @ consts.Vh_perp[co].conj().T
    return s_all, Ud, Vd, ULd, VRd


def _modified_corners(corners_tilde, s_all, nrows: int, ncols: int):
    """``iCi[co] = s[prev_coordinate(co)] · C̃[co] · s[co]``.

    The full inverse on *both* corner legs is what puts the singular values
    explicitly into the contraction environment, so that it transforms
    covariantly under Eq. 84.  Edges already carry their ``s`` from the Eq. 82
    map and are left alone.
    """
    return {
        co: s_all[prev_coordinate(co, nrows, ncols)] @ C @ s_all[co]
        for co, C in corners_tilde.items()
    }


def characteristic_residual_multisite(y, a_by_cell, consts: CellRoot, chi: int):
    """``F(y, p)`` for the whole unit cell.

    ``y = (corners_tilde, edges_tilde, u, S, v)``.  Normalisation is the
    reference's ``X'/λ - X``; ``λ`` is deliberately not real-projected, since
    ``dot(X, X')`` is genuinely complex for a complex state and taking the real
    part alone moves ``|F1|`` by thirteen orders (Phase 1, #721).
    """
    corners_t, edges_t, u_all, S_all, v_all = y
    nrows, ncols = consts.nrows, consts.ncols
    d2 = next(iter(a_by_cell.values())).shape[0]
    n = chi * d2

    s_all, Ud, Vd, ULd, VRd = _covariant_pieces_multisite(
        consts, S_all, u_all, v_all, d2
    )
    corners_mod = _modified_corners(corners_t, s_all, nrows, ncols)

    eye_d2 = jnp.eye(d2, dtype=next(iter(s_all.values())).dtype)
    K_is = {co: jnp.kron(s, eye_d2) for co, s in s_all.items()}

    EC, cols = {}, {}
    for co in coordinates(nrows, ncols):
        k, r, c = co
        EC[co] = _as_matrix(
            enlarged_corner(corners_mod, edges_t, a_by_cell, co, nrows, ncols)
        )
        # The same edge that enters EC[co], with the sandwich attached:
        # (chi_l, a_l | a_out | chi_r, a_r).
        T = edges_t[above(co, nrows, ncols)]
        a_k = rotate_a_times(a_by_cell[(r, c)], k)
        cols[co] = jnp.einsum("xfy,fjlr->xljyr", T, a_k).reshape(n, d2, n)

    M, P_R, P_L = {}, {}, {}
    for co in coordinates(nrows, ncols):
        nxt = next_coordinate(co, nrows, ncols)
        kis = K_is[proj_sinv_indices(co, nrows, ncols)]
        M[co] = EC[co] @ kis @ EC[nxt]
        P_R[co] = Ud[co] @ EC[co] @ kis
        P_L[co] = kis @ EC[nxt] @ Vd[co]

    R_C, R_E, R_u, R_S, R_v = {}, {}, {}, {}, {}
    for co in coordinates(nrows, ncols):
        M_co, s_inv = M[co], consts.s_star_inv[co]

        core = Ud[co] @ M_co @ Vd[co]
        lam_S = jnp.vdot(S_all[co], core)
        R_S[co] = core / lam_S - S_all[co]
        R_u[co] = (ULd[co] @ M_co @ Vd[co]) @ s_inv / lam_S - u_all[co]
        R_v[co] = s_inv @ (Ud[co] @ M_co @ VRd[co]) / lam_S - v_all[co]

        C_new = P_R[prev_coordinate(co, nrows, ncols)] @ EC[co] @ P_L[co]
        C_cur = corners_t[co]
        R_C[co] = C_new / jnp.vdot(C_cur, C_new) - C_cur

        E_new = jnp.einsum(
            "ax,xjy,yb->ajb",
            P_R[left_projector(co, nrows, ncols)],
            cols[co],
            P_L[co],
        )
        E_cur = edges_t[co]
        R_E[co] = E_new / jnp.vdot(E_cur, E_new) - E_cur

    return (R_C, R_E, R_u, R_S, R_v)


def root_parametrize_multisite(
    corners,
    edges,
    a_by_cell,
    chi: int,
    nrows: int,
    ncols: int,
    *,
    prev_projs=None,
    pinv_rtol: float = 1e-10,
    polish_steps: int = 40,
    polish_tol: float = 1e-10,
):
    """Extract ``y* = (C̃, Ẽ, 0, S*, 0)`` and the frozen isometries.

    Every tensor is rescaled to unit Frobenius norm so the ``λ`` defined as an
    inner product really is the eigenvalue Eqs. 76-77 need.

    Pass ``prev_projs`` whenever the environment came from a sweep chain: it
    carries that chain's bond gauge, and a cold pin fixes a *different* one,
    leaving ``y*`` describing an environment it was not extracted from.  For a
    real state one polish sweep absorbs the difference; for a complex state the
    gauge is a continuous phase and the corner residual plateaus instead
    (#721).
    """
    from tenax.algorithms._ctm_root_implicit_asym import _normalize

    del _normalize
    best = None
    d2 = next(iter(a_by_cell.values())).shape[0]
    n = chi * d2
    dtype = next(iter(corners.values())).dtype

    for _step in range(max(int(polish_steps), 1)):
        corners = {co: C / (jnp.linalg.norm(C) + 1e-300) for co, C in corners.items()}
        edges = {co: E / (jnp.linalg.norm(E) + 1e-300) for co, E in edges.items()}

        ECs = all_enlarged_corners(corners, edges, a_by_cell, nrows, ncols)
        projs = all_projectors_multisite(ECs, chi, nrows, ncols, prev_projs)
        prev_projs = projs

        U_star, U_perp, Vh_star, Vh_perp, s_map, s_inv = {}, {}, {}, {}, {}, {}
        for co in coordinates(nrows, ncols):
            _pl, _pr, U, S_keep, Vh = projs[co]
            U_star[co] = U[:, :chi]
            U_perp[co] = U[:, chi:]
            Vh_star[co] = Vh[:chi]
            Vh_perp[co] = Vh[chi:]
            s_map[co] = S_keep
            diag = jnp.diag(S_keep).real
            cutoff = pinv_rtol * jnp.max(diag)
            inv_diag = jnp.where(
                diag > cutoff, 1.0 / jnp.where(diag > cutoff, diag, 1.0), 0.0
            )
            s_inv[co] = jnp.diag(inv_diag).astype(S_keep.dtype)

        corners_t, edges_t = remove_inverse_roots_multisite(
            corners, edges, s_map, nrows, ncols
        )
        root = CellRoot(
            corners=corners_t,
            edges=edges_t,
            u={co: jnp.zeros((n - chi, chi), dtype=dtype) for co in s_map},
            s=s_map,
            v={co: jnp.zeros((chi, n - chi), dtype=dtype) for co in s_map},
            U_star=U_star,
            U_perp=U_perp,
            Vh_star=Vh_star,
            Vh_perp=Vh_perp,
            s_star_inv=s_inv,
            nrows=nrows,
            ncols=ncols,
        )
        R = characteristic_residual_multisite(root.y, a_by_cell, root, chi)
        residual = float(
            jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(R)))
        )
        if best is None or residual < best[1]:
            best = (root, residual)
        if residual <= polish_tol:
            break
        corners, edges, prev_projs = sweep_multisite(
            corners, edges, a_by_cell, chi, nrows, ncols, projs
        )

    assert best is not None
    return best
