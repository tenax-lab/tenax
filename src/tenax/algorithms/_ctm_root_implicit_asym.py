"""Root implicit differentiation for asymmetric CTMRG (#715, Phase 1).

Implements Section V of Burgelman, Francuz, Brehmer, Devos, Haegeman,
Verstraete and Vanhecke, *Implicit differentiation of tensor network
algorithms*, arXiv:2607.15030 — the general case with four distinct corner
and four distinct edge tensors, no spatial symmetry.

Relation to :mod:`tenax.algorithms._ctm_c4v_root_implicit` (Phase 0): same
idea, bigger variable set.  Where C4v needs ``y = (C, E, u)`` and one
``eigh``, the asymmetric case needs

    y = ({C_a}, {E_a}, {u_a}, {S_a}, {v_a})           (paper Eq. 69)

— 20 variables and 20 characteristic equations for a 1x1 unit cell, because
the truncated *SVD* has two isometries to pin instead of one.

Characteristic equations, per direction ``a``, with ``M_a`` the half-infinite
environment of Eq. 65 and ``U_a = U*_a + U_perp,a u_a``,
``Vh_a = Vh*_a + v_a Vh_perp,a`` (Eq. 88):

    R_C = P_bot Q P_top - λ_C C                  (Eq. 76)
    R_E = P_bot (E·a) P_top - λ_E E              (Eq. 77)
    R_u = (U_perp† M Vh†) S*^-1 - λ_S u          (Eq. 78)
    R_S = diag(U† M Vh†) - λ_S S                 (Eq. 79)
    R_v = S*^-1 (U† M Vh_perp†) - λ_S v          (Eq. 80)

Where this stands (#715 Phase 1, incomplete)
-------------------------------------------
``S`` is a general complex chi x chi matrix, and the projectors use a genuine
matrix inverse square root (Denman-Beavers, so no decomposition enters ``F``).
That much is required, not optional: with a *diagonal* ``S`` the in-space
rotation of the isometries has nowhere to go, the projector closure breaks at
first order, and Eq. 88's null-space restriction then discards a physical
contribution instead of a gauge one — the gradient came out 120% wrong.
Promoting ``S`` brings it to 1-2.5%.

Still missing is the rest of Eqs. 73-82: the modified corners and edges that
carry ``s`` explicitly on the bonds, and the quartic roots
``s^L = (S S†)^-1/4``, ``s^R = (S† S)^-1/4`` on the *cut legs* of that
environment.  Putting those roots inside the projectors instead was tried and
is worse (20%): they are equivariant under independent left/right rotations,
but their product is not ``S^-1``, so the closure breaks for a non-diagonal
``S``.  See ``docs/plans/2026-07-29-715-phase1-modified-variables.md``.

Conventions (all dense ``jnp`` arrays, ``d2 = D²``)
---------------------------------------------------
``a[u, d, l, r]`` double layer; environment laid out exactly as
:class:`~tenax.algorithms._ctm_tensor_init.CTMTensorEnv`::

    C1[d, r]      C2[l, d]      C3[u, l]      C4[r, u]
    T1[l, u2, r]  T2[u, r2, d]  T3[r, d2, l]  T4[d, l2, u]

The middle index of each edge contracts the correspondingly-named leg of
``a``.  A 90° counter-clockwise rotation is a pure cyclic relabel of the
eight environment tensors with **no axis permutation** — only ``a`` is
transposed — which is what lets one ``left`` move serve all four directions.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

__all__ = [
    "AsymEnv",
    "asym_characteristic_residual",
    "asym_root_implicit_energy_and_grad",
    "asym_root_parametrize",
]


class AsymEnv(NamedTuple):
    """Eight-tensor CTM environment as dense arrays."""

    C1: jax.Array
    C2: jax.Array
    C3: jax.Array
    C4: jax.Array
    T1: jax.Array
    T2: jax.Array
    T3: jax.Array
    T4: jax.Array


def rotate_env(env: AsymEnv) -> AsymEnv:
    """Rotate the environment 90° counter-clockwise.

    Old top-right becomes new top-left, and so on.  Every tensor's axis
    order is already the rotated one (``C2[l, d]`` read as ``C1[d, r]``
    is the same array), so this is a pure relabel.
    """
    return AsymEnv(
        C1=env.C2,
        C2=env.C3,
        C3=env.C4,
        C4=env.C1,
        T1=env.T2,
        T2=env.T3,
        T3=env.T4,
        T4=env.T1,
    )


def rotate_a(a: jax.Array) -> jax.Array:
    """Rotate the double-layer tensor 90° counter-clockwise.

    new-up = old-right, new-down = old-left, new-left = old-up,
    new-right = old-down.
    """
    return jnp.transpose(a, (3, 2, 0, 1))


def _unrotate_index(slot: int, k: int) -> int:
    """Which original tensor sits in ``slot`` after ``k`` rotations.

    Slots and tensors are both numbered 1..4 in the ``C1..C4`` order, and
    one rotation advances the label by one, so this is modular arithmetic.
    """
    return (slot - 1 + k) % 4 + 1


# ---------------------------------------------------------------------------
# Enlarged corners and the half-infinite environment (paper Eq. 65)
# ---------------------------------------------------------------------------


def _upper_left_quadrant(env: AsymEnv, a: jax.Array) -> jax.Array:
    """``C1·T1·T4·a`` with axes ``(chi_r, a_r, chi_d, a_d)``.

    The ``(chi_d, a_d)`` pair is the vertical bond to be truncated; the
    ``(chi_r, a_r)`` pair is what remains open to the right.
    """
    # C1[c,e] T1[e,f,g] T4[h,i,c] a[f,j,i,k]
    #   c = C1.d = T4.u   e = C1.r = T1.l   f = T1's a-leg = a.u
    #   g = T1.r          h = T4.d          i = T4's a-leg = a.l
    #   j = a.d           k = a.r
    return jnp.einsum("ce,efg,hic,fjik->gkhj", env.C1, env.T1, env.T4, a)


def _lower_left_quadrant(env: AsymEnv, a: jax.Array) -> jax.Array:
    """``C4·T3·T4·a`` with axes ``(chi_u, a_u, chi_r, a_r)``.

    The ``(chi_u, a_u)`` pair is the vertical bond to be truncated.
    """
    # C4[m,n] T3[p,q,m] T4[n,i,t] a[u,q,i,k]
    #   m = C4.r = T3.l   n = C4.u = T4.d   p = T3.r
    #   q = T3's a-leg = a.d               i = T4's a-leg = a.l
    #   t = T4.u          u = a.u          k = a.r
    return jnp.einsum("mn,pqm,nit,uqik->tupk", env.C4, env.T3, env.T4, a)


def _denman_beavers(A: jax.Array, n_iter: int = 24):
    """Principal square root and inverse square root, together.

    Deliberately *not* ``eigh``.  ``S`` is a general matrix here, and while a
    matrix root is smooth even where the spectrum is degenerate (its Frechet
    derivative is a Loewner divided difference, which tends to ``f'(x)``),
    JAX's ``eigh`` VJP still divides by eigenvalue differences and would NaN
    exactly where this method is supposed to be safe.

    Denman-Beavers is matrix multiplications and inverses only, so ``F``
    stays free of decompositions and differentiates cleanly.  Convergence is
    quadratic; the spectral range seen here (1 to ~1e-3) is comfortable.
    """
    eye = jnp.eye(A.shape[0], dtype=A.dtype)
    # Frobenius scaling only — deliberately no decomposition here, not even
    # for the scale factor, or an ``svd`` reappears in the jaxpr of ``F`` and
    # the whole point of the method is lost.  Rank deficiency is handled
    # where it arises, by flooring the retained spectrum in
    # :func:`all_projectors`.
    scale = jnp.sqrt(jnp.linalg.norm(A) + 1e-300)
    Y, Z = A / (scale**2), eye
    for _ in range(n_iter):
        Yn = 0.5 * (Y + jnp.linalg.inv(Z))
        Zn = 0.5 * (Z + jnp.linalg.inv(Y))
        Y, Z = Yn, Zn
    return Y * scale, Z / scale


def _inv_sqrt(A: jax.Array, n_iter: int = 24) -> jax.Array:
    return _denman_beavers(A, n_iter)[1]


def _inv_quartic_root(A: jax.Array, n_iter: int = 24) -> jax.Array:
    """``A^-1/4`` for Hermitian positive ``A``, as ``(A^1/2)^-1/2``."""
    return _denman_beavers(_denman_beavers(A, n_iter)[0], n_iter)[1]


def _pin_bond_gauge(U, Vh, P_top, P_bot, chi, prev_P_top=None):
    """Pin the residual phase freedom on each renormalised bond.

    An SVD fixes the singular subspaces but leaves one phase per retained
    index free (one *sign*, for real input).  That phase is a gauge of the
    CTM bond, so the corner spectra converge without it being fixed — and
    indeed the environment converges element-wise in ``|·|`` to 1e-14 while
    individual signs keep flipping from sweep to sweep.  A characteristic
    equation cannot have a root under those conditions: ``F`` compares
    tensors, not their magnitudes.

    Fixing the phase on ``U`` alone is not enough, because the projectors
    also inherit the *previous* bond gauge through the quadrants.  Pinning
    the largest-magnitude entry of each ``P_top`` column to be real-positive
    fixes the new bond directly, and pushing the conjugate phase onto
    ``P_bot`` keeps ``P_bot @ P_top = 1``.

    The same phase is folded into ``(U, Vh)`` so that projectors rebuilt
    from them inside the characteristic equations land in the same gauge.
    """
    if prev_P_top is None:
        # Cold start: pin the largest-magnitude entry of each column.
        idx = jnp.argmax(jnp.abs(P_top), axis=0)
        ref = P_top[idx, jnp.arange(P_top.shape[1])]
    else:
        # Warm: align to the previous sweep.  ``argmax`` is discontinuous —
        # when two retained singular values are close its row index hops
        # between sweeps and the pinned phase oscillates with period two,
        # which is exactly what a near-degenerate pair produced here.
        ref = jnp.sum(jnp.conj(prev_P_top) * P_top, axis=0)
    psi = jnp.where(jnp.abs(ref) > 0, jnp.conj(ref) / jnp.abs(ref), 1.0)
    P_top = P_top * psi[None, :]
    P_bot = jnp.conj(psi)[:, None] * P_bot
    U = U.at[:, :chi].multiply(psi[None, :])
    Vh = Vh.at[:chi, :].multiply(jnp.conj(psi)[:, None])
    return U, Vh, P_top, P_bot


def half_infinite_environment(env: AsymEnv, a: jax.Array) -> jax.Array:
    """Paper Eq. 65: upper-left quadrant glued to lower-left along the cut.

    Returns a ``(chi*d2, chi*d2)`` matrix indexed by the two quadrants'
    outward-facing leg pairs.
    """
    chi, d2 = env.C1.shape[0], a.shape[0]
    top = _upper_left_quadrant(env, a).reshape(chi * d2, chi * d2)
    bot = _lower_left_quadrant(env, a).reshape(chi * d2, chi * d2)
    return top @ bot


def _fishman_projectors(
    env: AsymEnv,
    a: jax.Array,
    U: jax.Array,
    s: jax.Array,
    Vh: jax.Array,
    chi: int,
):
    """Paper Eqs. 66-67 from a *given* decomposition of the cut.

    ``P_bot`` acts on the upper piece's downward leg, ``P_top`` on the lower
    piece's upward leg, and ``P_bot @ P_top`` reconstructs the identity on
    the retained subspace by construction.

    Taking ``(U, s, Vh)`` as arguments rather than recomputing them is what
    makes this usable inside the characteristic equations: there the
    isometries are the *variables*, not the output of a decomposition.
    """
    n = env.C1.shape[0] * a.shape[0]
    top = _upper_left_quadrant(env, a).reshape(n, n)
    bot = _lower_left_quadrant(env, a).reshape(n, n)
    # ``s`` is a general chi x chi matrix, not a vector of singular values.
    # That is what makes the pair gauge-covariant: under the bond rotation
    # U -> U W, Vh -> W† Vh, S -> W† S W the matrix root is equivariant,
    # (W† S W)^-1/2 = W† S^-1/2 W, so P_bot -> W† P_bot and P_top -> P_top W
    # and the closure survives.  With a *diagonal* S the in-space rotation is
    # not representable, the closure breaks at first order, and the
    # null-space restriction of Eq. 88 then discards a real contribution
    # instead of a gauge one.
    # A genuine *matrix* inverse square root, symmetric on both ends.  This
    # is what makes the pair gauge-covariant: under the bond rotation
    # U -> U W, Vh -> W† Vh, S -> W† S W the matrix root is equivariant,
    # (W† S W)^-1/2 = W† S^-1/2 W, so P_bot -> W† P_bot, P_top -> P_top W and
    # P_bot @ P_top = S^-1/2 (U† M Vh†) S^-1/2 = S^-1/2 S S^-1/2 = 1 survives
    # for *any* S, diagonal or not.
    #
    # The two-sided roots of paper Eq. 73, (S S†)^-1/4 and (S† S)^-1/4, were
    # tried here and are worse (2e-1 vs 2.5e-2 gradient error): they are
    # equivariant under independent left/right rotations, but their product
    # is not S^-1, so the closure breaks at first order in a non-diagonal S.
    # In the paper they sit on the *cut legs* of the modified environment,
    # where no closure condition applies — not inside the projectors.
    inv_sqrt = _inv_sqrt(s)

    P_top = bot @ Vh[:chi].conj().T @ inv_sqrt
    P_bot = inv_sqrt @ (U[:, :chi].conj().T @ top)
    return P_top, P_bot


# ---------------------------------------------------------------------------
# Forward: one left move, four rotations to a sweep
# ---------------------------------------------------------------------------


def _left_move_pieces(env: AsymEnv, a: jax.Array):
    """The three tensors a left move renormalises, before projection."""
    chi, d2 = env.C1.shape[0], a.shape[0]
    # C1·T1 -> (C1.d, a.u) x T1.r: the fused pair is the bond above the row.
    c1g = jnp.einsum("ce,efg->cfg", env.C1, env.T1).reshape(chi * d2, -1)
    # C4·T3 -> (C4.u, a.d) x T3.r: the bond below the row.
    c4g = jnp.einsum("mn,pqm->nqp", env.C4, env.T3).reshape(chi * d2, -1)
    # T4·a -> (T4.u, a.u) x a.r x (T4.d, a.d)
    t4g = jnp.einsum("hit,ujik->tukhj", env.T4, a).reshape(chi * d2, d2, chi * d2)
    return c1g, c4g, t4g


def _apply_left_projectors(c1g, c4g, t4g, P_top, P_bot, chi):
    """Insert ``P_top P_bot ≈ 1`` on both vertical bonds of the column."""
    # Which projector goes where follows from the insertion
    # ``Q_upper · (P_top P_bot) · Q_lower ≈ Q_upper · Q_lower``: the piece
    # *above* a bond carries P_top, the piece *below* carries P_bot.  (P_top
    # is indexed by the lower quadrant's legs and P_bot by the upper
    # quadrant's, which reads backwards until you write out the insertion.)
    #
    # C1' sits above its bond.
    C1_new = jnp.einsum("ia,ir->ar", P_top, c1g)
    # C4' sits below its bond; result is C4[chi_r, chi_new].
    C4_new = jnp.einsum("ir,ai->ra", c4g, P_bot)
    # T4' is below its upper bond and above its lower one.  Result is
    # T4[chi_down, a_r, chi_up].
    T4_new = jnp.einsum("ui,ixj,jd->dxu", P_bot, t4g, P_top)
    del chi
    return C1_new, C4_new, T4_new


def _normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.max(jnp.abs(x)) + 1e-300)


def all_projectors(env: AsymEnv, a: jax.Array, chi: int, prev=None):
    """Decompose the cut in all four directions, from the *same* environment.

    Paper Eq. 65 and "their corresponding rotated versions": every projector
    in a sweep is built from one environment, and every corner and edge is
    then renormalised simultaneously.  This matters — a sequential
    (Gauss-Seidel) sweep, where move ``k+1`` sees the output of move ``k``,
    has a fixed point that does *not* satisfy Eqs. 76-77, because those
    equations evaluate all four moves at the same ``y``.
    """
    out = []
    env_k, a_k = env, a
    for k in range(4):
        M = half_infinite_environment(env_k, a_k)
        U, s, Vh = jnp.linalg.svd(M, full_matrices=True)
        # Floor the retained spectrum before it becomes a matrix: early
        # sweeps start from a near-identity environment whose half-infinite
        # matrix is rank deficient, and a singular S makes the matrix
        # inverse square root below produce NaNs.
        s_k = s[:chi]
        s_k = jnp.maximum(s_k, 1e-12 * s_k[0])
        S_keep = jnp.diag(s_k / (jnp.linalg.norm(s_k) + 1e-300))
        P_top, P_bot = _fishman_projectors(env_k, a_k, U, S_keep, Vh, chi)
        U, Vh, P_top, P_bot = _pin_bond_gauge(
            U, Vh, P_top, P_bot, chi, None if prev is None else prev[k][0]
        )
        out.append((P_top, P_bot, U, S_keep, Vh))
        env_k, a_k = rotate_env(env_k), rotate_a(a_k)
    return out


def _renormalised_corner(env_k, a_k, P_top_k, P_bot_next, chi):
    """Paper Eq. 68: the quadrant projected on *both* open legs.

    The corner sits in the ``C1`` slot of move ``k`` (above its vertical
    bond, so ``P_top`` of move ``k``) and in the ``C4`` slot of move
    ``k+1`` (below its bond in the rotated frame, so ``P_bot`` of move
    ``k+1``).  Both projectors act on the same upper-left quadrant.
    """
    n = env_k.C1.shape[0] * a_k.shape[0]
    Q = _upper_left_quadrant(env_k, a_k).reshape(n, n)
    del chi
    return jnp.einsum("br,rd,da->ab", P_bot_next, Q, P_top_k)


def _renormalised_edge(env_k, a_k, P_top_k, P_bot_k, chi):
    """Paper Eq. 69: the edge absorbs one ``a`` and is projected on both bonds."""
    _c1g, _c4g, t4g = _left_move_pieces(env_k, a_k)
    del chi
    return jnp.einsum("ui,ixj,jd->dxu", P_bot_k, t4g, P_top_k)


def sweep(env: AsymEnv, a: jax.Array, chi: int, prev=None):
    """One simultaneous CTMRG sweep: all four directions from one environment.

    Returns ``(env, projectors)``; the projectors feed the next sweep's
    gauge alignment.
    """
    projs = all_projectors(env, a, chi, prev)
    corners: list = [None] * 4
    edges: list = [None] * 4
    env_k, a_k = env, a
    for k in range(4):
        P_top_k, P_bot_k = projs[k][0], projs[k][1]
        P_bot_next = projs[(k + 1) % 4][1]
        corners[_unrotate_index(1, k) - 1] = _normalize(
            _renormalised_corner(env_k, a_k, P_top_k, P_bot_next, chi)
        )
        edges[_unrotate_index(4, k) - 1] = _normalize(
            _renormalised_edge(env_k, a_k, P_top_k, P_bot_k, chi)
        )
        env_k, a_k = rotate_env(env_k), rotate_a(a_k)
    return AsymEnv(*corners, *edges), projs


def _init_env(A: Tensor, chi: int) -> tuple[AsymEnv, jax.Array]:
    env_t = initialize_ctm_tensor_env(A, chi)
    a_t = _build_double_layer_tensor(A)
    labels = list(a_t.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    a = jnp.asarray(a_t.transpose(perm).todense())
    env = AsymEnv(
        C1=jnp.asarray(env_t.C1.todense()),
        C2=jnp.asarray(env_t.C2.todense()),
        C3=jnp.asarray(env_t.C3.todense()),
        C4=jnp.asarray(env_t.C4.todense()),
        T1=jnp.asarray(env_t.T1.todense()),
        T2=jnp.asarray(env_t.T2.todense()),
        T3=jnp.asarray(env_t.T3.todense()),
        T4=jnp.asarray(env_t.T4.todense()),
    )
    return env, a


def converge(
    A: Tensor,
    chi: int,
    *,
    max_iter: int = 200,
    conv_tol: float = 1e-12,
    min_iter: int = 4,
) -> tuple[AsymEnv, jax.Array, dict[str, Any]]:
    """Run sweeps until the corner spectra stop moving."""
    env, a = _init_env(A, chi)
    prev = None
    prev_projs = None
    residual = float("inf")
    converged = False
    iters = 0
    for it in range(int(max_iter)):
        env, prev_projs = sweep(env, a, chi, prev_projs)
        iters = it + 1
        # Element-wise, not spectral.  Corner *singular values* are invariant
        # under independent rotations of each bond, so a spectral criterion
        # calls convergence while the tensors are still moving — and the
        # characteristic equations compare tensors.
        cur = tuple(t / (jnp.linalg.norm(t) + 1e-300) for t in env)
        if prev is not None and all(c.shape == q.shape for c, q in zip(cur, prev)):
            residual = float(max(jnp.max(jnp.abs(c - q)) for c, q in zip(cur, prev)))
            if iters >= min_iter and residual < conv_tol:
                converged = True
                break
        prev = cur
    return env, a, {"iters": iters, "residual": residual, "converged": converged}


# ---------------------------------------------------------------------------
# Characteristic equations (paper Eqs. 76-80)
# ---------------------------------------------------------------------------


class AsymRoot(NamedTuple):
    """Root variables plus the constants held fixed while differentiating."""

    env: AsymEnv
    u: tuple  # 4 x ((n - chi) x chi)
    s: tuple  # 4 x (chi,)
    v: tuple  # 4 x (chi x (n - chi))
    U_star: tuple
    U_perp: tuple
    Vh_star: tuple
    Vh_perp: tuple
    s_star_inv: tuple

    @property
    def y(self):
        return (self.env, self.u, self.s, self.v)


def asym_characteristic_residual(y, a: jax.Array, consts: AsymRoot, chi: int):
    """Evaluate ``F(y, p)`` for all four directions (paper Eqs. 76-80).

    Returns a pytree matching ``y``, so the system is square: 4 corners,
    4 edges, and per direction one ``u``, one ``S`` and one ``v``.
    """
    env, u_all, s_all, v_all = y

    # Pass 1: every projector from the same y, mirroring ``all_projectors``.
    P_top, P_bot, M_all = [], [], []
    env_k, a_k = env, a
    for k in range(4):
        U = consts.U_star[k] + consts.U_perp[k] @ u_all[k]  # Eq. 71
        Vh = consts.Vh_star[k] + v_all[k] @ consts.Vh_perp[k]  # Eq. 72
        M_all.append((half_infinite_environment(env_k, a_k), U, Vh))
        pt, pb = _fishman_projectors(env_k, a_k, U, s_all[k], Vh, chi)
        P_top.append(pt)
        P_bot.append(pb)
        env_k, a_k = rotate_env(env_k), rotate_a(a_k)

    # Pass 2: residuals.
    corners: list = [None] * 4
    edges: list = [None] * 4
    R_u, R_S, R_v = [None] * 4, [None] * 4, [None] * 4
    env_k, a_k = env, a
    for k in range(4):
        M, U, Vh = M_all[k]
        s_inv = consts.s_star_inv[k]
        core = U.conj().T @ M @ Vh.conj().T
        lam_S = jnp.vdot(s_all[k], core).real

        # Eq. 79 as the full chi x chi block, not just its diagonal.  With U
        # and Vh pinned to their null-space variations (Eq. 88) the in-space
        # rotation has nowhere to go except into S, so S has to be free to
        # leave the diagonal in the reverse pass.  Phase 0's analogue is
        # "treat C as a generic complex Hermitian matrix".
        R_S[k] = core - lam_S * s_all[k]
        R_u[k] = (consts.U_perp[k].conj().T @ M @ Vh.conj().T) @ s_inv - lam_S * u_all[
            k
        ]
        R_v[k] = (
            s_inv @ (U.conj().T @ M @ consts.Vh_perp[k].conj().T) - lam_S * v_all[k]
        )

        C_new = _renormalised_corner(env_k, a_k, P_top[k], P_bot[(k + 1) % 4], chi)
        C_cur = env_k.C1
        lam_C = jnp.vdot(C_cur, C_new).real
        corners[_unrotate_index(1, k) - 1] = C_new - lam_C * C_cur

        E_new = _renormalised_edge(env_k, a_k, P_top[k], P_bot[k], chi)
        E_cur = env_k.T4
        lam_E = jnp.vdot(E_cur, E_new).real
        edges[_unrotate_index(4, k) - 1] = E_new - lam_E * E_cur

        env_k, a_k = rotate_env(env_k), rotate_a(a_k)

    return (AsymEnv(*corners, *edges), tuple(R_u), tuple(R_S), tuple(R_v))


def asym_root_parametrize(
    env: AsymEnv,
    a: jax.Array,
    chi: int,
    *,
    pinv_rtol: float = 1e-10,
    polish_steps: int = 40,
    polish_tol: float = 1e-10,
) -> tuple[AsymRoot, float]:
    """Extract ``y* = ({C}, {E}, 0, {S*}, 0)`` and the frozen isometries.

    Each environment tensor is rescaled to unit Frobenius norm so that the
    ``λ`` defined as an inner product in Eqs. 76-77 really is the
    eigenvalue those equations need.  Rescaling is harmless: the energy is
    a ratio with equal numbers of corners and edges above and below.
    """
    best: tuple[AsymRoot, float] | None = None
    prev_projs = None
    for _step in range(max(int(polish_steps), 1)):
        env = AsymEnv(*[t / (jnp.linalg.norm(t) + 1e-300) for t in env])
        projs = all_projectors(env, a, chi, prev_projs)
        prev_projs = projs
        U_star, U_perp, Vh_star, Vh_perp, s_list, s_inv = [], [], [], [], [], []
        for k in range(4):
            _pt, _pb, U, S_keep, Vh = projs[k]
            U_star.append(U[:, :chi])
            U_perp.append(U[:, chi:])
            Vh_star.append(Vh[:chi])
            Vh_perp.append(Vh[chi:])
            s_list.append(S_keep)
            diag = jnp.diag(S_keep).real
            cutoff = pinv_rtol * jnp.max(diag)
            inv_diag = jnp.where(
                diag > cutoff, 1.0 / jnp.where(diag > cutoff, diag, 1.0), 0.0
            )
            # Constant right/left preconditioner for Eqs. 78 and 80; diagonal
            # because the root S* is.
            s_inv.append(jnp.diag(inv_diag).astype(S_keep.dtype))

        n = env.C1.shape[0] * a.shape[0]
        root = AsymRoot(
            env=env,
            u=tuple(jnp.zeros((n - chi, chi), dtype=env.C1.dtype) for _ in range(4)),
            s=tuple(s_list),
            v=tuple(jnp.zeros((chi, n - chi), dtype=env.C1.dtype) for _ in range(4)),
            U_star=tuple(U_star),
            U_perp=tuple(U_perp),
            Vh_star=tuple(Vh_star),
            Vh_perp=tuple(Vh_perp),
            s_star_inv=tuple(s_inv),
        )
        R = asym_characteristic_residual(root.y, a, root, chi)
        residual = float(
            jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jax.tree.leaves(R)))
        )
        if best is None or residual < best[1]:
            best = (root, residual)
        if residual <= polish_tol:
            break
        env, prev_projs = sweep(env, a, chi, projs)

    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Energy and gradient
# ---------------------------------------------------------------------------


def _to_ctm_env(env: AsymEnv, template):
    return type(template)(
        **{
            name: DenseTensor(getattr(env, name), getattr(template, name).indices)
            for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
        }
    )


def asym_energy(A: Tensor, env: AsymEnv, template, gate) -> jax.Array:
    return compute_energy_ctm_tensor(A, _to_ctm_env(env, template), gate)


def asym_root_implicit_energy_and_grad(
    A: Tensor,
    gate,
    *,
    chi: int = 8,
    max_iter: int = 200,
    conv_tol: float = 1e-12,
    min_iter: int = 4,
    polish_steps: int = 40,
    polish_tol: float = 1e-10,
    solve_tol: float = 1e-8,
    solve_maxiter: int = 400,
    solve_restart: int = 30,
    root_residual_warn: float = 1e-6,
    return_diagnostics: bool = False,
):
    """Energy and ``dE/dA`` for a 1x1 unit cell via asymmetric root implicit AD."""
    from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint

    if isinstance(A, SymmetricTensor):
        raise TypeError("Asymmetric root implicit AD is dense-only (#715 Phase 3).")

    A_const = DenseTensor(jax.lax.stop_gradient(A.todense()), A.indices)
    env, a_arr, meta = converge(
        A_const, chi, max_iter=max_iter, conv_tol=conv_tol, min_iter=min_iter
    )
    root, root_residual = asym_root_parametrize(
        env, a_arr, chi, polish_steps=polish_steps, polish_tol=polish_tol
    )
    if root_residual > root_residual_warn:
        warnings.warn(
            f"Asymmetric root implicit AD: ‖F(y*)‖ = {root_residual:.3e} exceeds "
            f"{root_residual_warn:.1e}; the implicit-function gradient is "
            "correspondingly inaccurate (paper Fig. 1).",
            RuntimeWarning,
            stacklevel=2,
        )

    template = initialize_ctm_tensor_env(A_const, chi)
    A_data = jnp.asarray(A.todense())

    def energy_of(a_data, env_arrays):
        A_live = DenseTensor(a_data, A.indices)
        return asym_energy(A_live, env_arrays, template, gate)

    energy, vjp_energy = jax.vjp(energy_of, A_data, root.env)
    grad_direct, env_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    y_bar = (
        env_bar,
        tuple(jnp.zeros_like(x) for x in root.u),
        tuple(jnp.zeros_like(x) for x in root.s),
        tuple(jnp.zeros_like(x) for x in root.v),
    )

    def F_of_y(y):
        return asym_characteristic_residual(y, a_arr, root, chi)

    _, vjp_y = jax.vjp(F_of_y, root.y)
    F_bar, solve_resid = _solve_root_adjoint(
        lambda v: vjp_y(v)[0],
        y_bar,
        tol=solve_tol,
        maxiter=solve_maxiter,
        restart=solve_restart,
    )

    def F_of_p(a_data):
        A_live = DenseTensor(a_data, A.indices)
        a_t = _build_double_layer_tensor(A_live)
        labels = list(a_t.labels())
        perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
        a_live = a_t.transpose(perm).todense()
        return asym_characteristic_residual(root.y, a_live, root, chi)

    _, vjp_p = jax.vjp(F_of_p, A_data)
    grad = grad_direct - vjp_p(F_bar)[0]

    if return_diagnostics:
        return (
            energy,
            grad,
            {
                **meta,
                "root_residual": root_residual,
                "adjoint_residual": float(solve_resid),
            },
        )
    return energy, grad
