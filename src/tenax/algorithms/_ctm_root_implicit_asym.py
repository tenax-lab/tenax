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

Where this stands (#715 Phase 1, complete)
-----------------------------------------
Phase 1 is done for the dense asymmetric 1x1 cell, real and complex.  Measured
gradient parity against finite differences is ``< 1e-5`` real
(``test_gradient_parity_needs_the_modified_variables``) and ``< 1e-9`` complex
(``test_gradient_parity_for_a_complex_state``), with ``||F(y*)|| = 2.5e-16``.

Two ingredients get it there, and both were arrived at the hard way, so neither
is worth re-litigating from the equations alone:

``S`` is a general complex chi x chi matrix, and the projectors use a genuine
matrix inverse square root (Denman-Beavers, so no decomposition enters ``F``).
That is required, not optional: with a *diagonal* ``S`` the in-space rotation
of the isometries has nowhere to go, the projector closure breaks at first
order, and Eq. 88's null-space restriction then discards a physical
contribution instead of a gauge one — the gradient came out 120% wrong.
Promoting ``S`` alone brings it to 1-2.5%.

The rest of Eqs. 73-82 closes that residual: the modified corners and edges
carrying ``s`` explicitly on the bonds (:func:`_modified_env`) and the quartic
roots ``s^L = (S S†)^-1/4``, ``s^R = (S† S)^-1/4`` on the *cut legs* of that
environment (:func:`_quartic_root`, :func:`_inv_quartic_root`), assembled by
:func:`asym_characteristic_residual_covariant`.  Putting those roots inside the
projectors instead was tried and is worse (20%): they are equivariant under
independent left/right rotations, but their product is not ``S^-1``, so the
closure breaks for a non-diagonal ``S``.  See
``docs/plans/2026-07-29-715-phase1-modified-variables.md``.

What is *not* done is #715 as a whole: none of this is wired into the
production gradient path.  ``optimize_gs_ad`` still differentiates the CTM
fixed point through :mod:`tenax.algorithms._ctm_energy_ad`, SVD backward and
all.  Phase 3 (:mod:`tenax.algorithms._ctm_root_implicit_symmetric`) is the
phase that would pay off #566/#687 and is blocked on #731 (8.4 GB peak in the
GMRES solve at D=2, chi=4).

Complex states (#721)
---------------------
Everything above holds for a complex site tensor, but three things about the
real case are accidents of real data and were fixed only in #721.  ``S`` is real
because ``svd`` returns a real spectrum, not because the equations want it real,
and the reverse pass needs it widened to the environment's dtype.  A converged
environment carries the bond gauge of the sweep chain that produced it, and
re-pinning from cold — which a real state's sign gauge survives and a complex
state's continuous phase does not — leaves ``y*`` describing a different
environment; :func:`asym_root_parametrize` takes the chain now.  And ``∂_y F``
is genuinely singular: an independent phase on each environment tensor is an
exact null direction, so the root is a gauge orbit rather than a point.  That
last one is not a defect — see :func:`asym_root_implicit_energy_and_grad`.

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

from tenax.algorithms._ad_primitives import (
    _check_root_residual_policy,
    _report_root_residual,
)
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


def _quartic_root(A: jax.Array, n_iter: int = 24) -> jax.Array:
    """``A^1/4`` for Hermitian positive ``A``, as ``(A^1/2)^1/2``."""
    return _denman_beavers(_denman_beavers(A, n_iter)[0], n_iter)[0]


# ---------------------------------------------------------------------------
# The y <-> x map (paper Eq. 82)
# ---------------------------------------------------------------------------
#
# The characteristic equations are written in *modified* corners and edges,
# which carry the inverse singular values explicitly on their environment
# legs.  Only that form transforms covariantly under the eight gauge
# unitaries of Eq. 84, and only then is the null-space restriction of Eq. 88
# — hence holding ``U*`` and ``V*`` constant — legitimate.
#
# The forward contraction is untouched: it produces the regular ``x``, and
# these two functions move between the two descriptions.  Crucially the
# *adjoint* has to travel the same way, which is what makes ``S̆`` nonzero;
# see :func:`asym_root_implicit_energy_and_grad`.
#
# Direction bookkeeping at a 1x1 unit cell (paper indices reduce to pure
# direction arithmetic, cf. PEPSKit ``_prev_coordinate``): corner ``k`` has
# its first environment leg on direction ``k-1``'s edge and its second on
# direction ``k``'s, and edge ``k`` takes ``S_k`` on both chi legs.


def _corner_leg_directions(k: int) -> tuple[int, int]:
    """``(prev, own)`` singular-value slots for corner ``k`` (0-based)."""
    return (k - 1) % 4, k


def _apply_corner_roots(C: jax.Array, left: jax.Array, right: jax.Array) -> jax.Array:
    return left @ C @ right


def _apply_edge_roots(E: jax.Array, left: jax.Array, right: jax.Array) -> jax.Array:
    """Absorb on both chi legs of ``E[chi_in, d2, chi_out]``."""
    return jnp.einsum("ai,ixj,jb->axb", left, E, right)


def _map_env_roots(env: AsymEnv, roots: tuple, *, normalize: bool) -> AsymEnv:
    """Absorb ``roots[k]`` onto corner/edge environment legs (Eq. 82)."""
    corners, edges = [], []
    for k in range(4):
        prev_dir, own_dir = _corner_leg_directions(k)
        C = _apply_corner_roots(
            getattr(env, f"C{k + 1}"), roots[prev_dir], roots[own_dir]
        )
        E = _apply_edge_roots(getattr(env, f"T{k + 1}"), roots[k], roots[k])
        if normalize:
            C = C / (jnp.linalg.norm(C) + 1e-300)
            E = E / (jnp.linalg.norm(E) + 1e-300)
        corners.append(C)
        edges.append(E)
    return AsymEnv(*corners, *edges)


def remove_inverse_roots(env: AsymEnv, S: tuple) -> AsymEnv:
    """Regular ``x`` -> modified ``(C̃, Ẽ)``: multiply by ``sqrt(S)``.

    Undoes the inverse square roots the forward projectors put on the
    environment legs, leaving the modified tensors of Eq. 82.
    """
    roots = tuple(_denman_beavers(s)[0] for s in S)
    return _map_env_roots(env, roots, normalize=True)


def absorb_inverse_roots(env_tilde: AsymEnv, S: tuple) -> AsymEnv:
    """Modified ``(C̃, Ẽ)`` -> regular ``x``: multiply by ``sqrt(S^-1)``.

    This is the differentiable direction.  ``S`` enters here, so the energy
    — which is evaluated on the regular environment — depends on ``S``
    through this map, and ``S̆`` picks that up.  A matrix square root of the
    inverse is used rather than a diagonal power because ``dS`` is a general
    non-diagonal matrix in the reverse pass.
    """
    roots = tuple(_inv_sqrt(s) for s in S)
    return _map_env_roots(env_tilde, roots, normalize=True)


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


def _rank_capped_spectrum(s: jax.Array, chi: int, *, rel_floor: float | None = None):
    """Truncate to ``chi`` and clamp the numerically-null tail (#772).

    Returns ``(s_capped, usable_rank)``.  ``usable_rank`` counts the retained
    directions that carry real weight; when it is below ``chi`` the extra
    directions are noise and ``chi`` exceeds what the state's environment
    supports.

    **Why the clamp is raised, not lowered.**  The covariant characteristic
    equations carry ``S^-1`` on *both* corner legs (:func:`_modified_env`) plus
    the Eq. 73 quartic roots on the isometries, so their dependence on ``S`` is
    cubic.  A retained direction whose weight is below ``eps^(1/3)`` relative to
    the largest cannot be resolved in the working precision, and retaining one
    makes ``‖F(y*)‖`` explode — measured 5.7e+05 with NaN gradients on a
    physical simple-update state at ``chi=6`` (#772).

    The distinction that matters is *small-but-real* versus *floored-and-empty*.
    A direction at or below the clamp is harmless: the environment carries
    approximately no weight there, so the large ``S^-1`` multiplies
    approximately nothing.  A direction carrying genuine but tiny weight is what
    breaks the equations.  Raising the old ``1e-12`` clamp to ``eps^(1/3)``
    converts the second kind into the first, which is why it *improves*
    accuracy rather than costing it: measured ``‖F‖`` 5.7e+05 -> 1.9e-05 at
    ``chi=6``, and the gradient goes from NaN to finite and FD-correct to 6e-08.

    The clamp is two-sided in its effect and must not be raised further: pushing
    it above the genuinely-weighted directions makes ``S`` disagree with the
    environment it came from, which breaks well-conditioned states that were
    fine (a ``1e-2`` clamp takes the random-tensor fixture from 2e-14 to 4e-03).
    ``eps^(1/3)`` sits below any physically meaningful weight and above the
    unresolvable band; ``eps^(1/2)`` is measurably too low (1.3e+04 at chi=6).

    This also subsumes the guard the old constant existed for — early sweeps
    start from a rank-deficient near-identity environment, and a singular ``S``
    would make :func:`_inv_sqrt` produce NaNs.  The clamp is strictly positive,
    so that cannot happen.
    """
    s_k = s[:chi]
    if rel_floor is None:
        rel_floor = float(jnp.finfo(s_k.dtype).eps ** (1.0 / 3.0))
    cut = rel_floor * s_k[0]
    usable_rank = jnp.sum(s_k > cut)
    return jnp.maximum(s_k, cut), usable_rank


# #779: how far a *clamped* cut can be polished, and no further.
#
# Once :func:`_rank_capped_spectrum` clamps a retained direction, ``y*`` cannot
# satisfy the characteristic equations below the clamp level however long it is
# polished — the clamp is part of the map being solved.  Each clamped direction
# contributes about that level to the residual norm, and the norm adds them in
# quadrature, so the floor grows as the square root of their number:
#
#     ‖F(y*)‖  ≈  C · eps^(1/3) · sqrt(chi - usable_rank)
#
# Measured on the D=2 simple-update Heisenberg state, whose environment
# supports usable_rank=3 at *every* chi (so every production chi over-provisions
# and the clamp always fires):
#
#     chi            4        6        8       10       12       16       24
#     forward   1.26e-6  2.81e-6  3.77e-6  4.53e-6  5.18e-6  6.28e-6  8.04e-6
#     covariant 8.49e-6  1.91e-5  2.57e-5  3.09e-5  3.53e-5  4.28e-5  5.48e-5
#
# Dividing by sqrt(chi-3) converges to C = 0.29 (forward) and C = 1.95
# (covariant).  The law was checked by prediction rather than by fit: from the
# chi<=12 rows alone, chi=16 was predicted at 6.2e-6 / 4.3e-5 and measured
# 6.279e-6 / 4.280e-5, and chi=24 predicted at 7.9e-6 / 5.4e-5 and measured
# 8.042e-6 / 5.482e-5 -- both within 1.5%.
#
# The covariant coefficient is close to state-INDEPENDENT, which is what makes
# this a property of the clamp rather than a fit to one fixture: a squashed
# random state (usable_rank=1) gives 1.03/1.11/1.14e-5 against this state's
# 1.18e-5, and at chi=24 a squashed state at usable_rank=3 reads 5.417e-5
# against this state's 5.482e-5 -- 1.2% apart on entirely different physics.
#
# The factor carries ~4x margin over the measured covariant coefficient so one
# tolerance serves both residuals (their ratio is *not* universal — 6.8 on this
# state but 200-370 on the others, so a per-residual factor would be fitting
# noise).
_CLAMPED_RESIDUAL_FACTOR = 8.0


def _root_residual_tolerance(base_tol, usable_rank, chi, dtype) -> float:
    """Residual tolerance for a cut of rank ``usable_rank`` out of ``chi``.

    Returns ``base_tol`` unchanged while the clamp is inert — a full-rank cut
    has no intrinsic floor and reaches roundoff (measured 1.5e-16 forward /
    3.5e-14 covariant on a well-conditioned random state), so the strict
    default is both meaningful and achievable and must not be weakened there.

    When the clamp has fired the floor above applies and the tolerance follows
    it.  This only ever *relaxes*: it returns at least ``base_tol``.

    **This gate does not measure gradient quality**, and must not be described
    as if it did.  Measured against a converged directional finite difference,
    the residual mispredicts in both directions: the simple-update state is
    rejected by the strict forward gate at 1.26e-6 while its gradient is
    correct to 3.1e-8, and a squashed random state *passes* it at 1.7e-8 while
    its gradient is off by 4.4e-5.  What the residual reports is whether
    ``y*`` solves the equations, which is a different question — see #785.
    """
    if int(usable_rank) >= int(chi):
        return float(base_tol)
    eps13 = float(jnp.finfo(dtype).eps ** (1.0 / 3.0))
    n_clamped = max(1, int(chi) - int(usable_rank))
    floor = _CLAMPED_RESIDUAL_FACTOR * eps13 * float(n_clamped) ** 0.5
    return max(float(base_tol), floor)


def all_projectors(env: AsymEnv, a: jax.Array, chi: int, prev=None):
    """Decompose the cut in all four directions, from the *same* environment.

    Paper Eq. 65 and "their corresponding rotated versions": every projector
    in a sweep is built from one environment, and every corner and edge is
    then renormalised simultaneously.  This matters — a sequential
    (Gauss-Seidel) sweep, where move ``k+1`` sees the output of move ``k``,
    has a fixed point that does *not* satisfy Eqs. 76-77, because those
    equations evaluate all four moves at the same ``y``.

    Each entry is ``(P_top, P_bot, U, S_keep, Vh, usable_rank)``.  The trailing
    ``usable_rank`` is what :func:`_rank_capped_spectrum` resolved for that
    direction; it used to be discarded here.  It is load-bearing for #779: once
    a retained direction is clamped, ``y*`` cannot satisfy the characteristic
    equations to better than the clamp level no matter how long it is polished,
    so the root-residual gate has to know whether the clamp fired before it can
    decide what residual is acceptable.
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
        s_k, usable_rank = _rank_capped_spectrum(s, chi)
        # Cast to the environment's dtype.  ``S`` is a *variable* of the
        # characteristic equations, and the reverse pass needs it free to leave
        # the reals for the same reason Eq. 79 needs it free to leave the
        # diagonal: ``R_S = core/λ - S`` compares it against a complex
        # contraction, and a cotangent inherits its primal's dtype, so a real
        # ``S`` hands the adjoint solve a ``float64`` block where ``F``
        # produced ``complex128`` and JAX refuses outright (#721).
        # ``jnp.linalg.svd`` only ever returns a real spectrum, hence the cast
        # rather than a genuinely complex construction.
        S_keep = jnp.diag(s_k / (jnp.linalg.norm(s_k) + 1e-300)).astype(M.dtype)
        P_top, P_bot = _fishman_projectors(env_k, a_k, U, S_keep, Vh, chi)
        U, Vh, P_top, P_bot = _pin_bond_gauge(
            U, Vh, P_top, P_bot, chi, None if prev is None else prev[k][0]
        )
        # Kept as the raw traced scalar, NOT int(): ``all_projectors`` runs
        # inside ``sweep``, which callers put under ``jax.jit``, and
        # concretising here raises ConcretizationTypeError.  The conversion
        # belongs in ``asym_root_parametrize``, which is already un-jittable
        # (it does ``float(residual)`` and breaks out of a Python loop).
        out.append((P_top, P_bot, U, S_keep, Vh, usable_rank))
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
    # Numerically a no-op on this initialiser (symmetric C4, palindromic
    # T3/T4), but this is the other side of the same convention boundary as
    # ``_to_ctm_env``; see :func:`swap_env_convention`.
    return swap_env_convention(env), a


def converge(
    A: Tensor,
    chi: int,
    *,
    max_iter: int = 200,
    conv_tol: float = 1e-12,
    min_iter: int = 4,
    return_projectors: bool = False,
):
    """Run sweeps until the corner spectra stop moving.

    With ``return_projectors`` the final projector set comes back as a fourth
    element.  That is not a diagnostic: the converged environment sits in the
    bond gauge of the chain that built it, and
    :func:`asym_root_parametrize` needs the same chain to extract a root in that
    gauge rather than re-pinning a different one — see its docstring.
    """
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
    meta: dict[str, Any] = {
        "iters": iters,
        "residual": residual,
        "converged": converged,
    }
    if return_projectors:
        return env, a, meta, prev_projs
    return env, a, meta


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


def asym_root_to_covariant_convention(root: AsymRoot) -> AsymRoot:
    """Relabel a forward-convention root into the one paper §V.3 uses.

    The forward sweep truncates with the *left* half of the plane, gluing the
    upper-left quadrant to the lower-left one at the same rotation.  §V.3
    truncates with the *upper* half, gluing enlarged corner ``k`` to enlarged
    corner ``k+1``.  Those are the same truncation, not two different ones:

        _lower_left_quadrant(env_k, a_k) == _upper_left_quadrant(env_{k-1}, a_{k-1})

    holds exactly (1e-16), and therefore

        M_left(k) == M_up(k-1).T .

    Transposing a decomposition exchanges its isometries — ``(U S Vh).T =
    Vh.T S.T U.T``, and both factors stay isometric, since ``Vh Vh† = 1``
    gives ``(Vh.T)† Vh.T = conj(Vh Vh†) = 1`` — so the §V.3 data at direction
    ``j`` is the forward data at rotation ``j+1`` with ``U`` and ``Vh``
    swapped and every block transposed.  Plain transpose, not adjoint: no
    conjugation enters.

    The environment itself is untouched; only which cut each ``S`` belongs to
    changes.  Feed the result to :func:`remove_inverse_roots` and
    :func:`asym_characteristic_residual_covariant`, which are §V.3-indexed;
    :func:`asym_characteristic_residual` wants the un-relabelled root.
    """

    def shift(xs: tuple) -> tuple:
        return tuple(xs[(j + 1) % 4] for j in range(4))

    def shiftT(xs: tuple) -> tuple:
        return tuple(x.T for x in shift(xs))

    return AsymRoot(
        env=root.env,
        u=shiftT(root.v),
        s=shiftT(root.s),
        v=shiftT(root.u),
        U_star=shiftT(root.Vh_star),
        U_perp=shiftT(root.Vh_perp),
        Vh_star=shiftT(root.U_star),
        Vh_perp=shiftT(root.U_perp),
        s_star_inv=shiftT(root.s_star_inv),
    )


def _covariant_pieces(consts: AsymRoot, S_all: tuple, u_all: tuple, v_all: tuple):
    """Per-direction covariant building blocks (paper Eqs. 71-75).

    ``s = S^-1``, and the Eq. 73 roots go on the *isometries*, not on the
    projectors.  Their placement is fixed by the Eq. 87 transformation laws
    rather than chosen: with ``U -> U Q_L†``, ``V -> Q_R V`` and
    ``s -> Q_R s Q_L†``,

        (s† s)^1/4 -> Q_L (s† s)^1/4 Q_L†,
        (s s†)^1/4 -> Q_R (s s†)^1/4 Q_R†,

    which is what makes them legitimate factors to attach to the isometries.

    They attach to the ``n = chi * d2`` *cut* leg, on its ``chi`` sub-leg —
    not to the outer ``chi`` index.  That is the leg §V.3 describes as
    carrying "a dangling sqrt(s) on outer legs of the edge tensors that are
    being cut", and it is what the reference's ``absorb_right`` reaches
    (``domainind(P)[1]``, the first of the split ``(chi, D, D)`` triple).
    Attaching to the outer ``chi`` instead is shape-legal for ``Ud`` and
    ``Vd`` but not for the null-space blocks, which is the giveaway.

    The direction offsets follow the reference's
    ``_leftvec_invfroot_indices`` / ``_rightvec_invfroot_indices``, which at a
    1x1 unit cell reduce to ``k-1`` and ``k+1``: the cut legs belong to the
    neighbouring directions' edges.
    """
    s_all = tuple(jnp.linalg.inv(S) for S in S_all)
    root_L = tuple(_quartic_root(s.conj().T @ s) for s in s_all)
    root_R = tuple(_quartic_root(s @ s.conj().T) for s in s_all)

    d2 = consts.U_star[0].shape[0] // S_all[0].shape[0]
    eye_d2 = jnp.eye(d2, dtype=consts.U_star[0].dtype)
    # chi is the slow index of the fused (chi, d2) cut leg.
    K_L = tuple(jnp.kron(r, eye_d2) for r in root_L)
    K_R = tuple(jnp.kron(r, eye_d2) for r in root_R)

    Ud, Vd, ULd, VRd = [], [], [], []
    for k in range(4):
        km, kp = (k - 1) % 4, (k + 1) % 4
        U = consts.U_star[k] + consts.U_perp[k] @ u_all[k]  # Eq. 71
        Vh = consts.Vh_star[k] + v_all[k] @ consts.Vh_perp[k]  # Eq. 72
        chi = S_all[k].shape[0]
        Ud.append(U[:, :chi].conj().T @ K_L[km])
        Vd.append(K_R[kp] @ Vh[:chi].conj().T)
        ULd.append(consts.U_perp[k].conj().T @ K_L[km])
        VRd.append(K_R[kp] @ consts.Vh_perp[k].conj().T)
    return s_all, tuple(Ud), tuple(Vd), tuple(ULd), tuple(VRd)


def _modified_env(env_tilde: AsymEnv, s_all: tuple) -> AsymEnv:
    """``iCi = s_{k-1} C̃_k s_k`` with the edges left alone.

    The full inverse goes on *both* corner legs (reference ``iCi``), which is
    what puts the singular values explicitly into the contraction environment
    so that it transforms covariantly.  Edges already carry their ``s`` from
    the Eq. 82 map.
    """
    corners = []
    for k in range(4):
        prev_dir, own_dir = _corner_leg_directions(k)
        corners.append(
            s_all[prev_dir] @ getattr(env_tilde, f"C{k + 1}") @ s_all[own_dir]
        )
    edges = [getattr(env_tilde, f"T{k + 1}") for k in range(4)]
    return AsymEnv(*corners, *edges)


def asym_characteristic_residual_covariant(y, a: jax.Array, consts: AsymRoot, chi: int):
    """``F(y, p)`` in the covariant parametrisation of paper §V.3.

    ``y = (env_tilde, u, S, v)`` with *modified* corners and edges, so that
    holding ``U*`` and ``V*`` constant is licensed by Eq. 88.  Normalisation
    follows the reference implementation's ``X'/lambda - X`` rather than
    ``X' - lambda X``; the two differ by a row scaling of ``F``, which leaves
    the implicit gradient invariant but not the Jacobian conditioning.  It
    does rely on every tensor in ``y`` having unit Frobenius norm, which
    :func:`asym_root_parametrize` arranges.

    ``lambda`` is deliberately *not* real-projected.  ``dot(X, X')`` is a
    genuinely complex quantity for a complex state, and since ``F`` normalises
    by it rather than subtracting it, dropping the phase is a different ``F``,
    not a rescaled one.

    How visible that is depends on the gauge, which an earlier version of this
    note did not say.  At the root :func:`asym_root_parametrize` polishes to,
    the bond phases are aligned and all twelve ``lambda`` come out real to
    ``|arg| < 8e-16``, so real-projecting there is a no-op (6.50e-14 either
    way).  The eight environment phases are exact null directions, so phasing
    ``env_tilde`` lands on an equally valid root with ``|arg(lambda)|`` up to
    3.0 — and there the same edit moves ``‖F‖`` to 3.6e0 and 1.2e1.
    ``test_the_dense_root_carries_lambdas_phase_rather_than_dropping_it``
    measures both halves.

    The assembly mirrors the reference implementation
    (``contract_asymmetric_characteristic_equation``, ``Val{:implicit}``)
    rather than the forward sweep in :func:`asym_characteristic_residual`,
    and the two use *different halves of the plane*:

    * The forward sweep truncates with the left half, gluing the upper-left
      quadrant to the lower-left one at the same rotation.
    * §V.3 truncates with the upper half, gluing enlarged corner ``k`` to
      enlarged corner ``k+1``.  So there is only one quadrant primitive here,
      and ``_lower_left_quadrant`` is not used at all.

    Every step below is pinned numerically against the reference's own dumped
    fixed point (``docs/plans/reference/718-dump.jl``): the enlarged corners
    to 8e-16, the projector pair to 5e-16, and all five residual blocks to
    2e-12 — the same order as the reference's ``|F|`` at its own root.
    """
    env_tilde, u_all, S_all, v_all = y
    s_all, Ud, Vd, ULd, VRd = _covariant_pieces(consts, S_all, u_all, v_all)
    env_mod = _modified_env(env_tilde, s_all)

    d2 = a.shape[0]
    n = chi * d2

    # Pass 1: one enlarged corner per direction, the ``s`` that sits on the
    # leg being cut, and the column the renormalised edge is built from.
    EC, K_is, cols = [], [], []
    env_k, a_k = env_mod, a
    for k in range(4):
        # The reference orders an enlarged corner (cut | outer) while the
        # helper returns (outer | cut), hence the transpose.  Verified to
        # 7.7e-16 against its dumped ``EC`` by
        # ``docs/plans/reference/718-ecmap.py``.
        EC.append(_upper_left_quadrant(env_k, a_k).reshape(n, n).T)
        # ``s_k`` lands on the cut leg's chi sub-leg, not on the outer chi:
        # that is what the reference's ``absorb_right`` reaches
        # (``domainind(P)[1]``, the chi of the split ``(chi, D, D)`` triple).
        # ``chi`` is the slow index of the fused leg, so ``kron(s, I)``.
        K_is.append(jnp.kron(s_all[k], jnp.eye(d2, dtype=s_all[k].dtype)))
        # The same north edge that enters ``EC[k]``, with the sandwich
        # attached: (chi_l, a_l | a_d | chi_r, a_r).
        cols.append(jnp.einsum("xfy,fjlr->xljyr", env_k.T1, a_k).reshape(n, d2, n))
        env_k, a_k = rotate_env(env_k), rotate_a(a_k)

    # Pass 2: the half-infinite environment at direction ``k`` glues ``EC[k]``
    # to ``EC[k+1]`` across the cut, carrying exactly one ``s_k``.
    M_all, P_R, P_L = [], [], []
    for k in range(4):
        kp = (k + 1) % 4
        M_all.append(EC[k] @ K_is[k] @ EC[kp])
        P_R.append(Ud[k] @ EC[k] @ K_is[k])
        P_L.append(K_is[k] @ EC[kp] @ Vd[k])

    # Pass 3: residuals.  Both projectors of a corner belong to the *same*
    # enlarged corner, one per group, so the corner takes ``P_R[k-1]`` — the
    # partner of the cut that ``EC[k]``'s first group sits on.
    corners: list = [None] * 4
    edges: list = [None] * 4
    R_u, R_S, R_v = [None] * 4, [None] * 4, [None] * 4
    for k in range(4):
        km = (k - 1) % 4
        M_k, s_inv = M_all[k], consts.s_star_inv[k]

        core = Ud[k] @ M_k @ Vd[k]
        lam_S = jnp.vdot(S_all[k], core)
        R_S[k] = core / lam_S - S_all[k]
        R_u[k] = (ULd[k] @ M_k @ Vd[k]) @ s_inv / lam_S - u_all[k]
        R_v[k] = s_inv @ (Ud[k] @ M_k @ VRd[k]) / lam_S - v_all[k]

        # Direction ``k`` renormalises the corner and the edge that its own
        # enlarged corner is built from, i.e. ``C1``/``T1`` of the rotated
        # frame, which is index ``k+1`` unrotated.
        idx = _unrotate_index(1, k)
        C_new = P_R[km] @ EC[k] @ P_L[k]
        C_cur = getattr(env_tilde, f"C{idx}")
        corners[idx - 1] = C_new / jnp.vdot(C_cur, C_new) - C_cur

        E_new = jnp.einsum("ax,xjy,yb->ajb", P_R[k], cols[k], P_L[k])
        E_cur = getattr(env_tilde, f"T{idx}")
        edges[idx - 1] = E_new / jnp.vdot(E_cur, E_new) - E_cur

    return (AsymEnv(*corners, *edges), tuple(R_u), tuple(R_S), tuple(R_v))


def asym_root_parametrize(
    env: AsymEnv,
    a: jax.Array,
    chi: int,
    *,
    prev_projs=None,
    pinv_rtol: float = 1e-10,
    polish_steps: int = 40,
    polish_tol: float = 1e-10,
    return_usable_rank: bool = False,
) -> tuple[AsymRoot, float]:
    """Extract ``y* = ({C}, {E}, 0, {S*}, 0)`` and the frozen isometries.

    Each environment tensor is rescaled to unit Frobenius norm so that the
    ``λ`` defined as an inner product in Eqs. 76-77 really is the
    eigenvalue those equations need.  Rescaling is harmless: the energy is
    a ratio with equal numbers of corners and edges above and below.

    Pass ``prev_projs`` — :func:`converge`'s fourth return value — whenever the
    environment came from a sweep chain.  A converged environment carries that
    chain's bond gauge, and :func:`_pin_bond_gauge`'s cold start pins a
    *different* one, which leaves ``y*`` describing an environment it was not
    extracted from.  The polish loop is meant to absorb that, and for a real
    state it does in a single sweep, because there the bond gauge is a sign and
    one sweep reproduces it exactly.  For a complex state the gauge is a
    continuous phase, the warm alignment only ever recovers it to best fit, and
    the corner residual — the one equation that couples two directions'
    projectors, hence two bond phases — plateaus at 6e-5 instead of falling,
    while the edges converge geometrically past 1e-8.  Warm-started off the
    forward chain the same environment is a root to 6e-17 (#721).
    """
    best: tuple[AsymRoot, float] | None = None
    best_rank = chi
    for _step in range(max(int(polish_steps), 1)):
        env = AsymEnv(*[t / (jnp.linalg.norm(t) + 1e-300) for t in env])
        projs = all_projectors(env, a, chi, prev_projs)
        prev_projs = projs
        U_star, U_perp, Vh_star, Vh_perp, s_list, s_inv = [], [], [], [], [], []
        for k in range(4):
            _pt, _pb, U, S_keep, Vh, _rank = projs[k]
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
            # The binding direction: one clamped cut is enough to floor the
            # residual, so the minimum is what the #779 gate must react to.
            best_rank = min(int(p[5]) for p in projs)
        if residual <= polish_tol:
            break
        env, prev_projs = sweep(env, a, chi, projs)

    assert best is not None
    if return_usable_rank:
        return (*best, best_rank)
    return best


# ---------------------------------------------------------------------------
# Energy and gradient
# ---------------------------------------------------------------------------


def swap_env_convention(env: AsymEnv) -> AsymEnv:
    """Convert between this module's rotation-uniform convention and the CTM one.

    This module stores every tensor in the frame of its own direction, which is
    what makes :func:`rotate_env` a pure relabel: corner ``k`` is always
    ``(leg towards direction k-1, leg towards direction k)`` and edge ``k``
    always ``(towards the previous corner, physical, towards the next)``.  So
    the ring closes uniformly::

        C1.1-T1.0  T1.2-C2.0  C2.1-T2.0  T2.2-C3.0
        C3.1-T3.0  T3.2-C4.0  C4.1-T4.0  T4.2-C1.0

    :class:`~tenax.algorithms._ctm_tensor_env.CTMTensorEnv` is *not* uniform —
    per the connectivity documented on ``_rdm2x1_tensor``, it closes as::

        C1.0-T4.0  C1.1-T1.0  T1.2-C2.0  C2.1-T2.0
        T2.2-C3.0  C3.1-T3.2  T3.0-C4.1  C4.0-T4.2

    which is the same ring with ``C4`` transposed and ``T3``, ``T4`` reversed.
    Reinterpreting one as the other glues the network wrongly: at ``D=2``,
    ``chi=4`` on the test state it moved the energy by 2.8e-3 (1.5%), and it
    broke the per-bond gauge invariance of the energy on exactly the two bonds
    that touch ``C4`` — the ±2.121e-3 antisymmetry of #718, which looked like an
    unlicensed gauge in Eq. 88 but was this relabelling all along.

    The map is an involution, so the same function converts either way.  It is
    a no-op on :func:`initialize_ctm_tensor_env` output, whose ``C4`` is
    symmetric and whose ``T3``/``T4`` are palindromic, which is why the mismatch
    stayed invisible until the environment became genuinely asymmetric.
    """
    return env._replace(
        C4=env.C4.T,
        T3=jnp.transpose(env.T3, (2, 1, 0)),
        T4=jnp.transpose(env.T4, (2, 1, 0)),
    )


def _to_ctm_env(env: AsymEnv, template):
    env = swap_env_convention(env)
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
    on_root_residual: str = "raise",
    return_diagnostics: bool = False,
):
    """Energy and ``dE/dA`` for a 1x1 unit cell via asymmetric root implicit AD.

    Root implicit differentiation of paper §V: the environment is characterised
    by ``F(y, p) = 0`` in the modified variables ``y = (C̃, Ẽ, u, S, v)`` of
    Eqs. 76-80, and the gradient comes from Eq. 18 without back-propagating a
    single SVD.  That is the point of the construction — see #566 and #687 for
    the block-sparse SVD/eigh VJP compile wall and accuracy floor it avoids.

    Parity against explicit backprop through the same sweep is 4e-8 relative at
    ``D=2``, ``chi=4``.  It was 3.06e-2 until 2026-07-30, and #718 spent a long
    time hunting that in the characteristic equations; the cause turned out to
    be neither the equations nor Eq. 88's null-space restriction but the
    environment convention at the *energy* boundary — see
    :func:`swap_env_convention`.

    A complex state reaches 2.5e-12 against the same reference at 40 sweeps
    (#721).  Compare only against a reference that has actually converged: the
    explicit map is truncated, and this state needs 53 forward iterations where
    the real one needs 25, so at 12 sweeps the reference is itself 2.1e-5 away
    and finite differences will agree with it to four digits.

    ``diagnostics["gauge_consistency"]`` reports how far the energy's cotangent
    is from orthogonal to the environment-phase gauge, which is what makes the
    singular adjoint system solvable; it is ~1e-16 when the energy boundary is
    right and is worth watching whenever that boundary changes.
    """
    _check_root_residual_policy(on_root_residual)
    from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint

    if isinstance(A, SymmetricTensor):
        raise TypeError("Asymmetric root implicit AD is dense-only (#715 Phase 3).")

    A_const = DenseTensor(jax.lax.stop_gradient(A.todense()), A.indices)
    env, a_arr, meta, forward_projs = converge(
        A_const,
        chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        min_iter=min_iter,
        return_projectors=True,
    )
    root, root_residual, usable_rank = asym_root_parametrize(
        env,
        a_arr,
        chi,
        prev_projs=forward_projs,
        polish_steps=polish_steps,
        polish_tol=polish_tol,
        return_usable_rank=True,
    )
    # #779: a clamped cut has an intrinsic residual floor that no amount of
    # polishing removes, so the tolerance has to know whether the clamp fired.
    # Unchanged (strict) whenever it did not.
    residual_tol = _root_residual_tolerance(
        root_residual_warn, usable_rank, chi, env.C1.dtype
    )
    if root_residual > residual_tol:
        _report_root_residual(
            on_root_residual,
            f"Asymmetric root implicit AD: ‖F(y*)‖ = {root_residual:.3e} exceeds "
            f"{residual_tol:.1e}, so y* does not solve the characteristic "
            f"equations (usable_rank={usable_rank} of chi={chi}). The gradient "
            "differentiates equations the environment does not satisfy. Note "
            "this residual is not a measure of gradient accuracy in either "
            "direction -- see #785.",
            residual=float(root_residual),
            tolerance=float(residual_tol),
        )

    # §V.3 works in the modified variables, and indexes its cuts by the upper
    # half of the plane rather than the left half the forward sweep uses.
    root_cov = asym_root_to_covariant_convention(root)
    S_star = root_cov.s
    tilde = remove_inverse_roots(root_cov.env, S_star)
    y_star = (tilde, root_cov.u, S_star, root_cov.v)

    template = initialize_ctm_tensor_env(A_const, chi)
    A_data = jnp.asarray(A.todense())

    # The energy is a function of the *regular* environment, so the last step
    # of the forward pass is the Eq. 82 absorption; differentiating through it
    # is what gives ``S`` an adjoint at all.  ``e`` depends on ``S`` only this
    # way, and setting that adjoint to zero — which is what happens if ``F``
    # is written in the regular variables — is the #718 bug: ``|S̆|`` comes out
    # the same order as ``|C̆|``, not negligible.
    def energy_of(a_data, env_tilde, S_all):
        A_live = DenseTensor(a_data, A.indices)
        return asym_energy(
            A_live, absorb_inverse_roots(env_tilde, S_all), template, gate
        )

    energy, vjp_energy = jax.vjp(energy_of, A_data, tilde, S_star)
    grad_direct, tilde_bar, S_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    # ``u`` and ``v`` carry no cotangent: the energy does not see the
    # null-space coordinates, only their effect through the root.
    y_bar = (
        tilde_bar,
        tuple(jnp.zeros_like(x) for x in root_cov.u),
        S_bar,
        tuple(jnp.zeros_like(x) for x in root_cov.v),
    )

    # An independent phase on *each* environment tensor is an exact null
    # direction of ``∂_y F``.  The normalisation is a ratio, and
    # ``X'/⟨X, X'⟩`` is invariant under any phase on ``X'`` while picking up
    # ``e^{iγ}`` from ``X``, so ``R = X'/⟨X, X'⟩ - X`` is phase-*covariant*
    # tensor by tensor and vanishes along all eight orbits at once; ``R_u``,
    # ``R_S`` and ``R_v`` are ratios throughout and are invariant outright.
    # ``∂_y F`` is therefore singular — measured nullity 12 at ``D=2``,
    # ``chi=4`` — and the implicit function theorem does not apply as literally
    # written.
    #
    # Two things make that harmless, and only the first can fail here.  The
    # adjoint system is consistent iff ``ў`` is orthogonal to every null
    # direction, which holds because the energy is invariant along each orbit —
    # but that invariance lives at the *energy* boundary, not in this module,
    # and #718 is a standing reminder that the boundary is where conventions go
    # wrong, so measure it rather than trust it.  The leftover freedom in
    # ``F̆`` then cannot reach the gradient at all: differentiating
    # ``F(y*(p), p) = 0`` puts ``∂_p F`` in the range of ``∂_y F``, and the
    # freedom is orthogonal to that range (verified to 6e-15 in the tests).
    y_bar_norm = float(
        jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(y_bar)))
    )
    gauge_consistency = 0.0
    for bar, tensor in zip(tilde_bar, tilde):
        pairing = float(jnp.real(jnp.sum(bar * (1j * tensor))))
        scale = y_bar_norm * float(jnp.linalg.norm(tensor)) + 1e-300
        gauge_consistency = max(gauge_consistency, abs(pairing) / scale)
    if gauge_consistency > 1e-8:
        warnings.warn(
            f"Asymmetric root implicit AD: the energy cotangent has a "
            f"{gauge_consistency:.3e} relative component along an environment "
            "tensor's phase, which is a null direction of ∂F/∂y. The adjoint "
            "system is inconsistent by that much and the gradient is "
            "correspondingly unreliable; the energy is supposed to be "
            "invariant under every such phase (#721).",
            RuntimeWarning,
            stacklevel=2,
        )

    def F_of_y(y):
        return asym_characteristic_residual_covariant(y, a_arr, root_cov, chi)

    F_at_root, vjp_y = jax.vjp(F_of_y, y_star)
    covariant_residual = float(
        jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(F_at_root)))
    )
    if covariant_residual > residual_tol:
        _report_root_residual(
            on_root_residual,
            f"Asymmetric root implicit AD: the covariant ‖F(y*)‖ = "
            f"{covariant_residual:.3e} exceeds {residual_tol:.1e} "
            f"(usable_rank={usable_rank} of chi={chi}). The gradient solves "
            "the adjoint of equations that y* does not satisfy. As above, "
            "this does not by itself imply an inaccurate gradient (#785).",
            residual=float(covariant_residual),
            tolerance=float(residual_tol),
        )
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
        return asym_characteristic_residual_covariant(y_star, a_live, root_cov, chi)

    _, vjp_p = jax.vjp(F_of_p, A_data)
    grad = grad_direct - vjp_p(F_bar)[0]

    if return_diagnostics:
        return (
            energy,
            grad,
            {
                **meta,
                "root_residual": root_residual,
                "covariant_residual": covariant_residual,
                "adjoint_residual": float(solve_resid),
                "gauge_consistency": gauge_consistency,
            },
        )
    return energy, grad
