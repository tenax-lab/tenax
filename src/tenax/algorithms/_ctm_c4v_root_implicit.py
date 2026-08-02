"""Root implicit differentiation for dense C4v CTMRG (#715, Phase 0).

Implements Section III of Burgelman, Francuz, Brehmer, Devos, Haegeman,
Verstraete and Vanhecke, *Implicit differentiation of tensor network
algorithms*, arXiv:2607.15030.

Where the production path in :mod:`tenax.algorithms._ctm_energy_ad`
characterises the converged environment as a **fixed point** of one CTM
sweep and solves ``(1 - J^T) λ = dE/denv`` — differentiating the sweep,
truncated SVD backward and all — this module characterises it as the
**root** of an algebraic equation ``F(y, p) = 0`` in the enlarged variable
set ``y = (C, E, u)`` and solves ``F̆ ∂_y F = ў`` (paper Eqs. 14-18).

The payoff is that ``F`` is a handful of plain contractions, so no ``eigh``
or ``svd`` backward appears anywhere in the gradient path.  Degeneracies in
the kept spectrum, which make the truncated-``eigh`` VJP diverge, are
invisible here: the isometry only varies along its null space (Eq. 31),

    U = U* + U_perp @ u,

with ``U*`` and ``U_perp`` held constant.  The conditioning of the ``u``
block is set by the *truncation gap* — the separation between kept and
discarded eigenvalues — not by gaps inside the kept block.

Characteristic equations (paper Eqs. 26-28), with ``M`` the Hermitian
enlarged corner and ``N`` the enlarged edge:

    R_C = U† M U - λ_C C                        (Eq. 26)
    R_E = U† N U - λ_E E                        (Eq. 27)
    R_u = (U_perp† M U) C*^-1 - λ_C u           (Eq. 28)

``λ_C`` and ``λ_E`` are *differentiable* functions of ``y`` — the inner
products of the first term of Eqs. 26 and 27 with ``C`` and ``E`` — which
is what removes the zero mode associated with the overall normalisation
(paper §III.3, Appendix B).  ``C*^-1`` is a constant right preconditioner.

The forward contraction is unchanged: this module reuses
:func:`tenax.algorithms._ctm_tensor_c4v._c4v_sweep`.

Conventions match ``_c4v_sweep``:

* ``C``  — corner, labels ``(c_a, c_b)``, shape ``(chi, chi)``
* ``T``  — edge, labels ``(t_l, D2, t_r)``, shape ``(chi, d2, chi)``
* ``a``  — double layer, labels ``(u2, d2, l2, r2)``, each of dim ``d2 = D²``
* the fused environment leg is ordered ``(chi, d2)`` on both the corner
  (``c_a``, ``D2``) and the edge (``t_l``, ``l2``), which is what lets the
  same ``U`` act on the enlarged corner and the enlarged edge.
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
from tenax.algorithms._ctm_tensor_c4v import _c4v_sweep, _c4v_to_full_env
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

__all__ = [
    "C4vRoot",
    "c4v_characteristic_residual",
    "c4v_root_implicit_energy",
    "c4v_root_implicit_energy_and_grad",
    "c4v_root_parametrize",
]


class C4vRoot(NamedTuple):
    """Root of the characteristic equations plus its constant tensors.

    ``y = (C, E, u)`` are the differentiable variables; ``U_star``,
    ``U_perp`` and ``C_star_inv`` are held constant when differentiating
    ``F`` (paper Eq. 25 and the Eq. 28 preconditioner).
    """

    C: jax.Array
    E: jax.Array
    u: jax.Array
    U_star: jax.Array
    U_perp: jax.Array
    C_star_inv: jax.Array

    @property
    def y(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (self.C, self.E, self.u)


# ---------------------------------------------------------------------------
# Enlarged corner / edge — the two contractions the sweep is built from.
# ---------------------------------------------------------------------------


def _enlarged_corner(C: jax.Array, E: jax.Array) -> jax.Array:
    """Hermitian enlarged corner ``M = 2 Cg Cg†`` with ``Cg = (C·E)`` fused.

    Mirrors steps 1 and 3 of :func:`_c4v_sweep`: the corner is grown by one
    edge and the density matrix accumulated from the two (identical, by C4v)
    corner slots.  Returns a ``(chi*d2, chi*d2)`` Hermitian PSD matrix whose
    dominant eigenvectors are the CTM projector.
    """
    chi, d2 = C.shape[0], E.shape[1]
    Cg = jnp.einsum("ab,bmc->amc", C, E).reshape(chi * d2, chi)
    M = 2.0 * (Cg @ Cg.conj().T)
    # eigh below assumes exact hermiticity; the product form is Hermitian up
    # to rounding, so symmetrise rather than let the asymmetry leak into the
    # eigenvectors.
    return 0.5 * (M + M.conj().T)


def _enlarged_edge(E: jax.Array, a: jax.Array) -> jax.Array:
    """Enlarged edge ``N[fl, d2, fr]`` with ``fl = (t_l, l2)``, ``fr = (t_r, r2)``.

    Mirrors step 2 of :func:`_c4v_sweep`.
    """
    chi, d2 = E.shape[0], E.shape[1]
    N = jnp.einsum("xuy,uvlr->xlvyr", E, a)
    return N.reshape(chi * d2, d2, chi * d2)


def _project_corner(M: jax.Array, U: jax.Array) -> jax.Array:
    return U.conj().T @ M @ U


def _project_edge(N: jax.Array, U: jax.Array) -> jax.Array:
    return jnp.einsum("fi,fmg,gj->imj", U.conj(), N, U)


# ---------------------------------------------------------------------------
# Characteristic equations (paper Eqs. 26-28)
# ---------------------------------------------------------------------------


def c4v_characteristic_residual(
    y: tuple[jax.Array, jax.Array, jax.Array],
    a: jax.Array,
    U_star: jax.Array,
    U_perp: jax.Array,
    C_star_inv: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate ``F(y, p)`` — paper Eqs. 26, 27 and 28.

    Args:
        y:          ``(C, E, u)``; ``C`` is ``(chi, chi)``, ``E`` is
                    ``(chi, d2, chi)``, ``u`` is ``(chi*d2 - chi, chi)``.
        a:          Double-layer tensor, axes ordered ``(u2, d2, l2, r2)``.
        U_star:     Fixed-point isometry ``(chi*d2, chi)``, constant.
        U_perp:     Its null space ``(chi*d2, chi*d2 - chi)``, constant.
        C_star_inv: Constant right preconditioner ``C*^-1`` for Eq. 28.

    Returns:
        ``(R_C, R_E, R_u)``, matching the shapes of ``(C, E, u)``.
    """
    C, E, u = y
    U = U_star + U_perp @ u  # Eq. 25 — differentiable only through ``u``

    M = _enlarged_corner(C, E)
    N = _enlarged_edge(E, a)

    proj_C = _project_corner(M, U)
    proj_E = _project_edge(N, U)

    # λ as a differentiable function of y, not an independent variable
    # (paper §III.3 / Appendix B): this is what kills the normalisation
    # zero mode in ∂_y F.
    lam_C = jnp.vdot(C, proj_C).real
    lam_E = jnp.vdot(E, proj_E).real

    R_C = proj_C - lam_C * C
    R_E = proj_E - lam_E * E
    R_u = (U_perp.conj().T @ M @ U) @ C_star_inv - lam_C * u
    return (R_C, R_E, R_u)


def _align_isometry(U: jax.Array, U_prev: jax.Array | None) -> jax.Array:
    """Rotate ``U`` inside the kept subspace to align it with ``U_prev``.

    ``eigh`` fixes the kept subspace but not the basis inside it: column
    signs flip and near-degenerate eigenvectors rotate freely between
    calls on infinitesimally different matrices.  That is harmless for the
    *subspace* but it makes the corner and edge jump between gauges, so
    the polish loop below would oscillate instead of converging.

    Maximal alignment is the unitary polar factor of the overlap
    ``O = U_prev† U``: writing ``O = W S Vh``, the rotation ``Q = W Vh``
    makes ``U Q†`` the element of the gauge orbit closest to ``U_prev``.

    This is forward-pass bookkeeping only — the backward never sees it,
    which is the point of the null-space parametrisation (paper §III.3).
    """
    if U_prev is None:
        return U
    overlap = U_prev.conj().T @ U
    w_o, _s_o, vh_o = jnp.linalg.svd(overlap, full_matrices=False)
    return U @ (w_o @ vh_o).conj().T


def c4v_root_parametrize(
    C: jax.Array,
    E: jax.Array,
    a: jax.Array,
    chi: int,
    *,
    pinv_rtol: float = 1e-12,
    polish_steps: int = 60,
    polish_tol: float = 1e-10,
) -> tuple[C4vRoot, float]:
    """Map a converged environment to the root ``y* = (C*, E*, 0)``.

    Performs the ``parametrize`` step of the paper's Algorithm 2: a full
    ``eigh`` of the enlarged corner supplies both the kept isometry ``U*``
    and its null space ``U_perp``, after which ``C*`` and ``E*`` are
    redefined in that basis.

    ``(C, E, U)`` must be *mutually* self-consistent for ``F(y*) = 0`` to
    hold — it is not enough for the incoming ``(C, E)`` to be converged in
    some other basis, because Eq. 28 is amplified by ``C*^-1`` and so
    resolves a mismatch of order ``1e-12`` into a residual of order one.
    The loop below therefore re-applies the (enlarged corner → ``eigh`` →
    project) map, with the isometry gauge-aligned each step, until the
    characteristic residual stops improving.

    Returns:
        ``(root, residual)`` where ``residual`` is ``‖F(y*, p)‖``.  This is
        the CTM convergence error and is what limits gradient accuracy
        (paper Fig. 1) — callers should surface or gate on it.
    """
    U_prev: jax.Array | None = None
    best: tuple[C4vRoot, float] | None = None

    for _step in range(max(int(polish_steps), 1)):
        M = _enlarged_corner(C, E)
        # eigh returns ascending eigenvalues; keep the chi dominant ones.
        w, V = jnp.linalg.eigh(M)
        order = jnp.argsort(-w)
        V = V[:, order]
        U_star = _align_isometry(V[:, :chi], U_prev)
        U_perp = V[:, chi:]

        proj_C = _project_corner(M, U_star)
        C_star = proj_C / jnp.linalg.norm(proj_C)

        proj_E = _project_edge(_enlarged_edge(E, a), U_star)
        E_star = proj_E / jnp.linalg.norm(proj_E)

        # C* is diagonal and positive when U* is aligned with the eigenbasis;
        # after a gauge rotation it is Hermitian, so invert it as such with a
        # relative floor on the spectrum.
        C_star_inv = _hermitian_pinv(C_star, pinv_rtol)

        u_zero = jnp.zeros((U_perp.shape[1], chi), dtype=C_star.dtype)
        root = C4vRoot(C_star, E_star, u_zero, U_star, U_perp, C_star_inv)
        residual = float(
            jnp.sqrt(
                sum(
                    jnp.sum(jnp.abs(r) ** 2)
                    for r in c4v_characteristic_residual(
                        root.y, a, U_star, U_perp, C_star_inv
                    )
                )
            )
        )
        if best is None or residual < best[1]:
            best = (root, residual)
        if residual <= polish_tol:
            break
        C, E, U_prev = C_star, E_star, U_star

    assert best is not None
    return best


def _hermitian_pinv(H: jax.Array, rtol: float) -> jax.Array:
    """Pseudo-inverse of a Hermitian matrix with a relative spectral floor.

    Modes below the floor are dropped rather than inverted.  A kept mode
    with vanishing weight carries no environment amplitude, so zeroing its
    row of Eq. 28 leaves ``R_u = -λ_C u`` there — well conditioned, and it
    correctly contributes nothing to the gradient.
    """
    w, V = jnp.linalg.eigh(0.5 * (H + H.conj().T))
    cutoff = rtol * jnp.max(jnp.abs(w))
    safe = jnp.where(jnp.abs(w) > cutoff, w, 1.0)
    inv_w = jnp.where(jnp.abs(w) > cutoff, 1.0 / safe, 0.0)
    return (V * inv_w[None, :]) @ V.conj().T


# ---------------------------------------------------------------------------
# Real-embedded linear algebra
#
# ``F`` is not holomorphic (it contains conjugations), so its VJP is only
# real-linear.  Krylov solvers assume complex-linearity, so the solve runs on
# the real embedding (real and imaginary parts stacked).  For real inputs this
# reduces to the plain real solve at no extra cost.
# ---------------------------------------------------------------------------


def _real_struct(tree) -> list[tuple[Any, bool]]:
    return [
        (leaf.shape, bool(jnp.iscomplexobj(leaf))) for leaf in jax.tree.leaves(tree)
    ]


def _to_real_vec(tree) -> jax.Array:
    parts = []
    for leaf in jax.tree.leaves(tree):
        if jnp.iscomplexobj(leaf):
            parts.append(jnp.concatenate([leaf.real.ravel(), leaf.imag.ravel()]))
        else:
            parts.append(leaf.ravel())
    return jnp.concatenate(parts)


def _from_real_vec(vec: jax.Array, treedef, struct) -> Any:
    leaves = []
    off = 0
    for shape, is_complex in struct:
        n = 1
        for s in shape:
            n *= s
        if is_complex:
            re = vec[off : off + n].reshape(shape)
            im = vec[off + n : off + 2 * n].reshape(shape)
            leaves.append(re + 1j * im)
            off += 2 * n
        else:
            leaves.append(vec[off : off + n].reshape(shape))
            off += n
    return jax.tree.unflatten(treedef, leaves)


def _solve_root_adjoint(
    matvec,
    rhs,
    *,
    tol: float,
    maxiter: int,
    restart: int,
):
    """Solve ``matvec(F̆) = ў`` on the real embedding.

    Returns ``(F̆, info)`` where ``info`` carries the achieved relative
    residual so callers can surface a diagnostic rather than silently
    returning a bad gradient.
    """
    leaves, treedef = jax.tree.flatten(rhs)
    struct = _real_struct(rhs)
    b_vec = _to_real_vec(rhs)

    @jax.jit
    def op(vec):
        tree = _from_real_vec(vec, treedef, struct)
        return _to_real_vec(matvec(tree))

    # The solve is the hot loop: GMRES evaluates ``op`` up to ``maxiter``
    # times, and each evaluation is a VJP through the whole of ``F``.  Run
    # eagerly, the per-operation Python dispatch dominates — enough that
    # adding the Eq. 73 cut-leg roots to ``F`` made a single measurement
    # fail to return.  Compiling the whole solve leaves one XLA program per
    # call instead.
    @jax.jit
    def _solve(b):
        sol, _info = jax.scipy.sparse.linalg.gmres(
            op,
            b,
            x0=b,
            tol=tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            solve_method="batched",
        )
        resid = jnp.linalg.norm(op(sol) - b) / (jnp.linalg.norm(b) + 1e-30)
        return sol, resid

    sol, resid = _solve(b_vec)
    del leaves
    return _from_real_vec(sol, treedef, struct), resid


# ---------------------------------------------------------------------------
# Forward contraction
# ---------------------------------------------------------------------------


def _converge_c4v(
    A: Tensor,
    chi: int,
    *,
    max_iter: int,
    conv_tol: float,
    min_iter: int,
    projector_method: str,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Run the existing C4v sweep to convergence (no gradient tracked)."""
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    C = env.C1.relabels({"c1_d": "c_a", "c1_r": "c_b"})
    T = env.T1.relabels({"t1_l": "t_l", "u2": "D2", "t1_r": "t_r"})

    prev = None
    residual = float("inf")
    converged = False
    iters = 0
    for it in range(int(max_iter)):
        C, T = _c4v_sweep(C, T, a, chi, projector_method)
        iters = it + 1
        cur = jnp.linalg.svd(C.todense(), compute_uv=False)
        if prev is not None and cur.shape == prev.shape:
            residual = float(jnp.linalg.norm(cur / cur[0] - prev / prev[0]))
            if iters >= min_iter and residual < conv_tol:
                converged = True
                break
        prev = cur
    meta = {"iters": iters, "residual": residual, "converged": converged}
    return C, T, a, meta


def _a_dense(a: Tensor) -> jax.Array:
    """Double-layer array with axes ordered ``(u2, d2, l2, r2)``."""
    labels = list(a.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    return jnp.asarray(a.transpose(perm).todense())


def _energy_from_env_arrays(
    A: Tensor,
    C_arr: jax.Array,
    E_arr: jax.Array,
    C_tmpl: Tensor,
    T_tmpl: Tensor,
    gate,
) -> jax.Array:
    C_t = DenseTensor(C_arr, C_tmpl.indices)
    T_t = DenseTensor(E_arr, T_tmpl.indices)
    env = _c4v_to_full_env(C_t, T_t)
    return compute_energy_ctm_tensor(A, env, gate)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def c4v_root_implicit_energy_and_grad(
    A: Tensor,
    gate,
    *,
    chi: int = 16,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    min_iter: int = 4,
    projector_method: str = "eigh",
    polish_steps: int = 60,
    polish_tol: float = 1e-10,
    solve_tol: float = 1e-8,
    solve_maxiter: int = 200,
    solve_restart: int = 30,
    root_residual_warn: float = 1e-6,
    on_root_residual: str = "raise",
    solve_residual_warn: float = 1e-6,
    return_diagnostics: bool = False,
):
    """Energy and ``dE/dA`` for dense C4v iPEPS via root implicit AD.

    The gradient is assembled as (paper Eqs. 17-18)::

        dE/dp = ∂_p e  -  F̆ ∂_p F,      F̆ solves  F̆ ∂_y F = ў

    with ``ў = (C̆, Ĕ, 0)``: the isometry never enters the energy, so its
    adjoint vanishes.

    Args:
        A:                 iPEPS site tensor ``(u, d, l, r, phys)``, dense.
        gate:              2-site Hamiltonian gate.
        chi:               Environment bond dimension.
        max_iter/conv_tol/min_iter: Forward CTM convergence controls.
        projector_method:  Forward projector, ``"eigh"`` (default) or ``"qr"``.
                           Only affects the forward — the backward never
                           differentiates it.
        solve_tol/solve_maxiter/solve_restart: GMRES controls for Eq. 17.
        return_diagnostics: Also return a dict with the forward CTM meta,
                           ``‖F(y*)‖`` and the adjoint solve residual.

    Returns:
        ``(energy, grad_A)`` where ``grad_A`` is a dense array shaped like
        ``A``; plus a diagnostics dict when ``return_diagnostics`` is set.
    """
    _check_root_residual_policy(on_root_residual)
    if isinstance(A, SymmetricTensor):
        raise TypeError(
            "c4v_root_implicit_energy_and_grad supports dense tensors only "
            "(#715 Phase 3 covers SymmetricTensor)."
        )
    if not isinstance(A, DenseTensor):
        raise TypeError(f"Expected DenseTensor, got {type(A).__name__}.")

    # --- Forward: converge, then parametrize the root (no grad tracked) ---
    A_const = DenseTensor(jax.lax.stop_gradient(A.todense()), A.indices)
    C_t, T_t, a_t, meta = _converge_c4v(
        A_const,
        chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        min_iter=min_iter,
        projector_method=projector_method,
    )
    a_arr = _a_dense(a_t)
    root, root_residual = c4v_root_parametrize(
        jnp.asarray(C_t.todense()),
        jnp.asarray(T_t.todense()),
        a_arr,
        chi,
        polish_steps=polish_steps,
        polish_tol=polish_tol,
    )

    if root_residual > root_residual_warn:
        _report_root_residual(
            on_root_residual,
            f"C4v root implicit AD: ‖F(y*)‖ = {root_residual:.3e} exceeds "
            f"{root_residual_warn:.1e}. The environment is not a root of the "
            "characteristic equations, so the implicit-function gradient is "
            "correspondingly inaccurate (paper Fig. 1). Usual cause: chi cuts "
            "through a numerically degenerate part of the corner spectrum — "
            "lower chi, or converge the forward CTM further.",
            residual=float(root_residual),
            tolerance=float(root_residual_warn),
        )

    A_data = jnp.asarray(A.todense())

    def energy_of(a_data, c_arr, e_arr):
        A_live = DenseTensor(a_data, A.indices)
        return _energy_from_env_arrays(A_live, c_arr, e_arr, C_t, T_t, gate)

    # --- ∂_p e (direct) and ў = (C̆, Ĕ, 0) ---
    energy, vjp_energy = jax.vjp(energy_of, A_data, root.C, root.E)
    grad_direct, C_bar, E_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    y_bar = (C_bar, E_bar, jnp.zeros_like(root.u))

    # --- ∂_y F at the root, with a and the constants frozen ---
    def F_of_y(y):
        return c4v_characteristic_residual(
            y, a_arr, root.U_star, root.U_perp, root.C_star_inv
        )

    _, vjp_y = jax.vjp(F_of_y, root.y)

    def matvec(v):
        return vjp_y(v)[0]

    F_bar, solve_resid = _solve_root_adjoint(
        matvec,
        y_bar,
        tol=solve_tol,
        maxiter=solve_maxiter,
        restart=solve_restart,
    )

    # --- ∂_p F (indirect) ---
    def F_of_p(a_data):
        A_live = DenseTensor(a_data, A.indices)
        a_live = _a_dense(_build_double_layer_tensor(A_live))
        return c4v_characteristic_residual(
            root.y, a_live, root.U_star, root.U_perp, root.C_star_inv
        )

    _, vjp_p = jax.vjp(F_of_p, A_data)
    grad_indirect = vjp_p(F_bar)[0]

    grad = grad_direct - grad_indirect

    # An unconverged adjoint solve does not give an approximate gradient — it
    # gives an arbitrary one, because F̆ is then not the solution of Eq. 17 at
    # all.  Surfacing the residual only under ``return_diagnostics`` hid that:
    # ``solve_maxiter=1`` returns a finite, badly wrong gradient in silence.
    # Checked unconditionally.
    if float(solve_resid) > solve_residual_warn:
        warnings.warn(
            f"C4v root implicit AD: adjoint solve did not converge "
            f"(relative residual {float(solve_resid):.3e} > "
            f"{solve_residual_warn:.1e} after {solve_maxiter} restart(s) of "
            f"a {solve_restart}-dimensional Krylov space). The returned "
            "gradient does not solve F̆ ∂_y F = ў and should not be used; "
            "raise solve_maxiter/solve_restart or loosen solve_tol.",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_diagnostics:
        diag = {
            **meta,
            "root_residual": root_residual,
            "adjoint_residual": float(solve_resid),
        }
        return energy, grad, diag
    return energy, grad


def c4v_root_implicit_energy(
    A: Tensor,
    gate,
    *,
    chi: int = 16,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    min_iter: int = 4,
    projector_method: str = "eigh",
    polish_steps: int = 60,
    polish_tol: float = 1e-10,
    **_ignored,
) -> jax.Array:
    """Energy only — forward contraction, parametrisation and evaluation.

    Deliberately does not build either VJP or run the adjoint solve.  Callers
    that only want an energy (line searches, diagnostics, convergence probes)
    would otherwise pay the whole backward cost, and could fail on an adjoint
    solve they never asked for.
    """
    if isinstance(A, SymmetricTensor):
        raise TypeError(
            "c4v_root_implicit_energy supports dense tensors only "
            "(#715 Phase 3 covers SymmetricTensor)."
        )
    if not isinstance(A, DenseTensor):
        raise TypeError(f"Expected DenseTensor, got {type(A).__name__}.")

    C_t, T_t, a_t, _meta = _converge_c4v(
        A,
        chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        min_iter=min_iter,
        projector_method=projector_method,
    )
    root, _residual = c4v_root_parametrize(
        jnp.asarray(C_t.todense()),
        jnp.asarray(T_t.todense()),
        _a_dense(a_t),
        chi,
        polish_steps=polish_steps,
        polish_tol=polish_tol,
    )
    return _energy_from_env_arrays(A, root.C, root.E, C_t, T_t, gate)
