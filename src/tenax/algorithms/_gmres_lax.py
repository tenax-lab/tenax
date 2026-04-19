"""GMRES(m) solver using ``jax.lax.while_loop`` for full JIT compatibility.

This module provides a restarted GMRES implementation that can run entirely
inside ``jax.jit``.  It is used by the implicit-differentiation backward pass
to solve ``(I - J^T) lambda = dE/denv`` without requiring the Jacobian to be
contractive (unlike Neumann series).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def gmres_lax(
    matvec,
    b: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    *,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 30,
) -> tuple[jnp.ndarray, int]:
    """Solve ``A x = b`` via restarted GMRES(m).

    Parameters
    ----------
    matvec : callable
        Matrix-vector product ``x -> A @ x``.
    b : jnp.ndarray
        Right-hand side vector, shape ``(n,)``.
    x0 : jnp.ndarray or None
        Initial guess.  Defaults to zeros.
    tol : float
        Relative tolerance: converge when ``||r|| / ||b|| < tol``.
    maxiter : int
        Maximum total number of Arnoldi iterations (across all restarts).
    restart : int
        Number of inner Arnoldi iterations before restarting (the *m* in
        GMRES(m)).

    Returns
    -------
    x : jnp.ndarray
        Approximate solution.
    info : int
        0 if converged, 1 otherwise.
    """
    n = b.shape[0]
    m = restart

    if x0 is None:
        x0 = jnp.zeros_like(b)

    bnorm = jnp.linalg.norm(b)
    # Guard against zero RHS — return zeros immediately via where.
    bnorm_safe = jnp.where(bnorm > 0.0, bnorm, 1.0)

    # ------------------------------------------------------------------
    # Inner GMRES cycle: run *m* Arnoldi steps from current x.
    # Returns updated x, residual norm, and number of iterations used.
    # ------------------------------------------------------------------

    def _inner_gmres(x, total_iters, max_total):
        """One GMRES(m) cycle starting from *x*."""
        r = b - matvec(x)
        r_norm = jnp.linalg.norm(r)
        r_norm_safe = jnp.where(r_norm > 0.0, r_norm, 1.0)

        # Arnoldi basis V: (m+1, n), upper Hessenberg H: (m+1, m)
        V = jnp.zeros((m + 1, n), dtype=b.dtype)
        V = V.at[0].set(r / r_norm_safe)
        H = jnp.zeros((m + 1, m), dtype=b.dtype)

        # Givens rotation coefficients
        cs = jnp.zeros(m, dtype=b.dtype)
        sn = jnp.zeros(m, dtype=b.dtype)

        # Right-hand side of the least-squares problem
        g = jnp.zeros(m + 1, dtype=b.dtype)
        g = g.at[0].set(r_norm)

        # Inner state: (j, V, H, cs, sn, g, converged, total_iters)
        inner_state0 = (0, V, H, cs, sn, g, False, total_iters)

        def _inner_cond(state):
            j, _V, _H, _cs, _sn, g, converged, t_iters = state
            not_done = (~converged) & (j < m) & (t_iters < max_total)
            return not_done

        def _inner_body(state):
            j, V, H, cs, sn, g, _converged, t_iters = state

            # Arnoldi step: compute A * v_j and orthogonalize
            w = matvec(V[j])

            # Modified Gram-Schmidt via fori_loop
            def _orth_body(i, carry):
                w_, H_ = carry
                h_ij = jnp.vdot(V[i], w_)
                H_ = H_.at[i, j].set(h_ij)
                w_ = w_ - h_ij * V[i]
                return (w_, H_)

            w, H = lax.fori_loop(0, j + 1, _orth_body, (w, H))

            h_jp1_j = jnp.linalg.norm(w)
            h_jp1_j_safe = jnp.where(h_jp1_j > 0.0, h_jp1_j, 1.0)
            H = H.at[j + 1, j].set(h_jp1_j)
            V = V.at[j + 1].set(w / h_jp1_j_safe)

            # Apply previous Givens rotations to column j of H
            def _apply_prev_givens(i, H_col):
                temp = cs[i] * H_col[i] + sn[i] * H_col[i + 1]
                H_col = H_col.at[i + 1].set(-sn[i] * H_col[i] + cs[i] * H_col[i + 1])
                H_col = H_col.at[i].set(temp)
                return H_col

            h_col = H[:, j]
            h_col = lax.fori_loop(0, j, _apply_prev_givens, h_col)
            H = H.at[:, j].set(h_col)

            # Compute new Givens rotation for row (j, j+1)
            a_ = H[j, j]
            b_ = H[j + 1, j]
            denom = jnp.sqrt(a_**2 + b_**2)
            denom_safe = jnp.where(denom > 0.0, denom, 1.0)
            cs_j = a_ / denom_safe
            sn_j = b_ / denom_safe
            cs = cs.at[j].set(cs_j)
            sn = sn.at[j].set(sn_j)

            # Apply to H
            H = H.at[j, j].set(cs_j * a_ + sn_j * b_)
            H = H.at[j + 1, j].set(0.0)

            # Apply to g
            g_j = g[j]
            g = g.at[j].set(cs_j * g_j)
            g = g.at[j + 1].set(-sn_j * g_j)

            # Check convergence
            res = jnp.abs(g[j + 1])
            converged = res / bnorm_safe < tol

            return (j + 1, V, H, cs, sn, g, converged, t_iters + 1)

        j_final, V, H, cs, sn, g, converged, t_iters = lax.while_loop(
            _inner_cond, _inner_body, inner_state0
        )

        # Back-substitution: solve H[:j, :j] y = g[:j]
        # Use j_final as the number of completed iterations.
        # We solve the full m x m system but mask using j_final.
        H_upper = H[:m, :m]
        g_rhs = g[:m]

        # Zero out rows/cols beyond j_final so solve_triangular is safe
        mask = jnp.arange(m) < j_final
        g_rhs = jnp.where(mask, g_rhs, 0.0)
        # Make H diagonal = 1 beyond j_final to avoid singular system
        diag_fill = jnp.where(mask, 0.0, 1.0)
        H_upper = H_upper + jnp.diag(diag_fill)

        y = jax.scipy.linalg.solve_triangular(H_upper, g_rhs)
        # Zero out components beyond j_final
        y = jnp.where(mask, y, 0.0)

        # Update solution: x_new = x + V[:m] @ y
        x_new = x + V[:m].T @ y

        res_norm = jnp.abs(g[j_final])
        return x_new, res_norm, converged, t_iters

    # ------------------------------------------------------------------
    # Outer restart loop
    # ------------------------------------------------------------------

    # State: (x, converged, total_iters)
    outer_state0 = (x0, False, 0)

    def _outer_cond(state):
        _x, converged, total_iters = state
        return (~converged) & (total_iters < maxiter)

    def _outer_body(state):
        x, _converged, total_iters = state
        x_new, res_norm, converged, t_iters = _inner_gmres(x, total_iters, maxiter)
        return (x_new, converged, t_iters)

    x_final, converged, _total = lax.while_loop(_outer_cond, _outer_body, outer_state0)

    # Handle zero RHS
    x_final = jnp.where(bnorm > 0.0, x_final, x0)
    info = jnp.where(converged, 0, 1)

    return x_final, info


def gmres_pytree(
    matvec,
    b_tree,
    x0_tree=None,
    *,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 30,
) -> tuple:
    """GMRES(m) for pytree-valued linear systems.

    Flattens the pytree to a single vector, solves, then unflattens.
    ``matvec`` operates on pytrees: matvec(tree) -> tree.
    """
    b_leaves, treedef = jax.tree.flatten(b_tree)
    shapes = [leaf.shape for leaf in b_leaves]
    sizes = [leaf.size for leaf in b_leaves]
    # Use Python ints for split indices so they stay concrete under JIT.
    offsets = []
    acc = 0
    for sz in sizes[:-1]:
        acc += sz
        offsets.append(acc)

    def flatten(tree):
        return jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(tree)])

    def unflatten(vec):
        parts = jnp.split(vec, offsets)
        return jax.tree.unflatten(
            treedef, [p.reshape(s) for p, s in zip(parts, shapes)]
        )

    def flat_matvec(v):
        return flatten(matvec(unflatten(v)))

    b_flat = flatten(b_tree)
    x0_flat = flatten(x0_tree) if x0_tree is not None else None

    x_flat, info = gmres_lax(
        flat_matvec, b_flat, x0_flat, tol=tol, maxiter=maxiter, restart=restart
    )
    return unflatten(x_flat), info


def gmres_pytree_jax(
    matvec,
    b_tree,
    x0_tree=None,
    *,
    tol: float = 1e-7,
    solve_method: str | None = None,
) -> tuple:
    """GMRES for pytree-valued linear systems using JAX's built-in solver.

    Uses ``jax.scipy.sparse.linalg.gmres`` — the same solver variPEPS uses.
    No iteration cap (runs until ``tol`` is met).

    Args:
        matvec: Pytree-valued linear operator.
        b_tree: Right-hand side pytree.
        x0_tree: Initial guess (optional).
        tol: Absolute tolerance.
        solve_method: ``"batched"`` (GPU) or ``"incremental"`` (CPU).
    """
    from jax.scipy.sparse.linalg import gmres as jax_gmres

    b_leaves, treedef = jax.tree.flatten(b_tree)
    shapes = [leaf.shape for leaf in b_leaves]
    sizes = [leaf.size for leaf in b_leaves]
    offsets = []
    acc = 0
    for sz in sizes[:-1]:
        acc += sz
        offsets.append(acc)

    def flatten(tree):
        return jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(tree)])

    def unflatten(vec):
        parts = jnp.split(vec, offsets)
        return jax.tree.unflatten(
            treedef, [p.reshape(s) for p, s in zip(parts, shapes)]
        )

    def flat_matvec(v):
        return flatten(matvec(unflatten(v)))

    b_flat = flatten(b_tree)
    x0_flat = flatten(x0_tree) if x0_tree is not None else b_flat

    kwargs = {"atol": tol}
    if solve_method is not None:
        kwargs["solve_method"] = solve_method

    x_flat, info = jax_gmres(flat_matvec, b_flat, x0_flat, **kwargs)
    return unflatten(x_flat), info
