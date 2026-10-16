"""GMRES(m) solver using ``jax.lax.while_loop`` for full JIT compatibility.

This module provides a restarted GMRES implementation that can run entirely
inside ``jax.jit``.  It is used by the implicit-differentiation backward pass
to solve ``(I - J^T) lambda = dE/denv`` without requiring the Jacobian to be
contractive (unlike Neumann series).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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

        # Short-circuit when initial residual is already zero (exact solution).
        already_converged = r_norm / bnorm_safe < tol

        # Inner state: (j, V, H, cs, sn, g, converged, total_iters)
        inner_state0 = (0, V, H, cs, sn, g, already_converged, total_iters)

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

            # Apply previous Givens rotations to column j of H.
            # Complex-safe form: [c̄  s̄; -s  c] zeroes the lower entry.
            def _apply_prev_givens(i, H_col):
                temp = jnp.conj(cs[i]) * H_col[i] + jnp.conj(sn[i]) * H_col[i + 1]
                H_col = H_col.at[i + 1].set(-sn[i] * H_col[i] + cs[i] * H_col[i + 1])
                H_col = H_col.at[i].set(temp)
                return H_col

            h_col = H[:, j]
            h_col = lax.fori_loop(0, j, _apply_prev_givens, h_col)
            H = H.at[:, j].set(h_col)

            # Compute new Givens rotation for row (j, j+1).
            # Complex-safe: denom = sqrt(|a|² + |b|²), c = a/r, s = b/r.
            a_ = H[j, j]
            b_ = H[j + 1, j]
            denom = jnp.sqrt(jnp.abs(a_) ** 2 + jnp.abs(b_) ** 2)
            denom_safe = jnp.where(denom > 0.0, denom, 1.0)
            cs_j = a_ / denom_safe
            sn_j = b_ / denom_safe
            cs = cs.at[j].set(cs_j)
            sn = sn.at[j].set(sn_j)

            # Apply to H: [c̄  s̄; -s  c] @ [a; b] = [r; 0]
            H = H.at[j, j].set(jnp.conj(cs_j) * a_ + jnp.conj(sn_j) * b_)
            H = H.at[j + 1, j].set(0.0)

            # Apply to g
            g_j = g[j]
            g = g.at[j].set(jnp.conj(cs_j) * g_j)
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
    maxiter: int | None = None,
    restart: int | None = None,
    solve_method: str | None = None,
) -> tuple:
    """GMRES for pytree-valued linear systems using JAX's built-in solver.

    Uses ``jax.scipy.sparse.linalg.gmres`` — the same solver variPEPS uses.

    Args:
        matvec: Pytree-valued linear operator.
        b_tree: Right-hand side pytree.
        x0_tree: Initial guess.  ``None`` means **zeros**, matching JAX and
            SciPy.  It used to mean ``b``, which is not a neutral default --
            see the warning below.
        tol: Absolute tolerance.
        maxiter: Maximum GMRES iterations (outer restarts).
        restart: Inner Krylov dimension before restart.
        solve_method: ``"batched"`` (GPU) or ``"incremental"`` (CPU).

    .. warning::
        **A non-zero ``x0`` removes the guarantee that the answer beats
        ``x = 0``.**  GMRES minimises ``||b - A x||`` over ``x0 + Krylov``, so
        its final residual is bounded by ``||b - A x0||`` -- its *starting*
        residual -- and not by ``||b||``.  With ``x0 = 0`` those coincide and
        the returned ``x`` can never be worse than the trivial answer.  With
        ``x0 = b`` the bound is ``||b - A b||``, which exceeds ``||b||``
        whenever the operator is not contracting.

        That is #858: this wrapper substituted ``b`` for a missing ``x0``, so
        adjoint solves on a poorly conditioned CTM fixed point reported
        relative residuals of 1.58 and 2.10 -- a ``lambda`` that solves the
        adjoint equation *worse than* ``lambda = 0``.  It read as GMRES
        diverging, which GMRES cannot do; it was the seed.  The sibling
        :func:`gmres_pytree` never had this (it passes ``None`` through), so
        the two implementations of one interface disagreed on their default.

        ``b`` is a reasonable seed -- it is the first Neumann term of
        ``(I - J^T)^-1 b`` -- but only where ``||J^T b|| < ||b||``.  Callers
        that want it should pass it explicitly, having checked that it helps;
        :func:`~tenax.algorithms._ctm_energy_ad._best_adjoint_seed` does this
        by measurement.
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
    # ``None`` -> let JAX use its own zeros default.  See the warning above:
    # substituting ``b`` here is what let an adjoint solve come back worse
    # than ``lambda = 0`` (#858).
    x0_flat = flatten(x0_tree) if x0_tree is not None else None

    kwargs = {"tol": tol, "atol": tol}
    if maxiter is not None:
        kwargs["maxiter"] = maxiter
    if restart is not None:
        kwargs["restart"] = restart
    if solve_method is not None:
        kwargs["solve_method"] = solve_method

    x_flat, info = jax_gmres(flat_matvec, b_flat, x0_flat, **kwargs)
    return unflatten(x_flat), info


# Phases of :func:`gmres_single_matvec`.  The loop body applies the matvec
# once and the phase decides what the product means.
_SEED, _RESID, _ARNOLDI = 0, 1, 2


def gmres_single_matvec(matvec, b_tree, candidates, *, tol, maxiter, restart):
    """Restarted GMRES whose compiled program applies ``matvec`` at ONE site.

    JAX's GMRES calls the matvec at three places (initial residual, Arnoldi
    step, residual after each restart), and XLA inlines every call.  When the
    matvec is a CTM-sweep VJP that is three compiled copies of the most
    expensive program in the backward -- measured at D=2 chi=8 on CPU:
    2,458 s for the GMRES program against 611 s for the fused Neumann module
    with one copy.  Here a single ``lax.while_loop`` body computes
    ``y = matvec(v_in)`` once and a ``lax.switch`` on ``phase`` consumes it:

    * ``_SEED``    -- ``y = A c_k`` for candidate ``k``; keep ``c_k`` if its
      residual is *strictly* below the incumbent's.  The incumbent starts as
      zero at ``||b||``, so zero wins ties and an earlier candidate beats a
      later equal one -- the :func:`_best_adjoint_seed` rules (#858).
    * ``_RESID``   -- ``y = A x``: the true residual.  Stop when
      ``||r|| <= max(tol*||b||, tol)``, when ``maxiter`` cycles are spent, or
      when ``||r||`` is not finite; otherwise start a Krylov cycle.
    * ``_ARNOLDI`` -- ``y = A V[j]``: CGS2 orthogonalisation, Givens update,
      and at the end of a cycle (``restart`` steps, estimated convergence, or
      breakdown) the triangular solve and the update of ``x``.

    Every exit passes through ``_RESID``, so the returned ``abs_resid`` is the
    measured ``||b - A x||`` of the returned ``x``, never an estimate.  The one
    exception: if a ``_RESID`` residual is non-finite before any finite
    residual of the operator has been measured (the operator produced NaN/inf),
    the best iterate is returned but ``abs_resid`` is NaN, so the solve fails
    closed (``best_res`` would then be the assumed ``||b||`` of ``x = 0``,
    which was never measured).  A later non-finite cycle (e.g. a singular
    system) keeps the best *measured* pair.  A non-finite *seed* candidate
    residual does not set this: a bad warm start must not poison a good solve.

    Args:
        matvec: pytree -> pytree, the operator ``A``.
        b_tree: right-hand side pytree.
        candidates: list of seed pytrees shaped like ``b_tree``; ``None``
            entries are skipped.  Zero is always in the running.
        tol: absolute and relative tolerance, as ``_adjoint_converged``.
        maxiter: maximum restart cycles.  ``0`` returns the best seed.
        restart: Arnoldi steps per cycle.

    Returns:
        ``(x_tree, abs_resid, b_norm)``.
    """
    b_leaves, treedef = jax.tree.flatten(b_tree)
    shapes = [leaf.shape for leaf in b_leaves]
    offsets = [int(o) for o in np.cumsum([leaf.size for leaf in b_leaves])[:-1]]

    def flatten(tree):
        return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)])

    def unflatten(vec):
        parts = jnp.split(vec, offsets)
        return jax.tree.unflatten(
            treedef, [p.reshape(s) for p, s in zip(parts, shapes)]
        )

    b = flatten(b_tree)
    dtype = b.dtype
    rdtype = jnp.zeros((), dtype).real.dtype
    n = b.shape[0]
    m = int(restart)
    if m < 1:
        raise ValueError(f"restart must be >= 1, got {restart}")

    # Fail closed on a real problem with complex data: ``.astype`` would drop
    # the imaginary part and certify a residual that is not the true one.
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        out_shapes = jax.eval_shape(matvec, b_tree)
        if any(
            jnp.issubdtype(leaf.dtype, jnp.complexfloating)
            for leaf in jax.tree.leaves(out_shapes)
        ):
            raise TypeError("gmres_single_matvec: b is real but matvec returns complex")
        for c in candidates:
            if c is not None and any(
                jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.complexfloating)
                for leaf in jax.tree.leaves(c)
            ):
                raise TypeError(
                    "gmres_single_matvec: b is real but a candidate is complex"
                )

    def apply(v):
        return flatten(matvec(unflatten(v))).astype(dtype)

    cands = [flatten(c).astype(dtype) for c in candidates if c is not None]
    n_cand = len(cands)
    C = jnp.stack(cands) if cands else jnp.zeros((1, n), dtype)

    def norm(v):
        return jnp.sqrt(jnp.sum(jnp.abs(v) ** 2)).astype(rdtype)

    b_norm = norm(b)
    tol_r = jnp.asarray(tol, rdtype)
    thresh = jnp.maximum(tol_r * b_norm, tol_r)
    eps = jnp.finfo(rdtype).eps
    zeros_n = jnp.zeros((n,), dtype)
    basis_idx = jnp.arange(m + 1)
    coef_idx = jnp.arange(m)

    state0 = {
        "phase": jnp.int32(_SEED if n_cand else _RESID),
        "k": jnp.int32(0),
        "v_in": C[0] if n_cand else zeros_n,
        "x": zeros_n,
        "best_x": zeros_n,
        "best_res": b_norm,
        "V": jnp.zeros((m + 1, n), dtype),
        "H": jnp.zeros((m + 1, m), dtype),
        "cs": jnp.zeros((m,), rdtype),
        "sn": jnp.zeros((m,), dtype),
        "g": jnp.zeros((m + 1,), dtype),
        "j": jnp.int32(0),
        "cycles": jnp.int32(0),
        "done": jnp.array(False),
        "nonfinite": jnp.array(False),
        # True once any finite residual of the operator has been measured, so
        # ``best_res`` is a measured value and not the assumed ``||b||``.
        "measured": jnp.array(False),
    }

    def seed(s, y):
        res = norm(b - y)
        better = res < s["best_res"]  # NaN compares False: never chosen
        best_x = jnp.where(better, C[s["k"]], s["best_x"])
        best_res = jnp.where(better, res, s["best_res"])
        k1 = s["k"] + 1
        last = k1 >= n_cand
        nxt = C[jnp.minimum(k1, max(n_cand - 1, 0))]
        return {
            **s,
            "k": k1,
            "best_x": best_x,
            "best_res": best_res,
            "measured": s["measured"] | jnp.isfinite(res),
            "x": jnp.where(last, best_x, s["x"]),
            "v_in": jnp.where(last, best_x, nxt),
            "phase": jnp.where(last, _RESID, _SEED).astype(jnp.int32),
        }

    def resid(s, y):
        r = b - y
        rn = norm(r)
        bad = ~jnp.isfinite(rn)
        done = (rn <= thresh) | (s["cycles"] >= maxiter) | bad
        v0 = r / jnp.where(rn > 0, rn, 1).astype(dtype)
        # Best *measured* iterate (strict <; NaN never wins): a bad cycle
        # must not replace a better x, and (x, abs_resid) stays a measured pair.
        better = rn < s["best_res"]
        return {
            **s,
            "best_x": jnp.where(better, s["x"], s["best_x"]),
            "best_res": jnp.where(better, rn, s["best_res"]),
            "nonfinite": s["nonfinite"] | (bad & ~s["measured"]),
            "measured": s["measured"] | ~bad,
            "done": done,
            "V": jnp.zeros_like(s["V"]).at[0].set(v0),
            "H": jnp.zeros_like(s["H"]),
            "cs": jnp.zeros_like(s["cs"]),
            "sn": jnp.zeros_like(s["sn"]),
            "g": jnp.zeros_like(s["g"]).at[0].set(rn.astype(dtype)),
            "j": jnp.int32(0),
            "v_in": v0,
            "phase": jnp.int32(_ARNOLDI),
        }

    def arnoldi(s, y):
        j = s["j"]
        V = s["V"]
        mask = (basis_idx <= j).astype(dtype)
        # Classical Gram-Schmidt applied twice (CGS2): vectorised over the
        # basis, as stable as modified Gram-Schmidt in practice.
        h1 = mask * (jnp.conj(V) @ y)
        w = y - h1 @ V
        h2 = mask * (jnp.conj(V) @ w)
        w = w - h2 @ V
        hnext = norm(w)
        col = (h1 + h2).at[j + 1].set(hnext.astype(dtype))

        def rotate(i, c):
            a, bb = c[i], c[i + 1]
            ci, si = s["cs"][i], s["sn"][i]
            return (
                c.at[i].set(ci * a + si * bb).at[i + 1].set(-jnp.conj(si) * a + ci * bb)
            )

        col = lax.fori_loop(0, j, rotate, col)
        a, bb = col[j], col[j + 1]
        denom = jnp.sqrt(jnp.abs(a) ** 2 + jnp.abs(bb) ** 2)
        safe_denom = jnp.where(denom > 0, denom, 1)
        abs_a = jnp.abs(a)
        phase_a = jnp.where(abs_a > 0, a / jnp.where(abs_a > 0, abs_a, 1), 1)
        c_new = jnp.where(denom > 0, abs_a / safe_denom, 1).astype(rdtype)
        s_new = jnp.where(denom > 0, phase_a * jnp.conj(bb) / safe_denom, 0).astype(
            dtype
        )
        col = col.at[j].set(c_new * a + s_new * bb).at[j + 1].set(0)
        gj = s["g"][j]
        g = s["g"].at[j].set(c_new * gj).at[j + 1].set(-jnp.conj(s_new) * gj)
        H = s["H"].at[:, j].set(col)
        cs = s["cs"].at[j].set(c_new)
        sn = s["sn"].at[j].set(s_new)

        est = jnp.abs(g[j + 1])
        breakdown = hnext <= 10 * eps * norm(y)
        end = (j + 1 >= m) | (est <= thresh) | breakdown

        # End of cycle: solve the leading (j+1)x(j+1) upper-triangular system.
        active = coef_idx <= j
        Hs = jnp.where(
            active[:, None] & active[None, :], H[:m, :m], jnp.eye(m, dtype=dtype)
        )
        coef = jax.scipy.linalg.solve_triangular(
            Hs, jnp.where(active, g[:m], 0), lower=False
        )
        x_new = s["x"] + coef @ V[:m]
        V_next = V.at[j + 1].set(w / jnp.where(hnext > 0, hnext, 1).astype(dtype))
        return {
            **s,
            "H": H,
            "cs": cs,
            "sn": sn,
            "g": g,
            "V": jnp.where(end, V, V_next),
            "x": jnp.where(end, x_new, s["x"]),
            "v_in": jnp.where(end, x_new, V_next[j + 1]),
            "j": jnp.where(end, j, j + 1).astype(jnp.int32),
            "cycles": s["cycles"] + end.astype(jnp.int32),
            "phase": jnp.where(end, _RESID, _ARNOLDI).astype(jnp.int32),
        }

    def body(s):
        y = apply(s["v_in"])  # the one and only matvec call site
        return lax.switch(s["phase"], (seed, resid, arnoldi), s, y)

    out = lax.while_loop(lambda s: ~s["done"], body, state0)
    abs_resid = jnp.where(out["nonfinite"], jnp.nan, out["best_res"])
    return unflatten(out["best_x"]), abs_resid, b_norm
