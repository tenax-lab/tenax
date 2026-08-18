"""Restarted GMRES(m) whose *loop* runs in Python, driving a jitted matvec.

Why this exists (#731).  ``jax.scipy.sparse.linalg.gmres`` traces the operator
into a ``lax.while_loop`` body, and ``custom_linear_solve`` then needs it in
several places at once: ``solve`` holds three copies (``A(x0)``, ``A(v)`` in
the Arnoldi body, ``A(x)`` for the restart residual), ``transpose_solve``
another three, and a caller that measures an honest residual one more.
Measured on the root-implicit adjoint (#715), where the operator is the VJP of
the characteristic equations: the matvec's jaxpr is **76,884** equations and
the jitted solve's is **700,013**, i.e. 9.1x the operator, at roughly +1.0 GB
and +50 s of *compile* per extra copy.

At ``D=2, chi=4`` -- the smallest interesting case -- that peaked at **8.6 GB**
of host RAM against GitHub Linux runners of ~7 GB.  None of it was data.  The
real embedding there is ``n = 384``, so the whole 30-dimensional Krylov basis
is 91 KB and ``memory_analysis()`` reports 0.5 MB of runtime temporaries;
*executing* the compiled program added nothing to the high-water mark, while
*lowering* it added 5.1 GB in 148 s.

Running the *loop* eagerly and keeping the *matvec* jitted compiles one program
for one matvec instead of one program for the whole solve: 8.6 -> 4.8 GB and
303 -> 180 s on that fixture, gradient identical to ten digits.  The Krylov
bookkeeping costs ``O(m^2 n)`` arithmetic and ``(m+1) x n`` memory, which is
nothing next to the operator.

What is left is the same kind of thing one level out, and worth knowing before
anyone reads this as solved: ~3.0 GB of that 4.8 is committed *before* the
solve, 1.8 GB of it compiling the energy's own VJP.  Deleting the adjoint solve
entirely would still leave ~3.7 GB.

This is **not** the "run the whole thing eagerly" that
``_ctm_c4v_root_implicit._solve_root_adjoint`` used to warn against.  There the
*matvec itself* was un-jitted, so per-operation Python dispatch dominated -- a
single measurement failed to return once ``F`` grew the Eq. 73 cut-leg roots.
Here each iteration is exactly one compiled call.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

__all__ = ["gmres_eager"]

# A subdiagonal entry this far below the column it terminates means the Krylov
# space is exhausted (a "happy breakdown"): the current subspace already
# contains the solution, and normalising by it would divide by noise.
_BREAKDOWN_REL = 1e-14


def gmres_eager(
    matvec: Callable[[Any], Any],
    b,
    *,
    x0=None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 200,
    restart: int = 30,
) -> tuple[Any, float]:
    """Solve ``matvec(x) = b`` on a real vector by restarted GMRES(m).

    Semantics mirror ``jax.scipy.sparse.linalg.gmres(..., solve_method="batched")``
    so this is a drop-in replacement: ``maxiter`` counts *restarts*, ``restart``
    is the Krylov dimension ``m``, ``x0`` defaults to ``b`` (not to zero), the
    convergence test is ``||b - A x|| <= max(tol * ||b||, atol)`` and is applied
    **between** restarts only -- each restart builds the full ``m``-dimensional
    subspace, which is why the achieved residual routinely lands well below
    ``tol`` (see the note in the inner loop).

    Parameters
    ----------
    matvec
        The linear operator.  Called once per Krylov step, plus once per restart
        for the true residual.  Jit it -- that is the entire point of this
        module.
    b
        Right-hand side, a real 1-D array.  Complex systems must be passed
        through their real embedding: ``matvec`` here is only assumed
        real-linear, which is exactly the case for a VJP that conjugates, and a
        complex-linear Krylov method does not apply to it.

    Returns
    -------
    (x, resid)
        ``resid`` is the achieved *relative* residual ``||b - A x|| / ||b||``,
        measured with a real matvec rather than read off the Givens recurrence.
        The two differ after a breakdown or a stagnating restart, and it is the
        measured one a caller's gate must see.

    Notes
    -----
    Orthogonalisation is iterated classical Gram-Schmidt, two passes, matching
    what JAX's ``"batched"`` method does.  One pass loses orthogonality on the
    ill-conditioned systems this solver is pointed at; two is the standard
    "twice is enough" result and costs one extra ``(j+1) x n`` product.

    ``partial_dot`` work (the Hessenberg, the Givens rotations, the triangular
    solve) runs in host NumPy on ``(m+1) x m`` arrays.  Only the length-``n``
    vectors stay on the device, so nothing here transfers ``O(n)`` data per
    iteration.
    """
    b = jnp.asarray(b)
    if b.ndim != 1:
        raise ValueError(
            f"gmres_eager expects a 1-D right-hand side; got shape {b.shape}"
        )
    if jnp.iscomplexobj(b):
        raise TypeError(
            "gmres_eager is real-only: pass the real embedding of a complex "
            "system.  Its callers' operators are real-linear (a VJP that "
            "conjugates is not complex-linear), so a complex Krylov space "
            "would not span the right thing."
        )

    n = int(b.shape[0])
    m = int(min(int(restart), n))
    if m < 1:
        raise ValueError(
            f"restart must be >= 1 (and b non-empty); got restart={restart}, n={n}"
        )

    bnorm = float(jnp.linalg.norm(b))
    if not math.isfinite(bnorm):
        # Nothing to solve against.  Return the start point and a residual the
        # caller's gate cannot mistake for convergence.
        x_bad = b if x0 is None else jnp.asarray(x0)
        return x_bad, float("inf")
    if bnorm == 0.0:
        return jnp.zeros_like(b), 0.0

    target = max(float(tol) * bnorm, float(atol))

    x = b if x0 is None else jnp.asarray(x0)
    r = b - matvec(x)
    beta = float(jnp.linalg.norm(r))

    for _restart_index in range(int(maxiter)):
        if not math.isfinite(beta):
            return x, float("inf")
        if beta <= target:
            break

        # V holds the orthonormal Krylov basis, one row per vector, on device.
        V = jnp.zeros((m + 1, n), dtype=b.dtype).at[0].set(r / beta)
        H = np.zeros((m + 1, m), dtype=np.float64)
        cs = np.zeros(m, dtype=np.float64)
        sn = np.zeros(m, dtype=np.float64)
        g = np.zeros(m + 1, dtype=np.float64)
        g[0] = beta

        k = 0  # Krylov dimension actually built this restart
        for j in range(m):
            w = matvec(V[j])

            basis = V[: j + 1]
            h1 = basis @ w
            w = w - h1 @ basis
            h2 = basis @ w  # second pass: "twice is enough"
            w = w - h2 @ basis
            H[: j + 1, j] = np.asarray(h1 + h2, dtype=np.float64)

            h_next = float(jnp.linalg.norm(w))
            H[j + 1, j] = h_next

            # Apply the rotations built so far, then build the one that kills
            # the new subdiagonal entry.
            col = H[: j + 2, j].copy()
            for i in range(j):
                upper = cs[i] * col[i] + sn[i] * col[i + 1]
                col[i + 1] = -sn[i] * col[i] + cs[i] * col[i + 1]
                col[i] = upper
            denom = math.hypot(col[j], col[j + 1])
            if denom == 0.0:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j], sn[j] = col[j] / denom, col[j + 1] / denom
            col[j] = cs[j] * col[j] + sn[j] * col[j + 1]
            col[j + 1] = 0.0
            H[: j + 2, j] = col

            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k = j + 1

            # NOTE the missing early exit.  Textbook GMRES stops the inner loop
            # once ``abs(g[k])`` -- its own residual estimate -- reaches the
            # target.  JAX's ``"batched"`` method does not: ``_gmres_batched``
            # opens with ``del ptol  # unused`` and loops on ``k < restart``
            # alone, so it always builds the whole subspace and only tests
            # convergence *between* restarts.  That is not a detail to tidy up
            # while replacing it.  On the root-implicit adjoint the extra
            # directions take the achieved residual six orders past the request
            # -- 3.5e-15 against a ``tol`` of 1e-8 -- and every caller's
            # gradient has been measured against that.  Loosening the adjoint
            # by six orders inside a change whose purpose is *memory* would be
            # untraceable later, especially as #785 established that no cheap
            # diagnostic predicts root-implicit gradient quality.  Restoring
            # the early exit is a one-line change for whoever measures its
            # effect on the gradient first.
            #
            # Happy breakdown is the one inner exit, and it is not optional:
            # the subspace is A-invariant, so extending it divides by noise.
            scale = max(abs(H[j, j]), 1.0)
            if h_next <= _BREAKDOWN_REL * scale:
                break
            V = V.at[j + 1].set(w / h_next)

        # Back-substitute the (upper-triangular) least-squares problem and step.
        y = _solve_upper_triangular(H[:k, :k], g[:k])
        x = x + jnp.asarray(y, dtype=b.dtype) @ V[:k]

        r = b - matvec(x)
        beta = float(jnp.linalg.norm(r))

    # Fail closed, one place, on every exit.  ``nan > tolerance`` is False, so a
    # NaN handed to a caller's gate reads as *converged* -- the exact shape of
    # #796 / #787 / #784, which had to be fixed on four engines separately.
    return x, (beta / bnorm if math.isfinite(beta) else float("inf"))


def _solve_upper_triangular(R: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Back-substitution that tolerates an exactly singular ``R``.

    A restart that stagnates -- which the singular ``∂_y F`` of the root-implicit
    adjoint can produce, since the environment phase gauge is an exact null
    direction -- leaves a zero on the diagonal.  Dropping that component keeps
    the step in the range of what the subspace can represent instead of
    returning ``inf`` and poisoning ``x`` for every later restart.
    """
    k = R.shape[0]
    y = np.zeros(k, dtype=np.float64)
    for i in range(k - 1, -1, -1):
        acc = rhs[i] - R[i, i + 1 :] @ y[i + 1 :]
        if R[i, i] == 0.0:
            y[i] = 0.0
        else:
            y[i] = acc / R[i, i]
    return y
