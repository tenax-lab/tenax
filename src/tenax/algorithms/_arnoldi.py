"""Arnoldi iteration for spectral-radius estimation.

Used as a precheck before GMRES in implicit differentiation:
if rho(J^T) >= 1, the linear system (I - J^T) lam = rhs is ill-conditioned.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def arnoldi_spectral_radius(matvec, v0, n_iter: int = 20) -> float:
    """Estimate the spectral radius of a linear operator via Arnoldi iteration.

    Builds an upper Hessenberg matrix H of size ``n_iter x n_iter`` and returns
    the largest absolute eigenvalue of H as an approximation to the spectral
    radius of the operator.

    Parameters
    ----------
    matvec : callable
        Function mapping a flat vector to a flat vector (v -> A @ v).
    v0 : jax.Array
        Initial vector. May be real or complex.
    n_iter : int
        Number of Arnoldi steps (default 20).

    Returns
    -------
    float
        Estimated spectral radius max|eig(H)|.
    """
    n = v0.shape[0]
    n_iter = min(n_iter, n)
    dtype = v0.dtype
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        dtype = jnp.float64

    # Normalise starting vector
    v0_norm = jnp.sqrt(jnp.real(jnp.vdot(v0, v0)))
    q = v0 / v0_norm

    # Storage: Q columns and Hessenberg matrix
    Q = jnp.zeros((n, n_iter), dtype=dtype)
    Q = Q.at[:, 0].set(q)
    H = jnp.zeros((n_iter, n_iter), dtype=dtype)

    for j in range(n_iter):
        w = matvec(Q[:, j])

        # Orthogonalise against previous basis vectors
        for i in range(j + 1):
            h_ij = jnp.vdot(Q[:, i], w)  # Hermitian inner product
            H = H.at[i, j].set(h_ij)
            w = w - h_ij * Q[:, i]

        h_next = jnp.sqrt(jnp.real(jnp.vdot(w, w)))

        if j + 1 < n_iter:
            H = H.at[j + 1, j].set(h_next)
            # Guard against breakdown (h_next ~ 0)
            safe_h = jnp.where(h_next > 1e-14, h_next, 1.0)
            Q = Q.at[:, j + 1].set(w / safe_h)

    # Spectral radius from eigenvalues of H
    eigs = jnp.linalg.eigvals(H[:n_iter, :n_iter])
    rho = float(jnp.max(jnp.abs(eigs)))
    return rho


def arnoldi_spectral_radius_pytree(matvec, v0, n_iter: int = 20) -> float:
    """Estimate the spectral radius for a pytree-valued linear operator.

    Flattens the pytree to a single vector, runs :func:`arnoldi_spectral_radius`,
    and returns the result. The ``matvec`` callable should map a pytree to a
    pytree of the same structure (e.g. ``tuple[Array, ...] -> tuple[Array, ...]``).

    Parameters
    ----------
    matvec : callable
        Function mapping a pytree to a pytree of the same structure.
    v0 : pytree
        Initial pytree (used to infer structure and dtype).
    n_iter : int
        Number of Arnoldi steps (default 20).

    Returns
    -------
    float
        Estimated spectral radius.
    """
    leaves, treedef = jax.tree.flatten(v0)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.size for leaf in leaves]

    def _flatten(pytree):
        flat_leaves = jax.tree.leaves(pytree)
        return jnp.concatenate([leaf.ravel() for leaf in flat_leaves])

    def _unflatten(vec):
        parts = jnp.split(vec, jnp.cumsum(jnp.array(sizes[:-1])))
        rebuilt = [p.reshape(s) for p, s in zip(parts, shapes)]
        return treedef.unflatten(rebuilt)

    def flat_matvec(vec):
        pytree_in = _unflatten(vec)
        pytree_out = matvec(pytree_in)
        return _flatten(pytree_out)

    v0_flat = _flatten(v0)
    return arnoldi_spectral_radius(flat_matvec, v0_flat, n_iter=n_iter)
