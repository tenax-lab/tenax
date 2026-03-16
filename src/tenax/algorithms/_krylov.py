"""Lanczos-based Krylov matrix exponential for TDVP time evolution.

Computes exp(dt * A) @ v without forming the full matrix exponential,
using a Krylov subspace approximation via the Lanczos algorithm.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def krylov_expm(
    matvec: Callable[[jax.Array], jax.Array],
    v: jax.Array,
    dt: complex,
    krylov_dim: int = 20,
    tol: float = 1e-12,
) -> jax.Array:
    """Compute ``exp(dt * A) @ v`` via Lanczos iteration.

    Builds an orthonormal Krylov basis V and tridiagonal matrix T, then
    diagonalizes T and exponentiates in the small Krylov subspace.

    Args:
        matvec:     Function applying the (Hermitian) matrix A to a vector.
        v:          Starting vector.
        dt:         Time step (complex for real-time, real negative for
                    imaginary-time evolution).
        krylov_dim: Maximum dimension of the Krylov subspace.
        tol:        Convergence tolerance on the Lanczos residual norm.

    Returns:
        The vector ``exp(dt * A) @ v``.
    """
    norm_v = jnp.linalg.norm(v)

    # Handle zero vector
    if float(norm_v) < 1e-15:
        return v

    # Handle dt=0: exp(0) = I
    if dt == 0:
        return v

    q = v / norm_v

    # Krylov basis and tridiagonal coefficients
    basis: list[jax.Array] = [q]
    alphas: list[jax.Array] = []
    betas: list[jax.Array] = []

    for step in range(krylov_dim):
        w = matvec(basis[-1])
        alpha = jnp.dot(basis[-1].conj(), w).real
        alphas.append(alpha)

        w = w - alpha * basis[-1]
        if step > 0:
            w = w - betas[-1] * basis[-2]

        beta = jnp.linalg.norm(w).real
        betas.append(beta)

        if float(beta) < tol:
            break

        basis.append(w / beta)

    n = len(alphas)
    if n == 0:
        return v

    # Build tridiagonal matrix T
    alphas_arr = jnp.stack(alphas)
    T = jnp.diag(alphas_arr)
    if n > 1:
        betas_arr = jnp.stack(betas[: n - 1])
        T = T + jnp.diag(betas_arr, k=1) + jnp.diag(betas_arr, k=-1)

    # Diagonalize T: T = P diag(lambda) P^dag
    eigvals, P = jnp.linalg.eigh(T)

    # Exponentiate: exp(dt * T) = P diag(exp(dt * lambda)) P^dag
    exp_eigvals = jnp.exp(dt * eigvals)
    exp_T_e1 = P @ (exp_eigvals * P[0, :].conj())

    # Map back to full space: result = ||v|| * V @ exp(dt*T) @ e_1
    basis_stacked = jnp.stack(basis[:n], axis=0)  # (n, vec_dim)
    result = norm_v * jnp.tensordot(exp_T_e1, basis_stacked, axes=1)

    return result
