"""JIT-compatible dense DMRG operations for GPU/TPU acceleration.

Functions in this module operate on raw JAX arrays (not Tensor objects) and
produce padded outputs with fixed shapes, making them compatible with
``jax.jit`` and ``jax.lax.scan``.

Public API:
    update_left_env_dense_jit  -- left environment update
    update_right_env_dense_jit -- right environment update
    effective_ham_matvec_dense  -- 2-site effective Hamiltonian matvec
    lanczos_ground_state_dense -- JIT-compatible Lanczos eigensolver
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def update_left_env_dense_jit(
    L_env: jax.Array,
    A: jax.Array,
    W: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Compute a left environment update and pad output to fixed shape.

    Contracts: new_L[d,e,f] = L[a,b,c] * A[a,p,d] * W[b,p,x,e] * conj(A)[c,x,f]

    This matches the einsum ``"abc,apd,bpxe,cxf->def"`` from ``_update_left_env``.

    Args:
        L_env: Left environment, shape ``(chi_l, D_w_in, chi_l)``.
            May already be padded to ``(chi_max, D_w, chi_max)`` -- zero
            padding is handled correctly by einsum.
        A: MPS site tensor, shape ``(chi_l, d, chi_r)``.
            May be padded to ``(chi_max, d, chi_max)``.
        W: MPO site tensor, shape ``(D_w_in, d, d, D_w_out)``.
            Not padded (MPO bond dimension is fixed).
        chi_max: Maximum bond dimension for padding.

    Returns:
        Padded left environment of shape ``(chi_max, D_w_out, chi_max)``.
    """
    D_w_out = W.shape[3]

    new_L = jnp.einsum("abc,apd,bpxe,cxf->def", L_env, A, W, jnp.conj(A))

    # Pad to (chi_max, D_w_out, chi_max)
    chi_r = new_L.shape[0]
    padded = jnp.zeros((chi_max, D_w_out, chi_max), dtype=new_L.dtype)
    padded = padded.at[:chi_r, :D_w_out, :chi_r].set(new_L)
    return padded


def update_right_env_dense_jit(
    R_env: jax.Array,
    B: jax.Array,
    W: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Compute a right environment update and pad output to fixed shape.

    Contracts: new_R[d,e,f] = R[a,b,c] * B[d,p,a] * W[e,p,x,b] * conj(B)[f,x,c]

    This matches the einsum ``"abc,dpa,epxb,fxc->def"`` from ``_update_right_env``.

    Args:
        R_env: Right environment, shape ``(chi_r, D_w_in, chi_r)``.
            May already be padded to ``(chi_max, D_w, chi_max)``.
        B: MPS site tensor, shape ``(chi_l, d, chi_r)``.
            May be padded to ``(chi_max, d, chi_max)``.
        W: MPO site tensor, shape ``(D_w_out, d, d, D_w_in)``.
            Not padded (MPO bond dimension is fixed).
        chi_max: Maximum bond dimension for padding.

    Returns:
        Padded right environment of shape ``(chi_max, D_w_out, chi_max)``.
    """
    D_w_out = W.shape[0]

    new_R = jnp.einsum("abc,dpa,epxb,fxc->def", R_env, B, W, jnp.conj(B))

    # Pad to (chi_max, D_w_out, chi_max)
    chi_l = new_R.shape[0]
    padded = jnp.zeros((chi_max, D_w_out, chi_max), dtype=new_R.dtype)
    padded = padded.at[:chi_l, :D_w_out, :chi_l].set(new_R)
    return padded


# ------------------------------------------------------------------ #
# Effective Hamiltonian matvec (padded dense)                         #
# ------------------------------------------------------------------ #


def effective_ham_matvec_dense(
    theta_flat: jax.Array,
    L_env: jax.Array,
    W_l: jax.Array,
    W_r: jax.Array,
    R_env: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Apply the 2-site effective Hamiltonian to theta via einsum.

    All MPS/environment arrays are padded to ``chi_max`` on their virtual
    bond dimensions. The MPO tensors ``W_l`` and ``W_r`` are *not* padded
    (their bond dimension is fixed by the Hamiltonian).

    The einsum subscripts match exactly those in
    :func:`tenax.algorithms.dmrg._effective_hamiltonian_matvec`::

        "abc,apqd,bpse,eqtf,dfg->cstg"

    Args:
        theta_flat: Flattened 2-site wavefunction of length
            ``chi_max * d_l * d_r * chi_max``.
        L_env: Left environment, shape ``(chi_max, D_w_l, chi_max)``.
        W_l: Left MPO site, shape ``(D_w_l, d_l, d_l, D_w_m)``.
        W_r: Right MPO site, shape ``(D_w_m, d_r, d_r, D_w_r)``.
        R_env: Right environment, shape ``(chi_max, D_w_r, chi_max)``.
        chi_max: Maximum bond dimension (static for JIT).

    Returns:
        Flattened result of H_eff @ theta, length
        ``chi_max * d_l * d_r * chi_max``.
    """
    d_l = W_l.shape[1]
    d_r = W_r.shape[1]
    theta = theta_flat.reshape(chi_max, d_l, d_r, chi_max)

    result = jnp.einsum(
        "abc,apqd,bpse,eqtf,dfg->cstg",
        L_env,
        theta,
        W_l,
        W_r,
        R_env,
    )
    return result.ravel()


# ------------------------------------------------------------------ #
# Lanczos eigensolver (padded dense, fully JIT-compatible)            #
# ------------------------------------------------------------------ #


def lanczos_ground_state_dense(
    matvec: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    max_iter: int,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compatible Lanczos eigensolver for the smallest eigenvalue.

    Uses ``jax.lax.fori_loop`` for iteration (static number of steps) and
    ``jax.lax.scan`` for full reorthogonalization. Pre-allocates a basis
    matrix of shape ``(max_iter, n)`` and solves the tridiagonal Ritz
    problem via ``jnp.linalg.eigh``.

    No host-device synchronization (no ``float()`` calls), so this function
    is fully compatible with ``jax.jit``.

    Args:
        matvec: Function that applies the (effective) Hamiltonian to a
            flattened vector and returns the result as a flat vector of
            the same length.
        v0: Initial guess vector (will be normalized internally).
        max_iter: Number of Lanczos steps (static, not data-dependent).

    Returns:
        ``(eigenvalue, eigenvector)`` where *eigenvalue* is a scalar
        JAX array (not a Python float) and *eigenvector* is the
        corresponding ground-state vector (same shape as ``v0``).
    """
    n = v0.shape[0]
    v0_normed = v0 / (jnp.linalg.norm(v0) + 1e-15)

    # Pre-allocate Krylov basis and tridiagonal coefficients
    basis = jnp.zeros((max_iter, n), dtype=v0.dtype)
    basis = basis.at[0].set(v0_normed)
    alphas = jnp.zeros(max_iter, dtype=v0.dtype)
    betas = jnp.zeros(max_iter, dtype=v0.dtype)

    def body(step, state):
        basis, alphas, betas = state
        v = basis[step]
        w = matvec(v)

        # Diagonal element of the tridiagonal matrix
        alpha = jnp.dot(v.conj(), w).real
        alphas = alphas.at[step].set(alpha)

        # Subtract projections onto current and previous basis vectors
        w = w - alpha * v
        v_prev = jnp.where(step > 0, basis[step - 1], jnp.zeros_like(v))
        beta_prev = jnp.where(step > 0, betas[step - 1], 0.0)
        w = w - beta_prev * v_prev

        # Full reorthogonalization via scan (prevents ghost eigenvalues)
        def reorth_step(w, q):
            overlap = jnp.dot(q.conj(), w)
            return w - overlap * q, None

        w, _ = jax.lax.scan(reorth_step, w, basis)

        # Off-diagonal element
        beta = jnp.linalg.norm(w)
        betas = betas.at[step].set(beta)

        # Normalized new basis vector (zero vector if beta ~ 0)
        v_new = jnp.where(beta > 1e-15, w / beta, jnp.zeros_like(w))
        basis = jnp.where(
            step + 1 < max_iter,
            basis.at[step + 1].set(v_new),
            basis,
        )
        return basis, alphas, betas

    basis, alphas, betas = jax.lax.fori_loop(0, max_iter, body, (basis, alphas, betas))

    # Solve the tridiagonal Ritz eigenproblem
    T = jnp.diag(alphas) + jnp.diag(betas[:-1], k=1) + jnp.diag(betas[:-1], k=-1)
    eigvals, eigvecs = jnp.linalg.eigh(T)

    # Ground state: smallest eigenvalue (eigh returns sorted)
    coefs = eigvecs[:, 0]
    eigenvector = coefs @ basis  # (max_iter,) @ (max_iter, n) -> (n,)
    eigenvector = eigenvector / (jnp.linalg.norm(eigenvector) + 1e-15)

    return eigvals[0], eigenvector
