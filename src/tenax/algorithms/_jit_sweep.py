"""JIT-compatible dense DMRG operations for GPU/TPU acceleration.

Functions in this module operate on raw JAX arrays (not Tensor objects) and
produce padded outputs with fixed shapes, making them compatible with
``jax.jit`` and ``jax.lax.scan``.

Public API:
    update_left_env_dense_jit  -- left environment update
    update_right_env_dense_jit -- right environment update
    effective_ham_matvec_dense  -- 2-site effective Hamiltonian matvec
    lanczos_ground_state_dense -- JIT-compatible Lanczos eigensolver
    jit_dmrg_sweep_dense       -- full DMRG sweep via lax.scan
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from tenax.algorithms._padded_linalg import padded_svd_dense


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


# ------------------------------------------------------------------ #
# Full lax.scan DMRG sweep (dense path)                               #
# ------------------------------------------------------------------ #


def _update_left_env_scan(
    L_env: jax.Array,
    A: jax.Array,
    W: jax.Array,
) -> jax.Array:
    """Left env update for scan (no padding -- shapes are already padded).

    Contracts: new_L[d,e,f] = L[a,b,c] * A[a,p,d] * W[b,p,x,e] * conj(A)[c,x,f]
    """
    return jnp.einsum("abc,apd,bpxe,cxf->def", L_env, A, W, jnp.conj(A))


def _update_right_env_scan(
    R_env: jax.Array,
    B: jax.Array,
    W: jax.Array,
) -> jax.Array:
    """Right env update for scan (no padding -- shapes are already padded).

    Contracts: new_R[d,e,f] = R[a,b,c] * B[d,p,a] * W[e,p,x,b] * conj(B)[f,x,c]
    """
    return jnp.einsum("abc,dpa,epxb,fxc->def", R_env, B, W, jnp.conj(B))


def _build_initial_left_envs(
    mps_stack: jax.Array,
    W_stack: jax.Array,
    L: int,
    chi_max: int,
    D_w: int,
) -> jax.Array:
    """Build all left environments from scratch.

    Args:
        mps_stack: ``(L, chi_max, d, chi_max)`` padded MPS tensors.
        W_stack:   ``(L, D_w, d, d, D_w)`` padded MPO tensors.
        L:         Number of sites.
        chi_max:   Maximum bond dimension.
        D_w:       Padded MPO bond dimension.

    Returns:
        Left environments ``(L+1, chi_max, D_w, chi_max)`` where
        ``left_envs[0]`` is the trivial boundary.
    """
    dtype = mps_stack.dtype
    left_envs = jnp.zeros((L + 1, chi_max, D_w, chi_max), dtype=dtype)
    # Trivial left boundary: identity at (0,0,0)
    left_envs = left_envs.at[0, 0, 0, 0].set(1.0)

    def scan_fn(L_env, site_data):
        A, W = site_data
        new_L = _update_left_env_scan(L_env, A, W)
        return new_L, new_L

    # Scan over sites 0..L-2 (we need L_envs[1]..L_envs[L-1])
    init_L = left_envs[0]
    _, all_left = jax.lax.scan(scan_fn, init_L, (mps_stack[: L - 1], W_stack[: L - 1]))
    # all_left: (L-1, chi_max, D_w, chi_max)
    left_envs = left_envs.at[1:L].set(all_left)
    return left_envs


def _build_initial_right_envs(
    mps_stack: jax.Array,
    W_stack: jax.Array,
    L: int,
    chi_max: int,
    D_w: int,
) -> jax.Array:
    """Build all right environments from scratch.

    Args:
        mps_stack: ``(L, chi_max, d, chi_max)`` padded MPS tensors.
        W_stack:   ``(L, D_w, d, d, D_w)`` padded MPO tensors.
        L:         Number of sites.
        chi_max:   Maximum bond dimension.
        D_w:       Padded MPO bond dimension.

    Returns:
        Right environments ``(L+1, chi_max, D_w, chi_max)`` where
        ``right_envs[L]`` is the trivial boundary.
    """
    dtype = mps_stack.dtype
    right_envs = jnp.zeros((L + 1, chi_max, D_w, chi_max), dtype=dtype)
    # Trivial right boundary: identity at (0,0,0)
    right_envs = right_envs.at[L, 0, 0, 0].set(1.0)

    def scan_fn(R_env, site_data):
        B, W = site_data
        new_R = _update_right_env_scan(R_env, B, W)
        return new_R, new_R

    # Scan from site L-1 down to site 1 (reversed)
    # We need right_envs[L-1]..right_envs[1]
    mps_rev = mps_stack[1:][::-1]  # sites L-1, L-2, ..., 1
    W_rev = W_stack[1:][::-1]
    init_R = right_envs[L]
    _, all_right_rev = jax.lax.scan(scan_fn, init_R, (mps_rev, W_rev))
    # all_right_rev: (L-1, chi_max, D_w, chi_max) for sites L-1..1
    # Reverse back so index 0 = site 1, index L-2 = site L-1
    all_right = all_right_rev[::-1]
    right_envs = right_envs.at[1:L].set(all_right)
    return right_envs


def _scan_left_to_right(
    mps_stack: jax.Array,
    left_envs: jax.Array,
    right_envs: jax.Array,
    W_stack: jax.Array,
    chi_max: int,
    D_w: int,
    d: int,
    L: int,
    lanczos_max_iter: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Left-to-right half-sweep via ``jax.lax.scan``.

    For each bond ``i`` in ``0..L-2``:
      1. Build 2-site theta from ``mps[i]`` and ``mps[i+1]``
      2. Lanczos eigensolver with effective Hamiltonian
      3. SVD + truncation
      4. Update ``mps[i]`` (left-canonical A) and ``mps[i+1]`` (sVh)
      5. Update ``left_envs[i+1]``

    Args:
        mps_stack:   ``(L, chi_max, d, chi_max)``
        left_envs:   ``(L+1, chi_max, D_w, chi_max)``
        right_envs:  ``(L+1, chi_max, D_w, chi_max)``
        W_stack:     ``(L, D_w, d, d, D_w)``
        chi_max:     Maximum bond dimension.
        D_w:         Padded MPO bond dimension.
        d:           Physical dimension.
        L:           Number of sites.
        lanczos_max_iter: Number of Lanczos steps.

    Returns:
        ``(mps_stack, left_envs, energy)`` with updated MPS and left environments.
    """

    def step_fn(carry, idx):
        mps, L_envs = carry
        i = idx

        # Load tensors for this 2-site update
        A_i = mps[i]  # (chi_max, d, chi_max)
        A_ip1 = mps[i + 1]  # (chi_max, d, chi_max)
        L_env = L_envs[i]  # (chi_max, D_w, chi_max)
        R_env = right_envs[i + 2]  # (chi_max, D_w, chi_max)
        W_i = W_stack[i]  # (D_w, d, d, D_w)
        W_ip1 = W_stack[i + 1]  # (D_w, d, d, D_w)

        # Build theta = contract(A_i, A_ip1) = sum_j A_i[a,p,j] * A_ip1[j,q,b]
        theta = jnp.einsum("ipj,jqk->ipqk", A_i, A_ip1)
        # theta: (chi_max, d, d, chi_max)
        theta_flat = theta.ravel()

        # Lanczos: solve for ground state of effective Hamiltonian
        def matvec(v):
            return effective_ham_matvec_dense(v, L_env, W_i, W_ip1, R_env, chi_max)

        E, theta_opt_flat = lanczos_ground_state_dense(
            matvec, theta_flat, lanczos_max_iter
        )
        theta_opt = theta_opt_flat.reshape(chi_max, d, d, chi_max)

        # SVD + truncation
        U, s, Vh = padded_svd_dense(theta_opt, chi_max)
        # U:  (chi_max * d, chi_max)
        # s:  (chi_max,)
        # Vh: (chi_max, d * chi_max)

        # Reshape to MPS site tensors
        # A_new = U reshaped to (chi_max, d, chi_max) -- left canonical
        A_new = U.reshape(chi_max, d, chi_max)

        # sVh = diag(s) @ Vh, reshaped to (chi_max, d, chi_max)
        sVh = (s[:, None] * Vh).reshape(chi_max, d, chi_max)

        # Update MPS
        mps = mps.at[i].set(A_new)
        mps = mps.at[i + 1].set(sVh)

        # Update left environment
        new_L = _update_left_env_scan(L_env, A_new, W_i)
        L_envs = L_envs.at[i + 1].set(new_L)

        return (mps, L_envs), E

    init_carry = (mps_stack, left_envs)
    indices = jnp.arange(L - 1)
    (mps_out, L_envs_out), energies = jax.lax.scan(step_fn, init_carry, indices)

    # Last energy from the sweep
    last_energy = energies[-1]
    return mps_out, L_envs_out, last_energy


def _scan_right_to_left(
    mps_stack: jax.Array,
    left_envs: jax.Array,
    right_envs: jax.Array,
    W_stack: jax.Array,
    chi_max: int,
    D_w: int,
    d: int,
    L: int,
    lanczos_max_iter: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Right-to-left half-sweep via ``jax.lax.scan``.

    Mirrors _scan_left_to_right but processes bonds from L-2 down to 0.

    Returns:
        ``(mps_stack, right_envs, energy)`` with updated MPS and right environments.
    """

    def step_fn(carry, scan_idx):
        mps, R_envs = carry
        # Map scan index 0..L-2 to actual site index L-2..0
        i = (L - 2) - scan_idx

        A_i = mps[i]
        A_ip1 = mps[i + 1]
        L_env = left_envs[i]
        R_env = R_envs[i + 2]
        W_i = W_stack[i]
        W_ip1 = W_stack[i + 1]

        # Build theta
        theta = jnp.einsum("ipj,jqk->ipqk", A_i, A_ip1)
        theta_flat = theta.ravel()

        # Lanczos
        def matvec(v):
            return effective_ham_matvec_dense(v, L_env, W_i, W_ip1, R_env, chi_max)

        E, theta_opt_flat = lanczos_ground_state_dense(
            matvec, theta_flat, lanczos_max_iter
        )
        theta_opt = theta_opt_flat.reshape(chi_max, d, d, chi_max)

        # SVD + truncation
        U, s, Vh = padded_svd_dense(theta_opt, chi_max)

        # Right-to-left: B = Vh (right-canonical), absorb s into A = Us
        A_new = U.reshape(chi_max, d, chi_max)
        B_new = Vh.reshape(chi_max, d, chi_max)

        # Us = A_new with singular values absorbed
        # Us[a, p, k] = sum_k' A_new[a, p, k'] * diag(s)[k', k]
        Us = A_new * s[None, None, :]

        mps = mps.at[i].set(Us)
        mps = mps.at[i + 1].set(B_new)

        # Update right environment
        new_R = _update_right_env_scan(R_env, B_new, W_ip1)
        R_envs = R_envs.at[i + 1].set(new_R)

        return (mps, R_envs), E

    init_carry = (mps_stack, right_envs)
    indices = jnp.arange(L - 1)
    (mps_out, R_envs_out), energies = jax.lax.scan(step_fn, init_carry, indices)

    last_energy = energies[-1]
    return mps_out, R_envs_out, last_energy


def jit_dmrg_sweep_dense(
    mps_tensors: list[jax.Array],
    mpo_tensors: list[jax.Array],
    chi_max: int,
    num_sweeps: int = 10,
    lanczos_max_iter: int = 20,
) -> list[float]:
    """Run full DMRG sweeps compiled to a single XLA program via ``jax.lax.scan``.

    Operates on raw JAX arrays (not Tensor objects). All MPS and MPO tensors
    are padded to uniform shapes so the entire sweep can be JIT-compiled.

    Args:
        mps_tensors:  List of L MPS site tensors, each ``(chi_l, d, chi_r)``.
        mpo_tensors:  List of L MPO site tensors, each ``(D_w_l, d, d, D_w_r)``.
        chi_max:      Maximum bond dimension for MPS.
        num_sweeps:   Number of full (L->R + R->L) sweeps.
        lanczos_max_iter: Number of Lanczos iterations per site update.

    Returns:
        List of energies, one per sweep (energy from the last bond update of
        the right-to-left half-sweep).
    """
    L = len(mps_tensors)
    d = mps_tensors[0].shape[1]  # physical dimension

    # Determine max MPO bond dimension for padding
    D_w_max = max(max(W.shape[0], W.shape[3]) for W in mpo_tensors)

    # Pad and stack MPS tensors to (L, chi_max, d, chi_max)
    dtype = mps_tensors[0].dtype
    mps_stack = jnp.zeros((L, chi_max, d, chi_max), dtype=dtype)
    for i, M in enumerate(mps_tensors):
        chi_l, _, chi_r = M.shape
        mps_stack = mps_stack.at[i, :chi_l, :, :chi_r].set(M)

    # Pad and stack MPO tensors to (L, D_w_max, d, d, D_w_max)
    W_stack = jnp.zeros((L, D_w_max, d, d, D_w_max), dtype=dtype)
    for i, W in enumerate(mpo_tensors):
        dw_l, _, _, dw_r = W.shape
        W_stack = W_stack.at[i, :dw_l, :, :, :dw_r].set(W)

    # Run sweeps via the JIT-compiled inner function
    energies = _jit_sweep_loop(
        mps_stack, W_stack, L, chi_max, D_w_max, d, num_sweeps, lanczos_max_iter
    )

    return [float(e) for e in energies]


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def _jit_sweep_loop(
    mps_stack: jax.Array,
    W_stack: jax.Array,
    L: int,
    chi_max: int,
    D_w: int,
    d: int,
    num_sweeps: int,
    lanczos_max_iter: int,
) -> jax.Array:
    """Inner JIT-compiled sweep loop.

    All integer parameters are static (traced at compile time) so that
    shapes are known. The function body uses ``jax.lax.fori_loop`` over
    sweeps, with ``jax.lax.scan`` inside each half-sweep.

    Returns:
        ``(num_sweeps,)`` array of energies.
    """
    # Build initial environments
    left_envs = _build_initial_left_envs(mps_stack, W_stack, L, chi_max, D_w)
    right_envs = _build_initial_right_envs(mps_stack, W_stack, L, chi_max, D_w)
    energies = jnp.zeros(num_sweeps, dtype=mps_stack.dtype)

    def sweep_body(sweep_idx, state):
        mps, L_envs, R_envs, Es = state

        # Left-to-right half-sweep
        mps, L_envs, _ = _scan_left_to_right(
            mps, L_envs, R_envs, W_stack, chi_max, D_w, d, L, lanczos_max_iter
        )

        # Rebuild right environments from updated MPS
        R_envs = _build_initial_right_envs(mps, W_stack, L, chi_max, D_w)

        # Right-to-left half-sweep
        mps, R_envs, E_rl = _scan_right_to_left(
            mps, L_envs, R_envs, W_stack, chi_max, D_w, d, L, lanczos_max_iter
        )

        # Rebuild left environments from updated MPS for next sweep
        L_envs = _build_initial_left_envs(mps, W_stack, L, chi_max, D_w)

        Es = Es.at[sweep_idx].set(E_rl)
        return mps, L_envs, R_envs, Es

    init_state = (mps_stack, left_envs, right_envs, energies)
    _, _, _, energies = jax.lax.fori_loop(0, num_sweeps, sweep_body, init_state)

    return energies
