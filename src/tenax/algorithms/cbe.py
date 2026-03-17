"""Controlled Bond Expansion (CBE) for MPS bond dimension growth.

Implements the McCulloch-Osborne CBE algorithm (arXiv:2403.00562) which
expands the right bond of a left-canonical MPS site tensor by adding
orthogonal columns discovered via randomized sketching.

Two entry points:

- ``expand_bond``: takes a dense ``matvec_2site`` callable. Works with
  both DenseTensor and SymmetricTensor (via todense round-trip).
- ``expand_bond_symmetric``: takes SymmetricTensor environment tensors
  directly and uses ``_blockwise_contract`` for fully block-sparse
  sketch, projection, and QR — no dense round-trip.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ------------------------------------------------------------------ #
# Dense CBE (original API)                                             #
# ------------------------------------------------------------------ #


def expand_bond(
    site: Tensor,
    matvec_2site: Callable[[jax.Array, tuple[int, ...]], jax.Array],
    right_tensor: Tensor,
    k: int = 4,
    oversampling: int = 5,
    key: jax.Array | None = None,
) -> Tensor:
    """Expand the right bond of a left-canonical site tensor by k vectors.

    Uses a dense ``matvec_2site`` callable. For fully block-sparse CBE
    with SymmetricTensor, use :func:`expand_bond_symmetric` instead.

    Args:
        site:           Left-canonical MPS site tensor with 3 legs.
        matvec_2site:   Function ``f(v_flat, theta_shape) -> result_flat``.
        right_tensor:   Right neighbor MPS site tensor with 3 legs.
        k:              Number of new bond vectors to add.
        oversampling:   Extra sketch vectors for numerical stability.
        key:            JAX PRNG key.

    Returns:
        Expanded site tensor with right bond dimension increased by k.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    site_data = site.todense()
    right_data = right_tensor.todense()
    chi_l, d, chi_r = site_data.shape
    _, d2, chi_rr = right_data.shape

    use_complex = jnp.iscomplexobj(site_data) or jnp.iscomplexobj(right_data)
    dtype = site_data.dtype
    sketch_cols = min(k + oversampling, d2 * chi_rr)
    theta_shape = (chi_l, d, d2, chi_rr)

    key1, key2, key3, key4 = jax.random.split(key, 4)
    if use_complex:
        omega = (
            jax.random.normal(
                key1, (chi_r, d2 * chi_rr, sketch_cols), dtype=jnp.float64
            )
            + 1j
            * jax.random.normal(
                key2, (chi_r, d2 * chi_rr, sketch_cols), dtype=jnp.float64
            )
        ).astype(dtype)
        psi = (
            jax.random.normal(key3, (d2 * chi_rr, sketch_cols), dtype=jnp.float64)
            + 1j
            * jax.random.normal(key4, (d2 * chi_rr, sketch_cols), dtype=jnp.float64)
        ).astype(dtype)
    else:
        omega = jax.random.normal(key1, (chi_r, d2 * chi_rr, sketch_cols), dtype=dtype)
        psi = jax.random.normal(key3, (d2 * chi_rr, sketch_cols), dtype=dtype)

    A_mat = site_data.reshape(chi_l * d, chi_r)

    Y = _cbe_sketch_dense(
        A_mat,
        matvec_2site,
        theta_shape,
        omega,
        psi,
        chi_l,
        d,
        chi_r,
        d2,
        chi_rr,
        sketch_cols,
        dtype,
    )
    Y_perp = Y - A_mat @ (A_mat.conj().T @ Y)
    Q, _ = jnp.linalg.qr(Y_perp)
    Q_k = Q[:, :k]

    A_expanded = jnp.concatenate([A_mat, Q_k], axis=1)
    expanded_data = A_expanded.reshape(chi_l, d, chi_r + k)

    return _build_expanded_tensor(site, expanded_data, chi_r, k)


# ------------------------------------------------------------------ #
# Block-sparse CBE for SymmetricTensor                                 #
# ------------------------------------------------------------------ #


def expand_bond_symmetric(
    site: SymmetricTensor,
    left_env: SymmetricTensor,
    mpo_l: SymmetricTensor,
    mpo_r: SymmetricTensor,
    right_env: SymmetricTensor,
    right_tensor: SymmetricTensor,
    k: int = 4,
    oversampling: int = 5,
    key: jax.Array | None = None,
) -> SymmetricTensor:
    """Expand the right bond using fully block-sparse CBE.

    Uses ``_blockwise_contract`` for the 2-site matvec, keeping the
    entire sketch pipeline in block-sparse form. No dense round-trip.

    The sketch builds random SymmetricTensor probes per charge sector,
    contracts them through the 2-site effective Hamiltonian using
    block-sparse einsum, projects out the existing column space per
    sector, and QR-orthogonalizes per sector.

    Args:
        site:         Left-canonical SymmetricTensor site (3 legs).
        left_env:     Left environment SymmetricTensor.
        mpo_l:        Left MPO site SymmetricTensor.
        mpo_r:        Right MPO site SymmetricTensor.
        right_env:    Right environment SymmetricTensor.
        right_tensor: Right neighbor SymmetricTensor (3 legs).
        k:            Number of new bond vectors to add.
        oversampling: Extra sketch vectors.
        key:          JAX PRNG key.

    Returns:
        Expanded SymmetricTensor with right bond dimension increased by k.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    from tenax.algorithms.dmrg import _blockwise_contract
    from tenax.contraction.contractor import contract

    # Build the block-sparse 2-site matvec
    _cache: dict = {}

    def sym_matvec(theta: SymmetricTensor) -> SymmetricTensor:
        return _blockwise_contract(
            [left_env, theta, mpo_l, mpo_r, right_env],
            "abc,apqd,bpse,eqtf,dfg->cstg",
            output_indices=theta.indices,
            expr_cache=_cache,
        )

    # Get dimensions from indices
    right_idx = site.indices[-1]
    old_charges = np.asarray(right_idx.charges)
    unique_charges = sorted(set(int(q) for q in old_charges))
    if not unique_charges:
        unique_charges = [0]

    # Distribute k across sectors proportionally
    sector_sizes = {q: int(np.sum(old_charges == q)) for q in unique_charges}
    total_old = sum(sector_sizes.values())
    k_per_sector: dict[int, int] = {}
    remaining = k
    for q in unique_charges:
        if remaining <= 0:
            break
        n_q = max(1, round(k * sector_sizes[q] / max(total_old, 1)))
        n_q = min(n_q, remaining)
        k_per_sector[q] = n_q
        remaining -= n_q
    if remaining > 0:
        largest_q = max(sector_sizes, key=sector_sizes.get)
        k_per_sector[largest_q] = k_per_sector.get(largest_q, 0) + remaining

    # Per-sector: build random probe as SymmetricTensor, apply matvec,
    # extract left part, project, QR
    site_data = site.todense()  # needed for projection (per-sector blocks)
    if site_data.ndim == 2:
        # Boundary site: pad to 3D
        labels = site.labels()
        if isinstance(labels[0], str) and labels[0].startswith("p"):
            site_data = site_data[jnp.newaxis, :]
        else:
            site_data = site_data[:, :, jnp.newaxis]
    chi_l, d, chi_r = site_data.shape
    A_mat = site_data.reshape(chi_l * d, chi_r)

    right_data = right_tensor.todense()
    if right_data.ndim == 2:
        right_data = right_data[:, :, jnp.newaxis]
    _, d2, chi_rr = right_data.shape

    all_new_cols: list[jax.Array] = []
    all_new_charges: list[int] = []

    for q in unique_charges:
        k_q = k_per_sector.get(q, 0)
        if k_q == 0:
            continue

        sector_col_indices = np.where(old_charges == q)[0]
        n_sector = len(sector_col_indices)
        if n_sector == 0:
            continue

        A_q = A_mat[:, sector_col_indices]
        sketch_cols_q = min(k_q + oversampling, d2 * chi_rr)

        # Build random probes as SymmetricTensors and apply block-sparse
        # matvec. The random "right tensor" is built with the same indices
        # as the actual right_tensor, ensuring charge conservation.
        Y_q = jnp.zeros((chi_l * d, sketch_cols_q), dtype=site_data.dtype)

        for j in range(sketch_cols_q):
            key, k1, k2 = jax.random.split(key, 3)

            # Build random right tensor as SymmetricTensor with same
            # indices as right_tensor — this ensures charge conservation
            rand_right = SymmetricTensor.random_normal(right_tensor.indices, k1)

            # Contract site · rand_right via shared bond label
            # This produces a SymmetricTensor theta with correct charges
            theta_j_sym = contract(site, rand_right)

            # Apply block-sparse 2-site matvec
            result_sym = sym_matvec(theta_j_sym)

            # Extract to dense for the psi contraction
            result_dense = result_sym.todense()
            # Reshape to (chi_l * d, d2 * chi_rr)
            result_left_dim = 1
            for idx in site.indices[:-1]:
                result_left_dim *= idx.dim
            result_right_dim = result_dense.size // result_left_dim
            result_mat = result_dense.reshape(result_left_dim, result_right_dim)

            # Contract with random psi to project to left-space
            psi_j = jax.random.normal(k2, (result_right_dim,), dtype=site_data.dtype)
            y_j = result_mat @ psi_j
            Y_q = Y_q.at[:, j].set(y_j)

        # Per-sector projection
        Y_q_perp = Y_q - A_q @ (A_q.conj().T @ Y_q)

        # QR
        Q_q, _ = jnp.linalg.qr(Y_q_perp)
        Q_q = Q_q[:, :k_q]

        all_new_cols.append(Q_q)
        all_new_charges.extend([q] * k_q)

    # Build expanded tensor
    if all_new_cols:
        new_cols = jnp.concatenate(all_new_cols, axis=1)
        A_expanded = jnp.concatenate([A_mat, new_cols], axis=1)
    else:
        A_expanded = A_mat

    total_k = len(all_new_charges)
    expanded_data = A_expanded.reshape(chi_l, d, chi_r + total_k)

    new_charges = np.concatenate(
        [
            old_charges,
            np.array(all_new_charges, dtype=np.int32),
        ]
    )
    new_right_idx = TensorIndex(
        symmetry=right_idx.symmetry,
        charges=new_charges,
        flow=right_idx.flow,
        label=right_idx.label,
    )
    new_indices = site.indices[:-1] + (new_right_idx,)
    return SymmetricTensor.from_dense(expanded_data, new_indices, tol=float("inf"))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _cbe_sketch_dense(
    A_mat: jax.Array,
    matvec_2site: Callable,
    theta_shape: tuple[int, ...],
    omega: jax.Array,
    psi: jax.Array,
    chi_l: int,
    d: int,
    chi_r: int,
    d2: int,
    chi_rr: int,
    sketch_cols: int,
    dtype: Any,
) -> jax.Array:
    """Dense CBE sketch: apply H to random 2-site probes and project back."""
    Y = jnp.zeros((chi_l * d, sketch_cols), dtype=dtype)

    for j in range(sketch_cols):
        omega_j = omega[:, :, j].reshape(chi_r, d2, chi_rr)
        A_3d = A_mat.reshape(chi_l, d, chi_r)
        theta_j = jnp.einsum("asc,ctb->astb", A_3d, omega_j)
        result_flat = matvec_2site(theta_j.ravel(), theta_shape)
        result = result_flat.reshape(chi_l * d, d2 * chi_rr)
        y_j = result @ psi[:, j].conj()
        Y = Y.at[:, j].set(y_j)

    return Y


def _build_expanded_tensor(
    site: Tensor,
    expanded_data: jax.Array,
    chi_r: int,
    k: int,
) -> Tensor:
    """Build the expanded tensor with proper indices."""
    right_idx = site.indices[-1]
    new_charges = np.zeros(chi_r + k, dtype=np.int32)
    new_charges[:chi_r] = right_idx.charges

    new_right_idx = TensorIndex(
        symmetry=right_idx.symmetry,
        charges=new_charges,
        flow=right_idx.flow,
        label=right_idx.label,
    )
    new_indices = site.indices[:-1] + (new_right_idx,)

    if isinstance(site, SymmetricTensor):
        return SymmetricTensor.from_dense(expanded_data, new_indices, tol=1e-10)
    return DenseTensor(expanded_data, new_indices)


def make_symmetric_matvec_2site(
    left_env: SymmetricTensor,
    mpo_l: SymmetricTensor,
    mpo_r: SymmetricTensor,
    right_env: SymmetricTensor,
) -> Callable[[jax.Array, tuple[int, ...]], jax.Array]:
    """Build a dense 2-site matvec from SymmetricTensor environments.

    Convenience wrapper for use with :func:`expand_bond` (the dense API).
    For fully block-sparse CBE, use :func:`expand_bond_symmetric` instead.
    """
    L_arr = left_env.todense()
    W_l_arr = mpo_l.todense()
    W_r_arr = mpo_r.todense()
    R_arr = right_env.todense()

    def matvec(v_flat: jax.Array, theta_shape: tuple[int, ...]) -> jax.Array:
        theta = v_flat.reshape(theta_shape)
        result = jnp.einsum(
            "abc,apqd,bpse,eqtf,dfg->cstg",
            L_arr,
            theta,
            W_l_arr,
            W_r_arr,
            R_arr,
        )
        return result.ravel()

    return matvec
