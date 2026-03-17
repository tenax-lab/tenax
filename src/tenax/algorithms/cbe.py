"""Controlled Bond Expansion (CBE) for MPS bond dimension growth.

Implements the McCulloch-Osborne CBE algorithm (arXiv:2403.00562) which
expands the right bond of a left-canonical MPS site tensor by adding
orthogonal columns discovered via randomized sketching.

The key idea: instead of blindly adding random directions to the bond space,
we apply the two-site Hamiltonian to random probes and project out the
existing bond space, capturing directions that the Hamiltonian would mix in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def expand_bond(
    site: Tensor,
    matvec_2site: Callable[[jax.Array, tuple[int, ...]], jax.Array],
    right_tensor: Tensor,
    k: int = 4,
    oversampling: int = 5,
    key: jax.Array | None = None,
) -> Tensor:
    """Expand the right bond of a left-canonical site tensor by k vectors.

    Uses the McCulloch-Osborne CBE algorithm: sketch random 2-site vectors
    through the Hamiltonian matvec, project out the existing column space,
    and QR-orthogonalize to get new bond directions.

    Args:
        site:           Left-canonical MPS site tensor with 3 legs
                        (left, phys, right).
        matvec_2site:   Function ``f(v_flat, theta_shape) -> result_flat``
                        applying the 2-site effective Hamiltonian.
                        ``theta_shape = (chi_l, d, d2, chi_rr)``.
        right_tensor:   Right neighbor MPS site tensor with 3 legs,
                        sharing the right bond label with ``site``.
        k:              Number of new bond vectors to add.
        oversampling:   Extra sketch vectors for numerical stability.
        key:            JAX PRNG key. Uses PRNGKey(0) if None.

    Returns:
        Expanded site tensor with right bond dimension increased by k.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    is_symmetric = isinstance(site, SymmetricTensor)

    # Extract dense arrays
    site_data = site.todense()
    right_data = right_tensor.todense()

    # Dimensions: site is (chi_l, d, chi_r), right is (chi_r, d2, chi_rr)
    chi_l, d, chi_r = site_data.shape
    _, d2, chi_rr = right_data.shape

    # Determine dtype
    use_complex = jnp.iscomplexobj(site_data) or jnp.iscomplexobj(right_data)
    dtype = site_data.dtype

    # Number of sketch columns (clamp if right space is small)
    sketch_cols = min(k + oversampling, d2 * chi_rr)

    # theta_shape for 2-site tensor
    theta_shape = (chi_l, d, d2, chi_rr)

    # Step 1: Sketch
    # Generate random Omega in (chi_r, d2, chi_rr, sketch_cols) to form
    # 2-site vectors theta_j[a,s,t,b] = A[a,s,c] * Omega[c,t,b,j]
    # Apply matvec_2site to each, then contract right legs with random Psi
    # to project back to left-site space.
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

    # A_mat: site reshaped to (chi_l*d, chi_r)
    A_mat = site_data.reshape(chi_l * d, chi_r)

    # Build Y of shape (chi_l*d, sketch_cols)
    Y = _cbe_sketch(
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

    # Step 2: Project out existing column space: Y_perp = (I - A A^dagger) Y
    Y_perp = Y - A_mat @ (A_mat.conj().T @ Y)

    # Step 3: QR factorization, take first k columns
    Q, _ = jnp.linalg.qr(Y_perp)
    Q_k = Q[:, :k]

    # Step 4: Expand: A_expanded = [A_mat | Q_k]
    A_expanded = jnp.concatenate([A_mat, Q_k], axis=1)
    expanded_data = A_expanded.reshape(chi_l, d, chi_r + k)

    # Build output tensor with proper indices
    return _build_expanded_tensor(site, expanded_data, chi_r, k, is_symmetric)


def _cbe_sketch(
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
    """Perform the CBE sketch: apply H to random 2-site probes and project back.

    For each sketch column j:
    1. Form 2-site vector theta_j[a,s,t,b] = A[a,s,c] * Omega[c,(t,b),j]
    2. Apply matvec_2site to get H @ theta_j
    3. Contract result with psi_j on the right (d2*chi_rr) legs
    """
    Y = jnp.zeros((chi_l * d, sketch_cols), dtype=dtype)

    for j in range(sketch_cols):
        # Omega_j has shape (chi_r, d2*chi_rr), reshape to (chi_r, d2, chi_rr)
        omega_j = omega[:, :, j].reshape(chi_r, d2, chi_rr)

        # theta_j[a,s,t,b] = A[a,s,c] * Omega_j[c,t,b]
        A_3d = A_mat.reshape(chi_l, d, chi_r)
        theta_j = jnp.einsum("asc,ctb->astb", A_3d, omega_j)
        theta_flat = theta_j.reshape(-1)

        # Apply 2-site matvec
        result_flat = matvec_2site(theta_flat, theta_shape)
        result = result_flat.reshape(chi_l * d, d2 * chi_rr)

        # Contract with psi_j on right legs to get left-space vector
        psi_j = psi[:, j]
        y_j = result @ psi_j.conj()

        Y = Y.at[:, j].set(y_j)

    return Y


def _build_expanded_tensor(
    site: Tensor,
    expanded_data: jax.Array,
    chi_r: int,
    k: int,
    is_symmetric: bool,
) -> Tensor:
    """Build the expanded tensor with proper indices."""
    old_indices = site.indices

    # Find the right (OUT) bond index -- last leg
    right_idx = old_indices[-1]

    # Build new right index with expanded dimension
    new_charges = np.zeros(chi_r + k, dtype=np.int32)
    new_charges[:chi_r] = right_idx.charges

    new_right_idx = TensorIndex(
        symmetry=right_idx.symmetry,
        charges=new_charges,
        flow=right_idx.flow,
        label=right_idx.label,
    )

    new_indices = old_indices[:-1] + (new_right_idx,)

    if is_symmetric:
        return SymmetricTensor.from_dense(expanded_data, new_indices, tol=1e-10)
    else:
        return DenseTensor(expanded_data, new_indices)
