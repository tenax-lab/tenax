"""JIT-compatible SVD + truncation for dense and block-sparse tensors.

These functions are designed for the GPU/TPU DMRG path where the entire
sweep must compile to a single XLA program via ``jax.lax.scan``.

Key insight: ``jax.lax.top_k`` returns static-shape output, making the
SVD+truncation fully JIT-compatible without host-device synchronization.

Public API:
    padded_svd_dense(theta, chi_max) -> (U, s, Vh)
    padded_svd_blocks(pba, chi_max) -> (top_values, per_block_keeps)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ------------------------------------------------------------------ #
# Dense path                                                           #
# ------------------------------------------------------------------ #


def padded_svd_dense(
    theta: jax.Array,
    chi_max: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """SVD + truncation on a dense 2-site theta tensor, JIT-compatible.

    Args:
        theta: ``(chi_l, d_l, d_r, chi_r)`` two-site wavefunction.
        chi_max: Keep at most this many singular values.

    Returns:
        U:  ``(chi_l * d_l, chi_max)`` left singular vectors (zero-padded if
            rank < chi_max).
        s:  ``(chi_max,)`` singular values (zero-padded if rank < chi_max).
        Vh: ``(chi_max, d_r * chi_r)`` right singular vectors (zero-padded if
            rank < chi_max).
    """
    shape = theta.shape
    m = shape[0] * shape[1]
    n = shape[2] * shape[3]
    mat = theta.reshape(m, n)

    # Full (thin) SVD -- always returns min(m, n) singular values
    U_full, s_full, Vh_full = jnp.linalg.svd(mat, full_matrices=False)
    # U_full: (m, k), s_full: (k,), Vh_full: (k, n) where k = min(m, n)
    k = s_full.shape[0]

    # Truncate to chi_max -- static slice (always valid, pads if k < chi_max)
    U_trunc = U_full[:, :chi_max]
    s_trunc = s_full[:chi_max]
    Vh_trunc = Vh_full[:chi_max, :]

    # Pad to exactly chi_max if rank k < chi_max.
    # Since k = min(m, n) is known at trace time, the pad sizes are
    # static and jnp.pad produces a no-op when the pad size is 0.
    U_out = jnp.pad(U_trunc, ((0, 0), (0, max(chi_max - k, 0))))
    s_out = jnp.pad(s_trunc, (0, max(chi_max - k, 0)))
    Vh_out = jnp.pad(Vh_trunc, ((0, max(chi_max - k, 0)), (0, 0)))

    return U_out, s_out, Vh_out


# ------------------------------------------------------------------ #
# Block-sparse path                                                    #
# ------------------------------------------------------------------ #


def padded_svd_blocks(
    pba,  # PaddedBlockArray
    chi_max: int,
) -> tuple[jax.Array, jax.Array]:
    """Block-sparse SVD with global truncation via ``jax.lax.top_k``.

    Vmaps ``jnp.linalg.svd`` over all blocks in the PaddedBlockArray,
    then selects the globally largest ``chi_max`` singular values across
    all charge sectors.

    This is fully JIT-compatible: ``jax.lax.top_k`` returns static-shape
    output, so no host-device sync is needed.

    Args:
        pba: PaddedBlockArray with ``(num_blocks, M_max, N_max)`` data.
        chi_max: Maximum total bond dimension across all sectors.

    Returns:
        top_values:      ``(chi_max,)`` globally sorted singular values
                         (zero-padded if fewer than chi_max nonzero values
                         exist).
        per_block_keeps: ``(num_blocks,)`` int32 array giving the number of
                         singular values kept from each charge sector.
    """
    num_blocks = pba.data.shape[0]
    M_max = pba.data.shape[1]
    N_max = pba.data.shape[2]
    sv_per_block = min(M_max, N_max)

    # vmap SVD over all blocks
    # Each block is (M_max, N_max); SVD produces U (M_max, sv_per_block),
    # s (sv_per_block,), Vh (sv_per_block, N_max).
    _, s_all, _ = jax.vmap(lambda m: jnp.linalg.svd(m, full_matrices=False))(pba.data)
    # s_all: (num_blocks, sv_per_block)

    # Mask out padding: singular values from padded rows/cols are spurious.
    # A block with actual shape (m_i, n_i) has at most min(m_i, n_i) real
    # singular values. The rest (from zero-padded region) should be zeroed.
    # We build a per-block mask: s_all[i, j] is valid iff j < min(m_i, n_i).
    block_sv_limits = jnp.array(
        [min(m, n) for m, n in pba.block_shapes], dtype=jnp.int32
    )
    sv_range = jnp.arange(sv_per_block)
    sv_valid_mask = sv_range[None, :] < block_sv_limits[:, None]
    # (num_blocks, sv_per_block) bool

    # Zero out spurious singular values from padding
    s_masked = jnp.where(sv_valid_mask, s_all, 0.0)

    # Global truncation: find top chi_max singular values across all sectors
    s_flat = s_masked.ravel()  # (num_blocks * sv_per_block,)
    total_sv = s_flat.shape[0]
    k = min(chi_max, total_sv)

    top_values, top_indices = jax.lax.top_k(s_flat, k)

    # Pad to chi_max if fewer total singular values available
    if k < chi_max:
        top_values = jnp.pad(top_values, (0, chi_max - k))
        top_indices = jnp.pad(top_indices, (0, chi_max - k), constant_values=-1)

    # Compute per-block keep counts from top_indices.
    # top_indices are flat indices into s_masked.ravel().
    # Block id = index // sv_per_block.
    block_ids = top_indices // sv_per_block

    # Count how many of the top-k belong to each block
    # Using one_hot + sum for JIT compatibility (no dynamic indexing)
    one_hot = jax.nn.one_hot(block_ids, num_blocks, dtype=jnp.int32)
    # Handle padded indices (value -1): they map to block_id < 0, but
    # one_hot with num_classes=num_blocks will produce all-zeros for
    # out-of-range indices, which is correct.
    per_block_keeps = jnp.sum(one_hot, axis=0)  # (num_blocks,)

    return top_values, per_block_keeps
