"""Tests for padded SVD functions (JIT-compatible truncation)."""

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor
from tenax.linalg import svd as tenax_svd

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_symmetric_matrix(seed=42, chi=8):
    """Build a 2-index SymmetricTensor (matrix) for SVD testing.

    Creates a random U(1)-symmetric matrix with charge sectors that have
    varying multiplicities, giving blocks of different sizes.
    """
    sym = U1Symmetry()
    # Build charges with multiplicity > 1 so blocks are non-trivial.
    # e.g. chi=8 -> charges [-1,-1,0,0,0,0,1,1] giving blocks 2x2, 4x4, 2x2
    half = chi // 4 if chi >= 4 else 1
    mid = chi - 2 * half
    charges = np.array([-1] * half + [0] * mid + [1] * half, dtype=np.int32)

    indices = (
        TensorIndex(sym, charges, FlowDirection.IN, label="row"),
        TensorIndex(sym, sym.dual(charges), FlowDirection.OUT, label="col"),
    )
    key = jax.random.PRNGKey(seed)
    return SymmetricTensor.random_normal(indices, key)


# ------------------------------------------------------------------ #
# Dense SVD tests                                                      #
# ------------------------------------------------------------------ #


class TestPaddedSVDDense:
    def test_dense_svd_truncation_matches_numpy(self):
        """Padded SVD with truncation matches numpy SVD + truncation."""
        from tenax.algorithms._padded_linalg import padded_svd_dense

        key = jax.random.PRNGKey(0)
        theta = jax.random.normal(key, (8, 2, 2, 8))
        chi_max = 6

        # Reference: numpy SVD
        mat = np.array(theta).reshape(16, 16)
        U_ref, s_ref, Vh_ref = np.linalg.svd(mat, full_matrices=False)
        U_ref = U_ref[:, :chi_max]
        s_ref = s_ref[:chi_max]
        Vh_ref = Vh_ref[:chi_max, :]

        # Padded JIT version
        U_jit, s_jit, Vh_jit = padded_svd_dense(theta, chi_max)

        assert U_jit.shape == (16, chi_max)
        assert s_jit.shape == (chi_max,)
        assert Vh_jit.shape == (chi_max, 16)

        np.testing.assert_allclose(np.array(s_jit), s_ref, atol=1e-10)
        # U and Vh may differ by column sign, so check reconstruction
        recon = U_jit @ jnp.diag(s_jit) @ Vh_jit
        recon_ref = U_ref @ np.diag(s_ref) @ Vh_ref
        np.testing.assert_allclose(np.array(recon), recon_ref, atol=1e-10)

    def test_dense_svd_padding_when_rank_small(self):
        """When matrix rank < chi_max, output is zero-padded to chi_max."""
        from tenax.algorithms._padded_linalg import padded_svd_dense

        # Rank-2 matrix embedded in (4, 2, 2, 4) theta
        key = jax.random.PRNGKey(1)
        a = jax.random.normal(key, (8, 1))
        b = jax.random.normal(jax.random.PRNGKey(2), (1, 8))
        mat_rank1 = a @ b
        theta = mat_rank1.reshape(4, 2, 2, 4)
        chi_max = 6

        U, s, Vh = padded_svd_dense(theta, chi_max)
        assert U.shape == (8, chi_max)
        assert s.shape == (chi_max,)
        assert Vh.shape == (chi_max, 8)

        # Only 1 nonzero singular value; rest should be ~0
        s_np = np.array(s)
        assert s_np[0] > 1e-5
        np.testing.assert_allclose(s_np[1:], 0.0, atol=1e-10)

    def test_padded_svd_is_jittable(self):
        """padded_svd_dense works inside jax.jit."""
        from tenax.algorithms._padded_linalg import padded_svd_dense

        @jax.jit
        def f(theta):
            return padded_svd_dense(theta, chi_max=4)

        theta = jax.random.normal(jax.random.PRNGKey(1), (4, 2, 2, 4))
        U, s, Vh = f(theta)
        assert U.shape == (8, 4)
        assert s.shape == (4,)
        assert Vh.shape == (4, 8)

        # Singular values should be non-negative (or very close to 0)
        assert jnp.all(s >= -1e-14)

    def test_dense_svd_chi_max_exceeds_rank(self):
        """When chi_max > min(m, n), pad with zeros."""
        from tenax.algorithms._padded_linalg import padded_svd_dense

        # (2, 2, 2, 2) theta -> 4x4 matrix, max rank = 4
        theta = jax.random.normal(jax.random.PRNGKey(3), (2, 2, 2, 2))
        chi_max = 8

        U, s, Vh = padded_svd_dense(theta, chi_max)
        assert s.shape == (chi_max,)
        # Only 4 nonzero singular values
        s_np = np.array(s)
        assert np.sum(s_np > 1e-10) <= 4
        np.testing.assert_allclose(s_np[4:], 0.0, atol=1e-12)


# ------------------------------------------------------------------ #
# Block-sparse SVD tests                                               #
# ------------------------------------------------------------------ #


class TestPaddedSVDBlocks:
    def test_block_sparse_svd_matches_symmetric_svd(self):
        """Padded block SVD with vmap + top_k matches tenax.linalg.svd."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray
        from tenax.algorithms._padded_linalg import padded_svd_blocks

        sym = _make_symmetric_matrix(seed=42, chi=6)
        chi_max = 4

        # Reference: tenax SVD
        U_ref, s_ref, Vh_ref, s_full_ref = tenax_svd(
            sym,
            left_labels=[sym.labels()[0]],
            right_labels=[sym.labels()[1]],
            max_singular_values=chi_max,
        )

        # Padded path
        pba = PaddedBlockArray.from_symmetric(sym)
        top_values, per_block_keeps = padded_svd_blocks(pba, chi_max)

        # Compare singular values (sorted descending)
        s_ref_arr = np.sort(np.array(s_ref))[::-1]
        s_jit_arr = np.sort(np.array(top_values))[::-1]
        # Filter out zeros from padding
        s_jit_nonzero = s_jit_arr[s_jit_arr > 1e-14]
        s_ref_nonzero = s_ref_arr[s_ref_arr > 1e-14]
        n = min(len(s_ref_nonzero), len(s_jit_nonzero))
        np.testing.assert_allclose(s_jit_nonzero[:n], s_ref_nonzero[:n], atol=1e-8)

    def test_block_sparse_svd_is_jittable(self):
        """padded_svd_blocks works inside jax.jit."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray
        from tenax.algorithms._padded_linalg import padded_svd_blocks

        sym = _make_symmetric_matrix(seed=7, chi=6)
        pba = PaddedBlockArray.from_symmetric(sym)
        chi_max = 3

        @jax.jit
        def f(data):
            pba_inner = PaddedBlockArray(
                data=data,
                block_charges=pba.block_charges,
                block_shapes=pba.block_shapes,
                indices=pba.indices,
                symmetry=pba.symmetry,
            )
            return padded_svd_blocks(pba_inner, chi_max)

        top_values, per_block_keeps = f(pba.data)
        assert top_values.shape == (chi_max,)
        # All returned singular values should be non-negative
        assert jnp.all(top_values >= -1e-14)

    def test_per_block_keeps_sum_to_chi_max(self):
        """The per-block keep counts sum to at most chi_max."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray
        from tenax.algorithms._padded_linalg import padded_svd_blocks

        sym = _make_symmetric_matrix(seed=99, chi=8)
        pba = PaddedBlockArray.from_symmetric(sym)
        chi_max = 5

        top_values, per_block_keeps = padded_svd_blocks(pba, chi_max)
        total_kept = int(jnp.sum(per_block_keeps))
        assert total_kept <= chi_max
        # Should keep exactly chi_max if enough singular values exist
        total_available = sum(
            np.sum(np.linalg.svd(np.array(pba.data[i]), compute_uv=False) > 1e-14)
            for i in range(pba.data.shape[0])
        )
        if total_available >= chi_max:
            assert total_kept == chi_max

    def test_block_sparse_svd_preserves_sector_structure(self):
        """The returned per_block_keeps correctly partitions chi_max across sectors."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray
        from tenax.algorithms._padded_linalg import padded_svd_blocks

        sym = _make_symmetric_matrix(seed=123, chi=8)
        pba = PaddedBlockArray.from_symmetric(sym)
        chi_max = 6

        top_values, per_block_keeps = padded_svd_blocks(pba, chi_max)

        # per_block_keeps should have one entry per block
        assert per_block_keeps.shape == (pba.data.shape[0],)
        # All keeps should be non-negative
        assert jnp.all(per_block_keeps >= 0)
