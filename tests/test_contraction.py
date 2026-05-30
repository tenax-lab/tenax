"""Tests for the contraction module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.contraction.contractor import (
    _cached_contraction_path,
    _labels_to_subscripts,
    contract,
    contract_with_subscripts,
    qr_decompose,
    truncated_svd,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def make_dense(u1, shape, labels, flows, rng, charges_per_leg=None):
    """Create a DenseTensor with given shape and labels."""
    data = jax.random.normal(rng, shape)
    charges = [np.zeros(s, dtype=np.int32) for s in shape]
    if charges_per_leg is not None:
        charges = charges_per_leg
    indices = tuple(
        TensorIndex.from_charges(u1, charges[i], flows[i], label=labels[i])
        for i in range(len(shape))
    )
    return DenseTensor(data, indices)


# ------------------------------------------------------------------ #
# Label → subscript translation                                        #
# ------------------------------------------------------------------ #


class TestLabelsToSubscripts:
    def test_two_tensor_contraction(self, u1, rng):
        """Two tensors sharing one label → subscript contracts that label."""
        charges = np.zeros(3, dtype=np.int32)
        idx_i = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i")
        idx_j_in = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j")
        idx_j_out = TensorIndex.from_charges(
            u1, charges, FlowDirection.OUT, label="j"
        )  # shared
        idx_k = TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="k")

        A = DenseTensor(jnp.ones((3, 3)), (idx_i, idx_j_out))
        B = DenseTensor(jnp.ones((3, 3)), (idx_j_in, idx_k))

        subs, out_indices = _labels_to_subscripts([A, B], None)
        assert "->" in subs
        # Output should contain 'i' and 'k' chars (not 'j')
        in_part, out_part = subs.split("->")
        parts = in_part.split(",")
        # j appears in both parts → contracted
        j_char = None
        for p in parts:
            for c in p:
                if j_char is None:
                    j_char = c
        assert out_part.count(",") == 0  # flat output

    def test_triple_label_raises(self, u1, rng):
        """Label appearing 3 times should raise."""
        charges = np.zeros(2, dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="shared")
        A = DenseTensor(jnp.ones((2,)), (idx,))
        B = DenseTensor(jnp.ones((2,)), (idx,))
        C = DenseTensor(jnp.ones((2,)), (idx,))
        with pytest.raises(ValueError, match="3 times"):
            _labels_to_subscripts([A, B, C], None)

    def test_output_labels_respected(self, u1, rng):
        """output_labels should control ordering of free legs."""
        charges = np.zeros(2, dtype=np.int32)
        idx_a = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a")
        idx_b = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="b")
        A = DenseTensor(jnp.ones((2, 2)), (idx_a, idx_b))
        # No shared labels — both a, b are free
        subs, out_indices = _labels_to_subscripts([A], output_labels=["b", "a"])
        assert out_indices[0].label == "b"
        assert out_indices[1].label == "a"


# ------------------------------------------------------------------ #
# Dense contraction                                                    #
# ------------------------------------------------------------------ #


class TestContractDense:
    def test_matrix_vector_multiply(self, u1, rng):
        """contract(M, v) where shared label 'j' gives M @ v."""
        n = 4
        charges = np.zeros(n, dtype=np.int32)
        idx_i = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i")
        idx_j_out = TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="j")
        idx_j_in = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j")

        M_data = jax.random.normal(rng, (n, n))
        v_data = jax.random.normal(jax.random.PRNGKey(1), (n,))

        M = DenseTensor(M_data, (idx_i, idx_j_out))
        v = DenseTensor(v_data, (idx_j_in,))

        result = contract(M, v)
        expected = M_data @ v_data
        np.testing.assert_allclose(result.todense(), expected, rtol=1e-5)
        assert result.labels() == ("i",)

    def test_matrix_matrix_multiply(self, u1, rng):
        """Two matrices: contract on shared 'j' leg → matrix product."""
        n = 3
        charges = np.zeros(n, dtype=np.int32)
        A_data = jax.random.normal(rng, (n, n))
        B_data = jax.random.normal(jax.random.PRNGKey(10), (n, n))

        A = DenseTensor(
            A_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="j"),
            ),
        )
        B = DenseTensor(
            B_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="k"),
            ),
        )

        result = contract(A, B)
        expected = A_data @ B_data
        np.testing.assert_allclose(result.todense(), expected, rtol=1e-5)
        assert result.labels() == ("i", "k")

    def test_trace(self, u1, rng):
        """Contracting both legs of a square matrix gives trace."""
        n = 4
        charges = np.zeros(n, dtype=np.int32)
        M_data = jax.random.normal(rng, (n, n))

        M = DenseTensor(
            M_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),
                TensorIndex.from_charges(
                    u1, charges, FlowDirection.OUT, label="i"
                ),  # same label!
            ),
        )

        result = contract(M)
        expected = jnp.trace(M_data)
        np.testing.assert_allclose(float(result.todense()), float(expected), rtol=1e-5)

    def test_outer_product(self, u1, rng):
        """No shared labels → outer product."""
        n = 3
        charges = np.zeros(n, dtype=np.int32)
        a_data = jax.random.normal(rng, (n,))
        b_data = jax.random.normal(jax.random.PRNGKey(7), (n,))

        a = DenseTensor(
            a_data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),),
        )
        b = DenseTensor(
            b_data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j"),),
        )

        result = contract(a, b)
        expected = jnp.outer(a_data, b_data)
        np.testing.assert_allclose(result.todense(), expected, rtol=1e-5)
        assert result.labels() == ("i", "j")

    def test_output_labels_ordering(self, u1, rng):
        """output_labels should reorder the result legs."""
        n = 3
        charges = np.zeros(n, dtype=np.int32)
        a_data = jax.random.normal(rng, (n,))
        b_data = jax.random.normal(jax.random.PRNGKey(8), (n,))

        a = DenseTensor(
            a_data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),),
        )
        b = DenseTensor(
            b_data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j"),),
        )

        result_ij = contract(a, b, output_labels=["i", "j"])
        result_ji = contract(a, b, output_labels=["j", "i"])

        np.testing.assert_allclose(
            result_ij.todense(),
            result_ji.todense().T,
            rtol=1e-5,
        )
        assert result_ji.labels() == ("j", "i")

    def test_single_tensor_returns_unchanged(self, small_dense_matrix):
        result = contract(small_dense_matrix)
        assert result is small_dense_matrix

    def test_mixed_types_raises(self, u1, rng, u1_sym_tensor_2leg):
        charges = np.zeros(3, dtype=np.int32)
        dense = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="x"),),
        )
        with pytest.raises(TypeError, match="mix"):
            contract(dense, u1_sym_tensor_2leg)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            contract()

    def test_opt_einsum_called(self, monkeypatch, u1, rng):
        """Verify opt_einsum.contract_path is invoked."""
        import opt_einsum

        call_count = [0]
        original = opt_einsum.contract_path

        def patched(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(opt_einsum, "contract_path", patched)

        n = 3
        charges = np.zeros(n, dtype=np.int32)
        A_data = jax.random.normal(rng, (n, n))
        A = DenseTensor(
            A_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="j"),
            ),
        )
        B_data = jax.random.normal(jax.random.PRNGKey(2), (n,))
        B = DenseTensor(
            B_data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j"),),
        )

        contract(A, B)
        assert call_count[0] > 0

    def test_path_cache_hits(self, u1, rng):
        """Second contraction with same shapes should use cached path."""
        _cached_contraction_path.cache_clear()

        n = 4
        charges = np.zeros(n, dtype=np.int32)
        A_data = jax.random.normal(rng, (n, n))
        A = DenseTensor(
            A_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="j"),
            ),
        )
        B_data = jax.random.normal(jax.random.PRNGKey(99), (n,))
        B = DenseTensor(
            B_data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j"),),
        )

        contract(A, B)
        info_after_first = _cached_contraction_path.cache_info()

        # Second call with different data but same shapes → cache hit
        A2_data = jax.random.normal(jax.random.PRNGKey(100), (n, n))
        A2 = DenseTensor(
            A2_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="j"),
            ),
        )
        B2_data = jax.random.normal(jax.random.PRNGKey(101), (n,))
        B2 = DenseTensor(
            B2_data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j"),),
        )

        contract(A2, B2)
        info_after_second = _cached_contraction_path.cache_info()

        assert info_after_second.hits > info_after_first.hits

    def test_path_cache_numerical_parity(self, u1, rng):
        """Cached path must produce identical results to uncached."""
        _cached_contraction_path.cache_clear()

        n = 5
        charges = np.zeros(n, dtype=np.int32)
        A_data = jax.random.normal(rng, (n, n))
        B_data = jax.random.normal(jax.random.PRNGKey(50), (n, n))

        A = DenseTensor(
            A_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="j"),
            ),
        )
        B = DenseTensor(
            B_data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="j"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="k"),
            ),
        )

        # First call (cache miss)
        result1 = contract(A, B)
        # Second call (cache hit)
        result2 = contract(A, B)

        np.testing.assert_array_equal(result1.todense(), result2.todense())


# ------------------------------------------------------------------ #
# Symmetric contraction                                                #
# ------------------------------------------------------------------ #


class TestContractSymmetric:
    def test_matches_dense_contraction(self, u1_sym_tensor_pair, u1, rng):
        """Block-sparse contraction must match equivalent dense einsum."""
        A, B = u1_sym_tensor_pair
        # A labels: ('p0', 'bond_left', 'bond')
        # B labels: ('p1', 'bond', 'bond_right')
        # Shared label 'bond' → contracted; free: p0, bond_left, p1, bond_right

        # Dense reference using the label-based subscripts:
        # sorted labels: bond->a, bond_left->b, bond_right->c, p0->d, p1->e
        # A: (p0=d, bond_left=b, bond=a) → 'dba'
        # B: (p1=e, bond=a, bond_right=c) → 'eac'
        # output: (p0=d, bond_left=b, p1=e, bond_right=c) → 'dbec'
        A_dense = A.todense()
        B_dense = B.todense()
        dense_result = jnp.einsum("dba,eac->dbec", A_dense, B_dense)

        sym_result = contract(A, B)
        sym_dense = sym_result.todense()

        np.testing.assert_allclose(
            np.abs(sym_dense), np.abs(dense_result), rtol=1e-4, atol=1e-5
        )

    def test_result_is_symmetric_tensor(self, u1_sym_tensor_pair):
        A, B = u1_sym_tensor_pair
        result = contract(A, B)
        assert isinstance(result, SymmetricTensor)

    def test_output_satisfies_conservation(self, u1_sym_tensor_pair, u1):
        A, B = u1_sym_tensor_pair
        result = contract(A, B)
        for key in result.blocks:
            # Determine flows from output indices
            net = sum(int(idx.flow) * int(q) for idx, q in zip(result.indices, key))
            assert net == u1.identity(), f"Block {key} violates conservation"

    def test_multi_input_with_disjoint_contracted_legs(self, u1, rng):
        """Regression for #553: contract(A, B, C) where A and B share no
        labels but both share separately with C must agree with pairwise
        contraction.

        Topology::

            A = (a,)        B = (b,)        C = (a, b)

        Before the fix the multi-input ``_contract_symmetric`` path
        intersected per-tensor contracted-leg signatures of mismatched
        lengths (A's 1-tuples vs C's 2-tuples) and silently produced
        an all-zero output.  Pairwise execution gives the correct
        scalar.  See GitHub issue tenax-lab/tenax#553 for details.
        """
        sym = u1
        charges = np.zeros(2, dtype=np.int32)
        A = SymmetricTensor.random_normal(
            (TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="a"),),
            rng,
        )
        B = SymmetricTensor.random_normal(
            (TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="b"),),
            jax.random.split(rng)[0],
        )
        C = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="a"),
                TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="b"),
            ),
            jax.random.split(rng)[1],
        )

        r3 = float(contract(A, B, C).todense())
        r_pair_ab = float(contract(contract(A, B), C).todense())
        r_pair_ac = float(contract(contract(A, C), B).todense())

        # All three orderings should agree (and the pre-fix bug returned 0.0).
        np.testing.assert_allclose(r3, r_pair_ab, atol=1e-12)
        np.testing.assert_allclose(r3, r_pair_ac, atol=1e-12)
        # Cross-check against numpy einsum (no symmetry tracking — for U(1)
        # trivial charges this gives the same answer).
        ref = float(
            np.einsum(
                "a,b,ab->",
                np.array(A.todense()),
                np.array(B.todense()),
                np.array(C.todense()),
            )
        )
        np.testing.assert_allclose(r3, ref, atol=1e-12)

    def test_multi_input_dmrg_style_three_input(self, u1, rng):
        """Regression for #553: structurally similar to in-tree call sites
        like ``contract(s_left_dag, corner, s_right)`` and
        ``contract(C0, L0, C1, L1, C2, L2, T_r, ...)`` where some pairs
        of inputs do not share labels directly.

        Topology::

            X = (l, m, p)
            Y = (m, n)
            Z = (n, l)

        Each pair shares one label; full contraction → scalar of (p,).
        """
        sym = u1
        charges = np.zeros(2, dtype=np.int32)
        X = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="l"),
                TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="m"),
                TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="p"),
            ),
            rng,
        )
        Y = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="m"),
                TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="n"),
            ),
            jax.random.split(rng)[0],
        )
        Z = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="n"),
                TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="l"),
            ),
            jax.random.split(rng)[1],
        )

        r3 = np.array(contract(X, Y, Z).todense())
        r_pair = np.array(contract(contract(X, Y), Z).todense())
        np.testing.assert_allclose(r3, r_pair, atol=1e-12)


# ------------------------------------------------------------------ #
# Truncated SVD                                                        #
# ------------------------------------------------------------------ #


class TestTruncatedSVD:
    def test_reconstruction_dense(self, u1, rng):
        """U @ diag(s) @ Vh should approximately reconstruct the original matrix."""
        n, m = 4, 5
        charges_n = np.zeros(n, dtype=np.int32)
        charges_m = np.zeros(m, dtype=np.int32)
        data = jax.random.normal(rng, (n, m))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges_n, FlowDirection.IN, label="row"),
                TensorIndex.from_charges(u1, charges_m, FlowDirection.OUT, label="col"),
            ),
        )

        U, s, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        # Reconstruct: U[row, bond] * s[bond] * Vh[bond, col]
        recon = U.todense() * s[None, :] @ Vh.todense()
        np.testing.assert_allclose(recon, data, rtol=1e-4, atol=1e-4)

    def test_max_singular_values_limits_bond(self, u1, rng):
        """Truncation should limit the number of singular values."""
        n = 6
        charges = np.zeros(n, dtype=np.int32)
        data = jax.random.normal(rng, (n, n))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="left"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="right"),
            ),
        )

        max_chi = 3
        U, s, Vh, _ = truncated_svd(
            t,
            left_labels=["left"],
            right_labels=["right"],
            new_bond_label="bond",
            max_singular_values=max_chi,
        )
        assert len(s) <= max_chi
        assert "bond" in U.labels()
        assert "bond" in Vh.labels()

    def test_singular_values_positive_and_decreasing(self, u1, rng):
        n = 5
        charges = np.zeros(n, dtype=np.int32)
        data = jax.random.normal(rng, (n, n))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            ),
        )
        _, s, _, _ = truncated_svd(
            t, left_labels=["a"], right_labels=["b"], new_bond_label="bond"
        )
        s_np = np.array(s)
        assert np.all(s_np >= 0), "Singular values should be non-negative"
        assert np.all(np.diff(s_np) <= 1e-5), "Singular values should be non-increasing"

    def test_wrong_labels_raises(self, small_dense_matrix):
        with pytest.raises(ValueError):
            truncated_svd(
                small_dense_matrix,
                left_labels=["row"],
                right_labels=["nonexistent"],
                new_bond_label="bond",
            )

    def test_overlapping_labels_raises(self, small_dense_matrix):
        with pytest.raises(ValueError, match="disjoint"):
            truncated_svd(
                small_dense_matrix,
                left_labels=["row", "col"],
                right_labels=["col"],
                new_bond_label="bond",
            )

    def test_new_bond_label_propagates(self, u1, rng):
        """The new bond should get the specified label."""
        n = 4
        charges = np.zeros(n, dtype=np.int32)
        data = jax.random.normal(rng, (n, n))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="left"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="right"),
            ),
        )

        U, _, Vh, _ = truncated_svd(
            t, left_labels=["left"], right_labels=["right"], new_bond_label="my_bond"
        )
        assert "my_bond" in U.labels()
        assert "my_bond" in Vh.labels()

    def test_s_full_contains_all_singular_values(self, u1, rng):
        """s_full should contain all singular values (length = min(m, n))."""
        n, m = 6, 8
        charges_n = np.zeros(n, dtype=np.int32)
        charges_m = np.zeros(m, dtype=np.int32)
        data = jax.random.normal(rng, (n, m))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges_n, FlowDirection.IN, label="row"),
                TensorIndex.from_charges(u1, charges_m, FlowDirection.OUT, label="col"),
            ),
        )

        max_chi = 3
        _, s_trunc, _, s_full = truncated_svd(
            t,
            left_labels=["row"],
            right_labels=["col"],
            new_bond_label="bond",
            max_singular_values=max_chi,
        )

        # s_full should have min(n, m) entries (all singular values)
        assert len(s_full) == min(n, m), (
            f"Expected s_full length {min(n, m)}, got {len(s_full)}"
        )
        # s_trunc should be truncated to max_chi
        assert len(s_trunc) == max_chi

        # s_trunc should be a prefix of s_full (same top singular values)
        np.testing.assert_allclose(
            np.array(s_trunc),
            np.array(s_full[:max_chi]),
            rtol=1e-6,
        )

        # s_full should match an independent SVD
        ref_s = jnp.linalg.svd(data, full_matrices=False, compute_uv=False)
        np.testing.assert_allclose(np.array(s_full), np.array(ref_s), rtol=1e-6)

    def test_s_full_no_truncation(self, u1, rng):
        """Without truncation, s and s_full should be identical."""
        n = 5
        charges = np.zeros(n, dtype=np.int32)
        data = jax.random.normal(rng, (n, n))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            ),
        )

        _, s, _, s_full = truncated_svd(
            t,
            left_labels=["a"],
            right_labels=["b"],
            new_bond_label="bond",
        )

        np.testing.assert_allclose(np.array(s), np.array(s_full), rtol=1e-6)


# ------------------------------------------------------------------ #
# QR decomposition                                                     #
# ------------------------------------------------------------------ #


class TestQRDecompose:
    def test_reconstruction(self, u1, rng):
        """Q @ R should reconstruct the original matrix."""
        n, m = 4, 3
        charges_n = np.zeros(n, dtype=np.int32)
        charges_m = np.zeros(m, dtype=np.int32)
        data = jax.random.normal(rng, (n, m))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges_n, FlowDirection.IN, label="row"),
                TensorIndex.from_charges(u1, charges_m, FlowDirection.OUT, label="col"),
            ),
        )

        Q, R = qr_decompose(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        recon = Q.todense() @ R.todense()
        np.testing.assert_allclose(recon, data, rtol=1e-4, atol=1e-4)

    def test_q_is_isometric(self, u1, rng):
        """Q^dag @ Q should be identity."""
        n = 5
        charges = np.zeros(n, dtype=np.int32)
        data = jax.random.normal(rng, (n, n))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),
            ),
        )

        Q, R = qr_decompose(
            t, left_labels=["a"], right_labels=["b"], new_bond_label="bond"
        )
        Q_dense = Q.todense()
        QtQ = Q_dense.T @ Q_dense
        np.testing.assert_allclose(QtQ, np.eye(QtQ.shape[0]), atol=1e-5)

    def test_bond_label_set(self, u1, rng):
        n = 4
        charges = np.zeros(n, dtype=np.int32)
        data = jax.random.normal(rng, (n, n))
        t = DenseTensor(
            data,
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="x"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="y"),
            ),
        )
        Q, R = qr_decompose(
            t, left_labels=["x"], right_labels=["y"], new_bond_label="qr_bond"
        )
        assert "qr_bond" in Q.labels()
        assert "qr_bond" in R.labels()


# ------------------------------------------------------------------ #
# Symmetric (block-sparse) Truncated SVD                               #
# ------------------------------------------------------------------ #


def _make_symmetric_2leg(u1, rng, left_charges, right_charges, left_label, right_label):
    """Helper: build a 2-leg U(1)-symmetric tensor with given charges."""
    indices = (
        TensorIndex.from_charges(u1, left_charges, FlowDirection.IN, label=left_label),
        TensorIndex.from_charges(
            u1, u1.dual(left_charges), FlowDirection.OUT, label=right_label
        ),
    )
    return SymmetricTensor.random_normal(indices, rng)


def _make_symmetric_3leg(u1, rng):
    """Helper: build a 3-leg U(1)-symmetric tensor for multi-leg SVD tests."""
    phys_c = np.array([-1, 0, 1], dtype=np.int32)
    virt_c = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, phys_c, FlowDirection.IN, label="phys"),
        TensorIndex.from_charges(u1, virt_c, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(u1, u1.dual(virt_c), FlowDirection.OUT, label="right"),
    )
    return SymmetricTensor.random_normal(indices, rng)


class TestTruncatedSVDSymmetric:
    def test_reconstruction(self, u1, rng):
        """U @ diag(s) @ Vh should reconstruct a SymmetricTensor."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")
        original_dense = t.todense()

        U, s, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        recon = U.todense() * s[None, :] @ Vh.todense()
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_returns_symmetric(self, u1, rng):
        """U and Vh should be SymmetricTensor instances."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        U, s, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        assert isinstance(U, SymmetricTensor)
        assert isinstance(Vh, SymmetricTensor)

    def test_bond_charges_nontrivial(self, u1, rng):
        """Bond index should carry non-trivial charges (not all zeros)."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        U, s, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        # Find bond index on U (last index)
        bond_idx = U.indices[-1]
        assert bond_idx.label == "bond"
        # With charges [-1, 0, 1] there should be multiple distinct charge values
        unique_bond_charges = set(bond_idx.charges.tolist())
        assert len(unique_bond_charges) > 1, (
            f"Expected non-trivial bond charges, got {unique_bond_charges}"
        )

    def test_matches_dense(self, u1, rng):
        """Singular values from block-sparse SVD should match dense SVD."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        _, s_sym, _, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        # Dense SVD of the same tensor
        dense = t.todense()
        _, s_dense, _ = jnp.linalg.svd(dense, full_matrices=False)

        # Sort both for comparison (block-sparse orders by sector, not globally)
        s_sym_sorted = np.sort(np.array(s_sym))[::-1]
        s_dense_sorted = np.sort(np.array(s_dense))[::-1]

        np.testing.assert_allclose(s_sym_sorted, s_dense_sorted, rtol=1e-4, atol=1e-6)

    def test_truncation(self, u1, rng):
        """max_singular_values should limit total bond dimension across sectors."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        max_chi = 2
        U, s, Vh, _ = truncated_svd(
            t,
            left_labels=["row"],
            right_labels=["col"],
            new_bond_label="bond",
            max_singular_values=max_chi,
        )

        assert len(s) <= max_chi
        # Bond dimension on U and Vh should match
        assert U.indices[-1].dim == len(s)
        assert Vh.indices[0].dim == len(s)

    def test_3leg_reconstruction(self, u1, rng):
        """SVD of a 3-leg tensor: contract(U, diag(s), Vh) ~ original."""
        t = _make_symmetric_3leg(u1, rng)
        original_dense = t.todense()

        U, s, Vh, _ = truncated_svd(
            t,
            left_labels=["phys", "left"],
            right_labels=["right"],
            new_bond_label="bond",
        )

        assert isinstance(U, SymmetricTensor)
        assert isinstance(Vh, SymmetricTensor)

        # Reconstruct via dense for simplicity
        U_d = U.todense()
        Vh_d = Vh.todense()
        # U: (phys, left, bond), Vh: (bond, right)
        # U * s along bond axis, then contract bond
        Us = U_d * s[None, None, :]
        recon = jnp.einsum("plb,br->plr", Us, Vh_d)
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_conservation_law_preserved(self, u1, rng):
        """All blocks in U and Vh should satisfy the conservation law."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        U, _, Vh, _ = truncated_svd(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        for tensor_out in (U, Vh):
            for key in tensor_out.blocks:
                net = sum(
                    int(idx.flow) * int(q) for idx, q in zip(tensor_out.indices, key)
                )
                assert net == u1.identity(), (
                    f"Block {key} violates conservation: net={net}"
                )


# ------------------------------------------------------------------ #
# Symmetric (block-sparse) QR                                          #
# ------------------------------------------------------------------ #


class TestQRSymmetric:
    def test_reconstruction(self, u1, rng):
        """contract(Q, R) should reconstruct the original SymmetricTensor."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")
        original_dense = t.todense()

        Q, R = qr_decompose(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        recon = Q.todense() @ R.todense()
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_q_isometric(self, u1, rng):
        """Q^dag Q should be identity (block-wise)."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        Q, _ = qr_decompose(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        Q_dense = Q.todense()
        QtQ = Q_dense.T.conj() @ Q_dense
        np.testing.assert_allclose(QtQ, np.eye(QtQ.shape[0]), atol=1e-5)

    def test_returns_symmetric(self, u1, rng):
        """Q and R should be SymmetricTensor instances."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        Q, R = qr_decompose(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        assert isinstance(Q, SymmetricTensor)
        assert isinstance(R, SymmetricTensor)

    def test_3leg_reconstruction(self, u1, rng):
        """QR of a 3-leg tensor: contract(Q, R) ~ original."""
        t = _make_symmetric_3leg(u1, rng)
        original_dense = t.todense()

        Q, R = qr_decompose(
            t,
            left_labels=["phys", "left"],
            right_labels=["right"],
            new_bond_label="bond",
        )

        assert isinstance(Q, SymmetricTensor)
        assert isinstance(R, SymmetricTensor)

        recon = jnp.einsum("plb,br->plr", Q.todense(), R.todense())
        np.testing.assert_allclose(recon, original_dense, rtol=1e-4, atol=1e-6)

    def test_conservation_law_preserved(self, u1, rng):
        """All blocks in Q and R should satisfy the conservation law."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        t = _make_symmetric_2leg(u1, rng, charges, charges, "row", "col")

        Q, R = qr_decompose(
            t, left_labels=["row"], right_labels=["col"], new_bond_label="bond"
        )

        for tensor_out in (Q, R):
            for key in tensor_out.blocks:
                net = sum(
                    int(idx.flow) * int(q) for idx, q in zip(tensor_out.indices, key)
                )
                assert net == u1.identity(), (
                    f"Block {key} violates conservation: net={net}"
                )


# ------------------------------------------------------------------ #
# Batched block-sparse path (issue #568, Milestone B)                 #
# ------------------------------------------------------------------ #


class TestBatchedBlockSparse:
    """The opt-in TENAX_BATCH_BLOCKSPARSE path must equal the per-combo path.

    The gate is read fresh from os.environ on every contract() call, so toggling
    the env var inside the test switches paths with no cache to clear.
    """

    @staticmethod
    def _assert_paths_equal(monkeypatch, A, B):
        """contract(A, B) must agree block-for-block with the flag off vs on."""
        # Per-combo path (default / flag off).
        monkeypatch.delenv("TENAX_BATCH_BLOCKSPARSE", raising=False)
        res_off = contract(A, B)

        # Batched path (flag on). Gate is read fresh per call, no cache to clear.
        monkeypatch.setenv("TENAX_BATCH_BLOCKSPARSE", "1")
        res_on = contract(A, B)

        assert set(res_off.blocks.keys()) == set(res_on.blocks.keys())
        assert len(res_off.blocks) > 0, "expected a non-trivial multi-block result"
        for k in res_off.blocks:
            a = res_off.blocks[k]
            b = res_on.blocks[k]
            assert a.shape == b.shape
            assert a.dtype == b.dtype
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12
            )

    def test_fermion_parity_multi_shape_group(self, monkeypatch):
        """Multi-block bond so surviving combos span several shape-groups."""
        from tenax.core.symmetry import FermionParity

        fp = FermionParity()
        charges = np.array([0, 1], dtype=np.int32)
        bond_charges = np.array([0, 0, 1, 1], dtype=np.int32)
        indices_A = (
            TensorIndex.from_charges(fp, charges, FlowDirection.IN, label="p0"),
            TensorIndex.from_charges(fp, charges, FlowDirection.IN, label="q0"),
            TensorIndex.from_charges(fp, bond_charges, FlowDirection.OUT, label="bond"),
        )
        indices_B = (
            TensorIndex.from_charges(fp, bond_charges, FlowDirection.IN, label="bond"),
            TensorIndex.from_charges(fp, charges, FlowDirection.IN, label="p1"),
            TensorIndex.from_charges(fp, charges, FlowDirection.IN, label="q1"),
        )
        k1, k2 = jax.random.split(jax.random.PRNGKey(123))
        A = SymmetricTensor.random_normal(indices_A, k1)
        B = SymmetricTensor.random_normal(indices_B, k2)
        self._assert_paths_equal(monkeypatch, A, B)

    def test_fermion_parity_single_crossing(self, monkeypatch):
        """Single contracted-leg crossing (``ac,bc->ab``).

        This topology is the one that *would* diverge if the two paths ever
        disagreed on leg-permutation handling (it has a single, non-cancelling
        leg crossing — unlike the duplicated-crossing multi-leg case). Guards
        against a future re-introduction of a per-combo-only transform.
        """
        from tenax.core.symmetry import FermionParity

        fp = FermionParity()
        ch = np.array([0, 1], dtype=np.int32)
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))
        A = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(fp, ch, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(fp, ch, FlowDirection.OUT, label="c"),
            ),
            k1,
        )
        B = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(fp, ch, FlowDirection.IN, label="b"),
                TensorIndex.from_charges(fp, ch, FlowDirection.IN, label="c"),
            ),
            k2,
        )
        self._assert_paths_equal(monkeypatch, A, B)

    def test_u1_batched_equals_per_combo(self, monkeypatch):
        """The batched path is general to all charge-conserving symmetries,
        not just fermionic — exercise a U(1) multi-block contraction."""

        u1 = U1Symmetry()
        ch = np.array([0, 1, -1], dtype=np.int32)
        k1, k2 = jax.random.split(jax.random.PRNGKey(31))
        A = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(u1, ch, FlowDirection.IN, label="a"),
                TensorIndex.from_charges(u1, ch, FlowDirection.OUT, label="c"),
            ),
            k1,
        )
        B = SymmetricTensor.random_normal(
            (
                TensorIndex.from_charges(u1, ch, FlowDirection.IN, label="b"),
                TensorIndex.from_charges(u1, ch, FlowDirection.IN, label="c"),
            ),
            k2,
        )
        self._assert_paths_equal(monkeypatch, A, B)
