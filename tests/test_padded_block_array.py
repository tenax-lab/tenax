"""Tests for PaddedBlockArray data structure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._padded_block_array import (
    PaddedBlockArray,
    pad_dense,
    pba_add,
    pba_axpy,
    pba_inner,
    pba_norm,
    pba_scale,
    pba_zeros_like,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor


@pytest.fixture
def u1():
    return U1Symmetry()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def sym_matrix(u1, rng):
    """2-leg U(1) symmetric tensor with charges [-1, 0, 1] on each leg."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex(u1, charges, FlowDirection.IN, label="in"),
        TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
    )
    return SymmetricTensor.random_normal(indices, rng)


@pytest.fixture
def sym_3leg(u1, rng):
    """3-leg U(1) symmetric tensor: (phys, left, right) like an MPS tensor."""
    phys_c = np.array([-1, 1], dtype=np.int32)
    virt_c = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex(u1, phys_c, FlowDirection.IN, label="phys"),
        TensorIndex(u1, virt_c, FlowDirection.IN, label="left"),
        TensorIndex(u1, u1.dual(virt_c), FlowDirection.OUT, label="right"),
    )
    return SymmetricTensor.random_normal(indices, rng)


class TestFromSymmetricCreatesPaddedArray:
    """Verify from_symmetric produces a 3D padded array with correct shape."""

    def test_2leg_shape(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        # Data should be 3D: (num_blocks, M_max, N_max)
        assert pba.data.ndim == 3
        assert pba.data.shape[0] == sym_matrix.n_blocks

    def test_3leg_shape(self, sym_3leg):
        pba = PaddedBlockArray.from_symmetric(sym_3leg)
        # For a 3-leg tensor, blocks are reshaped to 2D (fused_left, fused_right)
        # Data should still be 3D: (num_blocks, M_max, N_max)
        assert pba.data.ndim == 3
        assert pba.data.shape[0] == sym_3leg.n_blocks

    def test_padded_dimensions_are_max(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        # M_max and N_max should be >= every block's row/col dimension
        for shape in pba.block_shapes:
            assert pba.data.shape[1] >= shape[0]
            assert pba.data.shape[2] >= shape[1]


class TestRoundTripSymmetric:
    """from_symmetric -> to_symmetric should be identity."""

    def test_2leg_round_trip(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        recovered = pba.to_symmetric()
        # Check same indices
        assert recovered.indices == sym_matrix.indices
        # Check same block keys
        assert recovered._block_keys == sym_matrix._block_keys
        # Check same block data (dense comparison)
        np.testing.assert_allclose(
            recovered.todense(), sym_matrix.todense(), atol=1e-12
        )

    def test_3leg_round_trip(self, sym_3leg):
        pba = PaddedBlockArray.from_symmetric(sym_3leg)
        recovered = pba.to_symmetric()
        assert recovered.indices == sym_3leg.indices
        assert recovered._block_keys == sym_3leg._block_keys
        np.testing.assert_allclose(recovered.todense(), sym_3leg.todense(), atol=1e-12)


class TestPytreeRegistration:
    """PaddedBlockArray should work inside jax.jit."""

    def test_jit_identity(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)

        @jax.jit
        def identity(x):
            return x

        result = identity(pba)
        np.testing.assert_allclose(result.data, pba.data, atol=1e-12)

    def test_jit_scale(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)

        @jax.jit
        def scale(x):
            return PaddedBlockArray(
                data=x.data * 2.0,
                block_charges=x.block_charges,
                block_shapes=x.block_shapes,
                indices=x.indices,
                symmetry=x.symmetry,
            )

        result = scale(pba)
        np.testing.assert_allclose(result.data, pba.data * 2.0, atol=1e-12)

    def test_tree_leaves(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        leaves = jax.tree_util.tree_leaves(pba)
        # Should have exactly 1 leaf: data (mask is derived from static aux data)
        assert len(leaves) == 1


class TestMaskZerosPadding:
    """Padding region should have mask=False and data=0."""

    def test_mask_shape_matches_data(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        assert pba.mask.shape == pba.data.shape

    def test_padding_is_zero(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        # Where mask is False, data must be 0
        padding_data = pba.data[~pba.mask]
        np.testing.assert_allclose(padding_data, 0.0, atol=1e-15)

    def test_mask_true_in_block_region(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        # For each block, the mask should be True in the block's actual region
        for i, shape in enumerate(pba.block_shapes):
            block_mask = pba.mask[i, : shape[0], : shape[1]]
            assert jnp.all(block_mask), (
                f"Block {i} with shape {shape} has False values in its data region"
            )

    def test_mask_false_outside_block(self, sym_matrix):
        pba = PaddedBlockArray.from_symmetric(sym_matrix)
        M_max = pba.data.shape[1]
        N_max = pba.data.shape[2]
        for i, shape in enumerate(pba.block_shapes):
            # Check padding rows
            if shape[0] < M_max:
                assert not jnp.any(pba.mask[i, shape[0] :, :])
            # Check padding cols
            if shape[1] < N_max:
                assert not jnp.any(pba.mask[i, :, shape[1] :])


class TestPadDense:
    """pad_dense should pad a dense (chi_l, d, chi_r) tensor to (chi_max, d, chi_max)."""

    def test_basic_padding(self, rng):
        data = jax.random.normal(rng, (3, 2, 4))
        padded = pad_dense(data, chi_max=8)
        assert padded.shape == (8, 2, 8)

    def test_original_data_preserved(self, rng):
        data = jax.random.normal(rng, (3, 2, 4))
        padded = pad_dense(data, chi_max=8)
        np.testing.assert_allclose(padded[:3, :, :4], data, atol=1e-12)

    def test_padding_is_zero(self, rng):
        data = jax.random.normal(rng, (3, 2, 4))
        padded = pad_dense(data, chi_max=8)
        # Padding region should be zero
        np.testing.assert_allclose(padded[3:, :, :], 0.0, atol=1e-15)
        np.testing.assert_allclose(padded[:, :, 4:], 0.0, atol=1e-15)

    def test_no_op_when_already_max(self, rng):
        data = jax.random.normal(rng, (5, 2, 5))
        padded = pad_dense(data, chi_max=5)
        assert padded.shape == (5, 2, 5)
        np.testing.assert_allclose(padded, data, atol=1e-12)


class TestEmptyTensor:
    """PaddedBlockArray from an empty SymmetricTensor (0 blocks)."""

    def test_empty_shape(self, u1):
        # Use a target charge that no block can satisfy to get 0 blocks
        charges = np.array([0], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
        )
        empty = SymmetricTensor.zeros(indices, target=999)
        assert empty.n_blocks == 0

        pba = PaddedBlockArray.from_symmetric(empty)
        assert pba.data.shape == (0, 0, 0)
        assert pba.block_charges == ()
        assert pba.block_shapes == ()

    def test_empty_round_trip(self, u1):
        charges = np.array([0], dtype=np.int32)
        indices = (
            TensorIndex(u1, charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
        )
        empty = SymmetricTensor.zeros(indices, target=999)
        pba = PaddedBlockArray.from_symmetric(empty)
        recovered = pba.to_symmetric()
        assert recovered.n_blocks == 0
        assert recovered.indices == empty.indices


class TestVariedBlockSizes:
    """PaddedBlockArray with blocks of genuinely different 2D sizes."""

    def test_varied_shapes_padding(self, u1, rng):
        # IN: charge 0 has multiplicity 3, charge 1 has multiplicity 2
        in_charges = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        # OUT: charge 0 has multiplicity 2, charge 1 has multiplicity 3
        out_charges = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, in_charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, out_charges, FlowDirection.OUT, label="out"),
        )
        tensor = SymmetricTensor.random_normal(indices, rng)
        assert tensor.n_blocks == 2

        pba = PaddedBlockArray.from_symmetric(tensor)
        # Block shapes should be (3, 2) and (2, 3) -- different dimensions
        shapes = set(pba.block_shapes)
        assert (3, 2) in shapes
        assert (2, 3) in shapes
        # M_max = 3, N_max = 3 (max of each dimension)
        assert pba.data.shape == (2, 3, 3)
        # Mask should match data shape
        assert pba.mask.shape == pba.data.shape

    def test_varied_shapes_round_trip(self, u1, rng):
        in_charges = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        out_charges = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, in_charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, out_charges, FlowDirection.OUT, label="out"),
        )
        tensor = SymmetricTensor.random_normal(indices, rng)
        pba = PaddedBlockArray.from_symmetric(tensor)
        recovered = pba.to_symmetric()
        np.testing.assert_allclose(recovered.todense(), tensor.todense(), atol=1e-12)

    def test_varied_shapes_padding_is_zero(self, u1, rng):
        in_charges = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        out_charges = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        indices = (
            TensorIndex(u1, in_charges, FlowDirection.IN, label="in"),
            TensorIndex(u1, out_charges, FlowDirection.OUT, label="out"),
        )
        tensor = SymmetricTensor.random_normal(indices, rng)
        pba = PaddedBlockArray.from_symmetric(tensor)
        # Padding region (where mask is False) must be zero
        padding_data = pba.data[~pba.mask]
        np.testing.assert_allclose(padding_data, 0.0, atol=1e-15)


# ===== Tests for PaddedContractionPlan and contract_padded =====


from tenax.algorithms._padded_block_array import PaddedContractionPlan, contract_padded
from tenax.contraction.contractor import contract


def _make_matrix_pair(u1, rng, charges_i, charges_j, charges_k):
    """Create two SymmetricTensors A(i,j) and B(j,k) that can be multiplied."""
    idx_i = TensorIndex(u1, charges_i, FlowDirection.IN, label="i")
    idx_j_out = TensorIndex(u1, u1.dual(charges_j), FlowDirection.OUT, label="j")
    idx_j_in = TensorIndex(u1, charges_j, FlowDirection.IN, label="j")
    idx_k = TensorIndex(u1, u1.dual(charges_k), FlowDirection.OUT, label="k")

    key_a, key_b = jax.random.split(rng)
    A = SymmetricTensor.random_normal((idx_i, idx_j_out), key_a)
    B = SymmetricTensor.random_normal((idx_j_in, idx_k), key_b)
    return A, B


class TestPaddedContractionPlanBuild:
    """Verify PaddedContractionPlan.build produces correct static metadata."""

    def test_build_basic(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)
        plan = PaddedContractionPlan.build(A, B)

        assert isinstance(plan.left_indices, tuple)
        assert isinstance(plan.right_indices, tuple)
        assert isinstance(plan.output_indices, tuple)
        # All index arrays must have the same length (one entry per block pair)
        assert len(plan.left_indices) == len(plan.right_indices)
        assert len(plan.left_indices) == len(plan.output_indices)
        assert plan.num_output_blocks > 0
        assert isinstance(plan.subscripts, str)
        assert "->" in plan.subscripts

    def test_build_has_correct_subscripts(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)
        plan = PaddedContractionPlan.build(A, B)
        # For a 2-leg x 2-leg matmul, subscripts should have the form
        # "XY,YZ->XZ" where Y is the contracted leg and X, Z are free.
        lhs, rhs = plan.subscripts.split("->")
        sub_a, sub_b = lhs.split(",")
        assert len(sub_a) == 2  # A has 2 legs
        assert len(sub_b) == 2  # B has 2 legs
        assert len(rhs) == 2  # output has 2 free legs
        # Contracted char appears in both sub_a and sub_b
        assert sub_a[1] == sub_b[0]  # j is the contracted leg
        # Free chars appear in output
        assert rhs[0] == sub_a[0]
        assert rhs[1] == sub_b[1]

    def test_output_charges_match_valid_blocks(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)
        plan = PaddedContractionPlan.build(A, B)

        # The output charges should form valid charge-sector keys
        assert len(plan.output_charges) == plan.num_output_blocks
        assert len(plan.output_shapes) == plan.num_output_blocks


class TestContractPaddedMatchesSymmetricContract:
    """contract_padded must produce the same result as tenax.contract on SymmetricTensors."""

    def test_matmul_simple(self, u1, rng):
        """Simple case: charges [-1, 0, 1] on all legs."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)

        # Reference: SymmetricTensor contraction
        C_ref = contract(A, B)

        # Padded path
        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)
        pC = contract_padded(plan, pA, pB)

        # Convert back and compare
        C_padded = pC.to_symmetric()
        np.testing.assert_allclose(C_padded.todense(), C_ref.todense(), atol=1e-12)

    def test_matmul_varied_charges(self, u1, rng):
        """Different charge multiplicities on each leg."""
        charges_i = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        charges_j = np.array([-1, 0, 0, 1], dtype=np.int32)
        charges_k = np.array([0, 1, 1, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges_i, charges_j, charges_k)

        C_ref = contract(A, B)

        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)
        pC = contract_padded(plan, pA, pB)
        C_padded = pC.to_symmetric()

        np.testing.assert_allclose(C_padded.todense(), C_ref.todense(), atol=1e-12)

    def test_matmul_single_charge(self, u1, rng):
        """Single charge sector: only one block each."""
        charges = np.array([0, 0, 0], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)

        C_ref = contract(A, B)

        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)
        pC = contract_padded(plan, pA, pB)
        C_padded = pC.to_symmetric()

        np.testing.assert_allclose(C_padded.todense(), C_ref.todense(), atol=1e-12)

    def test_matmul_larger_charges(self, u1, rng):
        """Larger charge range to test more block pairs."""
        charges = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)

        C_ref = contract(A, B)

        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)
        pC = contract_padded(plan, pA, pB)
        C_padded = pC.to_symmetric()

        np.testing.assert_allclose(C_padded.todense(), C_ref.todense(), atol=1e-12)


class TestContractPaddedJITCompatible:
    """contract_padded must work inside jax.jit."""

    def test_plan_is_jit_compatible(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)

        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)

        @jax.jit
        def do_contract(a, b):
            return contract_padded(plan, a, b)

        pC = do_contract(pA, pB)

        # Verify result matches non-jitted path
        C_ref = contract(A, B)
        C_padded = pC.to_symmetric()
        np.testing.assert_allclose(C_padded.todense(), C_ref.todense(), atol=1e-12)

    def test_jit_reuse_plan(self, u1, rng):
        """Same plan, different data: no recompilation needed."""
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A1, B1 = _make_matrix_pair(u1, rng, charges, charges, charges)

        plan = PaddedContractionPlan.build(A1, B1)

        @jax.jit
        def do_contract(a, b):
            return contract_padded(plan, a, b)

        pA1 = PaddedBlockArray.from_symmetric(A1)
        pB1 = PaddedBlockArray.from_symmetric(B1)
        pC1 = do_contract(pA1, pB1)

        # Create different data with the same charge structure
        key2 = jax.random.PRNGKey(99)
        A2, B2 = _make_matrix_pair(u1, key2, charges, charges, charges)
        pA2 = PaddedBlockArray.from_symmetric(A2)
        pB2 = PaddedBlockArray.from_symmetric(B2)
        pC2 = do_contract(pA2, pB2)

        # Both should match their respective reference contractions
        C_ref1 = contract(A1, B1)
        C_ref2 = contract(A2, B2)
        np.testing.assert_allclose(
            pC1.to_symmetric().todense(), C_ref1.todense(), atol=1e-12
        )
        np.testing.assert_allclose(
            pC2.to_symmetric().todense(), C_ref2.todense(), atol=1e-12
        )


class TestContractPaddedOutputStructure:
    """Verify the output PaddedBlockArray has correct structure."""

    def test_output_has_correct_charges(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)

        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)
        pC = contract_padded(plan, pA, pB)

        # Output block charges should match the plan
        assert pC.block_charges == plan.output_charges

    def test_output_padding_is_zero(self, u1, rng):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        A, B = _make_matrix_pair(u1, rng, charges, charges, charges)

        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)
        pC = contract_padded(plan, pA, pB)

        # Padding regions must be zero
        padding_data = pC.data[~pC.mask]
        np.testing.assert_allclose(padding_data, 0.0, atol=1e-12)

    def test_zero_pair_non_overlapping_charges(self, u1, rng):
        """A has only charge 0, B has only charge 1: no matching contracted charge."""
        charges_i = np.array([0, 0], dtype=np.int32)
        charges_j_a = np.array([0, 0], dtype=np.int32)  # A's contracted leg: charge 0
        charges_j_b = np.array([1, 1], dtype=np.int32)  # B's contracted leg: charge 1
        charges_k = np.array([1, 1], dtype=np.int32)

        idx_i = TensorIndex(u1, charges_i, FlowDirection.IN, label="i")
        idx_j_out = TensorIndex(u1, u1.dual(charges_j_a), FlowDirection.OUT, label="j")
        idx_j_in = TensorIndex(u1, charges_j_b, FlowDirection.IN, label="j")
        idx_k = TensorIndex(u1, u1.dual(charges_k), FlowDirection.OUT, label="k")

        key_a, key_b = jax.random.split(rng)
        A = SymmetricTensor.random_normal((idx_i, idx_j_out), key_a)
        B = SymmetricTensor.random_normal((idx_j_in, idx_k), key_b)

        plan = PaddedContractionPlan.build(A, B)
        pA = PaddedBlockArray.from_symmetric(A)
        pB = PaddedBlockArray.from_symmetric(B)
        pC = contract_padded(plan, pA, pB)

        # No matching contracted charges means no contributing pairs
        assert len(plan.left_indices) == 0
        # Output should have valid structure (possibly non-zero output blocks
        # from valid_out_keys, but all data should be zero)
        assert pC.block_charges == plan.output_charges
        if pC.data.size > 0:
            np.testing.assert_allclose(pC.data, 0.0, atol=1e-15)


# ===== Tests for PBA vector operations =====


class TestPBAVectorOps:
    """Test PBA vector operations needed for block-sparse Lanczos."""

    @pytest.fixture
    def pba_pair(self, u1):
        """Two PBAs with same structure, random data."""
        idx_a = TensorIndex(
            u1,
            np.array([0, 0, 1, 1, 1], dtype=np.int32),
            FlowDirection.IN,
            label="a",
        )
        idx_b = TensorIndex(
            u1,
            np.array([0, 0, 0, 1, 1], dtype=np.int32),
            FlowDirection.OUT,
            label="b",
        )
        t1 = SymmetricTensor.random_normal((idx_a, idx_b), jax.random.PRNGKey(0))
        t2 = SymmetricTensor.random_normal((idx_a, idx_b), jax.random.PRNGKey(1))
        return PaddedBlockArray.from_symmetric(t1), PaddedBlockArray.from_symmetric(t2)

    def test_inner_matches_symmetric(self, pba_pair):
        """PBA inner product matches SymmetricTensor block-wise inner product."""
        pba_a, pba_b = pba_pair
        result = pba_inner(pba_a, pba_b)
        # Reference: convert back and compute with numpy
        ta = pba_a.to_symmetric()
        tb = pba_b.to_symmetric()
        expected = 0.0
        for i, key in enumerate(ta._block_keys):
            ba = np.asarray(ta._get_block(i))
            bb = np.asarray(tb._get_block(i))
            expected += np.sum(np.conj(ba) * bb)
        np.testing.assert_allclose(float(result), float(expected.real), rtol=1e-10)

    def test_inner_self_positive(self, pba_pair):
        pba_a, _ = pba_pair
        result = pba_inner(pba_a, pba_a)
        assert float(result) > 0

    def test_inner_jit_compatible(self, pba_pair):
        pba_a, pba_b = pba_pair
        jit_inner = jax.jit(pba_inner)
        result = jit_inner(pba_a, pba_b)
        expected = pba_inner(pba_a, pba_b)
        np.testing.assert_allclose(float(result), float(expected), rtol=1e-12)

    def test_scale(self, pba_pair):
        pba_a, _ = pba_pair
        result = pba_scale(pba_a, 3.0)
        np.testing.assert_allclose(
            np.asarray(result.data), np.asarray(pba_a.data) * 3.0
        )

    def test_add(self, pba_pair):
        pba_a, pba_b = pba_pair
        result = pba_add(pba_a, pba_b)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.asarray(pba_a.data) + np.asarray(pba_b.data),
        )

    def test_axpy(self, pba_pair):
        pba_a, pba_b = pba_pair
        result = pba_axpy(2.5, pba_a, pba_b)
        expected = np.asarray(pba_b.data) + 2.5 * np.asarray(pba_a.data)
        np.testing.assert_allclose(np.asarray(result.data), expected)

    def test_norm(self, pba_pair):
        pba_a, _ = pba_pair
        result = pba_norm(pba_a)
        expected = np.sqrt(float(pba_inner(pba_a, pba_a)))
        np.testing.assert_allclose(float(result), expected, rtol=1e-10)

    def test_zeros_like(self, pba_pair):
        pba_a, _ = pba_pair
        z = pba_zeros_like(pba_a)
        assert z.block_charges == pba_a.block_charges
        assert z.block_shapes == pba_a.block_shapes
        np.testing.assert_array_equal(np.asarray(z.data), 0.0)

    def test_ops_jit_compatible(self, pba_pair):
        """All ops work under jax.jit."""
        pba_a, pba_b = pba_pair

        @jax.jit
        def do_ops(a, b):
            s = pba_scale(a, 2.0)
            added = pba_add(a, b)
            axpy_result = pba_axpy(0.5, a, b)
            n = pba_norm(a)
            return s, added, axpy_result, n

        s, added, axpy_result, n = do_ops(pba_a, pba_b)
        np.testing.assert_allclose(np.asarray(s.data), np.asarray(pba_a.data) * 2.0)
        assert float(n) > 0
