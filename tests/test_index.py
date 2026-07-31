"""Tests for TensorIndex and FlowDirection."""

import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex, net_charge
from tenax.core.symmetry import ProductSymmetry, U1Symmetry, ZnSymmetry


class TestFlowDirection:
    def test_values(self):
        assert int(FlowDirection.IN) == 1
        assert int(FlowDirection.OUT) == -1

    def test_negation(self):
        assert -FlowDirection.IN == FlowDirection.OUT
        assert -FlowDirection.OUT == FlowDirection.IN

    def test_names(self):
        assert FlowDirection.IN.name == "IN"
        assert FlowDirection.OUT.name == "OUT"


class TestTensorIndexCreation:
    def test_basic_creation(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="test")
        assert idx.dim == 3
        assert idx.flow == FlowDirection.IN
        assert idx.label == "test"

    def test_from_charges(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="test")
        assert idx.dim == 3
        assert idx.flow == FlowDirection.IN
        assert idx.label == "test"
        np.testing.assert_array_equal(idx.charges, charges)

    def test_default_label(self, u1):
        charges = np.array([0, 1], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        assert idx.label == ""

    def test_integer_label(self, u1):
        charges = np.array([0, 1], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label=42)
        assert idx.label == 42

    def test_int32_coercion(self, u1):
        """charges should be coerced to int32."""
        charges = np.array([0, 1, -1], dtype=np.int64)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        assert idx.charges.dtype == np.int32

    def test_float_coercion(self, u1):
        """float charges are coerced to int32."""
        charges = np.array([0.0, 1.0, -1.0])
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        assert idx.charges.dtype == np.int32

    def test_multidim_raises(self, u1):
        """2D charge array should raise."""
        charges = np.array([[0, 1], [1, 0]], dtype=np.int32)
        with pytest.raises(ValueError, match="1-D"):
            TensorIndex.from_charges(u1, charges, FlowDirection.IN)

    def test_dim_property(self, u1):
        charges = np.array([0, 1, 2, 3], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.OUT)
        assert idx.dim == 4

    def test_frozen(self, u1):
        """TensorIndex is immutable (frozen dataclass)."""
        charges = np.array([0], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        with pytest.raises((AttributeError, TypeError)):
            idx.flow = FlowDirection.OUT

    def test_slots(self, u1):
        """TensorIndex uses __slots__ — no __dict__ attribute."""
        charges = np.array([0], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        assert not hasattr(idx, "__dict__")


class TestTensorIndexDual:
    def test_dual_flips_flow(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="test")
        d = idx.dual()
        assert d.flow == FlowDirection.OUT

    def test_dual_negates_u1_charges(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN)
        d = idx.dual()
        np.testing.assert_array_equal(d.charges, -u1_charges_3)

    def test_dual_preserves_label(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(
            u1, u1_charges_3, FlowDirection.IN, label="myleg"
        )
        d = idx.dual()
        assert d.label == "myleg"

    def test_dual_of_dual_is_original(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN)
        dd = idx.dual().dual()
        assert dd.flow == idx.flow
        np.testing.assert_array_equal(dd.charges, idx.charges)

    def test_dual_zn(self, z3):
        charges = np.array([0, 1, 2], dtype=np.int32)
        idx = TensorIndex.from_charges(z3, charges, FlowDirection.OUT)
        d = idx.dual()
        assert d.flow == FlowDirection.IN
        np.testing.assert_array_equal(d.charges, z3.dual(charges))


class TestTensorIndexRelabel:
    def test_relabel(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="old")
        new_idx = idx.relabel("new")
        assert new_idx.label == "new"
        assert idx.label == "old"  # original unchanged

    def test_relabel_preserves_everything_else(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.OUT, label="old")
        new_idx = idx.relabel("new")
        assert new_idx.flow == FlowDirection.OUT
        np.testing.assert_array_equal(new_idx.charges, u1_charges_3)
        assert new_idx.symmetry is u1


class TestTensorIndexCompatibility:
    def test_is_dual_of_true(self, u1_index_pair):
        idx_in, idx_out = u1_index_pair
        assert idx_in.is_dual_of(idx_out)
        assert idx_out.is_dual_of(idx_in)

    def test_is_dual_of_same_flow_false(self, u1, u1_charges_3):
        a = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN)
        b = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN)
        assert not a.is_dual_of(b)

    def test_is_dual_of_wrong_charges_false(self, u1):
        a = TensorIndex.from_charges(
            u1, np.array([0, 1], dtype=np.int32), FlowDirection.IN
        )
        b = TensorIndex.from_charges(
            u1, np.array([0, 2], dtype=np.int32), FlowDirection.OUT
        )
        assert not a.is_dual_of(b)

    def test_compatible_with_opposite_flows(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        a = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        b = TensorIndex.from_charges(u1, charges, FlowDirection.OUT)
        assert a.compatible_with(b)

    def test_compatible_with_same_flow_false(self, u1, u1_charges_3):
        a = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN)
        b = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN)
        assert not a.compatible_with(b)

    def test_compatible_with_different_dim_false(self, u1):
        a = TensorIndex.from_charges(
            u1, np.array([0, 1], dtype=np.int32), FlowDirection.IN
        )
        b = TensorIndex.from_charges(
            u1, np.array([0, 1, 2], dtype=np.int32), FlowDirection.OUT
        )
        assert not a.compatible_with(b)

    def test_compatible_with_different_symmetry_false(self, u1, z2):
        charges = np.array([0, 1], dtype=np.int32)
        a = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        b = TensorIndex.from_charges(z2, charges, FlowDirection.OUT)
        assert not a.compatible_with(b)


class TestTensorIndexHashEquality:
    def test_equal_indices(self, u1, u1_charges_3):
        a = TensorIndex.from_charges(
            u1, u1_charges_3.copy(), FlowDirection.IN, label="x"
        )
        b = TensorIndex.from_charges(
            u1, u1_charges_3.copy(), FlowDirection.IN, label="x"
        )
        assert a == b

    def test_different_label_not_equal(self, u1, u1_charges_3):
        a = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="x")
        b = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="y")
        assert a != b

    def test_different_flow_not_equal(self, u1, u1_charges_3):
        a = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN)
        b = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.OUT)
        assert a != b

    def test_hashable(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="test")
        d = {idx: "value"}
        assert d[idx] == "value"

    def test_usable_in_set(self, u1, u1_charges_3):
        a = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="x")
        b = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="y")
        s = {a, b}
        assert len(s) == 2

    def test_repr(self, u1, u1_charges_3):
        idx = TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="test")
        r = repr(idx)
        assert "IN" in r
        assert "test" in r
        assert "dim=3" in r


class TestTensorIndexSectors:
    """Tests for the sector-based representation."""

    def test_sectors_from_charges(self, u1):
        idx = TensorIndex.from_charges(
            u1, np.array([-1, 0, 1, 0], dtype=np.int32), FlowDirection.IN, label="test"
        )
        np.testing.assert_array_equal(idx.sectors, np.array([-1, 0, 1]))
        np.testing.assert_array_equal(idx.multiplicities, np.array([1, 2, 1]))
        assert idx.dim == 4

    def test_charges_preserves_original_order(self, u1):
        original = np.array([1, -1, 0], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, original, FlowDirection.IN)
        # from_charges preserves original ordering for from_dense compat
        np.testing.assert_array_equal(idx.charges, original)

    def test_already_sorted_roundtrip(self, u1):
        charges = np.array([-1, 0, 1], dtype=np.int32)
        idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN)
        np.testing.assert_array_equal(idx.charges, charges)

    def test_fuse_info_none_by_default(self, u1):
        idx = TensorIndex.from_charges(
            u1, np.array([0, 1], dtype=np.int32), FlowDirection.IN
        )
        assert idx.fuse_info is None

    def test_direct_construction(self, u1):
        sectors = np.array([-1, 0, 1], dtype=np.int32)
        mults = np.array([1, 1, 1], dtype=np.int32)
        idx = TensorIndex(u1, sectors, mults, FlowDirection.IN, label="test")
        assert idx.dim == 3
        np.testing.assert_array_equal(idx.sectors, sectors)
        np.testing.assert_array_equal(idx.multiplicities, mults)

    def test_n_sectors(self, u1):
        idx = TensorIndex.from_charges(
            u1, np.array([-1, 0, 0, 1], dtype=np.int32), FlowDirection.IN
        )
        assert idx.n_sectors == 3

    def test_multiplicity_helper(self, u1):
        idx = TensorIndex.from_charges(
            u1, np.array([-1, 0, 0, 1], dtype=np.int32), FlowDirection.IN
        )
        assert idx.multiplicity(0) == 2
        assert idx.multiplicity(-1) == 1
        assert idx.multiplicity(99) == 0

    def test_sector_offset_helper(self, u1):
        idx = TensorIndex.from_charges(
            u1, np.array([-1, 0, 0, 1], dtype=np.int32), FlowDirection.IN
        )
        assert idx.sector_offset(-1) == 0
        assert idx.sector_offset(0) == 1
        assert idx.sector_offset(1) == 3

    def test_has_sector(self, u1):
        idx = TensorIndex.from_charges(
            u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN
        )
        assert idx.has_sector(0)
        assert not idx.has_sector(2)

    def test_dual_sectors(self, u1):
        idx = TensorIndex.from_charges(
            u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN
        )
        d = idx.dual()
        # U(1) dual negates: [-1,0,1] → [1,0,-1], sorted → [-1,0,1]
        np.testing.assert_array_equal(d.sectors, np.array([-1, 0, 1]))
        np.testing.assert_array_equal(d.multiplicities, np.array([1, 1, 1]))


def test_net_charge_reduces_a_single_out_leg():
    """The #733 case: one OUT leg, nothing to fuse against, so no reduction ran."""
    sym = ZnSymmetry(2)
    idx = TensorIndex.from_charges(
        sym, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="a"
    )
    assert net_charge((idx,), (1,)) == 1  # not -1


def test_net_charge_agrees_between_one_leg_and_two_legs():
    sym = ZnSymmetry(3)
    out = TensorIndex.from_charges(
        sym, np.array([0, 1, 2], dtype=np.int32), FlowDirection.OUT, label="a"
    )
    trivial = TensorIndex.from_charges(
        sym, np.array([0], dtype=np.int32), FlowDirection.IN, label="b"
    )
    for q in (0, 1, 2):
        assert net_charge((out,), (q,)) == net_charge((out, trivial), (q, 0))


def test_net_charge_is_identity_for_a_conserving_product_symmetry_block():
    ps = ProductSymmetry(ZnSymmetry(2), U1Symmetry())
    q = ProductSymmetry.encode(1, 1)
    secs = np.array([ProductSymmetry.encode(0, 0), q], dtype=np.int32)
    a = TensorIndex.from_charges(ps, secs, FlowDirection.IN, label="a")
    b = TensorIndex.from_charges(ps, secs, FlowDirection.OUT, label="b")
    assert net_charge((a, b), (q, q)) == ps.identity()


def test_net_charge_matches_plain_summation_for_u1():
    """U(1) is the case the old hand-rolled arithmetic got right; keep it right."""
    sym = U1Symmetry()
    a = TensorIndex.from_charges(
        sym, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    b = TensorIndex.from_charges(
        sym, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    for qa in (-1, 0, 1):
        for qb in (-1, 0, 1):
            assert net_charge((a, b), (qa, qb)) == qa - qb
