"""Tests for TensorIndex and FlowDirection."""

import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry, ZnSymmetry


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
