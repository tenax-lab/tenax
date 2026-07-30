"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3)."""

import numpy as np
import pytest

from tenax import FlowDirection, TensorIndex, U1Symmetry, ZnSymmetry
from tenax.algorithms._ctm_root_implicit_sym_sectors import (
    BondLayout,
    bond_index_from_layout,
)


def test_bond_index_from_layout_has_one_sector_per_retained_charge():
    layout = BondLayout(dims={-1: 2, 0: 3, 1: 2})
    idx = bond_index_from_layout(layout, U1Symmetry(), FlowDirection.OUT, "chi_new")
    assert list(idx.sectors) == [-1, 0, 1]
    assert list(idx.multiplicities) == [2, 3, 2]
    assert idx.flow is FlowDirection.OUT
    assert idx.label == "chi_new"
    assert int(np.sum(idx.multiplicities)) == layout.total == 7


def test_bond_index_from_layout_drops_empty_sectors():
    # A sector that retained nothing must not appear as a zero-width sector:
    # a zero multiplicity is a shape of 0 that propagates into every
    # downstream contraction.
    layout = BondLayout(dims={-1: 0, 0: 4, 1: 0})
    idx = bond_index_from_layout(layout, U1Symmetry(), FlowDirection.IN, "b")
    assert list(idx.sectors) == [0]
    assert list(idx.multiplicities) == [4]
    assert layout.total == 4
