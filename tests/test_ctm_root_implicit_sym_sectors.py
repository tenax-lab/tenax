"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3)."""

import jax
import numpy as np
import pytest

from tenax import FlowDirection, TensorIndex, U1Symmetry, ZnSymmetry
from tenax.algorithms._ctm_root_implicit_sym_sectors import (
    BondLayout,
    bond_index_from_layout,
)


def test_bond_index_from_layout_has_one_sector_per_retained_charge():
    layout = BondLayout.from_dims({-1: 2, 0: 3, 1: 2})
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
    layout = BondLayout.from_dims({-1: 0, 0: 4, 1: 0})
    idx = bond_index_from_layout(layout, U1Symmetry(), FlowDirection.IN, "b")
    assert list(idx.sectors) == [0]
    assert list(idx.multiplicities) == [4]
    assert layout.total == 4


def test_bond_index_from_layout_raises_on_empty_dict():
    layout = BondLayout.from_dims({})
    with pytest.raises(ValueError, match="no charges"):
        bond_index_from_layout(layout, U1Symmetry(), FlowDirection.IN, "b")


def test_bond_index_from_layout_raises_on_all_zero_dict():
    # All sectors retain nothing: distinct input from the empty dict, but
    # normalises to the same "no charges retained" BondLayout — both must
    # raise the same way rather than one silently building a rank-0 index.
    layout = BondLayout.from_dims({-1: 0, 0: 0, 1: 0})
    with pytest.raises(ValueError, match="no charges"):
        bond_index_from_layout(layout, U1Symmetry(), FlowDirection.IN, "b")


def test_from_dims_rejects_negative_dims():
    with pytest.raises(ValueError, match="non-negative"):
        BondLayout.from_dims({-1: 2, 0: -1})


def test_from_dims_sorts_charges_regardless_of_input_order():
    # Both existing tests above write their dicts in already-sorted key
    # order, so the sort step is a no-op and never exercised. Feed it charges
    # out of order and check the sort actually happens, and that the built
    # index's multiplicities line up with the *right* charges: TensorIndex's
    # multiplicity()/sector_offset() use np.searchsorted, which silently
    # returns wrong results (not an error) if sectors aren't truly sorted.
    layout = BondLayout.from_dims({1: 2, -1: 3, 0: 1})
    assert layout.sectors == [-1, 0, 1]
    assert layout.dim_of(-1) == 3
    assert layout.dim_of(0) == 1
    assert layout.dim_of(1) == 2

    idx = bond_index_from_layout(layout, U1Symmetry(), FlowDirection.IN, "b")
    assert list(idx.sectors) == [-1, 0, 1]
    assert list(idx.multiplicities) == [3, 1, 2]
    assert idx.multiplicity(-1) == 3
    assert idx.multiplicity(0) == 1
    assert idx.multiplicity(1) == 2


def test_bond_index_from_layout_with_zn_symmetry():
    # Non-U(1) coverage: Z_2 charges live in {0, 1}, fused mod 2.
    layout = BondLayout.from_dims({0: 5, 1: 3})
    idx = bond_index_from_layout(layout, ZnSymmetry(2), FlowDirection.OUT, "chi")
    assert list(idx.sectors) == [0, 1]
    assert list(idx.multiplicities) == [5, 3]
    assert idx.symmetry.n == 2
    assert layout.total == 8


def test_bond_layout_is_not_a_pytree_and_is_hashable():
    # BondLayout must be frozen, opaque metadata under jax.tree_util so its
    # sector dimensions never become tracers, and it must be hashable so it
    # can be passed as a static/nondiff argument (e.g. custom_vjp's
    # nondiff_argnums). A NamedTuple with a dict field fails both.
    layout = BondLayout.from_dims({-1: 2, 0: 3, 1: 2})

    leaves = jax.tree.leaves(layout)
    assert leaves == [layout] or leaves == []

    hash(layout)  # must not raise

    jitted = jax.jit(lambda lay: lay.total, static_argnums=0)
    assert jitted(layout) == 7
