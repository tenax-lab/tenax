"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax import (
    FlowDirection,
    SymmetricTensor,
    TensorIndex,
    U1Symmetry,
    ZnSymmetry,
    fuse_indices,
)
from tenax.algorithms._ctm_root_implicit_sym_sectors import (
    BondLayout,
    bond_index_from_layout,
    sector_svd,
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


def _matrix_tensor(seed=0, sectors=(-1, 0, 1), mults=(1, 2, 1)):
    """A fused 2-leg tensor shaped like a half-infinite environment cut."""
    sym = U1Symmetry()

    def leg(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.asarray(sectors),
            multiplicities=np.asarray(mults),
            flow=flow,
            label=lbl,
        )

    ec = SymmetricTensor.random_normal_np(
        (
            leg(FlowDirection.OUT, "chi_r"),
            leg(FlowDirection.OUT, "a_r"),
            leg(FlowDirection.IN, "chi_d"),
            leg(FlowDirection.IN, "a_d"),
        ),
        np.random.RandomState(seed),
    )
    fused = fuse_indices(ec, 2, 3, "row", FlowDirection.IN)
    return fuse_indices(fused, 0, 1, "col", FlowDirection.OUT)


def test_sector_svd_truncates_globally_not_per_sector():
    m = _matrix_tensor()
    chi = 6
    sectors, layout = sector_svd(m, chi, row_axis=1, col_axis=0)

    assert layout.total == chi
    # Global truncation: the retained values are exactly the top chi of the
    # union over sectors.  A per-sector rule would keep chi/n_sectors each.
    kept = sorted(
        (float(s) for q in layout.sectors for s in sectors[q].s[: layout.dim_of(q)]),
        reverse=True,
    )
    every = sorted((float(s) for q in sectors for s in sectors[q].s), reverse=True)
    assert kept == pytest.approx(every[:chi], rel=1e-12)


def test_sector_svd_null_space_is_the_exact_complement():
    m = _matrix_tensor()
    sectors, layout = sector_svd(m, 6, row_axis=1, col_axis=0)
    for q in layout.sectors:
        blk = sectors[q]
        k = layout.dim_of(q)
        u_star, u_perp = blk.U[:, :k], blk.U[:, k:]
        assert float(jnp.max(jnp.abs(u_star.conj().T @ u_perp))) < 1e-12
        # U_perp must actually span the rest, not be empty by accident.
        assert u_perp.shape[1] == blk.U.shape[0] - k


def _nonsquare_matrix_tensor(seed=0):
    """A fused matrix whose sector blocks are NOT square.

    Square blocks cannot detect a U/Vh orientation swap, which is why the
    original sector_svd tests missed one.
    """
    sym = U1Symmetry()

    def leg(flow, lbl, mults):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array([-1, 0, 1]),
            multiplicities=np.array(mults),
            flow=flow,
            label=lbl,
        )

    ec = SymmetricTensor.random_normal_np(
        (
            leg(FlowDirection.OUT, "chi_r", [1, 2, 1]),
            leg(FlowDirection.OUT, "a_r", [1, 1, 1]),
            leg(FlowDirection.IN, "chi_d", [2, 1, 2]),
            leg(FlowDirection.IN, "a_d", [1, 1, 1]),
        ),
        np.random.RandomState(seed),
    )
    fused = fuse_indices(ec, 2, 3, "row", FlowDirection.IN)
    return fuse_indices(fused, 0, 1, "col", FlowDirection.OUT)


def test_sector_svd_orients_u_to_row_axis_and_vh_to_col_axis():
    # Regression for a U/Vh orientation bug: sector_svd used row_axis/col_axis
    # only to compute each block's bond charge, never to orient the block
    # before jnp.linalg.svd -- so U ended up spanning whichever axis the
    # SymmetricTensor happened to store first, not the caller-requested
    # row_axis. Square sector blocks can't see this (U and Vh have the same
    # shape either way, and generic SVD algebra like orthogonality is
    # invariant under swapping them), which is why the original tests above
    # missed it. This fixture is deliberately non-square.
    m = _nonsquare_matrix_tensor()
    sectors, _ = sector_svd(m, 4, row_axis=1, col_axis=0)

    row_index, col_index = m.indices[1], m.indices[0]
    assert row_index.label == "row"
    assert col_index.label == "col"

    saw_nonsquare_sector = False
    for q, blk in sectors.items():
        row_dim = row_index.multiplicity(q)
        col_dim = col_index.multiplicity(q)
        if row_dim != col_dim:
            saw_nonsquare_sector = True

        # (1) Shapes must match the *requested* row/col axes, not whichever
        # axis happened to be stored first. On its own this could pass by a
        # lucky shape coincidence, which is why (2) also pins the content.
        assert blk.U.shape[0] == row_dim
        assert blk.Vh.shape[1] == col_dim

        # (2) The reconstruction must equal the block oriented as (row, col).
        # The tensor's stored axis order is (col, row) -- labels are
        # ('col', 'row') -- so the raw stored block has shape
        # (col_dim, row_dim) and the correctly oriented reconstruction is its
        # transpose, not the raw block itself.
        k = len(blk.s)
        recon = blk.U[:, :k] @ jnp.diag(blk.s) @ blk.Vh[:k, :]
        raw_block = m.blocks[(q, q)]
        assert raw_block.shape == (col_dim, row_dim)
        np.testing.assert_allclose(
            np.asarray(recon), np.asarray(raw_block).T, atol=1e-10
        )

    assert saw_nonsquare_sector  # fixture must actually exercise one


def test_sector_svd_floors_against_the_global_maximum():
    # A sector whose own singular values are all tiny must not have its noise
    # promoted: the floor is relative to the largest SV of the whole cut, not
    # of the sector.
    m = _matrix_tensor()
    sectors, layout = sector_svd(m, 6, row_axis=1, col_axis=0)
    biggest = max(float(sectors[q].s[0]) for q in sectors)
    for q in layout.sectors:
        k = layout.dim_of(q)
        assert float(jnp.min(sectors[q].S_keep_diag[:k])) >= 1e-12 * biggest * 0.5
