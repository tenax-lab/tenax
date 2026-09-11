"""``svd(bond_order="sector")``: the traceable ordering of a symmetric SVD.

The twin of ``test_eigh_bond_order.py``, for the other half of the same
obstruction.  ``_truncated_svd_symmetric`` lays out its bond by ranking the
whole spectrum on the host (``np.array`` per sector), which raises on a
tracer; the tracer dispatch that papers over this reroutes to
``_truncated_svd_symmetric_traced``, whose per-sector SVD is the AD primitive
``truncated_svd_ad`` -- and that primitive zeroes any singular value below
``1e-12 * (s_max + 1e-30)``.  On a 1x1 sector the relative arm can never fire,
so the ``+1e-30`` term acts as an *absolute* ~1e-42 cutoff: a real singular
value of 4.6e-43 came back exactly 0.0, which is how the BP gauge solve
"stopped being a gauge" (3.0e-01 drift) whenever its sweep ran under ``jit``.

Without a truncation the ranking decides nothing, so the order is a
convention, and ``bond_order="sector"`` picks the one that needs no host read
-- and, unlike the tracer reroute, it is the *same code path* eager and
traced, so there is no second implementation to disagree with the first: no
floor, no sign fixing, no allocation heuristic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.linalg import svd

IN, OUT = FlowDirection.IN, FlowDirection.OUT


def _two_sector(charges=(0, 0, 1, 1), seed=0):
    """A generic (non-Hermitian) SymmetricTensor with two charge sectors."""
    sym = U1Symmetry()
    ch = np.asarray(charges, dtype=np.int32)
    row = TensorIndex.from_charges(sym, ch, OUT, label="row")
    col = TensorIndex.from_charges(sym, ch, IN, label="col")
    return SymmetricTensor.random_normal((row, col), jax.random.PRNGKey(seed))


def _diagonal(block_diags):
    """Sectors with hand-picked singular values, one diagonal block each."""
    sym = U1Symmetry()
    ch = np.concatenate(
        [np.full(len(d), q, dtype=np.int32) for q, d in sorted(block_diags.items())]
    )
    row = TensorIndex.from_charges(sym, ch, OUT, label="row")
    col = TensorIndex.from_charges(sym, ch, IN, label="col")
    blocks = {
        (q, q): jnp.asarray(np.diag(np.asarray(d, dtype=np.float64)))
        for q, d in block_diags.items()
    }
    return SymmetricTensor._from_blocks_unchecked(blocks, (row, col))


def _reconstruct(U, s, Vh):
    """``U diag(s) Vh`` as a dense matrix, in the operands' own row order."""
    Um = np.asarray(U.todense()).reshape(-1, len(s))
    Vm = np.asarray(Vh.todense()).reshape(len(s), -1)
    return Um @ np.diag(np.asarray(s)) @ Vm


def test_both_orders_are_the_same_decomposition():
    """The mode is a permutation of the bond, not a different factorisation."""
    t = _two_sector()

    U_d, s_d, Vh_d, _ = svd(t, ["row"], ["col"], new_bond_label="k")
    U_s, s_s, Vh_s, _ = svd(
        t, ["row"], ["col"], new_bond_label="k", bond_order="sector"
    )

    assert sorted(np.asarray(s_d).tolist()) == pytest.approx(
        sorted(np.asarray(s_s).tolist())
    )
    assert _reconstruct(U_d, s_d, Vh_d) == pytest.approx(
        _reconstruct(U_s, s_s, Vh_s), abs=1e-12
    )


def test_descending_is_ranked_and_sector_is_grouped():
    """Default: global value order.  Sector: charge groups, descending inside.

    ``{0: [3, 1], 1: [2, 0.5]}`` separates the two: the default interleaves
    the sectors ([3, 2, 1, 0.5]) while sector order keeps each together with
    its own values still descending.  The sector keys are flow-weighted, so
    the charge-1 rows emit bond charge -1 and that group sorts *first*
    ([2, 0.5, 3, 1]) -- the same convention the eigh twin documents.  Either
    way the array is not monotone, and code that read ``s[0]`` as "the
    largest" must not be handed this layout unknowingly.
    """
    t = _diagonal({0: [3.0, 1.0], 1: [2.0, 0.5]})

    _U, s_d, _Vh, _ = svd(t, ["row"], ["col"], new_bond_label="k")
    U_s, s_s, _Vh_s, _ = svd(
        t, ["row"], ["col"], new_bond_label="k", bond_order="sector"
    )

    assert np.asarray(s_d).tolist() == [3.0, 2.0, 1.0, 0.5]
    assert np.asarray(s_s).tolist() == [2.0, 0.5, 3.0, 1.0]
    assert not np.all(np.diff(np.asarray(s_s)) <= 0), (
        "sector order should not be globally monotone on this fixture"
    )
    charges = np.asarray(U_s.indices[-1].charges).tolist()
    assert charges == sorted(charges), f"bond is not charge-grouped: {charges}"


def test_sector_mode_returns_the_spectrum_as_its_own_full_spectrum():
    """``s_full`` is ``s`` in sector mode: nothing was truncated to differ.

    The descending path returns the pre-truncation spectrum separately; with
    no truncation allowed, keeping the two identical is the honest reading,
    and it is what the traced convention already did.
    """
    t = _two_sector()

    _U, s, _Vh, s_full = svd(
        t, ["row"], ["col"], new_bond_label="k", bond_order="sector"
    )

    assert np.array_equal(np.asarray(s), np.asarray(s_full))


def _sum_s(t, alpha, order):
    blocks = {k: alpha * b for k, b in t.blocks.items()}
    scaled = SymmetricTensor._from_blocks_unchecked(blocks, t.indices)
    _U, s, _Vh, _ = svd(scaled, ["row"], ["col"], new_bond_label="k", bond_order=order)
    return jnp.sum(s)


def test_sector_order_survives_tracing():
    """The point of the mode: this one can go inside ``jax.jit``."""
    t = _two_sector()

    eager = float(_sum_s(t, jnp.asarray(1.0), "sector"))
    traced = jax.jit(lambda a: _sum_s(t, a, "sector"))(jnp.asarray(1.0))

    assert jnp.isfinite(traced)
    assert traced == pytest.approx(eager)


def test_traced_sector_mode_is_the_eager_code_path():
    """Same layout, same values -- there is no second implementation.

    The tracer reroute to ``_truncated_svd_symmetric_traced`` is what made
    eager and traced disagree (floor, sign gauge, allocation); sector mode
    must not take it.  Layout is compared exactly; values to 1e-13, which a
    reroute would fail by ~twelve orders on the fixture below.
    """
    t = _two_sector()

    def factor(t):
        return svd(t, ["row"], ["col"], new_bond_label="k", bond_order="sector")

    U_e, s_e, Vh_e, _ = factor(t)
    U_t, s_t, Vh_t, _ = jax.jit(factor)(t)

    assert np.asarray(U_t.indices[-1].charges).tolist() == (
        np.asarray(U_e.indices[-1].charges).tolist()
    )
    assert sorted(U_t.blocks) == sorted(U_e.blocks)
    assert sorted(Vh_t.blocks) == sorted(Vh_e.blocks)
    assert np.asarray(s_t) == pytest.approx(np.asarray(s_e), rel=1e-13)
    assert _reconstruct(U_t, s_t, Vh_t) == pytest.approx(
        _reconstruct(U_e, s_e, Vh_e), abs=1e-13
    )


def test_no_floor_is_applied_under_trace():
    """A 4.6e-43 singular value must come back as itself, not as 0.0.

    This is the defect that broke the BP gauge: ``truncated_svd_ad`` zeroes
    ``s < 1e-12 * (s_max + 1e-30)`` per sector, and on a 1x1 sector the
    ``+1e-30`` arm is an absolute ~1e-42 cutoff.  A gauge transformation
    built from a floored SVD is not a gauge transformation -- the zeroed
    direction carried 13.6% of the 2-site norm in the measured failure --
    so the sector path must apply no floor whatsoever, at any scale f64
    can represent.
    """
    tiny = 4.6e-43
    t = _diagonal({0: [tiny], 1: [1.0]})

    def factor(t):
        return svd(t, ["row"], ["col"], new_bond_label="k", bond_order="sector")[1]

    s_eager = np.asarray(factor(t))
    s_traced = np.asarray(jax.jit(factor)(t))

    assert np.all(s_eager > 0.0)
    assert np.all(s_traced > 0.0), (
        f"a singular value was floored to exact zero under trace: {s_traced}"
    )
    assert np.min(s_traced) == pytest.approx(tiny, rel=1e-12)


def test_descending_order_still_cannot_be_traced_as_one_code_path():
    """Descending under a tracer reroutes to the static-allocation variant.

    That variant applies the subrank floor and its own layout, so it is not
    the eager path -- the reroute is pinned as *reachable* here (no crash),
    and the sector tests above pin that sector mode never takes it.
    """
    t = _two_sector()

    s = jax.jit(lambda a: _sum_s(t, a, "descending"))(jnp.asarray(1.0))
    assert jnp.isfinite(s)


def test_sector_order_is_refused_where_the_ranking_is_load_bearing():
    """Both truncation knobs rank sectors against each other, so both refuse."""
    t = _two_sector()

    with pytest.raises(ValueError, match="cannot be combined with max_singular"):
        svd(t, ["row"], ["col"], max_singular_values=2, bond_order="sector")

    with pytest.raises(ValueError, match="cannot be combined with max_truncation"):
        svd(t, ["row"], ["col"], max_truncation_err=1e-3, bond_order="sector")


def test_the_dense_path_ignores_bond_order_rather_than_refusing_it():
    """Dense has no sectors to group by, so the incompatibility does not apply.

    Same placement as ``eigh``'s (Codex P2 on #939): the guard lives inside
    the SymmetricTensor branch, and a dense truncation with a do-nothing
    ``bond_order`` still runs.
    """
    t = _two_sector()
    dense = DenseTensor(t.todense(), t.indices)

    _U, s, _Vh, _ = svd(
        dense, ["row"], ["col"], max_singular_values=2, bond_order="sector"
    )

    assert len(s) == 2
    assert np.all(np.diff(np.asarray(s)) <= 1e-12), (
        "the dense path ignores bond_order, so it must still rank descending"
    )
    # ... and the membership check still applies on both paths.
    with pytest.raises(ValueError, match="must be 'descending' or 'sector'"):
        svd(dense, ["row"], ["col"], bond_order="ascending")

    with pytest.raises(ValueError, match="must be 'descending' or 'sector'"):
        svd(t, ["row"], ["col"], bond_order="ascending")
