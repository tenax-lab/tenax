"""``eigh(bond_order="sector")``: the traceable ordering of a symmetric eigh.

``_eigh_symmetric`` ranks the whole spectrum by magnitude to lay out its bond,
and it does that by reading the eigenvalues on the host (``np.array``), which
raises on a tracer.  That single line is what pinned a ``SymmetricTensor`` pair
to :func:`ipeps_bp_gauge._bp_solve_eager`.

Without a truncation the ranking decides nothing -- every eigenvalue is kept
either way -- so the order is a convention, and ``bond_order="sector"`` picks
the one that needs no host read.  The two differ by a permutation of the bond,
with ``V`` and ``eigenvalues`` permuted together.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor
from tenax.linalg import eigh

IN, OUT = FlowDirection.IN, FlowDirection.OUT


def _psd(charges=(0, 0, 1, 1), seed=0):
    """A PSD SymmetricTensor with two charge sectors, as ``m = M M^dag``."""
    sym = U1Symmetry()
    ch = np.asarray(charges, dtype=np.int32)
    row = TensorIndex.from_charges(sym, ch, OUT, label="row")
    col = TensorIndex.from_charges(sym, ch, IN, label="col")
    M = SymmetricTensor.random_normal((row, col), jax.random.PRNGKey(seed))
    blocks = {k: b @ b.conj().T for k, b in M.blocks.items()}
    return SymmetricTensor._from_blocks_unchecked(blocks, (row, col))


def _reconstruct(V, w):
    """``V diag(w) V^dag`` as a dense matrix, in V's own row order."""
    Vd = np.asarray(V.todense())
    Vm = Vd.reshape(-1, Vd.shape[-1])
    return Vm @ np.diag(np.asarray(w)) @ Vm.conj().T


def test_both_orders_reconstruct_the_same_operator():
    """The mode is a permutation of the bond, not a different decomposition."""
    m = _psd()

    V_d, w_d = eigh(m, ["row"], ["col"], new_bond_label="k")
    V_s, w_s = eigh(m, ["row"], ["col"], new_bond_label="k", bond_order="sector")

    assert sorted(np.asarray(w_d).tolist()) == pytest.approx(
        sorted(np.asarray(w_s).tolist())
    )
    assert _reconstruct(V_d, w_d) == pytest.approx(_reconstruct(V_s, w_s), abs=1e-12)


def test_descending_is_ranked_and_sector_is_grouped():
    m = _psd()

    _V_d, w_d = eigh(m, ["row"], ["col"], new_bond_label="k")
    V_s, _w_s = eigh(m, ["row"], ["col"], new_bond_label="k", bond_order="sector")

    w = np.asarray(w_d)
    assert np.all(np.diff(w) <= 1e-12), f"default order is not descending: {w}"

    charges = np.asarray(V_s.indices[-1].charges).tolist()
    assert charges == sorted(charges), (
        f"sector order must group the bond by charge, got {charges}"
    )


def _sum_eigs(m, alpha, order):
    blocks = {k: alpha * b for k, b in m.blocks.items()}
    scaled = SymmetricTensor._from_blocks_unchecked(blocks, m.indices)
    _V, w = eigh(scaled, ["row"], ["col"], new_bond_label="k", bond_order=order)
    return jnp.sum(w)


def test_sector_order_survives_tracing():
    """The whole point: this one can go inside ``jax.jit``."""
    m = _psd()

    eager = float(_sum_eigs(m, jnp.asarray(1.0), "sector"))
    traced = jax.jit(lambda a: _sum_eigs(m, a, "sector"))(jnp.asarray(1.0))

    assert jnp.isfinite(traced)
    assert traced == pytest.approx(eager)


def test_descending_order_still_cannot_be_traced():
    """And this one cannot -- which is the reason the option exists.

    Its own ``m`` rather than the fixture above's: the host read leaks a tracer
    into whatever tensor it was reading, so a shared operand would carry the
    damage into the next test and report it there.
    """
    m = _psd()

    with pytest.raises(
        (jax.errors.TracerArrayConversionError, jax.errors.UnexpectedTracerError)
    ):
        jax.jit(lambda a: _sum_eigs(m, a, "descending"))(jnp.asarray(1.0))


def test_sector_order_is_refused_where_the_ranking_is_load_bearing():
    """Truncation has to compare sectors, so the two options are exclusive."""
    m = _psd()

    with pytest.raises(ValueError, match="cannot be combined with max_eigenvalues"):
        eigh(m, ["row"], ["col"], max_eigenvalues=2, bond_order="sector")

    with pytest.raises(ValueError, match="must be 'descending' or 'sector'"):
        eigh(m, ["row"], ["col"], bond_order="ascending")
