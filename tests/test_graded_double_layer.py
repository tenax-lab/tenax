"""Graded fusion and the graded double layer (#1035, design §5 step 3).

The physics check against the exact Fock oracle is in
``test_graded_contract_oracle.py``; this file pins the algebra: a graded
fuse must commute with graded contraction, invert under split, and build a
double layer with production's structure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_open_tensor,
    _build_double_layer_tensor,
    _fuse_pair_by_label,
)
from tenax.algorithms._graded_double_layer import (
    build_graded_double_layer,
    graded_fuse_pair,
    graded_split_pair,
)
from tenax.core._graded import graded_bar, graded_contract, graded_reorder
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1, FermionParity, U1Symmetry
from tenax.core.tensor import SymmetricTensor

IN, OUT = FlowDirection.IN, FlowDirection.OUT
SYMS = [
    pytest.param(FermionParity(), [0, 1, 0, 1], id="FermionParity"),
    pytest.param(FermionicU1(), [-1, 0, 1, 0, 1], id="FermionicU1"),
]


def _idx(sym, charges, flow, label):
    return TensorIndex.from_charges(
        sym, np.asarray(charges, np.int32), flow, label=label
    )


def _rand(indices, seed):
    return SymmetricTensor.random_normal(
        indices=tuple(indices), key=jax.random.PRNGKey(seed)
    )


def _maxdiff(s, t):
    return float(jnp.max(jnp.abs(s.todense() - t.todense())))


def _bond_pair(sym, ch):
    """``T1`` holds ket leg ``k`` (IN) and bra leg ``K`` (OUT) with a free leg
    between them, so the fuse's reorder is a real permutation; ``T2`` holds
    their partners (flows flipped) in the opposite order."""
    T1 = _rand(
        [
            _idx(sym, ch, IN, "k"),
            _idx(sym, ch, OUT, "a"),
            _idx(sym, ch, OUT, "K"),
            _idx(sym, ch, IN, "b"),
        ],
        0,
    )
    T2 = _rand(
        [_idx(sym, ch, IN, "c"), _idx(sym, ch, IN, "K"), _idx(sym, ch, OUT, "k")], 1
    )
    return T1, T2


@pytest.mark.parametrize("sym,ch", SYMS)
@pytest.mark.parametrize("first", ["T1", "T2"])
def test_contracting_fused_legs_equals_contracting_the_pairs(sym, ch, first):
    T1, T2 = _bond_pair(sym, ch)
    F1 = graded_fuse_pair(T1, "k", "K", "f")
    F2 = graded_fuse_pair(T2, "k", "K", "f")
    x, y, fx, fy = (T1, T2, F1, F2) if first == "T1" else (T2, T1, F2, F1)
    ref = graded_contract(x, y)
    got = graded_contract(fx, fy)
    assert _maxdiff(graded_reorder(got, ref.labels()), ref) < 1e-12


def test_regime_a_sign_free_fuse_does_not_commute_with_graded_contraction():
    """Production's fuse (``fuse_indices`` after a sign-free move) is what the
    graded fuse replaces; it must fail the property above."""
    T1, T2 = _bond_pair(FermionParity(), [0, 1, 0, 1])
    ref = graded_contract(T1, T2)
    F1 = _fuse_pair_by_label(T1, "k", "K", "f", OUT)
    F2 = _fuse_pair_by_label(T2, "k", "K", "f", IN)
    got = graded_contract(F1, F2)
    assert _maxdiff(graded_reorder(got, ref.labels()), ref) > 1e-3


@pytest.mark.parametrize("sym,ch", SYMS)
def test_split_inverts_fuse(sym, ch):
    for T in _bond_pair(sym, ch):
        F = graded_fuse_pair(T, "k", "K", "f")
        back = graded_split_pair(F, "f")
        assert set(back.labels()) == set(T.labels())
        assert _maxdiff(graded_reorder(back, T.labels()), T) == 0.0


def test_the_fused_leg_takes_the_bra_legs_flow_and_place():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    F = graded_fuse_pair(T1, "k", "K", "f")
    assert F.labels() == ("f", "a", "b")
    assert F.indices[0].flow == OUT


def test_a_pair_with_equal_flows_is_refused():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="same flow"):
        graded_fuse_pair(T1, "k", "b", "f")


def test_a_fused_label_already_in_use_is_refused():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="already"):
        graded_fuse_pair(T1, "k", "K", "a")  # "a" is a remaining label


def test_an_unknown_label_is_refused():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="no leg labelled"):
        graded_fuse_pair(T1, "k", "nope", "f")
    with pytest.raises(ValueError, match="no leg labelled"):
        graded_split_pair(T1, "nope")


def test_splitting_a_leg_whose_flow_changed_since_fusion_is_refused():
    """``graded_bar`` of a fused tensor flips the fused leg's flow; splitting
    it would silently use the fusion-time flow instead, mis-splitting."""
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    F = graded_fuse_pair(T1, "k", "K", "f")
    barred = graded_bar(F)
    with pytest.raises(ValueError, match="flow"):
        graded_split_pair(barred, "f")


def test_splitting_a_leg_that_was_never_fused_is_refused():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="not a fused leg"):
        graded_split_pair(T1, "a")


def test_a_bosonic_pair_fuses_as_production_does():
    u1 = U1Symmetry()
    q = [-1, 0, 1]
    T = _rand([_idx(u1, q, IN, "k"), _idx(u1, q, OUT, "a"), _idx(u1, q, OUT, "K")], 3)
    graded = graded_fuse_pair(T, "k", "K", "f")
    plain = _fuse_pair_by_label(graded_reorder(T, ["k", "K", "a"]), "k", "K", "f", OUT)
    assert graded.labels() == plain.labels()
    assert _maxdiff(graded, plain) == 0.0


def _site(sym, ch, seed):
    """A site tensor in production's convention: ``(u, d, l, r, phys)``
    with flows ``(OUT, IN, OUT, IN, IN)``."""
    flows = (OUT, IN, OUT, IN)
    legs = [_idx(sym, ch, f, x) for f, x in zip(flows, "udlr")]
    legs.append(_idx(sym, [0, 1], IN, "phys"))
    return _rand(legs, seed)


# Dimension-2 bonds: a double layer squares every bond, and FermionicU1 at
# dimension 5 costs minutes of block-sparse compile for a structure check.
SITE_SYMS = [
    pytest.param(FermionParity(), [0, 1], id="FermionParity"),
    pytest.param(FermionicU1(), [0, 1], id="FermionicU1"),
]


@pytest.mark.parametrize("sym,ch", SITE_SYMS)
@pytest.mark.parametrize("open_phys", [False, True])
def test_the_graded_double_layer_has_productions_structure(sym, ch, open_phys):
    A = _site(sym, ch, 5)
    if open_phys:
        prod = _build_double_layer_open_tensor(A)
        graded = build_graded_double_layer(A, phys_bra="phys_bra")
    else:
        prod = _build_double_layer_tensor(A)
        graded = build_graded_double_layer(A)
    assert graded.labels() == prod.labels()
    for g, p in zip(graded.indices, prod.indices):
        assert g.flow == p.flow
        assert list(g.charges) == list(p.charges)


def test_regime_the_graded_double_layer_differs_from_productions():
    A = _site(FermionParity(), [0, 1], 5)
    assert _maxdiff(build_graded_double_layer(A), _build_double_layer_tensor(A)) > 1e-3


def test_the_graded_double_layer_traces_under_jit():
    A = _site(FermionParity(), [0, 1], 5)
    eager = build_graded_double_layer(A, phys_bra="phys_bra")
    jitted = jax.jit(lambda t: build_graded_double_layer(t, phys_bra="phys_bra"))(A)
    assert _maxdiff(jitted, eager) < 1e-12
