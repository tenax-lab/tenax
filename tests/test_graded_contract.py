"""Algebra of the reference graded contractor (#1035, design §3.4).

The physics check against the exact Fock oracle is in
``test_graded_contract_oracle.py``; this file pins the algebraic properties a
graded contraction must have and the ones a sign-free one lacks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.contraction.contractor import contract
from tenax.core._graded import graded_bar, graded_contract, graded_reorder
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1, FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

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


def _triple(sym, ch):
    A = _rand(
        [
            _idx(sym, ch, IN, "a"),
            _idx(sym, ch, OUT, "x"),
            _idx(sym, ch, IN, "y"),
            _idx(sym, ch, OUT, "b"),
        ],
        0,
    )
    B = _rand(
        [_idx(sym, ch, IN, "x"), _idx(sym, ch, OUT, "c"), _idx(sym, ch, OUT, "y")], 1
    )
    C = _rand([_idx(sym, ch, IN, "c"), _idx(sym, ch, IN, "d")], 2)
    return A, B, C


def _maxdiff(s, t):
    return float(jnp.max(jnp.abs(s.todense() - t.todense())))


@pytest.mark.parametrize("sym,ch", SYMS)
def test_even_tensors_commute_up_to_a_graded_reorder(sym, ch):
    A, B, _ = _triple(sym, ch)
    ab, ba = graded_contract(A, B), graded_contract(B, A)
    assert ab.labels() == ("a", "b", "c")
    assert _maxdiff(graded_reorder(ba, ab.labels()), ab) < 1e-12


@pytest.mark.parametrize("sym,ch", SYMS)
def test_contraction_is_associative(sym, ch):
    A, B, C = _triple(sym, ch)
    left = graded_contract(graded_contract(A, B), C)
    right = graded_contract(A, graded_contract(B, C))
    assert _maxdiff(graded_reorder(right, left.labels()), left) < 1e-12


def test_regime_the_graded_result_is_not_the_sign_free_one():
    A, B, _ = _triple(FermionParity(), [0, 1, 0, 1])
    ab = graded_contract(A, B)
    assert _maxdiff(contract(A, B, output_labels=ab.labels()), ab) > 1e-3


def test_bosonic_operands_fall_back_to_plain_contract():
    u1 = U1Symmetry()
    q = [-1, 0, 1]
    A = _rand([_idx(u1, q, IN, "a"), _idx(u1, q, OUT, "x")], 3)
    B = _rand([_idx(u1, q, IN, "x"), _idx(u1, q, OUT, "b")], 4)
    assert (
        _maxdiff(graded_contract(A, B), contract(A, B, output_labels=("a", "b"))) == 0.0
    )


def test_no_shared_labels_is_the_ordered_outer_product():
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    z = _rand([_idx(FermionParity(), [0, 1], IN, "z")], 9)
    assert graded_contract(A, z).labels() == ("a", "x", "y", "b", "z")


def test_graded_bar_is_the_full_reversal_kept_in_storage_order():
    """Rule 3: conj + flip + reverse the generators, then reorder back with
    the Koszul sign -- which is the (-1)^{sum_{i<j} p_i p_j} phase."""
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    rev = tuple(reversed(range(A.ndim)))
    reversed_form = A.bar().permute_legs(rev)  # generators in reverse order
    assert _maxdiff(graded_bar(A), graded_reorder(reversed_form, A.labels())) < 1e-12
    assert (
        _maxdiff(graded_bar(A), A.bar()) > 1e-3
    )  # regime: the phase is not trivial here


def test_graded_bar_is_an_involution():
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    assert _maxdiff(graded_bar(graded_bar(A)), A) == 0.0


def test_it_traces_under_jit():
    A, B, _ = _triple(FermionParity(), [0, 1, 0, 1])
    jitted = jax.jit(lambda a, b: graded_contract(a, b).todense())
    assert (
        float(jnp.max(jnp.abs(jitted(A, B) - graded_contract(A, B).todense()))) == 0.0
    )


def test_a_dense_operand_is_refused_when_either_is_fermionic():
    A, B, _ = _triple(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(TypeError, match="DenseTensor"):
        graded_contract(A, DenseTensor(B.todense(), B.indices))


def _scaled(U, S, bond):
    from tenax.algorithms._ctm_tensor_projector_2x2 import _scale_bond_by_diag

    return _scale_bond_by_diag(U, S, bond)


def test_graded_svd_reconstructs_when_storage_order_is_not_left_plus_right():
    from tenax.core._graded import graded_svd

    fp, ch = FermionParity(), [0, 1, 0, 1]
    T = _rand(
        [
            _idx(fp, ch, IN, "p"),
            _idx(fp, ch, OUT, "q"),
            _idx(fp, ch, IN, "r"),
            _idx(fp, ch, OUT, "s"),
        ],
        5,
    )
    left, right = ["p", "r"], ["q", "s"]
    U, S, Vh, _ = graded_svd(T, left, right, "k")
    assert U.labels() == ("p", "r", "k") and Vh.labels() == ("k", "q", "s")
    target = graded_reorder(T, left + right)
    assert _maxdiff(graded_contract(_scaled(U, S, "k"), Vh), target) < 1e-12


def test_regime_plain_svd_reorders_sign_free_and_does_not_reconstruct():
    from tenax.linalg import svd

    fp, ch = FermionParity(), [0, 1, 0, 1]
    T = _rand(
        [
            _idx(fp, ch, IN, "p"),
            _idx(fp, ch, OUT, "q"),
            _idx(fp, ch, IN, "r"),
            _idx(fp, ch, OUT, "s"),
        ],
        5,
    )
    U, S, Vh, _ = svd(T, ["p", "r"], ["q", "s"], new_bond_label="k")
    target = graded_reorder(T, ["p", "r", "q", "s"])
    assert _maxdiff(graded_contract(_scaled(U, S, "k"), Vh), target) > 1e-3
