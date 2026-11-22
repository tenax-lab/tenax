"""Mechanism tests of the charged C4v CTM (#762, #763), with no converged CTM.

The CTM and energy tests are in ``test_ctm_tensor_c4v.py`` (algorithm bucket).
Design: ``docs/superpowers/specs/2026-11-22-c4v-charged-u1-design.md``.
"""

from __future__ import annotations

import functools

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import c4v_fixture

from tenax.algorithms._ctm_c4v_charges import (
    _first_unequal_c4v_face,
    _is_charged_c4v,
    _negate_charges,
    _pairs,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

IN, OUT = FlowDirection.IN, FlowDirection.OUT


def _layout(sym, faces, pc, key=0):
    """A random tensor with the face layout ``faces = ((label, flow, charges), ...)``."""
    indices = tuple(
        TensorIndex.from_charges(sym, np.asarray(q, dtype=np.int32), flow, label=label)
        for label, flow, q in faces
    )
    phys = TensorIndex.from_charges(
        sym, np.asarray(pc, dtype=np.int32), IN, label="phys"
    )
    return SymmetricTensor.random_normal((*indices, phys), jax.random.PRNGKey(key))


@functools.cache
def _site(name):
    """The site tensors of these tests, each built once."""
    u1, z4 = U1Symmetry(), ZnSymmetry(4)
    if name == "762":  # the #762 fixture: charged U(1), 10 blocks
        return c4v_fixture(u1, [-1, 1], [-2, 0])
    if name == "test":  # the layout and data of small_peps_u1
        return c4v_fixture(u1, [0, 1], [0, 1])
    if name == "z3":  # Z3: charge 1 has dual 2
        return c4v_fixture(ZnSymmetry(3), [0, 1], [0, 1])
    if name == "z4":  # Z4: the pair (1, 3) and the self-partner charge 2
        return c4v_fixture(z4, [0, 1], [0, 1])
    if name == "z4_self_dual":  # Z4 charges 0 and 2 only
        return c4v_fixture(z4, [0, 2], [0, 1])
    if name == "z2":
        return c4v_fixture(ZnSymmetry(2), [0, 1], [0, 1])
    if name == "zero_u1":
        return c4v_fixture(u1, [0, 0], [0, 0])
    if name == "dense_762":
        A = _site("762")
        return DenseTensor(A.todense(), A.indices)
    q, qd = [-1, 1], [1, -1]
    if name == "fermionic":  # the layout of small_peps_fermionic
        f = [0, 1]
        faces = (("u", OUT, f), ("d", IN, f), ("l", OUT, f), ("r", IN, f))
        return _layout(FermionParity(), faces, f, key=7)
    if name == "uniform_u1":  # conserves the uniform U(1): d IN with u's charges
        f = [0, 1]
        faces = (("u", OUT, f), ("d", IN, f), ("l", OUT, f), ("r", IN, f))
        return _layout(u1, faces, [0, 1])
    if name == "multiset_equal":  # faces equal as multisets, not per position
        faces = (("u", OUT, q), ("d", IN, q), ("l", OUT, q), ("r", IN, q))
        return _layout(u1, faces, [-2, 0])
    if name == "l_flow":  # the charged layout with l IN instead of OUT
        faces = (("u", OUT, q), ("d", IN, qd), ("l", IN, q), ("r", IN, qd))
        return _layout(u1, faces, [-2, 0])
    raise KeyError(name)


def _conserves(sym, key, indices):
    """True when the block key fuses to the identity charge, weighted by flow."""
    parts = [
        sym.flow_charge(idx.flow, np.asarray([q], dtype=np.int32))
        for q, idx in zip(key, indices)
    ]
    return int(sym.fuse_many(parts)[0]) == sym.identity()


# ------------------------------------------------------------------ #
# Charge helpers (spec test 1, and the face and charged checks of     #
# spec section 1)                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", ["762", "z3"])
def test_negate_is_a_data_preserving_involution(name):
    A = _site(name)
    sym = A.indices[0].symmetry
    B = _negate_charges(A)
    for ia, ib in zip(A.indices, B.indices):
        assert (ib.label, ib.flow) == (ia.label, ia.flow)
        np.testing.assert_array_equal(ib.charges, sym.dual(ia.charges))
    np.testing.assert_array_equal(np.asarray(B.todense()), np.asarray(A.todense()))
    assert all(_conserves(sym, key, B.indices) for key in B.blocks)
    A2 = _negate_charges(B)
    assert set(A2.blocks) == set(A.blocks)
    for key, block in A.blocks.items():
        np.testing.assert_array_equal(np.asarray(A2.blocks[key]), np.asarray(block))


def test_negate_refuses_fermionic_and_dense_input():
    with pytest.raises(NotImplementedError, match="fermionic"):
        _negate_charges(_site("fermionic"))
    with pytest.raises(TypeError, match="SymmetricTensor"):
        _negate_charges(_site("dense_762"))


def test_pairs_needs_opposite_flows_and_equal_charges():
    sym = U1Symmetry()
    q = np.asarray([-1, 1], dtype=np.int32)
    x = TensorIndex.from_charges(sym, q, OUT, label="x")
    assert _pairs(x, TensorIndex.from_charges(sym, q, IN, label="y"))
    assert not _pairs(x, TensorIndex.from_charges(sym, q, OUT, label="y"))
    assert not _pairs(x, TensorIndex.from_charges(sym, sym.dual(q), IN, label="y"))


@pytest.mark.parametrize(
    ("name", "face"),
    [
        ("762", None),
        ("test", None),
        ("z3", None),
        ("z2", None),
        ("uniform_u1", "d"),
        ("multiset_equal", "d"),
        ("l_flow", "l"),
    ],
)
def test_first_unequal_c4v_face(name, face):
    assert _first_unequal_c4v_face(_site(name)) == face


@pytest.mark.parametrize(
    ("name", "charged"),
    [
        ("762", True),
        ("test", True),
        ("z3", True),
        ("z4", True),
        ("z2", False),
        ("zero_u1", False),
        ("z4_self_dual", False),
        ("uniform_u1", False),
        ("multiset_equal", False),
        ("dense_762", False),
        ("fermionic", False),
    ],
)
def test_is_charged_c4v(name, charged):
    assert _is_charged_c4v(_site(name)) is charged


def test_is_charged_c4v_needs_the_five_labels():
    assert _is_charged_c4v(_site("762").relabel("phys", "p")) is False
