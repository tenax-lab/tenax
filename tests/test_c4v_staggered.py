"""Mechanism tests of the charged C4v CTM (#762, #763), with no converged CTM.

The CTM and energy tests are in ``test_ctm_tensor_c4v.py`` (algorithm bucket).
Design: ``docs/superpowers/specs/2026-11-22-c4v-charged-u1-design.md``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import c4v_fixture, leg, pairs

from tenax.algorithms import _ctm_tensor_c4v as c4v
from tenax.algorithms import _ctm_tensor_energy as energy
from tenax.algorithms._ctm_c4v_charges import (
    _first_unequal_c4v_face,
    _is_charged_c4v,
    _negate_charges,
    _pairs,
)
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_initial_state,
    _c4v_staggered_projectors,
    _c4v_staggered_sweep,
    _c4v_staggered_to_full_env,
    _check_c4v_staggered_invariants,
    _ctm_tensor_c4v_staggered,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import _flip_leg_flow
from tenax.contraction.contractor import contract
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


# ------------------------------------------------------------------ #
# The block projector (spec test 4)                                    #
# ------------------------------------------------------------------ #


def _synthetic_q(sym, fused_charges, blocks):
    """An enlarged corner with the block structure of the 2-sublattice sweep.

    ``fused`` (IN) carries ``fused_charges`` and ``col`` (OUT) their duals, so
    the block (q, q) joins the fused states of charge q with those of charge
    ``dual(q)``.  ``blocks`` maps q to that block.
    """
    fc = np.asarray(fused_charges, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, fc, IN, label="fused"),
        TensorIndex.from_charges(sym, sym.dual(fc), OUT, label="col"),
    )
    data = {(q, q): jnp.asarray(m, dtype=jnp.float64) for q, m in blocks.items()}
    return SymmetricTensor(data, indices)


# U(1): the mirror pair (-1, 1) with values 3 and 1, and the self-partner
# charge 0 with value 2.  Each block is the transpose of its mirror image.
_U1_Q = {
    "fused_charges": [1, 1, 0, -1, -1],
    "blocks": {1: np.diag([3.0, 1.0]), -1: np.diag([3.0, 1.0]), 0: [[2.0]]},
}


@pytest.mark.parametrize(("chi", "dim"), [(5, 5), (4, 3), (3, 3), (2, 2)])
def test_projector_keeps_mirror_pairs_whole(chi, dim):
    sym = U1Symmetry()
    P_left, P_right = _c4v_staggered_projectors(_synthetic_q(sym, **_U1_Q), chi)
    assert [(i.label, i.flow) for i in P_left.indices] == [
        ("fused", OUT),
        ("chi_new", IN),
    ]
    assert [(i.label, i.flow) for i in P_right.indices] == [
        ("col", IN),
        ("chi_new_r", OUT),
    ]
    charges = np.asarray(leg(P_left, "chi_new").charges)
    assert charges.size == dim
    assert np.sum(charges == 1) == np.sum(charges == -1)  # no pair is split
    np.testing.assert_array_equal(leg(P_right, "chi_new_r").charges, sym.dual(charges))
    if chi == 3:
        assert 0 in charges.tolist()  # the self-partner value 2 is kept


def test_projector_refuses_a_chi_that_splits_the_top_pair():
    with pytest.raises(ValueError, match="chi=1"):
        _c4v_staggered_projectors(_synthetic_q(U1Symmetry(), **_U1_Q), 1)


def test_projected_corner_is_block_diagonal():
    Qf = _synthetic_q(U1Symmetry(), **_U1_Q)
    P_left, P_right = _c4v_staggered_projectors(Qf, 3)
    C = contract(contract(P_left, Qf), P_right)
    values = []
    for block in C.blocks.values():
        b = np.asarray(block)
        np.testing.assert_allclose(b, np.diag(np.diag(b)), atol=1e-12)
        values += np.abs(np.diag(b)).tolist()
    assert sorted(values, reverse=True) == pytest.approx([3.0, 3.0, 2.0])


@pytest.mark.parametrize(("chi", "charges"), [(1, [2]), (2, [2]), (3, [1, 2, 3])])
def test_projector_self_partner_block_needs_no_partner(chi, charges):
    # Z4: the pair (1, 3) has value 4; the self-partner charge 2 has value -5.
    blocks = {1: [[4.0]], 3: [[4.0]], 2: [[-5.0]]}
    Qf = _synthetic_q(ZnSymmetry(4), [1, 2, 3], blocks)
    P_left, P_right = _c4v_staggered_projectors(Qf, chi)
    assert np.asarray(leg(P_left, "chi_new").charges).tolist() == charges
    if chi == 1:  # the eigenvalue keeps its sign
        C = contract(contract(P_left, Qf), P_right)
        assert np.asarray(C.todense()).tolist() == [[-5.0]]


# ------------------------------------------------------------------ #
# One step and its invariants (spec test 5, first half)               #
# ------------------------------------------------------------------ #


@functools.cache
def _swept(name, chi=4, steps=3):
    """``(A, a, a_B, states)``: the initial state and the state after each step."""
    A = _site(name)
    a = _build_double_layer_tensor(A)
    a_B = _build_double_layer_tensor(_negate_charges(A))
    states = [_c4v_staggered_initial_state(a)]
    for _ in range(steps):
        states.append(_c4v_staggered_sweep(*states[-1], a, chi))
    return A, a, a_B, states


@pytest.mark.parametrize("name", ["762", "z3", "z4"])
def test_invariants_hold_after_each_step(name):
    A, a, a_B, states = _swept(name)
    for C, T in states:
        _check_c4v_staggered_invariants(C, T, a, a_B)
    ca = leg(states[-1][0], "c_a")
    sym = ca.symmetry
    s = np.asarray(sym.flow_charge(ca.flow, ca.charges))
    # The regime: the corner holds charges that are not self-dual.
    assert np.any(s != sym.dual(s)), f"vacuous: corner charges {s.tolist()}"


def test_invariant_check_catches_an_unnegated_edge():
    A, a, a_B, states = _swept("762")
    C, T = states[0]
    with pytest.raises(RuntimeError, match="I1"):
        _check_c4v_staggered_invariants(C, _negate_charges(T), a, a_B)


def test_enlarged_corner_faces_carry_dual_charges():
    A, a, a_B, states = _swept("762")
    Qf = _c4v_staggered_enlarged_corner(*states[1], a)
    fused, col = leg(Qf, "fused"), leg(Qf, "col")
    assert (fused.flow, col.flow) == (IN, OUT)
    np.testing.assert_array_equal(col.charges, fused.symmetry.dual(fused.charges))


# ------------------------------------------------------------------ #
# Env expansion and the loop (spec test 5, second half)               #
# ------------------------------------------------------------------ #

# The 16 joins of spec section 3.
_JOINS = (  # 8 chi bonds
    ("C1", "c1_r", "T1", "t1_l"),
    ("T1", "t1_r", "C2", "c2_l"),
    ("C2", "c2_d", "T2", "t2_u"),
    ("T2", "t2_d", "C3", "c3_u"),
    ("C3", "c3_l", "T3", "t3_l"),
    ("T3", "t3_r", "C4", "c4_u"),
    ("C4", "c4_r", "T4", "t4_u"),
    ("T4", "t4_d", "C1", "c1_d"),
)
_D2_JOINS = (("T1", "u2"), ("T2", "r2"), ("T3", "d2"), ("T4", "l2"))
_NEG_JOINS = (  # env to negate(env)
    ("T1", "t1_r", "t1_l"),
    ("T3", "t3_l", "t3_r"),
    ("T4", "t4_u", "t4_d"),
    ("T2", "t2_d", "t2_u"),
)
# The #905 layout: labels in the documented order, with their flows.
_ENV_LAYOUT = {
    "C1": [("c1_d", IN), ("c1_r", OUT)],
    "C2": [("c2_l", IN), ("c2_d", OUT)],
    "C3": [("c3_u", IN), ("c3_l", OUT)],
    "C4": [("c4_r", OUT), ("c4_u", IN)],
    "T1": [("t1_l", IN), ("u2", OUT), ("t1_r", OUT)],
    "T2": [("t2_u", IN), ("r2", IN), ("t2_d", OUT)],
    "T3": [("t3_r", OUT), ("d2", IN), ("t3_l", IN)],
    "T4": [("t4_d", OUT), ("l2", OUT), ("t4_u", IN)],
}


def _layout_of(env):
    return {
        name: [(i.label, i.flow) for i in field.indices]
        for name, field in env._asdict().items()
    }


def test_layout_table_is_the_init_env_layout():
    assert _layout_of(initialize_ctm_tensor_env(_site("762"), 4)) == _ENV_LAYOUT


@pytest.mark.parametrize("name", ["762", "z3"])
def test_full_env_joins_and_layout(name):
    A, a, a_B, states = _swept(name)
    env = _c4v_staggered_to_full_env(*states[-1])
    env_B = CTMTensorEnv(*(_negate_charges(t) for t in env))
    for n1, l1, n2, l2 in _JOINS:
        assert pairs(leg(getattr(env, n1), l1), leg(getattr(env, n2), l2)), (
            n1,
            l1,
            n2,
            l2,
        )
    for n, lab in _D2_JOINS:
        assert pairs(leg(getattr(env, n), lab), leg(a, lab)), (n, lab)
    for n, l1, l2 in _NEG_JOINS:
        assert pairs(leg(getattr(env, n), l1), leg(getattr(env_B, n), l2)), (
            n,
            l1,
            l2,
        )
    assert _layout_of(env) == _ENV_LAYOUT


def test_staggered_loop_returns_a_symmetric_env():
    env = _ctm_tensor_c4v_staggered(_site("762"), chi=4, max_iter=2, conv_tol=1e-10)
    assert isinstance(env, CTMTensorEnv)
    for field in env:
        assert isinstance(field, SymmetricTensor)
        assert np.all(np.isfinite(np.asarray(field.todense())))


# ------------------------------------------------------------------ #
# Energy routing and the env guard (spec test 6; Review Focus 3)      #
# ------------------------------------------------------------------ #

GATE = jnp.zeros((2, 2, 2, 2))  # the routing tests mock what consumes it


def _staggered_init_env(A):
    """The 2-sublattice env of the trace boundary: cheap, and built for A."""
    a = _build_double_layer_tensor(A)
    return _c4v_staggered_to_full_env(*_c4v_staggered_initial_state(a))


def _same_metadata(X, Y):
    """Same labels, flows and charges per index, and the same block keys."""
    return (
        [(i.label, i.flow) for i in X.indices] == [(i.label, i.flow) for i in Y.indices]
        and all(
            np.array_equal(i.charges, j.charges) for i, j in zip(X.indices, Y.indices)
        )
        and set(X.blocks) == set(Y.blocks)
    )


def test_charged_energy_takes_the_2site_branch(monkeypatch):
    A = _site("762")
    env = _staggered_init_env(A)
    seen = {}

    def two_site(A_, B_, env_A, env_B, gate, d=None, **kwargs):
        seen.update(A=A_, B=B_, env_A=env_A, env_B=env_B, d=d)
        return jnp.asarray(0.25)

    monkeypatch.setattr(energy, "compute_energy_ctm_tensor_2site", two_site)
    assert float(energy.compute_energy_ctm_tensor(A, env, GATE, d=2)) == 0.25
    assert seen["A"] is A and seen["env_A"] is env and seen["d"] == 2
    assert _same_metadata(seen["B"], _negate_charges(A))
    for got, field in zip(seen["env_B"], env):
        assert _same_metadata(got, _negate_charges(field))


@pytest.mark.parametrize("name", ["uniform_u1", "dense_762", "z2"])
def test_other_input_keeps_the_1site_path(monkeypatch, name):
    A = _site(name)
    env = initialize_ctm_tensor_env(A, 2)
    calls = []

    def two_site(*args, **kwargs):
        raise AssertionError("the 2-site branch was taken")

    def fake_rdm(tag):
        def rdm(A_, env_):
            calls.append(tag)
            return jnp.zeros((2, 2, 2, 2))

        return rdm

    monkeypatch.setattr(energy, "compute_energy_ctm_tensor_2site", two_site)
    monkeypatch.setattr(energy, "_rdm2x1_tensor", fake_rdm("h"))
    monkeypatch.setattr(energy, "_rdm1x2_tensor", fake_rdm("v"))
    assert float(energy.compute_energy_ctm_tensor(A, env, GATE, d=2)) == 0.0
    assert calls == ["h", "v"]


def test_guard_refuses_a_dense_env():
    env = c4v.ctm_tensor_c4v(_site("dense_762"), chi=2, max_iter=1)
    with pytest.raises(ValueError, match="not built for this A"):
        energy.compute_energy_ctm_tensor(_site("762"), env, GATE, d=2)


def test_guard_refuses_a_mispaired_d2_leg():
    A = _site("762")
    env = _staggered_init_env(A)
    bad = env._replace(T1=_flip_leg_flow(env.T1, "u2"))
    with pytest.raises(ValueError, match="T1.u2 does not pair"):
        energy.compute_energy_ctm_tensor(A, bad, GATE, d=2)
