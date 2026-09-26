"""The fermionic simple update against the Fock oracle (#1035, design §5 step 6).

A bond update is, before truncation, the gate applied to the two sites:
``A_new . lambda . B_new == exp(-tau h) . (A . B)``.  On a finite cluster that
is checkable exactly -- the cluster state after the update must equal the
Fock-space gate applied to the cluster state before it.  With all bond
weights 1 and no truncation, the update is an exact local move, and slicing
the open boundary afterwards commutes with it.

The vertical bonds are the most sign-sensitive: their sites are not
Jordan-Wigner neighbours on the row-major line, so a string runs between
them.  The sign-free update this replaces gives the hard-core-boson gate.
"""

from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest
from _fermionic_fock_oracle import sites_of
from _graded_cluster import fock_of_kets, production_site
from test_graded_contract_oracle import _fock_gate

import tenax.algorithms.ipeps_simple_update as su
from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _trotter_gate,
    spinless_fermion_gate,
)

jax.config.update("jax_enable_x64", True)

TAU = 0.4
R, C = 2, 2


def _full_even_site(rng) -> np.ndarray:
    """A random parity-even site with every leg of dimension 2."""
    A = rng.standard_normal((2, 2, 2, 2, 2))
    for k in itertools.product(*[range(2)] * 5):
        if sum(k) % 2:
            A[k] = 0.0
    return A


def _ket(t, s):
    """A production site tensor as a cluster ket: boundary legs sliced to
    their parity-0 slot (no Koszul sign) and dropped, the rest renamed to the
    oracle's bond labels ``b{k}`` and ``p{n}``."""
    from _fermionic_fock_oracle import bonds_of

    n = sites_of(R, C).index(s)
    name = {}
    for b, (a, x, u, y) in enumerate(bonds_of(R, C)):
        name[(a, x)] = f"b{b}"
        name[(u, y)] = f"b{b}"
    dense = np.asarray(t.todense())
    idx = []
    for k, (lab, ix) in enumerate(zip(t.labels(), t.indices)):
        if lab == "phys":
            idx.append(ix.relabel(f"p{n}"))
        elif (s, lab) in name:
            idx.append(ix.relabel(name[(s, lab)]))
        else:
            assert int(ix.charges[0]) == 0
            dense = np.take(dense, [0], axis=k)
    keep = [k for k, lab in enumerate(t.labels()) if lab == "phys" or (s, lab) in name]
    dense = dense.reshape([dense.shape[k] for k in keep])
    from tenax.core.tensor import SymmetricTensor

    return SymmetricTensor.from_dense(jax.numpy.asarray(dense), tuple(idx))


def _state(tensors: dict) -> np.ndarray:
    return fock_of_kets(R, C, {s: _ket(t, s) for s, t in tensors.items()})


@pytest.fixture
def cluster():
    rng = np.random.default_rng(7)
    return {s: production_site(_full_even_site(rng)) for s in sites_of(R, C)}


@pytest.fixture
def gate():
    return _trotter_gate(spinless_fermion_gate(FPEPSConfig(D=2, t=1.0, V=0.0)), TAU)


@pytest.fixture
def exact(monkeypatch):
    """No truncation: lift the fermionic layout pin, so the SVD keeps every
    singular value and the update is exact."""
    monkeypatch.setattr(su, "_truncation_base_charges", lambda A, leg: None)


ONES = np.ones(2)


#: All four bonds of the 2x2 cluster.  Mutants (each kills the listed cases):
#: no twist on ``U``, a sign-free ``permute_legs`` in place of
#: ``graded_reorder``, and the old sign-free update -- all four; a sign-free
#: ``theta`` -- the two horizontal ones only.  On a vertical bond that mutant
#: is off by ``(-1)**(p_d * p_u)`` on A (the Koszul move of ``A.d`` past
#: ``l, r, phys`` times rule 2's pair sign, using that A is even), and A's
#: ``u`` is open boundary (parity 0) on a 2x2 cluster, so it is invisible
#: there; a 3x2 cluster would see it.  Equivalent, as predicted: the twist on
#: ``Vh``'s end instead of ``U``'s (one sign per bond, either end), and the
#: gate as the right operand (an even operator supercommutes; measured 0.0).
BONDS = [
    ("h", (0, 0), (0, 1)),
    ("h", (1, 0), (1, 1)),
    ("v", (0, 0), (1, 0)),
    ("v", (0, 1), (1, 1)),
]


@pytest.mark.parametrize("kind,s,t", BONDS, ids=[f"{k}{s}{t}" for k, s, t in BONDS])
def test_a_bond_update_is_the_fermionic_gate(cluster, gate, exact, kind, s, t):
    update = (
        su._simple_update_2site_horizontal_tensor
        if kind == "h"
        else su._simple_update_2site_vertical_tensor
    )
    psi0 = _state(cluster)
    A_new, B_new, lam = update(cluster[s], cluster[t], gate, ONES, ONES, 16)
    # Vidal form: the new bond's weight lives in lam; put it back on A.
    A_new = scale_bond_axis(A_new, "r" if kind == "h" else "d", lam)
    psi1 = _state({**cluster, s: A_new, t: B_new})

    n = sites_of(R, C)
    want = _fock_gate(psi0, n.index(s), n.index(t), TAU)
    cos = abs(np.vdot(psi0, want)) / np.linalg.norm(psi0) / np.linalg.norm(want)
    assert cos < 1 - 1e-3  # regime: the gate moves the state

    for x in (A_new, B_new):  # flows in == flows out
        assert [i.flow for i in x.indices] == [i.flow for i in cluster[s].indices]
    scale = np.vdot(psi1, want) / np.vdot(psi1, psi1)  # the SU's normalisation
    np.testing.assert_allclose(scale * psi1, want, atol=1e-12 * np.linalg.norm(want))
