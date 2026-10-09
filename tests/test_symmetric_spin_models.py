"""U(1)-Sz and Z2 XXZ gates and the Z2 init pair."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps import (
    xxz_gate,
    xxz_gate_u1sz,
    xxz_gate_z2,
    xxz_z2_init_pair,
)
from tenax.core.tensor import SymmetricTensor


def _conserves(t):
    signs = [int(ix.flow.value) for ix in t.indices]
    n = getattr(t.indices[0].symmetry, "n", None)
    for key in t.blocks:
        tot = sum(s * q for s, q in zip(signs, key))
        assert (tot % n == 0) if n else (tot == 0), key


@pytest.mark.parametrize("delta", [0.3, 1.0])
@pytest.mark.parametrize("build", [xxz_gate_u1sz, xxz_gate_z2])
def test_gate_equals_the_dense_xxz_gate(build, delta):
    g = build(delta)
    assert isinstance(g, SymmetricTensor)
    np.testing.assert_array_equal(
        np.asarray(g.todense()), np.asarray(xxz_gate(delta).todense())
    )
    _conserves(g)
    assert len(g.blocks) > 1
    # Check physical charges
    if build is xxz_gate_u1sz:
        assert set(np.asarray(g.indices[0].charges).tolist()) == {1, -1}
    else:  # xxz_gate_z2
        assert np.asarray(g.indices[0].charges).tolist() == [0, 1]


def test_z2_physical_charges_are_down_spin_parity():
    g = xxz_gate_z2(1.0)
    assert np.asarray(g.indices[0].charges).tolist() == [0, 1]


def test_z2_pair_is_symmetric_and_blocked():
    A, B = xxz_z2_init_pair(D=2, key=jax.random.PRNGKey(0))
    for t in (A, B):
        assert isinstance(t, SymmetricTensor)
        assert [ix.label for ix in t.indices] == ["u", "d", "l", "r", "phys"]
        # Check flows: (u=OUT(-1), d=IN(1), l=OUT(-1), r=IN(1), phys=IN(1))
        flows = [int(ix.flow.value) for ix in t.indices]
        assert flows == [-1, 1, -1, 1, 1]
        # Check virtual charges [0, 1] (D=2)
        for label in ["u", "d", "l", "r"]:
            idx = next(i for i, ix in enumerate(t.indices) if ix.label == label)
            assert np.asarray(t.indices[idx].charges).tolist() == [0, 1]
        # Check physical charges [0, 1]
        phys_idx = 4
        assert np.asarray(t.indices[phys_idx].charges).tolist() == [0, 1]
        assert len(t.blocks) > 1
        _conserves(t)


def test_z2_pair_refuses_d1():
    with pytest.raises(ValueError, match="D must be >= 2"):
        xxz_z2_init_pair(D=1, key=jax.random.PRNGKey(0))
