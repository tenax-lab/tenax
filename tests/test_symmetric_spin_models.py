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


def test_z2_physical_charges_are_down_spin_parity():
    g = xxz_gate_z2(1.0)
    assert np.asarray(g.indices[0].charges).tolist() == [0, 1]


def test_z2_pair_is_symmetric_and_blocked():
    A, B = xxz_z2_init_pair(D=2, key=jax.random.PRNGKey(0))
    for t in (A, B):
        assert isinstance(t, SymmetricTensor)
        assert [ix.label for ix in t.indices] == ["u", "d", "l", "r", "phys"]
        assert len(t.blocks) > 1
        _conserves(t)


def test_z2_pair_refuses_d1():
    with pytest.raises(ValueError, match="D must be >= 2"):
        xxz_z2_init_pair(D=1, key=jax.random.PRNGKey(0))
