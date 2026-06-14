"""Tests for U(1)-Sz–symmetric Heisenberg helpers (issue #570 follow-up)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import heisenberg_gate
from tenax.algorithms.ipeps import heisenberg_gate_u1sz
from tenax.core.tensor import SymmetricTensor


class TestHeisenbergGateU1Sz:
    def test_returns_symmetric_tensor(self):
        gate = heisenberg_gate_u1sz()
        assert isinstance(gate, SymmetricTensor)

    def test_dense_roundtrip_matches_plain_gate(self):
        """Same physics as heisenberg_gate(); only the charges differ.

        from_dense only reorders the same float values into blocks (no
        arithmetic), so the dense round-trip is bit-for-bit identical.
        """
        gate_u1 = heisenberg_gate_u1sz()
        gate_plain = heisenberg_gate()
        np.testing.assert_array_equal(
            np.asarray(gate_u1.todense()),
            np.asarray(gate_plain.todense()),
        )

    def test_physical_charges_are_sz(self):
        """Physical legs carry Sz charges [+1, -1] (units of 2*Sz)."""
        gate = heisenberg_gate_u1sz()
        charges = np.asarray(gate.indices[0].charges)
        assert sorted(charges.tolist()) == [-1, 1]

    def test_is_nontrivially_blocked(self):
        """Sz conservation splits H into more than one charge block."""
        gate = heisenberg_gate_u1sz()
        assert len(gate.blocks) > 1

    def test_blocks_conserve_sz(self):
        """Every block conserves Sz: flow-signed charges sum to zero.

        This tests the symmetry property itself (not just that >1 block
        exists): for each block key, sum_leg(flow_sign * charge) == 0
        with flows IN=+1, OUT=-1.
        """
        gate = heisenberg_gate_u1sz()
        signs = [int(ix.flow.value) for ix in gate.indices]
        for key in gate.blocks:
            assert sum(s * q for s, q in zip(signs, key)) == 0


class TestHeisenbergU1SzInit:
    def test_pair_are_symmetric_tensors(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        assert isinstance(A, SymmetricTensor)
        assert isinstance(B, SymmetricTensor)

    def test_pair_have_five_legs_and_nontrivial_blocks(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        assert len(A.indices) == 5  # u, d, l, r, phys
        assert len(A.blocks) > 1    # non-trivially blocked -> exercises absorb step
        assert len(B.blocks) > 1

    def test_physical_leg_is_sz_charged(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, _ = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        phys = np.asarray(A.indices[4].charges)  # phys is the 5th leg
        assert sorted(phys.tolist()) == [-1, 1]

    def test_blocks_conserve_sz(self):
        """Every block of each site tensor conserves Sz (flow-signed sum 0)."""
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        for tensor in (A, B):
            signs = [int(ix.flow.value) for ix in tensor.indices]
            for key in tensor.blocks:
                assert sum(s * q for s, q in zip(signs, key)) == 0

    def test_rejects_d_below_2(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        with pytest.raises(ValueError, match="D must be >= 2"):
            heisenberg_u1sz_init_pair(D=1, key=jax.random.PRNGKey(0))
