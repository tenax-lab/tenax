"""End-to-end block-sparse CTM integration tests.

Verifies that the CTM forward on a U(1) SymmetricTensor iPEPS stays
block-sparse throughout and that its energy matches the dense path.  (The
slow trivial-charge implicit-AD gradient/energy checks were deleted in the
slow-bucket prune: symmetric implicit-AD gradients are pinned on real charged
states by test_symmetric_2site_ad_gradient.py and test_joint_chain_rule.py.)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_tensor import (
    CTMTensorEnv,
    compute_energy_ctm_tensor,
    ctm_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _make_u1_symmetric_ipeps(seed=42):
    """Create a U(1)-symmetric iPEPS tensor with trivial charges, D=2, d=2.

    Uses trivial charges (all zeros) so that the SymmetricTensor has a single
    block containing all elements -- this ensures the energy matches the dense
    path exactly while still exercising the full block-sparse code path.
    """
    D, d = 2, 2
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    rng = np.random.RandomState(seed)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
    data = data / jnp.linalg.norm(data)
    return SymmetricTensor.from_dense(data, indices)


def _make_dense_from_symmetric(A_sym):
    """Create a DenseTensor with the same data and indices as a SymmetricTensor."""
    return DenseTensor(A_sym.todense(), A_sym.indices)


# ------------------------------------------------------------------ #
# Test 1: CTM convergence stays block-sparse                          #
# ------------------------------------------------------------------ #


class TestBlockSparseCTMForward:
    def test_u1_ctm_env_stays_symmetric_tensor(self):
        """CTM convergence with U(1) SymmetricTensor preserves block-sparse type."""
        A_sym = _make_u1_symmetric_ipeps()
        chi = 4

        env, _ = ctm_tensor(A_sym, chi=chi, max_iter=30, conv_tol=1e-6)

        assert isinstance(env, CTMTensorEnv)
        for field in env:
            assert isinstance(field, SymmetricTensor), (
                f"Environment tensor {field.labels()} should be SymmetricTensor, "
                f"got {type(field).__name__}"
            )
            assert jnp.all(jnp.isfinite(field.todense())), (
                f"Non-finite values in env tensor {field.labels()}"
            )

    # The "energy < 0" assertion on a random U(1) iPEPS was removed:
    # CTM gives the environment of the supplied A, not an optimized A,
    # and a random Gaussian iPEPS has no reason to land at AFM-sign
    # energy. The U(1) forward path is covered by
    # test_u1_ctm_env_stays_symmetric_tensor (block-sparse type
    # preservation + finiteness) and the production AD trifecta
    # benchmarks (D=3 chi=16 E=-0.6521).


# ------------------------------------------------------------------ #
# Test 3: Dense vs block-sparse energy match                          #
# ------------------------------------------------------------------ #


class TestDenseVsBlockSparseMatch:
    def test_ctm_forward_energy_match(self):
        """Forward CTM energy from dense and block-sparse paths should match."""
        A_sym = _make_u1_symmetric_ipeps(seed=99)
        A_dense = _make_dense_from_symmetric(A_sym)
        gate = heisenberg_gate()
        chi = 6

        env_sym, _ = ctm_tensor(A_sym, chi=chi, max_iter=40, conv_tol=1e-8)
        E_sym = float(compute_energy_ctm_tensor(A_sym, env_sym, gate, d=2))

        env_dense, _ = ctm_tensor(A_dense, chi=chi, max_iter=40, conv_tol=1e-8)
        E_dense = float(compute_energy_ctm_tensor(A_dense, env_dense, gate, d=2))

        np.testing.assert_allclose(
            E_sym,
            E_dense,
            atol=1e-4,
            err_msg=f"Block-sparse energy {E_sym} != dense energy {E_dense}",
        )
