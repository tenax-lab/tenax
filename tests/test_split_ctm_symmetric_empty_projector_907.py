"""#907: a confined symmetric CTM environment made ``ctm_split_tensor_2site``
raise a bare ``StopIteration`` four frames down (``next(iter(U_T.blocks))`` on
an empty SVD output), and -- once that was hardened -- an ``IndexError`` one
line later (``S[0]`` on a size-0 spectrum).

The fix has two parts, tested here:

1. :func:`_gauge_fix_symmetric_svd` reads its dtype off the tensor rather than
   a sample block, so it is a clean no-op on an empty projector instead of
   crashing.  (Unit test; the mutation anchor for reverting to ``next(iter)``.)

2. :func:`_compute_2x2_projector_symmetric` guards every projector SVD: an empty
   (rank-0) bond -- the confined-environment limitation of #905 -- raises a
   named diagnosis rather than letting an empty projector flow into the sweep
   and certify a wrong environment (#907).  The dense path densifies and never
   hits this, which the integration test pins as the contrast.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_projector_2x2 import (
    _gauge_fix_symmetric_svd,
    _require_svd_connected,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor
from tenax.linalg import svd as tensor_svd

# Bucket is assigned by ``tests/conftest.py`` (registered ``core``); the
# end-to-end integration case below carries its own ``@pytest.mark.slow``,
# which the conftest rule honours by withholding ``core`` from it.


def _empty_svd_outputs():
    """A genuinely empty SVD: left charge {0} vs right charge {1} share no
    sector, so ``M`` has no blocks and ``U``/``Vh`` come back with an empty
    bond -- the exact shape the confined environment produced in the wild.
    """
    sym = U1Symmetry()
    left = TensorIndex.from_charges(
        sym, np.array([0], dtype=np.int32), FlowDirection.IN, label="l"
    )
    right = TensorIndex.from_charges(
        sym, np.array([1], dtype=np.int32), FlowDirection.OUT, label="r"
    )
    # All-zero data: with no shared charge sector every element is forbidden,
    # so zeros is the only value ``from_dense`` accepts and M comes out empty.
    M = SymmetricTensor.from_dense(jnp.zeros((1, 1)), (left, right))
    assert not M.blocks, "control: M must be empty (no shared charge sector)"
    U, S, Vh, _ = tensor_svd(
        M, left_labels=("l",), right_labels=("r",), new_bond_label="bond"
    )
    return M, U, S, Vh


def test_gauge_fix_symmetric_svd_passes_empty_through_without_crashing():
    """Part 1: the dtype detection no longer needs a sample block.

    Reverting line 99 to ``next(iter(U_T.blocks.values()))`` makes this raise
    ``StopIteration`` -- so this is the mutation anchor for the dtype fix.
    """
    _M, U, _S, Vh = _empty_svd_outputs()
    assert not U.blocks and not Vh.blocks  # regime: genuinely empty

    U_out, Vh_out = _gauge_fix_symmetric_svd(U, Vh)  # must not raise

    assert not U_out.blocks and not Vh_out.blocks
    assert U_out.labels() == U.labels()
    assert Vh_out.labels() == Vh.labels()


def test_require_svd_connected_raises_on_empty_bond():
    """Part 2: an empty projector SVD raises a named diagnosis, not silence."""
    M, U, _S, _Vh = _empty_svd_outputs()
    with pytest.raises(ValueError) as exc:
        _require_svd_connected(
            M,
            U,
            left_labels=("l",),
            right_labels=("r",),
            direction="left",
            matrix_name="M_test",
        )
    msg = str(exc.value)
    assert "rank-0" in msg
    assert "#907" in msg and "#905" in msg
    assert "M_test" in msg


def test_require_svd_connected_is_a_noop_when_the_bond_is_populated():
    """The guard must not fire on a healthy projector -- else it would break
    every working symmetric CTM.  A charge-conserving M has a non-empty SVD.
    """
    sym = U1Symmetry()
    left = TensorIndex.from_charges(
        sym, np.array([0, 1], dtype=np.int32), FlowDirection.IN, label="l"
    )
    right = TensorIndex.from_charges(
        sym, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="r"
    )
    M = SymmetricTensor.from_dense(jnp.eye(2), (left, right))
    U, _S, _Vh, _ = tensor_svd(
        M, left_labels=("l",), right_labels=("r",), new_bond_label="bond"
    )
    assert U.blocks, "control: this M must have a populated SVD bond"
    # Returns None, does not raise.
    assert (
        _require_svd_connected(
            M,
            U,
            left_labels=("l",),
            right_labels=("r",),
            direction="left",
            matrix_name="M_healthy",
        )
        is None
    )


def _confined_symmetric_pair(steps: int = 100):
    """Build the #907 state: a U(1)-Sz Heisenberg pair evolved by simple update
    until its symmetric CTM environment is confined to a charge sector.
    """
    from tenax.algorithms.ipeps import (
        heisenberg_gate_u1sz,
        heisenberg_u1sz_init_pair,
    )
    from tenax.algorithms.ipeps_simple_update import (
        _make_trotter_gate_tensor,
    )
    from tenax.algorithms.ipeps_simple_update import (
        _simple_update_2site_horizontal_tensor as suh,
    )
    from tenax.algorithms.ipeps_simple_update import (
        _simple_update_2site_vertical_tensor as suv,
    )

    D = 2
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    gate = _make_trotter_gate_tensor(heisenberg_gate_u1sz(), 0.05, site_tensor=A)
    lam_h = jnp.ones(D)
    lam_v = jnp.ones(D)
    for step in range(steps):
        if step % 2 == 0:
            A, B, lam_h = suh(A, B, gate, lam_h, lam_v, D)
        else:
            A, B, lam_v = suv(A, B, gate, lam_h, lam_v, D)
    return A, B


@pytest.mark.slow
def test_symmetric_split_2site_raises_meaningfully_on_confined_env():
    """Integration: symmetric input raises a #907 diagnosis; the densified
    *same* state runs, so the failure is symmetric-path-specific (#905), not a
    bad state.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        ctm_split_tensor_2site,
    )
    from tenax.core.tensor import DenseTensor

    A, B = _confined_symmetric_pair()

    with pytest.raises(ValueError) as exc:
        ctm_split_tensor_2site(A, B, chi=8, max_iter=3, conv_tol=1e-10, recipe="2x2")
    msg = str(exc.value)
    assert "rank-0" in msg and "#907" in msg
    # No bare StopIteration / IndexError leaking through.
    assert "projector" in msg.lower()

    # The densified same state must succeed -- proves it is symmetric-specific.
    Ad = DenseTensor(A.todense(), A.indices)
    Bd = DenseTensor(B.todense(), B.indices)
    env, _info = ctm_split_tensor_2site(
        Ad, Bd, chi=8, max_iter=3, conv_tol=1e-10, recipe="2x2"
    )
    assert env is not None
