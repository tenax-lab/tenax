"""Faithfulness guard for the #610 C-lever prototype (Stage 2 prereq)."""
import jax
import numpy as np

from examples.u1sz_defrag_prototype_610 import sector_dropping_truncation
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair


def test_prototype_ctm_converges_and_energy_is_sane():
    jax.config.update("jax_enable_x64", True)
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_truncation(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7)
        for name in env._fields:
            t = getattr(env, name)
            assert np.all(np.isfinite(np.asarray(t._data))), f"{name} non-finite"
        e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert np.isfinite(e), "prototype energy non-finite"
    assert -2.0 < e < 0.0, f"prototype energy {e} outside sane Heisenberg window"


def test_prototype_actually_drops_sectors():
    # Under the prototype, the env block counts must drop toward the Gate-A
    # static prediction: corners 5->3, edges 19->9 at D=3 chi=12.
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_truncation(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, max_iter=8, conv_tol=1e-6)
    nblocks = {n: len(getattr(env, n)._block_keys) for n in env._fields}
    # edges must have strictly fewer blocks than the fragmented baseline (19)
    assert nblocks["T1"] < 19, f"edge sectors not dropped: {nblocks}"
    assert nblocks["C1"] <= 5, f"corner sectors not dropped: {nblocks}"
