"""U(1)-Sz arm prerequisites for the #566 feasibility spike."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair


@pytest.mark.parametrize("D,chi", [(2, 8), (3, 8)])
def test_u1sz_ctm_forward_runs(D, chi):
    """#605/#608: D>=3 U(1)-Sz CTM must not raise (the unfused-projector fix)."""
    A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    env, _trunc = ctm_tensor(A, chi=chi, max_iter=4, conv_tol=1e-4)
    for name in env._fields:
        t = getattr(env, name)
        assert np.all(np.isfinite(np.asarray(t._data))), f"{name} non-finite"


import os

from tenax import compute_energy_ctm_tensor  # re-exported from _ctm_tensor_energy


def _u1sz_energy(D, chi, stack: str):
    """CTM energy for a U(1)-Sz site with TENAX_STACK_BLOCKSPARSE = stack."""
    prev = os.environ.get("TENAX_STACK_BLOCKSPARSE")
    os.environ["TENAX_STACK_BLOCKSPARSE"] = stack
    try:
        jax.clear_caches()
        A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
        env, _ = ctm_tensor(A, chi=chi, max_iter=8, conv_tol=1e-6)
        gate = heisenberg_gate_u1sz()
        return float(compute_energy_ctm_tensor(A, env, gate))  # add d=2 if required
    finally:
        if prev is None:
            os.environ.pop("TENAX_STACK_BLOCKSPARSE", None)
        else:
            os.environ["TENAX_STACK_BLOCKSPARSE"] = prev


def test_stack_flag_energy_drift_is_bounded():
    """Quantify the flagged ~4.6e-4 stacked-core drift on the U(1)-Sz path."""
    e_off = _u1sz_energy(D=2, chi=8, stack="0")
    e_on = _u1sz_energy(D=2, chi=8, stack="1")
    drift = abs(e_on - e_off)
    print(f"\nU1Sz D=2 chi=8 energy drift |on-off| = {drift:.3e} "
          f"(off={e_off:.8f}, on={e_on:.8f})")
    assert drift < 1e-2, f"stacked drift {drift:.3e} too large to trust off/on grid"


def test_profiler_u1sz_arm_builds_symmetric_site_and_gate():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "profile_ctm_ad_wall_566",
        pathlib.Path(__file__).parent.parent / "examples" / "profile_ctm_ad_wall_566.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    site, gate = mod.make_site_and_gate("u1sz", D=2, seed=0)
    from tenax.core.tensor import SymmetricTensor
    assert isinstance(site, SymmetricTensor)
    assert len(site._block_keys) > 1
    assert isinstance(gate, SymmetricTensor)
