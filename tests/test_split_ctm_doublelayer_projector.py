import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._split_ctm_tensor_energy import _split_env_to_tensor_standard

pytestmark = pytest.mark.core


def _oracle():
    # import helper module robustly whether or not `tests` is a package
    try:
        from tests._split_ctm_oracle import (
            fused_env_to_split,
            heisenberg_gate,
            make_site,
        )
    except ModuleNotFoundError:
        from _split_ctm_oracle import fused_env_to_split, heisenberg_gate, make_site
    return make_site, heisenberg_gate, fused_env_to_split


@pytest.mark.parametrize("D", [2, 3])
def test_fused_to_split_roundtrip(D):
    make_site, heisenberg_gate, fused_env_to_split = _oracle()
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()
    fused_env, _ = ctm_tensor(A, chi=8, max_iter=200, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor(A, fused_env, gate))
    split_env = fused_env_to_split(fused_env, D, chi_I=8 * D)
    rt_env = _split_env_to_tensor_standard(split_env)
    E_rt = float(compute_energy_ctm_tensor(A, rt_env, gate))
    np.testing.assert_allclose(E_rt, E_fused, atol=1e-8)
