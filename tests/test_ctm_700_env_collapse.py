"""Regression: U(1)-Sz single-site CTM env must not collapse to zero (#700).

PR #671's sorted-tail chi-leg tiling drove the D=3 U(1)-Sz ``ctm_tensor`` env to
exact zero (E=0) on the first absorption sweep at partial-tile chi (chi=12/14/16,
where chi - D**2 is neither 0 nor a full multiset), while leaving chi=10 and
chi=18 nonzero.  The collapse was silent (no error, just E=0).  The fix pads the
chi bonds beyond D**2 with the identity (charge-0) sector, which reproduces the
pre-regression energy and is chi-independent.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

# Pre-#671 reference energy for the raw D=3 init (issue #700).
_E_REF = -0.061722


@pytest.mark.parametrize("chi", [10, 12, 14, 16, 18])
def test_u1sz_env_does_not_collapse(chi):
    A, _ = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=chi, max_iter=30, conv_tol=1e-10)
    c1_norm = float(np.linalg.norm(np.asarray(env.C1._data)))
    assert c1_norm > 1e-6, f"chi={chi}: env collapsed to zero (|C1|={c1_norm})"
    e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert e == pytest.approx(_E_REF, abs=1e-4), f"chi={chi}: E={e} != {_E_REF}"
