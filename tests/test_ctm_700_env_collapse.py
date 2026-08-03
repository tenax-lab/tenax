"""Regression: U(1)-Sz single-site CTM env must not collapse to zero (#700).

PR #671's sorted-tail chi-leg tiling drove the D=3 U(1)-Sz ``ctm_tensor`` env to
exact zero (E=0) on the first absorption sweep at partial-tile chi (chi=12/14/16,
where chi - D**2 is neither 0 nor a full multiset), while leaving chi=10 and
chi=18 nonzero.  The collapse was silent (no error, just E=0).  The fix pads the
chi bonds beyond D**2 with the identity (charge-0) sector, which reproduces the
pre-regression energy and is chi-independent.

Updated for #723: ``ctm_tensor`` now defaults to ``recipe="2x2"``, so the
chi-frozen reference energy this test used to pin is gone — see the note below.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

# The former ``_E_REF = -0.061722`` pin is deliberately gone.  It was measured
# on the legacy ``recipe="1x1"`` path, which collapses the environment to
# rank-1 corners (#723/#726/#747), and it was a *chi-frozen* value: -0.061721808
# at chi=10 and bit-identical at chi=12, because a rank-1 corner is a chi_eff=1
# mean-field boundary that cannot respond to chi at all.  Pinning one energy
# across chi=10..18 only type-checked because the environment was broken.
#
# On the corrected 2x2 recipe the corner has rank 4 and the energy genuinely
# moves with chi (+0.112962196 at chi=10, +0.113096902 at chi=12), so a single
# pinned value is the wrong shape for this test.  A corrected reference should
# be re-pinned as part of the #747 re-run; until then this asserts what #700
# actually guards — the environment must not go to *exact zero* — plus the
# anti-collapse invariant that would have caught #723.


@pytest.mark.parametrize("chi", [10, 12, 14, 16, 18])
def test_u1sz_env_does_not_collapse(chi):
    A, _ = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=chi, max_iter=30, conv_tol=1e-10)
    c1_norm = float(np.linalg.norm(np.asarray(env.C1._data)))
    assert c1_norm > 1e-6, f"chi={chi}: env collapsed to zero (|C1|={c1_norm})"

    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    rank = int((s / (s[0] + 1e-300) > 1e-10).sum())
    assert rank > 1, f"chi={chi}: env collapsed to a rank-{rank} corner (#723)"

    e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert np.isfinite(e) and e != 0.0, f"chi={chi}: E={e}"
