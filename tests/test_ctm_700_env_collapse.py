"""Regression: the U(1)-Sz single-site CTM env must not collapse (#700, #723).

#700: PR #671's sorted-tail chi-leg padding drove the D=3 U(1)-Sz ``ctm_tensor``
env to exact zero on the first sweep at partial-tile chi (12/14/16).  That
defect lives in ``_tile_fused_to_chi`` and is pinned there, directly, by
``tests/test_ctm_tensor_tiling.py::test_tile_pads_with_vacuum_not_asymmetric_sectors``
(0.1 s).  This end-to-end test can no longer see it: since #723 made
``recipe="2x2"`` the default, the first sweep's projector renormalises the
padded sectors away before the edges ever meet them.  Measured with the
sorted-tail padding restored (2026-09-27): E and |C1| bit-identical to the fixed
code at chi=10..18 after 1 and 2 sweeps, and still passing all of the old
assertions after the full 30 sweeps at chi=12.

#723: the legacy ``recipe="1x1"`` collapses the environment to a rank-1 corner
(a chi_eff=1 mean-field boundary).  That is what this test guards.  One sweep is
enough: the 1x1 recipe gives rank 1 at every chi in 10..18 after the first
sweep, the 2x2 recipe rank D**2 = 9.  The old test ran 30 sweeps at five chi
values; the block-sparse sectors grow every sweep and each new block shape is a
fresh XLA compile (77% of the wall time), so it took ~2.5 h per chi on CI and
timed its slow shard out at the 240 min job cap on every run.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

CHI = 12  # partial tile (chi - D**2 = 3), the #700 regime


def test_u1sz_env_does_not_collapse():
    A, _ = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=CHI, max_iter=1, conv_tol=1e-10)
    c1_norm = float(np.linalg.norm(np.asarray(env.C1._data)))
    assert c1_norm > 1e-6, f"env collapsed to zero (|C1|={c1_norm})"

    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    rank = int((s / (s[0] + 1e-300) > 1e-10).sum())
    assert rank > 1, f"env collapsed to a rank-{rank} corner (#723)"
