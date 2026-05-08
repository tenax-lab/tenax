"""End-to-end integration test for variPEPS §2.8.2 auto-χ_E bump.

Lives in its own file (mapped ``"algorithm"`` in ``conftest.py``) because
``optimize_gs_ad`` is JAX-JIT-heavy and the test takes ~3 minutes; it
should not slow down ``pytest -m core`` (the CI required-check set).
"""

from __future__ import annotations

import math

import jax

from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad


def test_optimize_gs_ad_auto_bump_raises_chi_under_pressure():
    """Heisenberg D=2 at chi=2: auto-bump must raise chi during optimization.

    Asserts:
        - Returned env's corner χ axis is strictly larger than the initial chi=2.
        - Final energy is finite and a Python float.

    Why this works: chi=2 with D=2 (-> chi_eff = D²×chi = 8 needed for full
    information). The CTM SVD projector's cross-product M = C1g^H @ C4g is
    chi×chi and cannot measure discarded modes.  To get a meaningful eps_T > 0,
    _update_env_cache runs one non-JIT eigh sweep (rho = C1g @ C1g^H + C4g @
    C4g^H is chi*D² × chi*D² = 8×8) and discards 6 eigenmodes -> eps_T >> 1e-5,
    triggering the bump on the first L-BFGS step.
    """
    gate = heisenberg_gate()
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    A_init = jax.random.normal(k1, (2, 2, 2, 2, 2)) + 1j * jax.random.normal(
        k2, (2, 2, 2, 2, 2)
    )

    cfg = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=2,
            chi_auto_bump=True,
            chi_auto_bump_eps=1e-5,
            chi_auto_bump_step=2,
            chi_max=8,
            max_iter=10,
            min_iter=2,
            conv_tol=1e-3,
        ),
        gs_num_steps=3,
        gs_implicit_ad=True,
        gs_verbose=False,
        su_init=False,
    )

    A_opt, env, e_opt = optimize_gs_ad(gate, A_init, cfg)

    # env is a CTMTensorEnv (NamedTuple of DenseTensor); corners are (chi, chi).
    final_chi = env.C1._data.shape[0]

    assert final_chi > 2, f"auto-bump never fired (final_chi={final_chi})"
    assert final_chi <= 8, f"chi exceeded chi_max=8 (final_chi={final_chi})"
    assert math.isfinite(float(e_opt)), f"final energy is not finite: {e_opt}"
