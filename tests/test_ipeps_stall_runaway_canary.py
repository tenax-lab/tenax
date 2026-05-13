"""Production canary for #454: a real 20-step Heisenberg D=2 χ=8 2-site
run should now see ≤ 3 stalls (was unbounded before the cap+rollback fix).
"""

import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.slow
def test_heisenberg_d2_chi8_stall_count_under_cap(capsys):
    """20-step D=2 χ=8 2-site Heisenberg AD; assert ≤ 3 stall events.

    No monkeypatch — exercises the real production line search and CTM.
    Before the cap+rollback fix, this scenario could log 18+ consecutive
    resets (issue #454). After the fix, the cap (default 5) caps total
    resets and rollback prevents the mathematical fixed point that
    caused the runaway.
    """
    d = 2
    D = 2
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    gate = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real

    rng = np.random.default_rng(42)
    A = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    B = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    A = A / jnp.linalg.norm(A)
    B = B / jnp.linalg.norm(B)

    cfg = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=D,
        ctm=CTMConfig(chi=8),
        gs_num_steps=20,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=5,
        gs_verbose=True,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _opt.optimize_gs_ad(gate, (A, B), cfg)

    out = capsys.readouterr().out
    stalls = re.findall(r"stall #(\d+)", out)
    n_stalls = max((int(s) for s in stalls), default=0)
    assert n_stalls <= 3, (
        f"expected ≤ 3 stalls on D=2 χ=8 production canary, got {n_stalls}. "
        f"This is a regression in stall recovery — investigate.\n"
        f"last 4000 chars of stdout:\n{out[-4000:]}"
    )
