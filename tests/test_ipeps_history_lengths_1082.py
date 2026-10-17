"""#1082: history lists stay aligned after a gradient-spike rollback.

The 1-site loop appended its CTM-forward lists (``ctm_converged``,
``ctm_sv_diff``) for every step but ``continue``d out of the spike branch
before appending the energy, so ``energies`` drifted one short per rollback
(3 vs 4 on a 4-step run).  The 2-site loop records the spiked step.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

jax.config.update("jax_enable_x64", True)


def _gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    g = (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    )
    return g.real


def _rand(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal((2, 2, 2, 2, 2))
    return jnp.asarray(a / np.linalg.norm(a))


@pytest.mark.parametrize("uc", ["1x1", "2site"])
def test_history_lists_aligned_after_spike_rollback(uc, monkeypatch):
    real = _opt._grad_l2_norm
    calls = {"n": 0}

    def fake(g):
        calls["n"] += 1
        v = real(g)
        return 1e6 if calls["n"] == 2 else v

    monkeypatch.setattr(_opt, "_grad_l2_norm", fake)
    cfg = iPEPSConfig(
        unit_cell=uc,
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4, max_iter=40, min_iter=1, conv_tol=1e-14, on_unconverged="warn"
        ),
        gs_num_steps=4,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=2,
        su_init=False,
        gs_conv_criterion="grad_norm",
        return_history=True,
        gs_grad_spike_ratio=2.0,
    )
    init = (_rand(0), _rand(1)) if uc == "2site" else _rand(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _opt.optimize_gs_ad(_gate(), init, cfg)
    h = out[-1]
    assert calls["n"] >= 2  # the spike was actually injected
    assert len(h["energies"]) == len(h["ctm_converged"]) == len(h["ctm_sv_diff"])
