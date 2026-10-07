"""1-site metric L-BFGS / CG read the current iterate, not the initial tensor.

The 1-site optimizer starts from ``params = A`` and only ``params`` moves.
The metric branches read ``A``: L-BFGS built ``s = A - A_prev`` from it, so
``s = 0`` at every step and no curvature pair was stored (the "L-BFGS" was
metric-preconditioned steepest descent), and ``precondition_gradient`` built
the metric ``<A|N|A>`` of the initial tensor.  The 2-site and multisite
paths read the current params.

Mechanism tests: spies record the L-BFGS history length and the tensor the
metric is built from at each step.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._metric_precond as _mp
from tenax import CTMConfig, heisenberg_gate, iPEPSConfig, optimize_gs_ad

# χ=4 does not always reach conv_tol; the spies, not convergence, are tested.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NUM_STEPS = 4


def _rand(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal((2, 2, 2, 2, 2))
    return jnp.asarray(a / np.linalg.norm(a))


def _run(monkeypatch, optimizer):
    hist_lens, metric_As = [], []
    real_two_loop = _mp.lbfgs_two_loop
    real_precond = _mp.precondition_gradient

    def spy_two_loop(grad, history, h0_matvec):
        hist_lens.append(len(history))
        return real_two_loop(grad, history, h0_matvec)

    def spy_precond(A, env, grad, delta, config):
        metric_As.append(np.asarray(A.todense()))
        return real_precond(A, env, grad, delta, config)

    monkeypatch.setattr(_mp, "lbfgs_two_loop", spy_two_loop)
    monkeypatch.setattr(_mp, "precondition_gradient", spy_precond)
    cfg = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4, max_iter=300, min_iter=1, conv_tol=1e-8, on_unconverged="warn"
        ),
        gs_num_steps=NUM_STEPS,
        gs_optimizer=optimizer,
        gs_metric_precond=True,
        gs_line_search_method="hager_zhang",
        su_init=False,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-12,  # run every step
    )
    A0 = _rand()
    optimize_gs_ad(heisenberg_gate(), A0, cfg)
    return hist_lens, metric_As, np.asarray(A0)


def test_lbfgs_stores_curvature_pairs(monkeypatch):
    hist_lens, _, _ = _run(monkeypatch, "lbfgs")
    # The regime: the optimizer took every step through the two-loop.
    assert len(hist_lens) == NUM_STEPS, hist_lens
    assert hist_lens[0] == 0
    # Every later step has at least one (s, y) pair.
    assert all(n >= 1 for n in hist_lens[1:]), hist_lens


@pytest.mark.parametrize("optimizer", ["lbfgs", "cg"])
def test_metric_is_built_at_the_current_iterate(monkeypatch, optimizer):
    _, metric_As, A0 = _run(monkeypatch, optimizer)
    assert len(metric_As) >= 2, len(metric_As)
    # The first metric is at the start point; every later one has moved off it.
    assert np.allclose(metric_As[0], A0)
    for k, A in enumerate(metric_As[1:], start=1):
        assert not np.allclose(A, A0), f"metric call {k} still uses the initial A"
