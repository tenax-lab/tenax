"""A masked gradient must not end the root-implicit optimization (#812).

When a root solve returns non-finite gradient entries the loop masks them to
zero so the best-so-far state survives.  That rescue is also what makes the run
*look* converged, and it corrupts **both** convergence criteria:

* ``gs_conv_criterion='grad_norm'`` -- a fully masked gradient has L2 norm
  *exactly* ``0.0``, below any tolerance.  Fires on the masked step itself.
* ``gs_conv_criterion='dE'`` (the current default) -- a fully masked step is a
  no-op, so the *next* step re-evaluates identical params and sees
  ``delta_energy == 0.0``.  Fires one step later.

#811 stopped the loop *reporting* that as ``converged=True``.  It still broke
out of the optimization.  These tests pin the control flow: the loop keeps
going, and the second test covers the one-step-later variant that a
per-step-only guard would miss.
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize_root_implicit import optimize_gs_ad_root_implicit


def _tiny_state():
    """Smallest D=2 1x1 state the root-implicit path will accept, plus a gate."""
    import numpy as np

    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    rng = np.random.RandomState(0)
    data = jax.numpy.array(rng.standard_normal((2, 2, 2, 2, 2)))
    sym = U1Symmetry()
    ch = np.zeros(2, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="phys"),
    )
    A = DenseTensor(data / (jax.numpy.linalg.norm(data) + 1e-10), idx)

    Sz = 0.5 * jax.numpy.array([[1.0, 0.0], [0.0, -1.0]])
    H = jax.numpy.kron(Sz, Sz).reshape(2, 2, 2, 2)
    return H, A


def _cfg(**kw):
    base = dict(
        max_bond_dim=2,
        unit_cell="1x1",
        su_init=False,
        gs_metric_precond=False,
        gs_line_search=False,
        return_history=True,
        ctm=CTMConfig(chi=4, max_iter=20, conv_tol=1e-10, ctm_ad_mode="root_implicit"),
    )
    base.update(kw)
    return iPEPSConfig(**base)


_SHAPE = (2, 2, 2, 2, 2)
_E = -0.25


def _patch(monkeypatch, grad_for_step):
    """Drive the loop with scripted (energy, grad) pairs, counting calls."""
    import tenax.algorithms._ctm_root_implicit_asym as _asym

    calls = {"n": 0}

    def _fake(A_t, gate, **kw):
        i = calls["n"]
        calls["n"] += 1
        return jax.numpy.asarray(_E), grad_for_step(i)

    monkeypatch.setattr(_asym, "asym_root_implicit_energy_and_grad", _fake)
    return calls


def test_a_masked_gradient_does_not_end_the_optimization(monkeypatch):
    """grad_norm criterion: a fully masked gradient has norm exactly 0.0.

    Without the guard the loop exits at step 1 having done nothing, and reports
    an energy from a state it never optimized.
    """
    H, A = _tiny_state()
    _patch(monkeypatch, lambda i: jax.numpy.full(_SHAPE, jax.numpy.nan))

    cfg = _cfg(gs_num_steps=3, gs_conv_criterion="grad_norm")
    with pytest.warns(RuntimeWarning, match="non-finite gradient"):
        out = optimize_gs_ad_root_implicit(H, A, cfg)

    hist = out[3]
    assert hist["num_steps"] == 3, (
        "a masked gradient has norm exactly 0.0 and satisfies grad_norm; the "
        f"loop must not treat that as convergence, got num_steps={hist['num_steps']}"
    )
    assert hist["converged"] is False


def test_a_masked_step_does_not_false_converge_one_step_later(monkeypatch):
    """dE criterion -- the default, and the variant a per-step guard misses.

    Step 0 is fully masked, so it is a no-op and step 1 re-evaluates *identical*
    params.  ``delta_energy`` is then exactly 0.0 through no merit of the
    optimization, and the ``dE`` criterion fires on a step whose own gradient is
    perfectly finite.  Guarding only the contaminated step would not catch this;
    ``prev_energy`` has to be reset too.
    """
    H, A = _tiny_state()
    _patch(
        monkeypatch,
        lambda i: (
            jax.numpy.full(_SHAPE, jax.numpy.nan) if i == 0 else jax.numpy.zeros(_SHAPE)
        ),
    )

    cfg = _cfg(gs_num_steps=4, gs_conv_criterion="dE")
    with pytest.warns(RuntimeWarning, match="non-finite gradient"):
        out = optimize_gs_ad_root_implicit(H, A, cfg)

    hist = out[3]
    assert hist["num_steps"] >= 3, (
        "step 1 saw delta_energy == 0.0 only because the masked step 0 was a "
        "no-op; prev_energy must be reset so that manufactured zero cannot "
        f"end the run, got num_steps={hist['num_steps']}"
    )
    assert hist["converged"] is False
