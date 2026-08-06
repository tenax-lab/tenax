"""#792: the root-implicit path must not silently ignore a documented knob.

Four settings were accepted and then dropped, so the caller got a materially
different optimizer with no signal.  They do **not** all get the same
treatment, and the distinction is the point:

* ``gs_optimizer='cg'`` and ``gs_c4v`` are **opt-in** — you only get them by
  asking — so an unsatisfiable request is an error.
* ``gs_line_search`` is effectively **default-on** (``gs_optimizer`` defaults
  to ``'lbfgs'`` and ``gs_line_search=None`` means auto-on for lbfgs/cg), so
  raising would break every existing run, including the documented Path 5
  example.  It warns, exactly as ``gs_metric_precond`` already does on this
  path.
* ``return_history`` is cheap to honour and is a documented API, so it is
  *implemented* rather than refused.

That rule — refuse what was explicitly requested, warn about what arrived by
default — is what keeps the fix from breaking the path it is fixing.
"""

from __future__ import annotations

import warnings

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize_root_implicit import (
    optimize_gs_ad_root_implicit,
    validate_root_implicit_config,
)


def _cfg(**kw):
    base = dict(
        max_bond_dim=2,
        unit_cell="1x1",
        su_init=False,
        gs_num_steps=1,
        gs_metric_precond=False,
        ctm=CTMConfig(chi=4, max_iter=20, conv_tol=1e-10, ctm_ad_mode="root_implicit"),
    )
    base.update(kw)
    return iPEPSConfig(**base)


# --- opt-in knobs: refuse ------------------------------------------------


def test_cg_is_refused_rather_than_run_as_gradient_descent():
    """``_build_optimizer`` returns None for 'cg' and the loop then does plain
    ``params - lr * grad``: no retained previous gradient, no Polak-Ribiere
    coefficient, no conjugate direction.  Asking for CG and getting gradient
    descent is a materially different optimizer.
    """
    with pytest.raises(NotImplementedError, match="gs_optimizer"):
        validate_root_implicit_config(_cfg(gs_optimizer="cg"))


def test_c4v_is_refused_rather_than_silently_unenforced():
    """Nothing on this path projects into the C4v basis, so the first update
    can leave the requested symmetry sector while the caller believes it is
    constrained.
    """
    with pytest.raises(NotImplementedError, match="gs_c4v"):
        validate_root_implicit_config(_cfg(gs_c4v=True))


def test_the_error_still_names_every_offending_knob_at_once():
    """A caller fixing one knob at a time round-trips once per knob."""
    with pytest.raises(NotImplementedError) as exc:
        validate_root_implicit_config(_cfg(gs_optimizer="cg", gs_c4v=True))
    msg = str(exc.value)
    assert "gs_optimizer" in msg and "gs_c4v" in msg, msg


def test_the_default_optimizer_is_not_refused():
    """The guard must not reject the path's own default configuration.

    ``gs_optimizer='lbfgs'`` with ``gs_line_search=None`` is what every
    existing root-implicit run uses, the documented Path 5 example included.
    """
    validate_root_implicit_config(_cfg())  # must not raise


# --- default-on knob: warn ------------------------------------------------


def test_an_effective_line_search_warns_that_steps_are_unconditional():
    """``_use_line_search`` is True by default for lbfgs, but no line search
    runs: ``_build_optimizer`` returns a plain ``scale_by_lbfgs`` chain, and
    ``value_fn=`` is only consumed by ``optax.lbfgs(linesearch=...)``.

    So the default takes unconditional quasi-Newton steps and can accept
    energy-increasing ones.  The pre-existing in-code comment reads as though
    a line search is running; it is not.
    """
    cfg = _cfg(
        unit_cell="2site"
    )  # "cell" variant: refused after the warning is emitted
    with pytest.warns(UserWarning, match="line search"):
        with pytest.raises(NotImplementedError):
            optimize_gs_ad_root_implicit(None, None, cfg)


def test_disabling_the_line_search_is_silent():
    """Without this, the warning could fire unconditionally and be noise.

    The documented Path 5 example sets ``gs_line_search=False``, so that
    configuration in particular must stay quiet.
    """
    cfg = _cfg(unit_cell="2site", gs_line_search=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError):
            optimize_gs_ad_root_implicit(None, None, cfg)
    bad = [str(w.message) for w in rec if "line search" in str(w.message)]
    assert not bad, bad


# --- documented API: implement --------------------------------------------


@pytest.mark.slow
def test_return_history_yields_the_documented_four_tuple():
    """``return_history`` was accepted and dropped, so callers of the
    documented 1x1 history API got a 3-tuple and lost the trajectory.

    The loop already computes every field, so this is honoured rather than
    refused.  ``jit_compile_time`` is 0.0 by construction: this path drives
    eager per-step solves and has no single compile event to attribute -- the
    key is present so consumers do not have to special-case the path.
    """
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

    cfg = _cfg(gs_num_steps=2, gs_line_search=False, return_history=True)
    out = optimize_gs_ad_root_implicit(H, A, cfg)

    assert len(out) == 4, f"expected a 4-tuple with return_history, got {len(out)}"
    hist = out[3]
    assert set(hist) == {
        "energies",
        "step_times",
        "jit_compile_time",
        "num_steps",
        "converged",
    }, sorted(hist)
    assert hist["num_steps"] == len(hist["energies"])
    assert len(hist["step_times"]) == len(hist["energies"])
    assert hist["energies"], "no energies recorded"
    assert all(isinstance(e, float) for e in hist["energies"])


def test_history_is_absent_by_default():
    """Default must keep the 3-tuple: this is a shape change on an opt-in flag.

    Cheap -- gs_num_steps=0 short-circuits before any root solve.
    """
    cfg = _cfg(gs_num_steps=0, gs_line_search=False)
    assert cfg.return_history is False
