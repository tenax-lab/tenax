"""``gs_conv_criterion`` defaults to ``"grad_norm"`` (v0.8.4, issue #448).

The legacy ``"dE"`` default, ``|E_k - E_{k-1}| < gs_conv_tol``, declared
convergence with ``||grad E||`` between 1e-2 and 0.69 in all four D=2/3
square-Heisenberg validation runs: a tiny ``dE`` happens whenever the line
search barely moves, or right after a rollback / noise injection.

These tests pin the *mechanism*, not a physical convergence: the CTM energy
function is replaced by a scripted one whose value and gradient are set
independently, so "energy repeats while the gradient stays large" is exact
rather than hoped for.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad
from tenax.algorithms import ipeps_ad_policy

_E_CONST = -0.5


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _script_energy(monkeypatch, grad_scale: float):
    """Replace the CTM energy with ``E == _E_CONST`` and ``|grad| ~ grad_scale``.

    ``s - stop_gradient(s)`` is identically zero in value but carries the
    gradient of ``s``, so the energy repeats *exactly* on every step (dE == 0
    from step 1 on) while the gradient norm is whatever ``grad_scale`` makes
    it.  ``W`` is a fixed random tensor so the gradient is not parallel to the
    (normalised) parameters and survives the projection onto the sphere.
    """
    calls = {"n": 0}

    def _factory(**_kw):
        def _energy(site_tensors):
            calls["n"] += 1
            s = 0.0
            for i, key in enumerate(sorted(site_tensors)):
                a = site_tensors[key].todense()
                w = jax.random.normal(jax.random.PRNGKey(11 + i), a.shape)
                s = s + jnp.real(jnp.sum(a * w))
            s = grad_scale * s
            return _E_CONST + (s - jax.lax.stop_gradient(s))

        return _energy

    monkeypatch.setattr(ipeps_ad_policy, "make_ctm_energy_fn", _factory)
    return calls


def _cfg(unit_cell: str, **overrides) -> iPEPSConfig:
    base = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=4),
        unit_cell=unit_cell,
        gs_num_steps=5,
        gs_learning_rate=1e-3,
        gs_optimizer="adam",
        gs_line_search=False,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        su_init=False,
        return_history=True,
    )
    return replace(base, **overrides)


def _run(unit_cell: str, cfg: iPEPSConfig) -> dict:
    out = optimize_gs_ad(_heisenberg_gate(), None, cfg)
    return out[-1]


# --- (a) the default -------------------------------------------------------


def test_default_criterion_is_grad_norm():
    assert iPEPSConfig().gs_conv_criterion == "grad_norm"
    assert iPEPSConfig().gs_grad_norm_tol == 1e-5


# --- (d) warnings ----------------------------------------------------------


def test_default_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        iPEPSConfig()
        iPEPSConfig(max_bond_dim=3, gs_num_steps=10)


def test_explicit_dE_still_selectable_and_warns():
    with pytest.warns(DeprecationWarning, match="gs_conv_criterion='dE'") as rec:
        cfg = iPEPSConfig(gs_conv_criterion="dE")
    assert cfg.gs_conv_criterion == "dE"
    assert "before v0.8.4" in str(rec[0].message)


# --- (b) dE == 0 with a large gradient does not converge -------------------


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_default_ignores_zero_dE_when_gradient_is_large(monkeypatch, unit_cell):
    """dE == 0 exactly from step 1, |g| ~ O(1): the default must keep going."""
    _script_energy(monkeypatch, grad_scale=1.0)
    cfg = _cfg(unit_cell)
    assert cfg.gs_conv_criterion == "grad_norm"  # the default under test
    hist = _run(unit_cell, cfg)
    assert hist["converged"] is False
    assert hist["num_steps"] == cfg.gs_num_steps
    # Regime guard: the scripted energy really did repeat, so the legacy
    # criterion *would* have fired (see the dE control below).
    energies = [e for e in hist["energies"] if e is not None]
    assert len(energies) >= 2 and max(energies) - min(energies) < 1e-12


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_explicit_dE_converges_on_the_same_signal(monkeypatch, unit_cell):
    """Control: the same scripted run under ``"dE"`` stops at step 2 —
    the false convergence the default switch removes."""
    _script_energy(monkeypatch, grad_scale=1.0)
    with pytest.warns(DeprecationWarning):
        cfg = _cfg(unit_cell, gs_conv_criterion="dE")
    hist = _run(unit_cell, cfg)
    assert hist["converged"] is True
    assert hist["num_steps"] == 2


# --- (c) a small gradient converges under the default ----------------------


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_default_converges_when_gradient_is_below_tol(monkeypatch, unit_cell):
    """|g| ~ 1e-9 << gs_grad_norm_tol: exit on step 0, before any dE exists."""
    _script_energy(monkeypatch, grad_scale=1e-9)
    hist = _run(unit_cell, _cfg(unit_cell))
    assert hist["converged"] is True
    assert hist["num_steps"] == 1


# --- finite safety: a masked NaN gradient is not stationarity ---------------


def test_c4v_reference_nan_gradient_does_not_converge(monkeypatch):
    """The C4v-reference loop masks non-finite gradient entries to 0 before
    taking the norm.  An all-NaN gradient then has ``||g|| == 0.0`` -- below
    any tolerance -- so, under the ``grad_norm`` default, the run would stop
    on step 0 and report the untouched initial state as converged.

    The energy is scripted (value ``_E_CONST``, gradient NaN everywhere) so
    the signal is exact; the real CTM still runs, only its energy is replaced.
    """
    import tenax.algorithms._ctm_tensor as _ctm_tensor

    @jax.custom_vjp
    def _nan_grad(x):
        return jnp.zeros((), x.dtype)

    def _fwd(x):
        return _nan_grad(x), x

    def _bwd(x, g):
        return (jnp.full_like(x, jnp.nan),)

    _nan_grad.defvjp(_fwd, _bwd)

    def _fake_energy(A_tensor, env, gate, d_phys):
        return _E_CONST + _nan_grad(A_tensor.todense())

    monkeypatch.setattr(_ctm_tensor, "compute_energy_ctm_tensor", _fake_energy)

    cfg = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=8, min_iter=2, ctm_ad_mode="c4v_reference"),
        gs_num_steps=3,
        gs_learning_rate=1e-2,
        gs_implicit_ad=True,
        gs_c4v=True,
        unit_cell="1x1",
        su_init=False,
        gs_optimizer="adam",
        gs_verbose=True,
    )
    assert cfg.gs_conv_criterion == "grad_norm"
    A0 = jax.random.normal(jax.random.PRNGKey(7), (2, 2, 2, 2, 2))
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        optimize_gs_ad(_heisenberg_gate(), A0, cfg)
    assert "converged at step" not in buf.getvalue(), buf.getvalue()


# --- SU-init plateau: a stationary *simple-update* start is not converged ---
#
# Under "dE" step 0 can never converge (prev_energy = inf).  Under
# "grad_norm" it can, and a simple-update start that lands on a stationary
# point (a saddle such as the |up up> product state, or the documented SU
# plateau) would be returned as converged before the line search could fail
# and trigger stall recovery (Codex P1 on #1075).  The guard skips the test on
# the first evaluation of an SU-derived start only; a user-supplied A_init
# that is already stationary is a warm start and must converge at once.


def _su_cfg(unit_cell: str, **overrides) -> iPEPSConfig:
    base = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=5,
        ctm=CTMConfig(chi=4, max_iter=4),
        unit_cell=unit_cell,
        gs_num_steps=5,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        su_init=True,
        return_history=True,
        gs_verbose=True,
    )
    return replace(base, **overrides)


def _run_captured(cfg, A_init=None):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        out = optimize_gs_ad(_heisenberg_gate(), A_init, cfg)
    return out[-1], buf.getvalue()


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_su_init_stationary_start_is_not_converged_at_step_zero(monkeypatch, unit_cell):
    """(a) SU start with |g| << tol: step 0 must not converge; the optimizer
    step runs (its line search fails, so the default "reset" stall recovery
    fires and rolls back to the same state); the next evaluation, still
    stationary, converges normally."""
    _script_energy(monkeypatch, grad_scale=1e-9)
    hist, log = _run_captured(_su_cfg(unit_cell))
    assert hist["converged"] is True
    assert hist["num_steps"] == 2, log
    assert "converged at step 2" in log, log
    assert "stall #1, reset L-BFGS history" in log, log


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_user_A_init_stationary_start_converges_at_step_zero(monkeypatch, unit_cell):
    """(b) A user-supplied stationary start is a warm start: exit on step 0."""
    _script_energy(monkeypatch, grad_scale=1e-9)
    A = jax.random.normal(jax.random.PRNGKey(3), (2, 2, 2, 2, 2))
    A_init = A if unit_cell == "1x1" else (A, A[::-1])
    from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

    if unit_cell == "2site":
        A_init = tuple(_wrap_as_dense_tensor(a) for a in A_init)
    hist, log = _run_captured(_su_cfg(unit_cell), A_init)
    assert hist["converged"] is True
    assert hist["num_steps"] == 1, log


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_su_init_guard_leaves_dE_alone(monkeypatch, unit_cell):
    """(c) Under "dE" step 0 never converges anyway (dE = inf); step 1 sees
    dE == 0 and converges -- the guard must not add a step."""
    _script_energy(monkeypatch, grad_scale=1e-9)
    with pytest.warns(DeprecationWarning):
        cfg = _su_cfg(unit_cell, gs_conv_criterion="dE")
    hist, log = _run_captured(cfg)
    assert hist["converged"] is True
    assert hist["num_steps"] == 2, log


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_su_guard_also_withholds_the_chi_stage_advance(monkeypatch, unit_cell):
    """The skipped first evaluation must not reach ``_advance_chi_stage_if_due``
    as a convergence signal either: with a chi schedule, the end-of-step
    stage-advance call re-tests the same small |g| and would leave the first
    stage after one step (Codex P2 on #1075)."""
    from tenax import optimize_gs_ad_chi_schedule

    _script_energy(monkeypatch, grad_scale=1e-9)
    cfg = _su_cfg(unit_cell, ctm=CTMConfig(chi=4, chi_max=6, max_iter=4))
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        optimize_gs_ad_chi_schedule(
            _heisenberg_gate(), None, cfg, chi_schedule=[(4, 3), (6, 3)]
        )
    log = buf.getvalue()
    assert "[iPEPS-AD step 1] schedule advance" not in log, log


# --- root-implicit engine: same guard, its own SU init ---------------------


def _script_root_energy(monkeypatch, unit_cell: str, grad_scale: float):
    if unit_cell == "1x1":
        import tenax.algorithms._ctm_root_implicit_asym as _mod

        def _fake(A_t, gate, **kw):
            a = A_t.todense()
            return jnp.asarray(_E_CONST), grad_scale * jnp.ones_like(a)

        monkeypatch.setattr(_mod, "asym_root_implicit_energy_and_grad", _fake)
    else:
        import tenax.algorithms._ctm_root_implicit_multisite as _mod

        def _fake(A_by_cell, **kw):
            return jnp.asarray(_E_CONST), {
                c: grad_scale * jnp.ones_like(t.todense()) for c, t in A_by_cell.items()
            }

        monkeypatch.setattr(_mod, "cell_root_implicit_energy_and_grad", _fake)


def _root_cfg(unit_cell: str, **kw) -> iPEPSConfig:
    base = dict(
        max_bond_dim=2,
        num_imaginary_steps=5,
        unit_cell=unit_cell,
        su_init=True,
        gs_num_steps=4,
        gs_metric_precond=False,
        gs_line_search=False,
        return_history=True,
        ctm=CTMConfig(chi=4, max_iter=20, conv_tol=1e-10, ctm_ad_mode="root_implicit"),
    )
    base.update(kw)
    return iPEPSConfig(**base)


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_root_implicit_su_start_is_not_converged_at_step_zero(monkeypatch, unit_cell):
    _script_root_energy(monkeypatch, unit_cell, 1e-9)
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    hist = optimize_gs_ad_root_implicit(_heisenberg_gate(), None, _root_cfg(unit_cell))[
        -1
    ]
    assert hist["converged"] is True
    assert hist["num_steps"] == 2


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_root_implicit_user_A_init_converges_at_step_zero(monkeypatch, unit_cell):
    _script_root_energy(monkeypatch, unit_cell, 1e-9)
    from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    A = _wrap_as_dense_tensor(jax.random.normal(jax.random.PRNGKey(3), (2,) * 5))
    A_init = A if unit_cell == "1x1" else (A, A)
    hist = optimize_gs_ad_root_implicit(
        _heisenberg_gate(), A_init, _root_cfg(unit_cell)
    )[-1]
    assert hist["converged"] is True
    assert hist["num_steps"] == 1
