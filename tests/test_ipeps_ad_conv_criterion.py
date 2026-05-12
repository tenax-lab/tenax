"""Regression tests for the AD outer-loop convergence criterion (issue #448).

``iPEPSConfig.gs_conv_criterion`` selects which condition exits
``optimize_gs_ad``:

* ``"dE"``       — legacy ``|E_step - E_step-1| < gs_conv_tol`` (default).
* ``"grad_norm"`` — variPEPS-style ``||grad||_2 < gs_grad_norm_tol``.
* ``"both"``      — require both to hold simultaneously.

These tests cover the helper, config validation, and end-to-end exit
behaviour on the 1-site dispatcher (the 2-site/multisite dispatchers
share the same helper and convergence shape).
"""

from dataclasses import replace

import jax.numpy as jnp
import pytest

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad
from tenax.algorithms.ipeps_optimize import _converged_outer, _grad_l2_norm

# --- unit: deprecation warning for legacy default --------------------------


def test_iPEPSConfig_dE_emits_deprecation_warning():
    """``gs_conv_criterion='dE'`` (current default) emits a DeprecationWarning.

    The warning is suppressed project-wide in ``pyproject.toml`` so the
    rest of the test suite stays quiet during the transition; this test
    un-suppresses it via ``pytest.warns`` to confirm it still fires.
    """
    with pytest.warns(DeprecationWarning, match="gs_conv_criterion='dE'"):
        iPEPSConfig()  # default
    with pytest.warns(DeprecationWarning, match="gs_conv_criterion='dE'"):
        iPEPSConfig(gs_conv_criterion="dE")  # explicit


def test_iPEPSConfig_grad_norm_no_warning():
    """Opting into ``'grad_norm'`` or ``'both'`` silences the deprecation."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        iPEPSConfig(gs_conv_criterion="grad_norm")
        iPEPSConfig(gs_conv_criterion="both")


# --- unit: _converged_outer ------------------------------------------------


def _cfg(criterion: str, *, dE_tol: float = 1e-8, gn_tol: float = 1e-5) -> iPEPSConfig:
    return iPEPSConfig(
        gs_conv_criterion=criterion,
        gs_conv_tol=dE_tol,
        gs_grad_norm_tol=gn_tol,
    )


def test_converged_outer_dE_default_ignores_grad_norm():
    cfg = _cfg("dE", dE_tol=1e-5)
    # Small dE: converged regardless of (huge) grad norm.
    assert _converged_outer(cfg, delta_energy=1e-7, grad_norm=1e3) is True
    # Large dE: not converged even with tiny grad norm.
    assert _converged_outer(cfg, delta_energy=1e-3, grad_norm=1e-12) is False
    # None grad norm is fine for the dE-only criterion.
    assert _converged_outer(cfg, delta_energy=1e-7, grad_norm=None) is True


def test_converged_outer_grad_norm():
    cfg = _cfg("grad_norm", gn_tol=1e-5)
    # Issue #448 motivating case: tiny dE, large grad norm → NOT converged.
    assert _converged_outer(cfg, delta_energy=1e-12, grad_norm=1e-2) is False
    # Genuine stationarity → converged.
    assert _converged_outer(cfg, delta_energy=1e-2, grad_norm=1e-7) is True
    # A missing grad norm cannot satisfy the grad-norm criterion.
    assert _converged_outer(cfg, delta_energy=1e-12, grad_norm=None) is False


def test_converged_outer_both_requires_both():
    cfg = _cfg("both", dE_tol=1e-5, gn_tol=1e-5)
    # dE alone is not enough.
    assert _converged_outer(cfg, delta_energy=1e-9, grad_norm=1e-2) is False
    # grad norm alone is not enough.
    assert _converged_outer(cfg, delta_energy=1e-2, grad_norm=1e-9) is False
    # Both → converged.
    assert _converged_outer(cfg, delta_energy=1e-9, grad_norm=1e-9) is True
    # Missing grad norm cannot satisfy "both".
    assert _converged_outer(cfg, delta_energy=1e-9, grad_norm=None) is False


def test_grad_l2_norm_matches_jnp_norm():
    g1 = jnp.array([3.0, 4.0])
    g2 = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    # Plain pytree (tuple). ||(3,4)||_2 = 5.
    assert _grad_l2_norm((g1, g2)) == pytest.approx(5.0)
    # Empty pytree → 0.0.
    assert _grad_l2_norm([]) == 0.0


# --- unit: config validation -----------------------------------------------


def test_iPEPSConfig_rejects_invalid_conv_criterion():
    with pytest.raises(ValueError, match="gs_conv_criterion must be one of"):
        iPEPSConfig(gs_conv_criterion="grad")  # type: ignore[arg-type]


def test_iPEPSConfig_rejects_nonpositive_grad_norm_tol():
    with pytest.raises(ValueError, match="gs_grad_norm_tol must be positive"):
        iPEPSConfig(gs_grad_norm_tol=0.0)


# --- integration: 1-site dispatcher end-to-end -----------------------------


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _fast_base_cfg() -> iPEPSConfig:
    """Cheap explicit-AD config that still exercises the outer loop.

    ``su_init=False`` skips simple update and uses the seeded random
    initializer in ``optimize_gs_ad``. The SU-init at small ``D``/``chi``
    can produce SVDs degenerate enough to NaN the AD backward; the
    random init keeps the convergence-criterion logic the focus of the
    test rather than upstream numerics.
    """
    return iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=5),
        gs_num_steps=6,
        gs_learning_rate=1e-2,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        gs_optimizer="adam",
        su_init=False,
        return_history=True,
    )


@pytest.mark.algorithm
def test_grad_norm_criterion_exits_when_grad_is_tiny():
    """A huge ``gs_grad_norm_tol`` forces an immediate grad-norm exit."""
    cfg = replace(
        _fast_base_cfg(),
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e30,  # any finite grad norm satisfies this.
        # Make the dE criterion impossibly tight so it cannot trip first.
        gs_conv_tol=1e-30,
    )
    out = optimize_gs_ad(_heisenberg_gate(), None, cfg)
    history = out[-1]
    assert history["converged"] is True
    # Exited well before the step budget.
    assert history["num_steps"] < cfg.gs_num_steps


@pytest.mark.algorithm
def test_dE_default_unchanged_when_dE_tol_loose():
    """``"dE"`` (legacy default) still exits via the dE branch."""
    cfg = replace(
        _fast_base_cfg(),
        gs_conv_criterion="dE",
        gs_conv_tol=1.0,  # arbitrarily loose → exits on first step.
    )
    out = optimize_gs_ad(_heisenberg_gate(), None, cfg)
    history = out[-1]
    assert history["converged"] is True


@pytest.mark.algorithm
def test_both_criterion_requires_both_to_pass():
    """``"both"`` does NOT exit if only dE is loose enough."""
    # Loose dE, tight grad-norm — should NOT trigger early exit.
    cfg = replace(
        _fast_base_cfg(),
        gs_conv_criterion="both",
        gs_conv_tol=1.0,
        gs_grad_norm_tol=1e-30,
    )
    out = optimize_gs_ad(_heisenberg_gate(), None, cfg)
    history = out[-1]
    # Did not converge under "both" because the grad-norm half can't be met.
    assert history["converged"] is False
    assert history["num_steps"] == cfg.gs_num_steps
