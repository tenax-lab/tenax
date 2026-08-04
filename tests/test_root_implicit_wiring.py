"""Wiring for ``ctm_ad_mode="root_implicit"`` (#715).

These cover the *dispatch and guard* surface, not the characteristic equations
themselves -- those live in ``test_ctm_root_implicit_asym.py`` and friends.

The guards are the point.  The root-implicit ``*_energy_and_grad`` entry points
replace the whole ``value_and_grad``: each runs its own CTM convergence and
takes no warm-start environment, so every knob that rides on a warm-started env
would be *silently ignored* if it were merely passed through.  This repository
has been burned by exactly that shape of bug in #723, #760 and #762, so the
policy is to hard-error at config time.
"""

from __future__ import annotations

import dataclasses

import pytest

from tenax.algorithms.ipeps_ad_policy import use_root_implicit_path
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize_root_implicit import (
    root_implicit_variant,
    validate_root_implicit_config,
)


def _cfg(**ctm_kw):
    """A minimal root-implicit config; ``ctm_kw`` overrides CTM fields."""
    base = dict(chi=4, max_iter=50, conv_tol=1e-10, ctm_ad_mode="root_implicit")
    base.update(ctm_kw)
    return iPEPSConfig(max_bond_dim=2, unit_cell="1x1", ctm=CTMConfig(**base))


# ------------------------------------------------------------------ #
# Config surface                                                       #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "mode", ["root_implicit", "root_implicit_symmetric", "c4v_reference", None]
)
def test_valid_ctm_ad_modes_are_accepted(mode):
    iPEPSConfig(max_bond_dim=2, ctm=CTMConfig(chi=4, ctm_ad_mode=mode))


def test_unknown_ctm_ad_mode_still_rejected():
    with pytest.raises(ValueError, match="ctm_ad_mode must be one of"):
        iPEPSConfig(max_bond_dim=2, ctm=CTMConfig(chi=4, ctm_ad_mode="root"))


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("root_implicit", True),
        ("root_implicit_symmetric", True),
        ("c4v_reference", False),
        (None, False),
    ],
)
def test_path_predicate(mode, expected):
    cfg = iPEPSConfig(max_bond_dim=2, ctm=CTMConfig(chi=4, ctm_ad_mode=mode))
    assert use_root_implicit_path(cfg) is expected


def test_variant_selection_follows_the_unit_cell():
    """``"root_implicit"`` picks the dense engine from the unit cell."""
    assert root_implicit_variant(_cfg()) == "asym"
    cell = dataclasses.replace(_cfg(), unit_cell="2site")
    assert root_implicit_variant(cell) == "cell"
    sym = _cfg(ctm_ad_mode="root_implicit_symmetric")
    assert root_implicit_variant(sym) == "symmetric"


def test_variant_rejects_a_non_root_mode():
    cfg = iPEPSConfig(max_bond_dim=2, ctm=CTMConfig(chi=4, ctm_ad_mode=None))
    with pytest.raises(ValueError, match="not a root-implicit mode"):
        root_implicit_variant(cfg)


# ------------------------------------------------------------------ #
# Guards: every one of these would otherwise be silently ignored       #
# ------------------------------------------------------------------ #


def test_a_plain_config_validates():
    validate_root_implicit_config(_cfg())


@pytest.mark.parametrize(
    "field,value",
    [
        ("chi_auto_bump", True),
        ("chi_ramp", [(4, 10)]),
        ("fuse_virtual_legs", False),
    ],
)
def test_ctm_knobs_that_cannot_be_honoured_are_rejected(field, value):
    cfg = _cfg(**{field: value})
    with pytest.raises(NotImplementedError, match="silently ignored"):
        validate_root_implicit_config(cfg)


def test_in_ctm_chi_bump_is_rejected():
    """Separate because ``ctmrg_heuristic_increase_chi`` needs ``chi_max`` set
    at construction, so it cannot ride the parametrised case above."""
    cfg = _cfg(ctmrg_heuristic_increase_chi=True, chi_max=16)
    with pytest.raises(NotImplementedError, match="silently ignored"):
        validate_root_implicit_config(cfg)


def test_checkpoint_path_is_rejected():
    cfg = dataclasses.replace(_cfg(), gs_checkpoint_path="/tmp/ckpt.pkl")
    with pytest.raises(NotImplementedError, match="silently ignored"):
        validate_root_implicit_config(cfg)


def test_metric_precond_warns_rather_than_rejecting():
    """It defaults to True, so rejecting it would make the mode unusable.

    Every other unhonourable knob defaults off, so hard-erroring on those costs
    the user nothing.  ``gs_metric_precond=True`` is the default, and rejecting
    it would force an unrelated opt-out before root-implicit could be tried at
    all.  The split-CTM path warns and falls back for the same knob for the
    same reason; this follows it.
    """
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    cfg = _cfg(ctm_ad_mode="root_implicit_symmetric")
    assert cfg.gs_metric_precond is True, "precondition: it defaults on"
    validate_root_implicit_config(cfg)  # must NOT raise

    # It surfaces as a warning on the way to the (unwired) symmetric refusal.
    with pytest.warns(UserWarning, match="gs_metric_precond"):
        with pytest.raises(NotImplementedError):
            optimize_gs_ad_root_implicit(None, None, cfg)


def test_the_error_names_every_offending_knob_at_once():
    """One run should not require N round-trips to discover N bad settings."""
    # chi_auto_bump and chi_ramp are mutually exclusive at construction, so
    # pair the bump with the split-CTM flag instead.
    cfg = dataclasses.replace(
        _cfg(chi_auto_bump=True, fuse_virtual_legs=False),
        gs_checkpoint_path="/tmp/ckpt.pkl",
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_root_implicit_config(cfg)
    msg = str(exc.value)
    assert "chi_auto_bump" in msg
    assert "fuse_virtual_legs" in msg
    assert "gs_checkpoint_path" in msg


# ------------------------------------------------------------------ #
# Unwired variants must say so rather than run something else          #
# ------------------------------------------------------------------ #


def test_multisite_variant_is_refused_with_a_reason():
    """Phase 2's shifted-cell tables are the risk; it gets its own increment."""
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    cfg = dataclasses.replace(_cfg(), unit_cell="2site")
    with pytest.raises(NotImplementedError, match="dense 1x1"):
        optimize_gs_ad_root_implicit(None, None, cfg)


def test_symmetric_variant_is_refused_and_cites_its_blocker():
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    cfg = _cfg(ctm_ad_mode="root_implicit_symmetric")
    with pytest.raises(NotImplementedError, match="8.4 GB"):
        optimize_gs_ad_root_implicit(None, None, cfg)


# ------------------------------------------------------------------ #
# The gap that blocks production use (#772)                            #
# ------------------------------------------------------------------ #


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "#772, NARROWED by the rank cap: the covariant residual on a physical "
        "simple-update state is now 1.9e-05 at chi=6 (was 5.7e+05) and the "
        "gradient is finite and FD-correct to 6.9e-08 (was NaN). What still "
        "fails is the GATE, not the gradient: clamping the unusable directions "
        "leaves the forward root residual at 2.8e-06, and the default "
        "root_residual_warn=1e-06 rejects that by 2.8x. The residual floor is "
        "intrinsic to a clamped spectrum, so closing this needs a decision on "
        "the tolerance -- note the gradient is 40x more accurate than the gate "
        "implies, so 1e-06 is not predictive of gradient quality here."
    ),
    strict=False,
)
def test_production_heisenberg_run_through_optimize_gs_ad():
    """The production case this wiring exists for: a real Heisenberg state.

    Left in as an executable record of #772 rather than deleted, so whoever
    fixes the engine gets an immediate signal here (an XPASS) instead of having
    to reconstruct the case.

    The failure is at least *loud*: ``on_root_residual="raise"`` (#759) turns a
    non-vanishing root residual into ``RootResidualError``, so the optimizer
    stops rather than stepping on a NaN gradient.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms.ipeps import heisenberg_gate, sublattice_rotate_gate
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=40,
        dt=0.05,
        unit_cell="1x1",
        su_init=True,
        gs_num_steps=3,
        gs_optimizer="adam",
        gs_learning_rate=1e-2,
        gs_line_search=False,
        gs_metric_precond=False,
        ctm=CTMConfig(chi=6, max_iter=100, conv_tol=1e-10, ctm_ad_mode="root_implicit"),
    )
    _A, _env, E = optimize_gs_ad(gate, None, cfg)
    assert E == pytest.approx(-0.4886, abs=5e-3)
