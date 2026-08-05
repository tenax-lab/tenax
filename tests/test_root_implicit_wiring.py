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
import math

import jax.numpy as jnp
import numpy as np
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


def test_ctm_config_carries_the_rank_clamp():
    assert CTMConfig(chi=6).rel_floor is None


def test_ctm_config_rejects_a_nonsensical_rank_clamp():
    with pytest.raises(ValueError, match="rel_floor"):
        CTMConfig(chi=6, rel_floor=0.0)
    with pytest.raises(ValueError, match="rel_floor"):
        CTMConfig(chi=6, rel_floor=1.5)


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
# rel_floor forwarding + the NaN-gradient warnings (#772 Task 4)       #
# ------------------------------------------------------------------ #


def test_rel_floor_forwards_and_nonfinite_grads_warn(monkeypatch):
    """Pins the ``rel_floor=ctm_cfg.rel_floor`` forwarding and both new
    warnings that replaced the old silent NaN-gradient mask.

    ``asym_root_implicit_energy_and_grad`` is imported *inside* the body of
    ``optimize_gs_ad_root_implicit``, so monkeypatching it on
    ``ipeps_optimize_root_implicit`` itself would do nothing -- the import
    statement rebinds the name from its *source* module,
    ``tenax.algorithms._ctm_root_implicit_asym``, at call time.  Patching
    there is what actually takes effect.

    The stub records the ``rel_floor`` it was called with and returns
    deliberately non-finite gradients, so a single ``gs_num_steps=1`` run
    exercises the per-step warning and the end-of-run summary without paying
    for a real root-implicit CTM solve (the point of stubbing it out).
    """
    import tenax.algorithms._ctm_root_implicit_asym as asym_mod
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    captured = {}

    def _stub_energy_and_grad(A, gate, *, rel_floor, **kwargs):
        captured["rel_floor"] = rel_floor
        data = A.todense()
        return jnp.asarray(-0.5 + 0.0j), jnp.full_like(data, jnp.nan)

    monkeypatch.setattr(
        asym_mod, "asym_root_implicit_energy_and_grad", _stub_energy_and_grad
    )

    D, d = 2, 2
    rng = np.random.RandomState(0)
    A_init = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    gate = jnp.eye(d * d, dtype=jnp.float64).reshape(d, d, d, d)

    cfg = iPEPSConfig(
        max_bond_dim=D,
        unit_cell="1x1",
        gs_num_steps=1,
        gs_optimizer="adam",
        gs_learning_rate=1e-2,
        gs_conv_criterion="grad_norm",
        ctm=CTMConfig(
            chi=4,
            max_iter=5,
            conv_tol=1e-10,
            ctm_ad_mode="root_implicit",
            rel_floor=1e-5,
        ),
    )

    with pytest.warns(RuntimeWarning) as record:
        optimize_gs_ad_root_implicit(gate, A_init, cfg)

    assert captured["rel_floor"] == 1e-5, "rel_floor was not forwarded"

    messages = [str(w.message) for w in record]
    assert any("non-finite gradient entries" in m for m in messages), messages
    assert any("optimizer steps made no progress" in m for m in messages), messages


# ------------------------------------------------------------------ #
# The gap that blocks production use (#772)                            #
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_production_heisenberg_run_through_optimize_gs_ad():
    """The production case this wiring exists for: a real Heisenberg state.

    This carried #772 as an xfail through two rounds.  First the covariant
    equations NaN'd the gradient on a physical simple-update state; #778's rank
    clamp fixed that.  What remained was the *gate*: clamping leaves an
    O(rel_floor) inconsistency in the equations by construction, and the fixed
    1e-06 threshold rejected it.  The gate is now set from the clamp.

    **What this asserts, and why not an energy target.**  It used to end at
    ``E == approx(-0.4886, abs=5e-3)``.  That number was written in the same
    commit that marked the test xfail.  Its provenance is now known: ``-0.4886``
    matches ``-0.488638504625`` recorded at
    ``tests/test_ctm_723_single_site_collapse.py:16`` as the pre-optimization
    2x2 CTM energy of the same D=2 sublattice-rotated Heisenberg simple-update
    state (that file uses ``num_imaginary_steps=60`` against this test's
    ``40``, which accounts for the small offset from the measured
    ``E_start = -0.48198`` below).  So the old assertion demanded that three
    Adam steps barely move the energy, and the run in fact lands at
    ``-0.5066``.
    Pinning an unvalidated constant would either freeze in whatever the
    optimizer happens to do today or keep failing for a reason that has
    nothing to do with the wiring.  So this asserts what the test can actually
    certify: the whole path runs end to end without tripping the residual gate
    (no ``RootResidualError`` -- the point of #772), the energy is finite, and
    the optimizer *descends*.

    The descent baseline is measured, not quoted.  It cannot come from
    ``PHYSICAL_SU_D2_E_SU``: that is ``ipeps()``'s own simple-update energy
    evaluation, whereas ``E`` here is a converged CTM energy from
    ``compute_energy_ctm_tensor``, and the two are different quantities (the
    SU number is *lower*, so comparing them fails for an apples-to-oranges
    reason).  Instead the same config with ``gs_num_steps=0`` is run first,
    which returns the SU-initialised starting tensor through the identical
    ``_final_env`` + ``compute_energy_ctm_tensor`` path.  Measured: baseline
    ``-0.48198``, optimized ``-0.50660``, a descent of 2.5e-02 over three adam
    steps.
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

    # Pre-optimization energy of the same starting tensor on the same CTM path.
    _A0, _env0, E_start = optimize_gs_ad(
        gate, None, dataclasses.replace(cfg, gs_num_steps=0)
    )

    _A, _env, E = optimize_gs_ad(gate, None, cfg)
    assert math.isfinite(E), E
    assert E < E_start, f"no descent: {E!r} did not improve on {E_start!r}"
