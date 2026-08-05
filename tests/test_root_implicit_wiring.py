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
# The production case: an optimizer running through the path (#715)    #
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_production_heisenberg_run_through_optimize_gs_ad():
    """A real Heisenberg state optimized through the root-implicit path.

    This is the gate #715 actually had to clear.  Every other verification in
    the suite is a single gradient at a single point; what was never shown is
    that an optimizer *descends* through this path.  Until #779 it could not
    even take its first step -- the root-residual gate raised on a physical
    simple-update state, whose environment supports only three directions at
    any chi so the rank clamp always fires.

    The assertions are physical rather than a pinned number: at three Adam
    steps the exact value is a property of the optimizer schedule, but the
    energy must go *down* from the simple-update start and must not fall below
    the exact ground state.  A run that breaches the latter is reporting a
    non-variational energy, which is the failure mode that matters and the one
    a hard-coded ``approx`` would have hidden behind a tolerance.
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

    assert math.isfinite(E), E
    # The simple-update start sits at -0.48198 at this chi; three Adam steps
    # reach -0.5066.  Requiring a clear improvement rather than the exact value
    # keeps the test about the optimizer working, not about its schedule.
    assert E < -0.49, f"the optimizer did not improve on the SU start: {E}"
    # Square-lattice spin-1/2 Heisenberg AFM, Sandvik QMC.  A D=2 chi=6 state
    # cannot legitimately go below this.
    assert E > -0.669437, f"non-variational energy below the exact ground state: {E}"


def test_the_root_implicit_gradient_descends_the_energy():
    """A few plain gradient steps must lower the energy monotonically.

    The slow test above exercises the whole ``optimize_gs_ad`` stack but is
    deselected from the required gate, which runs ``-m core`` only.  This is
    the same claim -- the gradient points downhill -- at a size the required
    gate can afford, using the engine directly so no optimizer schedule sits
    between the gradient and the assertion.

    A wrong-sign or badly-scaled gradient shows up here immediately; that was
    the failure mode #718 spent a long time on, where the energy boundary was
    mis-glued and the gradient was off by 3e-2 relative while every residual
    looked healthy.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms._ctm_root_implicit_asym import (
        asym_root_implicit_energy_and_grad,
    )
    from tenax.algorithms.ipeps import heisenberg_gate, sublattice_rotate_gate
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    D = d = 2
    sym = U1Symmetry()
    zeros = [0] * D
    zphys = [0] * d
    indices = (
        TensorIndex.from_charges(sym, list(zeros), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, list(zeros), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, list(zeros), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, list(zeros), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, list(zphys), FlowDirection.IN, label="phys"),
    )
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (D, D, D, D, d))
    params = params / jnp.linalg.norm(params)

    gate = sublattice_rotate_gate(heisenberg_gate())
    energies = []
    lr = 0.05
    for _ in range(3):
        E, g = asym_root_implicit_energy_and_grad(
            DenseTensor(params, indices), gate, chi=4, max_iter=40, conv_tol=1e-10
        )
        assert bool(jnp.all(jnp.isfinite(g))), "non-finite gradient"
        energies.append(float(jnp.real(E)))
        params = params - lr * g
        params = params / jnp.linalg.norm(params)

    assert all(energies[i + 1] < energies[i] for i in range(len(energies) - 1)), (
        f"energy did not decrease monotonically: {energies}"
    )
    assert energies[-1] > -0.669437, f"below the exact ground state: {energies}"
