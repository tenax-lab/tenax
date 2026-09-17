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

    # It surfaces as a warning before the run gets far enough to need a state.
    # ``A_init=None`` then stops it: the symmetric variant has no way to build
    # one (see ``test_root_implicit_symmetric_wiring.py``), which makes this a
    # cheap place to observe the warning without running an optimization.
    with pytest.warns(UserWarning, match="gs_metric_precond"):
        with pytest.raises(ValueError, match="needs an explicit A_init"):
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


def test_a_general_lattice_is_refused_with_the_grid_reason():
    """#894 wired the 2-site cell; a general Lattice stays refused, and for the
    real reason.

    The multisite engine's neighbour map is ``cell_neighbors`` -- a rectangular
    periodic grid.  ``unit_cell='2site'`` is that grid (a 2x2 bipartite cell),
    so it now runs.  A ``Lattice`` whose topology is *not* a rectangular grid --
    kagome, honeycomb, triangular -- has no root-implicit objective here, and
    the refusal must say so rather than silently optimising the wrong model on a
    grid it does not have.
    """
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )
    from tenax.core.lattice import kagome

    cfg = dataclasses.replace(
        _cfg(), unit_cell=kagome(), gs_metric_precond=False, gs_line_search=False
    )
    gate = jnp.eye(4, dtype=jnp.float64).reshape(2, 2, 2, 2)
    with pytest.raises(NotImplementedError, match="rectangular periodic grid"):
        optimize_gs_ad_root_implicit(gate, None, cfg)


def test_the_symmetric_variant_is_no_longer_refused():
    """It was blocked on #731 (8.4 GB in the GMRES solve); that is fixed.

    The rest of its surface -- the initial state, the parameter type, the NaN
    guard through the pytree -- lives in
    ``tests/test_root_implicit_symmetric_wiring.py``.  What is pinned *here* is
    only that dispatch no longer dead-ends, because a refusal is exactly the
    kind of thing that gets reinstated by a merge.
    """
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    cfg = _cfg(ctm_ad_mode="root_implicit_symmetric")
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="needs an explicit A_init"):
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
    assert any("had a non-finite gradient" in m for m in messages), messages
    # A fully-masked step is the only kind that is genuinely a no-op, and the
    # end-of-run summary has to say which kind it saw rather than calling every
    # masked step no progress -- a partly-masked step still moves the state.
    summary = next(m for m in messages if "had a non-finite gradient" in m)
    assert "fully masked" in summary and "partly masked" in summary, summary


# ------------------------------------------------------------------ #
# The gap that blocks production use (#772)                            #
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

    # Pre-optimization energy of the same starting tensor on the same CTM path.
    _A0, _env0, E_start = optimize_gs_ad(
        gate, None, dataclasses.replace(cfg, gs_num_steps=0)
    )

    _A, _env, E = optimize_gs_ad(gate, None, cfg)
    assert math.isfinite(E), E
    assert E < E_start, f"no descent: {E!r} did not improve on {E_start!r}"
    # The simple-update start sits at -0.48198 at this chi; three Adam steps
    # reach -0.5066.  Requiring a clear improvement rather than the exact value
    # keeps the test about the optimizer working, not about its schedule.
    assert E < -0.49, f"the optimizer did not improve on the SU start: {E}"
    # Square-lattice spin-1/2 Heisenberg AFM, Sandvik QMC.  A D=2 chi=6 state
    # cannot legitimately go below this.
    assert E > -0.669437, f"non-variational energy below the exact ground state: {E}"


@pytest.mark.slow
def test_the_root_implicit_gradient_descends_the_energy():
    """A few plain gradient steps must lower the energy monotonically.

    Uses the engine directly, so no optimizer schedule sits between the
    gradient and the assertion.  A wrong-sign or badly-scaled gradient shows
    up here immediately -- that was the failure mode #718 spent a long time
    on, where the energy boundary was mis-glued and the gradient was off by
    3e-2 relative while every residual looked healthy.

    **Marked slow, reluctantly.**  This was written to give the *required*
    gate (``-m core``) coverage of the claim that the gradient points
    downhill, which it otherwise has none of.  It cost 4.2 GB in-suite when
    that was written, against a 210 MB baseline for the rest of this file.
    ``-m core`` already peaks at 6.35 GB against ~7 GB runners (#732), and JAX
    caches persist across tests within a session, so landing 4 GB on top of an
    accumulated cache risked an OOM presenting as a confusing flake.

    **Two things in the original note were wrong, and #731 fixed the second.**
    The cost was never "the adjoint solve's Krylov basis": the real embedding
    at D=2 chi=4 is n = 384, so a 30-dimensional basis is **91 KB**.  It was
    XLA compiling the operator inside GMRES's ``lax.while_loop``.  Taking the
    loop out of the jit and leaving the matvec in it drops one isolated
    gradient here from **2.94 GB / 90.0 s to 1.74 GB / 39.0 s**, gradient
    identical to ten digits (measured in a fresh process, which is why both
    numbers sit below the 4.2 GB in-suite figure).

    So the memory argument for this marker is gone and the wall time is what
    is left.  Promoting it into ``-m core`` is a CI-budget decision rather
    than a memory one now, and is deliberately not taken here.
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
            DenseTensor(params, indices),
            gate,
            chi=4,
            max_iter=40,
            conv_tol=1e-10,
        )
        assert bool(jnp.all(jnp.isfinite(g))), "non-finite gradient"
        energies.append(float(jnp.real(E)))
        params = params - lr * g
        params = params / jnp.linalg.norm(params)

    assert all(energies[i + 1] < energies[i] for i in range(len(energies) - 1)), (
        f"energy did not decrease monotonically: {energies}"
    )
    assert energies[-1] > -0.669437, f"below the exact ground state: {energies}"


# ------------------------------------------------------------------ #
# The 2-site checkerboard cell path (#894)                            #
# ------------------------------------------------------------------ #


def _site_indices(D=2, d=2):
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry

    sym = U1Symmetry()
    z = [0] * D
    zp = [0] * d
    return (
        TensorIndex.from_charges(sym, list(z), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, list(z), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, list(z), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, list(z), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, list(zp), FlowDirection.IN, label="phys"),
    )


def test_fan_checkerboard_places_the_two_sublattices_on_the_neel_diagonals():
    """``a`` on (0,0)/(1,1), ``b`` on (0,1)/(1,0) -- the bipartite (Neel)
    placement, not stripe.

    Under ``cell_neighbors(2, 2)`` this is exactly what makes every neighbour of
    an ``a`` site a ``b`` site.  A stripe placement -- ``a`` on a whole row --
    would put like on like along one axis and silently optimise a different
    model on the same grid, so the diagonal pairing is a correctness property,
    not a cosmetic choice.
    """
    from tenax.algorithms.ipeps_optimize_root_implicit import _fan_checkerboard

    idx = _site_indices()
    a = jnp.ones((2, 2, 2, 2, 2))
    b = 2.0 * jnp.ones((2, 2, 2, 2, 2))
    cell = _fan_checkerboard((a, b), idx, idx)

    assert set(cell) == {(0, 0), (1, 1), (0, 1), (1, 0)}
    # a-sublattice: the two main-diagonal cells carry the same tensor ...
    assert jnp.array_equal(cell[(0, 0)].todense(), cell[(1, 1)].todense())
    assert float(cell[(0, 0)].todense().reshape(-1)[0]) == 1.0
    # ... b-sublattice the anti-diagonal, and the two sublattices differ.
    assert jnp.array_equal(cell[(0, 1)].todense(), cell[(1, 0)].todense())
    assert float(cell[(0, 1)].todense().reshape(-1)[0]) == 2.0
    assert not jnp.array_equal(cell[(0, 0)].todense(), cell[(0, 1)].todense())


def test_tie_checkerboard_sums_each_sublattices_two_partials():
    """``dE/da = dE/dA_(0,0) + dE/dA_(1,1)``; ``dE/db = dE/dA_(0,1) + dE/dA_(1,0)``.

    The engine sees four independent cells and returns four cotangents; the tie
    makes each parameter's gradient the sum of the two partials it drives.
    Powers of ten make every key's contribution uniquely identifiable, so 11 and
    1100 are reachable *only* by the correct pairing -- any wrong pair, or a
    dropped partial, lands on a different number.
    """
    from tenax.algorithms.ipeps_optimize_root_implicit import _tie_checkerboard_grad

    grad = {
        (0, 0): jnp.array([1.0]),
        (1, 1): jnp.array([10.0]),
        (0, 1): jnp.array([100.0]),
        (1, 0): jnp.array([1000.0]),
    }
    ga, gb = _tie_checkerboard_grad(grad)
    assert float(ga.reshape(-1)[0]) == 11.0
    assert float(gb.reshape(-1)[0]) == 1100.0


def test_2site_a_init_must_be_a_pair():
    """The cell path optimises two sublattice tensors, so ``A_init`` is ``None``
    or a ``(A, B)`` tuple; a bare tensor is an ambiguous single-site start and
    is refused rather than silently placed on both sublattices.
    """
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    cfg = dataclasses.replace(
        _cfg(), unit_cell="2site", gs_metric_precond=False, gs_line_search=False
    )
    gate = jnp.eye(4, dtype=jnp.float64).reshape(2, 2, 2, 2)
    lone = jnp.ones((2, 2, 2, 2, 2))
    with pytest.raises(TypeError, match="tuple of the two"):
        optimize_gs_ad_root_implicit(gate, lone, cfg)


@pytest.mark.slow
def test_2site_checkerboard_run_through_optimize_gs_ad():
    """A real 2-site Heisenberg state optimised through the root-implicit cell
    path (#894).

    The single-site slow test above is the 1x1 analogue; this is the gate #894
    had to clear -- that an optimizer *descends* a genuine two-site energy whose
    RDMs span adjacent cells, with an SVD-free backward.  As on the 1x1 path the
    assertions are physical, not a pinned number: this path runs no line search
    (documented), so the trajectory can rise on a step while best-so-far falls,
    and the returned energy is the best state reached.  It must improve on the
    random start and must stay above the exact ground state; a value below it
    would be non-variational, the failure a hard-coded ``approx`` would hide.

    The return shape is pinned too: ``((A, B), (env_A, env_B), E)`` matching
    ``_optimize_gs_ad_2site``, so a caller can drop the cell path in for the
    fixed-point 2-site optimizer.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms.ipeps import heisenberg_gate
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad
    from tenax.core.tensor import DenseTensor

    gate = heisenberg_gate()  # plain AFM: the ground state is Neel, A != B
    cfg = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        su_init=False,  # deterministic random start (PRNGKey(0))
        gs_num_steps=4,
        gs_optimizer="lbfgs",
        gs_line_search=False,
        gs_metric_precond=False,
        return_history=True,
        ctm=CTMConfig(chi=4, max_iter=30, conv_tol=1e-9, ctm_ad_mode="root_implicit"),
    )

    (A, B), envs, E, hist = optimize_gs_ad(gate, None, cfg)

    energies = hist["energies"]
    assert all(math.isfinite(e) for e in energies), energies
    assert math.isfinite(E), E
    # Descended: the best state reached beats the random start.  (Not
    # monotonic -- no line search on this path.)
    assert min(energies) < energies[0], f"no descent: {energies}"
    assert E < energies[0], f"returned energy did not improve on the start: {E}"
    # Square-lattice spin-1/2 Heisenberg AFM, Sandvik QMC.  A D=2 chi=4 state
    # cannot legitimately go below this.
    assert E > -0.669437, f"non-variational energy below the ground state: {E}"
    # Two distinct sublattice tensors and their two environments, per
    # _optimize_gs_ad_2site's contract.
    assert isinstance(A, DenseTensor) and isinstance(B, DenseTensor)
    assert isinstance(envs, tuple) and len(envs) == 2
