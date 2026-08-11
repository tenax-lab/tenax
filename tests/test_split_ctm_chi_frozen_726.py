"""Detectors for the #726 rank-1 corner collapse on the ``1x1`` CTM recipe.

The ``1x1`` corner-pair (Fishman) projector -- shared *verbatim* by the fused
and split single-site paths, so this is not a split-CTM defect -- collapses the
environment to rank-1 corners.  ``_ctm_projector.py`` forms ``M = C1g† C4g``,
which is ``chi x chi``: the ``chi·D²`` seam is summed away, so ``M`` sees only
the outer chi legs.  Since ``C1_new = S^{1/2} U^H``, the updated corner's whole
spectrum is ``sqrt(spec(M))`` and ``rank(C_new) <= rank(C1g)``, which is 1 at
cold init.  The corner can never grow rank.

The detector is the **rank of the corner**, measured directly.

An earlier version of this file used *energy frozen in chi* instead, on the
reasoning that a genuine corner transfer matrix improves as the boundary grows
while a chi_eff=1 mean-field boundary does not.  That proxy does not work, and
the file now says so rather than carrying it: a collapsed environment gives a
chi-independent *wrong* energy and a converged one gives a chi-independent
*right* energy, so flatness is consistent with both and discriminates neither.
On this fixture the 2x2 energy moves by **2 ULP** between chi=8 and chi=16 and
by exactly zero between 4 and 8, with the corner at rank 4/6/6 throughout --
so the bit-inequality assertion was decided by rounding, and split by platform
accordingly.  ``frozen_chi_pairs`` in ``_ctm_diagnostics`` carries the same
warning from the same lesson.

Two guards live here:

* ``test_the_2x2_recipe_corner_is_not_rank_1`` -- the positive control, swept
  across chi.  It must keep passing; if the *working* recipe ever returns a
  rank-1 corner it has regressed into the same failure mode.
* ``test_single_site_split_honours_the_default_2x2_recipe`` -- the production
  guard.  ``gs_recipe="2x2"`` is the default, and the single-site split path
  used to run ``1x1`` moves unconditionally, so before #726 the default config
  silently returned mean-field numbers.  #726 made that raise; #746 routed the
  path through the 2x2 plaquette sweep, so it now has to *run* and return a
  rank>1 corner.  ``test_single_site_split_still_accepts_1x1_for_bisection``
  keeps the opt-in legacy path honest.

The ``1x1`` collapse itself is deliberately *not* asserted as expected
behaviour here.  Three existing tests were shaped around that symptom (see
#726); pinning it again would entrench the bug.  Fixing the projector is
tracked separately.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import _ctm_tensor_multisite
from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

# A 1-site unit cell expressed as a multisite problem, so the same entry point
# serves both recipes (``ctm_tensor`` has no ``recipe`` knob -- it is the fused
# 1x1 path by construction).
_ORIGIN = (0, 0)
_NEIGHBORS = {
    _ORIGIN: {"left": _ORIGIN, "right": _ORIGIN, "top": _ORIGIN, "bottom": _ORIGIN}
}


@pytest.fixture(scope="module")
def _su_state():
    """A physical simple-update state -- not a random tensor.

    The collapse is only meaningful on a state with genuine corner structure:
    the 2x2 spectrum of this one is ``1, 0.128, 0.127, 0.016, 2.1e-3, 2.0e-3``,
    whose degenerate pairs are the expected Neel structure.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=60,
        dt=0.05,
        unit_cell="1x1",
        ctm=CTMConfig(chi=8, max_iter=100, conv_tol=1e-10),
    )
    _E, tensors, _envs = ipeps(gate, None, cfg)
    return tensors[0], heisenberg_gate()


def _env_at(A, chi, recipe):
    envs = _ctm_tensor_multisite(
        {_ORIGIN: A},
        _NEIGHBORS,
        chi=chi,
        max_iter=200,
        conv_tol=1e-12,
        recipe=recipe,
    )
    return envs[_ORIGIN]


@pytest.mark.parametrize("chi", [4, 8, 16])
def test_the_2x2_recipe_corner_is_not_rank_1(chi, _su_state):
    """Positive control: the working recipe must not collapse, at any chi.

    This replaces ``test_the_2x2_recipe_energy_is_not_frozen_in_chi``, which
    asserted ``E(chi_lo) != E(chi_hi)`` -- bit-inequality -- as a proxy for "the
    boundary is doing something".  That proxy has no discriminating power here,
    and the parametrisation was decided by rounding.  Measured on this fixture::

        chi  corner_rank  energy (repr)
          4      4        0.49912538691954667
          8      6        0.49912538691954667
         16      6        0.49912538691954655

        chi 4->8 : delta = 0.0            bit-identical  -> old test FAILED
        chi 8->16: delta = 1.110223e-16   = 2.000 ULP    -> old test PASSED

    The corner is rank 4/6/6 -- never 1 -- so the collapse the old message
    announced was simply absent.  ``[8-16]`` passed on a **two-ULP** difference:
    not a detection, just noise that happened to move the last bit.  Which way
    each case fell was therefore a property of the platform's floating point,
    and CI split accordingly (ubuntu-3.11 and macOS-3.12 red, ubuntu-3.12
    green).  The old docstring's "real 2x2 movement ... is ~2.3e-7" is off by
    seven orders of magnitude against this state.

    The reason no threshold would have rescued it: **energy flatness cannot
    discriminate.**  A collapsed environment gives a chi-independent *wrong*
    energy; a converged one gives a chi-independent *right* energy.  Both are
    flat, so the signal is empty in both directions.  ``frozen_chi_pairs`` in
    ``_ctm_diagnostics`` already documents this ("a fully converged environment
    is flat in chi too -- that is what convergence means") and states that
    ``ctm_corner_rank`` is the only sound detector.  That is what this asserts,
    now swept across chi so it still covers the range the old parametrisation
    was reaching for.
    """
    A, _gate = _su_state
    env = _env_at(A, chi=chi, recipe="2x2")
    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    rank = int((s > 1e-10 * s[0]).sum())
    assert rank > 1, (
        f"2x2 corner collapsed to rank {rank} at chi={chi} (spectrum {s[:6]}); "
        "this is the #726 signature on the recipe that is supposed to work."
    )


@pytest.mark.parametrize("chi", [4, 8, 16])
def test_the_1x1_recipe_collapses_here_at_every_chi(chi, _su_state):
    """Negative control: the detector above must be *able* to fail.

    Same helper, same state, same chi -- only the recipe differs::

        recipe  chi     corner_rank  spectrum[:4]
        1x1     4/8/16       1       [1.      0.      0.      0.     ]
        2x2     4            4       [1.      0.1276  0.1266  0.0164 ]
        2x2     8/16         6       [1.      0.1276  0.1266  0.0164 ]

    The ``1x1`` spectrum is *exactly* ``[1, 0, 0, 0]``, and ``2x2`` shows the
    expected Neel degenerate pairs.  So ``rank > 1`` is discriminating at every
    chi rather than passing for incidental reasons -- which is the property the
    energy-flatness assertion it replaced never had, at any chi.

    This overlaps ``test_single_site_split_still_accepts_1x1_for_bisection`` in
    intent, but reaches the collapse through ``_ctm_tensor_multisite`` directly
    rather than through ``optimize_gs_ad``, so it pins the projector itself
    rather than the production wiring around it.
    """
    A, _gate = _su_state
    env = _env_at(A, chi=chi, recipe="1x1")
    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    rank = int((s > 1e-10 * s[0]).sum())
    assert rank == 1, (
        f"1x1 corner is rank {rank} at chi={chi} (spectrum {s[:6]}), not 1. "
        "If the 1x1 projector has been fixed this is good news, but the #726 "
        "documentation in this module and the positive control's negative "
        "control both go stale -- update them together."
    )


def _split_cfg(gs_num_steps):
    from tenax.algorithms.ipeps_config import iPEPSConfig as Cfg

    return Cfg(
        max_bond_dim=2,
        num_imaginary_steps=2,
        dt=0.05,
        unit_cell="1x1",
        gs_num_steps=gs_num_steps,
        ctm=CTMConfig(chi=4, max_iter=20, conv_tol=1e-8, fuse_virtual_legs=False),
    )


@pytest.mark.parametrize("gs_num_steps", [1, 0])
def test_single_site_split_honours_the_default_2x2_recipe(gs_num_steps):
    """The default config must actually *run* 2x2 on the single-site split path.

    History: ``gs_recipe`` defaults to ``"2x2"`` while the single-site split
    path ran the ``1x1`` moves unconditionally, so a default-configured
    ``fuse_virtual_legs=False`` run returned a chi_eff=1 mean-field energy with
    no indication anything was wrong.  #726 made that *raise* rather than lie.
    #746 then routed the path through the 2x2 plaquette sweep, so the honest
    behaviour is now to run it.

    ``gs_num_steps=0`` is the evaluate-only case: ``loss_fn`` is never invoked
    and the final ``_eval_fresh`` reaches ``_split_forward`` ->
    ``converge_split_env`` directly.  That path used to bypass a loss-only
    guard (Codex review, PR #749), so it is still covered here — it must now
    succeed by the same route it used to escape through.
    """
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = _split_cfg(gs_num_steps)
    assert cfg.gs_recipe == "2x2", "precondition: 2x2 is the default recipe"

    _E, tensors, _envs = ipeps(gate, None, cfg)
    _A_opt, envs, E = optimize_gs_ad(gate, tensors[0], cfg)
    assert np.isfinite(float(E)), f"split 2x2 optimize returned {E!r}"

    # The point of the recipe change: a non-collapsed environment.
    env = envs[(0, 0)] if isinstance(envs, dict) else envs
    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    rank = int((s > 1e-10 * s[0]).sum())
    assert rank > 1, (
        f"single-site split env still rank {rank} under gs_recipe='2x2' "
        f"(spectrum {s[:6]}); the #726 collapse survived the #746 reroute"
    )


def test_single_site_split_still_accepts_1x1_for_bisection():
    """``gs_recipe='1x1'`` stays reachable, and still collapses.

    Kept so the regression-bisection path does not rot: if this starts
    returning a rank>1 corner, the 1x1 projector itself changed and the
    surrounding #726/#746 documentation is stale.
    """
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = _split_cfg(1)
    cfg = dataclasses.replace(cfg, gs_recipe="1x1")

    _E, tensors, _envs = ipeps(gate, None, cfg)
    _A_opt, envs, E = optimize_gs_ad(gate, tensors[0], cfg)
    assert np.isfinite(float(E))
    env = envs[(0, 0)] if isinstance(envs, dict) else envs
    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    assert int((s > 1e-10 * s[0]).sum()) == 1
