"""Detectors for the #726 rank-1 corner collapse on the ``1x1`` CTM recipe.

The ``1x1`` corner-pair (Fishman) projector -- shared *verbatim* by the fused
and split single-site paths, so this is not a split-CTM defect -- collapses the
environment to rank-1 corners.  ``_ctm_projector.py`` forms ``M = C1g† C4g``,
which is ``chi x chi``: the ``chi·D²`` seam is summed away, so ``M`` sees only
the outer chi legs.  Since ``C1_new = S^{1/2} U^H``, the updated corner's whole
spectrum is ``sqrt(spec(M))`` and ``rank(C_new) <= rank(C1g)``, which is 1 at
cold init.  The corner can never grow rank.

The cheapest possible detector for that is **energy frozen in chi**: a genuine
corner transfer matrix improves as the boundary grows, a chi_eff=1 mean-field
boundary does not.  On a physical D=2 SU state the ``1x1`` energy is
bit-identical from chi=2 to chi=32 (delta exactly 0.0), while the ``2x2``
recipe converges normally over the same range.

Two guards live here:

* ``test_the_2x2_recipe_energy_is_not_frozen_in_chi`` -- the positive control.
  It must keep passing; if it ever goes chi-frozen, the *working* recipe has
  regressed into the same failure mode.
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
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
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


def _energy_at(A, gate, chi, recipe):
    return float(compute_energy_ctm_tensor(A, _env_at(A, chi, recipe), gate))


@pytest.mark.parametrize("chi_lo,chi_hi", [(4, 8), (8, 16)])
def test_the_2x2_recipe_energy_is_not_frozen_in_chi(chi_lo, chi_hi, _su_state):
    """Positive control: a working recipe must respond to chi.

    Not an accuracy assertion -- only that the boundary is doing something.  A
    chi_eff=1 environment returns *bit-identical* energies across any chi range
    (delta exactly 0.0), which is what this catches.  The real 2x2 movement
    between chi=4 and chi=8 on this state is ~2.3e-7, so the threshold sits far
    below it and far above float noise.
    """
    A, gate = _su_state
    E_lo = _energy_at(A, gate, chi_lo, "2x2")
    E_hi = _energy_at(A, gate, chi_hi, "2x2")
    assert E_lo != E_hi, (
        f"2x2 energy is bit-identical at chi={chi_lo} and chi={chi_hi} "
        f"({E_lo!r}); the working recipe has regressed into the #726 "
        "chi_eff=1 collapse."
    )


def test_the_2x2_recipe_corner_is_not_rank_1(_su_state):
    """The direct form of the same check, on the corner itself."""
    A, _gate = _su_state
    env = _env_at(A, chi=8, recipe="2x2")
    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    rank = int((s > 1e-10 * s[0]).sum())
    assert rank > 1, (
        f"2x2 corner collapsed to rank {rank} (spectrum {s[:6]}); this is the "
        "#726 signature on the recipe that is supposed to work."
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
