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

One guard lives here: ``test_single_site_split_honours_the_default_2x2_recipe``
-- the production guard.  ``gs_recipe="2x2"`` is the default, and the
single-site split path used to run ``1x1`` moves unconditionally, so before
#726 the default config silently returned mean-field numbers.  #726 made that
raise; #746 routed the path through the 2x2 plaquette sweep, so it now has to
*run* and return a rank>1 corner.

The rank detector's positive control (2x2 corner rank > 1 and growing with
chi), its negative control (1x1 rank 1 at every chi) and the legacy-1x1
reachability check that used to live here were removed in the slow-bucket
prune (#1149): ``test_ctm_723_single_site_collapse.py`` and
``test_split_ctm_746_single_site_collapse.py`` (both core) assert the same
claims on the fused and split paths, including the rank growth with chi, and
``test_recipe_1x1_deprecation.py`` pins the opt-in legacy recipe.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig


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

    _E, tensors, _envs = ipeps(gate, None, cfg, compute_energy=False)
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
