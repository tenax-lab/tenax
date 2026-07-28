"""Multisite cell-size consistency of the CTM recipes (``core`` bucket).

``gs_recipe`` selects the CTM projector scheme: ``"2x2"`` (Fishman plaquette,
the :class:`iPEPSConfig` default) or ``"1x1"`` (reduced-corner single-site).
Both are production paths, but only ``"1x1"`` had coverage --
``test_split_ctm_production_correctness`` pins ``"1x1"`` energies, the kagome
multisite tests skip without a saved ``logs/d4_ad_optimum.npz``, and the
bipartite ``E < -0.66`` checks in ``test_ipeps.py`` exercise simple update
rather than the AD path.  So the config *default* recipe had no green test.

This module is the cheap, forward-only half of closing that gap; the
end-to-end physical anchor lives in
``test_ctm_recipe_2x2_production_correctness.py`` (slow bucket).  They are
kept in separate files on purpose: ``conftest.py`` applies its file-level
marker to every test in a file, so a ``slow`` test sharing this file would
also inherit ``core`` and would then run in required CI.

Context: issues #676 / #702.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
    make_neighbors,
)
from tenax.algorithms.ipeps import heisenberg_gate
from tests.test_split_ctm_fuse_flag import _make_site


@pytest.mark.parametrize("recipe", ["2x2", "1x1"])
def test_uniform_tiling_consistency(recipe):
    """A uniform tiling must give the same per-site energy at every cell size.

    Uses a *direction-dependent* random ``A`` (``A.l != A.r``), not a C4v
    tensor: a symmetric tensor can mask a left/right or up/down bond mix-up,
    which is precisely the failure mode this guards.
    """
    # chi and max_iter are deliberately small: this asserts a *structural*
    # invariant (cell-size independence), not an accurate energy, and all
    # three cells run the identical fixed-point iteration, so the comparison
    # is exact whether or not the CTM has fully converged.
    D, d, chi = 2, 2, 4
    A = _make_site(D, d, seed=0)
    gate = heisenberg_gate()

    def energy(site_tensors, neighbors):
        e = ctm_energy_implicit(
            site_tensors,
            neighbors,
            gate,
            chi=chi,
            max_iter=30,
            conv_tol=1e-12,
            recipe=recipe,
        )
        return float(jnp.real(e))

    e1 = energy({(0, 0): A}, SINGLE_SITE_NEIGHBORS)
    e2 = energy({(0, 0): A, (1, 0): A}, CHECKERBOARD_NEIGHBORS)
    e4 = energy(
        {(0, 0): A, (1, 0): A, (0, 1): A, (1, 1): A},
        make_neighbors(2, 2),
    )

    worst = max(abs(e1 - e2), abs(e1 - e4), abs(e2 - e4))
    assert worst < 1e-10, (
        f"recipe={recipe!r}: uniform tiling is cell-size dependent -- "
        f"multisite bond bookkeeping is broken. "
        f"E(1-site)={e1:.12f} E(2-site)={e2:.12f} E(2x2)={e4:.12f} "
        f"worst gap={worst:.3e}"
    )
