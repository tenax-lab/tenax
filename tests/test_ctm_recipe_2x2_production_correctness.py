"""End-to-end physical anchor for the ``gs_recipe="2x2"`` CTM path (slow).

``gs_recipe="2x2"`` (Fishman plaquette) is the :class:`iPEPSConfig` default,
yet before this test no green test pinned a physical energy through it --
``test_split_ctm_production_correctness`` uses ``"1x1"``, the kagome multisite
tests skip without a saved ``logs/d4_ad_optimum.npz``, and the bipartite
``E < -0.66`` checks in ``test_ipeps.py`` exercise simple update rather than
the AD path.

The cheap forward-only half of this coverage lives in
``test_ctm_recipe_2x2_consistency.py`` (``core`` bucket).  That test proves
the multisite machinery is *self-consistent* across cell sizes; it cannot
catch a scheme that is self-consistent but converges to the wrong fixed
point.  This test closes that hole by optimizing a genuine bipartite cell and
pinning the result inside the QMC variational band.

Kept in a separate file from the consistency test on purpose: ``conftest.py``
applies its file-level marker to every test in a file, so a ``slow`` test
sharing the ``core``-registered file would also inherit ``core`` and would
then run in required CI.

Reference values (D=2, chi=8, implicit AD, A100):
  recipe="2x2": E/site = -0.658779   (634s)
  recipe="1x1": E/site = -0.659967  (1997s)
both with ``||A-B||/||A|| = sqrt(2)`` (orthogonal sublattices), against the
Sandvik QMC ground state -0.6694 and the repo's ``fused+c4v = -0.6601``
anchor.  The ~1e-3 spread between recipes is expected: they are different
truncation schemes and need not agree exactly at finite chi.

Runtime note: this takes ~100s on CPU, i.e. ~6x *faster* than the 634s A100
run above.  At D=2 / chi=8 there is not enough work per kernel to amortize
GPU launch overhead, so the step is host-orchestration-bound (same regime as
#566 / #618).  CI runs the slow bucket on CPU, which is the fast path here.

Context: issues #676 / #702.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tests.test_split_ctm_fuse_flag import _make_site

QMC_FLOOR = -0.6694
ORDERED_CEIL = -0.60


@pytest.mark.slow
def test_bipartite_checkerboard_physical_energy():
    """recipe="2x2" on a genuine bipartite cell must land in the QMC band.

    ``unit_cell="2site"`` drives the real multisite production path: it builds
    ``site_tensors={(0,0): A, (1,0): B}`` over ``CHECKERBOARD_NEIGHBORS`` with
    ``recipe=config.gs_recipe``, so the CTM carries two *distinct* sublattice
    tensors rather than a uniform tiling.  The gate stays unrotated -- with
    ``gs_c4v=True`` the 2-site path derives B from A by sublattice rotation on
    the physical leg, so Neel order lives in the A/B distinction.
    """
    D, d = 2, 2
    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(
            chi=8,
            chi_I=8,
            fuse_virtual_legs=True,
            max_iter=60,
            conv_tol=1e-9,
            min_iter=4,
        ),
        unit_cell="2site",
        gs_recipe="2x2",
        gs_implicit_ad=True,
        gs_c4v=True,
        gs_metric_precond=False,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-3,
        gs_num_steps=40,
        gs_log_interval=10,
        su_init=False,
    )

    gate = heisenberg_gate()
    AB = (_make_site(D, d, seed=3), _make_site(D, d, seed=11))
    (A_opt, B_opt), _envs, E = optimize_gs_ad(gate, AB, config)
    E = float(E)

    # Guard against a vacuous pass: if the optimizer collapsed A and B onto
    # each other this degenerates to a uniform tiling and proves nothing that
    # the consistency test does not already cover.
    A_d, B_d = A_opt.todense(), B_opt.todense()
    distinctness = float(jnp.linalg.norm(A_d - B_d) / jnp.linalg.norm(A_d))
    assert distinctness > 0.1, (
        f"sublattices collapsed (||A-B||/||A||={distinctness:.3e}); "
        "this is no longer a bipartite test"
    )

    assert QMC_FLOOR - 1e-3 <= E <= ORDERED_CEIL, (
        f"recipe='2x2' bipartite energy {E:.6f}/site is outside the physical "
        f"band [{QMC_FLOOR}, {ORDERED_CEIL}] -- either non-variational "
        f"(below QMC) or not ordered (above {ORDERED_CEIL})"
    )
