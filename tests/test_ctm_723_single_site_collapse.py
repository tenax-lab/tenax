"""Regression: single-site ``ctm_tensor`` must not collapse to a rank-1 env (#723).

The legacy ``1x1`` recipe builds its projector from the corner-pair cross
product ``M = C1g^H C4g``, which is ``chi x chi`` — the ``chi * D**2`` seam is
summed away, so ``rank(P) <= rank(C1g)``.  With the cold ``chi_init=1`` seed
(``_make_rank1_dense_corner`` plus the ``T[0, diag, 0]`` edge) that is 1, and
rank-1 is an *absorbing* state: the environment can never grow.

The signature is an energy that does not move with chi at all — bit-identical
across a 4x change — because a rank-1 corner is a chi_eff = 1 mean-field
boundary rather than a corner transfer matrix.  Nothing raises: the collapsed
environment is still finite, Hermitian and PSD, so it is silently wrong.

Switching the projector does **not** rescue the ``1x1`` recipe.  Measured on the
D=2 sublattice-rotated Heisenberg simple-update state, ``2x2`` gives
``-0.488638504625`` at every chi, while ``1x1`` gives:

===========  ==============  ==============  ==============
projector    chi=4           chi=8           chi=16
===========  ==============  ==============  ==============
``"svd"``    -0.485745       -0.485745       -0.485745      (bit-identical, rank 1)
``"eigh"``   -0.530886       -0.723684       -0.272534      (full rank, divergent)
``"qr"``     -0.500077       -0.527835       -0.301451      (full rank, divergent)
===========  ==============  ==============  ==============

``eigh``/``qr`` escape the rank collapse but are wildly non-convergent, and
-0.7237 is below the exact ground-state energy of this model.  So the defect is
the recipe, not the projector, and the fix routes the single-site path through
the 2x2 plaquette projector that the multisite path already uses.

The oracle here is ``ctm_tensor_2site(A, A)``: a uniform 1-site lattice is
exactly the checkerboard with both sublattices equal, and the 2x2 recipe is the
known-good path (#726, #746, #747).
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor, ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.fixture(scope="module")
def su_state():
    """Physical D=2 simple-update Heisenberg state (the #726/#747 reproducer)."""
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=60,
        dt=0.05,
        # This CTM is dead weight: ``ipeps()`` runs simple update first and the
        # tensors are fixed before it starts, so ``config.ctm`` cannot affect
        # them -- and this fixture discards both the energy and the env.  It
        # was spending its whole budget without converging (the
        # "CTM did not converge in ipeps()" warning), then throwing the
        # result away.  chi is unchanged; only the sweep count is cut (#933).
        ctm=CTMConfig(chi=8, max_iter=2),
    )
    _, (A, _B), _ = ipeps(gate, None, cfg)
    return A, gate


def _corner_rank(env, tol=1e-10):
    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    return int((s / (s[0] + 1e-300) > tol).sum())


def test_single_site_env_is_not_rank_one(su_state):
    """A rank-1 corner is a chi_eff=1 mean-field boundary, not a CTM env."""
    A, _ = su_state
    env, _ = ctm_tensor(A, chi=8, max_iter=100, conv_tol=1e-12)
    rank = _corner_rank(env)
    assert rank > 1, (
        f"single-site CTM env collapsed to rank {rank} (#723): the corner "
        f"spectrum is [1, 0, 0, ...], i.e. a product environment"
    )


def test_chi_frozen_energy_is_only_a_bug_together_with_a_rank_1_corner(su_state):
    """The environment must *grow* with chi; the energy need not.

    Two corrections are baked into this test, both found by required CI on
    macOS after these files were promoted to `core`:

    1. The original asserted ``E(chi=4) != E(chi=16)`` on 2x2.  False for a
       correct implementation -- a converged environment is flat in chi, and on
       this D=2 state 2x2 converges by chi=4.  It failed on macOS reporting the
       *correct* value.
    2. The replacement asserted the 1x1 energy was bit-identical across chi.
       Also not portable: the split 1x1 energies differ by 1 ULP on macOS
       (0.49620072949960814 vs ...803) while agreeing exactly on Linux.

    Energy-vs-chi cannot discriminate on this fixture in *either* direction.
    The portable statement is about the corner rank, which is what the collapse
    actually destroys:

    ==========  =====  =====  ======  ======
    recipe      chi=4  chi=8  chi=16  chi=24
    ==========  =====  =====  ======  ======
    ``"1x1"``   1      1      1       1
    ``"2x2"``   4      6      6       6
    ==========  =====  =====  ======  ======

    Rank-1 is *absorbing* under 1x1 -- the environment can never grow, at any
    chi.  Under 2x2 it grows until it saturates at the true environment rank.
    Singular values here are either ~1e-17 or >=2e-3, far from the 1e-10
    threshold, so the ranks are robust to last-bit platform differences.
    """
    A, _gate = su_state

    def rank_at(chi, recipe):
        env, _ = ctm_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12, recipe=recipe)
        return _corner_rank(env)

    assert rank_at(4, "1x1") == 1 and rank_at(16, "1x1") == 1, (
        "the 1x1 corner is expected to stay rank-1 at every chi (rank-1 is an "
        "absorbing state); if it now grows, the projector was fixed and this "
        "whole file is stale"
    )

    r4, r16 = rank_at(4, "2x2"), rank_at(16, "2x2")
    assert r4 > 1, f"2x2 corner collapsed at chi=4 (rank {r4})"
    assert r16 > r4, (
        f"2x2 environment did not grow with chi (rank {r4} -> {r16}); a real "
        f"corner transfer matrix gains boundary entanglement as chi rises"
    )


def test_single_site_matches_2site_oracle(su_state):
    """A uniform 1-site lattice IS the checkerboard with both sublattices equal."""
    A, gate = su_state
    chi = 8
    env_1, _ = ctm_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12)
    env_A, _env_B = ctm_tensor_2site(A, A, chi=chi, max_iter=100, conv_tol=1e-12)
    E_1 = float(compute_energy_ctm_tensor(A, env_1, gate, d=2))
    E_2 = float(compute_energy_ctm_tensor(A, env_A, gate, d=2))
    assert E_1 == pytest.approx(E_2, rel=1e-6), (
        f"single-site CTM energy {E_1!r} != 2x2 oracle {E_2!r}"
    )


def test_legacy_1x1_recipe_still_reachable(su_state):
    """``recipe='1x1'`` is kept for regression bisection, and still collapses."""
    A, _ = su_state
    env, _ = ctm_tensor(A, chi=8, max_iter=100, conv_tol=1e-12, recipe="1x1")
    assert _corner_rank(env) == 1, (
        "the legacy 1x1 recipe is expected to still collapse; if this now "
        "passes, the 1x1 projector itself was fixed and this test is stale"
    )
