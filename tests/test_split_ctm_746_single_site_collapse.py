"""Regression: single-site ``ctm_split_tensor`` must not collapse (#746).

The split half of #723/#726.  The ``1x1`` recipe's corner-pair projector is
shared *verbatim* by the fused and split single-site paths — ``M = C1g^H C4g``
is ``chi x chi``, so the ``chi * D**2`` seam is summed away and
``rank(P) <= rank(C1g)``, which is 1 at the cold rank-1 seed.  Rank-1 is
therefore absorbing, and the environment is a chi_eff = 1 mean-field boundary
rather than a corner transfer matrix.

#723 fixed the *fused* entry point (``ctm_tensor``).  ``ctm_split_tensor`` was
left on the collapsing ``1x1`` moves, so it still returned a chi-frozen energy.
Measured on the D=2 sublattice-rotated Heisenberg simple-update state:

==================  ====================  ====================  =========
path                chi=4                 chi=16                rank(C1)
==================  ====================  ====================  =========
split ``"1x1"``     0.49620072949960814   *bit-identical*       1
split ``"2x2"``     0.4991254745638001    0.49912538701724773   4 -> 6
fused ``"2x2"``     0.4991253869195439    0.4991253869195441    4 -> 6
==================  ====================  ====================  =========

The split ``2x2`` corner spectrum reproduces the fused one digit-for-digit
(``1, 0.12764, 0.12659, 0.01638, 0.00208, 0.00202``) and the energies agree to
1e-10 at chi=16, so the two paths converge to the same fixed point — which is
the whole point of the split representation.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
from tenax.algorithms._split_ctm_tensor_energy import compute_energy_split_ctm_tensor
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
        unit_cell="1x1",
        ctm=CTMConfig(chi=8, max_iter=100, conv_tol=1e-10),
    )
    _E, tensors, _envs = ipeps(gate, None, cfg)
    return tensors[0], heisenberg_gate()


def _corner_rank(env, tol=1e-10):
    s = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
    return int((s / (s[0] + 1e-300) > tol).sum())


def test_split_env_is_not_rank_one(su_state):
    """A rank-1 corner is a chi_eff=1 mean-field boundary, not a CTM env."""
    A, _ = su_state
    env = ctm_split_tensor(A, chi=8, max_iter=100, conv_tol=1e-12)
    rank = _corner_rank(env)
    assert rank > 1, (
        f"single-site split-CTM env collapsed to rank {rank} (#746): the "
        f"corner spectrum is [1, 0, 0, ...], i.e. a product environment"
    )


def test_split_energy_moves_with_chi(su_state):
    """#747's cheap detector: a chi-frozen energy is a broken environment."""
    A, gate = su_state
    env_lo = ctm_split_tensor(A, chi=4, max_iter=100, conv_tol=1e-12)
    env_hi = ctm_split_tensor(A, chi=16, max_iter=100, conv_tol=1e-12)
    E_lo = float(compute_energy_split_ctm_tensor(A, env_lo, gate))
    E_hi = float(compute_energy_split_ctm_tensor(A, env_hi, gate))
    assert E_lo != E_hi, (
        f"split energy bit-identical across a 4x change in chi ({E_lo!r}); "
        f"that is the rank-1 collapse signature (#726/#746), not convergence"
    )


def test_split_matches_fused_oracle(su_state):
    """The split path must reach the same fixed point as the fused one.

    This is the *non-circular* oracle: the fused side runs its own (2x2)
    default rather than the shared broken projector, so agreement is a real
    physics check, unlike the split-1x1-vs-fused-1x1 comparison #746 flags.
    """
    A, gate = su_state
    chi = 8
    env_split = ctm_split_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12)
    env_fused, _ = ctm_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12)
    E_split = float(compute_energy_split_ctm_tensor(A, env_split, gate))
    E_fused = float(compute_energy_ctm_tensor(A, env_fused, gate))
    assert E_split == pytest.approx(E_fused, rel=1e-7), (
        f"split-CTM energy {E_split!r} != fused 2x2 oracle {E_fused!r}"
    )


def test_split_corner_spectrum_matches_fused(su_state):
    """The stronger form: the whole corner spectrum, not just the energy."""
    A, _ = su_state
    chi = 8
    env_split = ctm_split_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12)
    env_fused, _ = ctm_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12)
    s_split = np.linalg.svd(np.asarray(env_split.C1.todense()), compute_uv=False)
    s_fused = np.linalg.svd(np.asarray(env_fused.C1.todense()), compute_uv=False)
    n = min(len(s_split), len(s_fused))
    np.testing.assert_allclose(
        s_split[:n] / s_split[0], s_fused[:n] / s_fused[0], atol=1e-6
    )


def test_legacy_1x1_recipe_still_reachable(su_state):
    """``recipe='1x1'`` is kept for regression bisection, and still collapses."""
    A, _ = su_state
    env = ctm_split_tensor(A, chi=8, max_iter=100, conv_tol=1e-12, recipe="1x1")
    assert _corner_rank(env) == 1, (
        "the legacy 1x1 recipe is expected to still collapse; if this now "
        "passes, the 1x1 projector itself was fixed and this test is stale"
    )


def test_unknown_recipe_raises(su_state):
    A, _ = su_state
    with pytest.raises(ValueError, match="Unknown split CTM recipe"):
        ctm_split_tensor(A, chi=4, max_iter=1, recipe="3x3")
