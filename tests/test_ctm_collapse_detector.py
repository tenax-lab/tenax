"""The #747 collapse detectors: rank-1 corners and chi-frozen energies.

#747 audited four benchmark campaigns that reported a collapsed ``1x1``
environment as a convergence success.  Two independent detectors would have
caught every one of them, and both are cheap enough to run unconditionally:

* the *direct* check -- one SVD of the ``chi x chi`` ``C1`` corner;
* the *indirect* check -- bit-identical energies across a chi scan.

The D=8 split arm of PR #650 reported an energy "identical to 13 digits across
chi 48 -> 384" and read it as clean convergence; it was a rank-1 boundary.
"""

import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_diagnostics import (
    CollapsedEnvironmentError,
    check_ctm_env,
    ctm_corner_rank,
    env_is_collapsed,
    frozen_chi_pairs,
)
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.fixture(scope="module")
def su_state():
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=60,
        dt=0.05,
        unit_cell="1x1",
        ctm=CTMConfig(chi=8, max_iter=100, conv_tol=1e-10),
    )
    _E, tensors, _envs = ipeps(gate, None, cfg)
    return tensors[0]


# ------------------------------------------------------------------ #
# Direct detector: corner rank                                         #
# ------------------------------------------------------------------ #


def test_detects_the_collapsed_1x1_environment(su_state):
    """The whole point: ``1x1`` must be reported as collapsed."""
    env, _ = ctm_tensor(su_state, chi=8, max_iter=100, conv_tol=1e-12, recipe="1x1")
    assert ctm_corner_rank(env) == 1
    assert env_is_collapsed(env)


def test_passes_the_working_2x2_environment(su_state):
    """And the working recipe must not be flagged -- no false positive."""
    env, _ = ctm_tensor(su_state, chi=8, max_iter=100, conv_tol=1e-12)
    assert ctm_corner_rank(env) > 1
    assert not env_is_collapsed(env)


def test_works_on_the_split_env_too(su_state):
    """``SplitCTMTensorEnv`` exposes ``C1`` as well, so one helper covers both."""
    collapsed = ctm_split_tensor(
        su_state, chi=8, max_iter=100, conv_tol=1e-12, recipe="1x1"
    )
    healthy = ctm_split_tensor(su_state, chi=8, max_iter=100, conv_tol=1e-12)
    assert env_is_collapsed(collapsed)
    assert not env_is_collapsed(healthy)


def test_check_warns_and_returns_rank(su_state):
    env, _ = ctm_tensor(su_state, chi=8, max_iter=100, conv_tol=1e-12, recipe="1x1")
    with pytest.warns(RuntimeWarning, match="collapsed to a rank-1 corner"):
        rank = check_ctm_env(env, context="D=2 chi=8 unit-test")
    assert rank == 1


def test_check_is_silent_on_a_healthy_env(su_state):
    env, _ = ctm_tensor(su_state, chi=8, max_iter=100, conv_tol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        rank = check_ctm_env(env, context="healthy")
    assert rank > 1


def test_strict_mode_raises(su_state):
    env, _ = ctm_tensor(su_state, chi=8, max_iter=100, conv_tol=1e-12, recipe="1x1")
    with pytest.raises(CollapsedEnvironmentError, match="chi_eff=1"):
        check_ctm_env(env, context="strict", strict=True)


def test_context_appears_in_the_message(su_state):
    """A sweep has to say *which* cell collapsed, not just that one did."""
    env, _ = ctm_tensor(su_state, chi=8, max_iter=100, conv_tol=1e-12, recipe="1x1")
    with pytest.raises(CollapsedEnvironmentError, match=r"\[D=8 chi=384 split\]"):
        check_ctm_env(env, context="D=8 chi=384 split", strict=True)


def test_rejects_a_non_env():
    with pytest.raises(TypeError, match="exposing 'C1'"):
        ctm_corner_rank(object())


# ------------------------------------------------------------------ #
# Indirect detector: chi-frozen energies                               #
# ------------------------------------------------------------------ #


def test_frozen_scan_is_flagged():
    """The recorded D=8 split signature: identical E across a 8x change in chi."""
    scan = {48: -0.60053487745, 96: -0.60053487745, 384: -0.60053487745}
    pairs = frozen_chi_pairs(scan)
    assert (48, 96) in pairs and (48, 384) in pairs and (96, 384) in pairs


def test_a_converging_scan_is_not_flagged():
    """The real D=4 scan (PR #646, measured at 2x2) must stay clean.

    These are the recorded values; they differ in the last digits at every
    chi, which is what a genuine corner transfer matrix does.
    """
    scan = {
        16: -0.6633791865197316,
        24: -0.6633937000473232,
        32: -0.6633967779107297,
        48: -0.6633979620602313,
        64: -0.6633980735550820,
        96: -0.6633980947093154,
        128: -0.6633980966128770,
    }
    assert frozen_chi_pairs(scan) == []


def test_near_identical_is_not_flagged():
    """Exact equality only: agreeing to 12 digits is convergence, not collapse.

    From the #638 showcase, -0.4953177474 recurs at chi=16 and chi=48 but is
    not bit-identical.  Flagging that would make the detector cry wolf on every
    well-converged scan.
    """
    scan = {16: -0.49531774742346624, 48: -0.495317747423358}
    assert frozen_chi_pairs(scan) == []


def test_accepts_pairs_and_skips_none():
    pairs = frozen_chi_pairs([(8, -1.5), (16, None), (32, -1.5)])
    assert pairs == [(8, 32)]


def test_single_point_scan_is_vacuously_clean():
    assert frozen_chi_pairs({8: -1.0}) == []
    assert frozen_chi_pairs({}) == []


def test_frozen_scan_alone_can_false_positive_and_rank_disambiguates(su_state):
    """Documented limitation: a *converged* scan is flat too.

    `frozen_chi_pairs` is the indirect detector and cannot tell "collapsed"
    from "converged" on its own -- both are flat in chi.  On this D=2 state the
    2x2 energy saturates by chi=4, so a chi=4/chi=8/chi=16 scan can be
    bit-identical while the environment is perfectly healthy.

    The conjunction is what indicts: frozen AND rank-1.  This test pins the
    limitation so nobody re-derives it the hard way -- a #723 regression test
    asserted the chi response alone and failed on macOS while reporting the
    correct converged energy.

    It errs the *other* way too, which cannot be pinned portably here: a
    collapsed environment is not reliably bit-identical either.  The split
    ``1x1`` path returns ``0.49620072949960814`` at chi=4 and ``...803`` at
    chi=16 on macOS while agreeing exactly on Linux, so exact equality can
    miss a genuine collapse differing in the last ULP.  Only
    :func:`ctm_corner_rank` is sound in both directions.
    """
    A = su_state
    gate = heisenberg_gate()

    energies, ranks = {}, {}
    for chi in (4, 8, 16):
        env, _ = ctm_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12)
        energies[chi] = float(compute_energy_ctm_tensor(A, env, gate, d=2))
        ranks[chi] = ctm_corner_rank(env)

    # Healthy: every corner carries real boundary entanglement ...
    assert all(r > 1 for r in ranks.values()), ranks
    # ... yet if the energies happen to saturate exactly, the indirect detector
    # fires anyway.  That is the false positive, and it is fine *because* the
    # rank check overrules it.
    if frozen_chi_pairs(energies):
        assert min(ranks.values()) > 1, (
            "frozen + rank-1 would be a real collapse; this fixture is healthy"
        )
