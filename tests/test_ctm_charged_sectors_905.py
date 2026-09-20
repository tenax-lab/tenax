"""The symmetric CTM must keep its charged environment sectors (#905).

Before #905 the block-sparse CTM converged to an environment in which every
corner and edge block with a non-zero U(1) charge was allocated but exactly
``0.0``.  The environment was confined to the ``q = 0`` sector, which made the
energy wrong by ~0.16 at D=2 and — the tell — completely **flat in chi**: the
symmetric energy agreed to ten digits between chi=8 and chi=24 while the dense
energy kept moving.

Three independent flow-convention faults produced it, and each one on its own is
enough to annihilate the charged sectors, because ``_contract_symmetric`` pairs
blocks by charge *value* and drops any product whose output key falls outside
the output legs' conservation law:

1. ``_apply_proj_unfused`` flow-flipped the projector, but ``split_index``
   restores the fused leg's parents with their recorded flows, so the flip
   survived on ``chi_new`` alone and broke that tensor's conservation law.
2. Each edge's D² leg was declared with the *same* flow as the double-layer
   tensor's matching leg instead of the opposite one, so every ``T · a``
   contraction added the two flow-weighted charges instead of cancelling them.
3. Four of the initial environment's eight chi bonds were same-flow.

The guards here are on the physics, not on any of those three mechanisms: the
charged blocks carry weight, and the symmetric energy tracks the flow-insensitive
dense one.  ``tests/test_enlarged_corner_flow_invariant_834.py`` holds the
structural (per-bond duality) half.

**On the solver budget.**  These guards do NOT require a converged fixed
point, and pretending they did cost the required gate 15.5 minutes: the
sector annihilation is *structural* — conservation-law refusal kills the
charged products from the first absorption sweep — and the energy guard is a
same-budget PARITY between the symmetric and dense paths, not an absolute
value.  Scanned before freezing the budget (2026-09-10, this box):

    max_iter   sym-vs-dense gap   charged  dead  rank>n_q0
    60         2.2e-08            2        0     8>4
    20         8.3e-07            2        0     8>4
    10         7.4e-06            2        0     8>4
    6          2.3e-05            2        0     8>4

Every structural signal is budget-invariant, and the parity gap stays two
orders of magnitude inside the 5e-3 tolerance (four inside the 0.16 defect)
at every point of the scan.  ``_MAX_ITER`` below picks the scan point with a
3x margin to the tolerance-relevant one; the absolute-convergence oracle
(flatness in chi) lives in the ``slow`` test, unchanged, where convergence is
genuinely the property under test.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

# See the module docstring's scan table before changing either number.  The
# conv_tol is unreachable at this max_iter by design: the budget is the
# max_iter, and both paths run the SAME budget, which is what makes the
# energy comparison a parity rather than a convergence claim.
_MAX_ITER = 20
_CHI = 8


def _pair(D=2, seed=0):
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

    return heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))


def _converge(A, B, chi, max_iter=_MAX_ITER):
    from tenax.algorithms._ctm_tensor import ctm_tensor_2site

    return ctm_tensor_2site(
        A, B, chi=chi, recipe="2x2", max_iter=max_iter, conv_tol=1e-10
    )


@pytest.fixture(scope="module")
def swept_envs():
    """One symmetric and one dense sweep, shared by every test here.

    The two gate tests used to converge this identical fixture independently
    (same tensors, same seed, same chi), at 60 sweeps each — three ~4-minute
    block-sparse convergences where one symmetric + one (nearly free) dense
    sweep carries every guard.  The dense pair uses the same budget so the
    energy comparison stays same-system, same-budget.
    """
    from tenax.core.tensor import DenseTensor

    A, B = _pair()
    envA, envB = _converge(A, B, chi=_CHI)
    Ad = DenseTensor(A.todense(), A.indices)
    Bd = DenseTensor(B.todense(), B.indices)
    envAd, envBd = _converge(Ad, Bd, chi=_CHI)
    return A, B, envA, envB, Ad, Bd, envAd, envBd


def test_swept_corner_has_non_zero_charged_blocks(swept_envs):
    """C1's ``q != 0`` blocks must carry weight.

    This is the direct observable of the bug: pre-#905 those blocks existed in
    the block table with norm exactly 0.0, so ``rank(C1)`` equalled the number
    of charge-0 slots on its legs and nothing else.
    """
    _A, _B, envA, _envB, *_ = swept_envs
    C1 = envA.C1

    charged = {
        key: float(np.linalg.norm(np.asarray(block)))
        for key, block in C1.blocks.items()
        if any(int(q) != 0 for q in key)
    }
    assert charged, "C1 has no charged blocks at all — fixture is not charged"
    dead = [k for k, n in charged.items() if n <= 1e-12]
    assert not dead, (
        f"charged C1 blocks annihilated: {dead} (all norms: {charged}). The "
        "environment collapsed into the q=0 sector — see the module docstring."
    )

    # rank(C1) must exceed the number of q=0 slots; equality is the signature of
    # the confined environment.
    n_q0 = int(np.sum(np.asarray(C1.indices[0].charges) == 0))
    rank = int(np.linalg.matrix_rank(np.asarray(C1.todense()), tol=1e-10))
    assert rank > n_q0, f"rank(C1)={rank} == charge-0 slot count {n_q0}"


def test_symmetric_energy_matches_the_dense_reference(swept_envs):
    """``DenseTensor`` contraction ignores flow, so E_dense is the ground truth.

    The symmetric and dense paths are not expected to agree bit-for-bit — the
    block-sparse SVD allocates chi per charge sector rather than taking a global
    top-chi — but a confined environment misses by ~0.16, two orders of
    magnitude above the tolerance here, which itself sits three orders above
    the measured same-budget gap (8.3e-07 at ``_MAX_ITER=20``; scan table in
    the module docstring).
    """
    from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor_2site
    from tenax.algorithms.ipeps import heisenberg_gate

    A, B, envA, envB, Ad, Bd, envAd, envBd = swept_envs
    gate = heisenberg_gate().todense()

    E_sym = float(compute_energy_ctm_tensor_2site(A, B, envA, envB, gate, d=2))
    E_dense = float(compute_energy_ctm_tensor_2site(Ad, Bd, envAd, envBd, gate, d=2))

    assert np.isfinite(E_sym) and abs(E_sym) > 1e-6
    assert abs(E_sym - E_dense) < 5e-3, (
        f"E_sym={E_sym!r} vs E_dense={E_dense!r} (gap {abs(E_sym - E_dense):.3e})"
    )
    # Regime pin: at this budget the two paths agree far inside the guard's
    # tolerance, so a future gap NEAR the tolerance is already suspicious --
    # it means the fixture drifted (different budget, different init) rather
    # than that the guard is comfortably green (#884's fixture lesson).
    assert abs(E_sym - E_dense) < 1e-4, (
        f"gap {abs(E_sym - E_dense):.3e} is an order above the measured "
        "regime (8.3e-07); the fixture or budget drifted — re-run the scan "
        "in the module docstring before touching the 5e-3 guard."
    )


@pytest.mark.slow
def test_the_symmetric_energy_still_moves_with_chi():
    """Flatness in chi was the diagnostic that the sectors were dead.

    A confined environment has nothing left to resolve, so raising chi changes
    the energy by less than 1e-10.  A live one behaves like the dense path and
    keeps moving.  Convergence is genuinely the property under test here, so
    this keeps its own full-budget sweeps and stays ``slow``.
    """
    from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor_2site
    from tenax.algorithms.ipeps import heisenberg_gate

    A, B = _pair()
    gate = heisenberg_gate().todense()

    energies = []
    for chi in (8, 16):
        envA, envB = _converge(A, B, chi=chi, max_iter=60)
        energies.append(
            float(compute_energy_ctm_tensor_2site(A, B, envA, envB, gate, d=2))
        )

    assert abs(energies[1] - energies[0]) > 1e-9, (
        f"symmetric energy is flat in chi: {energies} — the environment is "
        "confined to the q=0 sector (#905)."
    )
