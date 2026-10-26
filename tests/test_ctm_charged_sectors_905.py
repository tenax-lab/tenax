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

    chi  max_iter   sym-vs-dense gap   charged  dead  rank>n_q0
    8    60         2.2e-08            2        0     8>4
    8    20         8.3e-07            2        0     8>4
    8    10         7.4e-06            2        0     8>4
    8    6          2.3e-05            2        0     8>4

Every structural signal is budget-invariant.  The cost is the block-sparse
compile of each new sector layout (~20 s per sweep here), not the sweeps, so
the test-trim pass (2026-10-26) re-scanned the cheap end, cold JAX cache,
together with the #905 defect re-introduced (the ``_flow_flip_no_conj`` of
``P_un`` put back in ``_apply_proj_unfused``):

    chi  max_iter   gap      dead  rank>n_q0   with the defect: gap  dead  rank
    8    3          4.5e-05  0     8>4                           -     -     -
    8    2          2.2e-04  0     7>4                         1.2e-02  2   4==4
    4    3          1.1e-16  0     4>2                           -     -     -
    4    2          0.0      0     4>2                         1.1e-02  2   2==2

``chi=4`` is ``D**2``, so the two-sweep environment is untruncated and the two
paths agree exactly, while the defect still kills both charged C1 blocks and
misses by ~1e-2 (above the 5e-3 guard and two orders above the 1e-4 regime
pin).  ``chi=8, max_iter=2`` would trip the regime pin (2.2e-04), which is why
the 2-site guards run at ``_CHI_2SITE=4``: 43 s symmetric instead of the
~6 minutes ``chi=8, max_iter=20`` cost.  (A slow flatness-in-chi scan used to
live here too; it was the diagnostic, not the guard, and was deleted in the
slow-bucket prune -- ``test_ctm_chi_truncation_policy_922.py`` pins converged
D=3 chi=16 parity to 1e-9, which no confined environment can satisfy.)
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

# See the module docstring's scan tables before changing any of these.  The
# conv_tol is unreachable at this max_iter by design: the budget is the
# max_iter, and both paths run the SAME budget, which is what makes the
# energy comparison a parity rather than a convergence claim.
_MAX_ITER = 2
#: The 2-site guards' chi (``D**2``; see the second scan table).
_CHI_2SITE = 4
#: The 1-site anchor's chi.
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
    envA, envB = _converge(A, B, chi=_CHI_2SITE)
    Ad = DenseTensor(A.todense(), A.indices)
    Bd = DenseTensor(B.todense(), B.indices)
    envAd, envBd = _converge(Ad, Bd, chi=_CHI_2SITE)
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

    The symmetric and dense paths are not expected to agree bit-for-bit once
    the cut truncates — the block-sparse SVD allocates chi per charge sector
    rather than taking a global top-chi — but a confined environment misses by
    ~0.16 at convergence (1.1e-02 at this fixture's two-sweep budget, still
    above the 5e-3 tolerance), while the measured same-budget gap here is 0.0
    (``chi = D**2`` is untruncated; scan tables in the module docstring).
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
        f"gap {abs(E_sym - E_dense):.3e} is far above the measured "
        "regime (0.0); the fixture or budget drifted — re-run the scan "
        "in the module docstring before touching the 5e-3 guard."
    )


# --------------------------------------------------------------------------- #
# The 1-site ``ctm_tensor`` path (the #853 control's subject)                   #
# --------------------------------------------------------------------------- #

#: ``compute_energy_ctm_tensor`` returns ``E_h + E_v`` for ``H = S_i . S_j``
#: (eigenvalues -3/4 and +1/4), so any genuine expectation value lies in
#: [-1.5, +0.5] however poorly converged the environment is.
_E_ATTAINABLE = (-1.5, 0.5)

#: Same-budget sweeps for the 1-site anchor.  Measured sym-vs-dense gap
#: (2026-10-26, CPU): 2.4e-09 at 3 sweeps, 6.2e-11 at 6, 5.6e-17 at 20 -- all
#: far inside the 1e-6 tolerance, which is itself far inside the ~0.16 #905
#: miss.  The cost is the block-sparse compile of each new sector layout
#: (70 s / 94 s / 120 s symmetric at 3 / 6 / 20 sweeps), so keep it small.
_SINGLE_SITE_SWEEPS = 4


def _rdm_min_eig_rel(rdm) -> float:
    """Smallest eigenvalue of a trace-normalised RDM, over its spectral radius."""
    M = np.asarray(rdm)
    M = M.reshape(M.shape[0] * M.shape[1], -1)
    M = M / np.trace(M)
    w = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
    return float(w.min()) / max(float(np.max(np.abs(w))), 1e-300)


def test_the_single_site_symmetric_ctm_matches_dense_and_keeps_the_rdm_a_state():
    """1-site ``ctm_tensor`` on a charged U(1)-Sz site: sym == dense, RDMs PSD.

    The tests above run ``ctm_tensor_2site``; this is the same guard on the
    1-site entry point, which the #853 control (deleted in the slow-bucket
    prune: one D=3 chi=12 run, ~96 min of slow-bucket time) was the only test
    of on a *charged* site -- the other 1-site sym-vs-dense checks
    (``test_ctm_tensor.py``, ``test_block_sparse_ctm_ad.py``) use trivial
    charges, i.e. a single block, which cannot drop a sector.

    Same-budget PARITY, as above: both paths run ``_SINGLE_SITE_SWEEPS`` sweeps
    at chi=8, so the comparison does not depend on reaching a fixed point (the
    sector annihilation #905 fixed is structural from the first absorption).
    The #905 defect missed by ~0.16, so the 1e-6 tolerance is loose against
    roundoff and tight against any dropped sector.  The RDM-positivity and
    attainable-energy assertions are #853's: an inaccurate environment still
    yields a density matrix, a block-pairing defect does not.
    """
    from tenax import compute_energy_ctm_tensor
    from tenax.algorithms._ctm_diagnostics import ctm_corner_rank
    from tenax.algorithms._ctm_tensor import ctm_tensor
    from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor, _rdm2x1_tensor
    from tenax.algorithms.ipeps import heisenberg_gate_u1sz
    from tenax.core.tensor import DenseTensor, SymmetricTensor

    A, _B = _pair()
    assert isinstance(A, SymmetricTensor) and len(A.blocks) > 1, "not charged"
    gate = heisenberg_gate_u1sz()

    n = _SINGLE_SITE_SWEEPS
    env, _ = ctm_tensor(A, chi=_CHI, max_iter=n, conv_tol=1e-10)
    Ad = DenseTensor(A.todense(), A.indices)
    env_d, _ = ctm_tensor(Ad, chi=_CHI, max_iter=n, conv_tol=1e-10)

    # Regime: a charged, non-collapsed environment (#723, #905).
    assert isinstance(env.C1, SymmetricTensor)
    chi_charges = set(np.asarray(env.C1.indices[0].charges).tolist())
    assert chi_charges - {0}, f"chi bond carries only q=0: {chi_charges}"
    assert ctm_corner_rank(env) > 1, "collapsed to a rank-1 corner (#723)"

    E_sym = float(compute_energy_ctm_tensor(A, env, gate))
    E_dense = float(compute_energy_ctm_tensor(Ad, env_d, gate))
    assert abs(E_sym - E_dense) < 1e-6, (
        f"1-site E_sym={E_sym!r} vs E_dense={E_dense!r} "
        f"(gap {abs(E_sym - E_dense):.3e}): the symmetric CTM drops weight "
        "the flow-insensitive dense contraction keeps (#905 class)."
    )
    lo, hi = _E_ATTAINABLE
    assert lo <= E_sym <= hi, f"E_sym={E_sym} is not attainable by any state (#853)"

    for name, rdm in (
        ("2x1 horizontal", _rdm2x1_tensor(A, env)),
        ("1x2 vertical", _rdm1x2_tensor(A, env)),
    ):
        min_eig = _rdm_min_eig_rel(rdm)
        assert min_eig >= -1e-8, (
            f"the {name} RDM of the symmetric 1-site CTM reaches {min_eig:.4e} "
            "of its spectral radius below zero: not a density matrix (#853)."
        )
