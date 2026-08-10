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
        env = ctm_split_tensor(A, chi=chi, max_iter=100, conv_tol=1e-12, recipe=recipe)
        return _corner_rank(env)

    assert rank_at(4, "1x1") == 1 and rank_at(16, "1x1") == 1, (
        "the 1x1 split corner is expected to stay rank-1 at every chi; if it "
        "now grows, the projector was fixed and this whole file is stale"
    )

    r4, r16 = rank_at(4, "2x2"), rank_at(16, "2x2")
    assert r4 > 1, f"2x2 split corner collapsed at chi=4 (rank {r4})"
    assert r16 > r4, f"2x2 split environment did not grow with chi (rank {r4} -> {r16})"


def _split_fused_energies(A, gate, chi):
    env_split = ctm_split_tensor(A, chi=chi, max_iter=200, conv_tol=1e-12)
    env_fused, _ = ctm_tensor(A, chi=chi, max_iter=200, conv_tol=1e-12)
    return (
        float(compute_energy_split_ctm_tensor(A, env_split, gate)),
        float(compute_energy_ctm_tensor(A, env_fused, gate)),
    )


def test_split_matches_fused_oracle(su_state):
    """The split path must reach the same fixed point as the fused one.

    This is the *non-circular* oracle: the fused side runs its own (2x2)
    default rather than the shared broken projector, so agreement is a real
    physics check, unlike the split-1x1-vs-fused-1x1 comparison #746 flags.
    It is the standing validity check on split-CTM and stays armed until that
    path is stable.

    **Compared at chi=48, not chi=8** (#667).  Split and fused are different
    algorithms that truncate differently, so at finite chi they need not agree:
    measured 1.67e-05 at chi=8 falling to 9.62e-09 at chi=48, and the gap is
    identical to 12 digits at max_iter 100/400/2000, so it is truncation and not
    non-convergence.  The old chi=8 / rel=1e-7 form passed only because the
    simple-update state it ran on had collapsed to a near-product state, which
    made both paths trivially identical -- the same #667 defect that had
    ``test_2site_heisenberg_D2_energy`` recommending dt=0.3.

    At chi=48 >> D**2=4 the boundary is essentially exact, so both schemes must
    reproduce the *same* contraction.  That makes this a stronger oracle than
    the chi=8 version: any disagreement here is a defect, with no truncation
    left to excuse it.
    """
    A, gate = su_state
    E_split, E_fused = _split_fused_energies(A, gate, chi=48)
    assert E_split == pytest.approx(E_fused, rel=1e-7), (
        f"split-CTM energy {E_split!r} != fused 2x2 oracle {E_fused!r} at "
        f"chi=48, where truncation cannot explain it"
    )


def test_split_fused_gap_closes_with_chi(su_state):
    """A defect is flat in chi; truncation shrinks.  (#762's own standard.)

    The chi=48 oracle above would still pass if split-CTM were wrong by a fixed
    amount too small to see there.  This pins the *trend*, which is what
    separates a finite-chi truncation difference from a real disagreement.

    Deliberately **not** a monotonicity assertion: the measured gap is
    1.67e-05 / 2.49e-05 / 3.71e-06 / 1.72e-06 / 1.93e-06 / 9.62e-09 at
    chi = 8 / 12 / 16 / 24 / 32 / 48, which rises at chi=12 and again at 32.
    Only the endpoints are compared, with two orders of margin on a measured
    1.7e+03 ratio.
    """
    A, gate = su_state
    gaps = []
    for chi in (8, 48):
        E_split, E_fused = _split_fused_energies(A, gate, chi)
        gaps.append(abs(E_split - E_fused) / abs(E_fused))
    assert gaps[1] < gaps[0] / 10, (
        f"split-vs-fused gap did not close with chi ({gaps[0]:.2e} at chi=8 -> "
        f"{gaps[1]:.2e} at chi=48); a gap flat in chi is a defect, not truncation"
    )


def test_split_corner_spectrum_matches_fused(su_state):
    """The stronger form: the whole corner spectrum, not just the energy.

    At chi=48 for the reason given in ``test_split_matches_fused_oracle``: at
    chi=8 the two schemes' spectra differ by up to 7.3e-05 on the small
    singular values (2.3% relative) purely from truncation, which the 1e-6
    tolerance here cannot accommodate.  Measured 5.49e-08 at chi=48.
    """
    A, _ = su_state
    chi = 48
    env_split = ctm_split_tensor(A, chi=chi, max_iter=200, conv_tol=1e-12)
    env_fused, _ = ctm_tensor(A, chi=chi, max_iter=200, conv_tol=1e-12)
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


# ------------------------------------------------------------------ #
# AD path (#746): the reroute has to survive differentiation, not just  #
# the forward.  Before #746 the single-site split AD path was rejected  #
# outright under the default gs_recipe="2x2" (ipeps_ad_policy).         #
# ------------------------------------------------------------------ #


def test_split_2x2_implicit_grad_matches_explicit(su_state):
    """Implicit (Neumann) and explicit (unrolled) backward must agree.

    Two independent gradient mechanisms through the same 2x2 split forward.
    Measured agreement on this state: 7.1e-10 relative.  A directional finite
    difference along ``g/|g|`` matches the analytic ``g.v`` to 5.3e-5, stable
    across eps=1e-4..1e-6 (so the residual is CTM fixed-point accuracy, not FD
    noise) -- kept out of the test because a converged-enough FD costs minutes.
    """
    import jax.numpy as jnp

    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit,
        ctm_energy_split_implicit,
    )
    from tenax.core.tensor import DenseTensor

    A, gate = su_state
    idx = A.indices
    x0 = A.todense()

    def E_imp(x):
        return ctm_energy_split_implicit(
            {(0, 0): DenseTensor(x, idx)},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            chi_I=4,
            max_iter=60,
            conv_tol=1e-11,
        ).real

    def E_exp(x):
        return ctm_energy_split_explicit(
            {(0, 0): DenseTensor(x, idx)},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            chi_I=4,
            warmup_steps=12,
            backprop_steps=12,
        ).real

    assert float(E_imp(x0)) == pytest.approx(float(E_exp(x0)), rel=1e-8)

    g_imp = np.asarray(jax.grad(E_imp)(x0))
    g_exp = np.asarray(jax.grad(E_exp)(x0))
    den = max(np.linalg.norm(g_imp), np.linalg.norm(g_exp), 1e-30)
    rel = np.linalg.norm(g_imp - g_exp) / den
    assert np.linalg.norm(g_imp) > 1e-8, "gradient is identically zero"
    assert rel < 1e-6, f"implicit vs explicit split-2x2 gradient differ by {rel:.3e}"
    assert jnp.all(jnp.isfinite(jnp.asarray(g_imp)))


def test_split_2x2_warm_start_matches_cold(su_state):
    """The 2x2 fixed point is seed-independent, on a state that reaches one.

    ``test_split_ctm_fuse_flag.py`` tests this on ``recipe="1x1"`` because its
    random ``_make_site`` fixture does not converge element-wise under 2x2
    (per-sweep diff 0.5..0.85 after 80 sweeps -- a property shared with the
    already-shipped 2-site path, not introduced by the #746 reroute -- #767).  A
    physical simple-update state does converge (1.3e-14), so the claim is
    testable here: warm-starting from a *different* tensor's converged env must
    not move the energy or the implicit gradient.

    Measured: |dE| = 0.0 exactly, gradient agreement 1.9e-13.
    """
    import jax.numpy as jnp

    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms._split_ctm_energy_ad import (
        converge_split_env,
        ctm_energy_split_implicit,
    )

    A, gate = su_state
    kw = dict(chi=4, chi_I=4, max_iter=80, conv_tol=1e-12, min_iter=2)

    # Seed from a *different* state so the warm start is a genuine perturbation.
    warm_env = converge_split_env(A * 1.3, **kw)

    def cold(a):
        return ctm_energy_split_implicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate, **kw
        ).real

    def warm(a):
        return ctm_energy_split_implicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate, env_init=warm_env, **kw
        ).real

    e_c, g_c = jax.value_and_grad(cold)(A)
    e_w, g_w = jax.value_and_grad(warm)(A)
    gc = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_c)])
    gw = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_w)])

    assert float(e_c) == pytest.approx(float(e_w), abs=1e-10), (
        f"warm start moved the 2x2 fixed point: cold={e_c!r} warm={e_w!r}"
    )
    rel = float(jnp.linalg.norm(gc - gw) / jnp.linalg.norm(gc))
    assert rel < 1e-9, f"warm vs cold implicit gradient differ by {rel:.3e}"
