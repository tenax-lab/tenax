"""Flow implicit AD on symmetric 2-site tensors matches a converged FD
(spec 2026-10-09 §7).  The fixture asserts its regime.

The dense-oracle gradient comparison of the spec is not here.  After #1098
the densified forward reaches the same energy (<= 6e-15), but the dense
``bond_phase`` gauge is not stationary on these states (cross-sector
degenerate singular values: residual 0.43 for U(1), 1.4 for Z2), so the
dense backward is no reference (relative error 0.31 / 1.0).  The dense
*energy* is used: it pins that the symmetric layout is not restricted.
See the task-4 report of the 2026-10-09 symmetric 2-site CTM AD plan.
"""

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms import _ctm_energy_ad
from tenax.algorithms._symmetric_ad_validation import (
    converged_energy,
    densify_pair,
    fd_directional,
    grad_dot,
    make_2site_loss,
)
from tenax.algorithms.ctm_layout import symmetric_layout_seed
from tenax.algorithms.ipeps import (
    heisenberg_u1sz_init_pair,
    xxz_gate,
    xxz_gate_u1sz,
    xxz_gate_z2,
    xxz_z2_init_pair,
)
from tenax.core.tensor import SymmetricTensor

D, DELTA = 2, 0.3
# chi per case: a clean cut in the adapted (#1098) spectrum.  U(1) chi=8 cuts
# inside a near-degenerate set; chi=6 has relative gap 0.62.  Z2 chi=8: 0.32.
_CASES = {
    "u1": (heisenberg_u1sz_init_pair, xxz_gate_u1sz, 6),
    "z2": (xxz_z2_init_pair, xxz_gate_z2, 8),
}


def _ctm(chi):
    return dataclasses.replace(
        CTMConfig(chi=chi), conv_tol=1e-10, max_iter=300, plateau_patience=None
    )


def _clear_diagnostics():
    """Drop the keys asserted below, so a stale value cannot pass."""
    for key in ("forward_iterations", "projector_backward_used"):
        _ctm_energy_ad._F3_LAST_DIAGNOSTICS.pop(key, None)


def _regime(A, B, seed):
    """Refuse a fixture that would make the test vacuous."""
    # Corner legs only: every corner leg is a chi leg.  The layout also holds
    # the T tensors' double-layer legs, whose sectors are fixed by the sites
    # and can be charged when every chi leg is not (Codex on #1134).
    chi_legs = [v for k, v in seed.layout.items() if k[1].startswith("C")]
    if not any(len(v) > 1 for v in chi_legs):
        raise AssertionError("out of regime: no charged environment sector")
    if not (isinstance(A, SymmetricTensor) and len(A.blocks) > 1):
        raise AssertionError("out of regime: site tensor not blocked")
    if not seed.cold_converged or not seed.stable:
        raise AssertionError("out of regime: forward not converged/stable")
    if not seed.gap_converged:
        raise AssertionError("out of regime: the cut-gap solve did not converge")
    if not seed.cut_gap > 1e-2:
        raise AssertionError(f"out of regime: cut gap {seed.cut_gap:.2e} <= 1e-2")


@pytest.fixture(scope="module", params=sorted(_CASES))
def point(request):
    init, gate_fn, chi = _CASES[request.param]
    A0, B0 = init(D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=chi)
    )
    # SU takes the dense gate; ipeps casts the Trotter gate to the site symmetry.
    _, (A, B), _ = ipeps(xxz_gate(DELTA), (A0, B0), su, compute_energy=False)
    gate = gate_fn(DELTA)
    seed = symmetric_layout_seed(A, B, gate, _ctm(chi), forced_sweeps=100)
    _regime(A, B, seed)
    return chi, A, B, gate, seed


@pytest.mark.slow
def test_gradient_matches_a_converged_fd(point):
    chi, A, B, gate, seed = point
    # The symmetric energy is the CTM energy, not a restricted-layout one
    # (#1098): the dense eager CTM reaches it (measured <= 6e-15).
    e_dense, dense_ok = converged_energy(*densify_pair(A, B), gate, _ctm(chi), None)
    assert dense_ok, "the dense reference CTM did not converge"
    assert abs(seed.energy_cold - e_dense) < 1e-10, (seed.energy_cold, e_dense)
    loss, _ = make_2site_loss(gate, _ctm(chi), seed.envs_init)
    _clear_diagnostics()
    e, (gA, gB) = jax.value_and_grad(loss)((A, B))
    diag = _ctm_energy_ad.get_last_implicit_ad_diagnostics()
    # Flow backward, not the frozen fallback (#1028).
    assert diag["projector_backward_used"] == "flow", diag
    # The loss starts from envs_init: a seeded forward stops at min_iter
    # (measured 10, against 17 / 22 cold seed iterations).
    assert diag["forward_iterations"] < seed.cold_iterations, diag
    # The loss is the seed's fixed point: same energy as the eager forward.
    # A cold-started loss misses it by 6.6e-6 (U(1)) / 8.3e-5 (Z2).
    assert abs(float(e) - seed.energy_cold) < 1e-10
    for k in range(2):
        kA, kB = jax.random.split(jax.random.PRNGKey(100 + k))
        dA = SymmetricTensor.random_normal(A.indices, kA)
        dB = SymmetricTensor.random_normal(B.indices, kB)
        nrm = float(np.sqrt(dA.norm() ** 2 + dB.norm() ** 2))
        dA, dB = dA * (1.0 / nrm), dB * (1.0 / nrm)
        r = fd_directional(A, B, dA, dB, gate, _ctm(chi), seed.envs_init)
        # The h scan found a plateau: the FD is a measurement, not noise.
        assert r["fd_unc"] < 1e-3 * abs(r["fd"]), r
        gd = grad_dot((gA, gB), (dA, dB))
        tol = max(1e-6 * abs(r["fd"]), r["fd_unc"])
        assert abs(gd - r["fd"]) < tol, (k, gd, r["fd"], r["fd_unc"], r["scan"])


def test_fixture_rejects_trivial_charges():
    from tenax.algorithms.ctm_layout import SeedResult

    class _T:
        blocks = {0: None}

    # Every chi leg has one sector; a double-layer leg of T1 has two.
    layout = {
        ((0, 0), "C1", "chi_a"): ((0, 4),),
        ((0, 0), "C1", "chi_b"): ((0, 4),),
        ((0, 0), "T1", "u2"): ((-1, 1), (1, 1)),
    }
    seed = SeedResult({}, layout, 0.0, 0.0, True, 1.0, True, 10, True)
    with pytest.raises(AssertionError, match="no charged environment sector"):
        _regime(_T(), _T(), seed)


def test_fixture_rejects_an_unconverged_cut_gap():
    from tenax.algorithms.ctm_layout import SeedResult

    A, B = heisenberg_u1sz_init_pair(D, jax.random.PRNGKey(0))
    assert len(A.blocks) > 1  # regime: the site check passes
    layout = {((0, 0), "C1", "chi_a"): ((-1, 2), (1, 2))}
    seed = SeedResult({}, layout, 0.0, 0.0, True, 1.0, True, 10, False)
    with pytest.raises(AssertionError, match="cut-gap solve did not converge"):
        _regime(A, B, seed)
