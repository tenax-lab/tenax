"""Flow implicit AD on symmetric 2-site tensors matches a converged FD
(spec 2026-10-09 §7).  The fixture asserts its regime.

The dense-oracle comparison of the spec is not here: on these fixtures the
densified forward is not a fixed point of the dense bond_phase-gauged CTM
(cross-sector degenerate singular values), so it is no reference.  See the
task-4 report of the 2026-10-09 symmetric 2-site CTM AD plan.
"""

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._symmetric_ad_validation import (
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

D, CHI, DELTA = 2, 8, 0.3
_CASES = {
    "u1": (heisenberg_u1sz_init_pair, xxz_gate_u1sz),
    "z2": (xxz_z2_init_pair, xxz_gate_z2),
}


def _ctm():
    return dataclasses.replace(
        CTMConfig(chi=CHI), conv_tol=1e-10, max_iter=300, plateau_patience=None
    )


def _regime(A, B, seed):
    """Refuse a fixture that would make the test vacuous."""
    if not any(len(v) > 1 for v in seed.layout.values()):
        raise AssertionError("out of regime: no charged environment sector")
    if not (isinstance(A, SymmetricTensor) and len(A.blocks) > 1):
        raise AssertionError("out of regime: site tensor not blocked")
    if not seed.cold_converged or not seed.stable:
        raise AssertionError("out of regime: forward not converged/stable")
    if not seed.cut_gap > 1e-2:
        raise AssertionError(f"out of regime: cut gap {seed.cut_gap:.2e} <= 1e-2")


@pytest.fixture(scope="module", params=sorted(_CASES))
def point(request):
    init, gate_fn = _CASES[request.param]
    A0, B0 = init(D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=CHI)
    )
    # SU takes the dense gate; ipeps casts the Trotter gate to the site symmetry.
    _, (A, B), _ = ipeps(xxz_gate(DELTA), (A0, B0), su, compute_energy=False)
    gate = gate_fn(DELTA)
    seed = symmetric_layout_seed(A, B, gate, _ctm(), forced_sweeps=100)
    _regime(A, B, seed)
    return request.param, A, B, gate, seed


@pytest.mark.slow
def test_gradient_matches_a_converged_fd(point):
    _, A, B, gate, seed = point
    loss, _ = make_2site_loss(gate, _ctm(), seed.envs_init)
    e, (gA, gB) = jax.value_and_grad(loss)((A, B))
    # The loss is the seed's fixed point: same energy as the eager forward.
    assert abs(float(e) - seed.energy_cold) < 1e-10
    for k in range(2):
        kA, kB = jax.random.split(jax.random.PRNGKey(100 + k))
        dA = SymmetricTensor.random_normal(A.indices, kA)
        dB = SymmetricTensor.random_normal(B.indices, kB)
        nrm = float(np.sqrt(dA.norm() ** 2 + dB.norm() ** 2))
        dA, dB = dA * (1.0 / nrm), dB * (1.0 / nrm)
        r = fd_directional(A, B, dA, dB, gate, _ctm(), seed.envs_init)
        gd = grad_dot((gA, gB), (dA, dB))
        tol = max(1e-6 * abs(r["fd"]), r["fd_unc"])
        assert abs(gd - r["fd"]) < tol, (k, gd, r["fd"], r["fd_unc"], r["scan"])


def test_fixture_rejects_trivial_charges():
    from tenax.algorithms.ctm_layout import SeedResult

    class _T:
        blocks = {0: None}

    seed = SeedResult({}, {("x",): ((0, 4),)}, 0.0, 0.0, True, 1.0, True, 10)
    with pytest.raises(AssertionError, match="no charged environment sector"):
        _regime(_T(), _T(), seed)
