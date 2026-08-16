"""#667: simple update must not converge to the product state.

The shipped simple update had two independent defects that together drove the
state to a product state:

1. The sweep covered only 2 of the 4 checkerboard bonds (``su_h(A,B)`` and
   ``su_v(A,B)``, never ``su_h(B,A)`` / ``su_v(B,A)``), so A was always the
   left/top site of every gate and only ever picked up ``sqrt(lam)`` on its
   ``r``/``d`` legs -- leaving half the lattice bonds with no Schmidt weight.
2. ``Gamma`` absorbed ``sqrt(sigma)`` that was *also* stored as the new lambda
   and re-absorbed in full on the next sweep, so the shared bond carried
   ``lambda**1.5``.

Symptom: lam_2 proportional to dt, lam_3 proportional to dt**2, and E -> -0.5
exactly (the product-state energy) as dt -> 0, i.e. *smaller dt was worse*.

**These tests deliberately do not assert on the energy ``ipeps()`` returns.**
That number comes from the legacy 2-site ``ctm_2site``, which does not converge
here (measured ``diff`` 1.6e-3 to 6.5e-2 after 600 sweeps at conv_tol=1e-10),
so freezing it would freeze noise -- the same trap #836 removed elsewhere.  The
state itself is well defined, so the assertions below measure the state with a
converged 1x1 CTM (``recipe="2x2"``; E is flat to 1e-12 from chi=16, corner rank
= chi, so it is neither unconverged nor #747-collapsed).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

# D=2 iPEPS Heisenberg reference. Simple update is not variationally optimal, so
# it lands slightly above the -0.6599 optimum; the product state sits at -0.5.
D2_SU_ENERGY = -0.65933
PRODUCT_STATE_ENERGY = -0.5


def _run_su(D: int, *, steps: int = 200, dt: float = 0.05):
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=steps,
        dt=dt,
        ctm=CTMConfig(chi=8, max_iter=40, conv_tol=1e-8),
    )
    # ipeps() may warn that its own (legacy) 2-site CTM did not converge (#839)
    # at the small chi used here.  That is a property of ctm_2site, not of the
    # state under test, and whether it fires depends on D -- so it is ignored
    # rather than asserted either way.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CTM did not converge.*")
        _, (A, B), _ = ipeps(gate, None, cfg)
    return gate, A, B


def _energy_1x1(A, gate, chi: int = 16) -> float:
    """Energy of the uniform 1-site ansatz built from a single site tensor."""
    env, _ = ctm_tensor(A, chi=chi, max_iter=200, conv_tol=1e-10, recipe="2x2")
    return float(compute_energy_ctm_tensor(A, env, gate, 2))


@pytest.fixture(scope="module")
def su_d2():
    """The D=2 reference run, shared -- this file is in the required gate.

    600 steps rather than the 200 that sufficed before #851, because that 200
    was never evidence of convergence.  With one horizontal and one vertical
    spectrum stamped onto all four bonds, ``A`` and ``B`` were handed the *same*
    gauge by construction, so ``E_1x1(A) == E_1x1(B)`` held to 1e-16 at any
    step count -- an identity, not a measurement.  Four independent spectra make
    the equality a real statement about reaching the uniform fixed point, and it
    converges geometrically: measured ``|E_A - E_B|`` at dt=0.05 is

        200 steps  4.7e-08     350 steps  2.5e-12     500 steps  8.9e-16
        250 steps  1.9e-09     400 steps  8.5e-14     600 steps  5.6e-16

    so 600 sits at the machine-precision floor, seven orders below the 1e-8 the
    uniformity test asserts.  The tolerance is deliberately left tight: a
    *structural* asymmetry between the AB and BA bonds would be flat in the step
    count, not decaying, and would still miss 1e-8 by orders of magnitude.
    Costs ~2.2s against ~0.9s; the 1x1 CTM in ``_energy_1x1`` dominates anyway.
    """
    return _run_su(2, steps=600)


def test_simple_update_does_not_converge_to_the_product_state(su_d2):
    """#667: D=2 must reach ~-0.659, not the -0.5 product state."""
    gate, A, _ = su_d2
    E = _energy_1x1(A, gate)
    assert E < -0.64, (
        f"simple update returned E={E:.6f}; the product state is "
        f"{PRODUCT_STATE_ENERGY} and the D=2 reference is {D2_SU_ENERGY} (#667)"
    )
    assert E == pytest.approx(D2_SU_ENERGY, abs=1e-3)


def test_simple_update_state_is_uniform_under_the_sublattice_rotation(su_d2):
    """The rotated gate makes the ground state uniform, so A and B must agree.

    Compared physically: a simple-update tensor is defined only up to a bond
    gauge, so ``||A-B||`` measures the gauge and not the state -- it stays ~1.7
    even when the two are the same physical tensor.  The uniform 1x1 energy
    built from each is gauge-invariant.
    """
    gate, A, B = su_d2
    E_A, E_B = _energy_1x1(A, gate), _energy_1x1(B, gate)
    assert E_A == pytest.approx(E_B, abs=1e-8), (
        f"sublattice-rotated SU should be uniform but E_1x1(A)={E_A:.10f} != "
        f"E_1x1(B)={E_B:.10f}"
    )


def test_simple_update_entanglement_is_not_a_trotter_artifact(su_d2):
    """#667's signature: the state got *worse* as dt shrank.

    With the defects present the only thing entangling the state was the Trotter
    step, so lam_2 was proportional to dt and E -> -0.5 as dt -> 0.  A correct
    simple update converges to a dt-independent state.
    """
    gate, A_coarse, _ = su_d2
    _, A_fine, _ = _run_su(2, steps=600, dt=0.01)
    E_coarse = _energy_1x1(A_coarse, gate)
    E_fine = _energy_1x1(A_fine, gate)
    assert E_fine < -0.64, (
        f"E={E_fine:.6f} at dt=0.01 -- shrinking dt drove the state toward the "
        f"product state, which is #667's signature"
    )
    assert E_fine == pytest.approx(E_coarse, abs=5e-3)


def test_simple_update_d3_bond_is_genuinely_rank_3():
    """A nominally-D=3 state must actually use its third bond direction.

    With the defects present lam_3 was ~2e-6 at D=3, so any 'D=3' result was
    really a D=2 state wearing a D=3 shape.
    """
    gate, A, _ = _run_su(3)
    a = np.asarray(A.todense())
    # Schmidt spectrum across one virtual leg of the physical site tensor.
    sv = np.linalg.svd(a.reshape(a.shape[0], -1), compute_uv=False)
    sv = sv / sv[0]
    assert sv[2] > 1e-3, (
        f"D=3 site tensor has a negligible third direction (spectrum {sv}); "
        f"the state is effectively D=2 (#667)"
    )
    E = _energy_1x1(A, gate, chi=24)
    assert E < D2_SU_ENERGY, (
        f"D=3 energy {E:.6f} is not below the D=2 reference {D2_SU_ENERGY}"
    )


@pytest.mark.slow
def test_d4_beats_d2():
    """D=4 must reach the ~-0.667 the literature reports.

    ``test_ipeps.py::test_2site_heisenberg_D4_energy`` used to assert this on
    the energy ``ipeps()`` returns, which comes from the non-convergent
    ``ctm_2site`` (-0.6350 for this state).  Measured on a converged 1x1 CTM the
    state does reach -0.6674.
    """
    gate, A, _ = _run_su(4, steps=400, dt=0.05)
    E = _energy_1x1(A, gate, chi=24)
    assert E < -0.66, f"D=4 energy {E:.6f} does not reach the -0.667 reference"
    assert E < D2_SU_ENERGY, f"D=4 {E:.6f} is not below D=2 {D2_SU_ENERGY}"


def test_energy_is_chi_converged_and_the_environment_is_not_collapsed(su_d2):
    """#747 discipline: an energy flat in chi with a rank-1 corner is a collapsed
    environment, not a converged one.  Here it must be flat *and* full rank.

    This one guards the energy the tests above pin rather than reproducing the
    #667 defect -- it holds both before and after the fix.
    """
    gate, A, _ = su_d2
    energies, ranks = [], []
    for chi in (8, 16, 24):
        env, _ = ctm_tensor(A, chi=chi, max_iter=200, conv_tol=1e-10, recipe="2x2")
        energies.append(float(compute_energy_ctm_tensor(A, env, gate, 2)))
        C1 = np.asarray(env.C1.todense())
        sv = np.linalg.svd(C1, compute_uv=False)
        ranks.append(int(np.sum(sv > 1e-10 * sv[0])))
    assert energies[1] == pytest.approx(energies[2], abs=1e-6), energies
    assert min(ranks) > 1, f"corner collapsed to rank {ranks} (#747)"
