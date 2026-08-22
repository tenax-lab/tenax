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

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor, ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import (
    _rdm2x1_tensor_2site,
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
)
from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

# D=2 iPEPS Heisenberg reference, measured on the **A-B checkerboard** -- the
# lattice ``_to_physical_pair`` returns.  Simple update is not variationally
# optimal, so it lands slightly above the -0.6599 optimum; the product state
# sits at -0.5.
#
# This was -0.65933 until #900.  That number is the energy of the *uniform A-A*
# lattice, which is a different state; see ``_energy_checkerboard``.  The two
# sit 3.3e-4 apart at D=2 -- close enough to hide for a year, and 30x too small
# to matter against the 1e-3 tolerances below -- but at D=4 they separate by
# 1.0e-2 and disagree about whether D=4 beats D=2 at all.
D2_SU_ENERGY = -0.65900
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
    """Energy of the uniform 1-site ansatz built from a single site tensor.

    **This is not the energy of the state simple update produced** (#900), and
    it is kept only so that ``test_the_uniform_reading_is_a_different_lattice``
    can pin the difference.  Every assertion in this file measures
    :func:`_energy_checkerboard` instead.
    """
    env, _ = ctm_tensor(A, chi=chi, max_iter=200, conv_tol=1e-10, recipe="2x2")
    return float(compute_energy_ctm_tensor(A, env, gate, 2))


def _energy_checkerboard(A, B, gate, chi: int = 16) -> float:
    """Energy per site of the A-B checkerboard -- the lattice SU actually built.

    ``_to_physical_pair`` returns a pair meant to tile as ``A-B-A-B``: the bond
    between ``A.r`` and ``B.l`` carries ``sqrt(lam_h)`` from each side.  Tiling
    ``A`` alone instead contracts ``A``'s right-leg gauge against ``A``'s own
    left-leg gauge, which is a different network -- and measurably a different
    state, even once the two sublattices are physically equivalent.

    **The 1x1 code is not wrong; it is answering a different question.**
    Measured through the *same* 2-site machinery, ``E_1x1(A)`` is bit-identical
    to ``E_2site(A, A)`` at every ``D`` and step count tested (12 digits).  And
    the sublattices really do become equivalent: the 1-site RDMs traced from the
    A-B two-site RDM converge geometrically, ``||rho_A - rho_B|| =`` 5.99e-04,
    2.26e-05, 8.54e-07, 1.73e-12 at 400/600/800/1600 steps (D=2).  The A-A
    energy nonetheless stays 3.3e-4 away and does **not** follow -- equivalent
    sites do not make the two tilings the same network.

    At D=4 the gap is not small.  Both columns below are converged CTMs:
    bit-identical at ``max_iter`` 200, 400 **and** 800, at full corner rank
    24/24 throughout, chi=24::

        steps   E_2site(A,B)      E_1x1(A) == E_2site(A,A)
          400   -0.667030038246   -0.667387772978
          500   -0.667064040096   -0.667389471190
          600   -0.667069740328   -0.656839942585   <-- the "transient"
          800   -0.667070856002   -0.656839993926
         1600   -0.667070888256   -0.656839995414

    The right-hand column is what #900 reported as a decay and what macOS CI has
    been failing on since 2026-08-12.  It is a different lattice, not a later
    time: the left-hand column is monotone, converged, and comfortably below the
    D=2 reference.

    The #882 Phase 2 engine -- a separate implementation that stores no bond
    spectrum at all -- reaches -0.658880 / -0.662839 / -0.667012 at D = 2/3/4 on
    this same checkerboard reading, agreeing with the shipped engine to 1.2e-4,
    1.8e-5 and 5.9e-5.  Two independent engines agree on this column; neither
    reproduces the -0.65684.

    Raises:
        AssertionError: if either corner is zero or rank-1 (#747/#898), which
            would make the number a mean-field artifact rather than an energy.
    """
    env_A, env_B = ctm_tensor_2site(
        A, B, chi=chi, max_iter=200, conv_tol=1e-10, recipe="2x2"
    )
    for name, env in (("A", env_A), ("B", env_B)):
        sv = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
        assert sv[0] > 0, f"the {name} corner C1 is identically zero"
        rank = int(np.sum(sv > sv[0] * 1e-10))
        assert rank > 1, (
            f"the {name} corner C1 has rank {rank} at chi={chi}: a rank-1 "
            f"corner returns a plausible wrong energy rather than failing "
            f"(#747), and the convergence criterion cannot see it (#898)"
        )
    return float(compute_energy_ctm_tensor_2site(A, B, env_A, env_B, gate, 2))


def _one_site_rdms(A, B, chi: int = 16):
    """The two 1-site RDMs traced from the **same** A-B two-site RDM.

    Tracing both out of one joint RDM is what makes the comparison meaningful:
    two separately-built uniform lattices would each be normalised on their own
    and could agree for reasons that say nothing about the pair (#900).
    """
    env_A, env_B = ctm_tensor_2site(
        A, B, chi=chi, max_iter=200, conv_tol=1e-10, recipe="2x2"
    )
    rho = np.asarray(_rdm2x1_tensor_2site(A, B, env_A, env_B)).reshape(2, 2, 2, 2)
    rho = rho / np.trace(rho.reshape(4, 4))
    return np.einsum("abcb->ac", rho), np.einsum("abad->bd", rho)


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
    gate, A, B = su_d2
    E = _energy_checkerboard(A, B, gate)
    assert E < -0.64, (
        f"simple update returned E={E:.6f}; the product state is "
        f"{PRODUCT_STATE_ENERGY} and the D=2 reference is {D2_SU_ENERGY} (#667)"
    )
    assert E == pytest.approx(D2_SU_ENERGY, abs=1e-3)


def test_simple_update_state_is_uniform_under_the_sublattice_rotation(su_d2):
    """The rotated gate makes the ground state uniform, so A and B must agree.

    ``||A-B||`` is not the way to check: a simple-update tensor is defined only
    up to a bond gauge, so that norm measures the gauge and stays ~1.7 even when
    the two are the same physical tensor.

    **Nor is ``E_1x1(A)`` vs ``E_1x1(B)``, which is what this test used to
    compare** (#900).  Those are the energies of two *separate* uniform
    lattices, and they agree to 5e-11 on a pair whose A-B checkerboard sits
    3.3e-4 away from either -- so the old assertion held whether or not the
    sublattices were equivalent, and could not have failed for the reason it
    named.  A test that passes on both branches of the question it asks is not
    measuring anything.

    The gauge-invariant statement is about the two 1-site reduced density
    matrices traced out of the **same** A-B two-site RDM.  Those converge
    geometrically, which makes the tolerance below a real measurement::

        steps    ||rho_A - rho_B||
          400        5.993e-04
          600        2.262e-05
          800        8.536e-07
         1600        1.731e-12

    A *structural* inequivalence between the sublattices would be flat in the
    step count rather than decaying, and would miss 1e-4 by orders of magnitude.
    """
    _, A, B = su_d2
    rho_A, rho_B = _one_site_rdms(A, B)
    diff = float(np.linalg.norm(rho_A - rho_B))
    assert diff < 1e-4, (
        f"sublattice-rotated SU should be uniform but the 1-site RDMs differ "
        f"by {diff:.3e}; at 600 steps this is 2.3e-5 and falling geometrically"
    )


def test_simple_update_entanglement_is_not_a_trotter_artifact(su_d2):
    """#667's signature: the state got *worse* as dt shrank.

    With the defects present the only thing entangling the state was the Trotter
    step, so lam_2 was proportional to dt and E -> -0.5 as dt -> 0.  A correct
    simple update converges to a dt-independent state.
    """
    gate, A_coarse, B_coarse = su_d2
    _, A_fine, B_fine = _run_su(2, steps=600, dt=0.01)
    E_coarse = _energy_checkerboard(A_coarse, B_coarse, gate)
    E_fine = _energy_checkerboard(A_fine, B_fine, gate)
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
    gate, A, B = _run_su(3)
    a = np.asarray(A.todense())
    # Schmidt spectrum across one virtual leg of the physical site tensor.
    sv = np.linalg.svd(a.reshape(a.shape[0], -1), compute_uv=False)
    sv = sv / sv[0]
    assert sv[2] > 1e-3, (
        f"D=3 site tensor has a negligible third direction (spectrum {sv}); "
        f"the state is effectively D=2 (#667)"
    )
    E = _energy_checkerboard(A, B, gate, chi=24)
    assert E < D2_SU_ENERGY, (
        f"D=3 energy {E:.6f} is not below the D=2 reference {D2_SU_ENERGY}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("steps", [400, 800])
def test_d4_beats_d2(steps):
    """D=4 must beat D=2, and must still beat it once the sweep has settled.

    Two step counts rather than one, because a single one cannot tell a
    converged number from a transient and this test froze a transient for a
    year (#900, #836).  On the checkerboard reading D=4 is flat to 5e-7 across
    them -- -0.667030038246 at 400 and -0.667070856002 at 800, still
    -0.667070888256 at 1600.

    **The measurement is ``_energy_checkerboard``, not ``_energy_1x1``**, and
    that is the whole of #900.  The old assertion read the uniform A-A lattice,
    which at 400 steps sits a harmless 3.6e-4 away and from ~500 steps drops to
    -0.65684 -- *above* the D=2 reference.  macOS CI reported exactly that from
    the day this test was written; the platform difference was only which side
    of the split the 400-step state landed on, and the number it printed
    (-0.6568381) is Linux's own value at 600+ steps to 1.8e-6.

    So `-0.667` was never "the literature": the square-lattice Heisenberg QMC
    value is -0.669437 and a variational D=4 iPEPS optimum is a third quantity
    again.  What is asserted here is only what simple update produces, measured
    on the lattice it produces it for.
    """
    gate, A, B = _run_su(4, steps=steps, dt=0.05)
    E = _energy_checkerboard(A, B, gate, chi=24)
    assert E < -0.666, f"D=4 energy {E:.6f} does not reach the -0.667 reference"
    assert E < D2_SU_ENERGY, f"D=4 {E:.6f} is not below D=2 {D2_SU_ENERGY}"


def test_energy_is_chi_converged_and_the_environment_is_not_collapsed(su_d2):
    """#747 discipline: an energy flat in chi with a rank-1 corner is a collapsed
    environment, not a converged one.  Here it must be flat *and* full rank.

    This one guards the energy the tests above pin rather than reproducing the
    #667 defect -- it holds both before and after the fix.  It therefore has to
    guard the **checkerboard** environment, since that is what they now measure
    (#900): a chi-scan of the uniform A-A lattice says nothing about the
    convergence of a reading nobody takes.
    """
    gate, A, B = su_d2
    energies, ranks = [], []
    for chi in (8, 16, 24):
        env_A, env_B = ctm_tensor_2site(
            A, B, chi=chi, max_iter=200, conv_tol=1e-10, recipe="2x2"
        )
        energies.append(
            float(compute_energy_ctm_tensor_2site(A, B, env_A, env_B, gate, 2))
        )
        for env in (env_A, env_B):
            sv = np.linalg.svd(np.asarray(env.C1.todense()), compute_uv=False)
            ranks.append(int(np.sum(sv > 1e-10 * sv[0])))
    assert energies[1] == pytest.approx(energies[2], abs=1e-6), energies
    assert min(ranks) > 1, f"corner collapsed to rank {ranks} (#747)"


def test_the_uniform_reading_is_a_different_lattice(su_d2):
    """Pin the trap itself, so the 1x1 reading cannot quietly come back (#900).

    Three claims, each of which had to be true for #900 to be a measurement
    error rather than an engine defect:

    1. ``E_1x1(A)`` is exactly ``E_2site(A, A)``.  The 1-site machinery is not
       broken -- it computes the uniform A-A lattice correctly, through
       different code.  If this ever stops holding, one of the two paths has a
       real bug and the rest of this reasoning is void.
    2. ``E_2site(A, B)`` differs from both.  That gap is the finding: the pair
       ``_to_physical_pair`` returns does not tile as ``A-A``.
    3. The gap is not shrinking with the sweep -- measured at **two** step
       counts, not asserted in prose.  It is a property of the network, not a
       transient, which is why more steps never rescued the 1x1 number and why
       #900 read that flatness as convergence.  A test that checked one step
       count could not tell those apart, and would endorse the very reasoning
       this file exists to correct.

    Measured at D=2, chi=24: the checkerboard sits at -0.659003527529 and the
    uniform lattice at -0.659334299578 (600 steps).  The gap is 3.3102e-04,
    3.3077e-04, 3.3077e-04, 3.3077e-04 at 400/600/800/1600 steps -- flat to
    2.5e-08 from 400 onward, against a 1e-2 separation at D=4.  Deliberately
    D=2: the effect is not a large-``D`` pathology, it is there at the smallest
    size and merely too small to notice until D=4 makes it 30x bigger.

    **In the required gate, not the ``slow`` bucket.**  #900 reached CI in the
    first place because ``@pytest.mark.slow`` withheld this file's ``core``
    marker from ``test_d4_beats_d2``, so no required job ran it on any platform
    for fifteen days (#740/#805 -- *"a guard that runs in no required job is not
    one"*).  The D=4 arm genuinely costs too much to gate; this one does not,
    because it reuses ``su_d2`` rather than repeating the 600-step run it had
    already paid for, and reads at the fixture's own chi.
    """
    gate, A, B = su_d2

    # (1) the two paths agree about the uniform lattice.  su_d2's state only;
    # this is a statement about the code, not about the sweep length.
    e_uniform_1x1 = _energy_1x1(A, gate)
    e_uniform_2site = _energy_checkerboard(A, A, gate)
    assert e_uniform_1x1 == pytest.approx(e_uniform_2site, abs=1e-10), (
        f"E_1x1(A)={e_uniform_1x1:.12f} should be the same lattice as "
        f"E_2site(A,A)={e_uniform_2site:.12f}; if these disagree, one of the "
        f"two energy paths is wrong and #900's diagnosis needs re-checking"
    )

    # (2) and (3): the gap is real and it is not a transient.  600 comes from
    # the fixture; 400 is the extra run, and it is the cheap end of the scan
    # deliberately -- if the gap were closing, the *earlier* state is where it
    # would be widest, so this pair brackets the claim rather than repeating it.
    gaps = {}
    for steps, pair in ((400, _run_su(2, steps=400)), (600, (gate, A, B))):
        g, X, Y = pair
        gaps[steps] = abs(_energy_checkerboard(X, Y, g) - _energy_1x1(X, g))

    for steps, gap in gaps.items():
        assert gap > 1e-4, (
            f"at {steps} steps the checkerboard and uniform readings have "
            f"converged to each other (gap {gap:.3e}); if that is real, #900's "
            f"premise is gone and _energy_1x1 would be a valid reading again"
        )
    assert gaps[400] == pytest.approx(gaps[600], abs=1e-6), (
        f"the gap moved with the sweep ({gaps[400]:.6e} at 400 steps vs "
        f"{gaps[600]:.6e} at 600) -- it is measured flat to 2.5e-08 from 400 "
        f"onward, and a gap that decays would make it a transient rather than "
        f"a property of the network, which is the whole of #900's diagnosis"
    )
