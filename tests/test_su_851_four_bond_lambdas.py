"""#851: a checkerboard has four bonds, and each needs its own Schmidt spectrum.

The four-phase simple-update sweep evolves ``A.r<->B.l``, ``A.d<->B.u``,
``B.r<->A.l`` and ``B.d<->A.u``.  Storing one horizontal and one vertical
spectrum for those four made phases 0 and 2 write the same slot, so
``num_imaginary_steps % 4`` decided which bond's gauge was stamped onto the
whole lattice::

    steps % 4 | last horizontal phase | lam_h applied to all four legs
    1, 2      | phase 0               | the AB bond's spectrum
    3, 0      | phase 2               | the BA bond's spectrum

``num_imaginary_steps`` is an ordinary convergence knob.  Nobody expects it to
select a gauge.

Independent bonds are **opt-in** (``su_independent_bond_lambdas``), and the
tests below pass ``independent_bonds=True`` explicitly.  The default shares one
spectrum between each pair, which is what the code always did.  That is a
measured trade rather than an unfinished migration: on a translation-invariant
Hamiltonian the paired bonds coincide at the fixed point, so sharing costs
nothing at convergence -- while four free bonds can follow a dimerising
direction that the shared spectrum projects out.  See
``test_sharing_the_spectrum_is_the_conservative_default`` and the config
docstring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import _wrap_as_dense_tensor, sublattice_rotate_gate
from tenax.algorithms.ipeps_simple_update import (
    BondWeights,
    _make_trotter_gate_tensor,
    _simple_update_checkerboard_sweep,
    _to_physical_pair,
)
from tenax.algorithms.ipeps_simple_update import (
    _simple_update_2site_horizontal_tensor as _horizontal,
)
from tenax.algorithms.ipeps_simple_update import (
    _simple_update_2site_vertical_tensor as _vertical,
)

jax.config.update("jax_enable_x64", True)

_LEGS = ("u", "d", "l", "r")


def _heisenberg_gate():
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    H = np.kron(Sz, Sz) + 0.5 * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp))
    return jnp.asarray(H).reshape(2, 2, 2, 2)


def _random_pair(D, seed=0, d=2):
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    A = _wrap_as_dense_tensor(jax.random.normal(kA, (D, D, D, D, d)))
    B = _wrap_as_dense_tensor(jax.random.normal(kB, (D, D, D, D, d)))
    return A * (1.0 / float(A.norm())), B * (1.0 / float(B.norm()))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def _bond_fingerprint(T):
    """Eigenvalues of each virtual leg's reduced matrix, largest first.

    ``M_leg[i,j] = sum_rest T[i,rest] conj(T[j,rest])``.  Its spectrum is
    invariant under any unitary acting on that leg, which is exactly the
    simple-update bond-gauge freedom -- so this reads off *how much weight* a
    leg carries without being confused by which basis the SVD happened to pick.
    """
    labels = T.labels()
    M = np.asarray(T.todense())
    out = []
    for leg in _LEGS:
        ax = labels.index(leg)
        X = np.moveaxis(M, ax, 0).reshape(M.shape[ax], -1)
        out.append(np.sort(np.abs(np.linalg.eigvalsh(X @ X.conj().T)))[::-1])
    return np.concatenate(out)


# ------------------------------------------------------------------ #
# 1. The premise: the four bonds really are inequivalent              #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("D", [2, 3])
def test_the_four_checkerboard_bonds_carry_different_spectra(D):
    """Without this, #851 would be a distinction without a difference.

    On the translation-uniform fixed point the AB and BA bonds are related by
    the sublattice symmetry and their spectra coincide, which is why sharing
    one slot went unnoticed.  Away from it -- an asymmetric initial PEPS, a
    dimerized phase, or simply a sweep that has not converged, which is the
    normal state of affairs mid-run -- they do not.

    Measured ``|h_AB - h_BA| / |h_AB|`` on this fixture: 0.093 at 8 steps,
    0.279 at D=3/60 steps.  The gap decays toward the symmetric fixed point
    (1.1e-05 at D=2, dt=0.05, 200 steps), so the assertion is made at a step
    count where the state is still evolving.
    """
    A, B = _random_pair(D)
    gate = _make_trotter_gate_tensor(_heisenberg_gate(), 0.05, site_tensor=A)
    _A, _B, lam = _simple_update_checkerboard_sweep(
        A, B, gate, D, 8, independent_bonds=True
    )

    assert _rel(lam.h_AB, lam.h_BA) > 1e-2, (
        f"the two horizontal bonds have the same spectrum to "
        f"{_rel(lam.h_AB, lam.h_BA):.3e}, so this fixture cannot distinguish "
        f"whether they are tracked separately. Pick a step count further from "
        f"the symmetric fixed point."
    )
    assert _rel(lam.v_AB, lam.v_BA) > 1e-2, (
        f"the two vertical bonds have the same spectrum to "
        f"{_rel(lam.v_AB, lam.v_BA):.3e} -- see above."
    )


# ------------------------------------------------------------------ #
# 2. The defect: the state must not move with ``steps % 4``           #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("D", [2, 3])
def test_the_physical_state_does_not_move_with_steps_mod_4(D):
    """A gate that does nothing must not change the state, at any step count.

    ``dt = 0`` makes the Trotter gate the identity, so the sweep evolves
    nothing -- it only gauge-fixes toward the Vidal canonical form.  Once that
    has settled, the physical tensors are a fixed point and running further
    phases must leave them alone.  That turns #851 into an exact statement with
    no convergence tolerance to argue about: any dependence on ``steps % 4``
    under an identity gate is pure bookkeeping.

    Measured spread of the leg fingerprint over 8 further identity phases:

    ===========  =============  ==================
    D            four spectra   one h + one v
    ===========  =============  ==================
    2            1.4e-05        **2.5e-01**
    3            5.1e-06        **1.5e-01**
    ===========  =============  ==================

    and the two-spectrum column is periodic with period 4 exactly as the table
    in the module docstring predicts: flat at ``extra in {1, 2, 6}`` (the last
    horizontal write was phase 0, matching the reference) and 0.14-0.25 at
    ``{3, 4, 5, 7, 8}`` (phase 2 overwrote it with the other bond).

    The residual ~1e-05 is the identity sweep still drifting toward the
    canonical form, not a phase effect -- it does not step with the phase
    index.  The threshold below sits 100x above it and 150x below the defect.
    """
    A, B = _random_pair(D)

    # A physically meaningful state first, so the bonds are inequivalent...
    evolve = _make_trotter_gate_tensor(_heisenberg_gate(), 0.05, site_tensor=A)
    A, B, lam = _simple_update_checkerboard_sweep(
        A, B, evolve, D, 60, independent_bonds=True
    )

    # ...then freeze the Hamiltonian off and let the gauge settle.
    identity = _make_trotter_gate_tensor(_heisenberg_gate(), 0.0, site_tensor=A)
    A, B, lam = _simple_update_checkerboard_sweep(
        A, B, identity, D, 24, lam, independent_bonds=True
    )

    assert _rel(lam.h_AB, lam.h_BA) > 1e-2, (
        "the AB and BA bonds coincide after the warm-up, so stamping one onto "
        "the other would be invisible and this test proves nothing."
    )

    ref = None
    spreads = {}
    for extra in range(9):
        A2, B2, lam2 = _simple_update_checkerboard_sweep(
            A, B, identity, D, extra, lam, independent_bonds=True
        )
        fp = np.concatenate(
            [_bond_fingerprint(t) for t in _to_physical_pair(A2, B2, lam2)]
        )
        if ref is None:
            ref = fp
        spreads[extra] = float(np.max(np.abs(fp - ref)) / np.max(np.abs(ref)))

    worst = max(spreads, key=spreads.get)
    assert spreads[worst] < 1e-3, (
        f"the physical state moved by {spreads[worst]:.3e} after {worst} "
        f"further phases of an *identity* gate, which evolves nothing. Per-step "
        f"spread: { {k: f'{v:.2e}' for k, v in spreads.items()} }. A jump that "
        f"switches on at phase 2 and repeats with period 4 is #851: the AB and "
        f"BA bonds sharing one lambda slot, so where the sweep stops selects "
        f"which bond's gauge is applied to the whole lattice."
    )


# ------------------------------------------------------------------ #
# 3. The mapping: each leg gets its own bond, and shared bonds agree  #
# ------------------------------------------------------------------ #


def test_each_leg_receives_its_own_bond_spectrum():
    """Pins the site-to-bond mapping, which invariance alone cannot.

    A *consistently* wrong mapping -- say ``h_BA`` on ``A.r`` and ``h_AB`` on
    ``A.l``, mirrored on ``B`` -- produces a different physical state but is
    still independent of ``steps % 4``, so the test above would pass it.  This
    one reads the applied weights straight back off the tensors.

    ``A`` and ``B`` are mirror images, not copies: ``A.r`` and ``B.l`` are the
    *same* bond, so both must receive ``sqrt(h_AB)`` for the bond to carry
    ``h_AB`` exactly once.  Getting that wrong would weight one bond by
    ``sqrt(h_AB . h_BA)`` and is the reason the mapping lives in one function.
    """
    D, d = 3, 2
    lam = BondWeights(
        h_AB=jnp.asarray([1.0, 2.0, 4.0]),
        h_BA=jnp.asarray([1.0, 3.0, 9.0]),
        v_AB=jnp.asarray([1.0, 5.0, 25.0]),
        v_BA=jnp.asarray([1.0, 7.0, 49.0]),
    )
    ones = _wrap_as_dense_tensor(jnp.ones((D, D, D, D, d)))
    A_phys, B_phys = _to_physical_pair(ones, ones, lam)

    def applied(T, leg):
        """Read back the weight applied to ``leg``, normalised to entry 0.

        The tensor is all-ones before scaling and is normalised afterwards, so
        every slice along ``leg`` differs only by ``sqrt(lam[i])`` -- square
        the ratio to recover ``lam`` up to its overall scale.
        """
        labels = T.labels()
        M = np.asarray(T.todense())
        ax = labels.index(leg)
        prof = np.moveaxis(M, ax, 0).reshape(M.shape[ax], -1)[:, 0]
        return (prof / prof[0]) ** 2

    def spectrum(lam_vec):
        v = np.asarray(lam_vec)
        return v / v[0]

    expected = {
        ("A", "r"): lam.h_AB,
        ("B", "l"): lam.h_AB,
        ("A", "l"): lam.h_BA,
        ("B", "r"): lam.h_BA,
        ("A", "d"): lam.v_AB,
        ("B", "u"): lam.v_AB,
        ("A", "u"): lam.v_BA,
        ("B", "d"): lam.v_BA,
    }
    tensors = {"A": A_phys, "B": B_phys}
    for (site, leg), lam_vec in expected.items():
        np.testing.assert_allclose(
            applied(tensors[site], leg),
            spectrum(lam_vec),
            rtol=1e-10,
            err_msg=(
                f"{site}.{leg} did not receive its own bond's spectrum. The "
                f"checkerboard mapping is A.u=v_BA A.d=v_AB A.l=h_BA A.r=h_AB, "
                f"and B mirrored; A.r and B.l are one bond and must match."
            ),
        )


# ---------------------------------------------------------------------------
# The default, and why it is the default.
# ---------------------------------------------------------------------------


def test_the_shared_default_reproduces_the_historical_loop_exactly():
    """Off, this must be the code that shipped -- not merely close to it.

    The shared mode writes each freshly computed spectrum to its partner bond
    too, which is what one ``lam_h`` written by phases 0 and 2 did.  Asserting
    bit-equality against a hand-rolled two-lambda loop is what makes that claim
    checkable rather than a comment.
    """
    D = 3
    A0, B0 = _random_pair(D, seed=0)
    gate = _make_trotter_gate_tensor(_heisenberg_gate(), 0.05, site_tensor=A0)

    A, B = _random_pair(D, seed=0)
    lam_h, lam_v = jnp.ones(D), jnp.ones(D)
    for step in range(40):
        phase = step % 4
        if phase == 0:
            A, B, lam_h = _horizontal(A, B, gate, lam_h, lam_v, D)
        elif phase == 1:
            A, B, lam_v = _vertical(A, B, gate, lam_h, lam_v, D)
        elif phase == 2:
            B, A, lam_h = _horizontal(B, A, gate, lam_h, lam_v, D)
        else:
            B, A, lam_v = _vertical(B, A, gate, lam_h, lam_v, D)

    A2, B2, lams = _simple_update_checkerboard_sweep(
        *_random_pair(D, seed=0), gate, D, 40, independent_bonds=False
    )

    assert float((A - A2).norm()) == 0.0, "site A drifted from the shipped loop"
    assert float((B - B2).norm()) == 0.0, "site B drifted from the shipped loop"
    for name, want in (
        ("h_AB", lam_h),
        ("h_BA", lam_h),
        ("v_AB", lam_v),
        ("v_BA", lam_v),
    ):
        got = getattr(lams, name)
        assert float(jnp.max(jnp.abs(got - want))) == 0.0, f"{name} differs"


def test_sharing_the_spectrum_is_the_conservative_default():
    """The default is off because four free bonds are less robust, not more.

    Measured at D=3 over eight random seeds: independent bonds converged to a
    dimerised state (``|h_AB - h_BA|`` of 10-50%, flat spectrum) on 3 of them
    against 1 for the shared spectrum, and seed 0 -- which ``ipeps()``
    hardcodes -- is one of the three.  This pins that seed, so the default
    cannot be flipped without the trade being re-measured.
    """
    D = 3
    A, B = _random_pair(D, seed=0)
    # The rotated gate: the physical choice for Heisenberg, what ipeps() users
    # pass, and what the 8-seed trade was measured on.  With the unrotated gate
    # a random start traps *both* modes, so it cannot separate them.
    gate = _make_trotter_gate_tensor(
        sublattice_rotate_gate(_heisenberg_gate()), 0.05, site_tensor=A
    )

    _, _, shared = _simple_update_checkerboard_sweep(
        *_random_pair(D, seed=0), gate, D, 400, independent_bonds=False
    )
    _, _, free = _simple_update_checkerboard_sweep(
        *_random_pair(D, seed=0), gate, D, 400, independent_bonds=True
    )

    shared_min = min(float(jnp.min(v)) for v in shared)
    free_min = min(float(jnp.min(v)) for v in free)
    assert shared_min < 0.05, (
        f"the shared default no longer reaches a decaying spectrum "
        f"(min lambda {shared_min:.3e}); the trade this default encodes has "
        f"changed and needs re-measuring"
    )
    assert free_min > 0.05, (
        f"independent bonds no longer fall into the dimerised state on seed 0 "
        f"(min lambda {free_min:.3e}). If this is a genuine improvement the "
        f"default should be revisited -- do not just delete this assertion"
    )


def test_the_config_default_is_off():
    """A one-line guard: the flag is the whole contract of this change."""
    from tenax.algorithms.ipeps_config import iPEPSConfig

    assert iPEPSConfig().su_independent_bond_lambdas is False
