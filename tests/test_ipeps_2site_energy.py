"""The 2-site energy of ``ipeps()`` is a function of the state.

#845 added a guard that warns when a bond's RDM contracts to zero, and the
guard fires in ``ipeps()``'s own smoke tests: at ``chi = D**2`` one raw 2-site
RDM (``_rdm2x1_2site`` or ``_rdm1x2_2site``) comes back exactly zero, and which
bond dies depends on the host.  The guard reports the collapse; it does not
repair it.  These tests pin the two causes underneath.

**The environment.**  The legacy ``ctm_2site`` builds the new boundary of each
sublattice from that sublattice's *own* corners and parallel edge, and
renormalises each environment with a projector from its *own* grown corners.
A checkerboard boundary must come from the cell whose column or row is
absorbed, and the two environments must share one basis on every bond between
them, because the 2-site RDM networks join ``env_A`` to ``env_B`` across those
bonds.  Neither holds, so the energy depends on the gauge of the site tensors.
Measured at ``2cb7c575`` on three D=2 simple-update states, each in two bond
gauges at chi=6 and 8 (12 runs): ``ipeps()`` reported energies from ``-0.7815``
to ``+0.8171``, several below the ground-state energy ``-0.6694`` (QMC), while
``ctm_tensor_2site`` agreed with the 1-site CTM of each state to 4e-11 in every
run.

**The bond count.**  The cell has four bonds and two sites; the energy per site
is half the sum of all four.  ``compute_energy_ctm_2site`` adds the two bonds
that start on ``A`` and drops the two that start on ``B`` -- the defect #493
fixed in ``compute_energy_ctm_tensor_2site`` and not in this copy.  On a state
whose two kinds of bond differ, the result is not the energy: singlets on every
``A.r-B.l`` bond have exactly ``-0.375`` per site, and ``ipeps()`` reported
``-0.750``.  On the D=2 simple-update state of the plain Heisenberg gate the
bonds that start on ``A`` give ``-0.6393`` and those that start on ``B``
``-0.6743``; that difference, not a dropped bond, is most of the "~0.02 above
the truth" that the guide reported before 0.9.3.

Every state here is measured with ``num_imaginary_steps=0``: simple update then
leaves the pair unchanged (all bond weights stay 1), so ``ipeps()`` measures
exactly the pair it is given.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor, ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import (
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor_2site,
    compute_energy_ctm_tensor,
)
from tenax.algorithms.ipeps import (
    _wrap_as_dense_tensor,
    heisenberg_gate,
    ipeps,
    sublattice_rotate_gate,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_ctm_convergence import ctm_2site
from tenax.algorithms.ipeps_rdm import compute_energy_ctm_2site

D = 2
DPHYS = 2
CHI = 6  # above D**2 = 4, so the environment truncates
CTM = CTMConfig(chi=CHI, max_iter=200, conv_tol=1e-10)
#: Energy of singlets on every ``A.r-B.l`` bond, product elsewhere: one
#: singlet (``-3/4``) per two sites.
DIMER_ENERGY = -0.375


def _measure(gate, pair, ctm=CTM):
    """The energy ``ipeps()`` reports for ``pair``, which it leaves unchanged."""
    config = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=0, dt=0.1, ctm=ctm, unit_cell="2site"
    )
    energy, _, _ = ipeps(gate, pair, config)
    return energy


def _gauge_pair(A0, seed, eps=0.3):
    """``(A0, A0)`` with ``G G^-1`` on each of the four bonds of the cell.

    The four bonds are ``A.r-B.l``, ``B.r-A.l``, ``A.d-B.u`` and ``B.d-A.u``.
    Each gets its own random invertible ``G``, so the physical state is
    unchanged while ``A`` and ``B`` (and their double layers) differ.
    """
    Gh1, Gh2, Gv1, Gv2 = (
        jnp.eye(D) + eps * jax.random.normal(k, (D, D))
        for k in jax.random.split(jax.random.PRNGKey(seed), 4)
    )
    inv = jnp.linalg.inv
    A = jnp.einsum("udlrs,rR->udlRs", A0, Gh1)  # A.r G_h1
    A = jnp.einsum("udlrs,Ll->udLrs", A, inv(Gh2))  # G_h2^-1 A.l
    A = jnp.einsum("udlrs,dD->uDlrs", A, Gv1)  # A.d G_v1
    A = jnp.einsum("udlrs,Uu->Udlrs", A, inv(Gv2))  # G_v2^-1 A.u
    B = jnp.einsum("udlrs,Ll->udLrs", A0, inv(Gh1))  # G_h1^-1 B.l
    B = jnp.einsum("udlrs,rR->udlRs", B, Gh2)  # B.r G_h2
    B = jnp.einsum("udlrs,Uu->Udlrs", B, inv(Gv1))  # G_v1^-1 B.u
    B = jnp.einsum("udlrs,dD->uDlrs", B, Gv2)  # B.d G_v2
    return A, B


def _dimer_pair():
    """Singlets on every ``A.r-B.l`` bond; every other bond carries index 0."""
    A = jnp.zeros((D, D, D, D, DPHYS)).at[0, 0, 0, :, :].set(jnp.eye(DPHYS))
    singlet = jnp.array([[0.0, 1.0], [-1.0, 0.0]]) / jnp.sqrt(2.0)
    B = jnp.zeros((D, D, D, D, DPHYS)).at[0, 0, :, 0, :].set(singlet)
    return A, B


@pytest.fixture(scope="module")
def uniform_state():
    """A uniform D=2 state ``A0`` and its energy from the 1-site CTM.

    ``A0`` is the ``A`` tensor of a short simple update on the
    sublattice-rotated Heisenberg gate, used on every site.  The reference is
    ``ctm_tensor``, which never sees ``B``, so the 2-site code is not checked
    against itself.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, (A_su, _), _ = ipeps(
            gate,
            None,
            iPEPSConfig(max_bond_dim=D, num_imaginary_steps=40, dt=0.1),
            compute_energy=False,
        )
    A0 = A_su.todense()
    A0 = A0 / jnp.linalg.norm(A0)
    At = _wrap_as_dense_tensor(A0)
    env, _, info = ctm_tensor(
        At, CHI, max_iter=CTM.max_iter, conv_tol=CTM.conv_tol, return_meta=True
    )
    assert bool(info.converged), "the 1-site reference did not converge"
    E1 = float(compute_energy_ctm_tensor(At, env, gate.todense(), DPHYS))
    return gate, A0, E1


@pytest.mark.parametrize("seed", [None, 1, 2], ids=["uniform", "gauge1", "gauge2"])
def test_ipeps_energy_does_not_depend_on_the_bond_gauge(uniform_state, seed):
    """One physical state, one energy, whatever the gauge of ``(A, B)``.

    At ``2cb7c575``: ``-0.658663`` (uniform; the legacy sweep does not
    converge even there), ``-0.725908`` and ``-0.626903`` (the two gauges),
    against ``-0.658644`` from the 1-site CTM.
    """
    gate, A0, E1 = uniform_state
    pair = (A0, A0) if seed is None else _gauge_pair(A0, seed)

    E = _measure(gate, pair)

    assert E == pytest.approx(E1, abs=1e-8)


def test_ipeps_counts_every_bond_of_the_cell():
    """Singlets on the ``A.r-B.l`` bonds: exactly ``-0.375`` per site.

    Summing only the two bonds that start on ``A`` gives ``-0.75``: the singlet
    bond is counted once per two sites, as if it were one per site.  The CTM
    cannot certify convergence on this rank-1 state, and says so; the energy
    is exact anyway, since every bond's RDM is a product or a singlet.
    """
    # The corners of this state have rank 1, so the criterion refuses to
    # certify them (#898), and ipeps() must say so -- once: the loop inside it
    # must not warn a second time.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        E = _measure(heisenberg_gate(), _dimer_pair(), CTMConfig(chi=4, max_iter=50))
    ctm_warnings = [w for w in caught if "CTM" in str(w.message)]
    assert len(ctm_warnings) == 1, [str(w.message)[:80] for w in ctm_warnings]
    assert issubclass(ctm_warnings[0].category, UserWarning)
    assert str(ctm_warnings[0].message).startswith("CTM did not converge in ipeps()")

    assert E == pytest.approx(DIMER_ENERGY, abs=1e-10)


def _ipeps_ctm_warning(gate, pair, ctm):
    """The text of the one "CTM did not converge in ipeps()" warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _measure(gate, pair, ctm)
    texts = [
        str(w.message)
        for w in caught
        if str(w.message).startswith("CTM did not converge in ipeps()")
    ]
    assert len(texts) == 1, texts
    return texts[0]


def test_ipeps_quotes_the_loop_when_a_rank_1_corner_blinds_the_criterion():
    """More sweeps never certify a corner that stays at rank 1.

    The criterion is ``inf`` whenever a compared corner spectrum has rank <= 1
    below what the state can reach (#898).  The loop knows which corners are
    still at rank 1 and which have recovered, and its diagnosis says, per
    corner, whether more sweeps can help; ``ipeps()`` dropped it and advised
    raising ``max_iter`` whatever the cause.  Both corners of the dimer stay at
    rank 1: the loop's verdict is the blind one at ``max_iter`` 1, 2 and 50,
    with both coordinates still at rank 1 (measured on 1aee49a5).

    The loop's text for a corner still at rank 1 also says the environment is
    not a converged fixed point.  That overstates this case: the dimer's own
    fixed point has rank-1 corners, and its energy is exact
    (``test_ipeps_counts_every_bond_of_the_cell``).  This test pins only that
    ``ipeps()`` passes the diagnosis on.
    """
    text = _ipeps_ctm_warning(
        heisenberg_gate(), _dimer_pair(), CTMConfig(chi=4, max_iter=50)
    )

    assert "STILL rank <= 1" in text, text
    assert "Raise max_iter" not in text, text


def test_ipeps_keeps_its_own_advice_when_one_sweep_compares_nothing(uniform_state):
    """With one sweep the criterion is ``inf`` too, and "Raise max_iter" is right.

    The criterion compares successive sweeps, so one sweep gives it nothing to
    compare, whatever the rank of the corners.  The loop's own text for that
    case names ``ctm_tensor_multisite()``, which the caller never called, and
    suggests another recipe, which ``ipeps()`` does not offer.
    """
    gate, A0, _ = uniform_state
    text = _ipeps_ctm_warning(gate, (A0, A0), CTMConfig(chi=CHI, max_iter=1))

    # Regime: the infinite criterion of a run with no comparison.
    assert "max_iter=1 sweeps" in text and "(final criterion inf)" in text, text
    assert "Raise max_iter" in text, text
    assert "ctm_tensor_multisite" not in text, text


def test_ipeps_advises_raising_chi_when_chi_1_blinds_the_criterion(uniform_state):
    """At ``chi=1`` the criterion is blind whatever the state does: raise chi.

    Every corner is a 1x1 matrix there, so its sum-normalised spectrum is
    ``[1]`` however the environment moves, and the loop reads each corner as
    still rank <= 1 (``_forced_corner_rank``, #903).  The loop's text for that
    case says the energy will not respond to ``chi``.  At ``chi=1`` it does:
    this state gives ``-0.64987`` at chi 1, then ``-0.65431`` and ``-0.65864``
    at chi 2 and 4, where the CTM converges (measured on eba0622d).
    """
    gate, A0, _ = uniform_state
    text = _ipeps_ctm_warning(gate, (A0, A0), CTMConfig(chi=1, max_iter=3))

    # Regime: an infinite criterion after sweeps that did compare spectra.
    assert "max_iter=3 sweeps" in text and "(final criterion inf)" in text, text
    assert "Raise chi" in text, text
    assert "will not respond to chi" not in text, text


def test_ipeps_energy_does_not_depend_on_which_sublattice_is_called_A():
    """``(A, B)`` and ``(B, A)`` are the same state, one site apart.

    The energy per site of a translation-invariant gate cannot change when
    the two sublattices swap names.  A sum over only the bonds that start on
    ``A`` does change, because the swap turns them into the bonds that start
    on ``B``.  The regime assertion keeps this test from going vacuous: it
    needs a state whose two kinds of bond really differ.  At ``2cb7c575``:
    ``-0.62569`` for ``(A, B)`` and ``-0.66244`` for ``(B, A)``.
    """
    gate = heisenberg_gate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, (A, B), _ = ipeps(
            gate,
            None,
            iPEPSConfig(max_bond_dim=D, num_imaginary_steps=200, dt=0.05),
            compute_energy=False,
        )
    A, B = A.todense(), B.todense()

    # Regime: the bonds that start on A and on B differ, measured with the
    # Tensor-protocol 2-site CTM, which this test does not exercise.
    At, Bt = _wrap_as_dense_tensor(A), _wrap_as_dense_tensor(B)
    eA, eB = ctm_tensor_2site(At, Bt, CHI, max_iter=CTM.max_iter, conv_tol=CTM.conv_tol)
    H = gate.todense().reshape(DPHYS, DPHYS, DPHYS, DPHYS)

    def bonds(X, Y, eX, eY):
        rdms = (_rdm2x1_tensor_2site(X, Y, eX, eY), _rdm1x2_tensor_2site(X, Y, eX, eY))
        return sum(float(jnp.einsum("ijkl,klij->", r, H).real) for r in rdms)

    assert abs(bonds(At, Bt, eA, eB) - bonds(Bt, At, eB, eA)) > 1e-2

    assert _measure(gate, (A, B)) == pytest.approx(_measure(gate, (B, A)), abs=1e-8)


# ---------------------------------------------------------------------------
# The public pieces ipeps() is built from: ctm_2site, compute_energy_ctm_2site.
# ---------------------------------------------------------------------------


def _measure_standalone(gate, A, B, ctm=CTM):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        env_A, env_B = ctm_2site(A, B, ctm)
    return float(compute_energy_ctm_2site(A, B, env_A, env_B, gate.todense(), DPHYS))


@pytest.mark.parametrize("seed", [1, 2])
def test_ctm_2site_energy_does_not_depend_on_the_bond_gauge(uniform_state, seed):
    """The standalone entry points obey the same rule as ``ipeps()``."""
    gate, A0, E1 = uniform_state
    A, B = _gauge_pair(A0, seed)

    assert _measure_standalone(gate, A, B) == pytest.approx(E1, abs=1e-8)


def test_compute_energy_ctm_2site_counts_every_bond():
    """The dimer state again, through ``ctm_2site`` and the raw energy."""
    A, B = _dimer_pair()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onverge", category=UserWarning)
        E = _measure_standalone(heisenberg_gate(), A, B, CTMConfig(chi=4, max_iter=50))

    assert E == pytest.approx(DIMER_ENERGY, abs=1e-10)
