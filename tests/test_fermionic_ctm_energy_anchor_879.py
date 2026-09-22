"""#879: a magnitude anchor on the fermionic CTM energy path.

Before this, every fermionic energy assertion in the tree was a finiteness /
positivity / shape check -- ``compute_energy_split_ctm_tensor_2site`` (the path
``fpeps()`` actually returns) could return *any* finite real number and no test
objected.  That is exactly how #878 hid for months: ``fpeps()`` returned exactly
zero while every ``isfinite`` check passed on the corpse.

Two anchors live here, and they fail for different reasons on purpose.

**The analytic-limit anchor (primary).**  The fully-occupied product state has a
per-site energy that is *exactly* ``2V``: the hopping term cannot act (there is
no empty site to hop into), ``n_i n_j = 1`` on every NN bond, and the square
lattice has two NN bonds per site.  That value is read off the gate itself
below, not asserted from prose.  The state is approached rather than used
directly -- see the next paragraph -- and the test asserts both that the energy
sits at ``2V`` and that the deviation *shrinks* as the state approaches the
product limit.  The shrink half is what makes it an anchor rather than a
tolerance: a path that is merely near ``2V`` by luck does not converge to it.

**Why the exact product state is approached, not used.**  Feeding the exact
product tensor to the CTM contracts the network to zero -- a product state has
rank-1 virtual bonds, so the corner is degenerate and the RDM comes back with
``|tr - 1| = 1`` (#845).  So the fixture is ``normalize(A_product + eps R)``,
whose energy is ``2V + O(eps^4)`` (measured: 1.7e-3, 2.2e-5, 1.9e-7, 2.4e-9,
2.4e-13 at eps = 0.3, 0.1, 0.03, 0.01, 0.001).  This is a *constructed* fixture:
no simple update runs, so unlike the CDW anchor below it cannot drift when the
SU truncation changes.

**A structural note, because the record here was wrong.**  The previous revision
of this file stated that "charge conservation forbids a hand-built product
state".  That is true of the *CDW* product state and false in general, and the
distinction matters because it is what forces the fixture above to be
all-occupied rather than the more obvious staggered state.  On a 2-site
checkerboard, sublattice ``A``'s four legs and sublattice ``B``'s four legs are
the *same four bonds* (``A.u`` pairs with ``B.d``, ``A.d`` with ``B.u``, and so
on), so both sites carry the same virtual parity ``q_u + q_d + q_l + q_r``.  A
fully-polarised CDW needs ``A`` odd (occupied) and ``B`` even (empty), which
that shared sum cannot deliver.  The all-occupied state needs *both* odd, which
it can: one bond at charge 1 and the rest at charge 0.  So the obstruction is
specific to the staggered pattern, and the all-occupied state builds fine --
what stops it being used bare is the degenerate corner, a different problem.

**The CDW anchor (secondary).**  This one pins the energy on a state the SU
actually produces, so it covers ``fpeps()`` end to end.  With the particle-hole
chemical potential ``mu = 2V`` grafted on (``-(mu/4)(n_i + n_j)`` per bond, the
/4 splitting each site's ``-mu n`` across its four NN bonds), the CDW's per-site
energy is ``-V``.  Its fixture is SU-evolved from a *finite* occupation bias, so
it is only approximately the polarised CDW -- the regime is therefore asserted
before the energy, so that an SU improvement which legitimately retains quantum
fluctuations reports "the fixture left its regime" instead of masquerading as a
broken CTM energy.

Both anchors gate on the RDMs being **PSD** first (the #854 rule).  A fermionic
energy is an expectation value ``tr(rho H)`` only when ``rho`` is a density
matrix; on a non-PSD environment neither the manual trace nor the function is
bounded by physics, so an energy assertion on one would be meaningless.  All
**four** bonds are checked: ``compute_energy_split_ctm_tensor_2site`` routes
through the multisite path, which evaluates horizontal and vertical RDMs for
both ``A -> B`` and ``B -> A``, so gating only the ``A -> B`` pair would let an
inversion-asymmetric regression through the validity gate it is supposed to
guard.  That widening is not cosmetic: on the CDW fixture the *worst* of the
four margins is ``v B -> A``, the one the narrower gate never looked at.

The PSD tolerance here is 1e-6 rather than the 1e-8 used elsewhere, because on
a near-product state the exact RDM is rank 1 and the measured minimum is CTM
error rather than physics -- see ``_PSD_TOL`` for the chi scan that shows a
1e-8 gate flipping sign between chi=4 and chi=6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
from tenax.algorithms._split_ctm_tensor_energy import (
    _rdm1x2_split_tensor_2site,
    _rdm2x1_split_tensor_2site,
    compute_energy_split_ctm_tensor_2site,
)
from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _fpeps_simple_update,
    _initialize_fpeps,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_simple_update import _to_physical_pair
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

#: min-eigenvalue / spectral-radius floor for "this RDM is a density matrix".
#:
#: Deliberately 100x looser than ``_ctm_diagnostics.RDM_PSD_TOL`` (1e-8), and
#: the reason is structural rather than a concession.  Both fixtures here are
#: *near-product* states, so the exact bond RDM is rank 1 -- three of its four
#: eigenvalues are exactly zero.  The measured minimum is therefore not a
#: physical negativity but the CTM's own error, which lands around 1e-8 and does
#: not shrink monotonically with chi.  Measured worst margin on the CDW fixture:
#:
#:     chi =  4   -4.09e-09      chi =  8   -7.07e-10
#:     chi =  6   -5.74e-08      chi = 12   -6.60e-09
#:
#: A 1e-8 gate passes chi=4 by 2.4x and *fails* chi=6 by 5.7x, i.e. it asserts
#: that roundoff happens to land on the favourable side of a threshold -- the
#: previous revision of this file was green for that reason and no better one.
#: The non-PSD environments #854 exists to catch are nothing like this size:
#: the ones in the tree report a smallest eigenvalue ~0.67 *of the spectral
#: radius* below zero.  1e-6 therefore keeps five orders of discrimination
#: against a real collapse while sitting two orders above the noise, and every
#: chi in the table above clears it.
_PSD_TOL = 1e-6

#: The two distances from the product limit used by the analytic anchor.  The
#: deviation from ``2V`` runs as ``O(eps^4)``, so a 3x step in ``eps`` buys ~80x
#: in accuracy; the test demands only 10x, which every seed clears by ~8x.
_EPS_COARSE = 0.3
_EPS_FINE = 0.1

#: Measured |E - 2V| at ``_EPS_FINE`` over seeds {0, 1, 17, 123, 2024}:
#: 6.2e-6, 4.2e-5, 2.2e-5, 4.1e-5, 4.2e-6.  1e-3 clears the worst by ~24x while
#: staying ~4 orders below the "≈ 0" failure mode it exists to catch.
_ANALYTIC_ABS_TOL = 1e-3


def _fpeps_indices(D: int, d: int) -> tuple[TensorIndex, ...]:
    """The (u, d, l, r, phys) index tuple ``_build_initial_fpeps_tensor`` uses."""
    sym = FermionParity()
    virt = np.array([i % 2 for i in range(D)], dtype=np.int32)
    phys = np.array([i % 2 for i in range(d)], dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, virt, FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, virt, FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, virt, FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, virt, FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys, FlowDirection.IN, label="phys"),
    )


def _all_occupied_pair(D: int = 2, d: int = 2):
    """The exact fully-occupied product state as a 2-site checkerboard pair.

    Both sites sit at physical index 1 (occupied).  The single vertical bond
    ``A.u <-> B.d`` carries charge 1 and every other bond carries charge 0,
    which is the parity assignment that closes on *both* sublattices (see the
    module docstring).
    """
    idx = _fpeps_indices(D, d)

    def site(u, dn, ll, r, p):
        dense = np.zeros((D, D, D, D, d))
        dense[u, dn, ll, r, p] = 1.0
        return SymmetricTensor.from_dense(jnp.array(dense), idx)

    return site(1, 0, 0, 0, 1), site(0, 1, 0, 0, 1)


def _perturbed_all_occupied_pair(eps: float, seed: int, D: int = 2, d: int = 2):
    """``normalize(A_product + eps R)`` -- the product limit, approached."""
    idx = _fpeps_indices(D, d)
    A0, B0 = _all_occupied_pair(D, d)
    kA, kB = jax.random.split(jax.random.PRNGKey(seed))
    out = []
    for base, key in ((A0, kA), (B0, kB)):
        R = SymmetricTensor.random_normal(idx, key)
        T = base + R * (eps / float(R.norm()))
        out.append(T * (1.0 / float(T.norm())))
    return out[0], out[1]


def _gate_with_mu(cfg: FPEPSConfig, mu: float) -> SymmetricTensor:
    """``spinless_fermion_gate`` plus a diagonal ``-(mu/4)(n_i + n_j)`` per bond.

    The added term is diagonal in the occupation basis, so it preserves the
    ``FermionParity`` block structure -- it is grafted straight onto the dense
    4x4 gate and re-wrapped with the same indices.
    """
    g = spinless_fermion_gate(cfg)
    h = np.array(g.todense()).reshape(4, 4)
    # basis order |00>, |01>, |10>, |11>  ->  n_i + n_j = 0, 1, 1, 2
    h = h + np.diag(-(mu / 4.0) * np.array([0.0, 1.0, 1.0, 2.0]))
    return SymmetricTensor.from_dense(jnp.array(h.reshape(2, 2, 2, 2)), g.indices)


def _seeded_cdw_pair(cfg: FPEPSConfig, gate: SymmetricTensor, steps: int):
    """A deterministically CDW-seeded, SU-evolved ``(A, B)`` checkerboard pair."""
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    # Strong (16x) opposite occupation bias on the two sublattices -- seeds the
    # symmetry breaking so the CDW is reproduced on every build (#999).
    a = scale_bond_axis(A0, "phys", jnp.array([4.0, 0.25]))
    b = scale_bond_axis(A0, "phys", jnp.array([0.25, 4.0]))
    A, B, lam = _fpeps_simple_update(a, gate, max_D=cfg.D, dt=cfg.dt, steps=steps, B=b)
    return _to_physical_pair(A, B, lam)


def _four_bond_rdms(A, B, env_A, env_B) -> dict[str, np.ndarray]:
    """Every bond RDM that enters ``compute_energy_split_ctm_tensor_2site``.

    That function routes through the multisite path, which counts 4 NN bonds --
    horizontal and vertical for ``A -> B`` *and* for ``B -> A``.  Gating only
    the ``A -> B`` pair would leave half the energy's inputs unvalidated.
    """
    return {
        "h A->B": _rdm2x1_split_tensor_2site(A, B, env_A, env_B),
        "v A->B": _rdm1x2_split_tensor_2site(A, B, env_A, env_B),
        "h B->A": _rdm2x1_split_tensor_2site(B, A, env_B, env_A),
        "v B->A": _rdm1x2_split_tensor_2site(B, A, env_B, env_A),
    }


def _as_matrix(rdm) -> np.ndarray:
    return np.array(rdm).reshape(4, 4)


def _psd_margin(rdm) -> float:
    """Smallest eigenvalue over spectral radius; ``>= -_PSD_TOL`` means PSD."""
    m = _as_matrix(rdm)
    m = 0.5 * (m + m.conj().T)
    ev = np.linalg.eigvalsh(m)
    return float(ev.min() / max(abs(ev).max(), 1e-300))


def _occupations(rdm) -> tuple[float, float]:
    """``(<n_first>, <n_second>)`` from a trace-normalised bond RDM.

    Basis order |00>, |01>, |10>, |11>: the first site is occupied in the last
    two entries, the second in entries 1 and 3.
    """
    m = _as_matrix(rdm)
    tr = np.trace(m)
    if abs(tr) > 1e-300:
        m = m / tr
    return float(np.real(m[2, 2] + m[3, 3])), float(np.real(m[1, 1] + m[3, 3]))


def _assert_all_four_rdms_psd(rdms: dict[str, np.ndarray], what: str) -> None:
    margins = {k: _psd_margin(v) for k, v in rdms.items()}
    bad = {k: m for k, m in margins.items() if m < -_PSD_TOL}
    assert not bad, (
        f"the {what} environment is not PSD on {sorted(bad)} "
        f"(margins { {k: f'{m:.2e}' for k, m in margins.items()} }) -- the #854 "
        f"gate.  The energy below is not an expectation value on a "
        f"non-density-matrix, so the anchor would be void"
    )


def _energy_and_rdms(A, B, gate, chi: int = 8, max_iter: int = 40):
    env_A, env_B = ctm_split_tensor_2site(
        A, B, chi=chi, max_iter=max_iter, conv_tol=1e-11
    )
    rdms = _four_bond_rdms(A, B, env_A, env_B)
    E = float(compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, gate, d=2))
    return E, rdms


# --------------------------------------------------------------------------- #
# The analytic-limit anchor                                                    #
# --------------------------------------------------------------------------- #


def test_the_all_occupied_fixture_is_the_analytic_case_it_claims_to_be():
    """Regime guard for the anchor below: the fixture and its expected value.

    Three things have to hold for ``E/site = 2V`` to be the right answer, and
    all three are checked here rather than asserted in the docstring:

    * the exact product tensors are *representable* -- charge conservation does
      not zero them (this is what the staggered CDW cannot manage, and getting
      it wrong would silently hand the anchor a zero tensor);
    * they are genuine product states -- exactly one nonzero amplitude each;
    * the gate really does give ``V`` on the doubly-occupied bond and can only
      hop between states of different occupation, so ``2 bonds/site x V`` is the
      gate's own value and not a number copied into the test.
    """
    D = d = 2
    A0, B0 = _all_occupied_pair(D, d)

    assert float(A0.norm()) == pytest.approx(1.0), (
        "the all-occupied A tensor came back with zero norm -- charge "
        "conservation rejected the block, so the parity assignment is wrong"
    )
    assert float(B0.norm()) == pytest.approx(1.0), (
        "the all-occupied B tensor came back with zero norm -- charge "
        "conservation rejected the block, so the parity assignment is wrong"
    )

    for name, T in (("A", A0), ("B", B0)):
        dense = np.array(T.todense())
        nnz = int(np.count_nonzero(np.abs(dense) > 1e-12))
        assert nnz == 1, (
            f"the {name} fixture has {nnz} nonzero amplitudes, so it is not a "
            f"product state and its energy is not analytically 2V"
        )
        # The one surviving amplitude must sit on the occupied physical state.
        assert np.argmax(np.abs(dense)) % d == 1, (
            f"the {name} fixture's amplitude is not on the occupied physical "
            f"index -- this is not the all-occupied state"
        )

    V = 4.0
    h = np.array(spinless_fermion_gate(FPEPSConfig(D=D, t=1.0, V=V, dt=0.05)).todense())
    h = h.reshape(4, 4)
    # |11> is basis entry 3: the only configuration the all-occupied state sees.
    assert float(np.real(h[3, 3])) == pytest.approx(V), (
        f"the gate's doubly-occupied diagonal is {h[3, 3]!r}, not V = {V}; the "
        f"anchor's expected value 2V is derived from this entry"
    )
    # Hopping is strictly off-diagonal in occupation, so it cannot contribute
    # to a product state -- that is why the expected value is pure interaction.
    assert float(np.real(h[0, 0])) == pytest.approx(0.0), (
        "the gate has a nonzero empty-bond diagonal, so the all-occupied "
        "energy is not 2V by the argument this anchor rests on"
    )


def test_fermionic_ctm_energy_converges_to_the_analytic_product_value():
    """``compute_energy_split_ctm_tensor_2site`` must tend to ``2V`` as the
    state tends to the fully-occupied product limit.

    Mutation coverage: an energy contraction that dropped a bond reads ``V``,
    one that double-counted reads ``4V``, the #878 collapse reads ``~0``, and a
    sign error reads ``-2V`` -- all fail the absolute check by >= 4.  A path
    that is near ``2V`` without converging to it fails the shrink check, which
    no single tolerance would catch.
    """
    V = 4.0
    cfg = FPEPSConfig(D=2, t=1.0, V=V, dt=0.05)
    gate = spinless_fermion_gate(cfg)
    expected = 2.0 * V

    E_coarse, _ = _energy_and_rdms(*_perturbed_all_occupied_pair(_EPS_COARSE, 17), gate)
    A, B = _perturbed_all_occupied_pair(_EPS_FINE, 17)
    E_fine, rdms = _energy_and_rdms(A, B, gate)

    _assert_all_four_rdms_psd(rdms, "near-product")

    # Regime guard: this must still be the all-occupied state, or 2V is not the
    # value it should be converging to.
    for name, rdm in rdms.items():
        n_first, n_second = _occupations(rdm)
        assert n_first == pytest.approx(1.0, abs=1e-2), (
            f"bond {name} has <n> = {n_first:.4f} on its first site, not ~1 -- "
            f"the fixture is no longer the all-occupied state, so the analytic "
            f"2V is the wrong target"
        )
        assert n_second == pytest.approx(1.0, abs=1e-2), (
            f"bond {name} has <n> = {n_second:.4f} on its second site, not ~1 -- "
            f"the fixture is no longer the all-occupied state"
        )

    dev_coarse, dev_fine = abs(E_coarse - expected), abs(E_fine - expected)
    assert dev_fine <= _ANALYTIC_ABS_TOL, (
        f"fermionic CTM energy of the near-product state is {E_fine:.10f}, not "
        f"the analytic 2V = {expected}.  A ~0 here is the #878 failure mode; "
        f"{V} is a dropped bond, {4 * V} a double-counted one, {-expected} a "
        f"sign error"
    )
    assert dev_fine * 10.0 <= dev_coarse, (
        f"the energy does not converge to the analytic limit: |E - 2V| went "
        f"{dev_coarse:.3e} -> {dev_fine:.3e} as eps went {_EPS_COARSE} -> "
        f"{_EPS_FINE}, a factor of {dev_coarse / max(dev_fine, 1e-300):.1f} "
        f"where the O(eps^4) scaling demands >= 10.  Sitting near 2V without "
        f"converging to it means the agreement is a coincidence"
    )


# --------------------------------------------------------------------------- #
# The CDW anchor                                                               #
# --------------------------------------------------------------------------- #


def test_fermionic_ctm_energy_equals_the_cdw_analytic_value():
    """``compute_energy_split_ctm_tensor_2site`` on the polarised CDW must read
    ``-V`` per site (mu = 2V), on a PSD environment.

    Unlike the analytic anchor above this runs the simple update, so it covers
    the state ``fpeps()`` actually returns.  The price is that the fixture is
    only approximately the polarised CDW -- it is SU-evolved from a *finite*
    occupation bias with the hopping term switched on -- so the polarisation is
    asserted before the energy.  If a future SU legitimately retains more
    quantum fluctuation, the regime assertion fires first and says so, instead
    of the energy assertion firing and implicating the CTM.
    """
    V = 4.0
    cfg = FPEPSConfig(D=2, t=1.0, V=V, dt=0.05)
    gate = _gate_with_mu(cfg, mu=2.0 * V)

    A, B = _seeded_cdw_pair(cfg, gate, steps=20)
    E, rdms = _energy_and_rdms(A, B, gate, chi=4, max_iter=30)

    _assert_all_four_rdms_psd(rdms, "CDW")

    # Regime guard: one sublattice full, the other empty.  ``-V`` follows from
    # the polarisation, so if the polarisation has gone the target has too.
    n_A, n_B = _occupations(rdms["h A->B"])
    assert abs(n_A - n_B) >= 0.9, (
        f"the fixture is no longer a polarised CDW: <n_A> = {n_A:.4f}, "
        f"<n_B> = {n_B:.4f}, staggering {abs(n_A - n_B):.4f} < 0.9.  The "
        f"analytic -V assumes full polarisation, so retune or retire this "
        f"fixture -- this is NOT evidence that the CTM energy is wrong"
    )

    assert E == pytest.approx(-V, abs=1e-3), (
        f"fermionic CTM energy of the polarised CDW is {E:.8f}, not the "
        f"analytic -V = {-V} (mu = 2V).  A ~0 here is the #878 failure mode; a "
        f"+V is a sign error; anything else is a broken contraction/normalisation"
    )


# --------------------------------------------------------------------------- #
# The gate's own guard                                                         #
# --------------------------------------------------------------------------- #


def test_the_psd_gate_still_rejects_a_real_collapse_at_this_tolerance():
    """``_PSD_TOL`` is loosened to 1e-6 above; prove that still catches #854.

    Loosening a guard is only safe if it keeps rejecting what it was built for.
    The non-PSD environments in the tree report a smallest eigenvalue around
    ``0.67`` *of the spectral radius* below zero -- five orders above the CTM
    noise the tolerance was raised past.  This feeds the gate one such matrix
    and requires it to fail, so a future tolerance change that quietly disarms
    the gate cannot pass unnoticed.

    Without this, ``_PSD_TOL`` could drift to 1e-1 and every assertion in this
    file would still be green.
    """
    radius = 2.68 / 0.669  # the real #854 report: -2.68, which was 0.669 of radius
    collapsed = np.diag([radius, 0.1, 0.05, -2.68])
    assert _psd_margin(collapsed) == pytest.approx(-0.669, abs=1e-3), (
        "the #854-scale fixture does not reproduce the reported margin, so "
        "this guard is not testing the case it claims to"
    )

    healthy = np.diag([1.0, 0.5, 0.25, 1e-9])
    # The gate must pass a healthy RDM ...
    _assert_all_four_rdms_psd({f"bond{i}": healthy for i in range(4)}, "healthy")

    # ... and must reject a collapsed one, wherever among the four it sits.
    for slot in range(4):
        rdms = {f"bond{i}": healthy for i in range(4)}
        rdms[f"bond{slot}"] = collapsed
        with pytest.raises(AssertionError, match="not PSD"):
            _assert_all_four_rdms_psd(rdms, "collapsed")
