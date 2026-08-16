"""#881 review: the sublattice diagnostic, and restarting from the pair.

``fpeps()`` returns two site tensors because the t-V ground state at finite
``V`` is a checkerboard charge-density wave, which is inherently two-site.  The
diagnostic that ships with it has one job: tell a caller how much **charge
order** the returned pair carries.

It is a **one-body** probe, and these tests are careful not to claim otherwise.
A nonzero value is evidence of charge order; a zero is not evidence that one
tensor would have sufficed, because a columnar-dimer or bond-ordered state has
identical on-site densities on both sublattices and reads zero while being
genuinely two-site.

The first version could not measure even that.  It compared the singular values
of each leg's Gram matrix ``M = T T†``, and those are not gauge invariant: under
a bond gauge ``T -> G T`` the matrix goes to ``G M G†``, whose spectrum moves
unless ``G`` is unitary -- and simple update's gauge is not.  So it reported a
difference between two *representations*, not between two states.  This is the
same trap as ``||A - B|| ~ 1.7`` on a provably uniform pair.

These tests pin the replacement: the trace distance between the two
sublattices' one-site reduced density matrices, traced out of the two-site RDM
the energy already uses.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
from tenax.algorithms._split_ctm_tensor_energy import (
    compute_energy_split_ctm_tensor_2site,
)
from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _fpeps_simple_update,
    _initialize_fpeps,
    fpeps,
    spinless_fermion_gate,
    sublattice_gap,
)
from tenax.algorithms.ipeps_simple_update import _to_physical_pair
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

CHI = 4


def _gram_gap(A, B):
    """The metric this replaced: leg-wise Gram singular values.

    Kept in the test file, not the library, precisely because the test below
    shows it moving under a transformation that does not touch the state.
    """

    def leg_matrix(t, leg):
        labels = t.labels()
        arr = np.asarray(t.todense())
        arr = np.moveaxis(arr, labels.index(leg), 0)
        arr = arr.reshape(arr.shape[0], -1)
        return arr @ arr.conj().T

    gaps = []
    for leg in ("u", "d", "l", "r"):
        sa, sb = (
            np.sort(np.linalg.svd(leg_matrix(t, leg), compute_uv=False))[::-1]
            for t in (A, B)
        )
        gaps.append(float(np.linalg.norm(sa - sb) / max(float(sa[0]), 1e-300)))
    return max(gaps)


#: The gauge factors, one per checkerboard bond. Diagonal in the charge basis so
#: the FermionParity block structure survives, and deliberately **not** unitary
#: -- a unitary gauge would leave even the Gram spectrum alone and prove nothing.
_GAUGE = (
    np.array([2.0, 0.5]),  # h_AB
    np.array([1.5, 0.8]),  # h_BA
    np.array([0.7, 1.3]),  # v_AB
    np.array([1.1, 2.2]),  # v_BA
)


def _bond_gauge(A, B, mispair=False):
    """Insert ``G G^-1`` on each of the four checkerboard bonds.

    Every bond of the infinite lattice gets a factor and its inverse on the two
    tensors it joins, so the contracted network -- and therefore every physical
    observable -- is unchanged.

    ``mispair=True`` puts ``h_AB``'s inverse on ``B.r`` instead of ``B.l``.  The
    per-leg factors are identical; only the *pairing* is wrong, so it is not a
    gauge at all and the state really does move.  It exists so the guards below
    can be shown to fail on it -- a witness that cannot fail is not a witness,
    which is the whole subject of this file.
    """
    g_hAB, g_hBA, g_vAB, g_vBA = _GAUGE
    A = scale_bond_axis(A, "r", g_hAB)  # h_AB: A.r <-> B.l
    B = scale_bond_axis(B, "r" if mispair else "l", 1.0 / g_hAB)
    B = scale_bond_axis(B, "r", g_hBA)  # h_BA: B.r <-> A.l
    A = scale_bond_axis(A, "l", 1.0 / g_hBA)
    A = scale_bond_axis(A, "d", g_vAB)  # v_AB: A.d <-> B.u
    B = scale_bond_axis(B, "u", 1.0 / g_vAB)
    B = scale_bond_axis(B, "d", g_vBA)  # v_BA: B.d <-> A.u
    A = scale_bond_axis(A, "u", 1.0 / g_vBA)
    return A, B


@pytest.fixture(scope="module")
def su_pair():
    """A short D=2 t-V simple-update run at V=4, physical (CTM-contractable).

    Strong CDW: the gap saturates at ~1.0 here, which is what makes it the right
    end of the V response in ``test_the_gap_tracks_the_charge_density_wave`` --
    and exactly what makes it the *wrong* fixture for the gauge test, see
    ``midgap_pair``.
    """
    cfg = FPEPSConfig(D=2, t=1.0, V=4.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    A, B, lam = _fpeps_simple_update(A0, H, max_D=cfg.D, dt=cfg.dt, steps=8)
    return _to_physical_pair(A, B, lam)


@pytest.fixture(scope="module")
def midgap_pair():
    """A pair whose gap sits **mid-range**, for the gauge tests.

    At ``V=4`` the gap is 1.000437 ungauged, 1.000059 under the correct gauge
    and 0.999419 under a *mispaired* one -- the observable is saturated, so it
    barely moves for a state change that is real and large.  An invariance test
    on a saturated observable proves close to nothing: it would pass on a
    diagnostic that had been replaced by ``return 1.0``.

    The energy witness is worse than weak there, it is **inverted**: on that
    fixture the mispaired gauge moves ``E`` by 2.080e-03 while the *correct*
    gauge moves it by 3.195e-03, so no bar separates them in the right
    direction at all.  The cause is #392 -- with no chemical potential ``E`` is
    ~0 at ``V=4`` (-1.9e-03 here), so the residuals are noise about nothing.

    ``V=1`` puts the gap at ~0.27, in the responsive part of its range, and the
    energy at ~1.5, which restores both witnesses.
    """
    cfg = FPEPSConfig(D=2, t=1.0, V=1.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    A, B, lam = _fpeps_simple_update(A0, H, max_D=cfg.D, dt=cfg.dt, steps=8)
    return (*_to_physical_pair(A, B, lam), H)


#: CTM settings and the bracket the two gauge tests share.  ``BAR_*`` sits
#: *between* the measured correct-gauge and mispaired-gauge residuals, and both
#: sides are asserted -- see ``test_a_mispaired_gauge_is_caught``.  Raising a bar
#: to rescue the invariance test breaks the mutation test and vice versa, which
#: is the property that makes this a guard rather than a decoration.
#:
#: 40 sweeps rather than 12, which buys a wider bracket on the gap: at 12 sweeps
#: with the constants of the day the separation was 2.06x on the mispaired side
#: (dgap 2.882e-01 against BAR_GAP 0.14), and here it is 3.6x.  That is what the
#: extra ~90 s per environment is for.
#:
#: **These constants are calibrated at exactly chi=4 / 40 sweeps and do not
#: survive a change of either.**  Do not raise ``GAUGE_SWEEPS`` for "better
#: convergence": the environment does not converge here at all (``conv_tol``
#: 1e-10 is met at no sweep in any of the three runs -- the corner spectrum is
#: still moving by ~1e-01 at sweep 40), and the residual is **non-monotone** in
#: sweeps.  Measured ``dgap`` for the correct gauge: 6.7e-02 (12), 5.4e-02 (20),
#: 1.2e-01 (30), 5.2e-02 (35), 1.6e-02 (38), 6.1e-02 (39), 1.3e-02 (40).  With
#: the bars below, the invariance test *fails* at 12, 20, 30, 35 and 39 sweeps
#: and passes only at 38 and 40.  The separation is a property of this operating
#: point, not a convergence trend, and an innocent-looking bump to the sweep
#: count will turn this file red for reasons that read as inexplicable.
#:
#: For the same reason the 3.6x below is a *separation*, not a robustness
#: margin: perturbing the input pair by 2.7e-12 relative moves the
#: mutation-side margin to 1.73x and ``E_mispaired`` from 5.79 to 0.42 -- an
#: O(1) response to a 1e-12 input.  Determinism holds within one binary; it
#: should not be expected to survive a different BLAS.  This file is
#: ``algorithm``-bucketed (see ``tests/conftest.py``), so it is deselected by
#: the ``-m core`` required checks and a flake lands on the full suite rather
#: than on the merge gate.
GAUGE_CHI, GAUGE_SWEEPS = 4, 40
#: correct 2.357e-02 (12.7x under), mispaired 4.307 (14.4x over).  0.3 rather
#: than the exact geometric mean 0.3186, so the two sides are comparable but
#: not equal.
BAR_E = 0.3
#: correct 1.295e-02 (3.6x under), mispaired 1.710e-01 (3.6x over).  This one
#: is the geometric mean to four figures (0.04705).
BAR_GAP = 0.047


def _observables(A, B, H):
    """``(E, gap)`` from a freshly converged environment for this pair."""
    envs = ctm_split_tensor_2site(
        A, B, GAUGE_CHI, max_iter=GAUGE_SWEEPS, conv_tol=1e-10
    )
    d = A.indices[A.labels().index("phys")].dim
    E = float(compute_energy_split_ctm_tensor_2site(A, B, *envs, H, d=d))
    return E, sublattice_gap(A, B, *envs)


@pytest.fixture(scope="module")
def midgap_baseline(midgap_pair):
    """``(A, B, H, E, gap)`` for the ungauged pair -- computed once, not twice.

    Both gauge tests need the ungauged numbers to compare against, and a CTM at
    these settings is ~90 s.
    """
    A, B, H = midgap_pair
    E, gap = _observables(A, B, H)
    return A, B, H, E, gap


def test_the_gap_is_invariant_under_a_bond_gauge(midgap_baseline):
    """The state does not change, so the diagnostic must not either.

    The gauge leaves every physical observable alone by construction -- each
    bond carries a factor and its inverse.  A diagnostic that moves under it is
    reading the representation, not the state.

    "By construction" is checked, not assumed: the energy from the same two
    environments must agree across the gauge before any claim is made about the
    diagnostic.  Otherwise a mis-written gauge would move the state, a *correct*
    diagnostic would move with it, and this test would fail on the fix and pass
    on the defect.

    **Why the bars are 3e-1 and 4.7e-2 rather than something tiny.**  The
    environment is re-converged on the gauged pair, and a CTM at finite chi
    truncates in a basis the gauge moves, so the two runs are not algebraically
    the same calculation and the correct gauge leaves a residual of its own:
    2.357e-02 (energy) and 1.295e-02 (gap) here.

    That residual is an artefact of *this* operating point and not a
    convergence error being driven down -- the environment never converges at
    these settings, and the residual is non-monotone in the sweep count.  See
    the note on ``GAUGE_SWEEPS`` above before changing anything about how the
    environments are built.

    A saturated fixture reports much smaller numbers -- at V=4 the gap moves by
    3.779e-04 -- but only because a saturated observable barely moves for
    anything, including for a state change that is real and large.  That is
    insensitivity, not precision, and on that fixture the energy witness is
    *inverted*: the mispaired gauge moves E by 2.080e-03 against the correct
    gauge's own 3.195e-03, so no bar separates them in the right direction at
    all.  This is why ``midgap_pair`` exists.

    What makes the bar meaningful is not its size but that it is **bracketed**:
    ``test_a_mispaired_gauge_is_caught`` requires the same constants to fail on
    a transformation that is *not* a gauge, so neither bar can be moved in
    either direction without breaking one of the two tests.
    """
    A, B, H, E, gap = midgap_baseline
    A_g, B_g = _bond_gauge(A, B)
    E_g, gap_g = _observables(A_g, B_g, H)

    assert 0.05 < gap < 0.95, (
        f"gap {gap:.4f} is at the edge of its range -- a saturated observable "
        f"is invariant under everything, so the assertions below would be weak "
        f"even if they passed"
    )
    assert abs(E - E_g) < BAR_E, (
        f"the 'gauge' moved the energy from {E:.8f} to {E_g:.8f} -- it is not a "
        f"gauge transformation, so nothing below is about gauge invariance"
    )
    assert abs(gap - gap_g) < BAR_GAP, (
        f"sublattice_gap moved from {gap:.8f} to {gap_g:.8f} under a pure bond "
        f"gauge -- it is measuring the gauge, not the state (#881 P2-3)"
    )

    # And the metric this replaced does move, by orders of magnitude more than
    # either bar.  This is the finding, pinned.
    gram, gram_g = _gram_gap(A, B), _gram_gap(A_g, B_g)
    assert abs(gram - gram_g) > 1.0, (
        f"the Gram metric read {gram:.6f} -> {gram_g:.6f} under the same gauge; "
        f"if it no longer moves, this test has stopped discriminating"
    )


def test_the_mispairing_stays_a_single_relocated_inverse(midgap_pair):
    """The severity of the mutation is pinned, so it cannot be inflated.

    ``BAR_E`` and ``BAR_GAP`` are bracketed from both sides, but the *violence*
    of the mutation that pins them from below is a free knob: making the
    mispairing more destructive would ease ``test_a_mispaired_gauge_is_caught``
    and nothing would object.  This closes that.

    The mispaired transform must be exactly "one bond's inverse on the wrong
    leg" and nothing more, which is a statement about the tensors and needs no
    environment:

    * ``A`` is untouched by the mutation -- it only ever moves a factor on ``B``.
    * ``B_mispaired`` is ``B_correct`` with ``h_AB``'s factor taken off ``l`` and
      put onto ``r``: the same numbers, relocated across one bond.

    If someone strengthens ``_GAUGE`` for the mutation only, mispairs a second
    bond, or reaches for a different perturbation entirely, this fails.
    """
    A, B, _H = midgap_pair
    A_c, B_c = _bond_gauge(A, B)
    A_m, B_m = _bond_gauge(A, B, mispair=True)

    np.testing.assert_allclose(
        np.asarray(A_m.todense()),
        np.asarray(A_c.todense()),
        rtol=1e-13,
        atol=1e-15,
        err_msg="the mispairing touched A; it must only relocate a factor on B",
    )

    g_hAB = _GAUGE[0]
    want = scale_bond_axis(scale_bond_axis(B_c, "l", g_hAB), "r", 1.0 / g_hAB)
    np.testing.assert_allclose(
        np.asarray(B_m.todense()),
        np.asarray(want.todense()),
        rtol=1e-13,
        atol=1e-15,
        err_msg=(
            "the mispairing is not a single relocated inverse -- it has been "
            "made more violent than the mutation the bars are calibrated "
            "against, which weakens test_a_mispaired_gauge_is_caught"
        ),
    )


def test_a_mispaired_gauge_is_caught(midgap_baseline):
    """The bars above must fail on something that is *not* a gauge.

    This is the mutation check, kept in the suite rather than run once by hand.
    ``_bond_gauge(mispair=True)`` applies the identical per-leg factors but puts
    ``h_AB``'s inverse on ``B.r`` instead of ``B.l``.  Nothing cancels on that
    bond, so the physical state genuinely moves -- and both witnesses must say
    so, or they are decorations.  How far it may move is itself pinned, by
    ``test_the_mispairing_stays_a_single_relocated_inverse``.

    Measured on ``midgap_pair`` at chi=4, 40 sweeps: the energy moves 4.307
    against the correct gauge's 2.357e-02, and the gap 1.710e-01 against
    1.295e-02.
    ``BAR_E`` and ``BAR_GAP`` sit between the two, so this test and the one
    above bracket them from opposite sides.  A previous version of the guard
    used ``abs(E - E_g) < 2e-2 * max(abs(E), 1.0)`` on the V=4 fixture, where
    ``E ~ 0`` (#392) collapsed the relative bar to an absolute 2e-2 -- ten times
    the whole magnitude of ``E`` -- and this mutation passed it.
    """
    A, B, H, E, gap = midgap_baseline
    A_m, B_m = _bond_gauge(A, B, mispair=True)
    E_m, gap_m = _observables(A_m, B_m, H)

    assert abs(E - E_m) > BAR_E, (
        f"a mispaired gauge moved the energy only {abs(E - E_m):.3e} "
        f"({E:.8f} -> {E_m:.8f}), inside BAR_E={BAR_E} -- the energy witness in "
        f"test_the_gap_is_invariant_under_a_bond_gauge cannot fail, so it is "
        f"not checking anything"
    )
    assert abs(gap - gap_m) > BAR_GAP, (
        f"a mispaired gauge moved the gap only {abs(gap - gap_m):.3e} "
        f"({gap:.8f} -> {gap_m:.8f}), inside BAR_GAP={BAR_GAP} -- the "
        f"invariance assertion cannot fail, so it is not checking anything"
    )


def test_the_gap_tracks_the_charge_density_wave(su_pair):
    """The diagnostic has to answer the question it exists for.

    Gauge invariance alone is satisfied by any constant.  What makes this the
    right probe is that it moves with the physics it claims to report: the t-V
    checkerboard CDW is driven by ``V``, and at ``V = 0`` the model is free
    fermions with no charge order at all.  Measured on the same 8-step D=2 sweep
    at chi=4: ``V=0`` gives **0.037**, ``V=1`` **0.270**, ``V=2`` **0.900** and
    ``V=4`` **1.000** -- from "these two sites are the same" to "these two sites
    are perfectly distinguishable", which for spinless fermions is the fully
    polarised occupied/empty checkerboard.
    """
    cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    A, B, lam = _fpeps_simple_update(A0, H, max_D=cfg.D, dt=cfg.dt, steps=8)
    A, B = _to_physical_pair(A, B, lam)
    kw = dict(max_iter=12, conv_tol=1e-10)
    gap_free = sublattice_gap(A, B, *ctm_split_tensor_2site(A, B, CHI, **kw))

    A4, B4 = su_pair
    gap_cdw = sublattice_gap(A4, B4, *ctm_split_tensor_2site(A4, B4, CHI, **kw))

    assert gap_free < 0.1, (
        f"gap {gap_free:.4f} at V=0, where free spinless fermions have no "
        f"charge order -- a probe that reports a CDW here is not reading one"
    )
    assert gap_cdw > 0.5, (
        f"gap {gap_cdw:.4f} at V=4, where the ground state is a strong "
        f"checkerboard CDW -- the probe is not seeing the order it exists for"
    )


def test_the_gap_never_densifies_the_site_tensor(su_pair, monkeypatch):
    """Block-sparse throughout: nothing of size ``D**4 * d`` is densified.

    The Gram version called ``todense()`` on the full rank-5 site tensor, twice
    per leg on both sites -- eight ``D**4 * d`` arrays for a diagnostic that
    only ever needed a ``d``-by-``d`` matrix.  On the fermionic path that is the
    memory advantage the whole block-sparse representation exists for (#881
    P1-2, and ``CLAUDE.md``'s standing rule).
    """
    A, B = su_pair
    envs = ctm_split_tensor_2site(A, B, CHI, max_iter=12, conv_tol=1e-10)

    original = SymmetricTensor.todense
    ranks = []

    def recording_todense(self, *args, **kwargs):
        ranks.append(len(self.indices))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SymmetricTensor, "todense", recording_todense)
    sublattice_gap(A, B, *envs)

    assert 5 not in ranks, (
        f"sublattice_gap densified a rank-5 tensor (todense ranks seen: "
        f"{sorted(set(ranks))}) -- that is the D**4 * d site tensor (#881 P1-2)"
    )


def test_sublattice_gap_is_exported():
    """New public API must be importable from the top-level package."""
    import tenax
    import tenax.algorithms as algos

    assert "sublattice_gap" in tenax.__all__
    assert "sublattice_gap" in algos.__all__
    assert tenax.sublattice_gap is sublattice_gap
    assert algos.sublattice_gap is sublattice_gap


def test_fpeps_restarts_from_its_own_returned_pair():
    """The returned pair goes straight back in as ``initial_tensor``.

    ``fpeps()`` returns two tensors, so a single-tensor-only initializer cannot
    warm-start from its own output: passing the tuple used to reach
    ``tuple.relabel`` and passing only ``A`` threw away the sublattice structure
    the pair exists to carry.

    The pair is returned in *physical* form for the same reason.  Handing back
    the bare Vidal ``Gamma`` would restart on a different state, because the
    bond weights live outside it and the restart resets them to ones.
    """
    cfg = FPEPSConfig(
        D=2,
        t=1.0,
        V=4.0,
        dt=0.05,
        num_imaginary_steps=4,
        ctm_chi=4,
        ctm_max_iter=12,
        ctm_conv_tol=1e-6,
    )
    H = spinless_fermion_gate(cfg)
    E1, (A1, B1), _ = fpeps(H, cfg, key=jax.random.PRNGKey(5))

    # A zero-step restart must reproduce the state it was handed, which is what
    # makes the round trip meaningful: same tensors in, same energy out.
    cfg0 = dataclasses.replace(cfg, num_imaginary_steps=0)
    E0, (A0, B0), _ = fpeps(H, cfg0, initial_tensor=(A1, B1))
    assert E0 == pytest.approx(E1, rel=1e-8, abs=1e-8), (
        f"restarting on the returned pair with 0 steps gave E={E0:.10f} where "
        f"the run that produced it gave E={E1:.10f} -- the pair does not "
        f"round-trip (#881 P2-4)"
    )
    for name, before, after in (("A", A1, A0), ("B", B1, B0)):
        np.testing.assert_allclose(
            np.asarray(after.todense()),
            np.asarray(before.todense()),
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"a 0-step restart changed sublattice {name}",
        )

    # And a further evolution from the pair runs and keeps both sublattices.
    E2, (A2, B2), _ = fpeps(H, cfg, initial_tensor=(A1, B1))
    assert np.isfinite(E2)
    for t in (A2, B2):
        assert isinstance(t, SymmetricTensor)
