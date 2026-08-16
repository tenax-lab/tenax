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

    At ``V=4`` the gap is 1.0004 ungauged, 1.0001 under the correct gauge and
    0.9998 under a *mispaired* one -- the observable is saturated, so it barely
    moves for a state change that is real and large.  An invariance test on a
    saturated observable proves close to nothing: it would pass on a diagnostic
    that had been replaced by ``return 1.0``.

    ``V=1`` puts the gap at ~0.27, in the responsive part of its range, and as a
    bonus puts the energy at ~1.5 rather than the ~0 that #392 produces at
    ``V=4`` -- which matters because the energy is the witness that the gauge is
    inert, and a witness keyed to a quantity that is ~0 degenerates into an
    absolute bar ten times its own magnitude.
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
GAUGE_CHI, GAUGE_SWEEPS = 4, 12
BAR_E = 0.25  # measured: correct 1.00e-01, mispaired 7.00e-01
BAR_GAP = 0.14  # measured: correct 6.75e-02, mispaired 2.88e-01


def _observables(A, B, H):
    """``(E, gap)`` from a freshly converged environment for this pair."""
    envs = ctm_split_tensor_2site(
        A, B, GAUGE_CHI, max_iter=GAUGE_SWEEPS, conv_tol=1e-10
    )
    E = float(compute_energy_split_ctm_tensor_2site(A, B, *envs, H, d=2))
    return E, sublattice_gap(A, B, *envs)


def test_the_gap_is_invariant_under_a_bond_gauge(midgap_pair):
    """The state does not change, so the diagnostic must not either.

    The gauge leaves every physical observable alone by construction -- each
    bond carries a factor and its inverse.  A diagnostic that moves under it is
    reading the representation, not the state.

    "By construction" is checked, not assumed: the energy from the same two
    environments must agree across the gauge before any claim is made about the
    diagnostic.  Otherwise a mis-written gauge would move the state, a *correct*
    diagnostic would move with it, and this test would fail on the fix and pass
    on the defect.

    **Why the bars are 1e-1 and not 1e-3.**  The environment is re-converged on
    the gauged pair, and a CTM at finite chi truncates in a basis the gauge
    moves, so the two runs are not algebraically the same calculation.  The
    residual is therefore CTM truncation, not the metric, and on an *unsaturated*
    fixture it is genuinely of order 1e-1: measured here 1.00e-01 (energy) and
    6.75e-02 (gap).  A saturated fixture would report much smaller numbers --
    at V=4 the gap moves by 4e-04 -- but only because a saturated observable
    barely moves for anything, including for a state change that is real and
    large.  That is not precision, it is insensitivity, and it is exactly what
    ``midgap_pair`` exists to avoid.

    What makes the bar meaningful is not its size but that it is **bracketed**:
    ``test_a_mispaired_gauge_is_caught`` requires the same constants to fail on
    a transformation that is *not* a gauge, so neither bar can be moved in
    either direction without breaking one of the two tests.
    """
    A, B, H = midgap_pair
    A_g, B_g = _bond_gauge(A, B)

    E, gap = _observables(A, B, H)
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


def test_a_mispaired_gauge_is_caught(midgap_pair):
    """The bars above must fail on something that is *not* a gauge.

    This is the mutation check, kept in the suite rather than run once by hand.
    ``_bond_gauge(mispair=True)`` applies the identical per-leg factors but puts
    ``h_AB``'s inverse on ``B.r`` instead of ``B.l``.  Nothing cancels on that
    bond, so the physical state genuinely moves -- and both witnesses must say
    so, or they are decorations.

    Measured on ``midgap_pair`` at chi=4, 12 sweeps: the energy moves 7.00e-01
    against the correct gauge's 1.00e-01, and the gap 2.88e-01 against 6.75e-02.
    ``BAR_E`` and ``BAR_GAP`` sit between the two, so this test and the one
    above bracket them from opposite sides.  A previous version of the guard
    used ``abs(E - E_g) < 2e-2 * max(abs(E), 1.0)`` on the V=4 fixture, where
    ``E ~ 0`` (#392) collapsed the relative bar to an absolute 2e-2 -- ten times
    the whole magnitude of ``E`` -- and this mutation passed it.
    """
    A, B, H = midgap_pair
    A_m, B_m = _bond_gauge(A, B, mispair=True)

    E, gap = _observables(A, B, H)
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
