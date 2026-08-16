"""#881 review: the sublattice diagnostic, and restarting from the pair.

``fpeps()`` returns two site tensors because the t-V ground state at finite
``V`` is a checkerboard charge-density wave, which is inherently two-site.  The
diagnostic that ships with it has one job: tell a caller whether the returned
pair really *is* a checkerboard, or whether the sweep collapsed to a uniform
state that a single tensor would describe just as well.

The first version could not do that job.  It compared the singular values of
each leg's Gram matrix ``M = T T†``, and those are not gauge invariant: under a
bond gauge ``T -> G T`` the matrix goes to ``G M G†``, whose spectrum moves
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


def _bond_gauge(A, B, g_hAB, g_hBA, g_vAB, g_vBA):
    """Insert ``G G^-1`` on each of the four checkerboard bonds.

    Every bond of the infinite lattice gets a factor and its inverse on the two
    tensors it joins, so the contracted network -- and therefore every physical
    observable -- is unchanged.  The gauges are diagonal in the charge basis, so
    the FermionParity block structure survives, and they are *not* unitary,
    which is the whole point: a unitary gauge would leave even the Gram spectrum
    alone and the test would prove nothing.
    """
    A = scale_bond_axis(A, "r", g_hAB)  # h_AB: A.r <-> B.l
    B = scale_bond_axis(B, "l", 1.0 / g_hAB)
    B = scale_bond_axis(B, "r", g_hBA)  # h_BA: B.r <-> A.l
    A = scale_bond_axis(A, "l", 1.0 / g_hBA)
    A = scale_bond_axis(A, "d", g_vAB)  # v_AB: A.d <-> B.u
    B = scale_bond_axis(B, "u", 1.0 / g_vAB)
    B = scale_bond_axis(B, "d", g_vBA)  # v_BA: B.d <-> A.u
    A = scale_bond_axis(A, "u", 1.0 / g_vBA)
    return A, B


@pytest.fixture(scope="module")
def su_pair():
    """A short D=2 t-V simple-update run, in physical (CTM-contractable) form."""
    cfg = FPEPSConfig(D=2, t=1.0, V=4.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    A, B, lam = _fpeps_simple_update(A0, H, max_D=cfg.D, dt=cfg.dt, steps=8)
    return _to_physical_pair(A, B, lam)


def test_the_gap_is_invariant_under_a_bond_gauge(su_pair):
    """The state does not change, so the diagnostic must not either.

    The gauge below leaves every physical observable alone by construction --
    each bond carries a factor and its inverse.  A diagnostic that moves under
    it is reading the representation, not the state, and cannot be used to
    decide whether the two sublattices differ.

    "By construction" is checked, not assumed: the energy from the same two
    environments must agree across the gauge before any claim is made about the
    diagnostic.  Otherwise a mis-written gauge -- an inverse on the wrong leg --
    would move the state, a *correct* diagnostic would move with it, and this
    test would fail on the fix and pass on the defect.

    The tolerance is 2e-2 rather than machine precision, and the reason is the
    CTM, not the metric: the environment is re-converged on the gauged pair, and
    a CTM at finite chi truncates in a basis the gauge moves, so the two runs
    are not algebraically the same calculation.  Measured on this fixture the
    residual runs 6.4e-05 (chi=8, 40 sweeps), 1.1e-04 (chi=8, 80), 3.8e-04
    (chi=4, 12), 1.1e-03 (chi=4, 8), 2.2e-03 (chi=4, 20), 3.8e-03 (chi=6, 12).
    Note it does **not** fall monotonically with chi -- it tracks how converged
    the two runs are, and sits at or below the CTM's own sweep-to-sweep motion
    at the same settings (1.5e-02 at chi=4 between 10 and 40 sweeps).  The
    metric this replaced moves by **27** on the same pair, three orders of
    magnitude above any of that, so no tolerance in this range confuses them.
    """
    A, B = su_pair
    A_g, B_g = _bond_gauge(
        A,
        B,
        np.array([2.0, 0.5]),
        np.array([1.5, 0.8]),
        np.array([0.7, 1.3]),
        np.array([1.1, 2.2]),
    )

    kw = dict(max_iter=12, conv_tol=1e-10)
    envs, envs_g = (
        ctm_split_tensor_2site(A, B, CHI, **kw),
        ctm_split_tensor_2site(A_g, B_g, CHI, **kw),
    )

    # Prove the premise before using it. `_bond_gauge` is *claimed* inert, and
    # everything below is worthless if it is not -- a gauge with a typo (an
    # inverse on the wrong leg, say) changes the state, and then a diagnostic
    # that moved would be reporting correctly and this test would be pinning a
    # bug as the fix. The energy is a physical observable computed from the same
    # environments, so it is the right witness.
    d = A.indices[A.labels().index("phys")].dim
    H = spinless_fermion_gate(FPEPSConfig(D=2, t=1.0, V=4.0))
    E = float(compute_energy_split_ctm_tensor_2site(A, B, *envs, H, d=d))
    E_g = float(compute_energy_split_ctm_tensor_2site(A_g, B_g, *envs_g, H, d=d))
    assert abs(E - E_g) < 2e-2 * max(abs(E), 1.0), (
        f"the 'gauge' moved the energy from {E:.10f} to {E_g:.10f} -- it is not "
        f"a gauge transformation, so nothing this test asserts below is about "
        f"gauge invariance"
    )

    gap = sublattice_gap(A, B, *envs)
    gap_g = sublattice_gap(A_g, B_g, *envs_g)

    assert gap > 1e-2, (
        f"gap {gap:.3e} on a V=4 t-V pair: the fixture is not exercising a "
        f"checkerboard at all, so the invariance below would be vacuous"
    )
    assert abs(gap - gap_g) < 2e-2, (
        f"sublattice_gap moved from {gap:.10f} to {gap_g:.10f} under a pure "
        f"bond gauge -- it is measuring the gauge, not the state (#881 P2-3)"
    )

    # And the metric this replaced does move, by orders of magnitude more than
    # the tolerance above.  This is the finding, pinned: 0.077 -> 27.0, a factor
    # of 350, on a pair whose physical state did not change at all.
    gram, gram_g = _gram_gap(A, B), _gram_gap(A_g, B_g)
    assert abs(gram - gram_g) > 1.0, (
        f"the Gram metric read {gram:.6f} -> {gram_g:.6f} under the same gauge; "
        f"if it no longer moves, this test has stopped discriminating"
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
