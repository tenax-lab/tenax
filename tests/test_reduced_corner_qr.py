"""Tests for the dense reduced-corner QR projector (issue #570, Phase 1).

The reduced-corner QR-CTMRG projector (Yang/Zhang/Corboz, arXiv:2505.00494)
builds an isometry directly from the enlarged corners with no large (chi D^2)
truncating SVD.  ``_reduced_qr_projector`` productionizes the spike-validated
"Candidate C" construction (concat both corners -> unpivoted QR -> tiny
2*chi x 2*chi Hermitian eigendecomposition -> top-chi directions).

Phase 1 is dense + forward only; these tests exercise the isometry property of
the returned projector on representative dense enlarged corners.
"""

from __future__ import annotations

from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax import CTMConfig, heisenberg_gate, ipeps, iPEPSConfig, optimize_gs_ad
from tenax.algorithms._ctm_projector import (
    _compute_projector_tensor,
    _gauge_fix_qr_dense,
    _reduced_qr_projector,
)
from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor, ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import (
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
)
from tenax.algorithms.ipeps import sublattice_rotate_gate, symmetrize_c4v
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _build_dense_enlarged_corners(chi: int, D: int = 2, seed: int = 0):
    """Build a representative dense enlarged-corner pair ``(C1g, C4g)``.

    Mirrors the spike's view of the 1x1 left move: each enlarged corner is a
    dense ``(fused | cut)`` matrix where ``dim(fused) = chi * D**2`` and the
    cut leg (``t1_r`` / ``t3_l``) is already dimension ``chi`` (the reduced
    corner).  ``C1g`` carries labels ``(fused, t1_r)``; ``C4g`` carries
    ``(fused, t3_l)``.  The two corners are made close-but-not-identical so the
    both-corners (off-C4v) path is genuinely exercised.
    """
    sym = U1Symmetry()
    fused_dim = chi * D * D
    rng = np.random.default_rng(seed)

    fused_idx = TensorIndex.from_charges(
        sym, np.zeros(fused_dim, dtype=np.int32), FlowDirection.IN, label="fused"
    )
    t1_r_idx = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), FlowDirection.OUT, label="t1_r"
    )
    t3_l_idx = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), FlowDirection.OUT, label="t3_l"
    )

    base = rng.standard_normal((fused_dim, chi))
    # C4g close to C1g (small asymmetric perturbation) so both corners matter.
    pert = 0.05 * rng.standard_normal((fused_dim, chi))
    C1g = DenseTensor(jnp.asarray(base), (fused_idx, t1_r_idx))
    C4g = DenseTensor(jnp.asarray(base + pert), (fused_idx, t3_l_idx))
    return C1g, C4g


@pytest.mark.core
@pytest.mark.parametrize("chi", [4, 8])
def test_reduced_qr_projector_is_isometry(chi):
    """``P`` has labels ``(fused, chi_new)`` and ``P^dagger P = I_chi``."""
    C1g, C4g = _build_dense_enlarged_corners(chi)

    P = _reduced_qr_projector(C1g, C4g, chi)

    assert P.labels() == ("fused", "chi_new")
    assert P._data.shape == (chi * 2 * 2, chi)

    # Isometry: contracting P.bar() with P over the fused leg yields I_chi.
    gram = P._data.conj().T @ P._data
    np.testing.assert_allclose(np.asarray(gram), np.eye(chi), atol=1e-9)


@pytest.mark.core
def test_compute_projector_tensor_qr_uses_reduced_qr_dense():
    """Dense forward ``projector_method='qr'`` dispatch routes through the
    canonical ``_reduced_qr_projector`` (consolidation, issue #570 Phase 1)."""
    chi = 6
    C1g, C4g = _build_dense_enlarged_corners(chi)

    P1, P2, eps = _compute_projector_tensor(C1g, C4g, chi, "qr", None, "auto")

    # Single isometry (P_1 = P_2), and equals the canonical reference output.
    P_ref = _reduced_qr_projector(C1g, C4g, chi)
    np.testing.assert_allclose(P1.todense(), P_ref.todense(), atol=1e-10)
    np.testing.assert_allclose(P1.todense(), P2.todense(), atol=1e-12)
    assert float(eps) == 0.0

    # Isometry holds through the dispatch:
    g = P1._data.conj().T @ P1._data
    np.testing.assert_allclose(np.asarray(g), np.eye(g.shape[0]), atol=1e-9)


@pytest.mark.core
def test_gauge_fix_qr_dense_makes_diag_R_nonneg_and_preserves_QR():
    key = jax.random.PRNGKey(3)
    M = jax.random.normal(key, (12, 6))
    Q, R = jnp.linalg.qr(M)
    Qf, Rf = _gauge_fix_qr_dense(Q, R)
    # Orthonormal columns preserved:
    np.testing.assert_allclose(Qf.conj().T @ Qf, np.eye(Qf.shape[1]), atol=1e-10)
    # Reconstruction preserved: Qf @ Rf == Q @ R == M
    np.testing.assert_allclose(Qf @ Rf, M, atol=1e-10)
    # diag(Rf) real, non-negative:
    d = jnp.diag(Rf)
    assert jnp.all(jnp.real(d) >= -1e-12)
    assert jnp.allclose(jnp.imag(d), 0.0, atol=1e-10)


@pytest.mark.core
def test_gauge_fix_qr_dense_is_smooth_under_perturbation():
    key = jax.random.PRNGKey(5)
    M = jax.random.normal(key, (12, 6))
    dM = 1e-7 * jax.random.normal(jax.random.PRNGKey(6), (12, 6))

    def q_of(scale):
        Q, R = jnp.linalg.qr(M + scale * dM)
        Qf, _ = _gauge_fix_qr_dense(Q, R)
        return Qf

    assert (
        jnp.max(jnp.abs(q_of(1.0) - q_of(0.0))) < 1e-3
    )  # no O(1) sign flip for O(eps) perturbation


@pytest.mark.core
def test_qr_projector_gradient_finite_dense():
    """jax.grad through the dense 'qr' projector dispatch (AD-tracer path) is
    finite, including a rank-deficient corner (regularized_qr keeps it finite).

    Finiteness guard: at this size/conditioning the gauge-fix + top-k SVD
    truncation in the dispatch happens to discard the ill-conditioned QR
    directions, so even raw ``jnp.linalg.qr`` yields a finite gradient here.
    The guard pins the contract (finite backward through the AD-tracer 'qr'
    path); ``regularized_qr`` is what keeps it finite at genuine
    rank-deficiency, exercised directly in ``tests/test_regularized_qr.py``
    (``test_regularized_qr_backward_finite_at_rank_deficiency``).
    """
    chi = 6
    C1g, C4g = _build_dense_enlarged_corners(chi)

    def loss(scale):
        c1 = type(C1g)(C1g._data * scale, C1g.indices)
        P1, _P2, _eps = _compute_projector_tensor(c1, C4g, chi, "qr", None, "auto")
        return jnp.real(jnp.sum(jnp.abs(P1._data) ** 2))

    g = jax.grad(loss)(1.3)
    assert jnp.isfinite(g)

    # Rank-deficient corner: zero out some fused-direction content so QR sees
    # a near-singular M; gradient must still be finite via regularized_qr.
    base = np.array(C1g._data)  # writable copy
    base[chi:, :] = 0.0  # kill part of the fused space -> rank-deficient stack
    C1g_rd = type(C1g)(jnp.asarray(base), C1g.indices)

    def loss_rd(scale):
        c1 = type(C1g_rd)(C1g_rd._data * scale, C1g_rd.indices)
        P1, _P2, _eps = _compute_projector_tensor(c1, C4g, chi, "qr", None, "auto")
        return jnp.real(jnp.sum(jnp.abs(P1._data) ** 2))

    g_rd = jax.grad(loss_rd)(1.1)
    assert jnp.isfinite(g_rd)


# --------------------------------------------------------------------------- #
# Energy-agreement harness (reduced-corner QR vs eigh), reused from the spike  #
# examples/probe_reduced_corner_qr_reconstruction_570.py.                      #
#                                                                              #
# The spike validated Candidate C (the production "qr" path now routed through #
# _reduced_qr_projector) on the spin-1/2 2D Heisenberg model at D=2 in the     #
# single-site (1x1) dense CTM, measuring |E_qr - E_eigh| ~ 1e-13.  These tests #
# turn that validation into a regression.                                      #
# --------------------------------------------------------------------------- #

# Building the physical state (simple update) is expensive; cache it once.
_PHYS_STATE = None


def _build_physical_state_heisenberg_D2():
    """Build a C4v-symmetrized D=2 single-site Heisenberg ``A`` + rotated gate.

    Copied/adapted from the spike harness ``build_physical_state``: sublattice
    rotation makes the Neel AFM ground state a *uniform* single-site iPEPS;
    simple update converges a physical ``A``; then ``A`` is **C4v-symmetrized**
    and renormalized.

    **Why C4v symmetrization is load-bearing — corrected (#911).**  This used
    to say it was load-bearing because *"otherwise the four directional 1x1
    moves are inequivalent and the single-site eigh sweep limit-cycles at the
    #425/#426 plateau, so the eigh oracle is untrustworthy"*.  #911 measured
    that claim and it is false in the direction that matters: the
    C4v-symmetrized state limit-cycles **as hard or harder** than the raw one
    — energy range over the last 40 of 240 sweeps 4.47e-3 (C4v) against
    4.86e-3 (raw), and C4v's terminal ``sv_diff`` is 4.41e-4 against raw's
    1.08e-4, the *worse* of the two.  Neither state converges on the generic
    1x1 sweep at any projector method.  Symmetrizing does not buy an oracle.

    What it actually buys is **admission to a different engine**.
    ``ctm_tensor_c4v`` requires a C4v-symmetric state, and it is the only
    entry point that runs ``eigh``/``qr`` at full rank to a genuine fixed
    point — #911's control has all three projector methods agreeing there to
    1e-12, against ``recipe="1x1"`` where ``svd`` collapses to rank 1 and
    ``eigh``/``qr`` limit-cycle.  So the symmetrization is what lets the
    tests below compare projector methods on an engine that honours the
    parameter at all.  See ``_heisenberg_D2_ctm_energy_c4v``.
    """
    global _PHYS_STATE
    if _PHYS_STATE is not None:
        return _PHYS_STATE

    gate = heisenberg_gate()
    gate_rot = sublattice_rotate_gate(gate)
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=400,
        dt=0.05,
        ctm=CTMConfig(chi=16, max_iter=80, projector_method="eigh"),
    )
    _E_su, (A, _B), _envs = ipeps(gate_rot, initial_peps=None, config=config)
    A = DenseTensor(symmetrize_c4v(A._data), A.indices)
    A = A * (1.0 / float(A.norm()))
    _PHYS_STATE = (A, gate_rot)
    return _PHYS_STATE


def _heisenberg_D2_ctm_energy_c4v(chi, projector_method, max_iter=200):
    """Converged dense CTM energy for the given projector, via ``ctm_tensor_c4v``.

    **This used to call ``ctm_tensor`` and test nothing.**  It was written
    when ``ctm_tensor``'s ``recipe`` defaulted to ``"1x1"``, and it never
    passed ``recipe=`` explicitly.  ``988c2a8`` (#765, 2026-08-03) flipped
    that default to ``"2x2"`` — which hardcodes Fishman SVD and *ignores*
    ``projector_method`` entirely — so from that commit every caller below
    silently compared ``2x2`` against itself.  Measured on the state this
    module builds, ``eigh``, ``qr`` and ``svd`` returned **bit-identical**
    energies:

        chi=8   all three  -0.659430578410895
        chi=16  all three  -0.659430578425110

    so ``test_reduced_qr_energy_matches_eigh_heisenberg_D2`` was asserting
    ``abs(0.0) < 1e-3`` and would have passed with the reduced-corner QR
    projector deleted from the codebase.  That is the whole point of this
    module gone quiet for a month.

    The fix is not to pass ``recipe="1x1"``: #911 established that recipe
    reaches no fixed point at any projector method (``svd`` collapses to
    rank 1, ``eigh``/``qr`` limit-cycle), so pinning it would pin an orbit
    sample.  ``ctm_tensor_c4v`` is the engine that genuinely honours
    ``projector_method`` — it is #911's own control, where all three methods
    hold full rank and agree with ``2x2`` to 1e-12 — and the module's state
    is already C4v-symmetrized, which is what admits it.  It is also exactly
    the migration the ``recipe="1x1"`` deprecation message points callers to.

    ``max_iter`` is exposed so a fixed-point-stability check can re-converge
    with a larger sweep budget (the CTM still stops early at ``conv_tol``).
    ``test_the_projector_method_actually_reaches_the_projector`` pins that
    this helper is on a path that consults the parameter, so the comparison
    below cannot quietly go vacuous again.
    """
    A, gate_rot = _build_physical_state_heisenberg_D2()
    env = ctm_tensor_c4v(
        A,
        chi=chi,
        max_iter=max_iter,
        conv_tol=1e-10,
        projector_method=projector_method,
    )
    return float(compute_energy_ctm_tensor(A, env, gate_rot))


def _c4v_probe_state(D=2, d=2, chi_seed=0):
    """A tiny C4v-symmetric state, for liveness probes that need no physics."""
    rng = np.random.default_rng(chi_seed)
    sym = U1Symmetry()
    idx = tuple(
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), flow, label=lbl)
        for lbl, flow in [
            ("u", FlowDirection.OUT),
            ("d", FlowDirection.IN),
            ("l", FlowDirection.OUT),
            ("r", FlowDirection.IN),
        ]
    ) + (
        TensorIndex.from_charges(
            sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )
    A = DenseTensor(jnp.asarray(rng.standard_normal((D, D, D, D, d))), idx)
    A = DenseTensor(symmetrize_c4v(A._data), A.indices)
    return A * (1.0 / float(A.norm()))


@pytest.mark.core
def test_the_projector_method_actually_reaches_the_projector(monkeypatch):
    """``projector_method`` must select a *different code path*, not a label.

    This is the guard the module lacked.  For a month
    ``_heisenberg_D2_ctm_energy_*`` ran ``ctm_tensor`` on the default
    ``recipe="2x2"``, which hardcodes Fishman SVD and ignores
    ``projector_method``, so every ``qr``-vs-``eigh`` comparison below was a
    value against itself and would have passed with ``_reduced_qr_projector``
    deleted.  An energy assertion cannot detect that — the two numbers agree
    whether or not the parameter did anything.  Reachability can.

    Sabotage the QR projector and require the failure to propagate on
    ``"qr"`` and *not* on ``"eigh"``.  Both halves matter: the first says QR
    is on the path, the second says the parameter chooses.
    """
    A = _c4v_probe_state()
    sentinel = RuntimeError("reduced-corner QR projector reached")

    def _boom(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr("tenax.algorithms._ctm_projector._reduced_qr_projector", _boom)

    with pytest.raises(RuntimeError, match="reduced-corner QR projector reached"):
        ctm_tensor_c4v(A, chi=4, max_iter=2, projector_method="qr")

    # ...and the selector genuinely selects: eigh must not touch it.
    ctm_tensor_c4v(A, chi=4, max_iter=2, projector_method="eigh")


@pytest.mark.core
def test_the_energy_helper_is_wired_to_the_engine_that_honours_the_selector(
    monkeypatch,
):
    """Pin the *wiring*, not just the engine.

    ``test_the_projector_method_actually_reaches_the_projector`` proves
    ``ctm_tensor_c4v`` consults ``projector_method``.  It does not prove the
    energy helper below calls ``ctm_tensor_c4v`` — so pointing the helper back
    at ``ctm_tensor`` would make every energy comparison vacuous again while
    that test carried on passing.  That is the exact shape of the original
    defect, so it gets its own guard.

    Stubbed rather than converged: this asserts a call, and the physics is
    covered by the ``algorithm``-marked tests below.
    """
    recorded = {}

    def _fake_c4v(A, **kwargs):
        recorded.update(kwargs)
        return "sentinel-env"

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "the energy helper called ctm_tensor, whose default recipe='2x2' "
            "ignores projector_method — the comparison is vacuous again"
        )

    # The module caches the SU state in a global; seed it so this stays cheap.
    import sys

    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "_PHYS_STATE", (_c4v_probe_state(), None))
    monkeypatch.setattr(mod, "ctm_tensor_c4v", _fake_c4v)
    monkeypatch.setattr(mod, "ctm_tensor", _forbidden)
    monkeypatch.setattr(mod, "compute_energy_ctm_tensor", lambda *a, **k: -1.0)

    e = _heisenberg_D2_ctm_energy_c4v(chi=4, projector_method="qr")

    assert e == -1.0
    assert recorded.get("projector_method") == "qr", (
        f"the selector did not reach the engine: {recorded}"
    )
    assert recorded.get("chi") == 4


@pytest.mark.core
def test_recipe_2x2_ignores_projector_method(monkeypatch):
    """Pin the trap that made this module vacuous (#911, #765).

    ``recipe="2x2"`` -- ``ctm_tensor``'s default since ``988c2a8`` --
    hardcodes Fishman SVD.  Passing ``projector_method="qr"`` there is
    accepted and silently ignored, which is why the energy tests above had to
    move to ``ctm_tensor_c4v``.  If this ever starts raising, ``2x2`` grew a
    QR path and the helper above can be reconsidered.
    """
    A = _c4v_probe_state()

    def _boom(*args, **kwargs):
        raise RuntimeError("should not be reached on the 2x2 recipe")

    monkeypatch.setattr("tenax.algorithms._ctm_projector._reduced_qr_projector", _boom)

    # No raise: the 2x2 recipe never consults projector_method.
    ctm_tensor(A, chi=4, max_iter=2, projector_method="qr", recipe="2x2")


@pytest.mark.algorithm
@pytest.mark.parametrize("chi", [8, 16])
def test_reduced_qr_energy_matches_eigh_heisenberg_D2(chi):
    e_eigh = _heisenberg_D2_ctm_energy_c4v(chi=chi, projector_method="eigh")
    e_qr = _heisenberg_D2_ctm_energy_c4v(chi=chi, projector_method="qr")
    assert abs(e_qr - e_eigh) < 1e-3  # loosened vs eps; different scheme, same physics


@pytest.mark.algorithm
def test_reduced_qr_energy_gap_shrinks_with_chi():
    g8 = abs(
        _heisenberg_D2_ctm_energy_c4v(8, "qr")
        - _heisenberg_D2_ctm_energy_c4v(8, "eigh")
    )
    g16 = abs(
        _heisenberg_D2_ctm_energy_c4v(16, "qr")
        - _heisenberg_D2_ctm_energy_c4v(16, "eigh")
    )
    assert g16 <= g8 + 1e-12  # gap does not grow as chi increases


@pytest.mark.algorithm
def test_reduced_qr_ctm_converges_with_warmup():
    """The ``qr`` 1x1 CTM (eigh warm-up active, ``qr_warmup_steps=6 > 0``)
    converges to a clean, stable fixed point.

    Two non-flaky assertions:

    * **Finite, real energy** — no NaN/Inf escapes the reduced-corner QR
      projector or its eigh warm-up.
    * **Fixed-point stability** — re-converging with a doubled sweep budget
      does not move the energy.  Since ``ctm_tensor`` stops early at
      ``conv_tol`` (the helper's ``conv_tol=1e-10``), both runs reach the
      *same* fixed point and their energies must agree to far tighter than the
      assertion threshold; any residual drift (a limit cycle, or QR failing to
      hold the eigh fixed point) would surface here.
    """
    # Finite, real energy (no NaN/Inf) from the qr + warm-up CTM:
    e = _heisenberg_D2_ctm_energy_c4v(chi=8, projector_method="qr")
    assert np.isfinite(e)

    # Converged fixed point: doubling max_iter does not move the energy.
    e_n = _heisenberg_D2_ctm_energy_c4v(chi=8, projector_method="qr", max_iter=100)
    e_2n = _heisenberg_D2_ctm_energy_c4v(chi=8, projector_method="qr", max_iter=200)
    assert np.isfinite(e_n) and np.isfinite(e_2n)
    assert abs(e_n - e_2n) < 1e-8


# --------------------------------------------------------------------------- #
# 2-site (A != B) energy-agreement harness — Phase 2, Task 4.                  #
#                                                                              #
# Phase 1 validated reduced-corner QR on a single-site (1x1) dense CTM, where  #
# the four directional cuts are equivalent after C4v symmetrization.  The open #
# question deferred from Phase 1 was whether the *single*-isometry QR          #
# projector (P_1 == P_2) is correct for genuinely *asymmetric* multisite cuts  #
# — i.e. a bipartite 2-site cell with A != B.  ``recipe="1x1"`` already        #
# iterates the QR projector per site / per bond, so no new projector code is   #
# needed; this is a forward-energy validation that QR reproduces the eigh      #
# subspace on the asymmetric cell.                                             #
#                                                                              #
# Construction mirrors ``tests/test_ctm_tensor.py`` (the ``test_2site_*``      #
# tests): a physical bipartite (A, B) is produced by the public ``ipeps``      #
# 2-site simple update on the *un-rotated* Heisenberg gate, so the Neel        #
# ground state leaves A and B genuinely distinct (A != B verified below).      #
# The converged 2-site dense CTM energy is then read out per projector method  #
# via ``ctm_tensor_2site(..., recipe="1x1")`` + ``compute_energy_ctm_tensor_   #
# 2site``.                                                                     #
# --------------------------------------------------------------------------- #

# Building the 2-site physical state (simple update) is expensive; cache once.
_PHYS_STATE_2SITE = None


def _build_physical_state_heisenberg_D2_2site():
    """Build a genuine bipartite (A != B) D=2 Heisenberg state + dense gate.

    Uses the public ``ipeps`` 2-site simple-update path (same construction the
    ``test_2site_*`` tests in ``tests/test_ctm_tensor.py`` exercise) on the
    *un-rotated* Heisenberg gate: the antiferromagnetic Neel order makes the two
    sublattices distinct, so ``A != B`` and the asymmetric multisite cut is
    genuinely exercised (asserted at call sites).  Returns ``(A, B, gate_dense)``
    with both site tensors renormalized.
    """
    global _PHYS_STATE_2SITE
    if _PHYS_STATE_2SITE is not None:
        return _PHYS_STATE_2SITE

    gate = heisenberg_gate()
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=400,
        dt=0.05,
        ctm=CTMConfig(chi=16, max_iter=80, projector_method="eigh"),
    )
    _E_su, (A, B), _envs = ipeps(gate, initial_peps=None, config=config)
    A = A * (1.0 / float(A.norm()))
    B = B * (1.0 / float(B.norm()))
    gate_dense = jnp.asarray(gate.todense() if hasattr(gate, "todense") else gate)
    _PHYS_STATE_2SITE = (A, B, gate_dense)
    return _PHYS_STATE_2SITE


def _heisenberg_D2_2site_energy(chi, projector_method, max_iter=200):
    """Converged 2-site (A != B) dense CTM energy for the given projector.

    Drives the per-site ``recipe="1x1"`` 2-site CTM (``ctm_tensor_2site``),
    which calls the QR projector per bond, and reads the bipartite per-site
    energy via ``compute_energy_ctm_tensor_2site``.  ``"qr"`` runs the
    ``qr_warmup_steps=6`` eigh warm-up before switching to reduced-corner QR,
    matching the single-site harness above.
    """
    A, B, gate_dense = _build_physical_state_heisenberg_D2_2site()
    # Sanity: the cell is genuinely asymmetric (A != B), so QR is being
    # validated on an asymmetric multisite cut, not a disguised C4v case.
    assert not bool(jnp.allclose(A.todense(), B.todense()))
    env_A, env_B = ctm_tensor_2site(
        A,
        B,
        chi=chi,
        max_iter=max_iter,
        conv_tol=1e-10,
        projector_method=projector_method,
        qr_warmup_steps=6,
        recipe="1x1",
    )
    return float(compute_energy_ctm_tensor_2site(A, B, env_A, env_B, gate_dense, d=2))


#: The converged energy of ``_build_physical_state_heisenberg_D2_2site``'s cell,
#: from ``recipe="2x2"``, which reaches an actual fixed point on it: identical to
#: 1e-13 across chi=6/8/10/16/24 and across max_iter=100/200/400.  This is the
#: number the 1x1 limit cycle orbits *around* -- it is not reachable by 1x1 at
#: any budget, which is the whole of #901.
_E_2SITE_TRUE = -0.6590003658935

#: How far the ``recipe="1x1"`` limit cycle wanders from ``_E_2SITE_TRUE``.
#: Measured 4e-3 to 1.1e-2 over chi=4..24 for BOTH projectors (#901).  The guard
#: below uses 2e-2 -- above the cycle, far below a real QR failure (O(1e-1)).
_1X1_CYCLE_BAND = 2e-2


@pytest.mark.algorithm
@pytest.mark.parametrize("chi", [6, 10])
def test_2site_1x1_still_limit_cycles_so_qr_vs_eigh_stays_unanswerable(chi):
    """#901: the QR-vs-eigh comparison this file wants cannot be made here yet.

    The original assertion was ``|E_qr - E_eigh| < 1e-3``.  It compared two
    numbers that are **both** points on a limit cycle: ``recipe="1x1"`` on a
    non-C4v 2-site cell has inequivalent directional moves and never reaches a
    fixed point, for *either* projector.  The disagreement it measured was the
    cycle amplitude (~6e-3, no chi trend), not a difference between the schemes.
    The 1e-3 threshold was inherited from the single-site test, where the
    measured agreement is ~1e-13; it was never re-measured here.

    It cannot be repaired by switching to ``recipe="2x2"`` -- the obvious move,
    and the one this issue originally recommended.  ``projector_method`` is
    *not consulted* on the 2x2 path (``_ctm_tensor_convergence.py:298``; the 2x2
    path always uses Fishman SVD), so all three methods return bit-identical
    energies there.  That would make the test green and permanently vacuous,
    which is the defect it exists to prevent.

    So this asserts the **regime** instead, which is a real property and the
    thing that has to change before the comparison becomes possible:

    * the 1x1 path does not converge on this cell -- and now says so;
    * both projectors nonetheless stay in the cycle band around the true
      energy, which is what a gross QR regression would break.

    **When this test fails, that is the good news**: it means ``recipe="1x1"``
    has started converging on an asymmetric cell (#425/#426), and the real
    comparison -- ``|E_qr - E_eigh|``, with a tolerance measured at that point,
    not inherited -- should be restored in its place.
    """
    for method in ("eigh", "qr"):
        with pytest.warns(UserWarning, match="did not converge"):
            e = _heisenberg_D2_2site_energy(chi=chi, projector_method=method)
        # Not a fixed point, but not diverging either: it orbits the true value.
        assert abs(e - _E_2SITE_TRUE) < _1X1_CYCLE_BAND, (
            f"{method} at chi={chi}: E={e} is {abs(e - _E_2SITE_TRUE):.3e} from "
            f"the 2x2 fixed point {_E_2SITE_TRUE}, outside the known 1x1 cycle "
            f"band {_1X1_CYCLE_BAND}. Either the cycle got worse or the "
            f"projector regressed -- both are real and neither is #901."
        )


@pytest.mark.algorithm
def test_nonconvergence_warning_quotes_the_budget_the_caller_actually_passed():
    """The #901 warning must report the caller's ``max_iter``, not the remainder.

    ``_ctm_tensor_multisite`` shortens ``max_iter`` by ``qr_warmup_steps``
    *before* the convergence loop::

        max_iter = max_iter - warmup      # QR warm-up
        ...
        budget = max_iter                 # <- pre-fix: the remainder

    so ``budget`` named a number the caller never passed.  That matters because
    the message's own advice is "Raise max_iter", which is unactionable against
    a value that is not the parameter: with ``qr_warmup_steps=6`` (what
    ``_heisenberg_D2_2site_energy`` uses) a caller passing ``max_iter=40`` was
    told it "ran the full max_iter=34 sweeps".

    The warm-up sweeps do run, so the caller's total is also the honest count
    of sweeps performed -- there is no reading on which 34 was right.

    ``projector_method="qr"`` is load-bearing here: the warm-up is the only
    thing that rewrites ``max_iter``, so ``eigh``/``svd`` cannot see this bug.
    """
    max_iter, warmup = 40, 6
    with pytest.warns(UserWarning, match="did not converge") as caught:
        _heisenberg_D2_2site_energy(chi=6, projector_method="qr", max_iter=max_iter)

    msgs = [str(w.message) for w in caught]
    assert any(f"max_iter={max_iter}" in m for m in msgs), (
        f"the warning does not quote the caller's budget max_iter={max_iter}: {msgs}"
    )
    assert not any(f"max_iter={max_iter - warmup}" in m for m in msgs), (
        f"the warning quotes the post-warm-up remainder "
        f"max_iter={max_iter - warmup} instead of the caller's "
        f"max_iter={max_iter}, so 'Raise max_iter' points at the wrong "
        f"number: {msgs}"
    )


@pytest.mark.algorithm
@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({"max_iter": 1, "projector_method": "svd"}, "one sweep, no warm-up"),
        (
            {"max_iter": 7, "projector_method": "qr", "qr_warmup_steps": 6},
            "warm-up leaves exactly one measured sweep",
        ),
        (
            {"max_iter": 6, "projector_method": "qr", "qr_warmup_steps": 6},
            "warm-up consumes the whole budget",
        ),
    ],
)
def test_warning_never_reports_a_criterion_it_did_not_measure(kwargs, label):
    """A criterion needs two spectra; with fewer, say so instead of printing 0.

    ``sweep_diff`` starts at ``0.0`` and is only raised by an actual comparison,
    so when every ``prev`` is ``None`` nothing is computed and it stays ``0.0``.
    Assigning that to ``final_diff`` anyway made the warning read::

        without reaching conv_tol=1e-10 (final criterion 0)

    which claims a perfectly converged criterion in the same sentence as
    reporting non-convergence -- and ``0`` is the *most* reassuring value it
    could have printed, so the failure is in the direction that misleads.

    The third case is the control: with zero measured sweeps the loop body never
    runs, ``final_diff`` keeps its pre-loop ``inf``, and this was already
    reported correctly.  The fix makes the one-measured-sweep cases agree with
    it rather than making the zero case agree with them.
    """
    A, B, _gate = _build_physical_state_heisenberg_D2_2site()
    with pytest.warns(UserWarning, match="did not converge") as caught:
        ctm_tensor_2site(A, B, chi=6, conv_tol=1e-10, recipe="2x2", **kwargs)

    msgs = [str(w.message) for w in caught if "ctm_tensor_multisite" in str(w.message)]
    assert msgs, f"{label}: no ctm_tensor_multisite warning was emitted"
    for m in msgs:
        assert "criterion 0)" not in m, (
            f"{label}: the warning reports 'criterion 0' -- a converged value -- "
            f"while saying conv_tol was not reached: {m}"
        )
        assert "never evaluated" in m, (
            f"{label}: fewer than two spectra were compared, so the warning must "
            f"say the criterion was never evaluated rather than print one: {m}"
        )


@pytest.mark.algorithm
def test_2x2_ignores_projector_method_so_it_cannot_host_this_comparison():
    """Pins the constraint that makes the test above the only option (#901).

    If ``projector_method`` ever *is* honoured on the 2x2 path, this fails --
    and at that point the QR-vs-eigh comparison should move there, where the
    energy is a fixed point to 1e-13 and a tight measured tolerance means
    something.  Without this, the next person to look at #901 re-derives the
    withdrawn recommendation from scratch, as I did.
    """
    A, B, gate_dense = _build_physical_state_heisenberg_D2_2site()
    energies = {}
    for method in ("svd", "eigh", "qr"):
        env_A, env_B = ctm_tensor_2site(
            A,
            B,
            chi=6,
            max_iter=100,
            conv_tol=1e-10,
            projector_method=method,
            qr_warmup_steps=6,
            recipe="2x2",
        )
        energies[method] = float(
            compute_energy_ctm_tensor_2site(A, B, env_A, env_B, gate_dense, d=2)
        )
    assert energies["svd"] == energies["eigh"] == energies["qr"], (
        "2x2 now distinguishes projector_method: "
        f"{energies}. Move the QR-vs-eigh comparison here (see #901)."
    )


def _build_single_site_dense(D: int = 2, d: int = 2, seed: int = 0) -> DenseTensor:
    """Minimal trivial-U(1) (d, D, D, D, D) single-site tensor for wiring probes."""
    rng = np.random.default_rng(seed)
    sym = U1Symmetry()
    bond = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.OUT, label="l"),
    )
    data = jnp.asarray(
        rng.standard_normal((d, D, D, D, D)).astype(np.float64), dtype=jnp.float64
    )
    return DenseTensor(data, indices)


@pytest.mark.core
def test_implicit_ad_recipe_threads_to_sweep(monkeypatch):
    """ctm_energy_implicit(recipe='1x1') routes the CTM sweep through recipe='1x1'.

    Without the recipe knob the implicit-AD forward/backward sweeps hardcode
    ``recipe='2x2'`` (Fishman plaquette), so ``projector_method='qr'`` is a
    silent no-op.  This proves the knob reaches the actual sweep call.
    """
    import tenax.algorithms._ctm_python_loop as loop
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS

    recipes_seen = []
    orig = loop._ctm_tensor_sweep_multisite

    def spy(*a, **k):
        recipes_seen.append(k.get("recipe"))
        return orig(*a, **k)

    # Patch where _make_jit_ctm_step looks it up (module-level reference in
    # _ctm_python_loop), so the jitted step closure calls the spy.
    monkeypatch.setattr(loop, "_ctm_tensor_sweep_multisite", spy)

    site_tensors = {(0, 0): _build_single_site_dense()}
    gate = heisenberg_gate()

    # The spy records ``recipe`` at sweep entry.  The 1x1 left-move currently
    # does a host-side ``float(eps_t)`` that is incompatible with the jitted
    # CTM step (a pre-existing 1x1-under-jit limitation, orthogonal to this
    # wiring task), so the call may raise *after* the recipe has been
    # recorded.  We only assert on what reached the sweep — the threading.
    try:
        ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=2,
            min_iter=1,
            recipe="1x1",
            projector_method="qr",
            qr_warmup_steps=0,
        )
    except Exception:  # noqa: BLE001 — threading is proven by recipes_seen
        pass

    assert recipes_seen, "the CTM sweep was never reached"
    assert "1x1" in recipes_seen
    assert "2x2" not in recipes_seen  # not falling back to the hardcoded default


# --------------------------------------------------------------------------- #
# Phase 2, Task 5b — recipe='1x1' + qr RUNS end-to-end under implicit-diff AD   #
#                                                                              #
# The wiring test above proves the recipe reaches the sweep, but the jitted    #
# CTM step previously raised ConcretizationTypeError because the 1x1 moves did  #
# ``float(eps_t)`` on a tracer.  This test proves the path now RUNS (finite     #
# energy) AND DIFFERENTIATES (finite gradient w.r.t. the site tensor).          #
# --------------------------------------------------------------------------- #


def _wrap_single_site_dense(arr: jax.Array) -> DenseTensor:
    """Wrap a raw (d, D, D, D, D) array as the trivial-U(1) DenseTensor used by
    :func:`_build_single_site_dense` (same index layout)."""
    d, D = arr.shape[0], arr.shape[1]
    sym = U1Symmetry()
    bond = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, bond.copy(), FlowDirection.OUT, label="l"),
    )
    return DenseTensor(jnp.asarray(arr), indices)


def _implicit_energy_of_A(A_arr):
    """Raw-array → implicit-AD energy via recipe='1x1' + projector_method='qr'.

    Tiny system (D=2, chi=4) and few iterations so it stays fast; the point is
    that it RUNS under jit and differentiates, not the physical accuracy.
    """
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS

    A = _wrap_single_site_dense(A_arr)
    gate = heisenberg_gate()
    return ctm_energy_implicit(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        gate,
        chi=4,
        max_iter=4,
        min_iter=1,
        recipe="1x1",
        projector_method="qr",
        qr_warmup_steps=2,
    )


def _implicit_energy(recipe="1x1", projector_method="qr"):
    """Finite-energy probe for the recipe='1x1' + qr implicit-AD forward."""
    A_arr = _build_single_site_dense().todense()
    return _implicit_energy_of_A(A_arr)


@pytest.mark.algorithm
def test_implicit_ad_qr_1x1_runs_and_is_differentiable():
    """ctm_energy_implicit(recipe='1x1', projector_method='qr') runs end-to-end
    under jit and yields a finite energy AND a finite gradient.

    Pre-fix this raised ConcretizationTypeError (the 1x1 moves did
    ``float(eps_t)`` on a tracer inside the jitted CTM step).
    """
    e = _implicit_energy(recipe="1x1", projector_method="qr")
    assert jnp.isfinite(e)

    A_arr = _build_single_site_dense().todense()
    g = jax.grad(lambda x: jnp.real(_implicit_energy_of_A(x)))(A_arr)
    assert jnp.all(jnp.isfinite(g))


# --------------------------------------------------------------------------- #
# Phase 2, Task 6 — implicit-AD QR gradient *correctness* (parity tests).       #
#                                                                              #
# Task 5b proved recipe='1x1' + qr RUNS and yields a *finite* gradient.  This   #
# validates the gradient is *correct* by two independent checks:                #
#   (a) finite-difference parity — the implicit-AD gradient matches a central   #
#       finite difference of the same energy (key correctness gate), and        #
#   (b) eigh parity — the QR-AD gradient agrees with the eigh-AD gradient on     #
#       the same state (same physics, a different isometric projector scheme),   #
#       converging together as chi grows.                                        #
#                                                                              #
# IMPORTANT — what makes a *genuine* FD check at this size (D=2), and why the    #
# spec's exact tolerances are not reachable here (a real finding, documented):   #
#                                                                              #
#  * Use a PHYSICAL, converged state.  The random tensor from                   #
#    ``_build_single_site_dense`` is not a near-fixed-point iPEPS; its CTM       #
#    fixed point is ill-conditioned and the implicit adjoint (the GMRES solve    #
#    of ``(I - J)``) is near-singular, giving O(1e2) gradients that no FD        #
#    tracks.  The C4v-symmetrized simple-update ``A`` from                       #
#    ``_build_physical_state_heisenberg_D2`` (E0 ~ -0.48, near the 2D Heisenberg #
#    ground state) gives sane O(1) gradients.                                   #
#                                                                              #
#  * The spec's first suggestion — a GLOBAL-SCALE parameter ``A -> A * theta``   #
#    — is UNUSABLE: the CTM energy is a *normalized* expectation value           #
#    ``<H>/<1>``, so the scale direction is NEARLY orthogonal to the gradient    #
#    (measured ``<grad, A0>`` ~= -4e-3, only ~0.6% of ||grad||).  It is not an    #
#    exact algebraic degeneracy — E does vary ~1.6e-3 across theta — but that     #
#    tiny scale-FD signal (~1e-3) is swamped by the kink noise (O(1e2)) from      #
#    projector reselection, so the scale FD cannot distinguish a correct          #
#    gradient from a wrong one.  This probe is therefore NOT usable as the FD     #
#    gate.                                                                        #
#                                                                              #
#  * A non-degenerate FD (perturbing along a real, energy-changing direction)    #
#    at D=2/chi<=16 is dominated by piecewise-flat-with-kinks structure: the     #
#    projector subspace is discretely reselected, so small-eps FD swings O(1e3)  #
#    and small-eps single-entry FD is meaningless.  A LARGE step (eps~1e-2)       #
#    along the AD gradient direction averages over the local flat regions and    #
#    recovers the slope's SIGN and ORDER OF MAGNITUDE (good to ~tens of percent, #
#    reproducibly) — the cleanest genuine FD available at this tiny size, used   #
#    as the ``test_implicit_qr_gradient_matches_fd`` correctness gate.           #
#                                                                              #
#  * qr-AD and eigh-AD gradients are NOT equal at finite chi (they amplify the   #
#    ~1e-3 forward energy gap into a ~10-40% gradient gap); they agree in sign/   #
#    structure (cosine ~0.8-0.9) and CONVERGE TOGETHER as chi grows, mirroring   #
#    the forward ``test_reduced_qr_energy_gap_shrinks_with_chi``.  So the eigh    #
#    parity is a directional + converge-with-chi statement, not an exact match.  #
# --------------------------------------------------------------------------- #


def _implicit_energy_of_A_phys(A_arr, *, recipe, projector_method, chi, **kw):
    """Implicit-AD energy of a raw ``(d,D,D,D,D)`` array on the *physical*
    (C4v-symmetrized simple-update) D=2 Heisenberg state's gate."""
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS

    _A, gate_rot = _build_physical_state_heisenberg_D2()
    A = _wrap_single_site_dense(A_arr)
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            gate_rot,
            recipe=recipe,
            projector_method=projector_method,
            chi=chi,
            max_iter=kw.get("max_iter", 80),
            min_iter=kw.get("min_iter", 20),
            qr_warmup_steps=kw.get("qr_warmup_steps", 6),
            conv_tol=kw.get("conv_tol", 1e-12),
        )
    )


def _phys_A0():
    """The physical C4v-symmetrized D=2 Heisenberg site tensor as a raw array."""
    A, _gate = _build_physical_state_heisenberg_D2()
    return A.todense()


def _implicit_energy_grad_scalar(recipe="1x1", projector_method="qr", chi=8):
    """Implicit-AD directional derivative along the AD gradient direction on the
    physical state — a single *scalar* gradient magnitude ``|g|`` per scheme.

    NB: the spec's first suggestion (a GLOBAL-SCALE parameter ``A -> A*theta``)
    is unusable here — the normalized CTM energy ``<H>/<1>`` makes the scale
    direction NEARLY orthogonal to the gradient (``<grad, A0>`` ~= -4e-3, ~0.6%
    of ||grad||), so its ~1e-3 scale-FD signal is swamped by O(1e2) kink noise
    and cannot distinguish a correct gradient from a wrong one.  Instead we take
    the energy derivative along the (energy-changing) gradient direction itself,
    which is the largest-signal scalar and exactly equals ``|g|`` analytically.
    """
    A0 = _phys_A0()
    g = np.asarray(
        jax.grad(
            lambda a: _implicit_energy_of_A_phys(
                a, recipe=recipe, projector_method=projector_method, chi=chi
            )
        )(A0)
    )
    return float(np.linalg.norm(g))  # == directional derivative along +grad


def _implicit_energy_grad_scalar_fd(
    recipe="1x1", projector_method="qr", eps=4e-2, chi=16
):
    """Central finite difference of the energy along the AD gradient direction.

    ``eps`` defaults to ``4e-2`` (large, plateau): at D=2 the energy landscape is
    piecewise-flat with kinks where the projector subspace is discretely
    reselected, so a small ``eps`` straddles individual kinks and is meaningless
    (the measured error is non-monotonic and noisy below ~1e-2).  In the
    large-step plateau ``eps in [2e-2, 8e-2]`` the central FD averages over the
    local flat regions and recovers the analytic directional slope ``||g||``
    stably to ~17% — the cleanest genuine FD available at this tiny, non-smooth
    size.  The FD systematically undershoots ``||g||`` (real curvature, not
    noise: it is reproducible and ~monotone across the plateau).
    """
    A0 = _phys_A0()
    g = np.asarray(
        jax.grad(
            lambda a: _implicit_energy_of_A_phys(
                a, recipe=recipe, projector_method=projector_method, chi=chi
            )
        )(A0)
    )
    d = g / np.linalg.norm(g)

    def E(step):
        return float(
            _implicit_energy_of_A_phys(
                A0 + step * d, recipe=recipe, projector_method=projector_method, chi=chi
            )
        )

    return (E(eps) - E(-eps)) / (2.0 * eps)


@pytest.mark.algorithm
def test_implicit_qr_gradient_matches_fd():
    """Implicit-AD gradient (recipe='1x1', qr) matches a directional finite diff.

    Key correctness gate, robust form.  The analytic directional derivative of
    the implicit-AD energy along the unit AD-gradient direction ``dir = g/||g||``
    is exactly ``<g, dir> == ||g||``.  We compare that to a central FD of the
    energy along the *same* direction with a LARGE plateau step.

    Why directional + large-eps: at D=2 the CTM energy landscape is
    kink-dominated (the projector subspace is discretely reselected), so a
    small-eps central FD is meaningless — its error is non-monotonic and noisy.
    Perturbing along the AD gradient and stepping into the large-eps plateau
    averages over the local flat regions and converges to the true slope.

    Measured (chi=16, eps=4e-2, deterministic across runs):
        ||g_ad|| = 0.4679, fd = 0.3902, relative error = 0.166.
    The plateau is stable: eps in {2e-2, 4e-2, 8e-2} -> rel {0.174, 0.166, 0.160}
    (tight, monotone).  TOL=0.35 leaves a comfortable >2x margin (measured error
    ~= half of TOL); it is NOT set just above the error.  A wrong backward (sign
    error, or a spurious O(1) leak) would blow past 0.35.

    NB: the spec's earlier suggestion of a ~0.3% plateau does not reproduce here;
    the genuine, reproducible agreement at this tiny non-smooth size is ~17%.
    """
    chi, eps, TOL = 16, 4e-2, 0.35
    g_ad = _implicit_energy_grad_scalar(
        recipe="1x1", projector_method="qr", chi=chi
    )  # == ||g||, the analytic directional derivative along +grad
    g_fd = _implicit_energy_grad_scalar_fd(
        recipe="1x1", projector_method="qr", eps=eps, chi=chi
    )
    assert np.isfinite(g_ad) and np.isfinite(g_fd)
    assert g_ad > 0.0  # ||g||
    assert g_fd > 0.0  # FD slope along +grad is positive (same sign as AD)
    rel_err = abs(g_fd - g_ad) / g_ad
    assert rel_err < TOL, f"directional FD off by {rel_err:.3f} (>= {TOL})"


@pytest.mark.algorithm
def test_implicit_qr_gradient_matches_eigh():
    """QR-AD gradient agrees with the eigh-AD gradient on the same physical state.

    Both ``qr`` and ``eigh`` are isometric projectors under implicit-AD
    recipe='1x1'.  At finite chi the two schemes are distinct fixed points (the
    forward energies differ at ~1e-3), so the gradient vectors are NOT identical
    — they amplify that gap to a ~10-40% difference at chi<=16.  The genuine
    parity statement is *directional*: each scheme's AD gradient, projected onto
    the OTHER scheme's gradient direction, reproduces that scheme's directional
    derivative to within the finite-chi scheme gap.
    """
    A0 = _phys_A0()
    g_qr = np.asarray(
        jax.grad(
            lambda a: _implicit_energy_of_A_phys(
                a, recipe="1x1", projector_method="qr", chi=16
            )
        )(A0)
    ).ravel()
    g_eigh = np.asarray(
        jax.grad(
            lambda a: _implicit_energy_of_A_phys(
                a, recipe="1x1", projector_method="eigh", chi=16
            )
        )(A0)
    ).ravel()

    # Sign/structure agreement: positively correlated gradients.
    cos = float(np.dot(g_qr, g_eigh) / (np.linalg.norm(g_qr) * np.linalg.norm(g_eigh)))
    assert cos > 0.8

    # Directional parity: qr-grad . eigh_dir ~ eigh dir-deriv (and symmetric).
    u_eigh = g_eigh / np.linalg.norm(g_eigh)
    u_qr = g_qr / np.linalg.norm(g_qr)
    np.testing.assert_allclose(np.dot(g_qr, u_eigh), np.dot(g_eigh, u_eigh), rtol=0.2)
    np.testing.assert_allclose(np.dot(g_eigh, u_qr), np.dot(g_qr, u_qr), rtol=0.2)


@pytest.mark.algorithm
@pytest.mark.xfail(
    reason=(
        "#858: the premise cannot hold on this configuration. The assertion "
        "'the gap does not grow as the environment is refined' requires the "
        "environment to refine, and recipe='1x1' gives corner rank 1 at EVERY "
        "chi (the #723/#726 chi_eff=1 collapse) while '2x2' grows 4->8->12->"
        "16->21 on the same state. It cannot be switched to '2x2' either: "
        "projector_method is 'Consulted only on the 1x1 recipe' (#795), so "
        "there is no non-collapsing configuration in which the QR adjoint can "
        "be compared at all. On top of that the implicit adjoint DIVERGES here "
        "-- relative residuals 9.0e-02, 1.581e+00 and 2.103e+00, i.e. a lambda "
        "that solves the adjoint equation worse than lambda=0 would -- so the "
        "gradients being compared carry no information about the true "
        "gradient. Measured gap vs chi: 0.0113, 0.0198, 0.0662, 0.0266, 0.531 "
        "at chi=4/8/12/16/24; the sampled pair (8,16) fails, (8,12) fails "
        "three times worse, and (12,16) PASSES. Re-tuning the pair would only "
        "move which coin flip is recorded."
    ),
    strict=False,
)
def test_implicit_qr_eigh_gradient_gap_shrinks_with_chi():
    """The qr-AD vs eigh-AD gradient gap does not grow as chi increases.

    Mirrors the forward ``test_reduced_qr_energy_gap_shrinks_with_chi``: the two
    isometric projector schemes are distinct at finite chi but converge to the
    same physics, so their gradient disagreement (1 - cosine similarity) must
    not grow with chi — confirming the qr backward tracks the eigh reference as
    the environment is refined rather than diverging from it.

    **Currently xfail (#858).** The intent is right and the design is sound --
    comparing two independent implementations is how a real adjoint bug would
    be caught -- but it asks the question on an environment that cannot answer
    it, with gradients too inaccurate to resolve the difference::

        chi  rank(1x1)  rank(2x2)   gap        |g_qr|     |g_eigh|
          4      1          4     0.011297895  4.4009e-01 4.3609e-01
          8      1          8     0.019805025  4.3434e-01 4.5293e-01
         12      1         12     0.066223983  4.6575e-01 4.5824e-01
         16      1         16     0.026562736  4.4307e-01 4.7237e-01
         24      1         21     0.530718621  4.9620e-01 9.6011e-01

    At chi=24 the two gradients are nearly orthogonal (cos ~ 0.47) with
    magnitudes differing by ~2x.  ``strict=False`` so that fixing either
    blocker flips this green without failing the suite in the meantime.
    """

    def cos_gap(chi):
        A0 = _phys_A0()
        g_qr = np.asarray(
            jax.grad(
                lambda a: _implicit_energy_of_A_phys(
                    a, recipe="1x1", projector_method="qr", chi=chi
                )
            )(A0)
        ).ravel()
        g_eigh = np.asarray(
            jax.grad(
                lambda a: _implicit_energy_of_A_phys(
                    a, recipe="1x1", projector_method="eigh", chi=chi
                )
            )(A0)
        ).ravel()
        cos = float(
            np.dot(g_qr, g_eigh) / (np.linalg.norm(g_qr) * np.linalg.norm(g_eigh))
        )
        return 1.0 - cos  # 0 == identical direction

    gap8 = cos_gap(8)
    gap16 = cos_gap(16)
    assert gap16 <= gap8 + 1e-9  # gap does not grow as chi increases


# --------------------------------------------------------------------------- #
# Phase 2, Task 7 — full optimize_gs_ad GS optimization under implicit AD.      #
#                                                                              #
# Tasks 5b/6 proved recipe='1x1' + qr RUNS and DIFFERENTIATES correctly under  #
# implicit-diff AD.  This validates the *whole* production GS optimizer: a few  #
# optimize_gs_ad steps with gs_recipe='1x1' + gs_projector_method='qr' must     #
# decrease the energy, stay finite, and track the eigh result on the same       #
# physical D=2 Heisenberg state.  The 1-site implicit adjoint uses the          #
# Neumann-series VJP (``ad_backward_method="vjp"``) with EMA divergence          #
# detection and a ``lam_norm`` safety truncation; we only require the run to     #
# complete without NaN / blow-up.                                                #
# --------------------------------------------------------------------------- #


def _short_optimize(gs_recipe, gs_projector_method, steps=5):
    """Run a short ``optimize_gs_ad`` (implicit AD) on the physical C4v
    D=2 Heisenberg state and return ``(initial_energy, final_energy, A_final)``.

    Starts from the C4v-symmetrized simple-update site tensor (near the 2D
    Heisenberg fixed point, E0 ~ -0.5) on the *sublattice-rotated* gate, so the
    single-site (1x1) uniform iPEPS is the correct ansatz.  Kept small/fast
    (``chi=8``, few CTM iters, few optimizer steps) — the point is the
    convergence *behavior* (decrease + finite + eigh-tracking), not a deep
    optimization.  ``su_init=False`` so the supplied ``A_init`` is honored
    (no extra simple-update rebuild).
    """
    A0, gate_rot = _build_physical_state_heisenberg_D2()
    config = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="1x1",
        gs_implicit_ad=True,
        gs_recipe=gs_recipe,
        gs_projector_method=gs_projector_method,
        su_init=False,
        gs_num_steps=steps,
        gs_learning_rate=1e-2,
        ctm=CTMConfig(
            chi=8,
            max_iter=40,
            min_iter=10,
            conv_tol=1e-10,
            projector_method=gs_projector_method,
            qr_warmup_steps=4,
        ),
    )
    # Initial energy: a zero-step run returns the energy of A_init unchanged.
    cfg0 = replace(config, gs_num_steps=0)
    _A_i, _env_i, e0 = optimize_gs_ad(gate_rot, A0, cfg0)
    A_f, _env_f, ef = optimize_gs_ad(gate_rot, A0, config)
    return float(e0), float(ef), A_f


@pytest.mark.algorithm
@pytest.mark.xfail(
    reason=(
        "#858, surfaced here by #844 -- a REAL optimizer failure, not a "
        "harness artifact, and deliberately left failing rather than retuned. "
        "The run now STARTS at e0=-0.65943 (essentially the converged answer; "
        "cf. the 2x2 reference -0.65900) and 5 Adam steps drive it UP to "
        "ef=-0.52262, an ascent of +0.1368. Before #844 changed the fixture "
        "state it started at -0.51363 and ended at -0.65949, so the same "
        "ascent read as a descent purely because the starting point was "
        "garbage; the assertion passed for the wrong reason for months. The "
        "cause is visible in the run: 'adjoint solve did not converge "
        "(relative residual 4.496e-01)' plus non-PSD RDMs at -0.0316 and "
        "-0.113 -- the C4v D=2 adjoint divergence of #858. The gradient is "
        "wrong by roughly the residual, so the optimizer is walking uphill on "
        "a direction that is not the gradient. Fixing this means fixing #858, "
        "not touching this file. strict=True on purpose: the ascent is 0.1368, "
        "far outside any BLAS variation, so if this ever passes the adjoint "
        "has genuinely been repaired and the xfail must come off."
    ),
    strict=True,
)
def test_optimize_gs_ad_qr_1x1_converges():
    """A short optimize_gs_ad run with gs_recipe='1x1' + gs_projector_method='qr'
    decreases the energy, stays finite, and reaches the physical Heisenberg
    fixed point.

    Core deliverable: the production implicit-diff GS optimizer runs end-to-end
    with the reduced-corner QR projector, the energy *decreases* (does not
    increase / NaN / blow up), and the QR-optimized state lands in the physical
    D=2 Heisenberg energy basin (~-0.66).

    Measured (chi=8, 5 Adam steps, lr=1e-2, deterministic):
        e0_qr = -0.5136, ef_qr = -0.6590 (decreased ~0.145).
    The 1-site implicit adjoint uses the Neumann-series VJP with
    divergence-truncation safeguards; the run stays finite (no NaN) and the
    energy descends.

    NOTE (#692): an earlier version re-ran a *forward eigh-CTM* oracle on the
    optimized tensor and asserted ``|ef_qr - e_eigh| < 5e-3``. That was both
    fragile and redundant. Fragile: eigh is uncertified under implicit AD
    precisely because it is unstable, and on the post-optimization tensor the
    eigh forward CTM can diverge (``e_eigh ~ 15`` on some CI BLAS/XLA builds)
    even though the certified QR result ``ef_qr`` stays physical. Redundant:
    forward QR-vs-eigh agreement on the base SU state is already covered by
    ``test_reduced_qr_energy_matches_eigh_heisenberg_D2``. We therefore assert
    the actual deliverable — ``ef_qr`` reaches the physical energy window —
    which depends only on the stable QR result.
    """
    e0_qr, ef_qr, _A_qr = _short_optimize(
        gs_recipe="1x1", gs_projector_method="qr", steps=5
    )
    assert np.isfinite(ef_qr)
    assert ef_qr <= e0_qr + 1e-9  # energy does not increase
    # Reached the physical D=2 Heisenberg basin (~-0.66); the wide window
    # excludes divergence without over-constraining the 5-step descent depth.
    assert -0.75 < ef_qr < -0.45, f"ef_qr={ef_qr} outside physical Heisenberg window"
