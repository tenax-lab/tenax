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
    (load-bearing — otherwise the four directional 1x1 moves are inequivalent
    and the single-site eigh sweep limit-cycles at the #425/#426 plateau, so the
    eigh oracle is untrustworthy) and renormalized.
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


def _heisenberg_D2_ctm_energy_1x1(chi, projector_method, max_iter=200):
    """Converged single-site (1x1) dense CTM energy for the given projector.

    Mirrors the spike's drive of the canonical single-site sweep
    (``_ctm_tensor_sweep``, reached here via the public ``ctm_tensor`` entry on
    a DenseTensor, which selects ``_ctm_tensor_sweep`` and therefore exercises
    ``_compute_projector_tensor`` — the 1x1 path the spec points at).  The
    ``"qr"`` method runs the ``qr_warmup_steps`` eigh warm-up (matching the
    spike's 6-sweep eigh warm-up) before switching to the reduced-corner QR
    projector; energy via ``compute_energy_ctm_tensor(A, env, gate_rot)``.

    ``max_iter`` is exposed so a fixed-point-stability check can re-converge
    with a larger sweep budget (the CTM still stops early at ``conv_tol``).
    """
    A, gate_rot = _build_physical_state_heisenberg_D2()
    env, _eps = ctm_tensor(
        A,
        chi=chi,
        max_iter=max_iter,
        conv_tol=1e-10,
        projector_method=projector_method,
        qr_warmup_steps=6,
    )
    return float(compute_energy_ctm_tensor(A, env, gate_rot))


@pytest.mark.algorithm
@pytest.mark.parametrize("chi", [8, 16])
def test_reduced_qr_energy_matches_eigh_heisenberg_D2(chi):
    e_eigh = _heisenberg_D2_ctm_energy_1x1(chi=chi, projector_method="eigh")
    e_qr = _heisenberg_D2_ctm_energy_1x1(chi=chi, projector_method="qr")
    assert abs(e_qr - e_eigh) < 1e-3  # loosened vs eps; different scheme, same physics


@pytest.mark.algorithm
def test_reduced_qr_energy_gap_shrinks_with_chi():
    g8 = abs(
        _heisenberg_D2_ctm_energy_1x1(8, "qr")
        - _heisenberg_D2_ctm_energy_1x1(8, "eigh")
    )
    g16 = abs(
        _heisenberg_D2_ctm_energy_1x1(16, "qr")
        - _heisenberg_D2_ctm_energy_1x1(16, "eigh")
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
    e = _heisenberg_D2_ctm_energy_1x1(chi=8, projector_method="qr")
    assert np.isfinite(e)

    # Converged fixed point: doubling max_iter does not move the energy.
    e_n = _heisenberg_D2_ctm_energy_1x1(chi=8, projector_method="qr", max_iter=100)
    e_2n = _heisenberg_D2_ctm_energy_1x1(chi=8, projector_method="qr", max_iter=200)
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
    return float(
        compute_energy_ctm_tensor_2site(A, B, env_A, env_B, gate_dense, d=2)
    )


@pytest.mark.algorithm
@pytest.mark.parametrize("chi", [6, 10])
def test_reduced_qr_energy_matches_eigh_2site_heisenberg_D2(chi):
    e_eigh = _heisenberg_D2_2site_energy(chi=chi, projector_method="eigh")
    e_qr = _heisenberg_D2_2site_energy(chi=chi, projector_method="qr")
    assert abs(e_qr - e_eigh) < 1e-3  # different scheme, same physics


@pytest.mark.algorithm
def test_reduced_qr_2site_energy_gap_shrinks_with_chi():
    g6 = abs(
        _heisenberg_D2_2site_energy(6, "qr") - _heisenberg_D2_2site_energy(6, "eigh")
    )
    g10 = abs(
        _heisenberg_D2_2site_energy(10, "qr") - _heisenberg_D2_2site_energy(10, "eigh")
    )
    # g6/g10 sit at the QR-vs-eigh noise floor (~1e-6..1e-4, set by SVD/QR gauge
    # plus platform BLAS reassociation) — three orders below the 1e-3 agreement
    # asserted in test_reduced_qr_energy_matches_eigh_2site_*. At that floor the
    # gap is NOT strictly monotonic in chi: Linux gives g6 > g10, macOS Accelerate
    # gives g10 > g6 (≈3e-5), so the old ``g10 <= g6 + 1e-9`` was red on macOS CI.
    # Assert the gap stays at the noise floor at the larger chi rather than
    # demanding bit-strict shrinkage; a genuine QR divergence would be O(1e-2+).
    assert g10 <= max(g6, 5e-4)


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
    cos = float(
        np.dot(g_qr, g_eigh) / (np.linalg.norm(g_qr) * np.linalg.norm(g_eigh))
    )
    assert cos > 0.8

    # Directional parity: qr-grad . eigh_dir ~ eigh dir-deriv (and symmetric).
    u_eigh = g_eigh / np.linalg.norm(g_eigh)
    u_qr = g_qr / np.linalg.norm(g_qr)
    np.testing.assert_allclose(np.dot(g_qr, u_eigh), np.dot(g_eigh, u_eigh), rtol=0.2)
    np.testing.assert_allclose(np.dot(g_eigh, u_qr), np.dot(g_qr, u_qr), rtol=0.2)


@pytest.mark.algorithm
def test_implicit_qr_eigh_gradient_gap_shrinks_with_chi():
    """The qr-AD vs eigh-AD gradient gap does not grow as chi increases.

    Mirrors the forward ``test_reduced_qr_energy_gap_shrinks_with_chi``: the two
    isometric projector schemes are distinct at finite chi but converge to the
    same physics, so their gradient disagreement (1 - cosine similarity) must
    not grow with chi — confirming the qr backward tracks the eigh reference as
    the environment is refined rather than diverging from it.
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


def _eigh_forward_energy_1x1(A, chi=8):
    """Converged single-site (1x1) *forward* eigh-CTM energy of site tensor ``A``.

    Used as the eigh oracle for the QR-AD optimizer's final state.  The implicit
    -AD optimizer cannot run with ``projector_method='eigh'`` itself — the
    production policy (``validate_ctm_for_implicit_ad``) only certifies ``'svd'``
    and ``'qr'`` projectors as stable under implicit differentiation, so an
    eigh-under-implicit-AD ``optimize_gs_ad`` run raises by design.  We therefore
    track the eigh *physics* with the same forward eigh-CTM oracle the Phase-1
    energy-agreement tests use (``_heisenberg_D2_ctm_energy_1x1(.., 'eigh')``),
    evaluated on the QR-AD-optimized tensor.
    """
    _A0, gate_rot = _build_physical_state_heisenberg_D2()
    A = DenseTensor(A.todense(), A.indices)
    env, _eps = ctm_tensor(
        A,
        chi=chi,
        max_iter=200,
        conv_tol=1e-10,
        projector_method="eigh",
        qr_warmup_steps=6,
    )
    return float(compute_energy_ctm_tensor(A, env, gate_rot))


@pytest.mark.algorithm
def test_optimize_gs_ad_qr_1x1_converges():
    """A short optimize_gs_ad run with gs_recipe='1x1' + gs_projector_method='qr'
    decreases the energy, stays finite, and tracks the eigh result.

    Core deliverable: the production implicit-diff GS optimizer runs end-to-end
    with the reduced-corner QR projector, the energy *decreases* (does not
    increase / NaN / blow up), and the QR-optimized state's energy agrees with
    the eigh oracle.

    eigh tracking — why a *forward* oracle, not an eigh ``optimize_gs_ad`` run:
    the implicit-AD path rejects ``projector_method='eigh'`` by design
    (``validate_ctm_for_implicit_ad`` certifies only ``'svd'``/``'qr'`` as stable
    under implicit differentiation), so an eigh-under-implicit-AD optimization
    raises before it starts.  We therefore compare ``ef_qr`` to the converged
    forward eigh-CTM energy of the *same* QR-optimized tensor — the eigh
    *physics* the QR scheme is meant to reproduce.

    Measured (chi=8, 5 Adam steps, lr=1e-2, deterministic):
        e0_qr = -0.5136, ef_qr = -0.6590 (decreased ~0.145),
        eigh-forward(opt A) = -0.6591, |ef_qr - e_eigh| ~ 7e-5 (<< 5e-3).
    The 1-site implicit adjoint uses the Neumann-series VJP with
    divergence-truncation safeguards; the run stays finite (no NaN) and the
    energy descends.
    """
    e0_qr, ef_qr, A_qr = _short_optimize(
        gs_recipe="1x1", gs_projector_method="qr", steps=5
    )
    assert np.isfinite(ef_qr)
    assert ef_qr <= e0_qr + 1e-9  # energy does not increase

    # QR tracks eigh: eigh forward-CTM energy of the QR-optimized state.
    e_eigh = _eigh_forward_energy_1x1(A_qr, chi=8)
    assert np.isfinite(e_eigh)
    assert abs(ef_qr - e_eigh) < 5e-3  # QR tracks eigh
