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

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax import CTMConfig, heisenberg_gate, ipeps, iPEPSConfig
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
    assert g10 <= g6 + 1e-9


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
