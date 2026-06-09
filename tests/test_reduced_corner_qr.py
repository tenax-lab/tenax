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

from tenax.algorithms._ctm_projector import (
    _compute_projector_tensor,
    _reduced_qr_projector,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

# --- energy-agreement harness imports (reused from the spike) ---------------
from tenax import CTMConfig, heisenberg_gate, ipeps, iPEPSConfig
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import sublattice_rotate_gate, symmetrize_c4v


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


@pytest.mark.parametrize("chi", [4, 8])
def test_reduced_qr_projector_is_isometry(chi):
    """``P`` has labels ``(fused, chi_new)`` and ``P^dagger P = I_chi``."""
    C1g, C4g = _build_dense_enlarged_corners(chi)

    P = _reduced_qr_projector(C1g, C4g, chi)

    assert P.labels() == ("fused", "chi_new")
    assert P._data.shape == (chi * 2 * 2, chi)

    # Isometry: contracting P.bar() with P over the fused leg yields I_chi.
    gram = P._data.conj().T @ P._data
    np.testing.assert_allclose(
        np.asarray(gram), np.eye(chi), atol=1e-9
    )


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


def test_gauge_fix_qr_dense_makes_diag_R_nonneg_and_preserves_QR():
    import jax, jax.numpy as jnp, numpy as np
    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms._ctm_projector import _gauge_fix_qr_dense
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


def test_gauge_fix_qr_dense_is_smooth_under_perturbation():
    import jax, jax.numpy as jnp, numpy as np
    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms._ctm_projector import _gauge_fix_qr_dense
    key = jax.random.PRNGKey(5)
    M = jax.random.normal(key, (12, 6))
    dM = 1e-7 * jax.random.normal(jax.random.PRNGKey(6), (12, 6))
    def q_of(scale):
        Q, R = jnp.linalg.qr(M + scale * dM)
        Qf, _ = _gauge_fix_qr_dense(Q, R)
        return Qf
    assert jnp.max(jnp.abs(q_of(1.0) - q_of(0.0))) < 1e-3  # no O(1) sign flip for O(eps) perturbation


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


def _heisenberg_D2_ctm_energy_1x1(chi, projector_method):
    """Converged single-site (1x1) dense CTM energy for the given projector.

    Mirrors the spike's drive of the canonical single-site sweep
    (``_ctm_tensor_sweep``, reached here via the public ``ctm_tensor`` entry on
    a DenseTensor, which selects ``_ctm_tensor_sweep`` and therefore exercises
    ``_compute_projector_tensor`` — the 1x1 path the spec points at).  The
    ``"qr"`` method runs the ``qr_warmup_steps`` eigh warm-up (matching the
    spike's 6-sweep eigh warm-up) before switching to the reduced-corner QR
    projector; energy via ``compute_energy_ctm_tensor(A, env, gate_rot)``.
    """
    A, gate_rot = _build_physical_state_heisenberg_D2()
    env, _eps = ctm_tensor(
        A,
        chi=chi,
        max_iter=200,
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
