"""Tests for ctm_energy_implicit: FD-AD gradient comparison."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit, ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate, ipeps
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _make_minimal_site_tensors_for_validation(seed: int = 0):
    """Build a minimal trivial-U(1) (D=2, d=2) site tensor for validation tests.

    Used by validation smoke tests that only need a syntactically valid
    site_tensors dict — the validation in ctm_energy_{implicit,explicit}
    fires before any CTM work, so a random tensor is fine.
    """
    import numpy as np

    from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

    rng = np.random.default_rng(seed)
    D, d = 2, 2
    sym = U1Symmetry()
    bond_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys_charges.copy(), FlowDirection.IN, label="p"),
    )
    site = DenseTensor(
        rng.standard_normal((D, D, D, D, d)).astype(np.float64),
        indices,
    )
    return {(0, 0): site}


def _make_su_tensor(D=2, d=2):
    """Create a simple-update initialized tensor (well-conditioned for CTM)."""
    gate = heisenberg_gate()
    config = iPEPSConfig(max_bond_dim=D, num_imaginary_steps=100, dt=0.05)
    _, (A_su, _), _ = ipeps(gate, None, config)
    data = A_su.todense()
    data = data / jnp.linalg.norm(data)
    return _wrap_as_dense_tensor(data)


@pytest.mark.slow
def test_ctm_energy_implicit_gradient_is_finite_and_nontrivial():
    """Gradient from GMRES backward is finite, non-zero, and well-scaled.

    Captures the production contract for implicit-AD: the optimizer needs
    a finite, non-zero gradient with magnitude comparable to the energy.
    Strict FD-parity is *not* checked here — at D=2 χ=4 the 2x2 plaquette
    projector's stop_gradient (PR #447) drops the basis-rotation
    contribution from ∂(projector)/∂A, giving ~25% FD bias.  The bias
    shrinks at larger bond dimension and does not block L-BFGS
    convergence (verified empirically up to D=3, χ=24).
    """
    A = _make_su_tensor(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS
    A_data = A.todense()

    def energy_fn(params_data):
        A_local = _wrap_as_dense_tensor(params_data)
        site_tensors = {(0, 0): A_local}
        return ctm_energy_implicit(
            site_tensors,
            neighbors,
            gate,
            chi=chi,
            max_iter=100,
            conv_tol=1e-12,
            gmres_tol=1e-10,
            gmres_maxiter=300,
            gmres_restart=50,
        )

    E = float(energy_fn(A_data))
    grad_ad = jax.grad(energy_fn)(A_data)

    assert jnp.all(jnp.isfinite(grad_ad)), "Gradient has NaN/Inf entries"
    g_norm = float(jnp.linalg.norm(grad_ad))
    assert g_norm > 1e-6 * (abs(E) + 1e-12), (
        f"Gradient norm too small: g_norm={g_norm:.3e}, E={E:.3e}"
    )
    assert g_norm < 1e6 * (abs(E) + 1e-12), (
        f"Gradient norm too large: g_norm={g_norm:.3e}, E={E:.3e}"
    )


@pytest.mark.slow
def test_ctm_energy_implicit_forward_runs():
    """Smoke test: forward pass produces a reasonable energy."""
    A = _make_su_tensor(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS

    site_tensors = {(0, 0): A}
    energy = ctm_energy_implicit(
        site_tensors,
        neighbors,
        gate,
        chi=chi,
        max_iter=40,
        conv_tol=1e-8,
    )
    assert jnp.isfinite(energy), f"Energy is not finite: {energy}"
    assert energy.shape == (), f"Energy should be scalar, got shape {energy.shape}"


def test_ctm_energy_implicit_chi_max_required_when_bump_enabled():
    """Passing ctmrg_heuristic_increase_chi=True without chi_max must raise.

    Validates that the four bump kwargs are plumbed through
    ctm_energy_implicit → dispatch → _make_implicit_vjp_fn →
    _sigma_gauged_ctm_converge.  The validation in
    _sigma_gauged_ctm_converge fires before any CTM work happens, so we
    only need a minimal valid call to confirm the ValueError surfaces.
    """
    site_tensors = _make_minimal_site_tensors_for_validation()
    gate = heisenberg_gate()

    with pytest.raises(ValueError, match="chi_max"):
        ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=2,
            ctmrg_heuristic_increase_chi=True,
            chi_max=None,
        )


def test_ctm_energy_explicit_chi_max_required_when_bump_enabled():
    """ctm_energy_explicit(..., ctmrg_heuristic_increase_chi=True, chi_max=None) raises.

    Mirrors the implicit-AD validation smoke (Task 5).  The explicit-AD
    forward (#514 Task 6) plumbs the same four bump kwargs and reuses the
    same validation contract, so the ValueError must fire before any CTM
    work happens.
    """
    site_tensors = _make_minimal_site_tensors_for_validation()
    gate = heisenberg_gate()

    with pytest.raises(ValueError, match="chi_max"):
        ctm_energy_explicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            warmup_steps=2,
            backprop_steps=1,
            ctmrg_heuristic_increase_chi=True,
            chi_max=None,
        )
