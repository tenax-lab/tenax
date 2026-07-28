"""Implicit-AD + in-CTM chi-bump correctness tests (#516 chi-lock).

These tests verify that ctm_energy_implicit produces correct gradients
when ctmrg_heuristic_increase_chi=True forces the forward CTM to grow
chi mid-convergence.  See docs/plans/2026-05-20-chi-lock-design.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry


def _build_site_tensor(D: int = 2, d: int = 2, seed: int = 0) -> DenseTensor:
    """Build a small trivial-U(1) (D, d) single-site tensor for D=2 probes.

    Uses a trivial U(1) symmetry (all charges zero) wrapped in DenseTensor,
    matching the validation-shim pattern used by ``test_ctm_energy_implicit``.
    """
    rng = np.random.default_rng(seed)
    sym = U1Symmetry()
    bond_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
    )
    data = jnp.asarray(
        rng.standard_normal((d, D, D, D, D)).astype(np.float64), dtype=jnp.float64
    )
    return DenseTensor(data, indices)


def test_implicit_ad_no_longer_raises_with_bump():
    """ctm_energy_implicit(..., ctmrg_heuristic_increase_chi=True) returns a value.

    Was a NotImplementedError before chi-lock (#516); now should run.
    """
    site_tensors = {(0, 0): _build_site_tensor()}
    gate = heisenberg_gate()

    energy = ctm_energy_implicit(
        site_tensors,
        SINGLE_SITE_NEIGHBORS,
        gate,
        chi=4,
        max_iter=4,
        ctmrg_heuristic_increase_chi=True,
        chi_max=8,
    )
    assert jnp.isfinite(energy)


def _central_diff(f, x: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """Central-difference gradient of scalar f at x (flat vector)."""
    grad = jnp.zeros_like(x)
    for i in range(x.shape[0]):
        ei = jnp.zeros_like(x).at[i].set(eps)
        f_plus = f(x + ei)
        f_minus = f(x - ei)
        grad = grad.at[i].set((f_plus - f_minus) / (2 * eps))
    return grad


@pytest.mark.slow
def test_forward_fixed_point_trajectory_independent():
    """#702 regression gate: bump 4->8 == direct chi=8 forward fixed point.

    The CTM fixed point at chi=8 is unique, so the bump path (start chi=4,
    grow in-CTM) and the direct chi=8 path must converge to the same energy.
    PR #676 broke this (|diff| 7e-5) by making the corner label convention
    non-uniform across the sweep; the sweep-invariant natural convention
    restores agreement to machine precision.  Forward-only — much cheaper
    than the AD gates below, and catches the forward regression class
    directly.
    """
    A = _build_site_tensor(D=2, d=2, seed=42)
    gate = heisenberg_gate()
    common = dict(max_iter=200, min_iter=2, conv_tol=1e-12, gmres_tol=1e-8)

    e_bump = ctm_energy_implicit(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        gate,
        chi=4,
        ctmrg_heuristic_increase_chi=True,
        ctmrg_heuristic_increase_chi_threshold=1e-12,
        ctmrg_heuristic_increase_chi_step_size=2,
        chi_max=8,
        **common,
    )
    e_fixed = ctm_energy_implicit(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        gate,
        chi=8,
        ctmrg_heuristic_increase_chi=False,
        **common,
    )
    assert abs(float(e_bump) - float(e_fixed)) < 1e-9, (
        f"trajectory-dependent fixed point: E_bump={float(e_bump):.10f} "
        f"E_fixed={float(e_fixed):.10f} |diff|={abs(float(e_bump) - float(e_fixed)):.3e}"
    )


@pytest.mark.slow
def test_ad_gradient_matches_fd_with_bump():
    """Numerical smoke at D=2 with forced bump: gradient is finite, FD-consistent.

    Confirms ctm_energy_implicit returns a finite, FD-consistent gradient
    (no NaNs, correct sign, correct order of magnitude) when the forward
    CTM grows chi 4 -> 8.

    The bulk-agreement check (90th-percentile relative error) is used
    instead of strict allclose because the 2x2 plaquette projector's
    stop_gradient (PR #447) creates a documented ~25% FD bias at D=2,
    and individual FD probes can occasionally cross the projector's
    near-degenerate SVD threshold — on macOS Accelerate this produced a
    single-index ~0.3 outlier in 32 elements (#529) while the bulk
    agreement was as tight as on Linux.

    This test does NOT verify the chi-lock contract.  The strict
    chi-lock check is test_ad_gradient_invariance_bump_vs_fixed_chi_max
    below, which factors out the D=2 projector bias by comparing the
    bump-path gradient against a fixed-chi=8 reference.
    """
    A = _build_site_tensor(D=2, d=2, seed=42)
    gate = heisenberg_gate()
    A_data = A.todense()
    flat_init = A_data.flatten()

    def loss(A_flat: jnp.ndarray) -> jnp.ndarray:
        A_perturbed = DenseTensor(A_flat.reshape(A_data.shape), A.indices)
        return ctm_energy_implicit(
            {(0, 0): A_perturbed},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=200,
            min_iter=2,
            conv_tol=1e-12,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-12,  # force bump on iter 1
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=8,
            gmres_tol=1e-8,
        )

    grad_ad = jax.grad(loss)(flat_init)
    grad_fd = _central_diff(loss, flat_init, eps=1e-4)

    assert jnp.all(jnp.isfinite(grad_ad)), "AD gradient contains non-finite values"
    # NaN in grad_fd would silently evade the violation count below
    # (NaN > x is False under IEEE 754), so it must be a hard failure.
    assert jnp.all(jnp.isfinite(grad_fd)), "FD gradient contains non-finite values"

    # Per-element allclose tolerance (the original strict check).
    atol, rtol = 1e-2, 1e-1
    abs_diff = jnp.abs(grad_ad - grad_fd)
    elem_tol = atol + rtol * jnp.abs(grad_fd)
    violations = abs_diff > elem_tol
    n_viol = int(jnp.sum(violations))

    # Allow up to 1 out of 32 indices to exceed the element-wise tolerance.
    # An FD probe can occasionally cross the 2x2 plaquette projector's
    # near-degenerate SVD threshold (PR #447) and produce a single-index
    # discontinuity; macOS Accelerate hit this for 1 index per #529, while
    # all other 31 indices agreed as tightly as on Linux.  A real AD
    # regression would corrupt many indices simultaneously and fail this
    # bound — single-outlier robustness does not cost regression sensitivity.
    assert n_viol <= 1, (
        f"AD/FD per-element disagreement: {n_viol} of {abs_diff.size} indices "
        f"exceed atol={atol} + rtol={rtol}*|grad_fd|.\n"
        f"per-index abs_diff = {abs_diff}\n"
        f"per-index elem_tol = {elem_tol}\n"
        f"grad_ad[:5] = {grad_ad[:5]}\n"
        f"grad_fd[:5] = {grad_fd[:5]}"
    )


@pytest.mark.slow
def test_ad_gradient_invariance_bump_vs_fixed_chi_max():
    """Bump path's gradient must equal fixed-chi=chi_max path's gradient.

    This is the chi-lock correctness gate.  Both calls converge CTM at
    chi=8: one starts at chi=4 and grows via in-CTM bump, the other
    starts directly at chi=8 and doesn't bump.  At convergence both reach
    the same CTM fixed point (energies agree to machine precision), and
    the gradients agree up to the degenerate-projector-subspace floor
    (~1e-3 on random D=2 probes — see the inline comment at the asserts).

    Falsification: if the chi-lock were broken (backward still uses
    closure-captured chi_initial=4 instead of chi_post=8), the bump-path
    gradient would be computed against a chi=4 truncated Jacobian and
    would not match the fixed-chi=8 reference.  This test would fail
    catastrophically (~100% relative error) in that case.

    """
    A = _build_site_tensor(D=2, d=2, seed=42)
    gate = heisenberg_gate()
    A_data = A.todense()
    flat_init = A_data.flatten()

    common_kwargs = dict(
        max_iter=200,
        min_iter=2,
        conv_tol=1e-12,
        gmres_tol=1e-8,
    )

    def loss_with_bump(A_flat: jnp.ndarray) -> jnp.ndarray:
        A_perturbed = DenseTensor(A_flat.reshape(A_data.shape), A.indices)
        return ctm_energy_implicit(
            {(0, 0): A_perturbed},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-12,
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=8,
            **common_kwargs,
        )

    def loss_fixed_chi_max(A_flat: jnp.ndarray) -> jnp.ndarray:
        A_perturbed = DenseTensor(A_flat.reshape(A_data.shape), A.indices)
        return ctm_energy_implicit(
            {(0, 0): A_perturbed},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=8,
            ctmrg_heuristic_increase_chi=False,
            **common_kwargs,
        )

    grad_bump = jax.grad(loss_with_bump)(flat_init)
    grad_fixed = jax.grad(loss_fixed_chi_max)(flat_init)

    # Both CTM fixed points are at chi=8, reached via different trajectories.
    # The fixed point is unique: both losses return the same energy to
    # machine precision (measured |dE| <= 2e-14 across seeds 0..42).
    assert jnp.allclose(
        loss_with_bump(flat_init), loss_fixed_chi_max(flat_init), atol=1e-10
    ), "CTM fixed points disagree — invariance test premise is broken"

    # Gradient agreement is floored at ~1e-3 (measured 6.7e-4..2.5e-2 across
    # seeds): the two trajectories land on energy-identical envs that differ
    # by a rotation inside (near-)degenerate projector-SV subspaces, and the
    # projector stop_gradient + degenerate-SV SVD backward feel that
    # rotation.  Both gradients are individually FD-consistent
    # (test_ad_gradient_matches_fd_with_bump).  The gates below sit above
    # the floor with >10x margin while still failing catastrophically on
    # the chi-lock breakage signature (backward differentiating the chi=4
    # Jacobian shrinks |grad_bump| ~10x — norm ratio would drop to ~0.1).
    norm_ratio = jnp.linalg.norm(grad_bump) / jnp.linalg.norm(grad_fixed)
    assert 0.9 < float(norm_ratio) < 1.1, (
        f"chi-lock contract broken: |grad_bump|/|grad_fixed| = {norm_ratio} "
        f"(expected ~1; ~0.1 is the chi_initial-Jacobian signature).\n"
        f"grad_bump[:5] = {grad_bump[:5]}\n"
        f"grad_fixed[:5] = {grad_fixed[:5]}"
    )
    assert jnp.allclose(grad_bump, grad_fixed, atol=5e-2, rtol=5e-2), (
        f"bump gradient disagrees with fixed-chi=8 reference beyond the "
        f"degenerate-subspace floor.\n"
        f"max |grad_bump - grad_fixed| = {jnp.max(jnp.abs(grad_bump - grad_fixed))}\n"
        f"grad_bump[:5] = {grad_bump[:5]}\n"
        f"grad_fixed[:5] = {grad_fixed[:5]}"
    )


def test_chi_bump_fires_when_smin_above_threshold():
    """With threshold=1e-12, the in-CTM bump fires on iter 1 → chi grows.

    Asserts the forward CTM grew env first-dim by reading the env-cache
    final_chi via the same path used internally by f_bwd's bounds check.
    """
    from tenax.algorithms._ctm_energy_ad import _sigma_gauged_ctm_converge

    site_tensors = {(0, 0): _build_site_tensor()}
    envs, chi_post = _sigma_gauged_ctm_converge(
        site_tensors,
        SINGLE_SITE_NEIGHBORS,
        chi=4,
        max_iter=6,
        conv_tol=1e-5,
        projector_method="svd",
        renormalize=True,
        projector_backward="auto",
        qr_warmup_steps=0,
        env_init=None,
        forward_gauge="phase",
        conv_method="sv",
        min_iter=2,
        ctmrg_heuristic_increase_chi=True,
        ctmrg_heuristic_increase_chi_threshold=1e-12,  # force bump
        ctmrg_heuristic_increase_chi_step_size=2,
        chi_max=8,
    )
    assert chi_post > 4, f"Expected bump fired (chi_post > 4), got chi_post={chi_post}"
    assert chi_post <= 8, f"chi_post={chi_post} exceeded chi_max=8"


def test_chi_bump_does_not_fire_when_below_threshold():
    """With threshold=1e10 (above any realistic smin), bump quiesces."""
    from tenax.algorithms._ctm_energy_ad import _sigma_gauged_ctm_converge

    site_tensors = {(0, 0): _build_site_tensor()}
    envs, chi_post = _sigma_gauged_ctm_converge(
        site_tensors,
        SINGLE_SITE_NEIGHBORS,
        chi=4,
        max_iter=6,
        conv_tol=1e-5,
        projector_method="svd",
        renormalize=True,
        projector_backward="auto",
        qr_warmup_steps=0,
        env_init=None,
        forward_gauge="phase",
        conv_method="sv",
        min_iter=2,
        ctmrg_heuristic_increase_chi=True,
        ctmrg_heuristic_increase_chi_threshold=1e10,  # never fires
        ctmrg_heuristic_increase_chi_step_size=2,
        chi_max=8,
    )
    assert chi_post == 4, f"Expected no bump (chi_post == 4), got chi_post={chi_post}"


def test_ad_gradient_equals_fixed_chi_when_no_bump_fires():
    """With chi_max == chi_initial, gradient is identical to bump=False.

    Verifies the chi-lock plumbing is a no-op when bump can't fire.
    Detects accidental drift in the chi_initial == chi_post path.
    """
    A = _build_site_tensor(D=2, d=2, seed=7)
    gate = heisenberg_gate()
    A_data = A.todense()
    flat_init = A_data.flatten()

    def loss_with_bump(A_flat: jnp.ndarray) -> jnp.ndarray:
        A_perturbed = DenseTensor(A_flat.reshape(A_data.shape), A.indices)
        return ctm_energy_implicit(
            {(0, 0): A_perturbed},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=6,
            min_iter=2,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-12,
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=4,  # == chi_initial → no room to bump
        )

    def loss_no_bump(A_flat: jnp.ndarray) -> jnp.ndarray:
        A_perturbed = DenseTensor(A_flat.reshape(A_data.shape), A.indices)
        return ctm_energy_implicit(
            {(0, 0): A_perturbed},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=6,
            min_iter=2,
            ctmrg_heuristic_increase_chi=False,
        )

    grad_bump = jax.grad(loss_with_bump)(flat_init)
    grad_no_bump = jax.grad(loss_no_bump)(flat_init)

    assert jnp.allclose(grad_bump, grad_no_bump, atol=1e-10, rtol=1e-10), (
        f"chi-lock plumbing leaks when bump can't fire.\n"
        f"max diff = {jnp.max(jnp.abs(grad_bump - grad_no_bump))}"
    )
