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
def test_ad_gradient_matches_fd_with_bump():
    """Numerical smoke at D=2 with forced bump: gradient is finite, FD-consistent.

    Confirms ctm_energy_implicit returns a finite, FD-consistent gradient
    (no NaNs, correct sign, correct order of magnitude) when the forward
    CTM grows chi 4 -> 8.  Tolerance is relaxed (atol=1e-2, rtol=1e-1)
    because the 2x2 plaquette projector's stop_gradient (PR #447) creates
    a documented ~25% FD bias at D=2.

    This test does NOT verify the chi-lock contract — at the relaxed
    tolerance a chi=4 backward Jacobian would still pass.  The strict
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

    # Tol: atol=1e-2 / rtol=1e-1 accommodates the ~25% D=2 FD bias from
    # PR #447's projector stop_gradient.  The chi-lock contract is checked
    # in test_ad_gradient_invariance_bump_vs_fixed_chi_max, not here.
    assert jnp.allclose(grad_ad, grad_fd, atol=1e-2, rtol=1e-1), (
        f"AD gradient diverges from FD reference.\n"
        f"max |grad_ad - grad_fd| = {jnp.max(jnp.abs(grad_ad - grad_fd))}\n"
        f"grad_ad[:5] = {grad_ad[:5]}\n"
        f"grad_fd[:5] = {grad_fd[:5]}"
    )


@pytest.mark.slow
def test_ad_gradient_invariance_bump_vs_fixed_chi_max():
    """Bump path's gradient must equal fixed-chi=chi_max path's gradient.

    This is the strict chi-lock correctness gate.  Both calls converge
    CTM at chi=8: one starts at chi=4 and grows via in-CTM bump, the
    other starts directly at chi=8 and doesn't bump.  At convergence both
    reach the same CTM fixed point, so the gradients must agree to
    floating-point noise.

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
    # The fixed point is unique (verified by both losses returning the same
    # energy to 1e-10); the gradients are therefore equal up to numerical
    # noise from independent GMRES solves.  atol=1e-6 gives 100x margin
    # over the chosen gmres_tol=1e-8.
    assert jnp.allclose(
        loss_with_bump(flat_init), loss_fixed_chi_max(flat_init), atol=1e-10
    ), "CTM fixed points disagree — invariance test premise is broken"
    assert jnp.allclose(grad_bump, grad_fixed, atol=1e-6, rtol=1e-6), (
        f"chi-lock contract broken: bump gradient at chi_max=8 disagrees with "
        f"fixed-chi=8 reference.\n"
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
