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
    """FD-vs-AD parity at D=2, chi_initial=4, chi_max=8 with forced bump.

    The correctness gate for chi-lock: confirms that when the forward CTM
    grows chi 4 -> 8 via in-CTM bump, the backward operates at chi_post=8
    (not the closure-captured chi_initial=4) and produces a gradient that
    matches finite-difference.
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
            max_iter=6,
            min_iter=2,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-12,  # force bump on iter 1
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=8,
            gmres_tol=1e-8,  # 100x margin vs atol=1e-5 below
        )

    grad_ad = jax.grad(loss)(flat_init)
    grad_fd = _central_diff(loss, flat_init, eps=1e-4)

    # Tol: 1e-5 abs is the chi-lock design target; rel 1e-3 absorbs FD
    # truncation error on small-magnitude components.
    assert jnp.allclose(grad_ad, grad_fd, atol=1e-5, rtol=1e-3), (
        f"AD gradient diverges from FD reference.\n"
        f"max |grad_ad - grad_fd| = {jnp.max(jnp.abs(grad_ad - grad_fd))}\n"
        f"grad_ad[:5] = {grad_ad[:5]}\n"
        f"grad_fd[:5] = {grad_fd[:5]}"
    )
