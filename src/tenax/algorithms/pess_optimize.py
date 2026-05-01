"""AD loss closure for kagome iPESS via the native rank-4 honeycomb CTM.

This module wires the convention-A kagome iPESS coarse-graining
(:func:`tenax.algorithms.pess.pess_to_honeycomb_supersites`) into Tenax's
native rank-4 honeycomb implicit-AD CTM
(:func:`tenax.algorithms.honeycomb_ctm.honeycomb_ctm_energy_implicit`).

Each sublattice supersite is one kagome triangle (up or down): three
physical legs and three honeycomb-edge legs. We fuse the three physical
legs into a single ``d_eff = d**3`` leg and feed
``sites = {(0, 0): A_u, (1, 0): A_d}`` (rank-4, labels
``("e0", "e1", "e2", "phys")``) to the native honeycomb path. The kagome
triangle Hamiltonian lives entirely inside one supersite, so the energy
contract is the intra-triangle 1-site sum provided by
:func:`compute_honeycomb_triangle_energy`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.honeycomb_ctm import (
    compute_honeycomb_triangle_energy,
    honeycomb_ctm_energy_implicit,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import IPESSState, pess_to_honeycomb_supersites
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

__all__ = ["build_pess_loss"]


def _make_honeycomb_indices(D: int, d_eff: int) -> tuple[TensorIndex, ...]:
    """Build the rank-4 ``(e0, e1, e2, phys)`` index tuple expected by the
    native honeycomb CTM. Trivial U(1) charges (single sector at charge 0)."""
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d_eff, dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )


def _supersite_to_honeycomb_tensor(
    A_super: jax.Array, indices: tuple[TensorIndex, ...]
) -> DenseTensor:
    """Reshape a ``(d, d, d, D, D, D)`` supersite into a ``(D, D, D, d^3)``
    rank-4 honeycomb site tensor.

    The three physical legs ``(p_a, p_b, p_c)`` are fused into the ``phys``
    leg in row-major (C) order so the basis matches
    ``np.kron(np.kron(s_a, s_b), s_c)`` used by
    :func:`kagome_triangle_xxz_hamiltonian`.
    """
    d_a, d_b, d_c, D_a, D_b, D_c = A_super.shape
    if not (D_a == D_b == D_c):
        raise ValueError(
            "Supersite virtual legs must all share the same bond dimension D; "
            f"got shape {A_super.shape}."
        )
    d_eff = d_a * d_b * d_c
    A_fused = A_super.reshape(d_eff, D_a, D_b, D_c)
    A_rank4 = jnp.moveaxis(A_fused, 0, -1)
    return DenseTensor(A_rank4, indices)


def build_pess_loss(
    hamiltonian: np.ndarray | jnp.ndarray,
    config: CTMConfig,
) -> Callable[[IPESSState], jnp.ndarray]:
    """Build the AD loss closure for kagome iPESS optimization.

    Args:
        hamiltonian: ``(d**3, d**3)`` Hermitian triangle Hamiltonian, e.g.
            from :func:`tenax.algorithms.pess.kagome_triangle_xxz_hamiltonian`.
        config: ``CTMConfig`` controlling chi, max_iter, conv_tol, projector
            method, gauge, and GMRES backward solver. Must use
            ``projector_method="biorthogonal"`` for the kagome path
            (A_u ≠ A_d in general; the ``"eigh"``/``"svd"`` projectors are
            isometric A=B opt-ins).

    Returns:
        ``loss_fn(state: IPESSState) -> jnp.ndarray`` returning the real
        scalar triangle energy per honeycomb unit cell. Differentiable via
        ``jax.grad`` through the implicit-AD CTM.
    """
    H = jnp.asarray(hamiltonian, dtype=jnp.complex128)

    def loss_fn(state: IPESSState) -> jnp.ndarray:
        A_u_super, A_d_super = pess_to_honeycomb_supersites(state)
        D = A_u_super.shape[-1]
        d_eff = A_u_super.shape[0] * A_u_super.shape[1] * A_u_super.shape[2]

        indices = _make_honeycomb_indices(D, d_eff)
        sites = {
            (0, 0): _supersite_to_honeycomb_tensor(A_u_super, indices),
            (1, 0): _supersite_to_honeycomb_tensor(A_d_super, indices),
        }

        return honeycomb_ctm_energy_implicit(
            sites,
            H,
            chi=config.chi,
            max_iter=config.max_iter,
            conv_tol=config.conv_tol,
            projector_method=config.projector_method,
            forward_gauge=config.forward_gauge,
            renormalize=config.renormalize,
            conv_method=config.ctm_conv_method,
            min_iter=config.min_iter,
            chi_ramp=config.chi_ramp,
            energy_fn=compute_honeycomb_triangle_energy,
            gmres_tol=config.gmres_tol,
            gmres_maxiter=config.gmres_maxiter,
            gmres_restart=config.gmres_restart,
            arnoldi_precheck=False,
        )

    return loss_fn
