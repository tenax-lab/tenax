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

import math
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

__all__ = ["build_pess_loss", "optimize_pess_ad"]


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
        # iPEPS states are projective — only direction matters. Normalize
        # so the un-normalized RDM contraction stays well above f64 zero
        # regardless of how IPESSState was initialized (random init at
        # scale=0.1 gives a supersite of norm ~1e-2, where Tr(ρ_un) ≈ 0).
        A_u_super = A_u_super / (jnp.linalg.norm(A_u_super) + 1e-12)
        A_d_super = A_d_super / (jnp.linalg.norm(A_d_super) + 1e-12)
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


def _tree_real_dot(a, b) -> float:
    """Real part of the Hermitian inner product over a pytree of arrays."""
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return float(
        jnp.real(sum(jnp.sum(jnp.conj(la) * lb) for la, lb in zip(leaves_a, leaves_b)))
    )


def _backtracking_line_search(
    params: dict,
    direction: dict,
    grad: dict,
    energy: float,
    loss_fn: Callable[[dict], jnp.ndarray],
    c1: float = 1e-4,
    rho: float = 0.5,
    max_steps: int = 8,
) -> tuple[dict, float, float]:
    """Armijo backtracking on the L-BFGS descent direction.

    Run at the Python level — the honeycomb CTM convergence loop has
    Python control flow that can't be ``jit``-traced, which rules out
    optax's bundled ``zoom_linesearch``.
    """
    slope = _tree_real_dot(grad, direction)
    if slope >= 0.0:
        direction = jax.tree.map(lambda g: -g, grad)
        slope = -_tree_real_dot(grad, grad)

    p_norm = math.sqrt(max(_tree_real_dot(params, params), 1e-30))
    d_norm = math.sqrt(max(_tree_real_dot(direction, direction), 1e-30))
    alpha = min(1.0, 0.1 * p_norm / d_norm)

    best_trial, best_f, best_alpha = params, energy, 0.0
    for _ in range(max_steps):
        trial = jax.tree.map(lambda p, d: p + alpha * d, params, direction)
        f_trial = float(loss_fn(trial))
        if f_trial < best_f:
            best_trial, best_f, best_alpha = trial, f_trial, alpha
        if f_trial <= energy + c1 * alpha * slope:
            return trial, f_trial, alpha
        alpha *= rho

    return best_trial, best_f, best_alpha


def optimize_pess_ad(
    initial_state: IPESSState,
    hamiltonian: np.ndarray | jnp.ndarray,
    config: CTMConfig,
    *,
    max_iter: int = 50,
    verbose: bool = False,
) -> tuple[IPESSState, float]:
    """L-BFGS optimization of kagome iPESS via the native rank-4 honeycomb CTM.

    Variational parameters are the 5 PESS primitives ``(R_a, R_b, R_c, T_u,
    T_d)``. SU ``lambdas`` are passed through unchanged — they are
    bookkeeping only and don't enter the supersite construction (see
    :func:`pess_to_honeycomb_supersites`).

    Inner step uses ``optax.scale_by_lbfgs`` (memory-10) to produce the
    quasi-Newton direction; line search is a Python-level Armijo
    backtracker since the honeycomb CTM forward pass uses Python control
    flow that can't be ``jit``-traced through ``optax.lbfgs``'s bundled
    zoom search.

    Args:
        initial_state: Starting :class:`IPESSState`. Typically the output of
            :func:`tenax.algorithms.pess.pess_simple_update`, but a freshly
            randomized state also works.
        hamiltonian: ``(d**3, d**3)`` Hermitian triangle Hamiltonian.
        config: ``CTMConfig`` for the inner CTM. Use
            ``projector_method="biorthogonal"`` (kagome supersites are
            non-isometric in general).
        max_iter: Maximum L-BFGS outer iterations.
        verbose: Print energy at each step.

    Returns:
        ``(optimized_state, final_energy)``.
    """
    import optax

    loss_fn_state = build_pess_loss(hamiltonian, config)
    lambdas = initial_state.lambdas

    params = {
        "R_a": initial_state.R_a,
        "R_b": initial_state.R_b,
        "R_c": initial_state.R_c,
        "T_u": initial_state.T_u,
        "T_d": initial_state.T_d,
    }

    def _params_to_state(p: dict) -> IPESSState:
        return IPESSState(
            R_a=p["R_a"],
            R_b=p["R_b"],
            R_c=p["R_c"],
            T_u=p["T_u"],
            T_d=p["T_d"],
            lambdas=lambdas,
        )

    def loss(p: dict) -> jnp.ndarray:
        return loss_fn_state(_params_to_state(p))

    optimizer = optax.chain(
        optax.scale_by_lbfgs(memory_size=10),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(loss)

    last_energy = float(loss(params))
    for step in range(max_iter):
        e_val, grads = grad_fn(params)
        last_energy = float(e_val)
        direction, opt_state = optimizer.update(grads, opt_state, params)
        params, last_energy, alpha = _backtracking_line_search(
            params, direction, grads, last_energy, loss
        )
        if verbose:
            print(
                f"[optimize_pess_ad] step {step + 1}/{max_iter}: "
                f"e = {last_energy:.10f}  alpha = {alpha:.3e}",
                flush=True,
            )
        if alpha == 0.0:
            break

    return _params_to_state(params), last_energy
