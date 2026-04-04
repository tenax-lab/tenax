"""Metric preconditioning (natural gradient) for iPEPS optimization.

Implements the norm-environment matvec needed for natural-gradient
preconditioning of CG/L-BFGS iPEPS optimization
(Rader et al., arXiv:2511.09546).
"""

from __future__ import annotations

__all__ = [
    "_contract_single_site_environment",
    "norm_environment_matvec",
    "precondition_gradient",
    "precondition_gradient_multisite",
    "lbfgs_two_loop",
]

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_sparse

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.contraction.contractor import contract
from tenax.core.tensor import Tensor

if TYPE_CHECKING:
    from tenax.algorithms.ipeps_config import iPEPSConfig


def _contract_single_site_environment(env: CTMTensorEnv) -> jax.Array:
    """Contract the 8 CTM boundary tensors into a single-site environment.

    Contracts the ring C1-T1-C2-T2-C3-T3-C4-T4, leaving the four D^2
    open legs (u2, d2, l2, r2) that couple to the central double-layer
    site tensor.

    Returns:
        Dense JAX array of shape ``(D^2, D^2, D^2, D^2)`` with index
        order ``(u2, d2, l2, r2)``.
    """
    # Step 1: Top row  C1 . T1 . C2  ->  (c1_d, u2, c2_d)
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    C1_T1 = contract(C1, env.T1)  # (c1_d, u2, t1_r)
    top = contract(C1_T1, C2)  # (c1_d, u2, c2_d)

    # Step 2: Bottom row  C4 . T3 . C3  ->  (c4_r, d2, c3_u)
    C4 = env.C4.relabel("c4_u", "t3_r")
    C3 = env.C3.relabel("c3_l", "t3_l")
    C4_T3 = contract(C4, env.T3)  # (c4_r, d2, t3_l)
    bot = contract(C4_T3, C3)  # (c4_r, d2, c3_u)

    # Step 3: Attach T4 to top  (contract c1_d <-> t4_d)
    T4 = env.T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    top_left = contract(top, T4)  # (u2, c2_d, l2, c4_r)

    # Step 4: Attach T2 to top_left  (contract c2_d <-> t2_u)
    T2 = env.T2.relabels({"t2_u": "c2_d", "t2_d": "c3_u"})
    top_lr = contract(top_left, T2)  # (u2, l2, c4_r, r2, c3_u)

    # Step 5: Close with bottom row  (contract c4_r, c3_u)
    E = contract(top_lr, bot, output_labels=["u2", "d2", "l2", "r2"])

    return E.todense()


def norm_environment_matvec(
    A: Tensor,
    env: CTMTensorEnv,
    v: jax.Array,
) -> jax.Array:
    """Apply the norm environment (metric) to a tangent vector.

    Computes ``(N . v)_{u',d',l',r',s} = E_{(u,u'),(d,d'),(l,l'),(r,r')} * v_{u,d,l,r,s}``

    where E is the single-site environment with D^2-dimensional legs
    (fused ket x bra), and contraction is over the ket indices.

    The fusing convention follows ``_build_double_layer_tensor``:
    ket = slow (axis_a), bra = fast (axis_b).

    Args:
        A:   iPEPS site tensor (used only to infer D).
        env: Converged CTMTensorEnv.
        v:   Dense tangent vector of shape ``(D, D, D, D, d)``.

    Returns:
        Dense array of shape ``(D, D, D, D, d)``.
    """
    D = A.indices[0].dim

    # E has shape (D^2, D^2, D^2, D^2) with index order (u2, d2, l2, r2)
    E = _contract_single_site_environment(env)

    # Unfuse: (D_ket, D_bra, D_ket, D_bra, D_ket, D_bra, D_ket, D_bra)
    E8 = E.reshape(D, D, D, D, D, D, D, D)

    # Contract ket indices of E with v, output bra indices
    # E8 indices: u_ket(a), u_bra(A), d_ket(b), d_bra(B), l_ket(c), l_bra(C), r_ket(d), r_bra(D)
    # v indices:  u_ket(a), d_ket(b), l_ket(c), r_ket(d), phys(s)
    return jnp.einsum("aAbBcCdD,abcds->ABCDs", E8, v)


def precondition_gradient(
    A: Tensor,
    env: CTMTensorEnv,
    grad: jax.Array,
    delta: float,
    config: iPEPSConfig,
) -> jax.Array:
    """Precondition the energy gradient using the norm environment metric.

    Solves ``(N + delta * I) g' = g`` via GMRES, where N is the norm
    environment (metric) and delta is a regularization parameter.

    Args:
        A:      iPEPS site tensor (used to infer D).
        env:    Converged CTM environment.
        grad:   Energy gradient, dense array of shape ``(D, D, D, D, d)``.
        delta:  Regularization parameter (positive scalar).
        config: iPEPS config with ``metric_gmres_tol`` and
                ``metric_gmres_maxiter``.

    Returns:
        Preconditioned gradient, dense array of shape ``(D, D, D, D, d)``.
    """
    D = A.indices[0].dim
    shape = grad.shape

    # Contract environment once and unfuse for reuse in the closure
    E = _contract_single_site_environment(env)
    E8 = E.reshape(D, D, D, D, D, D, D, D)

    g_flat = grad.ravel()

    def matvec(v_flat: jax.Array) -> jax.Array:
        v = v_flat.reshape(shape)
        Nv = jnp.einsum("aAbBcCdD,abcds->ABCDs", E8, v)
        return Nv.ravel() + delta * v_flat

    g_precond_flat, _ = jax_sparse.gmres(
        matvec,
        g_flat,
        x0=g_flat,
        tol=config.metric_gmres_tol,
        maxiter=config.metric_gmres_maxiter,
    )
    return g_precond_flat.reshape(shape)


def precondition_gradient_multisite(
    site_tensors: dict,
    envs: dict,
    grads: dict,
    delta: float,
    config: iPEPSConfig,
) -> dict:
    """Apply metric preconditioning independently per site.

    The local metric is block-diagonal in site index, so each site's
    gradient is preconditioned using only its own environment.

    Args:
        site_tensors: ``{(r, c): Tensor}`` current site tensors.
        envs: ``{(r, c): CTMTensorEnv}`` converged environments.
        grads: ``{(r, c): jax.Array}`` dense gradients per site.
        delta: Regularization parameter.
        config: iPEPS config with GMRES settings.

    Returns:
        ``{(r, c): jax.Array}`` preconditioned gradients per site.
    """
    return {
        key: precondition_gradient(
            site_tensors[key], envs[key], grads[key], delta, config
        )
        for key in grads
    }


def lbfgs_two_loop(
    grad: jax.Array,
    history: list[tuple[jax.Array, jax.Array, float]],
    h0_matvec: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """L-BFGS two-loop recursion (Nocedal Algorithm 7.4).

    Computes the L-BFGS search direction ``H_k @ grad`` using the stored
    history of ``(s_k, y_k, rho_k)`` pairs and an initial inverse-Hessian
    approximation ``h0_matvec``.

    Args:
        grad:       Flat 1-D gradient vector.
        history:    List of ``(s_k, y_k, rho_k)`` tuples, oldest first.
        h0_matvec:  Callable that applies the initial inverse Hessian H0
                    to a flat vector.

    Returns:
        The L-BFGS direction ``H_k @ grad`` (flat 1-D array).
    """
    q = grad  # JAX arrays are immutable, no copy needed
    alphas = []

    # First loop: newest to oldest
    for s, y, rho in reversed(history):
        alpha = rho * jnp.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
    alphas.reverse()

    # Apply initial inverse Hessian
    r = h0_matvec(q)

    # Second loop: oldest to newest
    for i, (s, y, rho) in enumerate(history):
        beta = rho * jnp.dot(y, r)
        r = r + s * (alphas[i] - beta)

    return r
