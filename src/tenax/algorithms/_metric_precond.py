"""Metric preconditioning for iPEPS AD optimization.

Implements the local tangent-space metric (quantum geometric tensor)
as an implicit matrix-vector product, and GMRES-based preconditioning
for CG and L-BFGS optimizers.

Reference: Rader et al., arXiv:2511.09546
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.contraction.contractor import contract

if TYPE_CHECKING:
    from tenax.algorithms.ipeps_config import iPEPSConfig
    from tenax.core.tensor import Tensor

# ---------------------------------------------------------------------------
# 1. Single-site environment contraction
# ---------------------------------------------------------------------------


def _contract_single_site_environment(env: CTMTensorEnv) -> jnp.ndarray:
    """Contract the 8 CTM tensors into E_{u2, d2, l2, r2}.

    Contracts C1-T1-C2-T2-C3-T3-C4-T4 around the central site,
    leaving the 4 D²-dimensional legs that connect to the site open.

    Returns:
        Dense array of shape (D², D², D², D²).
    """
    # Top row: C1·T1·C2
    C1 = env.C1.relabel("c1_r", "t1_l")
    top_left = contract(C1, env.T1)  # (c1_d, u2, t1_r)
    C2 = env.C2.relabel("c2_l", "t1_r")
    top = contract(top_left, C2)  # (c1_d, u2, c2_d)

    # Bottom row: C4·T3·C3
    C4 = env.C4.relabel("c4_u", "t3_r")
    bot_left = contract(C4, env.T3)  # (c4_r, d2, t3_l)
    C3 = env.C3.relabel("c3_l", "t3_l")
    bot = contract(bot_left, C3)  # (c4_r, d2, c3_u)

    # Left side: top · T4   (contract c1_d ↔ t4_d, leave c4_r for bot)
    T4 = env.T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    left = contract(top, T4)  # (u2, c2_d, l2, c4_r)

    # Right side: T2 relabelled to match top and bot chi bonds
    T2 = env.T2.relabels({"t2_u": "c2_d", "t2_d": "c3_u"})

    # Combine left with T2  (contract c2_d)
    left_right = contract(left, T2)  # (u2, l2, c4_r, r2, c3_u)

    # Close with bottom  (contract c4_r and c3_u)
    E_tensor = contract(left_right, bot, output_labels=["u2", "d2", "l2", "r2"])
    return E_tensor.todense()


# ---------------------------------------------------------------------------
# 2. Norm environment matvec
# ---------------------------------------------------------------------------


def norm_environment_matvec(
    A: Tensor,
    env: CTMTensorEnv,
    v: Tensor,
) -> jnp.ndarray:
    """Apply the single-site norm environment metric to vector v.

    Computes ``(N·v)_{u',d',l',r',s} = E_{(u,u'),(d,d'),(l,l'),(r,r')} v_{u,d,l,r,s}``
    where E is the CTM environment contracted around the central site.

    The physical index passes through unchanged (delta_{s,s'} under the metric).

    Fusing convention: D² = ket(slow) * D + bra(fast), matching
    ``_build_double_layer_tensor`` in ``_ctm_tensor_init.py``.

    Args:
        A: iPEPS site tensor (used only for shape metadata).
        env: Converged CTM environment.
        v: Input vector, same type and shape as A.

    Returns:
        Dense array of shape ``(D, D, D, D, d)``.
    """
    E = _contract_single_site_environment(env)
    v_dense = v.todense()
    D = v_dense.shape[0]
    d = v_dense.shape[-1]
    # Nv[A,B,C,D,s] = sum_{a,b,c,d} E[aA, bB, cC, dD] * v[a,b,c,d,s]
    #
    # Reshape E from (D², D², D², D²) to (D^4, D^4) matrix mapping
    # ket indices to bra indices, then apply as matmul.  This avoids
    # 8-dim einsum which crashes XLA autotuning on GPU for small D.
    #
    # E[i,j,k,l] with i=a*D+A (ket-slow fusing).  We need:
    #   E_mat[A*D³+B*D²+C*D+D_, a*D³+b*D²+c*D+d_]
    # = E[a*D+A, b*D+B, c*D+C, d_*D+D_]
    #
    # Reshape to (D,D, D,D, D,D, D,D), transpose bra indices out,
    # then flatten.  The 8-dim intermediate is fine as a reshape+transpose
    # (no einsum/fusion needed — XLA handles permutations natively).
    E_mat = (
        E.reshape(D, D, D, D, D, D, D, D)
        .transpose(1, 3, 5, 7, 0, 2, 4, 6)
        .reshape(D**4, D**4)
    )
    v_mat = v_dense.reshape(D**4, d)
    Nv_mat = E_mat @ v_mat  # (D^4, d)
    return Nv_mat.reshape(D, D, D, D, d)


# ---------------------------------------------------------------------------
# 3. GMRES-based gradient preconditioning
# ---------------------------------------------------------------------------


def precondition_gradient(
    A: Tensor,
    env: CTMTensorEnv,
    grad: Tensor,
    delta: float,
    config: iPEPSConfig,
) -> jnp.ndarray:
    """Solve ``(N + delta*I) g' = g`` via GMRES.

    Args:
        A: Current iPEPS site tensor.
        env: Converged CTM environment (from energy step).
        grad: Energy gradient w.r.t. A (same Tensor type).
        delta: Regularization parameter (|dE| or ||g||^2).
        config: iPEPS config with GMRES settings.

    Returns:
        Preconditioned gradient as dense array, shape ``(D, D, D, D, d)``.
    """
    E = _contract_single_site_environment(env)
    g_dense = grad.todense()
    D = g_dense.shape[0]
    d = g_dense.shape[-1]
    g_flat = g_dense.reshape(-1)

    # Build (D^4, D^4) metric matrix, avoiding 8-dim einsum that crashes
    # XLA autotuning on GPU.  Reshape+transpose is handled natively by XLA.
    E_mat = (
        E.reshape(D, D, D, D, D, D, D, D)
        .transpose(1, 3, 5, 7, 0, 2, 4, 6)
        .reshape(D**4, D**4)
    )

    def matvec(v_flat):
        v_mat = v_flat.reshape(D**4, d)
        Nv_mat = E_mat @ v_mat
        return Nv_mat.reshape(-1) + delta * v_flat

    g_precond, _ = jax_gmres(
        matvec,
        g_flat,
        x0=g_flat,
        tol=config.metric_gmres_tol,
        maxiter=config.metric_gmres_maxiter,
    )
    return g_precond.reshape(D, D, D, D, d)


def precondition_gradient_multisite(
    site_tensors: dict,
    envs: dict,
    grads: dict,
    delta: float,
    config: iPEPSConfig,
) -> dict:
    """Apply metric preconditioning independently per site.

    The metric is block-diagonal in site index (local approximation),
    so each site's gradient is preconditioned using its own environment.

    Args:
        site_tensors: ``{(r, c): Tensor}`` current site tensors.
        envs: ``{(r, c): CTMTensorEnv}`` converged environments.
        grads: ``{(r, c): Tensor}`` gradients per site.
        delta: Regularization parameter.
        config: iPEPS config.

    Returns:
        ``{(r, c): jnp.ndarray}`` preconditioned gradients per site.
    """
    return {
        key: precondition_gradient(
            site_tensors[key], envs[key], grads[key], delta, config
        )
        for key in grads
    }


# ---------------------------------------------------------------------------
# 4. L-BFGS two-loop recursion (Nocedal Algorithm 7.4)
# ---------------------------------------------------------------------------


def lbfgs_two_loop(
    grad: jnp.ndarray,
    history: list[tuple[jnp.ndarray, jnp.ndarray, float]],
    h0_matvec: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """L-BFGS two-loop recursion (Nocedal Algorithm 7.4).

    Computes ``H_k @ grad`` where ``H_k`` is the L-BFGS approximation to the
    inverse Hessian, with ``h0_matvec`` providing the initial ``H_0``.

    Args:
        grad: Gradient vector (flat 1D array).
        history: List of ``(s_k, y_k, rho_k)`` tuples, oldest first.
            ``s_k = x_{k+1} - x_k``, ``y_k = g_{k+1} - g_k``,
            ``rho_k = 1 / (y_k . s_k)``.
        h0_matvec: Function applying the initial inverse Hessian ``H_0``.
            Standard L-BFGS: ``lambda v: gamma * v``.
            Metric-preconditioned: GMRES solve of ``(N + delta*I) v = q``.

    Returns:
        L-BFGS direction ``H_k @ grad`` (flat 1D array).
    """
    q = grad.copy()
    alphas = []

    # First loop: newest to oldest
    for s, y, rho in reversed(history):
        alpha = rho * jnp.vdot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
    alphas.reverse()

    # Apply initial inverse Hessian
    r = h0_matvec(q)

    # Second loop: oldest to newest
    for i, (s, y, rho) in enumerate(history):
        beta = rho * jnp.vdot(y, r)
        r = r + s * (alphas[i] - beta)

    return r
