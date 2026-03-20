"""DMRG3S subspace expansion for 1-site DMRG.

Implements the Hubig-McCulloch-Schollwöck-Wolf (2015) enrichment step:
after each 1-site optimization, expand the site tensor by concatenating
P_i = α · L · M · W (L→R) or P_i = α · R · M · W (R→L), then
SVD-truncate back to the target bond dimension.

Reference: Phys. Rev. B 91, 155115 (2015), arXiv:1501.05504
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.core.tensor import Tensor


def build_expansion_tensor_dense(
    env: Tensor,
    site: Tensor,
    mpo: Tensor,
    alpha: float,
    direction: str,
) -> jax.Array:
    """Build the DMRG3S expansion tensor as a raw JAX array.

    For L→R (env = left environment):
      P[c, x, (e,d)] = α · L[a,b,c] · M[a,p,d] · W[b,p,x,e]
      Output shape: (chi_L_bra, d_phys, w_R * chi_R)

    For R→L (env = right environment):
      P[(b,a), x, f] = α · M[a,p,d] · W[b,p,x,e] · R[d,e,f]
      Output shape: (w_L * chi_L, d_phys, chi_R_bra)

    Args:
        env:       Left environment (L→R) or right environment (R→L).
        site:      Current MPS site tensor (3 legs).
        mpo:       MPO site tensor (4 legs).
        alpha:     Mixing factor.
        direction: ``"left_to_right"`` or ``"right_to_left"``.

    Returns:
        3D JAX array with the expansion directions.
    """
    E = env.todense()
    M = site.todense()
    W = mpo.todense()

    if direction == "left_to_right":
        # L[a,b,c] M[a,p,d] W[b,p,x,e] → P[c,x,e,d]
        P = alpha * jnp.einsum("abc,apd,bpxe->cxed", E, M, W)
        c, x, e, d = P.shape
        return P.reshape(c, x, e * d)
    else:
        # M[a,p,d] W[b,p,x,e] R[d,e,f] → P[b,a,x,f]
        P = alpha * jnp.einsum("apd,bpxe,def->baxf", M, W, E)
        b, a, x, f = P.shape
        return P.reshape(b * a, x, f)
