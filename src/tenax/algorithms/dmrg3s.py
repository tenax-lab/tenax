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
    env: Tensor | jax.Array,
    site: Tensor | jax.Array,
    mpo: Tensor | jax.Array,
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
    E = env.todense() if hasattr(env, "todense") else env
    M = site.todense() if hasattr(site, "todense") else site
    W = mpo.todense() if hasattr(mpo, "todense") else mpo

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


def expand_and_truncate_dense(
    site: jax.Array,
    neighbor: jax.Array,
    env: Tensor,
    mpo: Tensor,
    alpha: float,
    max_bond_dim: int,
    direction: str,
    svd_trunc_err: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Expand site with DMRG3S enrichment, SVD-truncate, absorb into neighbor.

    For L→R:
      M̃ = [M | P] along right bond → SVD → A, remainder
      B̃ = [B; 0] zero-padded → new_B = remainder @ B̃

    For R→L:
      M̃ = [P; M] along left bond → SVD → B, remainder
      Ã = [A | 0] zero-padded → new_A = Ã @ remainder

    Args:
        site:          3D JAX array of the optimized site tensor.
        neighbor:      3D JAX array of the neighbor site tensor.
        env:           Left env (L→R) or right env (R→L).
        mpo:           MPO site tensor.
        alpha:         Mixing factor.
        max_bond_dim:  Maximum bond dimension after truncation.
        direction:     ``"left_to_right"`` or ``"right_to_left"``.
        svd_trunc_err: Optional max truncation error (can reduce bond dim further).

    Returns:
        ``(new_site, new_neighbor)`` as JAX arrays with matching bond dims.
    """
    P = build_expansion_tensor_dense(env, site, mpo, alpha, direction)

    if direction == "left_to_right":
        expanded = jnp.concatenate([site, P], axis=-1)
        chi_l, d, chi_exp = expanded.shape
        mat = expanded.reshape(chi_l * d, chi_exp)
        U, S, Vh = jnp.linalg.svd(mat, full_matrices=False)

        k = _truncation_bond_dim(S, max_bond_dim, svd_trunc_err)
        A = U[:, :k].reshape(chi_l, d, k)
        remainder = jnp.diag(S[:k]) @ Vh[:k, :]

        chi_r_orig = site.shape[-1]
        pad_rows = chi_exp - chi_r_orig
        B_padded = jnp.concatenate(
            [
                neighbor,
                jnp.zeros((pad_rows,) + neighbor.shape[1:], dtype=neighbor.dtype),
            ],
            axis=0,
        )
        new_B = jnp.einsum("ij,jqf->iqf", remainder, B_padded)
        return A, new_B
    else:
        expanded = jnp.concatenate([P, site], axis=0)
        chi_exp, d, chi_r = expanded.shape
        mat = expanded.reshape(chi_exp, d * chi_r)
        U, S, Vh = jnp.linalg.svd(mat, full_matrices=False)

        k = _truncation_bond_dim(S, max_bond_dim, svd_trunc_err)
        B = Vh[:k, :].reshape(k, d, chi_r)
        remainder = U[:, :k] @ jnp.diag(S[:k])

        # Zero-pad neighbor: [0 | A] along right bond.
        # P occupies the leading rows of expanded, site the trailing rows,
        # so the neighbor's right bond aligns with the trailing block.
        chi_l_orig = site.shape[0]
        pad_cols = chi_exp - chi_l_orig
        A_padded = jnp.concatenate(
            [
                jnp.zeros(neighbor.shape[:-1] + (pad_cols,), dtype=neighbor.dtype),
                neighbor,
            ],
            axis=-1,
        )
        new_A = jnp.einsum("apj,jk->apk", A_padded, remainder)
        return B, new_A


def adapt_alpha(
    alpha: float,
    delta_e_opt: float,
    delta_e_trunc: float,
    target_ratio: float = 0.3,
    growth_factor: float = 2.0,
    shrink_factor: float = 0.5,
) -> float:
    """Adapt mixing factor based on optimization/truncation energy balance.

    Target: ``|ΔE_T| ≈ target_ratio · |ΔE_O|``.
    If truncation error too small → increase α.
    If truncation error too large → decrease α.

    Args:
        alpha:         Current mixing factor.
        delta_e_opt:   Energy change from optimization (negative = lowered energy).
        delta_e_trunc: Energy change from truncation (positive = raised energy).
        target_ratio:  Target |ΔE_T|/|ΔE_O| ratio.
        growth_factor: Multiply α by this when truncation error too small.
        shrink_factor: Multiply α by this when truncation error too large.

    Returns:
        Updated mixing factor.
    """
    if abs(delta_e_opt) < 1e-15:
        return alpha
    ratio = abs(delta_e_trunc) / abs(delta_e_opt)
    if ratio < target_ratio * 0.5:
        return alpha * growth_factor
    elif ratio > target_ratio * 2.0:
        return alpha * shrink_factor
    return alpha


def _truncation_bond_dim(
    singular_values: jax.Array, max_bond_dim: int, svd_trunc_err: float | None
) -> int:
    """Determine bond dimension from singular values, respecting both thresholds.

    Also caps at the numerical rank (drops near-zero singular values) to
    avoid inflating the bond dimension with null Schmidt directions.
    """
    # Cap at numerical rank: drop singular values below machine epsilon
    # relative to the largest
    s_max = float(singular_values[0]) if len(singular_values) > 0 else 0.0
    eps = s_max * 1e-14 if s_max > 0 else 1e-14
    rank = int(jnp.sum(singular_values > eps))
    k = min(max_bond_dim, rank, len(singular_values))

    if svd_trunc_err is not None:
        total = float(jnp.sum(singular_values**2))
        if total > 0:
            cumulative_discarded = jnp.cumsum(singular_values[::-1] ** 2)[::-1]
            for j in range(1, len(singular_values)):
                if float(cumulative_discarded[j]) / total < svd_trunc_err**2:
                    k = min(k, j)
                    break
    return max(k, 1)
