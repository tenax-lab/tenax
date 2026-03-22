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
import numpy as np

from tenax.core.index import TensorIndex
from tenax.core.tensor import SymmetricTensor, Tensor


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


def expand_and_truncate_hybrid(
    site: jax.Array,
    neighbor: jax.Array,
    env: Tensor,
    mpo: Tensor,
    max_bond_dim: int,
    direction: str,
    num_extra: int = 0,
    hybrid: bool = True,
    svd_trunc_err: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Improved post-expansion: null-space projection + adaptive mixing.

    Based on the MPToolkit algorithm (McCulloch):
      1. Build expansion P = env·M·W (full strength, no α scaling)
      2. SVD of site → kept space + null space
      3. Project P into the null space of the kept left singular vectors
      4. QR-orthogonalize the null-space projection
      5. Hybrid mode: scale by √(ε_trunc)/‖P_null‖, concatenate with site,
         re-SVD → keep χ states (smoothly mixed)
      6. Direct mode: keep χ + Δ states (extra expansion directions seeded
         with zero amplitude, optimized in next sweep)

    Args:
        site:          3D JAX array of the optimized site tensor.
        neighbor:      3D JAX array of the neighbor site tensor.
        env:           Left env (L→R) or right env (R→L).
        mpo:           MPO site tensor.
        max_bond_dim:  Maximum kept bond dimension χ.
        direction:     ``"left_to_right"`` or ``"right_to_left"``.
        num_extra:     Number of extra expansion states Δ. 0 = auto (10% of χ).
        hybrid:        If True, use √(ε_trunc) scaling + re-SVD (smooth mixing).
                       If False, keep χ + Δ states (direct expansion).
        svd_trunc_err: Optional max truncation error threshold.

    Returns:
        ``(new_site, new_neighbor)`` as JAX arrays with matching bond dims.
    """
    # Build expansion tensor at full strength (no α scaling)
    P = build_expansion_tensor_dense(env, site, mpo, alpha=1.0, direction=direction)

    if num_extra <= 0:
        num_extra = max(1, max_bond_dim // 10)

    if direction == "left_to_right":
        return _hybrid_ltr(
            site, neighbor, P, max_bond_dim, num_extra, hybrid, svd_trunc_err
        )
    else:
        return _hybrid_rtl(
            site, neighbor, P, max_bond_dim, num_extra, hybrid, svd_trunc_err
        )


def _hybrid_ltr(site, neighbor, P, max_bond_dim, num_extra, hybrid, svd_trunc_err):
    """L→R hybrid expansion: expand right bond."""
    chi_l, d, chi_r = site.shape
    M_mat = site.reshape(chi_l * d, chi_r)

    # Full SVD of site tensor
    U, S, Vh = jnp.linalg.svd(M_mat, full_matrices=False)

    # Truncate to χ, compute truncation error
    k = _truncation_bond_dim(S, max_bond_dim, svd_trunc_err)
    total_sq = float(jnp.sum(S**2))
    trunc_sq = float(jnp.sum(S[k:] ** 2)) if len(S) > k else 0.0
    trunc_err = trunc_sq / total_sq if total_sq > 0 else 0.0

    # Project expansion into null space of kept left singular vectors
    UKeep = U[:, :k]
    P_mat = P.reshape(chi_l * d, -1)
    P_null = P_mat - UKeep @ (UKeep.conj().T @ P_mat)

    # QR-orthogonalize the null-space projection
    Q_null, _ = jnp.linalg.qr(P_null)

    # Cap extra states at available null-space rank
    n_extra = min(num_extra, Q_null.shape[1])
    s_max_q = float(jnp.linalg.norm(Q_null[:, 0])) if Q_null.shape[1] > 0 else 0.0
    # Only keep columns with nonzero norm (numerical rank of null-space projection)
    if Q_null.shape[1] > 0:
        col_norms = jnp.linalg.norm(Q_null, axis=0)
        n_rank = int(jnp.sum(col_norms > s_max_q * 1e-12))
        n_extra = min(n_extra, n_rank)
    Q_extra = Q_null[:, :n_extra]

    if hybrid:
        # Scale expansion by sqrt(truncation error) — adaptive mixing
        norm_extra = float(jnp.linalg.norm(Q_extra))
        if norm_extra > 1e-15 and trunc_err > 0:
            Q_scaled = Q_extra * (jnp.sqrt(trunc_err) / norm_extra)
        else:
            Q_scaled = Q_extra * 0.0

        # Concatenate [M | Q_scaled] and re-SVD to smoothly mix
        expanded = jnp.concatenate([M_mat, Q_scaled], axis=1)
        U2, S2, Vh2 = jnp.linalg.svd(expanded, full_matrices=False)
        k2 = _truncation_bond_dim(S2, max_bond_dim, svd_trunc_err)
        A = U2[:, :k2].reshape(chi_l, d, k2)
        remainder = jnp.diag(S2[:k2]) @ Vh2[:k2, :]

        # Zero-pad neighbor to match expanded right bond
        pad_rows = expanded.shape[1] - chi_r
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
        # Direct expansion: keep χ + Δ states
        # The extra directions are seeded from the null-space projection
        U_expanded = jnp.concatenate([UKeep, Q_extra], axis=1)
        total_k = k + n_extra
        A = U_expanded.reshape(chi_l, d, total_k)

        # Project site into expanded basis → remainder for neighbor
        # Kept columns: U_Keep.T @ M_mat = diag(S[:k]) @ Vh[:k,:]
        # Extra columns: Q_extra.T @ M_mat ≈ 0 (null space)
        remainder = U_expanded.conj().T @ M_mat  # (total_k, chi_r)
        new_B = jnp.einsum("ij,jqf->iqf", remainder, neighbor)
        return A, new_B


def _hybrid_rtl(site, neighbor, P, max_bond_dim, num_extra, hybrid, svd_trunc_err):
    """R→L hybrid expansion: expand left bond."""
    chi_l, d, chi_r = site.shape
    M_mat = site.reshape(chi_l, d * chi_r)

    # Full SVD of site tensor
    U, S, Vh = jnp.linalg.svd(M_mat, full_matrices=False)

    # Truncate, compute truncation error
    k = _truncation_bond_dim(S, max_bond_dim, svd_trunc_err)
    total_sq = float(jnp.sum(S**2))
    trunc_sq = float(jnp.sum(S[k:] ** 2)) if len(S) > k else 0.0
    trunc_err = trunc_sq / total_sq if total_sq > 0 else 0.0

    # Project expansion into null space of kept right singular vectors
    VhKeep = Vh[:k, :]
    P_mat = P.reshape(-1, d * chi_r)
    P_null = P_mat - P_mat @ VhKeep.conj().T @ VhKeep

    # QR-orthogonalize (transpose to get row-space orthogonalization)
    Q_null_T, _ = jnp.linalg.qr(P_null.T)
    Q_null = Q_null_T.T  # rows are orthonormal expansion directions

    n_extra = min(num_extra, Q_null.shape[0])
    if Q_null.shape[0] > 0:
        row_norms = jnp.linalg.norm(Q_null, axis=1)
        r_max = float(row_norms[0])
        n_rank = int(jnp.sum(row_norms > r_max * 1e-12)) if r_max > 0 else 0
        n_extra = min(n_extra, n_rank)
    Q_extra = Q_null[:n_extra, :]

    if hybrid:
        norm_extra = float(jnp.linalg.norm(Q_extra))
        if norm_extra > 1e-15 and trunc_err > 0:
            Q_scaled = Q_extra * (jnp.sqrt(trunc_err) / norm_extra)
        else:
            Q_scaled = Q_extra * 0.0

        # Concatenate [Q_scaled; M] along left bond (rows) and re-SVD
        expanded = jnp.concatenate([Q_scaled, M_mat], axis=0)
        U2, S2, Vh2 = jnp.linalg.svd(expanded, full_matrices=False)
        k2 = _truncation_bond_dim(S2, max_bond_dim, svd_trunc_err)
        B = Vh2[:k2, :].reshape(k2, d, chi_r)
        remainder = U2[:, :k2] @ jnp.diag(S2[:k2])

        # Zero-pad neighbor: [0 | A] along right bond
        pad_cols = expanded.shape[0] - chi_l
        A_padded = jnp.concatenate(
            [
                jnp.zeros(neighbor.shape[:-1] + (pad_cols,), dtype=neighbor.dtype),
                neighbor,
            ],
            axis=-1,
        )
        new_A = jnp.einsum("apj,jk->apk", A_padded, remainder)
        return B, new_A
    else:
        # Direct expansion: keep χ + Δ states
        Vh_expanded = jnp.concatenate([VhKeep, Q_extra], axis=0)
        total_k = k + n_extra
        B = Vh_expanded.reshape(total_k, d, chi_r)

        remainder = M_mat @ Vh_expanded.conj().T  # (chi_l, total_k)
        new_A = jnp.einsum("apj,jk->apk", neighbor, remainder)
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


# ------------------------------------------------------------------ #
# Symmetric (SymmetricTensor) DMRG3S expansion                        #
# ------------------------------------------------------------------ #


def expand_and_truncate_symmetric(
    site: SymmetricTensor,
    neighbor: SymmetricTensor,
    env: SymmetricTensor,
    mpo: SymmetricTensor,
    alpha: float,
    max_bond_dim: int,
    direction: str,
    svd_trunc_err: float | None = None,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Expand and truncate for SymmetricTensor with charge-preserving SVD.

    Pipeline:
      1. Build 4-leg expansion tensor via ``_blockwise_contract``
      2. Fuse the two uncontracted legs via ``fuse_indices``
      3. Concatenate ``[M | P]`` as a SymmetricTensor (dense round-trip on
         the small site-sized tensors, with correct charge arrays)
      4. Block-sparse SVD via ``tenax.linalg.svd`` — preserves charge sectors
      5. Absorb singular values into the neighbor via ``contract``

    The ``todense()`` calls are only on site-sized tensors (bounded by
    ``max_bond_dim``), not on environments.
    """
    from tenax.algorithms._tensor_utils import fuse_indices
    from tenax.algorithms.dmrg import _blockwise_contract
    from tenax.contraction.contractor import contract
    from tenax.linalg import svd as _linalg_svd

    if direction == "left_to_right":
        return _expand_truncate_sym_ltr(
            site,
            neighbor,
            env,
            mpo,
            alpha,
            max_bond_dim,
            svd_trunc_err,
            fuse_indices,
            _blockwise_contract,
            contract,
            _linalg_svd,
        )
    else:
        return _expand_truncate_sym_rtl(
            site,
            neighbor,
            env,
            mpo,
            alpha,
            max_bond_dim,
            svd_trunc_err,
            fuse_indices,
            _blockwise_contract,
            contract,
            _linalg_svd,
        )


def _expand_truncate_sym_ltr(
    site,
    neighbor,
    env,
    mpo,
    alpha,
    max_bond_dim,
    svd_trunc_err,
    fuse_indices,
    _blockwise_contract,
    contract,
    _linalg_svd,
):
    """L→R symmetric expansion: expand right bond of site."""
    # Step 1: Build 4-leg expansion tensor P[c, x, e, d]
    # Using the 1-site matvec einsum minus right env:
    #   L[a,b,c] M[a,p,d] W[b,p,x,e] → P[c,x,e,d]
    output_indices_4leg = (
        env.indices[2],  # c: virt_bra from left env
        mpo.indices[2],  # x: phys_bra from MPO
        mpo.indices[3],  # e: mpo_right
        site.indices[2],  # d: virt_right from site
    )
    P4 = _blockwise_contract(
        [env, site, mpo],
        "abc,apd,bpxe->cxed",
        output_indices=output_indices_4leg,
    )
    P4 = P4 * alpha

    # Step 2: Fuse (e, d) → single expanded right bond
    bond_label = site.indices[2].label
    bond_flow = site.indices[2].flow
    P3 = fuse_indices(P4, 2, 3, bond_label, bond_flow)

    # Step 3: Concatenate [M | P] along right bond as SymmetricTensor
    M_dense = site.todense()
    P_dense = P3.todense()
    expanded_dense = jnp.concatenate([M_dense, P_dense], axis=-1)

    # Build combined right-bond charges
    orig_charges = np.asarray(site.indices[2].charges)
    exp_charges = np.asarray(P3.indices[2].charges)
    combined_charges = np.concatenate([orig_charges, exp_charges])
    combined_right_idx = TensorIndex(
        symmetry=site.indices[2].symmetry,
        charges=combined_charges,
        flow=site.indices[2].flow,
        label=site.indices[2].label,
    )
    expanded_indices = site.indices[:2] + (combined_right_idx,)
    expanded = SymmetricTensor.from_dense(
        expanded_dense, expanded_indices, tol=float("inf")
    )

    # Step 4: Block-sparse SVD
    left_labels = [site.indices[0].label, site.indices[1].label]
    right_labels = [combined_right_idx.label]
    svd_bond = "_dmrg3s_bond"
    A, s, Vh, _ = _linalg_svd(
        expanded,
        left_labels,
        right_labels,
        new_bond_label=svd_bond,
        max_singular_values=max_bond_dim,
    )

    # Step 5: Absorb s into Vh directly (scale rows by singular values)
    # Avoid label collision from s_diag by absorbing s into Vh's dense data
    Vh_data = Vh.todense()
    sVh_data = (
        s[:, None] * Vh_data
        if Vh_data.ndim == 2
        else jnp.einsum("i,i...->i...", s, Vh_data)
    )
    sVh = SymmetricTensor.from_dense(sVh_data, Vh.indices, tol=float("inf"))

    # Zero-pad neighbor's left bond to match expanded right bond,
    # then contract with sVh
    neigh_dense = neighbor.todense()
    chi_r_orig = site.todense().shape[-1]
    pad_rows = expanded_dense.shape[-1] - chi_r_orig
    B_padded_dense = jnp.concatenate(
        [
            neigh_dense,
            jnp.zeros((pad_rows,) + neigh_dense.shape[1:], dtype=neigh_dense.dtype),
        ],
        axis=0,
    )
    # Build SymmetricTensor for padded neighbor
    padded_left_idx = TensorIndex(
        symmetry=combined_right_idx.symmetry,
        charges=combined_charges,
        flow=neighbor.indices[0].flow,
        label=neighbor.indices[0].label,
    )
    padded_neigh_indices = (padded_left_idx,) + neighbor.indices[1:]
    B_padded = SymmetricTensor.from_dense(
        B_padded_dense, padded_neigh_indices, tol=float("inf")
    )

    new_neighbor = contract(sVh, B_padded)

    # Relabel the SVD bond to the original MPS bond label
    orig_bond_label = site.indices[2].label
    A = A.relabel(svd_bond, orig_bond_label)
    new_neighbor = new_neighbor.relabel(svd_bond, neighbor.indices[0].label)

    return A, new_neighbor


def _expand_truncate_sym_rtl(
    site,
    neighbor,
    env,
    mpo,
    alpha,
    max_bond_dim,
    svd_trunc_err,
    fuse_indices,
    _blockwise_contract,
    contract,
    _linalg_svd,
):
    """R→L symmetric expansion: expand left bond of site."""
    # Step 1: Build 4-leg expansion tensor P[b, a, x, f]
    #   M[a,p,d] W[b,p,x,e] R[d,e,f] → P[b,a,x,f]
    output_indices_4leg = (
        mpo.indices[0],  # b: mpo_left
        site.indices[0],  # a: virt_left from site
        mpo.indices[2],  # x: phys_bra from MPO
        env.indices[2],  # f: virt_bra from right env
    )
    P4 = _blockwise_contract(
        [site, mpo, env],
        "apd,bpxe,def->baxf",
        output_indices=output_indices_4leg,
    )
    P4 = P4 * alpha

    # Step 2: Fuse (b, a) → single expanded left bond
    bond_label = site.indices[0].label
    bond_flow = site.indices[0].flow
    P3 = fuse_indices(P4, 0, 1, bond_label, bond_flow)

    # Step 3: Concatenate [P; M] along left bond
    P_dense = P3.todense()
    M_dense = site.todense()
    expanded_dense = jnp.concatenate([P_dense, M_dense], axis=0)

    orig_charges = np.asarray(site.indices[0].charges)
    exp_charges = np.asarray(P3.indices[0].charges)
    combined_charges = np.concatenate([exp_charges, orig_charges])
    combined_left_idx = TensorIndex(
        symmetry=site.indices[0].symmetry,
        charges=combined_charges,
        flow=site.indices[0].flow,
        label=site.indices[0].label,
    )
    expanded_indices = (combined_left_idx,) + site.indices[1:]
    expanded = SymmetricTensor.from_dense(
        expanded_dense, expanded_indices, tol=float("inf")
    )

    # Step 4: Block-sparse SVD
    left_labels = [combined_left_idx.label]
    right_labels = [site.indices[1].label, site.indices[2].label]
    svd_bond = "_dmrg3s_bond"
    U, s, B, _ = _linalg_svd(
        expanded,
        left_labels,
        right_labels,
        new_bond_label=svd_bond,
        max_singular_values=max_bond_dim,
    )

    # Step 5: Absorb s into U directly (scale columns by singular values)
    U_data = U.todense()
    Us_data = (
        U_data * s[None, :]
        if U_data.ndim == 2
        else jnp.einsum("...i,i->...i", U_data, s)
    )
    Us = SymmetricTensor.from_dense(Us_data, U.indices, tol=float("inf"))

    # Zero-pad neighbor's right bond: [0 | A]
    neigh_dense = neighbor.todense()
    chi_l_orig = site.todense().shape[0]
    pad_cols = expanded_dense.shape[0] - chi_l_orig
    A_padded_dense = jnp.concatenate(
        [
            jnp.zeros(neigh_dense.shape[:-1] + (pad_cols,), dtype=neigh_dense.dtype),
            neigh_dense,
        ],
        axis=-1,
    )
    padded_right_idx = TensorIndex(
        symmetry=combined_left_idx.symmetry,
        charges=combined_charges,
        flow=neighbor.indices[-1].flow,
        label=neighbor.indices[-1].label,
    )
    padded_neigh_indices = neighbor.indices[:-1] + (padded_right_idx,)
    A_padded = SymmetricTensor.from_dense(
        A_padded_dense, padded_neigh_indices, tol=float("inf")
    )

    new_neighbor = contract(A_padded, Us)

    # Relabel
    orig_bond_label = site.indices[0].label
    B = B.relabel(svd_bond, orig_bond_label)
    new_neighbor = new_neighbor.relabel(svd_bond, neighbor.indices[-1].label)

    return B, new_neighbor
