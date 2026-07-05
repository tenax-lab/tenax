"""Stable automatic differentiation utilities for iPEPS.

Implements the solutions from Francuz et al., Phys. Rev. Research 7, 013237
(2025) for stable AD through CTM:

1. Custom truncated SVD with Lorentzian regularization for degenerate singular
   values and the full truncation correction term.
2. CTM fixed-point implicit differentiation (avoids storing all CTM iterations).
3. Gauge fixing for element-wise CTM convergence.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

from tenax.algorithms._ad_primitives import (
    CTMRGGradientError as CTMRGGradientError,
)
from tenax.algorithms._ad_primitives import (
    _fix_svd_signs as _fix_svd_signs,
)
from tenax.algorithms._ad_primitives import (
    _svd_sector_backward as _svd_sector_backward,
)
from tenax.algorithms._ad_primitives import (
    regularized_eigh as regularized_eigh,
)
from tenax.algorithms._ad_primitives import (
    regularized_svd as regularized_svd,
)
from tenax.algorithms._ad_primitives import (
    truncated_svd_ad as truncated_svd_ad,
)
from tenax.algorithms._ad_primitives import (
    truncated_svd_ad_vh_only as truncated_svd_ad_vh_only,
)
from tenax.algorithms._ad_primitives import (
    truncated_svd_symmetric_ad as truncated_svd_symmetric_ad,
)
from tenax.algorithms._ctm_tensor import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    _ctm_tensor_sweep_multisite,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor import (
    _ctm_sv_diff as _ctm_sv_diff_tensor,
)
from tenax.algorithms._split_ctm_tensor import (
    _split_ctm_tensor_sweep,
    ctm_split_tensor,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.linalg import _dense_svd

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2. Config tuple helpers (shared by all CTM AD paths)
# ---------------------------------------------------------------------------


_PM_STR_TO_INT = {"eigh": 0, "qr": 1, "svd": 2}
_PM_INT_TO_STR = {0: "eigh", 1: "qr", 2: "svd"}


_CONV_METHOD_STR_TO_INT = {"sv": 0, "elementwise": 1}
_CONV_METHOD_INT_TO_STR = {0: "sv", 1: "elementwise"}


_PB_STR_TO_INT = {"auto": 0, "standard": 1, "lorentzian": 2}
_PB_INT_TO_STR = {0: "auto", 1: "standard", 2: "lorentzian"}


def _config_to_tuple(config) -> tuple:
    """Pack CTMConfig into a hashable tuple for JAX tracing."""
    return (
        config.chi,
        config.max_iter,
        config.conv_tol,
        int(config.renormalize),
        _PM_STR_TO_INT.get(config.projector_method, 0),
        config.min_iter,
        int(getattr(config, "ad_regularize_svd", True)),
        int(getattr(config, "gmres_precondition", True)),
        {"vjp": 0, "gmres": 1}.get(getattr(config, "ad_backward_method", "vjp"), 0),
        _CONV_METHOD_STR_TO_INT.get(getattr(config, "ctm_conv_method", "sv"), 0),
        {"qr": 0, "sigma": 1, "phase": 2, "none": 3}.get(
            getattr(config, "forward_gauge", "qr"), 0
        ),
        _PB_STR_TO_INT.get(getattr(config, "projector_backward", "auto"), 0),
        int(getattr(config, "adjoint_arnoldi_precheck", True)),
        tuple(tuple(x) for x in config.chi_ramp)
        if getattr(config, "chi_ramp", None)
        else (),
    )


def _config_from_tuple(config_tuple: tuple):
    """Reconstruct CTMConfig from a packed tuple."""
    pm_int = config_tuple[4] if len(config_tuple) > 4 else 0
    min_iter = config_tuple[5] if len(config_tuple) > 5 else 10
    ad_regularize_svd = bool(config_tuple[6]) if len(config_tuple) > 6 else True
    gmres_precondition = bool(config_tuple[7]) if len(config_tuple) > 7 else False
    ad_bwd_int = config_tuple[8] if len(config_tuple) > 8 else 0
    ad_backward_method = {0: "vjp", 1: "gmres"}.get(ad_bwd_int, "vjp")
    conv_method_int = config_tuple[9] if len(config_tuple) > 9 else 0
    ctm_conv_method = _CONV_METHOD_INT_TO_STR.get(conv_method_int, "sv")
    forward_gauge_int = config_tuple[10] if len(config_tuple) > 10 else 0
    forward_gauge = {0: "qr", 1: "sigma", 2: "phase", 3: "none"}.get(
        forward_gauge_int, "qr"
    )
    pb_int = config_tuple[11] if len(config_tuple) > 11 else 0
    projector_backward = _PB_INT_TO_STR.get(pb_int, "auto")
    adjoint_arnoldi_precheck = (
        bool(config_tuple[12]) if len(config_tuple) > 12 else True
    )
    chi_ramp_encoded = config_tuple[13] if len(config_tuple) > 13 else ()
    chi_ramp = [tuple(x) for x in chi_ramp_encoded] if chi_ramp_encoded else None
    return CTMConfig(
        chi=config_tuple[0],
        max_iter=config_tuple[1],
        conv_tol=config_tuple[2],
        renormalize=bool(config_tuple[3]),
        projector_method=_PM_INT_TO_STR.get(pm_int, "eigh"),
        min_iter=min_iter,
        ad_regularize_svd=ad_regularize_svd,
        gmres_precondition=gmres_precondition,
        ad_backward_method=ad_backward_method,
        ctm_conv_method=ctm_conv_method,
        forward_gauge=forward_gauge,
        projector_backward=projector_backward,
        adjoint_arnoldi_precheck=adjoint_arnoldi_precheck,
        chi_ramp=chi_ramp,
    )


# ---------------------------------------------------------------------------
# 3. Standard CTM (Tensor protocol) fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _wrap_tensor(data, original):
    """Wrap dense data back into a Tensor preserving the original index structure."""
    from tenax.core.tensor import SymmetricTensor

    if isinstance(original, SymmetricTensor):
        return SymmetricTensor.from_dense(data, original.indices, tol=float("inf"))
    return type(original)(data, original.indices)


def _transfer_matrix_leading_eigvec(T_dense: jax.Array, n_iter: int = 30) -> jax.Array:
    """Compute leading right eigenvector of the double-layer transfer matrix.

    For a 3-leg edge tensor T of shape ``(chi, D2, chi)`` (left, phys, right),
    the double-layer transfer matrix acts on rho(chi, chi) as::

        rho' = sum_k T[:, k, :] @ rho @ T[:, k, :]^H

    Uses power iteration (matrix-free) which is fully JAX-differentiable,
    enabling sigma gauge to work with explicit AD backprop.

    Returns: leading eigenvector reshaped to (chi, chi).
    """
    chi = T_dense.shape[0]

    def _apply_tm(v_flat):
        """Apply transfer matrix TM @ v without building the full matrix."""
        rho = v_flat.reshape(chi, chi)
        # rho'[a,c] = sum_{k,b,d} T[a,k,b] * rho[b,d] * conj(T[c,k,d])
        rho_new = jnp.einsum("akb,bd,ckd->ac", T_dense, rho, T_dense.conj())
        return rho_new.reshape(-1)

    # Power iteration: start from uniform vector, converge to leading eigvec
    v = jnp.ones(chi * chi, dtype=T_dense.dtype) / chi
    for _ in range(n_iter):
        v = _apply_tm(v)
        v = v / (jnp.linalg.norm(v) + 1e-30)

    return v.real.reshape(chi, chi)


def arnoldi_spectral_radius(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    v0: jnp.ndarray,
    n_iter: int = 20,
) -> float:
    """Estimate the spectral radius of a linear operator via Arnoldi iteration.

    Builds an (n_iter x n_iter) upper Hessenberg matrix H from the Krylov
    subspace {v0, A v0, A^2 v0, ...} using Modified Gram-Schmidt, then
    returns max |eigenvalue(H)| as an approximation to rho(A).

    Args:
        matvec: Function applying the operator (e.g. J^T) to a flat vector.
        v0: Initial vector (typically the gradient g, flattened).
        n_iter: Number of Arnoldi iterations (default 20).

    Returns:
        Estimated spectral radius rho(A).
    """
    n = v0.shape[0]
    k = min(n_iter, n)
    Q = jnp.zeros((n, k + 1))
    H = jnp.zeros((k + 1, k))

    q = v0 / (jnp.linalg.norm(v0) + 1e-30)
    Q = Q.at[:, 0].set(q)

    actual_k = k
    for j in range(k):
        v = matvec(Q[:, j])
        for i in range(j + 1):
            h = jnp.dot(Q[:, i], v)
            H = H.at[i, j].set(h)
            v = v - h * Q[:, i]
        h_next = jnp.linalg.norm(v)
        H = H.at[j + 1, j].set(h_next)
        if h_next < 1e-14:
            actual_k = j + 1
            break
        Q = Q.at[:, j + 1].set(v / h_next)

    H_k = H[:actual_k, :actual_k]
    eigenvalues = jnp.linalg.eigvals(H_k)
    return float(jnp.max(jnp.abs(eigenvalues)))


def _sigma_gauge_fix_ctm_tensor(env_new, env_old):
    """Fix gauge of CTMTensorEnv via transfer-matrix eigenvector alignment.

    Computes sigma matrices from the leading eigenvectors of the transfer
    matrices of ``env_new`` and ``env_old``, then applies them so that
    ``env_new`` converges to ``env_old`` element-wise (not just spectrally).

    This is the gauge-fixing approach from arxiv:2311.11894, used by YASTN
    to make the VJP backward Neumann series converge.

    For each edge direction, computes:
        rho_old = leading eigvec of TM(T_old)   ->  Q_old via QR
        rho_new = leading eigvec of TM(T_new)   ->  Q_new via QR
        sigma = Q_new @ Q_old^dagger

    Then applies sigma to corners and edges so that the gauge aligns
    between iterations.
    """

    C1_n, C2_n, C3_n, C4_n = (
        c.todense() for c in (env_new.C1, env_new.C2, env_new.C3, env_new.C4)
    )
    T1_n, T2_n, T3_n, T4_n = (
        t.todense() for t in (env_new.T1, env_new.T2, env_new.T3, env_new.T4)
    )
    T1_o, T2_o, T3_o, T4_o = (
        t.todense() for t in (env_old.T1, env_old.T2, env_old.T3, env_old.T4)
    )

    def _compute_sigma(T_new, T_old):
        """Compute sigma = Q_new @ Q_old^H from transfer matrix eigenvectors."""
        rho_new = _transfer_matrix_leading_eigvec(T_new)
        rho_old = _transfer_matrix_leading_eigvec(T_old)

        # QR with sign convention (positive diagonal on R)
        Q_new, R_new = jnp.linalg.qr(rho_new)
        Q_old, R_old = jnp.linalg.qr(rho_old)

        # Fix sign: make diagonal of R positive
        signs_new = jnp.sign(jnp.diag(R_new))
        signs_old = jnp.sign(jnp.diag(R_old))
        Q_new = Q_new * signs_new[None, :]
        Q_old = Q_old * signs_old[None, :]

        return Q_new @ Q_old.conj().T

    # Compute sigma for each edge direction
    # Convention: sigma_X acts on the chi bond on the X side
    # T1 connects c1_r <-> c2_l (top bond)
    # T2 connects c2_d <-> c3_u (right bond)
    # T3 connects c4_r <-> c3_l (bottom bond)
    # T4 connects c1_d <-> c4_u (left bond)
    sigma_top = jax.lax.stop_gradient(_compute_sigma(T1_n, T1_o))
    sigma_right = jax.lax.stop_gradient(_compute_sigma(T2_n, T2_o))
    sigma_bottom = jax.lax.stop_gradient(_compute_sigma(T3_n, T3_o))
    sigma_left = jax.lax.stop_gradient(_compute_sigma(T4_n, T4_o))

    # Apply sigma to env_new:
    # Corner: C(row_bond, col_bond) -> sigma_row^H @ C @ sigma_col
    # Edge: T(left_bond, phys, right_bond) -> sigma_left^H @ T @ sigma_right
    #
    # CTM geometry (single-site, periodic):
    # C1(c1_d, c1_r)    connects to T4(left) and T1(top)
    # C2(c2_l, c2_d)    connects to T1(top) and T2(right)
    # C3(c3_u, c3_l)    connects to T2(right) and T3(bottom)
    # C4(c4_r, c4_u)    connects to T3(bottom) and T4(left)
    #
    # T1(t1_l, u2, t1_r) connects c1_r(left) and c2_l(right) via top bond
    # T2(t2_u, r2, t2_d) connects c2_d(top) and c3_u(bottom) via right bond
    # T3(t3_r, d2, t3_l) connects c4_r(left) and c3_l(right) via bottom bond
    # T4(t4_d, l2, t4_u) connects c1_d(top) and c4_u(bottom) via left bond

    sH_top = sigma_top.conj().T
    sH_right = sigma_right.conj().T
    sH_bottom = sigma_bottom.conj().T
    sH_left = sigma_left.conj().T

    # Corners: each corner sits at an intersection of two bonds
    C1_fixed = sH_left @ C1_n @ sigma_top  # c1_d(left bond), c1_r(top bond)
    C2_fixed = sH_top @ C2_n @ sigma_right  # c2_l(top bond), c2_d(right bond)
    C3_fixed = sH_right @ C3_n @ sigma_bottom  # c3_u(right bond), c3_l(bottom bond)
    C4_fixed = sH_bottom @ C4_n @ sigma_left  # c4_r(bottom bond), c4_u(left bond)

    # Edges: each edge has chi bonds on both sides + D^2 physical bond
    # T1(t1_l, u2, t1_r): left=top bond, right=top bond (same bond, wraps around)
    T1_fixed = jnp.einsum("ab,bdc,ce->ade", sH_top, T1_n, sigma_top)
    # T2(t2_u, r2, t2_d): top=right bond, bottom=right bond
    T2_fixed = jnp.einsum("ab,bdc,ce->ade", sH_right, T2_n, sigma_right)
    # T3(t3_r, d2, t3_l): left=bottom bond, right=bottom bond
    T3_fixed = jnp.einsum("ab,bdc,ce->ade", sH_bottom, T3_n, sigma_bottom)
    # T4(t4_d, l2, t4_u): top=left bond, bottom=left bond
    T4_fixed = jnp.einsum("ab,bdc,ce->ade", sH_left, T4_n, sigma_left)

    # Extract global U(1) phase per tensor from old env
    C1_o, C2_o, C3_o, C4_o = (
        c.todense() for c in (env_old.C1, env_old.C2, env_old.C3, env_old.C4)
    )

    def _fix_phase(fixed, old):
        """Remove residual U(1) phase by aligning with old tensor."""
        dot = jnp.sum(old.ravel().conj() * fixed.ravel())
        phase = dot / (jnp.abs(dot) + 1e-30)
        return fixed * jnp.conj(phase)

    C1_fixed = _fix_phase(C1_fixed, C1_o)
    C2_fixed = _fix_phase(C2_fixed, C2_o)
    C3_fixed = _fix_phase(C3_fixed, C3_o)
    C4_fixed = _fix_phase(C4_fixed, C4_o)
    T1_fixed = _fix_phase(T1_fixed, T1_o)
    T2_fixed = _fix_phase(T2_fixed, T2_o)
    T3_fixed = _fix_phase(T3_fixed, T3_o)
    T4_fixed = _fix_phase(T4_fixed, T4_o)

    return CTMTensorEnv(
        C1=_wrap_tensor(C1_fixed, env_new.C1),
        C2=_wrap_tensor(C2_fixed, env_new.C2),
        C3=_wrap_tensor(C3_fixed, env_new.C3),
        C4=_wrap_tensor(C4_fixed, env_new.C4),
        T1=_wrap_tensor(T1_fixed, env_new.T1),
        T2=_wrap_tensor(T2_fixed, env_new.T2),
        T3=_wrap_tensor(T3_fixed, env_new.T3),
        T4=_wrap_tensor(T4_fixed, env_new.T4),
    )


def _gauge_fix_ctm_tensor(env):
    """Fix gauge of CTMTensorEnv via QR decomposition of corners.

    Performs dense QR on corner and edge arrays, then wraps results back
    into Tensor objects preserving the original index structure.  All
    dense operations (``todense()``, ``jnp.linalg.qr``, ``jnp.einsum``)
    are JAX-differentiable.

    For SymmetricTensor with trivial charges (all zeros), the dense
    round-trip is cheap (single block).  For non-trivial charges, the
    ``from_dense(..., tol=inf)`` wrapping preserves the charge layout.
    """

    # Extract dense arrays — for SymmetricTensor with trivial charges
    # this is essentially free (single block covers the full tensor).
    C1, C2, C3, C4 = (c.todense() for c in (env.C1, env.C2, env.C3, env.C4))
    T1, T2, T3, T4 = (t.todense() for t in (env.T1, env.T2, env.T3, env.T4))

    def _sign_fixed_qr(M):
        """QR with positive diagonal on R (removes sign ambiguity)."""
        Q, R = jnp.linalg.qr(M)
        signs = jnp.sign(jnp.diag(R))
        # Replace zeros with 1 to avoid multiplying by 0
        signs = jnp.where(signs == 0, 1.0, signs)
        return Q * signs[None, :], R * signs[:, None]

    # C1 = Q1 @ R1 → C1_new = R1, absorb Q1^H into T1 (left) and T4 (left)
    Q1, R1 = _sign_fixed_qr(C1)
    C1_new = R1
    T1_new = jnp.einsum("ab,bdc->adc", Q1.conj().T, T1)
    T4_new = jnp.einsum("ab,bdc->adc", Q1.conj().T, T4)

    # C2 = Q2 @ R2 → C2_new = R2, absorb Q2 into T1 (right) and Q2^H into T2 (top)
    Q2, R2 = _sign_fixed_qr(C2)
    C2_new = R2
    T1_new = jnp.einsum("adb,bc->adc", T1_new, Q2)
    T2_new = jnp.einsum("ab,bdc->adc", Q2.conj().T, T2)

    # C3 = Q3 @ R3 → C3_new = R3, absorb Q3 into T2 (bottom) and T3 (right)
    Q3, R3 = _sign_fixed_qr(C3)
    C3_new = R3
    T2_new = jnp.einsum("adb,bc->adc", T2_new, Q3)
    T3_new = jnp.einsum("adb,bc->adc", T3, Q3)

    # C4 = Q4 @ R4 → C4_new = R4, absorb Q4^H into T3 (left) and Q4 into T4 (bottom)
    Q4, R4 = _sign_fixed_qr(C4)
    C4_new = R4
    T3_new = jnp.einsum("ab,bdc->adc", Q4.conj().T, T3_new)
    T4_new = jnp.einsum("adb,bc->adc", T4_new, Q4)

    return CTMTensorEnv(
        C1=_wrap_tensor(C1_new, env.C1),
        C2=_wrap_tensor(C2_new, env.C2),
        C3=_wrap_tensor(C3_new, env.C3),
        C4=_wrap_tensor(C4_new, env.C4),
        T1=_wrap_tensor(T1_new, env.T1),
        T2=_wrap_tensor(T2_new, env.T2),
        T3=_wrap_tensor(T3_new, env.T3),
        T4=_wrap_tensor(T4_new, env.T4),
    )


_EPS_PHASE = 0.1  # threshold fraction for "large" element (variPEPS default)


def _frob_phase_fix(arr):
    """Frobenius-normalize and fix the global U(1) phase of a dense array.

    1. Normalize by Frobenius norm (differentiable; norm wrapped in
       ``stop_gradient`` so the backward only sees the tangential gradient
       component — see #362 for why the radial path through divide-by-norm
       produces NaN on SymmetricTensor with empty charge sectors).
    2. Fix the phase by making the first "large" element (|x| >= EPS_PHASE *
       max|arr|) real-positive (variPEPS ``_post_process_CTM_tensors``).

    Shared by :func:`_phase_fix_ctm_tensor` (fused env) and
    :func:`_phase_fix_split_ctm_tensor` (split env).
    """
    norm = jnp.linalg.norm(arr)
    arr = arr / jax.lax.stop_gradient(norm + 1e-30)
    # Find first element with |x| >= EPS_PHASE * max(|arr|)
    flat = arr.ravel()
    abs_flat = jnp.abs(flat)
    abs_max = jnp.max(abs_flat)
    threshold = _EPS_PHASE * abs_max
    # Use argmax on a mask to find the first qualifying element
    mask = abs_flat >= threshold
    # jnp.argmax returns first True in the mask
    idx = jnp.argmax(mask)
    val = flat[idx]
    phase = val / (jnp.abs(val) + 1e-30)
    return arr * jnp.conj(phase)


def _phase_fix_ctm_tensor(env):
    """Fix gauge of CTMTensorEnv via Frobenius normalization + phase fixing.

    For each corner and edge tensor:
    1. Normalize by Frobenius norm (differentiable).
    2. Fix the global U(1) phase by making the first large element
       real-positive (variPEPS ``_post_process_CTM_tensors`` approach).

    This is simpler than sigma gauge (no transfer matrix eigenvector
    computation) and works identically in Python loops and JIT while_loop.

    Uses todense/from_dense round-trip.  Environment tensors are small
    (chi x chi corners, chi x D^2 x chi edges), so this is acceptable.
    The from_dense re-projection into block structure is needed for
    numerical stability over many CTM sweeps.
    """
    C1 = _frob_phase_fix(env.C1.todense())
    C2 = _frob_phase_fix(env.C2.todense())
    C3 = _frob_phase_fix(env.C3.todense())
    C4 = _frob_phase_fix(env.C4.todense())
    T1 = _frob_phase_fix(env.T1.todense())
    T2 = _frob_phase_fix(env.T2.todense())
    T3 = _frob_phase_fix(env.T3.todense())
    T4 = _frob_phase_fix(env.T4.todense())

    return CTMTensorEnv(
        C1=_wrap_tensor(C1, env.C1),
        C2=_wrap_tensor(C2, env.C2),
        C3=_wrap_tensor(C3, env.C3),
        C4=_wrap_tensor(C4, env.C4),
        T1=_wrap_tensor(T1, env.T1),
        T2=_wrap_tensor(T2, env.T2),
        T3=_wrap_tensor(T3, env.T3),
        T4=_wrap_tensor(T4, env.T4),
    )


def _phase_fix_split_ctm_tensor(env):
    """Fix gauge of a ``SplitCTMTensorEnv`` (Γ phase-fix over its 12 tensors).

    The split env carries the same residual U(1) gauge freedom as the fused
    :class:`CTMTensorEnv`, but on a 12-tensor (4 corners + 8 ket/bra edge
    halves) layout.  Without a per-tensor gauge fix the converged split env
    has no *element-wise* fixed point: the rank-1-seeded corner spectrum
    ``[0.5, 0.5, 0, 0]`` leaves a degenerate 2-d subspace that rotates each
    sweep, so the residual oscillates forever while the gauge-invariant
    energy is machine-stable (#463).  Applying :func:`_frob_phase_fix` to
    each tensor pins that gauge, giving the element-wise fixed point the
    implicit Neumann backward (variPEPS Eq. 18-19) needs to converge.

    Mirrors :func:`_phase_fix_ctm_tensor` exactly — same per-tensor
    Frobenius-normalize + first-large-element phase fix, same
    ``todense``/``from_dense`` round-trip — but over the split fields.
    """
    C1 = _frob_phase_fix(env.C1.todense())
    C2 = _frob_phase_fix(env.C2.todense())
    C3 = _frob_phase_fix(env.C3.todense())
    C4 = _frob_phase_fix(env.C4.todense())
    T1k = _frob_phase_fix(env.T1_ket.todense())
    T1b = _frob_phase_fix(env.T1_bra.todense())
    T2k = _frob_phase_fix(env.T2_ket.todense())
    T2b = _frob_phase_fix(env.T2_bra.todense())
    T3k = _frob_phase_fix(env.T3_ket.todense())
    T3b = _frob_phase_fix(env.T3_bra.todense())
    T4k = _frob_phase_fix(env.T4_ket.todense())
    T4b = _frob_phase_fix(env.T4_bra.todense())

    from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv

    return SplitCTMTensorEnv(
        C1=_wrap_tensor(C1, env.C1),
        C2=_wrap_tensor(C2, env.C2),
        C3=_wrap_tensor(C3, env.C3),
        C4=_wrap_tensor(C4, env.C4),
        T1_ket=_wrap_tensor(T1k, env.T1_ket),
        T1_bra=_wrap_tensor(T1b, env.T1_bra),
        T2_ket=_wrap_tensor(T2k, env.T2_ket),
        T2_bra=_wrap_tensor(T2b, env.T2_bra),
        T3_ket=_wrap_tensor(T3k, env.T3_ket),
        T3_bra=_wrap_tensor(T3b, env.T3_bra),
        T4_ket=_wrap_tensor(T4k, env.T4_ket),
        T4_bra=_wrap_tensor(T4b, env.T4_bra),
    )


def _needs_paired_sweep(A) -> bool:
    """Check if A is a SymmetricTensor with non-trivial virtual charges."""
    from tenax.core.tensor import SymmetricTensor

    if not isinstance(A, SymmetricTensor):
        return False
    import numpy as _np

    virtual_charges = [_np.sort(A.indices[i].charges) for i in range(4)]
    has_nontrivial = any(not _np.all(vc == 0) for vc in virtual_charges)
    all_same = all(
        _np.array_equal(virtual_charges[0], virtual_charges[i]) for i in range(1, 4)
    )
    return has_nontrivial and all_same


def _ctm_sv_diff_local(sv_new, sv_old):
    """Compute max abs diff between normalized SVs."""
    return _ctm_sv_diff_tensor(sv_new, sv_old)


def _flatten_envs(envs):
    """Flatten ``{Coord: CTMTensorEnv}`` to a flat tuple of leaves in coord-sorted order."""
    result = ()
    for c in sorted(envs):
        result = result + tuple(jax.tree.leaves(envs[c]))
    return result


def _unflatten_envs_init(env_init_leaves, site_tensors, chi):
    """Unflatten env_init_leaves into ``{Coord: CTMTensorEnv}``, or None."""
    if env_init_leaves is None:
        return None
    # Use first site tensor as template
    first_tensor = next(iter(site_tensors.values()))
    template = initialize_ctm_tensor_env(first_tensor, chi)
    treedef = jax.tree.structure(template)
    n = len(jax.tree.leaves(template))
    envs = {}
    offset = 0
    for c in sorted(site_tensors):
        envs[c] = jax.tree.unflatten(
            treedef, list(env_init_leaves[offset : offset + n])
        )
        offset += n
    return envs


def _ctm_tensor_step_multisite(
    site_leaves,
    env_leaves,
    neighbors,
    chi,
    renormalize,
    projector_method,
    site_treedefs,
    env_treedef,
    n_env_per_site,
    double_layers=None,
    sigma_gauge_ref_leaves=None,
    skip_gauge=False,
    gauge_mode="qr",
    projector_backward="auto",
):
    """One multisite CTM sweep + gauge fix, flat leaves to flat leaves.

    Works for any number of sites (1-site, 2-site, etc.).

    If *double_layers* is provided, it is used directly (avoids redundant
    recomputation when site tensors are constant, e.g. in the GMRES backward pass).

    If *sigma_gauge_ref_leaves* is provided, uses sigma gauge fixing
    (aligning output to the reference environment) instead of QR gauge.

    If *skip_gauge* is True, no gauge fix is applied (used in backward
    to avoid QR NaN on rank-deficient corners where D² < chi).

    *gauge_mode* selects the gauge fix when sigma_gauge_ref_leaves is None
    and skip_gauge is False: ``"qr"`` (default) or ``"phase"``
    (Frobenius normalization + phase fixing, variPEPS-style).
    """
    # Unflatten site tensors
    coords = sorted(site_treedefs)
    site_tensors = {}
    site_offset = 0
    for c in coords:
        td = site_treedefs[c]
        n_leaves = td.num_leaves
        site_tensors[c] = jax.tree.unflatten(
            td, site_leaves[site_offset : site_offset + n_leaves]
        )
        site_offset += n_leaves

    # Unflatten env tensors
    envs = {}
    env_offset = 0
    for c in coords:
        envs[c] = jax.tree.unflatten(
            env_treedef, list(env_leaves[env_offset : env_offset + n_env_per_site])
        )
        env_offset += n_env_per_site

    # Build double layers if not cached
    if double_layers is None:
        double_layers = {
            c: _build_double_layer_tensor(A) for c, A in site_tensors.items()
        }

    envs, _, _ = _ctm_tensor_sweep_multisite(
        envs,
        double_layers,
        neighbors,
        chi,
        renormalize,
        projector_method,
        projector_backward=projector_backward,
    )

    if sigma_gauge_ref_leaves is not None:
        # Sigma gauge: align output with reference (converged) environment
        ref_envs = {}
        ref_offset = 0
        for c in coords:
            ref_envs[c] = jax.tree.unflatten(
                env_treedef,
                list(sigma_gauge_ref_leaves[ref_offset : ref_offset + n_env_per_site]),
            )
            ref_offset += n_env_per_site
        envs = {c: _sigma_gauge_fix_ctm_tensor(envs[c], ref_envs[c]) for c in envs}
    elif skip_gauge:
        pass  # no gauge fix — used in backward to avoid QR NaN on rank-deficient corners
    elif gauge_mode == "phase":
        envs = {c: _phase_fix_ctm_tensor(e) for c, e in envs.items()}
    else:
        envs = {c: _gauge_fix_ctm_tensor(e) for c, e in envs.items()}

    return _flatten_envs(envs)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def ctm_tensor_converge(
    site_tensors,
    env_init_leaves,
    neighbors,
    config_tuple: tuple,
) -> tuple[jax.Array, ...]:
    """Unified multisite Tensor-protocol CTM with implicit differentiation.

    Handles 1-site (with ``SINGLE_SITE_NEIGHBORS``) and multi-site
    (e.g. ``CHECKERBOARD_NEIGHBORS``) unit cells through a single code path.

    Args:
        site_tensors:    Dict ``{Coord: Tensor}`` of iPEPS site tensors.
        env_init_leaves: Flat tuple of env leaf arrays for warm-start, or None.
        neighbors:       Neighbor map ``{Coord: {direction: Coord}}``.
        config_tuple:    CTMConfig fields packed as tuple for JAX tracing.

    Returns:
        Flat tuple of environment pytree leaf arrays (all sites, coord-sorted).
    """
    warnings.warn(
        "ctm_tensor_converge is deprecated. Use ctm_energy_implicit from "
        "tenax.algorithms._ctm_energy_ad instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    config = _config_from_tuple(config_tuple)
    envs_init = _unflatten_envs_init(env_init_leaves, site_tensors, config.chi)
    envs = _ctm_tensor_multisite_fixed_point(
        site_tensors, neighbors, config, envs_init=envs_init
    )
    return _flatten_envs(envs)


def _ctm_tensor_converge_fwd(site_tensors, env_init_leaves, neighbors, config_tuple):
    """Forward pass -- run multisite Tensor CTM, cache tensors and envs."""
    config = _config_from_tuple(config_tuple)
    envs_init = _unflatten_envs_init(env_init_leaves, site_tensors, config.chi)
    envs = _ctm_tensor_multisite_fixed_point(
        site_tensors, neighbors, config, envs_init=envs_init
    )
    out = _flatten_envs(envs)
    residuals = (site_tensors, envs, env_init_leaves)
    return out, residuals


def _precompute_sigma_matrices(envs):
    """Precompute sigma Q matrices from converged environment (constants).

    For each edge direction, compute the leading eigenvector of the
    double-layer transfer matrix and return its QR factor Q.

    Returns: ``{coord: (Q_T1, Q_T2, Q_T3, Q_T4)}`` where each Q is chi×chi.
    """
    result = {}
    for c in sorted(envs):
        env = envs[c]
        Qs = []
        for T in (env.T1, env.T2, env.T3, env.T4):
            rho = _transfer_matrix_leading_eigvec(T.todense())
            Q, R = jnp.linalg.qr(rho)
            signs = jnp.sign(jnp.diag(R))
            signs = jnp.where(signs == 0, 1.0, signs)
            Q = Q * signs[None, :]
            Qs.append(Q)
        result[c] = tuple(Qs)
    return result


def _apply_sigma_to_env_leaves(
    env_leaves, sigma_Qs, env_treedef, n_env_per_site, coords
):
    """Apply precomputed sigma gauge to env leaves (differentiable).

    The sigma Q matrices are stop_gradient constants; the application
    (matrix multiplications) is fully JAX-differentiable.
    """
    new_leaves = []
    offset = 0
    for c in coords:
        env = jax.tree.unflatten(
            env_treedef, list(env_leaves[offset : offset + n_env_per_site])
        )
        Q_T1, Q_T2, Q_T3, Q_T4 = sigma_Qs[c]
        Q_T1 = jax.lax.stop_gradient(Q_T1)
        Q_T2 = jax.lax.stop_gradient(Q_T2)
        Q_T3 = jax.lax.stop_gradient(Q_T3)
        Q_T4 = jax.lax.stop_gradient(Q_T4)

        C1, C2, C3, C4 = (x.todense() for x in (env.C1, env.C2, env.C3, env.C4))
        T1, T2, T3, T4 = (x.todense() for x in (env.T1, env.T2, env.T3, env.T4))

        # Corners: Q_row† @ C @ Q_col
        C1_f = Q_T4.conj().T @ C1 @ Q_T1
        C2_f = Q_T1.conj().T @ C2 @ Q_T2
        C3_f = Q_T2.conj().T @ C3 @ Q_T3
        C4_f = Q_T3.conj().T @ C4 @ Q_T4

        # Edges: Q† @ T @ Q (same bond on both sides for single-site)
        T1_f = jnp.einsum("ab,bdc,ce->ade", Q_T1.conj().T, T1, Q_T1)
        T2_f = jnp.einsum("ab,bdc,ce->ade", Q_T2.conj().T, T2, Q_T2)
        T3_f = jnp.einsum("ab,bdc,ce->ade", Q_T3.conj().T, T3, Q_T3)
        T4_f = jnp.einsum("ab,bdc,ce->ade", Q_T4.conj().T, T4, Q_T4)

        env_fixed = CTMTensorEnv(
            C1=_wrap_tensor(C1_f, env.C1),
            C2=_wrap_tensor(C2_f, env.C2),
            C3=_wrap_tensor(C3_f, env.C3),
            C4=_wrap_tensor(C4_f, env.C4),
            T1=_wrap_tensor(T1_f, env.T1),
            T2=_wrap_tensor(T2_f, env.T2),
            T3=_wrap_tensor(T3_f, env.T3),
            T4=_wrap_tensor(T4_f, env.T4),
        )
        new_leaves.extend(jax.tree.leaves(env_fixed))
        offset += n_env_per_site

    return tuple(new_leaves)


def _ctm_tensor_converge_bwd(neighbors, config_tuple, residuals, g):
    """Backward pass via implicit differentiation of multisite CTM fixed point."""
    site_tensors, envs, env_init_leaves = residuals
    config = _config_from_tuple(config_tuple)

    coords = sorted(site_tensors)

    # Build treedefs for each site tensor
    site_treedefs = {c: jax.tree.structure(site_tensors[c]) for c in coords}

    # All envs share the same treedef
    env_treedef = jax.tree.structure(envs[coords[0]])
    n_env_per_site = len(jax.tree.leaves(envs[coords[0]]))

    # Flatten site and env leaves
    site_leaves = ()
    for c in coords:
        site_leaves = site_leaves + tuple(jax.tree.leaves(site_tensors[c]))
    env_leaves = _flatten_envs(envs)

    use_sigma = getattr(config, "forward_gauge", "qr") == "sigma"

    if use_sigma:
        # --- YASTN-style backward: relative sigma-gauged step function ---
        #
        # The step function for the backward computes:
        #   g(A, env) = sigma_fix(CTM_step(A, env), stop_gradient(env))
        #
        # where sigma_fix aligns the sweep output to the (detached) input
        # via relative transfer-matrix eigenvector alignment.  This mirrors
        # the forward's _sigma_gauge_fix_ctm_tensor(env_new, env_old) and
        # the explicit-AD path's _one_sweep_sigma pattern (line ~1756).
        #
        # The stop_gradient on the reference ensures gradients flow only
        # through the forward map, not through the alignment target.
        #
        # Reference: YASTN fixed_pt.py FixedPoint.fixed_point_iter (arxiv:2311.11894)

        def step_fn_sigma(s_leaves, e_leaves):
            """One CTM sweep + relative sigma gauge alignment."""
            e_ref = tuple(jax.lax.stop_gradient(x) for x in e_leaves)
            swept = _ctm_tensor_step_multisite(
                s_leaves,
                e_leaves,
                neighbors,
                config.chi,
                config.renormalize,
                config.projector_method,
                site_treedefs,
                env_treedef,
                n_env_per_site,
                sigma_gauge_ref_leaves=e_ref,
                projector_backward=getattr(config, "projector_backward", "auto"),
            )
            return swept

        _, vjp_env_fn = jax.vjp(lambda e: step_fn_sigma(site_leaves, e), env_leaves)
        _, vjp_site_fn = jax.vjp(lambda s: step_fn_sigma(s, env_leaves), site_leaves)
    else:
        # Skip gauge fix in backward step to avoid QR NaN on rank-deficient
        # corners (D² < chi).  At the fixed point the gauge is already fixed.
        def step_fn(s_leaves, e_leaves):
            return _ctm_tensor_step_multisite(
                s_leaves,
                e_leaves,
                neighbors,
                config.chi,
                config.renormalize,
                config.projector_method,
                site_treedefs,
                env_treedef,
                n_env_per_site,
                skip_gauge=True,
                projector_backward=getattr(config, "projector_backward", "auto"),
            )

        _, vjp_env_fn = jax.vjp(lambda e: step_fn(site_leaves, e), env_leaves)
        _, vjp_site_fn = jax.vjp(lambda s: step_fn(s, env_leaves), site_leaves)

    max_fp_iter = min(config.max_iter, 50)

    # --- Arnoldi spectral-radius precheck ---
    if getattr(config, "adjoint_arnoldi_precheck", True):
        g_flat = jnp.concatenate([gi.ravel() for gi in g])
        shapes = [gi.shape for gi in g]
        sizes = [gi.size for gi in g]
        splits = jnp.cumsum(jnp.array(sizes[:-1]))

        def _flat_matvec(v_flat):
            chunks = jnp.split(v_flat, splits)
            v_tuple = tuple(c.reshape(s) for c, s in zip(chunks, shapes))
            jt_v = vjp_env_fn(v_tuple)[0]
            return jnp.concatenate([ji.ravel() for ji in jt_v])

        rho = arnoldi_spectral_radius(_flat_matvec, g_flat, n_iter=20)
        _logger.info("Arnoldi precheck: rho(J^T) = %.4f", rho)

        arnoldi_threshold = getattr(config, "adjoint_arnoldi_threshold", 5.0)
        if rho >= arnoldi_threshold:
            raise CTMRGGradientError(spectral_radius=rho)

    if config.ad_backward_method == "gmres":
        # --- GMRES path: solve (I - J^T) lam = g directly ---
        def apply_I_minus_Jt(v):
            Jt_v = vjp_env_fn(v)[0]
            return tuple(vi - ji for vi, ji in zip(v, Jt_v))

        precond = None
        lam, info = jax_gmres(
            apply_I_minus_Jt,
            g,
            x0=g,
            tol=config.conv_tol,
            maxiter=max_fp_iter,
            M=precond,
        )
        d_site_leaves = vjp_site_fn(lam)[0]
    else:
        # --- YASTN-style iterative VJP (Neumann series) ---
        #
        # Accumulate lam = sum_{n=0}^{inf} (J^T_env)^n @ g in env space,
        # then project to site space once: d_site = (dstep/dA)^T @ lam.
        #
        # With sigma gauge in the step function, J^T should have
        # spectral radius < 1 in the physical subspace.
        #
        # Convergence is checked via the projected site gradient (EMA),
        # following YASTN fixed_pt.py (arxiv:2311.11894).
        grads = g
        lam = g  # accumulated Neumann sum

        prev_site_grad = None
        diff_ema = None
        alpha_ema = 0.4

        for it in range(max_fp_iter):
            grads = vjp_env_fn(grads)[0]  # grads = J^T @ grads

            # Check if all cotangents are small (converged in env space)
            grads_inf = max(float(jnp.max(jnp.abs(gi))) for gi in grads)
            if grads_inf < config.conv_tol:
                break

            # Accumulate Neumann sum
            lam = tuple(li + gi for li, gi in zip(lam, grads))

            # Check convergence via projected site gradient (every 10 steps)
            if (it + 1) % 10 == 0 or it == max_fp_iter - 1:
                site_grad = vjp_site_fn(lam)[0]
                site_grad_flat = jnp.concatenate(
                    [si.ravel() for si in jax.tree.leaves(site_grad)]
                )
                if prev_site_grad is not None:
                    grad_diff = float(jnp.linalg.norm(site_grad_flat - prev_site_grad))
                    if grad_diff < config.conv_tol:
                        break
                    # EMA-based divergence detection
                    if diff_ema is not None and grad_diff > 2 * diff_ema:
                        break
                    diff_ema = (
                        alpha_ema * grad_diff + (1 - alpha_ema) * diff_ema
                        if diff_ema is not None
                        else grad_diff
                    )
                prev_site_grad = site_grad_flat

            # Safety: stop if lam is diverging
            lam_norm = sum(float(jnp.sum(li**2)) for li in lam) ** 0.5
            if not math.isfinite(lam_norm) or lam_norm > 1e15:
                lam = tuple(li - gi for li, gi in zip(lam, grads))  # undo last
                break

        d_site_leaves = vjp_site_fn(lam)[0]

    # Unflatten site gradients back into dict matching site_tensors structure
    d_site_tensors = {}
    offset = 0
    for c in coords:
        td = site_treedefs[c]
        n = td.num_leaves
        d_site_tensors[c] = jax.tree.unflatten(td, d_site_leaves[offset : offset + n])
        offset += n

    # Zero gradient for env_init
    if env_init_leaves is None:
        d_env_init = None
    else:
        d_env_init = tuple(jnp.zeros_like(x) for x in env_init_leaves)
    return (d_site_tensors, d_env_init)


ctm_tensor_converge.defvjp(_ctm_tensor_converge_fwd, _ctm_tensor_converge_bwd)


# ---------------------------------------------------------------------------
# 4. Multisite Tensor-protocol CTM fixed-point loop (shared by all paths)
# ---------------------------------------------------------------------------


def _env_chi(envs):
    """Infer chi from the first corner of the first site's environment."""
    first_env = next(iter(envs.values()))
    return first_env.C1.todense().shape[0]


def _ctm_tensor_multisite_fixed_point_chi_ramp(
    site_tensors, neighbors, config, envs_init=None
):
    """Run multisite CTM with chi-ramp schedule (warmup at small chi)."""
    from dataclasses import replace as _replace

    chi_ramp = config.chi_ramp
    envs = envs_init
    prev_chi = _env_chi(envs) if envs is not None else None

    for stage_idx, (stage_chi, stage_sweeps) in enumerate(chi_ramp):
        is_last = stage_idx == len(chi_ramp) - 1
        stage_config = _replace(config, chi=stage_chi, chi_ramp=None)

        if not is_last and stage_sweeps is not None:
            stage_config = _replace(
                stage_config, max_iter=stage_sweeps, min_iter=0, conv_tol=0.0
            )
        elif is_last and stage_sweeps is not None:
            stage_config = _replace(stage_config, max_iter=stage_sweeps)

        # Re-initialize when chi changes (including envs_init from a
        # previous optimizer step at a different chi).  Zero-padding the
        # old environment biases CTM toward a suboptimal fixed point;
        # identity initialization at the new chi converges correctly.
        if prev_chi is not None and stage_chi != prev_chi:
            envs = None

        envs = _ctm_tensor_multisite_fixed_point(
            site_tensors, neighbors, stage_config, envs_init=envs
        )
        prev_chi = stage_chi

    return envs


def _ctm_tensor_multisite_fixed_point(site_tensors, neighbors, config, envs_init=None):
    """Run multisite Tensor-protocol CTM to convergence with gauge fixing."""
    chi_ramp = getattr(config, "chi_ramp", None)
    if chi_ramp is not None:
        return _ctm_tensor_multisite_fixed_point_chi_ramp(
            site_tensors, neighbors, config, envs_init=envs_init
        )

    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    envs = (
        envs_init
        if envs_init is not None
        else {
            c: initialize_ctm_tensor_env(A, config.chi) for c, A in site_tensors.items()
        }
    )

    use_elementwise = getattr(config, "ctm_conv_method", "sv") == "elementwise"
    gauge_mode = getattr(config, "forward_gauge", "qr")
    use_sigma = gauge_mode == "sigma"
    use_none = gauge_mode == "none"
    use_phase = gauge_mode == "phase"
    prev_svs = {}
    prev_env_arrays = {}

    for i in range(config.max_iter):
        envs_old = envs if use_sigma else None
        envs, _, _ = _ctm_tensor_sweep_multisite(
            envs,
            double_layers,
            neighbors,
            config.chi,
            config.renormalize,
            config.projector_method,
            projector_backward=getattr(config, "projector_backward", "auto"),
        )
        if use_none:
            pass  # no gauge fix — rely on projector stability
        elif use_phase:
            envs = {c: _phase_fix_ctm_tensor(e) for c, e in envs.items()}
        elif use_sigma and i > 0:
            envs = {c: _sigma_gauge_fix_ctm_tensor(envs[c], envs_old[c]) for c in envs}
        else:
            envs = {c: _gauge_fix_ctm_tensor(e) for c, e in envs.items()}

        if i + 1 < config.min_iter:
            continue

        converged = True
        if use_elementwise:
            for c in sorted(envs):
                env_arrays = tuple(
                    t.todense()
                    for t in (
                        envs[c].C1,
                        envs[c].C2,
                        envs[c].C3,
                        envs[c].C4,
                        envs[c].T1,
                        envs[c].T2,
                        envs[c].T3,
                        envs[c].T4,
                    )
                )
                if c in prev_env_arrays:
                    for curr, prev in zip(env_arrays, prev_env_arrays[c]):
                        diff = float(jnp.max(jnp.abs(curr - prev)))
                        if diff >= config.conv_tol:
                            converged = False
                            break
                    if not converged:
                        break
                else:
                    converged = False
                prev_env_arrays[c] = env_arrays
        else:
            for c in sorted(envs):
                sv = _dense_svd(envs[c].C1.todense(), compute_uv=False)
                if c in prev_svs:
                    if float(_ctm_sv_diff_local(sv, prev_svs[c])) >= config.conv_tol:
                        converged = False
                else:
                    converged = False
                prev_svs[c] = sv
        if converged:
            break

    return envs


# ---------------------------------------------------------------------------
# 4b. Explicit differentiation CTM (backprop through unrolled iterations)
# ---------------------------------------------------------------------------


def ctm_tensor_converge_explicit(
    site_tensors,
    env_init_leaves,
    neighbors,
    config_tuple: tuple,
    num_steps: int | None = None,
    warmup_steps: int = 0,
) -> tuple[jax.Array, ...]:
    """CTM convergence with explicit (unrolled) autodiff.

    Runs *warmup_steps* iterations without gradient tracking, then
    *num_steps* fully differentiable iterations (including through
    projectors via regularized SVD).  Each backprop step is wrapped
    in ``jax.checkpoint`` to trade memory for recomputation.

    Args:
        site_tensors:    Dict ``{Coord: Tensor}`` of iPEPS site tensors.
        env_init_leaves: Flat tuple of env leaf arrays for warm-start, or None.
        neighbors:       Neighbor map.
        config_tuple:    CTMConfig fields packed as tuple.
        num_steps:       Backprop CTM iterations (default: config.max_iter).
        warmup_steps:    Warmup iterations with stop_gradient (default: 0).

    Returns:
        Flat tuple of environment pytree leaf arrays.
    """
    warnings.warn(
        "ctm_tensor_converge_explicit is deprecated. Use ctm_energy_explicit "
        "from tenax.algorithms._ctm_energy_ad instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    config = _config_from_tuple(config_tuple)
    envs_init = _unflatten_envs_init(env_init_leaves, site_tensors, config.chi)

    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    envs = (
        envs_init
        if envs_init is not None
        else {
            c: initialize_ctm_tensor_env(A, config.chi) for c, A in site_tensors.items()
        }
    )

    gauge_mode = getattr(config, "forward_gauge", "qr")
    use_sigma = gauge_mode == "sigma"
    use_phase = gauge_mode == "phase"
    use_none = gauge_mode == "none"
    if use_phase:
        _gauge_fn = _phase_fix_ctm_tensor
    elif use_none:
        _gauge_fn = lambda e: e  # noqa: E731 — diagnostic: no gauge fix
    else:
        _gauge_fn = _gauge_fix_ctm_tensor

    # Phase 1: Warmup — no gradient tracking
    for wi in range(warmup_steps):
        envs_old = envs if use_sigma else None
        envs, _, _ = _ctm_tensor_sweep_multisite(
            envs,
            double_layers,
            neighbors,
            config.chi,
            config.renormalize,
            config.projector_method,
            projector_backward=getattr(config, "projector_backward", "auto"),
        )
        if use_sigma and wi > 0:
            envs = {c: _sigma_gauge_fix_ctm_tensor(envs[c], envs_old[c]) for c in envs}
        else:
            envs = {c: _gauge_fn(e) for c, e in envs.items()}
    if warmup_steps > 0:
        envs = jax.tree.map(jax.lax.stop_gradient, envs)

    # Phase 2: Backprop — fully differentiable with checkpointing.
    # Sigma gauge aligns each sweep's output to its stop_gradient input,
    # stabilizing the backward graph.  The stop_gradient copy ensures
    # sigma compares post-sweep vs pre-sweep (not self vs self).
    n = num_steps if num_steps is not None else config.max_iter
    env_treedef = jax.tree.structure(envs)

    if use_sigma:

        @jax.checkpoint
        def _one_sweep_sigma(env_leaves_flat):
            envs_inner = jax.tree.unflatten(env_treedef, env_leaves_flat)
            envs_prev = jax.tree.map(jax.lax.stop_gradient, envs_inner)
            envs_inner, _, _ = _ctm_tensor_sweep_multisite(
                envs_inner,
                double_layers,
                neighbors,
                config.chi,
                config.renormalize,
                config.projector_method,
                projector_backward=getattr(config, "projector_backward", "auto"),
            )
            envs_inner = {
                c: _sigma_gauge_fix_ctm_tensor(envs_inner[c], envs_prev[c])
                for c in envs_inner
            }
            return tuple(jax.tree.leaves(envs_inner))

        env_leaves_flat = tuple(jax.tree.leaves(envs))
        for _ in range(n):
            env_leaves_flat = _one_sweep_sigma(env_leaves_flat)
    else:

        @jax.checkpoint
        def _one_sweep(env_leaves_flat):
            envs_inner = jax.tree.unflatten(env_treedef, env_leaves_flat)
            envs_inner, _, _ = _ctm_tensor_sweep_multisite(
                envs_inner,
                double_layers,
                neighbors,
                config.chi,
                config.renormalize,
                config.projector_method,
                projector_backward=getattr(config, "projector_backward", "auto"),
            )
            envs_inner = {c: _gauge_fn(e) for c, e in envs_inner.items()}
            return tuple(jax.tree.leaves(envs_inner))

        env_leaves_flat = tuple(jax.tree.leaves(envs))
        for _ in range(n):
            env_leaves_flat = _one_sweep(env_leaves_flat)

    return _flatten_envs(jax.tree.unflatten(env_treedef, env_leaves_flat))


# ---------------------------------------------------------------------------
# 5. Split CTM (Tensor protocol) fixed-point implicit differentiation
# ---------------------------------------------------------------------------


def _split_ctm_tensor_step(
    A_flat: jax.Array,
    env_tuple: tuple[jax.Array, ...],
    chi: int,
    chi_I: int,
    renormalize: bool,
    A_template,
    env_template,
) -> tuple[jax.Array, ...]:
    """One split-CTM sweep as function of (A_flat, env_flat).

    Reconstructs Tensor objects from flat arrays using templates,
    runs one sweep, and returns the flattened environment.
    """
    # Reconstruct A from flat
    A = jax.tree.unflatten(jax.tree.structure(A_template), (A_flat,))

    # Reconstruct env from tuple of arrays
    env_leaves = list(env_tuple)
    env = jax.tree.unflatten(jax.tree.structure(env_template), env_leaves)

    env_new = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)

    return tuple(jax.tree.leaves(env_new))


def ctm_split_tensor_fixed_point(
    A,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
):
    """Split-CTM with implicit differentiation at fixed point.

    Forward: run split-CTM to convergence.
    Backward: solve ``(I - J^T) lambda = g`` for the VJP via GMRES.

    Args:
        A:          iPEPS site tensor (DenseTensor or SymmetricTensor).
        chi:        Environment bond dimension.
        max_iter:   Maximum CTM iterations.
        conv_tol:   Convergence tolerance.
        chi_I:      Interlayer bond dimension.
        renormalize: Renormalize environment at each step.

    Returns:
        Converged SplitCTMTensorEnv.
    """
    return ctm_split_tensor(A, chi, max_iter, conv_tol, chi_I, renormalize)


def ctm_split_tensor_converge_explicit(
    A,
    chi: int,
    max_iter: int = 100,
    chi_I: int | None = None,
    renormalize: bool = True,
    num_steps: int | None = None,
    warmup_steps: int = 0,
):
    """Split-CTM with explicit (unrolled) autodiff.

    Same warmup/backprop structure as ``ctm_tensor_converge_explicit`` but
    for the split-CTM (Tensor) path: runs *warmup_steps* sweeps under
    ``stop_gradient``, then *num_steps* fully differentiable sweeps.

    Args:
        A:              iPEPS site tensor.
        chi:            Environment bond dimension.
        max_iter:       Default backprop steps if ``num_steps`` is None.
        chi_I:          Interlayer bond dimension (default: ``chi``).
        renormalize:    Renormalize environment at each step.
        num_steps:      Backprop iterations (overrides ``max_iter``).
        warmup_steps:   Warmup iterations wrapped in ``stop_gradient``.

    Returns:
        Converged SplitCTMTensorEnv.
    """
    from tenax.algorithms._split_ctm_tensor_init import (
        initialize_split_ctm_tensor_env,
    )

    if chi_I is None:
        chi_I = chi

    env = initialize_split_ctm_tensor_env(A, chi, chi_I)

    for _ in range(warmup_steps):
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)
    if warmup_steps > 0:
        env = jax.tree.map(jax.lax.stop_gradient, env)

    n = num_steps if num_steps is not None else max_iter
    for _ in range(n):
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)

    return env
