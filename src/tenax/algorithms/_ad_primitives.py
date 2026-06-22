"""Protocol-level AD primitives shared by CTM and other algorithms.

These pieces are deliberately CTM-agnostic so that low-level CTM modules
(``_ctm_projector`` etc.) can depend on them without pulling the broader
``ad_utils`` module — which would re-introduce the
``algorithms → linalg → contraction → algorithms`` SCC.

What lives here:
  - :class:`CTMRGGradientError` — sentinel exception.
  - :func:`_fix_svd_signs` — gauge-fix SVD vectors.
  - :func:`truncated_svd_ad` (+ helpers) — Lorentzian-regularized truncated SVD.
  - :func:`truncated_svd_ad_vh_only` — half-SVD variant returning ``(s, Vh)``.
  - :func:`regularized_svd` — full-SVD wrapper with regularized backward.
  - :func:`regularized_eigh` — Hermitian eigendecomposition with regularized
    backward.
  - :func:`truncated_svd_symmetric_ad` — SymmetricTensor wrapper around the
    dense regularized SVD.

Reference: Francuz et al., Phys. Rev. Research 7, 013237 (2025).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class CTMRGGradientError(RuntimeError):
    """Raised when the CTM adjoint system is non-contractive (rho(J^T) >= 1)."""

    def __init__(self, spectral_radius: float):
        self.spectral_radius = spectral_radius
        super().__init__(
            f"CTM adjoint non-contractive: spectral radius {spectral_radius:.4f} >= 1.0"
        )


# ---------------------------------------------------------------------------
# 0. SVD sign-fixing helper
# ---------------------------------------------------------------------------


def _fix_svd_signs(
    U: jax.Array, s: jax.Array, Vh: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fix SVD gauge: rotate each singular vector so max-|U| element is real-positive.

    For each column j of U, find the row with the largest absolute value and
    multiply both U[:, j] and Vh[j, :] by the sign (or phase for complex)
    that makes that element real-positive.

    Reference: YASTN fix_svd_signs, variPEPS gauge_fixed_svd.
    """
    max_idx = jnp.argmax(jnp.abs(U), axis=0)  # shape (k,)
    signs = U[max_idx, jnp.arange(U.shape[1])]
    phases = jnp.where(jnp.abs(signs) > 0, signs / jnp.abs(signs), 1.0)
    U = U * jnp.conj(phases)[None, :]
    Vh = Vh * jnp.conj(phases)[:, None]
    return U, s, Vh


def _zero_subrank_singular_values(
    s_trunc: jax.Array,
    s_full: jax.Array,
    k: int,
    rel_tol: float = 1e-12,
) -> jax.Array:
    """Zero out singular values below ``rel_tol * s_full[0]`` AND boundary multiplets.

    Combines two pruning rules:

    1. Rank-aware: any ``s_trunc[i]`` with ``s_trunc[i] < rel_tol * s_full[0]``
       is set to exactly 0. Preserves the static chi output dim (s_trunc.shape
       stays ``(k,)``) but marks rank-deficient modes for the backward to treat
       as discarded.
    2. Multiplet-aware boundary: when ``k < s_full.shape[0]`` and the chi cut
       lands inside a near-degenerate group, zero out any kept member matching
       ``s_full[k-1]`` within tolerance. (Original Tenax behaviour, preserved.)

    Returns ``s_trunc`` with the appropriate slots set to 0.
    """
    abs_floor = rel_tol * (s_full[0] + 1e-30)
    s_trunc = jnp.where(s_trunc < abs_floor, 0.0, s_trunc)

    if k < s_full.shape[0]:
        boundary_val = s_full[k - 1]
        next_val = s_full[k]
        gap = boundary_val - next_val
        mult_threshold = 1e-6 * (s_full[0] + 1e-30)
        is_in_split_multiplet = (gap < mult_threshold) & (
            jnp.abs(s_trunc - boundary_val) < mult_threshold
        )
        s_trunc = jnp.where(is_in_split_multiplet, 0.0, s_trunc)

    return s_trunc


# ---------------------------------------------------------------------------
# 1. Truncated SVD with stable backward pass
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def truncated_svd_ad(
    M: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Truncated SVD with correct and stable backward pass.

    Forward: standard SVD truncated to *chi* singular values.
    Backward: Lorentzian-regularized F-matrix + truncation correction.

    Args:
        M:   2-D matrix of shape ``(m, n)``.
        chi: Number of singular values/vectors to keep.

    Returns:
        ``(U, s, Vh)`` truncated to *chi*.
    """
    U, s_full, Vh = jnp.linalg.svd(M, full_matrices=False)
    k = min(chi, s_full.shape[0])
    s_trunc = s_full[:k]
    s_trunc = _zero_subrank_singular_values(s_trunc, s_full, k)

    U, s_trunc, Vh = U[:, :k], s_trunc, Vh[:k, :]
    U, s_trunc, Vh = _fix_svd_signs(U, s_trunc, Vh)
    return U, s_trunc, Vh


def _truncated_svd_ad_fwd(
    M: jax.Array,
    chi: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple]:
    """Forward pass — store full SVD for backward."""
    U_full, s_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
    U_full, s_full, Vh_full = _fix_svd_signs(U_full, s_full, Vh_full)
    k = min(chi, s_full.shape[0])
    s_trunc = s_full[:k]
    s_trunc = _zero_subrank_singular_values(s_trunc, s_full, k)

    U = U_full[:, :k]
    Vh = Vh_full[:k, :]
    # Store the *unmodified* full SVD in residuals for the backward pass
    residuals = (U_full, s_full, Vh_full, M, k)
    return (U, s_trunc, Vh), residuals


def _svd_sector_backward(
    U: jax.Array,
    s: jax.Array,
    Vh: jax.Array,
    dU: jax.Array,
    ds: jax.Array,
    dVh: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Lorentzian-regularized SVD backward for one dense matrix sector.

    Computes the gradient of the input matrix from the gradients of the
    truncated SVD factors, using the Lorentzian broadening from Francuz
    et al., PRR 7, 013237 to regularize degenerate singular values.

    *U*, *s*, *Vh* are the **full** (untruncated) SVD factors of the
    sector matrix.  *dU*, *ds*, *dVh* are the incoming gradients,
    which may be truncated to *k* values (``k <= len(s)``).

    Args:
        U:   Left singular vectors, shape ``(m, p)`` where ``p = min(m, n)``.
        s:   Singular values, shape ``(p,)``.
        Vh:  Right singular vectors, shape ``(p, n)``.
        dU:  Gradient w.r.t. truncated U, shape ``(m, k)``.
        ds:  Gradient w.r.t. truncated s, shape ``(k,)``.
        dVh: Gradient w.r.t. truncated Vh, shape ``(k, n)``.
        eps: Lorentzian broadening parameter.

    Returns:
        dM: Gradient w.r.t. the input matrix, shape ``(m, n)``.
    """
    k = ds.shape[0]
    m = U.shape[0]
    n = Vh.shape[1]

    # Kept subspace
    U_k = U[:, :k]
    s_k = s[:k]
    V_k = Vh[:k, :].conj().T  # (n, k)

    # Rank-aware F-matrix mask: zero F[i, j] where either sigma_i or sigma_j
    # is below the rank threshold. The unregularized F entry for (sigma>0,
    # sigma=0) pairs evaluates to ~1/sigma^2 — a gauge artifact that pumps
    # arbitrary upstream cotangent components on the kept-but-zero columns
    # of U/Vh into the gradient. Masking F drops those contributions while
    # leaving the well-defined sigma>0 / sigma>0 entries unchanged.
    #
    # NOTE: we deliberately do NOT mask U_k, V_k, or proj_U_perp. Doing so
    # would change the discarded-subspace projector for full-SVD on rank-
    # deficient inputs (square unitary U has U @ U^T = I → proj_U_perp = 0
    # natively; masking would make it nonzero and route upstream null-space
    # dU components through 1/sigma_kept, producing a wrong answer that
    # disagrees with finite-difference gradients).
    eps_rank = 1e-12 * jnp.maximum(s[0], 1e-30)
    keep_mask = (s_k > eps_rank).astype(s.dtype)  # (k,) -- 1.0 or 0.0

    # --- Lorentzian-regularized F-matrix ---
    s2 = s_k**2
    diff = s2[:, None] - s2[None, :]
    F = diff / (diff**2 + eps**2)
    F = F - jnp.diag(jnp.diag(F))
    F = F * keep_mask[:, None] * keep_mask[None, :]

    # Antisymmetric parts of projected cotangents
    UtdU = U_k.conj().T @ dU  # (k, k)
    VtdV = V_k.conj().T @ dVh.conj().T  # (k, k)
    UtdU_anti = 0.5 * (UtdU - UtdU.conj().T)
    VtdV_anti = 0.5 * (VtdV - VtdV.conj().T)

    # Inverse singular values — sanitize input so JAX backward never
    # evaluates 1/0 (jnp.where evaluates both branches during AD).
    s_safe = jnp.where(s_k > eps, s_k, 1.0)
    s_inv = jnp.where(s_k > eps, 1.0 / s_safe, 0.0)

    # Projectors onto complements of kept subspaces (UNMASKED — see note above)
    proj_U_perp = jnp.eye(m) - U_k @ U_k.conj().T
    proj_V_perp = jnp.eye(n) - V_k @ V_k.conj().T

    # Assemble gradient (Wan & Narayanan 2023 / Francuz et al.):
    Vh_k = Vh[:k, :]
    dM = jnp.zeros((m, n), dtype=U.dtype)

    # 1. Diagonal part from ds
    dM = dM + U_k @ jnp.diag(ds) @ Vh_k

    # 2. Off-diagonal from dU (within kept subspace)
    dM = dM + U_k @ (F * UtdU_anti) @ jnp.diag(s_k) @ Vh_k

    # 3. Off-diagonal from dVh (within kept subspace)
    dM = dM + U_k @ jnp.diag(s_k) @ (F * VtdV_anti) @ Vh_k

    # 4. Truncation correction from dU (kept-truncated coupling)
    dM = dM + proj_U_perp @ dU @ jnp.diag(s_inv) @ Vh_k

    # 5. Truncation correction from dVh (kept-truncated coupling)
    dM = dM + U_k @ jnp.diag(s_inv) @ dVh @ proj_V_perp

    return dM


def _truncated_svd_ad_bwd(
    chi: int,
    residuals: tuple,
    g: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array]:
    """Backward pass with Lorentzian regularization and truncation term.

    Implements the stable SVD adjoint from Francuz et al. PRR 7, 013237:
    - Lorentzian broadening ``s_i^2 - s_j^2 / ((s_i^2-s_j^2)^2 + eps^2)``
      prevents divergences from degenerate singular values.
    - Full truncation correction accounts for coupling between kept and
      discarded subspaces (the dominant error source identified by Francuz
      et al.).
    """
    U_full, s_full, Vh_full, M, k = residuals
    dU, ds, dVh = g
    dM = _svd_sector_backward(U_full, s_full, Vh_full, dU, ds, dVh)
    return (dM,)


truncated_svd_ad.defvjp(_truncated_svd_ad_fwd, _truncated_svd_ad_bwd)


# ---------------------------------------------------------------------------
# 1a-half. Half-SVD: only S and Vh (no U), saves memory in backward
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def truncated_svd_ad_vh_only(
    M: jax.Array,
    chi: int,
) -> tuple[jax.Array, jax.Array]:
    """Truncated SVD returning only ``(s, Vh)`` — U is discarded.

    Uses the same Lorentzian-regularized backward as ``truncated_svd_ad``
    but with ``dU=0``, saving memory by not storing U in residuals.
    """
    U, s_full, Vh = jnp.linalg.svd(M, full_matrices=False)
    k = min(chi, s_full.shape[0])
    s_trunc = s_full[:k]
    s_trunc = _zero_subrank_singular_values(s_trunc, s_full, k)

    _, s_trunc, Vh = _fix_svd_signs(U[:, :k], s_trunc, Vh[:k, :])
    return s_trunc, Vh


def _truncated_svd_ad_vh_only_fwd(M, chi):
    """Forward pass — store full SVD for backward but return only (s, Vh)."""
    U_full, s_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
    U_full, s_full, Vh_full = _fix_svd_signs(U_full, s_full, Vh_full)
    k = min(chi, s_full.shape[0])
    s_trunc = s_full[:k]
    s_trunc = _zero_subrank_singular_values(s_trunc, s_full, k)

    Vh = Vh_full[:k, :]
    residuals = (U_full, s_full, Vh_full, M, k)
    return (s_trunc, Vh), residuals


def _truncated_svd_ad_vh_only_bwd(chi, residuals, g):
    """Backward pass with dU=0 (U was not used)."""
    U_full, s_full, Vh_full, M, k = residuals
    ds, dVh = g
    dU = jnp.zeros((M.shape[0], k), dtype=M.dtype)
    dM = _svd_sector_backward(U_full, s_full, Vh_full, dU, ds, dVh)
    return (dM,)


truncated_svd_ad_vh_only.defvjp(
    _truncated_svd_ad_vh_only_fwd, _truncated_svd_ad_vh_only_bwd
)


# ---------------------------------------------------------------------------
# 1a-bis. Full SVD with regularized backward (used by QR projectors)
# ---------------------------------------------------------------------------


@partial(jax.custom_vjp)
def regularized_svd(M: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Full SVD with Lorentzian-regularized backward pass.

    Same as ``jnp.linalg.svd(M, full_matrices=False)`` but the VJP uses
    the Lorentzian-regularized F-matrix to prevent NaN gradients from
    degenerate singular values.

    Returns a plain ``(U, s, Vh)`` tuple (not ``SVDResult``).
    """
    result = jnp.linalg.svd(M, full_matrices=False)
    return _fix_svd_signs(result.U, result.S, result.Vh)


def _regularized_svd_fwd(M):
    result = jnp.linalg.svd(M, full_matrices=False)
    U, s, Vh = _fix_svd_signs(result.U, result.S, result.Vh)
    return (U, s, Vh), (U, s, Vh)


def _regularized_svd_bwd(residuals, g):
    U, s, Vh = residuals
    dU, ds, dVh = g
    dM = _svd_sector_backward(U, s, Vh, dU, ds, dVh)
    return (dM,)


regularized_svd.defvjp(_regularized_svd_fwd, _regularized_svd_bwd)


# ---------------------------------------------------------------------------
# 1a-quater. Thin QR with regularized backward (used by QR-CTMRG projectors)
# ---------------------------------------------------------------------------

# Floor applied to ``diag(R)`` in the backward triangular solve. This is the
# only place where ``diag(R) -> 0`` (a fully truncated bond) would divide by
# zero and produce NaN gradients; flooring keeps the backward finite without
# changing the well-conditioned answer (the floor only activates when
# ``|diag(R)| < _R_FLOOR``).
_R_FLOOR = 1e-12


def _qr_H(X):
    """Conjugate transpose."""
    return X.conj().T


@partial(jax.custom_vjp)
def regularized_qr(M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Thin QR with a backward stable through rank-deficient bonds.

    Same forward as ``jnp.linalg.qr(M)`` (reduced/thin mode), but the VJP
    floors ``diag(R)`` below ``_R_FLOOR`` in the backward triangular solve so
    gradients stay finite when a bond is near- or exactly rank-deficient.
    Raw ``jnp.linalg.qr``'s VJP divides by ``diag(R)`` and produces NaN when a
    bond is fully truncated.

    This is the *real* branch only: CTM projector matrices are real, so the
    complex diagonal correction in JAX's thin-QR JVP is intentionally omitted.

    Returns:
        ``(Q, R)`` as in ``jnp.linalg.qr(M)``.  Returned as a plain ``tuple``
        (not the ``QRResult`` namedtuple of newer ``jnp.linalg.qr``) so the
        primal output's pytree structure matches the ``custom_vjp`` fwd rule.
    """
    Q, R = jnp.linalg.qr(M)
    return (Q, R)


def _regularized_qr_fwd(M):
    Q, R = jnp.linalg.qr(M)
    return (Q, R), (Q, R)


def _regularized_qr_bwd(residuals, g):
    # Reverse-mode VJP of the thin QR M = Q R (m >= n), obtained by transposing
    # JAX's own thin-QR JVP rule (real branch). Verified to machine precision
    # (5e-16) against jax.vjp(jnp.linalg.qr, .) for square AND tall real
    # matrices in the Task 1 spike (#570).
    #
    #   P     = R̄ Rᴴ
    #   S     = Qᴴ Q̄
    #   under = S − P
    #   B̄     = (P − S) + tril(under − underᴴ, −1)
    #   Ā     = Q̄ + Q B̄
    #   M̄     = Ā R⁻ᴴ          ← regularized triangular solve (floor diag R)
    #
    # The R⁻ᴴ solve is the only place diag(R)→0 bites; flooring diag(R) there
    # keeps M̄ finite near rank-deficiency without changing the well-conditioned
    # answer (floor only activates when |diag(R)| < _R_FLOOR).
    Q, R = residuals
    dQ, dR = g
    P = dR @ _qr_H(R)
    S = _qr_H(Q) @ dQ
    under = S - P
    Bbar = (P - S) + jnp.tril(under - _qr_H(under), -1)
    Abar = dQ + Q @ Bbar
    d = jnp.diag(R)
    safe = jnp.where(jnp.abs(d) > _R_FLOOR, d, _R_FLOOR)
    R_reg = R - jnp.diag(d) + jnp.diag(safe)
    # M̄ = Ā R⁻ᴴ  ⟺  M̄ Rᴴ = Ā  ⟺  R M̄ᴴ = Āᴴ  (upper-tri solve in R).
    dM = _qr_H(
        jax.scipy.linalg.solve_triangular(R_reg, _qr_H(Abar), lower=False)
    )
    return (dM,)


regularized_qr.defvjp(_regularized_qr_fwd, _regularized_qr_bwd)


def truncated_lowrank_svd(
    M: jax.Array, k: int, *, oversample: int = 8, n_power_iterations: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """AD-stable top-*k* SVD, fast when ``rank(M) <= k`` and ``M`` is large.

    For a matrix whose numerical rank is ``<= k`` (e.g. the 2x2 CTM projector
    half-systems, which are exactly rank-χ), this returns the same top-*k*
    ``(U, s, Vh)`` as a full ``jnp.linalg.svd`` truncated to *k*, but avoids the
    full ``min(m, n)``-sized SVD: a randomized range finder reduces the problem
    to a stable truncated SVD of a small ``(k+oversample, n)`` matrix.

    Forward: ``Y = M Ω`` (deterministic Ω) → ``Q = qr(Y)`` → ``B = Qᴴ M`` →
    ``truncated_svd_ad(B, k)`` → ``U = Q U_B``. Backward inherits the stable
    Lorentzian VJP of :func:`truncated_svd_ad` on the small ``B`` plus the
    :func:`regularized_qr` VJP — far better-conditioned than a full-SVD VJP at
    ``χD²`` size. Exact (to round-off) whenever ``rank(M) <= k``.

    ``n_power_iterations`` (subspace iteration, re-orthogonalized each step)
    sharpens the captured subspace when the spectrum *decays* rather than having
    a sharp rank cliff; 0 is exact for a genuinely rank-``<= k`` matrix.

    Relationship to :func:`tenax.linalg.rsvd` / ``_rsvd_matrix``: same HMT 2011
    randomized SVD. This is the **AD-stable, layering-local** counterpart, kept
    separate for two reasons: (1) it sits on the CTM fixed-point AD backward, so
    it routes through :func:`regularized_qr` and :func:`truncated_svd_ad`
    (Lorentzian-regularized VJPs) — the plain ``jnp.linalg.svd`` VJP that ``rsvd``
    relies on produces NaN/blow-up through near-degenerate singular values (cf.
    #570); and (2) ``_ad_primitives`` deliberately does not import
    ``tenax.linalg`` (it would re-introduce the
    ``algorithms → linalg → contraction → algorithms`` SCC, see the module
    docstring), so it cannot call ``_rsvd_matrix`` directly. It also handles
    complex ``M`` (``conjᵀ`` vs ``rsvd``'s real-only ``.T``).

    Ω uses a fixed seed (a compile-time constant → no gradient, reproducible).
    Falls back to ``truncated_svd_ad(M, k)`` when ``M`` is already small
    (``k + oversample >= min(m, n)``), where the range finder gives no benefit.
    """
    m, n = M.shape
    ell = k + oversample
    if ell >= min(m, n):
        return truncated_svd_ad(M, k)
    omega = jax.random.normal(jax.random.PRNGKey(0), (n, ell), dtype=M.dtype)
    q, _r = regularized_qr(M @ omega)  # (m, ell)
    for _ in range(n_power_iterations):
        q, _r = regularized_qr(M.conj().T @ q)  # (n, ell)
        q, _r = regularized_qr(M @ q)  # (m, ell)
    b = q.conj().T @ M  # (ell, n)
    u_b, s, vh = truncated_svd_ad(b, k)  # (ell, k), (k,), (k, n)
    u = q @ u_b  # (m, k)
    return u, s, vh


# ---------------------------------------------------------------------------
# 1a-ter. Symmetric eigendecomposition with regularized backward pass
# ---------------------------------------------------------------------------

_EIGH_LORENTZ_EPS = 1e-12
_EIGH_LORENTZ_REL = (
    1e-7  # relative broadening: eps = max(_EIGH_LORENTZ_EPS, _REL * max|w|)
)


@partial(jax.custom_vjp)
def regularized_eigh(M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Symmetric eigendecomposition with Lorentzian-regularized backward.

    Same as ``jnp.linalg.eigh(M)`` but the VJP uses Lorentzian broadening
    to prevent NaN gradients from degenerate eigenvalues.

    Returns:
        ``(eigenvalues, eigenvectors)`` sorted ascending.
    """
    w, v = jnp.linalg.eigh(M)
    return w, v


def _regularized_eigh_fwd(M):
    w, v = jnp.linalg.eigh(M)
    return (w, v), (w, v)


def _regularized_eigh_bwd(residuals, g):
    """Lorentzian-regularized eigh backward.

    Standard eigh backward: F_ij = 1/(lambda_i - lambda_j) diverges for
    degenerate eigenvalues.
    Regularized: F_ij = (lambda_i - lambda_j) / ((lambda_i - lambda_j)^2 + eps^2)
    """
    w, v = residuals
    dw, dv = g
    # Adaptive Lorentzian broadening: scale with largest |w| to cap F at ~1/eps.
    # Hardcoded eps=1e-12 broadcast gradient spikes to O(1e12) when two
    # eigenvalues of the projector density matrix were numerically degenerate;
    # scaling eps with max|w| caps F at O(_EIGH_LORENTZ_REL^-1) and preserves
    # directional information for non-degenerate modes (issue #299).
    w_scale = jnp.maximum(jnp.max(jnp.abs(w)), _EIGH_LORENTZ_EPS)
    eps = jnp.maximum(_EIGH_LORENTZ_EPS, _EIGH_LORENTZ_REL * w_scale)

    # Lorentzian-regularized F-matrix.
    # JAX reverse-mode convention for symmetric eigh wants
    #   F_ij = 1 / (w_j - w_i)
    # so ``diff[i, j] = w_j - w_i``.  The previous formulation used
    # ``w_i - w_j`` which flipped the sign of the backward (issue #316).
    diff = w[None, :] - w[:, None]
    F = diff / (diff**2 + eps**2)
    F = F - jnp.diag(jnp.diag(F))  # zero diagonal

    # Backward: dM = V (diag(dw) + F * (V^T dV)) V^T
    inner = v.conj().T @ dv
    dM = v @ (jnp.diag(dw) + F * inner) @ v.conj().T

    # Symmetrize output (input was symmetric)
    dM = 0.5 * (dM + dM.conj().T)
    return (dM,)


regularized_eigh.defvjp(_regularized_eigh_fwd, _regularized_eigh_bwd)


# ---------------------------------------------------------------------------
# 1b. Truncated SVD with stable backward for SymmetricTensor
# ---------------------------------------------------------------------------


def truncated_svd_symmetric_ad(
    M,
    left_labels,
    right_labels,
    chi: int,
    new_bond_label: str = "bond",
):
    """Truncated SVD of a SymmetricTensor with Lorentzian-regularized backward.

    This is a convenience wrapper that densifies the SymmetricTensor, applies
    :func:`truncated_svd_ad` on the dense matrix, and reconstructs the result
    as a ``DenseTensor``.  The Lorentzian regularization from Francuz et al.
    (PRR 7, 013237) prevents NaN gradients from degenerate singular values
    within charge sectors.

    While a native per-sector custom_vjp would be more efficient (avoiding
    the dense round-trip), this implementation is correct and much simpler.
    The round-trip cost is acceptable for the moderate tensor sizes used in
    current AD iPEPS calculations.

    Args:
        M:               SymmetricTensor (or DenseTensor) to decompose.
        left_labels:     Labels forming the left (U) factor.
        right_labels:    Labels forming the right (Vh) factor.
        chi:             Number of singular values to keep.
        new_bond_label:  Label for the new virtual bond.

    Returns:
        ``(U, s, Vh)`` where U and Vh are ``DenseTensor`` objects and
        s is a 1-D ``jax.Array`` of length ``min(chi, min(m, n))``.
    """
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.tensor import DenseTensor

    # Resolve label ordering: left_labels then right_labels
    all_labels = list(left_labels) + list(right_labels)
    all_indices = []
    perm = []
    current_labels = M.labels()
    for lbl in all_labels:
        ax = current_labels.index(lbl)
        perm.append(ax)
        all_indices.append(M.indices[ax])

    # Densify and permute to (left_labels..., right_labels...)
    dense = M.todense()
    dense = jnp.transpose(dense, perm)

    # Reshape to matrix
    left_shape = tuple(dense.shape[i] for i in range(len(left_labels)))
    right_shape = tuple(
        dense.shape[i] for i in range(len(left_labels), len(all_labels))
    )
    m = 1
    for s in left_shape:
        m *= s
    n = 1
    for s in right_shape:
        n *= s
    matrix = dense.reshape(m, n)

    # Apply regularized SVD
    U_mat, s_vals, Vh_mat = truncated_svd_ad(matrix, chi)
    k = s_vals.shape[0]

    # Reshape back to tensor form
    U_data = U_mat.reshape(left_shape + (k,))
    Vh_data = Vh_mat.reshape((k,) + right_shape)

    # Build indices for the output tensors
    left_indices = tuple(all_indices[i] for i in range(len(left_labels)))
    right_indices = tuple(
        all_indices[i] for i in range(len(left_labels), len(all_labels))
    )

    # Determine symmetry from input (if available)
    sym = M.indices[0].symmetry
    bond_charges = np.zeros(k, dtype=np.int32)
    bond_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    U_tensor = DenseTensor(U_data, left_indices + (bond_out,))
    Vh_tensor = DenseTensor(Vh_data, (bond_in,) + right_indices)

    return U_tensor, s_vals, Vh_tensor
