"""Isometric CTM projector for the honeycomb lattice.

Per architectural decision **P1** (parallel module families), this module
mirrors the patterns of :mod:`tenax.algorithms._ctm_projector` WITHOUT
importing from it.  The square / split CTM projector handles a rank-4
"grown corner" with a 4-leg fused convention; the honeycomb projector
handles a rank-3 "boundary" with the fused-flow convention established
by :func:`tenax.algorithms._ctm_honeycomb_init._double_layer_honeycomb`.

Patterns mirrored from ``_ctm_projector.py``
--------------------------------------------

* ``S_safe`` clamp for ``1/sqrt(0)`` protection during AD::

      S_safe  = jnp.where(mask, S, 1.0)
      S_rsqrt = jnp.where(mask, 1.0 / jnp.sqrt(S_safe), 0.0)

  ``method='eigh'`` returns the plain isometric form and does **not**
  apply ``S_rsqrt`` to the projector itself, but the same NaN-safety
  logic is still used to guard the eigh forward.  ``method='svd'``
  returns the Fishman two-projector form where ``S_rsqrt`` IS load-
  bearing — both ``P1`` and ``P2`` carry an ``S^{-1/2}`` factor (see
  "isometric vs. Fishman" below).

* Phase-fix gauge wrapped in :func:`jax.lax.stop_gradient` so the gauge
  factor is treated as a constant by the optimizer.  The phase choice
  follows variPEPS' first-element-above-threshold convention (see
  ``_phase_fix_ctm_tensor`` in ``ad_utils.py``).

* Output isometric pair ``(P, P_dagger)`` returned as :class:`Tensor`
  objects with proper :class:`TensorIndex` metadata (labels + flows),
  not raw ``jax.Array``.

* Forward-only ``stop_gradient`` on the projector tensors themselves,
  matching the dense-fallback convention at ``_ctm_projector.py:1067``.
  Implicit-AD callers (Task 13) handle the backward through the CTM
  fixed point separately, so this is safe.

Patterns written fresh (no analog in ``_ctm_projector.py``)
-----------------------------------------------------------

* Rank-3 boundary handling — the input has a single ``(chi, d2, chi)``
  layout instead of the rank-4 grown-corner layout.  The ``(chi_in, d2)``
  pair is *jointly* truncated to a single ``chi_new`` axis.

* Output ``P_dagger`` is materialized as its own :class:`Tensor` (with
  flipped flows on the shared legs) rather than asking the caller to
  call ``.dagger()``.  Task 6 needs the explicit pair for the absorb
  step, and producing both here keeps the gauge / phase choice
  consistent across the two outputs.

Boundary input convention
-------------------------

A rank-3 :class:`Tensor` with labels ``(chi_in_label, e_alpha_d2_label,
chi_out_label)`` and flows ``(IN, IN, OUT)`` — i.e. exactly the layout
emitted by :func:`initialize_honeycomb_env` for ``L_alpha`` /
``R_alpha``.  The fused virtual-leg flow is ``IN`` (matches the bra
direction of the double-layer fused legs).  The first axis is the chi
axis to be truncated; the third axis is the "column" axis used to build
the density matrix.

Output projector pair
---------------------

The two return values are ``(P_or_P1, P_dagger_or_P2)``; their meaning
depends on ``method``.

``method='eigh'`` -- plain isometric pair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``P``        : labels ``(chi_in_label, e_alpha_d2_label, chi_new_in_label)``
                 flows  ``(IN,             IN,              OUT)``
                 dim    ``(chi_in,         d2,              chi_new)``

* ``P_dagger`` : labels ``(chi_new_out_label, chi_in_label, e_alpha_d2_label)``
                 flows  ``(IN,                OUT,          OUT)``
                 dim    ``(chi_new,           chi_in,       d2)``

where ``chi_new = min(chi, chi_in * d2)``.  Contracting ``P_dagger``
with ``P`` over the shared ``(chi_in, d2)`` axes gives the identity on
the new chi axis::

    einsum("kab,abm->km", P_dag.todense(), P.todense()) == eye(chi_new)

``method='svd'`` -- Fishman two-projector pair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``P1`` : labels ``(chi_in_label, e_alpha_d2_label, chi_new_in_label)``
           flows  ``(IN,             IN,              OUT)``
           dim    ``(chi_in,         d2,              chi_new)``

* ``P2`` : labels ``(chi_out_label, chi_new_out_label)``
           flows  ``(IN,            IN)``
           dim    ``(chi_out,       chi_new)``

P1 and P2 do **not** share their non-truncated legs.  The Fishman
identity is

    einsum("abk,abc,cm->km", conj(P1), boundary, P2) == eye(chi_new)

i.e. ``P1^dagger @ M @ P2 = I_chi_new`` where ``M`` is the matricised
boundary.  This matches Tenax's existing convention in
``_ctm_projector.py:_svd_projector_symmetric`` (P1 carries the left
singular vectors and S^{-1/2}; P2 carries the right singular vectors
and S^{-1/2}; both share the truncated ``chi_new`` axis).

Isometric vs. Fishman two-projector
-----------------------------------

* ``method='eigh'`` -- plain isometric: ``P = U``, ``P_dagger = U^dagger``
  with ``P_dagger @ P = I``.  ``S_rsqrt`` is computed but not applied.

* ``method='svd'`` -- Fishman: ``P1 = U S^{-1/2}``, ``P2 = V S^{-1/2}``
  with built-in ``S^{-1/2}`` weighting on both sides.  The
  ``S_safe / S_rsqrt`` clamp is load-bearing for this path (zeroing
  ``S_rsqrt`` for sub-cutoff singular values is what regularises the
  projector against rank-deficient boundaries).

Deferred (raises :class:`NotImplementedError`)
----------------------------------------------

* ``method='biorthogonal'`` — the two-projector biorthogonal variant
  (Paper 2 §II.C / Corboz 2014, Fishman et al. PRB 98 235148) is
  deferred to a follow-up.  See
  ``docs/plans/2026-04-25-honeycomb-ctm-design.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, Tensor
from tenax.linalg import _dense_svd

__all__ = [
    "compute_honeycomb_corner_biorthogonal_projector",
    "compute_honeycomb_projector",
]

IN = FlowDirection.IN
OUT = FlowDirection.OUT

_S_SAFE_EPS = 1e-12
_PHASE_EPS = 0.1  # variPEPS first-above-threshold fraction-of-max default


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _column_phase(U: jax.Array, eps_frac: float = _PHASE_EPS) -> jax.Array:
    """Return the per-column unit-modulus phase used by the gauge fix.

    For each column ``U[:, j]``, find the first row ``i`` whose
    ``|U[i, j]| >= eps_frac * max_i |U[i, j]|`` and return
    ``U[i, j] / |U[i, j]|``.  The factor is wrapped in
    :func:`jax.lax.stop_gradient` since it represents a gauge degree of
    freedom (mirrors the ``stop_gradient`` of the QR phase factor at
    ``_ctm_projector.py:953-964``).

    Args:
        U:        Matrix of shape ``(m, k)``.
        eps_frac: Fraction-of-max threshold for picking the reference
                  row.  Matches ``EPS_PHASE = 0.1`` in
                  :func:`ad_utils._phase_fix_ctm_tensor`.

    Returns:
        Per-column phase array of shape ``(k,)`` with unit modulus
        (or 1.0 for empty columns).  Apply with ``U * conj(phase)`` to
        make the reference row real-positive.
    """
    if U.shape[1] == 0:
        return jnp.ones((0,), dtype=U.dtype)
    abs_U = jnp.abs(U)
    abs_max = jnp.max(abs_U, axis=0, keepdims=True)  # (1, k)
    threshold = eps_frac * abs_max
    mask = abs_U >= threshold  # first True per column = reference row
    first_idx = jnp.argmax(mask.astype(jnp.int32), axis=0)  # (k,)
    rows = jnp.take_along_axis(U, first_idx[None, :], axis=0)[0]  # (k,)
    abs_rows = jnp.abs(rows)
    phase = jnp.where(
        abs_rows > 0,
        rows / jnp.where(abs_rows > 0, abs_rows, 1.0),
        jnp.ones_like(rows),
    )
    return jax.lax.stop_gradient(phase)


def _phase_fix_columns(U: jax.Array, eps_frac: float = _PHASE_EPS) -> jax.Array:
    """Apply the variPEPS first-above-threshold per-column phase fix.

    Convenience wrapper around :func:`_column_phase` that applies the
    phase to ``U`` directly.  For the SVD path we need the phase
    separately so it can also be applied to the right singular vectors
    (preserving the ``M = U S V^dagger`` invariant under the gauge).
    """
    if U.shape[1] == 0:
        return U
    phase = _column_phase(U, eps_frac=eps_frac)
    return U * jnp.conj(phase)[None, :]


def _make_chi_new_index(
    template_idx: TensorIndex, k: int, label: str, flow: FlowDirection
) -> TensorIndex:
    """Build a trivial ``(0, 0, ..., 0)`` charge index of dim ``k``.

    The honeycomb env at v1 uses trivial U(1) charges throughout (see
    :func:`initialize_honeycomb_env`), so the new chi axis inherits that
    structure.  ``template_idx`` is consulted only for its symmetry
    object — its charges are not copied.
    """
    chi_charges = np.zeros(k, dtype=np.int32)
    return TensorIndex.from_charges(
        template_idx.symmetry, chi_charges, flow, label=label
    )


# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #


def compute_honeycomb_projector(
    boundary: Tensor,
    *,
    method: str = "eigh",
    chi: int,
    s_safe_eps: float = _S_SAFE_EPS,
    chi_new_in_label: str = "chi_new_in",
    chi_new_out_label: str = "chi_new_out",
) -> tuple[Tensor, Tensor]:
    """Honeycomb CTM projector, ``(chi_in, d2)`` -> ``chi_new``.

    Two output conventions, selected by ``method``:

    * ``method='eigh'`` -- plain isometric pair ``(P, P_dagger)`` where
      ``P_dagger @ P = I_chi_new`` (contraction over the shared
      ``(chi_in, d2)`` legs).  ``P = U`` (left singular vectors of the
      matricised boundary).

    * ``method='svd'`` -- Fishman two-projector pair ``(P1, P2)`` where
      ``P1^dagger @ M @ P2 = I_chi_new`` with ``M`` the matricised
      boundary.  ``P1 = U @ diag(S^{-1/2})``, ``P2 = V @ diag(S^{-1/2})``;
      both projectors carry an ``S^{-1/2}`` factor.  Matches the
      convention of :func:`_svd_projector_symmetric` in
      ``_ctm_projector.py``.

    Args:
        boundary:           Rank-3 :class:`Tensor` with labels ``(chi_in,
                            e_alpha_d2, chi_out)`` and flows
                            ``(IN, IN, OUT)`` (matches the layout of
                            ``L_alpha`` / ``R_alpha`` in
                            :class:`HoneycombCTMEnv`).
        method:             ``'eigh'`` or ``'svd'``.  ``'biorthogonal'``
                            raises :class:`NotImplementedError`.
        chi:                Target bond dimension for the new chi axis.
                            Output dim is ``min(chi, chi_in * d2,
                            chi_out)`` for ``svd`` (bounded by SVD rank)
                            and ``min(chi, chi_in * d2)`` for ``eigh``.
        s_safe_eps:         Floor for singular values used to clamp
                            ``S_safe`` / ``S_rsqrt`` against NaNs.  For
                            ``eigh`` this also floors the eigenvalues
                            before ``sqrt(S)``.
        chi_new_in_label:   Label assigned to the truncated axis on
                            ``P`` / ``P1``.
        chi_new_out_label:  Label assigned to the truncated axis on
                            ``P_dagger`` / ``P2``.

    Returns:
        ``(P, P_dagger)`` for ``method='eigh'`` or ``(P1, P2)`` for
        ``method='svd'`` -- see module docstring for shapes / labels /
        flows.

    Raises:
        NotImplementedError: ``method='biorthogonal'`` (deferred).
        ValueError:          unknown ``method`` value or wrong boundary
                             rank.
    """
    if method == "biorthogonal":
        raise NotImplementedError(
            "Corboz biorthogonal projectors operate on the rank-4 enlarged "
            "corner, not on a rank-3 row tensor. Use "
            ":func:`compute_honeycomb_corner_biorthogonal_projector` instead, "
            "which takes the upper- and lower-half (chi*D², chi*D²) matrices "
            "of the enlarged corner. See "
            "docs/plans/2026-04-25-honeycomb-ctm-design.md."
        )
    if method not in ("eigh", "svd"):
        raise ValueError(
            f"Unknown projector method: {method!r}; expected 'eigh', 'svd', "
            "or 'biorthogonal'."
        )
    if boundary.ndim != 3:
        raise ValueError(f"honeycomb boundary must be rank-3, got rank {boundary.ndim}")

    # Pull index metadata.  The convention is FROZEN: leg order is
    # (chi_in, e_alpha_d2, chi_out) — Task 6 must respect this.
    chi_in_idx, e_d2_idx, chi_out_idx = boundary.indices
    chi_in = chi_in_idx.dim
    d2 = e_d2_idx.dim
    chi_out = chi_out_idx.dim

    # ---------------------------------------------------------------- #
    # Build the matrix M : (chi_in * d2, chi_out) by reshaping the      #
    # boundary so that the (chi_in, d2) pair is jointly truncated to    #
    # the new chi axis.                                                  #
    # ---------------------------------------------------------------- #
    boundary_arr = boundary.todense()  # small: (chi, d2, chi)
    M = boundary_arr.reshape(chi_in * d2, chi_out)  # (chi_in*d2, chi_out)

    if method == "svd":
        # NOTE: this local gauge fix predates #751.  ``_fix_svd_signs`` used
        # to apply ``conj(phase)`` to BOTH U and Vh per column/row, which
        # preserves ``M = U S Vh`` only for real matrices (where
        # conj(phase)^2 = 1) and broke the SVD identity for complex ones.
        # #751 fixed the shared helper to use this same convention -- the
        # inverse ``phase`` on V -- so calling it here would now be correct.
        # Left inlined to keep a correctness fix free of refactoring.
        U_full, S_full, Vh_full = _dense_svd(M, full_matrices=False)
        # SVD truncation is bounded by the matrix rank: min(chi_in*d2, chi_out).
        k = min(chi, S_full.shape[0])
        U_k = U_full[:, :k]
        # V_k = (V^†)^† has shape (chi_out, k) — these are the right singular
        # vectors as columns, the natural form for Fishman P2.
        V_k = jnp.conj(Vh_full[:k, :]).T  # (chi_out, k)
        S_k = S_full[:k]
    else:  # eigh path
        # rho = M M^† has eigenvalues = S^2.  Symmetrise to kill roundoff
        # asymmetry before diagonalising.
        rho = M @ M.conj().T  # (chi_in*d2, chi_in*d2)
        rho = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = jnp.linalg.eigh(rho)
        # eigh returns ascending; flip to descending so the largest
        # eigenvalues come first (matches the dense fallback in
        # _ctm_projector.py and the SVD natural ordering).
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        k = min(chi, eigvals.shape[0])
        U_k = eigvecs[:, :k]
        # Clamp to >= s_safe_eps before sqrt — eigh on a PSD matrix can
        # produce tiny negative roundoff values that turn sqrt into NaN.
        # This is the same NaN-safe spirit as the canonical S_safe pattern
        # at _ctm_projector.py:727-732.
        eig_k_clamped = jnp.where(eigvals[:k] > s_safe_eps, eigvals[:k], 0.0)
        S_k = jnp.sqrt(eig_k_clamped)

    # ---------------------------------------------------------------- #
    # S_safe / S_rsqrt — canonical NaN-safe pattern from                #
    # _ctm_projector.py (lines 727-732, 894-898).                       #
    # For ``method='eigh'`` this is computed but not applied to the     #
    # projector (the eigh path returns plain isometric ``P = U``).      #
    # For ``method='svd'`` this IS load-bearing — both ``P1`` and       #
    # ``P2`` carry the ``S_rsqrt`` factor (Fishman two-projector form). #
    # ---------------------------------------------------------------- #
    if k > 0:
        cutoff = jnp.maximum(s_safe_eps, 1e-14 * (S_k[0] + 1e-30))
        mask = S_k > cutoff
        S_safe = jnp.where(mask, S_k, 1.0)
        S_rsqrt = jnp.where(mask, 1.0 / jnp.sqrt(S_safe), 0.0)
    else:
        S_rsqrt = jnp.zeros_like(S_k)

    # ---------------------------------------------------------------- #
    # Phase-fix BEFORE multiplying by S_rsqrt, so the gauge factor is   #
    # captured cleanly in stop_gradient and not entangled with the      #
    # data-dependent singular-value scaling.                            #
    #                                                                    #
    # For SVD, the SAME per-column phase that we apply to U_k must also #
    # be applied to V_k (i.e. ``U S V^dagger`` is gauge-invariant only  #
    # when both U and V absorb the same diagonal phase).  Phase-fixing  #
    # V_k independently would compute a *different* phase from V_k's    #
    # entries and break the Fishman identity ``P1^dagger M P2 = I``.     #
    # ---------------------------------------------------------------- #
    if method == "svd":
        phase_U = _column_phase(U_k)  # (k,) unit modulus
        U_k = U_k * jnp.conj(phase_U)[None, :]
        V_k = V_k * jnp.conj(phase_U)[None, :]
    else:
        U_k = _phase_fix_columns(U_k)

    # Build TensorIndex metadata for the new chi axes (shared between
    # both methods — Task 6 expects identical labels/flows).
    chi_new_in_idx = _make_chi_new_index(chi_in_idx, k, chi_new_in_label, OUT)
    chi_new_out_idx = _make_chi_new_index(chi_in_idx, k, chi_new_out_label, IN)

    if method == "svd":
        # Fishman: P1 = U @ diag(S^{-1/2}), P2 = V @ diag(S^{-1/2}).
        P1_arr = (U_k * S_rsqrt[None, :]).reshape(chi_in, d2, k)
        P2_arr = V_k * S_rsqrt[None, :]  # (chi_out, k)

        # Stop-gradient through the projector forward path (see eigh
        # branch below for rationale).
        P1_arr = jax.lax.stop_gradient(P1_arr)
        P2_arr = jax.lax.stop_gradient(P2_arr)

        P1 = DenseTensor(
            P1_arr,
            (
                chi_in_idx,  # IN
                e_d2_idx,  # IN
                chi_new_in_idx,  # OUT
            ),
        )
        # P2's chi_out-facing leg has flow IN so that ``boundary @ P2``
        # contracts the OUT chi_out leg of the boundary against P2's IN.
        P2 = DenseTensor(
            P2_arr,
            (
                chi_out_idx.flip_flow(),  # IN  (boundary's chi_out is OUT)
                chi_new_out_idx,  # IN  (matches eigh's P_dagger truncated leg)
            ),
        )
        return P1, P2

    # ---------------------------------------------------------------- #
    # eigh: plain isometric form. P = U, P_dagger = U^†; P^† P = I_k.   #
    # ---------------------------------------------------------------- #
    P_arr = U_k.reshape(chi_in, d2, k)
    P_dag_arr = jnp.conj(U_k).T.reshape(k, chi_in, d2)

    # Stop-gradient through the *projector* itself in the forward path,
    # matching the convention of the dense fallbacks in
    # _ctm_projector.py (lines 979 and 1067).  Implicit-AD callers
    # (Task 13) handle the backward through the CTM fixed point
    # separately, so this is the right default.
    P_arr = jax.lax.stop_gradient(P_arr)
    P_dag_arr = jax.lax.stop_gradient(P_dag_arr)

    P = DenseTensor(
        P_arr,
        (
            chi_in_idx,  # IN
            e_d2_idx,  # IN
            chi_new_in_idx,  # OUT
        ),
    )
    # P_dagger flow on the shared (chi_in, d2) legs is flipped so that
    # ``contract(P_dagger, P)`` matches the IN/OUT bra-ket convention
    # (mirrors how ``DenseTensor.dagger`` produces dual indices on
    # tensor.py:523).  ``flip_flow`` keeps charges identical, so the
    # ``(k, chi_in, d2)`` data layout is unchanged.
    P_dag = DenseTensor(
        P_dag_arr,
        (
            chi_new_out_idx,  # IN
            chi_in_idx.flip_flow(),  # OUT (was IN on the boundary)
            e_d2_idx.flip_flow(),  # OUT (was IN on the boundary)
        ),
    )

    return P, P_dag


# ====================================================================== #
# Corboz biorthogonal projector on the rank-4 enlarged corner.            #
#                                                                         #
# Paper-faithful for the non-Hermitian A ≠ B honeycomb corner             #
# (Lukin-Sotnikov PRE 109, 045305 (2024) §II.C, Fig. 10(d); algorithm     #
# from Corboz et al. PRB 90, 115110 (2014)).                              #
#                                                                         #
# Distinct from ``compute_honeycomb_projector`` above:                    #
#   * Input: TWO matrices M_U and M_L (upper / lower halves of the        #
#     enlarged corner), not a single rank-3 row tensor.                   #
#   * Identity satisfied: ``P_L · P_R = I_chi`` directly (no boundary in  #
#     between), versus Fishman's ``P1^† · M · P2 = I``.                   #
#                                                                         #
# Memory: ``feedback_corboz_vs_fishman.md`` records this distinction.     #
# ====================================================================== #


def compute_honeycomb_corner_biorthogonal_projector(
    upper_half: jax.Array,
    lower_half: jax.Array,
    *,
    chi: int,
    s_safe_eps: float = _S_SAFE_EPS,
) -> tuple[jax.Array, jax.Array]:
    """Corboz biorthogonal projector pair for the honeycomb enlarged corner.

    Returns ``(P_L, P_R)`` satisfying

    .. math:: P_L \\, P_R = I_\\chi

    *directly* (no boundary tensor in between).  This is the defining
    identity of the Corboz biorthogonalization (Paper 2 §II.C / Corboz
    et al. 2014), and is the form required for paper-faithful CTMRG on a
    bipartite honeycomb lattice with non-Hermitian corner.

    The algorithm is QR + QR + truncated SVD on the QR factors:

    1. ``M_U = Q_U · R_U`` — reduced QR of the upper-half matrix.
    2. ``M_L^T = Q_L · R_L`` — reduced QR of the lower-half matrix
       transposed (i.e. ``M_L = R_L^T · Q_L^T``).  This is
       conventionally written ``LQ(M_L)`` and gives the lower half its
       "L"-side contribution to the projectors.
    3. ``U_svd, S, Vh = SVD(R_U · R_L^T)`` — truncate to the largest
       ``chi`` singular values.  ``S_safe`` clamps near-zero singular
       values for AD stability.
    4. Construct the pair:

       .. math::
          P_L &= S^{-1/2} \\, U_{\\chi}^\\dagger \\, R_U \\\\
          P_R &= R_L^T \\, V_{\\chi} \\, S^{-1/2}

       The biorthogonality identity then follows from
       ``R_U · R_L^T = U_χ · diag(S_χ) · V_χ^†`` (truncated SVD) and
       the isometry of ``U_χ`` and ``V_χ``::

           P_L · P_R = S^{-1/2} U_χ^† (R_U R_L^T) V_χ S^{-1/2}
                     = S^{-1/2} U_χ^† (U_χ S_χ V_χ^†) V_χ S^{-1/2}
                     = S^{-1/2} S_χ S^{-1/2}
                     = I_χ.

    The phase-fix gauge from :func:`_column_phase` is applied jointly to
    ``U_svd`` and ``Vh`` (the same per-column phase factor goes onto both)
    so that the SVD identity ``U S V^†`` is gauge-invariant under the
    fix, mirroring the convention used in the SVD branch of
    :func:`compute_honeycomb_projector`.

    Args:
        upper_half: Upper half of the enlarged corner, shape
            ``(chi*D², chi*D²)`` — typically the contraction of
            ``L_α^A · C_α^A · T_A`` reshaped to a square matrix.  See
            :func:`tenax.algorithms._ctm_honeycomb_moves.move_direction_alpha`
            for how Task 6 builds this from the env + bulk tensors.
        lower_half: Lower half of the enlarged corner, shape
            ``(chi*D², chi*D²)`` — typically ``T_B · C_α^B · R_α^B``.
        chi: Target truncation dim for the new chi axis.  Output dim is
            ``min(chi, upper_half.shape[1])`` (bounded by the SVD rank).
        s_safe_eps: Floor for singular values used to clamp ``S_safe``
            against ``1/sqrt(0)`` NaNs.

    Returns:
        ``(P_L, P_R)`` as raw :class:`jax.Array` matrices.

        * ``P_L`` shape ``(chi_out, n)`` where ``n = upper_half.shape[1]``.
        * ``P_R`` shape ``(n, chi_out)``.
        * ``chi_out = min(chi, n)``.

        Identity: ``P_L @ P_R == I_chi_out`` to floating-point tolerance.

    Notes:
        Returned as plain :class:`jax.Array` (not :class:`Tensor`) because
        Task 6 needs to absorb them into rank-4 enlarged corners of varying
        leg-flow conventions; the wrapping into :class:`Tensor` happens at
        the call site in ``_ctm_honeycomb_moves`` where the per-direction
        labels are known.
    """
    if upper_half.ndim != 2 or lower_half.ndim != 2:
        raise ValueError(
            "Corboz biorthogonal halves must be rank-2; got "
            f"upper_half.ndim={upper_half.ndim}, lower_half.ndim={lower_half.ndim}"
        )
    if upper_half.shape[1] != lower_half.shape[1]:
        raise ValueError(
            "Corboz biorthogonal: upper_half and lower_half must have matching "
            f"second axes; got {upper_half.shape} vs {lower_half.shape}"
        )

    # ------------------------------------------------------------------ #
    # Step 1 -- QR of the two halves.                                     #
    # ------------------------------------------------------------------ #
    _Q_U, R_U = jnp.linalg.qr(upper_half, mode="reduced")
    _Q_L, R_L = jnp.linalg.qr(lower_half.T, mode="reduced")

    # ------------------------------------------------------------------ #
    # Step 2 -- SVD of R_U @ R_L^T.                                       #
    #                                                                     #
    # The biorthogonality identity follows from this SVD: U_χ · diag(S) · #
    # V_χ^† == R_U · R_L^T (truncated).                                   #
    # ------------------------------------------------------------------ #
    U_svd, S, Vh = _dense_svd(R_U @ R_L.T, full_matrices=False)

    k = int(min(chi, S.shape[0]))
    U_k = U_svd[:, :k]  # (n_qr_U, k)
    Vh_k = Vh[:k, :]  # (k, n_qr_L)
    S_k = S[:k]

    # ------------------------------------------------------------------ #
    # Step 3 -- S_safe clamp + inverse-sqrt for the symmetric S^{-1/2}    #
    # weighting on both projectors. Mirrors the canonical pattern from    #
    # the SVD branch above.                                               #
    # ------------------------------------------------------------------ #
    if k > 0:
        cutoff = jnp.maximum(s_safe_eps, 1e-14 * (S_k[0] + 1e-30))
        mask = S_k > cutoff
        S_safe = jnp.where(mask, S_k, 1.0)
        S_rsqrt = jnp.where(mask, 1.0 / jnp.sqrt(S_safe), 0.0)
    else:
        S_rsqrt = jnp.zeros_like(S_k)

    # ------------------------------------------------------------------ #
    # Step 4 -- joint phase fix on U_k / Vh_k.                            #
    #                                                                     #
    # We apply the same per-column phase factor to U_k and V_k (= Vh_k.T  #
    # conjugated) so that ``R_U R_L^T = U S V^†`` is preserved exactly    #
    # under the gauge fix.  Phase-fixing V independently would compute    #
    # a *different* phase from V's columns and break the identity.        #
    # ------------------------------------------------------------------ #
    if k > 0:
        phase_U = _column_phase(U_k)  # (k,) unit modulus, stop_gradient'd
        U_k = U_k * jnp.conj(phase_U)[None, :]
        Vh_k = Vh_k * phase_U[:, None]  # apply same phase to V's columns

    # ------------------------------------------------------------------ #
    # Step 5 -- assemble the biorthogonal pair.                           #
    #                                                                     #
    #   P_L = S^{-1/2} · U_χ^† · R_U          shape (k, n_qr_U)            #
    #   P_R = R_L^T · V_χ · S^{-1/2}          shape (n_qr_L, k)           #
    # ------------------------------------------------------------------ #
    P_L = S_rsqrt[:, None] * (jnp.conj(U_k).T @ R_U)
    P_R = (R_L.T @ jnp.conj(Vh_k).T) * S_rsqrt[None, :]

    # Stop-gradient through the projector pair in the forward path,
    # matching the dense-fallback convention at _ctm_projector.py:1067.
    # The Task 13 implicit-AD wrapper handles the backward through the
    # CTM fixed point separately.
    P_L = jax.lax.stop_gradient(P_L)
    P_R = jax.lax.stop_gradient(P_R)

    return P_L, P_R
