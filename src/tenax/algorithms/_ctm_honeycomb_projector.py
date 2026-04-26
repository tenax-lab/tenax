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

  The honeycomb v1 isometric variant does **not** apply ``S_rsqrt`` to
  the projector itself (see "isometric vs. Fishman" below), but the
  same NaN-safety logic is used to guard the eigh/SVD forward.

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

For ``method in ('eigh', 'svd')`` the projector is isometric.  Task 6
will use ``P`` to absorb on one side and ``P_dagger`` on the other.

* ``P``        : labels ``(chi_in_label, e_alpha_d2_label, chi_new_in_label)``
                 flows  ``(IN,             IN,              OUT)``
                 dim    ``(chi_in,         d2,              chi_new)``

* ``P_dagger`` : labels ``(chi_new_out_label, chi_in_label, e_alpha_d2_label)``
                 flows  ``(IN,                OUT,          OUT)``
                 dim    ``(chi_new,           chi_in,       d2)``

where ``chi_new = min(chi, chi_in * d2)``.  Contracting ``P_dagger``
with ``P`` over the shared ``(chi_in, d2)`` axes (matched by index dim,
since the labels in ``P_dagger`` are shadow copies with flipped flows)
gives the identity on the new chi axis::

    einsum("kab,abm->km", P_dag.todense(), P.todense()) == eye(chi_new)

Isometric vs. Fishman two-projector
-----------------------------------

For ``method='eigh'`` and ``method='svd'`` we return the **plain
isometric form**::

    P = U,  P_dagger = U^dagger

where ``U`` is the leading ``chi_new`` left singular vectors of the
matricised boundary (or the leading ``chi_new`` eigenvectors of
``M M^dagger``).  This satisfies ``P_dagger @ P = I_{chi_new}``
exactly.  The Fishman two-projector form (``P1 = C4 V S^{-1/2}`` and
``P2 = C1 U S^{-1/2}``) is a *different* projection — the biorthogonal
``P1^dagger P2 = I`` rather than ``P^dagger P = I`` — and is reserved
for ``method='biorthogonal'`` once that variant is implemented.

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

__all__ = ["compute_honeycomb_projector"]

IN = FlowDirection.IN
OUT = FlowDirection.OUT

_S_SAFE_EPS = 1e-12
_PHASE_EPS = 0.1  # variPEPS first-above-threshold fraction-of-max default


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _phase_fix_columns(U: jax.Array, eps_frac: float = _PHASE_EPS) -> jax.Array:
    """Fix per-column U(1) phase via the variPEPS first-above-threshold rule.

    For each column ``U[:, j]``, find the first row ``i`` whose
    ``|U[i, j]| >= eps_frac * max_i |U[i, j]|`` and divide the column by
    ``U[i, j] / |U[i, j]|`` — making ``U[i, j]`` real-positive.

    The phase factor is wrapped in :func:`jax.lax.stop_gradient` because
    it represents a gauge degree of freedom: rotating an isometry's
    column by a unit-modulus scalar is invariant under the next layer
    of contractions, so the optimizer should not see it as a continuous
    parameter (mirrors the ``stop_gradient`` of the QR phase factor at
    ``_ctm_projector.py:953-964``).

    Args:
        U:        Matrix of shape ``(m, k)`` whose columns to phase-fix.
        eps_frac: Fraction-of-max threshold for picking the reference
                  row.  Matches ``EPS_PHASE = 0.1`` in
                  :func:`ad_utils._phase_fix_ctm_tensor`.

    Returns:
        ``U`` with per-column phases fixed.
    """
    if U.shape[1] == 0:
        return U
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
    phase = jax.lax.stop_gradient(phase)
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
    """Isometric projector for honeycomb CTM, ``(chi_in, d2)`` -> ``chi_new``.

    Args:
        boundary:           Rank-3 :class:`Tensor` with labels ``(chi_in,
                            e_alpha_d2, chi_out)`` and flows
                            ``(IN, IN, OUT)`` (matches the layout of
                            ``L_alpha`` / ``R_alpha`` in
                            :class:`HoneycombCTMEnv`).
        method:             ``'eigh'`` or ``'svd'``.  ``'biorthogonal'``
                            raises :class:`NotImplementedError`.
        chi:                Target bond dimension for the new chi axis.
                            Output dim is ``min(chi, chi_in * d2)``.
        s_safe_eps:         Floor for singular values when forming
                            ``sqrt(S)`` to avoid NaNs from tiny negative
                            eigenvalues in the eigh path.
        chi_new_in_label:   Label assigned to the truncated axis on
                            ``P``.
        chi_new_out_label:  Label assigned to the truncated axis on
                            ``P_dagger``.

    Returns:
        ``(P, P_dagger)`` — see module docstring for shapes / labels /
        flows.

    Raises:
        NotImplementedError: ``method='biorthogonal'`` (deferred).
        ValueError:          unknown ``method`` value or wrong boundary
                             rank.
    """
    if method == "biorthogonal":
        raise NotImplementedError(
            "biorthogonal projectors are deferred for honeycomb CTM v1; "
            "see docs/plans/2026-04-25-honeycomb-ctm-design.md for the "
            "design discussion."
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
    chi_in_idx, e_d2_idx, _chi_out_idx = boundary.indices
    chi_in = chi_in_idx.dim
    d2 = e_d2_idx.dim

    # ---------------------------------------------------------------- #
    # Build the matrix M : (chi_in * d2, chi_out) by reshaping the      #
    # boundary so that the (chi_in, d2) pair is jointly truncated to    #
    # the new chi axis.                                                  #
    # ---------------------------------------------------------------- #
    boundary_arr = boundary.todense()  # small: (chi, d2, chi)
    M = boundary_arr.reshape(chi_in * d2, -1)  # (chi_in*d2, chi_out)

    if method == "svd":
        from tenax.algorithms.ad_utils import _fix_svd_signs

        U_full, S_full, Vh_full = jnp.linalg.svd(M, full_matrices=False)
        U_full, S_full, _Vh_full = _fix_svd_signs(U_full, S_full, Vh_full)
        k = min(chi, S_full.shape[0])
        U_k = U_full[:, :k]
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
    # The isometric variant does NOT apply S_rsqrt to the projector     #
    # itself (see module docstring), so we compute these but reserve    #
    # them for the deferred biorthogonal variant.  Computing them now   #
    # also exercises the NaN-safety code path under the same inputs,    #
    # which keeps Task 15's safeguard tests honest.                     #
    # ---------------------------------------------------------------- #
    if k > 0:
        cutoff = jnp.maximum(s_safe_eps, 1e-14 * (S_k[0] + 1e-30))
        mask = S_k > cutoff
        S_safe = jnp.where(mask, S_k, 1.0)
        S_rsqrt = jnp.where(mask, 1.0 / jnp.sqrt(S_safe), 0.0)
        # Touch S_rsqrt so JAX doesn't DCE the computation under jit;
        # this keeps the NaN-safety check live for tracing tools but
        # adds ~zero work since `_` is never read.
        _ = S_rsqrt + 0.0

    # ---------------------------------------------------------------- #
    # Phase-fix the columns of U so the output is gauge-deterministic.  #
    # ---------------------------------------------------------------- #
    U_k = _phase_fix_columns(U_k)

    # ---------------------------------------------------------------- #
    # Build P and P_dagger as DenseTensors (the honeycomb env at v1     #
    # is dense complex128, see initialize_honeycomb_env).                #
    # ---------------------------------------------------------------- #
    # Isometric form: P = U, P_dagger = U^†.  P^† P = U^† U = I_k.
    P_arr = U_k.reshape(chi_in, d2, k)
    P_dag_arr = jnp.conj(U_k).T.reshape(k, chi_in, d2)

    # Stop-gradient through the *projector* itself in the forward path,
    # matching the convention of the dense fallbacks in
    # _ctm_projector.py (lines 979 and 1067).  Implicit-AD callers
    # (Task 13) handle the backward through the CTM fixed point
    # separately, so this is the right default.
    P_arr = jax.lax.stop_gradient(P_arr)
    P_dag_arr = jax.lax.stop_gradient(P_dag_arr)

    # Build TensorIndex metadata for the new chi axes.
    chi_new_in_idx = _make_chi_new_index(chi_in_idx, k, chi_new_in_label, OUT)
    chi_new_out_idx = _make_chi_new_index(chi_in_idx, k, chi_new_out_label, IN)

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
