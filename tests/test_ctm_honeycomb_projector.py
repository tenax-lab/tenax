"""Honeycomb projector isometry + S_safe NaN protection tests.

Mirrors the public-API style of ``_ctm_projector`` tests but for the
rank-3 honeycomb boundary returned by ``_double_layer_honeycomb`` /
``initialize_honeycomb_env``.

Tests use realistic boundaries built from random honeycomb sites
(``_make_random_honeycomb_site`` + ``_double_layer_honeycomb``) rather
than synthetic raw arrays — this exercises the actual TensorIndex / flow
metadata path that Task 6 will rely on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env
from tenax.algorithms._ctm_honeycomb_projector import (
    compute_honeycomb_corner_biorthogonal_projector,
    compute_honeycomb_projector,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_random_honeycomb_site(D: int, d: int, key: jax.Array) -> DenseTensor:
    """Inline copy of the fixture from ``tests/test_ctm_honeycomb_init.py``.

    Tests in this repo do not cross-import from other test modules
    (no ``tests`` package on the import path), so we duplicate the
    rank-4 site builder here.
    """
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
    data = (re + 1j * im).astype(jnp.complex128)
    return DenseTensor(data, indices)


# ------------------------------------------------------------------ #
# Fixtures: boundary tensors built the way Task 6 will build them.    #
# ------------------------------------------------------------------ #


def _column_boundary(D: int, chi_init: int, alpha: int, key: jax.Array) -> DenseTensor:
    """Build a column-edge boundary tensor matching Task 6's input shape.

    Returns a rank-3 ``Tensor`` with labels
    ``(chi_in_alpha, e_alpha_d2, chi_out_alpha)`` and flows
    ``(IN, IN, OUT)``.  This mirrors the ``L_alpha`` / ``R_alpha`` shape
    produced by :func:`initialize_honeycomb_env`.
    """
    A = _make_random_honeycomb_site(D=D, d=2, key=key)
    sites = {(0, 0): A}
    envs = initialize_honeycomb_env(sites, chi_init=chi_init, seed=int(key[0]))
    return getattr(envs[(0, 0)], f"L{alpha}")


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #


def test_projector_truncation_dim() -> None:
    """The new chi axis on P has dim ``min(chi, chi_in * d2)``."""
    D, chi_init, alpha = 2, 4, 0
    boundary = _column_boundary(D, chi_init, alpha, jax.random.PRNGKey(0))
    chi_in = boundary.indices[0].dim
    d2 = boundary.indices[1].dim

    chi = 5
    P, P_dag = compute_honeycomb_projector(boundary, method="eigh", chi=chi)
    expected = min(chi, chi_in * d2)
    # P: (chi_in, d2, chi_new_in) — last index is the truncated one.
    assert P.indices[2].dim == expected
    # P_dag: (chi_new_out, chi_in, d2) — first index is the truncated one.
    assert P_dag.indices[0].dim == expected


@pytest.mark.parametrize("method", ["eigh", "svd"])
def test_projector_left_right_identity(method: str) -> None:
    """eigh: ``P_dagger @ P = I``.  svd (Fishman): ``P1^dagger @ M @ P2 = I``.

    Both forms truncate ``(chi_in, d2)`` -> ``chi_new``, but the identity
    that holds depends on ``method``:

    * ``method='eigh'`` returns plain isometric ``(P, P_dagger)``; the
      contraction over the shared ``(chi_in, d2)`` legs gives
      ``I_chi_new`` directly.
    * ``method='svd'`` returns Fishman ``(P1, P2)``; both projectors
      carry an ``S^{-1/2}`` factor and the identity is
      ``P1^dagger @ boundary @ P2 = I_chi_new``.
    """
    D, chi_init, alpha = 2, 4, 1
    boundary = _column_boundary(D, chi_init, alpha, jax.random.PRNGKey(7))
    chi = 6  # smaller than chi_in*d2 = 4*4 = 16

    P_or_P1, P_dag_or_P2 = compute_honeycomb_projector(boundary, method=method, chi=chi)

    if method == "eigh":
        # Plain isometric: P_dagger @ P contracts on (chi_in, d2).
        P_arr = P_or_P1.todense()  # (chi_in, d2, chi_new_in)
        Pd_arr = P_dag_or_P2.todense()  # (chi_new_out, chi_in, d2)
        eye_check = jnp.einsum("kab,abm->km", Pd_arr, P_arr)
    else:
        # Fishman: P1^dagger @ boundary @ P2.
        # P1: (chi_in, d2, chi_new_in)         — like eigh's P
        # P2: (chi_out, chi_new_out)           — Fishman right-side projector
        # boundary: (chi_in, d2, chi_out)
        P1_arr = P_or_P1.todense()
        P2_arr = P_dag_or_P2.todense()
        b_arr = boundary.todense()
        eye_check = jnp.einsum("abk,abc,cm->km", jnp.conj(P1_arr), b_arr, P2_arr)

    n = eye_check.shape[0]
    assert jnp.allclose(eye_check, jnp.eye(n, dtype=eye_check.dtype), atol=1e-8), (
        f"identity failed for method={method!r}: "
        f"max|eye_check - I| = {jnp.max(jnp.abs(eye_check - jnp.eye(n))):.3e}"
    )


def test_projector_no_nan_at_degenerate_spectrum() -> None:
    """Boundary with rank << chi → no NaN in P or P_dagger."""
    # Build a deliberately rank-deficient boundary by zeroing all but one
    # outer-product slice. Keep the TensorIndex metadata from a real boundary.
    D, chi_init, alpha = 2, 4, 2
    boundary = _column_boundary(D, chi_init, alpha, jax.random.PRNGKey(3))
    chi_in, d2, chi_out = boundary.indices[0].dim, 1, boundary.indices[2].dim
    # Replace data with rank-1
    raw = boundary.todense()
    rank1 = jnp.zeros_like(raw)
    rank1 = rank1.at[0, 0, 0].set(1.0)
    rank1_boundary = DenseTensor(rank1, boundary.indices)

    chi_request = 8  # much larger than effective rank
    for method in ("eigh", "svd"):
        P, P_dag = compute_honeycomb_projector(
            rank1_boundary, method=method, chi=chi_request
        )
        assert jnp.all(jnp.isfinite(P.todense())), f"P NaN for {method!r}"
        assert jnp.all(jnp.isfinite(P_dag.todense())), f"P_dag NaN for {method!r}"

    _ = (chi_in, d2, chi_out)  # silence linter


def test_projector_phase_fix_idempotent() -> None:
    """Computing P twice with the same input returns bit-identical results."""
    D, chi_init, alpha = 2, 4, 0
    boundary = _column_boundary(D, chi_init, alpha, jax.random.PRNGKey(42))
    chi = 5

    P1, P1_dag = compute_honeycomb_projector(boundary, method="eigh", chi=chi)
    P2, P2_dag = compute_honeycomb_projector(boundary, method="eigh", chi=chi)
    assert jnp.array_equal(P1.todense(), P2.todense())
    assert jnp.array_equal(P1_dag.todense(), P2_dag.todense())


def test_isometric_api_rejects_biorthogonal_method() -> None:
    """The rank-3-boundary API only supports ``eigh`` and ``svd``.

    For Corboz biorthogonal projectors on the rank-4 enlarged corner, callers
    must use :func:`compute_honeycomb_corner_biorthogonal_projector` instead.
    """
    boundary = _column_boundary(2, 4, 0, jax.random.PRNGKey(0))
    with pytest.raises((ValueError, NotImplementedError), match="biorthogonal"):
        compute_honeycomb_projector(boundary, method="biorthogonal", chi=4)


# ------------------------------------------------------------------ #
# Task 5b: Corboz biorthogonal projector on the enlarged corner.       #
# Differs from compute_honeycomb_projector in:                         #
#   - Input shape: two (chi*d², chi*d²) halves, not one rank-3 tensor. #
#   - Identity satisfied: P_L · P_R = I_chi (Corboz), not P1^† M P2.   #
# Paper-faithful for the non-Hermitian A ≠ B corner; see               #
# memory: feedback_corboz_vs_fishman.md                                #
# ------------------------------------------------------------------ #


def _random_matrix(shape: tuple[int, ...], key: jax.Array) -> jax.Array:
    """Random complex128 matrix for Corboz tests."""
    re = jax.random.normal(key, shape)
    im = jax.random.normal(jax.random.fold_in(key, 1), shape)
    return (re + 1j * im).astype(jnp.complex128)


def test_corboz_biorthogonal_PL_PR_identity() -> None:
    """``P_L · P_R = I_chi`` directly (no boundary in between).

    This is the defining identity of the Corboz biorthogonal pair (Paper 2
    §II.C, Corboz et al. 2014). For non-Hermitian enlarged corners the
    Fishman ``P1^† M P2 = I`` form does not satisfy this identity; the
    explicit QR + SVD biorthogonalization does.
    """
    chi_in_d2 = 4 * 9  # chi * D² = 36 — typical "enlarged" dim
    chi_target = 6
    key = jax.random.PRNGKey(0)
    M_U = _random_matrix((chi_in_d2, chi_in_d2), key)
    M_L = _random_matrix((chi_in_d2, chi_in_d2), jax.random.fold_in(key, 100))
    P_L, P_R = compute_honeycomb_corner_biorthogonal_projector(M_U, M_L, chi=chi_target)
    # P_L: (chi_target, chi_in_d2)  — projects from chi_in_d2 to chi_target
    # P_R: (chi_in_d2, chi_target)
    assert P_L.shape == (chi_target, chi_in_d2)
    assert P_R.shape == (chi_in_d2, chi_target)
    prod = P_L @ P_R  # (chi_target, chi_target)
    assert jnp.allclose(prod, jnp.eye(chi_target, dtype=prod.dtype), atol=1e-6), (
        f"P_L @ P_R deviates from identity: "
        f"max|prod - I| = {float(jnp.max(jnp.abs(prod - jnp.eye(chi_target)))):.3e}"
    )


def test_corboz_biorthogonal_no_nan_on_degenerate() -> None:
    """Rank-deficient halves should not produce NaN P_L or P_R.

    The internal SVD on R_U · R_L^T can have near-zero singular values for
    rank-deficient inputs; the ``S_safe`` clamp in the implementation must
    keep the ``1/sqrt(S)`` factor finite.
    """
    chi_in_d2 = 4 * 9
    chi_target = 6
    M_U = jnp.zeros((chi_in_d2, chi_in_d2), dtype=jnp.complex128)
    M_U = M_U.at[0, 0].set(1.0)  # rank-1
    M_L = jnp.eye(chi_in_d2, dtype=jnp.complex128) * 1e-15  # near-zero
    P_L, P_R = compute_honeycomb_corner_biorthogonal_projector(M_U, M_L, chi=chi_target)
    assert jnp.all(jnp.isfinite(P_L))
    assert jnp.all(jnp.isfinite(P_R))


def test_corboz_biorthogonal_truncation_dim() -> None:
    """Output truncation dim equals ``min(chi, rank(R_U @ R_L^T))``."""
    chi_in_d2 = 16
    chi_request = 8
    key = jax.random.PRNGKey(5)
    M_U = _random_matrix((chi_in_d2, chi_in_d2), key)
    M_L = _random_matrix((chi_in_d2, chi_in_d2), jax.random.fold_in(key, 1))
    P_L, P_R = compute_honeycomb_corner_biorthogonal_projector(
        M_U, M_L, chi=chi_request
    )
    # Both halves are full-rank random complex matrices, so the SVD has
    # 16 nonzero singular values; truncating to chi_request=8 gives output
    # dim = 8.
    assert P_L.shape == (chi_request, chi_in_d2)
    assert P_R.shape == (chi_in_d2, chi_request)


def test_corboz_biorthogonal_phase_idempotent() -> None:
    """Computing twice on the same input returns bit-identical results.

    The phase fix is wrapped in stop_gradient; two calls with the same input
    must agree element-wise.
    """
    chi_in_d2 = 16
    chi_target = 6
    key = jax.random.PRNGKey(42)
    M_U = _random_matrix((chi_in_d2, chi_in_d2), key)
    M_L = _random_matrix((chi_in_d2, chi_in_d2), jax.random.fold_in(key, 1))
    P_L_1, P_R_1 = compute_honeycomb_corner_biorthogonal_projector(
        M_U, M_L, chi=chi_target
    )
    P_L_2, P_R_2 = compute_honeycomb_corner_biorthogonal_projector(
        M_U, M_L, chi=chi_target
    )
    assert jnp.array_equal(P_L_1, P_L_2)
    assert jnp.array_equal(P_R_1, P_R_2)
