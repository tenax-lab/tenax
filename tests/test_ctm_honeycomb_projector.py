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
from tenax.algorithms._ctm_honeycomb_projector import compute_honeycomb_projector
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
def test_projector_isometry_via_dense(method: str) -> None:
    """``P_dagger @ P == I_chi_new`` (within eps)."""
    D, chi_init, alpha = 2, 4, 1
    boundary = _column_boundary(D, chi_init, alpha, jax.random.PRNGKey(7))
    chi = 6  # smaller than chi_in*d2 = 4*4 = 16

    P, P_dag = compute_honeycomb_projector(boundary, method=method, chi=chi)
    P_arr = P.todense()  # shape (chi_in, d2, chi_new)
    Pd_arr = P_dag.todense()  # shape (chi_new, chi_in, d2)
    # P_dag @ P contracts on (chi_in, d2) → (chi_new_out, chi_new_in)
    eye_check = jnp.einsum("kab,abm->km", Pd_arr, P_arr)
    n = eye_check.shape[0]
    assert jnp.allclose(eye_check, jnp.eye(n, dtype=eye_check.dtype), atol=1e-8), (
        f"isometry failed for method={method!r}: "
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


def test_biorthogonal_raises_not_implemented() -> None:
    """``method='biorthogonal'`` raises ``NotImplementedError``."""
    boundary = _column_boundary(2, 4, 0, jax.random.PRNGKey(0))
    with pytest.raises(NotImplementedError, match="biorthogonal"):
        compute_honeycomb_projector(boundary, method="biorthogonal", chi=4)
