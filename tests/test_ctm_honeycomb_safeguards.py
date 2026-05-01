"""Numerical-safeguard tests for honeycomb CTM.

Mirrors the safeguard table in ``docs/plans/2026-04-25-honeycomb-ctm-design.md``:

* dtype must be complex128 (real float64 → non-variational drift, memory
  ``project_complex_tensors_variational.md``).
* sites must be rank-4 with labels ``(e0, e1, e2, phys)``.
* virtual bond dim ``D`` must agree across sublattices.
* non-convergence emits ``warnings.warn`` and returns a finite scalar
  (PR #343 contract: warn-not-raise).

The ``S_safe`` clamp + per-column phase fix are already exercised by
``tests/test_ctm_honeycomb_projector.py`` (see
``test_projector_no_nan_at_degenerate_spectrum`` and
``test_projector_phase_fix_idempotent``).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.honeycomb_ctm import honeycomb_ctm_energy_implicit
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _heisenberg_bond_xxz(d: int = 2) -> jnp.ndarray:
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return jnp.asarray(
        np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz),
        dtype=jnp.complex128,
    )


def _make_random_honeycomb_site(
    D: int, d: int, key: jax.Array, *, dtype=jnp.complex128
) -> DenseTensor:
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
    if dtype == jnp.complex128:
        im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
        data = (re + 1j * im).astype(jnp.complex128)
    else:
        data = re.astype(dtype)
    return DenseTensor(data, indices)


def _make_rank5_honeycomb_site(D: int, d: int, key: jax.Array) -> DenseTensor:
    """Bogus rank-5 site (extra leg) for the wrong-rank rejection test."""
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e3"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, D, d))
    data = (re + 1j * im).astype(jnp.complex128)
    return DenseTensor(data, indices)


# ------------------------------------------------------------------ #
# Input validation                                                     #
# ------------------------------------------------------------------ #


def test_rejects_real_dtype():
    """float64 input → ValueError mentioning complex128."""
    A = _make_random_honeycomb_site(
        D=2, d=2, key=jax.random.PRNGKey(0), dtype=jnp.float64
    )
    B = _make_random_honeycomb_site(
        D=2, d=2, key=jax.random.PRNGKey(1), dtype=jnp.float64
    )
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()
    with pytest.raises(ValueError, match="complex128"):
        honeycomb_ctm_energy_implicit(sites, H, chi=4, max_iter=2)


def test_rejects_mismatched_bond_dims():
    """A with D=4, B with D=2 → ValueError."""
    A = _make_random_honeycomb_site(D=4, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()
    with pytest.raises(ValueError, match="bond dims"):
        honeycomb_ctm_energy_implicit(sites, H, chi=4, max_iter=2)


def test_rejects_wrong_rank():
    """Rank-5 input → ValueError mentioning rank."""
    A = _make_rank5_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_rank5_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()
    with pytest.raises(ValueError, match="rank"):
        honeycomb_ctm_energy_implicit(sites, H, chi=4, max_iter=2)


def test_rejects_unexpected_labels():
    """Site with non-canonical label set → ValueError mentioning the label set."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    A_bad = A.relabels({"e0": "x0"})  # break the canonical label set
    sites = {(0, 0): A_bad, (1, 0): B}
    H = _heisenberg_bond_xxz()
    with pytest.raises(ValueError, match="labels"):
        honeycomb_ctm_energy_implicit(sites, H, chi=4, max_iter=2)


# ------------------------------------------------------------------ #
# Convergence-failure policy: warn, don't raise                        #
# ------------------------------------------------------------------ #


def test_nonconvergence_warns_returns_finite():
    """max_iter=1 + tight tol → RuntimeWarning + finite scalar (PR #343)."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        E = honeycomb_ctm_energy_implicit(
            sites,
            H,
            chi=4,
            max_iter=1,
            conv_tol=1e-12,
            min_iter=0,
            projector_method="biorthogonal",
            forward_gauge="phase",
        )
    assert jnp.isfinite(E), f"E = {E}"
    assert E.shape == ()
    assert any("did not converge" in str(wi.message).lower() for wi in w), [
        str(wi.message) for wi in w
    ]
