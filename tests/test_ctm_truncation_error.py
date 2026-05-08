"""Unit tests for the CTM truncation-error helper.

ϵ_T is the variPEPS §2.8.2 quantity: the L2 norm of the *normalized*
discarded singular values, i.e. ‖S[χ:]‖ / ‖S‖. We test against analytic
spectra so behavior is reproducible without a full CTM run.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_truncation_error import compute_truncation_error


def test_zero_when_chi_covers_full_spectrum():
    s = jnp.array([1.0, 0.5, 0.25, 0.125])
    assert float(compute_truncation_error(s, chi=4)) == pytest.approx(0.0)
    assert float(compute_truncation_error(s, chi=10)) == pytest.approx(0.0)


def test_matches_normalized_l2_of_discarded_tail():
    s = jnp.array([1.0, 0.5, 0.25, 0.125])
    expected = float(jnp.sqrt(jnp.sum(s[2:] ** 2) / jnp.sum(s**2)))
    assert float(compute_truncation_error(s, chi=2)) == pytest.approx(expected)


def test_one_when_chi_zero():
    s = jnp.array([1.0, 0.5])
    assert float(compute_truncation_error(s, chi=0)) == 1.0


def test_handles_zero_spectrum_safely():
    """A zero S vector (degenerate edge case) returns 0, not NaN."""
    s = jnp.zeros(4)
    assert float(compute_truncation_error(s, chi=2)) == 0.0


# ---------------------------------------------------------------------------
# Task 2: _compute_projector_tensor returns 3-tuple (P_1, P_2, eps_T)
# ---------------------------------------------------------------------------

import jax
import numpy as np

from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def test_compute_projector_tensor_returns_eps_t_dense_svd():
    """Dense non-tracer SVD path returns nontrivial ε_T when truncating."""
    key = jax.random.PRNGKey(0)
    chi_in, chi_target = 8, 4
    sym = U1Symmetry()
    charges = np.zeros(chi_in, dtype=np.int32)
    fused_idx = TensorIndex.from_charges(
        sym, charges.copy(), FlowDirection.IN, label="fused"
    )
    col_idx = TensorIndex.from_charges(
        sym, charges.copy(), FlowDirection.OUT, label="col"
    )

    C1_data = jax.random.normal(key, (chi_in, chi_in))
    C4_data = jax.random.normal(jax.random.fold_in(key, 1), (chi_in, chi_in))
    C1g = DenseTensor(C1_data, (fused_idx, col_idx))
    C4g = DenseTensor(C4_data, (fused_idx, col_idx))

    out = _compute_projector_tensor(C1g, C4g, chi_target, projector_method="svd")
    assert len(out) == 3
    P_1, P_2, eps_T = out
    assert eps_T.shape == ()
    assert float(eps_T) > 0.0  # truncating 8→4 must produce nonzero ε_T
    assert 0.0 <= float(eps_T) <= 1.0


def test_compute_projector_tensor_returns_zero_eps_t_for_eigh():
    """eigh path is out of scope for v1; returns 0.0 placeholder."""
    key = jax.random.PRNGKey(0)
    chi_in, chi_target = 4, 4
    sym = U1Symmetry()
    charges = np.zeros(chi_in, dtype=np.int32)
    fused_idx = TensorIndex.from_charges(
        sym, charges.copy(), FlowDirection.IN, label="fused"
    )
    col_idx = TensorIndex.from_charges(
        sym, charges.copy(), FlowDirection.OUT, label="col"
    )
    C1_data = jax.random.normal(key, (chi_in, chi_in))
    C4_data = jax.random.normal(jax.random.fold_in(key, 1), (chi_in, chi_in))
    C1g = DenseTensor(C1_data, (fused_idx, col_idx))
    C4g = DenseTensor(C4_data, (fused_idx, col_idx))

    out = _compute_projector_tensor(C1g, C4g, chi_target, projector_method="eigh")
    assert len(out) == 3
    _, _, eps_T = out
    assert float(eps_T) == 0.0
