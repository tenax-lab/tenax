"""Unit tests for the CTM truncation-error helper.

ϵ_T is the variPEPS §2.8.2 quantity: the L2 norm of the *normalized*
discarded singular values, i.e. ‖S[χ:]‖ / ‖S‖. We test against analytic
spectra so behavior is reproducible without a full CTM run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.algorithms._ctm_truncation_error import compute_truncation_error
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


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


def test_compute_projector_tensor_returns_eps_t_for_eigh():
    """Dense non-tracer eigh path computes eps_T from the density-matrix eigenspectrum.

    The density matrix rho = C1g @ C1g^H + C4g @ C4g^H is (chi_in × chi_in).
    When chi_target < chi_in, the discarded eigenvalues give eps_T > 0.
    This drives the variPEPS §2.8.2 auto-χ bump decision.
    """
    key = jax.random.PRNGKey(0)
    # chi_target=2 < chi_in=4: rho is 4×4, keep top-2 eigenvectors -> eps_T > 0
    chi_in, chi_target = 4, 2
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
    # eps_T must be positive: chi_target=2 discards 2 of 4 eigenvalues from rho.
    assert float(eps_T) > 0.0, f"Expected eps_T > 0 but got {float(eps_T)}"
    assert float(eps_T) <= 1.0, f"eps_T must be in [0, 1], got {float(eps_T)}"


# ---------------------------------------------------------------------------
# Issue #410: ε_T from the SymmetricTensor projector paths.
#
# The single-site dense auto-χ_E bump (PR #412) only surfaces ε_T from the
# dense projector paths.  The SymmetricTensor SVD and eigh paths still
# return ``jnp.asarray(0.0)`` as a placeholder, which means auto-χ_E bump
# cannot fire when the optimizer is run with SymmetricTensor inputs.
#
# Issue #410 asks for ε_T extraction from the SymmetricTensor SVD path;
# the eigh case is included here because ``optimize_gs_ad`` measures ε_T
# via a one-off ``_ctm_tensor_sweep(..., "eigh")`` call (see
# ``ipeps_optimize.py:881``) — so the eigh symmetric path is *the* path
# that decides the bump.  The QR symmetric path is not used for the
# auto-bump decision and is intentionally left as a 0.0 placeholder.
# ---------------------------------------------------------------------------


def _make_symmetric_corner(
    chi: int,
    fused_charges: np.ndarray,
    col_charges: np.ndarray,
    seed: int,
):
    """Build a random-block SymmetricTensor corner with labels ('fused', 'col').

    Random entries are placed in every symmetry-allowed block, so the
    underlying matrix is block-sparse but generic within each block.
    """
    from tenax.core.tensor import SymmetricTensor

    sym = U1Symmetry()
    fused_idx = TensorIndex.from_charges(
        sym, fused_charges.astype(np.int32), FlowDirection.IN, label="fused"
    )
    col_idx = TensorIndex.from_charges(
        sym, col_charges.astype(np.int32), FlowDirection.OUT, label="col"
    )
    return SymmetricTensor.random_normal((fused_idx, col_idx), jax.random.PRNGKey(seed))


def test_compute_projector_tensor_returns_eps_t_symmetric_svd():
    """SymmetricTensor SVD path must return ε_T from the merged per-sector spectrum.

    Two charge sectors (q=0, q=1) of dim 4 each → 8 singular values total.
    Truncating to chi_target=4 must discard at least 4 SVs, giving ε_T > 0.
    """
    chi_in = 8
    chi_target = 4
    # Two-sector charges: half q=0, half q=1.
    fused_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    col_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    C1g = _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=0)
    C4g = _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=1)

    P_1, P_2, eps_T = _compute_projector_tensor(
        C1g, C4g, chi_target, projector_method="svd"
    )
    assert eps_T.shape == ()
    assert 0.0 < float(eps_T) <= 1.0, (
        f"Expected 0 < ε_T <= 1 from symmetric SVD truncation, got {float(eps_T)}"
    )


def test_compute_projector_tensor_returns_eps_t_symmetric_eigh():
    """SymmetricTensor eigh path must return ε_T from the merged per-sector spectrum.

    This is the path optimize_gs_ad uses for the auto-bump decision
    (ipeps_optimize.py:881 calls _ctm_tensor_sweep with projector_method='eigh'
    purely to measure ε_T).  Without ε_T here, auto-bump can never fire on
    symmetric inputs.
    """
    chi_in = 8
    chi_target = 4
    fused_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    col_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    C1g = _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=0)
    C4g = _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=1)

    _, _, eps_T = _compute_projector_tensor(
        C1g, C4g, chi_target, projector_method="eigh"
    )
    assert eps_T.shape == ()
    assert 0.0 < float(eps_T) <= 1.0, (
        f"Expected 0 < ε_T <= 1 from symmetric eigh truncation, got {float(eps_T)}"
    )


def test_compute_projector_tensor_eps_t_symmetric_zero_when_no_truncation():
    """No truncation (chi >= n_keep) → ε_T = 0 on the symmetric paths too."""
    chi_in = 4
    chi_target = 8  # > spectrum length
    fused_charges = np.array([0, 0, 1, 1])
    col_charges = np.array([0, 0, 1, 1])
    C1g = _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=0)
    C4g = _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=1)

    _, _, eps_T_svd = _compute_projector_tensor(
        C1g, C4g, chi_target, projector_method="svd"
    )
    _, _, eps_T_eigh = _compute_projector_tensor(
        C1g, C4g, chi_target, projector_method="eigh"
    )
    assert float(eps_T_svd) == pytest.approx(0.0, abs=1e-12)
    assert float(eps_T_eigh) == pytest.approx(0.0, abs=1e-12)


def _scale_blocks_per_charge(t, scale_per_charge):
    """Return ``t`` with each block scaled by ``scale_per_charge[fq]``.

    Used by the tests below to amplify one charge sector so that the
    global top-N truncation differs from the per-sector forced allocation.
    """
    from tenax.core.tensor import SymmetricTensor

    new_blocks = {}
    for key, block in t.blocks.items():
        scale = float(scale_per_charge.get(int(key[0]), 1.0))
        new_blocks[key] = block * scale
    return SymmetricTensor._from_blocks_unchecked(new_blocks, t.indices)


def test_compute_projector_tensor_eps_t_symmetric_uses_actual_kept_set_svd():
    """Regression for PR #430 codex review: ε_T must reflect the kept set.

    When ``base_charges`` forces the per-sector allocation, the kept
    (fused_charge, index) pairs can differ from the global top-``n_keep``
    cutoff. ε_T must be computed against the COMPLEMENT of the actually
    kept pairs — otherwise it under-reports truncation whenever the forced
    allocation keeps a small singular value while discarding a larger one
    from another sector.

    Construction: an 8-dim corner with two sectors of dim 4 each, with
    the q=0 block amplified 10× over q=1. ``chi_target=4`` with
    ``base_charges=[0, 1]`` forces a 2+2 sector allocation, so two
    large q=0 singular values are discarded in favour of two small q=1
    values. The global top-4 cutoff keeps all 4 q=0 SVs and discards
    the 4 q=1 SVs (small), giving a much smaller ε_T. Pre-fix both
    paths used the global cutoff and reported the same ε_T.
    """
    chi_in = 8
    chi_target = 4
    fused_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    col_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    C1g = _scale_blocks_per_charge(
        _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=0),
        {0: 10.0, 1: 1.0},
    )
    C4g = _scale_blocks_per_charge(
        _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=1),
        {0: 10.0, 1: 1.0},
    )

    _, _, eps_global = _compute_projector_tensor(
        C1g, C4g, chi_target, projector_method="svd"
    )
    _, _, eps_forced = _compute_projector_tensor(
        C1g,
        C4g,
        chi_target,
        projector_method="svd",
        base_charges=np.array([0, 1], dtype=np.int32),
    )
    # Forced sector allocation discards dominant q=0 values that global
    # truncation would have kept → strictly more discarded weight.
    assert float(eps_forced) > 2.0 * float(eps_global), (
        f"ε_T(base_charges=[0,1])={float(eps_forced):.4e} should be much "
        f"larger than ε_T(no base_charges)={float(eps_global):.4e} when the "
        "forced 2+2 allocation discards dominant q=0 SVs. Pre-fix both used "
        "the global top-N cutoff and reported equal values."
    )


def test_compute_projector_tensor_eps_t_symmetric_uses_actual_kept_set_eigh():
    """eigh-path counterpart to the SVD regression above (PR #430 codex)."""
    chi_in = 8
    chi_target = 4
    fused_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    col_charges = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    C1g = _scale_blocks_per_charge(
        _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=2),
        {0: 10.0, 1: 1.0},
    )
    C4g = _scale_blocks_per_charge(
        _make_symmetric_corner(chi_in, fused_charges, col_charges, seed=3),
        {0: 10.0, 1: 1.0},
    )

    _, _, eps_global = _compute_projector_tensor(
        C1g, C4g, chi_target, projector_method="eigh"
    )
    _, _, eps_forced = _compute_projector_tensor(
        C1g,
        C4g,
        chi_target,
        projector_method="eigh",
        base_charges=np.array([0, 1], dtype=np.int32),
    )
    assert float(eps_forced) > 2.0 * float(eps_global), (
        f"ε_T(base_charges=[0,1])={float(eps_forced):.4e} should be much "
        f"larger than ε_T(no base_charges)={float(eps_global):.4e} when the "
        "forced 2+2 allocation discards dominant q=0 eigenvalues. Pre-fix "
        "both used the global top-N cutoff and reported equal values."
    )
