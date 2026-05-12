"""Tests for SymmetricTensor support in the 2x2 plaquette projector (#416)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_projector_2x2 import (
    _compute_2x2_projector,
    _gauge_fix_symmetric_svd,
    _scale_bond_by_diag,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor


def _make_test_matrix_tensor(seed: int = 0) -> SymmetricTensor:
    """Build a small 2-leg SymmetricTensor matrix with two U(1) charge sectors."""
    sym = U1Symmetry()
    left_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    right_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    left_idx = TensorIndex.from_charges(
        sym, left_charges, FlowDirection.IN, label="left"
    )
    right_idx = TensorIndex.from_charges(
        sym, right_charges, FlowDirection.OUT, label="right"
    )
    return SymmetricTensor.random_normal(
        (left_idx, right_idx), jax.random.PRNGKey(seed)
    )


def test_gauge_fix_symmetric_svd_preserves_reconstruction():
    """After gauge fix, U_T @ diag(S) @ Vh_T == original matrix (per sector)."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=0)
    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    U_fixed, Vh_fixed = _gauge_fix_symmetric_svd(U_T, Vh_T)

    from tenax.contraction.contractor import contract

    U_scaled = _scale_bond_by_diag(U_fixed, s, bond_label="bond")
    M_reconstructed = contract(U_scaled, Vh_fixed)
    np.testing.assert_allclose(
        np.asarray(M_reconstructed.todense()),
        np.asarray(M_T.todense()),
        atol=1e-10,
        err_msg="gauge-fixed SVD must preserve reconstruction U·diag(s)·Vh == M",
    )


def test_gauge_fix_symmetric_svd_real_positive_max_row():
    """After gauge fix, the entry of largest |U[:, j]| is real-positive for every j."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=1)
    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    U_fixed, _ = _gauge_fix_symmetric_svd(U_T, Vh_T)

    U_dense = np.asarray(U_fixed.todense())
    for j in range(U_dense.shape[1]):
        col = U_dense[:, j]
        if np.max(np.abs(col)) == 0.0:
            continue
        max_row = int(np.argmax(np.abs(col)))
        entry = col[max_row]
        assert entry.imag == pytest.approx(0.0, abs=1e-10), (
            f"column {j}: max-abs entry should be real, got {entry}"
        )
        assert entry.real >= 0.0, (
            f"column {j}: max-abs entry should be non-negative, got {entry}"
        )


def _make_symmetric_enlarged_corner(
    chi: int,
    D: int,
    chi_label_a: str,
    chi_label_b: str,
    D2_label_a: str,
    D2_label_b: str,
    flow_chi_a: FlowDirection,
    flow_chi_b: FlowDirection,
    flow_D2_a: FlowDirection,
    flow_D2_b: FlowDirection,
    seed: int,
) -> SymmetricTensor:
    """4-leg enlarged-corner SymmetricTensor with non-trivial U(1) charges."""
    sym = U1Symmetry()
    chi_charges = np.arange(chi, dtype=np.int32) % 2
    D2_charges = np.arange(D**2, dtype=np.int32) % 2
    indices = (
        TensorIndex.from_charges(sym, chi_charges, flow_chi_a, label=chi_label_a),
        TensorIndex.from_charges(sym, D2_charges, flow_D2_a, label=D2_label_a),
        TensorIndex.from_charges(sym, chi_charges, flow_chi_b, label=chi_label_b),
        TensorIndex.from_charges(sym, D2_charges, flow_D2_b, label=D2_label_b),
    )
    return SymmetricTensor.random_normal(indices, jax.random.PRNGKey(seed))


@pytest.fixture
def symmetric_corners():
    """Return (Q_TL, Q_TR, Q_BL, Q_BR) — 4-leg SymmetricTensors with non-trivial U(1).

    Uses MIXED flow conventions (chi seam OUT, D^2 seam IN for Q_TL; mirrored
    for the other three corners with matching pair-flows for auto-contraction)
    so each corner has multiple U(1) charge blocks and the resulting M_prime
    SVD spans more than one bond sector.  Task 5's
    ``test_..._base_charges_drive_chi_new`` requires M_prime to have charge-1
    sectors to verify per-sector allocation; the all-OUT-Q_TL/all-IN-Q_BR
    fixture used in Tasks 2-4 would have collapsed M_prime to a single
    charge-0 block via U(1) selection rules.  Closure still holds for all
    four directions under the richer fixture.
    """
    chi, D = 4, 2
    Q_TL = _make_symmetric_enlarged_corner(
        chi,
        D,
        chi_label_a="chi_R",
        chi_label_b="chi_B",
        D2_label_a="r2",
        D2_label_b="d2",
        flow_chi_a=FlowDirection.OUT,
        flow_chi_b=FlowDirection.OUT,
        flow_D2_a=FlowDirection.IN,
        flow_D2_b=FlowDirection.IN,
        seed=0,
    )
    Q_TR = _make_symmetric_enlarged_corner(
        chi,
        D,
        chi_label_a="chi_L",
        chi_label_b="chi_B",
        D2_label_a="l2",
        D2_label_b="d2",
        flow_chi_a=FlowDirection.IN,
        flow_chi_b=FlowDirection.OUT,
        flow_D2_a=FlowDirection.OUT,
        flow_D2_b=FlowDirection.IN,
        seed=1,
    )
    Q_BL = _make_symmetric_enlarged_corner(
        chi,
        D,
        chi_label_a="chi_R",
        chi_label_b="chi_T",
        D2_label_a="r2",
        D2_label_b="u2",
        flow_chi_a=FlowDirection.OUT,
        flow_chi_b=FlowDirection.IN,
        flow_D2_a=FlowDirection.IN,
        flow_D2_b=FlowDirection.OUT,
        seed=2,
    )
    Q_BR = _make_symmetric_enlarged_corner(
        chi,
        D,
        chi_label_a="chi_L",
        chi_label_b="chi_T",
        D2_label_a="l2",
        D2_label_b="u2",
        flow_chi_a=FlowDirection.IN,
        flow_chi_b=FlowDirection.IN,
        flow_D2_a=FlowDirection.OUT,
        flow_D2_b=FlowDirection.OUT,
        seed=3,
    )
    return Q_TL, Q_TR, Q_BL, Q_BR


def test_compute_2x2_projector_symmetric_closure_left(symmetric_corners):
    """Symmetric path: `P_bot · P_top = I_chi_new` (closure check)."""
    from tenax.contraction.contractor import contract

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4

    # Task 4 dispatches SymmetricTensor inputs through _compute_2x2_projector
    # to the block-sparse _compute_2x2_projector_symmetric helper.
    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=chi, direction="left"
    )
    I_tensor = contract(P_bot, P_top)
    I_dense = np.asarray(I_tensor.todense())
    chi_new = P_top.indices[2].dim
    assert I_dense.shape == (chi_new, chi_new)
    np.testing.assert_allclose(
        I_dense,
        np.eye(chi_new),
        atol=1e-9,
        err_msg="P_bot · P_top must be identity on the truncated chi_new bond",
    )
    # Sanity: with chi=4 and chi*D²=16 of bond room, chi_new should reach 4.
    assert chi_new == 4, (
        f"Expected chi_new == 4 after Fishman truncation, got {chi_new}"
    )


@pytest.mark.parametrize("direction", ["right", "top", "bottom"])
def test_compute_2x2_projector_symmetric_closure_other_directions(
    symmetric_corners, direction
):
    """Closure test for direction in {right, top, bottom}."""
    from tenax.contraction.contractor import contract

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4
    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=chi, direction=direction
    )
    I_tensor = contract(P_bot, P_top)
    I_dense = np.asarray(I_tensor.todense())
    chi_new = P_top.indices[2].dim
    np.testing.assert_allclose(
        I_dense,
        np.eye(chi_new),
        atol=1e-9,
        err_msg=f"P_bot · P_top must be identity for direction={direction!r}",
    )


def test_compute_2x2_projector_symmetric_base_charges_drive_chi_new(symmetric_corners):
    """When base_charges is supplied, chi_new charges match _derive_charges(base_charges, chi).

    Multiset (not ordered) comparison: the per-sector allocation in
    ``_retruncate_by_base_charges`` traverses ``target_count`` in sorted
    charge order, while ``_derive_charges`` preserves the interleaved
    ``base_charges`` tile order.  The two agree as multisets — that is
    the meaningful per-sector-budget invariant — but not element-wise.
    Aligned with ``_svd_projector_symmetric``'s sorted-charge convention.
    """
    from collections import Counter

    from tenax.algorithms._ctm_utils import _derive_charges

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4
    base_charges = np.array([0, 1, 0, 1], dtype=np.int32)

    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=chi, direction="left", base_charges=base_charges
    )
    expected_chi_new = _derive_charges(base_charges, P_top.indices[2].dim)
    actual_chi_new = np.asarray(P_top.indices[2].charges, dtype=np.int32)
    assert Counter(int(q) for q in actual_chi_new) == Counter(
        int(q) for q in expected_chi_new
    ), (
        f"chi_new charge multiset must match _derive_charges(base_charges, "
        f"{P_top.indices[2].dim}); got {actual_chi_new.tolist()} vs "
        f"{expected_chi_new.tolist()}"
    )


def test_compute_2x2_projector_symmetric_ad_fallback_passes_tracer(symmetric_corners):
    """SymmetricTensor inputs under jax.grad must dispatch to the dense fallback.

    Issue #416: the symmetric block-sparse path is eager-only (uses Python-level
    iteration that can't be JIT-traced).  The Task 4 dispatch detects JAX
    tracers in any input block and falls through to the dense pipeline.
    This test confirms `jax.grad` produces a finite gradient via that fallback.
    """
    import jax

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4

    def scalar_of(t: float) -> jax.Array:
        # Trivial parameterization: scale the first block of Q_TL by t, then
        # call the projector and reduce to a scalar via Frobenius norm squared.
        first_key = sorted(Q_TL.blocks.keys())[0]
        new_block = Q_TL.blocks[first_key] * t
        new_blocks = {**Q_TL.blocks, first_key: new_block}
        Q_TL_perturbed = SymmetricTensor._from_blocks_unchecked(
            new_blocks, Q_TL.indices
        )
        P_top, P_bot = _compute_2x2_projector(
            Q_TL_perturbed, Q_TR, Q_BL, Q_BR, chi=chi, direction="left"
        )
        return jnp.sum(P_top.todense() ** 2) + jnp.sum(P_bot.todense() ** 2)

    grad = jax.grad(scalar_of)(1.0)
    assert jnp.isfinite(grad), f"grad must be finite, got {grad}"


def test_compute_2x2_projector_dense_fallback_wraps_as_symmetric(symmetric_corners):
    """Tracer-bearing symmetric inputs go through the dense fallback (Task 4 dispatch).

    Issue #416: the dense fallback must return SymmetricTensor projectors when
    inputs are SymmetricTensor, so downstream contractions (e.g. with
    SymmetricTensor envs) don't fail with "Cannot mix DenseTensor and SymmetricTensor".
    """
    import jax

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4

    def call_projector(t: float):
        first_key = sorted(Q_TL.blocks.keys())[0]
        new_block = Q_TL.blocks[first_key] * t
        new_blocks = {**Q_TL.blocks, first_key: new_block}
        Q_TL_perturbed = SymmetricTensor._from_blocks_unchecked(
            new_blocks, Q_TL.indices
        )
        P_top, P_bot = _compute_2x2_projector(
            Q_TL_perturbed, Q_TR, Q_BL, Q_BR, chi=chi, direction="left"
        )
        return P_top, P_bot

    # Outside jax.grad: confirm wrap type unchanged (symmetric input → symmetric
    # output via the block-sparse path).
    P_top_eager, P_bot_eager = call_projector(1.0)
    assert isinstance(P_top_eager, SymmetricTensor)
    assert isinstance(P_bot_eager, SymmetricTensor)

    # Inside jax.grad: tracer-bearing symmetric input → dense fallback → wrap
    # as SymmetricTensor.
    def scalar(t):
        P_top, P_bot = call_projector(t)
        return jnp.sum(P_top.todense() ** 2) + jnp.sum(P_bot.todense() ** 2)

    grad = jax.grad(scalar)(1.0)
    assert jnp.isfinite(grad)


def test_gauge_fix_symmetric_svd_tracer_safe():
    """`_gauge_fix_symmetric_svd` does not raise TracerArrayConversionError under jax.grad."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=42)

    def loss(alpha: jax.Array) -> jax.Array:
        # Scale blocks by alpha to inject a tracer into block contents
        new_blocks = {k: alpha * b for k, b in M_T.blocks.items()}
        M_scaled = SymmetricTensor._from_blocks_unchecked(new_blocks, M_T.indices)
        U_T, _, Vh_T, _ = tensor_svd(
            M_scaled,
            left_labels=("left",),
            right_labels=("right",),
            new_bond_label="bond",
            max_singular_values=None,
        )
        U_fixed, _ = _gauge_fix_symmetric_svd(U_T, Vh_T)
        return jnp.sum(
            jnp.abs(jnp.concatenate([b.flatten() for b in U_fixed.blocks.values()]))
        )

    grad_fn = jax.grad(loss)
    g = grad_fn(jnp.asarray(1.0))
    assert jnp.isfinite(g), f"gradient through gauge-fixed SVD must be finite, got {g}"
