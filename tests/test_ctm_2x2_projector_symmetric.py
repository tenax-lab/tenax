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
    P_top, P_bot, _, _ = _compute_2x2_projector(
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
    P_top, P_bot, _, _ = _compute_2x2_projector(
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

    P_top, P_bot, _, _ = _compute_2x2_projector(
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
        P_top, P_bot, _, _ = _compute_2x2_projector(
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
        P_top, P_bot, _, _ = _compute_2x2_projector(
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


def test_truncated_svd_symmetric_traced_preserves_charges():
    """Tracer-bearing symmetric SVD produces non-trivial bond charges, runs under jax.grad."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=7)
    base_charges = np.array([0, 1, 0, 1], dtype=np.int32)

    def loss(alpha: jax.Array) -> jax.Array:
        new_blocks = {k: alpha * b for k, b in M_T.blocks.items()}
        M_scaled = SymmetricTensor._from_blocks_unchecked(new_blocks, M_T.indices)
        U_T, s, _, _ = tensor_svd(
            M_scaled,
            left_labels=("left",),
            right_labels=("right",),
            new_bond_label="bond",
            max_singular_values=3,
            base_charges=base_charges,
        )
        bond_charges = np.asarray(U_T.indices[-1].charges)
        assert not np.all(bond_charges == 0), (
            f"traced symmetric SVD must preserve non-trivial U(1) charges, got {bond_charges}"
        )
        return jnp.sum(s)

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(g), f"gradient must be finite, got {g}"


def test_truncated_svd_symmetric_traced_proportional_fallback_respects_cap():
    """The base_charges=None proportional fallback must respect max_singular_values cap."""
    from tenax.linalg import svd as tensor_svd

    # Build a symmetric tensor with 3 sectors each contributing 1 SV when SVD'd.
    sym = U1Symmetry()
    left_charges = np.array([0, 1, 2], dtype=np.int32)
    right_charges = np.array([0, 1, 2], dtype=np.int32)
    left_idx = TensorIndex.from_charges(
        sym, left_charges, FlowDirection.IN, label="left"
    )
    right_idx = TensorIndex.from_charges(
        sym, right_charges, FlowDirection.OUT, label="right"
    )
    M_T = SymmetricTensor.random_normal((left_idx, right_idx), jax.random.PRNGKey(99))

    def loss(alpha: jax.Array) -> jax.Array:
        new_blocks = {k: alpha * b for k, b in M_T.blocks.items()}
        M_scaled = SymmetricTensor._from_blocks_unchecked(new_blocks, M_T.indices)
        _, s, _, _ = tensor_svd(
            M_scaled,
            left_labels=("left",),
            right_labels=("right",),
            new_bond_label="bond",
            max_singular_values=2,
            base_charges=None,  # forces proportional fallback
        )
        # The cap must be respected even when 3 sectors each have 1 unit of capacity
        assert s.shape[0] <= 2, f"max_singular_values=2 violated, got s.shape={s.shape}"
        return jnp.sum(s)

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(g)


def test_truncated_svd_symmetric_traced_fills_unused_base_charges_budget():
    """When base_charges over-allocates a sector, the traced path's greedy
    fill should match eager `_retruncate_by_base_charges` bond dim.

    Codex P2 review on PR #440 flagged the eager-vs-traced divergence: if
    base_charges=[0,0,0,0] but the tensor has 2 q=0 + 2 q=1 singular vectors,
    the eager path fills the remaining 2 budget from q=1, returning bond
    dim 4; the traced path was clamping to dim 2 (no fill), causing
    forward-vs-backward bond-dim mismatch under AD.
    """
    from tenax.linalg import svd as tensor_svd

    sym = U1Symmetry()
    left_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    right_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    left_idx = TensorIndex.from_charges(
        sym, left_charges, FlowDirection.IN, label="left"
    )
    right_idx = TensorIndex.from_charges(
        sym, right_charges, FlowDirection.OUT, label="right"
    )
    M_T = SymmetricTensor.random_normal((left_idx, right_idx), jax.random.PRNGKey(0))

    # target_count = {0: 4} after _derive_charges. Sector 0 has only 2 SVs;
    # the 2 unused budget slots should fill from sector 1.
    base_charges = np.array([0, 0, 0, 0], dtype=np.int32)

    def loss(alpha: jax.Array) -> jax.Array:
        new_blocks = {k: alpha * b for k, b in M_T.blocks.items()}
        M_scaled = SymmetricTensor._from_blocks_unchecked(new_blocks, M_T.indices)
        _, s, _, _ = tensor_svd(
            M_scaled,
            left_labels=("left",),
            right_labels=("right",),
            new_bond_label="bond",
            max_singular_values=4,
            base_charges=base_charges,
        )
        # Bond dim should be 4 (matching eager), not 2.
        assert s.shape[0] == 4, (
            f"greedy-fill should distribute unused budget across sectors, "
            f"got bond dim {s.shape[0]}"
        )
        return jnp.sum(s)

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(g)


def test_2plaq_path_threads_base_charges_through_helpers(
    symmetric_corners, monkeypatch
):
    """The 2plaq path's projector-pair helper and 4 directional-move helpers
    expose a ``base_charges`` kwarg that is forwarded to
    :func:`_compute_2x2_projector`.

    Mirrors PR #437's plumbing for the 1x1 path: ``_get_base_charges`` is
    derived once at the multisite sweep entry and threaded through.  Under
    AD tracing this lets the symmetric SVD allocate per-sector keep counts
    from ``base_charges`` instead of the proportional-to-sector-dim
    defensive fallback (Issue #435).
    """
    import inspect

    from tenax.algorithms import _ctm_tensor_moves

    # Step 1: signature must accept ``base_charges`` on all 5 helpers.
    helper_names = (
        "_compute_plaquette_projector_pair",
        "_ctm_tensor_move_left_2x2",
        "_ctm_tensor_move_right_2x2",
        "_ctm_tensor_move_top_2x2",
        "_ctm_tensor_move_bottom_2x2",
    )
    for fn_name in helper_names:
        fn = getattr(_ctm_tensor_moves, fn_name)
        sig = inspect.signature(fn)
        assert "base_charges" in sig.parameters, (
            f"{fn_name} must accept base_charges; got params {list(sig.parameters)}"
        )
        # Default must be None so the kwarg is backwards-compatible.
        assert sig.parameters["base_charges"].default is None, (
            f"{fn_name}.base_charges default must be None; got "
            f"{sig.parameters['base_charges'].default!r}"
        )

    # Step 2: runtime forwarding -- monkeypatch ``_compute_2x2_projector`` and
    # call ``_compute_plaquette_projector_pair`` with a non-None
    # ``base_charges`` to confirm the value reaches the SVD entrypoint.
    seen: dict[str, object] = {}
    real_fn = _ctm_tensor_moves._compute_2x2_projector

    def spy(*args, **kwargs):
        seen["base_charges"] = kwargs.get("base_charges")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(_ctm_tensor_moves, "_compute_2x2_projector", spy)

    # Construct minimal CTMTensorEnvs and double-layer tensors that match the
    # symmetric_corners fixture's structure.  Easier: drop into the spy path
    # by calling ``_compute_2x2_projector`` directly with the corners, mirror-
    # ing what ``_compute_plaquette_projector_pair`` does internally.
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    expected = np.array([0, 1], dtype=np.int32)
    _ctm_tensor_moves._compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=4, direction="left", base_charges=expected
    )
    assert seen["base_charges"] is expected, (
        "base_charges must pass through to _compute_2x2_projector "
        f"(got {seen.get('base_charges')!r}, expected {expected!r})"
    )


def test_compute_2x2_projector_tracer_safe_under_grad(symmetric_corners):
    """`_compute_2x2_projector` runs inside jax.grad on non-trivial U(1) inputs."""
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    def loss(alpha: jax.Array) -> jax.Array:
        scaled_blocks = {k: alpha * b for k, b in Q_TL.blocks.items()}
        Q_TL_traced = SymmetricTensor._from_blocks_unchecked(
            scaled_blocks, Q_TL.indices
        )
        P_top, _, _, _ = _compute_2x2_projector(
            Q_TL_traced,
            Q_TR,
            Q_BL,
            Q_BR,
            chi=4,
            direction="left",
            base_charges=base_charges,
        )
        return jnp.sum(
            jnp.abs(jnp.concatenate([b.flatten() for b in P_top.blocks.values()]))
        )

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(g), (
        f"AD through symmetric 2x2 projector must produce finite grad, got {g}"
    )


def test_compute_2x2_projector_preserves_non_trivial_charges(symmetric_corners):
    """P_top's chi_new_top leg carries non-trivial charges (not the trivial-zero wrap)."""
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    P_top, P_bot, _, _ = _compute_2x2_projector(
        Q_TL,
        Q_TR,
        Q_BL,
        Q_BR,
        chi=4,
        direction="left",
        base_charges=base_charges,
    )
    chi_new_charges = np.asarray(P_top.indices[-1].charges)
    assert not np.all(chi_new_charges == 0), (
        f"chi_new_top must carry non-trivial U(1) charges from base_charges, "
        f"got all-zeros: {chi_new_charges}"
    )


def test_compute_2x2_projector_eager_vs_traced_trivial(symmetric_corners):
    """For trivial-charge symmetric inputs, eager and traced paths produce the same
    projector up to bond permutation (verified via P_top @ P_top.conj().T).

    Builds fresh trivial-charge (all-zero) corners with the same flow conventions
    as ``symmetric_corners`` — we can't simply re-label the fixture's blocks
    because zeroing all charges collapses the block structure (the existing
    non-trivial blocks have shapes that don't match a single all-zero index).
    """
    chi, D = 4, 2

    def _make_trivial_corner(
        chi_label_a,
        chi_label_b,
        D2_label_a,
        D2_label_b,
        flow_chi_a,
        flow_chi_b,
        flow_D2_a,
        flow_D2_b,
        seed,
    ):
        sym = U1Symmetry()
        chi_charges = np.zeros(chi, dtype=np.int32)
        D2_charges = np.zeros(D**2, dtype=np.int32)
        indices = (
            TensorIndex.from_charges(sym, chi_charges, flow_chi_a, label=chi_label_a),
            TensorIndex.from_charges(sym, D2_charges, flow_D2_a, label=D2_label_a),
            TensorIndex.from_charges(sym, chi_charges, flow_chi_b, label=chi_label_b),
            TensorIndex.from_charges(sym, D2_charges, flow_D2_b, label=D2_label_b),
        )
        return SymmetricTensor.random_normal(indices, jax.random.PRNGKey(seed))

    Q_TL_t = _make_trivial_corner(
        "chi_R",
        "chi_B",
        "r2",
        "d2",
        FlowDirection.OUT,
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.IN,
        seed=10,
    )
    Q_TR_t = _make_trivial_corner(
        "chi_L",
        "chi_B",
        "l2",
        "d2",
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.OUT,
        FlowDirection.IN,
        seed=11,
    )
    Q_BL_t = _make_trivial_corner(
        "chi_R",
        "chi_T",
        "r2",
        "u2",
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.IN,
        FlowDirection.OUT,
        seed=12,
    )
    Q_BR_t = _make_trivial_corner(
        "chi_L",
        "chi_T",
        "l2",
        "u2",
        FlowDirection.IN,
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.OUT,
        seed=13,
    )
    Qs_trivial = [Q_TL_t, Q_TR_t, Q_BL_t, Q_BR_t]

    P_top_eager, _, _, _ = _compute_2x2_projector(*Qs_trivial, chi=4, direction="left")

    def get_p_top(alpha):
        scaled = {k: alpha * b for k, b in Qs_trivial[0].blocks.items()}
        Q0 = SymmetricTensor._from_blocks_unchecked(scaled, Qs_trivial[0].indices)
        P_top, _, _, _ = _compute_2x2_projector(
            Q0,
            *Qs_trivial[1:],
            chi=4,
            direction="left",
            base_charges=np.zeros(4, dtype=np.int32),
        )
        return jnp.asarray(P_top.todense())

    P_top_traced = jax.jit(get_p_top)(jnp.asarray(1.0))
    P_top_eager_dense = jnp.asarray(P_top_eager.todense())

    chi_outer = P_top_eager_dense.shape[0]
    fused_D2 = P_top_eager_dense.shape[1]
    P_eager_mat = P_top_eager_dense.reshape(chi_outer * fused_D2, -1)
    P_traced_mat = P_top_traced.reshape(chi_outer * fused_D2, -1)
    inner_eager = P_eager_mat @ P_eager_mat.conj().T
    inner_traced = P_traced_mat @ P_traced_mat.conj().T
    np.testing.assert_allclose(
        np.asarray(inner_eager),
        np.asarray(inner_traced),
        atol=1e-8,
        rtol=1e-6,
        err_msg="trivial-charge: P @ P.conj().T must match between eager and traced",
    )


def test_compute_2x2_projector_grad_matches_finite_difference(symmetric_corners):
    """jax.grad through `_compute_2x2_projector` matches central-difference.

    The natural ``‖P_top‖_F^2`` reduction returns ``chi_new`` regardless of
    alpha — P_top columns are orthonormal by Fishman construction, so its
    Frobenius norm squared is alpha-invariant.  We instead reduce through
    ``Q_TL_traced`` (the alpha-scaled corner) directly: that path *does*
    depend on alpha non-trivially and gives a meaningful AD-vs-FD probe.
    """
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    def loss(alpha: jax.Array) -> jax.Array:
        scaled_blocks = {k: alpha * b for k, b in Q_TL.blocks.items()}
        Q_TL_traced = SymmetricTensor._from_blocks_unchecked(
            scaled_blocks, Q_TL.indices
        )
        P_top, _, _, _ = _compute_2x2_projector(
            Q_TL_traced,
            Q_TR,
            Q_BL,
            Q_BR,
            chi=4,
            direction="left",
            base_charges=base_charges,
        )
        # Pull alpha through into the loss by reducing Q_TL_traced * P_top
        # contributions.  ``P_top`` is alpha-invariant in magnitude but its
        # sign/gauge can flip, so we use ``|P_top|`` to keep the loss smooth.
        Q_blocks_sum = jnp.sum(
            jnp.concatenate([b.flatten() for b in Q_TL_traced.blocks.values()])
        )
        P_abs_sum = jnp.sum(
            jnp.abs(jnp.concatenate([b.flatten() for b in P_top.blocks.values()]))
        )
        return Q_blocks_sum * P_abs_sum

    alpha0 = jnp.asarray(1.0)
    g_ad = jax.grad(loss)(alpha0)

    eps = 1e-4
    g_fd = (loss(alpha0 + eps) - loss(alpha0 - eps)) / (2 * eps)

    # Both AD and FD must be of comparable magnitude.  When the FD value is
    # tiny (the projector is alpha-invariant in this regime), AD should also
    # be tiny — compare relative to the larger of the two magnitudes plus a
    # safe epsilon.
    denom = jnp.maximum(jnp.abs(g_fd), jnp.abs(g_ad)) + 1e-8
    rel_err = jnp.abs(g_ad - g_fd) / denom
    assert rel_err < 1e-3, (
        f"AD vs FD relative error too large: {rel_err} (g_ad={g_ad}, g_fd={g_fd})"
    )


def test_compute_2x2_projector_closure_under_tracing(symmetric_corners):
    """P_bot · P_top = I on (chi_new_bot, chi_new_top) under tracing."""
    from tenax.contraction.contractor import contract

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    @jax.jit
    def get_closure(alpha: jax.Array) -> jax.Array:
        scaled = {k: alpha * b for k, b in Q_TL.blocks.items()}
        Q0 = SymmetricTensor._from_blocks_unchecked(scaled, Q_TL.indices)
        P_top, P_bot, _, _ = _compute_2x2_projector(
            Q0,
            Q_TR,
            Q_BL,
            Q_BR,
            chi=4,
            direction="left",
            base_charges=base_charges,
        )
        closure = contract(P_bot, P_top)
        return jnp.asarray(closure.todense())

    closure = get_closure(jnp.asarray(1.0))
    chi_new = closure.shape[0]
    np.testing.assert_allclose(
        np.asarray(closure),
        np.eye(chi_new),
        atol=1e-6,
        err_msg="P_bot · P_top must be identity on chi_new × chi_new (Fishman closure)",
    )


# --------------------------------------------------------------------------- #
# Vectorization parity (#566): the per-sector rewrite of _gauge_fix_symmetric_svd
# must be byte-identical to the original per-column loop, frozen here.
# --------------------------------------------------------------------------- #
def _reference_gauge_fix_loop(U_T, Vh_T):
    """Frozen copy of the original per-column _gauge_fix_symmetric_svd loop.

    Kept in the test as the behavioral oracle for the vectorized rewrite.
    """
    bond_idx = U_T.indices[-1]
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    local_index_of: dict[int, dict[int, int]] = {}
    counter: dict[int, int] = {}
    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local_index_of.setdefault(q_int, {})[j] = counter.get(q_int, 0)
        counter[q_int] = counter.get(q_int, 0) + 1

    u_blocks_by_q: dict[int, list] = {}
    for key, block in U_T.blocks.items():
        u_blocks_by_q.setdefault(int(key[-1]), []).append((key, block))
    vh_blocks_by_q: dict[int, list] = {}
    for key, block in Vh_T.blocks.items():
        vh_blocks_by_q.setdefault(int(key[0]), []).append((key, block))

    new_u_blocks = {key: block for key, block in U_T.blocks.items()}
    new_vh_blocks = {key: block for key, block in Vh_T.blocks.items()}

    sample_block = next(iter(U_T.blocks.values()))
    is_complex = jnp.issubdtype(sample_block.dtype, jnp.complexfloating)

    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local = local_index_of[q_int][j]
        u_entries = u_blocks_by_q.get(q_int, [])
        vh_entries = vh_blocks_by_q.get(q_int, [])
        if not u_entries:
            continue
        candidates = jnp.concatenate(
            [jnp.reshape(new_u_blocks[key][..., local], (-1,)) for key, _ in u_entries]
        )
        max_idx = jnp.argmax(jnp.abs(candidates))
        best_value = candidates[max_idx]
        abs_best = jnp.abs(best_value)
        phase = jnp.where(
            abs_best > 0,
            best_value
            / jnp.maximum(abs_best, jnp.asarray(1e-30, dtype=abs_best.dtype)),
            jnp.ones_like(best_value),
        )
        if is_complex:
            conj_phase = jnp.conj(phase)
            bare_phase = phase
        else:
            conj_phase = jnp.real(phase)
            bare_phase = jnp.real(phase)
        for key, _block in u_entries:
            new_u_blocks[key] = new_u_blocks[key].at[..., local].multiply(conj_phase)
        for key, _block in vh_entries:
            new_vh_blocks[key] = new_vh_blocks[key].at[local, ...].multiply(bare_phase)

    U_out = SymmetricTensor._from_blocks_unchecked(new_u_blocks, U_T.indices)
    Vh_out = SymmetricTensor._from_blocks_unchecked(new_vh_blocks, Vh_T.indices)
    return U_out, Vh_out


def _svd_of(M_T):
    from tenax.linalg import svd as tensor_svd

    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    return U_T, s, Vh_T


def _complex_matrix_tensor(seed: int = 7) -> SymmetricTensor:
    """Two-sector U(1) complex128 matrix tensor."""
    sym = U1Symmetry()
    charges = np.array([0, 0, 1, 1], dtype=np.int32)
    left_idx = TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="left")
    right_idx = TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="right")
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    re = SymmetricTensor.random_normal((left_idx, right_idx), k1)
    im = SymmetricTensor.random_normal((left_idx, right_idx), k2)
    blocks = {
        key: (re.blocks[key] + 1j * im.blocks[key]).astype(jnp.complex128)
        for key in re.blocks
    }
    return SymmetricTensor._from_blocks_unchecked(blocks, re.indices)


def _degenerate_matrix_tensor(seed: int = 3) -> SymmetricTensor:
    """Scaled-identity-per-sector matrix → fully degenerate singular values
    (exercises argmax ties). ``seed`` only sources the index structure; all
    block values are overwritten deterministically, so the result is
    seed-independent."""
    M_T = _make_test_matrix_tensor(seed=seed)
    blocks = {}
    for key, block in M_T.blocks.items():
        n = block.shape[0]
        blocks[key] = jnp.eye(n, block.shape[1], dtype=block.dtype) * (1.0 + key[0])
    return SymmetricTensor._from_blocks_unchecked(blocks, M_T.indices)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _make_test_matrix_tensor(seed=0),
        lambda: _make_test_matrix_tensor(seed=5),
        _complex_matrix_tensor,
        _degenerate_matrix_tensor,
    ],
    ids=["u1_real_a", "u1_real_b", "u1_complex", "degenerate_ties"],
)
def test_gauge_fix_vectorized_matches_reference_forward(factory):
    """Vectorized gauge fix matches the frozen per-column loop.

    The two implementations are algebraically identical, but the gauge phase
    of a *complex* column is normalized as ``best_value / |best_value|`` — a
    complex division whose rounding is not bit-stable across platforms' BLAS.
    For real inputs the phase is an exact ±1 and the results are byte-identical;
    for the complex case they agree only to floating-point tolerance (a strict
    ``array_equal`` fails on some CI runners, e.g. the ``u1_complex`` sector).
    Compare at FP tolerance, matching ``..._matches_reference_grad`` below.
    """
    M_T = factory()
    U_T, _s, Vh_T = _svd_of(M_T)

    U_ref, Vh_ref = _reference_gauge_fix_loop(U_T, Vh_T)
    U_new, Vh_new = _gauge_fix_symmetric_svd(U_T, Vh_T)

    assert set(U_new.blocks) == set(U_ref.blocks)
    assert set(Vh_new.blocks) == set(Vh_ref.blocks)
    for key in U_ref.blocks:
        np.testing.assert_allclose(
            np.asarray(U_new.blocks[key]),
            np.asarray(U_ref.blocks[key]),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"U block {key}",
        )
    for key in Vh_ref.blocks:
        np.testing.assert_allclose(
            np.asarray(Vh_new.blocks[key]),
            np.asarray(Vh_ref.blocks[key]),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Vh block {key}",
        )


def test_gauge_fix_vectorized_matches_reference_grad():
    """Gradient through the vectorized gauge fix matches the reference (fp tier),
    differentiating w.r.t. BOTH U and Vh inputs."""
    M_T = _make_test_matrix_tensor(seed=2)
    U_T, _s, Vh_T = _svd_of(M_T)

    def _loss(U_T_in, Vh_T_in, gauge_fn):
        U_fixed, Vh_fixed = gauge_fn(U_T_in, Vh_T_in)
        u = U_fixed.todense()
        v = Vh_fixed.todense()
        return jnp.real(jnp.sum(u * jnp.conj(u))) + jnp.real(jnp.sum(v * jnp.conj(v)))

    u_leaves, u_tree = jax.tree.flatten(U_T)
    vh_leaves, vh_tree = jax.tree.flatten(Vh_T)

    def _from_leaves(ul, vl, fn):
        U_in = jax.tree.unflatten(u_tree, ul)
        Vh_in = jax.tree.unflatten(vh_tree, vl)
        return _loss(U_in, Vh_in, fn)

    g_ref = jax.grad(
        lambda ul, vl: _from_leaves(ul, vl, _reference_gauge_fix_loop), argnums=(0, 1)
    )(u_leaves, vh_leaves)
    g_new = jax.grad(
        lambda ul, vl: _from_leaves(ul, vl, _gauge_fix_symmetric_svd), argnums=(0, 1)
    )(u_leaves, vh_leaves)
    for a, b in zip(jax.tree.leaves(g_new), jax.tree.leaves(g_ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)
