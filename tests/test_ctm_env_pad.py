"""Unit tests for zero-padding CTMTensorEnv from chi_old to chi_new."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv, initialize_ctm_tensor_env
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _idx(dim: int, label: str, flow: FlowDirection) -> TensorIndex:
    sym = U1Symmetry()
    return TensorIndex.from_charges(
        sym, np.zeros(dim, dtype=np.int32), flow, label=label
    )


def _make_dense_env(chi: int, D: int, key=None) -> CTMTensorEnv:
    """Build a CTMTensorEnv with all 8 tensors as DenseTensor.

    Corners: shape (chi, chi). Edges: shape (chi, D**2, chi).
    The actual flow conventions follow CTMTensorEnv field comments —
    use IN/OUT pairs that produce sensible double-layer contractions.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 8)
    Csh = (chi, chi)
    Tsh = (chi, D**2, chi)
    chi_in = _idx(chi, "chi", FlowDirection.IN)
    chi_out = _idx(chi, "chi", FlowDirection.OUT)
    d_idx = _idx(D**2, "u2", FlowDirection.IN)

    def make_corner(k):
        return DenseTensor(jax.random.normal(k, Csh), (chi_in, chi_out))

    def make_edge(k):
        return DenseTensor(jax.random.normal(k, Tsh), (chi_in, d_idx, chi_out))

    return CTMTensorEnv(
        C1=make_corner(keys[0]),
        C2=make_corner(keys[1]),
        C3=make_corner(keys[2]),
        C4=make_corner(keys[3]),
        T1=make_edge(keys[4]),
        T2=make_edge(keys[5]),
        T3=make_edge(keys[6]),
        T4=make_edge(keys[7]),
    )


def test_pad_extends_corner_axes_to_new_chi():
    env_old = _make_dense_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    assert env_new.C1._data.shape == (6, 6)
    assert env_new.C4._data.shape == (6, 6)


def test_pad_extends_edge_chi_axes_only():
    env_old = _make_dense_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    # edges are (chi, D², chi); D² axis (=4) unchanged.
    assert env_new.T1._data.shape == (6, 4, 6)
    assert env_new.T4._data.shape == (6, 4, 6)


def test_pad_preserves_existing_block():
    env_old = _make_dense_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    assert jnp.allclose(env_new.C1._data[:4, :4], env_old.C1._data)
    assert jnp.allclose(env_new.T1._data[:4, :, :4], env_old.T1._data)


def test_pad_fills_new_block_with_zero():
    env_old = _make_dense_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=6)
    assert float(jnp.max(jnp.abs(env_new.C1._data[4:, :]))) == 0.0
    assert float(jnp.max(jnp.abs(env_new.T1._data[4:, :, :]))) == 0.0


def test_pad_noop_when_chi_unchanged():
    env_old = _make_dense_env(chi=4, D=2)
    env_new = pad_dense_env_chi(env_old, chi_new=4)
    # Identity check on the env: same object since nothing changed.
    assert env_new is env_old


def test_pad_rejects_shrink():
    env_old = _make_dense_env(chi=4, D=2)
    with pytest.raises(ValueError, match="must be >="):
        pad_dense_env_chi(env_old, chi_new=2)


# ---------------------------------------------------------------------------
# Issue #410: SymmetricTensor env padding.
#
# The single-site dense auto-χ_E bump (PR #412) only knows how to grow the
# χ axes of a DenseTensor env. Block-sparse envs raised NotImplementedError.
# Issue #410 lifts that restriction by tiling the existing χ-leg charges
# via ``_derive_charges`` (the same allocator used by the CTM projectors)
# and zero-padding each block along its χ axes.
# ---------------------------------------------------------------------------


def _make_u1_peps(D: int, d: int, seed: int = 42) -> SymmetricTensor:
    """Build a 5-leg U(1) iPEPS site tensor with non-trivial bond charges."""
    sym = U1Symmetry()
    # Non-trivial bond charges: half q=0, half q=1.  Tile/truncate to D.
    bond_charges = np.array([0, 1] * ((D + 1) // 2))[:D].astype(np.int32)
    phys_charges = np.array([0, 1] * ((d + 1) // 2))[:d].astype(np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return SymmetricTensor.random_normal(indices, jax.random.PRNGKey(seed))


def test_pad_symmetric_corner_axes_extend_to_new_chi():
    """Padding a symmetric corner grows both χ axes to ``chi_new``."""
    A_sym = _make_u1_peps(D=2, d=2)
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    env_new = pad_dense_env_chi(env_old, chi_new=8)
    assert env_new.C1.indices[0].dim == 8
    assert env_new.C1.indices[1].dim == 8
    assert env_new.C4.indices[0].dim == 8


def test_pad_symmetric_edge_chi_axes_only():
    """Padding a symmetric edge grows axes 0 and 2 (the χ axes); axis 1 (D²) untouched."""
    A_sym = _make_u1_peps(D=2, d=2)
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    D2_dim_old = env_old.T1.indices[1].dim
    env_new = pad_dense_env_chi(env_old, chi_new=8)
    assert env_new.T1.indices[0].dim == 8
    assert env_new.T1.indices[2].dim == 8
    assert env_new.T1.indices[1].dim == D2_dim_old  # D² leg unchanged


def test_pad_symmetric_uses_derive_charges_for_new_slots():
    """New χ-axis charges follow ``_derive_charges`` (tile of old charges)."""
    from tenax.algorithms._ctm_utils import _derive_charges

    A_sym = _make_u1_peps(D=2, d=2)
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    env_new = pad_dense_env_chi(env_old, chi_new=10)
    expected_charges = _derive_charges(env_old.C1.indices[0].charges, 10)
    assert np.array_equal(env_new.C1.indices[0].charges, expected_charges)
    # Edges share the same chi-charge tile as the corners.
    assert np.array_equal(env_new.T1.indices[0].charges, expected_charges)


def test_pad_symmetric_preserves_existing_blocks():
    """Existing block data is preserved under padding (only zeros appended)."""
    A_sym = _make_u1_peps(D=2, d=2)
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    env_new = pad_dense_env_chi(env_old, chi_new=8)

    # Densify both envs along the C1 corner and compare leading 4×4 block.
    C1_old_dense = env_old.C1.todense()
    C1_new_dense = env_new.C1.todense()
    # The padded corner should embed the old corner in its leading χ_old × χ_old block.
    assert C1_new_dense.shape == (8, 8)
    # Old charge order is preserved as a prefix of the new charge order (tile).
    chi_old = C1_old_dense.shape[0]
    assert jnp.allclose(C1_new_dense[:chi_old, :chi_old], C1_old_dense)


def test_pad_symmetric_fills_new_block_with_zero():
    """Newly grown rows/cols of the corner are zero."""
    A_sym = _make_u1_peps(D=2, d=2)
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    env_new = pad_dense_env_chi(env_old, chi_new=8)
    C1_new_dense = env_new.C1.todense()
    # Beyond χ_old, the corner must be all zero (since the only nonzero entry
    # in the rank-1 initial corner is at index (0, 0) which is already inside
    # the χ_old × χ_old block).
    assert float(jnp.max(jnp.abs(C1_new_dense[4:, :]))) == 0.0
    assert float(jnp.max(jnp.abs(C1_new_dense[:, 4:]))) == 0.0


def test_pad_symmetric_noop_when_chi_unchanged():
    A_sym = _make_u1_peps(D=2, d=2)
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    env_new = pad_dense_env_chi(env_old, chi_new=4)
    assert env_new is env_old


def test_pad_symmetric_rejects_shrink():
    A_sym = _make_u1_peps(D=2, d=2)
    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    with pytest.raises(ValueError, match="must be >="):
        pad_dense_env_chi(env_old, chi_new=2)


def test_pad_symmetric_uses_base_charges_when_supplied():
    """Regression for PR #430 codex review: when ``base_charges`` is
    supplied, padded χ charges follow ``sorted(_derive_charges(...))``.

    The legacy policy (no ``base_charges``) tiles the existing χ-leg
    pattern via ``_derive_charges``. After a CTM sweep the χ-leg is
    already grouped sector-by-sector (the projector's
    ``chi_new_charges`` is built as ``[fq] * n_q for fq in sorted``),
    so tiling that grouped pattern grows only the leading sectors and
    starves the trailing ones — the next projector then can't allocate
    its ideal per-sector budget. Deriving the new χ charges from the
    projector's ``base_charges`` (the u2-fused charges) avoids that.
    """
    from tenax.algorithms._ctm_tensor_convergence import _get_base_charges
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
    from tenax.algorithms._ctm_utils import _derive_charges

    A_sym = _make_u1_peps(D=2, d=2)
    # Same base_charges the symmetric projector consumes — fused
    # (ket, bra) charges from the double-layer tensor's u2 leg.
    base_charges = _get_base_charges(_build_double_layer_tensor(A_sym))
    assert base_charges is not None, "test premise: non-trivial U(1) charges"

    env_old = initialize_ctm_tensor_env(A_sym, chi=4)
    env_new_legacy = pad_dense_env_chi(env_old, chi_new=6)
    env_new_with_base = pad_dense_env_chi(env_old, chi_new=6, base_charges=base_charges)

    expected = np.sort(_derive_charges(base_charges, 6)).astype(np.int32)
    assert np.array_equal(
        np.asarray(env_new_with_base.C1.indices[0].charges), expected
    ), (
        f"χ-leg charges should be sorted(_derive_charges(base_charges, 6)) = "
        f"{expected}, got {np.asarray(env_new_with_base.C1.indices[0].charges)}"
    )
    # Edges share the same chi-charge allocation.
    assert np.array_equal(np.asarray(env_new_with_base.T1.indices[0].charges), expected)
    # The two policies must disagree on this χ-leg pattern — otherwise
    # the fix is a no-op for this configuration.
    assert not np.array_equal(
        np.asarray(env_new_legacy.C1.indices[0].charges),
        np.asarray(env_new_with_base.C1.indices[0].charges),
    ), (
        "Legacy tile policy and base_charges policy must produce different "
        "χ-charges on D=2 init; otherwise this regression test is vacuous."
    )
