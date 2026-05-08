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


def test_pad_raises_for_symmetric_tensor_env():
    """SymmetricTensor envs are out of scope for v1 — must raise cleanly."""
    sym = U1Symmetry()
    charges = np.zeros(2, dtype=np.int32)
    phys_charges = np.zeros(2, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    rng = np.random.RandomState(42)
    data = jnp.array(rng.standard_normal((2, 2, 2, 2, 2)))
    peps_sym = SymmetricTensor.from_dense(data, indices)
    env_sym = initialize_ctm_tensor_env(peps_sym, chi=4)
    with pytest.raises(NotImplementedError, match="v2 follow-up"):
        pad_dense_env_chi(env_sym, chi_new=6)
