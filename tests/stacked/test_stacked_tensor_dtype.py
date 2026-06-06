"""Regression: StackedSymmetricTensor scalar multiply must track promoted dtype.

A scalar that promotes the dtype (e.g. a real tensor times ``1j``) makes the
cached group arrays complex. The materialization buffer
(:func:`~tenax.core.stacked_view.scatter_stacked`) is allocated with the stored
dtype, so if ``__mul__`` recorded the STALE real dtype the scatter would cast
the complex rows back to real and silently drop the imaginary part (Codex P2).
"""

import jax
import jax.numpy as jnp
import pytest

from tenax.contraction.contractor import contract
from tenax.core.stacked_tensor import StackedSymmetricTensor
from tests.stacked._harness import assert_tiered, canonical_tensors


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_stacked(monkeypatch):
    """Produce a real-dtype StackedSymmetricTensor via the stacked contraction."""
    A = dict(canonical_tensors())["ferm_D2"]
    bra = A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})
    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    out = contract(A, bra)
    assert isinstance(out, StackedSymmetricTensor)
    return A, bra, out


def test_stacked_scalar_multiply_promotes_dtype(monkeypatch):
    A, bra, out = _make_stacked(monkeypatch)
    assert jnp.dtype(out.__dict__["_dtype"]) == jnp.float64

    out2 = out * 1j
    # Stored dtype must follow the promoted (complex) group arrays.
    assert jnp.dtype(out2.__dict__["_dtype"]) == jnp.complex128
    for grp in out2.stacked_blocks().groups.values():
        assert grp.array.dtype == jnp.complex128

    # Materialization must preserve the imaginary part, not cast it away.
    data = out2._data
    assert data.dtype == jnp.complex128
    assert bool(jnp.any(jnp.imag(data) != 0)), "imaginary part dropped on scatter"

    # Value matches the same op on the plain per-block path.
    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "0")
    ref2 = contract(A, bra) * 1j
    assert ref2._data.dtype == jnp.complex128
    assert_tiered(ref2._data, data, tier="fp")
