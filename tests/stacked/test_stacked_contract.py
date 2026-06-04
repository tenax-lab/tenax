"""Tests for the even-D stacked block-sparse contraction path (#566 P1b).

The stacked path (gated by ``TENAX_STACK_BLOCKSPARSE``) sources its operands
directly from the contiguous ``_data`` buffer via ``stacked_blocks()`` +
``jnp.take`` (NOT per-block ``.blocks``/``_get_block`` slicing) and emits one
batched einsum + segment_sum per input shape-group. This must be byte-correct
in VALUE and GRADIENT against the per-block path within the fp tier, and must
actually fire (counter) for even-D inputs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.contraction.contractor import _STACK_FIRED, contract
from tests.stacked._harness import assert_tiered, canonical_tensors


def _double_layer(A):
    """Contract ket A with bra A.bar() over ONLY the physical leg.

    Mirrors ``_phys_double_layer`` in tests/stacked/test_harness.py: virtual
    legs of the bra are relabelled (u/d/l/r -> U/D/L/R) so they survive as free
    output legs and the result stays a rank-8 SymmetricTensor.
    """
    bra = A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})
    return A, bra


@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_stacked_contract_matches_perblock(name, monkeypatch):
    A = dict(canonical_tensors())[name]
    A, Abar = _double_layer(A)

    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "0")
    ref = contract(A, Abar)
    n0 = _STACK_FIRED["n"]

    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    got = contract(A, Abar)

    assert _STACK_FIRED["n"] > n0, "stacked path did not fire"
    assert got._block_keys == ref._block_keys
    assert ref._data.size > 1, "double-layer collapsed to a scalar"
    assert_tiered(ref._data, got._data, tier="fp")


def test_stacked_contract_grad_matches_perblock(monkeypatch):
    A0 = dict(canonical_tensors())["ferm_D2"]
    A0bar_src = A0.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})

    indices = A0._indices
    block_keys = A0._block_keys
    block_shapes = A0._block_shapes
    block_offsets = A0._block_offsets

    bar_indices = A0bar_src._indices
    bar_keys = A0bar_src._block_keys
    bar_shapes = A0bar_src._block_shapes
    bar_offsets = A0bar_src._block_offsets

    def loss(data):
        A = type(A0)._raw(
            indices=indices,
            data=data,
            block_keys=block_keys,
            block_shapes=block_shapes,
            block_offsets=block_offsets,
        )
        # bra shares ket's data buffer (bar() conjugates; for real x64 it is
        # the same buffer, so differentiating w.r.t. one data vector is valid).
        Abar = type(A0)._raw(
            indices=bar_indices,
            data=jnp.conj(data),
            block_keys=bar_keys,
            block_shapes=bar_shapes,
            block_offsets=bar_offsets,
        )
        out = contract(A, Abar)
        return jnp.sum(out._data**2)

    data0 = A0._data

    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "0")
    g_ref = jax.grad(loss)(data0)
    n0 = _STACK_FIRED["n"]

    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    g_got = jax.grad(loss)(data0)

    assert _STACK_FIRED["n"] > n0, "stacked path did not fire under grad"
    assert_tiered(g_ref, g_got, tier="fp")


def test_stacked_falls_back_for_odd_d(monkeypatch):
    """D=3 has multiple shape-groups -> stacked path must return None (fall back)."""
    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _build_initial_fpeps_tensor,
    )

    A = _build_initial_fpeps_tensor(FPEPSConfig(D=3), jax.random.PRNGKey(0))
    A, Abar = _double_layer(A)

    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "0")
    ref = contract(A, Abar)
    n0 = _STACK_FIRED["n"]

    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    got = contract(A, Abar)

    assert _STACK_FIRED["n"] == n0, "stacked path fired on multi-shape (odd-D) input"
    assert got._block_keys == ref._block_keys
    assert_tiered(ref._data, got._data, tier="fp")
