"""Tests for the block-sparse contraction backend protocol + selection (#200, A2).

``select_backend`` chooses a backend (or ``None`` => per-block fallback) from the
(static) plan + tensors, honoring ``TENAX_BLOCKSPARSE_BACKEND`` precedence and the
legacy ``TENAX_STACK_BLOCKSPARSE`` flag. With NO backend env set the default is
``None`` (per-block, byte-identical to today). The pure-JAX
:class:`StackedJaxBackend` must fire exactly as the legacy stacked path did, and
its value+gradient must match the per-block path within the fp tier. The
``cutensornet`` env value must be ACCEPTED (no error) and resolve to ``None`` until
Task B registers a real backend.
"""

import jax
import jax.numpy as jnp
import pytest

from tenax.contraction.blocksparse_backend import (
    BlockSparseContractBackend,
    StackedJaxBackend,
    select_backend,
)
from tenax.contraction.blocksparse_plan import build_block_contract_plan
from tenax.contraction.contractor import _STACK_FIRED, contract
from tests.stacked._harness import assert_tiered, canonical_tensors


def _double_layer(A):
    """Contract ket A with bra A.bar() over ONLY the physical leg (rank-8 out)."""
    bra = A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})
    return A, bra


def _make_plan():
    """Build a real even-D plan to drive ``select_backend`` / ``supports``."""
    A = dict(canonical_tensors())["ferm_D2"]
    A, Abar = _double_layer(A)
    from tenax.contraction.contractor import _labels_to_subscripts

    subscripts, out_indices = _labels_to_subscripts([A, Abar])
    plan = build_block_contract_plan([A, Abar], subscripts, out_indices)
    assert plan is not None, "fixture must produce an in-scope (even-D) plan"
    return [A, Abar], plan


# ----------------------------------------------------------------------------
# select_backend precedence
# ----------------------------------------------------------------------------


def _clear_env(monkeypatch):
    monkeypatch.delenv("TENAX_BLOCKSPARSE_BACKEND", raising=False)
    monkeypatch.delenv("TENAX_STACK_BLOCKSPARSE", raising=False)


def test_select_perblock_returns_none(monkeypatch):
    tensors, plan = _make_plan()
    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "perblock")
    assert select_backend(tensors, plan) is None


def test_select_stacked_returns_stacked_backend(monkeypatch):
    tensors, plan = _make_plan()
    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "stacked")
    backend = select_backend(tensors, plan)
    assert isinstance(backend, StackedJaxBackend)
    assert isinstance(backend, BlockSparseContractBackend)
    assert backend.name == "stacked"


def test_select_cutensornet_accepted_returns_none(monkeypatch):
    tensors, plan = _make_plan()
    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "cutensornet")
    # Accepted (no error), resolves to None until Task B registers a backend.
    assert select_backend(tensors, plan) is None


def test_select_legacy_flag_returns_stacked_backend(monkeypatch):
    tensors, plan = _make_plan()
    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    backend = select_backend(tensors, plan)
    assert isinstance(backend, StackedJaxBackend)


def test_select_auto_falls_through_to_legacy(monkeypatch):
    tensors, plan = _make_plan()
    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "auto")
    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    assert isinstance(select_backend(tensors, plan), StackedJaxBackend)


def test_select_all_unset_returns_none(monkeypatch):
    tensors, plan = _make_plan()
    _clear_env(monkeypatch)
    assert select_backend(tensors, plan) is None


def test_explicit_backend_wins_over_legacy(monkeypatch):
    """TENAX_BLOCKSPARSE_BACKEND=perblock beats a truthy legacy flag."""
    tensors, plan = _make_plan()
    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "perblock")
    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    assert select_backend(tensors, plan) is None


# ----------------------------------------------------------------------------
# StackedJaxBackend protocol surface
# ----------------------------------------------------------------------------


def test_stacked_backend_available_and_supports():
    tensors, plan = _make_plan()
    backend = StackedJaxBackend()
    assert backend.available() is True
    assert backend.supports(tensors, plan) is True


# ----------------------------------------------------------------------------
# Equivalence: backend=stacked value + grad match per-block
# ----------------------------------------------------------------------------


def test_backend_stacked_value_matches_perblock(monkeypatch):
    A = dict(canonical_tensors())["ferm_D2"]
    A, Abar = _double_layer(A)

    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "perblock")
    ref = contract(A, Abar)
    n0 = _STACK_FIRED["n"]

    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "stacked")
    got = contract(A, Abar)

    assert _STACK_FIRED["n"] > n0, "stacked backend did not fire"
    assert got._block_keys == ref._block_keys
    assert_tiered(ref._data, got._data, tier="fp")


def test_backend_stacked_grad_matches_perblock(monkeypatch):
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

    _clear_env(monkeypatch)
    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "perblock")
    g_ref = jax.grad(loss)(data0)

    monkeypatch.setenv("TENAX_BLOCKSPARSE_BACKEND", "stacked")
    g_got = jax.grad(loss)(data0)

    assert_tiered(g_ref, g_got, tier="fp")


# ----------------------------------------------------------------------------
# Back-compat: the legacy flag still fires the stacked path
# ----------------------------------------------------------------------------


def test_legacy_flag_still_fires_and_matches(monkeypatch):
    A = dict(canonical_tensors())["ferm_D2"]
    A, Abar = _double_layer(A)

    _clear_env(monkeypatch)
    ref = contract(A, Abar)  # all unset => per-block default
    n0 = _STACK_FIRED["n"]

    monkeypatch.setenv("TENAX_STACK_BLOCKSPARSE", "1")
    got = contract(A, Abar)

    assert _STACK_FIRED["n"] > n0, "legacy flag did not fire the stacked path"
    assert got._block_keys == ref._block_keys
    assert_tiered(ref._data, got._data, tier="fp")


def test_default_unset_is_perblock(monkeypatch):
    """No backend env set => per-block default (stacked path does NOT fire)."""
    A = dict(canonical_tensors())["ferm_D2"]
    A, Abar = _double_layer(A)

    _clear_env(monkeypatch)
    n0 = _STACK_FIRED["n"]
    contract(A, Abar)
    assert _STACK_FIRED["n"] == n0, "stacked path fired with no backend env set"
