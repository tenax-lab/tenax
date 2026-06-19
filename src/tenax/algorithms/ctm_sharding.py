"""GSPMD sharding helpers for the dense CTM path (large-D, single-node multi-GPU).

Shards the D² virtual axis of the double-layer tensor and CTM edges over a 1-D
device mesh; corners (tiny, ~χ²) stay replicated. Used by the forward CTM to keep
per-device peak memory ≈1/N at large D. See the rung-1 design spec.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv

_AXIS = "d"
_CORNER_FIELDS = frozenset(("C1", "C2", "C3", "C4"))


def build_ctm_mesh(devices: Sequence[jax.Device] | None = None) -> Mesh:
    """1-D mesh named ``"d"`` over the given devices (default: all local devices)."""
    devs = list(devices) if devices is not None else jax.devices()
    return Mesh(np.asarray(devs), axis_names=(_AXIS,))


def edge_partition_spec() -> PartitionSpec:
    """Edge ``(χ, D², χ)`` → shard the D² axis."""
    return PartitionSpec(None, _AXIS, None)


def corner_partition_spec() -> PartitionSpec:
    """Corner ``(χ, χ)`` → replicated."""
    return PartitionSpec(None, None)


def double_layer_partition_spec() -> PartitionSpec:
    """Double-layer ``(D², D², D², D²)`` → shard the first D² axis."""
    return PartitionSpec(_AXIS, None, None, None)


def commit_double_layer(a: jax.Array, mesh: Mesh) -> jax.Array:
    """device_put the double-layer tensor onto its D²-sharded layout."""
    return jax.device_put(a, NamedSharding(mesh, double_layer_partition_spec()))


# Per-move surviving D²-leg of the double-layer tensor ``a`` (axes
# u2,d2,l2,r2 = 0,1,2,3).  Sharding ``a`` on this axis keeps the dominant
# χ²·D⁶ absorption intermediate at ≈1/N AND keeps the newly-absorbed edge
# sharded, so the memory win chains across moves.  See the rung-1 spec.
_MOVE_SURVIVING_AXIS = {"left": 3, "right": 2, "top": 1, "bottom": 0}


def double_layer_move_partition_spec(direction: str) -> PartitionSpec:
    """Shard the `a` D²-leg that survives the given move (keeps the χ²·D⁶
    intermediate ≈1/N and the new edge sharded). See the rung-1 spec."""
    spec = [None, None, None, None]
    spec[_MOVE_SURVIVING_AXIS[direction]] = _AXIS
    return PartitionSpec(*spec)


def constrain_double_layer_for_move(a, direction: str, mesh: Mesh):
    """with_sharding_constraint the double-layer Tensor `a` to its surviving-leg
    sharding for `direction`; operates on the single array leaf, returns a Tensor."""
    sh = NamedSharding(mesh, double_layer_move_partition_spec(direction))
    leaves, treedef = jax.tree_util.tree_flatten(a)
    leaves = [with_sharding_constraint(x, sh) for x in leaves]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def commit_env(env: CTMTensorEnv, mesh: Mesh) -> CTMTensorEnv:
    """device_put a CTMTensorEnv: edges D²-sharded, corners replicated.

    Operates on the Tensor pytree leaves (any registered Tensor — the helper is
    symmetry-agnostic) and rebuilds the env via the pytree so the
    wrapper/indices are preserved.
    """
    corner_sh = NamedSharding(mesh, corner_partition_spec())
    edge_sh = NamedSharding(mesh, edge_partition_spec())
    fields = {}
    for name in env._fields:
        t = getattr(env, name)
        sh = corner_sh if name in _CORNER_FIELDS else edge_sh
        leaves, treedef = jax.tree_util.tree_flatten(t)
        leaves = [jax.device_put(x, sh) for x in leaves]
        fields[name] = jax.tree_util.tree_unflatten(treedef, leaves)
    return env._replace(**fields)
