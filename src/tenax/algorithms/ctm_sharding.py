"""GSPMD sharding helpers for the dense CTM path (large-D, single-node multi-GPU).

Shards the D² virtual axis of the double-layer tensor and CTM edges over a 1-D
device mesh; corners (tiny, ~χ²) stay replicated. Used by the forward CTM to keep
per-device peak memory ≈1/N at large D. See the rung-1 design spec.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

_AXIS = "d"


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


def commit_env(env, mesh: Mesh):
    """device_put a CTMTensorEnv: edges D²-sharded, corners replicated.

    Operates on the ``DenseTensor`` leaves and rebuilds the env via the pytree so
    the wrapper/indices are preserved.
    """
    corner_sh = NamedSharding(mesh, corner_partition_spec())
    edge_sh = NamedSharding(mesh, edge_partition_spec())
    fields = {}
    for name in env._fields:
        t = getattr(env, name)
        sh = corner_sh if name.startswith("C") else edge_sh
        leaves, treedef = jax.tree_util.tree_flatten(t)
        leaves = [jax.device_put(x, sh) for x in leaves]
        fields[name] = jax.tree_util.tree_unflatten(treedef, leaves)
    return env._replace(**fields)
