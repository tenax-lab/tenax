"""GSPMD sharding helpers for the dense CTM path (large-D, single-node multi-GPU).

Shards the D² virtual axis of the double-layer tensor and CTM edges over a 1-D
device mesh; corners (tiny, ~χ²) stay replicated. Used by the forward CTM to keep
per-device peak memory ≈1/N at large D. See the rung-1 design spec.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

_AXIS = "d"


def build_ctm_mesh(devices=None) -> Mesh:
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
