import jax
import numpy as np
import pytest

from tenax.algorithms.ctm_sharding import (
    build_ctm_mesh,
    corner_partition_spec,
    double_layer_partition_spec,
    edge_partition_spec,
)


def test_mesh_and_specs():
    devs = jax.devices()
    mesh = build_ctm_mesh(devs)
    assert mesh.axis_names == ("d",)
    assert mesh.devices.size == len(devs)
    # edge (chi, D2, chi): shard axis 1 (D2) over "d"
    assert tuple(edge_partition_spec()) == (None, "d", None)
    # corner (chi, chi): replicated
    assert tuple(corner_partition_spec()) == (None, None)
    # double-layer (D2, D2, D2, D2): shard axis 0 over "d"
    assert tuple(double_layer_partition_spec()) == ("d", None, None, None)


def test_build_ctm_mesh_default_devices():
    mesh = build_ctm_mesh()
    assert mesh.axis_names == ("d",)
    assert mesh.devices.size == len(jax.devices())
