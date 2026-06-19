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


def test_commit_double_layer_is_sharded():
    # Requires >=2 devices; this test is run by the subprocess harness under
    # XLA_FLAGS=--xla_force_host_platform_device_count=4 (Task 4 driver). When
    # run on a single device it asserts the no-op (replicated) fallback.
    import jax.numpy as jnp

    from tenax.algorithms.ctm_sharding import build_ctm_mesh, commit_double_layer

    mesh = build_ctm_mesh()
    D2 = 4
    a = jnp.ones((D2, D2, D2, D2))
    a_sharded = commit_double_layer(a, mesh)
    n = mesh.devices.size
    # first axis is split across n devices (or 1 device → full shard = whole axis)
    shard_shape = a_sharded.sharding.shard_shape(a_sharded.shape)
    assert shard_shape[0] == D2 // n
    assert shard_shape[1:] == (D2, D2, D2)
