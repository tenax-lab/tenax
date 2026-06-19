import jax

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


def test_commit_env_shards_edges_replicates_corners():
    import jax.numpy as jnp

    from tenax.algorithms._ctm_tensor_init import (
        CTMTensorEnv,
        _make_dense_standard_edge,
        _make_rank1_dense_corner,
        _std_edge_specs_compat,
    )
    from tenax.algorithms._ctm_utils import _CORNER_SPECS
    from tenax.algorithms.ctm_sharding import (
        build_ctm_mesh,
        commit_env,
        corner_partition_spec,
        edge_partition_spec,
    )

    chi = 4
    D2 = 4
    dtype = jnp.float64
    corners = {
        name: _make_rank1_dense_corner(chi, la, lb, fa, fb, dtype)
        for name, (la, lb, fa, fb, _ref) in _CORNER_SPECS.items()
    }
    edges = {
        name: _make_dense_standard_edge(chi, D2, l1, l2, l3, f1, f2, f3, dtype)
        for name, (l1, l2, l3, f1, f2, f3, _rc1, _rda, _rdb) in (
            _std_edge_specs_compat().items()
        )
    }
    env = CTMTensorEnv(
        C1=corners["C1"],
        C2=corners["C2"],
        C3=corners["C3"],
        C4=corners["C4"],
        T1=edges["T1"],
        T2=edges["T2"],
        T3=edges["T3"],
        T4=edges["T4"],
    )

    mesh = build_ctm_mesh()
    committed = commit_env(env, mesh)

    def _leaf_spec(field):
        # The underlying jnp array is the (single) pytree leaf commit_env
        # device_puts; works for any registered Tensor regardless of symmetry.
        (leaf,), _ = jax.tree_util.tree_flatten(field)
        return leaf.sharding.spec

    # Corner fields: replicated (all-None partition spec).
    for name in ("C1", "C2", "C3", "C4"):
        spec = _leaf_spec(getattr(committed, name))
        assert spec == corner_partition_spec(), name
        assert all(p is None for p in spec), name
    # Edge fields: D² axis (axis 1) sharded over "d".
    for name in ("T1", "T2", "T3", "T4"):
        spec = _leaf_spec(getattr(committed, name))
        assert spec == edge_partition_spec(), name
        assert tuple(spec) == (None, "d", None), name


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
