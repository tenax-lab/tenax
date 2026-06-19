"""Spike: does GSPMD shard the dominant dense-CTM contraction across devices?

Run on N fake CPU devices:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        uv run python examples/spike_ctm_sharding.py

Asserts: (1) the sharded enlarged-corner-style contraction equals the
single-device result to 1e-10; (2) the output is sharded (shard_shape < full).
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from tenax.algorithms.ctm_sharding import (
    _AXIS,
    build_ctm_mesh,
    corner_partition_spec,
    edge_partition_spec,
)


def enlarged_corner(C, T_h, T_v, a):
    # toy stand-in for the enlarged-corner contraction: C(χ,χ), T_h(χ,D²,χ),
    # T_v(χ,D²,χ), a(D²,D²,D²,D²) → carries the D² legs GSPMD must partition.
    x = jnp.einsum("ij,jkl->ikl", C, T_h)  # (χ, D², χ)
    x = jnp.einsum("ikl,lmn->ikmn", x, T_v)  # (χ, D², D², χ)
    x = jnp.einsum("ikmn,kmpq->ipqn", x, a)  # (χ, D², D², χ)
    return x


def main():
    n = jax.device_count()
    chi, D2 = 8, 8
    key = jax.random.PRNGKey(0)
    k = jax.random.split(key, 4)
    C = jax.random.normal(k[0], (chi, chi))
    T_h = jax.random.normal(k[1], (chi, D2, chi))
    T_v = jax.random.normal(k[2], (chi, D2, chi))
    a = jax.random.normal(k[3], (D2, D2, D2, D2))

    single = jax.jit(enlarged_corner)(C, T_h, T_v, a)

    mesh = build_ctm_mesh()
    edge_sh = NamedSharding(mesh, edge_partition_spec())
    Cs = jax.device_put(C, NamedSharding(mesh, corner_partition_spec()))
    T_hs = jax.device_put(T_h, edge_sh)
    T_vs = jax.device_put(T_v, edge_sh)
    # NOTE: this toy einsum contracts away the *first* D² leg of ``a`` (index
    # ``k``). ``commit_double_layer`` shards exactly that leg, so GSPMD would
    # all-reduce it and the output would come back replicated — a true statement
    # about *this* contraction, not a defect in the D²-axis sharding choice.
    # The real enlarged corner always keeps a virtual bond in its output, so we
    # shard a *surviving* D² leg of ``a`` (the output index ``p``) to model that
    # and prove sharding propagates end-to-end. The double-layer commit helper is
    # exercised by Task 5's parity test where the real contraction is wired in.
    a_s = jax.device_put(a, NamedSharding(mesh, PartitionSpec(None, None, _AXIS, None)))
    sharded = jax.jit(enlarged_corner)(Cs, T_hs, T_vs, a_s)

    err = float(jnp.max(jnp.abs(single - sharded)))
    shard0 = sharded.sharding.shard_shape(sharded.shape)
    print(f"devices={n}  max|single-sharded|={err:.2e}  out_shard_shape={shard0}")
    assert err < 1e-10, err
    if n > 1:
        assert shard0 != sharded.shape, "output not sharded"
    print("SPIKE OK")


if __name__ == "__main__":
    main()
