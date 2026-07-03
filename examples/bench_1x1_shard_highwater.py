r"""Real per-device high-water peak of the reduced-corner 1×1 CTM path, at three
granularities, to localize where GSPMD sharding is (or isn't) retained. (#570/#632)

This is the FAITHFUL measurement that corrects the earlier
``gate_reduced_corner_shard_570.py`` (which used ``memory_analysis`` + a single-leaf
return and was fooled by dead-code elimination into reporting a spurious ~4× relief).
Use ``memory_stats()['peak_bytes_in_use']`` — one mode per process — as
``bench_ctm_sharding_memory.py`` does.

Finding (2×A100, D=10 χ=32; repl = sharding OFF, device_mesh=None):
  absorb (isolated χ²·D⁶ contraction) : 9.80 -> 5.71 GB  = 1.72x  (shards)
  move   (full _ctm_tensor_move_left) : 17.19 -> 9.40 GB = 1.83x  (shards)
  sweep  (full 4-direction 1×1 sweep) : 17.19 -> 17.60 GB = 0.98x (NO relief)
The single move shards, but the full sweep does NOT: each move's output env comes back
replicated (projector/isometry), so moves 2-4 run replicated; the internal _shard_a only
shards the small double-layer, not the env. Env sharding does not persist across moves.
=> reduced-corner 1×1 SWEEP is NOT a multi-GPU lever (~1× outcome, cf. #632). NO-GO.

Run (one mode/process; NCCL 4-way deadlocks on the DGX-Display box, use 2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_1x1_shard_highwater.py 10 32 shard sweep
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import (  # noqa: E402
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import _ctm_tensor_move_left  # noqa: E402
from tenax.algorithms.ctm_sharding import (  # noqa: E402
    build_ctm_mesh,
    commit_env,
    constrain_double_layer_for_move,
    double_layer_move_partition_spec,
    double_layer_partition_spec,
)
from tenax.contraction.contractor import contract  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402


def _build_A(D, d=2, seed=42):
    rng = np.random.RandomState(seed)
    data = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


def _randomize(tree, seed):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    out = []
    for i, x in enumerate(leaves):
        r = np.random.RandomState(seed + i).standard_normal(x.shape)
        r = jnp.asarray(r) / (float(np.linalg.norm(r)) + 1e-10)
        out.append(r.astype(x.dtype))
    return jax.tree_util.tree_unflatten(treedef, out)


def _sc(*ts):  # keep every leaf live -> no DCE
    return sum(jnp.sum(x * x) for t in ts for x in jax.tree_util.tree_leaves(t))


def main():
    D, chi, mode, layer = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    mesh = build_ctm_mesh()
    A = _build_A(D)
    env = _randomize(initialize_ctm_tensor_env(A, chi), 1)
    a = _randomize(_build_double_layer_tensor(A), 500)

    # True replicated baseline: sharding OFF (device_mesh=None, no manual
    # constraint). Only mode == "shard" turns GSPMD on, so the ratio is
    # sharding-off vs sharding-on rather than two GSPMD layouts.
    dm = mesh if mode == "shard" else None

    def _sha(a, direction="left"):
        return constrain_double_layer_for_move(a, direction, dm) if dm is not None else a

    if layer == "absorb":
        def fn(env, a):
            return _sc(contract(env.T4, _sha(a)))
    elif layer == "move":
        def fn(env, a):
            e, _ = _ctm_tensor_move_left(env, env, _sha(a), chi, "svd")
            return _sc(e)
    else:  # sweep
        def fn(envs, dls):
            e, _, _ = _ctm_tensor_sweep_multisite(
                envs, dls, SINGLE_SITE_NEIGHBORS, chi, False, "svd",
                recipe="1x1", device_mesh=dm,
            )
            return _sc(e[(0, 0)])

    if layer == "sweep":
        if mode == "repl":
            sh = NamedSharding(mesh, PartitionSpec())
            A1 = jax.device_put({(0, 0): env}, sh)
            A2 = jax.device_put({(0, 0): a}, sh)
        else:
            A1 = {(0, 0): commit_env(env, mesh)}
            A2 = {(0, 0): jax.device_put(a, NamedSharding(mesh, double_layer_partition_spec()))}
    else:
        if mode == "repl":
            sh = NamedSharding(mesh, PartitionSpec())
            A1 = jax.device_put(env, sh)
            A2 = jax.device_put(a, sh)
        else:
            A1 = commit_env(env, mesh)
            A2 = jax.device_put(a, NamedSharding(mesh, double_layer_move_partition_spec("left")))

    out = jax.jit(fn)(A1, A2)
    jax.block_until_ready(out)
    hw = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    print(f"LAYER={layer} MODE={mode} D={D} chi={chi} ndev={jax.device_count()} "
          f"peak_dev0={hw:.4f}GB")


if __name__ == "__main__":
    main()
