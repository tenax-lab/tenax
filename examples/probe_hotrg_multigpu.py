r"""Feasibility probe: can GSPMD shard a HOTRG (or TRG) coarse-graining step to
cut per-device peak memory and extend the single-GPU chi ceiling? (#663 post-1.0)

Why this might work where CTM multi-GPU did NOT:
  * TRG/HOTRG free energy is a FORWARD-only computation — no AD-through-SVD
    backward, which is exactly the wall that replicated the dominant CTM
    intermediate (see 2026-07-02 reduced-corner-shard-gate NO-GO).
  * tenax HOTRG explicitly forms T_merged = (up,down,left,U,D,right) = chi^6
    (hotrg._hotrg_step_horizontal step 3). That chi^6 intermediate is the
    single-GPU memory wall (chi=40 -> 33 GB, chi=48 -> 96 GB OOMs one 80GB GPU).
    The HOSVD operand is only chi^2 x chi^2 = chi^4. So if sharding a surviving
    leg keeps the chi^6 contraction at 1/N while only the chi^4 SVD replicates,
    per-device peak drops ~N x and the chi ceiling extends.

METHOD (learned the hard way on CTM): measure REAL peak_bytes_in_use high-water,
full output tensor live (no DCE), ONE mode per process. Do NOT trust
memory_analysis + single-leaf (DCE-inflates the ratio).

Run (one mode/process; 2 GPUs — the DGX-Display box deadlocks 4-way NCCL):
    CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/probe_hotrg_multigpu.py 40 up repl hotrg
    ... shard  ...
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.hotrg import _hotrg_step_horizontal  # noqa: E402
from tenax.algorithms.trg import _trg_step  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402

_LABELS = ("up", "down", "left", "right")
_FLOWS = (FlowDirection.IN, FlowDirection.OUT, FlowDirection.IN, FlowDirection.OUT)


def _build_T(chi, seed=0):
    rng = np.random.RandomState(seed)
    data = jnp.asarray(rng.standard_normal((chi, chi, chi, chi)))
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    idx = tuple(
        TensorIndex.from_charges(sym, np.zeros(chi, np.int32), fl, label=lb)
        for lb, fl in zip(_LABELS, _FLOWS)
    )
    return DenseTensor(data, idx)


def main():
    chi = int(sys.argv[1])
    shard_leg = sys.argv[2] if len(sys.argv) > 2 else "up"
    mode = sys.argv[3] if len(sys.argv) > 3 else "repl"
    algo = sys.argv[4] if len(sys.argv) > 4 else "hotrg"

    mesh = Mesh(np.asarray(jax.devices()), axis_names=("d",))
    n = jax.device_count()
    T = _build_T(chi)

    step = _hotrg_step_horizontal if algo == "hotrg" else _trg_step

    def fn(T):
        T_new, log_norm = step(T, chi, None)  # (T, max_bond_dim, svd_trunc_err)
        # keep the FULL output tensor + scalar live (no DCE)
        return jnp.sum(jax.tree_util.tree_leaves(T_new)[0] ** 2) + log_norm

    axis = _LABELS.index(shard_leg)
    if mode == "repl":
        spec = PartitionSpec()
    else:
        p = [None, None, None, None]
        p[axis] = "d"
        spec = PartitionSpec(*p)
    sh = NamedSharding(mesh, spec)
    leaves, treedef = jax.tree_util.tree_flatten(T)
    T = jax.tree_util.tree_unflatten(treedef, [jax.device_put(x, sh) for x in leaves])

    out = jax.jit(fn)(T)
    out.block_until_ready()
    try:
        hw = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        hw = float("nan")
    chi6 = chi**6 * 8 / 1e9
    chi4 = chi**4 * 8 / 1e9
    print(f"ALGO={algo} chi={chi} shard_leg={shard_leg} MODE={mode} ndev={n} "
          f"peak_dev0={hw:.4f}GB  [chi^6={chi6:.2f}GB chi^4={chi4:.4f}GB] "
          f"result={float(out):.4e}")


if __name__ == "__main__":
    main()
