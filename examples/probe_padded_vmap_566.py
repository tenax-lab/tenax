#!/usr/bin/env python3
"""#566 probe: does a padded-``vmap`` block-sparse representation give O(1)
compile WITHOUT converging to dense cost?

Settles the last un-measured JAX-native lever (the PaddedBlockArray + ``lax.scan``
/ ``vmap``-over-blocks port). Two orthogonal axes:

(ii) PADDING WASTE (the runtime verdict, analytic, CPU-ok).
   To make the per-block loop ONE ``vmap`` op (O(1) compile in block count) every
   block must be padded to a single common shape.  We measure, for the real
   fermionic CTM tensors (site A and the double-layer ``ac`` that dominates
   absorption) at D in {2,3,4}:
     - sparse_vol  = sum of block volumes   (true block-sparse cost)
     - dense_vol   = product of leg dims     (the dense baseline)
     - n_shapes    = #distinct block shapes  (= op count of the BATCHED path #568,
                     zero padding waste)
     - padded_vol  = n_blocks * max_block_vol (single-stack vmap: K=1 op, O(1)
                     compile, but pads ALL blocks to the max shape)
   Verdict per D: padded_vol/dense_vol.  < 1 => single-stack vmap beats dense
   (a real win); >= 1 => padding has reconstructed (or exceeded) dense, so the
   O(1)-compile form forfeits the sparsity advantage.

(i) COMPILE FLATNESS (the compile axis, A100).
   On the real block shapes/counts, build a representative per-block contraction
   two ways and measure cold ``jit`` compile + #XLA-compiles at D=2 vs D=3:
     - unrolled  : Python loop over n_blocks  (K = n_blocks ops)  -> should scale
     - vmap-stack: one vmap over the padded single stack (K = 1 op) -> should be flat
   Confirms the op-count -> compile-cost link and that vmap collapses it to O(1).
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor  # noqa: E402
from tenax.algorithms.ad_utils import (  # noqa: E402
    _ctm_tensor_multisite_fixed_point,
)
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402

# Reuse the profiler harness (site/gate construction, compile capture, cold timer).
_spec = importlib.util.spec_from_file_location(
    "prof566", pathlib.Path("examples/profile_ctm_ad_wall_566.py")
)
prof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prof)


# --------------------------------------------------------------------------- #
# (ii) padding-waste statistics
# --------------------------------------------------------------------------- #
def _vol(shape) -> int:
    return int(math.prod(shape)) if shape else 1


def block_stats(T, name: str, D: int) -> dict:
    """Volumes for tensor T under the three representations."""
    shapes = list(T._block_shapes)
    n_blocks = len(shapes)
    block_vols = [_vol(s) for s in shapes]
    sparse_vol = sum(block_vols)
    leg_dims = tuple(idx.dim for idx in T._indices)
    dense_vol = _vol(leg_dims)
    max_vol = max(block_vols) if block_vols else 0
    distinct_shapes = {tuple(s) for s in shapes}
    padded_vol = n_blocks * max_vol
    return {
        "name": name,
        "D": D,
        "n_blocks": n_blocks,
        "n_shapes": len(distinct_shapes),
        "leg_dims": leg_dims,
        "sparse_vol": sparse_vol,
        "dense_vol": dense_vol,
        "max_block_vol": max_vol,
        "padded_vol": padded_vol,
        # how much sparsity buys vs dense (lower = sparser):
        "sparse/dense": sparse_vol / dense_vol if dense_vol else float("nan"),
        # padding overhead of the single-stack (K=1) form vs true sparse:
        "padded/sparse": padded_vol / sparse_vol if sparse_vol else float("nan"),
        # THE VERDICT: does the O(1)-compile single-stack beat dense?
        "padded/dense": padded_vol / dense_vol if dense_vol else float("nan"),
    }


def run_padding_waste(Ds=(2, 3, 4)) -> list[dict]:
    rows: list[dict] = []
    for D in Ds:
        A, _gate = prof.make_site_and_gate("fermionic", D, seed=42)
        ac = _build_double_layer_tensor(A)
        rows.append(block_stats(A, "site_A", D))
        rows.append(block_stats(ac, "dbl_ac", D))
    return rows


def print_padding_table(rows: list[dict]) -> None:
    print("\n## (ii) padding-waste  [verdict: padded/dense < 1 => vmap beats dense]")
    hdr = (
        f"{'tensor':>7} {'D':>2} {'blk':>4} {'shp':>4} "
        f"{'sparse':>9} {'dense':>9} {'padded':>10} "
        f"{'spr/dns':>8} {'pad/spr':>8} {'pad/dns':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "  WIN" if r["padded/dense"] < 1.0 else "  ~dense+"
        print(
            f"{r['name']:>7} {r['D']:>2} {r['n_blocks']:>4} {r['n_shapes']:>4} "
            f"{r['sparse_vol']:>9} {r['dense_vol']:>9} {r['padded_vol']:>10} "
            f"{r['sparse/dense']:>8.3f} {r['padded/sparse']:>8.2f} "
            f"{r['padded/dense']:>8.3f}{flag}"
        )


# --------------------------------------------------------------------------- #
# (ii-env) the DECISIVE extension: do the CONVERGED env tensors (corners chi^2,
# edges chi*D^2*chi) stay shape-uniform at even D, or does SVD truncation
# fragment them regardless of parity?  The #566 wall is dominated by env tensors
# at chi, not the site/double-layer, so this is the real verdict.
# --------------------------------------------------------------------------- #
def run_env_heterogeneity(cases=((2, 8), (4, 16)), max_iter=10) -> list[dict]:
    rows: list[dict] = []
    for D, chi in cases:
        A, _gate = prof.make_site_and_gate("fermionic", D, seed=42)
        cfg = CTMConfig(chi=chi, max_iter=max_iter, conv_tol=1e-4)
        envs = _ctm_tensor_multisite_fixed_point(
            {(0, 0): A}, SINGLE_SITE_NEIGHBORS, cfg
        )
        env = envs[(0, 0)]
        for label, T in (
            ("C1", env.C1),
            ("C2", env.C2),
            ("T1", env.T1),
            ("T2", env.T2),
        ):
            r = block_stats(T, f"{label}", D)
            r["chi"] = chi
            rows.append(r)
    return rows


def print_env_table(rows: list[dict]) -> None:
    print("\n## (ii-env) CONVERGED env-tensor heterogeneity  [the dominant cost]")
    hdr = (
        f"{'env':>4} {'D':>2} {'chi':>3} {'blk':>4} {'shp':>4} "
        f"{'sparse':>9} {'dense':>9} {'padded':>10} "
        f"{'spr/dns':>8} {'pad/spr':>8} {'pad/dns':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "  WIN" if r["padded/dense"] < 1.0 else "  ~dense+"
        print(
            f"{r['name']:>4} {r['D']:>2} {r['chi']:>3} {r['n_blocks']:>4} "
            f"{r['n_shapes']:>4} "
            f"{r['sparse_vol']:>9} {r['dense_vol']:>9} {r['padded_vol']:>10} "
            f"{r['sparse/dense']:>8.3f} {r['padded/sparse']:>8.2f} "
            f"{r['padded/dense']:>8.3f}{flag}"
        )


# --------------------------------------------------------------------------- #
# (i) compile flatness: unrolled (K=n_blocks ops) vs vmap-stack (K=1 op)
# --------------------------------------------------------------------------- #
def _as_matrix(shape):
    """Split a block shape into (rows, cols) ~ a representative contraction axis."""
    n = len(shape)
    if n == 0:
        return (1, 1)
    half = max(1, n // 2)
    rows = _vol(shape[:half])
    cols = _vol(shape[half:])
    return (rows, cols)


def make_compile_arms(T):
    """Return (unrolled_fn, vmap_fn, blocks_list, padded_stack) for tensor T.

    Per-block op = a square matmul ``M @ M.T`` on the block reshaped to 2D.  The
    unrolled arm emits one matmul per block (K = n_blocks ops); the vmap arm pads
    every block to the max (rows, cols) and emits ONE batched matmul (K = 1 op).
    """
    mats = [
        T._get_block(i).reshape(_as_matrix(T._block_shapes[i]))
        for i in range(T.n_blocks)
    ]
    max_r = max(m.shape[0] for m in mats)
    max_c = max(m.shape[1] for m in mats)

    def _pad(m):
        return jnp.pad(m, ((0, max_r - m.shape[0]), (0, max_c - m.shape[1])))

    padded_stack = jnp.stack([_pad(m) for m in mats])  # (n_blocks, max_r, max_c)

    def unrolled(blocks):
        # K = n_blocks separate dot_general ops in the jaxpr
        return sum(jnp.sum(b @ b.T) for b in blocks)

    def vmapped(stack):
        # K = 1 batched dot_general op, independent of n_blocks
        return jnp.sum(jax.vmap(lambda m: m @ m.T)(stack))

    return unrolled, vmapped, mats, padded_stack


def run_compile_flatness(Ds=(2, 3)) -> list[dict]:
    cap = prof._install_compile_capture()
    rows: list[dict] = []
    for D in Ds:
        A, _gate = prof.make_site_and_gate("fermionic", D, seed=42)
        ac = _build_double_layer_tensor(A)
        unrolled, vmapped, mats, stack = make_compile_arms(ac)

        ju = jax.jit(unrolled)
        wall_u, ev_u, _ = prof._cold(ju, mats, cap)
        cu = sum(t for _, t in ev_u)

        jv = jax.jit(vmapped)
        wall_v, ev_v, _ = prof._cold(jv, stack, cap)
        cv = sum(t for _, t in ev_v)

        rows.append(
            {
                "D": D,
                "n_blocks": ac.n_blocks,
                "unrolled_compile_s": cu,
                "unrolled_n": len(ev_u),
                "vmap_compile_s": cv,
                "vmap_n": len(ev_v),
            }
        )
    return rows


def print_compile_table(rows: list[dict]) -> None:
    print("\n## (i) compile flatness  [dbl_ac per-block matmul; cold jit]")
    hdr = (
        f"{'D':>2} {'blk':>4} "
        f"{'unroll_s':>9} {'unroll_n':>9} {'vmap_s':>8} {'vmap_n':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['D']:>2} {r['n_blocks']:>4} "
            f"{r['unrolled_compile_s']:>9.3f} {r['unrolled_n']:>9} "
            f"{r['vmap_compile_s']:>8.3f} {r['vmap_n']:>7}"
        )
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        ur = b["unrolled_compile_s"] / max(a["unrolled_compile_s"], 1e-9)
        vr = b["vmap_compile_s"] / max(a["vmap_compile_s"], 1e-9)
        print(
            f"\n  D{a['D']}->D{b['D']} compile ratio:  "
            f"unrolled {ur:.2f}x   vmap {vr:.2f}x  "
            f"(flat ~1x => O(1) in block count)"
        )


def main():
    print(f"# #566 padded-vmap probe  [{jax.devices()[0].device_kind}]")
    waste = run_padding_waste()
    print_padding_table(waste)
    env = run_env_heterogeneity()
    print_env_table(env)
    comp = run_compile_flatness()
    print_compile_table(comp)


if __name__ == "__main__":
    main()
