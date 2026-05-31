#!/usr/bin/env python3
"""GPU/TPU benchmark for the TENAX_BATCH_BLOCKSPARSE batching gate (#566 / #569).

Issue #566: the symmetric/fermionic iPEPS AD backward (CTM sweep-VJP) is traced
and compiled once per optimizer run, but for block-sparse `SymmetricTensor` site
tensors that compile scales with the **number / size of charge blocks** --
block-sparse ops emit one set of XLA primitives *per block-combo*. PRs
#571/#572/#573 added a gated batched path (`TENAX_BATCH_BLOCKSPARSE`): same-shape
block-combos are grouped into a single batched `jnp.einsum` + `segment_sum`
(contraction) and a single `vmap`-ed `svd`/`qr`/`eigh` (decompositions). Default
off -> byte-identical.

The batching is **neutral-to-negative at small D on CPU** (stack/segment overhead
dominates the low per-call ceiling) and is expected to win on GPU/TPU at large D,
where many small per-block kernels are latency-bound and batching turns them into
a few large device-saturating kernels. This script measures exactly that crossover
so we can decide whether to flip the gate default-on (possibly device-aware).

What it measures, per bond dimension D and per gate mode (off / on):

  * **compile** time -- first `value_and_grad` call (trace + XLA compile + run).
  * **step** time    -- steady-state median over several subsequent calls
                        (compiled graph reused; pure device execution).

It drives the *real* production multi-block symmetric-AD path: `jax.value_and_grad`
of a CTM-converge + energy loss on a genuinely multi-block **FermionParity**
`SymmetricTensor` iPEPS site tensor (`_build_initial_fpeps_tensor`, 16 charge
blocks). This is the exact path validated by `tests/test_fpeps_ad.py`
(`test_symmetric_nontrivial_gradient_finite`) and the #567 fermionic AD smoke
test -- i.e. the multi-block regime #566/#565 are about.

Note: the U(1) single-site CTM path (unbounded charge sectors, the strongest
block-count stress) currently fails in the production absorb step for non-trivial
charges -- a known coverage gap (see the #566 memory note). FermionParity is the
multi-block path that works end-to-end, so the benchmark uses it.

Usage
-----
GPU (the run that matters)::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/bench_symmetric_ad_batching_566.py \
        --D 2 3 4 6 --chi-factor 3 --reps 5 --json bench_566_gpu.json

CPU sanity (slow -- D=3 compile is minutes)::

    JAX_PLATFORMS=cpu uv run python examples/bench_symmetric_ad_batching_566.py \
        --D 2 3 --chi-factor 2 --max-iter 8 --reps 2

The script prints a per-D table and a final summary with the speedup (off/on) at
each D and the smallest D at which batching becomes a net win. Save stdout (and
the --json) and attach it to issue #569.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import jax
import jax.numpy as jnp

# x64 is the realistic regime for iPEPS AD; enable before any array is created.
jax.config.update("jax_enable_x64", True)

# Import after x64 is set so internal constants pick up float64.
from tenax.algorithms._ctm_tensor import initialize_ctm_tensor_env  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_energy import (  # noqa: E402
    compute_energy_ctm_tensor,
)
from tenax.algorithms.ad_utils import (  # noqa: E402
    _config_to_tuple,
    ctm_tensor_converge,
)
from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _build_initial_fpeps_tensor,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig  # noqa: E402

# The batching path is gated by this env var, read fresh per contract/decomp
# call in contractor.py and linalg.py. Default off -> byte-identical.
FLAG = "TENAX_BATCH_BLOCKSPARSE"


def build_loss(config_tuple, env_treedef, prev_env_leaves, gate, d_phys):
    """value_and_grad-able CTM-converge + energy loss over one site tensor."""

    def loss_fn(A_param):
        A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
        env_leaves = ctm_tensor_converge(
            {(0, 0): A_norm},
            prev_env_leaves,
            SINGLE_SITE_NEIGHBORS,
            config_tuple,
        )
        env = jax.tree.unflatten(env_treedef, env_leaves)
        return compute_energy_ctm_tensor(A_norm, env, gate, d_phys)

    return jax.value_and_grad(loss_fn)


def setup(D: int, chi: int, max_iter: int, seed: int):
    """Build the site tensor, gate, and CTM config plumbing for one D."""
    fpeps_cfg = FPEPSConfig(D=D, t=1.0, V=0.0)
    ctm_cfg = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(
            chi=chi,
            max_iter=max_iter,
            conv_tol=1e-4,
            adjoint_arnoldi_precheck=False,
        ),
    )
    A = _build_initial_fpeps_tensor(fpeps_cfg, jax.random.PRNGKey(seed))
    gate = spinless_fermion_gate(fpeps_cfg).todense().reshape(2, 2, 2, 2)
    config_tuple = _config_to_tuple(ctm_cfg.ctm)
    env_template = initialize_ctm_tensor_env(A, chi)
    env_treedef = jax.tree.structure(env_template)
    prev_env_leaves = tuple(jax.tree.leaves(env_template))
    return A, gate, config_tuple, env_treedef, prev_env_leaves


def time_mode(A, plumbing, on: bool, reps: int):
    """Compile-once + steady-state timing for one gate mode.

    Returns (compile_s, step_median_s, step_min_s, energy, grad_finite).
    """
    gate, config_tuple, env_treedef, prev_env_leaves = plumbing
    os.environ[FLAG] = "1" if on else "0"  # flip the gate (read fresh per call)
    jax.clear_caches()  # force a fresh trace so the gated branch is re-read
    vg = build_loss(config_tuple, env_treedef, prev_env_leaves, gate, 2)

    t0 = time.perf_counter()
    E, g = vg(A)
    jax.block_until_ready((E, g))
    compile_s = time.perf_counter() - t0

    steps = []
    for _ in range(reps):
        t0 = time.perf_counter()
        E, g = vg(A)
        jax.block_until_ready((E, g))
        steps.append(time.perf_counter() - t0)

    grad_finite = bool(jnp.all(jnp.isfinite(g._data)))
    return (
        compile_s,
        statistics.median(steps),
        min(steps),
        float(E),
        grad_finite,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument(
        "--chi-factor",
        type=int,
        default=3,
        help="chi = chi_factor * D (env bond dim).",
    )
    ap.add_argument("--chi", type=int, default=None, help="Override: fixed chi.")
    ap.add_argument("--max-iter", type=int, default=12, help="CTM forward iters.")
    ap.add_argument("--reps", type=int, default=5, help="Steady-state timed reps.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--json", type=str, default=None, help="Optional path to dump results JSON."
    )
    args = ap.parse_args()

    devices = jax.devices()
    plat = devices[0].platform
    print("=" * 78)
    print(f"# Symmetric iPEPS CTM-AD batching benchmark ({FLAG}) -- #566 / #569")
    print(f"# platform   : {plat}  device0={devices[0].device_kind}")
    print(f"# host        : {platform.platform()}")
    print(f"# x64         : {jax.config.read('jax_enable_x64')}")
    print("# symmetry    : FermionParity (16-block iPEPS site tensor)")
    print(f"# D list      : {args.D}")
    chi_desc = f"{args.chi}" if args.chi else f"{args.chi_factor}*D"
    print(f"# chi         : {chi_desc}   max_iter={args.max_iter}")
    print(f"# reps        : {args.reps}")
    print("=" * 78)

    hdr = (
        f"{'D':>3} {'chi':>4} {'blocks':>7} "
        f"{'compile_off':>11} {'compile_on':>11} {'cmp_x':>6}  "
        f"{'step_off':>10} {'step_on':>9} {'step_x':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for D in args.D:
        chi = args.chi if args.chi else args.chi_factor * D
        A, gate, ct, td, pl = setup(D, chi, args.max_iter, args.seed)
        plumbing = (gate, ct, td, pl)
        n_blocks = A.n_blocks

        c_off, s_off, smin_off, e_off, fin_off = time_mode(
            A, plumbing, on=False, reps=args.reps
        )
        c_on, s_on, smin_on, e_on, fin_on = time_mode(
            A, plumbing, on=True, reps=args.reps
        )

        cmp_x = c_off / c_on if c_on else float("nan")
        step_x = s_off / s_on if s_on else float("nan")
        # Energies must match to many digits regardless of mode (correctness).
        e_match = abs(e_off - e_on) < 1e-9 * (1 + abs(e_off))

        print(
            f"{D:>3} {chi:>4} {n_blocks:>7} "
            f"{c_off:>10.3f}s {c_on:>10.3f}s {cmp_x:>5.2f}x  "
            f"{s_off * 1e3:>8.1f}ms {s_on * 1e3:>7.1f}ms {step_x:>5.2f}x"
        )
        if not e_match:
            print(
                f"    !! ENERGY MISMATCH off={e_off:.12g} on={e_on:.12g} "
                f"(diff={abs(e_off - e_on):.2e}) -- batching changed the result!"
            )
        if not (fin_off and fin_on):
            print(f"    !! NON-FINITE GRADIENT off_finite={fin_off} on_finite={fin_on}")

        results.append(
            {
                "D": D,
                "chi": chi,
                "n_blocks": n_blocks,
                "compile_off_s": c_off,
                "compile_on_s": c_on,
                "compile_speedup": cmp_x,
                "step_off_s": s_off,
                "step_on_s": s_on,
                "step_min_off_s": smin_off,
                "step_min_on_s": smin_on,
                "step_speedup": step_x,
                "energy_off": e_off,
                "energy_on": e_on,
                "energy_match": e_match,
                "grad_finite_off": fin_off,
                "grad_finite_on": fin_on,
            }
        )

    # ----------------------------------------------------------------- summary
    print("-" * len(hdr))
    print("\nSummary")
    print("  speedup > 1.0 means batching (gate ON) is FASTER.")
    cross_step = next((r["D"] for r in results if r["step_speedup"] > 1.0), None)
    cross_cmp = next((r["D"] for r in results if r["compile_speedup"] > 1.0), None)
    print(
        f"  crossover D (step time, ON faster)   : "
        f"{cross_step if cross_step is not None else 'none in range'}"
    )
    print(
        f"  crossover D (compile time, ON faster): "
        f"{cross_cmp if cross_cmp is not None else 'none in range'}"
    )
    all_match = all(r["energy_match"] for r in results)
    all_fin = all(r["grad_finite_off"] and r["grad_finite_on"] for r in results)
    print(f"  energies match (off == on) at every D : {all_match}")
    print(f"  gradients finite (both modes) at every D: {all_fin}")
    print(
        "\n  -> If ON wins clearly at large D on GPU/TPU, recommend flipping the\n"
        "     gate default-on for accelerators (keep CPU off/threshold). Attach\n"
        "     this output (+ --json) to issue #569."
    )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "platform": plat,
                    "device_kind": devices[0].device_kind,
                    "x64": bool(jax.config.read("jax_enable_x64")),
                    "symmetry": "FermionParity",
                    "chi_desc": chi_desc,
                    "max_iter": args.max_iter,
                    "reps": args.reps,
                    "crossover_step_D": cross_step,
                    "crossover_compile_D": cross_cmp,
                    "all_energies_match": all_match,
                    "all_grads_finite": all_fin,
                    "results": results,
                },
                fh,
                indent=2,
            )
        print(f"\n  JSON written to {args.json}")


if __name__ == "__main__":
    main()
