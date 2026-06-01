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

  * **first** step -- first `value_and_grad` call: traces and XLA-compiles the
                      internally-jitted CTM step (the one-time #566 compile
                      cost, and where the batching takes effect at trace time).
  * **step** time  -- median of subsequent calls: compiled inner steps are
                      reused, but the eager CTM-convergence + Neumann-backward
                      orchestration re-runs in Python each call. This is one
                      real production-style AD step, NOT a single fused device
                      kernel; the off/on ratio is the decision metric.

It drives the *actual default* production multi-block symmetric-AD path: a bare
`jax.value_and_grad` (no outer jax.jit) with the default
`ad_backward_method="vjp"` Neumann backward, on a genuinely multi-block
**FermionParity** `SymmetricTensor` iPEPS site tensor
(`_build_initial_fpeps_tensor`, 16 charge blocks) -- mirroring
`ipeps_optimize._optimize_gs_ad_tensor`. This is the path validated by
`tests/test_fpeps_ad.py::test_symmetric_nontrivial_gradient_finite` and the #567
fermionic AD smoke test -- i.e. the multi-block regime #566/#565 are about.

We deliberately do not jit the outer value_and_grad or switch to the gmres
backward: production does neither, and the "vjp" backward's host-side Python
loop is not jittable anyway (only the xfail-#292 gmres backend is). The
block-sparse batching still applies inside the internally-jitted CTM step.

Note: the U(1) single-site CTM path (unbounded charge sectors, the strongest
block-count stress) currently fails in the production absorb step for non-trivial
charges -- a known coverage gap (see the #566 memory note). FermionParity is the
multi-block path that works end-to-end, so the benchmark uses it.

Usage
-----
GPU (the run that matters)::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/bench_symmetric_ad_batching_566.py \
        --D 2 3 4 6 --chi-factor 3 --reps 5 --json bench_566_gpu.json

f32 latency-bound probe (consumer GPU proxy for the datacenter regime)::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/bench_symmetric_ad_batching_566.py \
        --no-x64 --D 2 3 4 6 --chi-factor 3 --reps 5 --json bench_566_gpu_f32.json

CPU sanity (slow -- D=3 first step is minutes)::

    JAX_PLATFORMS=cpu uv run python examples/bench_symmetric_ad_batching_566.py \
        --D 2 3 --chi-factor 2 --max-iter 8 --reps 2

The script prints a per-D table and a final summary with the speedup (off/on) at
each D and the smallest D at which batching becomes a net win. The --json file is
rewritten after every D (so a run killed on the large-D compile wall keeps its
completed rows). Save stdout (and the --json) and attach it to issue #569.
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
    """value_and_grad of the CTM-converge + energy loss (production path).

    This deliberately mirrors what the production fermionic optimizer does in
    ``ipeps_optimize._optimize_gs_ad_tensor``: it calls
    ``jax.value_and_grad(loss_fn)(params)`` **directly, without an outer
    jax.jit**, and uses the default ``ad_backward_method="vjp"`` (iterative
    Neumann VJP) backward. We do NOT wrap this in jax.jit, because:

      * production doesn't, so jitting would benchmark a different path; and
      * the "vjp" backward is a host-driven Python loop (``float()`` /
        data-dependent breaks) that cannot be traced under jax.jit at all --
        only the experimental ``ad_backward_method="gmres"`` backend is
        jittable, and that one is xfail (#292), i.e. not what users get.

    The heavy block-sparse work (the per-sweep CTM contraction + projector
    SVD) runs inside ``ctm_tensor_converge``'s *internally* jit-compiled CTM
    step, so the first call still pays a real XLA compile (the #566 cost, and
    where the TENAX_BATCH_BLOCKSPARSE batching takes effect at trace time) and
    later calls reuse those compiled inner steps. The outer convergence /
    Neumann-backward orchestration stays in Python -- exactly as in production
    -- so a timed call here is one real production-style AD step, not a single
    fused device kernel. See the timing labels in ``time_mode``.
    """

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
            # Leave ad_backward_method at its production default ("vjp"); see
            # build_loss for why we do not jit / switch to gmres.
            #
            # The Arnoldi spectral-radius precheck is the one production
            # default we override: on a *random* (un-optimized) iPEPS the
            # backward Jacobian can have rho(J^T) >= the 5.0 threshold, which
            # raises CTMRGGradientError and aborts the timing. Disabling it
            # only skips ~20 Arnoldi matvecs and does not affect the
            # block-sparse batching being measured.
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
    """Time one gate mode: first AD step (incl. inner-step compile) + warm steps.

    ``first_s`` is the first value_and_grad call: it traces and XLA-compiles
    the internally-jitted CTM step (the one-time #566 compile cost). ``step_s``
    is the median of subsequent calls: the compiled inner steps are reused, but
    the eager CTM-convergence + Neumann-backward orchestration re-runs in Python
    each call -- i.e. one real production-style AD step (NOT a single fused
    device kernel). The off/on ratio is the decision metric; the Python
    orchestration is ~constant across modes so the ratio still reflects the
    block-sparse batching delta.

    Returns (first_s, step_median_s, step_min_s, energy, grad_finite).
    """
    gate, config_tuple, env_treedef, prev_env_leaves = plumbing
    os.environ[FLAG] = "1" if on else "0"  # flip the gate (read fresh per call)
    jax.clear_caches()  # force a fresh trace so the gated branch is re-read
    vg = build_loss(config_tuple, env_treedef, prev_env_leaves, gate, 2)

    t0 = time.perf_counter()
    E, g = vg(A)
    jax.block_until_ready((E, g))
    first_s = time.perf_counter() - t0

    steps = []
    for _ in range(reps):
        t0 = time.perf_counter()
        E, g = vg(A)
        jax.block_until_ready((E, g))
        steps.append(time.perf_counter() - t0)

    grad_finite = bool(jnp.all(jnp.isfinite(g._data)))
    return (
        first_s,
        statistics.median(steps),
        min(steps),
        float(E),
        grad_finite,
    )


def _write_json(path, base_meta, results):
    """Dump base_meta + results-so-far + derived crossovers to ``path``.

    Called after every D (not just at the end) so a long run that is killed
    mid-sweep -- e.g. on the XLA:GPU compile wall at large D -- still leaves the
    completed rows on disk instead of losing everything.
    """
    cross_step = next((r["D"] for r in results if r["step_speedup"] > 1.0), None)
    cross_first = next((r["D"] for r in results if r["first_speedup"] > 1.0), None)
    payload = {
        **base_meta,
        "crossover_step_D": cross_step,
        "crossover_first_D": cross_first,
        "all_energies_match": all(r["energy_match"] for r in results),
        "all_grads_finite": all(
            r["grad_finite_off"] and r["grad_finite_on"] for r in results
        ),
        "results": results,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


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
        "--x64",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float64 (default; the realistic iPEPS regime). Pass --no-x64 to "
        "run float32 -- on a consumer GPU that flips the workload from "
        "f64-ALU-throughput-bound into the kernel-launch-latency-bound regime "
        "the batching gate targets (better datacenter-GPU/TPU proxy). May be "
        "less numerically stable (SVD/GMRES).",
    )
    ap.add_argument(
        "--json", type=str, default=None, help="Optional path to dump results JSON."
    )
    args = ap.parse_args()

    # Apply the dtype choice. tenax/__init__.py force-enables jax_enable_x64 on
    # import (which already happened above), so to honor --no-x64 we must
    # re-apply the flag here -- before any array is created in setup(). A
    # pre-import setting would be silently overridden by the tenax import.
    jax.config.update("jax_enable_x64", args.x64)

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

    # Off vs on are the same math but batching reorders the reduction
    # (stack + segment_sum vs per-block add), so they agree only up to
    # accumulation error -- ~5e-7 for f64 on GPU, ~1e-3 for f32. Use a
    # dtype-aware tolerance so the mismatch banner fires on real bugs, not on
    # benign reduction-order drift.
    e_tol = 1e-6 if args.x64 else 5e-3

    base_meta = {
        "platform": plat,
        "device_kind": devices[0].device_kind,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "symmetry": "FermionParity",
        "chi_desc": chi_desc,
        "max_iter": args.max_iter,
        "reps": args.reps,
        "energy_match_tol": e_tol,
    }

    hdr = (
        f"{'D':>3} {'chi':>4} {'blocks':>7} "
        f"{'first_off':>11} {'first_on':>11} {'first_x':>7}  "
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

        first_x = c_off / c_on if c_on else float("nan")
        step_x = s_off / s_on if s_on else float("nan")
        # Batching must not change the result beyond reduction-order drift.
        e_match = abs(e_off - e_on) < e_tol * (1 + abs(e_off))

        print(
            f"{D:>3} {chi:>4} {n_blocks:>7} "
            f"{c_off:>10.3f}s {c_on:>10.3f}s {first_x:>6.2f}x  "
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
                "first_off_s": c_off,
                "first_on_s": c_on,
                "first_speedup": first_x,
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

        # Persist after every D so a run killed on the large-D compile wall
        # still leaves the completed rows on disk.
        if args.json:
            _write_json(args.json, base_meta, results)

    # ----------------------------------------------------------------- summary
    print("-" * len(hdr))
    print("\nSummary")
    print("  speedup > 1.0 means batching (gate ON) is FASTER.")
    cross_step = next((r["D"] for r in results if r["step_speedup"] > 1.0), None)
    cross_first = next((r["D"] for r in results if r["first_speedup"] > 1.0), None)
    print(
        f"  crossover D (warm step, ON faster) : "
        f"{cross_step if cross_step is not None else 'none in range'}"
    )
    print(
        f"  crossover D (first step, ON faster): "
        f"{cross_first if cross_first is not None else 'none in range'}"
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
        print(f"\n  JSON written to {args.json}")


if __name__ == "__main__":
    main()
