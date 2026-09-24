"""Cost of the reference graded contractor vs today's sign-free ``contract``
(#1035 design §4 D0: B's cost is unmeasured until this runs).

A CTM-shaped pairwise contraction on FermionParity tensors: an edge tensor
``T(l, m, r)`` (chi, D^2, chi) against a double-layer-shaped ``a(m, u, d, s)``.
Randomizes arm evaluation order across trials and repeats compile timing to
account for order and cache effects: reports eager time (median of per-trial medians,
with min–max) and jit compile time (median and min–max over trials), each sampled fresh.

    JAX_PLATFORMS=cpu uv run python examples/profile_graded_contract_1035.py --chi 16 --d2 4
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import numpy as np

from tenax.contraction.contractor import contract
from tenax.core._graded import graded_contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor


def _idx(n, flow, label):
    return TensorIndex.from_charges(
        FermionParity(), np.arange(n, dtype=np.int32) % 2, flow, label=label
    )


def _operands(chi: int, d2: int):
    IN, OUT = FlowDirection.IN, FlowDirection.OUT
    T = SymmetricTensor.random_normal(
        indices=(_idx(chi, IN, "l"), _idx(d2, OUT, "m"), _idx(chi, OUT, "r")),
        key=jax.random.PRNGKey(0),
    )
    a = SymmetricTensor.random_normal(
        indices=(
            _idx(d2, IN, "m"),
            _idx(d2, OUT, "u"),
            _idx(d2, IN, "d"),
            _idx(d2, OUT, "s"),
        ),
        key=jax.random.PRNGKey(1),
    )
    return T, a


def _arms(T, a):
    out = tuple(lab for lab in T.labels() + a.labels() if lab != "m")
    return {
        "sign_free": lambda x, y: contract(x, y, output_labels=out).todense(),
        "graded": lambda x, y: graded_contract(x, y).todense(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--d2", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Disable JAX compilation cache to ensure each arm gets a fresh compile
    try:
        jax.config.update("jax_enable_compilation_cache", False)
    except (AttributeError, ValueError) as e:
        print(f"Note: Could not disable jax_enable_compilation_cache: {e}")

    T, a = _operands(args.chi, args.d2)
    arms = _arms(T, a)

    # One warm-up per arm
    for name, fn in arms.items():
        fn(T, a).block_until_ready()

    # Collect per-trial, per-arm timings
    res = {name: {"eager": [], "compile": []} for name in arms}
    rng = np.random.default_rng(args.seed)
    order_counts = {"sign_free_first": 0, "graded_first": 0}

    for trial in range(args.trials):
        # Randomize arm order per trial
        arm_names = list(arms.keys())
        arm_order = rng.permutation(arm_names)

        # Track which ran first in this trial
        if arm_order[0] == "sign_free":
            order_counts["sign_free_first"] += 1
        else:
            order_counts["graded_first"] += 1

        for name in arm_order:
            fn = arms[name]

            # Measure eager execution
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn(T, a).block_until_ready()
                times.append(time.perf_counter() - t0)
            eager_median = statistics.median(times)
            res[name]["eager"].append(eager_median)

            # Measure JIT compile time (fresh cache, fresh jit)
            jax.clear_caches()
            jitted = jax.jit(fn)
            t0 = time.perf_counter()
            jitted(T, a).block_until_ready()
            compile_s = time.perf_counter() - t0
            res[name]["compile"].append(compile_s)

    # Aggregate results across trials
    for name in arms:
        eager_trials = res[name]["eager"]
        compile_trials = res[name]["compile"]

        eager_median = statistics.median(eager_trials)
        eager_min = min(eager_trials)
        eager_max = max(eager_trials)

        compile_median = statistics.median(compile_trials)
        compile_min = min(compile_trials)
        compile_max = max(compile_trials)

        print(
            f"{name:10s} eager median {eager_median * 1e3:8.2f} ms "
            f"[{eager_min * 1e3:8.2f}–{eager_max * 1e3:8.2f}]   "
            f"jit compile {compile_median:7.3f} s [{compile_min:7.3f}–{compile_max:7.3f}]",
            flush=True,
        )

    # Ratio line
    eager_ratios = [
        statistics.median(res["graded"]["eager"]) / statistics.median(res["sign_free"]["eager"])
    ]
    compile_ratios = [
        statistics.median(res["graded"]["compile"]) / statistics.median(res["sign_free"]["compile"])
    ]
    eager_ratio = eager_ratios[0]
    compile_ratio = compile_ratios[0]

    print(
        f"ratio graded/sign_free: eager {eager_ratio:.2f}x   compile {compile_ratio:.2f}x   "
        f"(chi={args.chi}, D^2={args.d2}, trials={args.trials}, seed={args.seed})"
    )

    # Print order statistics
    print(
        f"Trials: {order_counts['sign_free_first']} sign_free first, {order_counts['graded_first']} graded first"
    )


if __name__ == "__main__":
    main()
