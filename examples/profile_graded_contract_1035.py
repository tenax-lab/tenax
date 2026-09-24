"""Cost of the reference graded contractor vs today's sign-free ``contract``
(#1035 design §4 D0: B's cost is unmeasured until this runs).

Two arm pairs, each run with the same randomized-trial method:

* ``sign_free`` vs ``graded`` -- a CTM-shaped pairwise contraction on
  FermionParity tensors: an edge tensor ``T(l, m, r)`` against a
  double-layer-shaped ``a(u, m, d, s)``.  ``T``'s contracted leg ``m`` is IN
  and ``T`` is the left operand of ``graded_contract``, so rule 2's twist
  (the shared legs that are IN on the left operand) runs; ``a`` is stored
  with ``m`` in the middle of its legs, so ``b``'s Koszul reorder onto
  ``reversed(shared) + free_b`` is a real permutation, not a no-op.  This
  orientation is chosen precisely so both of those graded-only costs are
  exercised, not skipped.
* ``bar`` vs ``graded_bar`` -- the plain ``SymmetricTensor.bar()`` against
  rule 3's sign-corrected ``graded_bar``, on ``T`` alone.

Randomizes arm evaluation order across trials and repeats first-call timing to
account for order and cache effects: reports eager time (median of per-trial medians,
with min–max) and jit first-call time (trace+compile+run, median and min–max over trials).

    JAX_PLATFORMS=cpu uv run python examples/profile_graded_contract_1035.py --chi 16 --d2 4
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import numpy as np

from tenax.contraction.contractor import contract
from tenax.core._graded import graded_bar, graded_contract
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
        indices=(_idx(chi, IN, "l"), _idx(d2, IN, "m"), _idx(chi, OUT, "r")),
        key=jax.random.PRNGKey(0),
    )
    a = SymmetricTensor.random_normal(
        indices=(
            _idx(d2, OUT, "u"),
            _idx(d2, OUT, "m"),
            _idx(d2, IN, "d"),
            _idx(d2, OUT, "s"),
        ),
        key=jax.random.PRNGKey(1),
    )
    return T, a


def _contract_out_labels(T, a):
    return tuple(lab for lab in T.labels() + a.labels() if lab != "m")


def _assert_same_labels(T, a):
    """One-time check that the two contraction arms produce the same legs
    in the same order, so the timing comparison is like for like."""
    out = _contract_out_labels(T, a)
    sign_free_labels = contract(T, a, output_labels=out).labels()
    graded_labels = graded_contract(T, a).labels()
    assert sign_free_labels == graded_labels, (sign_free_labels, graded_labels)


def _contract_arms(T, a):
    out = _contract_out_labels(T, a)
    arms = {
        "sign_free": lambda x, y: contract(x, y, output_labels=out).todense(),
        "graded": lambda x, y: graded_contract(x, y).todense(),
    }
    return arms, (T, a)


def _bar_arms(T):
    arms = {
        "bar": lambda x: x.bar().todense(),
        "graded_bar": lambda x: graded_bar(x).todense(),
    }
    return arms, (T,)


def _bench(
    arms: dict, operands: tuple, *, trials: int, repeats: int, rng
) -> tuple[dict, dict]:
    """Run every arm in ``arms`` (each called as ``fn(*operands)``) for
    ``trials`` trials, randomizing arm order per trial.  Returns the raw
    per-trial timings and a count of which arm ran first, keyed by name."""
    # One warm-up per arm
    for fn in arms.values():
        fn(*operands).block_until_ready()

    res = {name: {"eager": [], "first_call": []} for name in arms}
    order_counts = {name: 0 for name in arms}

    for _ in range(trials):
        arm_order = rng.permutation(list(arms.keys()))
        order_counts[arm_order[0]] += 1

        for name in arm_order:
            fn = arms[name]

            # Measure eager execution
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn(*operands).block_until_ready()
                times.append(time.perf_counter() - t0)
            res[name]["eager"].append(statistics.median(times))

            # Measure JIT first-call time (trace+compile+run, fresh cache, fresh jit)
            jax.clear_caches()
            jitted = jax.jit(fn)
            t0 = time.perf_counter()
            jitted(*operands).block_until_ready()
            res[name]["first_call"].append(time.perf_counter() - t0)

    return res, order_counts


def _report(name_a: str, name_b: str, res: dict, order_counts: dict, args) -> None:
    """Print per-arm timing lines, a ``ratio {a}/{b}:`` line, and the arm
    order counts -- for the pair ``(name_a, name_b)`` in ``res``."""
    for name in (name_b, name_a):
        eager_trials = res[name]["eager"]
        first_call_trials = res[name]["first_call"]

        eager_median = statistics.median(eager_trials)
        eager_min = min(eager_trials)
        eager_max = max(eager_trials)

        first_call_median = statistics.median(first_call_trials)
        first_call_min = min(first_call_trials)
        first_call_max = max(first_call_trials)

        print(
            f"{name:10s} eager median {eager_median * 1e3:8.2f} ms "
            f"[{eager_min * 1e3:8.2f}–{eager_max * 1e3:8.2f}]   "
            f"jit first call (trace+compile+run) {first_call_median:7.3f} s "
            f"[{first_call_min:7.3f}–{first_call_max:7.3f}]",
            flush=True,
        )

    eager_ratio = statistics.median(res[name_a]["eager"]) / statistics.median(
        res[name_b]["eager"]
    )
    first_call_ratio = statistics.median(res[name_a]["first_call"]) / statistics.median(
        res[name_b]["first_call"]
    )

    print(
        f"ratio {name_a}/{name_b}: eager {eager_ratio:.2f}x   first-call {first_call_ratio:.2f}x   "
        f"(chi={args.chi}, D^2={args.d2}, trials={args.trials}, seed={args.seed})"
    )

    parts = ", ".join(f"{order_counts[name]} {name} first" for name in order_counts)
    print(f"Trials: {parts}")


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
    _assert_same_labels(T, a)

    rng = np.random.default_rng(args.seed)

    contract_arms, contract_operands = _contract_arms(T, a)
    res, order_counts = _bench(
        contract_arms,
        contract_operands,
        trials=args.trials,
        repeats=args.repeats,
        rng=rng,
    )
    _report("graded", "sign_free", res, order_counts, args)

    bar_arms, bar_operands = _bar_arms(T)
    res, order_counts = _bench(
        bar_arms, bar_operands, trials=args.trials, repeats=args.repeats, rng=rng
    )
    _report("graded_bar", "bar", res, order_counts, args)


if __name__ == "__main__":
    main()
