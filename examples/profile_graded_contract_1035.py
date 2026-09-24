"""Cost of the reference graded contractor vs today's sign-free ``contract``
(#1035 design §4 D0: B's cost is unmeasured until this runs).

A CTM-shaped pairwise contraction on FermionParity tensors: an edge tensor
``T(l, m, r)`` (chi, D^2, chi) against a double-layer-shaped ``a(m, u, d, s)``.
Reports eager wall time (median of repeats, after a warm-up) and jit compile
time (``jax.clear_caches()`` before each arm, so each compile is fresh).

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
    args = ap.parse_args()
    T, a = _operands(args.chi, args.d2)
    res = {}
    for name, fn in _arms(T, a).items():
        fn(T, a).block_until_ready()  # warm-up
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn(T, a).block_until_ready()
            times.append(time.perf_counter() - t0)
        jax.clear_caches()
        jitted = jax.jit(fn)
        t0 = time.perf_counter()
        jitted(T, a).block_until_ready()
        compile_s = time.perf_counter() - t0
        res[name] = (statistics.median(times), compile_s)
        print(
            f"{name:10s} eager median {res[name][0] * 1e3:8.2f} ms   jit first call {compile_s:7.3f} s",
            flush=True,
        )
    (e0, c0), (e1, c1) = res["sign_free"], res["graded"]
    print(
        f"ratio graded/sign_free: eager {e1 / e0:.2f}x   compile {c1 / c0:.2f}x   (chi={args.chi}, D^2={args.d2})"
    )


if __name__ == "__main__":
    main()
