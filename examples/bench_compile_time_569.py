#!/usr/bin/env python3
"""Compile-time sweep: dense vs symmetric (batched/unbatched) for #569.

The #569 timing benchmark measured *runtime* of the eager CTM-AD step and found
batching never wins -- but that path is host-dispatch-bound, so it tests the
wrong layer. The reason the code moved to the eager vjp backward in the first
place is that the *dense* JIT compile time blows up at large D/chi. The real
question is therefore about COMPILE time, not runtime:

  Does a JITTED symmetric (block-sparse) CTM-AD step compile faster than the
  dense one at large D/chi -- and does batching (TENAX_BATCH_BLOCKSPARSE) help?

This harness measures exactly that. It times ``jax.jit(value_and_grad(loss))``
.lower().compile() -- pure trace + XLA compile, no execution -- for a loss that
exercises the operations that dominate CTM-AD compile: a couple of
``truncated_svd`` decompositions (and their backward graphs, the expensive part)
plus the ``contract`` that re-forms the tensor between them. The same
polymorphic loss runs on a ``DenseTensor`` and on a ``FermionParity``
``SymmetricTensor`` of matched total bond dimension, so the only difference is
the op structure: one big op (dense) vs several small per-block ops (symmetric),
collapsed into a few batched ops when the gate is on.

Usage (fixed bond dim ``--D``; sweep graph depth ``--rounds``)::

    CUDA_VISIBLE_DEVICES=0 uv run python examples/bench_compile_time_569.py \
        --D 16 --rounds 1 4 8 16 32 --chi-factor 2 --json compile_569.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._tensor_utils import scale_bond_axis  # noqa: E402
from tenax.contraction.contractor import contract, truncated_svd  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import FermionParity  # noqa: E402
from tenax.core.tensor import DenseTensor, SymmetricTensor  # noqa: E402

FLAG = "TENAX_BATCH_BLOCKSPARSE"
IN, OUT = FlowDirection.IN, FlowDirection.OUT


def make_dense(D, seed):
    idx = (
        TensorIndex.from_charges(FermionParity(), np.zeros(D, np.int32), IN, label="a"),
        TensorIndex.from_charges(FermionParity(), np.zeros(D, np.int32), IN, label="b"),
        TensorIndex.from_charges(FermionParity(), np.zeros(D, np.int32), OUT, label="c"),
        TensorIndex.from_charges(FermionParity(), np.zeros(D, np.int32), OUT, label="d"),
    )
    data = jax.random.normal(jax.random.PRNGKey(seed), (D, D, D, D))
    return DenseTensor(data, idx)


def make_sym(D, seed):
    ch = np.array([i % 2 for i in range(D)], dtype=np.int32)  # D/2 even, D/2 odd
    idx = (
        TensorIndex.from_charges(FermionParity(), ch, IN, label="a"),
        TensorIndex.from_charges(FermionParity(), ch, IN, label="b"),
        TensorIndex.from_charges(FermionParity(), ch, OUT, label="c"),
        TensorIndex.from_charges(FermionParity(), ch, OUT, label="d"),
    )
    return SymmetricTensor.random_normal(idx, jax.random.PRNGKey(seed))


def make_loss(chi, rounds):
    """``rounds`` rounds of [truncated_svd (+ backward) -> contract re-form].

    Each round SVD-truncates the (a,b,c,d) tensor and contracts it back to
    (a,b,c,d), alternating the leg grouping like CTM left/right moves. Growing
    ``rounds`` grows the unrolled value_and_grad graph linearly -- the knob that
    drives the *dense* CTM-AD compile wall (which is graph-size, not single-op,
    bound). Gauge-invariant scalar accumulator for stable AD.
    """

    def loss(T):
        X = T
        acc = 0.0
        for k in range(rounds):
            if k % 2 == 0:
                left, right = ["a", "b"], ["c", "d"]
            else:
                left, right = ["a", "c"], ["b", "d"]
            U, s, Vh, _ = truncated_svd(
                X, left, right, new_bond_label="m", max_singular_values=chi
            )
            acc = acc + (s ** 2).sum()
            # contract U.s.Vh back to a 4-leg (a,b,c,d) tensor for the next round
            X = contract(scale_bond_axis(U, "m", s), Vh)
        return acc

    return loss


def compile_seconds(loss, T, on):
    os.environ[FLAG] = "1" if on else "0"
    jax.clear_caches()
    f = jax.jit(jax.value_and_grad(loss))
    t0 = time.perf_counter()
    f.lower(T).compile()  # trace + XLA compile only (no execution)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, default=16, help="fixed bond dim per leg")
    ap.add_argument(
        "--rounds", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64],
        help="graph depth: # of SVD+contract rounds (sweeps the unrolled AD graph size)",
    )
    ap.add_argument("--chi-factor", type=int, default=2, help="chi = chi_factor * D")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    D = args.D
    chi = args.chi_factor * D
    Td = make_dense(D, args.seed)
    Ts = make_sym(D, args.seed)

    dev = jax.devices()[0]
    print("=" * 78)
    print("# Compile-time vs graph depth: dense vs symmetric (off/on) -- #569")
    print(f"# platform : {dev.platform}  device0={dev.device_kind}")
    print(f"# host     : {platform.platform()}")
    print(f"# D={D} chi={chi} blocks={Ts.n_blocks}   rounds = {args.rounds}")
    print("=" * 78)
    hdr = (f"{'rounds':>6} {'dense_s':>10} {'sym_off_s':>11} {'sym_on_s':>10} "
           f"{'sym_off/dense':>14} {'on/dense':>9}")
    print(hdr)
    print("-" * len(hdr))

    base = {
        "platform": dev.platform, "device_kind": dev.device_kind,
        "D": D, "chi": chi, "n_blocks": Ts.n_blocks, "results": [],
    }
    for R in args.rounds:
        loss = make_loss(chi, R)
        c_dense = compile_seconds(loss, Td, on=False)
        c_off = compile_seconds(loss, Ts, on=False)
        c_on = compile_seconds(loss, Ts, on=True)
        roff = c_off / c_dense if c_dense else float("nan")
        ron = c_on / c_dense if c_dense else float("nan")
        print(
            f"{R:>6} {c_dense:>9.2f}s {c_off:>10.2f}s {c_on:>9.2f}s "
            f"{roff:>13.2f}x {ron:>8.2f}x"
        )
        base["results"].append(
            {
                "rounds": R, "dense_compile_s": c_dense,
                "sym_off_compile_s": c_off, "sym_on_compile_s": c_on,
            }
        )
        if args.json:
            json.dump(base, open(args.json, "w"), indent=2)

    print("-" * len(hdr))
    print("\n-> Grows fastest with graph depth = loses. If dense_s blows up "
          "super-linearly while batched-symmetric stays sub-linear, symmetry "
          "escapes the dense compile wall; if symmetric grows faster, it does "
          "not. Attach to #569.")


if __name__ == "__main__":
    main()
