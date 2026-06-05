#!/usr/bin/env python3
"""Real-sweep op-count + persist-hit-rate measurement for the persisting
``StackedSymmetricTensor`` (#566 P1d slice 1). COMMITTABLE RECORD.

This is the go/no-go checkpoint for the persist-across-contractions mechanism.
It traces the FULL fused CTM-AD backward (env-sweep VJP + params-sweep VJP, the
compile-dominant ``_jit_fused_fixed_point_bwd`` unit) with
``TENAX_STACK_BLOCKSPARSE`` OFF vs ON-with-persist, and reports:

1. Backward TOTAL jaxpr op count OFF vs ON and the ON/OFF ratio (the realized
   op-count drop on the REAL sweep — the prior even-D within-call stacking gave
   ~1.00x here because every call re-gathered/re-scattered ``_data``).

2. PERSIST HIT-RATE: how many of the sweep's stacked ``_contract_symmetric``
   calls consumed an already-stacked input (PERSISTED, no ``_data`` gather)
   vs had to gather from ``_data`` (chain INTERRUPTED upstream by a
   fuse/bar/svd/_data-read). Reported as persisted/total operands and
   fully-persisted/total calls.

Trace-only (``jax.make_jaxpr``, no XLA compile). Reuses the probe's helpers.

Findings (fermionic D=2 and D=4, identical because the hit-rate is STRUCTURAL —
block-combo counts, not intra-block dense size):

* dot_general collapses 2344 -> 252 (~9.3x) — the batched einsum + segment_sum
  core works on the real sweep.
* backward TOTAL op-count barely moves: 0.995x (D=2) / 0.998x (D=4). The
  non-contraction ops (the ``.fuse()`` per-block reshapes, the SVD/eigh VJP, the
  scatter glue, and elementwise) dominate the graph and are unaffected.
* PERSIST hit-rate: 52/194 operands persisted (27%); only 8/97 calls fully
  persisted; 142 operands had to gather from ``_data``. The chains are short:
  the dominant interrupter is ``.fuse()`` (reads ``_data`` via ``_get_block``,
  ~41 materializations) — e.g. ``_build_double_layer_tensor`` does
  ``contract(A, A_bra)`` then immediately fuses the (u,U)/(d,D)/(l,L)/(r,R)
  pairs, materializing the stacked output one op later. A handful more come from
  ``.dtype`` reads (~8).

READ: contractions-only persistence does NOT clear the >=3-5x op-count target.
fuse/bar must be made stacked-aware (Task 5b) to lengthen the chains; the
contraction core itself is already ~9x leaner in dot_general.

PRE-EXISTING DEFECT (unrelated to this slice, flagged for follow-up): the landed
``TENAX_STACK_BLOCKSPARSE`` path makes the converged CTM energy drift ~4.6e-4 vs
the per-block path (reproduces on HEAD WITHOUT the persisting return; the
persisting return is byte-identical, dE=0.0, to the non-persisting stacked
path). Every individual contraction is byte-identical (0/97 divergent vs
per-block on concrete forward iterations), so the drift is a downstream
fixed-point sensitivity in the landed stacked core, not in this slice.

Run::

    JAX_PLATFORMS=cpu uv run python examples/measure_persist_sweep_566.py
    JAX_PLATFORMS=cpu uv run python examples/measure_persist_sweep_566.py 2 4
"""

from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from probe_backward_jaxpr_566 import (  # noqa: E402
    backward_vjp_jaxpr,
    count_primitives,
    make_site,
)

import tenax.algorithms._ctm_python_loop as _loop  # noqa: E402
import tenax.contraction.contractor as _ctr  # noqa: E402

STACK_FLAG = "TENAX_STACK_BLOCKSPARSE"


def _reset_trace_caches() -> None:
    """Force a fresh trace so the flag is re-read.

    ``_make_jit_ctm_step`` caches the jitted CTM step for the process lifetime
    (keyed by ``id(neighbors)``), and JAX caches the traced jaxpr. Without
    clearing both, the OFF trace is reused for the ON run and the flag flip is
    invisible. This is a measurement-harness concern only.
    """
    jax.clear_caches()
    _loop._JIT_STEP_CACHE.clear()


def total_ops(counter: collections.Counter) -> int:
    return int(sum(counter.values()))


def _snapshot_persist() -> dict:
    return dict(_ctr._STACK_PERSIST)


def measure(sym: str, D: int, chi_factor: int = 3, pbw: str = "auto") -> dict:
    chi = chi_factor * D
    A = make_site(sym, D)
    n_blocks = getattr(A, "n_blocks", 1)

    rows: dict[bool, tuple[int, int]] = {}
    persist_delta: dict | None = None
    for on in (False, True):
        os.environ[STACK_FLAG] = "1" if on else "0"
        _reset_trace_caches()
        if on:
            before = _snapshot_persist()
        env_jx, par_jx = backward_vjp_jaxpr(A, chi, pbw, full=True)
        c = count_primitives(env_jx) + count_primitives(par_jx)
        rows[on] = (total_ops(c), int(c.get("dot_general", 0)))
        if on:
            after = _snapshot_persist()
            persist_delta = {k: after[k] - before[k] for k in after}
    os.environ[STACK_FLAG] = "0"

    (off_tot, off_dg), (on_tot, on_dg) = rows[False], rows[True]
    ratio = on_tot / off_tot if off_tot else float("nan")

    calls = persist_delta["calls"] if persist_delta else 0
    persisted = persist_delta["persisted_inputs"] if persist_delta else 0
    gathered = persist_delta["gathered_inputs"] if persist_delta else 0
    full = persist_delta["fully_persisted"] if persist_delta else 0
    total_inputs = persisted + gathered

    # The persist snapshot accumulates over BOTH the forward sweep (run once to
    # produce ``out``) AND the env+params VJP tracing inside backward_vjp_jaxpr.
    # We report it as-is (the realized in-trace hit-rate).
    print(f"== {sym} D={D} chi={chi} blocks={n_blocks} ==")
    print(
        f"   backward TOTAL ops: off={off_tot:>7} on={on_tot:>7}  on/off={ratio:.3f}x"
    )
    print(f"   dot_general:        off={off_dg:>7} on={on_dg:>7}")
    if calls:
        pct_in = 100.0 * persisted / total_inputs if total_inputs else 0.0
        pct_call = 100.0 * full / calls if calls else 0.0
        print(
            f"   PERSIST hit-rate:   operands persisted {persisted}/{total_inputs} "
            f"({pct_in:.0f}%)  | fully-persisted calls {full}/{calls} "
            f"({pct_call:.0f}%)  | gathered(interrupted) {gathered}"
        )
    else:
        print("   PERSIST hit-rate:   no stacked contractions fired")
    print()
    return {
        "D": D,
        "ratio": ratio,
        "calls": calls,
        "persisted": persisted,
        "gathered": gathered,
        "fully_persisted": full,
        "off_tot": off_tot,
        "on_tot": on_tot,
    }


def main() -> None:
    print(
        f"# #566 P1d persist measurement | x64={jax.config.read('jax_enable_x64')} "
        f"| flag={STACK_FLAG} | unit=full fused backward (env-VJP + params-VJP)\n"
    )
    Ds = [int(x) for x in sys.argv[1:]] or [2, 4]
    recs = []
    for D in Ds:
        try:
            recs.append(measure("fermionic", D))
        except Exception as e:  # report partial on slow/failed D
            print(f"!! fermionic D={D} did not complete: {type(e).__name__}: {e}\n")

    # ---- VERDICT ----
    print("VERDICT")
    for r in recs:
        if r["calls"]:
            interrupted = r["gathered"]
            total_in = r["persisted"] + r["gathered"]
            read = (
                "contractions-only persistence is enough"
                if r["ratio"] <= 0.34
                else (
                    "fuse/bar interrupt most chains — Task 5b (stacked-aware "
                    "fuse/bar) needed to reach >=3-5x"
                    if interrupted > r["persisted"]
                    else "partial persistence; some interruptions remain"
                )
            )
            print(
                f"  D={r['D']}: backward op-count ON/OFF = {r['ratio']:.3f}x | "
                f"operands persisted {r['persisted']}/{total_in}, "
                f"fully-persisted calls {r['fully_persisted']}/{r['calls']}, "
                f"interrupted(gathered) {interrupted}\n"
                f"          read: {read}."
            )
        else:
            print(f"  D={r['D']}: no stacked contractions fired (even-D scope?).")
    print()
    print(
        "Note: a >=3-5x backward op-count drop is the target. If interruptions "
        "(fuse/bar/svd reading _data) dominate, every contraction chain is length "
        "~1 and persistence cannot help until those ops are made stacked-aware "
        "(Task 5b)."
    )


if __name__ == "__main__":
    main()
