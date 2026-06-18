#!/usr/bin/env python3
"""#618 step 2: clean separate-process backward op histogram for the U(1)-Sz
CTM-AD warm wall.

Run ONE gate value per process so jax.make_jaxpr trace-caching cannot reuse the
gate-OFF trace for the gate-ON call (the contamination that makes the in-process
probe_backward_jaxpr_566.py gate comparison report a spurious 1.00x identical).

Usage::

    JAX_PLATFORMS=cpu uv run python examples/trace_op_histogram_618.py 0   # gate OFF
    JAX_PLATFORMS=cpu uv run python examples/trace_op_histogram_618.py 1   # gate ON
"""

import os, sys, collections, importlib.util, pathlib, jax
jax.config.update("jax_enable_x64", True)
spec = importlib.util.spec_from_file_location("p", "examples/probe_backward_jaxpr_566.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
gate = sys.argv[1]
os.environ["TENAX_BATCH_BLOCKSPARSE"] = gate
A = P.make_site("u1sz", 3)
jx = P.backward_vjp_jaxpr(A, 12, "auto", full=True)
env_jx, par_jx = jx
c = P._combine(P.count_primitives(env_jx), P.count_primitives(par_jx))
b = P.bucketize(c)
print(f"GATE={gate} blocks={A.n_blocks} TOTAL={b['TOTAL']}")
for k in ["contraction","decomp(svd/eigh/qr)","block-pack(slice/scatter/reshape)","transpose","charge-mask / index arith","elementwise(add/mul/...)","other"]:
    print(f"  {k:<40} {b[k]:>8}")
for extra in ["segment_sum","broadcast_in_dim"]:
    print(f"  [raw] {extra:<34} {c.get(extra,0):>8}")
