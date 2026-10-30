"""Per-program compile and warm time of the implicit-AD backward.

Compares ``adjoint_chain_rule="separate"`` with ``"joint"``.  Run ONE mode per
process: JAX's trace caches make a second in-process run look cheaper.

    JAX_PLATFORMS=cpu uv run python benchmarks/bench_chain_rule_compile.py \
        --mode joint --D 2 --chi 8 --sym dense --path gmres --out joint.json
"""

from __future__ import annotations

import argparse
import collections
import json
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tenax.algorithms._ctm_energy_ad import (  # noqa: E402
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import (  # noqa: E402
    _wrap_as_dense_tensor,
    heisenberg_gate,
    heisenberg_gate_u1sz,
    heisenberg_u1sz_init_pair,
)
from tenax.core.tensor import SymmetricTensor  # noqa: E402

COMPILE = collections.defaultdict(float)
# Same events attributed by pairing with the preceding jaxpr_trace_duration;
# kept only to cross-check the direct attribution (see ``attribution_mismatch``).
COMPILE_PAIRED = collections.defaultdict(float)
_current = {"fun": "?"}


def _listener(event, duration, **kw):
    if event == "/jax/core/compile/jaxpr_trace_duration":
        _current["fun"] = kw.get("fun_name", "?")
    elif event == "/jax/core/compile/backend_compile_duration":
        # The event carries its own ``fun_name`` ("jit(name)"); that is exact.
        name = kw.get("fun_name") or f"jit({_current['fun']})"
        name = name[4:-1] if name.startswith("jit(") and name.endswith(")") else name
        COMPILE[name] += duration
        COMPILE_PAIRED[_current["fun"]] += duration


jm.register_event_duration_secs_listener(_listener)


def _site(D, sym, seed):
    """Random normalised site tensor; for ``u1`` the same Sz-conserving charge
    layout as ``heisenberg_u1sz_init_pair`` (virtual 0,+1,-1 cycle, phys +-1)."""
    if sym == "dense":
        rng = np.random.default_rng(seed)
        a = jnp.asarray(rng.standard_normal((D, D, D, D, 2)))
        return _wrap_as_dense_tensor(a / jnp.linalg.norm(a))
    A, _ = heisenberg_u1sz_init_pair(D, jax.random.PRNGKey(seed))
    nrm = jnp.sqrt(sum(jnp.sum(b**2) for b in A.blocks.values()))
    return SymmetricTensor({k: b / nrm for k, b in A.blocks.items()}, A.indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--persistent-cache",
        action="store_true",
        help="keep JAX's on-disk compilation cache (default: off, because a "
        "cache hit makes backend_compile_duration ~0 and voids the comparison)",
    )
    p.add_argument("--mode", choices=["separate", "joint"], required=True)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--chi", type=int, default=8)
    p.add_argument("--sym", choices=["dense", "u1"], default="dense")
    p.add_argument("--path", choices=["fused", "fallback", "gmres"], default="gmres")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if not a.persistent_cache:
        jax.config.update("jax_enable_compilation_cache", False)

    gate = heisenberg_gate_u1sz() if a.sym == "u1" else heisenberg_gate()
    method = "gmres" if a.path == "gmres" else "fixed_point"
    # "fallback": a fused cap of 1 iteration forces the GMRES fallback.
    extra = {"gmres_maxiter": 1} if a.path == "fallback" else {}
    site = _site(a.D, a.sym, a.seed)

    def loss(A):
        return jnp.real(
            ctm_energy_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                gate,
                chi=a.chi,
                max_iter=60,
                min_iter=8,
                conv_tol=1e-10,
                adjoint_method=method,
                adjoint_chain_rule=a.mode,
                **extra,
            )
        )

    g = jax.grad(loss)
    t0 = time.perf_counter()
    jax.block_until_ready(g(site))
    first = time.perf_counter() - t0
    diag = dict(get_last_implicit_ad_diagnostics())
    warm = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(g(site))
        warm.append(time.perf_counter() - t0)
    res = {
        "args": vars(a),
        "backend": jax.default_backend(),
        "first_grad_s": first,
        "warm_grad_s": sorted(warm)[len(warm) // 2],
        # The fallback cap also caps the GMRES fallback, so only its compile
        # numbers are meaningful.
        "warm_valid": a.path != "fallback",
        "compile_s_by_fun": dict(sorted(COMPILE.items(), key=lambda kv: -kv[1])),
        "compile_total_s": sum(COMPILE.values()),
        "attribution_mismatch_s": sum(
            abs(COMPILE.get(k, 0.0) - COMPILE_PAIRED.get(k, 0.0))
            for k in set(COMPILE) | set(COMPILE_PAIRED)
        ),
        "diag": {
            k: v for k, v in diag.items() if isinstance(v, (int, float, bool, str))
        },
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(
        json.dumps(
            {k: res[k] for k in ("first_grad_s", "warm_grad_s", "compile_total_s")}
        )
    )


if __name__ == "__main__":
    main()
