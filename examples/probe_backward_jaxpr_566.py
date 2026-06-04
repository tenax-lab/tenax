#!/usr/bin/env python3
"""Op-histogram of the #566 CTM-AD backward graph (trace-only, no XLA compile).

Phase-0 (A100, issue #566) localized the minutes-long symmetric-iPEPS-AD compile
wall to the jitted fixed-point BACKWARD ``_jit_fused_fixed_point_bwd``, dominated
by per-block emission that explodes with D. The single compile unit is the VJP of
ONE gauge-fixed CTM sweep (``apply_Jt``), built in ``_ctm_energy_ad.py`` Section B
as::

    def gauge_fixed_sweep_from_env(env_leaves):
        e = unflatten(env_leaves)
        e_out = jit_step_bwd(site_tensors, e, chi=...)   # one CTM sweep
        return tree_leaves(phase_fix(e_out))
    _, vjp_fn = jax.vjp(gauge_fixed_sweep_from_env, env_leaves)
    apply_Jt = lambda v: vjp_fn(v)[0]

This script reproduces exactly that VJP and counts the **jaxpr primitives**
(recursing into pjit/scan/while/cond sub-jaxprs) by op type, for fermionic vs
dense and ``TENAX_BATCH_BLOCKSPARSE`` off vs on. Tracing is cheap (no XLA
compile), so it runs at D=2/3/4 in seconds — unlike the minutes-long real
compile — and tells us WHERE the op count lives (per-block contraction VJP =
``dot_general``; SVD/eigh VJP = ``svd``/``eigh`` + their dense algebra; flat-
buffer block packing = ``dynamic_slice`` / ``dynamic_update_slice`` /
``reshape``) and whether the existing within-call batching shrinks or grows it.

Usage::

    JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py \
        --D 2 3 --chi-factor 3 --sym fermionic dense
"""

from __future__ import annotations

import argparse
import collections
import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_init import (  # noqa: E402
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ad_utils import _phase_fix_ctm_tensor  # noqa: E402
from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _build_initial_fpeps_tensor,
)

FLAG = "TENAX_BATCH_BLOCKSPARSE"


def make_site(sym: str, D: int, seed: int = 42):
    if sym == "fermionic":
        return _build_initial_fpeps_tensor(
            FPEPSConfig(D=D, t=1.0, V=0.0), jax.random.PRNGKey(seed)
        )
    if sym == "dense":
        import numpy as np

        from tenax.core.symmetry import U1Symmetry
        from tenax.core.tensor import DenseTensor, FlowDirection, TensorIndex

        data = jax.random.normal(jax.random.PRNGKey(seed), (D, D, D, D, 2))
        data = data / jnp.linalg.norm(data)
        sym_ = U1Symmetry()
        zD = np.zeros(D, dtype=np.int32)
        zd = np.zeros(2, dtype=np.int32)
        return DenseTensor(
            data,
            (
                TensorIndex.from_charges(sym_, zD, FlowDirection.OUT, label="u"),
                TensorIndex.from_charges(sym_, zD, FlowDirection.IN, label="d"),
                TensorIndex.from_charges(sym_, zD, FlowDirection.OUT, label="l"),
                TensorIndex.from_charges(sym_, zD, FlowDirection.IN, label="r"),
                TensorIndex.from_charges(sym_, zd, FlowDirection.IN, label="phys"),
            ),
        )
    raise ValueError(sym)


def _as_jaxpr(v):
    """Duck-type a value to its inner Jaxpr (one with ``.eqns``), or None.

    Handles both ClosedJaxpr (has ``.jaxpr``) and bare Jaxpr (has ``.eqns``),
    without importing version-specific jax.core / jax.extend.core classes.
    """
    if hasattr(v, "eqns"):
        return v
    inner = getattr(v, "jaxpr", None)
    if inner is not None and hasattr(inner, "eqns"):
        return inner
    return None


def count_primitives(jaxpr) -> collections.Counter:
    """Recursively count primitive applications in a jaxpr (incl. sub-jaxprs)."""
    counts: collections.Counter = collections.Counter()

    def walk(jx):
        for eqn in jx.eqns:
            counts[eqn.primitive.name] += 1
            # Recurse into nested jaxprs carried in params (pjit, scan, while,
            # cond, custom_vjp_call, closed_call, ...).
            for v in eqn.params.values():
                sub = _as_jaxpr(v)
                if sub is not None:
                    walk(sub)
                elif isinstance(v, (tuple, list)):
                    for item in v:
                        sub = _as_jaxpr(item)
                        if sub is not None:
                            walk(sub)

    walk(_as_jaxpr(jaxpr) or jaxpr.jaxpr)
    return counts


def backward_vjp_jaxpr(A, chi: int):
    """Make the jaxpr of apply_Jt = VJP of one gauge-fixed CTM sweep (Section B)."""
    A_norm = A * (1.0 / (A.norm() + 1e-10))
    site_tensors = {(0, 0): A_norm}
    env = initialize_ctm_tensor_env(A_norm, chi)
    envs = {(0, 0): env}
    treedef = jax.tree.structure(envs)
    env_leaves = tuple(jax.tree.leaves(envs))
    jit_step = _make_jit_ctm_step(SINGLE_SITE_NEIGHBORS)

    def gauge_fixed_sweep_from_env(env_leaves_flat):
        e = jax.tree.unflatten(treedef, env_leaves_flat)
        e_out, _eps, _smin = jit_step(
            site_tensors,
            e,
            chi=chi,
            projector_method="svd",
            renormalize=True,
            projector_backward="lorentzian",
        )
        e_fixed = {c: _phase_fix_ctm_tensor(e_out[c]) for c in e_out}
        return tuple(jax.tree.leaves(e_fixed))

    # cotangent with the output structure (ones-like each output leaf)
    out = gauge_fixed_sweep_from_env(env_leaves)
    v = tuple(jnp.ones_like(o) for o in out)

    def apply_Jt(vv):
        _, vjp_fn = jax.vjp(gauge_fixed_sweep_from_env, env_leaves)
        return vjp_fn(vv)[0]

    return jax.make_jaxpr(apply_Jt)(v)


# Op buckets that matter for the #566 backward-batching question.
BUCKETS = {
    "contraction": ["dot_general"],
    "decomp(svd/eigh/qr)": ["svd", "eigh", "qr", "geqrf", "householder_product"],
    "block-pack(slice/scatter/reshape)": [
        "dynamic_slice",
        "dynamic_update_slice",
        "slice",
        "reshape",
        "gather",
        "scatter",
        "scatter_add",
        "concatenate",
        "pad",
    ],
    "transpose": ["transpose"],
    "elementwise(add/mul/...)": [
        "add",
        "mul",
        "sub",
        "div",
        "neg",
        "abs",
        "select_n",
        "max",
        "integer_pow",
        "sqrt",
    ],
}


def bucketize(counts: collections.Counter) -> dict:
    out = {b: sum(counts.get(p, 0) for p in prims) for b, prims in BUCKETS.items()}
    bucketed = {p for prims in BUCKETS.values() for p in prims}
    out["other"] = sum(c for p, c in counts.items() if p not in bucketed)
    out["TOTAL"] = sum(counts.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--chi-factor", type=int, default=3)
    ap.add_argument("--sym", nargs="+", default=["fermionic", "dense"])
    args = ap.parse_args()

    print(
        f"# #566 backward-VJP jaxpr op-histogram (trace-only) | x64={jax.config.read('jax_enable_x64')}"
    )
    print(
        "# unit = apply_Jt = VJP of one gauge-fixed CTM sweep (the compile-dominant graph)\n"
    )

    for sym in args.sym:
        for D in args.D:
            chi = args.chi_factor * D
            A = make_site(sym, D)
            n_blocks = getattr(A, "n_blocks", 1)
            row = {}
            for on in (False, True):
                os.environ[FLAG] = "1" if on else "0"
                jx = backward_vjp_jaxpr(A, chi)
                row["on" if on else "off"] = bucketize(count_primitives(jx))
            off, onb = row["off"], row["on"]
            print(f"== {sym} D={D} chi={chi} blocks={n_blocks} ==")
            hdr = f"  {'bucket':<36} {'flag_off':>10} {'flag_on':>10} {'on/off':>8}"
            print(hdr)
            for b in list(BUCKETS) + ["other", "TOTAL"]:
                o, n = off[b], onb[b]
                ratio = (n / o) if o else float("nan")
                mark = "  <== TOTAL" if b == "TOTAL" else ""
                print(f"  {b:<36} {o:>10} {n:>10} {ratio:>7.2f}x{mark}")
            print()


if __name__ == "__main__":
    main()
