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
import importlib.util
import os
import pathlib

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

# Load the shared profiler so make_site_and_gate is the single source of truth
# for site construction (especially the u1sz arm added in Task 2).
_prof_spec = importlib.util.spec_from_file_location(
    "profile_ctm_ad_wall_566",
    pathlib.Path(__file__).parent / "profile_ctm_ad_wall_566.py",
)
_PROF = importlib.util.module_from_spec(_prof_spec)
_prof_spec.loader.exec_module(_PROF)

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
    if sym == "u1sz":
        # Reuse make_site_and_gate from the shared profiler for a single source of truth.
        site, _gate = _PROF.make_site_and_gate("u1sz", D, seed=seed)
        return site
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


def backward_vjp_jaxpr(
    A, chi: int, projector_backward: str = "auto", *, full: bool = False
):
    """Jaxpr(s) of the fixed-point backward's CTM-sweep VJP graph(s).

    ``apply_Jt`` (Section B) = VJP of one gauge-fixed CTM sweep w.r.t. the
    ENVIRONMENT — the unit applied every Neumann iteration, and the dominant
    repeated compile unit of ``_jit_fused_fixed_point_bwd``.

    The fused F3 backward also compiles a SECOND sweep VJP w.r.t. the SITE
    PARAMETERS (``gauge_fixed_sweep_from_params`` in ``_jit_chain_rule``) plus
    energy VJPs (Codex P2 #585).  With ``full=True`` we also trace that
    params-sweep VJP and return both, so the histogram covers the whole fused
    unit rather than apply_Jt alone.  ``projector_backward`` matches the profiled
    config (``CTMConfig`` default ``"auto"``); measured byte-identical to
    ``"lorentzian"`` here, but kept faithful.
    """
    A_norm = A * (1.0 / (A.norm() + 1e-10))
    site_tensors = {(0, 0): A_norm}
    env = initialize_ctm_tensor_env(A_norm, chi)
    envs = {(0, 0): env}
    treedef = jax.tree.structure(envs)
    env_leaves = tuple(jax.tree.leaves(envs))
    param_treedef = jax.tree.structure(A_norm)
    param_leaves = tuple(jax.tree.leaves(A_norm))
    jit_step = _make_jit_ctm_step(SINGLE_SITE_NEIGHBORS)

    def gauge_fixed_sweep_from_env(env_leaves_flat):
        e = jax.tree.unflatten(treedef, env_leaves_flat)
        e_out, _eps, _smin = jit_step(
            site_tensors,
            e,
            chi=chi,
            projector_method="svd",
            renormalize=True,
            projector_backward=projector_backward,
        )
        return tuple(
            jax.tree.leaves({c: _phase_fix_ctm_tensor(e_out[c]) for c in e_out})
        )

    out = gauge_fixed_sweep_from_env(env_leaves)
    v = tuple(jnp.ones_like(o) for o in out)
    env_jx = jax.make_jaxpr(
        lambda vv: jax.vjp(gauge_fixed_sweep_from_env, env_leaves)[1](vv)
    )(v)
    if not full:
        return env_jx

    def gauge_fixed_sweep_from_params(p_flat):
        Ap = jax.tree.unflatten(param_treedef, p_flat)
        e_out, _eps, _smin = jit_step(
            {(0, 0): Ap},
            envs,
            chi=chi,
            projector_method="svd",
            renormalize=True,
            projector_backward=projector_backward,
        )
        return tuple(
            jax.tree.leaves({c: _phase_fix_ctm_tensor(e_out[c]) for c in e_out})
        )

    par_jx = jax.make_jaxpr(
        lambda vv: jax.vjp(gauge_fixed_sweep_from_params, param_leaves)[1](vv)
    )(v)
    return env_jx, par_jx


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
    # Charge-mask / index arithmetic: the per-charge-sector comparisons, casts
    # and index iotas/argmaxes that the block-sparse SVD-VJP emits to mask and
    # route each sector.  This cluster scales with the NUMBER of chi-bond charge
    # sectors, so it is the one the #610 sector-drop is expected to shrink.
    "charge-mask / index arith": [
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "and",
        "or",
        "not",
        "convert_element_type",
        "iota",
        "argmax",
        "argmin",
    ],
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


def _combine(*counters: collections.Counter) -> collections.Counter:
    out: collections.Counter = collections.Counter()
    for c in counters:
        out.update(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--chi-factor", type=int, default=3)
    ap.add_argument(
        "--chi",
        type=int,
        default=None,
        help="Fixed chi override (overrides --chi-factor * D when set).",
    )
    ap.add_argument("--sym", nargs="+", default=["fermionic", "dense"])
    ap.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Accepted for CLI compatibility; ignored (trace-only, depth does not "
        "affect the jaxpr).",
    )
    ap.add_argument(
        "--projector-backward",
        default="auto",
        choices=["auto", "lorentzian", "standard"],
        help="Match the profiled CTMConfig (default 'auto'). Measured "
        "byte-identical to 'lorentzian' here.",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Also trace the params-sweep VJP (the _jit_chain_rule indirect "
        "term) so the histogram covers the whole fused F3 backward, not just "
        "apply_Jt (the Neumann matvec).",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Also print the per-primitive raw counts (top-20) alongside the bucket "
        "summary, useful for identifying dominant individual ops.",
    )
    ap.add_argument(
        "--defrag",
        action="store_true",
        help="#610 Gate B: wrap the backward trace in "
        "examples.u1sz_defrag_prototype_610.sector_dropping_truncation() so the "
        "traced CTM truncation drops the |Sz|>1 chi-bond sectors. Compare the "
        "'charge-mask / index arith' cluster vs the un-wrapped baseline.",
    )
    args = ap.parse_args()

    pbw = args.projector_backward
    print(
        f"# #566 backward-VJP jaxpr op-histogram (trace-only) | x64={jax.config.read('jax_enable_x64')}"
    )
    unit = (
        (
            "full fused backward = env-sweep VJP (apply_Jt) + params-sweep VJP "
            "(chain-rule indirect)"
        )
        if args.full
        else "apply_Jt = VJP of one gauge-fixed CTM sweep w.r.t. env (Neumann matvec)"
    )
    print(f"# unit = {unit}")
    print(f"# projector_backward = {pbw}\n")

    # #610 Gate B: optionally wrap the backward trace so the CTM truncation
    # drops |Sz|>1 chi-bond sectors. Mirrors backward_vjp_jaxpr's call signature
    # exactly; the only change is the surrounding sector_dropping_truncation().
    if args.defrag:
        import jax as _jax

        from tenax.algorithms._ctm_uniform_sector import keep_sectors_context

        # #615 Gate B: cold-trace under the REAL keep flag (not the #610
        # monkeypatch). clear_caches so the keep trace is not masked by a prior
        # flag-off jaxpr of the same jit unit (cold-trace caveat).
        _jax.clear_caches()
        _sector_drop = lambda: keep_sectors_context({-1, 0, 1})  # noqa: E731
    else:
        _sector_drop = None

    def _trace(A, chi):
        if args.full:
            env_jx, par_jx = backward_vjp_jaxpr(A, chi, pbw, full=True)
            return _combine(count_primitives(env_jx), count_primitives(par_jx))
        return count_primitives(backward_vjp_jaxpr(A, chi, pbw))

    def hist(A, chi, on):
        os.environ[FLAG] = "1" if on else "0"
        if _sector_drop is not None:
            with _sector_drop():
                raw = _trace(A, chi)
        else:
            raw = _trace(A, chi)
        return bucketize(raw), raw

    for sym in args.sym:
        for D in args.D:
            chi = args.chi if args.chi is not None else args.chi_factor * D
            A = make_site(sym, D)
            n_blocks = getattr(A, "n_blocks", 1)
            off, off_raw = hist(A, chi, on=False)
            onb, onb_raw = hist(A, chi, on=True)
            print(f"== {sym} D={D} chi={chi} blocks={n_blocks} ==")
            hdr = f"  {'bucket':<36} {'flag_off':>10} {'flag_on':>10} {'on/off':>8}"
            print(hdr)
            for b in list(BUCKETS) + ["other", "TOTAL"]:
                o, n = off[b], onb[b]
                ratio = (n / o) if o else float("nan")
                mark = "  <== TOTAL" if b == "TOTAL" else ""
                print(f"  {b:<36} {o:>10} {n:>10} {ratio:>7.2f}x{mark}")
            if args.raw:
                total_off = sum(off_raw.values())
                total_onb = sum(onb_raw.values())
                all_prims = sorted(
                    set(off_raw) | set(onb_raw),
                    key=lambda p: -(off_raw.get(p, 0) + onb_raw.get(p, 0)),
                )
                print("  -- raw primitive counts (top-20, flag_off) --")
                print(f"  {'primitive':<40} {'flag_off':>10} {'pct':>8} {'flag_on':>10} {'pct':>8}")
                for p in all_prims[:20]:
                    o, n = off_raw.get(p, 0), onb_raw.get(p, 0)
                    pct_o = 100.0 * o / total_off if total_off else 0.0
                    pct_n = 100.0 * n / total_onb if total_onb else 0.0
                    print(f"  {p:<40} {o:>10} {pct_o:>7.1f}% {n:>10} {pct_n:>7.1f}%")
            print()


if __name__ == "__main__":
    main()
