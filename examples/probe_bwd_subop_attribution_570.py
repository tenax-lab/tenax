#!/usr/bin/env python3
"""SOURCE-attributed op histogram of the #566 fixed-point backward (for #570).

The #200 P-B4 measurement (PR #587) settled that the symmetric-iPEPS CTM-AD
compile wall is the implicit-diff backward ``_jit_fused_fixed_point_bwd`` and is
*backend-invariant* (per-block / stacked / cuTensorNet contraction backends all
land at ~549 s at D=4/chi=12 on A100). The redirect is #570: the large-chi
block-sparse **SVD / eigh / projector VJP** inside that backward.

The existing ``probe_backward_jaxpr_566.py`` buckets ops by PRIMITIVE TYPE, which
HIDES the decomposition cost: an SVD-VJP emits dozens of ``dot_general`` /
``transpose`` / ``div`` / ``broadcast`` ops that get miscounted under
contraction/transpose/elementwise rather than under "SVD". This probe instead
attributes EVERY op to the SOURCE function that emitted it, via the equation's
traceback (``eqn.source_info.traceback.frames``), so the full op footprint of the
SVD / eigh / projector differentiation is counted where it belongs.

It is trace-only (no XLA compile), so it runs in seconds at any D/chi and the
op-count breakdown is platform-independent — exactly the CPU-faithful localization
the NO-GO handoff prescribes ("profile WHICH sub-ops dominate") before committing
#570 work.

Two modes:
  --mode raw     : top (file::function) emitters, data-driven (no preset buckets)
  --mode buckets : categorized (svd_vjp / eigh_vjp / projector / contraction /
                   structural / other) + a chi-sweep, the decisive #570 readout

Usage::

    JAX_PLATFORMS=cpu uv run python examples/probe_bwd_subop_attribution_570.py \
        --mode raw --D 2 --chi 6
    JAX_PLATFORMS=cpu uv run python examples/probe_bwd_subop_attribution_570.py \
        --mode buckets --D 2 --chi 4 6 8 12 --full
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


# --------------------------------------------------------------------------- #
# Construction of the fused backward's VJP graph(s) — mirrors                  #
# probe_backward_jaxpr_566.py so the unit profiled is the SAME compile unit.   #
# --------------------------------------------------------------------------- #
def make_site(D: int, seed: int = 42):
    return _build_initial_fpeps_tensor(
        FPEPSConfig(D=D, t=1.0, V=0.0), jax.random.PRNGKey(seed)
    )


def backward_vjp_jaxprs(A, chi: int, *, full: bool):
    """Return the fused-backward VJP jaxpr(s): env-sweep VJP (+ params-sweep)."""
    A_norm = A * (1.0 / (A.norm() + 1e-10))
    site_tensors = {(0, 0): A_norm}
    env = initialize_ctm_tensor_env(A_norm, chi)
    envs = {(0, 0): env}
    treedef = jax.tree.structure(envs)
    env_leaves = tuple(jax.tree.leaves(envs))
    param_treedef = jax.tree.structure(A_norm)
    param_leaves = tuple(jax.tree.leaves(A_norm))
    jit_step = _make_jit_ctm_step(SINGLE_SITE_NEIGHBORS)

    def sweep_from_env(env_leaves_flat):
        e = jax.tree.unflatten(treedef, env_leaves_flat)
        e_out, _eps, _smin = jit_step(
            site_tensors,
            e,
            chi=chi,
            projector_method="svd",
            renormalize=True,
            projector_backward="auto",
        )
        return tuple(
            jax.tree.leaves({c: _phase_fix_ctm_tensor(e_out[c]) for c in e_out})
        )

    out = sweep_from_env(env_leaves)
    v = tuple(jnp.ones_like(o) for o in out)
    env_jx = jax.make_jaxpr(lambda vv: jax.vjp(sweep_from_env, env_leaves)[1](vv))(v)
    if not full:
        return [env_jx]

    def sweep_from_params(p_flat):
        Ap = jax.tree.unflatten(param_treedef, p_flat)
        e_out, _eps, _smin = jit_step(
            {(0, 0): Ap},
            envs,
            chi=chi,
            projector_method="svd",
            renormalize=True,
            projector_backward="auto",
        )
        return tuple(
            jax.tree.leaves({c: _phase_fix_ctm_tensor(e_out[c]) for c in e_out})
        )

    par_jx = jax.make_jaxpr(lambda vv: jax.vjp(sweep_from_params, param_leaves)[1](vv))(
        v
    )
    return [env_jx, par_jx]


# --------------------------------------------------------------------------- #
# Source attribution                                                          #
# --------------------------------------------------------------------------- #
# Frames that are pure tracing machinery — never the "source" of an op.
_NOISE_FILES = (
    "source_info_util.py",
    "partial_eval.py",
    "core.py",
    "interpreters/ad.py",
    "/ad.py",
    "linear_util.py",
    "api.py",
    "api_util.py",
    "traceback_util.py",
    "pjit.py",
    "custom_derivatives.py",
    "tree_util.py",
    "util.py",
)


def _is_noise(file_name: str) -> bool:
    return any(n in file_name for n in _NOISE_FILES)


def _signif_frames(eqn):
    """Significant (file_name, function) frames for an eqn, innermost-first."""
    tb = eqn.source_info.traceback
    if tb is None:
        return []
    out = []
    for fr in tb.frames:
        fn = fr.file_name
        if _is_noise(fn):
            continue
        out.append((fn.split("/")[-1], fr.function_name))
    return out


def walk_eqns(jaxpr):
    """Yield every primitive eqn (recursing into nested sub-jaxprs)."""
    jpr = getattr(jaxpr, "jaxpr", jaxpr)

    def rec(jx):
        for eqn in jx.eqns:
            yield eqn
            for vparam in eqn.params.values():
                items = vparam if isinstance(vparam, (list, tuple)) else (vparam,)
                for it in items:
                    sub = getattr(it, "jaxpr", it)
                    if hasattr(sub, "eqns"):
                        yield from rec(sub)

    yield from rec(jpr)


# Ordered category rules (first match on ANY significant frame wins). Decomp is
# checked first so an op emitted *inside* svd/eigh (whose frames also include the
# outer projector) is credited to the decomposition, not the projector.
def _categorize(frames) -> str:
    files = [f for f, _ in frames]
    funcs = [c.lower() for _, c in frames]
    fc = list(zip(files, funcs))

    def any_match(pred):
        return any(pred(f, c) for f, c in fc)

    if any_match(lambda f, c: "svd" in c or f == "ad_utils.py" and "svd" in c):
        return "svd_vjp"
    if any_match(lambda f, c: "svd" in f.lower()):
        return "svd_vjp"
    if any_match(
        lambda f, c: "eigh" in c or f in ("_lorentzian_eigh.py", "_ad_primitives.py")
    ):
        return "eigh_vjp"
    if any_match(
        lambda f, c: (
            "_ctm_compiled_moves.py" in f
            or "projector" in f.lower()
            or "projector" in c
            or "_ctm_projector" in f
        )
    ):
        return "projector"
    if any_match(lambda f, c: f == "contractor.py" or "_contract" in c):
        return "contraction"
    if any_match(
        lambda f, c: (
            f
            in (
                "tensor.py",
                "stacked_view.py",
                "index.py",
                "symmetry.py",
                "stacked_tensor.py",
            )
            or "fuse" in c
        )
    ):
        return "structural"
    return "other"


# Expensive-to-COMPILE kernels (each XLA custom-call lowering costs far more
# compile time than a generic elementwise op — op COUNT under-weights these).
_KERNEL_PRIMS = (
    "svd",
    "eigh",
    "qr",
    "geqrf",
    "householder_product",
    "triangular_solve",
    "lu",
    "eig",
)


def analyze(jaxprs):
    by_cat: collections.Counter = collections.Counter()
    by_source: collections.Counter = collections.Counter()
    kernels: collections.Counter = collections.Counter()
    total = 0
    for jx in jaxprs:
        for eqn in walk_eqns(jx):
            total += 1
            frames = _signif_frames(eqn)
            by_cat[_categorize(frames)] += 1
            top = frames[0] if frames else ("<none>", "<none>")
            by_source[f"{top[0]}::{top[1]}"] += 1
            if eqn.primitive.name in _KERNEL_PRIMS:
                kernels[eqn.primitive.name] += 1
    return total, by_cat, by_source, kernels


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["raw", "buckets"], default="buckets")
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--chi", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument(
        "--full",
        action="store_true",
        help="Include the params-sweep VJP (whole fused backward), not just "
        "apply_Jt (the env-sweep Neumann matvec).",
    )
    args = ap.parse_args()

    os.environ.setdefault("TENAX_BATCH_BLOCKSPARSE", "0")
    unit = (
        "env+params sweep VJP (full fused backward)"
        if args.full
        else "env-sweep VJP (apply_Jt / Neumann matvec)"
    )
    print(f"# #570 source-attributed backward op histogram | D={args.D} | unit={unit}")
    print(f"# x64={jax.config.read('jax_enable_x64')} | projector=svd\n")

    A = make_site(args.D)

    if args.mode == "raw":
        chi = args.chi[0]
        jaxprs = backward_vjp_jaxprs(A, chi, full=args.full)
        total, by_cat, by_source, kernels = analyze(jaxprs)
        print(f"== RAW top emitters | D={args.D} chi={chi} | TOTAL ops={total} ==")
        print("  -- by category --")
        for cat, n in by_cat.most_common():
            print(f"    {cat:<28} {n:>8} {100 * n / total:>6.1f}%")
        print("  -- top 25 (file::function) --")
        for src, n in by_source.most_common(25):
            print(f"    {src:<52} {n:>7} {100 * n / total:>5.1f}%")
        print(f"  -- decomposition kernels (expensive to compile) -- {dict(kernels)}")
        return

    # buckets mode: chi-sweep, the decisive #570 readout
    cats = ["svd_vjp", "eigh_vjp", "projector", "contraction", "structural", "other"]
    print(
        f"{'chi':>5} {'TOTAL':>9} "
        + " ".join(f"{c:>11}" for c in cats)
        + f" {'decomp%':>8} {'kernels':>9}"
    )
    rows = []
    for chi in args.chi:
        jaxprs = backward_vjp_jaxprs(A, chi, full=args.full)
        total, by_cat, _src, kernels = analyze(jaxprs)
        decomp = by_cat["svd_vjp"] + by_cat["eigh_vjp"] + by_cat["projector"]
        rows.append((chi, total, by_cat, decomp, kernels))
        cells = " ".join(
            f"{by_cat[c]:>5}/{100 * by_cat[c] / total:>4.1f}%" for c in cats
        )
        nkern = sum(kernels.values())
        print(f"{chi:>5} {total:>9} {cells} {100 * decomp / total:>7.1f}% {nkern:>9}")

    print("\n# DECISIVE #570 READOUT:")
    print(
        "#  - If (svd_vjp+eigh_vjp+projector) op-SHARE GROWS with chi -> #570 "
        "(large-chi decomp VJP) IS the compile lever."
    )
    print(
        "#  - If decomp share stays small/flat while 'structural' dominates -> "
        "the wall is per-block pack/unpack, not the decomposition."
    )
    print(
        "#  - 'kernels' = count of svd/eigh custom-calls (each is "
        "disproportionately expensive to COMPILE; op-count under-weights them)."
    )


if __name__ == "__main__":
    main()
