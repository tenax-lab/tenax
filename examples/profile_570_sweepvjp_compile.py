#!/usr/bin/env python3
"""#570 compile-attribution: cold XLA-compile time of the CTM sweep-VJP unit,
projector_method = svd vs qr vs eigh, swept over chi.

Why this script (and not the full value_and_grad wall)
------------------------------------------------------
P-B4 (#200) localized the minutes-long fermionic CTM-AD compile to the jitted
fixed-point backward ``_jit_fused_fixed_point_bwd`` and proved it is
**backend-invariant** (swapping the contraction executor — perblock -> stacked ->
cuTensorNet — does not move it). The open #570 question is *which sub-op inside
that backward dominates the compile*: the block-sparse **decomposition VJP**
(SVD/eigh of the chi x chi corner/projector matrices) or the per-block
contraction/packing emission (#566).

The trace-only op-histogram (``probe_backward_jaxpr_566.py``) cannot answer it:
its ``decomp`` bucket counts only the literal ``svd``/``eigh`` *primitives*
(flat at 48, chi-blind), NOT their VJP expansion, which lowers into a large
chi-dependent pile of elementwise/matmul/select dense algebra (the SVD-gradient
F-matrix). So we need **compile time**, not op count.

We do NOT need the full ~1000 s value_and_grad compile. ``apply_Jt`` — the VJP of
ONE gauge-fixed CTM sweep w.r.t. the environment (``probe_backward_jaxpr_566``,
Section B) — is the dominant *repeated* compile unit of the fused backward. We
isolate it, ``jax.jit`` it, and time the cold ``lower().compile()`` (no run
needed), for each ``projector_method``, swept over chi. The fingerprint:

  * SVD/eigh sweep-VJP compile grows STEEPLY with chi, QR much flatter
        => decomposition VJP is the wall  => #570 (QR projector) is the lever.
  * all three scale alike (mild) and the growth is elsewhere
        => the wall is per-block emission  => #566, not #570.

``--full`` also compiles the params-sweep VJP (the ``_jit_chain_rule`` indirect
term) so the measured unit covers both sweep VJPs of the fused F3 backward.

Usage
-----
GPU (the box that shows the wall)::

    JAX_PLATFORMS=cuda,cpu uv run python examples/profile_570_sweepvjp_compile.py \
        --D 2 --chi-list 6 12 18 24 36 --methods svd qr eigh \
        --json profile_570_sweepvjp_a100.json

CPU smoke (validates the harness; small chi only)::

    JAX_PLATFORMS=cpu uv run python examples/profile_570_sweepvjp_compile.py \
        --D 2 --chi-list 6 12 --methods svd qr --reps 1

JSON is rewritten after every cell so a run killed on a large-chi compile keeps
its rows.
"""

from __future__ import annotations

import argparse
import atexit
import json
import platform
import shutil
import statistics
import tempfile
import time

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
# Fresh persistent-cache dir per cold compile (the #584/#585 lesson): tenax
# enables JAX's on-disk compile cache, so clear_caches() alone lets a repeat
# compile load from disk and corrupt cold timings. Reap previous dirs to keep
# /tmp from filling on long sweeps.
# --------------------------------------------------------------------------- #
_PRIOR_CACHE_DIRS: list[str] = []
atexit.register(
    lambda: [shutil.rmtree(d, ignore_errors=True) for d in _PRIOR_CACHE_DIRS]
)


def _fresh_cache_dir() -> None:
    while _PRIOR_CACHE_DIRS:
        shutil.rmtree(_PRIOR_CACHE_DIRS.pop(), ignore_errors=True)
    d = tempfile.mkdtemp(prefix="jax_cc_570_")
    _PRIOR_CACHE_DIRS.append(d)
    jax.config.update("jax_compilation_cache_dir", d)


def make_site(D: int, seed: int = 42):
    return _build_initial_fpeps_tensor(
        FPEPSConfig(D=D, t=1.0, V=0.0), jax.random.PRNGKey(seed)
    )


def build_sweep_vjps(A, chi: int, method: str, projector_backward: str = "auto"):
    """Return (apply_Jt_env, apply_Jt_params, v) for the requested projector.

    ``apply_Jt_env``   = VJP of one gauge-fixed sweep w.r.t. the ENVIRONMENT
                         (the Neumann matvec; dominant repeated compile unit).
    ``apply_Jt_params``= VJP w.r.t. the SITE PARAMS (the chain-rule indirect term).
    ``v``              = a cotangent of the right structure for lowering.
    Mirrors ``probe_backward_jaxpr_566.backward_vjp_jaxpr`` exactly, but
    parameterized by ``projector_method`` and returned as callables to compile.
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

    def sweep_from_env(env_leaves_flat):
        e = jax.tree.unflatten(treedef, env_leaves_flat)
        e_out, _eps, _smin = jit_step(
            site_tensors,
            e,
            chi=chi,
            projector_method=method,
            renormalize=True,
            projector_backward=projector_backward,
        )
        return tuple(
            jax.tree.leaves({c: _phase_fix_ctm_tensor(e_out[c]) for c in e_out})
        )

    def sweep_from_params(p_flat):
        Ap = jax.tree.unflatten(param_treedef, p_flat)
        e_out, _eps, _smin = jit_step(
            {(0, 0): Ap},
            envs,
            chi=chi,
            projector_method=method,
            renormalize=True,
            projector_backward=projector_backward,
        )
        return tuple(
            jax.tree.leaves({c: _phase_fix_ctm_tensor(e_out[c]) for c in e_out})
        )

    out = sweep_from_env(env_leaves)
    v = tuple(jnp.ones_like(o) for o in out)

    def apply_Jt_env(vv):
        return jax.vjp(sweep_from_env, env_leaves)[1](vv)

    def apply_Jt_params(vv):
        return jax.vjp(sweep_from_params, param_leaves)[1](vv)

    return apply_Jt_env, apply_Jt_params, v


def _hlo_instr_count(compiled) -> int:
    """Best-effort HLO instruction count of a compiled executable."""
    try:
        txt = compiled.as_text()
    except Exception:  # noqa: BLE001
        return -1
    # HLO instructions are the ' = ' assignment lines inside computations.
    return sum(1 for ln in txt.splitlines() if " = " in ln)


def _cold_compile(fn, v):
    """Fresh-cold ``lower().compile()`` of ``jax.jit(fn)``; return timings+stats.

    Trace+lower and compile are timed separately; compile is the XLA cost we
    care about. No execution — pure compile attribution.
    """
    _fresh_cache_dir()
    jax.clear_caches()
    jitted = jax.jit(fn)
    t0 = time.perf_counter()
    lowered = jitted.lower(v)
    t_lower = time.perf_counter() - t0
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t1
    return t_lower, t_compile, _hlo_instr_count(compiled)


def profile_cell(D, chi, method, *, full, reps, projector_backward):
    A = make_site(D)
    n_blocks = getattr(A, "n_blocks", 1)
    env_fn, par_fn, v = build_sweep_vjps(A, chi, method, projector_backward)

    def measure(fn):
        lowers, compiles, instrs = [], [], -1
        for _ in range(reps):
            tl, tc, ic = _cold_compile(fn, v)
            lowers.append(tl)
            compiles.append(tc)
            instrs = ic
        return (
            statistics.median(lowers),
            statistics.median(compiles),
            instrs,
        )

    env_lower, env_compile, env_instr = measure(env_fn)
    row = {
        "D": D,
        "chi": chi,
        "method": method,
        "n_blocks": int(n_blocks),
        "projector_backward": projector_backward,
        "env_lower_s": env_lower,
        "env_compile_s": env_compile,
        "env_hlo_instr": env_instr,
    }
    if full:
        par_lower, par_compile, par_instr = measure(par_fn)
        row.update(
            {
                "par_lower_s": par_lower,
                "par_compile_s": par_compile,
                "par_hlo_instr": par_instr,
                "total_compile_s": env_compile + par_compile,
            }
        )
    return row


def _write_json(path, meta, rows):
    with open(path, "w") as fh:
        json.dump({**meta, "rows": rows}, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--chi-list", type=int, nargs="+", default=[6, 12, 18, 24])
    ap.add_argument(
        "--methods", nargs="+", default=["svd", "qr", "eigh"]
    )
    ap.add_argument(
        "--projector-backward",
        default="auto",
        choices=["auto", "lorentzian", "standard"],
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Also compile the params-sweep VJP (whole fused backward unit).",
    )
    ap.add_argument("--reps", type=int, default=1, help="Cold-compile reps (median).")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    dev = jax.devices()[0]
    print("=" * 88)
    print("# #570 sweep-VJP compile-attribution (svd vs qr vs eigh)")
    print(f"# platform : {dev.platform}  device0={dev.device_kind}")
    print(f"# host     : {platform.platform()}")
    print(f"# x64      : {jax.config.read('jax_enable_x64')}  D={args.D}")
    print(f"# methods  : {args.methods}   full={args.full}")
    print("=" * 88)

    meta = {
        "platform": dev.platform,
        "device_kind": dev.device_kind,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "D": args.D,
        "methods": args.methods,
        "chi_list": args.chi_list,
        "full": args.full,
        "projector_backward": args.projector_backward,
    }

    col = "total_compile_s" if args.full else "env_compile_s"
    hdr = (
        f"{'method':>6} {'chi':>4} {'blk':>4} {'env_lower':>10} "
        f"{'env_cmp':>9} {'env_instr':>10}"
        + (f" {'par_cmp':>9} {'tot_cmp':>9}" if args.full else "")
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for chi in args.chi_list:
        for method in args.methods:
            try:
                r = profile_cell(
                    args.D,
                    chi,
                    method,
                    full=args.full,
                    reps=args.reps,
                    projector_backward=args.projector_backward,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{method:>6} {chi:>4}  !! {type(exc).__name__}: {exc}")
                rows.append(
                    {
                        "D": args.D,
                        "chi": chi,
                        "method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if args.json:
                    _write_json(args.json, meta, rows)
                continue
            line = (
                f"{r['method']:>6} {r['chi']:>4} {r['n_blocks']:>4} "
                f"{r['env_lower_s']:>9.2f}s {r['env_compile_s']:>8.2f}s "
                f"{r['env_hlo_instr']:>10}"
            )
            if args.full:
                line += f" {r['par_compile_s']:>8.2f}s {r['total_compile_s']:>8.2f}s"
            print(line)
            rows.append(r)
            if args.json:
                _write_json(args.json, meta, rows)

    print("-" * len(hdr))
    print("\nReading the table (the #570 gate):")
    print(
        f"  * svd {col} grows STEEPLY with chi, qr much flatter "
        "-> decomposition VJP is the wall -> #570 (QR projector) is the lever."
    )
    print(
        "  * all methods scale alike (mild) -> wall is per-block emission "
        "-> #566, not #570."
    )
    if args.json:
        print(f"\n  JSON -> {args.json}")


if __name__ == "__main__":
    main()
