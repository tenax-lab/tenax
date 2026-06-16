#!/usr/bin/env python3
"""Phase-0 profiler for the #566 symmetric CTM-AD *compile wall*.

Before building a scan-fused jittable CTM-AD step (the #566 redirect), we must
*measure* where the minutes-long compile time actually goes on the PRODUCTION
path -- not the synthetic ``proto_scan_ctm_ising.py`` harness.  Memory note
``symmetric-ipeps-ad-compile-cost-...-566`` is explicit: the wall is
UNCHARACTERIZED, and the earlier "accept the ~20s per-block cost" claim was
retracted.  This script attributes the wall along five axes so the redirect has
a measured target (and a go/no-go: scan-fusion vs the #570 QR/truncated-backprop
runtime axis).

Production path measured
------------------------
``ipeps_optimize._optimize_gs_ad_tensor`` builds its loss via
``ipeps_ad_policy.make_ctm_energy_fn`` and calls ``jax.value_and_grad`` on it
with **no outer jax.jit**.  Defaults (``iPEPSConfig``): ``gs_implicit_ad=True``
-> ``ctm_energy_implicit`` with ``adjoint_method="fixed_point"`` (the jitted F3
fused Neumann backward, ``_jit_fused_fixed_point_bwd``).  The forward is a
Python host loop over an *internally jit-compiled* CTM step
(``_make_jit_ctm_step``), so the "compile" cost is: (forward step compile) +
(F3 backward compile).  We reproduce exactly this dispatcher.

The five axes (each isolates one hypothesised cost)
---------------------------------------------------
  (1) depth sweep  -- ``--depth 4 8 16 32``.  Implicit: ``max_iter``; the
      forward step is reused across iters, so implicit COMPILE should be
      ~FLAT in depth.  Explicit (``--explicit``): ``backprop_steps``; each
      checkpointed sweep is a distinct traced unit, so explicit compile
      should grow ~LINEARLY.  This separates the Python-unroll wall (a)
      (explicit-only) from everything else.
  (2) symmetry   -- ``--sym fermionic dense`` at matched D/chi.  Ratio
      isolates per-block XLA op emission (b) (the #566 core finding).
  (3) chi sweep  -- ``--chi-list 8 12 16 24`` at fixed D.  Steep growth =>
      large-chi SVD/eigh VJP (c) dominates => pivot to #570, NOT scan.
  (4) fwd-vs-v&g -- each config times a cold forward-only call AND a cold
      value_and_grad call.  ``vg_compile - fwd_compile`` isolates the
      implicit-diff BACKWARD compile (d).
  (5) per-jit attribution -- ``jax_log_compiles`` captures every XLA
      compilation's name + duration, so we can see WHICH jitted unit
      (forward step vs F3 backward vs energy) is the single biggest cost.

Cold compiles are forced with ``jax.clear_caches()`` before every timed call.

Usage
-----
CPU smoke (light -- validates the harness; D=2 only)::

    JAX_PLATFORMS=cpu uv run python examples/profile_ctm_ad_wall_566.py \
        --D 2 --chi 6 --depth 4 8 16 --sym fermionic dense \
        --json profile_566_cpu_d2.json

The real wall (run on the box that shows it; D=3 first call is minutes)::

    uv run python examples/profile_ctm_ad_wall_566.py \
        --D 3 --chi 16 --depth 8 --sym fermionic dense --explicit \
        --json profile_566_d3.json

chi-scaling probe (the #570 decision)::

    uv run python examples/profile_ctm_ad_wall_566.py \
        --D 3 --chi-list 8 12 16 24 --depth 8 --sym fermionic \
        --json profile_566_chi.json

JSON is rewritten after every config so a run killed on the wall keeps its rows.
Attach output + JSON to issue #566.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import platform
import re
import shutil
import statistics
import tempfile
import time

import jax
import jax.numpy as jnp

# x64 is the realistic iPEPS-AD regime; set before any array is created.
jax.config.update("jax_enable_x64", True)

# Imported after x64 so internal constants pick up float64.  tenax/__init__.py
# force-enables x64 on import, so we never need to re-assert it for the f64 run.
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _build_initial_fpeps_tensor,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps import heisenberg_gate  # noqa: E402
from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn  # noqa: E402
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402

# --------------------------------------------------------------------------- #
# Per-XLA-compilation capture (axis 5)
# --------------------------------------------------------------------------- #
_COMPILE_RE = re.compile(r"compilation of (.+?) in ([\d.]+) sec", re.IGNORECASE)


class _CompileCapture(logging.Handler):
    """Capture JAX ``jax_log_compiles`` records: (name, seconds) per compile."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.events: list[tuple[str, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = _COMPILE_RE.search(record.getMessage())
        if m:
            self.events.append((m.group(1), float(m.group(2))))

    def reset(self) -> None:
        self.events.clear()


def _install_compile_capture() -> _CompileCapture:
    jax.config.update("jax_log_compiles", True)
    cap = _CompileCapture()
    lg = logging.getLogger("jax")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(cap)
    return cap


# --------------------------------------------------------------------------- #
# Site tensors (matched D/chi; dense = single-block baseline, fermionic = multi)
# --------------------------------------------------------------------------- #
def make_dense_site(D: int, seed: int):
    """U(1)-trivial single-block DenseTensor iPEPS site (the dense baseline)."""
    import numpy as np

    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor, FlowDirection, TensorIndex

    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, 2))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    zD = np.zeros(D, dtype=np.int32)
    zd = np.zeros(2, dtype=np.int32)
    return DenseTensor(
        data,
        (
            TensorIndex.from_charges(sym, zD, FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, zD, FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, zD, FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, zD, FlowDirection.IN, label="r"),
            TensorIndex.from_charges(sym, zd, FlowDirection.IN, label="phys"),
        ),
    )


def make_site_and_gate(sym: str, D: int, seed: int):
    """Return (site_tensor, gate) for the requested symmetry at bond dim D."""
    if sym == "fermionic":
        cfg = FPEPSConfig(D=D, t=1.0, V=0.0)
        A = _build_initial_fpeps_tensor(cfg, jax.random.PRNGKey(seed))
        gate = spinless_fermion_gate(cfg).todense().reshape(2, 2, 2, 2)
        return A, gate
    if sym == "dense":
        return make_dense_site(D, seed), heisenberg_gate()
    if sym == "u1sz":
        from tenax.algorithms.ipeps import (
            heisenberg_gate_u1sz,
            heisenberg_u1sz_init_pair,
        )
        A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))
        return A, heisenberg_gate_u1sz()
    raise ValueError(f"unknown sym {sym!r}")


# --------------------------------------------------------------------------- #
# Loss closure: the real production dispatcher
# --------------------------------------------------------------------------- #
def build_loss(
    gate, chi: int, depth: int, *, explicit: bool, warmup: int, backward_steps=None
):
    """``A -> energy`` via make_ctm_energy_fn (implicit fixed_point default).

    ``backward_steps=K`` (explicit only) enables TBPTT (#506): only the last K of
    the ``depth`` checkpointed sweeps are differentiated. Use to measure the
    truncated-backprop RUNTIME lever (#570): warm-step vs implicit on the A100.
    """
    ctm_cfg = CTMConfig(
        chi=chi,
        max_iter=depth,
        conv_tol=1e-4,
        # production defaults: adjoint_method="fixed_point", ad backward jitted.
    )
    energy_fn = make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache={},
        use_explicit=explicit,
        explicit_warmup=warmup,
        explicit_steps=depth,
        explicit_backward_steps=backward_steps,
    )

    def loss_fn(A_param):
        A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
        return energy_fn({(0, 0): A_norm})

    return loss_fn


_PRIOR_CACHE_DIRS: list[str] = []
atexit.register(
    lambda: [shutil.rmtree(d, ignore_errors=True) for d in _PRIOR_CACHE_DIRS]
)


def _fresh_cache_dir() -> str:
    """Point the persistent compile cache at a fresh temp dir, reaping old ones.

    Importing tenax enables JAX's *persistent* on-disk compilation cache, so
    ``jax.clear_caches()`` alone (in-process only) lets a repeated module compile
    load from disk instead of recompiling -- corrupting cold timings (the
    #584/035d694 lesson).  A fresh dir per cold call guarantees a genuine cold
    XLA compile.  But multi-minute A100 compiles leave large artifacts, and the
    long depth/chi/D grids would otherwise fill /tmp (Codex P2, #585): so we
    delete all *previous* cache dirs (their cold call is done) before opening a
    new one, and register an atexit reaper for the last one.  The current dir
    stays valid through this call's compile + block_until_ready.
    """
    while _PRIOR_CACHE_DIRS:
        shutil.rmtree(_PRIOR_CACHE_DIRS.pop(), ignore_errors=True)
    d = tempfile.mkdtemp(prefix="jax_cc_566_")
    _PRIOR_CACHE_DIRS.append(d)
    jax.config.update("jax_compilation_cache_dir", d)
    return d


def _cold(fn, A, cap: _CompileCapture):
    """Clear caches, run one cold call, return (wall_s, compile_events)."""
    _fresh_cache_dir()
    jax.clear_caches()
    cap.reset()
    t0 = time.perf_counter()
    out = fn(A)
    jax.block_until_ready(out)
    wall = time.perf_counter() - t0
    return wall, list(cap.events), out


def profile_config(
    sym, D, chi, depth, *, explicit, warmup, reps, cap, backward_steps=None
):
    """One (sym,D,chi,depth,path) cell: cold fwd + cold v&g + warm steps."""
    A, gate = make_site_and_gate(sym, D, seed=42)
    n_blocks = getattr(A, "n_blocks", 1)
    loss_fn = build_loss(
        gate,
        chi,
        depth,
        explicit=explicit,
        warmup=warmup,
        backward_steps=backward_steps,
    )
    vg = jax.value_and_grad(loss_fn)

    # (4a) cold forward-only: forward-step compile + run, NO backward graph.
    fwd_wall, fwd_events, E_fwd = _cold(loss_fn, A, cap)
    fwd_compile = sum(t for _, t in fwd_events)

    # (4b) cold value_and_grad: forward + implicit-diff backward compile + run.
    vg_wall, vg_events, (E, g) = _cold(vg, A, cap)
    vg_compile = sum(t for _, t in vg_events)
    # biggest single compilation (name truncated) -> which jitted unit dominates
    top = max(vg_events, key=lambda e: e[1], default=("-", 0.0))

    # warm steady-state (compiled units reused; eager orchestration re-runs)
    steps = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = vg(A)
        jax.block_until_ready(out)
        steps.append(time.perf_counter() - t0)

    grad_finite = bool(jnp.all(jnp.isfinite(g._data)))
    return {
        "sym": sym,
        "D": D,
        "chi": chi,
        "depth": depth,
        "path": "explicit" if explicit else "implicit",
        "backward_steps": backward_steps,
        "n_blocks": int(n_blocks),
        "fwd_wall_s": fwd_wall,
        "fwd_compile_s": fwd_compile,
        "fwd_n_compiles": len(fwd_events),
        "vg_wall_s": vg_wall,
        "vg_compile_s": vg_compile,
        "vg_n_compiles": len(vg_events),
        # backward-only compile estimate: total v&g compile minus fwd compile.
        "bwd_compile_s_est": max(vg_compile - fwd_compile, 0.0),
        "top_compile_name": top[0][:60],
        "top_compile_s": top[1],
        "warm_step_s": statistics.median(steps) if steps else float("nan"),
        "energy": float(E),
        "grad_finite": grad_finite,
    }


def _write_json(path, meta, rows):
    with open(path, "w") as fh:
        json.dump({**meta, "rows": rows}, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, nargs="+", default=[2])
    ap.add_argument(
        "--chi", type=int, default=None, help="Fixed chi (overrides factor)."
    )
    ap.add_argument("--chi-factor", type=int, default=3, help="chi = factor*D.")
    ap.add_argument(
        "--chi-list",
        type=int,
        nargs="+",
        default=None,
        help="Sweep these chi values at the first D (axis 3). Overrides --chi.",
    )
    ap.add_argument("--depth", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--sym", nargs="+", default=["fermionic", "dense"],
                    help="arms: fermionic | dense | u1sz")
    ap.add_argument(
        "--explicit",
        action="store_true",
        help="Also profile the explicit-AD path (checkpointed unroll).",
    )
    ap.add_argument("--warmup", type=int, default=3, help="Explicit warmup sweeps.")
    ap.add_argument("--reps", type=int, default=3, help="Warm steady-state reps.")
    ap.add_argument(
        "--backward-steps",
        type=int,
        default=None,
        help="Explicit TBPTT (#506/#570): differentiate only the last K sweeps. "
        "Implies the explicit path; use to measure the truncated-backprop RUNTIME "
        "lever (warm-step vs implicit). Default None = full backprop.",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    # --backward-steps only affects the explicit path; auto-enable it.
    if args.backward_steps is not None:
        args.explicit = True

    cap = _install_compile_capture()
    dev = jax.devices()[0]

    print("=" * 92)
    print("# Phase-0 CTM-AD compile-wall profiler -- #566")
    print(f"# platform : {dev.platform}  device0={dev.device_kind}")
    print(f"# host     : {platform.platform()}")
    print(f"# x64      : {jax.config.read('jax_enable_x64')}")
    print(
        f"# syms     : {args.sym}   paths: implicit{' + explicit' if args.explicit else ''}"
    )
    print("=" * 92)

    meta = {
        "platform": dev.platform,
        "device_kind": dev.device_kind,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "syms": args.sym,
        "depths": args.depth,
        "explicit": args.explicit,
        "backward_steps": args.backward_steps,
    }

    # Build the (sym, D, chi, depth, path) grid.
    paths = [False] + ([True] if args.explicit else [])
    configs = []
    for sym in args.sym:
        for D in args.D:
            if args.chi_list is not None and D == args.D[0]:
                chis = args.chi_list
            elif args.chi is not None:
                chis = [args.chi]
            else:
                chis = [args.chi_factor * D]
            for chi in chis:
                for depth in args.depth:
                    for explicit in paths:
                        configs.append((sym, D, chi, depth, explicit))

    hdr = (
        f"{'sym':>9} {'D':>2} {'chi':>4} {'dep':>4} {'path':>9} {'blk':>4} "
        f"{'fwd_cmp':>8} {'vg_cmp':>8} {'bwd_cmp':>8} {'vg_wall':>9} "
        f"{'warm_ms':>8} {'top(name:s)':>22}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for sym, D, chi, depth, explicit in configs:
        try:
            r = profile_config(
                sym,
                D,
                chi,
                depth,
                explicit=explicit,
                warmup=args.warmup,
                reps=args.reps,
                cap=cap,
                backward_steps=args.backward_steps if explicit else None,
            )
        except Exception as exc:  # noqa: BLE001 - record + continue the sweep
            print(
                f"{sym:>9} {D:>2} {chi:>4} {depth:>4} "
                f"{'explicit' if explicit else 'implicit':>9}  !! {type(exc).__name__}: {exc}"
            )
            rows.append(
                {
                    "sym": sym,
                    "D": D,
                    "chi": chi,
                    "depth": depth,
                    "path": "explicit" if explicit else "implicit",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if args.json:
                _write_json(args.json, meta, rows)
            continue

        print(
            f"{r['sym']:>9} {r['D']:>2} {r['chi']:>4} {r['depth']:>4} "
            f"{r['path']:>9} {r['n_blocks']:>4} "
            f"{r['fwd_compile_s']:>7.2f}s {r['vg_compile_s']:>7.2f}s "
            f"{r['bwd_compile_s_est']:>7.2f}s {r['vg_wall_s']:>8.2f}s "
            f"{r['warm_step_s'] * 1e3:>7.1f} "
            f"{r['top_compile_name'][:16] + ':' + format(r['top_compile_s'], '.1f'):>22}"
        )
        if not r["grad_finite"]:
            print("    !! NON-FINITE GRADIENT")
        rows.append(r)
        if args.json:
            _write_json(args.json, meta, rows)

    print("-" * len(hdr))
    print("\nReading the table (the Phase-0 gate):")
    print("  * implicit vg_cmp ~FLAT across depth  -> compile wall is NOT unroll;")
    print(
        "    scan-fusion buys runtime (eager dispatch), not compile, for the default path."
    )
    print("  * explicit vg_cmp grows with depth     -> unroll wall (a), explicit-only.")
    print(
        "  * fermionic vg_cmp >> dense at same D/chi -> per-block emission (b) = #566 core."
    )
    print(
        "  * vg_cmp grows steeply with chi        -> large-chi SVD/eigh VJP (c) -> pivot #570."
    )
    print(
        "  * bwd_cmp (= vg_cmp - fwd_cmp) large   -> implicit-diff backward (d) dominates."
    )
    if args.json:
        print(f"\n  JSON -> {args.json}")


if __name__ == "__main__":
    main()
