#!/usr/bin/env python3
"""#570 lever-2: truncated backprop (TBPTT) — gradient parity vs compile cost.

Context
-------
The #570 finding (`docs/superpowers/handoffs/2026-06-07-570-svd-vjp-compile-finding.md`)
showed the fermionic CTM-AD compile wall is the dense **SVD projector VJP**, growing
super-linearly in χ, and that lever-1 (QR projector) is a no-op as a config flip. Lever-2
— **truncated backprop** — differentiates through only the last ``K`` of ``backprop_steps``
CTM sweeps (the leading ones run under ``stop_gradient``; #506's ``backward_steps``). It
shrinks the backward jaxpr **independent of the decomposition**: the explicit backward is
≈ K stacked sweep-VJP units, so compile should scale ~linearly in K — cutting the wall
without touching the projector.

The catch the finding flagged: truncation changes the *gradient*. The forward is identical
for every K (same warmup + backprop_steps sweeps), so **energy is K-invariant** — the only
question is whether the truncated gradient tracks the true (full-adjoint) gradient closely
enough to optimize with. This script measures exactly that trade-off.

What it measures (fixed D, χ)
-----------------------------
  * gold gradients (computed once):
      - ``explicit-full`` : explicit unroll, backward_steps=None (full backprop through
        the SAME forward) — the matched reference isolating the truncation error.
      - ``implicit``      : the production fixed-point adjoint (``ctm_energy_implicit``) —
        the gradient truncated backprop must ultimately approximate.
  * for K in --K-list: explicit truncated (backward_steps=K). Per K:
      - rel_err(g_K, g_full)  = ||g_K - g_full|| / ||g_full||   (truncation cost)
      - cos(g_K, g_full)
      - rel_err(g_K, g_implicit), cos(g_K, g_implicit)          (vs production gold)
      - energy (sanity: K-invariant)
      - cold backward compile time (vg_cmp - fwd_cmp) and #compiles
  * also: rel_err(g_full, g_implicit) — how far explicit-full itself is from the adjoint.

Reading the result (the lever-2 gate)
-------------------------------------
  * rel_err(g_K, gold) falls fast with K AND bwd_cmp grows ~linearly with K
        -> there is a small K with acceptable parity at a fraction of the wall -> lever-2 WINS.
  * rel_err stays large until K≈backprop_steps
        -> truncation can't approximate the adjoint here -> lever-2 weak; need lever-3.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu uv run python examples/profile_570_truncated_backprop.py \
        --D 2 --chi 12 --warmup 4 --steps 8 --K-list 1 2 4 8

A100::

    JAX_PLATFORMS=cuda,cpu uv run python examples/profile_570_truncated_backprop.py \
        --D 2 --chi 12 --warmup 4 --steps 8 --K-list 1 2 3 4 6 8 \
        --json profile_570_tbptt_a100.json

JSON is rewritten after every K so a killed run keeps its rows.
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

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _build_initial_fpeps_tensor,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn  # noqa: E402
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402

# --------------------------------------------------------------------------- #
# Per-XLA-compilation capture (for cold compile timing)
# --------------------------------------------------------------------------- #
_COMPILE_RE = re.compile(r"compilation of (.+?) in ([\d.]+) sec", re.IGNORECASE)


class _CompileCapture(logging.Handler):
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


_PRIOR_CACHE_DIRS: list[str] = []
atexit.register(
    lambda: [shutil.rmtree(d, ignore_errors=True) for d in _PRIOR_CACHE_DIRS]
)


def _fresh_cache_dir() -> None:
    while _PRIOR_CACHE_DIRS:
        shutil.rmtree(_PRIOR_CACHE_DIRS.pop(), ignore_errors=True)
    d = tempfile.mkdtemp(prefix="jax_cc_570tb_")
    _PRIOR_CACHE_DIRS.append(d)
    jax.config.update("jax_compilation_cache_dir", d)


def make_site_and_gate(D: int, seed: int = 42):
    cfg = FPEPSConfig(D=D, t=1.0, V=0.0)
    A = _build_initial_fpeps_tensor(cfg, jax.random.PRNGKey(seed))
    gate = spinless_fermion_gate(cfg).todense().reshape(2, 2, 2, 2)
    return A, gate


def build_loss(
    gate,
    chi: int,
    *,
    explicit: bool,
    warmup: int,
    steps: int,
    backward_steps: int | None,
    max_iter: int,
):
    """``A -> energy`` via make_ctm_energy_fn (explicit TBPTT or implicit adjoint)."""
    ctm_cfg = CTMConfig(chi=chi, max_iter=max_iter, conv_tol=1e-4)
    energy_fn = make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache={},
        use_explicit=explicit,
        explicit_warmup=warmup,
        explicit_steps=steps,
        explicit_backward_steps=backward_steps,
    )

    def loss_fn(A_param):
        A_norm = A_param * (1.0 / (A_param.norm() + 1e-10))
        return energy_fn({(0, 0): A_norm})

    return loss_fn


def _flat_grad(g):
    """Flatten a gradient pytree (SymmetricTensor etc.) to one 1-D array."""
    leaves = jax.tree.leaves(g)
    parts = [jnp.asarray(x).ravel() for x in leaves if jnp.asarray(x).size]
    return jnp.concatenate(parts) if parts else jnp.zeros(())


def _rel_err(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


def _cos(a, b):
    denom = jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30
    return float(jnp.real(jnp.vdot(a, b)) / denom)


def _cold_vg(vg, A, cap):
    """Fresh-cold value_and_grad; return (E, g, fwd-less compile s, n_compiles)."""
    _fresh_cache_dir()
    jax.clear_caches()
    cap.reset()
    t0 = time.perf_counter()
    E, g = vg(A)
    jax.block_until_ready((E, g._data if hasattr(g, "_data") else g))
    wall = time.perf_counter() - t0
    compile_s = sum(t for _, t in cap.events)
    return E, g, wall, compile_s, len(cap.events)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--chi", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8, help="explicit backprop_steps")
    ap.add_argument("--K-list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--steps-list",
        type=int,
        nargs="+",
        default=None,
        help="COMPILE-LEVER mode: sweep backprop_steps S (full backward each), "
        "with warmup = --total-sweeps - S so total forward sweeps is fixed. "
        "This is the config that can actually shrink compile (short differentiated "
        "unroll + cheap forward-only warmup). Overrides --K-list / --warmup / --steps.",
    )
    ap.add_argument(
        "--total-sweeps",
        type=int,
        default=12,
        help="Fixed total forward sweeps for --steps-list mode (warmup+S).",
    )
    ap.add_argument(
        "--implicit-iter",
        type=int,
        default=None,
        help="max_iter for the implicit adjoint gold (default warmup+steps).",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    cap = _install_compile_capture()
    dev = jax.devices()[0]
    A, gate = make_site_and_gate(args.D)
    n_blocks = getattr(A, "n_blocks", 1)
    impl_iter = args.implicit_iter or (args.warmup + args.steps)

    # --- COMPILE-LEVER mode: sweep the differentiated unroll length ----------
    if args.steps_list is not None:
        total = args.total_sweeps
        impl_iter_s = args.implicit_iter or total
        print("=" * 92)
        print("# #570 lever-2: COMPILE-LEVER — sweep differentiated unroll (backprop_steps)")
        print(f"# platform : {dev.platform}  device0={dev.device_kind}")
        print(
            f"# D={args.D} chi={args.chi} blocks={n_blocks} total_sweeps={total} "
            f"impl_iter={impl_iter_s}  x64={jax.config.read('jax_enable_x64')}"
        )
        print("=" * 92)
        print("computing gold: implicit fixed-point adjoint ...", flush=True)
        vg_impl = jax.value_and_grad(
            build_loss(
                gate, args.chi, explicit=False, warmup=0, steps=0,
                backward_steps=None, max_iter=impl_iter_s,
            )
        )
        E_impl, g_impl, _, impl_cmp, impl_nc = _cold_vg(vg_impl, A, cap)
        gi = _flat_grad(g_impl)
        print(
            f"  implicit: E={float(E_impl):.6f}  compile={impl_cmp:.1f}s  "
            f"n_compiles={impl_nc}"
        )
        meta_s = {
            "platform": dev.platform, "device_kind": dev.device_kind,
            "mode": "steps_sweep", "D": args.D, "chi": args.chi,
            "n_blocks": int(n_blocks), "total_sweeps": total,
            "impl_iter": impl_iter_s, "steps_list": args.steps_list,
            "E_impl": float(E_impl), "impl_compile_s": impl_cmp,
            "impl_n_compiles": impl_nc,
        }
        hdr = (
            f"{'S':>3} {'warmup':>6} {'E':>11} {'compile':>9} {'nc':>5} "
            f"{'vs_impl_cmp':>11} {'relerr_vsImpl':>13} {'cos_vsImpl':>10}"
        )
        print("-" * len(hdr)); print(hdr); print("-" * len(hdr))
        rows = []
        for S in args.steps_list:
            warm = max(0, total - S)
            try:
                vg_s = jax.value_and_grad(
                    build_loss(
                        gate, args.chi, explicit=True, warmup=warm, steps=S,
                        backward_steps=None, max_iter=impl_iter_s,
                    )
                )
                E_s, g_s, _, cmp_s, nc_s = _cold_vg(vg_s, A, cap)
                gs = _flat_grad(g_s)
                row = {
                    "S": S, "warmup": warm, "energy": float(E_s),
                    "compile_s": cmp_s, "n_compiles": nc_s,
                    "compile_vs_impl": cmp_s / (impl_cmp + 1e-9),
                    "relerr_vs_impl": _rel_err(gs, gi),
                    "cos_vs_impl": _cos(gs, gi),
                    "grad_finite": bool(jnp.all(jnp.isfinite(gs))),
                }
            except Exception as exc:  # noqa: BLE001
                print(f"{S:>3}  !! {type(exc).__name__}: {exc}")
                rows.append({"S": S, "error": f"{type(exc).__name__}: {exc}"})
                if args.json:
                    with open(args.json, "w") as fh:
                        json.dump({**meta_s, "rows": rows}, fh, indent=2)
                continue
            print(
                f"{row['S']:>3} {row['warmup']:>6} {row['energy']:>11.6f} "
                f"{row['compile_s']:>8.1f}s {row['n_compiles']:>5} "
                f"{row['compile_vs_impl']:>10.2f}x {row['relerr_vs_impl']:>13.3e} "
                f"{row['cos_vs_impl']:>10.6f}"
            )
            if not row["grad_finite"]:
                print("    !! NON-FINITE GRADIENT")
            rows.append(row)
            if args.json:
                with open(args.json, "w") as fh:
                    json.dump({**meta_s, "rows": rows}, fh, indent=2)
        print("-" * len(hdr))
        print("\nGate: a short S with compile << implicit AND cos≈1 vs adjoint => lever-2 WINS.")
        if args.json:
            print(f"\n  JSON -> {args.json}")
        return

    print("=" * 92)
    print("# #570 lever-2: truncated backprop — gradient parity vs compile")
    print(f"# platform : {dev.platform}  device0={dev.device_kind}")
    print(f"# host     : {platform.platform()}")
    print(
        f"# D={args.D} chi={args.chi} blocks={n_blocks} warmup={args.warmup} "
        f"steps={args.steps} impl_iter={impl_iter}  x64={jax.config.read('jax_enable_x64')}"
    )
    print("=" * 92)

    meta = {
        "platform": dev.platform,
        "device_kind": dev.device_kind,
        "D": args.D,
        "chi": args.chi,
        "n_blocks": int(n_blocks),
        "warmup": args.warmup,
        "steps": args.steps,
        "impl_iter": impl_iter,
        "K_list": args.K_list,
    }

    # --- gold gradients -----------------------------------------------------
    print("computing gold: explicit-full (backward_steps=None) ...", flush=True)
    vg_full = jax.value_and_grad(
        build_loss(
            gate, args.chi, explicit=True, warmup=args.warmup, steps=args.steps,
            backward_steps=None, max_iter=impl_iter,
        )
    )
    E_full, g_full, _, full_cmp, full_nc = _cold_vg(vg_full, A, cap)
    gf = _flat_grad(g_full)

    print("computing gold: implicit fixed-point adjoint ...", flush=True)
    vg_impl = jax.value_and_grad(
        build_loss(
            gate, args.chi, explicit=False, warmup=args.warmup, steps=args.steps,
            backward_steps=None, max_iter=impl_iter,
        )
    )
    E_impl, g_impl, _, impl_cmp, impl_nc = _cold_vg(vg_impl, A, cap)
    gi = _flat_grad(g_impl)

    full_vs_impl = _rel_err(gf, gi)
    print(
        f"  E_full={float(E_full):.6f} (cmp {full_cmp:.1f}s, {full_nc} compiles)  "
        f"E_impl={float(E_impl):.6f} (cmp {impl_cmp:.1f}s, {impl_nc} compiles)"
    )
    print(
        f"  rel_err(explicit-full, implicit-adjoint) = {full_vs_impl:.3e}  "
        f"cos = {_cos(gf, gi):.6f}"
    )

    meta.update(
        {
            "E_full": float(E_full),
            "E_impl": float(E_impl),
            "full_compile_s": full_cmp,
            "impl_compile_s": impl_cmp,
            "relerr_full_vs_impl": full_vs_impl,
        }
    )

    # --- truncated sweep ----------------------------------------------------
    hdr = (
        f"{'K':>3} {'E':>11} {'bwd_cmp':>9} {'nc':>4} "
        f"{'relerr_vsFull':>13} {'cos_vsFull':>10} "
        f"{'relerr_vsImpl':>13} {'cos_vsImpl':>10}"
    )
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for K in args.K_list:
        if K > args.steps:
            print(f"{K:>3}  (skip: K>steps={args.steps})")
            continue
        try:
            vg_k = jax.value_and_grad(
                build_loss(
                    gate, args.chi, explicit=True, warmup=args.warmup,
                    steps=args.steps, backward_steps=K, max_iter=impl_iter,
                )
            )
            E_k, g_k, _, cmp_k, nc_k = _cold_vg(vg_k, A, cap)
            gk = _flat_grad(g_k)
            row = {
                "K": K,
                "energy": float(E_k),
                "bwd_compile_s": cmp_k,
                "n_compiles": nc_k,
                "relerr_vs_full": _rel_err(gk, gf),
                "cos_vs_full": _cos(gk, gf),
                "relerr_vs_impl": _rel_err(gk, gi),
                "cos_vs_impl": _cos(gk, gi),
                "grad_finite": bool(jnp.all(jnp.isfinite(gk))),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"{K:>3}  !! {type(exc).__name__}: {exc}")
            rows.append({"K": K, "error": f"{type(exc).__name__}: {exc}"})
            if args.json:
                with open(args.json, "w") as fh:
                    json.dump({**meta, "rows": rows}, fh, indent=2)
            continue
        print(
            f"{row['K']:>3} {row['energy']:>11.6f} {row['bwd_compile_s']:>8.1f}s "
            f"{row['n_compiles']:>4} {row['relerr_vs_full']:>13.3e} "
            f"{row['cos_vs_full']:>10.6f} {row['relerr_vs_impl']:>13.3e} "
            f"{row['cos_vs_impl']:>10.6f}"
        )
        if not row["grad_finite"]:
            print("    !! NON-FINITE GRADIENT")
        rows.append(row)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({**meta, "rows": rows}, fh, indent=2)

    print("-" * len(hdr))
    print("\nGate: rel_err(g_K, gold) small at small K AND bwd_cmp ~linear in K")
    print("      => a cheap K has adjoint-quality gradient => lever-2 WINS.")
    if args.json:
        print(f"\n  JSON -> {args.json}")


if __name__ == "__main__":
    main()
