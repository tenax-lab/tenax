# C-adjoint Feasibility Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide GO/NO-GO on the C-adjoint direction for the #566 symmetric CTM-AD compile wall by proving (Gate 1) that a `jax.custom_vjp` with `pure_callback` forward/backward makes `value_and_grad` compile O(1) in charge-block count, and (Gate 2) that the gradient still matches production to 1e-6.

**Architecture:** A single throwaway script `examples/spike_ctm_cadjoint_566.py` (zero production edits). It wraps the existing production `ctm_energy_implicit` (built via `make_ctm_energy_fn`) in a parallel `custom_vjp` whose forward and backward are `jax.pure_callback`s. The host callbacks run the production energy and its `jax.vjp` under `jax.disable_jit()`, so the per-block work happens eagerly inside the callback and XLA never emits per-block ops in the outer graph.

**Tech Stack:** JAX (`custom_vjp`, `pure_callback`, `disable_jit`, `value_and_grad`), Tenax (`SymmetricTensor` pytree, `make_ctm_energy_fn`, `ctm_energy_implicit`), `uv` for running, A100 for the measured runs.

---

## Spec

Implements `docs/superpowers/specs/2026-06-19-566-ctm-cadjoint-feasibility-spike-design.md`.

## Background the engineer needs

- **`SymmetricTensor` is a registered pytree with exactly one leaf**: its flat data
  buffer `_data`. `tree_flatten` returns `([self._data], (block_keys, block_shapes,
  block_offsets, indices))`. So given a template tensor `A`, you reconstruct a tensor
  with new data via `jax.tree_util.tree_unflatten(treedef, [new_data])` where
  `treedef = jax.tree_util.tree_structure(A)`. (Verified: `src/tenax/core/tensor.py:795`.)
- **`A * scalar` and `A.norm()` preserve `SymmetricTensor` type** (`tensor.py:1216`,
  `tensor.py:1148`). The production loss normalizes with
  `A * (1.0 / (A.norm() + 1e-10))`.
- **The production energy path:** `make_ctm_energy_fn(neighbors, gate, get_ctm_cfg,
  env_cache, use_explicit=False)` returns `energy_fn(site_tensors_dict)` which calls
  `ctm_energy_implicit(...)` (itself a `jax.custom_vjp` whose backward is the implicit
  fixed-point adjoint). For a 1×1 cell, `site_tensors_dict = {(0, 0): A}` and
  `neighbors = SINGLE_SITE_NEIGHBORS`. This is exactly what
  `examples/profile_ctm_ad_wall_566.py::build_loss` does.
- **`jax.disable_jit()` context** turns every internal `@jax.jit` (`_make_jit_ctm_step`,
  `_jit_fused_fixed_point_bwd`) into eager op-by-op dispatch. `ctm_energy_implicit`'s
  own `custom_vjp` still applies, so `jax.vjp` through it under `disable_jit` yields the
  production *implicit* gradient, computed eagerly (no fused per-block jaxpr).
- **The compile profiler we reuse:** `examples/profile_ctm_ad_wall_566.py` exposes
  `make_site_and_gate(sym, D, seed)`, `_install_compile_capture()` (returns a `cap`
  whose `.events` is a list of `(name, seconds)` per XLA compile), and
  `_cold(fn, A, cap)` (clears caches, points at a fresh on-disk cache dir, runs one cold
  call, returns `(wall_s, events, out)`). Load it the same way
  `examples/profile_warm_dispatch_618.py` does (via `importlib`).
- **Run commands use `uv`** (`uv run python ...`). `python` is not on PATH. The A100 box
  is the target for the measured runs; CPU smoke is fine for correctness-only checks.
- **This is a throwaway spike** living in `examples/` — it is NOT added to `tests/` and
  not collected by CI. "Validation" steps below are asserts inside the script or quick
  `uv run` smoke calls, not a persistent test suite.

## File Structure

- **Create:** `examples/spike_ctm_cadjoint_566.py` — the entire spike (reconstructor,
  `custom_vjp` callback wrapper, Gate-1 compile measurement, Gate-2 correctness check,
  JSON + stdout reporting). One file, one responsibility (this experiment).
- **Create (Task 6):** `examples/spike_ctm_cadjoint_566_summary.md` — findings + verdict.
- **Modify (Task 6):** `docs/superpowers/specs/2026-06-19-566-ctm-cadjoint-feasibility-spike-design.md`
  — flip Status line to the measured outcome.

All work lands on branch `spike/566-ctm-cadjoint-feasibility` (already created).

---

### Task 1: Scaffold + reconstructor round-trip

**Files:**
- Create: `examples/spike_ctm_cadjoint_566.py`

- [ ] **Step 1: Write the scaffold with the reconstructor and a round-trip self-check**

```python
#!/usr/bin/env python3
"""#566 C-adjoint feasibility spike — architectural GO/NO-GO (numpy callbacks, no C).

Wraps the production ``ctm_energy_implicit`` (via ``make_ctm_energy_fn``) in a
parallel ``jax.custom_vjp`` whose forward/backward are ``jax.pure_callback``s.
The host callbacks run the production energy and its ``jax.vjp`` under
``jax.disable_jit()`` so XLA never emits per-block ops in the outer graph.

Two staged gates (see the design spec):
  Gate 1 (compile collapse): spike vg_compile ~flat in block count, seconds not minutes.
  Gate 2 (AD-correctness):   spike grad vs production grad < 1e-6 at fermionic D=2.

Usage::
    uv run python examples/spike_ctm_cadjoint_566.py --self-check
    uv run python examples/spike_ctm_cadjoint_566.py --gate1 --json spike_gate1.json
    uv run python examples/spike_ctm_cadjoint_566.py --gate2 --json spike_gate2.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import platform
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reuse the production loss dispatcher + site/gate builders + compile capture.
_SPEC = importlib.util.spec_from_file_location(
    "profile_ctm_ad_wall_566",
    pathlib.Path(__file__).parent / "profile_ctm_ad_wall_566.py",
)
_PROF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROF)

from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn  # noqa: E402
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402

_EPS = 1e-10


def make_reconstructor(template):
    """Return ``reconstruct(data) -> Tensor`` reusing template's static pytree aux."""
    treedef = jax.tree_util.tree_structure(template)

    def reconstruct(data):
        return jax.tree_util.tree_unflatten(treedef, [data])

    return reconstruct


def leaf_of(tensor):
    """Return the single flat-buffer leaf of a SymmetricTensor/DenseTensor."""
    leaves, _ = jax.tree_util.tree_flatten(tensor)
    assert len(leaves) == 1, f"expected single leaf, got {len(leaves)}"
    return leaves[0]


def _self_check():
    A, _gate = _PROF.make_site_and_gate("fermionic", 2, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    A2 = reconstruct(data)
    d = float(jnp.max(jnp.abs(leaf_of(A2) - data)))
    assert d == 0.0, f"round-trip mismatch {d}"
    print(f"[self-check] reconstruct round-trip exact (n_blocks={A.n_blocks}, "
          f"leaf={data.shape}) OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--gate2", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if args.self_check:
        _self_check()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the round-trip self-check**

Run: `uv run python examples/spike_ctm_cadjoint_566.py --self-check`
Expected: prints `[self-check] reconstruct round-trip exact (n_blocks=16, leaf=(16,)) OK` and exits 0.

- [ ] **Step 3: Commit**

```bash
git add examples/spike_ctm_cadjoint_566.py
git commit -m "spike(#566): scaffold C-adjoint spike + SymmetricTensor reconstructor"
```

---

### Task 2: Forward callback + custom_vjp (stub backward) + forward correctness

**Files:**
- Modify: `examples/spike_ctm_cadjoint_566.py`

- [ ] **Step 1: Add the energy_fn builder, the custom_vjp callback wrapper, and the loss builders**

Insert after `leaf_of` (before `_self_check`):

```python
def build_energy_fn(gate, chi, depth):
    """Production 1x1 implicit-AD energy_fn: site_tensors_dict -> energy."""
    ctm_cfg = CTMConfig(chi=chi, max_iter=depth, conv_tol=1e-4)
    return make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache={},
        use_explicit=False,
    )


def make_ctm_energy_cb(energy_fn, reconstruct, *, stub_backward):
    """custom_vjp over the flat data buffer; fwd/bwd run via pure_callback.

    Host functions run the production energy (and its vjp) under disable_jit,
    so no fused per-block jaxpr is built; XLA sees one opaque op each direction.
    ``stub_backward=True`` returns a zero cotangent (Gate-1 compile test only).
    """

    def host_energy(data_np):
        with jax.disable_jit():
            A = reconstruct(jnp.asarray(data_np))
            return np.asarray(energy_fn({(0, 0): A}), dtype=np.float64)

    def host_grad(data_np, ct_np):
        with jax.disable_jit():
            data = jnp.asarray(data_np)

            def e_of_data(d):
                return energy_fn({(0, 0): reconstruct(d)})

            _, vjp = jax.vjp(e_of_data, data)
            (g,) = vjp(jnp.asarray(ct_np))
            return np.asarray(g, dtype=data_np.dtype)

    @jax.custom_vjp
    def ctm_energy_cb(data):
        return jax.pure_callback(
            host_energy, jax.ShapeDtypeStruct((), data.dtype), data
        )

    def _fwd(data):
        return ctm_energy_cb(data), data

    def _bwd(res, ct):
        data = res
        if stub_backward:
            return (jnp.zeros_like(data),)
        g = jax.pure_callback(
            host_grad, jax.ShapeDtypeStruct(data.shape, data.dtype), data, ct
        )
        return (g,)

    ctm_energy_cb.defvjp(_fwd, _bwd)
    return ctm_energy_cb


def make_losses(template, energy_fn, reconstruct, *, stub_backward):
    """Return (loss_spike, loss_prod), both flat-array -> scalar, with the
    SAME normalization the production loss uses."""
    cb = make_ctm_energy_cb(energy_fn, reconstruct, stub_backward=stub_backward)

    def _normalized(data):
        A = reconstruct(data)
        return A * (1.0 / (A.norm() + _EPS))

    def loss_spike(data):
        return cb(leaf_of(_normalized(data)))

    def loss_prod(data):
        return energy_fn({(0, 0): _normalized(data)})

    return loss_spike, loss_prod
```

- [ ] **Step 2: Add a forward-correctness check and wire a `--fwd-check` flag**

Add this function before `main`:

```python
def _fwd_check(sym="fermionic", D=2, chi=8, depth=8):
    A, gate = _PROF.make_site_and_gate(sym, D, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    energy_fn = build_energy_fn(gate, chi, depth)
    loss_spike, loss_prod = make_losses(
        A, energy_fn, reconstruct, stub_backward=True
    )
    e_spike = float(loss_spike(data))
    e_prod = float(loss_prod(data))
    print(f"[fwd-check] {sym} D={D} chi={chi}: spike={e_spike:.10f} "
          f"prod={e_prod:.10f} |Δ|={abs(e_spike - e_prod):.2e}")
    assert abs(e_spike - e_prod) < 1e-8, "forward energy mismatch"
    # value_and_grad must run without error even with the stub backward.
    val, grad = jax.value_and_grad(loss_spike)(data)
    assert bool(jnp.all(jnp.isfinite(grad))), "non-finite stub grad"
    print(f"[fwd-check] value_and_grad ran (stub grad finite, ||g||="
          f"{float(jnp.linalg.norm(grad)):.2e}) OK")
```

In `main`, add the flag and dispatch:

```python
    ap.add_argument("--fwd-check", action="store_true")
```
```python
    if args.fwd_check:
        _fwd_check()
```

- [ ] **Step 3: Run the forward-correctness check (CPU smoke is fine; D=2 compiles in minutes on first call)**

Run: `uv run python examples/spike_ctm_cadjoint_566.py --fwd-check`
Expected: two `[fwd-check]` lines; `|Δ|` ~1e-12 (well under 1e-8), stub grad finite, exit 0.

- [ ] **Step 4: Commit**

```bash
git add examples/spike_ctm_cadjoint_566.py
git commit -m "spike(#566): custom_vjp+pure_callback fwd (stub bwd) + forward parity"
```

---

### Task 3: Gate 1 — compile-collapse measurement (DECISION POINT)

**Files:**
- Modify: `examples/spike_ctm_cadjoint_566.py`

- [ ] **Step 1: Add the Gate-1 measurement routine and wire `--gate1`**

Add before `main`:

```python
_GATE1_GRID = [
    ("fermionic", 2, 8, 8),
    ("fermionic", 3, 12, 8),
    ("dense", 3, 12, 8),
]


def _measure_spike_compile(sym, D, chi, depth, cap):
    A, gate = _PROF.make_site_and_gate(sym, D, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    energy_fn = build_energy_fn(gate, chi, depth)
    loss_spike, _ = make_losses(A, energy_fn, reconstruct, stub_backward=True)
    vg = jax.value_and_grad(loss_spike)
    wall, events, _out = _PROF._cold(vg, data, cap)
    vg_compile = sum(t for _, t in events)
    return {
        "sym": sym, "D": D, "chi": chi, "depth": depth,
        "n_blocks": int(getattr(A, "n_blocks", 1)),
        "vg_wall_s": wall, "vg_compile_s": vg_compile,
        "n_compiles": len(events),
    }


def run_gate1(json_path=None):
    cap = _PROF._install_compile_capture()
    dev = jax.devices()[0]
    print("=" * 78)
    print("# Gate 1: spike compile collapse  "
          f"[{dev.platform} {dev.device_kind}]")
    print("=" * 78)
    rows = []
    for sym, D, chi, depth in _GATE1_GRID:
        r = _measure_spike_compile(sym, D, chi, depth, cap)
        rows.append(r)
        print(f"  {r['sym']:>9} D={r['D']} chi={r['chi']} blk={r['n_blocks']:>2}: "
              f"vg_compile={r['vg_compile_s']:7.2f}s  wall={r['vg_wall_s']:7.2f}s  "
              f"n_compiles={r['n_compiles']}")
        if json_path:
            with open(json_path, "w") as fh:
                json.dump({"platform": dev.platform, "rows": rows}, fh, indent=2)
    # Verdict
    fD2 = next(r for r in rows if r["sym"] == "fermionic" and r["D"] == 2)
    fD3 = next(r for r in rows if r["sym"] == "fermionic" and r["D"] == 3)
    dD3 = next(r for r in rows if r["sym"] == "dense" and r["D"] == 3)
    ratio = fD3["vg_compile_s"] / max(fD2["vg_compile_s"], 1e-9)
    go = (fD3["vg_compile_s"] < 30.0) and (ratio < 2.0)
    print("-" * 78)
    print(f"  fermionic D2->D3 compile ratio = {ratio:.2f} (GO if < 2.0)")
    print(f"  fermionic D3 compile = {fD3['vg_compile_s']:.2f}s (GO if < 30s)")
    print(f"  fermionic D3 vs dense D3 compile = "
          f"{fD3['vg_compile_s']:.2f}s vs {dD3['vg_compile_s']:.2f}s (want ~equal)")
    print(f"  baseline (recorded) fermionic vg_cmp: 206s -> 2111s (~10x)")
    print(f"\n  GATE 1: {'GO' if go else 'NO-GO'}")
    return go
```

In `main`, dispatch:

```python
    if args.gate1:
        run_gate1(args.json)
```

- [ ] **Step 2: Run Gate 1 on the A100**

Run: `uv run python examples/spike_ctm_cadjoint_566.py --gate1 --json examples/spike_ctm_cadjoint_566_gate1.json`
Expected: three rows; the decisive signal is **fermionic D3 vg_compile in single-digit-to-low-tens of seconds** and **≈ dense D3**, with the D2→D3 ratio `< 2`. Prints `GATE 1: GO` or `NO-GO`.

- [ ] **Step 3: Commit the result**

```bash
git add examples/spike_ctm_cadjoint_566.py examples/spike_ctm_cadjoint_566_gate1.json
git commit -m "spike(#566): Gate-1 compile-collapse measurement + A100 result"
```

- [ ] **Step 4: DECISION POINT** — If `GATE 1: NO-GO`, STOP here. Skip Tasks 4–5, go straight to Task 6 and record the NO-GO (the C-adjoint direction is closed; fall back to formalizing the symmetric NO-GO / dense pivot). If `GATE 1: GO`, continue to Task 4.

---

### Task 4: Real backward callback (host_grad)

**Files:**
- Modify: `examples/spike_ctm_cadjoint_566.py`

The real `host_grad` is already implemented in Task 2's `make_ctm_energy_cb`; it is only bypassed by the `stub_backward` flag. This task adds a smoke check that the **real** backward runs and returns a finite, non-zero gradient.

- [ ] **Step 1: Add a real-backward smoke check and wire `--bwd-smoke`**

Add before `main`:

```python
def _bwd_smoke(sym="fermionic", D=2, chi=8, depth=8):
    A, gate = _PROF.make_site_and_gate(sym, D, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    energy_fn = build_energy_fn(gate, chi, depth)
    loss_spike, _ = make_losses(
        A, energy_fn, reconstruct, stub_backward=False
    )
    g = jax.grad(loss_spike)(data)
    finite = bool(jnp.all(jnp.isfinite(g)))
    nrm = float(jnp.linalg.norm(g))
    print(f"[bwd-smoke] {sym} D={D}: real grad finite={finite} ||g||={nrm:.3e}")
    assert finite and nrm > 0.0, "real backward produced zero/non-finite grad"
```

In `main`:

```python
    ap.add_argument("--bwd-smoke", action="store_true")
```
```python
    if args.bwd_smoke:
        _bwd_smoke()
```

- [ ] **Step 2: Run the real-backward smoke (CPU or A100; D=2)**

Run: `uv run python examples/spike_ctm_cadjoint_566.py --bwd-smoke`
Expected: `[bwd-smoke] fermionic D=2: real grad finite=True ||g||=<positive>`; exit 0.

- [ ] **Step 3: Commit**

```bash
git add examples/spike_ctm_cadjoint_566.py
git commit -m "spike(#566): real host_grad backward smoke (disable_jit + jax.vjp)"
```

---

### Task 5: Gate 2 — AD-correctness vs production

**Files:**
- Modify: `examples/spike_ctm_cadjoint_566.py`

- [ ] **Step 1: Add the Gate-2 routine and wire `--gate2`**

Add before `main`:

```python
def run_gate2(sym="fermionic", D=2, chi=8, depth=8, json_path=None):
    dev = jax.devices()[0]
    print("=" * 78)
    print(f"# Gate 2: AD-correctness vs production  [{dev.platform}] {sym} D={D}")
    print("=" * 78)
    A, gate = _PROF.make_site_and_gate(sym, D, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    energy_fn = build_energy_fn(gate, chi, depth)
    loss_spike, loss_prod = make_losses(
        A, energy_fn, reconstruct, stub_backward=False
    )
    t0 = time.perf_counter()
    g_spike = jax.grad(loss_spike)(data)
    jax.block_until_ready(g_spike)
    t_spike = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_prod = jax.grad(loss_prod)(data)  # one production compile (~minutes at D=2)
    jax.block_until_ready(g_prod)
    t_prod = time.perf_counter() - t0
    max_abs = float(jnp.max(jnp.abs(g_spike - g_prod)))
    denom = float(jnp.max(jnp.abs(g_prod))) + _EPS
    rel = max_abs / denom
    go = max_abs < 1e-6
    print(f"  ||g_spike - g_prod||_inf = {max_abs:.3e}  (rel {rel:.3e})  GO if < 1e-6")
    print(f"  wall: spike grad {t_spike:.1f}s | production grad {t_prod:.1f}s")
    print(f"\n  GATE 2: {'GO' if go else 'NO-GO'}")
    if json_path:
        with open(json_path, "w") as fh:
            json.dump({"platform": dev.platform, "sym": sym, "D": D, "chi": chi,
                       "max_abs_diff": max_abs, "rel_diff": rel,
                       "t_spike_grad_s": t_spike, "t_prod_grad_s": t_prod,
                       "go": go}, fh, indent=2)
    return go
```

In `main`:

```python
    if args.gate2:
        run_gate2(json_path=args.json)
```

- [ ] **Step 2: Run Gate 2 on the A100**

Run: `uv run python examples/spike_ctm_cadjoint_566.py --gate2 --json examples/spike_ctm_cadjoint_566_gate2.json`
Expected: `||g_spike - g_prod||_inf` `< 1e-6` (typically ~1e-12 — same math, eager vs jit reorder); prints `GATE 2: GO`.

- [ ] **Step 3: Commit the result**

```bash
git add examples/spike_ctm_cadjoint_566.py examples/spike_ctm_cadjoint_566_gate2.json
git commit -m "spike(#566): Gate-2 AD-correctness vs production + A100 result"
```

---

### Task 6: Findings summary + spec status + PR

**Files:**
- Create: `examples/spike_ctm_cadjoint_566_summary.md`
- Modify: `docs/superpowers/specs/2026-06-19-566-ctm-cadjoint-feasibility-spike-design.md`

- [ ] **Step 1: Write the findings summary** from the actual Gate-1 / Gate-2 numbers

Create `examples/spike_ctm_cadjoint_566_summary.md` with: platform line; the Gate-1 table
(sym, D, blk, vg_compile, n_compiles) with the recorded baseline (fermionic 206→2111s) for
contrast; the Gate-1 verdict (ratio, D3 compile, fermionic-vs-dense); the Gate-2 line
(max-abs diff, verdict); and a 2-3 sentence conclusion. On GO: "green light to scope the
Phase-2 C kernel (warm-runtime win); compile wall removed for dev/cold-start/CI now." On
NO-GO: which gate failed, the measured number vs threshold, and that the C-adjoint
direction is closed → formalize the symmetric NO-GO / dense pivot.

- [ ] **Step 2: Flip the spec Status line** in
  `docs/superpowers/specs/2026-06-19-566-ctm-cadjoint-feasibility-spike-design.md`
  from `design approved; spec for a throwaway feasibility spike (architectural GO/NO-GO only)`
  to the measured outcome, e.g. `Gate 1 GO / Gate 2 GO (A100, 2026-06-19) — see
  examples/spike_ctm_cadjoint_566_summary.md` (or the NO-GO equivalent).

- [ ] **Step 3: Commit**

```bash
git add examples/spike_ctm_cadjoint_566_summary.md \
        docs/superpowers/specs/2026-06-19-566-ctm-cadjoint-feasibility-spike-design.md
git commit -m "spike(#566): findings summary + spec status (GO/NO-GO record)"
```

- [ ] **Step 4: Push the branch and open a PR** (record artifact; the `🤖` marker satisfies the AI-comment hook)

```bash
git push -u origin spike/566-ctm-cadjoint-feasibility
gh pr create --title "spike(#566): C-adjoint feasibility — architectural GO/NO-GO" \
  --body "$(cat <<'EOF'
> 🤖 **AI-generated PR** — written by Claude Code, posted by @yingjerkao.

Throwaway feasibility spike for the #566 symmetric CTM-AD compile wall. Tests whether a
`jax.custom_vjp` with `pure_callback` forward/backward (production `ctm_energy_implicit`
under `jax.disable_jit`) collapses `value_and_grad` compile to O(1) in charge-block count
(Gate 1) while keeping the gradient correct to 1e-6 (Gate 2). Design:
`docs/superpowers/specs/2026-06-19-566-ctm-cadjoint-feasibility-spike-design.md`.
Findings: `examples/spike_ctm_cadjoint_566_summary.md`. Zero production code touched.
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §3 architecture (custom_vjp + pure_callback + disable_jit, reconstruct via treedef, normalization in JAX) → Tasks 1–2. ✓
- §4 Gate 1 (fermionic D2/D3 + dense D3, jax_log_compiles, `<30s` & `<2×`) → Task 3. ✓
- §5 Gate 2 (grad vs production at fermionic D=2, `<1e-6`) → Tasks 4–5. ✓
- §6 scope/value-proposition (fermionic-only, 1×1, warm deferred) → encoded in the grid + summary (Task 6). ✓
- §8 outcome record (summary + spec status + PR) → Task 6. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"write tests for the above" — every code
step shows complete code; Task 6 prose steps describe exact file content/edits. ✓

**Type consistency:** `make_reconstructor`/`reconstruct`, `leaf_of`, `build_energy_fn`,
`make_ctm_energy_cb(..., stub_backward=)`, `make_losses(...) -> (loss_spike, loss_prod)`,
`_PROF._cold`/`_PROF._install_compile_capture`/`_PROF.make_site_and_gate` are named
identically across Tasks 1–5. `host_grad` returns `dtype=data_np.dtype`; `pure_callback`
result_shape_dtypes match (`()` for energy, `data.shape` for grad). ✓
