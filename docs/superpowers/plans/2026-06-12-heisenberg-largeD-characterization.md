# Dense 2D-Heisenberg large-D characterization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone study script that sweeps the dense 2D-Heisenberg iPEPS-AD path over (D, χ) and records final energy (vs −0.6694430), cold-compile time, and runtime — then run it on the A100 and write up where the dense wall hits.

**Architecture:** One self-contained `examples/bench_heisenberg_largeD.py` that mirrors the known-good `examples/heisenberg_ipeps_ad.py` configuration (sublattice-rotated Heisenberg, C4v 1-site, sigma gauge, SVD projector, SU-init) inside a `(D, χ)` grid loop with JSON checkpoint/resume and per-cell error capture. Verification is operational (CPU smoke + the D=2 χ=8 ≈ −0.6625 sanity anchor), not unit tests — consistent with the repo's other `bench_*`/`profile_*` scripts.

**Tech Stack:** Python, JAX (x64, CUDA on A100), tenax `optimize_gs_ad`.

**Spec:** `docs/superpowers/specs/2026-06-12-heisenberg-largeD-characterization-design.md`

> **No-unit-test note:** The spec explicitly excludes pytest tests for this `examples/` script. Each task's "verify" step runs the script and inspects stdout/JSON. This is a deliberate, spec-sanctioned deviation from default TDD, matching `examples/bench_qr_vs_svd_optimize_570.py` and `examples/profile_570_sweepvjp_compile.py` (neither has tests).

---

## File structure

- **Create:** `examples/bench_heisenberg_largeD.py` — the entire study script (problem builder, single-cell runner, grid loop, JSON I/O, table printer, CLI).
- **Generated (not committed by build tasks):** `examples/heisenberg_largeD_a100.json` — results, written by the run.
- **Create (Task 6):** `docs/superpowers/handoffs/2026-06-12-heisenberg-largeD-characterization.md` — the findings writeup.

No `src/` changes. No other files.

---

## Task 1: Lock the exact API from the existing example, then scaffold the single-cell runner

**Files:**
- Read first: `examples/heisenberg_ipeps_ad.py` (ground-truth API), `src/tenax/algorithms/ipeps.py` (gate constructors), `src/tenax/algorithms/ipeps_config.py` (`iPEPSConfig`, `CTMConfig` field names), `src/tenax/algorithms/ipeps_optimize.py:864` (`optimize_gs_ad` signature + `return_history` dict keys).
- Create: `examples/bench_heisenberg_largeD.py`

- [ ] **Step 1: Read the ground-truth example and confirm exact names**

Read `examples/heisenberg_ipeps_ad.py` end to end and note the EXACT:
- import paths for `heisenberg_gate`, `sublattice_rotate_gate`,
- how the SU-init `A_init` is produced (config flag `su_init=True` vs an explicit init call),
- the `iPEPSConfig` / `CTMConfig` field names actually used (`max_bond_dim`, `ctm`, `chi`, `gs_c4v`, `forward_gauge`, `gs_num_steps`, `gs_conv_criterion`, `gs_grad_norm_tol`, `return_history`),
- the exact return arity of `optimize_gs_ad(..., return_history=True)` and the history dict keys (`energies`, `step_times`, `jit_compile_time`, `num_steps`, `converged`).

If any name below differs from the example, **use the example's name** (it is the source of truth) and adjust the code in later steps accordingly.

- [ ] **Step 2: Write the scaffold with `build_problem` and `run_cell` (single cell)**

Create `examples/bench_heisenberg_largeD.py`:

```python
#!/usr/bin/env python3
"""#570 dense 2D-Heisenberg large-D characterization.

Sweeps the dense iPEPS-AD path (sublattice-rotated Heisenberg, C4v 1-site, sigma
gauge, SVD projector, SU-init) over (D, chi) and records final energy vs the
-0.6694430 reference, cold XLA-compile time, and runtime. Mirrors the known-good
examples/heisenberg_ipeps_ad.py configuration inside a grid with JSON
checkpoint/resume. See docs/superpowers/specs/2026-06-12-heisenberg-largeD-characterization-design.md.

Usage (A100):
    CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cuda,cpu \\
      uv run python examples/bench_heisenberg_largeD.py \\
      --D-list 2 3 4 --chi-list 8 16 24 32 --gs-steps 100 \\
      --json examples/heisenberg_largeD_a100.json

CPU smoke:
    JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_largeD.py \\
      --D-list 2 --chi-list 8 --gs-steps 30
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# NOTE: confirm these import paths against examples/heisenberg_ipeps_ad.py in Step 1.
from tenax.algorithms.ipeps import heisenberg_gate, sublattice_rotate_gate  # noqa: E402
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig  # noqa: E402
from tenax.algorithms.ipeps_optimize import optimize_gs_ad  # noqa: E402

REF_ENERGY = -0.6694430  # 2D Heisenberg E/site (Corboz QR-CTMRG / QMC)


def build_problem(D: int, chi: int, gs_steps: int):
    """Return (gate, A_init, config) for one (D, chi) cell.

    Mirrors examples/heisenberg_ipeps_ad.py: sublattice-rotated Heisenberg gate,
    SU-init at this D, C4v 1-site, sigma gauge, SVD projector, grad-norm exit.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    ctm = CTMConfig(chi=chi)  # defaults max_iter=100, conv_tol=1e-8, projector_method="svd"
    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=ctm,
        gs_c4v=True,
        forward_gauge="sigma",
        gs_num_steps=gs_steps,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-5,
        su_init=True,            # confirm this is the SU-init flag in Step 1
        return_history=True,
    )
    A_init = None  # su_init=True builds the initial tensor; confirm in Step 1
    return gate, A_init, config


def run_cell(D: int, chi: int, gs_steps: int) -> dict:
    """Optimize one (D, chi) cell; return a result row."""
    gate, A_init, config = build_problem(D, chi, gs_steps)
    t0 = time.perf_counter()
    out = optimize_gs_ad(gate, A_init, config)
    total_wall = time.perf_counter() - t0
    # return_history=True -> (A_opt, env, E_gs, history)
    *_head, history = out
    energies = list(history["energies"])
    step_times = list(history.get("step_times", []))
    e_final = float(min(energies)) if energies else float("nan")
    warm = step_times[1:] if len(step_times) > 1 else step_times
    return {
        "D": D,
        "chi": chi,
        "E_final": e_final,
        "dE": e_final - REF_ENERGY,
        "jit_compile_s": float(history.get("jit_compile_time", float("nan"))),
        "total_wall_s": total_wall,
        "warm_step_s": float(statistics.median(warm)) if warm else float("nan"),
        "num_steps": int(history.get("num_steps", len(energies))),
        "converged": bool(history.get("converged", False)),
        "below_ref": e_final < REF_ENERGY,  # variational-floor watch
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D-list", type=int, nargs="+", default=[2])
    ap.add_argument("--chi-list", type=int, nargs="+", default=[8])
    ap.add_argument("--gs-steps", type=int, default=100)
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"# heisenberg large-D | {dev.platform} {dev.device_kind} | ref={REF_ENERGY}")
    for D in args.D_list:
        for chi in args.chi_list:
            row = run_cell(D, chi, args.gs_steps)
            print(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: CPU smoke — verify one cell runs and lands near the reference**

Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_largeD.py \
  --D-list 2 --chi-list 8 --gs-steps 30
```
Expected: prints one row dict with `E_final` ≈ −0.66 (descended from a higher value), `converged` possibly False at 30 steps, finite `jit_compile_s`/`warm_step_s`. If `E_final` is positive or NaN, the API wiring is wrong — fix names against the Step-1 reading before proceeding.

- [ ] **Step 4: Commit**

```bash
git add examples/bench_heisenberg_largeD.py
git commit -m "bench(#570): heisenberg large-D study — single-cell runner scaffold"
```

---

## Task 2: Grid loop + JSON checkpoint/resume

**Files:**
- Modify: `examples/bench_heisenberg_largeD.py`

- [ ] **Step 1: Add JSON I/O + resume + cheap-first ordering to `main`**

Replace `main` with:

```python
def _write_json(path, meta, rows):
    with open(path, "w") as fh:
        json.dump({**meta, "rows": rows}, fh, indent=2)


def _load_rows(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh).get("rows", [])
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D-list", type=int, nargs="+", default=[2])
    ap.add_argument("--chi-list", type=int, nargs="+", default=[8])
    ap.add_argument("--gs-steps", type=int, default=100)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    dev = jax.devices()[0]
    meta = {
        "platform": dev.platform,
        "device_kind": dev.device_kind,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "ref_energy": REF_ENERGY,
        "D_list": args.D_list,
        "chi_list": args.chi_list,
        "gs_steps": args.gs_steps,
    }
    rows = _load_rows(args.json)
    done = {(r["D"], r["chi"]) for r in rows}

    print(f"# heisenberg large-D | {dev.platform} {dev.device_kind} | ref={REF_ENERGY}")
    hdr = f"{'D':>3} {'chi':>4} {'E_final':>11} {'dE':>9} {'cmp_s':>8} {'wall_s':>8} {'step_s':>8} {'steps':>6} {'cnv':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:  # reprint resumed rows
        _print_row(r)

    # cheap-first: smallest D*chi first so a kill leaves the expensive tail undone
    cells = sorted(
        ((D, chi) for D in args.D_list for chi in args.chi_list if (D, chi) not in done),
        key=lambda dc: dc[0] * dc[0] * dc[1],
    )
    for D, chi in cells:
        row = run_cell(D, chi, args.gs_steps)
        rows.append(row)
        _print_row(row)
        if args.json:
            _write_json(args.json, meta, rows)


def _print_row(r):
    if "error" in r:
        print(f"{r['D']:>3} {r['chi']:>4}  !! {r['error']}")
        return
    print(
        f"{r['D']:>3} {r['chi']:>4} {r['E_final']:>11.6f} {r['dE']:>9.5f} "
        f"{r['jit_compile_s']:>8.1f} {r['total_wall_s']:>8.1f} "
        f"{r['warm_step_s']:>8.3f} {r['num_steps']:>6} {str(r['converged'])[:1]:>4}"
    )
```

- [ ] **Step 2: Verify grid + JSON write**

Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_largeD.py \
  --D-list 2 --chi-list 8 16 --gs-steps 20 --json /tmp/hl_smoke.json
```
Expected: a 2-row table (D=2 χ=8 then χ=16), and `/tmp/hl_smoke.json` exists with `meta` + 2 `rows`. Check: `python -c "import json;print(len(json.load(open('/tmp/hl_smoke.json'))['rows']))"` prints `2`.

- [ ] **Step 3: Verify resume (skip done cells)**

Run the SAME command again. Expected: it reprints the 2 resumed rows and runs **no** new cells (both `(2,8)` and `(2,16)` already in JSON). Confirm wall-clock is near-instant (no optimization).

- [ ] **Step 4: Commit**

```bash
git add examples/bench_heisenberg_largeD.py
git commit -m "bench(#570): heisenberg large-D study — grid loop, JSON checkpoint + resume"
```

---

## Task 3: Per-cell error capture + cost-ceiling pre-skip

**Files:**
- Modify: `examples/bench_heisenberg_largeD.py`

- [ ] **Step 1: Wrap the cell run in error capture and add a cost ceiling**

In `main`, add the CLI arg and replace the per-cell call:

```python
    ap.add_argument(
        "--max-cost", type=int, default=None,
        help="Skip cells where D*D*chi exceeds this (records a skip row). "
             "Pre-skip guard for the expensive tail; hard hangs are handled "
             "externally via `timeout` + resume.",
    )
```

```python
    for D, chi in cells:
        cost = D * D * chi
        if args.max_cost is not None and cost > args.max_cost:
            row = {"D": D, "chi": chi, "error": f"skipped: cost {cost} > {args.max_cost}"}
        else:
            try:
                row = run_cell(D, chi, args.gs_steps)
            except Exception as exc:  # noqa: BLE001 — OOM / solver failure -> record, continue
                row = {"D": D, "chi": chi, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        _print_row(row)
        if args.json:
            _write_json(args.json, meta, rows)
```

- [ ] **Step 2: Verify error row + resume interaction**

Run (force a skip):
```bash
JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_largeD.py \
  --D-list 2 --chi-list 8 64 --gs-steps 10 --max-cost 100 --json /tmp/hl_err.json
```
Expected: `(2,8)` runs (cost 32 ≤ 100); `(2,64)` prints `!! skipped: cost 256 > 100` and its row has an `error` field. Confirm the sweep does **not** crash and `/tmp/hl_err.json` holds both rows.

- [ ] **Step 3: Commit**

```bash
git add examples/bench_heisenberg_largeD.py
git commit -m "bench(#570): heisenberg large-D study — error capture + cost-ceiling skip"
```

---

## Task 4: Sanity anchor — reproduce the example's D=2 χ=8 energy

**Files:**
- Run only (no code change unless the anchor fails).

- [ ] **Step 1: Run the anchor cell to convergence**

Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_largeD.py \
  --D-list 2 --chi-list 8 --gs-steps 60 --json /tmp/hl_anchor.json
```
Expected: `E_final` ≈ **−0.6625** (the value `examples/heisenberg_ipeps_ad.py` reports at D=2 χ=8), `dE` ≈ +0.007, `below_ref` = False.

- [ ] **Step 2: Decision gate**

If `E_final` is within ~0.002 of −0.6625 → harness is correct; proceed to Task 5.
If it differs materially (or is positive/NaN/below −0.6694), the configuration does not match the example — diff `build_problem` against `examples/heisenberg_ipeps_ad.py` (gauge, c4v, su_init, gate construction) and fix, then re-run this task. **Do not run the A100 sweep until the anchor passes.**

- [ ] **Step 3: Commit (only if a fix was needed)**

```bash
git add examples/bench_heisenberg_largeD.py
git commit -m "bench(#570): heisenberg large-D study — fix config to match example anchor"
```

---

## Task 5: Run the A100 sweep (core grid, then stretch)

**Files:**
- Run only. Produces `examples/heisenberg_largeD_a100.json`.

- [ ] **Step 1: Pick a free GPU**

Run `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` and choose an index with ~0 MiB used (other users share this box). Use it as `CUDA_VISIBLE_DEVICES` below.

- [ ] **Step 2: Run the core grid in the background with external timeout + resume**

```bash
cat > /tmp/run_hl.sh <<'EOF'
set -e
export CUDA_VISIBLE_DEVICES=<FREE_GPU>
export JAX_PLATFORMS=cuda,cpu
cd /home/yjkao/tenax
# `timeout` bounds each whole-sweep attempt; resume continues if a cell kills it.
for attempt in 1 2 3; do
  timeout 3600 uv run python examples/bench_heisenberg_largeD.py \
    --D-list 2 3 4 --chi-list 8 16 24 32 --gs-steps 100 \
    --json examples/heisenberg_largeD_a100.json && break
  echo "### attempt $attempt timed out/killed; resuming ###"
done
echo "### CORE DONE ###"
EOF
nohup bash /tmp/run_hl.sh > /tmp/hl_core.log 2>&1 &
echo "core PID $!"
```
Monitor with an until-loop on `### CORE DONE ###`/`!!` in `/tmp/hl_core.log`. Each cell appends to the JSON; a kill loses only the in-flight cell.

- [ ] **Step 3: Inspect the core grid, then attempt the stretch**

Read `examples/heisenberg_largeD_a100.json`. If D=4 cells completed within budget and energies are still improving with χ, run the stretch (reuses the same JSON; resume skips completed cells):
```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> JAX_PLATFORMS=cuda,cpu \
  timeout 5400 uv run python examples/bench_heisenberg_largeD.py \
  --D-list 5 6 --chi-list 32 48 64 --gs-steps 100 \
  --json examples/heisenberg_largeD_a100.json
```
Record which stretch cells hit the wall (OOM error row / timeout-killed-and-not-resumed / no energy gain).

- [ ] **Step 4: Commit the results JSON**

```bash
git add examples/heisenberg_largeD_a100.json
git commit -m "bench(#570): heisenberg large-D A100 sweep results (dense path)"
```

---

## Task 6: Write the characterization handoff

**Files:**
- Create: `docs/superpowers/handoffs/2026-06-12-heisenberg-largeD-characterization.md`

- [ ] **Step 1: Write the findings doc from the JSON**

Populate from `examples/heisenberg_largeD_a100.json`. Sections (fill with the actual numbers — no placeholders):
1. **Result table** — one row per (D, χ): `E_final`, `dE` vs −0.6694430, `jit_compile_s`, `total_wall_s`, `warm_step_s`, `num_steps`, `converged`.
2. **Energy vs (D, χ)** — how close the dense path gets to −0.6694430; best cell; whether energy is still improving with D and with χ at the largest cells.
3. **Runtime + compile scaling** — how `jit_compile_s` and `warm_step_s` grow with D and χ; the dominant cost.
4. **Where the dense wall hits** — first cell that OOM'd / timed out / stopped gaining energy; the practical (D, χ) ceiling of the dense path on an 80 GB A100.
5. **Conclusion** — the #570 answer for the dense path, and the explicit statement that **U(1)/Sz symmetry** is the lever to push toward the reference scale (D=7 χ=300) — out of this study's scope, candidate next step.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/handoffs/2026-06-12-heisenberg-largeD-characterization.md
git commit -m "docs(#570): dense Heisenberg large-D characterization — findings"
```

---

## Self-review (completed by plan author)

- **Spec coverage:** measurement protocol → Task 1+2 (`run_cell` row fields match the spec list); grid + cheap-first + resume → Task 2; time-boxing/error capture → Task 3 (external `timeout`+resume for hard hangs, documented deviation from the spec's in-process "cap" since XLA compile can't be interrupted mid-flight); sanity anchor → Task 4; variational-floor watch → `below_ref` field (Task 1) + writeup §4; A100 sweep + stretch → Task 5; outputs (script, JSON, writeup) → Tasks 1–3, 5, 6; non-goals respected (no U(1), no 2-site, no fermionic, no QR, no tests). `eps_T`/smallest-S omitted per spec ("only if already returned… otherwise omitted").
- **Placeholder scan:** none — all code shown; the only deliberate `<FREE_GPU>` token in Task 5 is an operator-supplied runtime value with explicit instructions to obtain it.
- **Type consistency:** `run_cell` returns the exact keys `_print_row` and the writeup consume (`D, chi, E_final, dE, jit_compile_s, total_wall_s, warm_step_s, num_steps, converged, below_ref`, or `error`); resume keys on `(D, chi)` consistently.
- **API caveat:** Task 1 Step 1 mandates confirming import paths and config field names against `examples/heisenberg_ipeps_ad.py` before relying on the scaffold, since those are reconstructed from exploration, not verified line-by-line.
