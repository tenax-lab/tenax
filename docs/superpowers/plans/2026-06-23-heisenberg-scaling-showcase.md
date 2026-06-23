# iPEPS Heisenberg Scaling/Perf Showcase — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, checkpointed driver that runs a wide (D, χ) scaling sweep of the 2D square-lattice Heisenberg iPEPS ground state on 4×A100, reporting per-step timing + per-device peak memory with an explicit 1-GPU vs 4-GPU comparison, using only production-merged levers (`gs_recipe="1x1"` + `CTMConfig.device_mesh`).

**Architecture:** A single example file `examples/showcase_heisenberg_scaling.py` is BOTH an orchestrator (default mode) and a per-cell worker (`--cell` mode). The cumulative `peak_bytes_in_use` high-water trap forces **one cell = one OS subprocess**: the orchestrator launches each `(D, χ, n_devices, gs_num_steps)` cell as a subprocess with the right `CUDA_VISIBLE_DEVICES`, collects a per-cell JSON (checkpoint/resume), and aggregates a table/CSV/plots. The worker calls `optimize_gs_ad(..., return_history=True)` once and reports `median(step_times[1:])` + per-device peak.

**Tech Stack:** Python, JAX, tenax (`optimize_gs_ad`, `heisenberg_gate`, `sublattice_rotate_gate`, `build_ctm_mesh`, `iPEPSConfig`/`CTMConfig`), pytest, matplotlib (Agg). Pure orchestration helpers are jax-free and unit-tested; the GPU path is validated by the Phase-0 gate.

**Key invariants (do not violate):**
- Pure helpers (`Cell`, `build_grid`, `cell_result_path`, `should_stop_row`, `cell_to_argv_env`, `results_to_markdown`, `results_to_csv_rows`) import only stdlib — NO `jax`, NO `matplotlib` at module top. `jax`/`tenax` imports live INSIDE `run_cell`; `matplotlib` inside `make_plots`. This keeps the helper unit tests fast (`core` bucket) and lets `CUDA_VISIBLE_DEVICES` (set by the parent in the child's env) take effect before the child imports jax.
- Module top-level must be import-safe (no side effects) and guarded by `if __name__ == "__main__":`.
- Reference energy constant: `REFERENCE_E = -0.669437` (Sandvik QMC, square-lattice spin-½ Heisenberg AFM).

---

## File Structure

| File | Responsibility |
|---|---|
| `examples/showcase_heisenberg_scaling.py` (create) | Orchestrator + worker + pure helpers + plotting, one CLI. |
| `examples/showcase_results/` (created at runtime) | Per-cell JSON results (resume checkpoint); git-ignored output. |
| `tests/test_showcase_scaling.py` (create) | Unit tests for the pure helpers (loaded via importlib by file path). |
| `tests/conftest.py` (modify) | Register `test_showcase_scaling.py` as `core` so CI runs it. |
| `docs/superpowers/handoffs/2026-06-23-heisenberg-scaling-showcase-findings.md` (create, final task) | Findings: scaling table, plots, Phase-0 outcome, ceilings. |

---

### Task 1: Module scaffold + `Cell` + `build_grid`

**Files:**
- Create: `examples/showcase_heisenberg_scaling.py`
- Create: `tests/test_showcase_scaling.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_showcase_scaling.py`:

```python
"""Unit tests for the pure orchestration helpers of the Heisenberg scaling
showcase. The example file is loaded by path (it is not an importable package)
so these tests stay jax-free and fast."""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "examples" / "showcase_heisenberg_scaling.py"
_spec = importlib.util.spec_from_file_location("showcase_heisenberg_scaling", _PATH)
showcase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(showcase)


def test_build_grid_enumerates_metrics_and_anchor_cells():
    cells = showcase.build_grid(
        D_list=[2, 3],
        chi_ramp=[16, 32],
        device_counts=[1, 4],
        anchors=[(2, 32)],
        metrics_steps=5,
        anchor_steps=80,
    )
    metrics = [c for c in cells if not c.is_anchor]
    anchors = [c for c in cells if c.is_anchor]
    # 2 device_counts * 2 D * 2 chi metrics cells:
    assert len(metrics) == 8
    # 2 device_counts * 1 anchor:
    assert len(anchors) == 2
    # spot-check a metrics cell carries the right fields:
    c = next(c for c in metrics if c.D == 3 and c.chi == 16 and c.n_devices == 4)
    assert c.gs_num_steps == 5 and c.is_anchor is False
    # anchor cells carry the anchor step budget:
    assert all(c.gs_num_steps == 80 for c in anchors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_showcase_scaling.py::test_build_grid_enumerates_metrics_and_anchor_cells -v`
Expected: FAIL — `FileNotFoundError` (example file absent) or `AttributeError: build_grid`.

- [ ] **Step 3: Write minimal implementation**

Create `examples/showcase_heisenberg_scaling.py`:

```python
"""iPEPS square-lattice Heisenberg scaling / perf showcase.

Orchestrator + per-cell worker in one file. Run modes:

    # full sweep (orchestrator): launches one subprocess per cell
    uv run python examples/showcase_heisenberg_scaling.py

    # single cell (worker; normally invoked by the orchestrator):
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/showcase_heisenberg_scaling.py --cell \
        --D 2 --chi 16 --n-devices 1 --gs-num-steps 5 --out /tmp/cell.json

Pure helpers (Cell/build_grid/...) import only stdlib; jax/tenax imports live
inside run_cell so CUDA_VISIBLE_DEVICES (set by the parent) takes effect before
the child initialises a JAX backend, and so the helper unit tests stay fast.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

REFERENCE_E = -0.669437  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM


@dataclass(frozen=True)
class Cell:
    D: int
    chi: int
    n_devices: int       # 1 or 4
    gs_num_steps: int     # small => metrics-only; large => anchor (trusted energy)
    is_anchor: bool


def build_grid(D_list, chi_ramp, device_counts, anchors, metrics_steps, anchor_steps):
    """Enumerate all cells: a metrics cell per (n_devices, D, chi), plus an anchor
    cell per (n_devices, (D, chi) in anchors). Deterministic order."""
    cells = []
    for n in device_counts:
        for D in D_list:
            for chi in chi_ramp:
                cells.append(Cell(D, chi, n, metrics_steps, is_anchor=False))
    for n in device_counts:
        for (D, chi) in anchors:
            cells.append(Cell(D, chi, n, anchor_steps, is_anchor=True))
    return cells
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_showcase_scaling.py::test_build_grid_enumerates_metrics_and_anchor_cells -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/showcase_heisenberg_scaling.py tests/test_showcase_scaling.py
git commit -m "feat(showcase): Cell + build_grid for Heisenberg scaling sweep"
```

---

### Task 2: `cell_result_path` + `should_stop_row` (resume + OOM-aware ramp)

**Files:**
- Modify: `examples/showcase_heisenberg_scaling.py`
- Test: `tests/test_showcase_scaling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_showcase_scaling.py`:

```python
def test_cell_result_path_is_unique_and_descriptive():
    metrics = showcase.Cell(D=3, chi=48, n_devices=4, gs_num_steps=5, is_anchor=False)
    anchor = showcase.Cell(D=3, chi=48, n_devices=4, gs_num_steps=80, is_anchor=True)
    pm = showcase.cell_result_path("results", metrics)
    pa = showcase.cell_result_path("results", anchor)
    assert pm.endswith("D3_chi48_n4_metrics.json")
    assert pa.endswith("D3_chi48_n4_anchor.json")
    assert pm != pa  # anchor and metrics at the same (D,chi,n) must not collide


def test_should_stop_row_on_oom_or_error_only():
    assert showcase.should_stop_row({"oom": True, "error": None}) is True
    assert showcase.should_stop_row({"oom": False, "error": "Boom"}) is True
    assert showcase.should_stop_row({"oom": False, "error": None}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_showcase_scaling.py -k "result_path or should_stop" -v`
Expected: FAIL — `AttributeError: cell_result_path` / `should_stop_row`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/showcase_heisenberg_scaling.py` (after `build_grid`):

```python
def cell_result_path(results_dir, cell):
    """Per-cell JSON path. Anchor and metrics cells at the same (D,chi,n) get
    distinct files so resume never confuses them."""
    kind = "anchor" if cell.is_anchor else "metrics"
    return str(Path(results_dir) / f"D{cell.D}_chi{cell.chi}_n{cell.n_devices}_{kind}.json")


def should_stop_row(result):
    """Stop ramping chi for a (D, n_devices) row once a cell OOMs or errors."""
    return bool(result.get("oom") or result.get("error"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_showcase_scaling.py -k "result_path or should_stop" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/showcase_heisenberg_scaling.py tests/test_showcase_scaling.py
git commit -m "feat(showcase): per-cell result paths + OOM-aware row stop"
```

---

### Task 3: `cell_to_argv_env` (subprocess mapping)

**Files:**
- Modify: `examples/showcase_heisenberg_scaling.py`
- Test: `tests/test_showcase_scaling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_showcase_scaling.py`:

```python
def test_cell_to_argv_env_sets_devices_and_flags():
    cell1 = showcase.Cell(D=4, chi=64, n_devices=1, gs_num_steps=5, is_anchor=False)
    cell4 = showcase.Cell(D=4, chi=64, n_devices=4, gs_num_steps=80, is_anchor=True)
    base_env = {"PATH": "/usr/bin", "HOME": "/home/u"}

    argv1, env1 = showcase.cell_to_argv_env(
        cell1, results_dir="results", python_exe="python",
        script_path="examples/showcase_heisenberg_scaling.py", base_env=base_env)
    argv4, env4 = showcase.cell_to_argv_env(
        cell4, results_dir="results", python_exe="python",
        script_path="examples/showcase_heisenberg_scaling.py", base_env=base_env)

    # single-GPU cell pins device 0; 4-GPU cell pins 0,1,2,3 (NOT the display GPU)
    assert env1["CUDA_VISIBLE_DEVICES"] == "0"
    assert env4["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    # preallocation must be off so peak_bytes_in_use is meaningful
    assert env1["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    # base env is preserved, not replaced
    assert env1["PATH"] == "/usr/bin"
    # argv carries the worker flag, the cell params, and the right out path
    assert "--cell" in argv1
    assert "--D" in argv1 and "4" in argv1
    assert argv1[-1].endswith("D4_chi64_n1_metrics.json")
    assert argv4[-1].endswith("D4_chi64_n4_anchor.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_showcase_scaling.py -k argv_env -v`
Expected: FAIL — `AttributeError: cell_to_argv_env`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/showcase_heisenberg_scaling.py`:

```python
def cell_to_argv_env(cell, results_dir, python_exe, script_path, base_env):
    """Map a Cell to (argv, env) for its subprocess. Pins CUDA_VISIBLE_DEVICES
    (device 0 for 1-GPU; 0..n-1 for n-GPU — never the display GPU at index 4)
    and disables XLA preallocation so peak memory is real."""
    out = cell_result_path(results_dir, cell)
    argv = [
        python_exe, script_path, "--cell",
        "--D", str(cell.D),
        "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices),
        "--gs-num-steps", str(cell.gs_num_steps),
        "--out", out,
    ]
    devices = "0" if cell.n_devices == 1 else ",".join(str(i) for i in range(cell.n_devices))
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return argv, env
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_showcase_scaling.py -k argv_env -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/showcase_heisenberg_scaling.py tests/test_showcase_scaling.py
git commit -m "feat(showcase): map cells to subprocess argv+env (device pinning)"
```

---

### Task 4: `results_to_markdown` + `results_to_csv_rows` (aggregation)

**Files:**
- Modify: `examples/showcase_heisenberg_scaling.py`
- Test: `tests/test_showcase_scaling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_showcase_scaling.py`:

```python
def _sample_results():
    return [
        {"D": 2, "chi": 16, "n_devices": 1, "is_anchor": False, "oom": False,
         "error": None, "ms_per_step": 12.5, "peak_gb": 1.2, "E_site": None,
         "converged": False},
        {"D": 2, "chi": 32, "n_devices": 4, "is_anchor": True, "oom": False,
         "error": None, "ms_per_step": 40.0, "peak_gb": 0.9, "E_site": -0.6690,
         "converged": True},
        {"D": 4, "chi": 96, "n_devices": 1, "is_anchor": False, "oom": True,
         "error": None, "ms_per_step": None, "peak_gb": None, "E_site": None,
         "converged": False},
    ]


def test_results_to_markdown_has_header_and_values_and_oom():
    md = showcase.results_to_markdown(_sample_results())
    assert "ms/step" in md and "peak GB" in md
    assert "12.5" in md          # a metrics timing
    assert "-0.6690" in md or "-0.669" in md  # an anchor energy
    assert "OOM" in md           # the OOM cell is shown, not dropped


def test_results_to_csv_rows_are_flat_and_stable():
    rows = showcase.results_to_csv_rows(_sample_results())
    assert len(rows) == 3
    for r in rows:
        assert set(["D", "chi", "n_devices", "ms_per_step", "peak_gb",
                    "E_site", "converged", "oom"]).issubset(r.keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_showcase_scaling.py -k "markdown or csv" -v`
Expected: FAIL — `AttributeError: results_to_markdown`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/showcase_heisenberg_scaling.py`:

```python
def _status(r):
    if r.get("oom"):
        return "OOM"
    if r.get("error"):
        return "ERR"
    return "ok"


def _fmt(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def results_to_csv_rows(results):
    """Flatten results to stable-keyed dicts for CSV export."""
    keys = ["D", "chi", "n_devices", "is_anchor", "gs_num_steps",
            "ms_per_step", "peak_gb", "E_site", "converged", "oom", "error"]
    return [{k: r.get(k) for k in keys} for r in results]


def results_to_markdown(results):
    """Render a scaling table grouped by device count, sorted by (D, chi)."""
    lines = []
    for n in sorted({r["n_devices"] for r in results}):
        lines.append(f"\n### {n}-GPU\n")
        lines.append("| D | χ | kind | status | ms/step | peak GB | E/site | dE_ref | conv |")
        lines.append("|---|---|------|--------|---------|---------|--------|--------|------|")
        rows = sorted([r for r in results if r["n_devices"] == n],
                      key=lambda r: (r["D"], r["chi"], r.get("is_anchor", False)))
        for r in rows:
            kind = "anchor" if r.get("is_anchor") else "metrics"
            e = r.get("E_site")
            d_ref = (e - REFERENCE_E) if isinstance(e, (int, float)) else None
            lines.append(
                f"| {r['D']} | {r['chi']} | {kind} | {_status(r)} | "
                f"{_fmt(r.get('ms_per_step'), '.1f')} | {_fmt(r.get('peak_gb'), '.2f')} | "
                f"{_fmt(e, '.6f')} | {_fmt(d_ref, '+.2e')} | "
                f"{'Y' if r.get('converged') else 'N'} |"
            )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_showcase_scaling.py -k "markdown or csv" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/showcase_heisenberg_scaling.py tests/test_showcase_scaling.py
git commit -m "feat(showcase): markdown table + CSV aggregation of cell results"
```

---

### Task 5: Register the test file as `core` so CI runs it

**Files:**
- Modify: `tests/conftest.py` (the `_FILE_MARKERS` dict, near line 80)

- [ ] **Step 1: Add the entry**

In `tests/conftest.py`, inside the `_FILE_MARKERS` dict, add (alongside the other `core` entries, e.g. right after `"test_ctm_sharding.py": "core",`):

```python
    "test_showcase_scaling.py": "core",
```

- [ ] **Step 2: Verify the file is collected under `-m core`**

Run: `uv run pytest -m core tests/test_showcase_scaling.py -v`
Expected: all helper tests from Tasks 1–4 are collected and PASS (not deselected).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test(showcase): mark test_showcase_scaling as core for CI"
```

---

### Task 6: `run_cell` worker + `--cell` CLI

**Files:**
- Modify: `examples/showcase_heisenberg_scaling.py`

> No unit test here — `run_cell` drives the full GPU/CPU library path and is validated by the Phase-0 gate (Task 9) and the real run. Keep it small and faithful.

- [ ] **Step 1: Implement `run_cell`**

Append to `examples/showcase_heisenberg_scaling.py`:

```python
def _peak_gb():
    import jax
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return None


def run_cell(D, chi, n_devices, gs_num_steps):
    """Run ONE cell and return a result dict. One faithful entry point:
    optimize_gs_ad(return_history=True). ms_per_step = median of warm
    step_times (compile excluded). Anchor cells (large gs_num_steps) yield a
    trusted converged E_site."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices, "gs_num_steps": gs_num_steps,
        "is_anchor": gs_num_steps >= 40,
        "ms_per_step": None, "peak_gb": None, "E_site": None,
        "converged": False, "jit_compile_time": None, "oom": False, "error": None,
    }
    try:
        import jax  # noqa: F401  (import after CUDA_VISIBLE_DEVICES is set)
        from tenax.algorithms.ipeps import heisenberg_gate, sublattice_rotate_gate
        from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
        from tenax.algorithms.ipeps_optimize import optimize_gs_ad

        mesh = None
        if n_devices > 1:
            from tenax.algorithms.ctm_sharding import build_ctm_mesh
            mesh = build_ctm_mesh()

        gate = sublattice_rotate_gate(heisenberg_gate())
        config = iPEPSConfig(
            max_bond_dim=D,
            ctm=CTMConfig(
                chi=chi, max_iter=100, conv_tol=1e-8,
                projector_method="svd", forward_gauge="sigma",
                device_mesh=mesh,
            ),
            unit_cell="1x1",
            gs_recipe="1x1",
            gs_optimizer="lbfgs",
            gs_implicit_ad=True,
            gs_num_steps=gs_num_steps,
            su_init=True,
            return_history=True,
            gs_verbose=False,
        )
        _, _, E_gs, history = optimize_gs_ad(gate, None, config)

        step_times = history.get("step_times") or []
        warm = step_times[1:] if len(step_times) > 1 else step_times
        if warm:
            result["ms_per_step"] = 1000.0 * statistics.median(warm)
        result["E_site"] = float(E_gs)
        result["converged"] = bool(history.get("converged"))
        result["jit_compile_time"] = (
            float(history["jit_compile_time"]) if history.get("jit_compile_time") is not None else None
        )
        result["peak_gb"] = _peak_gb()
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = _peak_gb()
    return result
```

- [ ] **Step 2: Add the `--cell` CLI branch and `main()` stub**

Append to `examples/showcase_heisenberg_scaling.py`:

```python
def _run_worker(args):
    res = run_cell(args.D, args.chi, args.n_devices, args.gs_num_steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def _build_argparser():
    p = argparse.ArgumentParser(description="iPEPS Heisenberg scaling showcase")
    p.add_argument("--cell", action="store_true", help="run a single cell (worker mode)")
    p.add_argument("--D", type=int)
    p.add_argument("--chi", type=int)
    p.add_argument("--n-devices", dest="n_devices", type=int, default=1)
    p.add_argument("--gs-num-steps", dest="gs_num_steps", type=int, default=5)
    p.add_argument("--out", type=str)
    p.add_argument("--results-dir", dest="results_dir", type=str,
                   default="examples/showcase_results")
    return p


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)  # defined in Task 7
```

- [ ] **Step 3: Sanity-check the module still imports (helpers unaffected)**

Run: `uv run pytest tests/test_showcase_scaling.py -v`
Expected: all Task 1–4 tests still PASS (importing the module must not import jax; `main` is referenced only under `__main__`).

- [ ] **Step 4: Commit**

```bash
git add examples/showcase_heisenberg_scaling.py
git commit -m "feat(showcase): run_cell worker (optimize_gs_ad) + --cell CLI"
```

---

### Task 7: Orchestrator `main()` (resume + row loop + aggregate)

**Files:**
- Modify: `examples/showcase_heisenberg_scaling.py`

> The subprocess launch is not unit-tested (it shells out to GPUs); its building blocks (`cell_to_argv_env`, `should_stop_row`, `cell_result_path`) are already tested. Keep `main` a thin wiring layer.

- [ ] **Step 1: Implement `main()` and grid defaults**

Append to `examples/showcase_heisenberg_scaling.py` (BEFORE the `if __name__` block — move that block to the end of the file):

```python
import csv          # noqa: E402  (kept local to orchestrator concerns)
import subprocess   # noqa: E402
import sys          # noqa: E402

# Default sweep envelope (override via main args if desired).
DEFAULT_D_LIST = [2, 3, 4, 5]
DEFAULT_CHI_RAMP = [16, 24, 32, 48, 64, 96, 128]
DEFAULT_DEVICE_COUNTS = [1, 4]
DEFAULT_ANCHORS = [(2, 32), (3, 48), (4, 64)]
DEFAULT_METRICS_STEPS = 5
DEFAULT_ANCHOR_STEPS = 80


def _load_or_run_cell(cell, results_dir):
    """Resume: if a result JSON exists, load it; else launch the worker
    subprocess and load what it wrote. Returns the result dict (with is_anchor
    annotated for the reporter)."""
    path = Path(cell_result_path(results_dir, cell))
    if path.exists():
        res = json.loads(path.read_text())
    else:
        argv, env = cell_to_argv_env(
            cell, results_dir=results_dir, python_exe=sys.executable,
            script_path=str(Path(__file__).resolve()), base_env=dict(__import__("os").environ))
        print(f"[run] {argv[-1]}", flush=True)
        subprocess.run(argv, env=env, check=False)
        if not path.exists():
            res = {"D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices,
                   "is_anchor": cell.is_anchor, "oom": False,
                   "error": "worker produced no result file", "ms_per_step": None,
                   "peak_gb": None, "E_site": None, "converged": False}
            path.write_text(json.dumps(res, indent=2))
        else:
            res = json.loads(path.read_text())
    res["is_anchor"] = cell.is_anchor
    return res


def main(args):
    import os
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    anchor_cells = [c for c in build_grid(
        [], [], DEFAULT_DEVICE_COUNTS, DEFAULT_ANCHORS,
        DEFAULT_METRICS_STEPS, DEFAULT_ANCHOR_STEPS) if c.is_anchor]

    results = []
    # Metrics: ramp chi ascending per (n_devices, D) row; stop the row on OOM/err.
    for n in DEFAULT_DEVICE_COUNTS:
        for D in DEFAULT_D_LIST:
            for chi in DEFAULT_CHI_RAMP:
                cell = Cell(D, chi, n, DEFAULT_METRICS_STEPS, is_anchor=False)
                res = _load_or_run_cell(cell, results_dir)
                results.append(res)
                if should_stop_row(res):
                    print(f"[stop] row n={n} D={D} stopped at chi={chi} "
                          f"({_status(res)})", flush=True)
                    break
    # Anchors (specific cells; run regardless of metrics ramp).
    for cell in anchor_cells:
        results.append(_load_or_run_cell(cell, results_dir))

    # Aggregate.
    md = results_to_markdown(results)
    (Path(results_dir) / "scaling_table.md").write_text(md)
    rows = results_to_csv_rows(results)
    with open(Path(results_dir) / "scaling_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    try:
        make_plots(results, results_dir)  # defined in Task 8
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plotting failed: {e}", flush=True)
    print(md)
    print(f"\n[done] wrote {results_dir}/scaling_table.md, scaling_results.csv, *.png")
```

- [ ] **Step 2: Move the `if __name__ == "__main__":` block to the end of the file** (so `main` and `make_plots` are defined before it runs). Confirm the block is the last thing in the file.

- [ ] **Step 3: Confirm helpers still pass and the module imports**

Run: `uv run pytest tests/test_showcase_scaling.py -v`
Expected: PASS (module import must not trigger jax/subprocess; those live under `__main__`/inside functions).

- [ ] **Step 4: Commit**

```bash
git add examples/showcase_heisenberg_scaling.py
git commit -m "feat(showcase): orchestrator main — resume, OOM-aware ramp, aggregate"
```

---

### Task 8: `make_plots` (matplotlib, smoke-tested)

**Files:**
- Modify: `examples/showcase_heisenberg_scaling.py`
- Test: `tests/test_showcase_scaling.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_showcase_scaling.py`:

```python
def test_make_plots_writes_pngs(tmp_path):
    results = [
        {"D": 2, "chi": 16, "n_devices": 1, "is_anchor": False, "oom": False,
         "error": None, "ms_per_step": 10.0, "peak_gb": 1.0, "E_site": None,
         "converged": False},
        {"D": 2, "chi": 32, "n_devices": 1, "is_anchor": False, "oom": False,
         "error": None, "ms_per_step": 22.0, "peak_gb": 2.0, "E_site": None,
         "converged": False},
        {"D": 2, "chi": 32, "n_devices": 4, "is_anchor": False, "oom": False,
         "error": None, "ms_per_step": 30.0, "peak_gb": 0.7, "E_site": None,
         "converged": False},
        {"D": 2, "chi": 32, "n_devices": 4, "is_anchor": True, "oom": False,
         "error": None, "ms_per_step": 40.0, "peak_gb": 0.7, "E_site": -0.6690,
         "converged": True},
    ]
    paths = showcase.make_plots(results, str(tmp_path))
    assert len(paths) >= 1
    for p in paths:
        assert showcase.Path(p).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_showcase_scaling.py -k make_plots -v`
Expected: FAIL — `AttributeError: make_plots`.

- [ ] **Step 3: Write minimal implementation**

Append to `examples/showcase_heisenberg_scaling.py` (before the `__main__` block):

```python
def make_plots(results, outdir):
    """Write the showcase plots. Returns the list of PNG paths written.
    Best-effort: cells without a metric are skipped, not errored."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if not r.get("oom") and not r.get("error")]
    written = []

    # Plot 1: ms/step vs chi per D (single-GPU scaling curves).
    fig, ax = plt.subplots()
    plotted = False
    for D in sorted({r["D"] for r in ok if r["n_devices"] == 1}):
        pts = sorted((r["chi"], r["ms_per_step"]) for r in ok
                     if r["D"] == D and r["n_devices"] == 1 and r.get("ms_per_step") is not None)
        if pts:
            ax.plot(*zip(*pts), marker="o", label=f"D={D}")
            plotted = True
    if plotted:
        ax.set_xlabel("χ"); ax.set_ylabel("ms / optimizer step")
        ax.set_yscale("log"); ax.legend(); ax.set_title("Per-step cost vs χ (1 GPU)")
        p = outdir / "ms_per_step_vs_chi.png"; fig.savefig(p, dpi=120); written.append(str(p))
    plt.close(fig)

    # Plot 2: peak GB vs chi, 1-GPU vs 4-GPU overlay (largest D with both present).
    fig, ax = plt.subplots()
    plotted = False
    for n in sorted({r["n_devices"] for r in ok}):
        for D in sorted({r["D"] for r in ok if r["n_devices"] == n}):
            pts = sorted((r["chi"], r["peak_gb"]) for r in ok
                         if r["D"] == D and r["n_devices"] == n and r.get("peak_gb") is not None)
            if pts:
                ax.plot(*zip(*pts), marker="s", label=f"D={D}, {n}-GPU")
                plotted = True
    if plotted:
        ax.set_xlabel("χ"); ax.set_ylabel("per-device peak GB")
        ax.legend(fontsize="small"); ax.set_title("Peak memory: 1-GPU vs 4-GPU")
        p = outdir / "peak_gb_vs_chi.png"; fig.savefig(p, dpi=120); written.append(str(p))
    plt.close(fig)

    # Plot 3: anchor E/site vs chi with the QMC reference line.
    anchors = [r for r in ok if r.get("is_anchor") and r.get("E_site") is not None]
    if anchors:
        fig, ax = plt.subplots()
        for D in sorted({r["D"] for r in anchors}):
            pts = sorted((r["chi"], r["E_site"]) for r in anchors if r["D"] == D)
            ax.plot(*zip(*pts), marker="o", label=f"D={D}")
        ax.axhline(REFERENCE_E, ls="--", color="k", label=f"QMC {REFERENCE_E}")
        ax.set_xlabel("χ"); ax.set_ylabel("E / site"); ax.legend()
        ax.set_title("Anchor energies vs QMC reference")
        p = outdir / "energy_vs_chi.png"; fig.savefig(p, dpi=120); written.append(str(p))
        plt.close(fig)

    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_showcase_scaling.py -k make_plots -v`
Expected: PASS (writes `ms_per_step_vs_chi.png`, `peak_gb_vs_chi.png`).

> If matplotlib is not installed in the env, first run `uv add --dev matplotlib` (or confirm it is already a dev dependency) before this task; the smoke test requires it.

- [ ] **Step 5: Run the FULL helper suite + ruff**

Run: `uv run pytest tests/test_showcase_scaling.py -v && uv run ruff check examples/showcase_heisenberg_scaling.py`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add examples/showcase_heisenberg_scaling.py tests/test_showcase_scaling.py
git commit -m "feat(showcase): scaling plots (ms/step, peak GB, energy) + smoke test"
```

---

### Task 9: Phase-0 compatibility gate (run on the 4×A100 box)

**Files:** none (execution + decision). This GATES the full run.

- [ ] **Step 1: Tiny single-GPU smoke (sanity)**

Run:
```bash
cd /home/yjkao/tenax
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/showcase_heisenberg_scaling.py --cell \
  --D 2 --chi 8 --n-devices 1 --gs-num-steps 5 --out /tmp/gate_n1.json
cat /tmp/gate_n1.json
```
Expected: `error` is null, `oom` false, `ms_per_step` is a finite number, `E_site` finite.

- [ ] **Step 2: The real gate — 4-GPU + recipe=1x1 + device_mesh + implicit AD**

Run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/showcase_heisenberg_scaling.py --cell \
  --D 2 --chi 8 --n-devices 4 --gs-num-steps 5 --out /tmp/gate_n4.json
cat /tmp/gate_n4.json
```
Expected (GO): `error` null, `oom` false, finite `ms_per_step`, finite `E_site` close to the n=1 value (same physics, different device layout).

- [ ] **Step 3: Decide and record**

- **If both succeed (GO):** `recipe="1x1"` composes with `device_mesh`. Proceed to Task 10 unchanged.
- **If Step 2 fails (NO-GO):** the four-way combo does not compose. Apply the documented fallback in `run_cell`: when `n_devices > 1`, set `gs_recipe="2x2"` (the sharding-validated path) instead of `"1x1"`; keep `"1x1"` for `n_devices == 1`. Make this a one-line branch:

```python
            gs_recipe="2x2" if n_devices > 1 else "1x1",
```

  Commit the fallback:
```bash
git add examples/showcase_heisenberg_scaling.py
git commit -m "fix(showcase): recipe=2x2 fallback for multi-GPU (1x1+device_mesh NO-GO)"
```
  Record the composition limit verbatim (the error string) for the findings doc.

- [ ] **Step 4: Capture the gate outcome** (paste both JSONs into the findings draft for Task 10).

---

### Task 10: Run the showcase + write the findings doc

**Files:**
- Create: `docs/superpowers/handoffs/2026-06-23-heisenberg-scaling-showcase-findings.md`

- [ ] **Step 1: Launch the full sweep (long-running; resumable)**

Run (from repo root; the orchestrator sets per-cell `CUDA_VISIBLE_DEVICES` itself, so do NOT pin it here — but DO leave the display GPU out of the machine-level visible set if the box enumerates it):
```bash
cd /home/yjkao/tenax
uv run python examples/showcase_heisenberg_scaling.py --results-dir examples/showcase_results
```
The run launches one subprocess per cell, writes `examples/showcase_results/*.json`, and on completion writes `scaling_table.md`, `scaling_results.csv`, and the PNGs. If interrupted, re-run the same command — existing per-cell JSONs are skipped (resume).

- [ ] **Step 2: Sanity-check the physics at anchors**

Confirm the anchor `E_site` values trend toward `REFERENCE_E = -0.669437` as D and χ grow. If the magnitude is off by a constant factor (per-bond vs per-site convention), cross-check against `examples/heisenberg_ipeps_ad.py`'s printed energy for the same gate and note the convention in the findings — do NOT silently rescale.

- [ ] **Step 3: Write the findings doc**

Create `docs/superpowers/handoffs/2026-06-23-heisenberg-scaling-showcase-findings.md` containing:
- The Phase-0 gate outcome (GO, or NO-GO + the verbatim composition error + the fallback used).
- The `scaling_table.md` content (paste).
- Embedded plot references (`ms_per_step_vs_chi.png`, `peak_gb_vs_chi.png`, `energy_vs_chi.png`).
- The **observed ceilings**: largest χ per D before OOM on 1-GPU vs 4-GPU, and the χ where 1-GPU OOMs but 4-GPU survives (the N^(1/6) demonstration).
- The empirical ms/step scaling exponent in χ per D (fit `ms ∝ χ^p`).
- An honest verdict: does the production stack make large-D/large-χ tractable, and where is the next wall (e.g. the projector SVD at large χ, per `chunked-einsum-ctm-lever`)?

- [ ] **Step 4: Commit + open PR**

```bash
git add docs/superpowers/handoffs/2026-06-23-heisenberg-scaling-showcase-findings.md
git commit -m "docs(showcase): Heisenberg scaling/perf findings (D,chi sweep, 1 vs 4 GPU)"
git push -u origin showcase/heisenberg-scaling
gh pr create --title "Heisenberg iPEPS scaling/perf showcase (recipe=1x1 + multi-GPU)" \
  --body "$(cat <<'EOF'
Production-merged-levers scaling showcase for the 2D square-lattice Heisenberg
iPEPS ground state: wide (D, χ) sweep with per-step timing + per-device peak
memory, explicit 1-GPU vs 4-GPU comparison. One-cell-per-process (peak-memory
trap). Phase-0 gate result + findings in docs/superpowers/handoffs/.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Update memory**

Add/extend a memory file recording the showcase outcome (ceilings, ms/step exponents, Phase-0 GO/NO-GO) and link it from `MEMORY.md`, cross-referencing `[[chunked-einsum-ctm-lever]]` and `[[632-multigpu-dense-ctm-measured]]`.

---

## Self-Review

**Spec coverage:**
- One-cell-per-process / peak trap → Tasks 3, 6, 7 (env flags, subprocess). ✓
- Phase-0 compatibility gate + fallback → Task 9. ✓
- Wide (D,χ) grid, step-level metrics → Tasks 1, 6, 7 (`build_grid`, `run_cell`, ramp). ✓
- Explicit 1-GPU vs 4-GPU → `DEFAULT_DEVICE_COUNTS=[1,4]`, device pinning (Task 3), overlay plot (Task 8). ✓
- Anchors → converged energy → Task 1 (anchor cells), Task 6 (large `gs_num_steps`), Task 10 (physics check). ✓
- recipe=1x1 + device_mesh + sigma gauge + implicit AD + sublattice rotation, C4v off → Task 6 config. ✓
- OOM-aware ramp + resume → Tasks 2, 7. ✓
- Deliverables (table, CSV, plots, findings) → Tasks 7, 8, 10. ✓
- Reference −0.669437 → `REFERENCE_E` (Task 1), table dE_ref (Task 4), plot line (Task 8). ✓
- Tests are `core` → Task 5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; the only deferred reference is `make_plots`/`main` used in earlier tasks but defined in Tasks 7/8 — flagged inline and the `__main__` block is moved to file end in Task 7.

**Type consistency:** `Cell` fields (`D, chi, n_devices, gs_num_steps, is_anchor`) are used consistently across `build_grid`/`cell_result_path`/`cell_to_argv_env`/`_load_or_run_cell`. Result dict keys (`ms_per_step, peak_gb, E_site, converged, oom, error, is_anchor, D, chi, n_devices`) are written by `run_cell` (Task 6) and read by `results_to_markdown`/`results_to_csv_rows` (Task 4) and `make_plots` (Task 8) — names match. `should_stop_row` reads `oom`/`error` which `run_cell` always sets.
