# D=4 Heisenberg AFM χ-scaling benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `examples/heisenberg_d4_chi_scaling.py`, a single driver that measures both ground-state energy convergence (E/site vs χ) and multi-GPU performance (ms/sweep, peak memory, speedup) for the D=4 spin-1/2 square-lattice Heisenberg AFM via iPEPS-AD.

**Architecture:** Orchestrator + worker in one file (JAX imported only inside the worker, after the parent pins `CUDA_VISIBLE_DEVICES`). Optimize the variational state **once** at moderate χ_opt (the validated d3 implicit-AD + C4v + grad-spike recipe), cache it, then run that **fixed** state through a forward-CTM scan at each χ for each `n_devices ∈ {1,2,4}`. The E column (device-independent) gives the convergence curve; the time/memory columns at identical (state, χ) give the speedup curve. Subprocess-per-cell with resume, per-cell timeout, and a device-safety guard that refuses to run on the 4 GB display GPU.

**Tech Stack:** Python 3.11+, JAX (x64), tenax iPEPS-AD + GSPMD CTM sharding (`build_ctm_mesh`), 4× A100-80GB.

**Spec:** `docs/superpowers/specs/2026-06-27-heisenberg-d4-chi-scaling-design.md`

---

## File structure

- **Create** `examples/heisenberg_d4_chi_scaling.py` — orchestrator + worker. Top-level body imports only stdlib (`argparse`, `csv`, `json`, `os`, `pickle`, `subprocess`, `sys`, `time`, `dataclasses`, `pathlib`); `jax`/`tenax` are imported *inside* worker functions. Mirrors the structure of `examples/showcase_heisenberg_scaling.py`.
- **Create** `tests/test_heisenberg_d4_chi_scaling.py` — jax-free unit tests for the pure helpers, path-loaded (mirrors `tests/test_showcase_scaling.py`).
- **Modify** `tests/conftest.py` — register the new test file as a `core` test.

Hardware fact baked into the driver: this box's A100-80GB GPUs are at CUDA/PCI indices **0, 1, 2, 4**; index **3 is a 4 GB DGX Display** GPU and must never be used. (The existing showcase's `range(n_devices)` would wrongly select index 3 for a 4-GPU run.)

---

### Task 1: Pure-helper scaffold — `cuda_visible_for` + A100 index map

**Files:**
- Create: `examples/heisenberg_d4_chi_scaling.py`
- Create: `tests/test_heisenberg_d4_chi_scaling.py`
- Modify: `tests/conftest.py` (add to `_FILE_MARKERS`)

- [ ] **Step 1: Write the failing test**

`tests/test_heisenberg_d4_chi_scaling.py`:
```python
"""Unit tests for the pure orchestration helpers of the D=4 χ-scaling driver.
The example file is path-loaded (it is not an importable package) so these
tests stay jax-free and fast."""

import importlib.util
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples"
    / "heisenberg_d4_chi_scaling.py"
)
_spec = importlib.util.spec_from_file_location("heisenberg_d4_chi_scaling", _PATH)
d4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d4)


def test_cuda_visible_for_maps_to_real_a100_indices():
    assert d4.cuda_visible_for(1) == "0"
    assert d4.cuda_visible_for(2) == "0,1"
    assert d4.cuda_visible_for(4) == "0,1,2,4"


def test_cuda_visible_for_never_emits_the_display_gpu():
    # Index 3 is the 4 GB DGX Display GPU — it must never appear.
    for n in (1, 2, 3, 4):
        assert "3" not in d4.cuda_visible_for(n).split(",")


def test_cuda_visible_for_rejects_out_of_range():
    with pytest.raises(ValueError):
        d4.cuda_visible_for(0)
    with pytest.raises(ValueError):
        d4.cuda_visible_for(5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: FAIL — `FileNotFoundError` / module load error (the example file does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `examples/heisenberg_d4_chi_scaling.py` with the module docstring and the first helper. NB: deliberately **no** `from __future__ import annotations` — same reason as the showcase (the path-loader test triggers a `@dataclass` crash on stringized annotations).

```python
"""iPEPS D=4 square-lattice Heisenberg AFM: χ-convergence + multi-GPU performance.

Orchestrator + per-cell worker in one file.

    # full sweep (orchestrator): optimize once, then scan χ × {1,2,4} GPU
    uv run python examples/heisenberg_d4_chi_scaling.py --outdir runs/d4_chi_scaling

    # quick validation (tiny D=4 run end-to-end)
    uv run python examples/heisenberg_d4_chi_scaling.py --smoke

    # single cell (worker; normally invoked by the orchestrator):
    CUDA_VISIBLE_DEVICES=0 uv run python examples/heisenberg_d4_chi_scaling.py \
        --cell --phase scan --chi 32 --n-devices 1 --outdir runs/d4_chi_scaling \
        --out /tmp/cell.json

Pure helpers import only stdlib; jax/tenax imports live inside the worker so the
parent's CUDA_VISIBLE_DEVICES takes effect before the child initialises a JAX
backend, and so the helper unit tests stay fast and jax-free.
"""

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REFERENCE_E = -0.669437  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM

# This box: A100-SXM4-80GB at CUDA/PCI indices 0,1,2,4. Index 3 is a 4 GB DGX
# Display GPU and must never be selected. Drivers set CUDA_DEVICE_ORDER=PCI_BUS_ID
# so these indices match nvidia-smi deterministically.
A100_INDICES = [0, 1, 2, 4]


def cuda_visible_for(n_devices):
    """CUDA_VISIBLE_DEVICES string for an n-GPU run: the first n A100 indices.

    Never emits the display GPU (index 3). Raises ValueError if n is out of the
    1..len(A100_INDICES) range."""
    if not 1 <= n_devices <= len(A100_INDICES):
        raise ValueError(
            f"n_devices must be 1..{len(A100_INDICES)}, got {n_devices}"
        )
    return ",".join(str(i) for i in A100_INDICES[:n_devices])
```

- [ ] **Step 4: Register the test file as `core` in `tests/conftest.py`**

In `tests/conftest.py`, find the line:
```python
    "test_showcase_scaling.py": "core",
```
and add immediately after it:
```python
    "test_heisenberg_d4_chi_scaling.py": "core",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py tests/test_heisenberg_d4_chi_scaling.py tests/conftest.py
git commit -m "feat(examples): D=4 χ-scaling driver scaffold + GPU index map

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Grid + resume helpers — `Cell`, `build_grid`, `cell_result_path`, `should_stop_row`

**Files:**
- Modify: `examples/heisenberg_d4_chi_scaling.py`
- Test: `tests/test_heisenberg_d4_chi_scaling.py`

- [ ] **Step 1: Write the failing tests** (append to the test file)

```python
def test_build_grid_enumerates_cells_in_device_then_chi_order():
    cells = d4.build_grid(chi_ladder=[16, 32], device_counts=[1, 2, 4])
    assert len(cells) == 6
    # device-major, chi-minor ordering:
    assert (cells[0].n_devices, cells[0].chi) == (1, 16)
    assert (cells[1].n_devices, cells[1].chi) == (1, 32)
    assert (cells[2].n_devices, cells[2].chi) == (2, 16)
    assert all(c.D == 4 for c in cells)


def test_cell_result_path_is_unique_per_chi_and_device():
    a = d4.Cell(D=4, chi=32, n_devices=1)
    b = d4.Cell(D=4, chi=32, n_devices=4)
    pa = d4.cell_result_path("runs/x", a)
    pb = d4.cell_result_path("runs/x", b)
    assert pa != pb
    assert pa.endswith("D4_chi32_n1.json")
    assert pb.endswith("D4_chi32_n4.json")


def test_should_stop_row_on_oom_or_error():
    assert d4.should_stop_row({"oom": True}) is True
    assert d4.should_stop_row({"error": "timeout after 600s"}) is True
    assert d4.should_stop_row({"oom": False, "error": None}) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_grid'`.

- [ ] **Step 3: Implement** (append to the example file, after `cuda_visible_for`)

```python
D = 4  # fixed bond dimension for this driver


@dataclass(frozen=True)
class Cell:
    """One scan cell: the fixed D=4 state contracted at (chi, n_devices)."""

    D: int
    chi: int
    n_devices: int


def build_grid(chi_ladder, device_counts):
    """Scan cells in device-major, chi-minor order (one row per n_devices)."""
    return [
        Cell(D=D, chi=chi, n_devices=n)
        for n in device_counts
        for chi in chi_ladder
    ]


def cell_result_path(results_dir, cell):
    """Per-cell JSON path, unique per (D, chi, n_devices)."""
    return str(
        Path(results_dir) / f"D{cell.D}_chi{cell.chi}_n{cell.n_devices}.json"
    )


def should_stop_row(result):
    """Stop ramping χ for a given n_devices row once a cell OOMs or errors
    (CTM cost is monotone in χ)."""
    return bool(result.get("oom") or result.get("error"))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py tests/test_heisenberg_d4_chi_scaling.py
git commit -m "feat(examples): D=4 driver grid + resume helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Reporting — convergence table, performance table, CSV rows

**Files:**
- Modify: `examples/heisenberg_d4_chi_scaling.py`
- Test: `tests/test_heisenberg_d4_chi_scaling.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def _sample_results():
    # Two device rows over χ∈{16,32}. E identical across n (device-independent);
    # 1-GPU is the speedup baseline.
    return [
        {"D": 4, "chi": 16, "n_devices": 1, "E_site": -0.6601, "err_vs_qmc": 0.0093,
         "ms_per_sweep": 100.0, "n_sweeps": 30, "peak_gb": 1.0, "converged": True,
         "oom": False, "error": None},
        {"D": 4, "chi": 32, "n_devices": 1, "E_site": -0.6640, "err_vs_qmc": 0.0054,
         "ms_per_sweep": 400.0, "n_sweeps": 40, "peak_gb": 4.0, "converged": True,
         "oom": False, "error": None},
        {"D": 4, "chi": 16, "n_devices": 4, "E_site": -0.6601, "err_vs_qmc": 0.0093,
         "ms_per_sweep": 50.0, "n_sweeps": 30, "peak_gb": 0.3, "converged": True,
         "oom": False, "error": None},
        {"D": 4, "chi": 32, "n_devices": 4, "E_site": -0.6640, "err_vs_qmc": 0.0054,
         "ms_per_sweep": 200.0, "n_sweeps": 40, "peak_gb": 1.1, "converged": True,
         "oom": False, "error": None},
    ]


def test_convergence_table_dedups_by_chi_and_shows_qmc_error():
    md = d4.results_to_convergence_md(_sample_results())
    assert "E/site" in md and "err_vs_QMC" in md
    # one row per distinct χ (device-independent), not one per (χ, n):
    assert md.count("| 16 |") == 1
    assert md.count("| 32 |") == 1
    assert "-0.660100" in md


def test_performance_table_reports_speedup_vs_one_gpu():
    md = d4.results_to_performance_md(_sample_results())
    assert "1-GPU" in md and "4-GPU" in md
    # 4-GPU at χ=16 is 100/50 = 2.00× the 1-GPU baseline:
    assert "2.00" in md


def test_csv_rows_have_stable_keys():
    rows = d4.results_to_csv_rows(_sample_results())
    assert len(rows) == 4
    assert set(rows[0]) == {
        "D", "chi", "n_devices", "E_site", "err_vs_qmc", "ms_per_sweep",
        "n_sweeps", "peak_gb", "converged", "oom", "error",
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: FAIL — `AttributeError: ... 'results_to_convergence_md'`.

- [ ] **Step 3: Implement** (append)

```python
def _status(r):
    if r.get("oom"):
        return "OOM"
    if r.get("error"):
        return "ERR"
    return "ok"


def _fmt(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def _e_by_chi(results):
    """First valid-energy result per χ (E is device-independent), χ-sorted."""
    by_chi = {}
    for r in results:
        if r.get("E_site") is None:
            continue
        by_chi.setdefault(r["chi"], r)
    return [by_chi[c] for c in sorted(by_chi)]


def results_to_convergence_md(results):
    """E/site vs χ on the fixed optimized state (device-independent)."""
    lines = [
        "### Convergence: E/site vs χ (D=4 square-lattice Heisenberg AFM)",
        "",
        f"QMC reference E/site = {REFERENCE_E}",
        "",
        "| χ | E/site | err_vs_QMC | sweeps | conv |",
        "|---|--------|------------|--------|------|",
    ]
    for r in _e_by_chi(results):
        e = r["E_site"]
        lines.append(
            f"| {r['chi']} | {e:.6f} | {e - REFERENCE_E:+.2e} | "
            f"{_fmt(r.get('n_sweeps'), 'd')} | "
            f"{'Y' if r.get('converged') else 'N'} |"
        )
    return "\n".join(lines)


def _ms_baseline(results):
    """ms/sweep of the 1-GPU run, keyed by χ — the speedup denominator."""
    return {
        r["chi"]: r["ms_per_sweep"]
        for r in results
        if r["n_devices"] == 1 and r.get("ms_per_sweep") is not None
    }


def results_to_performance_md(results):
    """Per-sweep CTM cost + peak memory vs χ, grouped by device count, with
    speedup against the 1-GPU baseline at the same χ."""
    base = _ms_baseline(results)
    lines = ["### Performance: per-sweep CTM cost & memory vs χ × n_devices"]
    for n in sorted({r["n_devices"] for r in results}):
        lines += [
            "",
            f"#### {n}-GPU",
            "",
            "| χ | status | ms/sweep | sweeps | peak GB | speedup vs 1-GPU |",
            "|---|--------|----------|--------|---------|------------------|",
        ]
        for r in sorted(
            (r for r in results if r["n_devices"] == n), key=lambda r: r["chi"]
        ):
            ms = r.get("ms_per_sweep")
            b = base.get(r["chi"])
            sp = (b / ms) if (ms and b) else None
            lines.append(
                f"| {r['chi']} | {_status(r)} | {_fmt(ms, '.2f')} | "
                f"{_fmt(r.get('n_sweeps'), 'd')} | {_fmt(r.get('peak_gb'), '.3f')} | "
                f"{_fmt(sp, '.2f')} |"
            )
    return "\n".join(lines)


def results_to_csv_rows(results):
    keys = [
        "D", "chi", "n_devices", "E_site", "err_vs_qmc", "ms_per_sweep",
        "n_sweeps", "peak_gb", "converged", "oom", "error",
    ]
    return [{k: r.get(k) for k in keys} for r in results]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py tests/test_heisenberg_d4_chi_scaling.py
git commit -m "feat(examples): D=4 driver convergence + performance reporting

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Worker safety guard + peak-memory probe

**Files:**
- Modify: `examples/heisenberg_d4_chi_scaling.py`

No unit test (needs a live JAX backend); exercised by the smoke run in Task 9.

- [ ] **Step 1: Implement** (append)

```python
def _peak_gb():
    """Per-device peak GB, or None. Import inside try: this is also called from
    the worker's except handler, where an unguarded `import jax` could re-raise
    a backend-init failure and break the record-and-resume contract."""
    try:
        import jax

        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return None


def _assert_only_a100s():
    """Refuse to run if any visible device is not an 80 GB A100 (e.g. the 4 GB
    DGX Display GPU). Backstops a wrong CUDA_VISIBLE_DEVICES / index-order
    mismatch so a run can never silently land on the display GPU."""
    import jax

    bad = []
    for dev in jax.devices():
        try:
            limit = dev.memory_stats().get("bytes_limit", 0)
        except Exception:
            limit = 0
        kind = getattr(dev, "device_kind", "")
        ok = limit > 40e9 or ("A100" in kind and "Display" not in kind)
        if not ok:
            bad.append(f"{dev} kind={kind!r} bytes_limit={limit}")
    if bad:
        raise RuntimeError(
            "refusing to run: non-A100/display GPU visible: " + "; ".join(bad)
        )
```

- [ ] **Step 2: Quick import sanity (no GPU work)**

Run: `uv run python -c "import importlib.util,pathlib; p=pathlib.Path('examples/heisenberg_d4_chi_scaling.py'); s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('ok', m._peak_gb is not None)"`
Expected: prints `ok True` (module still loads; helpers defined).

- [ ] **Step 3: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py
git commit -m "feat(examples): D=4 driver device-safety guard + peak-mem probe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Worker — `optimize_once` (D=4 d3 recipe + device_mesh)

**Files:**
- Modify: `examples/heisenberg_d4_chi_scaling.py`

Mirrors the validated recipe in `examples/heisenberg_d3_chi_convergence.py` (implicit AD, C4v, `gs_energy_floor`, `gs_grad_spike_ratio`, checkpoint/resume) at `max_bond_dim=4`, with `ctm.device_mesh` set when sharding.

- [ ] **Step 1: Implement** (append)

```python
def _build_mesh(n_devices):
    """An n-device GSPMD mesh, or None for single-device. Asserts the visible
    devices are A100s first."""
    _assert_only_a100s()
    if n_devices <= 1:
        return None
    from tenax.algorithms.ctm_sharding import build_ctm_mesh

    return build_ctm_mesh()  # over all visible devices (== the pinned A100s)


def optimize_once(outdir, chi_opt, opt_steps, n_devices, probe_max_iter=15):
    """Optimize the D=4 state once at χ_opt; cache the optimized tensor to
    `<outdir>/A_opt.pkl`. Resumes from its gs checkpoint; if A_opt.pkl already
    exists, returns immediately."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax import (
        CTMConfig,
        heisenberg_gate,
        iPEPSConfig,
        optimize_gs_ad,
        sublattice_rotate_gate,
    )

    tensor_path = os.path.join(outdir, "A_opt.pkl")
    if os.path.exists(tensor_path):
        print(f"[opt] cached {tensor_path}; skipping optimization", flush=True)
        return tensor_path

    mesh = _build_mesh(n_devices)
    ckpt = os.path.join(outdir, "ckpt_opt", "ckpt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    resume = os.path.exists(os.path.join(ckpt, "ckpt.last.pkl"))
    probe = None if probe_max_iter <= 0 else probe_max_iter

    cfg = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=200,
        dt=0.05,
        ctm=CTMConfig(
            chi=chi_opt,
            max_iter=100,
            conv_tol=1e-8,
            projector_method="svd",   # implicit AD requires svd/qr (not eigh)
            forward_gauge="phase",    # implicit AD requires phase gauge
            probe_max_iter=probe,     # #503 cap on HZ line-search CTM probes
            device_mesh=mesh,         # #632 GSPMD sharding when n_devices > 1
        ),
        unit_cell="1x1",
        gs_c4v=True,                  # removes bond-gauge freedom -> stable backward
        gs_implicit_ad=True,          # variational (true expectation value)
        gs_recipe="1x1",
        gs_optimizer="lbfgs",
        gs_line_search_method="hager_zhang",
        gs_metric_precond=True,
        gs_num_steps=opt_steps,
        gs_conv_criterion="grad_norm",
        gs_energy_floor=REFERENCE_E,  # reject sub-GS CTM-artifact spikes (#298)
        gs_grad_spike_ratio=5.0,      # roll back >5x gradient blowups (#524)
        gs_verbose=True,
        gs_log_interval=1,
        su_init=True,
        gs_checkpoint_path=ckpt,
        gs_checkpoint_every=2,
        gs_resume=resume,
    )
    gate = sublattice_rotate_gate(heisenberg_gate())
    print(
        f"[opt] D=4 optimize at χ={chi_opt} (resume={resume}, {opt_steps} steps, "
        f"n_devices={n_devices})",
        flush=True,
    )
    t0 = time.perf_counter()
    A_opt, _env, E = optimize_gs_ad(gate, None, cfg)
    print(
        f"[opt] done in {time.perf_counter() - t0:.0f}s; in-loop E_best={float(E):.6f}",
        flush=True,
    )
    with open(tensor_path, "wb") as fh:
        pickle.dump(A_opt, fh)
    return tensor_path
```

- [ ] **Step 2: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py
git commit -m "feat(examples): D=4 driver optimize_once (d3 recipe + device_mesh)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Worker — `scan_cell` (forward CTM, timing, energy, ms/sweep)

**Files:**
- Modify: `examples/heisenberg_d4_chi_scaling.py`

Uses the **same** CTM the d3 scan uses (`python_loop_ctm_converge` + `ctm_converge_kwargs`, phase gauge, elementwise convergence), with `device_mesh` set for sharding. Warms once (the process-lifetime jit cache makes the second converge compile-free), then times the converge. The small χ²·D⁴ env is gathered to one device before the energy eval (sidesteps any sharded-energy issue; does not affect the timed CTM region).

- [ ] **Step 1: Implement** (append)

```python
def scan_cell(tensor_path, chi, n_devices):
    """Converge forward CTM at χ on the fixed optimized state; return E/site +
    per-sweep timing + peak memory. Record-and-resume safe."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices,
        "E_site": None, "err_vs_qmc": None, "total_s": None, "n_sweeps": None,
        "ms_per_sweep": None, "peak_gb": None, "converged": False,
        "oom": False, "error": None,
    }
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from tenax import CTMConfig, compute_energy_ctm_tensor, heisenberg_gate, \
            sublattice_rotate_gate
        from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

        mesh = _build_mesh(n_devices)  # also runs the A100-only guard
        with open(tensor_path, "rb") as fh:
            A_opt = pickle.load(fh)
        H = sublattice_rotate_gate(heisenberg_gate())

        cfg = CTMConfig(
            chi=chi, max_iter=200, conv_tol=1e-10,
            projector_method="svd", forward_gauge="phase", device_mesh=mesh,
        )
        kwargs = ctm_converge_kwargs(cfg)  # forwards device_mesh

        # Warm-up: compile the χ-specific @jit step (reused via the process
        # cache), so the timed converge measures pure per-sweep compute.
        warm_envs, _ = python_loop_ctm_converge(
            {(0, 0): A_opt}, SINGLE_SITE_NEIGHBORS, **kwargs
        )
        jax.block_until_ready(warm_envs[(0, 0)])

        t0 = time.perf_counter()
        envs, info = python_loop_ctm_converge(
            {(0, 0): A_opt}, SINGLE_SITE_NEIGHBORS, **kwargs
        )
        jax.block_until_ready(envs[(0, 0)])
        total_s = time.perf_counter() - t0

        env = envs[(0, 0)]
        if mesh is not None:  # gather the tiny env to device 0 for energy eval
            env = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, jax.devices()[0]), env
            )
        E = float(compute_energy_ctm_tensor(A_opt, env, H, 2))
        sweeps = int(info.iterations)

        result.update(
            E_site=E, err_vs_qmc=E - REFERENCE_E, total_s=float(total_s),
            n_sweeps=sweeps, ms_per_sweep=1000.0 * total_s / max(sweeps, 1),
            converged=bool(info.converged), peak_gb=_peak_gb(),
        )
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = _peak_gb()
    return result
```

- [ ] **Step 2: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py
git commit -m "feat(examples): D=4 driver scan_cell (sharded forward CTM + ms/sweep)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Worker dispatch + argparse

**Files:**
- Modify: `examples/heisenberg_d4_chi_scaling.py`

- [ ] **Step 1: Implement** (append)

```python
def _run_worker(args):
    """Worker entry: run one phase, write its result JSON, echo it."""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "optimize":
        try:
            optimize_once(
                args.outdir, args.chi_opt, args.opt_steps, args.n_devices,
                probe_max_iter=args.probe_max_iter,
            )
            res = {"phase": "optimize", "ok": True, "error": None}
        except Exception as e:  # noqa: BLE001
            res = {"phase": "optimize", "ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        tensor_path = os.path.join(args.outdir, "A_opt.pkl")
        res = scan_cell(tensor_path, args.chi, args.n_devices)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def _build_argparser():
    p = argparse.ArgumentParser(description="iPEPS D=4 Heisenberg χ-scaling benchmark")
    p.add_argument("--cell", action="store_true", help="worker mode: run one phase")
    p.add_argument("--phase", choices=["optimize", "scan"], default="scan")
    p.add_argument("--chi", type=int, help="scan χ (worker scan phase)")
    p.add_argument("--n-devices", dest="n_devices", type=int, default=1)
    p.add_argument("--out", type=str, help="worker result JSON path")
    # shared / orchestrator:
    p.add_argument("--outdir", default="runs/d4_chi_scaling")
    p.add_argument("--smoke", action="store_true",
                   help="quick validation: tiny χ_opt, few steps, short scan")
    p.add_argument("--chi-opt", dest="chi_opt", type=int, default=32)
    p.add_argument("--opt-steps", dest="opt_steps", type=int, default=100)
    p.add_argument("--probe-max-iter", dest="probe_max_iter", type=int, default=15)
    p.add_argument("--opt-devices", dest="opt_devices", type=int, default=4,
                   help="GPUs for the one-time optimization")
    p.add_argument("--chi-ladder", dest="chi_ladder", type=str,
                   default="16,24,32,48,64,96,128")
    p.add_argument("--device-counts", dest="device_counts", type=str, default="1,2,4")
    p.add_argument("--cell-timeout-s", dest="cell_timeout_s", type=int, default=1800)
    return p
```

- [ ] **Step 2: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py
git commit -m "feat(examples): D=4 driver worker dispatch + CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Orchestrator `main` + plots + `__main__`

**Files:**
- Modify: `examples/heisenberg_d4_chi_scaling.py`

- [ ] **Step 1: Implement** (append)

```python
def _worker_env(n_devices, base_env):
    """Subprocess env: pin the n A100s deterministically, no XLA preallocation."""
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_for(n_devices)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # CUDA indices == nvidia-smi indices
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # so peak_gb is real
    return env


def _launch(argv, n_devices, timeout_s):
    """Run a worker subprocess; return True if it exited within the timeout."""
    env = _worker_env(n_devices, dict(os.environ))
    print(f"[run] {' '.join(argv[argv.index('--cell'):])}", flush=True)
    try:
        subprocess.run(argv, env=env, check=False, timeout=timeout_s)
        return True
    except subprocess.TimeoutExpired:
        return False


def _optimize_phase(outdir, chi_opt, opt_steps, opt_devices, probe_max_iter):
    """Run the one-time optimization in a subprocess pinned to opt_devices."""
    if os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[opt] A_opt.pkl present; optimization skipped", flush=True)
        return
    out = os.path.join(outdir, "optimize_status.json")
    argv = [
        sys.executable, str(Path(__file__).resolve()), "--cell",
        "--phase", "optimize", "--outdir", outdir,
        "--chi-opt", str(chi_opt), "--opt-steps", str(opt_steps),
        "--probe-max-iter", str(probe_max_iter),
        "--n-devices", str(opt_devices), "--out", out,
    ]
    # Optimization can be long; allow generous wall-clock (resume-safe anyway).
    _launch(argv, opt_devices, timeout_s=None)


def _load_or_run_scan(cell, outdir, timeout_s):
    """Resume: load an existing cell JSON, else launch the scan worker and load
    what it wrote. A timeout/no-file is recorded as an error so the row stops."""
    path = Path(cell_result_path(outdir, cell))
    if path.exists():
        return json.loads(path.read_text())
    argv = [
        sys.executable, str(Path(__file__).resolve()), "--cell",
        "--phase", "scan", "--outdir", outdir, "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices), "--out", str(path),
    ]
    ok = _launch(argv, cell.n_devices, timeout_s)
    if path.exists():
        return json.loads(path.read_text())
    res = {
        "D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices,
        "E_site": None, "err_vs_qmc": None, "ms_per_sweep": None,
        "n_sweeps": None, "peak_gb": None, "converged": False, "oom": False,
        "error": ("timeout" if not ok else "worker produced no result file"),
    }
    path.write_text(json.dumps(res, indent=2))
    return res


def make_plots(results, outdir):
    """Best-effort PNGs: E vs χ (with QMC line), ms/sweep vs χ per n, speedup vs
    χ per n, peak GB vs χ per n. Returns the list of paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    ok = [r for r in results if not r.get("oom") and not r.get("error")]
    written = []

    conv = _e_by_chi(ok)
    if conv:
        fig, ax = plt.subplots()
        ax.plot([r["chi"] for r in conv], [r["E_site"] for r in conv], marker="o")
        ax.axhline(REFERENCE_E, ls="--", color="k", label=f"QMC {REFERENCE_E}")
        ax.set_xlabel("χ"); ax.set_ylabel("E / site")
        ax.set_title("D=4 convergence: E/site vs χ"); ax.legend()
        p = outdir / "convergence_E_vs_chi.png"
        fig.savefig(p, dpi=120); written.append(str(p)); plt.close(fig)

    base = _ms_baseline(ok)
    for metric, ylabel, fname, logy in [
        ("ms_per_sweep", "ms / CTM sweep", "perf_ms_per_sweep_vs_chi.png", True),
        ("peak_gb", "per-device peak GB", "perf_peak_gb_vs_chi.png", False),
    ]:
        fig, ax = plt.subplots(); plotted = False
        for n in sorted({r["n_devices"] for r in ok}):
            pts = sorted((r["chi"], r[metric]) for r in ok
                         if r["n_devices"] == n and r.get(metric) is not None)
            if pts:
                ax.plot(*zip(*pts), marker="o", label=f"{n}-GPU"); plotted = True
        if plotted:
            ax.set_xlabel("χ"); ax.set_ylabel(ylabel)
            if logy:
                ax.set_yscale("log")
            ax.legend(); ax.set_title(f"D=4 {ylabel} vs χ")
            p = outdir / fname
            fig.savefig(p, dpi=120); written.append(str(p))
        plt.close(fig)

    fig, ax = plt.subplots(); plotted = False
    for n in sorted({r["n_devices"] for r in ok if r["n_devices"] > 1}):
        pts = sorted((r["chi"], base[r["chi"]] / r["ms_per_sweep"]) for r in ok
                     if r["n_devices"] == n and r.get("ms_per_sweep")
                     and base.get(r["chi"]))
        if pts:
            ax.plot(*zip(*pts), marker="o", label=f"{n}-GPU"); plotted = True
    if plotted:
        ax.axhline(1.0, ls=":", color="k")
        ax.set_xlabel("χ"); ax.set_ylabel("speedup vs 1-GPU")
        ax.legend(); ax.set_title("D=4 multi-GPU speedup vs χ")
        p = outdir / "perf_speedup_vs_chi.png"
        fig.savefig(p, dpi=120); written.append(str(p))
    plt.close(fig)
    return written


def _aggregate(results, outdir):
    conv_md = results_to_convergence_md(results)
    perf_md = results_to_performance_md(results)
    (Path(outdir) / "convergence.md").write_text(conv_md)
    (Path(outdir) / "performance.md").write_text(perf_md)
    rows = results_to_csv_rows(results)
    if rows:
        with open(Path(outdir) / "results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    try:
        make_plots(results, outdir)
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plotting failed: {e}", flush=True)
    print(conv_md); print(); print(perf_md)
    print(f"\n[done] wrote {outdir}/convergence.md, performance.md, results.csv, *.png")


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    chi_ladder = [int(x) for x in args.chi_ladder.split(",")]
    device_counts = [int(x) for x in args.device_counts.split(",")]

    # Phase 1: optimize once (pinned to opt_devices GPUs).
    _optimize_phase(outdir, args.chi_opt, args.opt_steps, args.opt_devices,
                    args.probe_max_iter)
    if not os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[abort] optimization produced no A_opt.pkl; see "
              f"{outdir}/optimize_status.json", flush=True)
        return

    # Phase 2: scan χ per device row; stop a row on OOM/error/timeout.
    results = []
    for n in device_counts:
        for chi in chi_ladder:
            res = _load_or_run_scan(Cell(D=D, chi=chi, n_devices=n), outdir,
                                    args.cell_timeout_s)
            results.append(res)
            if should_stop_row(res):
                print(f"[stop] n={n} row stopped at χ={chi} ({_status(res)})",
                      flush=True)
                break

    _aggregate(results, outdir)


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.smoke:
        _args.outdir = _args.outdir + "_smoke"
        _args.chi_opt = 8
        _args.opt_steps = 6
        _args.opt_devices = 2
        _args.chi_ladder = "8,12"
        _args.device_counts = "1,2"
        _args.cell_timeout_s = 1200
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)
```

- [ ] **Step 2: Re-run the pure-helper tests (nothing should have broken)**

Run: `uv run pytest tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: 9 passed.

- [ ] **Step 3: Commit**

```bash
git add examples/heisenberg_d4_chi_scaling.py
git commit -m "feat(examples): D=4 driver orchestrator (optimize-once + χ×GPU scan) + plots

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end smoke validation (the three open items)

**Files:** none changed unless a fix is needed.

This task runs the full pipeline tiny and confirms the three spec open items: (1) sharded-env energy eval works (the gather), (2) the CUDA index mapping is correct (the guard passes on real A100s), (3) `A_opt.pkl` round-trips and re-shards across device counts.

- [ ] **Step 1: Run the smoke sweep**

Run: `uv run python examples/heisenberg_d4_chi_scaling.py --smoke`
Expected: completes without an unhandled exception; prints a convergence table (χ∈{8,12}) and a performance table with 1-GPU and 2-GPU rows; writes `runs/d4_chi_scaling_smoke/{convergence.md,performance.md,results.csv,*.png}` and `A_opt.pkl`.

- [ ] **Step 2: Verify the cell JSONs are clean (energy present, no error)**

Run: `cat runs/d4_chi_scaling_smoke/D4_chi8_n1.json runs/d4_chi_scaling_smoke/D4_chi8_n2.json`
Expected: both have a non-null `E_site` near the D=4 ballpark (a tiny-χ_opt smoke energy, not necessarily below QMC), `error: null`, `oom: false`, and a positive `ms_per_sweep`/`n_sweeps`. The `n1` and `n2` `E_site` should agree to ~1e-6 (device-independence cross-check).

- [ ] **Step 3: Confirm sharding actually engaged (perf signal)**

Inspect the two JSONs' `peak_gb`. Expectation: the 2-GPU `peak_gb` is ≤ the 1-GPU `peak_gb` (sharding reduces per-device memory). If they are equal, the bare `python_loop_ctm_converge` path did not commit `A_opt`'s double-layer to the mesh — record this as a finding and, if pursuing the memory story, pre-commit via `tenax.algorithms.ctm_sharding.commit_double_layer` before the converge call. Correctness (E parity) holds regardless; only the memory-win measurement is affected.

- [ ] **Step 4: Run the core test bucket once more (CI parity)**

Run: `uv run pytest -m core tests/test_heisenberg_d4_chi_scaling.py -q`
Expected: 9 passed (confirms the conftest marker registration works).

- [ ] **Step 5: Commit any fix + the smoke evidence note**

```bash
git add -A
git commit -m "test(examples): D=4 driver smoke validation (energy parity + sharding)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(If no fix was needed, skip this commit.)

---

## After implementation

- Full run: `uv run python examples/heisenberg_d4_chi_scaling.py --outdir runs/d4_chi_scaling` (long; resume-safe). Review `convergence.md` (does E plateau as χ grows; how close to QMC −0.66944) and `performance.md` (where does multi-GPU cross 1× — expected modest at D=4 per #632/#638).
- Open a PR per CLAUDE.md: `gh pr create` (branch `feat/ipeps-d4-chi-scaling`), then `gh pr merge <n> --squash --delete-branch --auto`. CI runs `pytest -m core`, which now includes the new test file.
- Update `README.md`/`docs` only if this is promoted from an example to public API (it is an example script, so no `__all__` change needed).

## Self-review notes

- **Spec coverage:** convergence (Tasks 6/8 reporting) ✓; performance ms/sweep+peak+speedup (Tasks 6/8) ✓; A100 indices 0,1,2,4 + guard (Tasks 1/4/8) ✓; optimize-once recipe (Task 5) ✓; subprocess/resume/timeout/stop-row (Tasks 2/8) ✓; device_mesh through both CTM paths (Tasks 5/6) ✓; TDD pure helpers + smoke (Tasks 1–3, 9) ✓; three open items (Task 9) ✓.
- **Type consistency:** `Cell(D, chi, n_devices)`, `cell_result_path`, `build_grid`, `scan_cell`→result dict keys, and the reporting/csv key sets all match across tasks. Result dict keys (`E_site`, `err_vs_qmc`, `ms_per_sweep`, `n_sweeps`, `peak_gb`, `converged`, `oom`, `error`) are identical in `scan_cell`, `_load_or_run_scan` fallback, `results_to_csv_rows`, and the reporting tables.
- **No placeholders:** every code step is complete and runnable.
