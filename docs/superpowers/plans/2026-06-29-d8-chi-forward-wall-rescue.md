# D=8 χ-scaling forward-CTM wall + multi-GPU rescue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a D=8 Heisenberg χ-scaling driver that produces a simple-update single-site seed, then scans forward CTM at large χ across {1,2}-GPU rows so the single-GPU memory wall and the #632 GSPMD multi-GPU rescue both fall out.

**Architecture:** New sibling `examples/heisenberg_d8_chi_scaling.py` (orchestrator + per-cell worker in one file), path-loading and reusing the D-agnostic pure helpers from `heisenberg_d4_chi_scaling.py`. New code: runtime free-A100 pinning, a 2-site-SU→single-site-C4v seed phase, and the single-site sharded forward χ-scan. The merged D=4 driver is not modified.

**Tech Stack:** Python 3.11/3.12, JAX (x64), tenax (`ipeps`, `python_loop_ctm_converge`, `compute_energy_ctm_tensor`, `symmetrize_c4v`, `build_ctm_mesh`), pytest, `uv`.

---

## Background the engineer needs

- **Why a sibling, not a flag on the D=4 file:** `heisenberg_d4_chi_scaling.py` hardcodes `D=4` at module level threaded through many functions and couples its state phase to AD optimize. We reuse its *pure* helpers and keep the merged file untouched (zero regression risk).
- **Why single-site:** the #632 GSPMD sharding (`tenax.algorithms.ctm_sharding`) is wired only into the single-site tensor CTM path. The 2-site `ctm_2site` has no `device_mesh` support, so the rescue must run on the single-site backbone.
- **Why SU gives us a single-site state:** SU is intrinsically 2-site (`ipeps()`). The project convention (`ipeps_optimize.py:977`) is to run 2-site SU and take the **A-sublattice tensor** as the single-site seed — valid because the sublattice-rotated Heisenberg gate makes A and B equivalent. We replicate exactly that, then C4v-symmetrize.
- **Type flow (verified):** `ipeps(gate, None, cfg)` returns `(energy, (A_su, B_su), envs)` with `A_su` a real-float64 `DenseTensor`. `symmetrize_c4v(arr: jax.Array)` takes a raw array (labels assumed `(u,d,l,r,phys)`), so re-wrap via `_wrap_as_dense_tensor(...)`. Both `python_loop_ctm_converge(site_tensors=dict[Coord, Tensor], ...)` and `compute_energy_ctm_tensor(A: Tensor, env, H, d)` consume the `DenseTensor`.
- **GPU rules:** only 80 GB A100s, never the 4 GB DGX Display GPU (index 3). GSPMD needs `n_devices | D²=64`, so device counts ∈ {1,2,4}. The free-A100 set churns during a session, so pin to *currently-idle* A100s at launch.

## Reused pure helpers from the D=4 module (do NOT re-implement)

Path-load `heisenberg_d4_chi_scaling.py` as `d4` and reuse:
`d4.REFERENCE_E`, `d4.Cell`, `d4._atomic_write_bytes`, `d4._atomic_write_text`,
`d4._read_json_or_none`, `d4._status`, `d4._fmt`, `d4._e_by_chi`,
`d4.results_to_convergence_md`, `d4.results_to_performance_md`,
`d4.results_to_csv_rows`, `d4._ms_baseline`, `d4.make_plots`, `d4._peak_gb`,
`d4._assert_only_a100s`, `d4._build_mesh`, `d4.should_stop_row`,
`d4.cell_result_path`, `d4._aggregate`.

New/overridden in the D=8 module (because they touch `D`, the worker `__file__`,
or the new free-A100 pinning): `D=8`, `build_grid`, `select_free_a100s`,
`_parse_nvidia_smi`, `free_a100_indices`, `cuda_visible_for`, `_worker_env`,
`_launch`, `su_seed_once`, `scan_cell`, `_run_worker`, `_build_argparser`,
`_su_phase`, `_load_or_run_scan`, `main`.

## File structure

- Create: `examples/heisenberg_d8_chi_scaling.py` — the driver.
- Create: `tests/test_heisenberg_d8_chi_scaling.py` — jax-free unit tests for the pure helpers.
- Modify: `tests/conftest.py` — map the new test file to the `core` marker.

---

## Task 1: Free-A100 selection (pure, jax-free)

**Files:**
- Create: `examples/heisenberg_d8_chi_scaling.py` (initial module: imports + selection helpers)
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_heisenberg_d8_chi_scaling.py`:

```python
"""Unit tests for the pure orchestration helpers of the D=8 χ-scaling driver.
Path-loaded (the example is not an importable package) so these tests stay
jax-free and fast."""

import importlib.util
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples"
    / "heisenberg_d8_chi_scaling.py"
)
_spec = importlib.util.spec_from_file_location("heisenberg_d8_chi_scaling", _PATH)
d8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d8)


# nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu
# --format=csv,noheader,nounits  (this box: A100s at 0,1,2,4; display at 3)
_SMI = (
    "0, NVIDIA A100-SXM4-80GB, 56, 0\n"
    "1, NVIDIA A100-SXM4-80GB, 20, 0\n"
    "2, NVIDIA A100-SXM4-80GB, 20, 0\n"
    "3, NVIDIA DGX Display, 7, 0\n"
    "4, NVIDIA A100-SXM4-80GB, 4066, 91\n"
)


def test_parse_nvidia_smi_reads_index_name_mem_util():
    rows = d8._parse_nvidia_smi(_SMI)
    assert rows[0] == (0, "NVIDIA A100-SXM4-80GB", 56, 0)
    assert rows[4] == (4, "NVIDIA A100-SXM4-80GB", 4066, 91)
    assert len(rows) == 5


def test_select_free_a100s_picks_most_idle_first():
    rows = d8._parse_nvidia_smi(_SMI)
    # idle A100s are 0 (56), 1 (20), 2 (20); sort by (mem, index) -> 1,2,0
    assert d8.select_free_a100s(rows, 1) == [1]
    assert d8.select_free_a100s(rows, 2) == [1, 2]
    assert d8.select_free_a100s(rows, 3) == [1, 2, 0]


def test_select_free_a100s_never_picks_the_display_gpu():
    rows = d8._parse_nvidia_smi(_SMI)
    assert 3 not in d8.select_free_a100s(rows, 3)


def test_select_free_a100s_excludes_busy_a100():
    rows = d8._parse_nvidia_smi(_SMI)
    # index 4 is busy (4066 MiB, 91%) -> never selected, and only 3 are free
    assert 4 not in d8.select_free_a100s(rows, 3)
    with pytest.raises(RuntimeError):
        d8.select_free_a100s(rows, 4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -v`
Expected: FAIL — `ModuleNotFoundError`/exec error (the example file does not exist yet).

- [ ] **Step 3: Create the driver module with the selection helpers**

Create `examples/heisenberg_d8_chi_scaling.py`:

```python
"""iPEPS D=8 square-lattice Heisenberg AFM: simple-update seed + forward-CTM
χ-scan showing the single-GPU memory wall and the #632 multi-GPU rescue.

    # full run (orchestrator): SU seed once, then scan χ × {1,2} GPU
    uv run python examples/heisenberg_d8_chi_scaling.py --outdir runs/d8_chi_scaling

    # quick validation (tiny end-to-end)
    uv run python examples/heisenberg_d8_chi_scaling.py --smoke

    # single cell (worker; normally invoked by the orchestrator):
    uv run python examples/heisenberg_d8_chi_scaling.py --cell --phase scan \
        --chi 64 --n-devices 1 --outdir runs/d8_chi_scaling --out /tmp/cell.json

Pure helpers import only stdlib; jax/tenax imports live inside the worker so the
parent's CUDA_VISIBLE_DEVICES takes effect before the child initialises a JAX
backend, and so the helper unit tests stay fast and jax-free.

The D-agnostic formatting/plot/IO/mesh helpers are reused from the sibling D=4
driver (path-loaded as ``d4``) so the merged file is not modified.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

# Path-load the D=4 driver to reuse its D-agnostic pure helpers. Its top level
# imports only stdlib (jax/tenax live inside functions), so this stays jax-free.
_D4_PATH = pathlib.Path(__file__).resolve().parent / "heisenberg_d4_chi_scaling.py"
_d4_spec = importlib.util.spec_from_file_location("heisenberg_d4_chi_scaling", _D4_PATH)
d4 = importlib.util.module_from_spec(_d4_spec)
_d4_spec.loader.exec_module(d4)

REFERENCE_E = d4.REFERENCE_E  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM
D = 8  # fixed bond dimension for this driver


def _parse_nvidia_smi(text):
    """Parse `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu
    --format=csv,noheader,nounits` into (index, name, mem_used_mib, util_pct)
    tuples. Lines that don't have all four fields are skipped."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, mem, util = parts[0], parts[1], parts[2], parts[3]
        rows.append((int(idx), name, int(float(mem)), int(float(util))))
    return rows


def select_free_a100s(rows, n, mem_threshold_mib=2048, util_threshold=50):
    """The n most-idle 80 GB A100 indices from parsed nvidia-smi rows.

    Idle = an A100 (never the DGX Display GPU) with memory.used and
    utilization below the thresholds. Sorted by (memory.used, index) so the
    most-idle device comes first; deterministic tiebreak by index. Raises
    RuntimeError if fewer than n idle A100s are available (so a row stops
    rather than landing on a busy or display GPU)."""
    free = [
        r for r in rows
        if "A100" in r[1] and "Display" not in r[1]
        and r[2] <= mem_threshold_mib and r[3] <= util_threshold
    ]
    free.sort(key=lambda r: (r[2], r[0]))
    if len(free) < n:
        raise RuntimeError(
            f"need {n} idle A100s, found {len(free)}: "
            + ", ".join(f"gpu{r[0]}({r[2]}MiB,{r[3]}%)" for r in rows)
        )
    return [r[0] for r in free[:n]]


def free_a100_indices(n, mem_threshold_mib=2048, util_threshold=50):
    """Query nvidia-smi and return the n most-idle A100 indices."""
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return select_free_a100s(_parse_nvidia_smi(out), n, mem_threshold_mib, util_threshold)


def cuda_visible_for(n_devices):
    """CUDA_VISIBLE_DEVICES string pinning the n most-idle A100s right now."""
    return ",".join(str(i) for i in free_a100_indices(n_devices))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(examples): D=8 χ-scaling driver — free-A100 selection helpers"
```

---

## Task 2: D=8 scan grid + conftest marker

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py` (add `build_grid`)
- Modify: `tests/conftest.py` (map the new test file to `core`)
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_heisenberg_d8_chi_scaling.py`:

```python
def test_build_grid_is_device_major_chi_minor_at_D8():
    cells = d8.build_grid(chi_ladder=[64, 96], device_counts=[1, 2])
    assert [(c.D, c.chi, c.n_devices) for c in cells] == [
        (8, 64, 1), (8, 96, 1), (8, 64, 2), (8, 96, 2),
    ]


def test_build_grid_uses_D8():
    cells = d8.build_grid(chi_ladder=[128], device_counts=[1])
    assert cells[0].D == 8
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py::test_build_grid_is_device_major_chi_minor_at_D8 -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_grid'`.

- [ ] **Step 3: Add `build_grid` to the driver**

Append to `examples/heisenberg_d8_chi_scaling.py` (after `cuda_visible_for`):

```python
def build_grid(chi_ladder, device_counts):
    """Scan cells in device-major, chi-minor order (one row per n_devices).
    Reuses the D=4 module's frozen ``Cell`` dataclass with D=8."""
    return [
        d4.Cell(D=D, chi=chi, n_devices=n)
        for n in device_counts
        for chi in chi_ladder
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Map the new test file to the `core` marker**

In `tests/conftest.py`, in the `_FILE_MARKERS` dict, immediately after the line
`    "test_heisenberg_d4_chi_scaling.py": "core",` add:

```python
    "test_heisenberg_d8_chi_scaling.py": "core",
```

- [ ] **Step 6: Verify the marker is applied**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -m core -v`
Expected: PASS — all 6 tests collected under `-m core` (none deselected).

- [ ] **Step 7: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py tests/conftest.py
git commit -m "feat(examples): D=8 scan grid + core test marker"
```

---

## Task 3: Worker plumbing (env, launch, dispatch, argparser)

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_heisenberg_d8_chi_scaling.py`:

```python
def test_worker_env_pins_idle_a100s_and_disables_prealloc(monkeypatch):
    monkeypatch.setattr(d8, "cuda_visible_for", lambda n: "1,2")
    env = d8._worker_env(2, {"PATH": "/usr/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["PATH"] == "/usr/bin"  # base env preserved


def test_argparser_defaults_target_the_wall():
    args = d8._build_argparser().parse_args([])
    assert args.chi_ladder == "64,96,128,160,192,224,256"
    assert args.device_counts == "1,2"
    assert args.outdir == "runs/d8_chi_scaling"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py::test_argparser_defaults_target_the_wall -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_build_argparser'`.

- [ ] **Step 3: Add worker env, launch, dispatch, and argparser**

Append to `examples/heisenberg_d8_chi_scaling.py`:

```python
def _worker_env(n_devices, base_env):
    """Subprocess env: pin the n most-idle A100s, deterministic index order, no
    XLA preallocation so peak_gb is the real high-water mark."""
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_for(n_devices)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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


def _run_worker(args):
    """Worker entry: run one phase, write its result JSON, echo it."""
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "su":
        try:
            su_seed_once(args.outdir, args.chi_su, args.imaginary_steps, args.dt)
            res = {"phase": "su", "ok": True, "error": None}
        except Exception as e:  # noqa: BLE001
            res = {"phase": "su", "ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        tensor_path = os.path.join(args.outdir, "A_opt.pkl")
        res = scan_cell(tensor_path, args.chi, args.n_devices)
    d4._atomic_write_text(args.out, json.dumps(res, indent=2))
    print(json.dumps(res))


def _build_argparser():
    p = argparse.ArgumentParser(description="iPEPS D=8 Heisenberg χ-scaling wall+rescue")
    p.add_argument("--cell", action="store_true", help="worker mode: run one phase")
    p.add_argument("--phase", choices=["su", "scan"], default="scan")
    p.add_argument("--chi", type=int, help="scan χ (worker scan phase)")
    p.add_argument("--n-devices", dest="n_devices", type=int, default=1)
    p.add_argument("--out", type=str, help="worker result JSON path")
    # shared / orchestrator:
    p.add_argument("--outdir", default="runs/d8_chi_scaling")
    p.add_argument("--smoke", action="store_true",
                   help="quick validation: tiny SU seed, short χ ladder")
    p.add_argument("--chi-su", dest="chi_su", type=int, default=24,
                   help="CTM χ for the SU-phase energy eval (kept small/cheap)")
    p.add_argument("--imaginary-steps", dest="imaginary_steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--chi-ladder", dest="chi_ladder", type=str,
                   default="64,96,128,160,192,224,256")
    p.add_argument("--device-counts", dest="device_counts", type=str, default="1,2")
    p.add_argument("--cell-timeout-s", dest="cell_timeout_s", type=int, default=2400)
    return p
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -v`
Expected: PASS (8 tests). (`_worker_env` test monkeypatches `cuda_visible_for`, so no nvidia-smi call.)

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(examples): D=8 worker env/launch/dispatch + argparser"
```

---

## Task 4: Phase 1 — simple-update single-site seed

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- (No unit test — jax-bearing; covered by the smoke run in Task 7.)

- [ ] **Step 1: Add `su_seed_once`**

Append to `examples/heisenberg_d8_chi_scaling.py`:

```python
def su_seed_once(outdir, chi_su, imaginary_steps, dt):
    """Produce the single-site C4v seed via 2-site simple update and cache it to
    `<outdir>/A_opt.pkl`. SU is intrinsically 2-site; we take the A-sublattice
    tensor as the single-site seed (the optimize_gs_ad su_init convention) and
    C4v-symmetrize. No AD. Existence-cached: a present A_opt.pkl returns at once."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax import (
        CTMConfig,
        heisenberg_gate,
        iPEPSConfig,
        sublattice_rotate_gate,
        symmetrize_c4v,
    )
    from tenax.algorithms.ipeps import _wrap_as_dense_tensor, ipeps

    tensor_path = os.path.join(outdir, "A_opt.pkl")
    if os.path.exists(tensor_path):
        print(f"[su] cached {tensor_path}; skipping simple update", flush=True)
        return tensor_path

    os.makedirs(outdir, exist_ok=True)
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=imaginary_steps,
        dt=dt,
        ctm=CTMConfig(
            chi=chi_su, max_iter=50, conv_tol=1e-8,
            projector_method="svd", forward_gauge="phase",
        ),
        unit_cell="2site",  # ipeps() always runs 2-site SU
        su_init=True,
    )
    print(f"[su] D={D} simple update ({imaginary_steps} steps, dt={dt}, "
          f"χ_su={chi_su})", flush=True)
    t0 = time.perf_counter()
    e_su, (A_su, _B_su), _ = ipeps(gate, None, cfg)
    A_seed = _wrap_as_dense_tensor(symmetrize_c4v(A_su.todense()))
    print(f"[su] done in {time.perf_counter() - t0:.0f}s; SU E/site≈{float(e_su):.6f}",
          flush=True)
    A_host = jax.device_get(A_seed)  # numpy leaves -> device-agnostic, picklable
    d4._atomic_write_bytes(
        tensor_path, pickle.dumps(A_host, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return tensor_path
```

- [ ] **Step 2: Smoke-check the seed phase in isolation on a free A100**

Run:
```bash
CUDA_VISIBLE_DEVICES=$(uv run python -c "import importlib.util,pathlib; \
p=pathlib.Path('examples/heisenberg_d8_chi_scaling.py'); \
s=importlib.util.spec_from_file_location('d8',p); m=importlib.util.module_from_spec(s); \
s.loader.exec_module(m); print(m.cuda_visible_for(1))") \
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/heisenberg_d8_chi_scaling.py --cell --phase su \
  --outdir /tmp/d8_su_smoke --chi-su 8 --imaginary-steps 20 --dt 0.05 \
  --out /tmp/d8_su_smoke/su.json
```
Expected: prints `[su] D=8 simple update ...`, then `[su] done ...; SU E/site≈-0.6...`, and writes `/tmp/d8_su_smoke/A_opt.pkl` + `su.json` with `"ok": true`.

- [ ] **Step 3: Verify the pickled seed is a single-site (8,8,8,8,2) tensor**

Run:
```bash
uv run python -c "import pickle; A=pickle.load(open('/tmp/d8_su_smoke/A_opt.pkl','rb')); \
print(type(A).__name__, A.todense().shape, A.todense().dtype)"
```
Expected: `DenseTensor (8, 8, 8, 8, 2) float64`.

- [ ] **Step 4: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py
git commit -m "feat(examples): D=8 simple-update single-site C4v seed phase"
```

---

## Task 5: Phase 2 — single-site forward χ-scan cell

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- (No unit test — jax-bearing; covered by the smoke run in Task 7.)

- [ ] **Step 1: Add `scan_cell`**

Append to `examples/heisenberg_d8_chi_scaling.py`:

```python
def scan_cell(tensor_path, chi, n_devices):
    """Converge forward CTM at χ on the fixed SU seed; return E/site + per-sweep
    timing + per-device peak memory. The single-site path is the only one with
    #632 device_mesh sharding, so n_devices>1 shards the D²=64 axis. Record-and-
    resume safe: never raises (OOM/errors are recorded)."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices,
        "E_site": None, "err_vs_qmc": None, "total_s": None, "n_sweeps": None,
        "ms_per_sweep": None, "peak_gb": None, "converged": False,
        "oom": False, "error": None,
    }
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from tenax import (
            CTMConfig, compute_energy_ctm_tensor, heisenberg_gate,
            sublattice_rotate_gate,
        )
        from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
        from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
        from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

        mesh = d4._build_mesh(n_devices)  # A100-only guard + GSPMD mesh for n>1
        with open(tensor_path, "rb") as fh:
            A_opt = pickle.load(fh)
        H = sublattice_rotate_gate(heisenberg_gate())

        cfg = CTMConfig(
            chi=chi, max_iter=200, conv_tol=1e-10,
            projector_method="svd", forward_gauge="phase", device_mesh=mesh,
        )
        kwargs = ctm_converge_kwargs(cfg)  # forwards device_mesh; default recipe

        # Warm-up: compile the χ-specific @jit step (process-cached) so the timed
        # converge measures pure per-sweep compute.
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
            converged=bool(info.converged), peak_gb=d4._peak_gb(),
        )
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = d4._peak_gb()
    return result
```

- [ ] **Step 2: Smoke-check a single scan cell against the seed from Task 4**

Run (reuses `/tmp/d8_su_smoke/A_opt.pkl`):
```bash
CUDA_VISIBLE_DEVICES=$(uv run python -c "import importlib.util,pathlib; \
p=pathlib.Path('examples/heisenberg_d8_chi_scaling.py'); \
s=importlib.util.spec_from_file_location('d8',p); m=importlib.util.module_from_spec(s); \
s.loader.exec_module(m); print(m.cuda_visible_for(1))") \
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/heisenberg_d8_chi_scaling.py --cell --phase scan \
  --chi 16 --n-devices 1 --outdir /tmp/d8_su_smoke --out /tmp/d8_su_smoke/cell.json
cat /tmp/d8_su_smoke/cell.json
```
Expected: JSON with a finite `"E_site"` near the Heisenberg AFM range (≈ -0.5 to -0.67 at tiny χ), `"oom": false`, `"error": null`, a numeric `"ms_per_sweep"` and `"peak_gb"`.

- [ ] **Step 3: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py
git commit -m "feat(examples): D=8 single-site forward χ-scan cell (sharded)"
```

---

## Task 6: Orchestration (SU phase, scan loop, aggregation, smoke)

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_heisenberg_d8_chi_scaling.py`:

```python
def test_smoke_args_shrink_the_run():
    args = d8._build_argparser().parse_args(["--smoke"])
    d8._apply_smoke(args)
    assert args.outdir.endswith("_smoke")
    assert args.chi_ladder == "8,12"
    assert args.device_counts == "1"
    assert args.imaginary_steps <= 20


def test_load_or_run_scan_returns_cached_cell(tmp_path):
    cell = d4.Cell(D=8, chi=64, n_devices=1)
    path = d4.cell_result_path(str(tmp_path), cell)
    d4._atomic_write_text(path, '{"D": 8, "chi": 64, "n_devices": 1, "oom": false}')
    # cached file present -> no subprocess launched, returns the parsed dict
    res = d8._load_or_run_scan(cell, str(tmp_path), timeout_s=1)
    assert res["chi"] == 64 and res["oom"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py::test_smoke_args_shrink_the_run -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_apply_smoke'`.

- [ ] **Step 3: Add orchestration + smoke**

Append to `examples/heisenberg_d8_chi_scaling.py`:

```python
def _su_phase(outdir, chi_su, imaginary_steps, dt):
    """Run the one-time SU seed in a subprocess pinned to one idle A100."""
    if os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[su] A_opt.pkl present; simple update skipped", flush=True)
        return
    out = os.path.join(outdir, "su_status.json")
    argv = [
        sys.executable, str(pathlib.Path(__file__).resolve()), "--cell",
        "--phase", "su", "--outdir", outdir, "--chi-su", str(chi_su),
        "--imaginary-steps", str(imaginary_steps), "--dt", str(dt),
        "--n-devices", "1", "--out", out,
    ]
    _launch(argv, n_devices=1, timeout_s=None)  # resume-safe; allow long wall


def _load_or_run_scan(cell, outdir, timeout_s):
    """Resume: load an existing cell JSON, else launch the scan worker and load
    what it wrote. A timeout/no-file is recorded as an error so the row stops."""
    path = pathlib.Path(d4.cell_result_path(outdir, cell))
    cached = d4._read_json_or_none(path) if path.exists() else None
    if cached is not None:
        return cached
    argv = [
        sys.executable, str(pathlib.Path(__file__).resolve()), "--cell",
        "--phase", "scan", "--outdir", outdir, "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices), "--out", str(path),
    ]
    ok = _launch(argv, cell.n_devices, timeout_s)
    loaded = d4._read_json_or_none(path)
    if loaded is not None:
        return loaded
    res = {
        "D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices,
        "E_site": None, "err_vs_qmc": None, "ms_per_sweep": None,
        "n_sweeps": None, "peak_gb": None, "converged": False, "oom": False,
        "error": ("timeout" if not ok else "worker produced no result file"),
    }
    d4._atomic_write_text(str(path), json.dumps(res, indent=2))
    return res


def _apply_smoke(args):
    """Shrink an args namespace to a fast end-to-end validation run."""
    args.outdir = args.outdir + "_smoke"
    args.chi_su = 8
    args.imaginary_steps = 20
    args.chi_ladder = "8,12"
    args.device_counts = "1"
    args.cell_timeout_s = 1200


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    chi_ladder = [int(x) for x in args.chi_ladder.split(",")]
    device_counts = [int(x) for x in args.device_counts.split(",")]

    # Phase 1: simple-update seed once (single idle A100).
    _su_phase(outdir, args.chi_su, args.imaginary_steps, args.dt)
    if not os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[abort] SU produced no A_opt.pkl; see "
              f"{outdir}/su_status.json", flush=True)
        return

    # Phase 2: scan χ per device row; stop a row on OOM/error/timeout.
    results = []
    for n in device_counts:
        for chi in chi_ladder:
            res = _load_or_run_scan(d4.Cell(D=D, chi=chi, n_devices=n), outdir,
                                    args.cell_timeout_s)
            results.append(res)
            if d4.should_stop_row(res):
                print(f"[stop] n={n} row stopped at χ={chi} "
                      f"({d4._status(res)})", flush=True)
                break

    d4._aggregate(results, outdir)


if __name__ == "__main__":
    _args = _build_argparser().parse_args()
    if _args.smoke:
        _apply_smoke(_args)
    if _args.cell:
        _run_worker(_args)
    else:
        main(_args)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(examples): D=8 orchestration (SU phase, scan loop, aggregate, smoke)"
```

---

## Task 7: End-to-end smoke validation on a free A100

**Files:** none (validation only)

- [ ] **Step 1: Run the full smoke orchestrator**

Run:
```bash
uv run python examples/heisenberg_d8_chi_scaling.py --smoke
```
Expected: SU seed runs once (single A100), then a `{1}`-GPU scan over χ=8,12; prints `convergence` + `performance` tables; writes `runs/d8_chi_scaling_smoke/` with `A_opt.pkl`, `convergence.md`, `performance.md`, `results.csv`, and PNGs. No tracebacks; cells show finite `E_site`, `oom=false`.

- [ ] **Step 2: Verify the smoke artifacts**

Run:
```bash
ls runs/d8_chi_scaling_smoke/ && echo "---" && cat runs/d8_chi_scaling_smoke/results.csv
```
Expected: `A_opt.pkl convergence.md performance.md results.csv` + `*.png`; CSV rows for D=8, χ∈{8,12}, n_devices=1 with numeric `E_site`, `ms_per_sweep`, `peak_gb`.

- [ ] **Step 3: Run the full core test suite to confirm no regressions**

Run: `uv run pytest -m core -q`
Expected: PASS (existing suite + the 11 new D=8 helper tests; no failures).

- [ ] **Step 4: Commit any smoke-fixups (if Steps 1–3 required code changes)**

```bash
git add -A && git commit -m "fix(examples): D=8 driver smoke-validation fixups"
```
(Skip if no changes were needed.)

---

## Task 8: Launch the production {1,2}-GPU run

**Files:** none (launch only). Per the design, the 4-GPU rescue row is deferred until four A100s are simultaneously idle.

- [ ] **Step 1: Confirm at least two idle A100s right now**

Run: `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits`
Expected: at least two A100 rows with low memory.used (≲2 GB) and low utilization (the driver will pin the two most-idle).

- [ ] **Step 2: Background-launch the real run**

Run (background; the orchestrator is resume-safe, so a disconnect/kill can be re-launched and it picks up from the per-cell JSONs):
```bash
nohup uv run python examples/heisenberg_d8_chi_scaling.py \
  --outdir runs/d8_chi_scaling > runs/d8_chi_scaling.log 2>&1 &
echo "launched pid $!"
```
Expected: a PID prints; `runs/d8_chi_scaling.log` begins with the `[su]` seed lines.

- [ ] **Step 3: Report status and hand off**

Tail the log and report: the SU seed energy, then per-cell `[run]`/results, and the χ at which the 1-GPU row OOMs versus how far the 2-GPU row reaches (the rescue). Note that the 4-GPU row is deferred. Do NOT commit `runs/` artifacts here — promoting results into `docs/benchmarks/` is a separate follow-up PR (mirroring the D=4 PR #649), out of scope for this plan.

---

## Self-review notes

- **Spec coverage:** sibling driver reusing D=4 pure helpers (Tasks 1–6); free-A100 runtime pinning (Task 1, used in Tasks 3/6); SU 2-site→single-site C4v seed (Task 4); single-site sharded forward χ-scan with OOM record + row-stop (Tasks 5–6); χ ladder `64…256` and device counts `{1,2}` (Task 3 defaults); aggregation/outputs via `d4._aggregate` (Task 6); smoke mode (Task 6) + end-to-end validation (Task 7); jax-free unit tests + conftest `core` marker (Tasks 1–2, 6); launch {1,2}-GPU now, 4-GPU deferred (Task 8). All spec sections map to a task.
- **Placeholders:** none — every code step shows complete code; every run step shows the command and expected output.
- **Type consistency:** `A_su` (DenseTensor) → `symmetrize_c4v(A_su.todense())` (array) → `_wrap_as_dense_tensor(...)` (DenseTensor) → pickled → `python_loop_ctm_converge({(0,0): A})` / `compute_energy_ctm_tensor(A, env, H, 2)` (both consume DenseTensor). `select_free_a100s` returns `list[int]`; `cuda_visible_for` joins them to a CUDA string. `d4.Cell(D, chi, n_devices)` used consistently. Phase names `"su"`/`"scan"` match between `_build_argparser`, `_run_worker`, `_su_phase`, `_load_or_run_scan`.
