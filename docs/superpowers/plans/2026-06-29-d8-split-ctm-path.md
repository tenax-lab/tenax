# D=8 Split-CTM Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--path {dense,split,both}` forward-CTM selector to the D=8 χ-scaling benchmark so one run shows both the dense single-GPU wall (χ≈112) and the split-CTM rescue (χ≈448).

**Architecture:** A small backward-compatible library addition (`ctm_split_tensor(..., return_info=True)`) plus d8-local helpers (`Cell8`, `_cell_path`, path-aware `build_grid`/`scan_cell`/`_aggregate8`) that reuse d4's path-agnostic formatters. The d4 sibling driver is not modified.

**Tech Stack:** Python, JAX, tenax split-CTM (`ctm_split_tensor`, `compute_energy_split_ctm_tensor`), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-06-29-d8-split-ctm-path-design.md`

**Branch:** Work on `feat/d8-chi-forward-wall-rescue` (extends PR #650). A running orchestrator in the main checkout re-reads `examples/heisenberg_d8_chi_scaling.py` per cell — **execute in an isolated git worktree** (`superpowers:using-git-worktrees`) so edits don't corrupt a live worker. tenax is an editable install pointing at the main checkout's `src/`; tests that import tenax must run with `PYTHONPATH=<worktree>/src` so they exercise the worktree's library edit.

---

### Task 1: Library — `ctm_split_tensor(..., return_info=False)`

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_convergence.py`
- Test: `tests/test_split_ctm_tensor.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_split_ctm_tensor.py` inside `class TestSplitCTMTensorConvergence`:

```python
    def test_return_info_reports_iterations_and_converged(self, small_peps_dense):
        """return_info=True returns (env, info) with a sweep count and a bool
        converged flag; the default call still returns a bare env."""
        env_only = ctm_split_tensor(small_peps_dense, chi=8, max_iter=30, chi_I=4)
        assert isinstance(env_only, SplitCTMTensorEnv)  # default unchanged

        env, info = ctm_split_tensor(
            small_peps_dense, chi=8, max_iter=30, chi_I=4, return_info=True
        )
        assert isinstance(env, SplitCTMTensorEnv)
        assert isinstance(info.iterations, int) and info.iterations >= 1
        assert isinstance(info.converged, bool)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src uv run pytest tests/test_split_ctm_tensor.py -k return_info -v`
Expected: FAIL (`ctm_split_tensor() got an unexpected keyword argument 'return_info'`).

- [ ] **Step 3: Implement** — in `_split_ctm_tensor_convergence.py`:

Add a `NamedTuple` near the top (after the imports, before the sweep section), and extend `__all__`:

```python
from typing import NamedTuple


class _SplitCTMInfo(NamedTuple):
    """Convergence info for ctm_split_tensor (mirrors the dense path's info)."""

    iterations: int
    converged: bool
```

Add `"_SplitCTMInfo"` to `__all__`. Then change `ctm_split_tensor`'s signature and loop:

```python
def ctm_split_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    return_info: bool = False,
) -> SplitCTMTensorEnv | tuple[SplitCTMTensorEnv, _SplitCTMInfo]:
```

In the body, track progress and honor `return_info`:

```python
    if chi_I is None:
        chi_I = chi

    env = initialize_split_ctm_tensor_env(A, chi, chi_I)

    prev_sv = None
    converged = False
    iterations = 0
    for i in range(max_iter):
        iterations = i + 1
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)

        _, current_sv, _, _ = tensor_svd(
            env.C1,
            left_labels=[env.C1.labels()[0]],
            right_labels=[env.C1.labels()[1]],
            new_bond_label="_conv_bond",
        )
        if prev_sv is not None:
            sv1 = current_sv / (jnp.sum(current_sv) + 1e-15)
            sv2 = prev_sv / (jnp.sum(prev_sv) + 1e-15)
            min_len = min(len(sv1), len(sv2))
            diff = jnp.max(jnp.abs(sv1[:min_len] - sv2[:min_len]))
            if float(diff) < conv_tol:
                converged = True
                break
        prev_sv = current_sv

    if return_info:
        return env, _SplitCTMInfo(iterations=iterations, converged=converged)
    return env
```

Also add the docstring line for the new arg under `Args:` (after `renormalize`):

```
        return_info: If True, return ``(env, _SplitCTMInfo(iterations, converged))``
                     instead of just ``env``.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD/src uv run pytest tests/test_split_ctm_tensor.py -k "return_info or converges or energy" -v`
Expected: PASS (new test + existing convergence/energy tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_tensor_convergence.py tests/test_split_ctm_tensor.py
git commit -m "feat(split-ctm): optional return_info on ctm_split_tensor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: d8 — `Cell8` + `_cell_path`

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_heisenberg_d8_chi_scaling.py`:

```python
def test_cell8_has_path_and_distinct_paths_dont_collide(tmp_path):
    dense = d8.Cell8(D=8, chi=128, n_devices=1, path="dense")
    split = d8.Cell8(D=8, chi=128, n_devices=1, path="split")
    assert dense.path == "dense" and split.path == "split"
    pd = d8._cell_path(str(tmp_path), dense)
    ps = d8._cell_path(str(tmp_path), split)
    assert pd != ps
    assert pd.endswith("D8_chi128_n1_dense.json")
    assert ps.endswith("D8_chi128_n1_split.json")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k cell8 -v`
Expected: FAIL (`module 'heisenberg_d8_chi_scaling' has no attribute 'Cell8'`).

- [ ] **Step 3: Implement** — in `examples/heisenberg_d8_chi_scaling.py`, add after the `D = 8` line (and add `import dataclasses`/use a NamedTuple — use `typing.NamedTuple` to avoid new imports beyond stdlib):

```python
from typing import NamedTuple


class Cell8(NamedTuple):
    """One D=8 scan cell: the fixed SU seed contracted at (chi, n_devices) via
    the given forward-CTM ``path`` ('dense' or 'split')."""

    D: int
    chi: int
    n_devices: int
    path: str


def _cell_path(outdir, cell):
    """Per-cell JSON path, unique per (chi, n_devices, path) so dense and split
    cells at the same chi never collide (resume-safe)."""
    return os.path.join(
        outdir, f"D{cell.D}_chi{cell.chi}_n{cell.n_devices}_{cell.path}.json"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k cell8 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(#650): Cell8 + path-aware _cell_path for the D=8 driver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: d8 — path-aware `build_grid`

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Replace the existing `build_grid` test** — in `tests/test_heisenberg_d8_chi_scaling.py`, replace `test_build_grid_is_device_major_chi_minor_at_D8` and `test_build_grid_uses_D8` with:

```python
def test_build_grid_both_paths_dense_rows_then_split_n1():
    cells = d8.build_grid(
        chi_ladder=[64, 96], device_counts=[1, 2], paths=["dense", "split"]
    )
    quads = [(c.D, c.chi, c.n_devices, c.path) for c in cells]
    # dense: device-major chi-minor over {1,2}; split: n=1 only, appended last
    assert quads == [
        (8, 64, 1, "dense"), (8, 96, 1, "dense"),
        (8, 64, 2, "dense"), (8, 96, 2, "dense"),
        (8, 64, 1, "split"), (8, 96, 1, "split"),
    ]


def test_build_grid_split_only_ignores_multi_device():
    cells = d8.build_grid(chi_ladder=[128], device_counts=[1, 2], paths=["split"])
    assert [(c.chi, c.n_devices, c.path) for c in cells] == [(128, 1, "split")]
    assert all(c.D == 8 for c in cells)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k build_grid -v`
Expected: FAIL (`build_grid() got an unexpected keyword argument 'paths'`).

- [ ] **Step 3: Implement** — replace `build_grid` in `examples/heisenberg_d8_chi_scaling.py`:

```python
def build_grid(chi_ladder, device_counts, paths):
    """Scan cells as Cell8. Dense rows cross device_counts × chi_ladder (one row
    per n_devices); the split path has no device_mesh, so it emits n=1 rows only
    and is appended after all dense rows. Dense rows come first so the shared χ
    ladder lets each row self-stop at its own wall."""
    cells = []
    if "dense" in paths:
        cells += [
            Cell8(D=D, chi=chi, n_devices=n, path="dense")
            for n in device_counts
            for chi in chi_ladder
        ]
    if "split" in paths:
        cells += [Cell8(D=D, chi=chi, n_devices=1, path="split") for chi in chi_ladder]
    return cells
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k build_grid -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(#650): path-aware build_grid (dense rows + split n=1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: d8 — `--path` flag, default ladder, smoke

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Update the argparse + smoke tests** — replace `test_argparser_defaults_target_the_wall` and `test_smoke_args_shrink_the_run` in `tests/test_heisenberg_d8_chi_scaling.py`:

```python
def test_argparser_defaults_target_the_wall():
    args = d8._build_argparser().parse_args([])
    assert args.chi_ladder == "64,96,112,128,192,256,320,384,448"
    assert args.device_counts == "1,2"
    assert args.path == "both"
    assert args.outdir == "runs/d8_chi_scaling"


def test_argparser_path_choices_validated():
    for p in ("dense", "split", "both"):
        assert d8._build_argparser().parse_args(["--path", p]).path == p
    with pytest.raises(SystemExit):
        d8._build_argparser().parse_args(["--path", "bogus"])


def test_smoke_args_shrink_the_run():
    args = d8._build_argparser().parse_args(["--smoke"])
    d8._apply_smoke(args)
    assert args.outdir.endswith("_smoke")
    assert args.chi_ladder == "8,12"
    assert args.device_counts == "1"
    assert args.path == "both"  # smoke still exercises both paths
    assert args.imaginary_steps <= 20
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k "argparser or smoke" -v`
Expected: FAIL (default `chi_ladder` differs; no `path` attribute).

- [ ] **Step 3: Implement** — in `_build_argparser`, change the default ladder and add `--path`:

```python
    p.add_argument("--chi-ladder", dest="chi_ladder", type=str,
                   default="64,96,112,128,192,256,320,384,448")
    p.add_argument("--device-counts", dest="device_counts", type=str, default="1,2")
    p.add_argument("--path", choices=["dense", "split", "both"], default="both",
                   help="forward-CTM path: dense (chi^2*D^6, wall ~chi=112), "
                        "split (chi^2*D^4, wall ~chi=448), or both (comparison)")
```

In `_apply_smoke`, set the ladder/devices (path already defaults to `both`):

```python
    args.chi_ladder = "8,12"
    args.device_counts = "1"
```

(leave the rest of `_apply_smoke` unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k "argparser or smoke" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(#650): --path flag (dense/split/both) + split-reaching default ladder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: d8 — `scan_cell` path dispatch

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`

(No jax-free unit test — `scan_cell` imports jax; it is verified by the end-to-end smoke in Task 8. Keep the record-and-resume contract: never raise.)

- [ ] **Step 1: Replace `scan_cell`** with a path-dispatching version. Keep the dense branch byte-identical to the current body; add the split branch and the `path` field:

```python
def scan_cell(tensor_path, chi, n_devices, path):
    """Converge the forward CTM at χ on the fixed SU seed via ``path`` ('dense'
    or 'split'); return E/site + per-sweep timing + per-device peak memory.

    dense: closed χ²·D⁶ CTM (python_loop_ctm_converge); the single-site path is
    the only one with #632 device_mesh sharding, so n_devices>1 shards the D²=64
    axis. split: ctm_split_tensor — never forms the χ²·D⁶ edge (peak χ²·D³·d,
    forward χ²·D⁴-bounded, #641); single-GPU only (no device_mesh). Record-and-
    resume safe: never raises (OOM/errors are recorded)."""
    result = {
        "D": D, "chi": chi, "n_devices": n_devices, "path": path,
        "E_site": None, "err_vs_qmc": None, "total_s": None, "n_sweeps": None,
        "ms_per_sweep": None, "peak_gb": None, "converged": False,
        "oom": False, "error": None,
    }
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from tenax import heisenberg_gate, sublattice_rotate_gate

        with open(tensor_path, "rb") as fh:
            A_opt = pickle.load(fh)
        H = sublattice_rotate_gate(heisenberg_gate())

        if path == "split":
            E, total_s, sweeps, converged = _converge_split(A_opt, H, chi)
        else:
            E, total_s, sweeps, converged = _converge_dense(
                A_opt, H, chi, n_devices
            )

        result.update(
            E_site=E, err_vs_qmc=E - REFERENCE_E, total_s=float(total_s),
            n_sweeps=sweeps, ms_per_sweep=1000.0 * total_s / max(sweeps, 1),
            converged=bool(converged), peak_gb=d4._peak_gb(),
        )
    except Exception as e:  # noqa: BLE001 — record and resume, never crash the sweep
        msg = f"{type(e).__name__}: {e}"
        result["error"] = msg
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
            result["oom"] = True
        result["peak_gb"] = d4._peak_gb()
    return result


def _converge_dense(A_opt, H, chi, n_devices):
    """Dense closed-path forward CTM (χ²·D⁶); device_mesh-sharded for n>1."""
    import jax

    from tenax import CTMConfig, compute_energy_ctm_tensor
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

    mesh = d4._build_mesh(n_devices)  # A100-only guard + GSPMD mesh for n>1
    cfg = CTMConfig(
        chi=chi, max_iter=200, conv_tol=1e-10,
        projector_method="svd", forward_gauge="phase", device_mesh=mesh,
    )
    kwargs = ctm_converge_kwargs(cfg)

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
    return E, total_s, int(info.iterations), bool(info.converged)


def _converge_split(A_opt, H, chi):
    """Split forward CTM (χ²·D⁴-bounded); single-GPU only. chi_I=chi is already
    lossless at D=8 (the spike showed chi_I=2χ gives identical E and memory)."""
    import jax

    from tenax import compute_energy_split_ctm_tensor
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor

    # Warm-up compile (process-cached) so the timed converge is pure compute.
    warm = ctm_split_tensor(A_opt, chi=chi, max_iter=200, conv_tol=1e-10, chi_I=chi)
    jax.block_until_ready(list(warm))

    t0 = time.perf_counter()
    env, info = ctm_split_tensor(
        A_opt, chi=chi, max_iter=200, conv_tol=1e-10, chi_I=chi, return_info=True
    )
    jax.block_until_ready(list(env))
    total_s = time.perf_counter() - t0

    E = float(compute_energy_split_ctm_tensor(A_opt, env, H, 2))
    return E, total_s, int(info.iterations), bool(info.converged)
```

- [ ] **Step 2: Sanity-import** (no GPU needed; just confirms the module parses and helpers resolve):

Run: `uv run python -c "import importlib.util,pathlib; s=importlib.util.spec_from_file_location('d8', pathlib.Path('examples/heisenberg_d8_chi_scaling.py')); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.scan_cell, m._converge_split, m._converge_dense)"`
Expected: prints three function objects, no error.

- [ ] **Step 3: Run the jax-free helper tests** (regression — module still path-loads):

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -q`
Expected: PASS (all existing + new pure-helper tests).

- [ ] **Step 4: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py
git commit -m "feat(#650): scan_cell path dispatch (dense closed vs split forward CTM)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: d8 — thread `path` through worker + orchestrator

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Update the resume/launch tests** — replace `test_load_or_run_scan_returns_cached_cell`, `test_load_or_run_scan_returns_none_and_writes_no_poison` in `tests/test_heisenberg_d8_chi_scaling.py`:

```python
def test_load_or_run_scan_returns_cached_cell(tmp_path):
    cell = d8.Cell8(D=8, chi=64, n_devices=1, path="split")
    path = d8._cell_path(str(tmp_path), cell)
    d4._atomic_write_text(
        path, '{"D": 8, "chi": 64, "n_devices": 1, "path": "split", "oom": false}'
    )
    res = d8._load_or_run_scan(cell, str(tmp_path), timeout_s=1)
    assert res["chi"] == 64 and res["path"] == "split" and res["oom"] is False


def test_load_or_run_scan_returns_none_and_writes_no_poison(monkeypatch, tmp_path):
    monkeypatch.setattr(d8, "_launch", lambda *a, **k: None)
    cell = d8.Cell8(D=8, chi=128, n_devices=2, path="dense")
    res = d8._load_or_run_scan(cell, str(tmp_path), timeout_s=1, gpu_wait_s=0)
    assert res is None
    assert not pathlib.Path(d8._cell_path(str(tmp_path), cell)).exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k load_or_run -v`
Expected: FAIL (`Cell8`/`_cell_path` used by the test but `_load_or_run_scan` still calls `d4.cell_result_path` and the worker argv lacks `--path`).

- [ ] **Step 3: Implement** — make four edits in `examples/heisenberg_d8_chi_scaling.py`:

(a) `_run_worker`: pass `args.path` to `scan_cell`:

```python
    else:
        tensor_path = os.path.join(args.outdir, "A_opt.pkl")
        res = scan_cell(tensor_path, args.chi, args.n_devices, args.path)
```

(b) `_build_argparser`: the worker also needs `--path` as a value-carrying arg — it is already added in Task 4 (shared), so no change here beyond Task 4.

(c) `_load_or_run_scan`: key on `_cell_path` and forward `--path` in the worker argv:

```python
def _load_or_run_scan(cell, outdir, timeout_s, gpu_wait_s=1800):
    """Resume: load an existing cell JSON, else launch the scan worker and load
    what it wrote. Returns None (writing no cell JSON) when no idle A100s freed
    up within ``gpu_wait_s`` so the caller stops gracefully."""
    path = pathlib.Path(_cell_path(outdir, cell))
    cached = d4._read_json_or_none(path) if path.exists() else None
    if cached is not None:
        return cached
    argv = [
        sys.executable, str(pathlib.Path(__file__).resolve()), "--cell",
        "--phase", "scan", "--outdir", outdir, "--chi", str(cell.chi),
        "--n-devices", str(cell.n_devices), "--path", cell.path, "--out", str(path),
    ]
    ok = _launch(argv, cell.n_devices, timeout_s, gpu_wait_s)
    if ok is None:
        return None
    loaded = d4._read_json_or_none(path)
    if loaded is not None:
        return loaded
    res = {
        "D": cell.D, "chi": cell.chi, "n_devices": cell.n_devices, "path": cell.path,
        "E_site": None, "err_vs_qmc": None, "ms_per_sweep": None,
        "n_sweeps": None, "peak_gb": None, "converged": False, "oom": False,
        "error": ("timeout" if not ok else "worker produced no result file"),
    }
    d4._atomic_write_text(str(path), json.dumps(res, indent=2))
    return res
```

(d) `main`: build the grid with paths and iterate Cell8 rows:

```python
def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    chi_ladder = [int(x) for x in args.chi_ladder.split(",")]
    device_counts = [int(x) for x in args.device_counts.split(",")]
    paths = ["dense", "split"] if args.path == "both" else [args.path]

    # Phase 1: simple-update seed once (single idle A100).
    _su_phase(outdir, args.chi_su, args.imaginary_steps, args.dt, args.gpu_wait_s)
    if not os.path.exists(os.path.join(outdir, "A_opt.pkl")):
        print("[abort] SU produced no A_opt.pkl; see "
              f"{outdir}/su_status.json", flush=True)
        return

    # Phase 2: scan each (path, n_devices) row; stop a row at its wall. A None
    # result means no idle A100s freed up in time -> stop gracefully and
    # aggregate what completed (re-run to resume).
    cells = build_grid(chi_ladder, device_counts, paths)
    rows = {}
    for c in cells:
        rows.setdefault((c.path, c.n_devices), []).append(c)

    results = []
    aborted = False
    for key in rows:
        if aborted:
            break
        for cell in rows[key]:
            res = _load_or_run_scan(cell, outdir, args.cell_timeout_s, args.gpu_wait_s)
            if res is None:
                print(f"[abort] no {cell.n_devices} idle A100(s) for "
                      f"{cell.path} χ={cell.chi} within {args.gpu_wait_s}s; "
                      "aggregating completed cells (re-run to resume)", flush=True)
                aborted = True
                break
            results.append(res)
            if d4.should_stop_row(res):
                print(f"[stop] {cell.path} n={cell.n_devices} row stopped at "
                      f"χ={cell.chi} ({d4._status(res)})", flush=True)
                break

    _aggregate8(results, outdir)
```

(`build_grid` already emits cells grouped dense-then-split and device-major; the `rows` dict preserves first-seen order in CPython 3.7+, so dense rows run before the split row.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -q`
Expected: PASS (all helper tests; `_aggregate8` is added in Task 7 — if `main` references it before then, keep Task 7 next and do not run `main` here; the unit tests do not call `main`).

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(#650): thread --path through worker + orchestrator rows

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: d8 — `_aggregate8` + wall-comparison plot

**Files:**
- Modify: `examples/heisenberg_d8_chi_scaling.py`
- Test: `tests/test_heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_heisenberg_d8_chi_scaling.py`:

```python
def test_aggregate8_writes_both_path_sections(tmp_path):
    results = [
        {"D": 8, "chi": 96, "n_devices": 1, "path": "dense", "E_site": -0.605,
         "err_vs_qmc": 0.064, "total_s": 785.0, "n_sweeps": 14,
         "ms_per_sweep": 56000.0, "peak_gb": 59.7, "converged": False,
         "oom": False, "error": None},
        {"D": 8, "chi": 128, "n_devices": 1, "path": "dense", "E_site": None,
         "err_vs_qmc": None, "total_s": None, "n_sweeps": None,
         "ms_per_sweep": None, "peak_gb": 72.8, "converged": False,
         "oom": True, "error": "RESOURCE_EXHAUSTED"},
        {"D": 8, "chi": 128, "n_devices": 1, "path": "split", "E_site": -0.6005,
         "err_vs_qmc": 0.0689, "total_s": 14.2, "n_sweeps": 8,
         "ms_per_sweep": 1775.0, "peak_gb": 6.59, "converged": True,
         "oom": False, "error": None},
    ]
    d8._aggregate8(results, str(tmp_path))
    md = (tmp_path / "convergence.md").read_text()
    assert "Dense" in md and "Split" in md
    # comparison table lists the shared χ rows
    assert "128" in md
    csv_text = (tmp_path / "results.csv").read_text()
    assert "path" in csv_text.splitlines()[0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -k aggregate8 -v`
Expected: FAIL (`module ... has no attribute '_aggregate8'`).

- [ ] **Step 3: Implement** — add to `examples/heisenberg_d8_chi_scaling.py` (uses `csv` from stdlib — add `import csv` to the import block):

```python
def _comparison_table_md(dense, split):
    """One row per χ present in either path: dense vs split peak/E/status."""
    def by_chi(rs):
        return {r["chi"]: r for r in rs}

    d, s = by_chi(dense), by_chi(split)
    chis = sorted(set(d) | set(s))
    lines = [
        "### Wall comparison: dense (χ²·D⁶) vs split (χ²·D⁴)",
        "",
        "| χ | dense peak GB | dense E | dense | split peak GB | split E | split |",
        "|---|---------------|---------|-------|---------------|---------|-------|",
    ]
    for chi in chis:
        dr, sr = d.get(chi), s.get(chi)
        lines.append(
            f"| {chi} "
            f"| {d4._fmt(dr.get('peak_gb') if dr else None, '.2f')} "
            f"| {d4._fmt(dr.get('E_site') if dr else None, '.5f')} "
            f"| {d4._status(dr) if dr else '-'} "
            f"| {d4._fmt(sr.get('peak_gb') if sr else None, '.2f')} "
            f"| {d4._fmt(sr.get('E_site') if sr else None, '.5f')} "
            f"| {d4._status(sr) if sr else '-'} |"
        )
    return "\n".join(lines)


def _plot_wall_comparison(dense, split, outdir):
    """Two-panel PNG: per-device peak GB and converge wall-time vs χ, dense vs
    split, with the 80 GB device limit marked. Best-effort."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pts(rs, metric):
        return sorted((r["chi"], r[metric]) for r in rs
                      if not r.get("oom") and not r.get("error")
                      and r.get(metric) is not None)

    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(11, 4))
    for rs, label, color in [(dense, "dense", "C0"), (split, "split", "C1")]:
        m = pts(rs, "peak_gb")
        if m:
            ax_m.plot(*zip(*m), marker="o", label=label, color=color)
        t = pts(rs, "total_s")
        if t:
            ax_t.plot(*zip(*t), marker="o", label=label, color=color)
    ax_m.axhline(80.0, ls="--", color="k", label="80 GB A100")
    ax_m.set_xlabel("χ"); ax_m.set_ylabel("per-device peak GB")
    ax_m.set_title("D=8 single-GPU memory wall"); ax_m.legend()
    ax_t.set_xlabel("χ"); ax_t.set_ylabel("converge wall (s)")
    ax_t.set_yscale("log"); ax_t.set_title("D=8 converge time"); ax_t.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "d8_wall_comparison.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def _aggregate8(results, outdir):
    """Write per-path convergence + performance markdown, a dense-vs-split wall
    comparison table, results.csv (with a path column), and the comparison PNG."""
    dense = [r for r in results if r.get("path") == "dense"]
    split = [r for r in results if r.get("path") == "split"]

    sections = [_comparison_table_md(dense, split), ""]
    for name, rs in [("Dense", dense), ("Split", split)]:
        if rs:
            sections += [f"## {name} path", "",
                         d4.results_to_convergence_md(rs, d_label=D), "",
                         d4.results_to_performance_md(rs), ""]
    conv_md = "\n".join(sections)
    d4._atomic_write_text(os.path.join(outdir, "convergence.md"), conv_md)

    keys = ["D", "chi", "n_devices", "path", "E_site", "err_vs_qmc",
            "ms_per_sweep", "n_sweeps", "peak_gb", "converged", "oom", "error"]
    rows = [{k: r.get(k) for k in keys} for r in results]
    if rows:
        with open(os.path.join(outdir, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    try:
        _plot_wall_comparison(dense, split, outdir)
    except Exception as e:  # noqa: BLE001 — plotting is best-effort
        print(f"[warn] plotting failed: {e}", flush=True)
    print(conv_md)
    print(f"\n[done] wrote {outdir}/convergence.md, results.csv, "
          "d8_wall_comparison.png")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -q`
Expected: PASS (all helper tests including `aggregate8`).

- [ ] **Step 5: Commit**

```bash
git add examples/heisenberg_d8_chi_scaling.py tests/test_heisenberg_d8_chi_scaling.py
git commit -m "feat(#650): _aggregate8 dense-vs-split wall comparison (md + csv + plot)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: End-to-end smoke + docs

**Files:**
- Modify: `README.md` (benchmark/example mention, if present)
- Verify: `examples/heisenberg_d8_chi_scaling.py`

- [ ] **Step 1: Run the full jax-free helper suite**

Run: `uv run pytest tests/test_heisenberg_d8_chi_scaling.py -v`
Expected: PASS (all pure-helper tests).

- [ ] **Step 2: End-to-end smoke on one idle A100** (exercises SU seed + both dense and split paths at tiny χ). Pick an idle A100 index from `nvidia-smi` (never index 3):

Run:
```bash
CUDA_VISIBLE_DEVICES=<idle> CUDA_DEVICE_ORDER=PCI_BUS_ID \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
PYTHONPATH=$PWD/src uv run python examples/heisenberg_d8_chi_scaling.py --smoke
```
Expected: completes; writes `runs/d8_chi_scaling_smoke/convergence.md` with both a **Dense path** and a **Split path** section and a wall-comparison table; `d8_wall_comparison.png` present. (The smoke worker self-pins, so `CUDA_VISIBLE_DEVICES` is belt-and-suspenders; the orchestrator's `_worker_env` re-pins per cell.)

- [ ] **Step 3: Verify the split test from Task 1 still passes against the worktree library**

Run: `PYTHONPATH=$PWD/src uv run pytest tests/test_split_ctm_tensor.py -k return_info -v`
Expected: PASS.

- [ ] **Step 4: Update README** — if `README.md` references the D=8 benchmark, note the new `--path {dense,split,both}` flag and the split-CTM rescue (one sentence, consistent with the actual flag). If there is no D=8 mention, skip (do not invent a section).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(#650): note --path split-CTM rescue in the D=8 benchmark

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(If README needed no change, skip this commit.)

---

## Final Review

After all tasks: dispatch a final code-reviewer over the full diff (`git diff main...HEAD -- examples/heisenberg_d8_chi_scaling.py src/tenax/algorithms/_split_ctm_tensor_convergence.py tests/`), then use `superpowers:finishing-a-development-branch` to push and update PR #650.
