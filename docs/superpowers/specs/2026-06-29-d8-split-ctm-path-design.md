# D=8 Split-CTM Path — Design Spec

**Date:** 2026-06-29
**Branch:** `feat/d8-chi-forward-wall-rescue` (extends PR #650)
**Status:** Approved

## Goal

Add a selectable forward-CTM path to the D=8 χ-scaling benchmark
(`examples/heisenberg_d8_chi_scaling.py`) so a single run demonstrates **both**
the dense single-GPU memory wall (χ≈112) **and** the split-CTM rescue (χ≈448)
on one figure.

## Motivation (measured in the prototype spike)

Same SU seed, same `H`, one 80 GB A100, cuda_async allocator:

| χ | path | E/site | peak GB | converge | status |
|---|------|--------|---------|----------|--------|
| 112 | dense | −0.60625 | 80.9 | 1230 s | ✓ (at limit) |
| 128 | dense | — | 72.8 + 32 req | — | **OOM** |
| 128 | split | −0.60053 | 6.59 | 14 s | ✓ |
| 384 | split | −0.60053 | 59.3 | 21 s | ✓ |
| 512 | split | — | 79 + 16 req | — | **OOM** |

The dense path forms the χ²·D⁶ absorb intermediate (~19 GB f64 block) — that *is*
the OOM. `ctm_split_tensor` never materializes it (peak χ²·D³·d, forward
χ²·D⁴-bounded, issue #641), moving the single-GPU wall from χ≈112 to χ≈448
(~3.5–4×), at ~12× less memory and ~50–100× faster. This is a single-GPU,
**algorithmic** win — the lever GSPMD multi-GPU could not provide (it was inert;
per-device peak was unchanged).

**Correctness note (carried into the benchmark honestly):** split-CTM converges
to a χ-invariant E=−0.60053 (identical to 13 digits across χ 48→384). Dense
*oscillates* around −0.6055 and reports `converged:false` at every χ (it never
meets conv_tol=1e-10 on this raw SU seed). The ~0.8 % gap is a genuine
split-vs-dense boundary-truncation difference, with split the cleanly-converged
side. Both sit +0.06 above QMC because the SU seed is un-optimized — that is the
seed, not the CTM method.

## Design Decisions

1. **`--path {dense,split,both}`, default `both`.** Each run produces the
   side-by-side comparison.
2. **Split χ ladder reaches χ=448** (near the measured ~χ=440 wall; the top rung
   may OOM and is recorded gracefully). Default shared ladder:
   `64,96,112,128,192,256,320,384,448`.
3. **d4 sibling stays unmodified.** The path dimension is added with d8-local
   helpers (`Cell8`, `_cell_path`, `_aggregate8`), reusing d4's D-agnostic
   formatters where they already ignore the path field.
4. **One small library addition:** `ctm_split_tensor(..., return_info=False)`
   gains an optional `(env, info)` return so split cells report
   `n_sweeps`/`ms_per_sweep`/`converged` identically to dense. Backward-compatible.

## Components

### C1. `Cell8` + `_cell_path` (d8-local)
A frozen `Cell8(D, chi, n_devices, path)` (the d4 `Cell` has no path field).
`_cell_path(outdir, cell) -> "<outdir>/D8_chi{chi}_n{n}_{path}.json"` keeps dense
and split cells at the same χ from colliding (resume-safe). `should_stop_row`,
`_read_json_or_none`, `_atomic_write_text` are reused from d4 (they consume the
result dict / paths, not the cell type).

### C2. `build_grid(chi_ladder, device_counts, paths)`
Emits `Cell8` rows:
- for `dense` in paths: `device_counts` × `chi_ladder` (device-major, χ-minor);
- for `split` in paths: `chi_ladder` at **n=1 only** (split has no `device_mesh`).
Order: all dense rows, then the split row. `paths` derives from `--path`
(`both` → `["dense","split"]`).

### C3. `scan_cell(tensor_path, chi, n_devices, path)`
Common: load `A_opt`, build `H`. Dispatch on `path`:
- **dense** (unchanged): `mesh = _build_mesh(n_devices)`;
  `python_loop_ctm_converge` warm+timed; `compute_energy_ctm_tensor(A, env, H, 2)`;
  `n_sweeps`/`converged` from `info`.
- **split**: `mesh = None` (n_devices ignored); warm-up
  `ctm_split_tensor(A, chi, max_iter=200, conv_tol=1e-10, chi_I=chi)` to compile,
  then a timed converge with `return_info=True`;
  `compute_energy_split_ctm_tensor(A, env, H, 2)`; `n_sweeps`/`converged` from
  the returned info; `ms_per_sweep = 1000*total_s/max(sweeps,1)`.
Result dict gains `"path": path`; otherwise the existing schema (`E_site`,
`err_vs_qmc`, `total_s`, `n_sweeps`, `ms_per_sweep`, `peak_gb`, `converged`,
`oom`, `error`). Still never raises (record-and-resume).

### C4. `ctm_split_tensor(..., return_info=False)` (library, `_split_ctm_tensor_convergence.py`)
Track the sweep count and whether the loop broke on conv_tol. With
`return_info=True` return `(env, _SplitCTMInfo(iterations, converged))`; default
`False` returns `env` unchanged (no behavior change for existing callers).
`_SplitCTMInfo` is a small local `NamedTuple(iterations: int, converged: bool)`.

### C5. `_aggregate8(results, outdir)` (d8-local)
Split `results` by `path`. Write `convergence.md` with one
`d4.results_to_convergence_md(path_results, d_label=8)` section per path, plus a
**comparison header table** (χ, dense peak/E/status, split peak/E/status). Write
one figure `d8_wall_comparison.png`: peak_gb vs χ (dense & split series) with an
80 GB device-limit line and OOM walls marked; a second panel for total_s vs χ
(log-y). Reuses `d4._status`, `d4._fmt`, `REFERENCE_E`, matplotlib.

### C6. Orchestrator `main` + `_run_worker` + argparse + smoke
- `--path` flag (choices `dense|split|both`, default `both`).
- `main`: build rows from `build_grid(chi_ladder, device_counts, paths)`; sweep
  each row, stop a row on `should_stop_row`; aggregate with `_aggregate8`.
  Existing graceful-abort on `None` (no idle A100) is preserved.
- `_run_worker`/`_load_or_run_scan` thread `--path` through to `scan_cell` and
  `_cell_path`.
- `_apply_smoke`: `path` stays `both`, χ `8,12`, device_counts `1` — exercises
  both dense and split end-to-end fast.

## Data Flow

`main` → `build_grid` (Cell8 rows) → per row per χ: `_load_or_run_scan`
(resume cache via `_cell_path`) → `_launch` worker subprocess → `_run_worker` →
`scan_cell(path)` → result JSON → `_aggregate8` → `convergence.md` +
`d8_wall_comparison.png`.

## Error Handling / Resume

Unchanged contract: `scan_cell` records OOM/errors (never raises);
`should_stop_row` halts a row at its wall; `_cell_path` makes resume idempotent
per (χ, n, path); a `None` from `_load_or_run_scan` (no idle A100) aborts the
sweep gracefully and aggregates what completed. The split top rung (χ=448) may
OOM — recorded like any other wall.

## Testing

Pure-helper unit tests stay **jax-free** (path-loaded module):
- `build_grid` with `paths`: dense rows × device_counts + split row at n=1 only;
  order dense-then-split; split ignores device_counts>1.
- `_cell_path`: dense vs split at same χ produce distinct filenames.
- `_apply_smoke`: path stays `both`, ladder `8,12`, n=1.
- argparse: `--path` default `both`; choices validated.
- `_aggregate8`: a mixed dense+split results list yields a `convergence.md` with
  both sections and a comparison table (no jax/plot assertions — patch/skip the
  figure or assert the markdown only).
Library test (`tests/test_split_ctm_tensor.py`, uses jax): `ctm_split_tensor(...,
return_info=True)` returns `(env, info)` with `info.iterations >= 1` and a bool
`info.converged`; default call still returns a bare env.

CI required checks run `pytest -m core`; the new example tests are auto-marked
`core` by `conftest.py` (already registered for the d8 test file). The library
test lands in the existing `test_split_ctm_tensor.py` (core).

## Out of Scope

- Multi-GPU split (no `device_mesh` in `ctm_split_tensor`; the win is algorithmic).
- `chi_I` override flag (chi_I=chi is already lossless at D=8 per the spike;
  YAGNI).
- AD / energy-optimized states (the SU seed's +0.06-vs-QMC offset is the seed,
  out of scope for a memory-wall benchmark).
- Modifying the d4 sibling driver.
