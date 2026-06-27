# D=4 Heisenberg AFM χ-scaling benchmark (convergence + performance)

Date: 2026-06-27
Status: approved design, pending spec review

## Goal

A single benchmark driver for the spin-1/2 antiferromagnetic Heisenberg model on
the infinite square lattice at bond dimension **D=4**, measuring two things from
one optimized state:

1. **Convergence** — ground-state energy `E/site` vs CTM environment dimension χ
   (how large must χ be at D=4), against the QMC reference `E/site ≈ -0.669437`.
2. **Performance** — per-sweep CTM cost (ms/sweep), per-device peak memory, and
   the 1-GPU vs multi-GPU speedup, across χ.

Target hardware: this box's **four A100-SXM4-80GB GPUs at CUDA/PCI indices
0, 1, 2, 4**. Index 3 is a 4 GB DGX Display GPU and MUST be excluded.

## Non-goals

- Not a publication-grade energy (one optimization run, moderate χ_opt). The
  convergence curve answers "is χ large enough", not "is this the certified GS".
- No new physics or algorithm code — reuses the validated iPEPS-AD + CTM stack.
- No symmetric-tensor path (D≥3 is dense-pragmatic per prior studies).

## Design rationale

Two existing drivers each solve half the problem:

- `examples/heisenberg_d3_chi_convergence.py` — the **physics backbone**:
  optimize once at moderate χ_opt (implicit AD + C4v + grad-spike guard), then a
  *clean* `E(χ)` scan on the **fixed** optimized tensor via
  `python_loop_ctm_converge`. Separating optimization from the χ-scan is what
  makes the convergence curve clean (no optimizer noise per χ).
- `examples/showcase_heisenberg_scaling.py` — the **orchestration backbone**:
  subprocess-per-config (so `CUDA_VISIBLE_DEVICES` is set before JAX init),
  resume from per-cell JSON, per-cell timeout, `peak_gb`, `device_mesh` via
  `build_ctm_mesh()`.

The new driver fuses these. Crucially, the multi-GPU `device_mesh` already
threads through BOTH CTM entry points: `ctm_converge_kwargs(ctm_cfg)` forwards
`ctm_cfg.device_mesh`, and `python_loop_ctm_converge` accepts `device_mesh`. So
the fixed-state scan can be sharded with no new plumbing.

Why a clean separation of the two measurements works in one scan loop: run the
**same fixed `A_opt`** through forward CTM at each χ for each `n_devices ∈
{1,2,4}`. Then
- the **E(χ)** column is device-independent (E across n is a correctness
  cross-check, like #632's grad parity), giving the convergence curve;
- the **time/memory** columns at identical `(state, χ)` give an apples-to-apples
  speedup curve.

## Architecture

One file, `examples/heisenberg_d4_chi_scaling.py`, in orchestrator + worker form
(same rationale as the showcase: JAX/tenax imported only inside the worker, after
the parent has pinned `CUDA_VISIBLE_DEVICES`).

### Pure helpers (stdlib only — unit-testable without JAX, marked `core`)

- `A100_INDICES = [0, 1, 2, 4]` — the four A100s (display GPU 3 excluded).
- `cuda_visible_for(n_devices) -> str` — first `n` of `A100_INDICES`, e.g.
  `1→"0"`, `2→"0,1"`, `4→"0,1,2,4"`. Paired with `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  so CUDA indices match nvidia-smi indices deterministically.
- `build_grid(chi_ladder, device_counts) -> list[Cell]` — scan cells in
  deterministic order.
- `cell_result_path(...)`, `should_stop_row(...)`, `results_to_*` — resume +
  reporting, mirroring the showcase. Two tables emitted: a **convergence** table
  (χ → E/site, err_vs_qmc) and a **performance** table (χ × n_devices →
  ms/sweep, n_sweeps, peak_gb, speedup vs 1-GPU).

### Worker

- `optimize_once(outdir, chi_opt, opt_steps, n_devices)` — the d3 recipe
  verbatim (implicit AD, C4v, `gs_energy_floor`, `gs_grad_spike_ratio`,
  checkpoint/resume), but `max_bond_dim=4` and `ctm.device_mesh` set when
  `n_devices > 1`. Writes `A_opt.pkl`. Runs ONCE total (pinned to 4 GPUs); if
  `A_opt.pkl` exists it is skipped.
- `scan_cell(A_opt, chi, n_devices) -> dict` — build `CTMConfig(chi, …,
  forward_gauge="phase", projector_method="svd", device_mesh=mesh-or-None)`,
  converge via `python_loop_ctm_converge(..., **ctm_converge_kwargs(cfg))`,
  timing the call; read sweep count from the returned `info`; evaluate
  `compute_energy_ctm_tensor(A_opt, env, H, 2)`; record `peak_gb`. Returns
  `{D, chi, n_devices, E_site, err_vs_qmc, total_s, n_sweeps, ms_per_sweep,
  peak_gb, oom, error}`.

### Orchestrator (`main`)

1. **Optimize** — one subprocess pinned to 4 A100s → `A_opt.pkl` (resume-aware).
2. **Scan** — for `n in device_counts`, for `chi in chi_ladder`: launch a worker
   subprocess pinned to `cuda_visible_for(n)`; load/record its JSON; resume if
   present; per-cell timeout; stop the row on OOM/error.
3. **Aggregate** — convergence table + performance table + CSV + best-effort
   plots (`E vs χ` with QMC line; `ms/sweep vs χ` per n; `speedup vs χ`;
   `peak_gb vs χ` per n).

## Data flow

`optimize_once → A_opt.pkl` → (shared, read-only) → `scan_cell` × (χ × n) →
per-cell JSON → orchestrator aggregate → `convergence.md`, `performance.md`,
`results.csv`, `*.png`.

## Error handling & resume

- Worker is record-and-resume: any exception is caught and written to the cell
  JSON (`error`, `oom` on `RESOURCE_EXHAUSTED`/OOM); the sweep never crashes.
- A row (`fixed n`, ascending χ) stops on the first OOM/error/timeout (cost is
  monotone in χ).
- Resume: an existing cell JSON is loaded instead of re-run; the optimization
  resumes from its gs checkpoint and `A_opt.pkl` short-circuits the whole phase.
- **Device-safety guard**: the worker asserts every `jax.devices()` entry is an
  80 GB A100 (`device_kind`/memory check) and aborts with a clear error if a 4 GB
  display GPU ever appears — so a wrong index can never silently corrupt a run.

## Testing strategy (TDD)

- Pure-helper unit tests first (`tests/test_heisenberg_d4_chi_scaling.py`,
  marked `core`, no JAX): `cuda_visible_for` (incl. the 0,1,2,4 mapping and that
  3 is never emitted), `build_grid` ordering/resume, `results_to_markdown`
  shape, `should_stop_row`.
- `--smoke` end-to-end path: D=4, χ_opt=8, ~6 opt steps, scan χ∈{8,12},
  n∈{1,2}. Validates the full subprocess/sharding/energy pipeline cheaply and
  doubles as a manual sanity gate before the full sweep.

## CLI & defaults

```
uv run python examples/heisenberg_d4_chi_scaling.py            # full sweep
uv run python examples/heisenberg_d4_chi_scaling.py --smoke    # quick validation
```

- `--outdir`         default `runs/d4_chi_scaling`
- `--chi-opt`        default **32** (moderate; implicit-AD backward is
                     spike-prone at large χ — d3 lesson)
- `--opt-steps`      default **100**
- `--chi-ladder`     default **16,24,32,48,64,96,128**
- `--device-counts`  default **1,2,4**  (D²=16 divides 1/2/4 — GSPMD ok)
- worker flags: `--cell --phase {optimize,scan} --chi --n-devices --out`

## Open items to verify in the smoke run (before the full sweep)

1. `compute_energy_ctm_tensor` on a **sharded** env. Fallback if it misbehaves:
   gather the small (χ²·D⁴) env to one device before the energy contraction.
2. The exact CUDA↔nvidia-smi index mapping under `CUDA_DEVICE_ORDER=PCI_BUS_ID`
   (the device-safety guard backstops this regardless).
3. `A_opt.pkl` round-trips cleanly and re-shards across device counts.

## Validation outcome (2026-06-27 smoke, 4×A100)

Amendment to the architecture above. The smoke run resolved all three open items
and surfaced one design change:

- **Open item 1 (sharded energy):** resolved — the χ²·D⁴ env is gathered to one
  device before `compute_energy_ctm_tensor`; 1-GPU vs 2-GPU energies agree to
  ~1e-8 (FP reassociation from the sharded contraction order).
- **Open item 2 (index mapping):** resolved — `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus
  the A100-only guard; the run never touched the display GPU.
- **Open item 3 (A_opt re-shard):** resolved — `jax.device_get(A_opt)` before
  pickling makes the cached tensor device-agnostic; each scan worker re-shards.
- **Design change — the optimization runs single-GPU, not on 4 GPUs.** The gs
  checkpoint (`_checkpoint.save_checkpoint`) pickles the optimizer state, which
  under a `device_mesh` holds mesh-sharded `jax.Array`s referencing `Device`
  objects that pickle cannot serialise. Since the one-time optimization gains
  nothing from multi-GPU that the χ-scan doesn't already measure, it runs
  single-GPU (checkpoint-safe, the proven d3 path) while the scan carries the
  full multi-GPU story. The `--opt-devices` default is now **1**. Multi-GPU
  optimization would require a core `_checkpoint.py` fix (gather/replicate
  sharded leaves on save, re-shard on load) — a possible follow-up, out of scope.
- Multi-GPU compute genuinely engages in the scan: 1.66× per-sweep speedup at
  χ=12 even at the tiny smoke scale. Per-device peak memory does *not* drop at
  smoke χ (8, 12) — the memory win is a large-χ effect (the χ²·D⁶ intermediate);
  the speedup confirms sharding is active.
