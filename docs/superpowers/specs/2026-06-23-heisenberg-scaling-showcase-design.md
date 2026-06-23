# Design: iPEPS Square-Lattice Heisenberg Scaling / Perf Showcase

**Date:** 2026-06-23
**Branch:** `showcase/heisenberg-scaling`
**Status:** Approved design → ready for implementation plan

## Purpose

A production-quality, reproducible run that **demonstrates the large-D / large-χ
dense iPEPS-AD machinery working at scale** on the 2D square-lattice
Heisenberg antiferromagnet, with timing and memory reporting alongside the
physics. This is the payoff for the performance-characterization arc (#566,
#570, #595–#597 reduced-corner CTM, #632/#634/#635 multi-GPU sharding): it shows
the *production-merged* levers actually deliver.

It is explicitly **not** a benchmark-convergence study or a single hero number —
the deliverable is a **scaling table + plots** over a (D, χ) grid, with an
explicit single-GPU-vs-4-GPU comparison.

## Scope and decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Deliverable | Scaling / perf showcase | Demonstrate the machinery at scale with timing/memory beside the physics. |
| Levers | **Production-merged only**: `gs_recipe="1x1"` (reduced-corner CTM) + `CTMConfig.device_mesh` (GSPMD multi-GPU) | Runs today; no new core build. Chunked-einsum CTM move and 2×2 truncated-SVD remain spike-only and are out of scope. |
| Envelope | **Wide grid, step-level metrics**; full convergence only at a few anchor cells | A scaling demonstration is a grid of `ms/step` + `peak GB`, not a grid of full optimizations (which would be tens of GPU-hours). |
| Multi-GPU demo | **Explicit 1-GPU vs 4-GPU** at matched cells | The N^(1/6) ceiling-extension is the centerpiece of the perf story; show where 1-GPU OOMs and 4-GPU survives. |

### Physics setup (defaults — chosen, not asked)

- **Model:** `sublattice_rotate_gate(heisenberg_gate())` — sublattice rotation is
  required for a **1-site** iPEPS tensor to represent the bipartite Néel ground
  state. (`heisenberg_gate` = `Sz⊗Sz + ½(S⁺⊗S⁻ + S⁻⊗S⁺)`.)
- **Unit cell:** 1-site (`unit_cell="1x1"`), the path validated for `device_mesh`
  sharding in `bench_rung2_optimize.py`.
- **Decompositions / AD:** `projector_method="svd"` on `gs_recipe="1x1"` (the fast
  `svd1x1` path; ~100× faster than the default `svd2x2` at large χ — see
  `chunked-einsum-ctm-lever` memory), `gs_implicit_ad=True`,
  `forward_gauge="sigma"` (implicit-AD stability, #292), L-BFGS optimizer,
  `float64`.
- **C4v:** **off** — stay on the sharding-validated code path
  (`_optimize_gs_ad_tensor_reference_c4v` composition with `device_mesh` is
  unverified and not in scope).
- **Init:** `su_init=True` (simple-update warm start).
- **Reference energy:** E/site = **−0.669437** (Sandvik QMC, square-lattice
  spin-½ Heisenberg AFM) — the physics column is reported as a distance to this.

## Architecture

The cumulative-peak-memory trap is the central architectural driver:
`peak_bytes_in_use` is a high-water mark JAX **never resets**, so two configs
measured in one process report a contaminated (shared, monotone) peak. Therefore
**one cell = one OS subprocess.**

```
showcase_heisenberg_scaling.py
├── orchestrator (default mode)
│     • builds the (D, χ, n_devices, measurement-mode) cell list
│     • for each cell: launch `python showcase_heisenberg_scaling.py --cell ...`
│       with the correct CUDA_VISIBLE_DEVICES + XLA_PYTHON_CLIENT_PREALLOCATE=false
│     • RESUME: skip any cell whose result JSON already exists
│     • OOM-aware ramp: when a (D, n_devices) row OOMs at χ, stop ramping χ for that row
│     • aggregate all JSON → markdown table + CSV + plots
│
└── per-cell worker (`--cell` mode)
      • build sublattice-rotated gate, iPEPSConfig (recipe=1x1, device_mesh?),
      • measurement modes:
          - "metrics": SU-init → 1 warm forward-CTM convergence → time N warm CTM
            sweeps (ms/sweep) + one value_and_grad (ms/step, exercises AD backward)
            → record per-device peak GB. No full optimization.
          - "anchor": full optimize_gs_ad → converged E/site, plus the metrics.
      • write ONE JSON: {D, chi, n_devices, mode, ms_per_sweep, ms_per_step,
        peak_gb, E_site (anchor only), converged, oom, error}
```

### Components (each independently testable)

1. **Grid builder** — pure function `(D_list, chi_ramp, device_configs,
   anchors) → list[Cell]`. Deterministic; unit-tested.
2. **Cell runner (worker)** — given a `Cell`, produce a result dict. The physics
   calls reuse already-tested library code (`optimize_gs_ad`, CTM convergence).
3. **Subprocess launcher** — maps a `Cell` to `(argv, env)` and runs it; captures
   exit status; turns a non-zero/OOM exit into a result. Unit-tested for the
   argv/env mapping and result-file resume logic (no real subprocess in tests).
4. **OOM-aware ramp controller** — pure function deciding the next χ for a
   `(D, n_devices)` row given prior results; stops the row on first OOM.
   Unit-tested.
5. **Aggregator / reporter** — pure functions: `results → markdown table`,
   `results → CSV rows`, and a plotting entry that writes PNGs. Table/CSV
   formatting unit-tested; plotting is a thin matplotlib wrapper (smoke-only).

## Phase 0 — compatibility gate (precedes all GPU-hours)

Before launching the grid, smoke-test the four-way combination
**1-site + `gs_recipe="1x1"` + `device_mesh` (4 GPUs) + `gs_implicit_ad`** at
**D=2, χ=8** and assert:
- forward energy is finite and near the SU-init value,
- one optimizer step produces a **finite gradient** and a non-increasing energy.

Outcomes:
- **Composes** → the full grid uses `gs_recipe="1x1"` for both 1-GPU and 4-GPU.
- **Does not compose** → documented fallback: `gs_recipe="1x1"` **single-GPU**
  for the speed/scaling curves; `gs_recipe="2x2"` (the sharding-validated path)
  for the 4-GPU memory demonstration. The fallback is reported honestly in the
  findings doc as a discovered composition limit, not hidden.

This gate is consistent with the repo's gated-spike culture and protects against
spending GPU-hours on a broken combination.

## Grid and metrics

- **D ∈ {2, 3, 4}** core; **D = 5** as a ceiling probe (may OOM early — expected).
- **χ ramp** per D, e.g. {16, 24, 32, 48, 64, 96, 128}, truncated by the
  OOM-aware controller.
- **Device configs:** 1-GPU (`CUDA_VISIBLE_DEVICES=0`) and 4-GPU
  (`CUDA_VISIBLE_DEVICES=0,1,2,3` — **not** the CUDA-index-4 display GPU) at
  matched cells.
- **Metrics per cell (warm, compile excluded):** `ms_per_sweep` (forward CTM),
  `ms_per_step` (one value_and_grad), `peak_gb` (per device).
- **Anchor cells** (small set, e.g. D2χ32 / D3χ48 / D4χ64): full
  `optimize_gs_ad` → converged `E_site`.

## Error handling

- Worker wraps the run in try/except; an OOM (`RESOURCE_EXHAUSTED` /
  `XlaRuntimeError`) → `{oom: true}`, any other exception → `{error: "..."}`.
  The worker **always** writes a JSON so the orchestrator can resume and so a
  failed cell is visible rather than silent.
- Orchestrator treats a missing/`oom`/`error` cell as ramp-terminating for that
  `(D, n_devices)` row but continues other rows.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set for every cell so
  `peak_bytes_in_use` reflects actual high-water, not the preallocated pool.

## Deliverables

- `examples/showcase_heisenberg_scaling.py` — orchestrator + `--cell` worker.
- `examples/showcase_results/*.json` — per-cell results (checkpoint/resume).
- Scaling **table** (markdown) + **CSV** + **plots** (PNG): `ms/step vs χ` per D,
  `peak GB vs χ` with 1-GPU vs 4-GPU overlay, `E/site vs (D,χ)` → −0.669437.
- Findings doc:
  `docs/superpowers/handoffs/2026-06-23-heisenberg-scaling-showcase-findings.md`.

## Testing

GPU runs are not unit-testable, but the orchestration logic is. TDD the pure
helpers:
- grid builder (cell enumeration, anchor flagging),
- OOM-aware ramp controller (stops a row on first OOM; continues others),
- subprocess argv/env mapping + resume-skip (result-file presence),
- aggregator (results → markdown table / CSV rows).

The physics path reuses already-tested library code; the worker is exercised
end-to-end only by the Phase-0 gate and the real run.

## Out of scope

- Chunked-einsum CTM move (spike-only; not wired into `_compiled_move_*`).
- Truncated-SVD into the default 2×2 projector (deferred checkpoint; irrelevant
  once `recipe="1x1"` is used).
- U(1)-Sz symmetry (NO-GO for D≥3 per #566 line).
- Order-parameter / correlation-length observables (could be a follow-up; the
  chosen goal is the perf/scaling showcase, energy is the physics column).
