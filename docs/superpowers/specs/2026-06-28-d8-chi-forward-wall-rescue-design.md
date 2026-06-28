# D=8 large-χ forward-CTM wall + multi-GPU rescue — design

**Date:** 2026-06-28
**Status:** approved (brainstorming)
**Author:** Claude Code (Opus 4.8), with Ying-Jer Kao

## Goal

Demonstrate, at D=8 on the square-lattice Heisenberg AFM, that the single-site
**forward** CTM reaches a per-device memory wall on one 80 GB A100 at large χ,
and that #632 GSPMD sharding **rescues** it: a χ where 1-GPU OOMs but a 2-GPU
(and, deferred, 4-GPU) run still converges. The headline artifact is the χ at
which 1-GPU OOMs versus the largest χ a sharded run sustains.

This continues the χ-scaling line: D=3 (#645), D=4 (#646/#649), now D=8 as the
rung where single-GPU genuinely cannot hold the environment.

## Decisions (locked during brainstorming)

1. **Headline:** wall + multi-GPU rescue (not just mapping the 1-GPU wall).
2. **State:** simple-update only — no AD optimize.
3. **Execution:** launch the {1, 2}-GPU rows now on free A100s; defer the 4-GPU
   rescue row until four A100s are simultaneously idle.

## Key constraints discovered

- **SU is intrinsically 2-site.** A true 1×1 unit cell cannot host the
  alternating horizontal/vertical bond updates. `ipeps()` is the 2-site SU path.
- **Multi-GPU sharding is single-site-only.** The #632 GSPMD helpers
  (`ctm_sharding.commit_env`, `commit_double_layer`,
  `constrain_double_layer_for_move`) are wired into the single-site tensor CTM
  path (`_ctm_python_loop`, `_ctm_tensor_convergence`, `CTMTensorEnv`). The
  2-site `ctm_2site` has **no `device_mesh` support**. So the rescue can only be
  shown on the single-site backbone, which needs a single-site state.
- **Reconciliation (the project's own convention).** `optimize_gs_ad`'s 1×1
  `su_init` path (`ipeps_optimize.py:977`) does exactly:
  ```python
  _, (A_su, _B_su), _ = ipeps(gate, None, config)  # 2-site SU
  A_init = A_su                                      # take A-sublattice as 1-site seed
  ```
  Valid because the sublattice-rotated gate makes the A and B sublattices
  equivalent, so the A tensor alone parametrizes the uniform 1-site ansatz. We
  replicate this: run 2-site SU, take `A_su` as the single-site C4v seed. This
  honors SU-only (no AD), yields a single-site state, and enables the rescue.
- **GSPMD divisibility.** The sharded axis is the D²=64 fused virtual index;
  `n_devices` must divide 64, so device counts ∈ {1, 2, 4}. Three free GPUs
  cannot run a 4-GPU row.
- **GPU churn.** Free A100 set changes during the session (index 2 freed, index
  4 became busy). The driver must pin to **actually-free** A100s rather than a
  hardcoded index order, and never select the 4 GB DGX Display GPU (index 3).

## Architecture

New sibling driver `examples/heisenberg_d8_chi_scaling.py`, same orchestrator +
per-cell worker shape as `heisenberg_d4_chi_scaling.py`. It **imports the
D-agnostic pure helpers** from the D=4 module (md/csv formatters, `make_plots`,
`_atomic_write_*`, `_read_json_or_none`, `_peak_gb`, `_assert_only_a100s`,
`should_stop_row`, `_status`, `_fmt`, `_e_by_chi`, `REFERENCE_E`) and keeps its
**own worker dispatch + orchestration** (the worker subprocess re-invokes
`__file__`, so that part cannot be shared). The merged, tested D=4 file is not
modified.

Rationale for a sibling (vs parametrizing the D=4 file): the D=4 driver hardcodes
`D=4` at module level threaded through many functions and couples its state phase
to AD optimize. Parametrizing it risks regressing a freshly-merged tested
benchmark; a sibling reuses the pure machinery (DRY on formatting/plots) with
zero regression risk.

### Free-A100 device pinning

Replace the hardcoded `A100_INDICES = [0,1,2,4]` selection with runtime
discovery of idle 80 GB A100s (via `nvidia-smi` memory.used / utilization,
excluding the DGX Display GPU). For an `n`-GPU row, pick the `n` most-idle A100s.
Fail loudly if fewer than `n` are available (record the cell as error, stop the
row) rather than landing on a busy or display GPU. Keep the existing
`_assert_only_a100s` backstop inside the worker.

## Components / data flow

### Phase 1 — SU seed (single-GPU, cheap, resumable)

- Build gate `sublattice_rotate_gate(heisenberg_gate())`.
- `su_cfg = iPEPSConfig(max_bond_dim=8, num_imaginary_steps≈200, dt≈0.05,
  ctm=CTMConfig(chi=χ_su, ...))` with a modest SU-phase χ (e.g. 16–32) — only
  used by the 2-site CTM energy eval inside `ipeps()`, kept small/cheap.
- `_, (A_su, _B_su), _ = ipeps(gate, None, su_cfg)`; `A = symmetrize_c4v(A_su)`.
- Gather to host (`jax.device_get`) and pickle to `<outdir>/A_opt.pkl` via the
  atomic writer. Existence-cached: a present `A_opt.pkl` skips Phase 1.
- Real float64, single-site shape `(8,8,8,8,2)`.

### Phase 2 — single-site forward χ-scan (the wall + rescue)

For `n in {1, 2}` (4 deferred), for χ in `[64, 96, 128, 160, 192, 224, 256]`:

- Worker subprocess pinned to the `n` most-idle A100s, `CUDA_DEVICE_ORDER=
  PCI_BUS_ID`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` (so peak_gb is real).
- Build mesh via `build_ctm_mesh()` for n>1 (after the A100-only guard).
- Load `A_opt.pkl`; warm-up compile the χ-specific jit step; then time a
  `python_loop_ctm_converge({(0,0): A}, SINGLE_SITE_NEIGHBORS,
  device_mesh=mesh, ...)` to convergence.
- Record E/site (`compute_energy_ctm_tensor`, env gathered to device 0 under a
  mesh), ms/sweep, n_sweeps, converged, per-device peak GB, oom flag.
- Auto-stop the row at the first OOM / error / timeout (CTM cost is monotone in
  χ). Record-and-resume safe: each cell writes its own JSON atomically; a
  present cell JSON is reused on restart.

Expected wall (rough, real f64, dominant χ²·D⁶ intermediate): 1-GPU OOM near
χ≈128–160; sharding the D²=64 axis lifts the 2-GPU (and later 4-GPU) ceiling.
The #632 forward per-device headroom is ~1.6×, so the rescue window in χ is
narrow (~1.27× in χ); the 32-wide χ ladder near the wall is sized to catch it.

### Aggregation / outputs

`runs/d8_chi_scaling/`:
- `convergence.md` — E/site vs χ (device-independent), vs QMC reference.
- `performance.md` — ms/sweep, n_sweeps, peak GB per device row, with the
  rescue visible as a 1-GPU OOM row beside a finite 2-GPU row at the same χ.
- `results.csv`, and best-effort PNGs (E vs χ; ms/sweep; peak GB; speedup).

### Smoke mode

`--smoke`: tiny SU χ + few imaginary steps + small χ ladder (e.g. `8,12`) +
`{1,2}` rows, separate outdir, short cell timeout — fast end-to-end validation
before the real background launch.

## Error handling

- Worker never crashes the sweep: each cell catches all exceptions, classifies
  `RESOURCE_EXHAUSTED` / "out of memory" as `oom`, records peak_gb, returns.
- A timeout or missing result file is recorded as an error so the row stops.
- Atomic writes (temp + `os.replace`) so a kill mid-write never leaves a
  truncated `A_opt.pkl` or cell JSON that the existence short-circuits would
  trust.

## Testing

- jax-free unit tests for the new pure helpers: free-A100 selection (mock
  `nvidia-smi` output → most-idle picks; raises when too few; never picks the
  display GPU), χ-ladder/grid build, row-stop on OOM. Mirror
  `tests/test_heisenberg_d4_chi_scaling.py` structure; mark `core` (fast).
- `--smoke` is the end-to-end integration check (run manually on a GPU box, not
  in CI).

## Out of scope

- The 4-GPU rescue row (deferred until four A100s are idle; driver supports it).
- Adding `device_mesh` sharding to the 2-site `ctm_2site` path.
- Any AD optimize of the D=8 state (SU seed only).
- Promoting results into `docs/benchmarks/` (a follow-up, like the D=4 PR #649).

## Launch plan

1. Build driver + pure-helper unit tests; run `--smoke` on a free A100.
2. Background-launch the real `{1, 2}`-GPU run on the free A100s.
3. Add the 4-GPU row when four A100s are simultaneously idle.
