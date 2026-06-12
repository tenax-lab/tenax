# Design: Dense 2D-Heisenberg large-D characterization study

**Date:** 2026-06-12
**Status:** approved (design); spec under review
**Issue:** #570 (the never-evaluated "large-D GPU energy + runtime + compile" acceptance criterion)
**Branch:** `study/heisenberg-largeD-characterization`

## Goal

Answer the question #570 named but never evaluated, for the **dense** path:

> How does tenax's dense iPEPS-AD perform on the 2D spin-½ Heisenberg model as the PEPS
> bond dimension **D** and CTM bond dimension **χ** grow — the **final energy** relative to
> the **E/site = −0.6694430** reference (Corboz QR-CTMRG / QMC), and the **runtime + cold
> XLA-compile** cost — and **where does the dense wall hit** (OOM / compile-time / no further
> energy gain)?

This is a **characterization** study of the path that exists today. It builds **no** new
physics machinery. The deliverable is an honest scaling picture plus a statement of what
would be needed (U(1) symmetry) to push further.

## Non-goals (explicitly out of scope)

- U(1)/Sz-symmetric Heisenberg (the lever for genuinely large D — flagged, not built).
- 2-site bipartite unit cell (the sublattice-rotated C4v 1-site path is used instead).
- Fermionic iPEPS / the #392 certification gap (separate thread).
- The QR projector (block-sparse Phase 3 = NO-GO, PR #598; `projector_method="svd"` only).
- Reaching the reference scale D=7 χ=300 (dense is not expected to get there).
- TPU; plotting infrastructure; unit tests (this is an `examples/` study script, matching
  the existing `bench_*` / `profile_*` scripts which carry no tests).

## Configuration (single, fixed across all cells)

- **Model gate:** `sublattice_rotate_gate(heisenberg_gate())` — the sublattice rotation maps
  the Néel antiferromagnet to a uniform state representable by a single C4v tensor.
- **Init:** simple-update initialization at the target D (`su_init=True`), matching
  `examples/heisenberg_ipeps_ad.py`.
- **Optimizer:** `optimize_gs_ad` with
  - `gs_c4v=True` (single-tensor C4v, 1-site `unit_cell="1x1"`),
  - `forward_gauge="phase"` + `projector_method="svd"` — **forced** by
    `validate_ctm_for_implicit_ad` (`src/tenax/algorithms/ipeps_ad_policy.py`), which rejects
    the sigma-gauge / eigh / gmres path for implicit AD,
  - `gs_implicit_ad=True` (default; differentiate the CTM fixed point),
  - `gs_recipe="2x2"` (production default),
  - `gs_conv_criterion="grad_norm"`, `gs_grad_norm_tol=1e-5` (#448),
  - fixed `gs_num_steps` budget (CLI; default 100),
  - `return_history=True`.
- **CTM:** defaults (`max_iter=100`, `conv_tol=1e-8`) unless a cell needs a cap; no χ-schedule
  (fixed χ per cell — cleanest scaling signal).

**Correction (2026-06-12, found during implementation):** the existing
`examples/heisenberg_ipeps_ad.py` is **API-stale** — it uses `forward_gauge="sigma"` +
`projector_method="eigh"` + `ad_backward_method="gmres"`, all now rejected by
`validate_ctm_for_implicit_ad`. Its docstring's E/site ≈ −0.6625 came from that invalid
config. The current valid implicit-AD path (phase + svd) converges to **E/site ≈ −0.6602**
at D=2 χ=8 — and CPU validation shows the D=2 energy is **flat across χ ∈ {8,16,24}**
(−0.66019 → −0.66015), i.e. χ-saturated by χ=8 because D=2 ⇒ double-layer bond D²=4. The
study's energy signal therefore lives in **D-scaling**, not χ-scaling.

## Measurement protocol (per (D, χ) cell)

1. Build gate + SU-init `A_init` at D.
2. Time the full `optimize_gs_ad` call (wall-clock around it).
3. From the returned history dict, capture: energy trajectory, `step_times`,
   `jit_compile_time`, `num_steps`, `converged`.
4. Record one row:
   - `D`, `chi`
   - `E_final` — best (lowest) energy in the trajectory
   - `dE` = `E_final − (−0.6694430)`
   - `jit_compile_s` — `jit_compile_time` (cold XLA compile)
   - `total_wall_s` — wall around the call
   - `warm_step_s` — median of `step_times` excluding the first (compile-laden) step
   - `num_steps`, `converged`
   - `eps_T` / smallest-S only if already returned by the optimizer/CTM history without
     extra computation; otherwise omitted (not a blocker for v1)
   - `error` — populated instead of the metrics if the cell raised/OOM'd

## Grid and time-boxing

- **Core grid:** D ∈ {2, 3, 4} × χ ∈ {8, 16, 24, 32}. The full cross-product is run;
  cells with χ < D² are kept (they characterize the under-resolved regime) but expected to
  converge poorly — this is a reported observation, not a filter.
- **Stretch:** D ∈ {5, 6} and χ ∈ {48, 64} — attempted only if the core grid stays tractable.
- **Per-cell budget:** a `--time-budget-s` cap; a cell that exceeds it (or OOMs) is recorded
  with an `error` string and skipped, not allowed to hang the sweep.
- **Hardware:** A100, pinned to a free GPU (`CUDA_VISIBLE_DEVICES`), `JAX_PLATFORMS=cuda,cpu`,
  x64. A CPU smoke on a tiny grid validates the harness before the GPU sweep.
- **Resume:** the JSON is rewritten after every cell; a re-run skips cells already present
  (keyed by `(D, χ)`), so a sweep killed on a large cell keeps its rows.

## Script architecture

`examples/bench_heisenberg_largeD.py`:

- `build_problem(D) -> (gate, A_init, config_template)` — gate, SU-init, base `iPEPSConfig`.
- `run_cell(D, chi, args) -> dict` — one optimization, returns the row (or an error row).
- `main()` — parse CLI, load any existing JSON (resume), loop the grid, checkpoint JSON +
  print the running table after each cell.
- **CLI:** `--D-list`, `--chi-list`, `--gs-steps`, `--time-budget-s`, `--json`,
  `--su-init/--no-su-init`.
- **JSON schema:**
  ```json
  {
    "platform": "...", "device_kind": "...", "x64": true,
    "ref_energy": -0.6694430, "D_list": [...], "chi_list": [...],
    "rows": [ { "D": 2, "chi": 8, "E_final": -0.6625, "dE": 0.0069,
                "jit_compile_s": ..., "total_wall_s": ..., "warm_step_s": ...,
                "num_steps": ..., "converged": true } ]
  }
  ```

## Outputs / deliverables

1. `examples/bench_heisenberg_largeD.py` (the study script).
2. JSON result file(s) under `examples/` (the evidence; checkpointed).
3. A handoff writeup in `docs/superpowers/handoffs/2026-06-12-heisenberg-largeD-characterization.md`:
   - the energy-vs-(D, χ) table and how close to −0.6694430 the dense path gets,
   - runtime + cold-compile scaling vs (D, χ),
   - where the dense wall hits (OOM / compile budget / energy plateau),
   - the conclusion: what U(1) symmetry would unlock to approach the reference scale.

## Correctness guards

- **Sanity anchor:** the D=2 χ=8 cell must land at E/site ≈ −0.6602 (the current valid
  phase+svd implicit-AD value; **not** the stale −0.6625 from the API-invalid example) and the
  D=2 energy must be ~flat across χ. Validated on CPU 2026-06-12 (−0.66019 at χ=8, flat to
  χ=24). If a cell deviates materially, the harness is mis-wired — stop before trusting the sweep.
- **Variational-floor watch:** finite-χ CTM is not a strict variational bound, but any cell
  with `E_final` meaningfully **below** −0.6694430 is flagged in the row/writeup (signals a
  normalization/gauge problem), recorded — not crashed on.

## Testing

No unit tests (consistent with the other `examples/bench_*` / `profile_*` scripts). Validation
is operational: (1) the CPU smoke runs end-to-end on a tiny grid; (2) the D=2 χ=8 sanity
anchor reproduces the known ≈ −0.6625.
