# Design: GSPMD-sharded dense CTM-AD backward — rung-2 feasibility gate

**Date:** 2026-06-21
**Status:** Approved (brainstorming) — gate-first; full build deferred behind the GO.
**Parent:** #632 rung-1 (forward CTM sharded, merged). **Spec:**
`docs/superpowers/specs/2026-06-19-gspmd-sharded-dense-ctm-large-d-design.md`.
**Rung-1 result that motivates a gate, not a blind build:**
`docs/superpowers/handoffs/2026-06-21-632-multigpu-dense-ctm-findings.md` — forward sharding
gives only ~1.6× per-device memory, single→4-GPU forward ceiling D=10→12; backward is ~2–4×
forward, so the *optimization* ceiling is likely only ~D=10–11 on 4 GPUs. Marginal D gain, but
rung 2 is what makes multi-GPU `optimize_gs_ad` produce an actual ground state (not just energy).

## The one question this gate answers

Does the dense CTM-AD **backward** shard once the forward is sharded — i.e. does GSPMD propagate
through the `@jax.custom_vjp` adjoint (`_ctm_energy_ad.py`) and its `lax.while_loop` fixed-point
solve — producing **correct gradients** at **lower per-device memory**, and what **optimize-D
ceiling** does that buy on the 4× A100-80GB box? Answer GO/NO-GO before funding the full build.

## Why this is the binding uncertainty

The AD forward already calls `python_loop_ctm_converge` (`_ctm_energy_ad.py:918`), which has the
rung-1 `device_mesh` hook. If we thread a mesh in, `f_fwd` saves **sharded** env residuals, and
the jitted backward operators (`_jit_apply_Jt`, `_jit_fused_fixed_point_bwd`) receive sharded
inputs. GSPMD *should* then partition the backward with no further change — **but** the adjoint
runs a `lax.while_loop` whose carry must stay sharded across iterations, and `jit_step_bwd` re-runs
a CTM sweep **without** the rung-1 per-move `a` constraints. Whether the backward shards "for free"
is unknown and not safely inferable — hence measure it.

## Minimal wiring (the ONLY src change in the gate)

Thread an optional `device_mesh=None` through the implicit-AD entry to the forward inside the
custom_vjp:

- `ctm_energy_implicit(..., device_mesh=None)` (`_ctm_energy_ad.py:337`)
- → `_ctm_energy_implicit_dispatch(...)` / `_make_implicit_vjp_fn(..., device_mesh=None)`
- → inside `_run_forward`, pass `device_mesh=device_mesh` to `python_loop_ctm_converge`.

`device_mesh` is a static (non-traced) closure value, like rung-1. Default `None` → today's path,
bit-for-bit. **No** `CTMConfig`/`optimize_gs_ad` surface, **no** per-move `a` constraints in
`jit_step_bwd` — both deferred to the post-gate build. This isolates the GSPMD-propagation question
and keeps the gate to ~one wiring change + a probe.

## Measurements

A standalone grad probe (`examples/_rung2_grad_probe.py`, throwaway) builds a 1-site dense iPEPS
and computes `value_and_grad` of the CTM-AD energy via `ctm_energy_implicit`, optionally on a mesh.
It builds its **own** well-conditioned (near-product) tensor — it does NOT depend on the
unmerged-PR probe helpers.

- **2A — correctness (CPU fake devices, CI-able).** `value_and_grad` single vs sharded on a
  **well-conditioned** state; assert max gradient-leaf abs error ≤ ~1e-8 (well-conditioned so FP
  reassociation stays at machine precision — per the rung-1 parity lesson). Run under
  `--xla_force_host_platform_device_count=4`, `JAX_PLATFORMS=cpu`, in a subprocess. **Load-bearing.**
- **2B — does the backward shard? (4× A100).** Per-device `peak_bytes_in_use` of `value_and_grad`,
  single-device vs sharded, at a fixed D (e.g. D=8, χ=24). A sharded per-device peak materially
  below single-device demonstrates GSPMD partitioned the backward.
- **2C — optimize-D ceiling (4× A100).** Largest D where `value_and_grad` runs, single vs 4 GPUs.

## GO / NO-GO

- **GO** ⟺ (2A) grad parity ≤ ~1e-8 on a well-conditioned state **and** (2B) sharded backward
  per-device memory materially < single (≥ ~1.3×) **and** (2C) ceiling lifts ≥ +1 in D.
  → green-light the full build: `CTMConfig.device_mesh` end-to-end surface, per-move `a` constraints
  in `jit_step_bwd`, gradient-parity CI test, and a D=10–12 multi-GPU `optimize_gs_ad` benchmark.
- **NO-GO** ⟺ backward replicates (while-loop carry de-shards / GSPMD doesn't propagate), or grads
  wrong, or no ceiling gain. → document; forward-only (rung-1) stands; large-D stays eager/YASTN.

## Components

- **Modify** `src/tenax/algorithms/_ctm_energy_ad.py` — `device_mesh=None` kwarg on
  `ctm_energy_implicit`, threaded through the dispatch/factory to the `_run_forward`
  `python_loop_ctm_converge` call. (~3 small edits, default-None no-op.)
- **Create** `examples/_rung2_grad_probe.py` — throwaway `value_and_grad` probe (single vs mesh),
  well-conditioned tensor, prints grad-parity Δ + per-device peak; CLI `--D --chi --shard --mesh-n`.
- **Create** `tests/_rung2_grad_parity_subproc.py` + a marker-gated test (2A), mirroring the
  rung-1 fake-device subprocess pattern.

## Testing & success criteria

1. **No-regression:** `device_mesh=None` path is bit-identical to today (guard via an existing
   dense CTM-AD test + the probe with mesh off vs absent).
2. **Gate 2A** is the CI-able correctness check (subprocess, fake CPU devices).
3. **Gates 2B/2C** are manual GPU measurements (the deliverable), recorded in a findings handoff.

## Out of scope (deferred to the post-gate build, only if GO)

- `CTMConfig`/`optimize_gs_ad` config-level `device_mesh` surface.
- Per-move `a` `with_sharding_constraint` inside `jit_step_bwd` (the backward sweep).
- Eager-GMRES adjoint fallback path sharding (gate uses the default `fixed_point` adjoint).
- Multisite 2×2 unit cell; distributed SVD (rung 3); multi-node; throughput tuning.
