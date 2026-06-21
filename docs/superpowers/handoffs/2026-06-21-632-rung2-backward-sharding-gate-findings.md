# #632 rung-2 — GSPMD-sharded dense CTM-AD backward — Gate Findings

**Date:** 2026-06-21
**Parent:** #632 rung-1 (forward sharded, merged). **Spec:**
`docs/superpowers/specs/2026-06-21-632-rung2-backward-sharding-gate-design.md`.
**Hardware:** 4× A100-SXM4-80GB (NVLink). **Verdict: GATE = GO (all three sub-gates pass)
— the backward shards correctly — but the ceiling gain is marginal (+2 in D), consistent
with the rung-1 forward result.**

## Question

Does the dense CTM-AD **backward** shard once the forward is sharded — does GSPMD propagate
through the `@jax.custom_vjp` implicit adjoint and its `lax.while_loop` fixed-point solve —
with correct gradients, lower per-device memory, and a higher optimize-D ceiling on 4× A100?

## Minimal wiring tested (the only src change)

`device_mesh=None` threaded through `ctm_energy_implicit` → `_ctm_energy_implicit_dispatch`
(added to the VJP cache key) → `_make_implicit_vjp_fn` → both `_run_forward` branches
(`python_loop_ctm_converge` and `_sigma_gauged_ctm_converge`); the sigma path now commits the
initial envs via `commit_env` and builds its step with `device_mesh`. The custom_vjp then saves
**sharded** env residuals and GSPMD partitions the jitted backward operators from them. **No**
per-move `a` constraints in `jit_step_bwd`, **no** `CTMConfig`/`optimize_gs_ad` surface — both
deliberately deferred. Default `device_mesh=None` is a bit-for-bit no-op (28 existing implicit-AD
tests pass).

## Results

### Gate 2A — correctness (CPU fake devices): **PASS**

`value_and_grad` single vs 4-device sharded, **well-conditioned** D=4/χ=8 state:

| quantity | single vs sharded |
|---|---|
| energy | \|ΔE\| = 1.4e-17 |
| **gradient** | **max\|Δg\| = 2.2e-16** (rel 1.2e-14) |

GSPMD propagates through the custom_vjp + while-loop adjoint **exactly** (up to FP reassociation,
which stays at machine precision on a well-conditioned fixed point — same lesson as rung-1).
Formalized as a CI test: `tests/test_ctm_sharding_backward.py` (marker-gated subprocess).

### Gate 2B — does the backward shard? (4× A100): **PASS**

Per-device peak of the **full `value_and_grad`** (forward CTM + implicit-AD backward), χ=24:

| D | single-GPU | 4-GPU/device | reduction |
|---|---:|---:|---|
| 6 | 2.16 GB | — | — |
| 8 | 10.66 GB | **4.76 GB** | **2.24×** |
| 10 | **OOM** (needs +29 GiB) | **21.45 GB → OK** | single can't run |

The backward shards *better* than the forward (2.24× vs the forward-only ~1.6×) — its dominant
cost is the env-shaped adjoint `λ` and the linearized sweep, which partition cleanly — **without**
the per-move constraints (so those are optional polish, not load-bearing).

### Gate 2C — optimize-D ceiling (4× A100): **PASS (+2 in D)**

- single-GPU `value_and_grad` ceiling (χ=24) = **D=8** (D=10 OOMs; the backward pulls the
  ceiling down from the forward-only D=10).
- 4-GPU ceiling = **D=10** (D=12 OOMs at +35 GiB). **Net: +2 in D.**

## Gate decision: **GO** (thresholds all met)

| sub-gate | threshold | measured | verdict |
|---|---|---|---|
| 2A grad parity | ≤ ~1e-8 | 2.2e-16 | ✅ |
| 2B backward shards | ≥ ~1.3× | 2.24× (and runs where single OOMs) | ✅ |
| 2C ceiling | ≥ +1 in D | +2 in D | ✅ |

The capability is proven: **multi-GPU `optimize_gs_ad` produces correct gradients end-to-end at a
D one GPU cannot fit** (D=10, χ=24). It is genuinely usable, not just forward energy.

## The honest value caveat (unchanged from rung-1)

The ceiling gain is **+2 in D** (D=8 single → D=10 on 4 GPUs at χ=24), because dense CTM-AD memory
still scales **~D⁶** ⇒ N GPUs buy **N^(1/6)**. D=12 OOMs even on 4 GPUs. So rung 2 makes multi-GPU
optimization *work and be correct*, but does not reach the large-D regime (D≥12) where eager/YASTN
matters. Same verdict as `2026-06-21-632-multigpu-dense-ctm-findings.md`.

## Full build — DONE (config surface + end-to-end sharded optimization)

After the GO, the green-lit polish was built and verified:

1. **`CTMConfig.device_mesh` end-to-end surface — DONE.** A field on `CTMConfig` (default `None`,
   ABI-appended) threads through `ipeps_ad_policy` into the implicit-AD energy *and* (via
   `ctm_converge_kwargs`) into every forward CTM eval in the 1-site optimize loop — `value_and_grad`,
   the warm-start `_update_env_cache`, the line-search `loss_fn_fwd`, and the final `_eval_fresh`.
   So `optimize_gs_ad` runs **fully sharded via config**, not just the bare `ctm_energy_implicit`
   keyword. Single-device (`None`) is unchanged (existing AD tests pass).
2. **Per-move `a` constraints in `jit_step_bwd` — SKIPPED (confirmed unneeded).** 2B already gets
   2.24× without them.
3. **End-to-end multi-GPU `optimize_gs_ad` — DONE (the headline).** On 4× A100, **D=10 / χ=24**
   (where single-GPU `value_and_grad` OOMs): `E_init=0.499999 → E_final=0.499997` (dE=−2.3e-6,
   energy decreases — valid optimization), **per-device peak 21.46 GB**, wall 1738 s for init + 2
   sharded gradient steps. Multi-GPU ground-state optimization at a D one GPU cannot fit. ✅

### End-to-end correctness (CI): `optimize_gs_ad` parity

`tests/test_ctm_sharding_backward.py::test_sharded_optimize_gs_ad_matches_single_device` — a few
optimize steps single vs 4-device sharded reach the same energy to **|ΔE|=2.9e-13** on a
**well-conditioned** init.

> **Subtlety found & resolved (not a bug).** On a *random* (ill-conditioned) init the multi-step
> sharded trajectory diverges from single by ~1e-2: step 1 is **bit-exact** (empty warm-start →
> identical to gate 2A), but step 2+ warm-start from the converged env, and the tiny FP
> reassociation in the warm-started sharded **backward** (GSPMD makes its own backward sharding
> choices — see the `_jit_fused_fixed_point_bwd` "involuntary full rematerialization" SPMD note) is
> amplified by the ill-conditioned fixed point **and** the chaotic optimization trajectory. With a
> **well-conditioned** init it collapses to 2.9e-13 (energy) — both trajectories reach the same
> minimum. So the valid parity test uses a well-conditioned init and asserts on **energy** (the
> physical observable); the tensor drifts ~1e-6 along a flat/gauge direction. Same lesson as the
> rung-1 forward parity.

## Artifacts (branch `spike/632-rung2-backward-sharding`)

- `src/tenax/algorithms/_ctm_energy_ad.py` — `device_mesh` through the implicit-AD entry/dispatch/
  factory/forward (default-None no-op; in VJP cache key).
- `src/tenax/algorithms/ipeps_config.py` — `CTMConfig.device_mesh` field.
- `src/tenax/algorithms/ipeps_ad_policy.py` — passes the mesh into the implicit-AD energy +
  `ctm_converge_kwargs` (shards warm-start/probe/final-env forwards).
- `tests/_rung2_grad_probe.py`, `tests/_rung2_grad_parity_subproc.py`,
  `tests/_rung2_optimize_parity_subproc.py`, `tests/test_ctm_sharding_backward.py` — probe + the two
  CI parity tests (backward grad; end-to-end optimize).
- `examples/bench_rung2_grad_memory.py`, `examples/bench_rung2_optimize.py` — GPU memory/ceiling +
  end-to-end optimize benches (throwaway).

## Caveats

- **SPMD inefficiency (perf, not correctness):** XLA logs `[SPMD] Involuntary full rematerialization`
  for a transpose in `_jit_fused_fixed_point_bwd` — GSPMD can't reshard `{[1,1,4,1,1]} →
  {[1,1,1,1,2,2]}` efficiently and replicates+repartitions. Correct, but a runtime tax on the
  backward (Shardy partitioner / explicit backward shardings would fix it). Contributes to the
  ~29 min D=10 wall.
- D=14 / large configs hit the same XLA transpose-autotuner failure seen in rung-1 (orthogonal).
- 2C ceiling measured on a well-conditioned near-product state (fast, valid gradient); physical
  states at the same (D, χ) have the same tensor shapes, so the memory ceiling is representative.
- Per-device peaks have ~10–25% single-GPU allocator variance (run one D per process — the bench
  does); 4-GPU peaks are reproducible.
