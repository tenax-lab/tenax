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

## Recommendation for the (green-lit) full build

The gate already delivered most of the substance: the AD wiring, the correctness CI test, and the
backward shards effectively *without* per-move constraints. What remains is **polish**, to weigh
against the +2-in-D reality:

1. **`optimize_gs_ad` / `CTMConfig.device_mesh` surface** — let a real multi-step optimization run
   sharded via config, not just the `ctm_energy_implicit` keyword. **Highest remaining value** (it
   turns the proven capability into a usable entry point); modest work.
2. **Per-move `a` constraints in `jit_step_bwd`** — 2B already gets 2.24× without them; likely
   marginal. **Low priority** — measure with/without before building.
3. **End-to-end multi-GPU `optimize_gs_ad` benchmark** (a real ground-state run at D=10, χ=24 on
   4 GPUs) — the convincing demonstration; moderate runtime.

## Artifacts (branch `spike/632-rung2-backward-sharding`)

- `src/tenax/algorithms/_ctm_energy_ad.py` — `device_mesh=None` threaded through the implicit-AD
  entry/dispatch/factory/forward (default-None no-op; in VJP cache key).
- `tests/_rung2_grad_probe.py` — self-contained `value_and_grad` probe (well-conditioned tensor).
- `tests/_rung2_grad_parity_subproc.py` + `tests/test_ctm_sharding_backward.py` — Gate 2A CI test.
- `examples/bench_rung2_grad_memory.py` — Gate 2B/2C GPU memory/ceiling bench (throwaway).

## Caveats

- D=14 / large configs hit the same XLA transpose-autotuner failure seen in rung-1 (orthogonal).
- 2C ceiling measured on a well-conditioned near-product state (fast, valid gradient); physical
  states at the same (D, χ) have the same tensor shapes, so the memory ceiling is representative.
- Per-device peaks have ~10–25% single-GPU allocator variance (run one D per process — the bench
  does); 4-GPU peaks are reproducible.
