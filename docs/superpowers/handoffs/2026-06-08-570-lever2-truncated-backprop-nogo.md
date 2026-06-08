# #570 lever-2 (truncated backprop) — NO-GO for the compile wall (accurate, but doesn't move it)

**Date:** 2026-06-08 (A100) · **Follows:** [`2026-06-07-570-svd-vjp-compile-finding.md`](2026-06-07-570-svd-vjp-compile-finding.md) · **Issue:** #570

> **Correction (2026-06-08).** Where this doc says "**dense** block-SVD VJP" or "**dense**-corner
> SVD fallback," read **block-sparse (per-sector) SVD VJP**: the production 2×2 path already
> differentiates a per-sector block-sparse SVD (`_compute_2x2_projector_symmetric` →
> `_truncated_svd_symmetric_traced`, Issue #435), not a dense corner SVD. Consequently
> **"lever-3" is moot — already implemented** (see `2026-06-08-570-mechanism-correction.md`).
> The lever-2 NO-GO conclusion and all measurements are **unaffected**.
> Further (PR #589): the per-sweep cost is **#566 structural emission** (block-pack ~60% +
> gauge-fix ~25%, decomposition ~0%) — read "block-SVD VJP" here as that structural emission,
> not the decomposition math. See `2026-06-08-570-relocalized-not-decomposition.md`.

## TL;DR

The #570 SVD-VJP finding recommended **lever-2 (truncated backprop / TBPTT)** as the
first move: differentiate through only the last few CTM sweeps instead of the full
fixed-point adjoint, to shrink the backward jaxpr independent of the decomposition. It is
already implemented (`ctm_energy_explicit`, `backward_steps`, #506). Measured on A100
(D=2, χ=12, fermionic):

- **Accuracy: excellent.** Differentiating a single sweep (K=1 / S=1) gives a gradient with
  **cos = 0.9994** (rel-err 3.4%) vs the production implicit adjoint; truncation error vs
  full backprop vanishes by **K=2–4** (cos → 1.0). Validates the Corboz premise.
- **Compile: NO-GO.** Truncation does **not** reduce the compile wall. Best case (S=1,
  warmup=11) is **0.88× implicit** (249.7 s vs 283.8 s — 12 % faster); **S ≥ 2 is *worse***
  (1.27–1.28×). The explicit path emits **1254–1257 compile units** vs the implicit
  adjoint's **236**.

**Why:** the per-sweep **block-sparse (per-sector) SVD VJP** is irreducible in *both* paths — any gradient
needs ≥1 differentiated block-sparse SVD sweep (~250 s at D=2/χ=12), and the explicit unroll
only adds forward-unroll compile on top (it traces the *whole* forward into one XLA module;
`stop_gradient` removes backward ops but keeps the forward sweeps, while the implicit path
runs the forward as a host loop over **one** cached compiled step). Truncated backprop
changes *how many* sweeps are differentiated, not the per-sweep SVD-VJP cost.

This makes **three levers, all NO-GO for the compile wall** — #200 contraction backend,
#570 lever-1 (QR projector), #570 lever-2 (truncated backprop) — all because the wall is the
per-sweep block-sparse (per-sector) SVD VJP emission, which none of them touches.

## Method

`examples/profile_570_truncated_backprop.py` (new). Two modes, both cold-compile + gradient
parity vs gold (`jax_log_compiles` capture, fresh persistent-cache dir per cold call):

- **K-sweep** (fixed unroll, truncate the backward): `explicit_steps=8`, sweep
  `backward_steps=K`. Gold = explicit-full (matched forward) and implicit adjoint.
- **steps-sweep / COMPILE-LEVER** (`--steps-list`): sweep the *differentiated unroll*
  `backprop_steps=S` with `warmup = total_sweeps − S` (forward-only warmup absorbs the rest),
  full backward each. This is the config that *could* shrink compile. Gold = implicit adjoint.

The forward is identical for every K/S (same total sweeps), so **energy is invariant**
(confirmed: −0.202734 across all rows); the only variable is the gradient.

## Results (A100, D=2, χ=12, 16 blocks, x64)

### K-sweep — truncate backward of a fixed 8-sweep unroll (warmup=4)

implicit adjoint gold: E=−0.203204, compile 290.7 s, 236 compiles · explicit-full:
360.4 s, 1257 compiles · rel-err(explicit-full, adjoint)=2.44e-2, cos=0.99970.

| K | compile (s) | n_compiles | rel-err vs full | cos vs full | rel-err vs adjoint | cos vs adjoint |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 302.6 | 1255 | 2.87e-2 | 0.99961 | 3.38e-2 | 0.99944 |
| 2 | 413.0 | 1258 | 5.42e-3 | 0.99999 | 2.54e-2 | 0.99968 |
| 4 | 415.5 | 1258 | 1.85e-4 | 1.00000 | 2.45e-2 | 0.99970 |
| 8 | 366.3 | 1257 | 0 | 1.00000 | 2.44e-2 | 0.99970 |

Compile **flat** (302–415 s), not ∝K — `backward_steps` truncation doesn't touch the
forward-unroll-dominated compile. Truncation error →0 by K=2–4; the residual 2.4 % vs the
adjoint is the explicit-forward-vs-fixed-point gap (present even at K=8), not truncation.

### steps-sweep — shorten the differentiated unroll (the compile lever), total=12

implicit adjoint gold: E=−0.203204, compile **283.8 s**, 236 compiles.

| S | warmup | compile (s) | n_compiles | vs implicit | rel-err vs adjoint | cos vs adjoint |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11 | 249.7 | 1254 | **0.88×** | 3.38e-2 | 0.99944 |
| 2 | 10 | 360.5 | 1257 | 1.27× | 2.54e-2 | 0.99968 |
| 4 | 8  | 361.4 | 1257 | 1.27× | 2.45e-2 | 0.99970 |
| 8 | 4  | 363.1 | 1257 | 1.28× | 2.44e-2 | 0.99970 |

Even S=1 only reaches 0.88× implicit; S≥2 is worse. The explicit unroll's ~1254 compile
units (vs 236) is the forward-unroll overhead the implicit host-loop avoids.

Raw: `570_results/profile_570_tbptt_a100.json`, `570_results/profile_570_tbptt_stepsweep_a100.json`.

## Scope / caveat (important)

This measured **compile time + gradient accuracy** — the wall P-B4 identified (~549 s at
D=4). It did **not** measure **warm-step runtime or peak memory**, which is where the
truncated-backprop literature (Corboz QR-CTMRG) claims its win at large χ/D (avoiding the
implicit-solve iterations and the full-history adjoint memory). So lever-2 is a NO-GO **for
the compile wall specifically**; its runtime/memory benefit at large D is plausible but
unmeasured here. If the goal shifts from compile to warm-step, re-open with a runtime A/B
(this rig + warm-step timing). For the compile wall as posed, it does not help.

## Conclusion for #570

The compile wall is the **per-sweep block-sparse (per-sector) SVD VJP**, and it is
irreducible under both the implicit adjoint and any explicit unroll (you always
differentiate ≥1 block-SVD sweep). Projector-swap (lever-1) and backprop-truncation
(lever-2) both leave that per-sweep cost intact. **Lever-3 (a block-sparse per-sector
AD-traced SVD VJP) is NOT a new lever — the production 2×2 path already does exactly this**
(`_compute_2x2_projector_symmetric` → `_truncated_svd_symmetric_traced`; the "Task 2.2"
dense fallback in `_ctm_projector.py` is in the unused 1×1 recipe). So the per-sector
block-SVD VJP cost *is* the wall. The remaining structural lever is **batching the
equal-shaped per-sector SVD-VJP units** into one vmapped graph (the #566/#569
`TENAX_BATCH_BLOCKSPARSE` axis — built, benchmarked for runtime ["never a net win" through
D=6], compile effect unmeasured). Absent a compile win there, the wall appears intrinsic to
differentiating a block-sparse SVD in XLA.

## Artifacts

- `examples/profile_570_truncated_backprop.py` — TBPTT parity/compile rig (K-sweep + steps-sweep)
- `570_results/profile_570_tbptt_a100.json` — K-sweep raw
- `570_results/profile_570_tbptt_stepsweep_a100.json` — steps-sweep raw
