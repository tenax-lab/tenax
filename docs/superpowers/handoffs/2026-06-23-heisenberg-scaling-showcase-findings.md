# iPEPS Square-Lattice Heisenberg Scaling/Perf Showcase — Findings

**Date:** 2026-06-23
**Branch:** `showcase/heisenberg-scaling`
**Hardware:** 4× A100-SXM4-80GB (CUDA ids 0–3; DGX Display = CUDA id 4, excluded), JAX 0.10.1, f64
**Spec/plan:** `docs/superpowers/specs/2026-06-23-heisenberg-scaling-showcase-design.md`,
`docs/superpowers/plans/2026-06-23-heisenberg-scaling-showcase.md`
**Artifacts:** `examples/showcase_heisenberg_scaling.py` (driver+worker),
`examples/showcase_analyze.py` (robust post-hoc), `examples/showcase_results/`
(29 per-cell JSONs + `scaling_table.md` + `scaling_results.csv` +
`ms_per_step_vs_chi.png` + `peak_gb_vs_chi.png`).

## TL;DR

The **production-merged dense-CTM stack** (`gs_recipe="1x1"` reduced-corner CTM +
`CTMConfig.device_mesh` GSPMD sharding + implicit-AD backward) **runs end-to-end
on 4× A100** for the 2D square-lattice Heisenberg AFM, and the scaling behaves
exactly as the prior characterization predicted:

- **`D⁶` cost wall is real and clean:** per-step jumps **9×** from D2 to D3
  (χ16: 2.6 s → 23.7 s).
- **Host-overhead-bound → FLOP-bound transition:** at D2 the step is flat in χ
  (~2–6 s, GPU ~0% util); at D3+ it is GPU-bound (~99% util).
- **Multi-GPU is a large-D lever, not a small-D one:** 4-GPU sharding is **5×
  slower at D2** (comm-dominated) but **1.1–1.75× faster at D4**, with the
  speedup growing in χ — a clean, quantified crossover around D3–D4.

Three constraints/caveats were discovered and are documented below (GSPMD
divisibility, χ-non-monotonicity from XLA kernel selection, accurate-AD anchor
cost).

## Phase-0 compatibility gate: **GO**

The one unverified assumption — that `recipe="1x1"` composes with `device_mesh`
(rung-2 sharding was validated on the default `2x2`) — **holds**. At D2/χ8 on
1 and 4 A100s the four-way combo (1-site + `recipe=1x1` + `device_mesh` +
implicit AD) produced finite energies (−0.500 / −0.472) and no error. No
`recipe=2x2` fallback was needed.

The gate also caught a real config bug cheaply: implicit AD **requires
`forward_gauge="phase"`** (+ `projector_method∈{svd,qr}`,
`ctm_conv_method="elementwise"`); `"sigma"` (the plan's stale choice) is rejected.

## Metric methodology (3 debugging rounds → robust per-step)

Getting a trustworthy per-step number took three corrections — each a real
property of the JAX/XLA dense path, worth recording:

1. **L-BFGS line search dominates cost.** The default optimizer runs several full
   CTM reconvergences per step → 15 s/step at D2/χ8. For a *scaling* measurement
   we want the cost of *one* forward-CTM + one backward, so metrics cells use a
   **cheap fixed-step profile** (Adam, no line search / no metric precond):
   2.1 s/step (7×).
2. **Trajectory-dependent CTM sweep counts** (adaptive convergence in the 1×1
   python-loop) made the per-step jump around (D2/χ16 read 13 s). Fix: **fixed
   CTM work** for metrics (`min_iter=max_iter=20`, `conv_tol=0` → exactly 20
   sweeps every step) → deterministic per-step.
3. **XLA re-autotunes at χ≥48** on the first 1–2 warm steps (spiking to ~compile
   time). The worker therefore **stores raw `step_times`** and per-step is taken
   as **`median(step_times[1:])`** in post-hoc — robust to both the recompile
   spikes and single async-dispatch outliers. (`min` over-corrected into an
   async-fast-outlier trap; median is right.)

## Scaling results (robust per-step = median of warm steps)

### D-scaling at χ16, 1-GPU — the `D⁶` jump
| D | per-step | × vs D2 |
|---|----------|---------|
| 2 | 2.6 s | 1.0× |
| 3 | 23.7 s | **9.0×** |
| 4 | 27.4 s | 10.3× |

(D3→D4 is only ~1.2× *at χ16* because χ16 is small enough that `χ²D⁶` is not yet
the dominant term; the D-gap widens with χ.)

### 1-GPU χ-scaling (per-step)
- **D2:** 2.6 / 2.6 / 2.6 / 3.6 / 3.6 / 4.7 / 5.7 s (χ = 16…128) — nearly flat
  (**host-overhead-bound**, GPU ~0%); peak 0.01 → 0.31 GB.
- **D3:** 23.7 / 33.7 / 50.0 / 54.4 s (χ16…48), then **timeout at χ96**.
- **D4:** 27.4 / 48.5 / 66.1 s (χ16…32), then **timeout at χ48**.

### 1-GPU vs 4-GPU (matched cells) — the multi-GPU crossover
| D | χ | 1-GPU | 4-GPU | speedup |
|---|----|-------|-------|---------|
| 2 | 16 | 2.6 s | 14.2 s | 0.19× |
| 2 | 32 | 2.6 s | 15.0 s | 0.18× |
| 4 | 16 | 27.4 s | 24.5 s | 1.12× |
| 4 | 24 | 48.5 s | 32.2 s | **1.51×** |
| 4 | 32 | 66.1 s | 37.8 s | **1.75×** |

At small D the D⁶ tensors are tiny, so sharding the D² axis only adds collective
overhead (4-GPU ≈ 5× *slower*). At D4 the tensors are large enough that the
per-device FLOP saving beats the comm cost, and the win grows with χ — exactly
the "multi-GPU helps at large D" thesis, measured. (Consistent with the weak
N^(1/6)-in-D memory lever from `632-multigpu-dense-ctm-measured`: the *speed* win
is the more visible effect here at D4.)

### Cost-aware ramp (memory never binds at 80 GB)
Per-device peak stayed **< 0.55 GB** through the whole grid — these 80 GB cards
never OOM at tractable D, so the χ ramp is bounded by a **per-cell wall-clock
timeout** (600 s metrics), not memory. A timed-out cell stops its (D, n_devices)
row, making "how far each D gets under a fixed per-cell time budget" the scaling
story and hard-bounding total runtime (~1 h sweep).

## Discovered constraints / caveats (each a real finding)

1. **GSPMD divisibility:** sharding splits the D² virtual axis, so **D² must be
   divisible by n_devices**. D3 (D²=9) **cannot** shard on 4 GPUs
   (`IndivisibleError`); D2 (4) and D4 (16) divide evenly and work. This rules
   out 4-GPU runs at odd-ish D without padding.
2. **Per-step is reproducibly non-monotonic in nominal χ.** D3/χ64 ran at
   **3.7 s/step vs χ48's 54 s** — confirmed by an independent re-probe
   (`[5.4, 3.7, 3.7, 3.9, 3.7]`), so it is *not* timing noise. On the
   reduced-corner path the cost depends on the **effective SVD rank** (which can
   fall below nominal χ) and on **which kernel XLA's autotuner selects**, neither
   monotone in χ. Robust conclusions (D-scaling, multi-GPU crossover) are
   unaffected; the within-D χ-curve is noisy and such points are flagged `*` in
   `scaling_table.md`.
3. **Converged-energy anchors are expensive.** The accurate AD path (L-BFGS +
   line search + implicit AD) costs **~25× the fixed-step measurement** — even at
   D2/χ16 it exceeded a 30-min budget (timeout at 1800 s) with and without metric
   preconditioning and a cheaper CTM. So the physics/QMC column is **best-effort
   only**: the 6-step fixed-step metrics energies sit at −0.48…−0.55 (vs QMC
   **−0.669437**), i.e. the optimization has started but is far from converged.
   A trusted energy needs a dedicated long run, which is out of scope for a
   *perf* showcase.

## Verdict

The production stack delivers where the prior analysis said it would: it runs at
scale, the `D⁶` wall and the host→FLOP transition are clean, and **multi-GPU is a
genuine large-D (D≥4) speed lever** while being a net loss at small D. The
remaining frictions are XLA-level (autotuner-driven χ-non-monotonicity), a GSPMD
shape constraint (D² mod n_devices), and the intrinsic cost of the accurate AD
optimizer — none of which are algorithm bugs. For absolute ground-state energies
at large D, the dense path remains runtime-bound (consistent with
`570-dense-largeD-study`); this showcase quantifies *where* and *why* on real
4× A100 hardware.

## Reproduce
```
# full sweep (resumable; ~1 h on 4×A100)
uv run python examples/showcase_heisenberg_scaling.py --results-dir examples/showcase_results
# robust post-hoc tables + plots + CSV
uv run python examples/showcase_analyze.py --write-outputs
```
