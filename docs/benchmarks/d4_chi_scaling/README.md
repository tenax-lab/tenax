# D=4 Heisenberg AFM iPEPS — χ-scaling benchmark results

Production-run outputs for `examples/heisenberg_d4_chi_scaling.py` (PR #646), run on
4×A100-SXM4-80GB and completed **2026-06-28**. Spin-1/2 antiferromagnetic
Heisenberg model on the infinite square lattice at bond dimension **D=4**.

The driver optimizes the variational state **once** (single-GPU, χ_opt=32,
implicit-AD + C4v + grad-spike guard), caches `A_opt.pkl`, then runs that **fixed**
state through forward CTM at each χ for each `n_devices ∈ {1,2,4}`. The energy column
is device-independent (a correctness cross-check) and gives the convergence curve;
the timing/memory columns at identical `(state, χ)` give the multi-GPU speedup curve.

All 21 cells (7 χ × 3 device counts) completed `ok` — zero OOM, zero errors.

## Key findings

- **Convergence:** E/site saturates at **−0.663398 by χ≈32** (flat to 6 digits
  beyond). The residual **+0.006 vs the QMC reference −0.669437 is the D=4
  variational floor**, not under-convergence. So χ≈32 is sufficient at D=4.
- **Multi-GPU:** numerically exact (ΔE vs 1-GPU ≤ 7e-8, machine-epsilon at large χ)
  but **below the crossover at this scale** — 2-GPU is break-even (0.99–1.01×, flat
  in χ), 4-GPU markedly slower but improving monotonically with χ (0.21× at χ=16 →
  0.49× at χ=96) as the sharding overhead amortizes. Peak memory ≤ 2.8 GB even at
  χ=128, so dense D=4 CTM at χ≤128 is too small for sharding to pay off; the
  crossover needs larger D or χ≫128, and the 4-GPU trend is consistent with one
  existing above this range rather than with sharding being unusable.

## Corrections

**#781 (2026-08-06) — the per-sweep timings were inflated up to 3×.** The driver
divided elapsed time by `CTMLoopResult.iterations`, which on the `plateau_patience`
bail reported the best-metric iteration rather than the sweeps performed. All 21
cells here exited on that bail, so every `ms_per_sweep` was short by
`plateau_patience=20` sweeps in the denominator — by a *different* factor per cell
(1.00–3.00×), so it did not cancel in the speedup ratio.

`performance.md` is recomputed from the recorded `total_s`; the per-cell JSONs and
`results.csv` keep their original values as the record, so **do not read
`ms_per_sweep` out of them**. The 2-GPU speedup column moved from a published
0.61–1.32× to a flat 0.99–1.01× — the apparent χ-dependence there was entirely the
bug. Direction of both findings is unchanged. Energies, `peak_gb` and the
convergence table are unaffected.

**Open (#780):** the `converged` column is `false` in every cell, which is a
convergence-*criterion* artifact rather than a statement about the environments —
see `convergence.md`.

See the [merged PR #646](https://github.com/tenax-lab/tenax/pull/646) discussion for
the full write-up.

## Files

| File | Contents |
|------|----------|
| `convergence.md` | E/site vs χ table (vs QMC) |
| `performance.md` | per-sweep CTM cost, peak GB, speedup vs 1-GPU, grouped by device count |
| `results.csv` | all 21 cells, machine-readable — `ms_per_sweep`/`n_sweeps` as recorded, i.e. inflated (#781) |
| `D4_chi{χ}_n{devices}.json` | per-cell raw result (E, timing, memory, convergence flag); same #781 caveat |
| `*.png` | E-vs-χ (with QMC line), ms/sweep, speedup, peak-GB plots |
| `A_opt.pkl` | the optimized D=4 tensor (host/numpy leaves) — re-scan without re-optimizing |
| `optimize_status.json` | optimize-phase exit status |

Optimizer checkpoints (`ckpt_opt/`, ~1.7 MB binary) and the run log are intentionally
**not** committed (large, regenerable).

## Reproduce

```bash
# Full sweep (optimize once at χ_opt=32, then scan χ × {1,2,4} GPU). ~19 h on 4×A100.
uv run python examples/heisenberg_d4_chi_scaling.py --outdir runs/d4_chi_scaling

# Re-run the χ-scan only, reusing this committed A_opt.pkl (skips the ~18 h optimize):
mkdir -p runs/d4_chi_scaling && cp docs/benchmarks/d4_chi_scaling/A_opt.pkl runs/d4_chi_scaling/
uv run python examples/heisenberg_d4_chi_scaling.py --outdir runs/d4_chi_scaling

# Quick end-to-end validation (tiny):
uv run python examples/heisenberg_d4_chi_scaling.py --smoke
```
