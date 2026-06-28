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
  but **below the crossover at this scale** — 2-GPU ≈ break-even (median ~1.0×),
  4-GPU markedly slower (0.19–0.61×). Peak memory ≤ 2.8 GB even at χ=128, so dense
  D=4 CTM at χ≤128 is too small for sharding to pay off; the crossover needs larger
  D or χ≫128.

See the [merged PR #646](https://github.com/tenax-lab/tenax/pull/646) discussion for
the full write-up.

## Files

| File | Contents |
|------|----------|
| `convergence.md` | E/site vs χ table (vs QMC) |
| `performance.md` | per-sweep CTM cost, peak GB, speedup vs 1-GPU, grouped by device count |
| `results.csv` | all 21 cells, machine-readable |
| `D4_chi{χ}_n{devices}.json` | per-cell raw result (E, timing, memory, convergence flag) |
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
