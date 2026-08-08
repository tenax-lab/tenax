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

> ## ⚠️ The energies on this page are superseded — #747
>
> The state these numbers describe was optimized against a **collapsed**
> environment. `gs_recipe` was hard-coded to `1x1`, which drives the CTM
> environment to rank-1 corners — a `chi_eff = 1` mean-field boundary (#723,
> #726) — so the `A_opt.pkl` committed here is a badly-optimized state, not the
> D=4 variational optimum.
>
> **The χ-scan itself is sound.** It runs through `python_loop_ctm_converge`,
> which has defaulted to `recipe="2x2"` since PR #597, so the scan methodology,
> the device-independence cross-check, and every **multi-GPU timing and memory**
> finding below are unaffected. What does not survive is the **absolute energy**
> and any reading of it as a truncation error.
>
> **Superseded by Run 2** (#747 comment 4, 2026-08-05), which re-optimized at
> `gs_recipe="2x2"` and confirmed `corner_rank == χ` in all seven cells:
>
> | | this page (`1x1`) | Run 2 (`2x2`) |
> |---|---|---|
> | E/site at χ=128 | −0.6633981 | **−0.6639345178** |
> | err vs QMC | +6.04e-03 | **+5.50e-03** |
>
> Run 2's figure is **itself an upper bound**: its optimization was stopped at
> step 14 with `E_best` flat for ~4 h of wall clock, not run to convergence.
> Neither number is the D=4 truncation error.
>
> The data files here are left unmodified as the historical record. Run 2's
> artifacts (`runs/d4_rerun_2x2/`) live on the machine that produced them —
> 2×RTX 4070 Ti SUPER, so its timings are also not comparable to the A100
> numbers below — and are not committed here.

## Key findings

- **Convergence:** E/site saturates by **χ≈32** (flat to 6 digits beyond), so
  **χ≈32 is sufficient at D=4** — that is a property of the scan and it holds.
  The saturated *value* (−0.663398, +0.006 vs the QMC reference −0.669437) is
  **superseded**, and that residual is **not** the D=4 variational floor: it
  mixes iPEPS truncation with a state optimized against a collapsed
  environment. See the banner above.
- **Multi-GPU:** numerically exact (ΔE vs 1-GPU ≤ 7e-8, machine-epsilon at large χ)
  but **below the crossover at this scale** — 2-GPU is break-even (0.99–1.01×, flat
  in χ), 4-GPU markedly slower but improving monotonically with χ (0.21× at χ=16 →
  0.49× at χ=96) as the sharding overhead amortizes. Peak memory ≤ 2.8 GB even at
  χ=128, so dense D=4 CTM at χ≤128 is too small for sharding to pay off; the
  crossover needs larger D or χ≫128, and the 4-GPU trend is consistent with one
  existing above this range rather than with sharding being unusable.

## Corrections

**#747 (2026-08-09) — the optimized state was produced against a collapsed
environment, so every energy here is superseded.** The driver hard-coded
`gs_recipe="1x1"`, whose rank-1 corners give a `chi_eff = 1` mean-field boundary
(#723/#726, fixed in #749/#765). The recorded energies are therefore the
energies of a badly-optimized state, and **+6.04e-03 must not be quoted as the
D=4 truncation error**. Run 2 (#747 comment 4) re-optimized at
`gs_recipe="2x2"` — now the driver default, with `--gs-recipe` exposed — and
reports **−0.6639345178 / +5.50e-03**, itself an upper bound (see the banner).

The χ-scan path, the multi-GPU findings and `peak_gb` are **unaffected**: the
forward scan always ran `2x2` via `python_loop_ctm_converge`. Only the state
being scanned was bad, so the curve's *shape* and the device-independence
cross-check stand while its absolute *level* does not.

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
| `convergence.md` | E/site vs χ table (vs QMC) — **energies superseded (#747)** |
| `performance.md` | per-sweep CTM cost, peak GB, speedup vs 1-GPU, grouped by device count — unaffected by #747 |
| `results.csv` | all 21 cells, machine-readable — **energies superseded (#747)**; `ms_per_sweep`/`n_sweeps` as recorded, i.e. inflated (#781) |
| `D4_chi{χ}_n{devices}.json` | per-cell raw result (E, timing, memory, convergence flag); same #747 and #781 caveats |
| `convergence_E_vs_chi.png` | E-vs-χ with QMC line — **plots the superseded energies (#747)** |
| `perf_*.png` | ms/sweep, speedup, peak-GB plots — unaffected by #747 |
| `A_opt.pkl` | the D=4 tensor as optimized **against a collapsed `1x1` environment (#747)** — kept as the record; **do not use it as a D=4 ground state** |
| `optimize_status.json` | optimize-phase exit status |

Optimizer checkpoints (`ckpt_opt/`, ~1.7 MB binary) and the run log are intentionally
**not** committed (large, regenerable).

## Reproduce

`gs_recipe` now defaults to `2x2`, so a fresh run no longer reproduces the
collapsed state — it reproduces **Run 2**, not this page.

```bash
# Full sweep (optimize once at χ_opt=32, then scan χ × {1,2,4} GPU). ~19 h on 4×A100.
uv run python examples/heisenberg_d4_chi_scaling.py --outdir runs/d4_chi_scaling

# Quick end-to-end validation (tiny):
uv run python examples/heisenberg_d4_chi_scaling.py --smoke
```

**Do not seed a new run with the committed `A_opt.pkl`.** The optimize phase
returns immediately when it finds one, so copying this file in silently scans
the collapsed-environment state again and reproduces the superseded energies —
which is exactly how the invalid numbers survived to be published. Reusing it is
only valid for re-deriving *this page's* record, e.g. the #781 timing
correction.

To reproduce the page as recorded, or to bisect against it, ask for the old
recipe explicitly:

```bash
uv run python examples/heisenberg_d4_chi_scaling.py --gs-recipe 1x1 --outdir runs/d4_1x1_bisect
```
