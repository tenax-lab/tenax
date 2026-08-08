### Convergence: E/site vs χ (D=4 square-lattice Heisenberg AFM)

QMC reference E/site = -0.669437

> **⚠️ These energies are superseded — #747.** The scanned state was optimized
> at `gs_recipe="1x1"`, against a CTM environment that collapses to rank-1
> corners (#723/#726). The scan below is sound — it ran `2x2` throughout — but
> the state it scans is badly optimized, so **+6.04e-03 is not the D=4
> truncation error**.
>
> Run 2 (#747 comment 4, 2026-08-05) re-optimized at `gs_recipe="2x2"`, with
> `corner_rank == χ` in all seven cells:
>
> | χ | E/site | err vs QMC | sweeps | corner_rank |
> |---|--------|------------|--------|-------------|
> | 16 | -0.6638308456 | +5.606e-03 | 54 | 16 |
> | 24 | -0.6639137982 | +5.523e-03 | 54 | 24 |
> | 32 | -0.6639289593 | +5.508e-03 | 39 | 32 |
> | 48 | -0.6639337605 | +5.503e-03 | 42 | 48 |
> | 64 | -0.6639343531 | +5.503e-03 | 78 | 64 |
> | 96 | -0.6639345149 | +5.502e-03 | 14 | 96 |
> | 128 | **-0.6639345178** | **+5.502e-03** | 18 | 128 |
>
> That run's `A_opt` was stopped at step 14, so **+5.50e-03 is also an upper
> bound**, not a converged variational result. Both tables agree that χ≈32
> saturates the scan; they disagree on the level by 5.4e-04.

> **The `conv` column reads `N` everywhere, and that is a criterion artifact —
> issue #780.** This run inherited `CTMConfig.ctm_conv_method="elementwise"`,
> which compares raw environment tensor entries between sweeps. A CTM
> environment is defined only up to a gauge on each χ-bond, so that comparison
> measures gauge motion as much as convergence: a *pure* gauge (energy
> invariant to 2.2e-16) moves the element-wise metric to 1.0 while the
> gauge-invariant `sv` metric stays at 1.1e-16. On a converged D=4 environment
> elementwise plateaus around 2.6e-01 while `sv` reaches 6.5e-09 in 18 sweeps,
> the two agreeing on the energy to 9 digits. No `conv_tol` can flip the flag;
> `plateau_patience` is always the exit path.
>
> **The environments here are converged.** E/site is invariant to 10 digits at
> χ=32 between sweep 39 and sweep 400 — genuinely different environments, since
> the bail returns the best-metric env and the run-to-`max_iter` path returns
> the last one — and flat to 6 digits from χ=32 up. Judge convergence from that
> stability, not from this column.
>
> The χ-scaling drivers now pass `conv_method="sv"` explicitly and record the
> criterion and its achieved metric per cell, so a future `N` here is
> diagnosable. `sweeps` is also a best-iteration index in this table rather than
> the sweeps performed — see the #781 note in `README.md`.

| χ | E/site | err_vs_QMC | sweeps | conv | metric | crit |
|---|--------|------------|--------|------|--------|------|
| 16 | -0.663379 | +6.06e-03 | 21 | N | - | elementwise |
| 24 | -0.663394 | +6.04e-03 | 14 | N | - | elementwise |
| 32 | -0.663397 | +6.04e-03 | 13 | N | - | elementwise |
| 48 | -0.663398 | +6.04e-03 | 17 | N | - | elementwise |
| 64 | -0.663398 | +6.04e-03 | 33 | N | - | elementwise |
| 96 | -0.663398 | +6.04e-03 | 33 | N | - | elementwise |
| 128 | -0.663398 | +6.04e-03 | 27 | N | - | elementwise |

(The `metric` column is `-` because this run predates recording it; that is the
gap #780 closes.)
