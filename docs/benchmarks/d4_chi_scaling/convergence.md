### Convergence: E/site vs χ (D=4 square-lattice Heisenberg AFM)

QMC reference E/site = -0.669437

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
