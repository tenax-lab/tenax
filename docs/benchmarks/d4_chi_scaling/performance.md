### Performance: per-sweep CTM cost & memory vs χ × n_devices

> **Corrected 2026-08-06 — issue #781.** The `ms/sweep` and `sweeps` columns as
> originally generated were wrong. `CTMLoopResult.iterations` reported the
> *best-metric* iteration rather than the sweeps performed, and every one of these
> 21 cells exited on the `plateau_patience` stop-loss, so the driver divided the
> full elapsed time by an index `plateau_patience=20` sweeps short of the truth.
> The inflation ranged from 1.00× (the two cells that genuinely converged) to
> 3.00×, so it does **not** cancel in the speedup ratio.
>
> The tables below are recomputed from the recorded `total_s` as
> `1000 · total_s / (n_sweeps + 20)`, exact here because the run used
> `plateau_patience=20` with the χ-bump and QR warm-up off. The raw per-cell
> JSONs and `results.csv` are left **unmodified** as the record — do not read
> `ms_per_sweep` out of them. Runs after #781 record `best_iteration` alongside
> `n_sweeps`, so no such re-derivation is needed again.
>
> Energies and `peak_gb` are unaffected. Conclusions are unchanged in direction
> but not in magnitude: the 2-GPU column, published as a noisy 0.61–1.32×, is
> flat break-even (0.99–1.01×) once corrected — that scatter was purely the bug.


#### 1-GPU

| χ | status | ms/sweep | sweeps | peak GB | speedup vs 1-GPU | published ms/sweep (inflated) |
|---|--------|----------|--------|---------|------------------|-------------------------------|
| 16 | ok | 171.91 | 41 | 0.037 | 1.00 | 335.64 |
| 24 | ok | 320.90 | 34 | 0.071 | 1.00 | 779.33 |
| 32 | ok | 513.20 | 33 | 0.143 | 1.00 | 1302.74 |
| 48 | ok | 1010.32 | 37 | 0.285 | 1.00 | 2198.94 |
| 64 | ok | 1653.45 | 53 | 0.939 | 1.00 | 2655.54 |
| 96 | ok | 2321.80 | 53 | 1.703 | 1.00 | 3728.95 |
| 128 | ok | 3638.61 | 47 | 2.805 | 1.00 | 6333.87 |

#### 2-GPU

| χ | status | ms/sweep | sweeps | peak GB | speedup vs 1-GPU | published ms/sweep (inflated) |
|---|--------|----------|--------|---------|------------------|-------------------------------|
| 16 | ok | 173.56 | 51 | 0.110 | 0.99 | 285.53 |
| 24 | ok | 322.10 | 44 | 0.169 | 1.00 | 590.51 |
| 32 | ok | 518.12 | 30 | 0.193 | 0.99 | 1554.36 |
| 48 | ok | 1015.66 | 31 | 0.495 | 0.99 | 2862.31 |
| 64 | ok | 1661.58 | 56 | 0.726 | 1.00 | 2584.69 |
| 96 | ok | 2304.54 | 32 | 1.604 | 1.01 | 6145.45 |
| 128 | ok | 3632.50 | 47 | 2.806 | 1.00 | 6323.24 |

#### 4-GPU

| χ | status | ms/sweep | sweeps | peak GB | speedup vs 1-GPU | published ms/sweep (inflated) |
|---|--------|----------|--------|---------|------------------|-------------------------------|
| 16 | ok | 827.59 | 33 | 0.036 | 0.21 | 827.59 |
| 24 | ok | 1502.99 | 32 | 0.077 | 0.21 | 4007.98 |
| 32 | ok | 2231.87 | 44 | 0.118 | 0.23 | 4091.76 |
| 48 | ok | 3594.31 | 31 | 0.301 | 0.28 | 3594.31 |
| 64 | ok | 3614.11 | 33 | 0.441 | 0.46 | 9174.29 |
| 96 | ok | 4760.65 | 31 | 0.990 | 0.49 | 13416.37 |
| 128 | ok | 8329.86 | 45 | 1.731 | 0.44 | 14993.75 |
