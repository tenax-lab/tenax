# Heisenberg scaling showcase — results

> ## ⚠️ The energy columns are RETRACTED (#747)
>
> **`E/site` and `dE_ref` have been removed from the table below. Do not quote
> them, and do not reinstate them from `scaling_results.csv`.**
>
> Two independent defects, either of which alone disqualifies them as physics:
>
> 1. **Collapsed environment.** This sweep ran `gs_recipe="1x1"`, whose
>    corner-pair projector collapses the CTM environment to rank-1 corners
>    (#723, #726, #746). That is a `chi_eff = 1` mean-field boundary, not a
>    corner transfer matrix, so its energy carries no boundary entanglement and
>    does not respond to χ.
> 2. **Never converged.** All 29 recorded cells are `gs_num_steps=6`,
>    `converged=false`, `is_anchor=false` — the accurate-AD anchor cells were
>    never recorded at all. Six optimizer steps cannot support a physics claim
>    under *any* recipe.
>
> Because both are present, the recorded data cannot attribute the observed
> χ-non-monotonicity to either one. The `-0.174648` at D=4, χ=32, 4-GPU is not
> a physical Heisenberg energy by any reading.
>
> **The performance columns below are unaffected and remain valid.** `ms/step`
> and `peak GB` depend on tensor *shapes*, which the collapse does not change:
> the D⁶ cost jump, the host→FLOP transition, and the multi-GPU crossover all
> stand.
>
> The driver now refuses to print an energy for a cell that did not converge or
> whose environment collapsed, and records `corner_rank` per cell, so a rerun
> cannot republish this silently. See #747 for the full audit.



### 1-GPU

| D | χ | kind | status | ms/step | peak GB | conv |
|---|---|------|--------|---------|---------|------|
| 2 | 16 | metrics | ok | 2647.3 | 0.01 | N |
| 2 | 24 | metrics | ok | 2625.8 | 0.01 | N |
| 2 | 32 | metrics | ok | 2629.6 | 0.02 | N |
| 2 | 48 | metrics | ok | 3603.2 | 0.05 | N |
| 2 | 64 | metrics | ok | 3624.6 | 0.07 | N |
| 2 | 96 | metrics | ok | 4713.7 | 0.21 | N |
| 2 | 128 | metrics | ok | 5694.8 | 0.31 | N |
| 3 | 16 | metrics | ok | 23715.9 | 0.07 | N |
| 3 | 24 | metrics | ok | 33700.4 | 0.11 | N |
| 3 | 32 | metrics | ok | 49969.5 | 0.13 | N |
| 3 | 48 | metrics | ok | 54390.1 | 0.33 | N |
| 3 | 64 | metrics | ok | - | 0.54 | N |
| 3 | 96 | metrics | ERR | - | - | N |
| 4 | 16 | metrics | ok | 27361.2 | 0.14 | N |
| 4 | 24 | metrics | ok | 48528.9 | 0.27 | N |
| 4 | 32 | metrics | ok | 66074.4 | 0.48 | N |
| 4 | 48 | metrics | ERR | - | - | N |

### 4-GPU

| D | χ | kind | status | ms/step | peak GB | conv |
|---|---|------|--------|---------|---------|------|
| 2 | 16 | metrics | ok | 14164.8 | 0.11 | N |
| 2 | 24 | metrics | ok | 15035.2 | 0.11 | N |
| 2 | 32 | metrics | ok | 14977.7 | 0.11 | N |
| 2 | 48 | metrics | ok | - | 0.11 | N |
| 2 | 64 | metrics | ok | 17838.0 | 0.11 | N |
| 2 | 96 | metrics | ok | - | 0.13 | N |
| 2 | 128 | metrics | ok | 22975.4 | 0.25 | N |
| 3 | 16 | metrics | ERR | - | 0.03 | N |
| 4 | 16 | metrics | ok | 24501.4 | 0.20 | N |
| 4 | 24 | metrics | ok | 32200.9 | 0.27 | N |
| 4 | 32 | metrics | ok | 37840.3 | 0.48 | N |
| 4 | 48 | metrics | ERR | - | - | N |
