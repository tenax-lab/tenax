# Symmetric 2-site CTM implicit AD: validation harness

Spec: `docs/superpowers/specs/2026-10-09-symmetric-2site-ctm-ad-design.md`.
Run from the repository root (`uv run`, so the worktree's tenax is imported).
The scripts need `JAX_PLATFORMS=cuda,cpu CUDA_VISIBLE_DEVICES=0` on a GPU
host (`cuda` alone fails on a `jax.debug.callback`); run one job at a time.

## One point

```bash
uv run python -m benchmarks.symmetric_ad.pipeline \
    --model xxz03 --sym z2 --D 2 --chi 8 --seed 0 \
    --cache-dir /path/to/jax-cache --out OUT/xxz03_z2_D2_chi8_s0
```

`--model` is `xxz03` (XXZ, Delta=0.3) or `heis` (Delta=1); `--sym` is `u1`
(U(1)-Sz) or `z2`.  Gates are unrotated (#939).  `OUT` receives `S1.json` ..
`S5.json` and `summary.json`.  Each record has `stage`, `passed`, `measured`
and `wall`.  A failed stage stops the point; later stages are written as
`skipped`.  An exception in a stage is a failure with its traceback recorded.

| Stage | Runs | Gate |
|---|---|---|
| S1 SU | symmetric 2-site SU -> (A, B); converged eager-CTM energy `E_sym`. Same-state dense reference: densify (A, B), run the dense converged CTM (`E_dense`), and measure `viol`, the charge-violating weight `\|P(rho)\|_F / \|rho\|_F` of the four bond RDMs of that dense environment (`P` keeps elements with `q_a+q_b != q_c+q_d` in the symmetry's own arithmetic). Dense SU from the same start is recorded (`E_dense_su`, `diff_su`), never gated. | Non-empty blocks, and `viol < 1e-6` (dense env kept the symmetry) -> \|E_sym - E_dense\| < 1e-8. If `viol >= 1e-6` the dense env broke the symmetry and is a different fixed point: the energy difference is reported, not gated, and `summary.json` gets `s1_symmetry_broken: true`. NaN `diff` or `viol` fails. Convergence flags are reported, not gated. |
| S2 Seed | `symmetric_layout_seed`; per-corner `chi_binding` (rank of each corner at chi+4 above 1e-12 max, more than chi) | `stable`, cold CTM converged, `cut_gap` not NaN. `cut_gap` <= 1e-2 flags near-degenerate (not a failure). If no corner is binding the point is marked `vacuous` in `summary.json` (not failed, not a pass). |
| S3 Forward | the AD loss started from the seed | layout unchanged and E equals S2's cold energy to 1e-10 |
| S4 Gradient | flow implicit-AD `value_and_grad` | 3 random unit directions: \|g.d - FD\| < max(1e-6 \|FD\|, fd_unc) at the best h, and fd_unc < 1e-3 \|FD\|. The S4 loss uses `gmres_tol=1e-9` (recorded as `gmres_tol` in the S4 record); the optimizer keeps 1e-6. S4 certifies the gradient formula with a tight adjoint solve; it does not certify accuracy at the optimizer default (there the adjoint residual alone gave a 1.6e-6 relative error on one direction). Oracle (`oracle_rel` per site, its forward stationarity and backward-used) is reported, never gated. The symmetric backward must be `flow`; all values finite. On a near-degenerate point the tolerance comparison is reported only. |
| S5 Optimize | `optimize_gs_ad` from `envs_init`, up to 3 layout stages; drift check by the S2 procedure; one dense converged CTM energy at the densified final tensors | layout settled, drift seed stable, all energies finite, final E <= start E, cold (forced) and warm energies agree to 1e-10 with the warm CTM converged, and \|E_sym - E_dense\| < 1e-8. There is no dense optimizer arm (its CTM is not stationary here: hours per 10 steps). |

Why S1 compares the same state (R25): the dense SU from the same start drifts to a U(1)-broken state at Delta=0.3 (two different states: seed 1 D=2 chi=8 differed by -5.5e-6 while one state agrees to 7e-15). And on the same state the dense CTM can itself converge to an environment that breaks U(1) (u1 D=3 chi=9 seed 0: bond `<(S+S+ + S-S-)/2>` = -0.1324, energy 4.25e-3 below the symmetric CTM), so equality is gated only where the dense environment keeps the symmetry. `viol` uses the four bond RDMs separately, not `compute_energy_ctm_tensor_2site` (which sums the four bonds and divides by two, so an order staggered between A-B and B-A bonds could cancel).

The reference CTMs (S1, S4 FD, S5) use `conv_tol=1e-10` on the element-wise
singular values (`REF_TOL`): some states sit on a floor near 3e-11 where
1e-12 never reports convergence although the energy is fixed to 2e-16.

## The matrix

```bash
uv run python -m benchmarks.symmetric_ad.matrix --out OUT --cache-dir CACHE [--only SUBSTR]
```

Resumable: a point is skipped when its `summary.json` is complete and has the current code hash (`--no-resume` reruns). Before a point starts its old records are deleted, so a crashed point never shows a stale result. The table shows PASS / vacuous / FAILED at S# / CRASHED(rc) and the five stage flags.

Runs the 32 points (model x sym x D in {2,3} x chi in {8,16} x seed in {0,1})
one subprocess at a time with the GPU environment above, then writes
`OUT/matrix_summary.md` (return code, stopped-at stage, near-degenerate flag).
Per-point logs are `OUT/<point>/run.log`.

## The control (AD without the seed)

```bash
uv run python -m benchmarks.symmetric_ad.control_free_layout \
    --model xxz03 --sym z2 --D 2 --chi 8 --seed 0 --out OUT/control
```

Under AD the layout is fixed by tracing (#929), so a "free layout" AD run does
not exist.  The control measures what seeding buys: the loss is built with an
empty env cache (a cold start), and 20 plain gradient-descent steps (step
0.01 on normalised tensors) are taken.  After each step `control.json` records
the loss energy, the converged CTM energy at the same tensors (whose layout
adapts), their gap, `forward_converged` and the wall time.  Nothing is gated.

## Smoke test

`tests/test_symmetric_ad_harness_smoke.py` runs a tiny point (Z2, D=2, chi=4)
and checks that every stage record is written, and that a failing stage stops
the point.  It checks the plumbing, not the physics.
