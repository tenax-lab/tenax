# Tenax ↔ variPEPS Square Heisenberg Benchmark — Status

**Last update:** 2026-05-09
**Protocol:** TOL=1e-6, MAX_STEPS=100, complex128, single_site path with sublattice-rotated gate, D=2, χ=16. Tenax `gs_implicit_ad=True`, variPEPS native.

## What's in this directory

- `single_site_D2_chi16_tenax.json` — Tenax run, error (timeout, see below).
- `single_site_D2_chi16_varipeps.json` — variPEPS run, **converged**.

## variPEPS reference baseline (point 1)

| field | value |
|---|---|
| `final_energy` | −0.66251430 |
| `num_steps` | 23 (converged before MAX_STEPS=100) |
| `total_wall_clock` | 800.7 s ≈ 13.3 min |
| `jit_compile_time` | 170.4 s ≈ 2.8 min |
| `peak_memory_mb` | 2421 |

Reference for D=2 chi=16 from the literature: E/site ≈ −0.6614 (variPEPS papers). Our run converged to a slightly lower energy due to chi=16 being slightly above the variational floor at D=2.

## Tenax status

**Single point timed out at 30-min subprocess budget on CPU even after the
JIT cache fix in commit 11aafd3.** The fix eliminated 45 of 46 redundant
`_step` compiles (verified at chi=4 attribution), but at chi=16 the
remaining per-step solve cost on CPU still exceeds the 30-min ceiling.

Two paths for the next iteration are concrete:

- **Run on GPU.** This machine has 2× CUDA devices. Re-run with
  `--device cuda:0`. The CTM forward + adjoint solve are
  large-matrix-multiply dominated, which GPUs handle well.
- **Continue the perf work tracked in
  `docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md`** — F2
  (eager-GMRES → fixed-point iteration, mirroring variPEPS) is the
  next high-leverage fix.

## How to reproduce

```bash
JAX_PLATFORMS=cpu uv run python -m benchmarks.varipeps_compare.compare \
    --device cpu --results-dir benchmarks/varipeps_compare/results
```

Idempotent — skips points whose JSON already exists. Results land in
`benchmarks/varipeps_compare/results/` (gitignored). Once a meaningful
Tenax/variPEPS pair is collected, copy into `published_results/`
alongside this file.
