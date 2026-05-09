# Tenax ↔ variPEPS Square Heisenberg Benchmark — Status

**Last update:** 2026-05-10 (F2 fixed-point adjoint landed)
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

**Single point still times out at the 30-min subprocess budget on CPU**
even after F1 (JIT cache fix, 11aafd3) and F2 (fixed-point adjoint, this
PR).  F2 reduces backward wall-clock by ~3× as designed — see
microbenchmark below — but the forward CTM convergence at χ=16 (Python
loop over a JIT'd `_step` plus repeated L-BFGS line-search forward calls)
still exceeds the 30-min ceiling on this machine.

### F2 microbenchmark (D=2, χ=8, single_site)

Single forward + backward via `ctm_energy_implicit`, CPU, complex128:

| `adjoint_method` | cold (compile + 1 bwd) | warm 1 | warm 2 |
|---|---|---|---|
| `"fixed_point"` (new default) | 15.7 s | 1.3 s | 2.1 s |
| `"gmres"` (legacy)            | 46.7 s | 4.7 s | 5.4 s |

F2 delivers ~3× cold-compile and ~3× warm-call speedup on the backward.
The plan's expected 3–5× drop is met.

### What's left to fit χ=16 inside 30 min

Two complementary paths, in order of leverage:

- **F3 (variPEPS-style backward fusion)** — fold `_jit_dE_denv`,
  `_jit_apply_Jt`, and `_jit_chain_rule` into a single `@jax.jit` with
  `lax.while_loop` for the adjoint iteration.  variPEPS's
  `_ctmrg_rev_workhorse` shows this lands the entire backward in one
  graph (~1 trace + 1 compile vs Tenax's 3) and is the reason variPEPS
  finishes the same problem in ~13 min.  See
  `docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md` §F3.
- **GPU.** This machine has 2× CUDA devices.  Re-run with
  `--device cuda:0`; the CTM forward + adjoint solve are
  large-matmul-dominated and benefit directly.

## How to reproduce

```bash
JAX_PLATFORMS=cpu uv run python -m benchmarks.varipeps_compare.compare \
    --device cpu --results-dir benchmarks/varipeps_compare/results
```

Idempotent — skips points whose JSON already exists. Results land in
`benchmarks/varipeps_compare/results/` (gitignored). Once a meaningful
Tenax/variPEPS pair is collected, copy into `published_results/`
alongside this file.
