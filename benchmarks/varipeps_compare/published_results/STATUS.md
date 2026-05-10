# Tenax ↔ variPEPS Square Heisenberg Benchmark — Status

**Last update:** 2026-05-10 (F3 fused-backward landed; chi=16 budget still blocked, see profile attribution below)
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
after F1 (JIT cache fix, 11aafd3), F2 (fixed-point adjoint, 032ec80),
and F3 (fused @jax.jit backward, this PR). F3 lands the entire
backward in one trace+compile as designed, and the parity test
wall-clock confirms a real backward speedup (170 s F2 → 108 s F3 for
the gmres-vs-fixed-point test pair). But the chi=16 single_site
benchmark still exceeds the 30-min ceiling.

### Why — profile attribution under cProfile

Ran `optimize_gs_ad` for ~10 min at single_site D=2 chi=16 with the
F3 backward in place. cProfile cumulative-time breakdown:

| component | cumtime | share | calls | per-call |
|---|---|---|---|---|
| `hager_zhang_line_search` | 396 s | **62%** | — | — |
| ↳ `_safe_dphi` | 386 s | — | 2 | 193 s |
| ↳↳ `_dphi` → `_tree_dot` | 362 s | — | 5 | 72 s |
| ↳↳↳ `numpy.ufuncs.conj` | 362 s tottime | **57%** | 2658 | 136 ms |
| `value_and_grad_f` (the gradient itself) | 84.9 s | 13% | 3 | 28 s |
| `backward_pass3` (the actual VJP F3 fused) | 44.8 s | 7% | — | — |
| XLA `backend_compile_and_load` | 17.2 s tottime | 3% | 197 | — |

**The chi=16 bottleneck is not the backward.** It is `_tree_dot`
(`src/tenax/algorithms/ipeps_optimize.py:191`) inside the L-BFGS
Hager-Zhang line search's `_dphi` evaluator. `_tree_dot` is a
Python `sum(...)` over `jnp.conj(la) * lb`; cProfile attributes the
time to `numpy.ufuncs.conj` (2,658 calls × 136 ms each), suggesting
the gradient pytree leaves materialise to numpy in the line-search
host code. F3 saves real backward time, but the backward is only
~13–20% of total wall-clock at chi=16 — not the dominant cost.

### F3 microbenchmark (D=2, χ=8, single_site)

Single forward + backward via `ctm_energy_implicit`, CPU,
complex128. **chi=8 is small enough that the F3 fused-graph
overhead exceeds the saved Python-dispatch cost** — F3 underperforms
F2 here. The chi=8 microbench is **not** the load-bearing test for
F3; the chi=16 parity wall-clock and per-backward attribution are.

| metric | F2 baseline | F3 (this PR) |
|---|---|---|
| cold (compile + 1 bwd) | 62 s | 104 s (worse — fused graph dominates at small chi) |
| warm 1 | 35 s | 55 s (worse) |
| warm 2 | 38 s | 46 s (worse) |
| parity test (fixed_point + gmres, pytest) | 170 s | **108 s (35% faster)** |
| chi=16 single_site benchmark | 30-min timeout | 30-min timeout (different bottleneck — see profile) |

### What's left to fit χ=16 inside 30 min

In order of leverage based on the profile:

- **Fix `_tree_dot` host-numpy materialisation** — keep gradient
  leaves on-device through the line search. Eliminating the 362 s
  spent in `numpy.ufuncs.conj` would alone unblock the budget.
  Highest-leverage and most localized fix. New issue should be
  opened.
- **GPU.** This machine has 2× CUDA devices. The CTM forward and
  the F3 fused backward are large-matmul-dominated and benefit
  directly. Re-run with `--device cuda:0` once `_tree_dot` is
  fixed.
- **F4** (`jax.experimental.implicit_diff.custom_root`) remains
  out of scope for this benchmark.

## How to reproduce

```bash
JAX_PLATFORMS=cpu uv run python -m benchmarks.varipeps_compare.compare \
    --device cpu --results-dir benchmarks/varipeps_compare/results
```

Idempotent — skips points whose JSON already exists. Results land in
`benchmarks/varipeps_compare/results/` (gitignored). Once a meaningful
Tenax/variPEPS pair is collected, copy into `published_results/`
alongside this file.
