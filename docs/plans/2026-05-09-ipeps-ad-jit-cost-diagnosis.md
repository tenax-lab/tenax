# iPEPS AD JIT Cost Diagnosis (vs variPEPS)

**Date:** 2026-05-09
**Context:** While running `benchmarks/varipeps_compare/compare.py` at the
intended protocol (single_site D=2 χ=16, implicit AD, L-BFGS), Tenax's
`run_tenax` subprocess timed out at the 30-minute budget on **both CPU and
GPU**. variPEPS solves the same problem in ~5 minutes total on CPU
(JIT compile + 23 AD steps to convergence). The benchmark's
apples-to-apples deliverable is gated on Tenax fitting in a comparable
budget.

This doc captures the architectural diagnosis so the fix can be scoped as
a focused follow-up.

## Empirical timings

| run | platform | chi | result |
|---|---|---|---|
| variPEPS (single_site, D=2, χ=16, max-steps=100, tol=1e-6) | CPU | 16 | ✅ 312 s; 64 s JIT + 248 s solve; converged in 23 steps; E = −0.6625 |
| Tenax (same config) | CPU | 16 | ❌ 30-min subprocess timeout |
| Tenax (`gs_num_steps=1` — only one AD step needed) | GPU | 16 | ❌ killed at 18 min, GPU still 95% busy on JIT |

Conclusion: the gap is in **Tenax's JIT compilation cost**, not its
runtime per step or its hardware utilization. The JAX persistent cache is
not the cause (clearing `~/.cache/jax/` and re-running fresh shows the
same cold-compile time).

## Architectural comparison

Both libraries use `@jax.custom_vjp` to put a hard boundary between the
forward CTM convergence and the implicit-AD backward. The difference is
inside the backward.

### Tenax `f_bwd` (`src/tenax/algorithms/_ctm_energy_ad.py:809`)

```python
def f_bwd(residuals, g):
    params_data_tuple, env_leaves = residuals

    dE_denv = _jit_dE_denv(params_data_tuple, env_leaves)        # JIT 1

    if arnoldi_precheck:
        ... apply_Jt_only(v) -> _jit_apply_Jt(...)               # JIT 2
        rho = arnoldi_spectral_radius_pytree(apply_Jt_only, ...)

    def _eager_apply_I_minus_Jt(v):
        return _jit_apply_Jt(params_data_tuple, env_leaves, v)   # JIT 2 (reused)

    lam, _info = gmres_pytree_jax(                               # eager GMRES
        _eager_apply_I_minus_Jt, dE_denv, dE_denv,
        tol=gmres_tol, maxiter=gmres_maxiter, restart=gmres_restart,
    )

    return _jit_chain_rule(params_data_tuple, env_leaves, lam_leaves, g)  # JIT 3
```

There is also `_jit_gmres_solve` (JIT 4) defined in the same file but
**not currently called** in `f_bwd`. The comment notes it's reserved for
a fully fused Krylov solve but the eager path is preferred at large χ
"where the custom gmres_lax can produce wrong gradients."

So at runtime, the cold-compile path traces and lowers **3 separate JIT
functions** (`_jit_dE_denv`, `_jit_apply_Jt`, `_jit_chain_rule`). Each
contains its own VJP-through-CTM-sweep graph, so they are individually
heavy. They compile sequentially on first invocation.

### variPEPS `_ctmrg_rev_workhorse` (`varipeps/ctmrg/routine.py:1050`)

```python
@jit
def _ctmrg_rev_workhorse(peps_tensors, new_unitcell, new_unitcell_bar, config, state):
    _, vjp_peps_tensors = vjp(
        lambda t: do_absorption_step(t, new_unitcell, config, state), peps_tensors
    )
    vjp_env = tree_util.Partial(
        vjp(lambda u: do_absorption_step(peps_tensors, u, config, state), new_unitcell)[1]
    )

    # ... Python while-loop fixed-point iteration:
    #     bar_fixed_point += vjp_env(bar_fixed_point)
    #   until element-wise converged.

    # ... apply vjp_peps_tensors at the end to get gradient w.r.t. inputs.
```

The entire backward is **one `@jit` function**. Inside, the adjoint
fixed-point iteration is a simple `bar += vjp_env(bar)` loop — no Krylov
subspace, no orthogonalization, no restarts. The graph is dramatically
smaller than GMRES's.

### Why this matters for compile cost

JIT trace + lower + compile is roughly proportional to the number of
HLO ops. Tenax's 3-JIT structure traces three different graphs that each
include:

- a copy of the CTM absorption step,
- VJP through it (~2x the forward graph size),
- the residual chain rule.

variPEPS's single-JIT backward traces the absorption step + VJP **once**
and reuses the closure. The Python while-loop body is a Python-level
iteration over a JIT'd closure, so each iteration is a cheap cached
launch — no re-trace.

Tenax's eager `gmres_pytree_jax` does the same kind of Python-level
iteration, BUT each iteration calls `_jit_apply_Jt` (cached after first
call) and additionally executes Krylov-subspace bookkeeping (Arnoldi
orthogonalization, Givens rotations) that variPEPS's fixed-point method
avoids entirely.

## Hypothesised fix paths

In rough order of effort:

### F1. Eliminate `_jit_dE_denv` and `_jit_chain_rule` as separate JITs

Both are small; their bodies could be inlined into the forward-VJP closure
that's already constructed. Saves 2/3 of the JIT compile boundaries. The
work that's currently in three separate compiled programs would happen
in one. **Lowest-risk step.** Estimate: 1–2 hours of work + tests.

### F2. Switch the eager-GMRES adjoint loop to fixed-point iteration

Mirror variPEPS: replace `gmres_pytree_jax(...)` with a Python
`for _ in range(maxiter): lam = lam + _jit_apply_Jt(lam)` until
convergence. Eliminates the Krylov machinery from the trace.

This is a **gradient-correctness change** — fixed-point iteration on
`(I - J^T) λ = b` requires the spectral radius of `J^T` to be < 1. The
existing Arnoldi precheck (already present in Tenax) measures exactly this.
At chi ≥ ~16 with the phase gauge default, the spectral radius is < 1 and
fixed-point iteration converges (variPEPS evidence). At smaller chi or
sigma gauge, the precheck would correctly reject and fall back to GMRES.
Estimate: half a day of work + tests.

### F3. Fuse all backward steps into one `@jax.jit` (variPEPS-style)

Inline `_jit_dE_denv`, the adjoint loop body, and `_jit_chain_rule` into
a single `@jax.jit` boundary, with the adjoint loop as a `jax.lax.while_loop`
inside that JIT. This is what variPEPS does. Highest payoff but biggest
risk — needs careful refactor of the existing API surface (mutables dict,
`_VJP_CACHE`). Estimate: 2–3 days of work + extensive tests.

### F4. Deeper: replace per-step custom_vjp with proper implicit-function-theorem helper

`jax.experimental.implicit_diff.custom_root` or equivalent. Best long-term
but requires significant rewrite of how the forward CTM convergence is
expressed (must look like a fixed-point root-finder). Out of scope for
the benchmark.

## Recommended next steps

Open three issues / PRs (or three commits on a single PR):
1. **F1** as a no-behavior-change refactor — easy to land, moderate
   speedup. Should drop cold-compile time roughly proportionally.
2. **F2** as a separate behavior-change PR — measurable compile-time and
   per-step wins, but needs gradient-correctness tests at multiple χ on
   the `bench_ipeps_ad` cases (especially the existing implicit-AD
   regression in `tests/test_ipeps_excitations.py::test_runs_without_error`).
3. **F3** as a longer-running refactor track, only if F1+F2 don't close
   the gap to variPEPS.

The benchmark harness (`benchmarks/varipeps_compare/`) is already in place
and idempotent — once F1+F2 land, re-running the orchestrator should
produce a complete report with both libs' data, no other plumbing
required.

## What's blocking right now

Until F1 (or better) ships, the benchmark cannot produce Tenax data on
the intended protocol within a 30-minute subprocess budget. Workarounds:

- Lower χ to 8 (smaller graph → faster compile). Loses comparison
  fidelity since both libs run the variational regime at χ ≥ 16.
- Increase subprocess timeout to ~90 min (Tenax still slower but might
  fit). Burns hours of wall-clock per benchmark run.
- Switch the protocol to explicit AD (smaller graph). Defeats the point
  — variPEPS only does implicit AD.

None of these are good. F1 is the right unblock.

## Empirical attribution (2026-05-09)

Ran a `jax.log_compiles()` capture on `optimize_gs_ad` at the smallest
single_site implicit-AD config (D=2, χ=4, max-steps=1) to attribute
compile time per JIT'd function. Aggregated XLA compile times:

| component | compiles | total | per-compile avg |
|---|---|---|---|
| `while_loop` | 11 | 19.7 s | 1.8 s |
| `_step` (CTM step) | **46** | 13.8 s | 0.3 s |
| `_jit_apply_Jt` | 2 | 5.1 s | 2.6 s |
| `_jit_chain_rule` | 1 | 1.7 s | — |
| `_jit_dE_denv` | 1 | 0.13 s | — |

**The original F1 hypothesis was wrong.** The three backward JITs
(`_jit_dE_denv`, `_jit_apply_Jt`, `_jit_chain_rule`) together total ~7 s
and consolidating them into one JIT would not change total compile work
materially.

**Real cost:** `_make_jit_ctm_step(neighbors)` is a factory that returns
a fresh `@jit`'d Python function on each call. Every call site (forward
CTM convergence, implicit-AD `f_bwd`, line search, etc.) constructed a
new `_step` closure with its own JIT cache. JAX dispatches by Python
function identity, so 46 calls to `_make_jit_ctm_step` → 46 redundant
compiles of identical-shape `_step`. The 11 `while_loop` compiles trace
those `_step` functions and inherit the same redundancy.

### Fix: cache `_step` by `id(neighbors)`

Implemented in commit 11aafd3
(`src/tenax/algorithms/_ctm_python_loop.py`). Process-lifetime
`_JIT_STEP_CACHE` keyed by `id(neighbors)`. After-fix attribution at
the same chi=4 config:

| component | compiles (before → after) | total time (before → after) |
|---|---|---|
| `_step` | 46 → **1** | 13.8 s → 0.12 s |
| `while_loop` | 11 → out of top-10 | 19.7 s → < 1 s |

Saves ~33 s of redundant compile work at chi=4. At chi=16 each
individual `_step` compile is ~10× larger (chi enters as O(χ³) in the
projector SVDs); the absolute savings scale accordingly.

Verified by `uv run pytest -m core -x` (765 passed, 0 failed).

### Empirical result on the actual benchmark

Re-ran `compare.py` at the original protocol (TOL=1e-6, MAX_STEPS=100)
with the cache fix in place. **Tenax at single_site D=2 χ=16 still
exceeded the 30-min subprocess budget on CPU.** The fix removed the
redundant-compile tax, but the remaining per-step solve cost (eager
GMRES adjoint at chi=16, plus L-BFGS line search retries) on CPU is
genuinely slower than variPEPS's fixed-point adjoint by enough to push
the run past the 30-min ceiling.

variPEPS on the same protocol: E = −0.66251430 in 23 steps in
800 s (≈13.3 min, 170 s of which was JIT compile). See
`benchmarks/varipeps_compare/published_results/STATUS.md`.

### What this changes

- **F1-as-cache-by-id-of-neighbors** is a small win and **shipped** in
  11aafd3. It makes the benchmark *less wasteful* but not *fast enough*.
- **F2 (eager-GMRES → fixed-point iteration)** moves up the priority
  list. variPEPS's empirical convergence in 23 steps with fixed-point
  iteration is direct evidence the spectral radius of `J^T` is < 1
  here, which is what fixed-point needs.
- **Running on GPU** is a separate orthogonal lever — this machine has
  2× CUDA devices, but compare.py currently forces CPU to keep
  wall-clocks comparable. Once F2 lands, the remaining gap may close
  enough that GPU isn't strictly needed.
