# Single-matvec GMRES for the implicit-AD adjoint

Date: 2026-10-16. Status: design approved in conversation; spec under review.

## Problem

The eager-GMRES adjoint path in `src/tenax/algorithms/_ctm_energy_ad.py`
compiles the expensive J^T matvec (the VJP of one gauge-fixed CTM sweep) about
four times. The fused Neumann path compiles it once.

That path runs in two cases:

- `adjoint_method="gmres"`;
- the fallback after the fused Neumann loop diverges or does not converge.
  This is the common case near a slow forward fixed point.

Measured on CPU (AMD EPYC 7742), U(1) XXZ D=2 chi=8, main `e6d6c7e`, first
gradient:

| module | fixed_point | gmres |
|---|---|---|
| fused module (dE/denv + Neumann) | 611 s | - |
| `_jit_eager_gmres_solve` | - | 2,458 s |
| `_jit_apply_Jt` (standalone matvec) | - | 435 s |
| `_jit_dE_denv` | 48 s | 50 s |
| `_jit_chain_rule` | 498 s | 548 s |
| first gradient, total | 1,349 s | 3,729 s |

The HLO dumps show why:

- Before optimization, the GMRES program has 4 call sites to `_jit_apply_Jt`
  computations. Three of them are in JAX's GMRES: the initial residual, the
  Arnoldi step, and the residual at each restart.
- XLA inlines every call, so after optimization the program has 2.42M
  instructions. That is 2.8x the standalone matvec, which has 852k.
- The standalone `_jit_apply_Jt`, compiled for the seed choice
  (`_best_adjoint_seed`) and the residual check (`_measure_adjoint`), is a
  fourth copy.

On a U(1) D=3 chi=16 `optimize_gs_ad` run, measured by another session on a
GPU host, the first step spent 12,357 s compiling this fallback.

## Goal

The eager-GMRES adjoint path compiles the J^T matvec exactly once. Its
convergence contract stays the same.

Acceptance: gradients equal the current path to the solve tolerance. They do
not need to be bit-identical.

## Scope

In scope: the two eager-GMRES call sites in `_bwd_core`, `_ctm_energy_ad.py`:

- the `adjoint_method == "gmres"` branch;
- the fixed-point divergence/non-convergence fallback.

Out of scope, left unchanged:

- `_jit_gmres_solve` and `gmres_lax`;
- `gmres_pytree_jax` and its other callers (`_ctm_honeycomb_ad.py`);
- the C4v adjoint;
- `_best_adjoint_seed`, `_measure_adjoint` and `_warn_if_adjoint_unconverged`.
  They keep their other users.

## Design

### 1. Solver: `gmres_single_matvec` in `src/tenax/algorithms/_gmres_lax.py`

```python
def gmres_single_matvec(matvec, b_tree, candidates, *, tol, maxiter, restart):
    """-> (x_tree, abs_resid, b_norm)"""
```

**Representation.**

- Pytrees are flattened to one vector, as in `gmres_pytree_jax`.
- The Krylov basis `V` has shape `(restart + 1, n)`.
- Inner products use conjugating `vdot`, so complex inputs work.
- `candidates` is a list of pytrees shaped like `b_tree`; `None` entries are
  skipped.

**One loop, one matvec.**

- The solver is one `lax.while_loop`.
- The carry holds `phase`, `v_in`, `x`, `V`, the Hessenberg column store, the
  Givens `cs`/`sn`, `g`, the counters, the best seed, its residual, and `done`.
- The body computes `y = matvec(v_in)` exactly once.
- `lax.switch(phase, ...)` then consumes `y`. No branch calls `matvec`.

Phases:

- **SEED k.**
  - `res_k = ||b - y||` for candidate k.
  - The incumbent starts as zero with residual `||b||`, so zero costs no
    matvec.
  - A candidate replaces the incumbent only if it is *strictly* smaller. Zero
    wins ties, and an earlier candidate beats a later equal one. These are the
    `_best_adjoint_seed` rules (#858).
  - After the last candidate: `x = best`, `v_in = x`, go to RESID. If no
    candidate exists, enter RESID directly with `x = 0` (the first RESID
    matvec of zero is allowed).
- **RESID.**
  - `r = b - y` is the true residual of the current `x`.
  - Stop if `||r|| <= max(tol * ||b||, tol)`, the `_adjoint_converged` rule,
    or if `maxiter` cycles are spent.
  - Otherwise set `V[0] = r / ||r||` and `g = ||r|| e_1`, and go to ARNOLDI
    with `j = 0` and `v_in = V[0]`.
- **ARNOLDI j.**
  - Modified Gram-Schmidt of `y` against `V[0..j]`.
  - Apply the previous Givens rotations, form a new one, and update `g`.
  - The residual estimate is `|g[j+1]|`.
  - End the cycle if `j = restart - 1`, if the estimate meets the tolerance,
    or on breakdown (`h[j+1, j] <= eps * ||y||`). To end it: solve the
    `(j+1)x(j+1)` triangular system, update `x`, count one cycle, and go to
    RESID with `v_in = x`.
  - Otherwise set `V[j+1]` and `v_in = V[j+1]`.

Properties:

- Every exit passes through a RESID phase, so `abs_resid` is the measured
  true residual, never an estimate.
- `maxiter` counts restart cycles of up to `restart` Arnoldi steps each, the
  JAX GMRES meaning, so `gmres_maxiter` and `gmres_restart` keep their
  meaning.
- Matvecs per solve: one per non-`None` candidate, plus at most
  `(restart + 1)` per cycle. That is the same order as today.

### 2. Wiring in `_ctm_energy_ad.py`

**Compiled entry point.**

- `_jit_gmres_one_matvec_solve` replaces `_jit_eager_gmres_solve`.
- It keeps the decorator `block_sparse_jit(by_bond_dim=False,
  static_argnames=("apply_Jt", "chi", "pb", "tol", "maxiter", "restart"))`.
- It builds `matvec(v) = apply_Jt(params_data_tuple, env_leaves, v, chi=chi,
  pb=pb)` once and calls `gmres_single_matvec`.
- `apply_Jt` stays static (#1087), so the compiled loop is reused across
  backward calls.

**Call sites in `_bwd_core`.**

- The `adjoint_method == "gmres"` branch passes candidates `[warm, rhs]`, the
  same as today.
- The fixed-point fallback passes `[rhs]`, the same as today.
- Both drop their `_best_adjoint_seed(_eager_apply_I_minus_Jt, ...)` and
  `_measure_adjoint(_eager_apply_I_minus_Jt, ...)` calls.

**Reporting.**

- A new helper `_report_adjoint(abs_resid, b_norm, *, tol, maxiter, restart,
  warn) -> (rel, converged)` replaces `_measure_adjoint` at these two sites.
- It uses `_adjoint_converged`, which fails closed on NaN.
- It emits the same `RuntimeWarning` text when `warn` is true and the solve
  did not converge.
- `_measure_adjoint` delegates its tail to it, so the two share one code path
  for the threshold and the warning.

**The fallback reuses the fused module's `b`.**

- `_jit_fused_fixed_point_bwd` already computes `dE/denv` inside the program.
- It also returns that vector, and the fallback uses it as `rhs`.
- The fallback no longer needs `_eager_dE_denv()`, so `_jit_dE_denv` is not
  compiled on that path.
- The fused branch also called `_eager_dE_denv()` for its cold-start seed.
  The fused module therefore takes a traced `use_init` flag and seeds from
  its own `dE_denv` when the flag is false, so the fixed-point path never
  compiles `_jit_dE_denv`. The seed value is identical. (Ruling made while
  planning.)
- `adjoint_method="gmres"` still calls `_eager_dE_denv()`. It has no other
  source.

**Unchanged.**

- `adjoint_residual`, `prev_lam_leaves`, `warm_start_invalidated` and the
  other diagnostics.
- The chain-rule call.
- The #1028 flow/frozen retry.
- `arnoldi_precheck`, which still uses `_jit_apply_Jt` and is off by default.

**Compiles per path after the change.**

- `gmres`: dE/denv, one matvec (in the solver), chain rule.
- Fallback: fused module, one matvec, chain rule.
- Today: four matvecs, and dE/denv twice in the fallback.

## Testing

### Unit: `tests/test_gmres_single_matvec.py` (core)

- **Correctness.** On small dense systems (well-conditioned,
  ill-conditioned, non-normal, complex128): the residual meets
  `max(tol * ||b||, tol)` and `x` agrees with `numpy.linalg.solve` to the
  tolerance.
- **Budget.** With `maxiter` too small: the returned `abs_resid` is the
  honest residual above tol, and `_adjoint_converged` returns False.
- **#858.**
  - A candidate worse than zero is not chosen, so the result is never worse
    than `x = 0` after zero cycles.
  - A good warm start is chosen and needs fewer matvecs.
  - Ties go to zero.
- **Breakdown.** For `b` in a small invariant subspace, the solve stops early
  with the exact solution and does not divide by zero.
- **One matvec.**
  - Wrap the matvec in a named `jax.jit` function and lower
    `jax.jit(gmres_single_matvec-wrapper)`.
  - Assert exactly one call site to it in the pre-optimization HLO text.
  - Assert exactly one occurrence in the jaxpr, counted recursively through
    sub-jaxprs.

### Integration (existing fixtures)

- On the D=2 U(1) fixture: `adjoint_method="gmres"` and a forced fallback
  give the same gradient as `fixed_point` to 1e-8 relative.
- The existing adjoint tests must pass unchanged. These include the #858,
  #824 and #1028 tests and the gmres-vs-fixed_point parity tests (`-m "not
  slow"` regression set).

### Existing tests that name the replaced function

- `tests/test_adjoint_seed_858.py` spies on `_jit_eager_gmres_solve`. Its
  wrapper test checks that the caller's `x0` reaches GMRES.
- `tests/test_vjp_cache_key_1049.py` lists `_jit_eager_gmres_solve` among
  the per-entry compiled functions.
- Re-point both to `_jit_gmres_one_matvec_solve`, and keep what they assert:
  - the #858 spy must still see the candidates arrive, and the chosen seed
    must still never be worse than zero;
  - the #1049 list must still name the compiled solver.
- Do not delete either guard.

### Acceptance measurement (recorded in the PR)

- Same probe as the spike: CPU, D=2 chi=8 U(1), `adjoint_method="gmres"`, a
  fresh compile cache.
- The GMRES-solve compile drops from 2,458 + 435 s to at most about 800 s.
- The warm gradient is unchanged within noise.
- `|g|` matches 0.1872797387…
- **NO-GO** if the drop is less than 2x. Report it before merging.
- D=3 is optional, run when a quiet host is available.

## Docs

- CHANGELOG, under "Behavior Changes": the eager-GMRES adjoint compiles the
  matvec once; the gradient is unchanged to the solve tolerance.
- `docs/guide/algorithms/ipeps_ad_paths.md`: one paragraph in the
  compile-time section.

## Risks

- **Numerical differences from JAX's GMRES:** Gram-Schmidt order and restart
  bookkeeping. Covered by the tolerance-level acceptance and the parity tests.
- **`lax.switch` inside `while_loop` adds per-iteration overhead.** The
  branches are vector operations of size n, negligible next to a CTM-sweep
  VJP.
- **Memory.** `V` is `(restart + 1) x n`, the same as JAX's GMRES today.
