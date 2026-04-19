# JIT-Compiled GMRES Backward for CTM Implicit Differentiation

**Date:** 2026-04-16
**Status:** COMPLETED (mechanically) — JIT GMRES backward works but doesn't fix non-C4v 2-site (ρ(J^T) ≫ 1). `jit_ctm` path was subsequently removed in PR #337. Issue #328 resolved via shared-C4v + implicit AD instead.
**Issue:** #328 (non-C4v 2-site non-variational energy)
**Branch:** `exp/noc4v-no-loss-normalization-328`

## Problem

Non-C4v 2-site CTM backward (`_ctm_tensor_converge_bwd`) runs eagerly in
Python.  Each GMRES iteration dispatches a separate XLA kernel, making the
backward ~50× slower than variPEPS's JIT-fused approach.  The Neumann series
diverges on this problem (J^T spectral radius ≫ 1), so GMRES is the only
viable implicit-diff solver — but it must be fast enough to be practical.

## Solution

When `jit_ctm=True` and `ad_backward_method="gmres"`, wrap the backward
GMRES solve in `@jax.jit` so the VJP applications + GMRES while_loop +
site projection compile into a single XLA program.

### Changes to `_ctm_tensor_converge_bwd` (GMRES branch only)

1. **JIT the solve.**  Define an inner `@jax.jit` function that:
   - Recomputes VJPs via `jax.vjp` (traced symbolically under JIT)
   - Applies `(I − J^T + τI)` as a matrix-free operator
   - Calls `jax.scipy.sparse.linalg.gmres` with configurable iterations
   - Projects result to site space via `vjp_site_fn`

2. **Use `adjoint_maxiter`** instead of `min(config.max_iter, 50)`.
   Default stays 50; users set higher (200+) for non-C4v 2-site.

3. **Skip Arnoldi precheck** when `jit_ctm=True` — it raises Python
   exceptions (not JIT-traceable), and GMRES handles ill-conditioning
   via least-squares residual.

4. **Include Tikhonov damping** from existing `adjoint_tikhonov` config:
   `(I − J^T + τI)λ = g`.  Already in the config (default 1e-6), just
   not wired into the GMRES path yet.

5. **Convergence signaling.**  After JIT, materialize GMRES `info`
   (0 = converged, >0 = not) and raise `CTMRGGradientError` if
   non-zero — the optimizer's existing stall recovery handles it.

### What stays the same

- Neumann path (`ad_backward_method="vjp"`) — unchanged, still eager
- Non-JIT GMRES (`jit_ctm=False`) — unchanged, still uses current eager path
- Forward path — already has JIT support via `jit_ctm`
- Config fields — no new fields; reuses `jit_ctm`, `adjoint_maxiter`,
  `adjoint_tikhonov`, `adjoint_tol`

### Architecture reference

variPEPS (`varipeps/ctmrg/routine.py`) JIT-compiles `_ctmrg_rev_workhorse`
with `@jit`, containing GMRES via `jax.scipy.sparse.linalg.gmres` inside a
`jax.lax.cond` branch.  The VJP closure is passed as `tree_util.Partial`.
Tenax's approach is simpler: define the JIT function inside the backward
(capturing VJP closures), letting JAX trace through them.

## Success criterion

D=2 Heisenberg non-C4v 2-site, chi=16, `jit_ctm=True`,
`ad_backward_method="gmres"`, `adjoint_maxiter=200`: produces variational
energy (> −0.6694) within 30 L-BFGS steps at reasonable wall time.
