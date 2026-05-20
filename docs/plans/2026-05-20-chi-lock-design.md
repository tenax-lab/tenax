# chi-lock for `ctm_energy_implicit` — design

**Issue:** #516 (filed 2026-05-20)
**Branch:** `feat/chi-lock-516`
**Base:** `origin/main` @ `2e0b378` (post-PR #515 merge)

## Goal

Lift the defensive `NotImplementedError` raise in `ctm_energy_implicit` that
blocks `ctmrg_heuristic_increase_chi=True`, so the implicit-AD path can opt
into variPEPS-style in-CTM χ-bump without producing silently wrong gradients.

**Motivation is wall-clock, not correctness.** Fixed χ with χ-extrapolation
remains the community-standard reporting protocol. The bump is a variPEPS-style
optimization heuristic that avoids wasting L-BFGS steps at small χ early in
optimization, growing to `chi_max` once truncation error matters.

## Problem statement

PR #515 (Phase 2 of #492/#514) wired in-CTM bump into the explicit-AD warmup
path but left `ctm_energy_implicit` raising `NotImplementedError` because the
custom_vjp backward closure captures `chi` as a Python-int closure constant at
`_make_implicit_vjp_fn` build time:

- `_jit_apply_Jt` at `_ctm_energy_ad.py:895` — `jit_step_bwd(..., chi=chi, ...)`
- `_jit_chain_rule` at `_ctm_energy_ad.py:931` — same
- `_jit_fused_fixed_point_bwd` at `_ctm_energy_ad.py:1000` — same
- `_jit_gmres_solve` at `_ctm_energy_ad.py:1087+` — same

When the forward CTM bumps `chi=9 → chi=18`, `env_leaves` flow into `f_bwd` at
shape `(18, D², 18)` but the closure-captured `chi` is still 9. JAX retraces
on shape change but `chi=9` stays baked into `jit_step_bwd`, so the backward
re-truncates the env to χ=9 and the adjoint `(I − J^T)λ = b` is solved against
the wrong Jacobian.

## Approach chosen: Option D — lazy per-chi JIT cache via static_argnames

Three other approaches were considered (issue #516 documents A/B/C). D was
chosen because:

- **Forward bump semantics unchanged.** variPEPS-style in-CTM bump remains as
  shipped in PR #515; only the backward learns to follow.
- **No `_VJP_CACHE` surgery.** Outer custom_vjp closure builds once; per-chi
  backward traces are managed by JAX's own JIT cache.
- **No SVD-projector audit.** χ stays JIT-static (it controls truncation rank,
  which XLA wants static); we just multi-trace.
- **Compile-cost bounded.** Worst case 4 helpers × `(chi_max − chi_initial) /
  step_size + 1` traces. At defaults (9→24, step=2): 8 chi values, 32 traces.
  Pays once per benchmark run.

### Why not the alternatives

| Option | Why rejected |
|---|---|
| A — cache eviction between L-BFGS steps | Forces bump-between-steps instead of bump-within-CTM, re-opening the zero-padding hazard documented in `feedback_drop_chi_schedule_protocol`. Current call's backward still wrong if mid-CTM bump fires. |
| B — promote chi to a runtime arg in the JIT'd helpers | Requires splitting SVD projector into trace-static rank vs runtime-shape branches — deep audit. |
| C — pre-build closures for every reachable χ | Equivalent compile cost to D but eager; D is C done lazily. |

## Architecture

```
ctm_energy_implicit(...)              # public API
  └── (defensive raise lifted)
      ↓
_ctm_energy_implicit_dispatch(...)    # _VJP_CACHE keyed on chi_INITIAL
  ↓
_make_implicit_vjp_fn(chi=chi_INITIAL, ...)
  ├── @custom_vjp f(params)
  │     ├─ outside grad: runs forward, returns energy
  │     └─ inside grad: JAX pairs f_fwd / f_bwd
  │
  ├── f_fwd(params):
  │     envs, chi_post = _run_forward(params)        # NEW: chi_post out
  │     residuals = (params, env_leaves, chi_post)   # NEW: chi_post in residuals
  │     return energy, residuals
  │
  └── f_bwd(residuals, g):
        params, env_leaves, chi_post = residuals     # NEW: chi_post out
        # JAX JIT cache traces one binary per distinct chi_post:
        rhs   = _jit_dE_denv(params, env_leaves)              # chi-agnostic
        lam   = _jit_fused_fixed_point_bwd(..., chi=chi_post) # NEW: chi static
        grad  = _jit_chain_rule(..., lam, chi=chi_post)       # NEW: chi static
        return grad
```

### Key invariants

1. **`chi_initial`** (closure constant) — used for `_VJP_CACHE` key and forward
   env_init when caller passes `env_init=None`. Static across one dispatch
   identity (gate, neighbors, energy_fn).
2. **`chi_post`** (residual payload) — Python int extracted from
   `CTMLoopResult.final_chi`. Threads forward → backward per call. Possibly
   different on every call.
3. **Forward bump semantics unchanged.** `_run_ctm_loop_with_bump` keeps
   bumping mid-CTM exactly as it does today.
4. **JAX JIT cache handles per-chi compilation** for the four backward
   helpers automatically — no manual cache management on the Tenax side.

### Why `chi_post` works as a residual payload

`f_fwd` is called by `jax.custom_vjp` with concrete inputs (grad evaluates at
concrete points; `f_fwd` is not traced as JAX abstract values when called
outside a `@jax.jit` context). Inside `f_fwd`, the call to `_run_ctm_loop_with_bump`
runs as ordinary Python — calls to the JIT'd CTM step dispatch with concrete
shapes and return concrete arrays. `float(_max_S)` (the bump trigger) and
`chi_current` updates are pure Python (see `_ctm_loop_core.py:187-196`). By
the time `f_fwd` returns, `CTMLoopResult.final_chi` is a Python int that can
be carried in residuals.

`f_bwd` receives residuals exactly as `f_fwd` returned them, so `chi_post`
arrives as a Python int. Passing it as a `static_argnames=('chi',)` keyword to
the JIT'd helpers triggers a one-time trace per distinct chi value.

## Components

### Modified files

All changes in `src/tenax/algorithms/_ctm_energy_ad.py`:

| Function | Line(s) | Change |
|---|---|---|
| `ctm_energy_implicit` | 441-450 | Delete defensive raise; update docstring |
| `_sigma_gauged_ctm_converge` | 489-592 | Return `(envs, final_chi)` instead of `envs` |
| `_make_implicit_vjp_fn._run_forward` | 768-811 | Return `(envs, chi_post)` |
| `_make_implicit_vjp_fn.f_fwd` | 827-835 | Append `chi_post` to residuals tuple |
| `_make_implicit_vjp_fn.f` | 821-825 | Discard `chi_post` (no backward needed) |
| `_make_implicit_vjp_fn.f_bwd` | (existing) | Unpack `chi_post`; pass as static arg; bounds-check |
| `_jit_apply_Jt` | 871-905 | `@partial(jit, static_argnames=('chi',))`; replace closure `chi` with param |
| `_jit_chain_rule` | 907-943 | Same treatment |
| `_jit_fused_fixed_point_bwd` | 945+ | Same treatment |
| `_jit_gmres_solve` | 1087+ | Same treatment |
| `_jit_dE_denv` | 854-869 | Unchanged — chi-agnostic |

The closure-captured `chi` (renamed conceptually to `chi_initial`) stays in
scope for: `_VJP_CACHE` cache key, forward env_init, documentation.
Backward helpers no longer reference closure `chi`.

### `_VJP_CACHE` semantics

Unchanged. Still keyed on `chi_initial` + bump kwargs. Per-chi-post backward
traces are managed by JAX's JIT cache (separate from `_VJP_CACHE`), so
multiple chi_post values within one `_VJP_CACHE` entry coexist without
collision.

### Files NOT touched

- `_ctm_loop_core.py` — `CTMLoopResult.final_chi` already exposes what we need.
- `_ctm_python_loop.py` — different code path (env-cache warm-start, not
  custom_vjp); already works.
- `ipeps_ad_policy.py` — kwargs threading already in place from #515.
- `ipeps_optimize.py` — only the `gs_implicit_ad=True` warning text at
  2100-2111 needs a one-line update to drop the "NotImplementedError" note.

## Error handling & edge cases

### Cases that just work

| Case | Why |
|---|---|
| `chi_post == chi_initial` (no bump fired) | Backward static_arg = chi_initial; identical to pre-chi-lock behavior |
| `chi_post` differs across L-BFGS steps (e.g. 12, 16, 16) | JAX JIT cache traces once per visited chi; reuses on repeat |
| Sigma gauge across in-CTM bump | `_run_ctm_loop_with_bump` already captures `envs_at_iter_start` before bumping (`_ctm_loop_core.py:178`); pair semantics preserved |
| `prev_lam_leaves` warm-start at new chi | Existing shape-validation invalidation (PR #515 review fix) auto-clears the stale cache |
| Eager-GMRES fallback path | Same static-arg treatment on `_jit_gmres_solve`; same JIT cache benefit |

### Explicit handling required

1. **Existing defensive-raise tests** — deleted, not updated. Replace with new
   chi-bump tests in Section 4. Any analogous tests in
   `tests/test_ctm_in_loop_bump_ad_paths.py` (PR #515) are audited and updated.

2. **`chi_post` bounds sanity check.** `f_bwd` asserts
   `chi_initial <= chi_post <= chi_max` before passing to backward helpers.
   Defends against future `_run_ctm_loop_with_bump` regressions that could
   leak a malformed `final_chi`.

3. **`tenax.ctm.gmres` debug logger** (#499 already wired). Add one DEBUG line
   in `f_bwd` when `chi_post != chi_initial`:
   `"chi-lock: backward at chi_post={chi_post} (initial={chi_initial})"`.
   Lets us diagnose perf regressions and confirm bump is firing in benchmarks.

4. **JIT-trace spam upper bound.** Document in the chi-lock docstring that the
   number of distinct backward traces is bounded by
   `(chi_max - chi_initial) / step_size + 1`. If a regression causes per-call
   re-tracing, the wall-clock signature is unmistakable (~30s extra per call).

### Cases explicitly NOT handled

- **`chi_post` oscillating downward** — bump is monotonic; no downward path.
- **Concurrent calls with different `chi_post`** — Tenax doesn't support
  concurrent gradient evaluation on the same VJP cache entry.
- **`chi_post` exceeding `chi_max`** — `_run_ctm_loop_with_bump` already caps
  at `chi_max_eff` (`_ctm_loop_core.py:196`). Bounds-check above is
  defense-in-depth.

## Testing

### New file: `tests/test_ctm_energy_implicit_chi_bump.py`

| Test | Asserts |
|---|---|
| `test_implicit_ad_no_longer_raises_with_bump` | `ctm_energy_implicit(..., ctmrg_heuristic_increase_chi=True, chi_max=...)` returns a value |
| `test_chi_bump_fires_when_smin_above_threshold` | Force low threshold; `final_chi > chi_initial` via diagnostic readback |
| `test_chi_bump_does_not_fire_when_below_threshold` | High threshold; `final_chi == chi_initial` |
| **`test_ad_gradient_matches_fd_with_bump`** | **Correctness gate.** FD-vs-AD parity on D=2 χ_initial=4 χ_max=8 with forced bump. Tol 1e-5 abs, 1e-3 rel |
| `test_ad_gradient_equals_fixed_chi_when_no_bump_fires` | With `chi_max == chi_initial`, gradient matches pre-chi-lock fixed-chi baseline |
| `test_jit_trace_count_bounded_at_chi_max` | Optimizer traverses chi_initial → chi_max; distinct chi values traced ≤ bound |

### Correctness gate detail

```python
D, chi_initial, chi_max = 2, 4, 8
A = random_site_tensor(D)
gate = heisenberg_gate()

def loss(A_flat):
    A_ = A_flat.reshape(A.shape)
    return ctm_energy_implicit(
        {(0,0): A_, (1,0): A_}, neighbors, gate,
        chi=chi_initial,
        ctmrg_heuristic_increase_chi=True,
        ctmrg_heuristic_increase_chi_threshold=1e-12,  # forces bump
        chi_max=chi_max,
    )

grad_ad = jax.grad(loss)(A.flatten())
grad_fd = central_diff(loss, A.flatten(), eps=1e-4)
assert jnp.allclose(grad_ad, grad_fd, atol=1e-5, rtol=1e-3)
```

Threshold `1e-12` guarantees the bump fires on the first iteration. Confirms:
- Forward env grows χ=4 → χ_max=8
- Backward operates at χ_post=8
- Resulting gradient is correct against FD reference

### Modified test files

- **`tests/test_ctm_energy_implicit.py`** — delete the `NotImplementedError`
  regression test added in PR #515. Keep the
  `_make_minimal_site_tensors_for_validation` helper.
- **`tests/test_ctm_in_loop_bump_ad_paths.py`** — audit and update tests that
  asserted the implicit-AD raise.

### Bench-level acceptance gate (out of CI)

`examples/heisenberg_ipeps_ad_2x2_v9b_implicit_bump.py`:

- D=3, `unit_cell="2site"`, `gs_implicit_ad=True`, `gs_c4v=False`
- `chi_initial=9`, `chi_max=24`, `ctmrg_heuristic_increase_chi=True`,
  threshold=1e-6, step=2
- **Pass:** wall-clock ≤ v7b (8h52m); energy within numerical tolerance of v7b
  (~−0.6225)
- **NOT a QMC-parity gate.** Fixed χ remains the production protocol;
  this benchmark only verifies bump saves cycles without changing the
  fixed-point.

## Out of scope

- Lifting the `chi_ramp + ctmrg_heuristic_increase_chi` mutex.
- Multisite / 3-site PESS paths (covered by #411).
- Deleting `chi_ramp` entirely (covered by #512).
- Public API simplification (Reading C from the design session — drop
  `chi_initial` from public API in favor of `chi_max`-only). Deferred to a
  separate PR after chi-lock lands.

## Related

- #492 — original in-CTM bump feature request
- #513 — Phase 1 (env-cache warm-start path)
- #514 / PR #515 — Phase 2 (AD paths; explicit enabled, implicit raised)
- #516 — this issue (chi-lock for implicit-AD)
- #328 — explicit-AD non-variational drift (separate; not affected here)
- `feedback_drop_chi_schedule_protocol.md` — why the bump-between-steps
  variant (Option A) was rejected
- `project_v7_fixed_chi_results.md` — fixed-χ=24 baseline for v9b gate
