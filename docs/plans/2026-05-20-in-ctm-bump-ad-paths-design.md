# In-CTM χ-bump in implicit-AD and explicit-AD forward loops (#514) — design

**Date**: 2026-05-20
**Branch**: `feat/in-ctm-chi-bump-ad-paths-514`
**Stacked on**: PR #513 (merged to main 2026-05-20)
**Related**: #492 (in-CTM bump implementation), #511 (variational docs), #499 (GMRES log), #501 (warm-start adjoint), #512 (deprecation timeline)

## Summary

PR #513 (#492 Phase 1) landed variPEPS-style in-CTM χ-bump infrastructure in
`python_loop_ctm_converge`. Codex review flagged that the bump knobs flow into
`ctm_converge_kwargs` but never reach the **AD gradient evaluations**:
`ctm_energy_implicit` and `ctm_energy_explicit` both carry their own hand-rolled
CTM convergence loops with `chi=chi` (the original config value, not a bumped
one). Net effect: enabling `ctmrg_heuristic_increase_chi` produces no
observable change in gradient quality — every AD eval silently re-truncates the
cached env back to the original chi.

This PR (Phase 2) plumbs the bump knobs into both AD forwards via a shared
helper that consolidates the three nearly-identical CTM convergence loops.

## Acceptance (from issue #514)

- `ctm_energy_implicit` forward accepts the four new knobs.
- `ctm_energy_explicit` warmup similarly bump-aware; backprop forces fixed chi.
- Shared CTM loop helper extracted; bump logic lives in one place.
- 18 existing in-CTM bump tests still pass.
- New integration test: 1 L-BFGS step with `ctmrg_heuristic_increase_chi=True`,
  `chi_init=4`, `chi_max=12`; assert gradient evaluated at chi ≥ 5.
- #511 warning updated; #499 GMRES log instrumented; #501 adjoint warm-start landed.
- CI green on Python 3.11 / 3.12 / macOS.

## Architecture

### New file

`src/tenax/algorithms/_ctm_loop_core.py` — exports `_run_ctm_loop_with_bump`
and `CTMLoopResult` NamedTuple. New module rather than embedding in
`_ctm_python_loop.py` to avoid the `_ctm_energy_ad.py → _ctm_python_loop.py`
circular path (`_ctm_energy_ad.py` already imports `python_loop_ctm_converge`).

### Modified files

| File | Change |
|---|---|
| `src/tenax/algorithms/_ctm_loop_core.py` | NEW — helper |
| `src/tenax/algorithms/_ctm_python_loop.py` | `python_loop_ctm_converge` body replaced with helper call; `CTMConvergeInfo` populated from `CTMLoopResult` |
| `src/tenax/algorithms/_ctm_energy_ad.py` | `_sigma_gauged_ctm_converge` body replaced; `ctm_energy_explicit` warmup replaced; `ctm_energy_implicit` gains 4 knobs; `_VJP_CACHE` key extended; `_jit_fused_fixed_point_bwd` gains `init_lam` input (#501); GMRES debug log (#499) |
| `src/tenax/algorithms/ipeps_ad_policy.py` | `make_ctm_energy_fn` plumbs 4 knobs to both AD entry points |
| `src/tenax/algorithms/ipeps_optimize.py` | #511 warning update at `:2100-2107` |
| `tests/test_ctm_in_loop_bump_ad_paths.py` | NEW — ~13 tests |

### Out of scope

- `_ctm_honeycomb_ad.py` — has its own honeycomb-shaped forward CTM; deferred follow-up.
- Refactoring `_ctm_python_loop._python_loop_chi_ramp` — already disallowed with
  bump (mutex from PR #513).
- F3 backward fusion (`_jit_fused_fixed_point_bwd`) restructure — only adds the
  `init_lam` input.

## Helper

```python
def _run_ctm_loop_with_bump(
    jit_step,                         # caller-supplied JIT step
    site_tensors,
    envs_init,                        # already-initialised envs dict
    *,
    chi_current: int,
    chi_max: int | None,
    bump_enabled: bool,
    bump_threshold: float,
    bump_step_size: int,
    projector_method: str,
    renormalize: bool,
    projector_backward: str,
    gauge_fix_fn,                     # (envs_new, envs_old) -> envs or None
    max_iter: int,
    min_iter: int,
    conv_tol: float,
    conv_method: str,                 # "sv" | "elementwise"
    plateau_patience: int | None,
    bump_base_charges,
) -> CTMLoopResult:
    ...

class CTMLoopResult(NamedTuple):
    envs: dict[Coord, CTMTensorEnv]
    converged: bool
    iterations: int                   # physical sweeps incl bump-extras
    sv_diff: float
    max_truncation_error: float
    max_smallest_S: float
    final_chi: int
    bump_extra_sweeps: int
```

### Gauge-fn pair semantics

Helper always calls `gauge_fix_fn(envs_new, envs_old)`. Callers wrap:
- Phase: `lambda new, _old: {c: _phase_fix_ctm_tensor(new[c]) for c in new}`
- Sigma: `lambda new, old: {c: _sigma_gauge_fix_env(new[c], old[c]) for c in new}`
- None: `None`

### Caller responsibilities (outside the helper)

- QR warmup loop (different per caller).
- Env initialization (caller decides whether to honor `env_init` or fresh-init).
- Validation (chi_max present when bump on, etc.) — caller-specific error messages.
- Computing `bump_base_charges` — caller already builds double-layer tensors.

### Load-bearing design points

1. **JIT step passed in, not created** — `python_loop_ctm_converge` memoises
   `_make_jit_ctm_step(neighbors)` by `id(neighbors)`. Helper creating it would
   destroy the cache.
2. **`bump_base_charges` passed in** — caller computes once outside the loop.

## AD-path integration

### `ctm_energy_implicit` (`_sigma_gauged_ctm_converge`)

- Accepts the four new kwargs.
- Validation block mirrors `python_loop_ctm_converge`:
  - bump=True ⇒ chi_max required
  - bump=True ⇒ step_size > 0
  - env_init.chi > chi_max raises
  - chi_max < chi_current raises (post warm-start finalize)
- chi_current derived from env_init shape when bump is on (warm-start round-trip).
- Builds `bump_base_charges` from one site tensor.
- Builds gauge_fix_fn from `forward_gauge` argument.
- Calls helper, returns `result.envs`.

`ctm_energy_implicit` itself plumbs the four kwargs through to dispatch.
`_VJP_CACHE` key gains 4 entries: `(bump_enabled, threshold, step_size, chi_max)`.

### `ctm_energy_explicit`

```python
# WARMUP — bump-aware, no-grad
result = _run_ctm_loop_with_bump(
    jit_step, site_tensors, envs,
    chi_current=chi, chi_max=chi_max,
    bump_enabled=ctmrg_heuristic_increase_chi,
    ...,
    max_iter=warmup_steps,
    gauge_fix_fn=None,
    conv_method="sv", conv_tol=jnp.inf,
    plateau_patience=None,
    min_iter=warmup_steps + 1,        # disables convergence check
)
envs = jax.lax.stop_gradient(result.envs)
chi_post_warmup = result.final_chi    # ← LOCKED for backprop

# BACKPROP — fixed chi, no bump (tape integrity)
def _step_envs_only(st, e):
    envs_out, _, _ = jit_step(st, e, chi=chi_post_warmup, ...)
    return envs_out
for _ in range(backprop_steps):
    envs = jax.checkpoint(_step_envs_only)(site_tensors, envs)
```

**Key invariant**: backprop uses `chi_post_warmup` (resolved at trace time as a
Python int), so the checkpointed sweeps trace once and reuse. Without this lock
the bump would retrace every backprop step.

Warmup historically does not gauge-fix (line 67 uses bare `jit_step`). Keep
that. Adding gauge-fix in warmup is out of scope for #514.

## Bundled items

### #511 — variational warning update

`ipeps_optimize.py:2100-2107` current text says variational "at chi >= 16".
New text references the in-CTM bump as the automated path:

> "This is variational when the CTM environment is converged; pass
> `ctmrg_heuristic_increase_chi=True` with `chi_max` set (variPEPS-style
> in-CTM bump, issue #492) to grow chi automatically until the truncation gap
> closes. Without the bump, chi must be set high enough manually (chi >= 16
> for generic 2-site Heisenberg)."

### #499 — GMRES logging

Two log points in `_ctm_energy_ad.py`:

1. After `_F3_LAST_DIAGNOSTICS` populate in `f_bwd`: `_GMRES_LOGGER.debug` with
   `n_iter`, `converged`, `diverged`, `gmres_tol`.
2. Eager-GMRES fallback (line 1064) and `adjoint_method == "gmres"` branch
   (line 1079): debug-log `maxiter / tol / restart` and post-solve residual.

Diagnostics dict gains `gmres_n_iter` and `gmres_final_residual` keys.

### #501 — warm-start adjoint solve

Per-VJP-cache state (NOT module-level):

```python
# In _make_implicit_vjp_fn, add to _cached:
_cached["prev_lam_leaves"] = None
```

Extend `_jit_fused_fixed_point_bwd` to accept `init_lam` as a regular input
(default: `dE_denv`). `f_bwd` passes `_cached["prev_lam_leaves"]` when
available, falls back to `dE_denv` otherwise. After successful solve, store
`lam_leaves` back.

Invalidation rule: clear `prev_lam_leaves` when `f_bwd` diverges OR eager-GMRES
fallback fires. Warm-start is best-effort — never load-bearing for correctness.

## Validation distribution

| Check | Where |
|---|---|
| bump=True ∧ chi_max=None | CTMConfig + `python_loop_ctm_converge` + `ctm_energy_implicit` + `ctm_energy_explicit` |
| bump=True ∧ step_size <= 0 | same |
| bump=True ∧ chi_ramp not None | same |
| env_init.chi > chi_max | direct callers only |
| chi_max < chi_current | direct callers only |
| bump=True ∧ chi_auto_bump=True | CTMConfig only (from #513) |

Helper trusts its inputs. All raises live in callers and config.

## Testing strategy

New `tests/test_ctm_in_loop_bump_ad_paths.py` (~13 tests):

- `test_implicit_ad_forward_grows_chi` — bump fires during implicit-AD forward
- `test_explicit_ad_warmup_grows_chi` — bump fires during explicit-AD warmup
- `test_explicit_ad_backprop_chi_locked` — no retrace; backprop uses post-warmup chi
- `test_implicit_ad_chi_monotone_growth` — 5 sequential calls; monotone `final_chi`
- `test_ctm_energy_implicit_chi_max_none_raises`
- `test_ctm_energy_explicit_chi_max_none_raises`
- `test_implicit_ad_env_init_above_chi_max_raises`
- `test_implicit_ad_chi_ramp_plus_bump_raises`
- `test_implicit_ad_gradient_matches_finite_diff_after_bump` (acceptance)
- `test_adjoint_warm_start_grad_unchanged` (#501)
- `test_adjoint_warm_start_reduces_iters` (#501)
- `test_adjoint_warm_start_invalidated_on_divergence` (#501)
- `test_gmres_logging_emits_n_iter` (#499)
- `test_gmres_logging_records_eager_fallback` (#499)

All `@pytest.mark.core`. D=2, chi_init=4, 1-5 sweeps — small budget.

**Existing 18 in-CTM bump tests** (`tests/test_ctm_in_loop_chi_bump.py`) unchanged
and must still pass — they exercise the helper's logic via `python_loop_ctm_converge`.

## Estimated diff

~400-500 lines:
- `_ctm_loop_core.py`: ~200 lines
- `_ctm_python_loop.py`: -200 +30 lines (delegates to helper)
- `_ctm_energy_ad.py`: -100 +150 lines (2 forward rewrites + 2 bundled fixes)
- `ipeps_ad_policy.py`: +10 lines (knob plumbing)
- `ipeps_optimize.py`: 1-line warning update
- New test file: ~250 lines

## Risk

- **JIT recompile blowup**: 4 new knobs in `_VJP_CACHE` key could fragment caches.
  Mitigation: knobs are static config (don't change inside an L-BFGS run); cache
  per (bump_config, chi_max) is correct behavior.
- **Backprop tape integrity in explicit-AD**: `chi_post_warmup` is captured by
  closure on the loop iteration, then used as `static_argname="chi"` on
  `jit_step`. Single trace per chi value, retrace if `chi_post_warmup` changes
  across calls. Same behavior as today's `chi=chi` capture.
- **Adjoint warm-start staleness across stall-recovery**: invalidation on divergence
  + eager-fallback covers this. Tests pin the invalidation behavior.
