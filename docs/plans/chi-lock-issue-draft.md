# [Draft] feat(ipeps-ad): chi-lock for `ctm_energy_implicit` so implicit AD can opt into in-CTM χ-bump

**Status:** Draft for filing after PR #515 lands.

**Title (for `gh issue create --title`):**
> chi-lock for `ctm_energy_implicit` — unblock implicit AD + in-CTM χ-bump (#514 follow-up)

**Labels:** `area/ipeps`, `area/ctm`, `kind/feature`, `priority/high`

---

## Summary

PR #515 (Phase 2 of #492 / #514) wired the variPEPS-style in-CTM χ-bump into the explicit-AD warmup path, but left `ctm_energy_implicit` defensively raising `NotImplementedError` whenever `ctmrg_heuristic_increase_chi=True`. This issue tracks lifting that raise.

**Motivation is wall-clock, not correctness.** Fixed χ with χ-extrapolation remains the community-standard reporting protocol, and the variational property `E(χ) ≥ E_exact` holds at any fixed χ. The bump is a variPEPS-style optimization heuristic to avoid wasting L-BFGS steps at small χ early in optimization, then growing to a final χ_max once truncation error matters. This issue makes that heuristic *available* on the implicit-AD path; production benchmarks may still prefer fixed χ.

## The bug (why the raise exists)

`_make_implicit_vjp_fn` (`src/tenax/algorithms/_ctm_energy_ad.py:714-737`) is a factory that closes over `chi` as a Python int and uses it inside three `@jax.jit`'d backward helpers:

- `_jit_apply_Jt` — `_ctm_energy_ad.py:895` → `jit_step_bwd(..., chi=chi, ...)`
- `_jit_chain_rule` — `_ctm_energy_ad.py:931` → `jit_step_bwd(..., chi=chi, ...)`
- `_jit_fused_fixed_point_bwd` — `_ctm_energy_ad.py:1000` → `jit_step_bwd(..., chi=chi, ...)`

`jit_step_bwd = _make_jit_ctm_step(neighbors)` (`:838`) uses that `chi` as the truncation rank of the renormalization step's SVD projector.

When `ctmrg_heuristic_increase_chi=True`, the forward CTM may grow the env's first-dim from e.g. χ=9 → 18 mid-convergence. `f_fwd` then hands `env_leaves` of shape `(18, D², 18)` to `f_bwd` as residuals, but the closure-captured `chi` is still 9. Two failure modes:

1. **Silent retrace.** JAX retraces because abstract shapes changed, but `chi=9` still goes into `jit_step_bwd` inside the new trace. The backward CTM step truncates the env back to χ=9.
2. **Wrong VJP.** The adjoint `(I − J^T)λ = ∂E/∂env` solves with `J^T` of "step with χ=9 projector applied to a χ=18 env" — not the Jacobian of the actual forward step at χ=18. The chain-rule term `J_params^T λ` (`:931`, `:1000`) is similarly broken. The optimizer descends against a fictitious objective — same flavour of pathology as the v8b ghost-minimum (`feedback_drop_chi_schedule_protocol`) but baked into every L-BFGS step.

Defensive raise: `_ctm_energy_ad.py:441-450`.

## The cache layer

`_VJP_CACHE` (`:595`) keys on the static config including `chi`, `ctmrg_heuristic_increase_chi`, `chi_max`, etc. (`:641-666`). The cache is keyed by *build-time* chi, so a forward-side bump within a single call doesn't refresh the closure.

## Proposed approaches

### Option A — cache invalidation on `final_chi` change *(recommended first cut)*

After forward, read `final_chi` from the warmup result. If it differs from the closure's `chi_at_build`, evict the `_VJP_CACHE` entry and rebuild `_make_implicit_vjp_fn(..., chi=final_chi, ...)`. Backward then operates at the new chi.

- **Pros:** small diff, mechanical, easy to test, doesn't touch JIT internals.
- **Cons:** loses JIT trace reuse across L-BFGS steps that trigger bumps. Once χ stabilises at χ_max, reuse resumes.
- **Effort:** ~few hundred LoC + tests. 3–5 days.
- **Risk:** low. Worst case: extra compile time on bump steps.

### Option B — promote chi to a runtime arg

Drop `chi=chi` from the JIT'd helpers' static config; pass it as a runtime arg, or infer from `env_leaves[0].shape[0]`.

- **Pros:** variPEPS-correct fix. One closure handles every χ in `[chi_initial, chi_max]`. No recompile on bump.
- **Cons:** requires auditing `_make_jit_ctm_step` to confirm `chi` is not a true XLA-static knob (it goes into the SVD projector which **does** want a static truncation rank). May force splitting the projector into trace-static rank vs runtime-shape branches.
- **Effort:** ~1–2 weeks + careful tests. Touches the projector path.
- **Risk:** medium-high. Subtle bugs in the SVD projector's static-vs-runtime split could land silently.

### Option C — pre-build closures for every reachable χ

For `chi_max=24` and step size 2 starting from χ=9: build closures at χ∈{9,11,13,…,23,24}. Dispatch by `final_chi` lookup.

- **Pros:** preserves trace reuse without runtime-chi audit.
- **Cons:** multiplies compile cost by chi-step count (~8× here). Awkward when `chi_max` is large or step size small.
- **Effort:** lower than B, higher than A. Tests need to assert every closure is reachable.
- **Risk:** medium (cache-explosion footgun if defaults change).

**Recommendation:** Ship A first to unblock v9b benchmarks; revisit B as a perf follow-up if A's recompile cost dominates wall-clock at large `chi_max`.

## Acceptance criteria

1. `ctmrg_heuristic_increase_chi=True` no longer raises in `ctm_energy_implicit`.
2. Forward CTM grows χ when `norm_smallest_S > eps`; backward operates at the post-forward χ.
3. Gradient correctness test: FD-vs-AD parity on a small probe (D=2, χ_initial=4, χ_max=8) where bump is forced to trigger.
4. Sigma-gauge path remains correct (regression test mirroring `test_ctm_loop_core.py` sigma-gauge regression added in #515).
5. v9b benchmark (D=3, 2-site bipartite, implicit AD, `chi_initial=9`, `chi_max=24`, no schedule) runs to convergence with wall-clock **at or below** the equivalent fixed-χ=24 run from v7b (8h52m). This is the *wall-clock* gate: bump should save cycles at early L-BFGS steps without producing a worse fixed-point at the same χ_max. Energy should land within numerical tolerance of fixed-χ=24 v7b (~−0.6225); **not** a QMC-parity gate.
6. `_F3_LAST_DIAGNOSTICS` / `tenax.ctm.gmres` debug logger emits a "chi-rebuild" event when invalidation fires, so wall-clock-perf regression can be diagnosed.

## Test plan

- **Unit:** Add `test_ctm_energy_implicit_chi_bump.py`. Force a bump during forward, assert (a) no raise, (b) `f_bwd`'s captured chi matches `final_chi`, (c) AD gradient matches FD to 1e-5 at D=2 χ=4→8.
- **Cache:** Assert `_VJP_CACHE` entry evicted/rebuilt when forward chi changes; existing entries at fixed chi unchanged.
- **Regression:** Existing 856 `-m core` tests still pass; existing defensive-raise tests in `test_ctm_energy_implicit.py` are updated to reflect the new behaviour (raise should no longer fire).
- **Benchmark gate:** v9b reaches within tolerance of QMC, or we file a follow-up explaining the residual gap.

## Not in scope

- Lifting the bump prohibition for the explicit-`chi_ramp` path (`chi_ramp + ctmrg_heuristic_increase_chi` mutex stays).
- Multisite / 3-site PESS paths (covered by #411).
- Deleting `chi_ramp` entirely (covered by #512).

## Related

- #492 — original in-CTM bump feature request
- #513 — Phase 1 (env-cache warm-start path)
- #514 / PR #515 — Phase 2 (AD paths; explicit-AD enabled, implicit-AD raises)
- #512 — deprecate scheduled chi-ramp + end-of-step bump
- #328 — explicit-AD non-variational drift (separate; not affected by this issue)
- `feedback_drop_chi_schedule_protocol.md` — why scheduled chi-ramp is the wrong alternative
- `project_v7_fixed_chi_results.md` — fixed-χ=24 baseline (wall-clock 8h52m on the v5 protocol)
