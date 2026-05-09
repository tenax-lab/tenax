# F2: Fixed-Point Adjoint for Tenax Implicit-AD CTM

> **For Claude (new session):** This is the F2 follow-up to PR #414 (squash 510fca4).
> Start here. The diagnosis doc at
> `docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md` has the full
> background; you don't need to re-derive any of it.
>
> **Branch:** `ipeps-ad-fixed-point-adjoint` (already created off main).
>
> **REQUIRED SUB-SKILL:** Use `superpowers:executing-plans` to implement this plan task-by-task.

## Goal

Replace the eager Krylov-GMRES adjoint solver in
`src/tenax/algorithms/_ctm_energy_ad.py:f_bwd` with a Python-loop
fixed-point iteration, mirroring variPEPS's
`_ctmrg_rev_workhorse` design. The motivating evidence:

- variPEPS solves the same physics (same protocol, same chi) using a
  fixed-point iteration and converges in ~23 outer steps in ~13 min on
  CPU. Tenax's eager-GMRES adjoint exceeds the 30-min subprocess budget
  even after the JIT cache fix in PR #414.
- For the variational regime (chi ≥ 16) with the phase-gauge default,
  the spectral radius ρ(J^T) is < 1 (verified by the existing Arnoldi
  precheck and by variPEPS's empirical convergence). When ρ < 1,
  `λ_{k+1} = b + J^T λ_k` converges geometrically — no Krylov subspace
  needed.

Expected outcome:
- Tenax `f_bwd` per-call wall-clock drops 3–5×.
- Combined with the F1 cache fix from #414, the variPEPS-compare
  benchmark fits comfortably inside its 30-min subprocess budget.

## Code change scope

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (only `f_bwd` body and
  one new helper near it)

**Out of scope:**
- The forward CTM (`_run_forward`) is untouched.
- The custom_vjp boundary stays.
- `_jit_apply_Jt`, `_jit_chain_rule`, `_jit_dE_denv` stay as separate
  JIT'd functions — F1 already showed combining them doesn't help.
- The Arnoldi precheck stays (and becomes load-bearing — see below).
- The existing `_jit_gmres_solve` and `gmres_pytree_jax` callers stay
  available behind a config knob (see "Knob & fallback" below).

## Algorithm (drop-in replacement for the GMRES block)

Replace this block (currently lines 830–844 of `_ctm_energy_ad.py`):

```python
def _eager_apply_I_minus_Jt(v):
    return _jit_apply_Jt(params_data_tuple, env_leaves, v)

lam, _info = gmres_pytree_jax(
    _eager_apply_I_minus_Jt,
    dE_denv, dE_denv,
    tol=gmres_tol, maxiter=gmres_maxiter, restart=gmres_restart,
)
lam_leaves = tuple(jax.tree.leaves(lam))
```

with:

```python
# Fixed-point iteration: λ_{k+1} = b + J^T λ_k.
# Equivalent to solving (I - J^T) λ = b iff ρ(J^T) < 1, which the Arnoldi
# precheck above guarantees when arnoldi_precheck=True.
def _apply_Jt(v):
    i_minus_jt_v = _jit_apply_Jt(params_data_tuple, env_leaves, v)
    return tuple(vi - im for vi, im in zip(v, i_minus_jt_v))

lam = dE_denv  # initialize at b (one matvec saved)
prev_diff = float("inf")
for k in range(gmres_maxiter):
    jt_lam = _apply_Jt(lam)
    new_lam = tuple(b + j for b, j in zip(dE_denv, jt_lam))
    diff = sum(
        float(jnp.linalg.norm(n - p))
        for n, p in zip(new_lam, lam)
    )
    lam = new_lam
    if diff < gmres_tol:
        break
    if diff > prev_diff and k > 5:
        # Diverging — fall back to the GMRES path for safety.
        lam, _info = gmres_pytree_jax(
            _eager_apply_I_minus_Jt,
            dE_denv, dE_denv,
            tol=gmres_tol, maxiter=gmres_maxiter, restart=gmres_restart,
        )
        break
    prev_diff = diff
lam_leaves = tuple(jax.tree.leaves(lam))
```

The loop is short and entirely in Python; each iteration calls one
already-cached `_jit_apply_Jt` matvec (no per-iter compile cost after
the first). Convergence check is element-wise L2 norm of
(λ_{k+1} - λ_k), summed across pytree leaves.

## Knob & fallback

Add `adjoint_method: Literal["fixed_point", "gmres"] = "fixed_point"`
to `CTMConfig` in `src/tenax/algorithms/ipeps_config.py`. Wire it
through `_implicit_vjp_dispatch` and `_make_implicit_vjp_fn` to
`f_bwd`. When `adjoint_method == "gmres"`, keep the current
GMRES-eager path verbatim (same code that's there now). When
`"fixed_point"` (default), use the new loop above.

This way:
- New default behavior is the fast fixed-point path.
- Anyone bitten by a divergence edge case can opt out via
  `CTMConfig(..., adjoint_method="gmres")`.
- Existing call sites that pass `gmres_*` parameters continue to work
  — the fixed-point loop reuses `gmres_tol` and `gmres_maxiter` as
  its own tolerance and step cap.

## Tests

**Existing tests that must continue to pass (regression):**
- `tests/test_ipeps_excitations.py::TestOptimizeGsAd::test_runs_without_error`
- `tests/test_ipeps_excitations.py::TestOptimizeGsAd::test_energy_decreases`
- `tests/test_ipeps_excitations.py::TestOptimizeGsAd::test_heisenberg_negative_energy`
- `tests/test_ipeps_excitations.py::TestOptimizeGsAd::test_heisenberg_excitation_dispersion`
- `uv run pytest -m core -x` (765 passed before #414, 789 passed after)

**New test:** `tests/test_ipeps_ad_adjoint_methods.py`
- `test_fixed_point_matches_gmres_at_chi8`: run `optimize_gs_ad` once
  with `adjoint_method="gmres"` and once with `"fixed_point"` at the
  same seed, D=2, chi=8, gs_num_steps=2; assert |E_gs_diff| < 1e-6
  AND tensor-element-wise rtol < 1e-5 between the two final tensors.
  Marked `@pytest.mark.algorithm`.
- `test_fixed_point_arnoldi_rejects_high_rho`: contrive a config where
  ρ(J^T) ≥ 1 (e.g. set `forward_gauge="none"` at small chi); assert
  the existing `CTMRGGradientError` is raised before the fixed-point
  loop runs (so the precheck still catches divergent cases).

**Bench:** re-run the variPEPS-compare benchmark after the fix lands:
```bash
JAX_PLATFORMS=cpu uv run python -m benchmarks.varipeps_compare.compare \
    --device cpu --results-dir benchmarks/varipeps_compare/results
```
At single_site D=2 chi=16, Tenax should now complete inside 30 min on
this machine. Update `benchmarks/varipeps_compare/published_results/`
with the new data + summary.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Fixed-point diverges where GMRES would have converged | Low at chi ≥ 16 with phase gauge (ρ < 1 verified empirically); Medium at chi < 16 or sigma gauge | Arnoldi precheck rejects ρ ≥ 1 cases. In-loop divergence detector (`diff > prev_diff` after step 5) falls back to GMRES. Knob lets users force GMRES. |
| Slightly different gradient (rounding) breaks downstream tests | Low | Both methods solve the same linear system; differences should be at GMRES tolerance level (1e-6). Test asserts |ΔE| < 1e-6 and tensor rtol < 1e-5. |
| Per-step Python overhead is higher than GMRES per-iter cost | Low | Each fixed-point iter is one `_jit_apply_Jt` matvec, same as one GMRES inner iter. Fewer iters in fixed-point (no Krylov restart overhead). |
| `gmres_tol` / `gmres_maxiter` defaults are wrong for fixed-point | Low | Existing defaults (`gmres_tol=1e-6`, `gmres_maxiter=200`) are conservative. Bench will surface if 200 iters isn't enough. |

## Estimate

- Wire the config knob: ~30 min.
- Refactor `f_bwd`: ~30 min.
- New test file: ~30 min.
- Run regression + benchmark: ~1 hour wall-clock (mostly waiting).
- Iterate on any failures: variable.

Plan: half-day end-to-end including the bench.

## Definition of done

1. `uv run pytest -m core -x` clean.
2. `uv run pytest tests/test_ipeps_ad_adjoint_methods.py -v` passes.
3. `uv run pytest tests/test_ipeps_excitations.py::TestOptimizeGsAd -v` passes.
4. `python -m benchmarks.varipeps_compare.compare --device cpu` produces
   a Tenax JSON for at least `single_site D=2 chi=16` (no timeout).
5. `published_results/` updated with new data + a STATUS.md note.
6. PR opened, auto-merge enabled with `--squash --delete-branch --auto`.
