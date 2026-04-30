# Complex128 Default + Arnoldi Spectral-Radius Precheck

**Date:** 2026-04-21
**Branch:** `feat/python-level-ctm-ad`
**Goal:** Match variPEPS E=-0.6625 on non-C4v 2-site Heisenberg (D=2)

## Motivation

Real float64 tensors cause non-variational drift in 2-site non-C4v iPEPS AD
(E: -0.089 -> -1.498). Complex128 tensors stay variational (E: -0.086 -> -0.103)
because they double the parameter space and break symmetry of degenerate gauge
modes that make the implicit-diff linear system ill-conditioned.

variPEPS uses complex128 as default dtype + Arnoldi precheck before GMRES backward
to detect and recover from ill-conditioned environments.

## Scope

1. **Complex128 as default dtype** for both 1-site C4v and 2-site non-C4v AD paths
2. **Arnoldi spectral-radius precheck** (20 iterations) before GMRES backward
3. **Optimizer recovery** on precheck failure: noise kick + L-BFGS reset

## Design

### 1. Complex128 Initialization

**Random init:** `_random_ipeps_tensor()` generates complex128 (real + imag parts
independently sampled, then normalized).

**Simple-update warm start:** Cast SU tensors to complex128 before entering the
optimizer loop.

**C4v coefficient vector:** Made complex128 so the shared C4v tensor is complex.

### 2. Gradient Handling

JAX `value_and_grad` on a real-valued loss with complex parameters returns Wirtinger
derivatives (dL/dz_bar), which is the correct descent direction. No special handling
needed:

- `jnp.vdot` (already used) gives Hermitian inner products
- L-BFGS two-loop recursion with `jnp.vdot(s, y)` is correct
- Tangent projection `_tangent_project_unit` uses `jnp.vdot` -- correct
- `ravel()`/`reshape()` on complex arrays just works

### 3. CTM Gauge Fixing

**Phase fix** (`_phase_fix_normalize_raw`): Already uses `jnp.conj(phase)` -- correct.

**Sigma gauge** (`_sigma_gauge_fix_env`): Transfer matrix rho = T^dag T is Hermitian
even for complex T, so `eigh` still applies.

### 4. Arnoldi Spectral-Radius Precheck

New function in `_ctm_energy_ad.py`:

```python
def _arnoldi_spectral_radius(apply_Jt, v0, n_iter=20):
    """Estimate spectral radius of J^T via Arnoldi iteration.

    Builds upper Hessenberg matrix H via n_iter Arnoldi steps,
    returns max|eig(H)|.
    """
```

Integration in `f_bwd()`, before GMRES:
1. Run Arnoldi with n_iter=20 on `_jit_apply_Jt`
2. If rho >= 1.0: raise `CTMRGGradientNotConvergedError`

### 5. Optimizer Recovery

In `_optimize_gs_ad_tensor_2site()` and 1-site equivalent:

```python
try:
    energy_val, grads = jax.value_and_grad(loss_fn)(params)
except CTMRGGradientNotConvergedError:
    # 1. Add small random complex perturbation to params
    # 2. Reset L-BFGS history
    # 3. Log warning
    continue
```

This matches variPEPS's recovery strategy.

### 6. Tests

- **Unit:** Arnoldi spectral radius on known matrices (rho < 1 passes, rho > 1 raises)
- **Gradient:** Complex128 FD-vs-AD gradient check for both 1-site and 2-site
- **Integration:** 2-site non-C4v Heisenberg (D=2, chi=8, ~20 steps), assert E < -0.5
- **C4v:** 1-site C4v with complex128, verify convergence matches real case

## Files to Modify

| File | Change |
|------|--------|
| `ipeps_optimize.py` | Complex128 init, recovery logic, dtype handling |
| `_ctm_energy_ad.py` | Arnoldi precheck, CTMRGGradientNotConvergedError |
| `_ctm_tensor_energy.py` | Verify complex RDM handling (already Hermitianized) |
| `_metric_precond.py` | Verify complex L-BFGS two-loop (already uses vdot) |
| `tests/test_arnoldi_precheck.py` | New: Arnoldi unit tests |
| `tests/test_complex128_ad.py` | New: complex gradient + integration tests |
