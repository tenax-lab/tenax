# Python-Level CTM AD Loop Design

**Date:** 2026-04-17
**Goal:** Match variPEPS compilation speed by moving CTM convergence and VJP iteration to Python-level loops, JIT-compiling only single-step kernels.

## Problem

Tenax currently traces the entire CTM convergence loop as one JAX computation graph:
- **Explicit AD:** Unrolls N CTM steps inside the trace → graph size O(N × chi²)
- **Implicit AD:** Uses `lax.while_loop` for forward + Neumann/GMRES for backward → still one giant trace

This causes JIT compilation times of 20-90+ minutes for modest problems (D=2, chi=8), while variPEPS compiles in seconds for the same problem.

## variPEPS Architecture

variPEPS JIT-compiles only **atomic operations**, running convergence loops in Python:

```
Forward (Python loop):
  while not converged:
    env = jit_ctm_step(env, A)         # JIT: single CTM absorption + projection
    converged = check_convergence(env)  # Python

Backward (Python loop, via custom_vjp):
  while not converged:
    x = jit_vjp_step(x, J_T)           # JIT: single Neumann/GMRES iteration
    converged = check_gradient_conv(x)  # Python

Line search (Python loop):
  while not sufficient_decrease:
    E = jit_energy(A + alpha*d, env)    # JIT: CTM forward + energy eval
    alpha = update_alpha(E)             # Python
```

Each JIT-compiled kernel is small (one CTM step ≈ 4 projector SVDs + 4 absorptions). Compilation takes O(seconds). The Python loops add negligible overhead since each step does O(chi³ D⁴) FLOPs.

## Proposed Tenax Architecture

### Phase 1: `custom_vjp` CTM wrapper (minimal change)

Wrap the existing CTM convergence in a `jax.custom_vjp`:

```python
@jax.custom_vjp
def ctm_fixed_point(site_tensors, env_init, config):
    # Forward: Python-level convergence loop
    env = env_init
    for _ in range(config.max_iter):
        env = _jit_ctm_step(env, site_tensors, config)
        if _converged(env, prev_env):
            break
    return env

def ctm_fixed_point_fwd(site_tensors, env_init, config):
    env = ctm_fixed_point(site_tensors, env_init, config)
    return env, (site_tensors, env)  # residuals for backward

def ctm_fixed_point_bwd(res, g):
    site_tensors, env = res
    # Solve (I - J^T) x = g via Python-level GMRES/Neumann
    x = g
    for _ in range(max_iter):
        x = _jit_vjp_step(x, site_tensors, env)
        if converged:
            break
    return (x, None, None)  # gradient w.r.t. site_tensors only

ctm_fixed_point.defvjp(ctm_fixed_point_fwd, ctm_fixed_point_bwd)
```

**Key:** `_jit_ctm_step` and `_jit_vjp_step` are small JIT-compiled functions. The loops are Python.

### Phase 2: JIT-compiled single CTM step

Extract from `_ctm_tensor_sweep_single` a function that does one full CTM sweep (4 directions) and returns the updated environment:

```python
@jax.jit
def _jit_ctm_step(env_leaves, site_tensor_data, config_tuple):
    # One full CTM sweep: absorb + project in all 4 directions
    ...
    return new_env_leaves
```

This is the only part that needs JIT. The convergence check, chi ramping, and logging stay in Python.

### Phase 3: Python-level optimizer loop

The optimizer (`_optimize_gs_ad_tensor`) becomes:

```python
# JIT-compile once: energy + gradient at a given (params, env)
@jax.jit
def _energy_and_grad(params, env_leaves):
    A = _params_to_A(params)
    env = ctm_fixed_point(A, env_leaves, config)  # custom_vjp
    E = compute_energy(A, env)
    return E

_vag = jax.value_and_grad(_energy_and_grad)

# Python optimization loop
for step in range(num_steps):
    E, grad = _vag(params, env_leaves)    # JIT: one call
    direction = lbfgs_direction(grad)      # Python
    alpha = line_search(params, direction) # Python loop calling _vag
    params = params + alpha * direction    # Python
```

## Files Changed

| File | Change |
|------|--------|
| `src/tenax/algorithms/ad_utils.py` | Add `custom_vjp` CTM wrapper |
| `src/tenax/algorithms/_ctm_tensor_convergence.py` | Extract `_jit_ctm_step` |
| `src/tenax/algorithms/ipeps_optimize.py` | Use new CTM wrapper in loss_fn |

## Benefits

- **Compilation:** Seconds instead of minutes/hours
- **Flexibility:** Chi ramping, convergence monitoring, and logging in Python
- **Debugging:** Can inspect intermediate CTM states without retracing
- **2-tensor CG:** Tuple-param pytrees work naturally since the trace is small
- **All iPEPS paths benefit:** 1-site, 2-site, CG, fermionic

## Risks

- Breaking existing implicit/explicit AD paths that depend on current trace structure
- The `custom_vjp` residuals include the full environment, increasing memory
- Need careful testing against known-good energies at each step

## Reference

- variPEPS: `varipeps/ctmrg/routine.py` (forward loop), `varipeps/ctmrg/ctmrg_custom_rule.py` (custom_vjp)
- Tenax issue #328: spectral radius > 1 for non-C4v — GMRES backward needed regardless of architecture
