# Python-Level CTM AD with JIT-Fused GMRES Backward

**Date:** 2026-04-18
**Branch:** TBD
**Status:** Design approved

## Problem

The current iPEPS AD architecture JIT-traces the entire CTM convergence loop:
- **Explicit:** Unrolls N CTM sweeps into a single XLA graph
- **Implicit:** `lax.while_loop` + Neumann/GMRES inside one JIT scope

This causes 20–90+ min compilation at D=2 χ=8, blocks CG iPEPS (multi-tensor
parameterization explodes the trace graph), and prevents practical use of
Hager-Zhang line search (each energy eval recompiles). For multi-site unit
cells (2×2, 3×3), the backward also fails: J^T spectral radius >> 1 makes
Neumann VJP diverge, and the current capped GMRES doesn't converge.

## Design

### Architecture: Hybrid Forward/Backward

```
optimize_gs_ad()
  └─ for step in range(num_steps):              # Python
       ├─ energy, grads = value_and_grad(f)(params, env_init)
       │    └─ f = custom_vjp wrapper:
       │         ├─ FWD (Python loop, no JIT on outer loop):
       │         │    ├─ map_fn(*raw_params) → A_cg      # CG contraction (if CG)
       │         │    ├─ for i in range(max_iter):        # Python
       │         │    │    env = jit_ctm_step(A, env)     # JIT'd single sweep
       │         │    │    if converged: break
       │         │    └─ energy = jit_compute_energy(A, env, gate)
       │         │
       │         └─ BWD (fully JIT'd):
       │              ├─ g = d(energy)/d(env) at fixed point
       │              ├─ lam = GMRES_solve(               # lax.while_loop
       │              │     matvec = v → v - J^T @ v,     # VJP through one CTM sweep
       │              │     rhs = g, tol = gmres_tol)
       │              └─ d(energy)/d(params) via chain rule through lam + map_fn
       │
       ├─ direction = optimizer_step(grads)       # Python
       ├─ alpha = line_search(f, params, dir)     # Python + JIT evals
       └─ params = params + alpha * direction
```

### Key Properties

- **Forward compiles in seconds:** single CTM sweep + energy eval (~2 JIT traces)
- **Backward compiles once:** GMRES `lax.while_loop` + one CTM sweep VJP, reused across steps
- **Chi ramp, adaptive stopping, logging** live in Python — no recompilation
- **Unit-cell-agnostic:** works for 1×1, 2×2, 3×3 — just more tensors in env pytree
- **CG-compatible:** `map_fn` lives inside the `custom_vjp` boundary; gradients flow through automatically

### Two Dispatch Paths

**Implicit** (`gs_implicit_ad=True`, default):
- Forward: Python-loop CTM to convergence
- Backward: JIT-fused GMRES solving `(I - J^T) λ = ∂E/∂env`
- Wrapper: `ctm_energy_implicit` with `@jax.custom_vjp`

**Explicit** (`gs_implicit_ad=False`):
- Forward: Warmup (no grad) + checkpointed explicit CTM sweeps
- Backward: Standard autodiff through checkpointed sweeps
- Wrapper: `ctm_energy_explicit` with `@jax.custom_vjp`

### `custom_vjp` Boundary

```python
@jax.custom_vjp
def ctm_energy_implicit(raw_params, env_init, gate, config):
    """Forward: Python-loop CTM → energy. Backward: JIT-fused GMRES."""
    A = _params_to_A(raw_params, config)  # includes map_fn for CG
    env = env_init
    for i in range(config.max_iter):
        env = _jit_ctm_step(A, env, config)
        if _check_converged(env, prev_env, config.conv_tol):
            break
    return _jit_compute_energy(A, env, gate)
```

- `env_init` gets `None` gradient (no backprop through warm-start)
- `config` is static (not differentiated)
- `raw_params` is a pytree: plain arrays for standard iPEPS, sub-site tensor tuples for CG
- Converged `env` cached in residuals for the backward

### JIT-Fused GMRES Backward

```python
@jax.jit
def _jit_gmres_backward(raw_params, env_converged, gate, g_scalar, config):
    A = _params_to_A(raw_params, config)

    # 1. RHS: d(energy)/d(env)
    d_energy_d_env = jax.grad(
        lambda env: compute_energy(A, env, gate)
    )(env_converged)

    # 2. Matvec: v → v - J_env^T @ v
    def matvec(v):
        _, vjp_fn = jax.vjp(
            lambda env: _ctm_sweep(A, env, config),
            env_converged
        )
        return v - vjp_fn(v)[0]

    # 3. GMRES solve via lax.while_loop (uncapped, restart every 30)
    lam = gmres_lax(matvec, d_energy_d_env, tol=config.gmres_tol,
                    restart=config.gmres_restart)

    # 4. Chain rule: direct + indirect through env
    direct = jax.grad(lambda p: compute_energy(_params_to_A(p, config),
                                                env_converged, gate))(raw_params)
    _, vjp_step = jax.vjp(
        lambda p: _ctm_sweep(_params_to_A(p, config), env_converged, config),
        raw_params
    )
    indirect = vjp_step(lam)[0]

    return jax.tree.map(lambda d, i: g_scalar * (d + i), direct, indirect)
```

### Multi-Site Support

- **`raw_params`:** Pytree of per-site parameters. For 2×2: `(p_00, p_01, p_10, p_11)`.
  For CG: each `p_ij` is itself a tuple of sub-site tensors.
- **`env`:** Per-site environments. N² sets of `(C1..C4, T1..T4)`.
- **Neighbor map:** General `make_neighbors(nx, ny)` factory replacing
  `SINGLE_SITE_NEIGHBORS` and `CHECKERBOARD_NEIGHBORS`.
- **Energy:** Generalize `compute_energy` to sum over all nearest-neighbor bonds
  for arbitrary NxN unit cells.
- **GMRES backward:** Unit-cell-agnostic — pytree operations handle any size.

### Config Changes

| Field | Change |
|-------|--------|
| `gs_implicit_ad` | Keep — dispatches implicit vs explicit |
| `gs_explicit_ad_steps` | Keep — explicit sweep count |
| `ad_backward_method` | Remove — implicit always uses GMRES |
| `gs_line_search_method` | Default → `"hager_zhang"` (was `"armijo"`) |
| `gmres_tol` | Add to CTMConfig (default 1e-6) |
| `gmres_restart` | Add to CTMConfig (default 30) |
| `unit_cell` | Generalize to `(nx, ny)` tuple |

### What Stays Unchanged

- `_ctm_tensor_sweep_multisite()` — the sweep function used by both forward and backward
- Gauge fixing (sigma/phase/QR) — applied inside sweep
- Chi ramp — in the Python-level forward loop
- Projector methods (SVD/eigh) — inside sweep
- Optimizer loop (L-BFGS, CG, line search) — calls new `ctm_energy_*` wrappers
- C4v reference mode — separate path, already works
- Arnoldi precheck — optional diagnostic before GMRES

### Optimizer Entry Points

Unify `_optimize_gs_ad_tensor` (1-site) and `_optimize_gs_ad_2site` (2-site) into
a single function that takes `unit_cell=(nx, ny)` and handles CG via `config.cg_gates`.
The C4v reference path remains separate.

## Success Criteria

1. 1-site Heisenberg reproduces current results (E ≈ -0.6694 at large χ)
2. 2-site non-C4v implicit AD is variational (E > -0.6694)
3. Compilation time < 2 min for 2×2 unit cell at D=2 χ=16
4. CG honeycomb reproduces variPEPS reference (E ≈ -0.5376)
5. HZ line search works without recompilation penalty
