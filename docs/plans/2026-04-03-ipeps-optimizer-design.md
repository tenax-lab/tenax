# iPEPS AD Optimizer: CG/L-BFGS with Python Backtracking Line Search

## Problem

The CTM fixed-point iteration uses Python-level convergence checks (`float()`),
making it incompatible with JAX's traced `lax.while_loop` required by
`optax.lbfgs`'s built-in Wolfe line search. Without line search, L-BFGS is
ineffective and Adam with fixed learning rate oscillates on the iPEPS energy
landscape. The AD iPEPS community (peps-torch, Corboz et al.) uses L-BFGS with
line search as the standard optimizer.

## Design

### Armijo backtracking line search (Python-level)

```
Given: x_k, d_k (search direction), g_k (gradient), f_k = loss(x_k)
alpha = 1.0
c1 = 1e-4  (sufficient decrease)
rho = 0.5  (backtracking factor)

for i in range(max_backtracks):  # Python loop, not traced
    x_trial = x_k + alpha * d_k
    f_trial = loss_fwd(x_trial)  # forward CTM only, no backward
    if f_trial <= f_k + c1 * alpha * dot(g_k, d_k):
        return x_trial, alpha, f_trial
    alpha *= rho
```

Forward-only CTM passes are cheap (no GMRES backward), so 3-5 backtracks
cost roughly 1 full AD step.

### Search directions

1. **L-BFGS** via `optax.scale_by_lbfgs` — quasi-Newton direction
2. **Polak-Ribiere CG** — implemented directly, no optax needed
3. **Steepest descent** — fallback / baseline

### Config

```python
gs_optimizer: str = "adam"  # "adam", "lbfgs", or "cg"
gs_line_search: bool = False  # backtracking line search
gs_line_search_max_steps: int = 8
```

Line search is ignored for Adam. Defaults to True for lbfgs/cg.

### Implementation scope

- `_backtracking_line_search()` in ipeps_optimize.py
- `_cg_direction()` for Polak-Ribiere beta
- Forward-only `loss_fn_fwd` (no grad, no GMRES)
- Modify 1-site and 2-site loops
- Test: `test_2site_ad_cg_energy_decreases`
