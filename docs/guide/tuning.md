# Tuning guide

This page is a cross-cutting reference for the most-tuned parameters in
Tenax, grouped by *what you are trying to do* rather than by which
dataclass a field lives on.

A **machine-readable registry** of these parameters lives at
`src/tenax/tuning/registry.py`. Future autotuning / hyperparameter-search
tooling should consume `tenax.tuning.all_params()` rather than re-parsing
this page.

```python
from tenax.tuning import all_params, params_by_category, TuningCategory

for p in params_by_category(TuningCategory.ACCURACY):
    print(p.fully_qualified_name(), p.default, p.hint.range)
```

> **Authoritative defaults** live in the dataclasses
> (`tenax.algorithms.ipeps_config.CTMConfig`,
> `tenax.algorithms.ipeps_config.iPEPSConfig`, etc.). This page and the
> registry mirror them — if you see a mismatch, the dataclass wins.

---

## Quick-reference: what to tune first

| Goal | First knob | Then | Last resort |
|---|---|---|---|
| Energy not converging | `CTMConfig.chi` ↑ | `iPEPSConfig.gs_num_steps` ↑ | `gs_optimizer` → `lbfgs` |
| AD forward is slow | `CTMConfig.jit_ctm=True` | loosen `conv_tol` | reduce `chi` |
| AD backward `Krylov failed to converge` | `adjoint_tikhonov` → `1e-4` | `adjoint_maxiter` ↑ | loosen `adjoint_tol` |
| L-BFGS stalls at a plateau | `gs_stall_recovery="reset"` | `su_init=False` | `gs_metric_precond=False` |
| NaN during optimization | `ad_regularize_svd=True` (default) | `forward_gauge="phase"` | check for degenerate singular values |
| Benchmark for paper | `gs_num_steps ≥ 2000` | `adjoint_tikhonov=0` | tighten `conv_tol` to `1e-10` |

---

## Accuracy knobs

The two knobs that dominate everything else are `CTMConfig.chi` and
`iPEPSConfig.gs_num_steps`. Tune them first; everything else is
secondary.

### `CTMConfig.chi` — CTM environment bond dimension

- **Default:** `20`
- **Scale:** log₂ (typical values 4, 8, 16, 32, 64)
- **Cost:** `O(chi⁶)` runtime per CTM sweep, `O(chi⁴)` memory
- **When to raise:** the ground-state energy still changes as you
  increase `chi`. For D≤3 square iPEPS, `chi ∈ [16, 32]` is usually
  enough. Kagome or D>3 typically need `chi ≥ 64`.
- **When *not* to raise:** doubling `chi` is roughly 64× slower. Pin
  `chi` and tune optimizer knobs first.

### `CTMConfig.max_iter` + `CTMConfig.conv_tol`

Convergence budget for a single CTM sweep-loop.

- **Defaults:** `max_iter=100`, `conv_tol=1e-8`
- Raise `max_iter` to 200–500 when `conv_tol` is tight (1e-10) near
  criticality.
- **Schedule trick:** use `iPEPSConfig.gs_ctm_conv_tol_schedule` to
  ramp `conv_tol` from loose → tight during AD optimization:
  ```python
  iPEPSConfig(
      gs_ctm_conv_tol_schedule=[(0.0, 1e-5), (0.5, 1e-7), (0.8, 1e-9)],
      ...,
  )
  ```
  Saves 10–30% runtime on long runs with no accuracy loss.

---

## Stability knobs (AD paths)

The iPEPS AD paths are sensitive to gauge choices and to near-singular
linear systems in the backward pass. These knobs don't affect forward
accuracy but *do* determine whether the backward pass produces finite,
well-conditioned gradients.

See `docs/guide/algorithms/ipeps_ad_paths.md` for the full matrix of
which AD path pairs with which gauge.

### `CTMConfig.forward_gauge`

Values: `"qr"` (default), `"phase"`, `"sigma"`, `"none"`

- `"qr"`: forward-only CTM, notebooks, diagnostics. `optimize_gs_ad`
  auto-promotes this to `"phase"` when `gs_explicit_ad=True`.
- `"phase"`: explicit-AD default after promotion. Numerically stable
  for unrolled backprop.
- `"sigma"`: required for the implicit-diff path at D ≥ 2. Aligns each
  sweep's output to the previous sweep's input, stabilizing the VJP.
- `"none"`: diagnostic only. Expect instabilities.

### `CTMConfig.ad_regularize_svd` (default `True`)

Use the Lorentzian-regularized SVD backward pass. Prevents NaN from
degenerate / near-degenerate singular values in the projector
computation. **Do not turn off** unless you're deliberately debugging
a numerical issue.

Reference: Francuz et al., arXiv:2311.11894.

### `CTMConfig.adjoint_tikhonov` (default `1e-6`)

Tikhonov damping on the linear adjoint system

```
((I - J^T) + τ·I) λ = g
```

solved during implicit-diff CTM. Near a fixed point, `J` has
eigenvalues approaching 1 along the slowest modes, so `(I - J^T)` is
near-singular and the Krylov solver stalls. Adding `τ·I` bounds the
condition number at the cost of a small O(τ) gradient bias.

| τ | Use case |
|---|---|
| `0.0` | strictly exact gradient; use for publication runs *if* the solver converges |
| `1e-6` (default) | robustness floor — smaller than `adjoint_tol`, so it can't bias beyond already-accepted tolerance |
| `1e-4` | first fallback when you hit `"Krylov adjoint failed to converge"` |
| `1e-3` | benchmark / smoke-run setting; noticeable bias (~0.1%) but reliably stable |
| `≥ 1e-2` | too aggressive — gradient direction is wrong |

Applies only when `CTMConfig.ctm_ad_mode="c4v_reference"`.

References: Liao et al. (arXiv:1903.09650), Francuz et al.
(arXiv:2311.11894), variPEPS (arXiv:2308.12358).

### `CTMConfig.adjoint_tol`, `adjoint_maxiter`, `adjoint_solver`

Krylov solver knobs for the implicit-diff adjoint. The effective
residual target is `max(10 × adjoint_tol, 1e-12)`. The solver order is
BiCGStab → GMRES (automatic fallback).

- Loosen `adjoint_tol` to `1e-5` when the outer optimizer is near the
  ground state; tighten to `1e-10` for publication-quality gradients.
- `adjoint_maxiter=50` is tight. Raise to `200+` for tight `adjoint_tol`
  or ill-conditioned systems.

---

## Optimizer knobs

### `iPEPSConfig.gs_optimizer`

Values: `"cg"` (default), `"lbfgs"`, `"adam"`

- **CG** — preconditioned conjugate gradient. Best with
  `gs_metric_precond=True`. Fast, stable, recommended default.
- **L-BFGS** — often reaches slightly lower energies than CG but can
  stall in bad basins. Pair with `gs_stall_recovery="reset"` for
  robustness.
- **Adam** — robust to noisy gradients (useful for large-chi with
  loose CTM `conv_tol`) but slower to converge. Needs `gs_learning_rate`
  in `1e-3..1e-2`.

### `iPEPSConfig.gs_num_steps`

| Purpose | Typical |
|---|---|
| Smoke test / CI | 10–50 |
| Benchmark | 100–500 |
| Publication (variPEPS-style) | 2000+ |

### `iPEPSConfig.gs_metric_precond` (default `True`)

Metric preconditioning (natural gradient) via the local tangent-space
metric `N_ij = <∂ᵢψ | ∂ⱼψ>`. Applied through GMRES. Roughly 1.5–3×
slower per step but dramatically faster outer-loop convergence.

Reference: Rader et al., arXiv:2511.09546.

### `iPEPSConfig.gs_line_search_method`

Values: `"armijo"` (default), `"hager_zhang"`

Armijo is cheap and usually enough. Hager-Zhang is stronger and matches
variPEPS — try it when Armijo is rejecting too many steps or when you
want publication-quality step selection.

Reference: Hager & Zhang, SIAM J. Optim. 16(1):170–192 (2005).

### `iPEPSConfig.gs_stall_recovery`

Values: `None` (auto), `"noise"`, `"reset"`

- **`"noise"`** — inject a `gs_noise_amplitude` Frobenius perturbation
  when the line search stalls. Required for 1-site C4v to break out of
  SU-init plateaus.
- **`"reset"`** — clear L-BFGS `(s, y)` history, roll back to
  `best_params`, force steepest descent on the next step. This is what
  variPEPS does; best for 2-site runs.
- **`None`** (default) — auto-select: `"noise"` for 1-site, `"reset"`
  for 2-site.

---

## Initialization

### `iPEPSConfig.su_init` (default `True`)

Initialize via simple update before AD. Usually a good idea — SU gives
you a state already close to the ground state manifold.

**Exception:** for the sublattice-rotated Heisenberg Hamiltonian, SU
converges to the classical `|↑↑⟩` product state, which is a
gradient-zero extremum at `E = -0.5/site`. L-BFGS cannot escape it. Use
`su_init=False` (random init) for this model, at the cost of a less
informative start.

---

## Method selection

### `iPEPSConfig.gs_explicit_ad` (default `True`)

- **`True`** (explicit) — differentiate through unrolled CTM sweeps via
  `jax.checkpoint`. Recommended path post PR #291. Memory scales with
  `gs_explicit_ad_steps`.
- **`False`** (implicit) — solve the fixed-point adjoint equation.
  Required by the `ctm_ad_mode="c4v_reference"` path (Francuz et al.
  App. C–F). Lower memory but needs `adjoint_tikhonov > 0` near the GS.

### `iPEPSConfig.gs_c4v` (default `False`)

Enforce C4v symmetry on the site tensor during AD by parameterizing in
a C4v basis. For D=2, reduces the parameter count from 16 → ~4 and is
**the only stable 1-site iPEPS AD path in Tenax today** for
C4v-symmetric models (square Heisenberg after sublattice rotation,
XXZ, TFIM).

### `CTMConfig.projector_method`

Values: `"eigh"` (default), `"qr"`, `"svd"` (Fishman)

- `"eigh"`: default for forward CTM.
- `"qr"`: cheaper at large `chi` (> 32); use for forward-only runs.
- `"svd"`: Fishman projector; the only fully AD-stable choice post
  PR #291. If you're differentiating through CTM and not already
  using it, switch.

Reference: Fishman et al., arXiv:1711.01288.

---

## Performance-only knobs

These don't change numerics (within tolerance) but speed things up.

### `CTMConfig.jit_ctm` (default `False`)

Fuse the entire CTM convergence loop into a single `jax.lax.while_loop`
kernel. 5–30× speedup on GPU for `chi ≥ 16`. Falls back to the Python
loop for `SymmetricTensor` inputs (block-sparse ops aren't
JIT-traceable).

### `CTMConfig.gmres_precondition` (experimental, default `False`)

Diagonal scaling preconditioner for the GMRES implicit-diff backward.
Currently experimental — do not turn on for production.

---

## See also

- `docs/guide/algorithms/ipeps_ad_paths.md` — AD mode matrix
- `docs/ipeps-code-paths.md` — which function goes with which config
- `src/tenax/tuning/registry.py` — machine-readable parameter metadata
- `src/tenax/algorithms/ipeps_config.py` — authoritative dataclass
  defaults (always the source of truth)
