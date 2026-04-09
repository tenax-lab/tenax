# iPEPS AD Optimization Paths

This document summarizes the successful automatic differentiation (AD) paths
for iPEPS ground-state optimization in Tenax, based on extensive benchmarking
against the AFM Heisenberg model on the square lattice.

## Recommended Configuration

```python
from tenax import (
    CTMConfig, iPEPSConfig, heisenberg_gate,
    optimize_gs_ad, optimize_gs_ad_chi_schedule,
    sublattice_rotate_gate,
)

H_rot = sublattice_rotate_gate(heisenberg_gate())

config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.05,
    ctm=CTMConfig(
        chi=8,
        max_iter=100,
        conv_tol=1e-8,
        projector_method="eigh",
        forward_gauge="sigma",        # critical for stable CTM
    ),
    gs_explicit_ad=True,
    gs_explicit_ad_steps=30,
    gs_explicit_ad_warmup=10,
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_metric_precond=True,
    gs_c4v=True,
    su_init=True,
)

# Chi-ramping for progressive refinement
chi_schedule = [(8, 30), (16, 20), (32, 15)]
A_opt, env, E = optimize_gs_ad_chi_schedule(H_rot, None, config, chi_schedule)
```

## Benchmark Results

**Model**: AFM Heisenberg, D=2, chi=8, RTX 4070 Ti GPU, float64

| Path | Best E | Time | Notes |
|------|--------|------|-------|
| **eigh + sigma (explicit AD)** | **-0.6670** | 1184s | Best energy |
| eigh + sigma (GMRES implicit) | -0.6601 | — | Fast convergence |
| svd + sigma (explicit AD) | -0.6624 | 1315s | SVD projector works with sigma |
| svd + none (differentiable S) | -0.6265 | 472s | Partial: needs sigma |
| svd + none (stop_gradient S) | -0.5266 | 483s | Fails |
| eigh + none / eigh + qr | ~-0.537 | ~580s | Fails |
| Literature (chi=8) | -0.6625 | — | — |
| Exact (QMC, chi→∞) | -0.6694 | — | — |

## Two Working AD Paths

### Path 1: Explicit AD (Recommended)

**Architecture**: Warmup CTM sweeps (no gradient) → N backprop sweeps with
sigma gauge and `jax.checkpoint` → gradients accumulate through unrolled graph.

```
Forward:  A → warmup sweeps (stop_gradient) → N CTM sweeps (sigma gauge, checkpointed) → energy
Backward: dE/dA via backprop through all N sweeps
```

**Strengths**: Best energy (-0.6670), robust sigma gauge integration.

**Configuration**:
- `forward_gauge="sigma"` — stabilizes CTM iteration and backprop
- `gs_explicit_ad=True` — backprop through unrolled steps
- `gs_explicit_ad_steps=30` — number of backprop CTM sweeps
- `gs_explicit_ad_warmup=10` — warmup sweeps (stop_gradient)

**Why sigma gauge works in backprop**: Each backprop sweep uses
`jax.lax.stop_gradient` to create an independent copy of the environment
before the sweep. Sigma gauge then aligns the post-sweep output to this
frozen reference, providing a meaningful alignment signal (unlike the
convergence loop where dict mutation made it a no-op before the shallow-copy
fix).

### Path 2: Implicit AD via GMRES

**Architecture**: CTM converges to fixed point → GMRES solves the implicit
differentiation equation at the fixed point → gradients flow back to tensor.

```
Forward:  A → CTM sweeps (sigma gauge) → converged env → energy
Backward: dE/dA via GMRES solve at fixed point (no unrolling)
```

**Strengths**: Fast convergence (16 steps), memory-efficient.

**Configuration**:
- `forward_gauge="sigma"` — stabilizes CTM iteration
- `ad_backward_method="gmres"` — solves linear system at fixed point
- `gs_explicit_ad=False` — uses implicit differentiation

This is the YASTN approach (arxiv:2311.11894), adapted for JAX.

## Critical Components

### 1. Sigma Gauge Fixing

Without sigma gauge, the CTM iteration is **fundamentally unstable** for AD —
energy oscillates or diverges regardless of projector type (eigh, svd, or qr).
Sigma gauge aligns each iteration's environment to the previous one using
transfer matrix eigenvectors, making convergence monotonic and the backward
pass well-conditioned.

**Implementation**: Power iteration (30 iterations) computes the leading
eigenvector of the double-layer transfer matrix. This is fully
JAX-differentiable, unlike `jnp.linalg.eig`.

**Sweep mutation fix**: The multisite CTM sweep mutates the environment dict
in-place. A shallow copy (`envs = dict(envs)`) at the start of each sweep
ensures callers that saved a reference to the input dict (for sigma gauge
comparison) still see the pre-sweep environments.

### 2. Projector Methods

**eigh** (recommended): Forms density matrix ρ = C1g·C1g† + C4g·C4g† and
eigendecomposes. Best energy with sigma gauge. Block-sparse path available
for SymmetricTensor.

**svd** (Fishman): Cross-product M = C1g†·C4g, SVD, projector P = C4g·V·S^{-1/2}.
Works with sigma gauge (E=-0.6624). The S^{-1/2} weighting is differentiable
(no stop_gradient), allowing gradient flow through singular values.

**qr**: QR-factored small eigenproblem. Fails for AD without sigma gauge.

### 3. SVD Projector: Single vs Two-Projector

The standard Fishman formulation uses a single projector P = C4g·V·S^{-1/2}
applied as P† on both sides. This means only V (not U) from the SVD gets
gradients.

The full two-projector formulation (P_up = C1g·U·S^{-1/2}, P_down = C4g·V·S^{-1/2})
gives gradients through all SVD factors. However, individual projectors are
NOT isometric (P†P ≠ I) — only the cross-product P_up†·P_down = I holds.
Implementing this requires restructuring CTM absorption to always use
projector pairs, as in arXiv:2502.10298 (Naumann et al.).

**Status**: Single-projector SVD with differentiable S works. Two-projector
requires CTM move refactoring (planned).

### 4. L-BFGS + Hager-Zhang Line Search

Second-order optimization with approximate Wolfe conditions. Metric
preconditioning (Rader et al., arxiv:2511.09546) uses the environment metric
tensor as a natural gradient preconditioner.

### 5. C4v Symmetry Enforcement

For the square lattice with C4v-symmetric Hamiltonians, enforcing C4v symmetry
on the site tensor reduces the parameter space from D²d to ~D²d/8, improving
optimization stability and speed.

## Known Limitations

1. **while_loop (JIT CTM)**: Sigma gauge + while_loop reaches E=-0.6502 vs
   Python loop -0.6670. Under investigation.

2. **Large chi**: Not yet benchmarked at chi > 16. May need conv_tol tuning.

3. **Non-C4v models**: The 2-site optimizer supports explicit AD but has not
   been benchmarked with sigma gauge.

4. **SymmetricTensor**: Block-sparse tensors fall back to the Python loop
   (not JIT-traceable). Sigma gauge works but with dense fallbacks.

5. **Sigma gauge cost**: ~40% overhead from power iteration per sweep.
   Reducing frequency (every N sweeps) or cheaper alignment methods are
   potential optimizations.

## References

- Francuz, Schuch, Vanhecke, *PRR* **7**, 013237 (2025) — Stable AD through CTM
- Rader et al., arXiv:2511.09546 (2025) — Metric preconditioning
- Zhang, Yang, Corboz, arXiv:2505.00494 (2025) — Chi-ramping schedule
- Naumann et al., arXiv:2502.10298 (2025) — Split CTMRG with two-projector formulation
- Fishman et al., *PRB* **98**, 235148 (2018) — SVD (Fishman) projectors
