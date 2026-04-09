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
        ad_backward_method="gmres",   # recommended backward method
    ),
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

| Path | Best E | Steps to converge | Stable? |
|------|--------|-------------------|---------|
| **Sigma + GMRES (implicit AD)** | **-0.6601** | 16 | Yes |
| **Sigma + explicit AD** | **-0.6601** | 30 | Yes |
| Sigma + GMRES + while_loop | -0.6502 | 30 | Yes |
| QR gauge + any AD | diverges | — | No |
| Literature (chi=8) | -0.6625 | — | — |
| Exact (QMC, chi→∞) | -0.6694 | — | — |

## Two Working AD Paths

### Path 1: Implicit AD via GMRES (Recommended)

**Architecture**: CTM converges to fixed point → GMRES solves the implicit
differentiation equation at the fixed point → gradients flow back to tensor.

```
Forward:  A → CTM sweeps (sigma gauge) → converged env → energy
Backward: dE/dA via GMRES solve at fixed point (no unrolling)
```

**Strengths**: Fast convergence (16 steps), accurate gradients, memory-efficient.

**Configuration**:
- `forward_gauge="sigma"` — stabilizes CTM iteration
- `ad_backward_method="gmres"` — solves linear system at fixed point
- `gs_explicit_ad=False` — uses implicit differentiation

This is the YASTN approach (arxiv:2311.11894), adapted for JAX.

### Path 2: Explicit AD (Backprop through Unrolled CTM)

**Architecture**: Run N CTM sweeps → backpropagate through all sweeps →
gradients accumulate through the unrolled computation graph.

```
Forward:  A → N CTM sweeps (sigma gauge, checkpointed) → energy
Backward: dE/dA via backprop through all N sweeps
```

**Strengths**: No fixed-point assumption needed, conceptually simple.

**Configuration**:
- `forward_gauge="sigma"` — stabilizes forward CTM
- `gs_explicit_ad=True` — backprop through unrolled steps
- `gs_explicit_ad_steps=30` — number of backprop CTM sweeps
- `gs_explicit_ad_warmup=10` — warmup sweeps (stop_gradient)

This is the variPEPS approach (Naumann et al.), adapted for JAX.

## Critical Components

### 1. Sigma Gauge Fixing

Without sigma gauge, the eigh projector CTM iteration is **fundamentally
chaotic** — energy oscillates ±0.1 with spikes to -4.0. Sigma gauge aligns
each iteration's environment to the previous one using transfer matrix
eigenvectors, making convergence monotonic.

**Implementation**: Power iteration (30 iterations) computes the leading
eigenvector of the double-layer transfer matrix. This is fully
JAX-differentiable, unlike `jnp.linalg.eig`.

### 2. eigh Projector (Not SVD or QR)

The SVD (Fishman) projector converges to a **trivial fixed point** (identity
corners) with sign-fixed QR gauge. The eigh projector reaches the correct
physical fixed point but requires sigma gauge for stability.

### 3. GMRES Backward

The iterative VJP backward diverges because the Jacobian spectral radius
ρ(J^T) ≈ 4400 without sigma gauge. GMRES solves the implicit differentiation
linear system directly, avoiding this instability. With sigma gauge, VJP may
also work (YASTN achieves ρ(J^T) < 1 via sigma gauge), but GMRES is more
robust.

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
   Python loop -0.6601. The while_loop convergence criterion may not detect
   convergence reliably. Under investigation.

2. **Large chi**: Not yet benchmarked at chi > 16. May need conv_tol tuning.

3. **Non-C4v models**: The 2-site optimizer supports explicit AD but has not
   been benchmarked with sigma gauge.

4. **SymmetricTensor**: Block-sparse tensors fall back to the Python loop
   (not JIT-traceable). Sigma gauge works but with dense fallbacks.

## References

- Francuz, Schuch, Vanhecke, *PRR* **7**, 013237 (2025) — Stable AD through CTM
- Rader et al., arXiv:2511.09546 (2025) — Metric preconditioning
- Zhang, Yang, Corboz, arXiv:2505.00494 (2025) — Chi-ramping schedule
- Naumann et al., arXiv:2502.10298 (2025) — variPEPS explicit AD
