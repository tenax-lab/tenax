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
        chi=16,
        max_iter=80,
        conv_tol=1e-8,
        projector_method="qr",        # fastest, best energy, scales to chi=64+
        # forward_gauge auto-set to "phase" for explicit AD
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

# Chi-ramping for progressive refinement (recommended for chi > 16)
chi_schedule = [(8, 30), (16, 30), (32, 30)]
A_opt, env, E = optimize_gs_ad_chi_schedule(H_rot, None, config, chi_schedule)
```

## Benchmark Results

**Model**: AFM Heisenberg, D=2, C4v, explicit AD, 30 L-BFGS steps, RTX 4070 Ti GPU, float64

### Phase gauge (recommended) + projector comparison at D=2

| Projector | chi=8 | chi=16 | chi=32 | chi=48 | chi=64 |
|-----------|-------|--------|--------|--------|--------|
| **qr+phase** | -0.6610 (170s) | **-0.6628 (172s)** | -0.6602 (207s) | -0.6622 (259s) | -0.6541 (424s) |
| eigh+phase | -0.6602 (176s) | -0.6602 (836s) | -0.6599 (452s) | — | — |
| svd+phase | -0.6602 (195s) | -0.6602 (805s) | -0.6599 (1109s) | — | — |

**qr+phase** is the new recommended path: best energy at chi=16 (-0.6628),
scales well to chi=64 (2.5x slower than chi=8), and never NaNs.

### Sigma gauge (historical, slower)

| Path | Best E | Time | Notes |
|------|--------|------|-------|
| eigh + sigma (explicit AD) | -0.6601 | 1234s | Slower than phase |
| svd + sigma (two-proj) | -0.6623 | 1124s | Slower than phase |
| eigh + sigma (GMRES implicit) | -0.6601 | — | Implicit AD path |
| Literature (chi=8) | -0.6625 | — | — |
| Exact (QMC, chi→∞) | -0.6694 | — | — |

Phase gauge is **6-9x faster** than sigma gauge for explicit AD with equal
or better energy. Sigma gauge is still needed for implicit AD (GMRES backward).

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

### 3. SVD Projector: Two-Projector Fishman

The two-projector Fishman formulation (arXiv:2502.10298) computes a
bi-orthogonal pair:

- P_1 = C4g·V·S^{-1/2} (applied to C1g side)
- P_2 = C1g·U·S^{-1/2} (applied to C4g side)

satisfying P_1†·P_2 = I. Both corners get clean projections (S^{1/2}·U†
and S^{1/2}·V†), and gradients flow through all SVD factors (U, S, V).

For eigh/qr projectors, P_1 = P_2 = P (standard isometric projector).

This matches the variPEPS and YASTN implementations. The remaining
differences are in numerical conditioning: variPEPS uses 2×2 enlarged
corners with Fishman low-rank pre-truncation; YASTN uses QR pre-factoring
of half-corners; Tenax uses neither (QR backward is unstable for
rank-deficient matrices during AD).

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
