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

## Working AD Paths

After PR #291 the explicit-AD path is the recommended workflow and the
implicit/GMRES path is explicitly labeled experimental until its stability
gap is closed (tracked by issue #292). Both paths share the same forward
CTM stack; they differ only in how gradients are computed.

### Path 1: Explicit AD (Recommended)

**Architecture**: Warmup CTM sweeps (no gradient) → N backprop sweeps with
phase-gauge fixing and `jax.checkpoint` → gradients accumulate through the
unrolled graph.

```
Forward:  A → warmup sweeps (stop_gradient) → N CTM sweeps (phase gauge, checkpointed) → energy
Backward: dE/dA via backprop through all N sweeps
```

**Strengths**: Best reported energy at chi=16 (-0.6628 with qr+phase),
scales cleanly to chi=64+, never NaNs, and is 6–9× faster than sigma
gauge for equal or better energy.

**Configuration**:
- `gs_explicit_ad=True` — backprop through unrolled steps (default).
- `gs_projector_method="qr"` — QR projectors (recommended for explicit AD).
- `forward_gauge="qr"` (config default) — `optimize_gs_ad` auto-promotes to
  `"phase"` at runtime when `gs_explicit_ad=True`. Users can override with
  `forward_gauge="sigma"` (historical path) or `forward_gauge="none"`
  (diagnostic); see the mode table below.
- `gs_explicit_ad_steps=30` — number of backprop CTM sweeps.
- `gs_explicit_ad_warmup=10` — warmup sweeps (stop_gradient).

**Why phase gauge works in backprop**: Each phase-gauge step is a
Frobenius normalization plus a differentiable global-phase fix on each
environment tensor. Applied inside the checkpointed unrolled graph, it
removes the gauge ambiguity that causes element-wise CTM convergence to
drift without introducing the power-iteration cost of sigma gauge. The
Frobenius + phase fix is what variPEPS uses in `_post_process_CTM_tensors`.

**Sigma gauge as a fallback**: ``forward_gauge="sigma"`` is still a
first-class mode. It is slower (~40% per sweep from power iteration) but
remains the right choice when you need the exact transfer-matrix
alignment — most importantly on the implicit-diff path below. For the
explicit-AD path, leave the config at its default and let the optimizer
promote to ``"phase"``.

### Path 2: Implicit AD (Experimental)

**Architecture**: CTM converges to fixed point → backward solves the
implicit-differentiation linear system at the fixed point → gradients flow
back to the tensor without unrolling.

```
Forward:  A → CTM sweeps (sigma gauge) → converged env → energy
Backward: dE/dA via (I - J^T) λ = g  (VJP iteration or GMRES)
```

**Strengths on paper**: Fast convergence, memory-efficient (no unrolled
graph in memory).

**Current status**:
- `ad_backward_method="vjp"` (default) — iterative Neumann-series backward,
  YASTN-style. This is the regression-covered implicit backward.
- `ad_backward_method="gmres"` — direct Krylov solve. **Documented unstable
  without further stabilization**: the `test_gmres_path_agrees_with_vjp`
  regression is currently marked `xfail` in `tests/test_ad_utils.py`. The
  GMRES spectral radius exceeds 1 without careful sigma-gauge alignment and
  a tighter CTM fixed point. Tracked by issue #292.

**Configuration (for the VJP path only)**:
- `forward_gauge="sigma"` — required for stable element-wise convergence.
- `ad_backward_method="vjp"` — the supported implicit backward.
- `gs_explicit_ad=False` — use implicit differentiation.

This is the YASTN approach (arXiv:2311.11894), adapted for JAX. For new
code prefer Path 1 — the explicit-AD path does not exercise the implicit
backward at all.

### Path 3: Paper-Faithful Dense C4v (Appendix C-F, Opt-In)

This opt-in path follows the fixed-point differentiation structure in
YASTN (arXiv:2311.11894, App. C-F) for **dense 1-site C4v** runs:

```
Forward:  A (C4v-projected) -> dense C4v CTM fixed point (single C/T representation)
Backward: Appendix-C eigendifferential + Appendix-F implicit solve
          (I - J^T) lambda = g via bicgstab with gmres fallback
```

**Enable it with:**

```python
config = iPEPSConfig(
    gs_explicit_ad=False,
    gs_c4v=True,
    unit_cell="1x1",
    ctm=CTMConfig(
        chi=16,
        paper_ctm_ad="c4v_appendix_cf",
        paper_krylov_solver="bicgstab",  # or "gmres"
        paper_krylov_maxiter=50,
        paper_krylov_tol=1e-8,
    ),
)
```

**Current scope and constraints:**
- dense tensors only (no SymmetricTensor path yet),
- strict gate: `unit_cell="1x1"`, `gs_c4v=True`, `gs_explicit_ad=False`,
  `ctm.paper_ctm_ad="c4v_appendix_cf"`,
- supports `gs_num_steps>0` optimization with implicit gradients.

## Forward Gauge Mode Matrix

Post-PR-#291 Tenax supports four ``forward_gauge`` modes. Their intended
use is summarized below:

| Mode | Explicit AD (Path 1) | Implicit AD (Path 2, VJP) | Notes |
|------|----------------------|----------------------------|-------|
| ``"qr"`` (static default) | Auto-promoted to ``"phase"`` when ``gs_explicit_ad=True`` | Works for simple cases but not preferred | Conservative default for callers that construct ``CTMConfig`` directly. |
| ``"phase"`` | **Recommended** | Not validated | Cheapest gauge fix; Frobenius + differentiable phase fix. |
| ``"sigma"`` | Historical — still correct but ~6–9× slower than phase | **Required** for stable element-wise convergence | Power iteration (30 steps) per sweep. |
| ``"none"`` | Benchmark / diagnostic only | Unstable | Isolates gauge-fix cost from projector cost. |

The auto-promotion rule lives in ``optimize_gs_ad``: when
``gs_explicit_ad=True`` and ``ctm.forward_gauge == "qr"`` (the static default),
the optimizer replaces the forward gauge with ``"phase"`` for the run. If the
user explicitly sets ``forward_gauge`` to any other value (``"sigma"``,
``"phase"``, or ``"none"``), that choice is respected without modification.

The GMRES backward (``ad_backward_method="gmres"``) is tracked as an open
gap — see issue #292 and the ``xfail``-marked regression test in
``tests/test_ad_utils.py``.

## Critical Components

### 1. Phase Gauge Fixing (default for explicit AD)

The phase gauge fix is two differentiable steps applied to every
corner and edge after each CTM sweep:

1. **Frobenius normalization** — divides each tensor by its Frobenius norm
   so that absorbed-layer singular values cannot exponentially grow or
   shrink across sweeps.
2. **Global phase fix** — picks the first sufficiently large element of
   each tensor and rotates the global U(1) phase so that this element is
   real-positive (variPEPS ``_post_process_CTM_tensors`` convention).

Together they remove the dominant gauge ambiguity at negligible cost —
no power iteration, no eigensolve, fully differentiable — and are the
reason the qr+phase path scales to chi=64 without NaNs.

### 2. Sigma Gauge Fixing (implicit-diff path)

Sigma gauge aligns each iteration's environment to the previous one using
transfer matrix eigenvectors, making element-wise convergence monotonic.
This is required for the implicit-diff backward, where a well-conditioned
fixed-point environment is needed for the ``(I - J^T) λ = g`` solve to
behave well.

**Implementation**: Power iteration (30 iterations) computes the leading
eigenvector of the double-layer transfer matrix. This is fully
JAX-differentiable, unlike `jnp.linalg.eig`.

**Sweep mutation fix (PR #291)**: The multisite CTM sweep used to mutate
the environment dict in-place. A shallow copy (`envs = dict(envs)`) at
the start of each sweep ensures callers that saved a reference to the
input dict (for sigma gauge comparison) still see the pre-sweep
environments. Before this fix, sigma gauge in the Python convergence
loop silently degenerated into a no-op.

### 3. Projector Methods

**qr** (recommended for explicit AD): QR-factored small eigenproblem. Best
benchmarked energy at D=2 with the phase gauge and scales cleanly to
chi=64+. Fails for AD **without** a gauge fix (phase or sigma).

**eigh**: Forms density matrix ρ = C1g·C1g† + C4g·C4g† and eigendecomposes.
Best energy with sigma gauge; slower than qr for large chi. Block-sparse
path available for SymmetricTensor.

**svd** (Fishman): Cross-product M = C1g†·C4g, SVD, projector P = C4g·V·S^{-1/2}.
Works with sigma gauge (E=-0.6624). The S^{-1/2} weighting is differentiable
(no stop_gradient), allowing gradient flow through singular values.

### 4. SVD Projector: Two-Projector Fishman

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

### 5. L-BFGS + Hager-Zhang Line Search

Second-order optimization with approximate Wolfe conditions. Metric
preconditioning (Rader et al., arxiv:2511.09546) uses the environment metric
tensor as a natural gradient preconditioner.

### 6. C4v Symmetry Enforcement

For the square lattice with C4v-symmetric Hamiltonians, enforcing C4v symmetry
on the site tensor reduces the parameter space from D²d to ~D²d/8, improving
optimization stability and speed.

## Known Limitations

1. **GMRES implicit backward**: ``ad_backward_method="gmres"`` is documented
   unstable (spectral radius > 1 without tight sigma-gauge alignment) and
   its regression test is marked ``xfail`` in ``tests/test_ad_utils.py``.
   Tracked by issue #292. Prefer the explicit-AD path (Path 1) or
   ``ad_backward_method="vjp"`` until the GMRES path is stabilized.

2. **``forward_gauge="none"`` on the JIT CTM path**: The JIT
   ``jax.lax.while_loop`` CTM kernel only dispatches on ``"phase"`` /
   ``"qr"`` / ``"sigma"``; ``"none"`` falls through to the qr gauge on the
   JIT path. For the Python-loop CTM (``jit_ctm=False``) and the explicit-AD
   unrolled path, ``"none"`` is honored end-to-end as a benchmark /
   diagnostic mode.

3. **2-site explicit AD**: The 2-site optimizer supports the same
   auto-phase promotion as the 1-site C4v path, but its convergence has
   not been benchmarked as thoroughly as the 1-site C4v workflow. Treat
   the 2-site explicit-AD path as experimental for now.

4. **SymmetricTensor**: Block-sparse tensors fall back to the Python loop
   (not JIT-traceable). Both phase and sigma gauge work on this path, but
   with dense fallbacks for gauge fixing.

5. **Sigma gauge cost**: ~40% overhead from power iteration per sweep.
   Phase gauge is the recommended replacement for explicit AD; reserve
   sigma gauge for the implicit-diff path.

## Stall recovery (`gs_stall_recovery`)

When the L-BFGS / CG line search fails to make progress, the optimizer
runs a stall-recovery routine. Two modes are supported:

- ``"noise"`` — inject a ``gs_noise_amplitude`` (default 10 %) Frobenius
  perturbation on the current params and reset the L-BFGS history.
  **Required for the 1-site C4v production path**, which sits on an
  SU-init plateau with gradient norms around ``1e-10`` that would
  otherwise trip ``gs_conv_tol`` before the first real step.
- ``"reset"`` — clear the L-BFGS ``(s, y)`` history and the CG beta
  state so the next iteration is a plain (preconditioned) steepest
  descent step from the current iterate. No rollback, no randomness.
  **Default for the 2-site path** because the 10 % noise kick in the
  ~32-dimensional D=2 parameter space lands in non-variational CTM
  regions and drives the optimizer into unphysical "best" energies
  (see issue #298).

Leaving ``gs_stall_recovery=None`` (the default) auto-selects the
right mode for the unit cell at dispatch time. An explicit user
setting is never overridden.

For extra safety on 2-site runs, set ``gs_energy_floor`` to a value a
bit below the expected variational minimum (e.g. ``2 * E_literature``).
Any in-loop candidate energy at or below the floor is rejected as a
non-variational CTM artifact — this catches pathological "best"
states arising from the ``_rdm2x1_tensor_2site`` trace-normalization
at near-zero trace. The check is off by default
(``gs_energy_floor=None``).

## Known open problems

- **2-site L-BFGS convergence gap at χ=8** — The 2-site AD path with
  L-BFGS + Hager-Zhang + metric precond + SU init reaches only
  ``E/site ≈ -0.56`` at D=2 χ=8 in 20 steps, vs. the ≈-0.65 literature
  value documented in issue #298's trajectory study. The gap is not
  a stall-recovery problem (the reset branch never fires on this
  trajectory); root cause is under investigation. Tracked by issue
  #299.

## References

- Francuz, Schuch, Vanhecke, *PRR* **7**, 013237 (2025) — Stable AD through CTM
- Rader et al., arXiv:2511.09546 (2025) — Metric preconditioning
- Zhang, Yang, Corboz, arXiv:2505.00494 (2025) — Chi-ramping schedule
- Naumann et al., arXiv:2502.10298 (2025) — Split CTMRG with two-projector formulation
- Fishman et al., *PRB* **98**, 235148 (2018) — SVD (Fishman) projectors
