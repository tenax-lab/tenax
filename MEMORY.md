# Memory

## 2026-04-26: Gauge Selection and Normalization for Differentiable CTMRG

### Objective
Evaluate how gauge fixing and tensor normalization affect gradient stability in differentiable CTMRG for complex tensors under AD (Wirtinger calculus; e.g., JAX).

### Core Issue
Projector phases can drift iteration-to-iteration (eigensolver/SVD ambiguity). Physical observables may converge while environment entries do not, breaking smooth fixed-point behavior and destabilizing implicit differentiation (VJP/adjoint solves).

### Gauge Strategy
- Recommended for implicit AD: `phase` gauge (Frobenius normalization + phase fixing).
  - Gives the stable backward behavior observed in this codebase for complex AD runs.
- Discouraged in optimization loops: QR sign/phase gauge as the primary continuity mechanism.
  - Enforcing real-positive `diag(R)` can introduce branch-cut sensitivity near small diagonal entries.
  - Small perturbations can trigger abrupt phase flips and noisy/ill-conditioned gradients.

### Normalization Strategy (Complex AD)
- Safest default: Frobenius normalization `T <- T / max(||T||_F, eps)`.
  - Real scalar scale factor, robust global magnitude control, AD-friendly in Wirtinger frameworks.
- Contextual option: fixed-index normalization (`T / T[idx_fixed]`).
  - Holomorphic and phase-pinning, but unstable if anchor approaches zero.
  - Use only when nonzero anchor is guaranteed and control flow is static.
- Discouraged on AD path: max-abs normalization (`T / max(abs(T))`).
  - Nonsmooth max-crossings and non-holomorphic abs create gradient discontinuities.

### Recommended Default Pipeline
- Implicit AD must use:
  - `projector_method="svd"`
  - `forward_gauge="phase"`
  - `ctm_conv_method="elementwise"`
- Use Frobenius norm for tensor scaling (phase gauge path).
