# AD-Based iPEPS Excitations

Tenax implements the quasiparticle excitation method from
[Ponsioen, Assaad & Corboz, SciPost Phys. 12, 006 (2022)](https://scipost.org/10.21468/SciPostPhys.12.1.006),
using JAX automatic differentiation to construct the effective Hamiltonian
and norm matrices. The stable AD infrastructure follows
[Francuz et al., Phys. Rev. Research 7, 013237 (2025)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.013237).

## Stable AD Infrastructure

Naively differentiating through CTM has three problems. The solutions live
in `tenax.algorithms.ad_utils`.

### 1. Custom truncated SVD (`truncated_svd_ad`)

The standard CTM (used for simple-update iPEPS) builds projectors via
`eigh`. For AD-based optimization and excitations, the CTM instead uses
`truncated_svd_ad`, which provides a custom VJP with two key fixes:

The standard SVD adjoint has two failure modes:

- **Degenerate singular values**: the factor $1/(s_i^2 - s_j^2)$ diverges.
- **Truncation**: discarding singular values drops the coupling between kept
  and truncated subspaces.

**Lorentzian regularization** replaces the divergent F-matrix:

$$
F_{ij} = \frac{s_i^2 - s_j^2}{(s_i^2 - s_j^2)^2 + \varepsilon^2},
\qquad \varepsilon \sim 10^{-12}
$$

This smoothly approaches $1/(s_i^2 - s_j^2)$ when singular values are
well-separated but stays finite when they are degenerate. The diagonal is
zeroed (gauge freedom).

**Truncation correction** (Francuz et al., the dominant error source) adds
two terms to the backward pass:

$$
\bar{M}_{\text{trunc}} =
  (I - U U^\dagger)\, \bar{U}\, \text{diag}(1/s)\, V_h
  \;+\;
  U\, \text{diag}(1/s)\, \bar{V}_h\, (I - V V^\dagger)
$$

These project the cotangent onto the complement of the kept subspace,
accounting for how changes in $M$ rotate vectors *into* the truncated part.

The full backward pass assembles five terms:

| # | Term | Role |
|---|------|------|
| 1 | $U\,\text{diag}(\bar{s})\,V_h$ | Direct gradient through singular values |
| 2 | $U\,(F \odot U^\dagger \bar{U}_{\text{anti}})\,\text{diag}(s)\,V_h$ | Rotation of left singular vectors (kept subspace) |
| 3 | $U\,\text{diag}(s)\,(F \odot V^\dagger \bar{V}_{\text{anti}})\,V_h$ | Rotation of right singular vectors (kept subspace) |
| 4 | $(I - UU^\dagger)\,\bar{U}\,\text{diag}(1/s)\,V_h$ | Truncation correction from $\bar{U}$ |
| 5 | $U\,\text{diag}(1/s)\,\bar{V}_h\,(I - VV^\dagger)$ | Truncation correction from $\bar{V}_h$ |

### 2. CTM fixed-point differentiation (`ctm_tensor_converge`)

Instead of backpropagating through all CTM iterations (storing
$O(\text{max\_iter})$ intermediate environments), we use **implicit
differentiation** of the fixed-point equation $x^* = f(A, x^*)$.

**Forward pass**: run CTM to convergence, cache only the final $(A, x^*)$.

**Backward pass**: given cotangent $\bar{x}$ for the environment, solve

$$
(I - J_x^T)\,\lambda = \bar{x}
$$

where $J_x = \partial f / \partial x$ is the Jacobian of one CTM step.
This is solved by **fixed-point iteration**:

$$
\lambda_{n+1} = \bar{x} + J_x^T \lambda_n
$$

Each iteration computes $J_x^T \lambda$ via a single `jax.vjp` call (no
explicit Jacobian). Once converged, the gradient w.r.t. $A$ is:

$$
\bar{A} = \frac{\partial f}{\partial A}^T \lambda
$$

### 3. Gauge fixing

CTM environments have a gauge ambiguity -- the fixed point is only unique
up to invertible transformations on bond indices. Without fixing this,
element-wise convergence fails and the implicit differentiation equation is
ill-defined.

Tenax provides two gauge-fixing strategies, selected via
``CTMConfig.forward_gauge``:

#### QR gauge (``forward_gauge="qr"``, default)

Applies **QR decomposition** (via ``tenax.linalg.qr``, block-sparse
for ``SymmetricTensor``) to each corner after every CTM step:

$$
C = QR \quad\Longrightarrow\quad C_{\text{new}} = R, \quad
Q^\dagger \text{ absorbed into adjacent edge tensors}
$$

The R factor from QR has a unique sign convention (positive diagonal),
giving a unique fixed point. Using block-sparse QR directly on ``Tensor``
objects avoids ``todense()``/``from_dense()`` round-trips, giving cleaner
gradients during AD.

#### Sigma gauge (``forward_gauge="sigma"``)

The sigma gauge aligns the CTM environment to the *previous iteration's*
environment using the leading eigenvector of each edge's double-layer
transfer matrix. The algorithm:

1. For each edge direction, compute the leading eigenvector $\rho$ of the
   transfer matrix $T_{\text{old}}$ and $T_{\text{new}}$ via **power
   iteration** (matrix-free, fully JAX-differentiable).
2. QR-factorize each $\rho$ to get orthogonal bases $Q_{\text{old}}$ and
   $Q_{\text{new}}$.
3. Compute the gauge transformation
   $\sigma = Q_{\text{new}} Q_{\text{old}}^\dagger$ and apply it to corners
   and edges.

This ensures that the environment converges **element-wise**, not just
spectrally. Element-wise convergence is critical for two reasons:

- The implicit differentiation equation $(I - J_x^T)\lambda = \bar{x}$
  requires a well-defined fixed point in the full tensor space, not just
  the singular-value subspace.
- Without sigma gauge, the ``eigh`` projector CTM exhibits chaotic
  gauge wandering: corner singular values converge but the environments
  themselves do not, causing GMRES to diverge and VJP to accumulate noise.

**Recommended for all AD optimization** (both implicit and explicit).

### 4. Backward methods for implicit differentiation

The implicit differentiation backward pass solves
$(I - J_x^T)\lambda = \bar{x}$ using one of two methods:

#### Iterative VJP (``ad_backward_method="vjp"``, default)

Neumann series accumulation matching YASTN's approach
(Francuz et al., arXiv:2311.11894):

$$
\lambda_{n+1} = \bar{x} + J_x^T \lambda_n
$$

Each iteration uses a single ``jax.vjp`` call. With sigma gauge in the
step function, the spectral radius of $J_x^T$ is less than 1 in the
physical subspace, ensuring convergence. However, convergence can be slow
(many VJP iterations).

#### GMRES (``ad_backward_method="gmres"``)

Direct Krylov solve of the linear system. Converges much faster than
iterative VJP when the system is well-conditioned. **Recommended** when
``forward_gauge="sigma"`` is enabled.

Without sigma gauge, GMRES can diverge because the linear system is
ill-conditioned (the fixed point is not unique element-wise).

### 5. Explicit AD (``gs_implicit_ad=False``)

Instead of implicit differentiation, explicit AD backpropagates through the
unrolled CTM iteration graph. This is now the **default** mode
(``gs_implicit_ad=False``).

The forward pass has two phases:

1. **Warmup** (``gs_explicit_ad_warmup`` steps): CTM sweeps with
   ``stop_gradient`` -- no gradient tracking, just environment warm-up.
2. **Tracked** (``gs_explicit_ad_steps`` steps): CTM sweeps with full
   gradient tracking through the JAX computation graph.

Sigma gauge is applied at every sweep (both warmup and tracked phases) to
maintain element-wise stability. The power iteration for the transfer-matrix
eigenvector is fully differentiable, so sigma gauge works seamlessly with
explicit AD.

Explicit AD works well for the **1-site C4v path** (``gs_c4v=True``), where
each CTM sweep is a single directional move, keeping the unrolled graph
compact. It is generally slower than implicit GMRES but avoids linear-solve
convergence issues entirely.

## AD Ground State Optimization

`optimize_gs_ad` uses the stable AD pipeline to compute exact energy
gradients and optimize the iPEPS tensor with optax:

```python
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(
        chi=16,
        max_iter=100,
        forward_gauge="sigma",       # stabilize element-wise convergence
        ad_backward_method="gmres",  # recommended backward method
    ),
    gs_optimizer="cg",
    gs_num_steps=100,
    gs_verbose=True,
    gs_log_interval=10,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, A_init=None, config=config)
```

The gradient flows through the full CTM + energy pipeline: the
`ctm_tensor_converge` custom VJP handles implicit differentiation, and
`truncated_svd_ad` handles SVD stability.

### Chi-ramping schedule

For production runs, use ``optimize_gs_ad_chi_schedule`` to ramp the
environment bond dimension progressively. Each stage uses the optimized
tensor from the previous stage as initialization:

```python
from tenax import optimize_gs_ad_chi_schedule

A_opt, env, E_gs = optimize_gs_ad_chi_schedule(
    H_bond, None, config, [(8, 30), (16, 20)]
)
```

The ``chi_schedule`` argument is a list of ``(chi, num_steps)`` tuples.
The base config provides all other settings; only ``chi`` and
``gs_num_steps`` are overridden per stage. This avoids cold-starting at
large chi, which can be slow and prone to local minima
(Zhang, Yang & Corboz, arXiv:2505.00494).

## Excitation Spectrum

The excitation ansatz places a perturbation tensor $B$ (same shape as $A$)
at one site with Bloch momentum phase $e^{i\mathbf{k}\cdot\mathbf{r}}$.

### Key idea (Ponsioen et al.)

The effective Hamiltonian $H_{\text{eff}}(\mathbf{k})$ and norm
$N(\mathbf{k})$ matrices are built column-by-column via AD:

$$
N_{:,m} = \nabla_{B^*} \langle\Phi_k(B)|\Phi_k(B)\rangle\big|_{B=e_m},
\qquad
H_{:,m} = \nabla_{B^*} \langle\Phi_k(B)|(H-E_{\text{gs}})|\Phi_k(B)\rangle\big|_{B=e_m}
$$

Since the functionals are bilinear in $B$ and $B^*$, the gradient w.r.t.
$B^*$ at the $m$-th basis vector gives the $m$-th column directly.

The excitation energies come from the **generalized eigenvalue problem**:

$$
H_{\text{eff}}\, v = \omega\, N\, v
$$

solved after projecting out the null space of $N$.

### Example

```python
import numpy as np
from tenax import ExcitationConfig, compute_excitations, make_momentum_path

config = ExcitationConfig(num_excitations=3)

momenta = make_momentum_path("brillouin", num_points=20)
result = compute_excitations(A_opt, env, H_bond, E_gs, momenta, config)

# result.energies  -- shape (num_k, num_excitations)
# result.momenta   -- shape (num_k, 2)
```

### Mixed double-layer tensors

The norm and energy functionals require contracting CTM networks with $B$
substituted at various sites. The module provides:

- `_build_mixed_double_layer(A, B, "ket"/"bra")` -- closed (physical traced)
- `_build_mixed_double_layer_open(A, B, "ket"/"bra")` -- open (physical exposed)
- `_rdm2x1_mixed` / `_rdm1x2_mixed` -- 2-site RDMs with arbitrary
  `(ket, bra)` substitutions at each site

Each RDM variant specifies which tensor appears in the ket and bra layers:
`("A","A")` for ground state, `("B","A")` for $B$ in ket, etc.

## References

- Ponsioen, Assaad & Corboz, *SciPost Phys.* **12**, 006 (2022) --
  AD excitations method
- Francuz et al., *Phys. Rev. Research* **7**, 013237 (2025) --
  Stable AD of CTM (custom SVD VJP, implicit differentiation, gauge fixing)
- Rader et al., arXiv:2511.09546 --
  Metric preconditioning (natural gradient) for iPEPS optimization
- Zhang, Yang & Corboz, arXiv:2505.00494 --
  Chi-ramping schedule for stable convergence at large bond dimensions
