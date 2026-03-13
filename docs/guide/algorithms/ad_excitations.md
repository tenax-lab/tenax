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

### 2. CTM fixed-point differentiation (`ctm_converge`)

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

### 3. Gauge fixing (`_gauge_fix_ctm`)

CTM environments have a gauge ambiguity -- the fixed point is only unique
up to invertible transformations on bond indices. Without fixing this,
element-wise convergence fails and the implicit differentiation equation is
ill-defined.

The fix applies **QR decomposition** to each corner after every CTM step:

$$
C = QR \quad\Longrightarrow\quad C_{\text{new}} = R, \quad
Q^\dagger \text{ absorbed into adjacent edge tensors}
$$

The R factor from QR has a unique sign convention (positive diagonal),
giving a unique fixed point.

## AD Ground State Optimization

`optimize_gs_ad` uses the stable AD pipeline to compute exact energy
gradients and optimize the iPEPS tensor with optax:

```python
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    gs_verbose=True,
    gs_log_interval=10,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, A_init=None, config=config)
```

The gradient flows through the full CTM + energy pipeline: the
`ctm_converge` custom VJP handles implicit differentiation, and
`truncated_svd_ad` handles SVD stability.

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
