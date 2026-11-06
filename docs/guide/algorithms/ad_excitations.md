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

#### Phase gauge (``forward_gauge="phase"``; what the ``"auto"`` default resolves to off the implicit-AD path)

Applies **Frobenius normalization + global phase fix** (variPEPS-style)
to each corner and edge after every CTM step.  Cheapest gauge fix; applied
on the implicit-AD path and the legacy ``ad_utils`` paths, 1-site and
2-site.  ``optimize_gs_ad``'s explicit-AD energy applies no forward gauge,
so the setting has no effect there (#1074).  See ``docs/guide/algorithms/ctm.md`` for details.

#### QR gauge (``forward_gauge="qr"``)

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

```{warning}
`compute_excitations` is deprecated: it truncates the momentum sum (#1052), and its
lowest energies on published Heisenberg states are negative (-0.10 at D=2 and -0.27
at D=3 at (pi,0)). Use [`compute_spectrum`](#spectrum-and-spectral-function) where
its momenta apply.
```

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

(spectrum-and-spectral-function)=
## Spectrum and Spectral Function

`compute_spectrum` includes the complete momentum sum through the CTM, and gives
spectral weights. `compute_excitations` truncates that sum to nearest neighbours
(#1052), and its lowest energies on published Heisenberg states are negative.

### Example

```python
import numpy as np
from tenax import SpectrumConfig, compute_spectrum, spectral_function

Sy = np.array([[0.0, 0.5], [-0.5, 0.0]])  # i S^y: the real form
config = SpectrumConfig(chi=16, checkpoint_dir="runs/x_chi16")
result = compute_spectrum(A, gate, (np.pi, 0.0), {"Sy": Sy}, config)

result.omega                  # excitation energies above the ground state
result.weights["Sy"]          # |<lambda|S^y|0>|^2 for each omega
result.structure_factor["Sy"] # s(k)

grid = np.linspace(0.0, 5.0, 501)
S = spectral_function(result, "Sy", grid, eta=0.1)  # Lorentzian half-width eta
```

`A`, the gates, the operators and the momentum must share one frame. The
sublattice rotation keeps the momentum of `S^y`, and shifts `S^x` and `S^z` by
`(pi, pi)`. So in the rotated frame, physical Gamma for `S^x` and `S^z` is `(pi, pi)`.

### Limits

- Dense real `A` only. A `SymmetricTensor` or a complex `A` raises.
- Only `(pi, 0)`, `(0, pi)` and `(pi, pi)`, because the 2x2 cell reaches only
  those. Gamma raises: the method needs k != 0.
- The cost is one column for each entry of `A`. At D=3, chi=18 one H column takes
  about 30 s on 12 CPU cores. At D=4, chi=16 it takes about 135 s, which is about
  19 h per momentum for H alone. The norm adds up to the same again.
- Use `checkpoint_dir` on any long run. A rerun resumes, and changing only
  `operators` reuses all columns.
- `basis` restricts the tangent space, for example to a charge sector. An operator
  whose `O A` has weight outside `basis` warns, and its results cover only the
  part inside.

### chi convergence

One run cannot show convergence, so compare at least two chi values. The diagnostics
do not replace this. At D=3, chi=18 is too small: it gives negative modes (-4.42 and
-0.30 at X), and their Rayleigh quotients are positive at chi=27 and about +6 at
chi=54. Fixed chi=18 eigenvectors evaluated at larger chi give:

| chi | 18 | 27 | 36 | 54 |
|---|---|---|---|---|
| omega(X) | 2.2641 | 2.2665 | 2.2676 | 2.2679 |
| omega(M) | 0.2688 | | 0.3056 | 0.3083 |

Negative modes stay in `omega`. The warning tells you to raise chi.

### Diagnostics

| Field | Meaning | Warns if |
|---|---|---|
| `energy_density` | energy per site | the CTM did not converge |
| `n_asymmetry` | largest entry of N - N^T | above 1e-4 |
| `h_asymmetry` | largest entry of H - H^T | above 1e-3 |
| `n_min_eig` | lowest N eigenvalue over `max(largest, 1e-2)` | below -1e-4 |
| `n_negative_modes` | modes with omega < 0 | any |
| `sum_rule_deficit` | per operator, 1 - sum(weights)/s(k) | above 1e-3 |
| `gauge_residual` | largest singular value of N on A and the gauge vectors, `basis=None` only | above 1e-3 |
| `null_dim` | eigenvalues of N below the cut | never |

The cut is `null_tol * max(largest N eigenvalue, 1e-2)`. With the floor, a `basis`
inside the null space gives rank 0 and a warning, not noise reported as modes.

The H limit is looser because of one noisy reverse-mode column: at D=2, chi=10, M it
gives 6.6e-4, while omega stays at 0.64873 across `fd_step` 1e-3, 1e-4 and 30, 60
sweeps.

### Validation

Hasik's D=2 J2=0 state (`j1j2_ipeps_states`, CC BY 4.0), chi=10:

| | X = (pi, 0) | M = (pi, pi) | Reference |
|---|---|---|---|
| Lowest omega | 2.293 | 0.6487 | X: series ~2.18, QMC ~2.16; M: 0 as D grows |
| s(k) for S^y | 0.20399 | 0.80857 | window oracle in random directions, within 1e-4 |
| Top S^y weight | 0.1904 | 0.7744 | |
| Sum rule deficit | 1e-13 | 2e-15 | |

Projector derivatives matter for H, not for N. On the same state at X, the matrix
element of `H - E0 N` in the direction `S^y A` is:

| chi | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| `projector_backward="flow"` (used here) | 0.4976433 | 0.4976434 | 0.4976434 | 0.4976434 |
| `projector_backward="frozen"` | 0.4958283 | 0.4976095 | 0.4976100 | 0.4976751 |

`s(k)` agrees between the two to all 7 printed digits at each of these chi values.

## References

- Ponsioen, Assaad & Corboz, *SciPost Phys.* **12**, 006 (2022) --
  AD excitations method
- Francuz et al., *Phys. Rev. Research* **7**, 013237 (2025) --
  Stable AD of CTM (custom SVD VJP, implicit differentiation, gauge fixing)
- Rader et al., arXiv:2511.09546 --
  Metric preconditioning (natural gradient) for iPEPS optimization
- Zhang, Yang & Corboz, arXiv:2505.00494 --
  Chi-ramping schedule for stable convergence at large bond dimensions
