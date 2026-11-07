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

For a J1-J2 model pass `gate_nnn`, used on both diagonals of every site, and
`result.energy_density` then includes it. Next-nearest neighbours share a sublattice,
and `S.S` and XXZ do not change when both sites rotate. So in the rotated frame this
is `J2 * S.S` without the rotation. A gate that does change, such as a field `S^z`
term, differs between the two sublattices in that frame, and one `gate_nnn` cannot
express it. The diagonal RDM adds little to a column: 15 s against 14 s on one state at
D=3, chi=12, on 6 cores.

### Workflow

1. **Get the state in the rotated frame.** `A` is dense and real, and `gate` is the
   rotated gate.
2. **Find chi on a small basis.** Use `A` and each `O A` as columns, so four columns for
   three spin operators (`--basis sma` in `benchmarks/ipeps_spectrum_hasik.py`). Compare
   `s(k)` and the first moment at two or three chi values. These two need only `O A`,
   not the full tangent space.
3. **Reduce the basis.** The cost is one column for each basis vector. A `basis` must
   hold `O A` for each operator, and a symmetry block does this with far fewer columns
   than the full tensor. See [Benchmarks](#spectrum-benchmarks).
4. **Run with `checkpoint_dir`.** A kill costs one column.
5. **Scan `null_tol`.** The cut sets the size of the subspace, and it moves the modes.
   Report it with them. The checkpoint does not depend on `null_tol`, so each rerun
   reuses all columns: 83 s and 1.3 GB at D=4, chi=24, on 6 cores.

   ```python
   for null_tol in (1e-5, 1e-4, 5e-4, 1e-3):
       config = SpectrumConfig(chi=24, null_tol=null_tol, checkpoint_dir="runs/x_chi24")
       result = compute_spectrum(A, gate, (np.pi, 0.0), {"Sy": Sy}, config, basis=basis)
       print(null_tol, result.rank, result.omega[:3])
   ```

6. **Read the diagnostics**, then make `S(k, omega)` with `spectral_function`.

### Limits

- Dense real `A` only. A `SymmetricTensor` or a complex `A` raises.
- Only `(pi, 0)`, `(0, pi)` and `(pi, pi)`, because the 2x2 cell reaches only
  those. Gamma raises: the method needs k != 0.
- The cost is one column for each entry of `A`, or for each vector of `basis`. One
  column gives H and N together. At D=3, chi=36 it takes 106 s on 12 CPU cores, which
  is 4.8 h per momentum. At D=4, chi=24 it takes 217 s on 6 cores, which is 5 h for a
  block of 78 columns, and about 31 h for all 512. See
  [Benchmarks](#spectrum-benchmarks).
- Use `checkpoint_dir` on any long run. A rerun resumes, and changing only
  `operators` or `null_tol` reuses all columns.
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

(spectrum-benchmarks)=
### Benchmarks

`benchmarks/ipeps_spectrum_hasik.py` runs these on the states of `j1j2_ipeps_states`
(CC BY 4.0). The reference is Ponsioen, Hasik & Corboz, PRB **108**, 195111
(arXiv:2306.13327). The machine is one CPU, a Threadripper 7960X with 24 physical cores
and 251 GB. One column gives H and N together.

| Benchmark | State, k | chi | Columns | Time per column | Criterion | Measured | Result |
|---|---|---|---|---|---|---|---|
| 1 | A1 J2=0 D=3, X | 36 | 162 | 106 s on 12 cores | omega near 2.268, no negative modes | 2.2652, none | pass |
| 1 | A1 J2=0 D=3, M | 36 | 162 | 195 s on 12 cores, shared; 108 s at best | no negative modes | none (omega 0.2843) | pass |
| 2 | A1 and U1B states | | none | under 1 s for each state | no U(1) in A1 at D=3 and D=4 | [`best_xz`](#spectrum-datasets) | confirmed |
| 3 | A1 J2=0.3 D=4 | 32 | none | 9 min 51 s for E, on 12 cores | E within 1e-5 of -0.5549696161 | -0.5547673648 | miss, [energy window](#spectrum-energy-window) |
| 4 | U1B J2=0 D=4, X | 24 | 78 | 216 s on 6 cores | omega below D=3 (2.2444), toward 2.18 | 2.1886 | pass |
| 4 | U1B J2=0 D=4, M | 24 | 78 | 200 s on 6 cores | omega below D=3 (0.4835) | 0.2112 | pass |
| 5 | U1B J2=0.3 D=4, X | 24 | 78 | 217 s on 6 cores | peak within 0.05 of 1.05 at eta=0.1 | 1.040 | pass |
| 5 | same run | | | | s within 1% of 0.182 | 0.18556 | miss, [Fig. 1](#spectrum-fig-1) |

The A1 runs use the full tensor and the default `fd_step`. The U1B runs use
`--basis even`, in a random orthogonal basis of that block, and `fd_step=1e-3`. They ran
with the Lorentzian backward, before #1178. On the same state at chi=24, in four columns,
the backward of #1178 at `fd_step=1e-4` gives `s` = 0.18555965, and these settings give
0.18555959.

Benchmark 4 reads omega at the default cut. Omega moves with the cut, so compare each
state at more than one rank:

| Rank | 49 | 38 | 26 | 19 | 14 | 8 |
|---|---|---|---|---|---|---|
| omega(X), U1B J2=0 D=4, chi=24 | 2.18 | 2.1867 | 2.1886 | 2.1942 | 2.2129 | 2.2204 |
| omega(X), U1B J2=0 D=3, chi=27 | | | | 2.244 | 2.2444 | 2.2572 |

At M, omega barely moves with the cut: 0.2107 to 0.2188 from rank 35 to rank 3 at D=4, and
0.4826 to 0.4884 from rank 19 to rank 2 at D=3. `s` at M is 0.9037 at D=3 and 1.9701 at
D=4.

(spectrum-datasets)=
#### The two datasets

`j1j2_ipeps_states` has two single-site datasets at these parameters.
`single-site_pg-C4v-A1` (A1 here) has no exact U(1).
`single-site_pg-C4v-A1_internal-U1` (U1B here) has one. The task `u1` of the benchmark
gives the residual at the best axis. A synthetic U(1) tensor gives 1e-15, and a random
tensor gives 0.64:

| State | `best_xz` |
|---|---|
| A1 J2=0, D=2 | 3.8e-4 |
| A1 J2=0, D=3 | 0.66 |
| A1 J2=0, D=4 | 0.052 |
| A1 J2=0.3, D=4 | 0.71 |
| U1B J2=0, 0.25, 0.3 at D=3 and D=4; J2=0.3 at D=5 | 1.0e-15 to 1.7e-15 |

The paper's states are U1B: they reproduce its Fig. 3, and A1 at D=3 does not (`2 s_y` =
0.39306 at chi=36, against 0.406). In the rotated frame on a U(1) state, the physical
`s_+(k)` is `2 s_y(k)`, and the "Energy" of Fig. 3 is `2 f_y`, with `f_y` the first moment
of `S^y`. The Fig. 3 columns are read from the plot:

| U1B state, X | chi | `2 s_y` | Fig. 3(c, d) | `2 f_y` | Fig. 3(a, b) |
|---|---|---|---|---|---|
| J2=0, D=3 | 27 | 0.40633 | 0.406 | 1.00955 | 1.009 |
| J2=0, D=4 | 12 | 0.38730 | 0.387 | 0.98011 | 0.980 |
| J2=0.25, D=3 | 12 | 0.37953 | 0.379 | 0.65795 | 0.658 |
| J2=0.25, D=4 | 12 | 0.37241 | 0.372 | 0.66014 | 0.661 |

#### Reduced bases

On a U(1) state, `S^x A` and `S^y A` lie in one block: charge offset +-2, even under
`u <-> d` and `l <-> r`. `--basis even` takes this block: 31, 78, 173 and 317 columns at
D=3, 4, 5 and 6, against 162, 512, 1250 and 2592 for the full tensor. At D=3 its five top
weighted modes equal those of the whole charge block (72 columns) to 4 decimals in omega.

With no U(1), `--basis reflection` takes only the two reflections: 18, 72 and 200 columns
at D=2, 3 and 4. A run at D=2, and a projection of the stored full run at D=3, give the
top modes of the full result to all printed digits.

(spectrum-fig-1)=
#### Fig. 1

Fig. 1 of the paper is vector data in its PDF, so its curves can be read exactly: J2=0.3,
D=4, `k = (pi, 0)`, eta=0.1. They are unit-height Lorentzians,
`sum(w_n * eta**2 / ((omega - omega_n)**2 + eta**2))`, and `spectral_function` is
area-normalised, so multiply it by `pi * eta` to compare. To run it:

```bash
python benchmarks/ipeps_spectrum_hasik.py <U1B J2=0.3 D=4 state> spectrum --chi 24 --j2 0.3 \
    --momentum X --basis even --fd-step 1e-3 --checkpoint-dir runs/fig1 --out fig1.npz
```

The paper sets the size of the subspace by hand, and `null_tol` is that setting here.
On the same 78 columns:

| `null_tol` | Rank | First peak | Second peak | rms against the paper's curve |
|---|---|---|---|---|
| The paper's curve | | (1.051, 0.1081) | (2.132, 0.048) | |
| 1e-5 | 57 | (1.032, 0.1034) | (1.944, 0.0445) | 0.0107 |
| 1e-4 (default) | 32 | (1.040, 0.1061) | (2.056, 0.0495) | 0.0058 |
| 5e-4 | 22 | (1.047, 0.1082) | (2.099, 0.0516) | 0.0029 |
| 6e-4 | 20 | (1.051, 0.1093) | (2.118, 0.0518) | 0.0019 |
| 8e-4 | 17 | (1.051, 0.1094) | (2.124, 0.0496) | 0.0014 |
| 1e-3 | 15 | (1.068, 0.1147) | (2.350, 0.0455) | 0.0129 |
| 3e-3 | 8 | (1.077, 0.1164) | (2.485, 0.0442) | 0.0161 |

So at 6e-4 to 8e-4 the curve sits on the paper's, and the first peak at the default is
inside the criterion too. `s` does not follow. The sum of weights is 0.18542 to 0.18556
down to rank 8, 2.0% above the caption's 0.182. By one quadrature on the paper's points,
the area under this curve is 1.9% to 2.3% above the paper's. The caption and the paper's
curve agree: scaled by the same quadrature, that area gives 0.181 to 0.182, and 0.186 to
0.187 for its VMC curve (caption 0.186).

The cause is not known. Ruled out: the cut, a mean over the components (0.17228), and
chi:

| State | chi | `s_y` | E, 2x1 window |
|---|---|---|---|
| U1B J2=0.3, D=4 | 12, 24, 36 | 0.18557, 0.18556, 0.18556 | -0.5547188, -0.5549078, -0.5549195 |
| A1 J2=0.3, D=4 | 12, 24, 36 | 0.18506, 0.18476, 0.18467 | -0.5516633, -0.5542744, -0.5548363 |
| U1B J2=0.3, D=5 | 16, 25 | 0.18279, 0.18246 | -0.5541852, -0.5548632 |
| U1B J2=0.3, D=6 | 16 | 0.18281 | -0.5527211 |

D=5 is 0.25% above the caption at chi=25. That is a lead, not a finding: the caption says
D=4, `s` at D=5 still moves with chi, and these D=5 and D=6 runs use four columns, so they
do not test the curve. D=6 at chi=16 is far from its energy at large chi (-0.5555266).

(spectrum-energy-window)=
#### The energy window

`energy_density` takes the nearest-neighbour term on a 2x1 window. The `.dat` files of the
dataset take it on a 2x2 window. The limit is the same, and at finite chi the 2x1 error
is about twice as large: against chi=800, its ratio to the 2x2 error is 2.06 to 2.10 at
chi=32 to 64. `energy --window 2x2` gives the 10 digits of the `.dat` file (-0.5549696161
at chi=32). So Benchmark 3 compared two estimators. It stays a miss of its criterion, and
this is its cause.

#### Memory

Set these for a long run:

```bash
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2
```

Without them glibc fragments the heap. At D=2, chi=10, over 80 columns, RSS goes to 3.5 GB
and keeps rising. With them it stays at 2.54 GB, and a column takes the same time. RSS
reaches its peak in the reverse pass of a column, and falls between columns:

| D | chi | Peak RSS |
|---|---|---|
| 4 | 24 | 44 GB in 4 columns, 60 GB seen in 78 columns |
| 4 | 36 | 105 GB |
| 5 | 16 | 33 GB |
| 5 | 25 | 141 GB |
| 6 | 16 | 142 GB |

## References

- Ponsioen, Assaad & Corboz, *SciPost Phys.* **12**, 006 (2022) --
  AD excitations method
- Francuz et al., *Phys. Rev. Research* **7**, 013237 (2025) --
  Stable AD of CTM (custom SVD VJP, implicit differentiation, gauge fixing)
- Rader et al., arXiv:2511.09546 --
  Metric preconditioning (natural gradient) for iPEPS optimization
- Zhang, Yang & Corboz, arXiv:2505.00494 --
  Chi-ramping schedule for stable convergence at large bond dimensions
