# iPEPS

Infinite Projected Entangled Pair States (iPEPS) is a variational ansatz for
2D quantum lattice models. Tenax implements the **simple update** for
optimisation and the **Corner Transfer Matrix (CTM)** method for computing
observables.

## Background

An iPEPS represents a 2D quantum state as a tensor network where each site has
a local tensor $A[u,d,l,r,s]$ with four virtual bonds and one physical index.
For translationally invariant states, a single-site unit cell suffices.

### Simple update

Fast imaginary time evolution:

1. For each nearest-neighbour bond, apply $\exp(-\delta\tau\, H_{\text{bond}})$.
2. SVD to restore tensor-product form; truncate to bond dimension $D$.
3. Update diagonal $\lambda$ matrices that approximate the bond environment.

### CTM environment

The Corner Transfer Matrix method approximates the infinite environment of a
PEPS site using 8 tensors (4 corners + 4 edges):

```
C1 --- T1 --- C2
|             |
T4    [A]    T2
|             |
C4 --- T3 --- C3
```

CTM iteratively absorbs rows and columns until the corner singular values
converge. The projectors used to truncate the enlarged corners are built
from an eigendecomposition (`eigh`) of the half-row/half-column density
matrices. For AD-based optimization, use `truncated_svd_ad` instead
(see {doc}`ad_excitations`).

## Configuration

```python
from tenax import iPEPSConfig, CTMConfig

ctm_config = CTMConfig(
    chi=20,              # CTM environment bond dimension
    max_iter=100,        # maximum CTM iterations
    conv_tol=1e-8,       # convergence tolerance on corner singular values
    renormalize=True,
    forward_gauge="phase",  # "phase" (default), "qr", "sigma", or "none"
)

config = iPEPSConfig(
    max_bond_dim=2,            # PEPS virtual bond dimension D
    num_imaginary_steps=100,   # imaginary time evolution steps
    dt=0.05,                   # time step size
    ctm=ctm_config,
    gate_order="sequential",
)
```

### Forward gauge

The ``forward_gauge`` option in ``CTMConfig`` controls how gauge ambiguity is
fixed after each CTM sweep during the forward pass. Four modes are supported:

| Value | Description |
|-------|-------------|
| ``"phase"`` (default) | variPEPS-style Frobenius normalization + phase fixing. Cheapest gauge fix that still stabilizes unrolled AD. **Recommended for both implicit and explicit AD** (1-site and 2-site). |
| ``"qr"`` | Legacy QR decomposition on each corner with sign-fixed diagonal. Fast and stable for simple update and forward-only CTM. |
| ``"sigma"`` | Transfer-matrix eigenvector alignment via power iteration. Required for element-wise convergence at large chi (1-site path). |
| ``"none"`` | No gauge fix. Diagnostic / benchmark mode only. |

**Forward gauge default**: ``forward_gauge`` defaults to ``"phase"`` (the
variPEPS-style Frobenius + phase fix), which is AD-correct for both the
implicit and explicit paths — the implicit-AD path in fact *requires*
``"phase"`` and validates it (``projector_method`` in ``("svd", "qr")``,
``forward_gauge="phase"``, ``ctm_conv_method="elementwise"``). There is **no
silent gauge promotion**: if you set ``forward_gauge="sigma"`` or ``"none"``
explicitly, that choice is respected as-is.

See {doc}`ipeps_ad_paths` for the complete post-PR-#291 recommended
configuration, benchmark results, and the split between the explicit-AD
and implicit-diff paths.

### Choosing `dt`

Larger time steps (`dt=0.1`–`0.3`) converge faster but can overshoot;
smaller steps (`dt=0.01`) are safer but need more iterations. A good
strategy is to start with `dt=0.1` for quick exploration and reduce it
for final production runs. The 2-site unit cell often benefits from
larger `dt` because the two independent tensors converge more slowly.

## Example -- 2D Heisenberg model

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, ipeps

# Heisenberg gate: H = Sz Sz + 0.5 (S+ S- + S- S+)
Sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=jnp.float32)
Sp = jnp.array([[0, 1], [0, 0]], dtype=jnp.float32)
Sm = jnp.array([[0, 0], [1, 0]], dtype=jnp.float32)
I2 = jnp.eye(2, dtype=jnp.float32)

H_bond = (
    jnp.kron(Sz, Sz)
    + 0.5 * jnp.kron(Sp, Sm)
    + 0.5 * jnp.kron(Sm, Sp)
).reshape(2, 2, 2, 2)

config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.01,
    ctm=CTMConfig(chi=10, max_iter=50),
)

energy, peps, env = ipeps(H_bond, initial_peps=None, config=config)
print(f"Energy per site: {energy:.6f}")
```

## Result

`ipeps()` returns a 3-tuple:

| Element | Type | Description |
|---------|------|-------------|
| `energy` | `float` | Energy per site |
| `peps` | `TensorNetwork` | Optimised PEPS (1x1 unit cell) |
| `env` | `CTMEnvironment` | Converged CTM environment tensors |

## CTMEnvironment

The `CTMEnvironment` named tuple contains the 8 environment tensors:

- **Corners** (`C1`, `C2`, `C3`, `C4`): shape `(chi, chi)`
- **Edges** (`T1`, `T2`, `T3`, `T4`): shape `(chi, D^2, chi)`

## Using CTM standalone

The `ctm()` function can be called independently to compute the
environment for an existing PEPS tensor:

```python
from tenax import ctm, CTMConfig

env = ctm(A_tensor, CTMConfig(chi=20, max_iter=100))
```

## 2-site checkerboard unit cell

A single-site unit cell cannot capture antiferromagnetic (Néel) order
because both sublattices share the same tensor. Setting
`unit_cell="2site"` in `iPEPSConfig` uses a 2-site checkerboard unit cell
with independent tensors $A$ (sublattice 0) and $B$ (sublattice 1).

On the checkerboard every neighbour of $A$ is $B$ and vice versa, which
is the minimal unit cell for Néel-ordered states.

### `ctm_2site()` -- standalone 2-site CTM

Compute CTM environments for an existing 2-site iPEPS:

```python
from tenax import ctm_2site, CTMConfig

env_A, env_B = ctm_2site(A, B, CTMConfig(chi=20, max_iter=100))
```

> **Note:** ``ctm_2site()`` is the legacy dense CTM used internally by
> simple update (``ipeps()``).  For AD-based optimization, use
> ``optimize_gs_ad()`` with ``unit_cell="2site"`` — it routes through
> the Tensor-protocol multisite CTM which supports both ``DenseTensor``
> and ``SymmetricTensor``.

| Argument | Type | Description |
|----------|------|-------------|
| `A` | `jax.Array` | Site tensor for sublattice A, shape `(D, D, D, D, d)` |
| `B` | `jax.Array` | Site tensor for sublattice B, shape `(D, D, D, D, d)` |
| `config` | `CTMConfig` | CTM configuration |

Returns a tuple `(env_A, env_B)` of `CTMEnvironment` named tuples.

### `compute_energy_ctm_2site()` -- 2-site energy

Compute the energy per site for a 2-site checkerboard iPEPS given
converged environments:

```python
from tenax import compute_energy_ctm_2site

energy = compute_energy_ctm_2site(A, B, env_A, env_B, H_bond, d=2)
```

The energy includes one horizontal and one vertical bond per site:
$E/\text{site} = E_h + E_v$.

### AD ground-state optimization

`optimize_gs_ad()` uses automatic differentiation through the CTM
fixed-point equation to compute exact gradients of the energy with
respect to the site tensor, then optimises with optax:

```python
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=20, max_iter=100),
    gs_optimizer="adam",
    gs_learning_rate=1e-3,
    gs_num_steps=200,
    gs_verbose=True,      # print optimization progress
    gs_log_interval=10,   # print every 10 AD steps
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, A_init=None, config=config)
```

Set ``gs_verbose=False`` (default) to disable console output.

#### Simple update initialization

Starting AD optimization from a random tensor can cause large gradients
and slow convergence.  Setting ``su_init=True`` runs simple update first
(using the ``num_imaginary_steps`` and ``dt`` already in the config) to
produce a physically reasonable starting point:

```python
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.01,
    ctm=CTMConfig(chi=20, max_iter=100),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, A_init=None, config=config)
```

When ``A_init`` is provided explicitly, ``su_init`` is ignored.

#### Optimizer selection

The AD optimizer is chosen via ``gs_optimizer`` in ``iPEPSConfig``:

| Optimizer | Setting | Best for |
|-----------|---------|----------|
| Adam | ``gs_optimizer="adam"`` (default) | Stable convergence, noisy gradients |
| L-BFGS | ``gs_optimizer="lbfgs"`` | Fast convergence near minimum |
| Conjugate gradient | ``gs_optimizer="cg"`` | Memory-efficient alternative to L-BFGS |

L-BFGS and CG use **Armijo backtracking line search** by default
(``gs_line_search=True``). Each trial step runs a fresh CTM convergence
to evaluate the energy, avoiding stale-environment artifacts.

```python
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_optimizer="lbfgs",
    gs_num_steps=30,
    gs_line_search_max_steps=8,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, None, config)
```

For Adam, a **cosine learning rate schedule** (lr → lr/10) is automatically
applied when ``gs_num_steps > 20``.

#### Explicit CTM differentiation

Set ``gs_implicit_ad=False`` to backpropagate through unrolled CTM
iterations instead of using implicit differentiation (the default
``gs_implicit_ad=True`` uses implicit diff). The forward pass runs
``gs_explicit_ad_warmup`` CTM sweeps without gradient tracking, then
``gs_explicit_ad_steps`` sweeps with full backpropagation.

```python
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50, projector_method="qr"),
    gs_implicit_ad=False,      # opt into explicit AD (default is implicit)
    gs_explicit_ad_steps=20,   # CTM steps with gradient tracking
    gs_explicit_ad_warmup=3,   # warmup steps (no gradient)
    gs_projector_method="qr",  # QR projectors scale cleanly to chi >= 16
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_num_steps=50,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, None, config)
```

Explicit AD is the **recommended** AD path on the 1-site C4v workflow
(``gs_c4v=True``). Each CTM sweep is a single move, the unrolled graph stays
manageable, and the backward pass avoids the implicit-diff linear solve
entirely.

```{note}
``forward_gauge`` defaults to ``"phase"`` (no promotion needed). Phase gauge
is 6–9× faster than sigma gauge with equal or better energy and is the
post-PR-#291 recommended gauge for both explicit and implicit AD. See
{doc}`ipeps_ad_paths` for the full benchmark table.
```

#### CTM convergence tolerance schedule

``iPEPSConfig.gs_ctm_conv_tol_schedule`` ramps the CTM convergence tolerance
from loose to tight across the AD optimization. It accepts a list of
``(step_fraction, conv_tol)`` pairs: at each AD step the optimizer looks up
the tolerance corresponding to the current ``step_index / gs_num_steps``
fraction and rebuilds the CTM config accordingly.

```python
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=80, conv_tol=1e-7),
    gs_num_steps=50,
    gs_ctm_conv_tol_schedule=[(0.0, 1e-5), (0.5, 1e-6), (0.8, 1e-7)],
)
```

This is an advanced tuning knob — leaving it at ``None`` (the default) uses
``ctm.conv_tol`` throughout, which is fine for most runs.

#### Chi-ramping schedule

Starting AD optimization at a large chi can be slow and unstable.
``optimize_gs_ad_chi_schedule`` runs optimization at progressively
increasing chi values, using the converged tensor from each level to
warm-start the next:

```python
from tenax import optimize_gs_ad_chi_schedule, iPEPSConfig, CTMConfig

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=8, projector_method="qr"),
    gs_projector_method="qr",
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_c4v=True,
    su_init=True,
)

# (chi, num_steps) pairs — chi and gs_num_steps are overridden per stage
A_opt, env, E_gs = optimize_gs_ad_chi_schedule(
    H_bond, None, config, [(8, 30), (16, 20)]
)
```

This follows the approach of Zhang, Yang & Corboz (arXiv:2505.00494).
The base ``config`` provides all other settings (optimizer, line search,
metric preconditioning, etc.); only ``chi`` and ``gs_num_steps`` change
per stage.

#### Backward method selection

The backward pass for CTM implicit differentiation (``gs_implicit_ad=True``)
has two options:

| Method | Setting | Description |
|--------|---------|-------------|
| Iterative VJP | ``ad_backward_method="vjp"`` (default) | Neumann series accumulation of VJP (YASTN-style). The regression-covered backward for the implicit path. |
| GMRES | ``ad_backward_method="gmres"`` | Direct linear solve of ``(I - J^T) λ = g``. **Experimental / documented unstable** — the GMRES backward is currently tracked as an open gap and its regression test is marked ``xfail`` (see issue #292). |

**To avoid the implicit-diff linear solve entirely**: set
``gs_implicit_ad=False`` (explicit AD is opt-in; the default
``gs_implicit_ad=True`` uses implicit diff). Explicit AD does not use the
``(I - J^T)`` solve at all and is the fastest path on the 1-site C4v
workflow. If you use the implicit path, prefer ``ad_backward_method="vjp"``
(the default) until the GMRES backward is stabilized.

```python
# Explicit-AD configuration — explicit AD + QR projectors + phase gauge (default)
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=100, projector_method="qr"),
    gs_implicit_ad=False,  # opt into explicit AD (default is implicit)
    gs_projector_method="qr",
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_metric_precond=True,
    gs_c4v=True,
    gs_num_steps=100,
    su_init=True,
)
```

For AD-based excitation spectra on top of an optimised iPEPS, see
{doc}`ad_excitations`. For the full benchmarked recommendation (including
when to reach for ``forward_gauge="sigma"``, ``forward_gauge="none"``, or
the ``gs_ctm_conv_tol_schedule`` knob) see {doc}`ipeps_ad_paths`.

## Split-CTMRG with Tensor protocol

The `ctm_split_tensor()` function provides a polymorphic split-CTMRG that
works with both `DenseTensor` and `SymmetricTensor` iPEPS site tensors.
It uses `bar()` (conjugate + flip flows, no charge dual) for the bra layer
instead of `dagger()`, which ensures correct physical-trace block matching
for nontrivial U(1) or fermionic charges.

```python
from tenax import ctm_split_tensor, compute_energy_split_ctm_tensor

env = ctm_split_tensor(A, chi=20, max_iter=100, chi_I=10)
E = compute_energy_split_ctm_tensor(A, env, H_bond, d=2)
```

`A` can be either a `DenseTensor` or `SymmetricTensor` with 5 legs
`(u, d, l, r, phys)`.

## Fermionic iPEPS (fPEPS)

Tenax supports fermionic PEPS using `SymmetricTensor` with `FermionParity`
symmetry. All contractions and decompositions automatically handle Koszul
signs (fermionic anticommutation).

### Spinless fermion example

```python
import jax
from tenax import FPEPSConfig, spinless_fermion_gate, fpeps, sublattice_gap

config = FPEPSConfig(D=2, ctm_chi=8, num_imaginary_steps=200, dt=0.05, V=4.0)
gate = spinless_fermion_gate(config)
energy, (A, B), (env_A, env_B) = fpeps(gate, config, key=jax.random.PRNGKey(0))
print(f"Energy per site: {energy:.6f}")
print(f"CDW gap: {sublattice_gap(A, B, env_A, env_B):.4f}")
```

The `spinless_fermion_gate()` builds $H = -t \sum (c^\dagger_i c_j + \text{h.c.}) + V \sum n_i n_j$
as a `SymmetricTensor` with `FermionParity` charges. The simple update uses
`contract()` and `svd()` which automatically compute Koszul
signs at every leg crossing.

The state and environment are **pairs** (#878): the t-V ground state at finite
`V` is a checkerboard charge-density wave, which is inherently two-site. See
[Fermionic iPEPS (fPEPS)](fpeps.md) for `sublattice_gap`, the warm-restart form,
and the two standing caveats (seed dependence, and #392's uncertified energy).
