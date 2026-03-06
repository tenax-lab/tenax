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
    chi=20,          # CTM environment bond dimension
    max_iter=100,    # maximum CTM iterations
    conv_tol=1e-8,   # convergence tolerance on corner singular values
    renormalize=True,
)

config = iPEPSConfig(
    max_bond_dim=2,            # PEPS virtual bond dimension D
    num_imaginary_steps=100,   # imaginary time evolution steps
    dt=0.05,                   # time step size
    ctm=ctm_config,
    gate_order="sequential",
)
```

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
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, A_init=None, config=config)
```

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

For AD-based excitation spectra on top of an optimised iPEPS, see
{doc}`ad_excitations`.

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
from tenax import FPEPSConfig, spinless_fermion_gate, fpeps

config = FPEPSConfig(D=2, ctm_chi=8, num_imaginary_steps=200, dt=0.01)
gate = spinless_fermion_gate(config)
energy, A_opt, env = fpeps(gate, config, key=jax.random.PRNGKey(0))
print(f"Energy per site: {energy:.6f}")
```

The `spinless_fermion_gate()` builds $H = -t \sum (c^\dagger_i c_j + \text{h.c.}) + V \sum n_i n_j$
as a `SymmetricTensor` with `FermionParity` charges. The simple update uses
`contract()` and `svd()` which automatically compute Koszul
signs at every leg crossing. Energy is evaluated via dense CTM fallback.
