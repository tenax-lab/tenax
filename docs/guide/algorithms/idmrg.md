# iDMRG

Infinite DMRG (iDMRG) finds the ground-state energy per site of a
translationally invariant Hamiltonian directly in the thermodynamic limit.

## Background

iDMRG works by repeatedly inserting two new sites into the centre of a
growing chain and optimising them with a Lanczos eigensolver. After each
growth step the centre bond is truncated via SVD to a maximum bond
dimension $\chi$. The energy-per-site estimate converges as the effective
environment builds up around the unit cell.

Key properties of the Tenax implementation:

- **Sweep-based algorithm** with environment warmup for self-consistent
  infinite boundary conditions.
- **Transfer matrix fixed-point environments** — computes the dominant
  eigenvector of the MPS transfer matrix for converged left/right
  environments, replacing the growing-chain approach.
- The bulk Hamiltonian is specified as a single repeated MPO tensor
  $W[w_l, \text{ket}, \text{bra}, w_r]$.
- Cython BLAS fast path for block-sparse contractions when available.
- Built-in MPO constructors for the Heisenberg chain and cylinder.

## Configuration

```python
from tenax import iDMRGConfig

config = iDMRGConfig(
    max_bond_dim=100,       # maximum MPS bond dimension (chi)
    max_iterations=200,     # maximum number of growth steps
    convergence_tol=1e-8,   # stop when |dE/site| < tol
    lanczos_max_iter=50,    # Lanczos iteration cap
    lanczos_tol=1e-12,      # Lanczos convergence tolerance
    svd_trunc_err=None,     # SVD truncation error (None = use max_bond_dim)
    verbose=True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_bond_dim` | 100 | Maximum MPS bond dimension $\chi$ |
| `max_iterations` | 200 | Maximum 2-site growth steps |
| `convergence_tol` | 1e-8 | Energy-per-site convergence threshold |
| `lanczos_max_iter` | 50 | Maximum Lanczos iterations |
| `lanczos_tol` | 1e-12 | Lanczos convergence tolerance |
| `svd_trunc_err` | None | Max SVD truncation error (overrides `max_bond_dim` if set) |
| `verbose` | False | Print per-step diagnostics |

## Example -- Heisenberg chain

```python
from tenax import iDMRGConfig, idmrg, build_bulk_mpo_heisenberg

# Build the bulk W-tensor for the spin-1/2 XXZ Heisenberg model
bulk_mpo = build_bulk_mpo_heisenberg(Jz=1.0, Jxy=1.0, hz=0.0)

config = iDMRGConfig(max_bond_dim=64, max_iterations=200, verbose=True)
result = idmrg(bulk_mpo, config)

print(f"Energy/site: {result.energy_per_site:.10f}")
print(f"Converged:   {result.converged}")
print(f"Steps:       {len(result.energies_per_step)}")
# Reference: E/site ≈ 0.25 - ln(2) ≈ −0.4431471805
```

`build_bulk_mpo_heisenberg` returns a `DenseTensor` with legs
`("w_l", "mpo_top", "mpo_bot", "w_r")` and shape `(5, 2, 2, 5)`.

## Example -- Infinite cylinder

`build_bulk_mpo_heisenberg_cylinder` constructs a bulk MPO for a
Heisenberg model on an infinite cylinder of circumference $L_y$. Each
"super-site" encodes an entire ring of $L_y$ spins (physical dimension
$d = 2^{L_y}$).

```python
from tenax import iDMRGConfig, idmrg, build_bulk_mpo_heisenberg_cylinder

Ly = 4
bulk_mpo = build_bulk_mpo_heisenberg_cylinder(Ly=Ly, J=1.0)

config = iDMRGConfig(max_bond_dim=200, max_iterations=300)
result = idmrg(bulk_mpo, config)

print(f"E/site (Ly={Ly}): {result.energy_per_site / Ly:.8f}")
```

```{note}
`build_bulk_mpo_heisenberg_cylinder` only accepts **even** $L_y$.
Odd circumference frustrates antiferromagnetic order on the square
lattice.
```

## Result object

`idmrg()` returns an `iDMRGResult` named tuple:

| Field | Type | Description |
|-------|------|-------------|
| `energy_per_site` | `float` | Converged energy per site |
| `energies_per_step` | `list[float]` | Energy-per-site estimate at each iteration |
| `mps_tensors` | `list[Tensor]` | 2-site unit cell `[A_L, A_R]` |
| `singular_values` | `jax.Array` | Singular values on the centre bond |
| `converged` | `bool` | Whether the run converged within tolerance |

## MPO construction

`build_bulk_mpo_heisenberg` builds a single bulk $W$-matrix (shape
$5 \times d \times d \times 5$) for the spin-1/2 XXZ Heisenberg model:

$$H = J_z \sum_i S^z_i S^z_{i+1} + \frac{J_{xy}}{2} \sum_i (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) + h_z \sum_i S^z_i$$

For finite chains, use {func}`~tenax.build_mpo_heisenberg` instead (see
{doc}`dmrg`). For custom Hamiltonians, see {doc}`auto_mpo`.
