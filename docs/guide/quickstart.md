# Quickstart

This page walks through three end-to-end examples: creating and contracting
tensors, running DMRG for a spin chain, and computing the 2D Ising free energy
with TRG.

## Example 1 -- Tensor creation and contraction

```python
import jax
import numpy as np
from tenax import (
    U1Symmetry, TensorIndex, FlowDirection,
    DenseTensor, contract,
)

# Create two dense tensors with labeled legs
sym = U1Symmetry()
charges = np.zeros(3, dtype=np.int32)

A = DenseTensor(
    jax.random.normal(jax.random.PRNGKey(0), (3, 4)),
    indices=(
        TensorIndex(sym, charges, FlowDirection.IN, label="i"),
        TensorIndex(sym, np.zeros(4, dtype=np.int32), FlowDirection.OUT, label="bond"),
    ),
)
B = DenseTensor(
    jax.random.normal(jax.random.PRNGKey(1), (4, 5)),
    indices=(
        TensorIndex(sym, np.zeros(4, dtype=np.int32), FlowDirection.IN, label="bond"),
        TensorIndex(sym, np.zeros(5, dtype=np.int32), FlowDirection.OUT, label="j"),
    ),
)

# Shared label "bond" is automatically contracted
C = contract(A, B)
print(C.labels())  # ('i', 'j')
print(C.todense().shape)  # (3, 5)
```

## Example 2 -- DMRG ground state of the Heisenberg chain

```python
import jax
from tenax import DMRGConfig, dmrg, build_mpo_heisenberg, FiniteMPS

L = 10  # 10-site chain
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
mps = FiniteMPS.random(L=L, d=2, chi=4, key=jax.random.PRNGKey(0))

config = DMRGConfig(
    max_bond_dim=32,
    num_sweeps=20,
    convergence_tol=1e-8,
    two_site=True,
    verbose=True,
)

result = dmrg(mpo, mps, config)
print(f"Ground state energy: {result.energy:.6f}")
print(f"Converged: {result.converged}")
```

## Example 3 -- 2D Ising free energy with TRG

```python
import math
from tenax import TRGConfig, trg, compute_ising_tensor, ising_free_energy_exact

beta_c = math.log(1 + math.sqrt(2)) / 2  # critical temperature
tensor = compute_ising_tensor(beta_c)

config = TRGConfig(max_bond_dim=16, num_steps=20)
log_Z_per_site = trg(tensor, config)

exact = ising_free_energy_exact(beta_c)
print(f"TRG  log(Z)/N = {float(log_Z_per_site):.6f}")
print(f"Exact         = {exact:.6f}")
```

## Next steps

- {doc}`core_concepts` -- symmetries, indices, and tensor types in depth
- {doc}`contraction` -- label-based contraction, SVD, and QR
- {doc}`tensor_networks` -- the `TensorNetwork` graph container
- Algorithm tutorials: {doc}`algorithms/dmrg`, {doc}`algorithms/trg`, {doc}`algorithms/hotrg`, {doc}`algorithms/ipeps`
