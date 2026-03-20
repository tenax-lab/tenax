# AutoMPO

`AutoMPO` builds Matrix Product Operators from symbolic Hamiltonian
descriptions, enabling DMRG for arbitrary 1D Hamiltonians without manually
constructing MPO tensors.

## Background

AutoMPO uses a finite-automaton (left-partial-state) approach. Each
Hamiltonian term is a product of local operators at specified sites. At each
internal bond, in-flight terms that span across the bond receive a dedicated
MPO state index. The resulting MPO bond dimension equals
(number of in-flight terms) + 2 (done + vacuum states).

An optional SVD compression pass can reduce the bond dimension toward the
optimal value.

## Class-based API

```python
import jax
from tenax import AutoMPO, dmrg, DMRGConfig, FiniteMPS

L = 10
auto = AutoMPO(L=L, d=2)  # spin-1/2, d=2

# Heisenberg model
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)

# Add a magnetic field
for i in range(L):
    auto += (0.1, "Sz", i)

mpo = auto.to_mpo()
print(f"MPO bond dimensions: {auto.bond_dims()}")
print(f"Number of terms: {auto.n_terms()}")

# Use with DMRG
mps = FiniteMPS.random(L=L, d=2, chi=4, key=jax.random.PRNGKey(0))
result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=32))
```

### Adding terms

Terms are added with `add_term(coeff, op_name, site, ...)` or the `+=`
shorthand:

```python
# Single-site term
auto.add_term(0.5, "Sz", 0)

# Two-site term
auto.add_term(1.0, "Sz", 0, "Sz", 1)

# Three-body term
auto.add_term(0.1, "Sz", 0, "Sz", 1, "Sz", 2)

# Shorthand with +=
auto += (1.0, "Sp", 3, "Sm", 4)
```

### MPO compression

For Hamiltonians with many terms or long-range interactions, enable SVD
compression to reduce the MPO bond dimension:

```python
mpo = auto.to_mpo(compress=True, compress_tol=1e-12)
```

## Functional API

`build_auto_mpo` provides a one-call interface:

```python
from tenax import build_auto_mpo

L = 10
terms = (
    [(1.0, "Sz", i, "Sz", i + 1) for i in range(L - 1)]
    + [(0.5, "Sp", i, "Sm", i + 1) for i in range(L - 1)]
    + [(0.5, "Sm", i, "Sp", i + 1) for i in range(L - 1)]
)

mpo = build_auto_mpo(terms, L=L, d=2)
```

## Built-in operators

Tenax provides standard operator sets:

### `spin_half_ops()` (d=2)

| Key | Operator |
|-----|----------|
| `"Sz"` | $\frac{1}{2}\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ |
| `"Sp"` | $\begin{pmatrix}0&1\\0&0\end{pmatrix}$ |
| `"Sm"` | $\begin{pmatrix}0&0\\1&0\end{pmatrix}$ |
| `"Id"` | $\begin{pmatrix}1&0\\0&1\end{pmatrix}$ |

### `spin_one_ops()` (d=3)

Standard spin-1 operators in the $|m=+1\rangle, |m=0\rangle, |m=-1\rangle$
basis.

### Custom operators

Pass a `site_ops` dictionary to use your own operators:

```python
import numpy as np

my_ops = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.float64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.float64),
    "Id": np.eye(2, dtype=np.float64),
}

auto = AutoMPO(L=6, d=2, site_ops=my_ops)
for i in range(5):
    auto += (1.0, "X", i, "X", i + 1)
    auto += (0.5, "Z", i)

mpo = auto.to_mpo()
```

## HamiltonianTerm

Each term is stored as a frozen `HamiltonianTerm` dataclass:

- `coefficient`: complex scalar prefactor
- `ops`: tuple of `(site, operator_matrix)` pairs, sorted by site
