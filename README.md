# Tenax

A JAX-based tensor network library with symmetry-aware block-sparse tensors and label-based contraction.

The name **Tenax** combines **Ten**sor network + J**ax**, and is also Latin for "holding fast" — reflecting how tensor networks bind indices together through contraction.

## Features

- **Block-sparse symmetric tensors** — only symmetry-allowed charge sectors stored (U(1), Z_n)
- **Label-based contraction** — legs are identified by string/integer labels; shared labels are automatically contracted (Cytnx-style)
- **opt_einsum integration** — optimal contraction path finding for multi-tensor contractions
- **Network class** — graph-based tensor network container with contraction caching
- **`.net` file support** — cytnx-style declarative network topology; parse once, load tensors, contract repeatedly (template pattern)
- **Algorithms** — DMRG, iDMRG (1D chain & infinite cylinder), TRG, HOTRG, iPEPS (simple update with 1-site or 2-site unit cell & AD optimization), fermionic iPEPS (fPEPS), quasiparticle excitations
- **AutoMPO** — build Hamiltonian MPOs from symbolic operator descriptions (custom couplings, NNN, arbitrary spin); supports `symmetric=True` for U(1) block-sparse MPOs
- **AD-based iPEPS optimization** — gradient optimization via implicit differentiation through CTM fixed point, supporting 1-site and 2-site unit cells (Francuz et al. PRR 7, 013237)
- **QR-based CTMRG projectors** — optional QR projectors for faster CTM convergence (replaces expensive `eigh`)
- **Split-CTMRG** — ket/bra-separated CTM environment tensors for O(χ³D³) projector cost instead of O(χ³D⁶) (Rader & Läuchli, arXiv:2502.10298)
- **Quasiparticle excitations** — iPEPS excitation spectra at arbitrary Brillouin-zone momenta (Ponsioen et al. 2022)
- **Polymorphic tensor arithmetic** — `+`, `-`, `*`, `-T`, `max_abs`, `inner()` work identically on `DenseTensor` and `SymmetricTensor`, enabling algorithm code that is agnostic to the underlying storage
- **Block-sparse SVD and QR** — native symmetry-aware decompositions for `SymmetricTensor`
- **Extensible symmetry system** — non-Abelian symmetry interface for future SU(2) support
- **Benchmark suite** — CLI-driven performance benchmarks for all algorithms across CPU, CUDA, TPU, and Metal backends

## Installation

```bash
# CPU only (default)
pip install tenax-tn

# NVIDIA GPU (CUDA 13)
pip install tenax-tn[cuda13]

# NVIDIA GPU (CUDA 12)
pip install tenax-tn[cuda12]

# Google Cloud TPU
pip install tenax-tn[tpu]

# Apple Silicon GPU (macOS, experimental)
pip install tenax-tn[metal]
```

For development:

```bash
git clone https://github.com/tenax-lab/tenax.git
cd tenax
uv sync --all-extras --dev
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import numpy as np
from tenax import (
    U1Symmetry, TensorIndex, FlowDirection,
    SymmetricTensor, TensorNetwork, contract
)

# Define U(1) symmetric tensor indices with named legs
u1 = U1Symmetry()
phys_charges = np.array([-1, 1], dtype=np.int32)
bond_charges = np.array([-1, 0, 1], dtype=np.int32)
key = jax.random.PRNGKey(0)

A = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, phys_charges, FlowDirection.IN,  label="p0"),
        TensorIndex(u1, bond_charges, FlowDirection.IN,  label="left"),
        TensorIndex(u1, bond_charges, FlowDirection.OUT, label="bond"),
    ),
    key=key,
)
B = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, phys_charges, FlowDirection.IN,  label="p1"),
        TensorIndex(u1, bond_charges, FlowDirection.IN,  label="bond"),  # shared label
        TensorIndex(u1, bond_charges, FlowDirection.OUT, label="right"),
    ),
    key=jax.random.PRNGKey(1),
)

# Contract by matching shared labels — "bond" is summed over automatically
result = contract(A, B)
print(result.labels())  # ('p0', 'left', 'p1', 'right')

# Build a tensor network and contract
tn = TensorNetwork()
tn.add_node("A", A)
tn.add_node("B", B)
tn.connect_by_shared_label("A", "B")
result = tn.contract()
```

## Network Blueprint (`.net` file) Example

```python
from tenax import NetworkBlueprint

# Define network topology as a string (or read from a .net file)
bp = NetworkBlueprint("""
L: a, b, c
M: a, p, q, d
A: b, p, s, e
M2: e, q, t, f
R: d, f, g
TOUT: c, s, t, g
""")

# Load tensors (can be DenseTensor or SymmetricTensor)
bp.put_tensors({"L": L, "M": M, "A": A, "M2": M2, "R": R})
result = bp.launch()  # contracts the full network

# Reuse with different tensors (e.g. in a DMRG sweep)
bp.put_tensor("A", new_A)
result2 = bp.launch()
```

## DMRG Example

```python
from tenax.algorithms.dmrg import dmrg, build_mpo_heisenberg, DMRGConfig
from tenax.network.network import build_mps

L = 10  # chain length
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)

# Build random initial MPS
# ...

config = DMRGConfig(max_bond_dim=50, num_sweeps=10)
result = dmrg(mpo, initial_mps, config)
print(f"Ground state energy: {result.energy:.8f}")
```

## 2D Cylinder DMRG Example

```python
from tenax import AutoMPO, DMRGConfig, build_random_mps, dmrg

# Build Heisenberg Hamiltonian on a 6x3 cylinder via AutoMPO
Lx, Ly, N = 6, 3, 18
auto = AutoMPO(L=N, d=2)
for x in range(Lx):
    for y in range(Ly):
        # Within-ring bond (periodic y)
        i, j = x * Ly + y, x * Ly + (y + 1) % Ly
        auto += (1.0, "Sz", min(i,j), "Sz", max(i,j))
        auto += (0.5, "Sp", min(i,j), "Sm", max(i,j))
        auto += (0.5, "Sm", min(i,j), "Sp", max(i,j))
        # Between-ring bond (open x)
        if x < Lx - 1:
            i, j = x * Ly + y, (x + 1) * Ly + y
            auto += (1.0, "Sz", i, "Sz", j)
            auto += (0.5, "Sp", i, "Sm", j)
            auto += (0.5, "Sm", i, "Sp", j)

mpo = auto.to_mpo(compress=True)
mps = build_random_mps(N, physical_dim=2, bond_dim=16)
config = DMRGConfig(max_bond_dim=100, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)
print(f"E/N = {result.energy / N:.8f}")  # converges in a few sweeps
```

See `examples/heisenberg_cylinder.py` for a full working example with
4x2, 6x3, and 8x4 cylinders.

## iDMRG Example

```python
from tenax import idmrg, build_bulk_mpo_heisenberg, iDMRGConfig

W = build_bulk_mpo_heisenberg(Jz=1.0, Jxy=1.0)
config = iDMRGConfig(max_bond_dim=32, max_iterations=100, convergence_tol=1e-8)
result = idmrg(W, config)
print(f"Energy per site: {result.energy_per_site:.6f}")  # ~ -0.4431
print(f"Converged: {result.converged}")
```

## Infinite Cylinder iDMRG Example

```python
from tenax import build_bulk_mpo_heisenberg_cylinder, iDMRGConfig, idmrg

# Ly=4 cylinder: each super-site is a ring of 4 spins (d=16, D_w=14)
# Only even Ly is supported (odd Ly frustrates AFM order).
W = build_bulk_mpo_heisenberg_cylinder(Ly=4)
config = iDMRGConfig(max_bond_dim=200, max_iterations=200, convergence_tol=1e-4)
result = idmrg(W, config, d=16)
e_per_spin = result.energy_per_site / 4
print(f"Energy per spin: {e_per_spin:.6f}")
```

See `examples/heisenberg_infinite_cylinder.py` for Ly=2 and Ly=4 cylinders
with ED cross-checks.

## TRG Example

```python
from tenax import TRGConfig, trg, compute_ising_tensor, ising_free_energy_exact

beta = 0.44  # near critical temperature
T = compute_ising_tensor(beta)

config = TRGConfig(max_bond_dim=16, num_steps=20)
log_z_per_n = trg(T, config)
f_trg = float(-log_z_per_n / beta)
f_exact = ising_free_energy_exact(beta)
print(f"TRG:   {f_trg:.8f}")
print(f"Exact: {f_exact:.8f}")
```

See `examples/ising_trg.py` and `examples/ising_hotrg.py` for full TRG and HOTRG
examples at multiple temperatures compared against the Onsager exact solution.

## AutoMPO Example

```python
from tenax import AutoMPO, build_auto_mpo

# Class-based interface: build a Heisenberg chain
L = 10
auto = AutoMPO(L)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()

# Or use the functional interface with custom operators
import numpy as np
custom_ops = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "Id": np.eye(2),
}
terms = [(1.0, "Z", i, "Z", i + 1) for i in range(L - 1)]
terms += [(0.5, "X", i) for i in range(L)]
mpo = build_auto_mpo(terms, L=L, site_ops=custom_ops)

# Build a symmetric (U(1) block-sparse) MPO
mpo_sym = auto.to_mpo(symmetric=True)
```

## iPEPS Simple Update (2-site unit cell)

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, ipeps

# Build a 2-site Heisenberg gate
Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))

# 2-site checkerboard iPEPS — captures Neel order
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.3,
    ctm=CTMConfig(chi=10, max_iter=40),
    unit_cell="2site",
)
energy, peps, (env_A, env_B) = ipeps(gate, None, config)
print(f"Energy per site: {energy:.6f}")  # ~ -0.65
```

See `examples/heisenberg_ipeps_su.py` for 1-site and 2-site unit cell examples.

## iPEPS AD Optimization and Excitations

```python
import jax.numpy as jnp
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad,
    ExcitationConfig, compute_excitations, make_momentum_path,
)

# Build a 2-site Heisenberg gate
Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))

# AD ground-state optimization (Francuz et al. PRR 7, 013237)
# su_init=True runs simple update first for a better starting tensor
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
print(f"Ground-state energy: {E_gs:.6f}")

# 2-site AD optimization for antiferromagnets (Néel order)
config_2site = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    unit_cell="2site",
    su_init=True,
)
(A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(gate, None, config_2site)

# Use QR projectors for faster CTM convergence
config_qr = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50, projector_method="qr"),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config_qr)

# Quasiparticle excitations (Ponsioen et al. 2022)
momenta = make_momentum_path("brillouin", num_points=20)
exc_config = ExcitationConfig(num_excitations=3)
result = compute_excitations(A_opt, env, gate, E_gs, momenta, exc_config)
print(result.energies.shape)  # (20, 3)
```

See `examples/heisenberg_ipeps_ad.py` for AD optimization with random vs simple
update initialization, and `examples/heisenberg_ipeps_excitations.py` for the
full excitation spectrum along Gamma-X-M-Gamma.

## Split-CTMRG

```python
from tenax import CTMConfig, ctm_split, compute_energy_split_ctm

# Split-CTMRG keeps ket/bra layers separate for O(χ³D³) projector cost
# instead of O(χ³D⁶) — significant speedup at large bond dimension D
config = CTMConfig(chi=20, max_iter=100, chi_I=10)
env = ctm_split(A, config)
E = compute_energy_split_ctm(A, env, gate, d=2)
```

## Examples

Runnable example scripts are in the `examples/` directory:

| Script | Algorithm | Model |
|--------|-----------|-------|
| `heisenberg_cylinder.py` | DMRG | Heisenberg on 4x2, 6x3, 8x4 cylinders |
| `heisenberg_infinite_cylinder.py` | iDMRG | Heisenberg on infinite Ly=2, Ly=4 cylinders |
| `heisenberg_ipeps_su.py` | iPEPS simple update | Heisenberg (1x1 and 2-site unit cells) |
| `heisenberg_ipeps_ad.py` | iPEPS AD optimization | Heisenberg (random vs SU init) |
| `heisenberg_ipeps_excitations.py` | iPEPS excitations | Heisenberg dispersion along Γ-X-M-Γ |
| `spinless_fermion_fpeps.py` | fPEPS simple update | Spinless fermions (free and interacting) |
| `ising_trg.py` | TRG | 2D Ising vs Onsager exact |
| `ising_hotrg.py` | HOTRG | 2D Ising vs Onsager exact |

Run any example with:

```bash
uv run python examples/<script>.py
```

## Symmetry System

```python
from tenax import U1Symmetry, ZnSymmetry
import numpy as np

# U(1): integer charges, fusion by addition
u1 = U1Symmetry()
charges = np.array([-1, 0, 1], dtype=np.int32)
print(u1.fuse(charges, charges))  # [-2, 0, 2]
print(u1.dual(charges))           # [1, 0, -1]

# Z_3: charges mod 3
z3 = ZnSymmetry(3)
print(z3.fuse(np.array([1, 2], dtype=np.int32),
              np.array([2, 2], dtype=np.int32)))  # [0, 1]
```

## Gotchas

### Float64 precision and `JAX_ENABLE_X64`

Tenax defaults to `float64` for all tensors and algorithms. Importing
`tenax` automatically calls `jax.config.update("jax_enable_x64", True)`,
so 64-bit arithmetic is enabled out of the box.

If you import JAX *before* `tenax` and create arrays in that window, they
will still be `float32`. To avoid surprises, either import `tenax` first or
enable x64 manually:

```python
import jax
jax.config.update("jax_enable_x64", True)

import tenax
```

### MPO index convention

The MPO W-tensor uses the convention `W[w_l, ket, bra, w_r]` — the two
middle indices are physical (ket on top, bra on bottom) and the outer
indices are bond dimensions.

### NumPy >= 2.0 casting

Adding a Python `complex` scalar (even `1+0j`) into a `float64` array
raises `UFuncOutputCastingError` under NumPy >= 2.0. Use `.real` or an
explicit `complex128` dtype instead.

### Local test failures on macOS x86_64

`uv run pytest` may fail on macOS x86_64 if jaxlib has no wheel for that
platform.
## Benchmarks

A CLI-driven benchmark suite measures wall-clock performance of every algorithm
across hardware backends.

```bash
# Quick smoke test (TRG, small size, 1 trial)
python -m benchmarks.run --backend cpu --algorithm trg --size small --trials 1

# Full CPU baseline
python -m benchmarks.run --backend cpu -o benchmarks/results/cpu_baseline.json

# GPU comparison
python -m benchmarks.run --backend cuda -o benchmarks/results/cuda.json

# Specific algorithms and sizes
python -m benchmarks.run -b cpu -a dmrg idmrg -s small medium -n 5

# CSV output for analysis
python -m benchmarks.run -b cpu -a all -s all --csv results.csv

# Show available backends
python -m benchmarks.run --list-backends
```

Each run prints a summary table and saves full results (timings, parameters,
device info) to JSON. See `docs/guide/benchmarks.md` for the complete guide.

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/tenax-lab/tenax.git
cd tenax
uv sync --all-extras --dev

# Install pre-commit hooks (ruff lint + format on every commit)
uv run pre-commit install

# Run tests
uv run pytest -m core          # fast core tests only
uv run pytest -m "not slow"    # skip expensive tests
uv run pytest                  # full suite

# Lint
uv run ruff check src/ tests/
```

Work-in-progress design documents live in `design/`.

## Documentation

Full API documentation is built with Sphinx:

```bash
cd docs && make html
```

The generated HTML is in `docs/_build/html/`.

## References

- H.-J. Liao, J.-G. Liu, L. Wang, T. Xiang, *Phys. Rev. X* **9**, 031041 (2019) — AD-based iPEPS ground-state optimization
- A. Francuz, N. Schuch, B. Vanhecke, *PRR* **7**, 013237 (2025) — Stable AD through CTM (SVD regularization, truncation correction, implicit differentiation)
- L. Ponsioen, F. F. Assaad, P. Corboz, *SciPost Phys.* **12**, 006 (2022) — Quasiparticle excitations for iPEPS
- M. Rader, A. M. Läuchli, arXiv:2502.10298 (2025) — Split-CTMRG with factored projectors for efficient iPEPS environments

## License

MIT
