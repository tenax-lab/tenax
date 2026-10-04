# Tenax

[Website](https://tenax-lab.github.io) | [Docs](https://tenax.readthedocs.io) | [PyPI](https://pypi.org/project/tenax-tn/)

A JAX-based tensor network library with symmetry-aware block-sparse tensors and label-based contraction.

The name **Tenax** combines **Ten**sor network + J**ax**, and is also Latin for "holding fast" — reflecting how tensor networks bind indices together through contraction.

> **Experimental project** — This library is under active development and largely written with the assistance of Claude Code (AI). While we test extensively, AI-generated code can contain subtle bugs. Please verify results against known benchmarks before using them in research. Bug reports and contributions are welcome.

## Features

Each entry links to its guide page; the full documentation is at [tenax.readthedocs.io](https://tenax.readthedocs.io).

**Tensors and contraction**

- **Block-sparse symmetric tensors** — only allowed charge sectors stored: U(1), Z_n, products, fermion parity ([symmetry](docs/guide/symmetry.md))
- **Label-based contraction** — shared labels contract automatically (Cytnx-style), with opt_einsum path finding ([contraction](docs/guide/contraction.md))
- **Block-sparse SVD, QR and eigh** — native symmetry-aware decompositions in `tenax.linalg` ([contraction](docs/guide/contraction.md))
- **Polymorphic arithmetic** — the same algorithm code runs on `DenseTensor` and `SymmetricTensor` ([core concepts](docs/guide/core_concepts.md))
- **Tensor networks and `.net` files** — cached graph container and declarative topologies ([tensor networks](docs/guide/tensor_networks.md))

**1D algorithms**

- **DMRG** — finite DMRG with a Cython BLAS CPU path and JIT / multi-GPU sharded sweeps ([DMRG](docs/guide/algorithms/dmrg.md))
- **iDMRG** — infinite chains and infinite cylinders ([iDMRG](docs/guide/algorithms/idmrg.md))
- **TDVP and iTEBD** — time evolution; iTEBD with the inversion-free Hastings update ([TDVP](docs/guide/algorithms/tdvp.md), [capabilities](docs/guide/capabilities.md))
- **AutoMPO** — Hamiltonian MPOs from symbolic operator terms, dense or U(1) block-sparse ([AutoMPO](docs/guide/algorithms/auto_mpo.md))

**2D quantum algorithms**

- **iPEPS simple update** — 1-site and 2-site unit cells, with a belief-propagation gauge for the bond weights ([iPEPS](docs/guide/algorithms/ipeps.md))
- **AD ground-state optimization** — implicit, explicit, C4v and root-implicit AD through CTM; an unconverged CTM forward raises `CTMNotConvergedError` (or warns with `CTMNotConvergedWarning`) on the fused 1x1 and 2-site paths ([AD paths](docs/guide/algorithms/ipeps_ad_paths.md))
- **CTM environments** — SVD/eigh/QR projectors, in-CTM χ growth, convergence and saddle checks, optional mixing for two-state cycles, split-CTMRG ([CTM](docs/guide/algorithms/ctm.md))
- **Fermionic iPEPS (fPEPS)** — graded tensors for spinless fermions / the t-V model ([fPEPS](docs/guide/algorithms/fpeps.md))
- **Honeycomb and kagome** — native honeycomb CTM and kagome iPESS with AD ([guide](docs/guide/algorithms/honeycomb_kagome.md))
- **Quasiparticle excitations** — iPEPS excitation spectra at arbitrary momenta ([AD excitations](docs/guide/algorithms/ad_excitations.md))

**Classical statistical mechanics**

- **TRG and Gilt-TNR** — coarse-graining of 2D partition functions ([TRG](docs/guide/algorithms/trg.md))
- **HOTRG and Gilt-HOTRG** — with multi-GPU sharding and the q-state Potts model ([HOTRG](docs/guide/algorithms/hotrg.md))

**Tooling**

- **Hardware backends and benchmarks** — CPU, CUDA, TPU and Metal via JAX; CLI benchmark suite ([benchmarks](docs/guide/benchmarks.md))

See [capabilities](docs/guide/capabilities.md) for a map of what each path is (and is not) the right tool for.

## Installation

> **Note:** The PyPI package (`tenax-tn`) is not yet available. Install from source using the instructions below.

```bash
git clone https://github.com/tenax-lab/tenax.git
cd tenax

# With uv (recommended)
uv sync --all-extras --dev

# Or with pip
pip install -e .
```

### Hardware acceleration

Tenax uses JAX as its backend. To enable GPU or TPU acceleration, install
the appropriate JAX variant **before** installing Tenax:

```bash
# NVIDIA GPU (CUDA 13, recommended)
pip install -U "jax[cuda13]"

# NVIDIA GPU (CUDA 12)
pip install -U "jax[cuda12]"

# Google Cloud TPU
pip install -U "jax[tpu]"

# Apple Silicon GPU (macOS only, experimental)
pip install jax-metal
```

See the [installation guide](docs/guide/installation.md) and the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
more accelerator options. Importing `tenax` enables float64; see
[gotchas](docs/guide/gotchas.md) if you import JAX first.

## Quick Start

```python
import jax
import jax.numpy as jnp
import numpy as np
from tenax import (
    U1Symmetry,
    TensorIndex,
    FlowDirection,
    SymmetricTensor,
    TensorNetwork,
    contract,
)

# Define U(1) symmetric tensor indices with named legs
u1 = U1Symmetry()
phys_charges = np.array([-1, 1], dtype=np.int32)
bond_charges = np.array([-1, 0, 1], dtype=np.int32)
key = jax.random.PRNGKey(0)

A = SymmetricTensor.random_normal(
    indices=(
        TensorIndex.from_charges(u1, phys_charges, FlowDirection.IN, label="p0"),
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.OUT, label="bond"),
    ),
    key=key,
)
B = SymmetricTensor.random_normal(
    indices=(
        TensorIndex.from_charges(u1, phys_charges, FlowDirection.IN, label="p1"),
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.IN, label="bond"),  # shared label
        TensorIndex.from_charges(u1, bond_charges, FlowDirection.OUT, label="right"),
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

## Examples

One short example per algorithm family. Each guide page has the full
version, the options, and the caveats.

### DMRG

```python
from tenax import DMRGConfig, build_mpo_heisenberg, build_random_symmetric_mps, dmrg

L = 10  # chain length
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)  # U(1) block-sparse MPO
mps = build_random_symmetric_mps(L, bond_dim=8)  # matching block-sparse MPS

config = DMRGConfig(max_bond_dim=50, num_sweeps=10)
result = dmrg(mpo, mps, config)
print(f"Ground state energy: {result.energy:.8f}")
```

2D cylinders map onto a chain through AutoMPO; see the
[DMRG guide](docs/guide/algorithms/dmrg.md).

### iDMRG

```python
from tenax import idmrg, build_bulk_mpo_heisenberg, iDMRGConfig

W = build_bulk_mpo_heisenberg(Jz=1.0, Jxy=1.0)
config = iDMRGConfig(max_bond_dim=32, max_iterations=100, convergence_tol=1e-8)
result = idmrg(W, config)
print(f"Energy per site: {result.energy_per_site:.6f}")  # ~ -0.4431
print(f"Converged: {result.converged}")
```

Infinite cylinders: see the [iDMRG guide](docs/guide/algorithms/idmrg.md).

### TRG

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

HOTRG, Gilt-TNR and the Potts model follow the same pattern
([TRG](docs/guide/algorithms/trg.md), [HOTRG](docs/guide/algorithms/hotrg.md)).

### AutoMPO

```python
from tenax import AutoMPO

L = 10
auto = AutoMPO(L)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
mpo_sym = auto.to_mpo(symmetric=True)  # U(1) block-sparse MPO
```

Custom operators and the functional interface: see the
[AutoMPO guide](docs/guide/algorithms/auto_mpo.md).

### iPEPS simple update

```python
from tenax import iPEPSConfig, CTMConfig, heisenberg_gate, ipeps

# 2-site checkerboard iPEPS — captures Neel order
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.05,
    ctm=CTMConfig(chi=10, max_iter=40),
    unit_cell="2site",
)
energy, peps, (env_A, env_B) = ipeps(heisenberg_gate(), None, config)
print(f"Energy per site: {energy:.6f}")  # ~ -0.63
```

The energy `ipeps()` reports is a quick estimate; for an accurate number, and
for the bond-weight options, see the [iPEPS guide](docs/guide/algorithms/ipeps.md).

### iPEPS AD optimization

```python
from tenax import (
    iPEPSConfig, CTMConfig, heisenberg_gate, optimize_gs_ad, sublattice_rotate_gate,
)

# 1-site unit cell: rotate one sublattice so the Neel state is translation invariant
H = sublattice_rotate_gate(heisenberg_gate())
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.05,
    ctm=CTMConfig(chi=8, max_iter=40),
    gs_num_steps=10,  # implicit AD + L-BFGS (Hager-Zhang) by default
    su_init=True,     # warm-start from simple update
)
A_opt, env, E_gs = optimize_gs_ad(H, None, config)
print(f"Ground-state energy: {E_gs:.6f}")  # ~ -0.660
```

The [AD paths guide](docs/guide/algorithms/ipeps_ad_paths.md) compares the
implicit, explicit, C4v and root-implicit paths and gives a recommended
configuration; excitation spectra are in the
[excitations guide](docs/guide/algorithms/ad_excitations.md).

### Fermionic iPEPS (fPEPS)

```python
import jax
from tenax import FPEPSConfig, fpeps, spinless_fermion_gate, sublattice_gap

# mu = 2V is the half-filling point of the t-V model
config = FPEPSConfig(D=2, t=1.0, V=4.0, mu=8.0, dt=0.05, num_imaginary_steps=200,
                     ctm_chi=8, ctm_max_iter=60, ctm_conv_tol=1e-8)
H = spinless_fermion_gate(config)

energy, (A, B), (env_A, env_B) = fpeps(H, config, key=jax.random.PRNGKey(0))
print(energy, sublattice_gap(A, B, env_A, env_B))  # -4.0 1.0: CDW, E = -V per site
```

Read the [fPEPS guide](docs/guide/algorithms/fpeps.md) before trusting the
energy: it covers the sign convention, the chemical potential, and the
standing caveats.

### Example scripts

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
| `gilt_hotrg_ising.py` | Gilt-HOTRG | 2D Ising critical point |
| `kagome_spin12_pess_ad_benchmark.py` | iPESS AD | Spin-½ kagome AFM Heisenberg sweep |
| `kagome_spin1_pess_ad_benchmark.py` | iPESS AD | Spin-1 kagome Heisenberg sweep |
| `kagome_spin1_xxz_anisotropy_sweep.py` | iPESS AD | Spin-1 kagome XXZ Δ ∈ {0, 0.5, 1, 1.5, 2} |

Run any example with:

```bash
uv run python examples/<script>.py
```

## Documentation

The full documentation — user guide, algorithm tutorials and API reference —
is at [tenax.readthedocs.io](https://tenax.readthedocs.io). The sources are in
[`docs/`](docs/index.md); build them locally with Sphinx:

```bash
uv sync --extra docs
cd docs && uv run make html
```

The generated HTML is in `docs/_build/html/`.

## Benchmarks

A CLI-driven benchmark suite measures wall-clock performance of every algorithm
across hardware backends:

```bash
# Quick smoke test (TRG, small size, 1 trial)
python -m benchmarks.run --backend cpu --algorithm trg --size small --trials 1

# Show available backends
python -m benchmarks.run --list-backends
```

See the [benchmarks guide](docs/guide/benchmarks.md) for every option and output format.

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
uv run pytest -m algorithm     # algorithm tests (DMRG, TRG, iPEPS, integration)
uv run pytest -m "not slow"    # skip expensive tests
uv run pytest                  # full suite

# Lint
uv run ruff check src/ tests/
```

See the [contributing guide](docs/guide/contributing.md) for the workflow and
how to document new API. Work-in-progress design documents live in `design/`.

## References

- H.-J. Liao, J.-G. Liu, L. Wang, T. Xiang, *Phys. Rev. X* **9**, 031041 (2019) — AD-based iPEPS ground-state optimization
- A. Francuz, N. Schuch, B. Vanhecke, *PRR* **7**, 013237 (2025) — Stable AD through CTM (SVD regularization, truncation correction, implicit differentiation)
- M. Rader, L. Gresista, C. Hubig, S. Montangero, A. Weichselbaum, J. von Delft, arXiv:2511.09546 (2025) — Metric preconditioning and Hager-Zhang line search for iPEPS optimization
- L. Ponsioen, F. F. Assaad, P. Corboz, *SciPost Phys.* **12**, 006 (2022) — Quasiparticle excitations for iPEPS
- J. Naumann, E. L. Weerda, J. Eisert, M. Rizzi, P. Schmoll, arXiv:2502.10298 (2025) — Split-CTMRG with factored projectors for efficient iPEPS environments

Algorithm-specific references are listed on each guide page.

## License

Apache 2.0
