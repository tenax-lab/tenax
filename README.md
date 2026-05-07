# Tenax

[Website](https://tenax-lab.github.io) | [Docs](https://tenax.readthedocs.io) | [PyPI](https://pypi.org/project/tenax-tn/)

A JAX-based tensor network library with symmetry-aware block-sparse tensors and label-based contraction.

The name **Tenax** combines **Ten**sor network + J**ax**, and is also Latin for "holding fast" — reflecting how tensor networks bind indices together through contraction.

> **Experimental project** — This library is under active development and largely written with the assistance of Claude Code (AI). While we test extensively, AI-generated code can contain subtle bugs. Please verify results against known benchmarks before using them in research. Bug reports and contributions are welcome.

## Features

- **Block-sparse symmetric tensors** — only symmetry-allowed charge sectors stored (U(1), Z_n)
- **Label-based contraction** — legs are identified by string/integer labels; shared labels are automatically contracted (Cytnx-style)
- **opt_einsum integration** — optimal contraction path finding for multi-tensor contractions
- **Network class** — graph-based tensor network container with contraction caching
- **`.net` file support** — cytnx-style declarative network topology; parse once, load tensors, contract repeatedly (template pattern)
- **Algorithms** — DMRG, iDMRG (1D chain & infinite cylinder), TRG, HOTRG, iPEPS (simple update with 1-site or 2-site unit cell & AD optimization), fermionic iPEPS (fPEPS), quasiparticle excitations
- **GPU/TPU-accelerated DMRG** — JIT-compiled sweeps via `jax.lax.scan` for dense tensors and per-operation JIT for block-sparse symmetric tensors; automatic warmup-to-JIT transition when bond dimensions are growing; multi-GPU sharding via GSPMD for large bond dimensions (`DMRGConfig(accelerator="jit"|"sharded")`)
- **AutoMPO** — build Hamiltonian MPOs from symbolic operator descriptions (custom couplings, NNN, arbitrary spin); supports `symmetric=True` for U(1) block-sparse MPOs
- **AD-based iPEPS optimization** — gradient optimization via implicit differentiation through CTM fixed point, supporting 1-site and 2-site unit cells (Francuz et al. PRR 7, 013237); L-BFGS with Hager-Zhang line search and metric preconditioning (Rader et al.), Adam (with cosine lr decay), and conjugate gradient optimizers; implicit AD via iterative VJP (default) and optional GMRES route; explicit AD through unrolled CTM iterations for 1-site C4v path; **2-site shared-tensor C4v path** (`unit_cell="2site"` + `gs_c4v=True`) where a single C4v tensor is optimized and the second sublattice is derived by spin-π rotation, stable across χ=8–24 for spin-1/2 AFMs; opt-in reference-mode dense C4v Appendix C-F mode (`ctm_ad_mode="c4v_reference"`) with Krylov implicit backward (`bicgstab` + `gmres` fallback); sigma gauge fixing (`forward_gauge="sigma"`) for stable elementwise CTM convergence; C4v symmetry enforcement via explicit basis parameterization; chi-ramping schedule (`optimize_gs_ad_chi_schedule`) for progressive refinement- **SVD and QR CTMRG projectors** — SVD (Fishman) projectors (`projector_method="svd"`) and QR projectors for faster CTM convergence alongside the default `eigh`
- **Split-CTMRG** — ket/bra-separated CTM environment tensors for O(χ³D³) projector cost instead of O(χ³D⁶); works with both `DenseTensor` and `SymmetricTensor` via the Tensor protocol (Naumann et al., arXiv:2502.10298)
- **Split-CTM energy entry points** — `compute_energy_split_ctm_tensor_2site` and `compute_energy_split_ctm_tensor_multisite` for 2-site checkerboard and multisite unit cells (kagome PESS, etc.) at large D
- **Honeycomb iPEPS CTM (native)** — rank-4, 6-corner, 3-direction, 2-sublattice CTMRG for honeycomb iPEPS (replaces the dummy-bond brick-wall workaround). Public entry `honeycomb_ctm_energy_implicit` provides `jax.custom_vjp` with a JIT-fused GMRES backward; default Corboz biorthogonal projector + per-column phase fix; configurable `energy_fn` hook for kagome iPESS triangle energies. References: Lukin & Sotnikov, PRB 107, 054424 (2023) for the 6-corner CTMRG and the bipartite extension in PRE 109, 045305 (2024) §II.C.
- **Quasiparticle excitations** — iPEPS excitation spectra at arbitrary Brillouin-zone momenta (Ponsioen et al. 2022)
- **Polymorphic tensor arithmetic** — `+`, `-`, `*`, `-T`, `max_abs`, `inner()`, `conj()`, `dagger()`, `bar()` work identically on `DenseTensor` and `SymmetricTensor`, enabling algorithm code that is agnostic to the underlying storage
- **Block-sparse SVD, QR, and eigh** — native symmetry-aware decompositions in `tenax.linalg` for `SymmetricTensor`
- **Sector-based TensorIndex** — legs store sorted charge sectors and multiplicities for O(n_sectors) lookups; `FuseInfo` tracks parent legs so `split_index` can reverse `fuse_indices`
- **Cython BLAS fast path** — fused Cython Lanczos solver and block-sparse contractions via direct BLAS calls with zero Python reentry for high-performance CPU DMRG
- **iDMRG transfer matrix environments** — fixed-point environment computation for self-consistent infinite boundary conditions
- **Extensible symmetry system** — non-Abelian symmetry interface for future SU(2) support
- **Benchmark suite** — CLI-driven performance benchmarks for all algorithms across CPU, CUDA, TPU, and Metal backends

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

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the latest accelerator options.

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
        TensorIndex(u1, phys_charges, FlowDirection.IN, label="p0"),
        TensorIndex(u1, bond_charges, FlowDirection.IN, label="left"),
        TensorIndex(u1, bond_charges, FlowDirection.OUT, label="bond"),
    ),
    key=key,
)
B = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, phys_charges, FlowDirection.IN, label="p1"),
        TensorIndex(u1, bond_charges, FlowDirection.IN, label="bond"),  # shared label
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

> **Performance note:** Tenax's DMRG uses a fused Cython BLAS pipeline on CPU for high-throughput block-sparse contractions. GPU/TPU acceleration is available via `DMRGConfig(accelerator="jit")` for dense tensors and `accelerator="sharded"` for multi-GPU runs.

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
        auto += (1.0, "Sz", min(i, j), "Sz", max(i, j))
        auto += (0.5, "Sp", min(i, j), "Sm", max(i, j))
        auto += (0.5, "Sm", min(i, j), "Sp", max(i, j))
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
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) + 0.5 * (
    jnp.einsum("ij,kl->ikjl", Sp, Sm) + jnp.einsum("ij,kl->ikjl", Sm, Sp)
)

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
    iPEPSConfig,
    CTMConfig,
    optimize_gs_ad,
    optimize_gs_ad_chi_schedule,
    ExcitationConfig,
    compute_excitations,
    make_momentum_path,
)

# Build a 2-site Heisenberg gate
Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) + 0.5 * (
    jnp.einsum("ij,kl->ikjl", Sp, Sm) + jnp.einsum("ij,kl->ikjl", Sm, Sp)
)

# Recommended AD configuration: L-BFGS + explicit AD + QR projectors.
# forward_gauge defaults to "phase" (variPEPS-style Frobenius + phase
# fix), correct for both implicit and explicit AD. Reaches E=-0.6628
# at D=2, chi=16 (literature: -0.6548 at D=2).
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(
        chi=16,
        max_iter=80,
        projector_method="qr",  # recommended projector for explicit AD
    ),
    # gs_implicit_ad=False is the default (explicit AD)
    gs_projector_method="qr",
    gs_optimizer="lbfgs",  # L-BFGS with Hager-Zhang line search
    gs_line_search_method="hager_zhang",
    gs_metric_precond=True,  # metric preconditioning (Rader et al.)
    gs_c4v=True,  # C4v basis parameterization
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
print(f"Ground-state energy: {E_gs:.6f}")

# Chi-ramping schedule: progressively increase chi for faster convergence
chi_schedule = [4, 8, 16]
A_opt, env, E_gs = optimize_gs_ad_chi_schedule(gate, None, config, chi_schedule)

# 2-site shared-tensor C4v AD for antiferromagnets (Neel order)
# A single C4v-parameterized tensor is optimized; B is derived from A via
# sublattice rotation B = e^{i pi sigma^y/2} on the physical leg.  This
# ties the two sublattices together and avoids the A/B drift that makes
# the unconstrained 2-site AD path unstable.  Spin-1/2 (d=2) only.
config_2site = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=100, min_iter=50),
    gs_optimizer="lbfgs",
    gs_explicit_ad_steps=10,
    gs_explicit_ad_warmup=2,
    gs_num_steps=50,
    gs_line_search=True,
    unit_cell="2site",
    gs_c4v=True,
    su_init=True,
    num_imaginary_steps=100,
    dt=0.3,
)
(A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(gate, None, config_2site)

# SVD (Fishman) projectors — alternative to eigh and QR
config_svd = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50, projector_method="svd"),
    gs_num_steps=200,
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config_svd)

# Opt-in reference-mode dense C4v mode (Francuz et al., App. C-F)
config_reference = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(
        chi=16,
        max_iter=80,
        projector_method="eigh",
        ctm_ad_mode="c4v_reference",
        adjoint_solver="bicgstab",
        adjoint_maxiter=50,
        adjoint_tol=1e-8,
    ),
    gs_implicit_ad=True,
    gs_c4v=True,
    unit_cell="1x1",
    gs_num_steps=100,
    gs_optimizer="adam",
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config_reference)

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

## Honeycomb iPEPS CTM (native rank-4)

Native rank-4 CTMRG for honeycomb iPEPS — six corners, three edge
directions, two sublattices — without the dummy-bond brick-wall hack.
Custom `jax.custom_vjp` forward with a JIT-fused GMRES backward.

```python
import jax
import jax.numpy as jnp
import numpy as np
from tenax import (
    HONEYCOMB_DIRECTIONS,
    honeycomb_ctm_energy_implicit,
    honeycomb_ctm_run,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_site(D=2, d=2, key=jax.random.PRNGKey(0)):
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
    return DenseTensor((re + 1j * im).astype(jnp.complex128), indices)


# Spin-1/2 Heisenberg bond operator (4×4)
sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_bond = jnp.asarray(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))

# Honeycomb iPEPS uses two rank-4 sites at coords (0,0) and (1,0); legs
# (e0, e1, e2, phys). All virtuals OUT, phys IN.
A = _make_site(D=2, d=2, key=jax.random.PRNGKey(0))
B = _make_site(D=2, d=2, key=jax.random.PRNGKey(1))
sites = {(0, 0): A, (1, 0): B}

# Forward only: returns the converged per-sublattice env dict + info.
envs, info = honeycomb_ctm_run(
    sites, chi=8, max_iter=80, conv_tol=1e-8,
    projector_method="biorthogonal",  # default; eigh/svd are A=B opt-ins
    forward_gauge="phase",            # default; sigma reserved for A=B opt-in
)

# Implicit-AD energy: takes jax.grad through the CTM fixed point via
# JIT-fused GMRES on (I - dF/denv) lambda = dE/denv.
energy = honeycomb_ctm_energy_implicit(
    sites, H_bond, chi=8, max_iter=80, conv_tol=1e-8,
)
grad_fn = jax.grad(
    lambda Ad: honeycomb_ctm_energy_implicit(
        {(0, 0): DenseTensor(Ad, A.indices), (1, 0): B},
        H_bond, chi=8, max_iter=40,
    )
)
gA = grad_fn(A.todense())
```

The default energy is the 3-edge nearest-neighbor bond sum
`Σ_α Tr(ρ_α · H_bond)`. Pass `energy_fn=compute_honeycomb_triangle_energy`
for the kagome iPESS use case where each site is a 3-spin triangle and
the Hamiltonian is the intra-triangle 3-spin operator.

## Kagome iPESS with AD

Differentiable iPESS pipeline for kagome XXZ ground states (Liao et al.,
PRX 9, 031041, 2019). Two simplex tensors `T_u`, `T_d` and three site
tensors `R_a`, `R_b`, `R_c` define the variational state; triangle
simple update gives the SU warm start, then L-BFGS through the
square-coarse-grained CTM (Convention C) refines `(R_a, R_b, R_c, T_u,
lambdas)`. `T_d` is held frozen during AD — its variational role is
absorbed by the down-bond gauges.

```python
import jax
from tenax import (
    CTMConfig,
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    kagome_xxz_pess_cg_gates,
    pess_simple_update,
    optimize_pess_ad,
)

D, d = 2, 3  # spin-1
H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=d)
cg_gates = kagome_xxz_pess_cg_gates(delta=1.0, d=d)

state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))
state = pess_simple_update(state, H,
                           dt_schedule=[(0.1, 200), (0.01, 200), (0.001, 100)],
                           D_max=D)

config = CTMConfig(chi=8, max_iter=30, conv_tol=1e-7,
                   projector_method="svd", forward_gauge="phase",
                   ctm_conv_method="elementwise")
state, e_per_site = optimize_pess_ad(state, cg_gates, config, max_iter=30)
print(f"E/site = {e_per_site:.6f}")  # spin-1 D=2 lands around -1.0
```

The full kagome Hamiltonian (3 up-triangle bonds + 3 down-triangle
bonds per unit cell) is reconstructed via `compute_energy_cg`'s
intra-cell + horizontal/vertical/diagonal inter-cell 2-site RDMs; see
`examples/kagome_spin12_pess_ad_benchmark.py` and
`examples/kagome_spin1_pess_ad_benchmark.py` for full sweeps.

### Multisite path (3-site kagome on a square unit cell)

For the multisite encoding `pess_to_kagome_3site_multisite`, where the
kagome unit cell maps to three sites `(u, v, w)` on a square lattice and
the energy uses 4 NN bonds + 2 marginalised-3-site contributions, use
`build_pess_loss_3site_multisite` and `optimize_pess_3site_multisite_ad`:

```python
from tenax import (
    build_pess_loss_3site_multisite,
    optimize_pess_3site_multisite_ad,
    pess_to_kagome_3site_multisite,
)
from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates

bond_gates = kagome_3site_bond_gates(delta=1.0, d=d)
state, e_per_site = optimize_pess_3site_multisite_ad(
    state, bond_gates, config, max_iter=30,
)
```

The optimizer warm-starts CTM envs across L-BFGS steps via an internal
`env_cache`, returns the best-seen energy across the trajectory, and
gates `CTMConfig` at entry on the implicit-AD invariants
(`projector_method='svd'`, `forward_gauge='phase'`,
`ctm_conv_method='elementwise'`).

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
| `kagome_spin12_pess_ad_benchmark.py` | iPESS AD | Spin-½ kagome AFM Heisenberg sweep |
| `kagome_spin1_pess_ad_benchmark.py` | iPESS AD | Spin-1 kagome Heisenberg sweep |
| `kagome_spin1_xxz_anisotropy_sweep.py` | iPESS AD | Spin-1 kagome XXZ Δ ∈ {0, 0.5, 1, 1.5, 2} |

Run any example with:

```bash
uv run python examples/<script>.py
```

## Symmetry System

```python
from tenax import U1Symmetry, ZnSymmetry, ProductSymmetry, FermionParity
import numpy as np

# U(1): integer charges, fusion by addition
u1 = U1Symmetry()
charges = np.array([-1, 0, 1], dtype=np.int32)
print(u1.fuse(charges, charges))  # [-2, 0, 2]
print(u1.dual(charges))  # [1, 0, -1]

# Z_3: charges mod 3
z3 = ZnSymmetry(3)
print(
    z3.fuse(np.array([1, 2], dtype=np.int32), np.array([2, 2], dtype=np.int32))
)  # [0, 1]

# Product symmetry: combine two symmetries (e.g., charge × S_z)
sym = ProductSymmetry(U1Symmetry(), U1Symmetry())
packed = ProductSymmetry.encode_charges(
    np.array([0, 1, -1], dtype=np.int32),  # charge
    np.array([1, 0, -1], dtype=np.int32),  # S_z
)
q1, q2 = ProductSymmetry.decode_charges(packed)
```

**Limitations:** `ProductSymmetry` combines exactly two factors by bit-packing two int16 charges into one int32. Nesting is not supported, so three-factor groups (e.g., U(1)×U(1)×Z₂) require a future `MultiProductSymmetry`. Each factor charge must fit in the int16 range [-32768, 32767].

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
uv run pytest -m algorithm     # algorithm tests (DMRG, TRG, iPEPS, integration)
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
- M. Rader, L. Gresista, C. Hubig, S. Montangero, A. Weichselbaum, J. von Delft, arXiv:2511.09546 (2025) — Metric preconditioning and Hager-Zhang line search for iPEPS optimization
- L. Ponsioen, F. F. Assaad, P. Corboz, *SciPost Phys.* **12**, 006 (2022) — Quasiparticle excitations for iPEPS
- J. Naumann, E. L. Weerda, J. Eisert, M. Rizzi, P. Schmoll, arXiv:2502.10298 (2025) — Split-CTMRG with factored projectors for efficient iPEPS environments

## License

Apache 2.0
