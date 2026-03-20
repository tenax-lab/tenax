---
name: tenax-getting-started
description: >
  Help users install Tenax, configure JAX backends (CPU, CUDA, TPU, Metal),
  verify the installation, and run their first tensor network calculation.
  Use this skill for onboarding questions: "how do I install Tenax", "set up
  GPU", "first DMRG run", "getting started", "hello world", or any request
  from a new user who hasn't used the library before.
---

# Getting Started with Tenax

Walk new users from zero to a working Tenax installation and first calculation.

## Installation

### PyPI (recommended)

The PyPI package name is `tenax-tn`; the import name is `tenax`.

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

### Development install

```bash
git clone https://github.com/tenax-lab/tenax.git
cd tenax
uv sync --all-extras --dev
```

## Verify Installation

```python
import tenax
print(tenax.__version__)  # 0.1.0

# Check which backend JAX is using
import jax
print(jax.devices())      # [CpuDevice(...)] or [CudaDevice(...)]
```

Tenax automatically enables 64-bit mode (`jax_enable_x64`) on import. All
tensors and algorithms default to `float64`.

## Backend Selection

JAX picks the backend automatically (GPU if available, else CPU). To force
a specific backend:

```bash
# Force CPU (useful for debugging)
JAX_PLATFORMS=cpu python my_script.py

# Force GPU
JAX_PLATFORMS=cuda python my_script.py

# Force TPU
JAX_PLATFORMS=tpu python my_script.py
```

## First Calculation: Heisenberg Chain DMRG

This is the "hello world" of tensor networks — finding the ground state
energy of the spin-1/2 Heisenberg antiferromagnet.

```python
import jax
from tenax import (
    dmrg, build_mpo_heisenberg, FiniteMPS, DMRGConfig
)

# 1. Build the Hamiltonian as an MPO
L = 20
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)

# 2. Build a random initial MPS
key = jax.random.PRNGKey(0)
mps = FiniteMPS.random(L=L, d=2, chi=16, key=key)

# 3. Configure and run DMRG
config = DMRGConfig(max_bond_dim=64, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)

# 4. Check the result
print(f"Ground state energy: {result.energy:.10f}")
print(f"Energy per site:     {result.energy / L:.10f}")
print(f"Converged:           {result.converged}")
# Expected E/site ≈ -0.4431 for large L (Bethe ansatz)
```

## First Calculation: 2D Ising TRG

A complementary "hello world" for classical stat mech:

```python
from tenax import TRGConfig, trg, compute_ising_tensor, ising_free_energy_exact

beta = 0.44  # near critical temperature (beta_c ≈ 0.4407)
T = compute_ising_tensor(beta)

config = TRGConfig(max_bond_dim=16, num_steps=20)
log_z_per_n = trg(T, config)
f_trg = float(-log_z_per_n / beta)
f_exact = ising_free_energy_exact(beta)

print(f"TRG free energy:   {f_trg:.8f}")
print(f"Exact (Onsager):   {f_exact:.8f}")
```

## What to Try Next

| Goal | Skill / function |
|------|-----------------|
| Build a custom Hamiltonian | `AutoMPO` — see AutoMPO skill |
| Infinite chain | `idmrg` — see DMRG workflow skill |
| 2D ground state | `ipeps` / `optimize_gs_ad` — see iPEPS workflow skill |
| Classical phase transition | `trg` / `hotrg` — see TRG workflow skill |
| Understand symmetric tensors | `SymmetricTensor` — see symmetry skill |
| Custom tensor contractions | `NetworkBlueprint` — see blueprint skill |

## Common First-Time Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: tenax` | Wrong package name | `pip install tenax-tn` (not `tenax`) |
| Arrays are float32 | Imported JAX before tenax | Import `tenax` first, or set `jax_enable_x64` manually |
| Slow first run | JAX JIT compilation | Normal — subsequent runs are fast |
| GPU not detected | Missing jaxlib GPU build | Install with `[cuda12]` or `[cuda13]` extra |
| macOS Metal errors | Experimental backend | Use CPU for production; Metal for experimentation |

## Running Examples

The `examples/` directory has complete scripts:

```bash
uv run python examples/ising_trg.py
uv run python examples/heisenberg_cylinder.py
uv run python examples/heisenberg_ipeps_su.py
```
