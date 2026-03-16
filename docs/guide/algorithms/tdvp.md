# TDVP

The Time-Dependent Variational Principle (TDVP) performs time evolution of a
Matrix Product State (MPS) by projecting the Schrödinger equation onto the
MPS tangent space.

## Background

TDVP evolves ``|ψ(t)⟩ = exp(-iHt)|ψ₀⟩`` (real-time) or
``|ψ(τ)⟩ = exp(-Hτ)|ψ₀⟩`` (imaginary-time) while keeping the state in
MPS form at all times. The algorithm sweeps through the chain, solving
local time-evolution problems via a Krylov (Lanczos) matrix exponential.

Key properties of the Tenax implementation:

- **1-site TDVP**: second-order Lie-Trotter splitting. Fixed bond dimension.
  Forward-evolves each site tensor by ``dt/2``, QR-decomposes, back-evolves
  the bond matrix by ``-dt/2``.
- **2-site TDVP**: merges two sites, evolves, SVD-truncates. Allows bond
  dimension growth.
- **Krylov matrix exponential** (``krylov_expm``): computes ``exp(dt·H_eff)·v``
  via Lanczos iteration without forming the full matrix. Reusable utility in
  ``_krylov.py``.
- Supports both **real-time** (``time_type="real"``) and **imaginary-time**
  (``time_type="imaginary"``) evolution.
- The outer sweep loop is a Python for-loop (not JIT-compiled), matching the
  DMRG pattern.

## Configuration

```python
from tenax import TDVPConfig

config = TDVPConfig(
    mode="1site",          # "1site" or "2site"
    dt=0.05,               # time step magnitude
    time_type="real",      # "real" for e^{-iHt}, "imaginary" for e^{-Ht}
    num_steps=100,         # number of steps for tdvp() driver
    max_bond_dim=64,       # SVD truncation (2-site only)
    svd_trunc_err=None,    # SVD error threshold (2-site only)
    krylov_dim=20,         # Krylov subspace dimension
    krylov_tol=1e-12,      # Krylov convergence tolerance
    verbose=False,
)
```

## Example — real-time quench dynamics

```python
from tenax import (
    TDVPConfig, tdvp, tdvp_step,
    DMRGConfig, dmrg,
    build_mpo_heisenberg, build_random_mps,
)

# Prepare ground state via DMRG
L = 20
mpo = build_mpo_heisenberg(L)
mps = build_random_mps(L, physical_dim=2, bond_dim=16)
result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=32, num_sweeps=20))
mps_gs = result.mps

# Real-time evolution
config = TDVPConfig(mode="1site", dt=0.05, time_type="real", num_steps=200)
result = tdvp(mps_gs, mpo, config)

# Energy should be conserved
print(f"E(t=0) = {result.energies[0]:.10f}")
print(f"E(t=T) = {result.energies[-1]:.10f}")
```

## Example — imaginary-time ground state

```python
config = TDVPConfig(
    mode="2site", dt=0.1, time_type="imaginary",
    num_steps=100, max_bond_dim=32,
)
result = tdvp(mps_init, mpo, config)
# Energy decreases toward ground state
```

## API

- ``tdvp_step(mps, hamiltonian, config)`` — one time step, returns updated MPS.
- ``tdvp(mps, hamiltonian, config, measure=None)`` — multi-step driver with
  energy tracking and optional measurement callback.
- ``krylov_expm(matvec, v, dt, krylov_dim, tol)`` — standalone Krylov matrix
  exponential.

## References

- Haegeman et al., *Phys. Rev. B* **94**, 165116 (2016) — 1-site and 2-site TDVP.
- Paeckel et al., *Ann. Phys.* **411**, 167998 (2019) — review of MPS time evolution.
