#!/usr/bin/env python
"""Spinless fermion iPEPS example using fermionic PEPS (fPEPS).

Runs fPEPS simple update with SymmetricTensor (FermionParity symmetry)
for the spinless fermion model on a square lattice:

    H = -t sum (c†_i c_j + h.c.) + V sum n_i n_j

``fpeps()`` is a **2-site checkerboard** (#878), so the state and environment
come back as pairs -- the t-V ground state at finite ``V`` is a charge-density
wave, which no single tensor can represent.  ``sublattice_gap`` reports how much
charge order the run actually produced.

Two caveats this prints numbers under: the sweep is seed-dependent, and the
absolute energy is not certified (#392) -- ``H`` has no chemical potential, so
the empty state and the fully polarised checkerboard are both exact ``E = 0``
eigenstates that imaginary time can settle on.
"""

import jax

from tenax import FPEPSConfig, fpeps, spinless_fermion_gate, sublattice_gap

# --- Free fermions (t=1, V=0) ---
config = FPEPSConfig(
    D=2,
    t=1.0,
    V=0.0,
    dt=0.01,
    num_imaginary_steps=200,
    ctm_chi=8,
    ctm_max_iter=50,
)

gate = spinless_fermion_gate(config)
energy, (A, B), (env_A, env_B) = fpeps(gate, config, key=jax.random.PRNGKey(42))
print(f"Free fermion energy per site: {energy:.6f}")
print(f"Free fermion charge order:    {sublattice_gap(A, B, env_A, env_B):.4f}")

# --- With nearest-neighbour repulsion (V=1) ---
config_V = FPEPSConfig(
    D=2,
    t=1.0,
    V=1.0,
    dt=0.01,
    num_imaginary_steps=200,
    ctm_chi=8,
    ctm_max_iter=50,
)

gate_V = spinless_fermion_gate(config_V)
energy_V, (A_V, B_V), (env_A_V, env_B_V) = fpeps(
    gate_V, config_V, key=jax.random.PRNGKey(42)
)
print(f"Interacting (V=1) energy per site: {energy_V:.6f}")
print(
    f"Interacting (V=1) charge order:    "
    f"{sublattice_gap(A_V, B_V, env_A_V, env_B_V):.4f}"
)
