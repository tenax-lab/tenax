#!/usr/bin/env python
"""Spinless fermion iPEPS example using fermionic PEPS (fPEPS).

Runs fPEPS simple update with SymmetricTensor (FermionParity symmetry)
for the spinless fermion model on a square lattice:

    H = -t sum (c†_i c_j + h.c.) + V sum n_i n_j

Phase 1 uses dense CTM fallback for energy evaluation.
"""

import jax

from tenax import FPEPSConfig, fpeps, spinless_fermion_gate

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
energy, A_opt, env = fpeps(gate, config, key=jax.random.PRNGKey(42))
print(f"Free fermion energy per site: {energy:.6f}")

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
energy_V, _, _ = fpeps(gate_V, config_V, key=jax.random.PRNGKey(42))
print(f"Interacting (V=1) energy per site: {energy_V:.6f}")
