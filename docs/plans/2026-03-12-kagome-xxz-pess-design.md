# Kagome XXZ via PESS — Design

## Goal

Implement a PESS (Projected Entangled Simplex States) simulation for the spin-1/2 XXZ model on the Kagome lattice, with simple update optimization and CTM energy measurement.

## Architecture

- **PESS tensor network**: Rank-3 site tensors on Kagome vertices, rank-3 simplex tensors on up/down triangles, connected by virtual bonds of dimension D.
- **Simple update**: Imaginary time evolution on each triangle via the 3-site XXZ Hamiltonian, truncated by HOSVD on the simplex tensor.
- **Energy measurement**: Contract PESS into an effective square-lattice iPEPS (coarse-grain a triangle into a super-site with d=8), then use existing CTM for energy.
- **AD optimization (staged)**: Phase 1 — optimize the mapped iPEPS super-site tensor via existing `optimize_gs_ad()`. Phase 2 (future) — direct AD on PESS tensors through PESS→iPEPS→CTM pipeline.

## Components

| Component | Location | Description |
|-----------|----------|-------------|
| `xxz_gate(delta)` | `ipeps.py` | 2-site XXZ gate as `DenseTensor` |
| `kagome_pess_su()` | `examples/kagome_xxz_pess.py` | PESS simple update on Kagome |
| `pess_to_ipeps()` | same file | Contract PESS → effective square-lattice iPEPS |
| `save/load_pess()` | same file | Serialize/deserialize PESS state |
| Energy measurement | reuse `ctm()` + `compute_energy_ctm()` | Existing tenax CTM |
| AD optimization | reuse `optimize_gs_ad()` | Existing tenax AD with PESS-initialized A |

## Data Flow

1. Initialize random S (site) and T (simplex) tensors
2. For each imaginary time step, for each triangle:
   - Absorb lambda weights onto S tensors
   - Contract 3 site tensors + simplex tensor into theta
   - Apply exp(-dt * H_triangle) gate
   - HOSVD decompose back into S1, S2, S3, T_new, lambdas
3. Contract PESS → effective iPEPS super-site A (D_eff=D², d=8)
4. Run `ctm(A, config.ctm)` for environment
5. Compute energy via RDM

## PESS Tensor Structure

- `S_up`, `S_left`, `S_right` — 3 site tensors per unit cell, shape `(D, D, d)` with d=2
- `T_up`, `T_down` — simplex tensors, shape `(D, D, D)`
- `lambda_bonds` — singular value vectors on each bond

## XXZ Hamiltonian

H = sum_{<i,j>} [ Sx_i Sx_j + Sy_i Sy_j + delta * Sz_i Sz_j ]

Parameterized by anisotropy delta. delta=1 recovers isotropic Heisenberg.

## Serialization

Save PESS state via `jnp.savez` for reuse in AD optimization. The mapped super-site tensor A serves as `A_init` for `optimize_gs_ad()`.

## Reference

Xie et al., PRL 112, 147203 (2014) — original PESS formulation for Kagome.
