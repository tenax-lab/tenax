# Changelog

## v0.2.0 (2026-03-17)

### New Algorithms

- **TDVP** — Time-Dependent Variational Principle for MPS time evolution (#141, #146, #149)
  - 1-site TDVP with second-order Lie-Trotter splitting
  - 2-site TDVP with SVD truncation for bond dimension growth
  - Real-time, imaginary-time, and complex-time evolution
  - Lanczos-based Krylov matrix exponential (`krylov_expm`)

- **C4v CTM** — Single-move CTM exploiting C4v point-group symmetry (#142)
  - One projector per sweep eliminates charge-sector divergence
  - For 1-site unit cells without sublattice structure

- **Fermionic iPEPS (fPEPS)** — iPEPS with graded tensor formalism (#134)
  - `FermionParity` and `FermionicU1` symmetries with automatic Koszul signs
  - `spinless_fermion_gate` for the t-V model
  - `fpeps()` entry point for simple update + CTM + energy

### Major Improvements

- **Paired CTM moves** for SymmetricTensor charge consistency (#145)
  - Prevents block-size divergence in fermionic CTM after multiple sweeps
  - Uses `base_charges` from double-layer tensor for stable charge allocation

- **Lattice abstraction** and `ctm_multisite()` for general unit cells (#128)
  - Built-in factories: `square`, `checkerboard`, `honeycomb`, `triangular`, `kagome`

- **Fully block-sparse split CTM** sweeps and energy (#121)
  - SymmetricTensor CTM without densification for non-fermionic symmetries

- **iPEPS refactored** into focused submodules (#116–#120)
  - `ipeps_simple_update.py`, `ipeps_optimize.py`, `ipeps_ctm.py`, etc.
  - CTM projector extracted to shared `_ctm_projector.py`

- **AD optimizer fixes** — use converged energy, not best-tracking (#131)

### New Features

- `heisenberg_gate()` and `xxz_gate()` pre-built 2-site gates (#122, #127)
- Kagome XXZ PESS examples (spin-1/2 and spin-1) (#127, #138)
- iPEPS AD optimization progress logging (#129)
- `unit_cell` validation in `iPEPSConfig` (#147)
- Benchmark JSON plotting CLI (#123)

### Documentation

- Algorithm reference pages: TDVP, fPEPS, CTM (#148)
- Claude Code plugin guide and contributing guide (#135)
- Stale design plans removed; algorithm docs kept current

### Infrastructure

- Apache 2.0 license (#139)
- CI workflow to sync skills to tenax-toolkit plugin repo (#133)
- Architecture guard tests for CTM modules (#125)

## v0.1.0 (2026-03-10)

Initial release with DMRG, iDMRG, TRG, HOTRG, iPEPS (simple update + AD),
SymmetricTensor (U(1), Z_n), label-based contraction, and JAX integration.
