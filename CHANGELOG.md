# Changelog

## v0.3.0 (unreleased)

### Breaking Changes

- **3-leg boundary tensors** — All MPS boundary tensors are now uniformly
  3-leg with trivial dimension-1 bonds (#169)
  - Site 0: `(1, d, chi)` with labels `(v_-1_0, p0, v0_1)`
  - Site L-1: `(chi, d, 1)` with labels `(v{L-2}_{L-1}, p{L-1}, v{L-1}_{L})`
  - Code that accessed `mps_tensor.ndim == 2` to detect boundaries must be
    updated; all tensors are now `ndim == 3`

### New Features

- **`FiniteMPS` and `InfiniteMPS` classes** with canonical form tracking,
  singular values at every bond, and `log_norm` normalization (#163, #169)
  - `FiniteMPS.random()` replaces `build_random_mps()` / `build_random_symmetric_mps()`
  - `canonicalize(center)` with QR sweeps (block-sparse, no `todense()`)
  - `compute_singular_values()` populates all bonds in one SVD sweep
  - `target_charge` field for symmetric MPS sector tracking
  - `InfiniteMPS` with `qshift` and L+1 bond convention
- **Controlled Bond Expansion (CBE)** for 1-site DMRG/TDVP (#154, #157)
  - Dense and block-sparse (`expand_bond_symmetric`) implementations
- **Randomized SVD** (`rsvd`) for large-scale truncation (#151)

### Improvements

- **iPEPS CTM: non-trivial U(1) charges** — `SymmetricTensor` iPEPS with
  non-trivial charge sectors now work through the full CTM pipeline (#180)
  - Fixed `_flip_leg_flow` to dual charges + remap block keys
  - Standard and multisite CTM sweeps pass `base_charges` for stable projector truncation
- **Block-sparse gauge fix for CTM AD** — replaced `todense()`/`from_dense()`
  round-trip with direct `tenax.linalg.qr` + `contract`, giving cleaner
  gradients and closing the energy quality gap between dense and Tensor
  AD paths (#182)
- **Unified AD on Tensor protocol** — removed legacy dense AD paths
  (`ctm_converge`, `ctm_converge_2site`); all optimization now uses
  `ctm_tensor_converge` / `ctm_tensor_converge_2site` with `DenseTensor`
  or `SymmetricTensor` (#183)
- Eliminated ~23 boundary special-case code paths across DMRG, TDVP,
  observables, and CBE
- Deleted `_pad_boundary_symmetric` / `_unpad_boundary_symmetric` functions
- DMRG and TDVP accept and return `FiniteMPS` with canonical form contracts

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
