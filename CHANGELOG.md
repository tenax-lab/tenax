# Changelog

## v0.4.2 (2026-04-05)

### Improvements

- **Fix PyPI publish workflow** — add attestation permissions and update
  GitHub Actions to Node.js 24 (#258)

## v0.4.1 (2026-04-04)

### Improvements

- **GMRES diagonal scaling preconditioner** for faster implicit diff backward
  pass in iPEPS AD optimization (#231)
- **Fix TensorIndex API calls** for sector-based refactor (#234)
- **Remove xfail** from `test_ad_d2_chi_scaling` — chi-scaling now works
  correctly with L-BFGS optimizer and fresh CTM line search
- **Iterative VJP backward** for CTM implicit differentiation — replaces
  GMRES as default; robust to gauge instability that caused NaN (#240)

## v0.4.0 (2026-04-03)

### New Features

- **L-BFGS and CG optimizers** for iPEPS AD with Armijo backtracking line
  search (`gs_optimizer="lbfgs"` or `"cg"`). Line search runs fresh CTM
  convergence for each trial step to avoid stale-environment artifacts (#235)
- **Explicit CTM differentiation** (experimental) — backprop through unrolled
  CTM iterations via `gs_implicit_ad=False`, as an alternative to implicit
  differentiation (#235)
- **Cosine learning rate decay** for Adam iPEPS optimizer — lr decays to lr/10
  over the optimization when `gs_num_steps > 20` (#235)
- **GPU/TPU-accelerated DMRG** — JIT-compiled sweeps via `jax.lax.scan` for
  dense tensors and per-operation JIT for block-sparse symmetric tensors;
  multi-GPU sharding via GSPMD (`DMRGConfig(accelerator="jit"|"sharded")`) (#209)
- **cuTENSOR block-sparse contractions** for `SymmetricTensor` on GPU (#203)
- **cuTensorNet backend** for dense GPU contractions (#202)
- **Symmetric iPEPS simple update** with non-trivial U(1) charges (#206)
- **Sector-based TensorIndex** — legs store sorted charge sectors and
  multiplicities for O(n_sectors) lookups; `FuseInfo` tracks parent legs so
  `split_index` can reverse `fuse_indices` (#213)
- **AD-based fermionic iPEPS** (fPEPS) optimization (#214)
- **iDMRG transfer matrix** fixed-point environments for self-consistent
  infinite boundary conditions (#215, #217)
- **Fused Cython Lanczos** + matvec dispatch — single Cython call for the
  entire Lanczos solve, eliminating Python loop overhead (#226)

### Performance

- **Cython BLAS acceleration** for block-sparse contractions — NumPy BLAS
  calls from Cython with zero Python reentry (#205, #207, #212)
- **Finite DMRG 2.7–5.3x faster than TeNPy** on CPU with Cython pipeline (#226)
- **iDMRG 3–4.5x speedup** + fix chi>=96 divergence (#229)
- **Cython pipeline optimizations** — fused matvec, precomputed block plans,
  reduced dispatch overhead (#212)

### Bug Fixes

- Fix post-#226 solver/config bugs (#228)
- Fix 4 correctness bugs in Cython BLAS path (#218)
- Add Cython availability guards to BLAS regression tests (#224)
- Fix Codecov v5 coverage input key (#227)
- Mark `test_ad_d2_energy` as xfail for underconverged CTM (#210)
- Resolve full test suite CI failures (#201, #208)

## v0.3.0 (2026-03-27)

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
- **Sweep-based iDMRG** — replaced growing-chain algorithm with proper
  sweep-based iDMRG with environment warmup (#191, #197)
  - Environment warmup phase for self-consistent infinite environments
  - 1-site update with DMRG3S subspace expansion (`two_site=False`)
  - Energy monotonically improves with chi (fixed chi scaling issue)
  - QR-based orthogonalization for numerical stability
- **Gradient clipping for iPEPS AD** — `gs_max_grad_norm` field in
  iPEPSConfig (default 1.0) prevents gradient spikes from diverging
  the optimizer. lr=1e-2 now gives E=-0.663 for D=2 Heisenberg
  (previously diverged) (#198)
- **SymmetricTensor 23x speedup** — cached `blocks` dict + NumPy einsum
  in `_blockwise_contract` (#196)
  - `blocks` property returns immutable `MappingProxyType`, cached on
    first access (8.5x from avoiding redundant slice+reshape)
  - NumPy einsum for per-block contractions avoids JAX dispatch
    overhead (74x faster per operation)
- **JIT-compiled Lanczos** — `_lanczos_solve_jit` via `lax.fori_loop`,
  120x faster per call for dense tensors (#199)
- **Precomputed block plan** — `_precompute_block_plan` enumerates
  valid charge-sector combinations once before Lanczos loop (#199)
- **Non-trivial U(1) gauge fix** — dense QR + `from_dense()` wrapping
  preserves charge layout for 2-site iPEPS CTM (#193)
- **iPEPS regression benchmarks** — D=2 Heisenberg SU/AD energy and
  chi scaling tests (#192)
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
