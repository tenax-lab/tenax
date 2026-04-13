# DMRG / iDMRG Code Paths Overview

This document is the **architectural map** of Tenax's DMRG stack: which
entry points dispatch to which sweep variant, which eigensolver backend
runs where, and which tensor backend (`DenseTensor` vs
`SymmetricTensor`) each path supports. For **user-facing recipes** see
[`docs/guide/algorithms/dmrg.md`](guide/algorithms/dmrg.md),
[`docs/guide/algorithms/idmrg.md`](guide/algorithms/idmrg.md), and
[`docs/guide/algorithms/tdvp.md`](guide/algorithms/tdvp.md).

Paths marked **BROKEN** raise `NotImplementedError` on entry. Paths
marked **ABSENT** are not implemented in the repo and have no entry
point — a workaround is listed where one exists.

## Pipeline Graph

```
                ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
                │   dmrg()     │     │   idmrg()    │     │   tdvp()     │
                │   dmrg.py    │     │   idmrg.py   │     │   tdvp.py    │
                └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                       │                    │                    │
                       │         ┌──────────┴─────────┐           │
                       │         │ unit_cell_size=2   │           │
                       │         │ (fixed; >2 raises) │           │
                       │         └──────────┬─────────┘           │
                       └────────────────────┼─────────────────────┘
                                            │
                      ┌─────────────────────┼─────────────────────┐
                      │                                           │
               ┌──────▼───────┐                           ┌───────▼──────┐
               │  2-site      │                           │   1-site     │
               │  sweep       │                           │   sweep      │
               │  (default;   │                           │              │
               │   truncates) │                           │              │
               └──────┬───────┘                           └───────┬──────┘
                      │                                           │
                      │                         ┌─────────────────┼─────────────┐
                      │                         │                 │             │
                      │                 ┌───────▼──────┐  ┌───────▼──────┐     │
                      │                 │  DMRG3S      │  │    CBE       │     │
                      │                 │  subspace    │  │  controlled  │     │
                      │                 │  expansion   │  │  bond        │     │
                      │                 │  (dmrg3s.py) │  │  expansion   │     │
                      │                 │  1-site only │  │  (cbe.py)    │     │
                      │                 │              │  │  arXiv:      │     │
                      │                 │  iDMRG-sym:  │  │  2403.00562  │     │
                      │                 │  NotImpl     │  │              │     │
                      │                 └───────┬──────┘  └───────┬──────┘     │
                      └────────────────────┬────┴─────────────────┴────────────┘
                                           │
                                           │  (effective eigenproblem H_eff ψ = E ψ;
                                           │   TDVP replaces eigh with krylov_expm)
                                           │
                      ┌────────────────────┼────────────────────┐
                      │                    │                    │
              ┌───────▼──────┐     ┌───────▼──────┐     ┌───────▼──────┐
              │ Lanczos      │     │ Lanczos      │     │ Cython fused │
              │ (Python/JAX) │     │ (JIT via     │     │ Lanczos +    │
              │ _lanczos_    │     │  lax.while)  │     │ matvec       │
              │  solve       │     │ _lanczos_    │     │ (opt-in; CPU)│
              │              │     │  solve_jit   │     │ TENAX_DISABLE│
              │              │     │              │     │ _CYTHON_BLAS │
              └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                     └────────────────────┼────────────────────┘
                                          │
                      ┌───────────────────┼───────────────────┐
                      │                                       │
              ┌───────▼──────┐                        ┌───────▼──────┐
              │ DenseTensor  │                        │ Symmetric-   │
              │ path         │                        │ Tensor path  │
              │              │                        │              │
              │ JAX einsum;  │                        │ _blockwise_  │
              │ JIT OK;      │                        │  contract;   │
              │ Cython       │                        │ numpy-       │
              │ opt-in       │                        │  blockwise   │
              │              │                        │  (default)   │
              └──────┬───────┘                        └──────┬───────┘
                     │                                       │
            ┌────────┼──────────┐                            │
            │        │          │                            │
     ┌──────▼──┐ ┌──▼────┐ ┌───▼──────┐              ┌───────▼──────┐
     │accel=   │ │accel= │ │accel=    │              │ U(1) charge  │
     │ "off"   │ │ "jit" │ │"sharded" │              │ sectors via  │
     │ Python  │ │ JAX   │ │ multi-   │              │ target_charge│
     │ loop    │ │ jit   │ │ device   │              │              │
     │         │ │       │ │ (dense   │              │ FermionParity│
     │         │ │       │ │  2-site) │              │ not wired    │
     │         │ │       │ │          │              │ (no fermionic│
     │         │ │       │ │          │              │  DMRG path)  │
     └─────────┘ └───────┘ └──────────┘              └──────────────┘

Legend:
  ┌────────┐  working path
  └────────┘

  NotImpl = raises NotImplementedError on entry
```

Dispatch between `DenseTensor` and `SymmetricTensor` is **implicit**:
`dmrg()` / `idmrg()` inspect the MPS and MPO tensor types at entry,
require them to be uniform, and route to `_dense_ops()` or
`_symmetric_ops(config)` accordingly. Mixed types raise `TypeError`.

## Status Summary

| Path                                   | Status           | Notes                                                              |
|----------------------------------------|------------------|--------------------------------------------------------------------|
| Finite DMRG 2-site (dense)             | **Working**      | Default path; best-tested; JIT + sharded accelerators available.   |
| Finite DMRG 2-site (symmetric)         | **Working**      | Block-sparse via `_blockwise_contract`; `numpy_blockwise` default. |
| Finite DMRG 1-site (dense)             | **Working**      | No bond growth; pair with DMRG3S or CBE to grow `chi`.             |
| Finite DMRG 1-site (symmetric)         | **Working**      | Same dispatch; use DMRG3S/CBE for expansion.                       |
| DMRG3S subspace expansion              | **Working**      | Hubig–McCulloch–Schollwöck; 1-site only; dense + symmetric.        |
| Controlled Bond Expansion (CBE)        | **Working**      | McCulloch–Osborne, arXiv:2403.00562; dense + symmetric.            |
| iDMRG 2-site (dense)                   | **Working**      | Default; 2-site unit cell only.                                    |
| iDMRG 2-site (symmetric)               | **Working**      | Block-sparse fixed-point solve.                                    |
| iDMRG 1-site (dense)                   | **Working**      | DMRG3S not wired in iDMRG path.                                    |
| iDMRG 1-site (symmetric)               | **BROKEN**       | Raises `NotImplementedError`.                                      |
| iDMRG `unit_cell_size > 2`             | **BROKEN**       | Raises `NotImplementedError`; 2-site cell hard-coded.              |
| TDVP 1-site                            | **Working**      | Real + imaginary time via `krylov_expm`; not in `__all__` yet.     |
| TDVP 2-site                            | **Working**      | Allows bond growth via SVD truncation.                             |
| JIT sweeps (dense, GPU/TPU)            | **Working**      | `accelerator="jit"`; PR #209.                                      |
| Sharded DMRG (multi-device)            | **Working**      | `accelerator="sharded"`; dense 2-site only.                        |
| Cython fused Lanczos (CPU)             | **Working**      | Opt-in; disable via `TENAX_DISABLE_CYTHON_BLAS=1`; PR #226.        |
| Fermionic DMRG                         | **ABSENT**       | No dedicated path; build fermionic MPO on `SymmetricTensor` by hand.|
| TEBD                                   | **ABSENT**       | Not implemented; use TDVP or iPEPS simple update.                  |
| Multi-state targeting (`num_states>1`) | **ABSENT**       | Raises `NotImplementedError`.                                      |
| Perturbative noise (`noise != 0`)      | **ABSENT**       | Raises `NotImplementedError`; use DMRG3S or CBE instead.           |

## Config Cheat Sheet

| Setting              | Flag                           | Default    | Options / notes                              |
|----------------------|--------------------------------|------------|----------------------------------------------|
| Bond dimension       | `max_bond_dim`                 | —          | int; hard cap on MPS bond                    |
| Sweeps               | `num_sweeps` / `max_iterations`| —          | finite DMRG / iDMRG respectively             |
| Convergence tol      | `convergence_tol`              | —          | energy-change threshold                      |
| Truncation error     | `svd_trunc_err`                | —          | discarded-weight threshold per SVD           |
| Sweep width          | `two_site`                     | `True`     | `False` = 1-site sweep                       |
| Subspace expansion   | `subspace_expansion`           | `False`    | DMRG3S; 1-site only                          |
| Mixing factor        | `mixing_factor`                | —          | DMRG3S / 1-site iDMRG perturbation           |
| Expansion extras     | `expansion_num_extra`          | —          | extra states kept during DMRG3S              |
| Hybrid mixing        | `hybrid_mixing`                | —          | DMRG3S hybrid schedule                       |
| Target charge        | `target_charge`                | `None`     | U(1) sector for SymmetricTensor MPS          |
| Lanczos iters        | `lanczos_max_iter`             | —          | per site eigenproblem                        |
| Lanczos tol          | `lanczos_tol`                  | —          | Krylov residual                              |
| Accelerator          | `accelerator`                  | `"auto"`   | `"off"` / `"jit"` / `"sharded"`              |
| Block backend        | `numpy_blockwise`              | `True`     | SymmetricTensor path; `False` = JAX einsum   |
| Cython matvec        | `TENAX_DISABLE_CYTHON_BLAS`    | unset      | env var; set to `1` to force Python loops    |
| Multi-state          | `num_states`                   | `1`        | `>1` → `NotImplementedError`                 |
| Perturbative noise   | `noise`                        | `0`        | `!=0` → `NotImplementedError`                |
| iDMRG cell size      | `unit_cell_size`               | `2`        | `>2` → `NotImplementedError`                 |
| iDMRG fixed-pt tol   | `arnoldi_tol`                  | —          | left/right transfer-matrix solve             |
| iDMRG reorth         | `orthogonalize_interval`       | —          | full-reorth period                           |
| TDVP time type       | `time_type`                    | —          | `"real"` / `"imaginary"`                     |
| TDVP time step       | `dt`                           | —          | complex allowed                              |

## Key Files

| Responsibility                         | File                                                   |
|----------------------------------------|--------------------------------------------------------|
| Finite DMRG driver + `DMRGConfig`      | `src/tenax/algorithms/dmrg.py`                         |
| iDMRG driver + `iDMRGConfig`           | `src/tenax/algorithms/idmrg.py`                        |
| DMRG3S subspace expansion              | `src/tenax/algorithms/dmrg3s.py`                       |
| Controlled Bond Expansion (CBE)        | `src/tenax/algorithms/cbe.py`                          |
| TDVP time evolution + `TDVPConfig`     | `src/tenax/algorithms/tdvp.py`                         |
| Krylov matrix exponential (TDVP)       | `src/tenax/algorithms/_krylov.py`                      |
| JIT-compiled dense sweep kernel        | `src/tenax/algorithms/_jit_sweep.py`                   |
| Block-sparse matvec (symmetric path)   | `src/tenax/algorithms/dmrg.py` (`_blockwise_contract`) |
| Cython fused Lanczos / matvec          | `src/tenax/contraction/_cython_blas.pyx` (+ `.pxd`)    |

## Related Documents

- [`docs/guide/algorithms/dmrg.md`](guide/algorithms/dmrg.md) — user guide
  for finite DMRG.
- [`docs/guide/algorithms/idmrg.md`](guide/algorithms/idmrg.md) — user guide
  for iDMRG.
- [`docs/guide/algorithms/tdvp.md`](guide/algorithms/tdvp.md) — user guide
  for TDVP time evolution.
- [`docs/guide/algorithms/auto_mpo.md`](guide/algorithms/auto_mpo.md) —
  building the Hamiltonian MPO fed to `dmrg()` / `idmrg()`.
- [`docs/ipeps-code-paths.md`](ipeps-code-paths.md) — companion
  architectural map for the iPEPS stack.
