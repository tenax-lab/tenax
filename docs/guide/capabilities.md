# Capabilities

This page is a high-level map of what Tenax can do today: the core tensor
engine, the 1D / 2D / classical algorithm suites, hardware acceleration, and a
candid note on the performance regimes where each path is (and is not) the right
tool. For step-by-step tutorials see the {doc}`algorithm guides <algorithms/dmrg>`;
for the full symbol list see the {doc}`API reference </api/index>`.

```{admonition} Experimental project
:class: warning

Tenax is under active development and largely written with the assistance of
Claude Code (AI). Verify results against known benchmarks before using them in
research.
```

## Core tensor engine

- **Block-sparse symmetric tensors** -- `SymmetricTensor` stores only
  symmetry-allowed charge sectors. Abelian symmetries: U(1), Z_n, and
  `FermionParity` / `FermionicU1` for fermions. A non-Abelian interface is
  stubbed for future SU(2).
- **Sector-based indices** -- `TensorIndex` keeps sorted charge sectors and
  multiplicities for O(n_sectors) lookups; `FuseInfo` records parent legs so
  `split_index` exactly reverses `fuse_indices`.
- **Polymorphic arithmetic** -- `+`, `-`, `*`, transpose, `inner()`, `conj()`,
  `dagger()`, `bar()`, and `max_abs()` behave identically on `DenseTensor` and
  `SymmetricTensor`, so algorithm code is agnostic to the storage backend.
- **Label-based contraction** -- legs are identified by string/integer labels;
  shared labels are contracted automatically (Cytnx-style), with `opt_einsum`
  path optimization for multi-tensor networks.
- **Containers** -- a `TensorNetwork` graph with contraction caching, and
  Cytnx-style declarative `.net` topology files (parse once, load tensors,
  contract repeatedly).
- **Decompositions** -- native symmetry-aware block-sparse SVD, QR, and `eigh`
  in `tenax.linalg`.
- **JAX-native** -- every tensor type is a registered pytree, so `jit`, `grad`,
  and `vmap` compose throughout.

## 1D and quasi-1D algorithms

- **DMRG** (finite), **iDMRG** (1D chain and infinite cylinder), three-site
  DMRG, **TDVP**, and **iTEBD** (numerically stable infinite TEBD, including the
  inversion-free Hastings update). Controlled-bond-expansion (CBE) support.
- **AutoMPO** -- build Hamiltonian MPOs from symbolic operator descriptions
  (custom couplings, next-nearest-neighbour, arbitrary spin); `symmetric=True`
  emits U(1) block-sparse MPOs.
- **Accelerated DMRG** -- JIT-compiled sweeps via `jax.lax.scan` for dense
  tensors and per-operation JIT for block-sparse tensors, with an automatic
  warmup-to-JIT transition while bond dimensions are still growing. Multi-GPU
  **GSPMD sharding** for large bond dimensions
  (`DMRGConfig(accelerator="jit" | "sharded")`).
- **Cython BLAS fast path** -- a fused Cython Lanczos solver and block-sparse
  contractions that call BLAS directly with no Python reentry, for
  high-performance CPU DMRG.
- **iDMRG transfer-matrix environments** -- fixed-point environment computation
  for self-consistent infinite boundary conditions.

## 2D algorithms (iPEPS / PEPS)

- **Simple update** -- imaginary-time evolution with 1-site or 2-site unit
  cells.
- **AD ground-state optimization** (`optimize_gs_ad`) -- gradient optimization
  via implicit differentiation through the CTM fixed point (Francuz et al., PRR
  7, 013237), for 1-site and 2-site unit cells.
  - Optimizers: **L-BFGS** (Hager-Zhang line search + metric preconditioning),
    **Adam** (cosine learning-rate decay), and **conjugate gradient**.
  - Backward: implicit AD via iterative VJP (default) or an optional GMRES
    route; **explicit AD** through unrolled CTM iterations for the 1-site C4v
    path.
  - **C4v paths**: a 2-site shared-tensor C4v mode (`unit_cell="2site"`,
    `gs_c4v=True`) optimizing a single tensor with the second sublattice derived
    by spin-π rotation (stable across χ = 8-24 for spin-1/2 AFMs), plus an
    opt-in dense reference mode (`ctm_ad_mode="c4v_reference"`) with a Krylov
    implicit backward.
  - Stability knobs: sigma gauge fixing (`forward_gauge="sigma"`) and a
    chi-ramping schedule for progressive refinement.
- **CTMRG projectors** -- SVD/Fishman (default), `eigh`, and a reduced-corner
  **QR-CTMRG** projector (`projector_method="qr"`, arXiv:2505.00494), usable
  both forward-only and under AD on the dense single-site path.
- **In-CTM χ-bump** (variPEPS §2.8.2) -- reactive growth of the CTM bond
  dimension *inside* CTM convergence (`CTMConfig.ctmrg_heuristic_increase_chi`),
  so the environment is always reconverged at the new χ before the optimizer
  sees it.
- **Split-CTMRG** -- ket/bra-separated environment tensors that cut the
  projector cost from O(χ³D⁶) to O(χ³D³) (Naumann et al., arXiv:2502.10298).
  The forward and energy entry points work on both `DenseTensor` and
  `SymmetricTensor`, for 2-site checkerboard and multisite (kagome PESS) cells.
  The split **AD ground-state** path is narrower: enable it with
  `CTMConfig(fuse_virtual_legs=False)` **together with** the dense single-site
  optimizer (`unit_cell="1x1"`), on either `gs_recipe` -- the default `"2x2"`
  since #746, or `"1x1"`. `SymmetricTensor` / fermionic inputs raise
  `NotImplementedError` (a later phase), and χ is fixed on this path. Its
  implicit gradient matches the trusted explicit-AD gradient to ~1e-12; the
  memory win over the fused double layer is a large-D effect (D ≳ 16).

  :::{warning}
  `gs_recipe="1x1"` is retained only for regression bisection and **must not be
  used for physics**. Its corner-pair projector collapses the environment to
  rank-1 corners, giving a χ_eff = 1 mean-field boundary whose energy is
  bit-identical across any range of χ. Nothing raises -- the collapsed
  environment is still finite, Hermitian and PSD. See #723, #726, #746, #747.
  :::
- **Fermionic iPEPS (fPEPS)** -- graded tensors with Koszul signs,
  `FermionParity` / `FermionicU1`, and a `spinless_fermion_gate` (hopping +
  interaction). See {doc}`algorithms/fpeps`.
- **Native honeycomb CTM** -- a rank-4, 6-corner, 3-direction, 2-sublattice
  CTMRG (`honeycomb_ctm_energy_implicit`) with a JIT-fused GMRES backward,
  replacing the brick-wall workaround.
- **PESS** -- kagome projected entangled simplex states.
- **Quasiparticle excitations** -- iPEPS excitation spectra at arbitrary
  Brillouin-zone momenta (Ponsioen et al., 2022).
- **Model gates** -- `heisenberg_gate`, `heisenberg_gate_u1sz` (U(1)-Sz
  block-sparse), `xxz_gate`, and `spinless_fermion_gate`.

## Classical statistical mechanics

- **TRG** and **HOTRG** coarse-graining for 2D classical partition functions
  (free energy, critical temperature, critical exponents), plus the underlying
  coarse-graining utilities.

## Hardware acceleration

- **CPU / CUDA / TPU / Metal** through the JAX backend.
- **Multi-GPU GSPMD sharding** for large bond dimensions in DMRG and dense CTM.
- **Cython BLAS** fast path for CPU DMRG.
- A CLI-driven {doc}`benchmark suite </guide/benchmarks>` for reproducible
  performance studies across all four backends.

## Performance characteristics and practical guidance

The algorithm surface is broad; the regimes below summarize where each path is
the most effective tool today.

- **Sweet spot.** Small-bond-dimension fermionic iPEPS (D ≲ 3-4), bosonic
  **dense** iPEPS, and the 1D/RG algorithms (DMRG, iDMRG, TRG/HOTRG) -- with
  strong throughput on TPU and multi-GPU.
- **Symmetry vs. dense.** Block-sparse U(1)-Sz iPEPS optimization pays off at
  D = 2 (faster as χ grows) but is currently slower than the dense path at
  D ≥ 3, where per-block dispatch overhead dominates. For D ≥ 3 bosonic work the
  **dense** path is the pragmatic default.
- **Large-D dense forward.** The **split-CTM** forward is the lever past the
  single-GPU χ wall -- roughly an order of magnitude less memory than the fused
  double layer at large χ.
- **Multi-GPU CTM-AD.** Sharding gives limited per-device memory relief for the
  CTM-AD backward, because the projector SVD forces the χ²·D⁶ intermediate to be
  replicated; it raises the achievable D by a small amount rather than removing
  the D⁶ wall.
- **Large-D fermionic.** Large-bond fermionic iPEPS is currently impractical in
  the JAX path (steep compile wall, host-orchestration-bound, and no dense
  fallback for fermions). For large-D fermionic systems, an eager PyTorch
  fermionic-PEPS code (e.g. YASTN/peps-torch) is the better tool today.

## Where to go next

- {doc}`Installation </guide/installation>` and {doc}`Quickstart </guide/quickstart>`
- {doc}`Core concepts </guide/core_concepts>` -- symmetries, indices, tensors
- Algorithm tutorials: {doc}`DMRG </guide/algorithms/dmrg>`,
  {doc}`iPEPS </guide/algorithms/ipeps>`,
  {doc}`CTM </guide/algorithms/ctm>`,
  {doc}`fermionic iPEPS </guide/algorithms/fpeps>`,
  {doc}`AD optimization paths </guide/algorithms/ipeps_ad_paths>`
- {doc}`API reference </api/index>`
