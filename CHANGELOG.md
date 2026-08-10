# Changelog

## Unreleased

### ⚠️ Published performance claims retracted

- **The v0.8.2 iPEPS multi-GPU / split-CTM frontier numbers were measured
  against a collapsed CTM environment and are retracted** (#747, #825). The
  #672 benchmark ran `recipe="1x1"` on both arms; that projector drives the
  environment to a rank-1 corner (#723/#726), i.e. a χ_eff=1 mean-field
  boundary, so its (D,χ) reach figures described how cheaply *that* scales
  rather than the reach of a converged CTM. Re-derived at `recipe="2x2"` on
  A100-80GB, with `corner_rank` gated and every ceiling walled:

  | claim as published | re-measured |
  |---|---|
  | "split-CTM on 1 GPU beats dense on 2 across the whole (D,χ) frontier" | **ties** dense-2GPU at D=8 (χ=96) and D=10 (χ=48); wins **2× only at D=12** |
  | frontier reach D=10 χ≥128, D=12 χ96 | **D=8 χ=96, D=10 χ=48, D=12 χ=32** |
  | split-1GPU ceilings χ=224 (D=8) / χ=128 (D=10), compile-bound | **χ=96 / χ=48**, and **memory**-bound — every autotuner wall converts to a plain OOM under `--xla_gpu_autotune_level=0` |
  | dense shard relief ~1.1–1.4×/device, never extends D-reach | **0.88–1.77×/device** (at D=12 sharding *costs* 14%); extends reach once, D=8 χ=64→96 |

  **The multi-GPU CTM-AD NO-GO itself is unaffected** — it rests on the
  SVD-replication mechanism, which is recipe-independent. What changed is the
  split-vs-dense comparison layered on top of it. Split-CTM is still the better
  path on equal hardware (1.5× in χ at D=8, 2× at D=12) and still matches a
  2-GPU dense run using one GPU at D=12; the advantage is simply much smaller
  than published, and it *erodes with χ* — 2.66× in peak at χ=16 down to 1.02×
  at the ceiling.

  One caveat on attribution: a same-recipe control showed that replacing the
  BFC allocator with `cuda_async` alone moves the *dense* arm (D=12 χ=16 goes
  autotuner-fail → OK at unchanged `1x1`), so #672's "dense cannot compile even
  χ16 / split is the only path that runs D=12" falls to the allocator rather
  than the recipe. The split-side collapse is cleanly the recipe — its controls
  reproduce to ≤1.8%.

### Fixes

- **Simple update converged to the product state; every `ipeps()` result is
  affected** (#667). Two independent defects in the simple-update path:

  1. The sweep drove only 2 of the checkerboard's 4 bonds — `A.r↔B.l` and
     `A.d↔B.u`, never `B.r↔A.l` or `B.d↔A.u`. A was therefore always the
     left/top site of every gate and only ever picked up bond weight on its
     `r`/`d` legs, leaving **half the lattice bonds with no Schmidt weight at
     all**. That state is spuriously dimerized (the two horizontal bonds differ
     by 2.5e-2), its 1-site RDM depends on which bond you trace it from
     (‖ρ_A(h) − ρ_A(v)‖ = 0.37), and that RDM has a **negative eigenvalue**
     (−0.094).
  2. `Γ` absorbed `sqrt(σ)` that was *also* stored as the new λ and re-absorbed
     in full on the next sweep, so the shared bond carried `λ^1.5`. This is what
     drove the state to a product state: λ₂ ∝ dt, λ₃ ∝ dt², and E → **−0.5
     exactly** as dt → 0. It is why *smaller dt was worse* — dt was the only
     thing entangling the state.

  Neither correction works alone. Together, D=2 Heisenberg goes **−0.548 →
  −0.659334287637** (reference −0.6599; the residual is ordinary simple-update
  error), D=3 → −0.663196204307 and D=4 → −0.6674, χ-converged with corner rank
  = χ. The state is now dt-stable (−0.6570 / −0.6593 / −0.6591 at dt =
  0.3 / 0.05 / 0.01).

  **Two consequences for existing results.** Any nominally-D=3 state from
  `ipeps()` was really D=2 — its third Schmidt value was ~2e-6. And
  `sublattice_rotate_gate` now does what it promises: the rotated-frame state is
  uniform, so `A` and `B` are the same physical tensor (their 1×1 CTM energies
  agree to 1e-16), which makes the single-site/C4v route usable. `‖A−B‖` is not
  the way to check that — it measures the bond gauge and stays ≈1.7 either way.

  Not fixed here, and newly visible now that the state is genuinely entangled:
  the legacy 2-site `ctm_2site` that `ipeps()` reports its energy through does
  **not converge** on such a state (diff 1.6e-3…6.5e-2 after 600 sweeps at any
  χ up to 32), so `ipeps()`'s returned energy sits ~0.02 above the state's true
  value. Measure with `ctm_tensor(recipe="2x2")` instead. `fermionic_ipeps.py`
  absorbs `sqrt(σ)` the same way and is an untested lead for #392.

  One characterization worth knowing if you compare CTM paths: **split-CTM and
  fused-CTM agree only as χ→∞.** On a genuinely entangled D=2 state their
  energies differ by 1.67e-05 at χ=8, falling to 9.62e-09 by χ=48 — and the gap
  is identical to 12 digits at max_iter 100/400/2000, so it is finite-χ
  truncation, not non-convergence. This was invisible while the state was
  collapsing, because a near-product state made the two paths trivially
  identical.

## v0.8.3 (2026-08-09)

A correctness release. The headline feature is root implicit AD for CTMRG
(#715), but the reason to upgrade is the set of silent-wrong-answer defects
closed alongside it.

### ⚠️ Results affected — please re-check

Three defects in this cycle produced **plausible, non-crashing, wrong numbers**.
If you have results from v0.8.2 or earlier, check whether they came through any
of these paths.

- **DMRG returned a state that was not the one whose energy it reported**
  (#816). If you passed an explicit `phys_charges` to `build_auto_mpo`, the MPS
  and MPO could disagree on the *order* of the physical charges. Symmetric
  contraction pairs sectors by charge *value*, so DMRG still converged to the
  true ground state and reported the true energy — while `DMRGResult.mps` came
  back in a **permuted physical basis**. Occupations, correlators and
  entanglement read off that state were silently wrong, and the energy check
  that would normally catch it *passed*. Found while measuring orbital-cut
  entanglement of an FQH ground state, where energies reproduced ED exactly at
  every size. Not fermion-specific: an XXZ chain hides it only because its
  Sz=0 ground state is spin-flip symmetric.

- **The SVD and eigh adjoint kernels were wrong** (#750, #751, #752, #753) —
  AD gradients disagreed with finite differences. Affects any
  gradient-based optimization that went through those decompositions.

- **The 1x1 CTM recipe collapses the environment to rank-1 corners** (#723,
  #726, #747) — producing χ-independent, effectively mean-field energies on
  what was the default path. The showcase energies computed on it have been
  retracted (#771); the performance results stand. `ctm_tensor` now defaults to
  the 2x2 recipe. **Re-runs are still outstanding (#747), and the split-CTM
  rank-1 collapse (#726) remains open** — this cycle changed the default and
  retracted the affected numbers; it did not re-derive them.

### Behavior changes

Several configurations that previously ran now raise. In every case the old
behavior was to accept the setting and silently do something else.

- `ctm_ad_mode="root_implicit"` rejects `gs_optimizer="cg"` and `gs_c4v=True`
  (#792). CG ran plain gradient descent under the CG name; nothing projected
  into the C4v basis. `gs_line_search` **warns** rather than raising, because it
  is effectively default-on and refusing it would reject the path's own default
  configuration.
- `dmrg()` rejects an MPS and MPO whose physical legs carry different charge
  orders (#816). Remedy: `build_random_symmetric_mps` gained `phys_charges`, so
  both can be built from the same array.
- `FiniteMPS.norm()` raises on an overlap that is complex, negative, or exactly
  zero from non-zero tensors (#819) — all impossible for a true `<psi|psi>`,
  all previously laundered into a plausible float by `abs()`.
- A masked (non-finite) gradient no longer ends a root-implicit optimization
  (#812), so a contaminated run consumes more of its step budget instead of
  exiting early having optimized nothing.

### Features

- **Root implicit AD for CTMRG** (#715, arXiv:2607.15030) — the converged
  environment is characterised as the root of an algebraic equation and the
  gradient comes from a single un-nested linear solve, so **no SVD or eigh
  backward appears in the gradient path**. Phase 0 (C4v), Phase 1
  (asymmetric), Phase 2 (unit cell, 2x2 FD parity 5.6e-10), Phase 3 slice 1
  (`SymmetricTensor`), wired behind `ctm_ad_mode="root_implicit"`. This is a
  *stability* feature, not a speed one: it removes the degenerate-singular-value
  failure class outright. **#715 remains open**: the default `recipe="2x2"`
  performs three SVDs per direction while the paper derives the characteristic
  equations for a single truncated SVD, and nothing yet detects an inaccurate
  root-implicit gradient (#785).
- **Charge-arithmetic boundary on `BaseSymmetry`** (#733, #734) — all charge
  arithmetic goes through the symmetry, including mixed-flow `ProductSymmetry`
  and fusion.
- **Collapsed-CTM environment detection** (#770) instead of silently reporting
  collapsed environments.

### Fixes

- **`FiniteMPS.norm()` returned exactly 0.0 for symmetric MPS** (#819) — the
  bra was built with `conj()`, which conjugates the data but leaves the flows
  alone, so every physical pairing was OUT-against-OUT, no block matched, and
  the overlap collapsed to `0j`. Now uses `bar()`. `norm()` also takes the
  orthogonality-centre fast path when one is set (O(χ²d) instead of O(Lχ³)),
  following ITensor.
- **Mixed real/complex multi-operand einsum crashed on CUDA** (#813) — JAX
  propagates the final complex result type onto an intermediate `dot_general`
  whose own operands are both real, and cuBLASLt has no such kernel. Took out
  22 gauge tests on GPU while CI (CPU-only) stayed green.
- **Root-implicit gates failed open** (#796, #787, #784) — a NaN residual
  passed the gate; `gauge_consistency` reported a NaN cotangent as 0.0
  (perfect); the multisite gate kept a pre-#778 tolerance it could never meet.
- **Implicit-AD backward discarded the GMRES convergence flag** (#801) — an
  unconverged adjoint yielded a gradient wrong by roughly the residual, with
  nothing downstream able to tell.
- **Root-implicit NaN gradients on physical simple-update states** (#772, #779)
  — the retained CTM spectrum is now rank-capped and the residual gate follows
  the clamp that sets its floor.
- **RDM symmetrised before trace-normalising** (#725, #742, #748) — 8 further
  sites routed through `_normalise_rdm`, plus a gauge leak in `_normalise_rdm`
  itself.
- **Block-sparse decompositions ignored `left_labels`/`right_labels`** (#689).
- **C4v enlarged corner did not absorb the site tensor** (#760).
- **Asymmetric root-implicit AD on complex states** (#721), and the env
  convention at the energy boundary (#718) — gradient 3.06e-2 → 4.1e-8.
- **Elementwise CTM convergence criterion** could not converge on forward-only
  scans (#780, still open for the underlying D=4 case); `ms_per_sweep` reported the best-metric iteration rather than
  the sweeps performed (#781).
- **`ipeps_optimize` ↔ `ipeps_optimize_root_implicit` import cycle** (#790) —
  shared helpers hoisted into `_ipeps_optimize_shared`; the architecture guard
  had been red since #773 and so could not catch a new cycle.

### CI / tests

- **A test file in no bucket now fails CI** (#805, #806). Files absent from the
  bucket table were silently deselected by `-m core` and ran in no required
  job — which is how #790 and #803 sat red unnoticed. 95 files remain in
  `_UNBUCKETED_LEGACY` (#805).
- Phase 2 cell-shift tables gated in required CI (#730).

## v0.8.2 (2026-07-29)

A consolidation release: the direction-dependent CTM work from v0.8.1's cycle
is finished and its two silent single-site regressions are closed, alongside
additive features (multi-GPU HOTRG, 2-site split-CTM, Potts helpers).

### Fixes

- **Symmetric CTM χ-bond padding uses vacuum, not the sorted tail** (#700, #708) —
  the U(1)-Sz single-site environment collapsed to zero at D=3, χ=12 when the
  χ-bond was padded with sorted-tail charges instead of the vacuum sector.
- **Sweep-invariant 2x2 CTM corner label convention** (#702, #710) — the TOP
  move's C1, RIGHT's C2, and BOTTOM's C4+C3 wrote their corners with the two
  legs swapped relative to the init convention that `_build_enlarged_corner`,
  the move reads, and the energy/RDM contraction all assume. Pre-#676 the
  defect was uniform (a tolerated global transpose); #676 fixed only C3, making
  it non-uniform and rendering the single-site fixed point trajectory-dependent.
  One cause behind all three symptoms: the χ-bump forward drift (7e-5 → 2e-14),
  the ~10x-off implicit-AD gradient, and the D=2 sharding-parity blowup. Also
  resolves the long-standing **#425/#426 random-init plateau** — the sweep now
  converges to sv_diff ≈ 1e-6 instead of stalling at 0.05, so three plateau
  xfails became genuine convergence guards. Note the single-site fixed point
  moves (the prior value was the transposed-scheme artifact), and mutual
  bump-vs-fixed *gradient* agreement now floors at ~1e-3 because the restored
  ket-bra Z2 produces paired-degenerate projector SVs (#425 mechanism,
  backward-only; both gradients stay FD-consistent).
- **Direction-dependent 2x2 CTM bond bookkeeping** (#667, #670, #671, #676, #677) —
  canonical env tiling + 2x2 init corner consistency, `bottom_left` enlarged-corner
  pairing, and the fermionic/fused bottom-absorption C3 pairing (`c3_u`↔`t2_d`),
  so the block-sparse U(1) sweep runs on asymmetric-corner (`A.l != A.r`) states.
- **GPU SVD routed through the QR algorithm** (#691, #693) — avoids cuSOLVER
  `gesvdj` returning NaN.
- Split-CTM review fixes: 2-site TBPTT guard rejects only real truncation (#694, #696).

### New Features

- **2-site split-CTM, Phases 0–4** (#684, #685, #688, #690) — joint dense
  forward, dense AD, `SymmetricTensor` forward, and fermionic forward
  (merge → fused → resplit), extending the χ²·D⁴ split path to 2-site cells.
- **Multi-GPU HOTRG** (#681, #682) — `HOTRGConfig(device_mesh=...)` shards a
  surviving leg of the χ⁶ merge under GSPMD: ideal **2.00×/device** memory
  relief on 2×A100 with numerically identical results, and a real χ-ceiling
  extension (χ=40 OOMs replicated, fits sharded). Works where CTM-AD sharding
  did not because coarse-graining is forward-only — no AD-through-SVD backward
  to replicate the dominant operand.
- **Potts model helpers** (#686) — `compute_potts_tensor` and
  `potts_critical_beta` for q-state Potts TRG/HOTRG.
- **Chunked dense-CTM edge absorption** (#665, #668) — forward, 1×1.

### Research / Docs

- **Multi-GPU CTM-AD characterized as NO-GO** (#632, #672, #673, #678, #680).
  Root cause is path-agnostic: XLA lowers the projector SVD as unshardable and
  all-gathers whatever GSPMD sharded. TRG is likewise not a sharding target
  (no χ⁶ intermediate); HOTRG is, and shipped above. **The NO-GO stands, but
  two figures published in this entry were measured against a collapsed CTM
  environment and are retracted — see Unreleased (#747, #825).** They were:
  "dense sharding gives only ~1.1–1.4×/device and never extends D-reach"
  (re-measured: 0.88–1.77×/device, and it does extend reach once, D=8 χ=64→96)
  and "split-CTM on 1 GPU beats dense on 2 across the whole (D,χ) frontier"
  (re-measured: split *ties* dense-2GPU at D=8 and D=10 and wins 2× only at
  D=12).
- `CLAUDE.md` / `AGENTS.md` rightsized for Claude 5-generation context
  engineering (#709), and the documented merge command corrected — never pass
  `--delete-branch` with a merge queue, which closes the PR instead of merging
  it (#711).

### Tests / Chore

- #692 flake and regression triage across the sharding, AD, and CTM-plateau
  suites (#695, #697, #698, #699, #701, #703, #704, #706).
- CI coverage for `gs_recipe="2x2"`, the default CTM recipe (#707).

## v0.8.1 (2026-06-30)

### New Features

- **Single-site split-CTM AD** (#647, #648, #651, #652, #653, #657, #662) —
  opt-in via `CTMConfig(fuse_virtual_legs=False)`: the whole 1-site optimizer
  (`unit_cell="1x1"`, `gs_recipe="1x1"`) runs through the split χ²·D⁴
  double-layer forward instead of the fused χ²·D⁶. Explicit and implicit
  (Γ-gauge-fixed `custom_vjp`, Neumann backward) gradients agree to ~1e-12;
  the warm-start, line-search probe, and final environment all route through
  the split forward (returns `SplitCTMTensorEnv`), behind a single consolidated
  config validator. Validated production-correct with C4v (variational window).
  `DenseTensor` only, fixed χ; the memory win over the fused path is a large-D
  (D≳16) effect. (#644/#641 bound the dense split forward to χ²·D⁴; #640/#642
  add the CG-kagome large-D memory harness + gradient parity.)
- **1-site checkpoint/resume incl. the coarse-grained (`cg_gates`) path** (#643, #497).
- **Multi-GPU large-D dense CTM (GSPMD)** (#632, #635, #637) — GSPMD-sharded
  forward (rung 1) and backward (rung 2) for `optimize_gs_ad`, plus an
  AD-stable randomized truncated low-rank SVD. (A100 verdict: correct but a
  weak large-D lever at present — see #634.)
- **`compute_excitations` accepts `optimize_gs_ad` Tensor outputs** (#639, #636).

### Performance / Research

- D=3 χ-convergence study + 1-site grad-spike guard (#645); Heisenberg
  scaling/perf showcase (recipe=1x1 + multi-GPU) (#638).
- #566 compile-wall investigation — NO-GO findings: batched dispatch can't
  close the D≥3 U(1)-Sz CTM-AD warm wall (#625, #626, #627), C-adjoint NO-GO
  (#630), padded-vmap spikes (#631, #633).

### Docs

- Split-CTM AD optimization-path docs + corrected stale "explicit AD is the
  default" claims (#654, #656); Capabilities overview page (#661).
- Heisenberg χ-scaling benchmarks: D=4 (#646, #649), D=8 (#650).

### Fixes / Chore

- Green chronic flaky/macOS tests (#658, #659); remove scratch study/profiling
  artifacts (#660); revert scratch 2×2 benchmark scripts (#629).

## v0.8.0 (2026-06-18)

### New Features

- **1D iTEBD dense reference** (#583, #619) — translation-invariant
  infinite-MPS imaginary-time evolution with a 2-site unit cell:
  `itebd_groundstate` (Vidal `Γ-λ` form with a regularized `_safe_inv`
  pseudo-inverse, no `λ^-1` blow-up) and the inversion-free
  `itebd_groundstate_hastings` (`_update_bond_hastings`, arXiv:0903.3253).
  Both converge to the spin-½ Heisenberg `e₀ = ¼ − ln 2` and agree
  bit-for-bit across the truncation regime.
- **Reduced-corner QR-CTMRG** (#570) — faithful QR-based CTMRG: Phase 1
  dense 1×1 (#595) and Phase 2 dense multisite with implicit-diff AD
  (#597); `regularized_qr` and a `recipe`/`gs_recipe` knob. (Block-sparse
  QR ruled out as a compile lever — see Research.)
- **U(1)-Sz symmetric CTM enablement** (#566, #602) — chi-bond order
  canonicalization unblocks U(1)-Sz iPEPS-AD (#604) and the symmetric 2×2
  plaquette projectors are applied unfused (#608); D=3 Heisenberg runs.
- **Opt-in batched block-sparse execution** (#568, #569) behind the
  `TENAX_BATCH_BLOCKSPARSE` gate — batched per-sector contraction (#571)
  and batched SVD/QR/eigh over same-shape charge sectors (#572).
- **Block-sparse contraction backend seam + cuTensorNet GPU backend**
  (#200) — pluggable contraction backend (#586) and a validated,
  default-OFF cuTensorNet path (#587).
- **iPEPS optimizer knobs** — `gs_line_search_method='auto'` (Armijo at
  low χ, Hager–Zhang at high χ) (#549); `gs_ctm_max_iter_schedule` for a
  late-stage CTM-iter cap (#540); `return_history` extended to the
  multisite optimizer (#550); implicit-AD warm-start invalidation API
  (#535); in-CTM chi-bump kwargs threaded into the PESS AD loss (#548).

### Behavior Changes

- **`hatch-vcs` versioning** (#530) — the git tag now drives the package
  version.
- **CTM χ-control consolidation** (#547) — `chi_ramp` and `chi_auto_bump`
  deprecated (Phase 1) in favor of the unified schedule.
- **iPEPS AD** — symmetric/fermionic first-step compile is no longer
  silenced (#567); stall reset now requires a strict decrease above the
  noise floor (#536); `hager_zhang` `bracket_only_phi` default reverted to
  `False` (#541).
- **linalg** — `max_truncation_err` is honored when `base_charges` is
  supplied (#561).

### Bug Fixes

- **contractor** — multi-input `contract()` no longer silently returns 0
  on disjoint inputs (#554).
- **fermionic** — remove spurious auto-Koszul signs (#557); drop the
  split-CTM shim fallback now that the algebra is closed (#562); preserve
  the canonical bond layout in simple-update at D>2 (fPEPS #560, iPEPS
  #564).
- **U(1)-Sz CTM correctness** — bond-order fix (#604) and unfused
  2-plaquette projectors (#608).
- **iPEPS** — unblock in-CTM chi-bump on the 2-site implicit-AD path
  (#542); restore the Arnoldi precheck raise on high ρ (#532, #533).
- **AutoMPO** — `compress=True` is no longer incompatible with
  `symmetric=True`: charge-aware bond-charge tracking through compression
  fixes the `SymmetricTensor.from_dense` shape mismatch (#621, closes #620).
- **build** — pin upper bounds on `build-system.requires` to fix a twine
  Metadata-Version 2.5 failure (#577).

### Performance

- **`_gauge_fix_symmetric_svd` vectorized** (#593) — per-column → per-sector,
  ~11–23× fewer HLO instructions.
- **fuse/split vectorized** (#573) — scatter/gather loops replaced with
  vectorized ops.
- **iPEPS line search** — share the forward CTM env across HZ φ/φ′ probes
  (#538), skip φ′ during the bracket phase (#539), cap CTM iters for
  probes (#537); TBPTT truncation for the explicit-AD CTM backward (#546).
- **split-CTM** — restore the split-aware mixed-env RDM with a trace-floor
  fallback (#545).

### Tests / CI

- Bosonic split-vs-shim energy/gradient parity (#581); Jordan-Wigner ED
  reference scaffolding (#556); stall-runaway regression canary (#544);
  hardened macOS-fragile assertions (#531); D=3 U(1)-Sz energy-vs-dense
  guard (#611); fast reduced-corner QR tests marked `core` (#596).
- **AI-comment marker hook** (#623) — `shlex`-parsing `PreToolUse` hook
  that labels AI-authored `gh` comments with a 🤖 marker; fires only on an
  actual `gh` posting command and checks the `--body` value (dev tooling).

### Documentation & Research

- **Research spikes / NO-GO findings** (#566, #570, #610) — stacked
  block-sparse feasibility NO-GO (#609); CTM env de-fragmentation
  NO-GO-by-obstruction (#614); block-sparse QR Phase-3 NO-GO (#598);
  CTM-AD compile-wall diagnosis (the wall is #566 structural per-block
  emission, not the decomposition) (#588–#592); dense Heisenberg large-D
  characterization (#599, #600); cuTensorNet NO-GO measurement (#587).
- **Benchmarks** — #566 batching harness + A100/H100 results (#574–#580,
  #584, #585); U(1)-Sz vs dense D-χ scaling (#607).
- **iTEBD docs** — correct the Hastings attribution on the Vidal path
  (#616); document the `B_new` gauge in the Hastings update (#622).
- **Docs** — swept-env T1/T3 flow convention (#613); implicit-AD
  variational-guarantee precondition (#534); stall-recovery rationale
  (#552); minimal working fermionic fPEPS (AD) example + simple-update
  breakage note (#582).

## v0.7.0 (2026-05-25)

### New Features

- **2-site implicit-AD safety nets** (#524) — three optional
  `iPEPSConfig` knobs catch the chi-ceiling collapse mode on the 2-site
  bipartite implicit-AD path:
  - `gs_chi_ceiling_bailout: int` — exit to `best_params` after K
    consecutive steps at `chi == chi_max` with the bump indicator
    above `chi_auto_bump_eps` (variPEPS §2.8.2 mechanism).
  - `gs_grad_spike_ratio: float | None` + `gs_grad_spike_window: int`
    — reject any step whose `||grad||₂` exceeds
    `ratio × max(median, 1.0)`; roll back to `best_params` and clear
    L-BFGS history.
  - `gs_hz_max_iter: int` — cap Hager–Zhang inner iterations
    (default 40; recommend 15 on chi-saturated implicit-AD).
  - Per-step GMRES amplification diagnostic (`||lam||` /
    `amp = ||lam|| / ||initial_lam||`) logged at each `gs_log_interval`.
  - Production recipe lands at E/site ≈ −0.6690 (≈ QMC) for D=3
    spin-½ Heisenberg in ~1.5 h on a single GPU; see
    `docs/guide/algorithms/ipeps_ad_paths.md` for the v6–v9
    benchmark ladder.
- **`CTMConfig.chi_auto_bump_metric`** (#525) — selects between
  `"eps_T"` (default, discarded SV tail mass) and `"norm_smallest_S"`
  (variPEPS-literal trigger `s[χ−1]/s[0]`) for the reactive χ_E bump.
- **In-CTM χ auto-bump in AD forward loops** (#492, #513, #514, #515,
  #516, #517) — variPEPS §2.8.2 reactive χ_E bump runs inside
  `python_loop_ctm_converge` for both explicit and implicit AD
  forwards.  `CTMConfig.chi_auto_bump`, `chi_auto_bump_eps`,
  `chi_auto_bump_step` knobs, with `chi_max` ceiling enforcement
  and zero-padded warm-restart.
- **Convergence-triggered adaptive χ ramping** (#455, #459, #462,
  #464, #465, #466, #467) — replaces the fixed-step chi schedule
  with a unified `_maybe_scheduled_bump` helper that advances stage
  on convergence + stall criteria; per-stage state tracking; reset
  of stall_count and L-BFGS history at chi-bump boundaries.
- **Grad-norm AD outer convergence criterion** (#449) — close on
  `||grad||_2 < gs_grad_norm_tol` instead of `|dE|`; closes #448.
- **iPEPS checkpoint primitive** (#497) — long-run AD resume via
  serialized optimizer state, env cache, and L-BFGS history.
- **2x2 plaquette CTM projector** (#406, #416, #434, #447) — variPEPS
  multi-site projector that lifts the kagome 3-site multisite CTM
  above the spin-½ AFH floor; `SymmetricTensor` support; `stop_gradient`
  on the projector outputs to break the implicit-AD NaN cluster.
- **F2/F3 fixed-point and JIT-fused implicit-AD CTM backward** (#415,
  #421) — restructured backward as `(I − J^T) λ = ∂E/∂env` solved by
  a single JIT-fused GMRES call.
- **Warm-start implicit-AD adjoint** (#501) — reuse previous step's
  λ as initial guess; shape-validated against stale-cache (#515 review).
- **Coarse-grained iPEPS for honeycomb / kagome** (#352, #353) —
  CG-iPEPS supersite encoding with split-aware energy at the supersite;
  closes the gap to variPEPS honeycomb references.
- **Native rank-4 honeycomb iPEPS CTM with implicit AD** (#347) —
  6-corner CTMRG for honeycomb with `jax.custom_vjp` and JIT-fused
  GMRES backward; replaces the brick-wall dummy-bond workaround.
  Configurable `energy_fn` hook for kagome iPESS triangle energies.
- **Differentiable iPESS for kagome XXZ** (#387, #398, #403) — kagome
  iPESS via CG-iPEPS supersite with marginalised-3-site formula;
  reaches the Liao 2017 PRL 118 137202 reference at D=4 χ=16.
- **Split-aware CTM energy at large D** (#390, #392, #393, #394, #397)
  — replace the shim path with native split-CTM energy at the
  χ²·D⁴·d peak; fits D=8/10 kagome.
- **Rank-aware SVD truncation** (#400) — prune zero modes from the
  kept singular-value set so the F-matrix backward stays well-conditioned.
- **`info.max_truncation_error` and reactive-bump events on the 2x2
  plaquette path** (#474, #481, #484) — real `ε_T` (was 0.0 stub) +
  INFO-level logging of every chi-bump event.
- **HZ probe count + ||grad||_2 in step output** (#495, #498, #500) —
  visibility into Hager–Zhang inner-iter cost and per-step gradient
  norm trend.
- **iPEPS AD benchmark harness + JIT cache fix + history hook** (#414).
- **iPEPS-AD example series (v6–v9)** — examples covering chi-schedule
  + reactive bump composition (v6), fixed-χ implicit/explicit AD
  comparison (v7), chi-ramp on the 4-bond-fixed energy (v8), and
  safety-net validation across bipartite / C4v + metric variants (v9).
- **`bar_super()`** with super-algebra Koszul twist for fermionic
  `SymmetricTensor` (#361).

### Behavior Changes

- **Count all 4 NN bonds in 2-site bipartite energy** (#493, #494) —
  earlier 2-bond formula under-counted; explains historical
  `gs_c4v=False` / #328 sub-QMC drift on bipartite paths.
- **CTM sweep order standardised** to (left, top, right, bottom) (#407).
- **`chi_auto_bump` end-of-iter** ordering preserves Wolfe invariant
  in the surrounding line search (#419, #432).
- **`plateau_patience` early-bail** in `python_loop_ctm_converge`
  (#439).
- **L-BFGS stall recovery** — cap on consecutive CTM-error resets to
  avoid runaway rollback loops (#454, #457).
- **Stop-gradient on 2x2 plaquette projector outputs** to break the
  implicit-AD NaN cluster (#447).

### Bug Fixes

- **Stall-rollback env-cache desync** — clear `_env_cache_2s` on
  best-rollback so the next CTM call cold-starts at the current
  `ctm_cfg_2s.chi` instead of using a stale-χ snapshot (#518, #519).
- **`chi_auto_bump_metric` `getattr` fallback** in the chi-ceiling
  bail-out path so PR #524 stands alone before PR #525 lands (#524
  follow-up).
- **Reset `chi_ceiling_consecutive_2s` on CTMRGGradientError
  recovery** (#524 follow-up) — matches the docstring contract for
  stall recovery.
- **`gs_grad_spike_window=1`** honoured (#524 follow-up) — relax
  `len >= 2` gate to `len >= 1` so the minimal window doesn't
  silently disable the guard.
- **Implicit-AD diagnostics toggle** restored via `try/finally`
  (#524 follow-up) — `_F3_DIAG_COMPUTE_NORMS` no longer leaks `True`
  if an exception escapes the optimizer loop.
- **Fall back to `eps_T` when `norm_smallest_S` indicator is missing**
  (#525 follow-up) — non-2-site callers no longer silently disable
  reactive bumps when the variPEPS-literal metric is selected.
- **Three CTM-iteration bugs preventing random-init convergence**
  (#422, #423, #424).  Closes long-standing issues but per-quadrant
  ket/bra ordering vs variPEPS rotates SVD basis in degenerate
  subspaces; documented as known-limitation in #425/#426.
- **Tracer-safe symmetric 2x2 projector** (#440) — closes #435.
- **Fill unused base_charges budget in traced symmetric SVD** (#445).
- **Thread base_charges through `_ctm_tensor_sweep`** for ε_T parity
  (#437).
- **Pad stale `best_env_cache` to post-bump χ** (#476) — closes #469
  warm-start mismatch.
- **Restore Arnoldi precheck raise on high ρ** (#477) — closes #469.
- **Match `_qr_projector_symmetric` output to `eigh` path** (#478) —
  closes #469.
- **Rebalance vertical/diagonal RDM contractions** to the χ²·D⁴·d²
  floor for kagome (#389).
- **Route fermionic energy through the split-CTM shim path** (#392,
  #394).
- **Canonicalise SVD-bond charges** so ket and bra agree on the split
  path (#391, #393).
- **`_flow_flip_no_conj` works on `SymmetricTensor`** (#423).
- **`rank-1 chi_init`** for standard CTM (#424).
- **`stop_gradient` 2x2 projectors** to fix implicit-AD NaN cluster
  (#447).
- **PESS-AD `_initial_alpha` floor** when `|energy|` collapses to 0
  (#401, #404).
- **PESS-AD accepts `Tensor`-valued bond gates** in 3-site multisite
  (#405).
- **SymmetricTensor SVD projector gauge** + `n_blocks==0` fallback
  (#408, #409).
- **CTM-AD eager-GMRES fallback** on slow Neumann (#420, #427).
- **SV-baseline guard** + PESS `plateau_patience` pass-through (#442).
- **Auto-χ_E bump end-of-iter ordering** (#419, #432).
- **`split-ctm` `_rdm1x2`/`_rdm2x1` shim delegation** (#479, #485,
  #486).
- **Pair symmetric MPO with symmetric MPS** in DMRG benchmarks (#508).
- **Reset `stall_count` on reactive auto-bump** (#465).
- **Reset `stall_count` + clear L-BFGS state on chi-schedule bump**
  (#464).
- **Chi-schedule shim off-by-one** — bumps were one stage late (#462).
- **Codex follow-ups on #457 / #459** (#461).
- **#449 codex follow-ups** (c4v_reference + multisite warmup) (#451).

### Performance

- **`_tree_dot` via host NumPy** — ~143× faster per call (#450).
- **F2 fixed-point + F3 fused-backward** for implicit-AD CTM (#415,
  #421).
- **Split-aware energy at large D** — replaces shim's χ²·D⁶ cost
  with native χ²·D⁴·d (#390).
- **Unified χ-schedule via `_maybe_scheduled_bump`**, pad envs to
  chi_max (#453, #459).

### Refactoring

- **Per-stage state for chi schedule** (#466).
- **Extract shared CTM-policy helpers** used by AD dispatchers (#382).
- **Split `ad_utils` into `_ad_primitives`** to break CTM SCC (#381).
- **Move `BlockArray` to `tenax.core`** to defer Cython import (#380).
- **Replace wildcard re-exports** in `_split_ctm_tensor` shim (#379).
- **Split DMRG dispatch executors** to break import cycle (#378).
- **Derive `ParamSpec.default`** from the dataclass on access (#374).
- **CTM sweep order** switched to (left, top, right, bottom) (#407).

### Tests / CI

- **Gate JAX cache clear on RSS threshold** — saves ~8 min on
  required Tests CI (#386).
- **`merge_group` trigger** so workflows run on queued groups (#452).
- **Split fast bucket** into `fast-ipeps` and `fast-other` to dodge
  the GitHub runner reaper (#383).
- **Split Full tests matrix** into fast + slow buckets (#370).
- **Include workflow edits in change-detection filter** (#384).
- Rationale for fast/slow Full-tests bucket split (#375).
- **Skip AD loop in slow symmetric regression test** (#385) —
  unblocks `fast-ipeps` bucket.
- **Drop fragile mechanism/parity tests** (#487, #488).
- **Drop 13 perf-style unit tests** (timing + convergence-budget)
  (#468).
- **Restore stall-cap regression coverage** with behavioural
  assertion (#489).
- **Replace strict FD-parity with gradient smoke** (#469, #482).
- **Pin compiled-sweep parity test to 1x1 recipe** + align sweep
  order (#446).
- **Full-rank env in projector biorthogonality tests** (#443).
- **Re-tune seeds for two AD adjoint tests** (#428, #429).
- **xfail 7 SymmetricTensor tests** on `_compute_2x2_projector`
  (#416, #417) — now resolved by #434.
- **xfail 4 tests** blocked on non-trivial U(1) 2x2 dense fallback
  wrap (#435, #436).
- **xfail 3 random-init plateau tests** (#425, #426 known-limitation,
  #444).
- **Drop AD smoke tests** that block CI without regression value
  (#373, #360).
- **Drop python-loop CTM smoke tests** + fix macOS-only failure
  (#367).
- **Restore #328/#298 implicit-AD regression coverage** (#365).
- **Drop redundant smoke tests** (hotrg/idmrg/block-sparse, fpeps
  builders) (#366, #368).
- **Tighten lbfgs/cg optimizer smoke scope** to avoid macOS OOM
  (#364).
- **Unblock buckets A/B/F/C/D/E from #354** (#355, #356, #358,
  #359).
- **Bump explicit-AD gradient tolerance** to 0.10 (macOS Accelerate
  bias) (#371).
- **Delete `test_ctm_energy_explicit_gradient_matches_fd`** (#369,
  #377).
- **Reduce compute in `symmetric_matches_dense`** to fit fast-ipeps
  bucket (Tests required check).
- **Fix tuple-unpack regressions** in CTM sweep tests (#441).

### Documentation

- **2-site implicit-AD safety nets section** in
  `docs/guide/algorithms/ipeps_ad_paths.md` with the v6–v9 ladder
  and production recipe.
- **Safety-nets subsection** in `.claude/skills/tenax-ipeps-workflow`
  surface `gs_chi_ceiling_bailout`, `gs_grad_spike_ratio/window`,
  `gs_hz_max_iter`, and `chi_auto_bump_metric` for agent guidance.
- **README** bullet for safety nets + new metric in auto-χ_E bump.
- **Refresh code-paths doc** + add CTM gauge/normalization notes
  (#350).
- **Add CG iPEPS, native honeycomb CTM, `bar_super()`** to
  code-paths (#376).
- **Pin issue #425 design rationale** on `_compute_2x2_projector`
  (#426).
- **Design plans for #454 stall-runaway and #453 χ-ramp recompile**.
- **Implementation plans for #454 and #453**.
- **Correct split-CTM shim peak documentation** from χ²·D⁸ to
  χ²·D⁶ (#395).
- **2-site warning** updated to reference in-CTM bump + implicit
  limitation (#511, #514).

## v0.6.0 (2026-04-28)

### New Features

- **Python-loop CTM AD with JIT-fused GMRES backward** — explicit and
  implicit AD share a Python-loop CTM forward; backward solves
  `(I − J^T)λ = ∂E/∂env` via JIT-fused GMRES (#340, #341)
- **Block-sparse CTM/AD** for `SymmetricTensor` iPEPS — `SymmetricTensor`
  passes through `jax.custom_vjp` as a pytree leaf without `todense()`
  round-trip; block-sparse Fishman SVD projector
  (`_svd_projector_symmetric`); sigma gauge fixing and convergence
  checks preserve `SymmetricTensor` type (#341)
- **Lorentzian projector backward for explicit AD** (Approach A) (#318)
- **Polymorphic optimizer shell for `SymmetricTensor` AD** — closes #297
  by routing the AD optimizer through a single shell that handles both
  `DenseTensor` and `SymmetricTensor` parameter pytrees (#329)
- **`adjoint_tikhonov` damping** + FD-AD gradient-correctness tests
  for the implicit-diff backward (#311)
- **Tuning surface expanded** — registered AD backward and optimizer
  knobs grow from 21 → 36 parameters (#323)
- **`CTMConfig.gmres_maxiter`** — separate knob for the outer GMRES
  iteration budget (was wired to `gmres_restart`, capping the adjoint
  solve at the restart size) (#343)
- **`chi_ramp`** documentation in guides, tuning, and skills (#339)

### Behavior Changes

- **`CTMConfig.forward_gauge` default changed from `"qr"` to `"phase"`** —
  the AD-correct choice for both implicit and explicit AD (1-site and
  2-site).  `optimize_gs_ad` no longer silently promotes a user-supplied
  gauge: an explicit choice is preserved (was previously promoted
  `"qr"` → `"phase"` for AD paths) (#343)
- **`build_ad_ctm_config`** no longer promotes `forward_gauge="qr"` to
  `"phase"`.  Direct `CTMConfig()` users now see the same gauge the
  optimizer uses (#343)
- **Unified Python-loop CTM forward + chi ramp** — single forward path
  used by both explicit and implicit AD (#337, #339)
- **Centralized AD policy** with import-cycle guardrails (#338)

### Bug Fixes

- Implicit-diff backward now honors `forward_gauge="sigma"` end-to-end
  (#322)
- Tangent-project 2-site AD direction (#330) and Hermitian inner product
  in `_tangent_project_unit` (#331) — closes #328
- Arnoldi precheck + sigma gauge fixes for implicit AD (#334)
- Correct F-matrix sign in `regularized_eigh` backward (#319)
- Warm-start fresh CTM re-eval to match in-loop best (#320)
- `_phase_fix_normalize_tensor` no longer downcasts to `DenseTensor` —
  `SymmetricTensor` environments are preserved across CTM sweeps (#341)
- Correct symmetric `numpy_blockwise` DMRG energy regression (#326)
- Refactor to dedupe projector wrapping and env-diff logic (#342)
- Address P1/P2/P3 review comments from #311, #314, #318, #320 (#321)
- Document non-C4v 2-site AD instability at finite χ (#328, #332)

### Tests

- Unblock 2 xfail AD tests now passing:
  `test_vjp_gradient_finite_with_elementwise`,
  `test_optimize_gs_ad_symmetric_energy_decreases` (#344)
- Migrate baseline gauge-policy tests to spy on `ctm_energy_implicit`
  (the legacy `ctm_tensor_converge_explicit` is no longer routed
  through by the optimizer) (#344)
- Mirror iPEPS correctness tests into the core CI tier (#312)
- `test_ad_d2_energy` uses `gs_c4v=True` (#327)
- Unblock pre-existing CI failures (#309)

### CI

- `test-full` matrix now runs only `-m "not core"` to avoid duplicating
  the 731 core tests already covered by required jobs.  Eliminates OOM
  / "runner lost communication" flakes on push to main (#345)

### Documentation

- Post-#341 AD policy: guides, skills, and CHANGELOG (#346)
- Architectural map of DMRG/iDMRG code paths (#314)
- Design plan for multisite `c4v_reference` AD (#315)
- Sync `c4v_reference` docs and citations to PR #304/#306 (#310)

## v0.5.0 (2026-04-13)

### New Features

- **C4v shared-tensor path for 2-site AD** and paper-faithful Appendix
  C–F AD mode following Francuz et al., PRR 7, 013237 (#304, #306)
- **Differentiable sigma gauge fixing** via power iteration (#289) with
  variants for stable AD convergence (#284) and elementwise CTM
  convergence (#275)
- **Direct AD through CTM projectors** with warmup / checkpoint (#257)
- **Sublattice rotation, C4v symmetrization, and metric preconditioning**
  for iPEPS AD (#262)
- **Fishman SVD projectors**, QR-gauge NaN fix, and per-absorption
  normalization (#276)
- **Half-SVD projectors**, JIT CTM, and convergence fix (#277)
- **Explicit C4v basis parameterization** for stable L-BFGS (#273)
- **Improved AD convergence** — SVD sign fix, Hager–Zhang line search,
  regularized `eigh` (#274)

### Behavior Changes

- Lazy-load algorithm modules to reduce import-time overhead (#286)
- Replace wildcard re-export shims with explicit internal imports (#288)
- Eval config inherits projector and `jit_ctm` settings (#278)
- Move `scale_bond_axis` from algorithms to core to fix dependency
  inversion (#285)

### Bug Fixes

- variPEPS-style stall recovery for 2-site AD (#298, #300)
- Adaptive Lorentzian broadening in `regularized_eigh` backward (#301)
- Dense-wrap `SymmetricTensor` inputs in 1-site and 2-site AD (#296)
- Sweep mutation + differentiable SVD projector (#291)
- Close JIT `while_loop` sigma-gauge energy gap (#290)
- Remove duplicated sigma-gauge warmup block in JIT CTM (#302, #303)
- Cython Lanczos reorthogonalization writing to copy instead of original
  (#268, #271)

### Performance

- Warm-start CTM in line search from previous boundary tensors (#272)

### Infrastructure / CI

- Use cibuildwheel for manylinux/macOS wheels (#261, #263, #264)
- Skip full tests for CI-only pushes; add nightly PyPI builds (#265)
- Fix `hatch-cython` config (#266, #267)
- Stop tracking `uv.lock` (#295)
- Drop `--auto` from sync-skills merge (#307)
- Bump version to 0.5.0 (#308)

### Documentation

- Sync iPEPS AD guidance and tests with post-PR-#291 baseline (#293)
- Audit dense fallbacks in symmetric CTM AD path (#287)

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
