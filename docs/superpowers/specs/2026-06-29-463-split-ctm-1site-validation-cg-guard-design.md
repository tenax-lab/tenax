# #463 — Split-CTM 1-site production-correctness validation + CG guard lock-in

**Date:** 2026-06-29
**Issue:** #463 (arch(ctm): unify split-CTM as the canonical path)
**Status:** design — pre-implementation

## Context

Issue #463 migrates iPEPS CTM from the fused double-layer construction
(`_build_double_layer_tensor`, peak χ²·D⁴·d²) to the split ket/bra path
(peak χ²·D⁴·d, and Koszul-correct for fermions). The issue defines four
phases:

- **Phase 1** — split forward + split-aware RDM (shipped).
- **Phase 2** — migrate callers behind `CTMConfig.fuse_virtual_legs` (shipped
  for the **single-site dense bosonic** path: PRs #648/651/652/653/654/656/657).
- **Phase 3** — flip the `fuse_virtual_legs` default to `False`.
- **Phase 4** — archive the fused path (remove flag + delete
  `_build_double_layer_tensor`), gated on the split path being default for
  2+ release cycles.

The user's "phase 4 flip" maps to the **flip-the-default** step (issue Phase 3).

### Why we are *not* flipping in this work

A literal global flip of `fuse_virtual_legs` from `True → False` would, by the
current guards, raise `NotImplementedError` for **2-site bipartite** (the
validated production Heisenberg config), c4v, honeycomb, multisite, and PESS,
and is unsupported for SymmetricTensor/fermionic. Per issue #463's own
structure, Phase 3 (flip) is gated on Phase 2 (migrate callers) being complete —
and only the 1×1 dense bosonic path is migrated. Flipping now would regress
production. **The flip is deliberately out of scope here.**

This spec is the first, tightly-scoped prerequisite for an eventual flip:
**prove the 1-site split path is production-correct, and lock in the
(intended) CG rejection with a regression test.** The 2-site/2×2 split forward
— the substantive prerequisite for a meaningful flip — is its own later spec.

## Goals

1. Upgrade the 1-site split correctness bar from "finite energy + single-eval
   gradient-match" (today's tests) to "**faithful drop-in for the fused path
   over a full optimization**."
2. Lock in the intended CG-on-split rejection with a regression test, and
   document precisely what a future effort needs to enable CG on split.

## Non-goals (explicitly out of scope)

- Flipping the `fuse_virtual_legs` default (Phase 3). Stays `True`.
- 2-site / 2×2 / honeycomb / multisite split forward (separate spec).
- SymmetricTensor / fermionic split AD.
- **Enabling** CG on the split path (new wiring code — see below).
- Any change to the auto-sentinel-vs-hard-flip default semantics (deferred to
  the flip spec).

## Findings that shaped the design

1. **`chi_I=chi` is already lossless** with the rank-1 corner init. Existing
   test `test_split_production_chi_I_converges_to_lossless` shows the
   interlayer-truncation error (`chi_I=chi` vs lossless `chi_I=chi*D`) is ~0
   even at D=3. ⟹ split == fused at the *production* interlayer bond, so a
   clean split-vs-fused energy parity is achievable at default settings — we do
   **not** need to pin `chi_I=chi*D` for the parity test.

2. **(REVISED 2026-06-30 after empirical probing — supersedes the original
   "fused-parity over optimization" bar.)** Three measured facts reshaped Part 1:

   a. **Split ≠ fused over a full optimization, and it's structural (#425), not
      truncation.** At D=2 χ=4 the bare-Heisenberg 1-site optimization gives
      split → −0.0527 vs fused → +0.0311 (gap 0.084), **identical at
      `chi_I=chi` and lossless `chi_I=chi*D` to 12 digits**, and unchanged by
      `gs_metric_precond`. So the iterated split- and fused-CTM forwards select
      *different fixed points* in the degenerate-corner SVD subspace — even
      though a *single* evaluation is bit-identical at lossless χ_I (the existing
      `test_split_matches_fused_lossless_chi_I`, 1e-8). ⟹ **"split == fused to
      1e-6 over a full optimization" is known-false and cannot be a test bar.**

   b. **An absolute reference IS achievable — via the sublattice-rotated
      Heisenberg gate** `H_rot = −Sz⊗Sz − ½(S⁺⊗S⁺ + S⁻⊗S⁻)`, under which a 1-site
      iPEPS represents Néel order, so E/site is comparable to QMC (−0.6694). The
      original "absolute QMC meaningless at 1×1" claim was wrong — it only holds
      for the *bare* (frustrated) gate.

   c. **C4v is mandatory.** Without `gs_c4v=True` the unconstrained 1-site CTM is
      non-variational for *both* paths (split converges cleanly to −0.714 *below*
      the QMC floor; fused goes chaotic, |g|~1e7 stall-noise, E_best −0.763) —
      the documented 1-site-CTM unreliability (`project_c3_floor_breach_smoking_gun`),
      orthogonal to split. **With `gs_c4v=True`** both become stable and
      variational and track within ~0.01/site. Measured converged values
      (D=2, χ=10, grad_norm |g|<1e-3):
      `split+c4v = −0.6505` (variational, +0.019 above QMC),
      `fused+c4v = −0.6601` (variational, +0.009 above QMC). The 0.0096/site gap
      **persists at tight |g|** ⟹ genuine bounded #425, not under-convergence.
      Both sit in the physical window `[−0.6694, −0.60]` (above the QMC floor,
      below the disordered energy ⟹ real order).

3. **CG is blocked on the split path at two layers, despite the split-aware CG
   energy already existing:**
   - `coarse_grain.py` provides `compute_energy_cg_split` + `_split_rdm_dispatch`
     (`_rdm{1x2,2x1,diagonal}_split_tensor`) — the split-aware CG RDM/energy
     building blocks exist.
   - `_split_ctm_energy_ad.ctm_energy_split_{explicit,implicit}` **reject any
     custom `energy_fn`** ("custom energy_fn (e.g. coarse-grain) is not
     supported on the split path").
   - `ipeps_optimize._optimize_gs_ad_tensor` separately rejects `cg_gates` under
     `fuse_virtual_legs=False` (`ipeps_optimize.py:1318`).

   Enabling CG-on-split = threading `energy_fn` through the split AD path +
   lifting both guards + validating the gradient. That is new code and is
   **out of scope**; this spec only locks the rejection in place and documents
   the enablement path.

## Design

### Part 1 — 1-site production-correctness test (variational window, C4v)

New test in a focused new file `tests/test_split_ctm_production_correctness.py`
(keeps the slow optimizer test out of the `core`-marked `test_split_ctm_fuse_flag.py`).

**Mechanism (single split run — does NOT require running fused).**

1. Build the sublattice-rotated Heisenberg gate
   `H_rot = −Sz⊗Sz − ½(S⁺⊗S⁺ + S⁻⊗S⁻)` so a 1-site iPEPS represents Néel order.
2. Run `optimize_gs_ad(gate, A, config)` once with the **split** path AND C4v:
   - `CTMConfig(chi=10, chi_I=10, fuse_virtual_legs=False, max_iter=80,
     conv_tol=1e-10, min_iter=4)`,
   - `iPEPSConfig(unit_cell="1x1", gs_recipe="1x1", gs_implicit_ad=True,
     gs_c4v=True, gs_metric_precond=False, gs_conv_criterion="grad_norm",
     gs_grad_norm_tol=1e-3, gs_num_steps=100, su_init=False)`,
   - `A = _make_site(2, 2, seed=3)` (D=2).
3. Assert on the returned final energy `E` (per site = E_h + E_v):
   - **Variational:** `E >= QMC_FLOOR - 1e-3` with `QMC_FLOOR = -0.6694` — split
     stays above the QMC ground-state energy. *This is the assertion that
     catches the failure mode:* without C4v split breaches to −0.714; with the
     #425-spurious fixed point it would dip below the floor.
   - **Ordered:** `E <= -0.60` — below the disordered/product energy, proving the
     1-site Néel ansatz actually found order (not stuck high).
   - i.e. `−0.6694 ≤ E ≤ −0.60`. Measured split+c4v value: **−0.6505** (margin
     +0.019 above floor, −0.050 below the ordered bound — comfortable both ways).

**Why a window, not fused-parity.** Split and fused converge to *different*
fixed points (#425, gap ~0.01/site, both physical) — see finding 2a/2c. A
1e-6 fused-parity bar is known-false. The physical window `[−0.6694, −0.60]`
is the well-defined bar: it is exactly what a *correct* variational 1-site path
must satisfy, and it is violated by the spurious sub-QMC fixed point the
no-C4v probe exposed. Single-run ⟹ ~half the wall-clock of a split+fused pair.

**Optional companion (separate test, same file).** A `split-tracks-fused` check:
run fused+c4v with the same config and assert `abs(E_split - E_fused) <= 0.03`
(measured 0.0096, 3× margin). Marked `slow`; documents the bounded #425 gap.
Include only if the extra optimizer run is acceptable; the window test is the
primary deliverable.

**Marker.** `slow` (one full implicit-AD optimization, ~minutes on GPU, much
longer on CPU). Runs on push-to-main / `run-full-tests`, NOT the `core` CI gate.
Use `@pytest.mark.slow` explicitly (do not rely on filename auto-marking).

**Convergence caveat.** L-BFGS wobbles mid-run (line-search overshoot, stall
recovery) — the assertion must be on the **final/E_best** energy after
`grad_norm` convergence (|g|<1e-3, ~step 69 measured), never on an arbitrary
intermediate step.

### Part 2 — CG guard lock-in + documentation

1. **Regression test** (target: `tests/test_split_ctm_fuse_flag.py`): assert that
   a CG configuration (`cg_gates` set) with `fuse_virtual_legs=False` raises
   `NotImplementedError` with a message naming `cg_gates`. This pins the
   `ipeps_optimize.py:1318` optimizer guard as *intended* behavior so a future
   refactor can't silently drop it. (Optionally also assert the
   `ctm_energy_split_{explicit,implicit}` `energy_fn` guard raises, to lock the
   second layer.)
2. **Documentation** — a short note (in this spec's "CG enablement (future)"
   section below, and a one-line pointer where the guards live) recording that
   the split-aware CG energy already exists and what wiring enables CG-on-split.

#### CG enablement (future, NOT this spec)

To run CG on the split path later:
- thread a `compute_energy_cg_split`-backed `energy_fn` through
  `ctm_energy_split_{explicit,implicit}` (remove the custom-`energy_fn` reject);
- lift the `cg_gates` reject in `_optimize_gs_ad_tensor`
  (`ipeps_optimize.py:1318`);
- validate the CG gradient on split vs the fused CG path (cosine ≈ 1).
- **Caveat:** split gives ~no CG-kagome memory win until D ≳ 16 (the
  d_eff²=64 supersite factor dominates; see
  `project_split_ctm_cg_memory_reality`). Enablement is correctness/uniformity,
  not a memory lever at usable D.

## Testing

- Part 1: the new `slow` variational-window test (split+c4v, rotated Heisenberg).
- Part 2: the CG-rejection regression test(s) (both guard layers).
- Regression guard: the existing split-CTM suite
  (`test_split_ctm_fuse_flag.py`, `test_split_ctm_doublelayer_projector.py`,
  `test_split_ctm_tensor.py`) must stay green.

## Acceptance criteria

- [ ] `slow` split+c4v rotated-Heisenberg optimization lands in the physical
      window `−0.6694 ≤ E/site ≤ −0.60` (variational + ordered), at D=2 χ=10
      with `gs_c4v=True`, `gs_conv_criterion="grad_norm"`, `gs_grad_norm_tol=1e-3`.
- [ ] CG + `fuse_virtual_legs=False` raises a clear `NotImplementedError` at BOTH
      guard layers (optimizer `cg_gates` reject + split-AD `energy_fn` reject),
      regression-locked.
- [ ] Future CG-on-split enablement path is documented.
- [ ] Full split-CTM test suite green; no behavior change to any shipped path.

## Risks

- **The variational-window bar is config-sensitive.** It only holds *with*
  `gs_c4v=True`; without C4v split breaches to −0.714 (measured). The test MUST
  set `gs_c4v=True` and assert on the converged `E_best`, never an intermediate
  step (L-BFGS wobbles). The −0.60 ordered bound has −0.050 margin and the QMC
  floor +0.019 margin against the measured −0.6505; both comfortable.
- **Runtime.** One implicit-AD optimization to |g|<1e-3 took ~69 steps (~minutes
  on GPU, ~20+ min on CPU). Hence `slow`-marked and kept off the `core` gate. If
  CI runtime is a concern, χ=8 / `gs_grad_norm_tol=2e-3` still clears the window
  (split was already above the floor by step ~6) — the plan may lower these, but
  must re-measure that the window still holds before tightening.
- **CG guard message drift** — match on a stable substring (`"cg_gates"` /
  `"energy_fn"`), not the full message.
- **#425 is the real flip blocker, not this spec.** This validation *confirms*
  the 1-site split path is variational with C4v; it does NOT close the bounded
  ~0.01/site split-vs-fused fixed-point gap. Whether that gap is acceptable for
  making split canonical (the actual flip) is a separate decision, out of scope.
