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

2. **A 1×1 unit cell cannot represent Néel order.** An absolute QMC comparison
   is therefore meaningless at 1-site. The correct production-correctness bar
   is *split reproduces the trusted fused result on the same 1×1 problem*, not
   an absolute physical energy.

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

### Part 1 — 1-site production-correctness test

New test (target file: `tests/test_split_ctm_fuse_flag.py`, or a focused new
`tests/test_split_ctm_production_parity.py` — implementation-plan decides).

**Mechanism.** For D ∈ {2, 3}:

1. Build a single 1×1 site tensor `A` from a fixed seed; use the existing
   `_heisenberg_gate()` antiferromagnetic gate.
2. Run `optimize_gs_ad(gate, A, config)` **twice** with identical config except
   the flag:
   - split: `CTMConfig(chi, chi_I=chi, fuse_virtual_legs=False)`,
   - fused: `CTMConfig(chi, fuse_virtual_legs=True)`,
   - both `unit_cell="1x1"`, `gs_recipe="1x1"`, `gs_implicit_ad=True`,
     `forward_gauge="phase"`, same `gs_num_steps` (small, e.g. 5–10), same seed.
3. Assert:
   - `E_split_final ≈ E_fused_final` to `atol ≈ 1e-6` (faithful drop-in),
   - `E_split_final < E_split_initial` (the optimizer actually descends),
   - returned env is a `SplitCTMTensorEnv` (already covered elsewhere; keep or
     cross-reference).

**Tolerance rationale.** Single-eval split-vs-fused parity holds to ~1e-8 at
lossless `chi_I` (existing `test_split_matches_fused_lossless_chi_I`). Over a
full optimization, accumulated L-BFGS path divergence and the
`chi_I=chi`-vs-lossless residual (~0 but nonzero) loosen this; `1e-6` is the
proposed bar, to be tightened in the implementation plan if the empirical gap
is smaller. The bar must be justified by a measured number, not guessed.

**Marker.** `algorithm`-tier (runs an optimizer); not `core` — keeps CI-required
checks fast.

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

- Part 1: the new optimizer-parity test (D=2, D=3).
- Part 2: the CG-rejection regression test(s).
- Regression guard: the existing split-CTM suite
  (`test_split_ctm_fuse_flag.py`, `test_split_ctm_doublelayer_projector.py`,
  `test_split_ctm_tensor.py`) must stay green.

## Acceptance criteria

- [ ] 1-site split optimization matches fused to ≤1e-6 in final energy at
      D∈{2,3}, and both descend from their initial energy.
- [ ] CG + `fuse_virtual_legs=False` raises a clear `NotImplementedError`
      (regression-locked).
- [ ] Future CG-on-split enablement path is documented.
- [ ] Full split-CTM test suite green; no behavior change to any shipped path.

## Risks

- **Optimizer-path divergence** between split and fused could exceed 1e-6 even
  though single-eval parity is ~1e-8 (different L-BFGS trajectories from
  floating-point-order differences). Mitigation: keep step count small, seed
  fixed; if the empirical gap is larger, the parity bar should compare
  *per-step* energies for the first step (where paths can't yet diverge) and
  loosen the final-energy tolerance with a measured justification — decided in
  the implementation plan, not guessed here.
- **CG guard message drift** — the test should match on a stable substring
  (`"cg_gates"`), not the full message.
