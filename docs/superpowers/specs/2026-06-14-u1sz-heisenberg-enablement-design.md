# Design: U(1)-Sz Heisenberg enablement + feasibility spike

**Date:** 2026-06-14
**Status:** approved (design); spec under review
**Issue:** #570 (follow-up — the lever identified by the dense large-D study)
**Predecessor:** `2026-06-12-heisenberg-largeD-characterization-design.md` (dense path = runtime-bound; U(1)/Sz named as the lever)

## Goal

Establish whether tenax can run the **U(1)-Sz–symmetric** 2D spin-½ Heisenberg model through
the iPEPS ground-state AD path, and gate further investment on a **feasibility spike** with an
explicit GO/NO-GO verdict.

This is the scoped follow-up the dense study pointed to: the dense single-site path is
**runtime-bound** (per-step CTM re-convergence ~χ^1.7; D=3 χ=16 = 108 min, D=7 χ=300 out of
reach), and U(1)/Sz block-sparsity is the lever to shrink per-step cost and unlock larger D/χ.
Before committing to a (D,χ) sweep we must first confirm the symmetric path **runs end-to-end,
is correct, and shows a block-sparsity signal** — because a documented coverage gap may block it.

## Context from exploration (2026-06-14)

Most symmetric machinery already exists:

- `SymmetricTensor` (`src/tenax/core/tensor.py:630`), `U1Symmetry`
  (`src/tenax/core/symmetry.py:182`), block-sparse `svd`/`qr`/`eigh` (`src/tenax/linalg.py`).
- `optimize_gs_ad` accepts `SymmetricTensor` for both 1-site (`_optimize_gs_ad_tensor`,
  `ipeps_optimize.py:1202`) and 2-site (`_optimize_gs_ad_2site`, `ipeps_optimize.py:2369`;
  regression test `tests/test_ipeps.py:827`).
- CTM supports symmetric environments (`_ctm_tensor_init.py:354`). `validate_ctm_for_implicit_ad`
  imposes **no** symmetry-specific constraints (phase+svd+elementwise apply to all tensor types).
- Fermionic fPEPS is the working template for a **charged** gate + symmetric init
  (`spinless_fermion_gate`, `fermionic_ipeps.py:60-108`).

Two gaps:

1. **No U(1)-Sz Heisenberg gate.** `heisenberg_gate()` (`ipeps.py:41`) uses **trivial** physical
   charges `[0,0]`. A charged version with physical charges `[+1,−1]` (units of 2·Sz) is needed.
2. **Documented absorb-step coverage gap.** `examples/bench_symmetric_ad_batching_566.py:57`
   notes the *U(1) single-site CTM path with non-trivial charges currently fails in the
   production absorb step* — the fermionic benchmark sidesteps it with bounded FermionParity
   charges. Sz is unbounded, so the symmetric Heisenberg path may hit exactly this.

## Formulation (forced, not a free choice)

U(1)-Sz Heisenberg **must** use the **unrotated 2-site bipartite cell**:

- The **1-site rotated C4v** trick the dense study used is incompatible: the sublattice rotation
  (π spin rotation on one sublattice) turns `S·S` into an anisotropic `−SˣSˣ + SʸSʸ − SᶻSᶻ`
  bond that conserves **no** single-axis Sz — the rotated model has no U(1) symmetry to exploit.
- A **1-site unrotated Sz** ansatz forces a uniform (non-Néel) state and cannot represent the
  AFM order — wrong (higher) energy.

So: **2-site `(A, B)`, both physical legs charged `[+1,−1]`, total Sz=0, Néel order carried by
opposite virtual-bond charge offsets.** 2-site symmetric AD is already supported.

## Non-goals (explicitly out of scope)

- The full (D,χ) **characterization sweep** of U(1)-Sz vs dense — that is the *next* study,
  gated on a GO verdict here. (Pulling it in now repeats the dense study's runtime-wall trap
  before knowing the path runs.) This study produces, on GO, a *recommended* sweep design only.
- Non-Abelian (SU(2)) symmetry.
- Fermionic / Hubbard models (separate threads).
- The dense C4v 1-site path (kept only as a rough energy sanity scale).
- Reaching the −0.6694430 reference scale (D=7 χ=300).

## Components

| # | Component | Location | Production? |
|---|-----------|----------|-------------|
| 1 | `heisenberg_gate_u1sz()` — physical charges `[+1,−1]`, returns `SymmetricTensor` | `src/tenax/algorithms/ipeps.py` | **Yes** — public API; tests + README + `__init__.__all__` |
| 2 | Symmetric 2-site init helper — `(A,B)` with virtual Sz sectors, Néel offset | `src/` or spike script | Decided at plan time (prefer src/ if reusable) |
| 3 | Spike script — small-(D,χ) smoke + dense-2-site cross-check + 1-cell perf signal → JSON + verdict | `examples/bench_heisenberg_u1sz_spike.py` | No (study script, like `bench_*`) |
| 4 | **Conditional** absorb-step fix for non-trivial U(1) charges + regression test | `src/` | **Yes** — only if the gap blocks |

### Component 1 — `heisenberg_gate_u1sz()`

- Physical index charges `[+1, −1]` for `{↑, ↓}` (units of 2·Sz, matching the `S+`/`S−`
  charge-±2 convention in `tests/test_observables.py:146-155`).
- Same numeric Hamiltonian as `heisenberg_gate()` (unrotated `S·S`), built as a
  4-leg `SymmetricTensor` via `SymmetricTensor.from_dense(H, indices)` (pattern:
  `spinless_fermion_gate`, `fermionic_ipeps.py:108`).
- Charge conservation: the `S+S−`/`S−S+` hopping terms must land in allowed blocks; assert the
  dense round-trip `gate.todense()` reproduces `heisenberg_gate().todense()` exactly.

### Component 3 — spike script

- Grid: **D=2, χ∈{8,16}** (cheap; the spike is about feasibility, not scale).
- For each cell, run **both** a U(1)-Sz 2-site optimization and a **dense 2-site** optimization
  (unrotated, same gate numerics) with identical config, and record both energies + timings.
- Config: `optimize_gs_ad`, 2-site unit cell, phase-gauge + SVD-projector + implicit AD
  (forced by `validate_ctm_for_implicit_ad`), `gs_conv_criterion="grad_norm"`, fixed `gs_steps`
  budget, `return_history=True`. SU-init or random symmetric init (decided at plan time).
- JSON-checkpointed, resumable (same pattern as `bench_heisenberg_largeD.py`).

## GO / NO-GO gate (the spike's three checks)

1. **Runs** — the U(1)-Sz 2-site path completes CTM-AD end-to-end (the absorb step survives
   non-trivial charges) without raising.
2. **Correct** — two separable checks, because two independent L-BFGS runs need not agree to
   machine precision even at the same minimum:
   - **(2a) contraction correctness (tight, ~1e-8):** take a *single fixed* symmetric state,
     evaluate its energy through the symmetric contraction and through its densified
     (`.todense()`) contraction — they must agree to ~1e-8. This proves the block-sparse
     energy/CTM path is numerically correct, independent of optimization.
   - **(2b) optimization reaches the right energy (loose, ~1e-3):** the U(1)-Sz optimization
     converges near the **dense 2-site** optimum at the same (D,χ) (symmetric ≳ dense within
     optimizer noise, since the GS is Sz=0), and D=2 lands near **−0.66** (dense study's D=2
     scale). This guards against a "runs but wrong basin" false GO.
3. **Perf signal** — at one (D,χ), the U(1)-Sz path shows block-sparsity reducing per-step cost
   vs dense: report median warm-step time **and** a structural proxy (number/size of charge
   blocks vs the dense tensor sizes). A modest reduction is sufficient for a GO; a regression
   (symmetric slower with no structural benefit) is a yellow flag to note, not an automatic
   NO-GO at D=2 (block overhead can dominate at tiny D).

## Branches / disposition

- **Runs + correct + signal → GO.** Write the verdict + a recommended (D,χ) characterization
  sweep design; that sweep is the next study.
- **Absorb-step gap blocks (check 1 fails) → FIX IT PROPERLY** (component 4): reproduce with a
  bounded-charge proxy (cap virtual charges / Z2) to localize whether the bug is
  unbounded-charge-specific or general, root-cause the production absorb step, fix with a
  regression test, then resume the gate and re-run the gate checks.
- **Runs but wrong energy (check 2 fails) → diagnose** the charge assignment / normalization /
  gauge before any perf claim; do not report a GO.

## Measurement protocol (per cell)

Record, for both the symmetric and dense runs: `E_final` (min over trajectory),
`dE` vs −0.6694430, `jit_compile_s`, `total_wall_s`, `warm_step_s` (median of post-compile
steps), `num_steps`, `converged`, and for the symmetric run a `num_blocks` / largest-block-size
structural proxy. Also record `dE_sym_vs_dense = E_sym − E_dense` (the correctness check).

## Correctness guards

- **Gate round-trip:** `heisenberg_gate_u1sz().todense()` must equal `heisenberg_gate().todense()`
  (same physics, only charges differ). Asserted in the gate's unit test.
- **Contraction correctness:** symmetric vs densified energy of a fixed state to ~1e-8 (check 2a)
  — the primary, optimization-independent GO gate.
- **Optimization agreement:** symmetric optimum near the dense 2-site optimum (check 2b, ~1e-3).
- **Variational-floor watch:** any cell with `E_final` materially below −0.6694430 is flagged
  (signals normalization/gauge error), recorded — not crashed on.

## Testing

- Component 1 (gate): unit tests (charge structure, dense round-trip, charge conservation on
  blocks) — production code, so real tests required.
- Component 4 (absorb fix, if built): a regression test that reproduces the original failure.
- Spike script: no unit tests (consistent with `examples/bench_*`); validation is operational
  (CPU smoke at D=2 χ=8 before any GPU run).

## Outputs / deliverables

1. `heisenberg_gate_u1sz()` in `src/` + tests + README/`__init__` updates (production capability).
2. `examples/bench_heisenberg_u1sz_spike.py` + JSON evidence.
3. *Conditional:* the absorb-step fix + regression test (if the gap blocks).
4. A handoff writeup `docs/superpowers/handoffs/2026-06-14-u1sz-heisenberg-spike.md` with the
   GO/NO-GO verdict, the symmetric-vs-dense correctness result, the first perf/block signal,
   and (on GO) a recommended characterization-sweep design.
