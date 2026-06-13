# Dense 2D-Heisenberg large-D characterization — findings

**Date:** 2026-06-12
**Issue:** #570 (the never-evaluated "large-D energy + runtime + compile" acceptance criterion)
**Branch:** `study/heisenberg-largeD-characterization`
**Script:** `examples/bench_heisenberg_largeD.py` · **Data:** `examples/heisenberg_largeD_a100.json`
**Spec/plan:** `docs/superpowers/{specs,plans}/2026-06-12-heisenberg-largeD-characterization*`

## TL;DR

The dense single-site (C4v, phase-gauge, implicit-AD, SVD-projector) iPEPS path **works and
scales in the right direction** — energy improves with D toward the −0.6694430 reference — but
it is **runtime-bound**, not compile-bound, and the per-cell cost explodes fast enough that the
dense path **cannot reach the regime needed for competitive energies** (the literature reference
is D=7 χ=300; here a single **D=2 χ=32** cell already costs **42 min**). The lever to go further
is **U(1)/Sz symmetry**, which is out of this study's scope and is the recommended next step.

A second, contrasting finding: the dense path's wall is **per-step runtime**
(cold XLA-compile is only ~10–20 s/cell), whereas the fermionic / block-sparse CTM-AD path that
the rest of #570 / #566 chased is **compile-bound**. Dense and block-sparse hit *different* walls.

## Setup

- Model: `sublattice_rotate_gate(heisenberg_gate())` — Néel → uniform, single C4v tensor.
- Path: `optimize_gs_ad`, `gs_c4v=True`, `forward_gauge="phase"`, `projector_method="svd"`,
  implicit AD, `gs_conv_criterion="grad_norm"` (tol 1e-5), SU-init, `gs_num_steps=150`.
  (phase+svd is **forced** by `validate_ctm_for_implicit_ad`; the older
  `examples/heisenberg_ipeps_ad.py` sigma+eigh+gmres config is now API-invalid — see the spec
  correction. Its docstring's −0.6625 came from that invalid config; the valid path gives −0.6602.)
- Hardware: A100-SXM4-80GB, x64, cold-compile + runtime per cell, JSON-checkpointed.
- Reference: E/site = **−0.6694430** (Corboz QR-CTMRG / QMC).

## Results

| D | χ | E_final | dE vs −0.6694430 | jit_compile (s) | total wall (s) | median warm-step (s) | steps | conv |
|---|---|---------|------------------|-----------------|----------------|----------------------|-------|------|
| 2 | 8  | −0.660229 | +0.00921 | 19.8 | 551.6  | 0.549 | 150 | no |
| 2 | 16 | −0.660231 | +0.00921 | 19.5 | 783.5  | 0.699 | 150 | no |
| 2 | 24 | −0.660214 | +0.00923 |  8.5 | 1409.7 | 1.365 | 150 | no |
| 2 | 32 | −0.660197 | +0.00925 |  8.6 | 2544.6 | 2.205 | 150 | no |
| 3 | 8  | −0.663993 | +0.00545 |  8.1 | 2493.2 | 1.636 | 150 | no |
| 3 | 16 | −0.664192 | +0.00525 | 27.2 | 6475.5 | 3.993 | 150 | no |
| 4 | 8  | −0.667109 | +0.00233 |  9.0 | 6637.0 | 2.787 | 150 | no |

Every cell past D=3 χ=8 exceeds the 1-hour per-attempt budget — D=3 χ=16 took **108 min** and
D=4 χ=8 took **111 min** — and the full 12-cell core grid did not finish in the kill/resume
budget (≈ one expensive cell per hour). The larger χ stretch (χ≥24 at D≥3, all D≥5) was not
pursued: the per-cell wall makes it impractical, and that intractability *is* the runtime-wall
finding.

## 1. Energy vs (D, χ)

- **D-scaling works, monotonically.** D=2 → −0.6602 (dE +0.0092), D=3 → −0.6642 (dE +0.0053),
  D=4 χ=8 → −0.6671 (dE +0.0023): each bond-dimension step closes a large fraction of the
  remaining gap to −0.6694430 (≈43% at D=2→3, then ≈56% at D=3→4), the expected variational
  improvement with D. The path reaches sensible, literature-consistent energies (D=2 iPEPS for
  2D Heisenberg sits at ≈ −0.660; D=4 is already within 0.0023 of the QMC reference).
- **χ-scaling is flat at D=2, marginal at D=3.** At D=2, E is constant to <4e-5 across
  χ ∈ {8,16,24,32} — correct physics: the double-layer bond is D²=4, so the CTM environment is
  χ-saturated by χ≈8. At D=3 (double-layer bond D²=9) χ=8 is slightly under-resolved and
  χ=8→16 buys a small but real gain (−0.663993 → −0.664192, ≈2e-4). The dominant energy signal
  still lives in **D**, not χ. (D=2 χ-flatness CPU-validated separately before the sweep.)
- `converged=no` everywhere: `grad_norm` does not reach 1e-5 within 150 steps because the
  near-minimum landscape is flat/CTM-noisy, but the **energy plateaus** well before the budget,
  so `E_final` (min over the trajectory) is the meaningful number. No cell fell below the
  reference (the variational-floor watch never tripped).

## 2. Runtime + compile scaling

- **Compile is cheap and ~flat:** `jit_compile_s` ≈ 8–20 s regardless of D/χ. The dense bosonic
  backward does **not** suffer the block-sparse compile wall.
- **Runtime is the cost and it explodes:** total wall at D=2 grows 552 → 784 → 1410 → 2545 s as
  χ goes 8 → 32 (≈ χ^1.7); median warm-step grows 0.55 → 2.2 s. D=3 χ=8 (2493 s) already costs as
  much as D=2 χ=32, and D=3 χ=16 (6475 s, **108 min**, warm-step 3.99 s) is the single most
  expensive completed cell. The per-step cost is the implicit-AD step: a full CTM re-convergence
  (`max_iter` up to 100 sweeps) inside each L-BFGS line-search evaluation, and that scales
  steeply in both D and χ.

## 3. Where the dense wall hits

The practical ceiling of this dense path (gs_steps=150, A100): **~D=3 χ=8 within an hour.** D=3
χ=16 (108 min) and D=4 χ=8 (111 min) each take ~2 h per cell; larger χ at D≥3 and all D≥5 were
not pursued. The reference scale (D=7 χ=300) is many orders of magnitude out of reach (D=2 χ=32
alone = 42 min). The wall is **wall-clock runtime**, set by per-step CTM convergence × line
search × steps — not memory and not compile (compile stays ≤27 s even at the largest cell;
notably D=4 χ=8 compiled in 9 s yet still ran 111 min, underscoring that the cost is *runtime*).

## 4. Conclusion / next step

The dense path is **correct and directionally competitive** (energy improves with D toward the
QMC reference) but **does not scale**: it is runtime-bound and impractical beyond small D. This
directly answers #570 for the dense path. The lever to push toward competitive large-D energies
is **U(1)/Sz symmetry**: block-diagonalizing the site/CTM tensors shrinks per-step contraction
and decomposition cost and unlocks larger D/χ — the natural, scoped follow-up study (the builtin
`heisenberg_gate()` currently carries trivial charges). The block-sparse CTM-AD compile wall
explored elsewhere in #570/#566 is a *separate* axis: that path is compile-bound; this dense one
is runtime-bound.
