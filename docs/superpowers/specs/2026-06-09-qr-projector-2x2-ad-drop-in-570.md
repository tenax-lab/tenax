# Drop-in QR projector for the 2×2 Fishman AD path — runtime/backward win for CTM-AD

**Date:** 2026-06-09
**Issue:** #570 (CTM-AD compile/runtime wall = block-sparse SVD VJP, confirmed PR #589)
**Umbrella:** #566 · **Branch:** `feat/qr-projector-2x2-570` (off `main`)
**Status:** DESIGN — awaiting build

---

## Goal

Reduce the cost of the compile-dominant, χ-scaling **block-sparse SVD VJP** in the
symmetric iPEPS CTM-AD path by replacing the two **non-truncating** half-system SVDs
of the Fishman 2×2 two-projector with **QR decompositions** (cheap backward, no
`1/(sᵢ²−sⱼ²)` degeneracy F-matrix). This is the **drop-in** step: it stays inside the
validated Fishman two-projector structure and the existing implicit-diff fixed-point
adjoint, and is opt-in behind `projector_method="qr"`.

It is deliberately **not** the faithful reduced-corner QR-CTMRG rewrite (the
order-of-magnitude, new-fixed-point change — logged as a follow-up below).

## Background / why this is the chosen lever

- PR #589 established the fused backward's only χ-scaling term is the block-sparse
  **SVD VJP** (61% at D=4/χ=12); the decomposition math, not the structural emission,
  is what QR attacks.
- Isolated decomposition-VJP cost (probe_decomp_vjp_cost_570): production SVD VJP
  **261 ops** vs QR **99** (~2.6×) vs eigh **113**. QR avoids the Lorentzian/gauge
  degeneracy backward entirely.
- The **falsified** SVD-via-eigh spec (2026-06-08) showed a *cheaper decomposition*
  that keeps the same `1/(sᵢ²−sⱼ²)` F-matrix (eigh-of-Gram) is **not** a real lever;
  QR is different precisely because it has **no** singular-value degeneracy backward.
- The published QR-CTMRG (Yang/Zhang/Corboz, arXiv 2505.00494) reaches the big wins
  (D=6 χ=600 ~140× on H100; D=7 in ~30 min) — but via a **reduced-corner** scheme
  that removes *all* SVDs and changes the projector to the standard single-isometry
  form. That is out of scope here (see Follow-ups).

## Existing code reality (verified 2026-06-09)

- The AD CTM sweep uses `_compute_2x2_projector_symmetric`
  (`src/tenax/algorithms/_ctm_tensor_projector_2x2.py`), which performs **three**
  per-sector SVDs via `tenax.linalg.svd`: **M1**, **M2** (full, untruncated, gauge-
  fixed by `_gauge_fix_symmetric_svd`), and **M′** (the χ-truncation SVD producing the
  `chi_new` bond).
- `projector_method="qr"` **already exists but is a misnomer**: on the standard/split
  CTM path (`_ctm_projector.py`) it is "identical to eigh path; 'qr' label retained
  for API compat" (docstring lines 18–19). There is **no real QR projector** anywhere.
  `qr_warmup_steps` (config default 3) is "eigh warm-up before QR kicks in" —
  scaffolding built in anticipation of exactly this work.
- Block-sparse QR exists: `tenax.linalg.qr` → `_qr_symmetric`
  (`src/tenax/linalg.py:1282`), label-based, per-sector, JAX-traceable
  (`_qr_symmetric_np` is the eager twin). `_ad_primitives.py` has `regularized_svd`
  and `regularized_eigh` custom-VJPs but **no `regularized_qr`**.

## Sizing (the honest caveat)

For the `left`/`bottom` move (others are mirror images) the per-sector matrices are
all **(χD × χD)** — the χ environment leg × the D virtual leg (`r2`/`u2`), not χD².

SVD path today:
```
M1 (χD×χD) ─SVD→ U1 S1 V1h ─→ first_half  = U1·√S1
M2 (χD×χD) ─SVD→ U2 S2 V2h ─→ second_half = √S2·V2h
M′ = second_half ⊗seam first_half        (χD×χD)
M′ ─SVD,trunc→χ→ U' S' V'h
  P_first  = first_half·V'·S'^{-½}
  P_second = S'^{-½}·U'†·second_half
```

QR drop-in:
```
M1 ─QR→ Q1 R1  (gauge-fix diag(R1)≥0)     first_half  = Q1   (R1 folded forward)
M2 ─QR/LQ→ Q2 R2 (gauge-fix)              second_half = Q2
M′ = R1 ⊗seam R2                          (χD×χD)  ← still needs truncation
M′ ─SVD,trunc→χ→ U' S' V'h
  P_first  = Q1·U'·S'^{-½}
  P_second = S'^{-½}·V'†·Q2†
```

- **Removed:** the two large M1/M2 SVD-VJPs → QR-VJPs (~2.6× cheaper each, no
  F-matrix).
- **Kept:** the M′ truncation SVD-VJP — QR cannot truncate to χ; only the deferred
  reduced-corner rewrite removes it.
- **Expected win (back-of-envelope):** svd_vjp share ~61% → ~35% (≈25–26% fewer total
  backward ops). The largest available win **without** the structural rewrite.
  **⚠ OBSOLETE post-#593 — see "Task 2 result" below:** PR #593 (per-sector gauge-fix
  vectorization) already captured most of this; svd_vjp is now only ~10% of the
  backward and the remaining reachable end-to-end op reduction is **~4.4%**, not ~25%.
- **Biorthogonality** `P_first† P_second = I_χ` is preserved: Q1,Q2 isometric,
  `S'^{-½}` balances — the two-projector generalization of the dense single-projector
  QR fallback already in `_ctm_projector.py`.

## Architecture (5 components, smallest blast radius first)

### C1. Gauge-fixed, differentiable block-sparse QR primitive
- `_gauge_fix_symmetric_qr(Q, R)` in `_ctm_tensor_projector_2x2.py`: per-sector phase-
  fix so `diag(R)` is real-nonnegative (zero-diagonal → phase 1, leave untouched),
  mirroring the dense QR sign-fix (`_ctm_projector.py:1064–1071`) and the per-sector
  style of the #593-vectorized `_gauge_fix_symmetric_svd`. Vectorize per bond-charge
  sector from the start (avoid re-introducing a per-column scatter loop).
- Confirm `_qr_symmetric` traces cleanly under `jax.linearize`/`jax.vjp`. If the raw
  per-sector `jnp.linalg.qr` VJP is unstable for near-rank-deficient sectors, add
  `regularized_qr` in `_ad_primitives.py` (custom-VJP analogous to `regularized_svd`).
  **Decision deferred to the C1 spike** — do not add it speculatively.

### C2. QR variant of `_compute_2x2_projector_symmetric`
- Add parameter `decomp: str = "svd"` accepting `"svd"|"qr"`. With `"svd"` the function
  is **byte-identical** to today.
- `"qr"`: Stage 2 M1/M2 → QR (+ `_gauge_fix_symmetric_qr`); halves become the Q
  isometries with R folded into M′; Stage 4 M′ → unchanged truncation SVD on the
  R-product; Stage 5 cross-projectors re-derived in QR terms (formulas above).
- Mirror the existing `direction ∈ {left,right,top,bottom}` orientation handling and
  the `base_charges` eager/traced dispatch already in the function.

### C3. Wiring
- `projector_method="qr"` on the **2×2 AD path** routes to `decomp="qr"`. Keep the
  non-AD standard/split CTM eigh-equivalent block path (`_ctm_projector.py`) untouched.
- Thread `decomp`/method through `_ctm_energy_ad.py` and `ipeps_config.py`. Preserve
  `qr_warmup_steps` semantics (eigh/SVD warm-up grows χ, then QR).
- Update the `projector_method="qr"` docstring/changelog: it now runs a real QR
  projector on the 2×2 AD path (behavior change for any config that relied on
  `"qr"`==eigh).

### C4. Default & API
- **SVD remains the default.** QR is opt-in. No change to existing default runs.

### C5. Measurement harness
- Extend `examples/probe_bwd_subop_attribution_570.py` to split the three SVDs
  (M1/M2 vs M′) — the go/no-go number (see Testing T2).
- Reuse `examples/profile_570_sweepvjp_compile.py` for the A100 HLO/compile/runtime
  comparison (Testing T7).

## Testing gate (inverts #593's byte-parity convention)

QR changes the projector ⇒ a *different but physically equivalent* fixed point, so
**byte-parity vs SVD fails by construction.** The gate is **physical agreement**,
cheapest-first (TDD):

- **T1 (gates all). QR primitive spike** — `jax.test_util.check_grads` on the gauge-
  fixed block-sparse QR for several random sectors incl. near-rank-deficient; a
  smoothness test (perturb input by ε, `diag(R)≥0` keeps Q continuous, no sign flips).
  Red→green before any projector wiring. Outcome decides whether `regularized_qr` is
  needed.
- **T2 (go/no-go). Per-SVD cost attribution** — trace-only, via C5; confirms the
  M1/M2 share justifies the build and reports the expected win. A number, not a test;
  explicit off-ramp if M1/M2 are a small slice.
- **T3. Biorthogonality unit test** — `P_first† P_second = I_χ` per sector on random
  enlarged corners. Proves the QR cross-projector derivation independent of any CTM run.
- **T4. Forward energy agreement (physics gate)** — converge CTM `svd` vs `qr` on 2D
  Heisenberg **D=2 and D=3** at fixed χ; `|E_qr − E_svd| < tol` (≈ few×10⁻⁵, loosened
  vs machine-eps), and the gap shrinks as χ grows (both → same value).
- **T5. Gradient agreement** — (a) finite-difference vs AD on the QR path (self-
  consistent); (b) QR-AD ≈ SVD-AD within tol on the same small model.
- **T6. Regression** — `tests/test_block_sparse_ctm_ad.py` green with
  `projector_method="qr"` (multi-block, not just trivial charge) + a fermionic
  FermionParity smoke case (the #565/#566 surfacing model).
- **T7. Perf deliverable (not pass/fail)** — A100 D=4, χ∈{8,12,16}: backward HLO
  instruction count + compile time + warm-step runtime, QR vs SVD, via C5. Reported in
  the PR table.

**Acceptance** = T1, T3, T4, T5, T6 green + T2 says go + T7 shows a real win.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| QR-VJP unstable on near-rank-deficient sectors | High | `regularized_qr` if T1 shows it; T1 gates the build |
| Cost attribution disappoints (M1/M2 small slice) | Med | T2 is the explicit off-ramp; write up, don't ship a non-win |
| Gauge discontinuity across iters breaks AD | Med | `diag(R)≥0` phase-fix + T1 smoothness test |
| Warm-up coupling | Low | Reuse existing `qr_warmup_steps`; don't redesign |

## Out of scope (logged follow-ups)

- **Faithful reduced-corner QR-CTMRG** — removes all SVDs via a rank-χ reduced corner
  + unpivoted QR (standard single-isometry scheme, *not* Fishman two-projector); the
  140×-class win, new approximate fixed point, full convergence revalidation. Separate
  project.
- **Truncated/fixed-step backprop** (the paper's "2 untracked + k tracked" AD) — the
  orthogonal depth lever; Tenax keeps the implicit-diff adjoint here.
- The 1×1/standard-CTM `_ctm_projector.py` eigh-alias `"qr"` path — left as-is.
- GPU-specific QR kernel tuning.

## References

- Yang, Zhang, Corboz — QR-CTMRG, arXiv:2505.00494 (also 2509.05090, PRB 113 085109).
- PR #589 / `2026-06-08-570-relocalized-not-decomposition.md` — wall = block-sparse
  SVD VJP, structural-vs-decomposition split.
- `2026-06-08-svd-via-eigh-fishman-projector-570.md` — falsified cheaper-decomposition
  lever (why QR ≠ eigh-of-Gram).
- PR #593 — `_gauge_fix_symmetric_svd` per-sector vectorization (the gauge-fix pattern
  C1 mirrors).

## Task 1 result

SPIKE probe `examples/probe_qr_vjp_stability_570.py` run with
`JAX_PLATFORMS=cpu uv run python examples/probe_qr_vjp_stability_570.py`
(x64 enabled), exact output:

```
PASS  well-conditioned 12x12
PASS  tall 16x8
FAIL  near-rank-deficient 12x12: AssertionError:
Not equal to tolerance rtol=0.0001, atol=0.0001
VJP cotangent projection
Mismatched elements: 1 / 1 (100%)
Max absolute
```

- **well-conditioned 12×12** — PASS
- **tall 16×8** — PASS
- **near-rank-deficient 12×12** (trailing `drop=4` singular values zeroed) — FAIL
  (`check_grads` reverse-mode cotangent mismatch beyond `rtol=atol=1e-4`)

JAX's raw QR backward is fine for full-rank sectors but, exactly like the raw
SVD backward, is **unstable when singular values collide / vanish** — the
gauge-fixed `Q` cotangent blows up as `diag(R)` elements approach zero, which is
precisely the regime CTM projector sectors hit near truncation.

**DECISION: add `regularized_qr`.** A custom-VJP `regularized_qr` (analogous to
the existing `regularized_svd` in `src/tenax/algorithms/_ad_primitives.py`) is
required before the QR projector can be differentiated through safely. The raw
`jnp.linalg.qr` VJP is **not** sufficient.

**Task 1 Step 5 is TRIGGERED** (near-rank-deficient case FAILED): a
`regularized_qr` custom-VJP must be implemented as a follow-up. Per the tight
spike scope it is *not* implemented in this task — only flagged here.

## Task 2 result

SPIKE — per-SVD cost attribution (`examples/probe_svd_split_attribution_570.py`).
Reuses the fused-backward jaxpr and `svd_vjp` source-attribution of
`examples/probe_bwd_subop_attribution_570.py`; sub-buckets every
`svd_vjp`-attributed backward equation into {M1, M2, M_prime} by the source line
of its originating `tensor_svd(...)` call inside `_ctm_tensor_projector_2x2.py`
(M1≈L879/886, M2≈L889/896, M_prime≈L943/962/971/981), bucketed by proximity to
the three call-site regions. **Zero unattributed svd_vjp ops** — every one
carried a projector-body frame, so the split is complete (not a residual-bucket
estimate).

Re-run at **production scale D=4** (2026-06-09) and reconciled against #589.

Command:
`JAX_PLATFORMS=cpu uv run python examples/probe_svd_split_attribution_570.py --D 4 --chi 8 12 16 --full`

```
  chi   M1_ops   M2_ops  Mprime_ops  total_svd_vjp  total_bwd  (M1+M2)%  svd/bwd%
    8     1980     1692        1512           5184      51581     70.8%     10.1%
   12     1980     1692        1512           5184      51581     70.8%     10.1%
   16     1980     1692        1512           5184      51581     70.8%     10.1%
```

The D=2 χ-sweep (`--D 2 --chi 4 8 12 24`) gives **byte-identical** numbers
(total_bwd=51,581, svd_vjp=5,184 at every χ). The op count is **block-COUNT
driven** (16 charge blocks, identical for even D and saturated already at χ=4),
so the traced jaxpr is invariant across D∈{2,4} and all χ — exactly #589 fact #2
("structural … constant across all χ AND identical at D=2 and D=4").

### Reconciliation with #589 (ROOT CAUSE of the apparent contradiction)

The prior D=2 run looked like a 5–6× **undercount** vs #589 (svd_vjp 10 % vs
36–61 %, FLAT vs growing). It is **not** an undercount — it is the genuine effect
of **PR #593**, which merged between #589 and this branch and vectorized
`_gauge_fix_symmetric_svd` from a **per-column** scatter loop to a **per-sector**
batched op:

- The gauge-fix backward is categorized `svd_vjp` (it is part of each SVD
  decomposition). **Pre-#593** its per-column emission scaled with the number of
  surviving singular values (≈ χ × block size) and was BOTH the dominant svd_vjp
  mass AND the χ-scaling driver #589 measured.
- **#593** collapsed it to one op per charge sector, cutting svd_vjp
  **92,368 → 5,184** at D=4/χ=12 (−18×) and the whole backward **150,621 →
  51,581**, and removing the χ-growth (no per-column loop ⇒ nothing scales with χ
  at fixed block count).
- **Empirically re-verified**: swapping the pre-#593 projector
  (`git show a366165:…/_ctm_tensor_projector_2x2.py`) back in and re-running
  `probe_bwd_subop_attribution_570.py` reproduces #589 **exactly** —
  D=4/χ=12 → total 150,621 / svd_vjp 61.3 %; D=2/χ=12 → 76,893 / 36.2 %. So #589
  is correct for its source snapshot; it is simply **stale**. The lever it sized
  (per-column gauge-fix) is already gone; what remains of svd_vjp is the
  per-sector SVD backward proper (3,216 truncated-SVD ops + 1,680 residual
  per-sector gauge-fix + 288 misc = 5,184).

The probe now prints the `total_bwd` and `svd/bwd%` anchor columns plus a
reconciliation banner, so this comparison is self-evident in its output.

### Split (complete partition, zero unattributed)

- **M1 + M2 = 3672 ops = 70.8 %** of the svd_vjp ops.
- **M_prime = 1512 ops = 29.2 %** of the svd_vjp ops.
- So **(M1+M2) is 2.43× the M_prime share** — the within-svd ratio (~71/29) is
  unchanged from the prior run; M1+M2 is the clear majority of the
  differentiated-SVD op cost.

### Backward-op reduction estimate (D=4/χ=12)

- Post-#593, svd_vjp is **10.1 %** of the whole fused backward
  (total_backward = 51,581; total_svd_vjp = 5,184).
- The QR drop-in replaces M1+M2 (3672 ops) with QR but keeps M_prime as SVD.
- Using the per-sector QR-vs-SVD VJP op ratio ≈ **2.6×** from Task 1
  (`probe_decomp_vjp_cost_570.py`): expected end-to-end backward-op reduction
  ≈ `(M1+M2)/total_backward × (1 − 1/2.6)` = `3672/51581 × 0.615` ≈ **4.4 %**.
- Within the svd_vjp slice alone: `0.708 × (1 − 1/2.6)` ≈ **43.5 %** of svd_vjp
  ops, i.e. svd_vjp shrinks from ~10.1 % → ~5.7 % of the backward.

**The headline-win picture CHANGED vs the original spec.** The spec's
"svd_vjp 61 % → ~35 %, ≈25–26 % fewer total backward ops" (lines 84–85) was
sized against the **pre-#593** 61 % baseline and is now **obsolete**: #593
already captured the bulk of that win (the per-column gauge-fix) for the SVD
path. The QR drop-in's *remaining* reachable end-to-end op reduction is **~4.4 %**
(not ~25 %).

### Verdict: **GO on attribution, but DOWNGRADED end-to-end expectation**

GO on the *attribution* criterion: M1+M2 (70.8 %) is the dominant slice of the
remaining differentiated-SVD op cost (~2.4× M_prime), so a QR drop-in still
targets the right two SVDs.

**Material caveat (was a minor caveat pre-#593, now the headline):** post-#593,
svd_vjp is only ~10 % of the whole backward, so the projected end-to-end
backward-**op** reduction is ~4.4 %, not the ~25 % the spec assumed. The case for
proceeding now rests on the *compile-time* (not op-count) argument: SVD
custom-call lowerings are disproportionately expensive to compile (op COUNT
under-weights them — see the `kernels` column; 48 SVD kernels, χ-flat), and the
QR drop-in removes 2 of the 3 per-sector SVD lowerings — a benefit op-count does
not capture. **The go/no-go for the full build (Tasks 3–9) should therefore hinge
on the A100 compile-time measurement (Task 8), not this op fraction.** If the
A100 number does not show a real compile win, this is a NO-GO — the op-count case
alone (~4.4 %) no longer justifies the build post-#593.

## Compile micro-spike result

Probe `examples/probe_svd_vs_qr_compile_570.py` (CPU, x64). The decisive
**compile-time** test the Task-2 caveat deferred to: it isolates a *single*
symmetric decomposition — the exact `tenax.linalg.svd(M, …, max_singular_values=None)`
the projector calls for M1/M2 vs `tenax.linalg.qr(M, …)` — and measures the
forward (`jax.jit(f)`) and backward (`jax.jit(jax.grad(f))`) **lowered-HLO
instruction count** (deterministic) and **cold wall compile time** (fresh on-disk
cache dir per compile, warm-once then median of N cold reps). No projector build,
no CTM run — pure decomposition compile attribution.

**M is synthetic-representative** (the spec's sanctioned fallback): a 4-leg
`(chi_L,d_L)×(chi_R,d_R)` U(1) `SymmetricTensor` with χ-leg charges {-1,0,0,1}
repeated to length χ and D-leg charges {-1,0,0,1} (D=4) / {0,1} (D=2), grouped to
a (χD × χD) matrix carrying the multi-sector block structure of a real M1/M2
(D=4/χ=12 → χD=48, 9 fused-charge sectors), decomposed by the *same* production
`tenax.linalg.svd`/`qr`. (Driving the full enlarged-corner fixture for a
byte-real M1_T is unnecessary for an isolated-decomposition compile measurement
and was avoided to keep the spike cheap; the synthetic M's per-sector lowering is
production-faithful.)

Commands:
```
JAX_PLATFORMS=cpu uv run python examples/probe_svd_vs_qr_compile_570.py --D 4 --chi 12 --reps 5
JAX_PLATFORMS=cpu uv run python examples/probe_svd_vs_qr_compile_570.py --D 2 --chi 12 --reps 3
```

| decomp | size (χD) | fwd_HLO | bwd_HLO | fwd_compile_s | bwd_compile_s |
|---|---|---|---|---|---|
| **D=4, χ=12** | | | | | |
| SVD | 48×48 | 1144 | 3253 | 0.21 | 0.45 |
| QR  | 48×48 |  541 | 1451 | 0.16 | 0.31 |
| 3×SVD (current, distinct inputs) | 48×48 | — | 9679 | — | 1.27 |
| 2×QR+1×SVD (drop-in) | 48×48 | — | 6108 | — | 0.89 |
| **D=2, χ=12** | | | | | |
| SVD | 24×24 | 761 | 2299 | 0.16 | 0.29 |
| QR  | 24×24 | 309 |  911 | 0.11 | 0.20 |
| 3×SVD (current) | 24×24 | — | 6812 | — | 0.72 |
| 2×QR+1×SVD (drop-in) | 24×24 | — | 4061 | — | 0.55 |

**Headline — SVD/QR backward ratios:**

| metric | D=4/χ=12 | D=2/χ=12 |
|---|---|---|
| single-decomp backward **HLO** ratio | **2.24×** | **2.52×** |
| single-decomp backward **compile-time** ratio | **1.47×** | **1.43×** |
| composite per-projector **HLO** ratio (3×SVD ÷ 2QR+1SVD) | **1.58×** | **1.68×** |
| composite per-projector **compile-time** ratio | **1.42×** | **1.32×** |

Reliability: HLO instruction counts are **byte-deterministic** (identical across
reps and across the D=2/D=4 verification runs). Cold compile-time ratios reproduce
within ≈0.05× across reps (single-decomp 1.43–1.48×, composite 1.32–1.42×), so the
ratios' magnitude is trustworthy. The composite uses three *distinct* rescaled
inputs (1.0/1.3/0.7·M) — an earlier reuse-`m` version let XLA CSE 3×SVD → 1×SVD
and reported a false ~1.0× composite; with distinct inputs the 3×SVD HLO is
≈3× the single SVD, as expected.

### Verdict: **MARGINAL → recommend the small end-to-end A100 check (Task 8) before the full build**

- **A single symmetric SVD backward does compile to ≈2.2–2.5× the HLO of a QR
  backward** — the per-decomposition op-emission belief is *confirmed* and crosses
  the GO line on HLO. So XLA does lower the SVD VJP into materially more HLO than
  QR, as hypothesized.
- **But the quantity that actually matters — the per-projector composite compile
  time — is only ≈1.3–1.4×, and the composite HLO only ≈1.6×.** Two reasons the
  per-decomp 2.2× does not carry through: (a) the drop-in keeps the M′ truncating
  SVD (QR cannot truncate to χ), and (b) two QR backwards still emit real HLO, so
  replacing 2-of-3 SVDs nets ≈37 % fewer HLO, not 56 %. Cold compile time scales
  *sub-linearly* in HLO here (the SVD/QR compile-time ratio 1.47× is well below the
  HLO ratio 2.24×), so the realized per-projector compile win is even smaller than
  the HLO delta.
- Both the composite compile-time (≈1.3–1.4×) and composite HLO (≈1.6×) ratios sit
  **inside the spec's MARGINAL band (1.3×–2×)**, stable across D∈{2,4}. This is
  **not** the ≳2× the GO rule requires, but it is clearly **above the <1.3× NO-GO
  floor**.

**Recommendation:** MARGINAL. The drop-in plausibly shaves a per-projector
compile fraction (~25–30 % of the differentiated-decomposition compile, gated by
the retained M′ SVD), but the spike cannot promise the ≳2× that would make it a
slam-dunk. Per the decision rule, run a **small end-to-end A100 compile-time check
(Task 8: `profile_570_sweepvjp_compile.py`, svd vs a prototype qr projector, D=4
χ∈{8,12,16})** *before* committing to the full Tasks 3–9 build. If that
end-to-end number lands ≳1.5× on the whole fused backward compile, proceed; if it
washes out to <1.3× (likely, since the projector decomposition compile is itself
only a slice of the fused-backward compile, and #593 already removed the
per-column gauge-fix mass), it is a NO-GO and #593 captured the reachable win.
