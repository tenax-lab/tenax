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
