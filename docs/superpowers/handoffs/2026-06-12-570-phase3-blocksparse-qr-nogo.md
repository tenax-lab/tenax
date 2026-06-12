# #570 Phase 3 — block-sparse reduced-corner QR: feasibility spike = **NO-GO**

**Date:** 2026-06-12
**Branch:** `feat/qr-ctmrg-phase3-blocksparse-570` (off `main` @ #597)
**Verdict:** **NO-GO.** The block-sparse (SymmetricTensor) reduced-corner QR projector
does **not** move the CTM-AD sweep-VJP compile wall. Measured on A100: QR is ~5–7%
**worse** HLO than the per-sector SVD baseline, with **no** χ-scaling advantage. Do
**not** build the full Phase 3. The decomposition is the wrong lever; the wall is the
#566 structural (pack/unpack + contraction) emission, which is identical for SVD and QR.

This closes the QR-CTMRG-as-compile-win thread for #570. Phases 1–2 (dense forward +
implicit-AD) remain correct and merged (#594–#597) — they are *usable* (a faithful
reduced-corner QR-CTMRG under AD) but confer **no compile or runtime win**, as the
dense end-to-end benchmark already showed (a wash) and this block-sparse spike now
confirms at the per-sector level.

---

## The question the spike settled

Phases 1–2 proved QR-CTMRG correct under AD but **not faster** (dense, small-D: a wash).
The Phase-3 hypothesis (the only remaining one for #570) was: going **block-sparse**
flips the wash, because the isolated per-sector **SVD-VJP is 2.2–2.5× the HLO of the
per-sector QR-VJP**, and that advantage might finally dominate the sweep-VJP at
production scale (D=4, χ≥12, fermionic).

The kickoff GO/NO-GO rule: full **sweep-VJP** (not isolated decomposition) bwd HLO /
compile ratio **≥ ~1.5× → build**; **< ~1.2× → NO-GO**.

## What was measured

All on the A100 box, fermionic fPEPS site (`_build_initial_fpeps_tensor`, FermionParity
/ FermionicU1, 16 charge blocks), `jax_enable_x64`, cold `lower().compile()` of the
gauge-fixed CTM **sweep-VJP** (`apply_Jt_env`, the dominant repeated unit of the fused
backward), via `examples/profile_570_sweepvjp_compile.py` (now `--recipe`-aware).

### 1. Isolated decomposition (re-confirmed, CPU) — premise holds in isolation
`examples/probe_svd_vs_qr_compile_570.py --D 4 --chi 12`:
- symmetric **SVD/QR backward HLO = 2.24×**, compile 1.47×.
- reduced-corner composite (2QR+1SVD vs 3SVD) HLO 1.58×, compile 1.41×.

The per-sector decomposition advantage is real. The spike tests whether it *survives*
the full sweep-VJP.

### 2. Baselines (A100, D=4 χ=12, env sweep-VJP)
| path | HLO | compile |
|---|---|---|
| 2×2 SVD (production wall, Fishman) | 28786 | 14.16s |
| 1×1 SVD (current dense fallback) | 22401 | 9.95s |
| 1×1 QR (current dense fallback) | 24042 | 10.39s |
| 1×1 eigh (current dense fallback) | 21671 | 11.20s |

Note: at the **dense** 1×1 level QR is already *slightly worse* than SVD (it runs
`regularized_qr` **and** `regularized_svd`) — the Phase-2 "wash" mechanism. All three
1×1 methods are near-identical because the traced backward **densifies** regardless of
`projector_method`.

### 3. **Decisive: block-sparse traced sweep-VJP, 1×1, D=4 χ-sweep**
Built a traced block-sparse reduced-corner projector with a `decomp ∈ {qr, svd}` switch
(see instrument below), wired into the *traced* SymmetricTensor branches, so svd-vs-qr
share **identical** block structure and isolate **exactly** the per-sector decomposition VJP:

| χ | svd HLO | qr HLO | **svd/qr HLO** | svd cmp | qr cmp | **svd/qr cmp** |
|---|---|---|---|---|---|---|
| 12 | 22401 | 24042 | **0.93×** | 9.95s | 10.39s | 0.96× |
| 18 | 20846 | 22474 | **0.93×** | 11.46s | 11.90s | 0.96× |
| 24 | 21593 | 22584 | **0.96×** | 11.93s | 12.26s | 0.97× |

JSON: `examples/profile_570_phase3_spike_1x1_blocksparse.json`,
`..._baseline_1x1.json`, `..._baseline_2x2svd.json`.

## Why NO-GO (the mechanism)

- The ratio is **inverted** (≈0.93–0.96×): block-sparse QR is *worse*, not ≥1.5× better.
  It fails the GO gate and is even below the NO-GO floor (it never reaches 1.2×).
- **No χ-scaling.** Both svd and qr sweep-VJP HLO are flat-to-declining in χ — the
  decomposition is **not** the χ-driver (consistent with `probe_svd_split_attribution_570`:
  post-#593 the per-sector decomposition VJP is block-count-driven, flat in χ).
- **Amdahl.** The 2.24× lives on a tiny, χ-flat slice of a ~22k-HLO sweep-VJP that is
  dominated by per-sector pack/unpack + gauge-fix + contraction emission (#566), which is
  **byte-identical** between the SVD and QR paths. 2.24× on ~5% of the work ≈ nothing.
- The reduced-corner QR path does **two** per-sector decompositions (thin QR **and** a
  tiny `eigh(R Rᴴ)`) plus a QR gauge-fix, so its decomposition slice is actually a hair
  *larger* than a single per-sector SVD — tipping the full ratio below 1.0.
- Even the most optimistic variant (Candidate A: pure `QR(C1g)`, drop the tiny eigh —
  but it only holds at the C₄ᵥ fixed point, not general multisite cuts) would at best
  remove a small eigh slice → land at a **wash**, not 1.5×.

This is the same conclusion the whole #570 saga converged on from the other direction:
**the compile wall is structural (#566), not the decomposition.** #593 already banked the
one decomposition-adjacent win (the per-column gauge-fix → per-sector vectorization, 6.5×).
Swapping the decomposition itself (SVD→QR), in *any* form — dense drop-in, dense faithful
reduced-corner, or block-sparse reduced-corner — does not move it.

## The instrument (env-gated; reverted from `src` to keep core pristine)

`examples/profile_570_sweepvjp_compile.py` gained a `--recipe {2x2,1x1}` flag (kept — useful
infra). The traced block-sparse projector hook was added to
`src/tenax/algorithms/_ctm_projector.py` behind `TENAX_P3_BLOCKSPARSE_SPIKE`
(off ⇒ byte-identical; verified: `tests/test_reduced_corner_qr.py -m core` = 7 passed)
and then **reverted** (NO-GO ⇒ no production value). Full diff archived at
`docs/superpowers/handoffs/570_results/2026-06-12-blocksparse-qr-instrument.diff` —
re-apply with `git apply` to reproduce. Core of it:

```python
def _reduced_proj_symmetric_traced(C1g, C4g, chi, base_charges, decomp="qr"):
    # Per fused sector q: M_q = concat(both corners' column blocks).
    #   decomp="qr":  Q,R = qr(M_q); gauge-fix diag(R)>=0; eigh(R Rᴴ); top-k_q -> P_q = Q@V
    #   decomp="svd": U,_,_ = svd(M_q); P_q = U[:, :k_q]
    # k_q allocated STATICALLY from base_charges via _derive_charges (concrete under
    # tracing — no host argsort of tracer eigenvalues), so the whole thing is traceable.
    # Single isometry P=(fused, chi_new); identical block structure for svd & qr.
```
Wired into the `svd`/`qr` branches of `_compute_projector_tensor` to intercept the
**traced** SymmetricTensor case (which otherwise densifies) when the env var is set.

Key enabler: `base_charges` (from `_get_base_charges` — the double-layer `u2` index
charges) is **structural/concrete**, so per-sector k_q truncation is static even under
`jax.vjp`. This is exactly how the 2×2 traced wall (`_retruncate_by_base_charges`)
stays block-sparse + traceable.

## Disposition / what to do next

- **Do not build full Phase 3.** Hypothesis falsified.
- #570 "QR projector as compile lever" is **exhausted** across all forms.
- The genuine remaining compile lever is unchanged: the **cross-call full-sweep stacked
  rep** (#586 territory — stack the per-sector structural pack/unpack + contraction ops
  across sectors *and across the ~387 sweep contract() calls*), high blast radius, out of
  any QR scope. That is the only thing that touches the #566 structural ~60% bucket.
- Keep: `--recipe` profiler flag, the JSON evidence, this writeup. Phases 1–2 stay
  merged as a correct (if not faster) reduced-corner QR-CTMRG option.
