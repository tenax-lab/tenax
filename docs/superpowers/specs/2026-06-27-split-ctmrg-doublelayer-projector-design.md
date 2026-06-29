# Split-CTM correct double-layer projector (DenseTensor, corner-pair) — design

**Date:** 2026-06-27
**Issue:** #463 (split-CTM canonicalization) — Phase-1-completion prerequisite that unblocks Phase 2 (`fuse_virtual_legs`).
**Scope:** DenseTensor (bosonic) split-CTM **forward** only. SymmetricTensor/fermionic and split-AD stability are deferred.
**Status:** fix direction **proven by spike** (2026-06-27); see §Spike evidence.

## Problem (root cause — verified)

The split-CTM forward (`ctm_split_tensor`, moves in `_split_ctm_tensor_moves.py`) converges to the **wrong fixed point**. On a random D=2 iPEPS at χ=8, native split energy = 0.0743 vs the trusted fused path 0.00186 — **~40× wrong**.

**Cause:** the directional moves build renormalization projectors from **single-layer** grown corners — e.g. `_split_ctm_move_left` feeds `_compute_projector_tensor` the **ket-only** corner `C1g_ket = contract(C1_r, T1_ket)`. A single-layer corner never sees the bra layer, so the projector truncates the environment bond using only ket information and selects the wrong χ directions. The fused path is correct because it builds the projector from the **double-layer** grown corner.

**Why missed:** `test_compute_energy_split_*_matches_shim` only checks the split energy fn vs the fused RDM *on the same (wrong) env* (self-consistency ~1e-16); and `ctm_split_tensor`'s `conv_tol` check false-triggers on an early transient plateau.

> **Corner rank is NOT a valid signal.** For a 1-site uniform iPEPS the *correct* boundary is genuinely **rank-1** (unique dominant transfer eigenvalue), verified across 6 seeds × D∈{2,3}. The native bug shows as a *rank-inflated* corner **and** wrong energy; energy is the discriminator.

## Spike evidence (2026-06-27)

A hybrid sweep (split-stored env, proven fused move under the hood, lossless χ_I=χ·D) established:
- **(A)** fused→split→`compute_energy_split_ctm_tensor` equals the fused energy to **|Δ|=7.6e-17** → split **storage + energy fn are correct**; the bug is purely the moves' projector.
- The `_fused_env_to_split` round-trip is exact (1e-17) → **at lossless χ_I the split path equals the fused path at the same χ**.
- **(B)** native per-layer split is ~40× wrong.

Conclusion: a correct **double-layer corner-pair** projector with split-stored envs reproduces the fused path exactly at lossless χ_I, with no paper figures required.

## Goal & correctness oracle

Replace the per-layer projector in the split moves with the correct **double-layer corner-pair** projector (the same construction Tenax's fused path uses and `_compute_projector_tensor` implements), keeping the environment stored/applied in split (ket/bra) form so the memory stays `O(χ²·D⁴)`.

**Oracle (exact parity at lossless χ_I — proven achievable):**
1. **Primary:** `ctm_split_tensor(A, χ, χ_I=χ·D)` energy via `compute_energy_split_ctm_tensor` **==** the fused single-site energy at the **same χ**, to **1e-8**, for random DenseTensor A (several seeds), D∈{2,3}, χ∈{4,6,8}. (Exact because corner-pair split at lossless χ_I is a faithful factorization of the fused move — spike (A).)
2. **Production χ_I:** with `χ_I = χ`, the split energy *converges* toward the fused/large-χ value as χ grows (no exact equality required); no unphysical (>0.75/bond) values.
3. **Convergence honesty:** `ctm_split_tensor` energy is stable / sweep-count-independent at increasing `max_iter` (transient-plateau false break removed).

(Corner-rank tests dropped — invalid for the 1-site ansatz, per above.)

## Design (Approach A): native double-layer corner-pair projector

Localized to **Phase A** (projector construction + corner renormalization) of the four directional moves; the edge SVD-split (Phase C) and bounded edge application are reused.

### Component 1 — double-layer grown corners

For each move, grow the corner with **both** ket and bra edges instead of one. Left move (mirror the fused `_ctm_tensor_move_left`, `_ctm_tensor_moves.py:997`): build the double-layer grown corner `C1g_dl` by contracting `C1` with `T1_ket` **and** `T1_bra` joined over their interlayer (`_I`) bond, fusing `(env, u_ket, u_bra)` → the `fused` leg (dim χ·D²) the projector truncates. Likewise `C4g_dl`. Memory: the grown corner is χ·D² — cheap; the χ²·D⁴ budget lives in the edge step (Component 3).

### Component 2 — projector via the existing kernel

Call `_compute_projector_tensor(C1g_dl_fused, C4g_dl_fused, chi, projector_method="svd", projector_backward=...)` — the *same* Fishman two-projector kernel the fused path uses. Returns the biorthogonal pair `(P_1, P_2)`, `P_1†P_2=I`, each mapping `(env, u_ket, u_bra)` → χ. Apply `P_1` to the first-corner side and `P_2` to the second, following the fused move's `_apply_projector_with_reembed` convention (the current split code's single-`P` application is replaced).

### Component 3 — keep the edge split: factorize the projector

The projector `P` acts on `(env, u_ket, u_bra)`. To renormalize the **split** edge without forming the χ²·D⁶ double-layer edge, SVD-factorize each projector across `(env, u_ket) | (u_bra, χ)`:
```
P[(env,ketD),(braD,χ)]  --SVD-->  P_first[(env,ketD), m] · P_second[m, (braD,χ)]   (m ≤ χ·D)
```
`P_first · P_second = P` exactly, so feeding `(P_first, P_second)` into the existing bounded machinery (`_grow_and_project_bounded`, the two-step corner projection) reproduces the true double-layer projection while keeping the edge step χ²·D⁴-bounded. Identity used: `P† C_dl = P_second†(P_first† C_dl)`.

> **Implementation sequencing (de-risk):** first land a *correct* version using the **closed** edge path (`_grow_edge_no_double_layer` + `_project_grown_edge_tensor`, forms the χ²·D⁶ edge) and verify oracle §1; then switch to the factorized bounded path (Component 3) and re-verify §1 unchanged. This separates "is the projector correct" from "is the memory-bounded application correct."

### Component 4 — convergence-criterion guard

Replace `ctm_split_tensor`'s transient-prone `conv_tol` break with a `min_iter` floor (mirror the fused loop) so it tracks the genuine fixed point.

## Data flow (left move)

```
split env (C1..C4, T*_ket, T*_bra), A, A.bar()
  ├ C1g_dl = C1·T1_ket·T1_bra (join interlayer) → fused=(env,ketD,braD), χ·D²   [Component 1]
  ├ (P_1,P_2) = _compute_projector_tensor(C1g_dl, C4g_dl, χ)                      [Component 2]
  ├ factorize each P across (env,ketD)|(braD,χ) → (P_first, P_second), m≤χ·D       [Component 3]
  ├ corners: two-step project → C1_new, C4_new (χ×χ)
  └ edge: _grow_and_project_bounded(T4_ket, T4_bra, A, A_bar, P_first, P_second) → SVD-split → T4_ket/bra (χ_I)
```

## Error handling

- Projector/factorization SVD on rank-deficient inputs reuses the AD-stable rank-aware kernel already used in `_svd_split_edge_tensor` (`truncated_svd_symmetric_ad`) — finite adjoint (matters for later Phase-2 AD; harmless to the forward).
- Default `χ_I = χ`; the oracle §1 pins `χ_I = χ·D` (lossless).

## Testing (DenseTensor, `pytest -m core`, small D/χ)

1. **Exact parity at lossless χ_I (load-bearing).** `ctm_split_tensor(A, χ, χ_I=χ·D)` split energy **==** fused single-site energy at the same χ, **1e-8**, random A (≥3 seeds), D∈{2,3}, χ∈{4,6,8}.
2. **Production-χ_I convergence.** With χ_I=χ, split energy → fused/large-χ value as χ grows; no >0.75/bond values.
3. **Convergence honesty.** Sweep-count-independent energy at increasing `max_iter`.
4. **Bounded == closed.** The factorized bounded edge path equals the closed-edge path (Component 3 sequencing) to 1e-10.
5. **No regression** in `tests/test_split_ctm_tensor.py` and `pytest -m core`.

## Memory & computational scaling (corner-pair vs half-system)

χ = environment bond, D = PEPS bond (paper's χ_B). All *split* variants share the
same leading order; corner-pair vs half-system differ only in prefactor and
accuracy-per-χ, **not** in asymptotic cost.

| | Time | Peak memory |
|---|---|---|
| Fused / conventional (Tenax current) | `O(χ³ D⁶)` | `O(χ² D⁶)` |
| **Corner-pair split (this design)** | `O(χ³ D⁴)` | `O(χ² D⁴)` |
| Half-system split (paper "full" projectors) | `O(χ³ D⁴)` | `O(χ² D⁴)` |
| Half-projectors (paper App. B) | `O(χ³ D⁴)`, smaller prefactor | `O(χ² D⁴)` |

- **Both split variants deliver the full D² advantage** over fused (paper
  Eqs. 11–12). Choosing corner-pair vs half-system is not a scaling decision.
- **Prefactor — corner-pair wins:** its projector is built from a double-layer
  *corner* (χ²D² intermediate, *subleading* to the χ²D⁴ edge step). The
  half-system projector construction sits *at* the leading χ³D⁴ order (the
  paper's "teal" bottleneck).
- **Accuracy per χ — half-system wins:** a half-system/RDM projector is a better
  truncation than a corner (`C·C`) projector, so it reaches a target accuracy at
  smaller χ. If corner-pair needs a modestly larger χ to match, that can erode
  the prefactor edge (cost ~χ³).
- **Implication for the deferral:** corner-pair *fully achieves the memory goal*
  (χ²D⁴ — the thing blocking large-D fermionic/CG iPEPS); the half-system upgrade
  is a pure accuracy-per-χ refinement with no change to the asymptotic budget.
  Caveat: report any large-D accuracy benchmark as "corner-pair at this χ," not
  "paper-equivalent at this χ" (corner-pair may sit at slightly higher truncation
  error at fixed χ).

## Scope / out of scope

- **In:** DenseTensor split forward — double-layer corner-pair projector in the four moves, the factorized bounded application, the convergence guard.
- **Out (deferred):**
  - The **paper's half-system projectors** (Naumann et al. PRB 111 235116, App. A; [[reference_split_ctmrg_paper]]) — more accurate at *fixed* χ_E but a different (convergence-only) truncation and figure-dependent. File as a follow-up accuracy upgrade.
  - SymmetricTensor / fermionic projectors (closed χ²·D⁶ Koszul path stays).
  - Split backward AD-stability (Phase-2 concern).
  - `fuse_virtual_legs` flag wiring (#463 Phase 2 plan) — resumes once oracle §1 passes.

## Risks

- **Factorization/bounded application** (Component 3) is the main new validation; mitigated by the closed-first sequencing + Test 4 (bounded==closed) + Test 1 (parity).
- **Projector-pair convention:** mirror the fused move's `(P_1,P_2)` application (`_apply_projector_with_reembed`) rather than the current split single-`P`.
- **Accuracy vs paper:** corner-pair (this design) is exact-parity with the fused path but the paper reports half-system projectors are more accurate at fixed χ_E; acceptable for unblocking Phase 2, upgrade tracked separately.
