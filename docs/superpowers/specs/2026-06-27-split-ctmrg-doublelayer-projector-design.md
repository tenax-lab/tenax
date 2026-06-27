# Split-CTMRG correct double-layer projector (DenseTensor) — design

**Date:** 2026-06-27
**Issue:** #463 (split-CTM canonicalization) — this is the **Phase-1-completion prerequisite** that unblocks Phase 2 (`fuse_virtual_legs`).
**Scope:** DenseTensor (bosonic) split-CTM forward only. SymmetricTensor / fermionic Koszul path explicitly deferred.

## Problem (root cause, verified 2026-06-26)

The split-CTM forward converges to the **wrong fixed point** — energy off by 16–108% vs the fused path (unphysical >0.75/bond at χ=3 for spin-½), independent of χ and χ_I. The converged corner collapses to **effective rank 1–2** (normalized SVs `[1,0,0,0]` at χ=4).

**Cause:** the directional moves in `_split_ctm_tensor_moves.py` build **per-layer** renormalization projectors. E.g. `_split_ctm_move_left` does
`P_ket = _compute_projector_tensor(C1g_ket_fused, C4g_ket_fused, chi)` where `C1g_ket` is the **ket-only** grown corner of shape `(chi, D)`. A single-layer corner's new-bond rank is bounded by D, and truncating ket and bra **independently** discards the cross-layer correlations that a double-layer projector encodes. The result is a rank-collapsed, physically wrong environment.

**Why it was missed:** `test_compute_energy_split_*_matches_shim` only checks the split energy function against the fused RDM *on the same (wrong) environment* (self-consistency ~1e-16), never physical correctness; and `ctm_split_tensor`'s `conv_tol` check false-triggers on an early transient plateau, returning before the (wrong) true fixed point.

## Goal

Replace the per-layer projector with a **correct double-layer projector** so the split forward reproduces the fused CTM, while preserving the χ²·D⁴ memory bound that is the entire reason the split path exists. Concretely:

**Correctness oracle (oracle-independent):** with a *lossless* interlayer bond `χ_I = χ·D`, the split path is an exact factorization of the fused path, so
`compute_energy_split_ctm_tensor(A, ctm_split_tensor(A, χ, χ_I=χ·D))` **==** the fused single-site energy at the **same χ**, to ~1e-10, for random DenseTensor A. (Random A is fine here precisely because we compare split-vs-fused at *equal* χ, not against a converged physical reference.)

## Design (Approach A): double-layer projector, SVD-factorized into the existing sequential pair

The fix is localized to **Phase A** (projector construction + corner renormalization) of the four directional moves. Phases B (bounded edge growth/projection) and C (SVD split of the edge) are reused unchanged.

### Component 1 — double-layer grown corners

For each move, grow the two corners with the **full double-layer** edge instead of a single layer. Take the left move: currently it grows `C1` with `T1_ket` only. Instead build the double-layer grown corner by contracting `C1` with **both** `T1_ket` and `T1_bra` joined over their shared interlayer (`_I`) bond (and the edge's physical trace already implicit in the ket/bra factorization). The grown corner has a fused leg `(env, ket-D, bra-D)` of dimension `χ·D²` and the surviving column leg(s) used to form the projector cross-product.

- Memory: the double-layer corner is `χ·D²` — cheap; it is **not** the split path's bottleneck. The χ²·D⁴ concern lives only in the *edge* growth, which Phase B already bounds.
- Build both `C1g_dl` and `C4g_dl` (left move); analogously `(C1,C2)` top, `(C2,C3)` right, `(C4,C3)` bottom — mirroring the existing per-move corner pairs.

### Component 2 — correct projector via the existing `_compute_projector_tensor`

Call `_compute_projector_tensor(C1g_dl_fused, C4g_dl_fused, chi, projector_method="svd", projector_backward=...)` on the **double-layer** fused leg. This returns the Fishman biorthogonal pair `(P_1, P_2)` with `P_1†P_2 = I`, each mapping the double-layer leg `(env, ket-D, bra-D)` → `χ`. Apply `P_1` to the first-corner side and `P_2` to the second-corner side, **following the fused move's convention** in `_ctm_tensor_moves.py` (the current split code applies a single `P_1` to both sides — mirror the fused two-projector application instead).

### Component 3 — SVD-factorize each projector into a sequential (ket, bra) pair

The downstream bounded machinery (`_grow_and_project_bounded`, `_precombine_projector_pair`, `_project_grown_edge_tensor`, and the two-step corner projection) consumes a **sequential** projector pair: `P_first` acting on `(env, ket-D)` then `P_second` on `(intermediate, bra-D)`. Produce these by SVD-factorizing each double-layer projector `P` across the `(env, ket-D) | (bra-D, χ)` partition:

```
P[(env,ketD), (braD, χ)]  --SVD-->  P_first[(env,ketD), m] · P_second[m, (braD, χ)]
```

with the factorization bond `m = rank ≤ χ·D` (no truncation — this is an exact rewrite of `P`). Because `P_first · P_second = P` exactly, the existing two-step corner projection and the bounded edge application reproduce the true double-layer projection `P†·C_dl`. The intermediate bond `m ≤ χ·D` keeps the corner and edge steps χ²·D⁴-bounded.

> Key correctness identity: `P† C1g_dl = P_second†(P_first† C1g_dl)`, and `P_first†` applied after ket-growth + `P_second†` after bra-growth is exactly the existing two-step structure — so only *how `P_first`/`P_second` are computed* changes, not how they are applied.

### Component 4 — convergence criterion fix (forward only)

`ctm_split_tensor`'s `conv_tol` check currently false-triggers on a transient plateau. Once Component 1–3 make the corner non-degenerate, re-verify convergence: the corner singular-value-change criterion should now track a genuine fixed point. If a transient plateau still trips it, require a minimum iteration count (mirror `min_iter` from the fused loop) before allowing an early break. This is a small forward-loop guard, not part of the projector math.

## Data flow (left move, illustrative)

```
C1, T1_ket, T1_bra ─join interlayer→ C1g_dl  (fused=(env,ketD,braD), χ·D²)
C4, T3_ket, T3_bra ─join interlayer→ C4g_dl
        │
        └─ _compute_projector_tensor(C1g_dl, C4g_dl, χ) → (P_1, P_2)   [double-layer, correct spectrum]
                │  SVD-factorize each across (env,ketD)|(braD,χ)
                └→ (P1_first, P1_second), (P2_first, P2_second)   [m ≤ χ·D]
                        │
   corners:  two-step project C1g/C4g with (P_first, P_second)  → C1_new, C4_new  (χ×χ)
   edge:     _grow_and_project_bounded(T4_ket, T4_bra, A, A_bar, P_first, P_second, …) → χ²·D⁴-bounded
                        │  Phase C: SVD-split edge → T4_ket_new, T4_bra_new (interlayer χ_I)
```

## Error handling

- The factorization SVD on rank-deficient `P` reuses the AD-stable kernel already used in `_svd_split_edge_tensor` (`truncated_svd_symmetric_ad`) so the backward stays finite (relevant later for Phase 2 AD; harmless for the forward).
- Keep `χ_I` configurable; default `χ_I = χ` for production memory, but the **correctness test pins `χ_I = χ·D`** (lossless).

## Testing

DenseTensor, `pytest -m core`, small D/χ.

1. **Faithful-factorization parity (load-bearing).** For random DenseTensor A (several seeds), D∈{2,3}, χ∈{D²…}, with `χ_I = χ·D`:
   `compute_energy_split_ctm_tensor(A, ctm_split_tensor(A, χ, χ_I=χ·D, max_iter=large, conv_tol tight))` **==** fused single-site energy at the same χ, to **1e-10**. This is the primary correctness gate.
2. **Corner non-degeneracy (regression for the root cause).** The converged split corner C1 has effective rank > 2 at χ ≥ 4 (i.e. the `[1,0,0,0]` collapse is gone): assert the normalized singular spectrum has ≥ min(χ, D²) entries above a threshold.
3. **χ / χ_I monotone refinement.** Split energy changes with χ (no longer pinned) and converges as χ_I → χ·D; no unphysical (>0.75/bond) values.
4. **Convergence honesty.** `ctm_split_tensor` at increasing `max_iter` gives a stable, sweep-count-independent energy (the transient-plateau false break is gone).
5. **No regression** to existing split tests (`test_split_ctm_tensor.py`) and `pytest -m core`.

## Scope / out of scope

- **In:** DenseTensor split forward (the four moves' Phase A + the convergence guard).
- **Out:** SymmetricTensor / fermionic projector (the closed χ²·D⁶ Koszul path stays as-is; its correctness is a separate follow-up). The `fuse_virtual_legs` flag wiring (#463 Phase 2) resumes *after* this lands — its plan (`docs/superpowers/plans/2026-06-26-...`) is unblocked once the oracle in §Testing passes.
- **Out:** AD-stability of the split backward (gauge fixing, etc.) — a Phase-2 concern; this design fixes only the *forward* correctness.

## Risks

- **Projector-pair convention:** the current split code applies a single `P_1` to both corners. Mirroring the fused move's `(P_1, P_2)` biorthogonal application is required; if the fused convention is subtle, read `_ctm_tensor_moves.py` carefully (the plan's first task).
- **Leg-order bookkeeping:** the SVD factorization partition `(env,ketD)|(braD,χ)` must match the leg labels each move's bounded application expects (`left_fuse`/`right_fuse` tuples). Per-move relabel maps are an implementation detail for the plan.
- **χ_I = χ·D cost in the test:** lossless χ_I raises edge memory in the *test* (still tiny at D=2,3). Production keeps χ_I = χ.
