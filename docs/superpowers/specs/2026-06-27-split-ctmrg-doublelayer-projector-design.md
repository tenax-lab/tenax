# Split-CTMRG correct double-layer projectors (DenseTensor) — design

**Date:** 2026-06-27
**Issue:** #463 (split-CTM canonicalization) — Phase-1-completion prerequisite that unblocks Phase 2 (`fuse_virtual_legs`).
**Scope:** DenseTensor (bosonic) split-CTM **forward** only. SymmetricTensor/fermionic and split-AD stability are deferred.
**Reference:** Naumann, Weerda, Eisert, Rizzi, Schmoll, *"Variationally optimizing iPEPS at large bond dimensions: A split-CTMRG approach,"* PRB **111**, 235116 (2025) / arXiv:2502.10298 — the paper `_compute_projector_tensor` already cites (Eq. 10). Main-text §II B + Eqs. 8–10 + Fig. 3; full projector set in **App. A** (Figs. 6–9).

## Problem (root cause — verified 2026-06-26, paper-confirmed)

The split-CTM forward converges to the **wrong fixed point**: energy off 16–108% vs the fused path, unphysical (>0.75/bond) at χ=3, independent of χ and χ_I; the converged corner collapses to **effective rank 1–2** (normalized SVs `[1,0,0,0]` at χ=4).

**Cause:** the directional moves (`_split_ctm_tensor_moves.py`) build renormalization projectors from **single-layer** grown corners — e.g. `_split_ctm_move_left` calls `_compute_projector_tensor(C1g_ket_fused, C4g_ket_fused, chi)` where `C1g_ket` is the **ket-only** grown corner `(chi, D)`. A single-layer corner's new-bond rank is bounded by D, and truncating ket/bra independently discards cross-layer correlation → rank collapse.

**Paper confirms the fix:** in split-CTMRG every projector is built from `M = (ρᴮ)ᵀρᵀ` (Eq. 8) where `ρᴮ, ρᵀ` are **double-layer half-systems** — patches containing *both* ket and bra edges ("approximate a large part of the double-layer network", §II B, Fig. 3). Projectors built from a single layer are not part of the algorithm.

**Why missed:** `test_compute_energy_split_*_matches_shim` only checks the split energy fn vs the fused RDM *on the same (wrong) env* (self-consistency ~1e-16); and `ctm_split_tensor`'s `conv_tol` false-triggers on an early transient plateau.

## Goal & correctness oracle

Reproduce the paper's split-CTMRG so the forward converges to the **correct** environment, while preserving the `O(χ_E³·D⁴)` scaling (vs conventional `O(χ_E³·D⁶)`, paper Eqs. 11–12) that is the reason the split path exists.

**Oracle (convergence, not exact parity).** Per the paper's Fig. 4 benchmark, split-CTMRG is a *different valid* CTMRG truncation that **converges to conventional CTMRG as χ_E grows** (relative energy difference decreasing toward ~1e-8), with **χ_I = χ_E**. It is **not** an exact factorization of the fused path at finite χ_E. So correctness is judged by:
1. On a **physical** iPEPS (a few simple-update steps of the 2D Heisenberg AFM, D=2), split energy `E_split(χ_E, χ_I=χ_E)` **→ the fused converged energy** as χ_E increases, with the relative difference **decreasing monotonically** to ≤1e-4 at modest χ_E (e.g. χ_E≈16–24). (NOT `split(χ)==fused(χ)` at a single χ.)
2. The converged corner is **non-degenerate** (effective rank > 2 at χ_E≥4) — direct regression for the rank-collapse root cause.
3. Energy is **physical** (≤0.75/bond magnitude; near the known D=2 Heisenberg value ≈ −0.66/site once optimized) — no unphysical values at any χ.

Random-PEPS equal-χ parity is **dropped** (the prior spec's `χ_I=χ·D` exact-parity oracle was wrong: random PEPS makes both CTMs erratic, and split≠fused at finite χ by construction).

## Design — the paper's four-projector full scheme

Each directional move uses **four** double-layer projectors (App. A, Fig. 6). For the **left** move:

| projector | applied | truncates | network ρ |
|---|---|---|---|
| **green** | after ket-layer absorption | env bond `χ_E·D → χ_E` | double-layer half-systems `ρᵀ_green, ρᴮ_green` (Fig. 3): top/bottom halves of the vertical strip, each containing C-corners + **both** `T_ket` and `T_bra` edges + ψ and ψ* |
| **teal** | after bra-layer absorption | env bond `→ χ_E` | `ρ_teal` **incorporating the green projectors** (Fig. 7) |
| **yellow** | interlayer | `χ_I·D → χ_I` (no physical) | `ρ_yellow` (Fig. 8); SVD each ρ to rank `χ_I·D` *before* multiplying (precondition + cost, App. A) |
| **red** | interlayer | `→ χ_I` **including the physical leg** | `ρ_red` **incorporating the yellow projectors** (Fig. 9); the only projector that truncates the physical space |

Dependency order per move: **green → teal**, **yellow → red** (bra-side incorporates ket-side). χ_I = χ_E by default.

### Component 1 — reuse the Eq.-10 kernel `_compute_projector_tensor`

The paper's Eqs. 8–10 are exactly the Fishman two-projector construction `_compute_projector_tensor` already implements: `M = (ρᴮ)ᵀρᵀ = U S V†`, then projectors `P_B = ρᵀ Ṽ S̃^{-1/2}`, `P_T = ρᴮ Ũ S̃^{-1/2}` truncated to the target bond. So **no new SVD/projector kernel is written** — each of the four projectors is one `_compute_projector_tensor(ρᴮ, ρᵀ, χ_target)` call, with `ρᴮ, ρᵀ` the appropriate double-layer half-systems (fused on the to-truncate legs) and `χ_target ∈ {χ_E, χ_I}`.

### Component 2 — half-system builders (the new work)

Add per-direction builders that assemble `ρᵀ, ρᴮ` for each of green/teal/yellow/red from the split env tensors (`C1..C4`, `T*_ket`, `T*_bra`) and `A`, `A.bar()`, keeping ket/bra layers separate during contraction so the build stays `O(χ_E³·D⁴)` (never forming the closed χ²·D⁶ object). Exact leg wiring follows Figs. 3, 6–9 of the paper. The four directions (left/top/right/bottom) are rotations of the left-move construction.

### Component 3 — restructure the four moves to the 4-projector flow

Replace the current 2-projector (`P_ket`, `P_bra` from single-layer corners) flow in each `_split_ctm_move_*` with: build the four half-system projector pairs (Component 2 + 1), then absorb ket → apply green (corners) + green/yellow (edge), absorb bra → apply teal + red, producing the new corners and the SVD-split ket/bra edges. Reuse the existing bounded edge contraction (`_grow_and_project_bounded`) and edge SVD-split (`_svd_split_edge_tensor`) where the leg structure matches.

### Component 4 — honest convergence criterion

Once corners are non-degenerate, replace `ctm_split_tensor`'s transient-prone `conv_tol` break with a guard that requires a `min_iter` floor (mirror the fused loop) before an early break, so it tracks the genuine fixed point rather than an early plateau.

## Data flow (left move)

```
split env (C1..C4, T*_ket, T*_bra), A, A.bar()
  │  Component 2: build half-systems ρᵀ,ρᴮ for green, teal, yellow, red (double-layer, O(χ_E³D⁴))
  │  Component 1: P_green = _compute_projector_tensor(ρᴮ_green, ρᵀ_green, χ_E)   (and teal/yellow/red)
  │                (teal uses green; red uses yellow)
  ├ absorb ket layer  → apply green (env, χ_E) + yellow (interlayer, χ_I) ──┐
  ├ absorb bra layer  → apply teal (env, χ_E) + red (interlayer+phys, χ_I) ─┤
  └ new C1,C4 (χ_E×χ_E) and T4_ket,T4_bra (interlayer χ_I)  ←───────────────┘
```

## Error handling

- Reuse the AD-stable rank-aware SVD (`truncated_svd_symmetric_ad`) already used in `_svd_split_edge_tensor` for the projector SVDs on rank-deficient `M` (keeps a finite adjoint; matters for the later Phase-2 AD, harmless to the forward).
- Guard `χ_I ≥ 1`; default `χ_I = χ_E`.

## Testing (DenseTensor, `pytest -m core`, small D/χ)

1. **Convergence to fused (load-bearing).** Physical D=2 iPEPS (≈5 simple-update steps of 2D Heisenberg): `E_split(χ_E, χ_I=χ_E)` relative difference to the fused converged energy **decreases monotonically** across χ_E∈{4,8,12,16,24} and reaches ≤1e-4. (Mirrors paper Fig. 4.)
2. **Corner non-degeneracy (root-cause regression).** Converged split corner C1 has effective rank > 2 at χ_E≥4 (normalized SV spectrum not `[1,0,0,…]`).
3. **Physical energy.** No bond energy exceeds 0.75 in magnitude at any χ; optimized energy approaches ≈ −0.66/site.
4. **Convergence honesty.** `ctm_split_tensor` energy is stable / sweep-count-independent at increasing `max_iter` (transient-plateau false break gone).
5. **χ_I sensitivity.** Energy converges as χ_I → χ_E and beyond (paper Fig. 4(b)); χ_I=χ_E is near-converged.
6. **No regression** in `tests/test_split_ctm_tensor.py` and `pytest -m core`.

## Scope / out of scope

- **In:** DenseTensor split forward — the four half-system projector builders, the four projector applications, the move restructure, the convergence-criterion guard.
- **Out:** SymmetricTensor / fermionic projectors (closed χ²·D⁶ Koszul path stays); the App. B "half projectors" cheaper variant (a possible later optimization); split backward AD-stability; the `fuse_virtual_legs` flag wiring (#463 Phase 2 plan, resumes after this lands and oracle §Testing.1 passes).

## Risks

- **Figure-accurate leg wiring is the dominant risk.** The half-system definitions live in the paper's *figures* (Figs. 3, 6–9), not fully in extractable text. Mitigation: (a) build the half-systems referencing the paper figures directly; (b) the convergence oracle (Test 1) + non-degeneracy (Test 2) + physical-energy (Test 3) tests catch wiring errors as wrong/degenerate fixed points; (c) optionally cross-check a single projector's action against the fused move on a controlled input during implementation.
- **Effort:** this is a faithful reimplementation of a published algorithm's projector set — the largest of the candidate fixes. Decomposable per projector (green first, validated, then teal/yellow/red).
- **teal/red dependency:** bra-side projectors incorporate ket-side ones; build order matters (green→teal, yellow→red).
