# Honeycomb CTM — G1–G5 resolution from papers 1 & 2

**Sources** (PDFs now in `papers/`):
- **Paper 1.** Lukin & Sotnikov, *Variational optimization of tensor-network states with the honeycomb-lattice corner transfer matrix*, PRB **107**, 054424 (2023); arXiv:2209.03428.
- **Paper 2.** Lukin & Sotnikov, *Corner transfer matrix renormalization group approach in the zoo of Archimedean lattices*, PRE **109**, 045305 (2024); arXiv:2401.07274. §II.C "Remarks on the honeycomb lattice".

(Both authored by the same group. Design doc citation list omitted Paper 2's authors.)

## Algorithm summary as actually published

### Uniform case (A = B), Paper 1 — Fig. 1

Per direction α, environment is **3 tensors**: one corner `C` (rank-2, χ × χ) and two row tensors `L, R` (rank-3, χ × D² × χ). With C₃ + reflection symmetry these collapse to a single (C, L, R) tuple by symmetry; storing 3 per direction is the general (non-symmetric-A) form.

Update equations (Paper 1, Eqs. 2–4):

```
C → L · C · R · T²       (corner absorbs both row tensors and two bulk T's)
L → L · T²               (row tensors absorb two bulk T's)
R → R · T²
```

`T²` = two copies of the bulk double-layer tensor (one A-sublattice, one B-sublattice) absorbed in the same step. Truncate via eigh of updated C (or SVD); absorb truncation isometry into L and R.

This is exactly the design's `(C_α, L_α, R_α)` triple and confirms our existing env structure.

### Two-site case (A ≠ B), Paper 2 — Fig. 10

Per direction α, environment doubles: `(C^A_α, L^A_α, R^A_α)` and `(C^B_α, L^B_α, R^B_α)`. Total = 9 fields × 2 sublattices = **18 fields**. Matches design.

**Critical change vs uniform:** projectors are **biorthogonal**, not isometric. Direct quote from Paper 2 §II.C:

> "The key difference from the case with a one-site unit cell is that there are two different projection tensors P_L and P_R, which are no longer isometric, but just biorthogonal, P_L P_R = 1."

Construction (Fig. 10(d)): biorthogonalization via QR + SVD on the enlarged corner matrices.

## G1–G5 resolutions

### G1. Absorbed boundary block for an α-direction move

For each sublattice s ∈ {A, B}, the C-block update (per Paper 1 Eq. 2, extended to A ≠ B) absorbs **5 tensors**:

```
C^s_α ← L^s_α · C^s_α · R^s_α · T^s · T^(other s)
```

The two T's come from the bipartite alternation across the absorbed honeycomb edge. The L- and R-block updates absorb **one T each** (Path 1 resolution, 2026-04-27):

```
L^s_α ← L^s_α · T^s             (L absorbs the same-sublattice bulk site)
R^s_α ← R^s_α · T^(other s)     (R absorbs the bipartite neighbor)
```

This asymmetric pairing is what gives the 5-tensor corner formula naturally: `C^s^new_unproj = L^s_new · C^s · R^s_new = L^s · C^s · R^s · T^s · T^(other s)` — one T_self from the L side, one T_other from the R side. The L→R chirality is the Lukin-Sotnikov 6-corner convention (Paper 1 Fig. 1(b)).

Total tensors entering per α-move: 5 + 1 + 1 = 7 per sublattice, ×2 sublattices = 14; with sharing, unique tensor count is 5 per sublattice (L, C, R, T_A, T_B).

### G2. Updated env fields per α-move

Six fields per α-move (six of 18 total):

```
{C^A_α, L^A_α, R^A_α, C^B_α, L^B_α, R^B_α}
```

Other 12 fields (β ∈ {α', α''} × 6 fields each) are untouched by this move.

### G3. Joint vs separate projectors

**Two sublattice-separate projectors per direction**, P_L^α and P_R^α — paper's wording:

> "two different projection tensors P_L and P_R, which are no longer isometric, but just biorthogonal, P_L P_R = 1."

Constructed from the enlarged corner matrix via QR + SVD biorthogonalization (Fig. 10(d)).

**Note on design choice S3:** Design picked "isometric SVD/eigh projectors" with biorthogonal as a follow-up. Paper says biorthogonal is required for A ≠ B. Trade-off:
- Isometric (S3): simpler, reuses existing `_ctm_projector.py` machinery, may underconverge or fail to reach the correct fixed point for highly A ≠ B states.
- Biorthogonal (paper-faithful): correct algorithm, requires new biorthogonalization helper. Roughly Task 5's complexity again.

### G4. Per-absorption phase fix

Paper 1 and Paper 2 don't discuss phase fixing explicitly — the QR + SVD biorthogonalization is itself gauge-fixing. For the existing variPEPS-style implementation: **once per sublattice update during the α-move** (i.e., applied separately when computing P_L^α and P_R^α). Two phase fixes per α-move, one per projector.

If we keep S3's isometric simplification: phase fix once per sublattice update (so still 2 per α-move).

### G5. Sigma gauge wiring

Paper 2 doesn't use sigma gauge explicitly — the biorthogonalization replaces it. For the design's S3 (isometric) variant, sigma gauge would be applied to the new C^s_α corners after each α-move. For the paper-faithful biorthogonal variant, sigma gauge is unnecessary: biorthogonalization absorbs the gauge dof.

## Implications for committed Tasks 1–5

| Component | Status | Paper-faithful? |
|---|---|---|
| `HoneycombCTMEnv` (9 fields/sublattice) | committed (Task 1) | ✅ yes (matches Paper 1 + Paper 2 with S=A,B) |
| `HONEYCOMB_NEIGHBORS` map | committed (Task 2) | ✅ yes |
| `_double_layer_honeycomb` (rank-3 T) | committed (Task 3) | ✅ yes |
| `initialize_honeycomb_env` | committed (Task 4) | ✅ yes |
| `compute_honeycomb_projector` (isometric) | committed (Task 5) | ⚠️ S3 simplification — paper requires biorthogonal for A ≠ B |

**Tasks 1–4 stand as-is.** Only Task 5 (projector) is in tension with the paper.

## Open decision

Task 5 is already merged with isometric projectors. Two paths for v1:

**Path A — Keep isometric (S3 as-is, defer biorthogonal).** Lower risk on the existing scaffolding. v1 may underconverge for strongly A ≠ B states; we accept this and track via the M2a Lukin-Sotnikov regression (uniform A=B, where isometric works). M2b kagome iPESS smoke (which IS A ≠ B) becomes a known-risk gate — if it diverges, we add biorthogonal in v2.

**Path B — Replace isometric with biorthogonal now.** Faithful to Paper 2 for A ≠ B. Adds ~Task-5-complexity to the schedule (build biorthogonalization via QR + SVD on enlarged corners; rewrite Task 5 tests for the biorthogonal API). Reduces risk of M2b failure.

**Path C — Hybrid.** Keep isometric as the default, add biorthogonal as a `projector_method="biorthogonal"` option (currently raises `NotImplementedError`). Lets v1 ship faster, allows opt-in to the paper-faithful path for A ≠ B. Roughly +50% Task 5 work on top of A.
