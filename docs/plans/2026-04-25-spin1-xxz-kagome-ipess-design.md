# Spin-1 XXZ iPESS on Kagome with AD — Design

**Status:** Design approved. Ready for implementation plan.

**Relation to prior docs:**
- Extends `2026-04-24-pess-kagome-module-design.md` by adding AD optimization of the PESS parameterization.
- The 04-24 design's Convention C (`pess_to_ipeps` → square iPEPS with dummy bond and fused d³ physical leg) is replaced by Convention A (kagome → honeycomb supersites). The existing example code (`examples/kagome_xxz_spin1_pess.py`) is kept as a reference but its `pess_to_ipeps` coarse-graining is not reused.

## Goal

Land a Liao-2019-style differentiable iPESS pipeline for spin-1 XXZ on the kagome lattice. The optimization variables are the PESS tensors themselves — two simplex tensors (T_u, T_d) and three site tensors (R_a, R_b, R_c) — and gradients flow through the coarse-graining and CTM environment back to those primitives.

Hamiltonian:
$$ H = \sum_{\langle i,j \rangle} \left[ S^x_i S^x_j + S^y_i S^y_j + \Delta\, S^z_i S^z_j \right], \qquad d = 3 $$

## Approach

**Ansatz**: standard iPESS on kagome.
- T_u, T_d ∈ ℂ^{D × D × D} — one simplex tensor per up-/down-triangle.
- R_a, R_b, R_c ∈ ℂ^{D × D × d} — one rank-3 site tensor per kagome sublattice, with one virtual leg to T_u, one to T_d, and one physical leg of dim d=3.
- Tensors stored as dense complex128 (variational stability — see `project_complex_tensors_variational.md`). U(1) S^z symmetric variant deferred to a follow-up.

**Coarse-graining (Convention A — kagome → honeycomb supersites)**:
- Up-supersite $A_u^{p_a p_b p_c}_{\ell_1 \ell_2 \ell_3}$: contract T_u with R_a, R_b, R_c on their T_u-facing virtual legs. Three physical legs (the three sublattice physical legs) and three D-bonds going to the three neighboring down-supersites.
- Down-supersite $A_d^{p_a p_b p_c}_{\ell_1 \ell_2 \ell_3}$: same construction with T_d.
- Two-sublattice honeycomb iPEPS. Reuses Tenax's existing honeycomb CTM (memory: `project_varipeps_2site_honeycomb_works.md`).

**Optimization pipeline**:
1. Initialize PESS tensors (random complex128, normalized).
2. Triangle simple-update warm start: alternate up/down triangle gates $e^{-\tau H_\triangle}$ with HOSVD truncation back to D. Several hundred steps decreasing dt.
3. Hand off (T_u, T_d, R_a, R_b, R_c) to L-BFGS via `optimize_gs_ad`. The loss function builds the two honeycomb supersites from PESS primitives, runs implicit-diff CTM with sigma gauge + GMRES backward, computes triangle energy, returns scalar.
4. Use SVD projectors with the $S_{\rm safe}$ NaN-protection pattern (already in `_ctm_projector.py`).

**Reused infrastructure** (no changes needed):
- `optimize_gs_ad` and L-BFGS plumbing.
- Honeycomb 2-sublattice CTM with implicit AD (sigma gauge + JIT-fused GMRES backward, PR #341).
- SVD projectors with phase fix.
- complex128 path.

**New code surface**:
- `src/tenax/algorithms/pess.py` — `IPESSState` (frozen dataclass), `pess_simple_update_triangle`, `pess_simple_update`, `pess_to_honeycomb_supersites`, `kagome_triangle_xxz_hamiltonian`.
- `src/tenax/algorithms/pess_optimize.py` (or extend `ipeps_optimize`) — `optimize_pess_ad(initial_state, hamiltonian, config)` returning optimized `IPESSState` and final energy.
- Loss closure that calls `pess_to_honeycomb_supersites` then existing CTM AD.
- Tests in `tests/test_pess.py` (SU correctness vs example) and `tests/test_pess_ad.py` (AD gradient sanity, energy improvement vs SU).

## Validation

Three milestones, each in a small benchmark script under `examples/`:

1. **Reduce to spin-½, Δ=1** — spin-1/2 kagome AFM. Reproduce Liao 2019 trend (E should improve with D and beat pure SU). Target: $E/N \approx -0.4365$ at D=8 ish, asymptotic to $-0.4378$ at $D=13$.
2. **Spin-1 Heisenberg, Δ=1** — compare against Picot et al. 2015 ($E/N \approx -1.41$ at large D). Confirm AD beats SU at the same D.
3. **Anisotropy sweep** — Δ ∈ {0, 0.5, 1.0, 1.5, 2.0} at fixed D=4, χ=16. Sanity-check sublattice magnetization shows expected XY-vs-Ising trend across Δ=1.

Each validation run produces a JSON with energies, gradient norms, and step counts; checked into `examples/`.

## Out of scope for v1 (deferred follow-ups)

- U(1) S^z-symmetric iPESS (block-sparse simplex/site tensors).
- Excitations / dispersion via `compute_excitations`.
- Spin-spin correlation functions and structure factor (only sublattice magnetization in v1).
- Larger unit cells (chiral $\sqrt{3}\times\sqrt{3}$, 9-site).

## Tradeoffs and risks

- **Why honeycomb (Convention A) over square w/ dummy-bond (Convention C)**: the dummy bond inflates the iPEPS virtual manifold past what the parameterization actually spans, so SVD projectors at AD time act on a degenerate spectrum. Honeycomb supersites avoid this and let the implicit-diff CTM operate on a clean, full-rank ansatz.
- **Why dense not symmetric**: minimizes new code surface for v1. SymmetricTensor AD is merged but vmap-fused performance is still pending; once tuned, lift the parameterization layer onto it.
- **Risk: triangle SU conditioning**. Three-body gates with HOSVD truncation can be ill-conditioned at small D; mitigation is decreasing-dt schedule and bond-matrix regularization (already used in the existing example).
- **Risk: honeycomb-CTM corner-cases at the kagome geometry**. Kagome's 3-fold local symmetry may not be fully captured by 2-sublattice honeycomb CTM if the optimizer drifts off the symmetric submanifold. If observed, fall back to enforcing $A_u = $ permutation of $A_d$ as a soft constraint (does not affect physics for symmetric phases).
