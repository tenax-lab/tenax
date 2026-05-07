# Multisite-CTM RDM helpers — surgical fix attempt

**Date:** 2026-05-07
**Branch:** `worktree-fix-multisite-ctm-rdm-helpers` (worktree at `.claude/worktrees/fix-multisite-ctm-rdm-helpers`)
**Status:** brainstorming → planning

## Problem

`c5_smoking_gun_bf_vs_ctm_at_optimum.py` (uncommitted in `.worktrees/multisite-kagome-pess`) at D=4 χ=16:

| state | E_BF/site | E_CTM/site | gap |
|---|---|---|---|
| AD optimum | −0.044 | −0.913 | −0.869 |

Multisite-CTM RDMs misrepresent uncorrelated states as deeply ordered, and the L-BFGS optimiser exploits the bias. Five of six BF bond energies are zero-to-positive at the optimum; CTM reports all six between −0.37 and −0.57.

Ruled out: AD gradient (Wirtinger-correct), warm-start, optimizer convention, χ truncation. Remaining suspects: `_rdm2x1_tensor_2site` / `_rdm1x2_tensor_2site` (in `_ctm_tensor_energy.py:531/601`), `_rdm_3site_marginal_vw_{row,col}` (in `_pess_multisite_energy.py:96/191`), and/or `ctm_multisite` env construction on the kagome 3-site neighbour map.

## Time budget

2 hours on Approach A (surgical localise). Pivot to honeycomb-native CTM (M2b, PR #347 scaffolding) if exit conditions are not met.

## Approach C, two tracks in parallel

### Track A — surgical (GPU 0)

`test_pess_3site_multisite_rdm_invariants.py` baseline (just run): all 3 gates **PASS** at D=2 χ=16 SU-warmstart. The bug does not show up at the witness state — it only manifests at D=4 AD-optimum. Investigation must therefore use the smoking-gun probe state, not the SU-warmstart state.

1. **Failing-state diagnostic probe.** From the AD-optimised state at D=4 χ=16: compute all 6 multisite-CTM RDMs. For each, report Hermiticity error, λ_min, trace, and ‖ρ_CTM − ρ_BF‖_F where ρ_BF is from the 3×3 torus brute force. Sort by severity. The single worst offender is the localised suspect. Also: the `_rdm_3site_marginal_vw_{row,col}` helpers should agree with each other on a v-w bond when they consume the same envs ("marginalisation-consistency" gate from PR #398's design); if not, that points at a per-helper bug independent of envs.
2. **Investigate the worst helper.** Element-wise compare ρ_CTM against ρ_BF. Likely failure modes: (i) wrong axis when tracing out u in `marginal_vw_*`, (ii) wrong env corner in 2x1/1x2 helpers when neighbours are heterogeneous (u/v/w), (iii) leg-ordering bug surfacing only when bond dimension > 1 and state has nontrivial entanglement structure (D=1 passes because all envs are trivial).
3. **Patch + verify.** Apply minimal change. Re-run smoking-gun probe at D=4. Pass bar: per-site `|E_BF − E_CTM| < 0.05` at the new optimum.

### Track D — variPEPS cross-check (GPU 1)

variPEPS 1.4.2 is installed at `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps`. Has `expectation/three_sites.py::_three_site_triangle_workhorse` + `calc_three_sites_triangle_without_{top_left,top_right,bottom_left,bottom_right}_*` and `expectation/two_sites.py::_two_site_workhorse` for 2-site horizontal/vertical/diagonal — exact analogs of the broken Tenax helpers.

1. **Map state.** Build a `PEPS_Unit_Cell` from a kagome 3-site multisite SU-warmstart state at D=4. Assigning u/v/w supersites to a 3-cell unit cell on a square sub-lattice with the same neighbour map.
2. **Run variPEPS CTM** at χ=16.
3. **Extract RDMs** via variPEPS triangular helpers; convert to a per-bond energy and compare to BF (ground truth).
4. **Verdict.** If variPEPS ≈ BF, that confirms the bug is in Tenax helpers and gives a target answer to match. If variPEPS also disagrees with BF, the multisite encoding *itself* may be wrong (escalates to redesign).

## Exit conditions

- ✅ Done: smoking-gun probe shows |E_BF − E_CTM| < 0.05/site at AD-optimum after patch.
- ❌ 2-hour budget elapsed without clear localisation in A: pivot to honeycomb-native CTM (M2b, PR #347) — write up findings.
- ❌ A patch fails to close the gap: roll back, treat as if A never localised, pivot.

## Out of scope

- Refactoring helper architecture beyond the minimum needed for the patch.
- Performance work.
- D > 4 testing.
- AD-path retesting (already known correct via Wirtinger AD-vs-FD).
- License-dependent variPEPS code redistribution. Track D uses variPEPS as a runtime cross-check only; if a port is needed, license review precedes any code copy.
