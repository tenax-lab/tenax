# #670 — Symmetric 2×2 CTM carried-bond threading fix (design)

**Date:** 2026-07-01
**Issue:** [#670](https://github.com/tenax-lab/tenax/issues/670)
**Branch:** `fix/670-symmetric-2x2-carried-bond`
**Approach:** A — oracle-anchored minimal correction (approved 2026-07-01)

## Goal

Make the symmetric (block-sparse U(1)) **2×2** forward CTM sweep
(`ctm_tensor_2site(..., recipe="2x2")` / `_ctm_tensor_sweep_multisite`) run to
convergence on a genuine multi-charge U(1)-Sz environment without the
per-sector block-size mismatch it currently raises, matching the dense result.
On success, un-xfail
`tests/test_ctm_direction_dependent_bonds.py::test_symmetric_2site_ctm_matches_dense_on_direction_dependent_bonds`
(retargeting it to `recipe="2x2"`).

## Background — confirmed root cause

The symmetric 2×2 sweep crashes for **any** genuine multi-charge U(1) env (not
just direction-dependent `A.l != A.r`; reproduced on a direction-uniform
multi-charge pair). The crash is in the enlarged-corner build inside the
plaquette projector, e.g. `bottom_left`'s `C4.c4_u ↔ T4.t4_u` contraction, with a
per-sector count mismatch.

Established by investigation (`docs/superpowers/plans/2026-07-01-direction-dependent-symmetric-ctm.md`
Phase 2 + follow-up, and #670):

- After a `left` absorption `_ctm_tensor_absorb_left_2plaq` produces `new C4` with
  one **projector-compressed** leg (`c4_r`, via `P_top_curr`) and one
  **carried-over** leg (`c4_u`, a relabel of the old edge leg `T3.t3_l`).
- The next move's enlarged corner glues that carried leg to an edge leg whose
  charge structure differs → mismatch. Dense (trivial-charge) tolerates it
  because there is a single block of size χ.
- A localized swap of `bottom_left`'s C4 pairing was tried and **refuted**: it
  merely relocates the crash to the other C4 leg. So both the pairing *and* the
  carried leg's bond identity are involved.

### variPEPS oracle finding (the design anchor)

variPEPS 1.4.2 is importable in `.venv` (GPL — reference/test only). Its
`do_left_absorption` produces `new C4 = (carried_old_T3_chi, compressed_new_chi)`
— **the same one-carried/one-compressed structure Tenax already has**. It does
*not* crash because the carried leg is a **verbatim copy of an *unmodified* edge
tensor's leg**, and the env consumer glues it back to that *same unmodified
tensor* (the left sweep does not recompute `T3`; `replace_left_env_tensors` keeps
it). So the carried leg needs no per-sector reconciliation — it is bit-identical
to the tensor it later contracts.

**Conclusion:** Tenax's absorption *structure* is correct. The bug is that
Tenax's carried leg and the leg the enlarged corner glues it to are **not the
same bond** — a bookkeeping error (which cell's edge is carried, which
`t3_l`/`t3_r` end, and/or the enlarged-corner pairing), not a structural
redesign.

## Approach A — oracle-anchored minimal correction

Treat this as a bond-bookkeeping bug. Use variPEPS as a **numerical oracle** to
pin the exact divergence, then apply the minimal correction, and verify Tenax's
whole sweep reproduces the correct env bond structure.

### Components

1. **variPEPS oracle harness (test-only, `scripts/` or `tests/` helper).**
   A thin wrapper that builds a 2-site unit cell in variPEPS from the *same*
   site tensors Tenax uses (or an equivalent random D=2 pair), runs variPEPS
   dense CTMRG to convergence, and exposes: (a) the converged corner/edge tensor
   bond dimensions and axis meaning, and (b) — where feasible — the intermediate
   post-`left`-absorption `C4`/`T4`/`T3` bond structure. variPEPS is GPL and is
   **never imported by `src/`**; it lives only in test/diagnostic code, guarded
   by an import skip if unavailable (so CI without variPEPS still passes the rest
   of the suite).

2. **Diagnostic (red) step — pin the exact divergent bond.**
   With Tenax and variPEPS both run on the same/equivalent pair, numerically diff
   the post-`left`-move `C4` (and `T4`, `T3`) bond charge structure to identify
   the precise leg where Tenax diverges from the reference: is the carried leg
   sourced from the wrong cell's `T3`, the wrong (`t3_l`/`t3_r`) end, or does the
   enlarged corner glue the wrong pair? This converts the current guesswork into
   a concrete, quoted target. Output: a written note of the exact divergence.

3. **Fix — minimal carried-leg/pairing correction.**
   In `_ctm_tensor_absorb_*_2plaq` and/or `_build_enlarged_corner`, correct the
   carried leg's source/relabel (and, if needed, the enlarged-corner pairing) so
   the carried leg is the verbatim unmodified edge leg the enlarged corner glues
   to — mirroring variPEPS's convention. The fix must be symmetric across all
   four directions (left/right/top/bottom) and internally consistent with the
   env-consumption convention already used by the energy/RDM path
   (`C4.c4_u↔T3.t3_r`, `C4.c4_r↔T4.t4_u`, etc.).

4. **Scope guard.** The non-fused (dense + bosonic-U(1)) absorption path only.
   The fermionic `_ctm_tensor_absorb_*_2plaq_fused` path is **out of scope**
   (separate follow-up if needed) — but if the same bookkeeping bug exists there,
   note it in a follow-up issue.

### Data flow (unchanged shape; corrected labels)

`initialize_ctm_tensor_env` → `_ctm_tensor_sweep_multisite(recipe="2x2")`
→ per direction: `_compute_plaquette_projector_pair` (builds 4 enlarged corners
→ `_compute_2x2_projector`) → `_ctm_tensor_absorb_*_2plaq` (applies projector
halves, carries the unmodified edge leg). The fix changes only *which leg is
carried / how it's paired*, not the contraction topology or the projector math.

## Testing strategy

- **Unit (red→green):** a test that builds the post-`left`-move env on a
  multi-charge (direction-uniform first, then direction-dependent) pair and
  asserts the enlarged corners for the *next* direction all build without a block
  mismatch (extends the existing Phase 1
  `test_2x2_enlarged_corners_build_on_direction_dependent_init` to post-absorption
  envs).
- **Oracle parity (where variPEPS available):** assert Tenax's converged 2×2
  symmetric env bond structure matches variPEPS's (dimensions/charges), skipped
  when variPEPS is absent.
- **Acceptance:** un-xfail + retarget
  `test_symmetric_2site_ctm_matches_dense_on_direction_dependent_bonds` to
  `recipe="2x2"`; require `abs(E_sym - E_dense) < 1e-6`, `C1 norm > 1e-8`.
- **Regression / no-op for the working paths:**
  - Dense parity: the direction-dependent *dense* 2×2 energy stays −0.542116.
  - `uv run pytest -m core -q` plus `tests/test_ctm_tensor.py`,
    `tests/test_ctm_tensor_projector_2x2.py`, `tests/test_ipeps_u1sz.py` green.
  - The 2×2 projector closure test (`P_bot · P_top = I`) unchanged.

## Acceptance criteria

1. `ctm_tensor_2site(recipe="2x2")` completes on a direction-uniform multi-charge
   U(1) env (the general #670 repro) — no block mismatch.
2. `abs(E_sym − E_dense) < 1e-6` on the direction-dependent pair with
   `recipe="2x2"`; corner norm > 1e-8.
3. The #667 acceptance test is **un-xfailed** and retargeted to `recipe="2x2"`
   and passes.
4. Dense and direction-uniform-dense results and all existing core/CTM/projector
   tests are unchanged (fix is a no-op for the dense path).
5. #670 closed; fermionic path scoped to a follow-up if the same bug exists there.

## Non-goals

- Fermionic (`FermionParity`/`FermionicU1`) 2×2 absorption (separate follow-up).
- The `1x1` recipe (has a distinct fundamental edge-orientation wall; the #667
  test moves to `2x2`).
- Any projector-math change (`_compute_2x2_projector` SVD/closure stays as is).
- Copying variPEPS code — it is a read-only GPL reference/oracle only.

## Risks

- **The divergence may span more than the carried leg** (e.g. multi-site storage
  target). Mitigation: the diagnostic step pins it before coding; if it turns out
  structural after all, we stop and re-scope with evidence rather than guessing
  (as we did when the localized swap was refuted).
- **variPEPS index-convention mapping** to Tenax is nontrivial; the oracle is a
  guide, and the primary acceptance is dense-parity + block-consistency, which
  hold independent of variPEPS.
