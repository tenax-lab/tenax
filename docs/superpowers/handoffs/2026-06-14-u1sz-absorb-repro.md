# U(1)-Sz symmetric CTM: charged-block collapse (the #570 coverage gap)

**Date:** 2026-06-14
**Branch:** `study/u1sz-heisenberg-enablement`
**Status:** confirmed bug, fix pending (Task 5 of the enablement plan)
**Repro test:** `tests/test_ipeps_u1sz.py::TestU1SzSymmetricMatchesDense::test_one_step_symmetric_matches_dense` (currently `xfail`)

## Symptom

Running 2-site `optimize_gs_ad` with **non-trivially-charged** U(1)-Sz `SymmetricTensor`
site tensors (`heisenberg_u1sz_init_pair`) and comparing against a dense run from the *same*
densified init:

- `E_sym = 0.0`
- `E_dense = −0.4945439…`  (the dense run from the densified init; lopsided because the D=2
  `[0,+1]` virtual scheme is variationally restricted — not the bug, just the init)

The symmetric path **runs without raising** but returns energy 0 — it is a silent
correctness failure, not a crash. This is the documented coverage gap
(`examples/bench_symmetric_ad_batching_566.py:57`: "U(1) single-site CTM path with
non-trivial charges currently fails in the production absorb step") manifesting as
charge-sector collapse.

## Mechanism (from the Task-3 investigation)

The CTM environment's non-trivial charge blocks zero out after the first sweep:

```
Iter 0 (after sweep 1):  C1 block (-1, 1) norm = 0.4965   ← correct, charged sector present
Iter 1 (after sweep 2):  C1 block (-1, 1) norm = 0.0000   ← charged blocks collapse
Iter 2 (after sweep 3):  C1 block  (0, 0) norm = 0.0000   ← (0,0) then dies against zeroed edges
```

The `(0,0)` block survives sweep 2 but zeros on sweep 3 because it contracts against the
already-zero `(±1,∓1)` blocks of the edge tensors. Once all blocks are zero,
`compute_energy_ctm_tensor_2site` returns 0.

## Candidate root cause (hypothesis — verify before fixing)

The symmetric CTM **edge/corner initialisation** seeds the chi legs with only the trivial
`(0,0)` charge sector, while the D² (= u⊗ū) fused legs carry non-trivial charges. The first
sweep's projectors populate the non-trivial chi blocks of the corners, but the edge tensors are
re-absorbed from the **old** trivial-chi edges crossed with the double layer, so on the next
sweep the `T4·a_src` contraction has a charge mismatch on the chi/D² legs and produces zeros.

Frames implicated by the trace:
1. `_ctm_python_loop.py:179` `_run_ctm_loop_with_bump` → `jit_step(...)`
2. `_ctm_tensor_convergence.py:342` `_ctm_tensor_sweep_multisite` (recipe="2x2", Phase 2 absorb)
3. `_ctm_tensor_moves.py:372-441` `_ctm_tensor_absorb_left_2plaq` (+ right/top/bottom analogs):
   `contract(env_src.T4, a_src)` — `a_src.u2` carries `[-1,0,0,1]` but `T4.l2` (chi side) is
   trivial-charge `[0]` only.
4. `_ctm_tensor_init.py:252` `_init_symmetric_standard_edge` /
   `_init_symmetric_standard_corner` — seed only trivial chi charges.

## Files to focus the fix on

- `src/tenax/algorithms/_ctm_tensor_init.py` — `_init_symmetric_standard_edge`,
  `_init_symmetric_standard_corner` (may need charge-aware chi seeding).
- `src/tenax/algorithms/_ctm_tensor_moves.py:372-441` — the four `_ctm_tensor_absorb_*_2plaq`
  charge flow in the `T·a` contraction.
- `src/tenax/algorithms/_ctm_tensor_convergence.py:412-434` — Phase 2 absorb dispatch.

## Note

The above mechanism is one agent's trace and is plausible but **not independently confirmed**.
Task 5 must reproduce minimally and verify the root cause before patching, and should localize
whether the bug is specific to unbounded U(1) charges or general to any non-trivial charge
(re-run with a bounded `Zn`/capped-charge proxy).
