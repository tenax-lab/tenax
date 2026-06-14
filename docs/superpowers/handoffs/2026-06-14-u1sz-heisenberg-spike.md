# U(1)-Sz Heisenberg feasibility spike — VERDICT: NO-GO (blocked by a core block-sparse bug)

**Date:** 2026-06-14
**Issue:** #570 (follow-up — the lever named by the dense large-D study)
**Branch:** `study/u1sz-heisenberg-enablement`
**Spec/plan:** `docs/superpowers/{specs,plans}/2026-06-14-u1sz-heisenberg-enablement*`
**Root-cause record:** `docs/superpowers/handoffs/2026-06-14-u1sz-absorb-repro.md`

## TL;DR

The feasibility spike did exactly its job: it **gated the study before any perf work** and
surfaced — then precisely characterized — a genuine **core block-sparse correctness bug** that
blocks unbounded-U(1) symmetric CTM. **Verdict: NO-GO** for the U(1)-Sz perf study until the core
bug is fixed.

The bug is **not** in anything this study built. The two new helpers
(`heisenberg_gate_u1sz`, `heisenberg_u1sz_init_pair`) are correct, tested, and land as working
production API. The blocker is upstream: the block-sparse contractor pairs contracted legs
**position-by-position within each charge sector** and never realigns intra-sector basis order,
while the **traced** symmetric SVD (used under jit/AD) emits its bond in **sector-block** order
vs the eager SVD's **SV-descending** order. For unbounded U(1) the two orderings diverge, the
intra-sector pairing breaks, charged CTM sectors cancel to zero, and the energy collapses to 0.

## What was built (lands regardless of the NO-GO)

- `heisenberg_gate_u1sz()` — U(1)-Sz charged Heisenberg gate (`SymmetricTensor`, physical charges
  `[+1,−1]`). Dense values bit-identical to `heisenberg_gate()`; 6 charge blocks; Sz-conserving.
  Production API (README + `__all__`), 5 unit tests.
- `heisenberg_u1sz_init_pair(D, key)` — random U(1)-Sz-symmetric 2-site iPEPS pair. Non-trivially
  blocked (8 blocks at D=2); virtual charges `[0,+1,−1,…]` (a `[+1,−1]`-only scheme is provably
  empty — parity obstruction). Production API, 5 unit tests incl. per-block Sz conservation.
- `tests/test_ipeps_u1sz.py::...::test_one_step_symmetric_matches_dense` — the GO-gate, committed
  as a **strict xfail** that doubles as the executable repro of the core bug.

## The GO/NO-GO gate (3 checks)

| Check | Result |
|-------|--------|
| **1. Runs** end-to-end through CTM-AD | ✅ runs (no crash) |
| **2a. Contraction correct** (symmetric == dense) | ❌ `E_sym = 0` vs `E_dense ≈ −0.49` |
| **2b. Right energy** (≈ −0.66 / matches dense) | ❌ (consequence of 2a) |
| **3. Perf / block signal** | ⛔ not reached (path is incorrect) |

Check 2 fails → **NO-GO**. The symmetric path executes but silently returns the wrong answer.

## Root cause (definitive — full detail in the repro doc)

`_contract_symmetric` (`contraction/contractor.py:599`) sums contracted legs position-by-position
within each charge sector; it never realigns intra-sector basis order. A fused leg's intra-sector
layout follows its chi sub-leg's charge order (`_compute_fused_charges`, `_tensor_utils.py:155`).

- Eager SVD (`linalg.py:170`) → **SV-descending** chi order → fused legs feeding the absorb
  contraction stay consistent → correct (E ≈ −0.05 eager).
- Traced SVD (`linalg.py:583`, taken under jit/AD) → **sector-block** chi order → intra-sector
  pairing across the two fused legs diverges → charge-0 weight misroutes/cancels → charged sectors
  collapse → **E = 0**.

Isolated to a single contraction in `_ctm_tensor_absorb_bottom_2plaq` (`_ctm_tensor_moves.py:639`):
identical inputs, only chi order differs, charge-0 output is 0.069 (eager) vs 0.0 (traced).

**Fermionic (FermionParity = Z2) is unaffected** — 2 bounded charges, the two orderings coincide;
`tests/test_fpeps_ad.py` passes. The bug is specific to richer (unbounded-U(1)) charge sets.

**No focused fix exists** (all three candidates rejected with evidence): SV-descending-under-jit
isn't jit-safe (SVs are traced); the projector is already consistent (bug is downstream in the
contraction); a contractor-level intra-sector realignment is a cross-cutting change to core code
shared with the working fermionic path. See the repro doc for the full rejection evidence.

## Recommendation

1. **File a core tenax issue** (done: **#602**) — "block-sparse contraction assumes consistent
   intra-sector basis order; traced SVD's sector-block bond order breaks unbounded-U(1) symmetric
   CTM." Scope it as a core-linalg correctness fix with the fermionic suite as a regression gate.
   The fix must thread a canonical intra-sector ordering through SVD → projector → all four
   absorbs, and stay stable as per-sector multiplicities drift between sweeps.
2. **Resume the U(1)-Sz Heisenberg perf study only after that lands** — at which point the GO-gate
   xfail flips to pass and the (D,χ) symmetric-vs-dense scaling sweep (the original next study)
   becomes meaningful. The dense path remains runtime-bound (`[[570-dense-largeD-study]]`); whether
   U(1) block-sparsity moves that wall is still the open question — now gated on the core fix.

## Honest framing

This is the same shape of outcome as the #570 Phase-3 NO-GO: a feasibility spike that paid for
itself by killing a path *before* a large build, with a precise, evidence-backed root cause and a
concrete pointer to what would unblock it. The two charged-tensor helpers are a net-positive
byproduct that ship regardless.
