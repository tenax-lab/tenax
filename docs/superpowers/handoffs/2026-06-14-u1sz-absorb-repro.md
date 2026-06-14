# U(1)-Sz symmetric CTM: charged-block collapse (the #570 coverage gap)

**Date:** 2026-06-14
**Branch:** `study/u1sz-heisenberg-enablement`
**Status:** ✅ **RESOLVED 2026-06-14** — fixed by a ~20-line init canonicalization (see "RESOLVED" section immediately below). #602's "no focused fix exists" verdict was wrong.

---

## RESOLVED (final, 2026-06-14): init↔SVD bond-order mismatch at sweep 1

**Pinned root cause.** The symmetric CTM env init
(`_init_symmetric_standard_corner` / `_init_symmetric_standard_edge`,
`algorithms/_ctm_tensor_init.py`) emits chi-bond charges in **enumeration order**
(D²-fused-charge order, e.g. `[0,-1,1,0,...]`), while the block-sparse SVD that grows the env under
jit/AD (`_truncated_svd_symmetric_traced`, the production path) emits the bond in **charge-grouped**
order (`sorted(sectors)`). For unbounded U(1) these differ. The CTM absorb pairs fused legs
**positionally** (no realignment) and a fused leg's intra-block layout follows its chi sub-leg order
(`_compute_fused_charges`), so the **sweep-1** absorb pairs the projector's grouped fused leg
against the env's enumeration-ordered fused leg by position → charged sectors cancel → `E_sym=0`.

**Decisive facts (this session's experiments):**
- Eager vs traced SVD differ **only inter-sector**; per-sector they are **bit-identical** (so the
  3rd investigation's "intra-sector pairing diverges" framing was wrong).
- Forcing the **init** into charge-grouped order recovers `E_sym=0 → −0.4998` (== the eager value).
- The symmetric contraction is **order-INVARIANT** when all bonds agree: the converged energy is
  unchanged to **6e-14** under a faithful virtual-bond permutation. So charge-grouped order is fine
  *when consistent*; the bug is the **init↔SVD mismatch**, not grouped-order itself. (The 2nd
  investigation's "forcing grouped broke eager" was *incomplete* forcing — it changed the SVD but
  not the init/env, leaving them inconsistent.)
- **Fermions unaffected** because bounded FermionParity charges make enumeration and grouped order
  coincide → the canonicalization is a no-op there.

**The fix (shipped).** Canonicalize the env-init chi bonds into the same charge-grouped order the
traced SVD uses, via a faithful (data+index) reorder in `_init_symmetric_standard_corner` /
`_init_symmetric_standard_edge` (helper `_grouped_chi_perm`). The shared eager
`_truncated_svd_symmetric` is **left untouched** — globally charge-sorting it breaks the
`base_charges` position-order contract that simple-update relies on (#560), so the fix lives in the
CTM init only.

**Scope / limitation.** Production iPEPS-AD jits → traced SVD → grouped → now consistent with the
grouped init → fixed. The **non-jit eager** symmetric CTM for unbounded U(1) remains inconsistent
(grouped init + interleaved eager SVD) — but that path was *never* consistent (it gave a wrong-ish
nonzero value, not a correct one) and is not a production path.

**Why `E_sym ≠ E_dense` exactly (and the GO-gate reframe).** The symmetric path enforces charge
conservation (block-diagonal, per-sector truncation); the dense path truncates globally without
charge structure; and the CTM is not fully converged for this variationally-restricted D=2/χ=8 init.
So the original test's `atol=1e-8` "matches dense" premise was unachievable *independent of #602*.
The GO-gate is reframed (`test_one_step_symmetric_charged_ctm_no_collapse`) to guard the real
property: charged sectors survive (`E_sym` finite and clearly negative) and agree with dense to a
physical tolerance (`2e-2`).

**Verification:** GO-gate passes; bosonic CTM 65 pass; fermionic `test_fpeps_ad.py::TestTodense…`
3 pass; `base_charges` contract intact; order-invariance 6e-14.

---
**Repro test:** `tests/test_ipeps_u1sz.py::TestU1SzSymmetricMatchesDense::test_one_step_symmetric_matches_dense` (currently `xfail`)

---

## CORRECTED ROOT CAUSE — 4th investigation (2026-06-14): a focused, jit-safe fix DOES exist

The three earlier root-cause sections below progressively narrowed the bug but the
"FINAL/DEFINITIVE" conclusion (**"no focused fix exists"**) is **wrong**, and it was wrong for a
specific, checkable reason. A 4th investigation reproduced #602, then verified every link of the
mechanism in code + a controlled SVD experiment.

**Decisive new evidence (controlled eager-vs-traced SVD on a charged U(1) tensor):**
the two SVD paths differ ONLY in *inter-sector* bond arrangement — **per-sector the SVDs are
bit-identical**:

```
EAGER  bond charge order: [1, 0, -1, 1, -1, 0, 0]   (SV-interleaved, magnitude-dependent)
TRACED bond charge order: [-1, -1, 0, 0, 0, 1, 1]   (charge-grouped, deterministic)
per-charge singular values: eager == traced (exact)   reconstruct err: 8e-15 both
```

So the 3rd investigation's framing ("the intra-sector basis *pairing* across the two fused legs
*diverges*") mislocates the cause. The SVD does not diverge intra-sector. The **inter-sector order
of the chi bond differs**, and that becomes *intra-block layout* only by passing through **fusion**.

**The verified mechanism (all code-checked):**
1. The contractor sums a contracted axis **positionally** over the full block array
   (`opt_einsum.contract_expression`), with **no intra-block realignment** — `contractor.py:850-863`.
2. Fusion bakes the chi sub-leg's enumeration order into each fused block's internal row layout:
   `fused[i*db + j]` — `_tensor_utils.py:177-189`.
3. Eager SVD emits the chi bond SV-interleaved; traced (under jit/AD) emits charge-grouped
   (evidence above). Per-sector identical.
4. `_reembed_fused` — the *only* fused-leg alignment step — reconciles per-charge **dimension only**;
   its `tgt_dim == cur_dim` branch copies the block **verbatim**, never reordering —
   `_ctm_projector.py:243-244`.
5. ⇒ Under jit/AD the projector's fused leg (charge-grouped) and the env corner's independently-fused
   leg are paired positionally by the absorb `contract(P_top_curr_bar, C3g)`
   (`_ctm_tensor_moves.py:640`) → mismatched basis rows → charged sectors cancel → E=0.

**Why "no focused fix exists" was a strawman.** It rejected (a) "make traced SV-descending — not
jit-safe (SVs are traced)" and (b) "realign by basis identity — impossible, indistinguishable by
charge." Neither is the actual requirement. We need **one canonical chi order shared by every
bond-creating site**; canonical = **charge-sorted**, which is **static aux_data ⇒ jit-safe**
(`_build_unified_fused_idx` already charge-sorts, `_ctm_projector.py:131`). The 2nd investigation's
"forcing charge-grouped order broke eager too" was **incomplete forcing** (it changed the SVD output
but not all carried env chi legs / didn't split truncation *selection* from bond *storage order*),
not proof of impossibility.

**The fix (Fix A, jit-safe, contractor/fusion/absorbs untouched):** decouple truncation *selection*
from bond *storage order* and make EVERY chi-bond creator emit the same canonical charge-sorted
order — `_truncated_svd_symmetric` (eager), `_truncated_svd_symmetric_traced`, the eigh/SVD
projectors in `_ctm_projector.py`, and the init env chi legs. Regression gate: full
`tests/test_fpeps_ad.py` (FermionParity — must be a verified no-op there, since its 2 bounded
charges already make the orderings coincide) + bosonic/dense CTM + the U(1) GO-gate flipping
`xfail`→pass.

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

## Root cause — FINAL / DEFINITIVE (2026-06-14, third investigation; pins the WHY)

The bond-order finding below is correct; this pins **why** order matters and why no focused fix exists.

**The block-sparse contractor pairs contracted legs position-by-position within each charge
sector and never realigns intra-sector basis ordering.** (`_contract_symmetric`,
`src/tenax/contraction/contractor.py:599`; `_reembed_fused`, `_ctm_projector.py:213`, only
pads/truncates per-sector *dimension*, not order.) A fused leg's intra-sector layout is derived
from the chi sub-leg's charge order via `_compute_fused_charges` (`_tensor_utils.py:155`).

- Eager SVD (`linalg.py:170`) emits chi in **SV-descending** order → the two independently-built
  fused legs feeding the absorb contraction stay mutually consistent → the per-sector `einsum`
  pairs the correct basis states → correct.
- Traced SVD (`linalg.py:583`, used under jit/AD) emits **sector-block** order → the intra-sector
  basis pairing across the two fused legs diverges → the dominant charge-0 content gets misrouted
  and cancels → charged sectors collapse → E=0.

**Decisive single-contraction isolation:** in `_ctm_tensor_absorb_bottom_2plaq`
(`_ctm_tensor_moves.py:639-641`), with *identical* inputs (`env_src.C3 per c3_u = {0:1.0}`,
`C3g = {0:0.103}`) the only difference is the projector's chi order:
`contract(P_top_curr_bar, C3g)` gives charge-0 weight 0.069 (eager, SV-descending) vs **0.0**
(traced, sector-block). The zeroed C3 then propagates: sweep-2 `Q_BR` loses its `chi_L=-1` sector
→ `M2 = Q_BR·Q_BL` charged blocks → 0 → energy 0.

**Why no focused fix exists (all three candidates rejected with evidence):**
1. Make traced chi_new SV-descending → requires sorting by singular-value magnitude, which are
   **traced values under jit** — not statically sortable, not jit-safe.
2. "Root" projector fix → the projector is already internally consistent; the inconsistency is in
   the *downstream contraction's positional intra-sector pairing*, not in
   `_compute_2x2_projector_symmetric`. No local index/data swap there fixes it.
3. Core contractor/fuse change to realign contracted legs by intra-sector basis identity → within
   a sector, basis states are indistinguishable by charge alone, so realignment needs a consistent
   canonical intra-sector ordering threaded through `fuse_indices` + projector + all four absorb
   directions: a cross-cutting change to core block-sparse code **shared with the working
   fermionic fPEPS path** — high blast radius.

**Why fermionic (FermionParity = Z2) is unaffected:** only 2 bounded charge values {0,1}, small
stable per-sector multiplicities → sector-block and SV-descending orderings coincide (or are
positionally degenerate), so the intra-sector pairing never diverges. Unbounded U(1) with
{−1,0,+1} and larger multiplicities is where they differ.

**Verdict:** this is a **core block-sparse-contraction correctness item** (consistent intra-sector
basis ordering for fused legs across SVD → projector → absorb), to be scoped and reviewed
separately with the full fermionic suite as a regression gate — NOT a CTM-local or study-local
patch. The U(1)-Sz Heisenberg perf study is **blocked** on it. Files for the eventual fix:
`contraction/contractor.py:599`, `_tensor_utils.py:155/230`, `_ctm_tensor_moves.py` absorbs,
`linalg.py:170/583`. Caveat for any fix: per-sector *dimensions* (not just order) drift between
sweeps for unbounded U(1), so the canonical ordering must stay stable as multiplicities change.

---

## Root cause — REVISED (2026-06-14, second investigation; bond-order, jit-independent)

A second, independent fix attempt ran a controlled experiment that **refutes the jit-retrace
mechanism** below and is more decisive:

- **The jit step does NOT retrace every sweep.** `jax.tree_util.tree_structure` shows the env
  treedef is *stable* from sweep 1 onward (one unavoidable init→steady retrace, then stable). The
  "static aux_data treedef changes every sweep" claim is wrong.
- **The bug is bond-ORDER dependence in the block-sparse 2×2 projector, independent of jit.**
  Forcing the SVD bond into **sector-block (charge-grouped)** order collapses the charged sectors
  to zero **even in pure eager mode (no jit)** → E=0. Leaving it in **SV-descending** order
  (eager's default, `_truncated_svd_symmetric`, `linalg.py:170`) works → E≈-0.05. The jit path
  takes the *traced* SVD (`_truncated_svd_symmetric_traced`, `linalg.py:583`, whose own comment at
  `:616` says "bond axis emerges in sector-block order"), so it inherits the buggy order.
- Consequence: the "canonical chi-order" fix (sort into sector-block order) is **actively wrong** —
  it forces the buggy ordering and collapsed even the previously-correct eager path. Verified the
  relabelling was faithful (`U·diag(S)·Vh == M` to 1e-16), so this is a genuine projector
  order-dependence bug, not a botched permutation.

**The real fix lead:** either make `_truncated_svd_symmetric_traced` emit the **same SV-descending
bond order as the eager SVD**, or fix `_compute_2x2_projector_symmetric`
(`_ctm_tensor_projector_2x2.py:925–1070`, Stages 4–6) to be correct regardless of bond order. This
is a **core block-sparse linalg/projector correctness bug** shared with the symmetric CTM path.
Residual open question: why the fermionic FermionParity (Z2) path is unaffected — likely its
sector-block order coincides with (or is order-insensitive for) the few-charge bounded case;
needs confirming as part of the fix.

**Confidence:** high on the controlled eager-order experiment; the exact projector wiring bug is
localized to the two files above but not yet pinned to a single line.

---

## Root cause — (first investigation; jit-retrace theory, now DISPUTED by the above)

**The initial absorb-contraction hypothesis was REFUTED.** Independent diagnosis (eager-vs-jit
isolation + retrace counter + bounded-symmetry cross-check) found the real mechanism: a
**JIT + drifting static charge-layout interaction**, specific to **unbounded U(1)**.

- The CTM step is `jax.jit`-wrapped (`_make_jit_ctm_step`, `_ctm_python_loop.py:59`), called once
  per sweep in a Python loop by the implicit-AD forward.
- `SymmetricTensor.tree_flatten` (`core/tensor.py:795`) puts `_block_keys/_block_shapes/
  _block_offsets/_indices` (the per-leg **charge vectors**) into pytree **aux_data** — static
  under jit. Only the flat `_data` buffer is a traced leaf.
- For unbounded U(1), the env chi-leg charge **ordering** is recomputed from the projector SVD
  each sweep and **drifts**:
  - init C1 chi charges: `[0,-1,1,0, 0,-1,1,0]`
  - after sweep 1:        `[0,-1,1,0, -1,0,0,1]`  (reordered)
  - after sweep 2:        `[0,-1,1,0, 0,-1,1,0]`  (reordered again)
- Each new layout is a different treedef → **jit re-traces every sweep**, and the recompiled
  block routing no longer matches the carried `_data` buffer, so the charged sectors land in
  mismatched/empty offsets and zero out.

**Evidence:** eager `_ctm_tensor_sweep_multisite` (no jit) keeps the charged `(-1,1)` block alive
(~0.43–0.46) and gives the correct **E=-0.327**; the jit'd step retraces on sweeps 1→2 and the
`(-1,1)` block goes 0.359 → 0.0 exactly when it retraces. **Bounded** symmetries do NOT drift:
Z2/Z3 (and the working fermionic **FermionParity** path, `tests/test_fpeps_ad.py`) keep charged
blocks alive across sweeps and give finite energy. So the absorb/projector/init code is correct;
only unbounded-U(1) chi-layout drift breaks it. **Confidence: high.**

## Why fermionic (FermionParity = Z2) works but U(1) doesn't

Bounded groups have a small fixed set of charge values, so the per-sector projector SVD returns a
**stable** chi-leg layout sweep-to-sweep → jit traces once, no buffer mismatch. Unbounded U(1)
charge sets reorder, triggering the retrace/mismatch.

## Fix options (design)

1. **Canonicalize chi-index charge ordering after each projector build** (recommended, localized,
   ~30–60 lines in `_ctm_tensor_moves.py` `_half_to_chi_new_*` / `_compute_2x2_projector`): sort
   new chi charges into a fixed canonical order and permute the projector data to match, so the
   post-sweep env layout is deterministic → jit traces once. Symmetric path only; dense untouched.
2. **Pre-declare a fixed padded chi-leg charge basis at init** (moderate, most robust): projectors
   always emit a fixed-length/fixed-order charge vector (zero-size blocks for empty sectors).
   Safer if a sector's *dimension* (not just order) drifts at larger D/chi.
3. **(Fallback) run the U(1) sweep eagerly** (skip `_make_jit_ctm_step` for non-trivial unbounded
   charges): trivially correct but loses jit performance and implicit-AD through the jit'd step.

Residual uncertainty: whether canonical sort alone (option 1) suffices when a sector's dimension
changes between sweeps at larger D/chi — option 2 is the safe fallback there.

---

## (Superseded) initial hypothesis — kept for the record

The first trace guessed an absorb-contraction charge mismatch in `_ctm_tensor_absorb_left_2plaq`
with trivial-chi edge seeding. This was **refuted** by the eager-vs-jit isolation above: eager
runs the same absorb code correctly. Do not pursue the absorb/init seeding theory.
