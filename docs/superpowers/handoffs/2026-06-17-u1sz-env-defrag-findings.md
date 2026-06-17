# U(1)-Sz CTM env de-fragmentation spike — Findings (#610)

**Date:** 2026-06-17
**Branch:** `spike/610-u1sz-env-defrag` (off `origin/main` @ 3b51689)
**Spec:** `docs/superpowers/specs/2026-06-16-u1sz-env-defrag-design.md`
**Plan:** `docs/superpowers/plans/2026-06-17-u1sz-env-defrag-spike.md`
**Verdict:** **NO-GO-by-obstruction.** The make-or-break Gate-B measurement is **structurally
unattainable via runtime monkeypatch**; confirming the lever requires the structural CTM change the
spike was meant to gate. No committed `src/` change (per scope).

## Question

Does reducing the distinct charge-sector count on the U(1)-Sz CTM **environment** tensors (the
sector-dropping "C-lever": keep only `|Sz| ≤ 1` on the chi bonds) shrink the backward **charge-mask
op cluster** (#609's ~35%), and buy D=3 χ=12 iPEPS-AD runtime to ≥ parity-with-dense within 1%
energy?

## Answer: the forward drop is real, but it cannot reach the backward that matters — without a structural change.

Three-gate staircase, cheapest-decisive-first. Gate A passed; Gate B is **blocked**, not failed.

### Gate A — static sector census (PASS)

Baseline (D=3 χ=12, `census_u1sz_baseline_610.json`): env block counts **C1–C4 = 5, T1–T4 = 19**
(median collapse ceiling 1.90× — heavy fragmentation, vs ~16× even-D). `E_frag = −0.0617` (a
**random, un-optimized** `heisenberg_u1sz_init_pair` site — a same-input baseline, *not* a ground
state; see Risks).

Static candidate-C prediction (`predict_sectors_under_keep`, dropping `|Sz|>1` on the chi bonds,
`census_u1sz_candidateC_610.json`):

| tensor | n_blocks | kept | reduction |
|---|---|---|---|
| C1–C4 | 5 | 3 | 1.667× |
| T1–T4 | 19 | 9 | 2.111× |

**Median = 2.111× ≥ 2×** → Gate A PASS. Total env blocks 96 → 48 (exactly 2×). The forward
prototype reproduces this exactly: under `sector_dropping_truncation()` the converged `ctm_tensor`
env has block counts `{C: 3, T: 9}`, energy finite/sane — the representation change **is
constructible and faithful in the forward path** (`examples/u1sz_defrag_prototype_610.py`,
`tests/test_u1sz_defrag_prototype_610.py`).

### Gate B — backward charge-mask re-profile (BLOCKED by structural obstruction)

The forward drop bites the **paired-moves** projector path (`ctm_tensor`), but the **backward** the
spike must measure — `_make_jit_ctm_step` → `_ctm_tensor_sweep_multisite`, the **2×2 plaquette**
path used by both the Gate-B probe **and** production `_ctm_energy_ad` — does **not** honor the drop.
We attacked it with three coordinated monkeypatches (no `src/` edits):

1. filter `_ctm_tensor_paired_moves._get_base_charges` (forward), and
2. filter the **separate** `_ctm_tensor_convergence._get_base_charges` (the multisite/backward
   base_charges source — a distinct function; this was the first reason the drop never reached the
   backward), and
3. restrict the Phase-2 χ-backfill in `tenax.linalg._truncated_svd_symmetric_traced`
   (`src/tenax/linalg.py:732–747`) to `keep` sectors, so it stops re-adding ±2 to saturate χ
   (`chi_new` honestly shrinks to ~10 at D=3 χ=12).

Even with all three, the **cold** backward VJP trace raises (independently reproduced):

```
ValueError: Size of label 'd' for operand 1 (4) does not match previous terms (5).
  _build_enlarged_corner -> contract(C_r, T_v)   (_ctm_tensor_projector_2x2.py:238)
```

**Root cause (structural):** `initialize_ctm_tensor_env` seeds every chi bond at the full 5-sector
`{−2..+2}`, and the 2×2 multisite sweep refreshes only the legs along the *current* absorption
direction per move. So a single sweep that transitions 5→3 sectors mid-sweep contracts a
freshly-dropped 3-sector leg (`t4_u` ∈ `{−1,0,1}`) against an un-refreshed 5-sector perpendicular
corner leg (`c4_u` ∈ `{−2..+2}`) — a **mixed-generation chi-bond mismatch**. This is *not* the
earlier post-hoc-bond-drop trap; the truncation is self-consistent per move, but the sweep contracts
tensors from different truncation generations. A monkeypatch cannot fix it: making the backward
honor the drop needs **env-init to seed a uniform sector set** *and* the 2×2 sweep to refresh **all**
chi bonds to that uniform set per sweep — a structural change to the most-used symmetric CTM paths.

Because the defrag jaxpr never builds, the charge-mask reduction is **undefined** — there is no
measured drop to assess against the ≥25% bar. Baseline backward (built fine): **63,612** total ops,
**7,398** charge-mask/index-arith ops (`probe_u1sz_baseline_610.txt`).

**Cold-trace caveat (load-bearing for any future re-test):** the monkeypatch only bites if the
patched `_make_jit_ctm_step` unit is traced **cold** under the patch. A prior un-patched trace of
the same jit unit (identical avals) seeds the `jax.jit` cache, so a later prototype-wrapped call
silently reuses the un-patched trace — neither erroring nor dropping. Any "it traced fine" claim must
verify cold tracing (`jax.clear_caches()`), or it is measuring the baseline.

### Gate C — not reached.

## The honest signal (inference, not measurement)

The forward env-block count drops **exactly 2×**, and #609 **measured** the backward op count to be
~linear in block count (~1.1 ops/block; D=2→D=3 was 19,319→63,612). The baseline here (63,612 ops,
7,398 charge-mask) is consistent with that scaling. So a *faithful* sector-dropped backward would
**plausibly** cut the charge-mask cluster ~proportionally (~40–50%, clearing the 25% bar). But this
is **inference**, exactly what the brief warned against relying on — and the structural obstruction
is precisely what prevents converting it to a measurement cheaply.

## Recommendation / next lever

**Do not fund the structural build on inference within a measure-first spike** (cheapest-kill
discipline, mirroring #609). The de-fragmentation idea is *not* refuted — it is **un-measurable
without building most of the real thing**. Recommended scoped follow-up (filed as a separate issue):

> Implement a **uniform-sector env representation** for the multisite 2×2 CTM: seed
> `initialize_ctm_tensor_env` chi bonds with the chosen `keep` sector set, and make the 2×2 sweep
> refresh **all** chi bonds to that uniform set per sweep (no mixed-generation contraction). Then the
> backward becomes traceable under the drop and Gate B + Gate C can be measured for real. Gate the
> build on the inferred ~2× backward-block reduction being judged worth the blast radius (env-init +
> sweep refresh + projector truncation — the most-used symmetric paths).

For D≥3 U(1)-Sz today, the **dense** path remains pragmatic (memory `u1sz-perf-study-d3-findings`).

## Artifacts (all on the branch; no `src/` touched)

- `examples/census_u1sz_block_shapes_566.py` — extended with `predict_sectors_under_keep`,
  `candidate_report`, `_chi_axes_for` (chi-bond discriminator: underscore-in-label), `--candidate-keep`.
- `tests/test_u1sz_defrag_census_610.py` — static predictor unit tests.
- `examples/u1sz_defrag_prototype_610.py` — the throwaway C-lever prototype (3 coordinated
  monkeypatches + the surgical traced-backfill copy). Never imported by `src/`.
- `tests/test_u1sz_defrag_prototype_610.py` — faithfulness guard (forward drop + energy) + the
  pinned `test_backward_trace_does_not_survive_surgical_drop` obstruction test.
- `examples/probe_backward_jaxpr_566.py` — added `--defrag` flag + a dedicated charge-mask bucket.
- `examples/_e_frag_610.py`, `census_u1sz_baseline_610.json`, `census_u1sz_candidateC_610.json`,
  `probe_u1sz_baseline_610.txt`, `probe_u1sz_defrag_610.txt`.

## Risks / caveats recorded

- **`E_frag` is a random-site baseline** (≈ −0.06, not a ground state). Gate C's 1%-relative energy
  tolerance would have been hypersensitive against this near-zero scale; the follow-up's energy gate
  must use an **optimized** state (a few AD steps), not this init. (Moot here — Gate C not reached.)
- **Verdict scope:** NO-GO is for the *cheap measure-first route*, not for the physics of the lever.
  The forward evidence is favorable; the obstruction is a CTM-architecture property, not a refutation.
