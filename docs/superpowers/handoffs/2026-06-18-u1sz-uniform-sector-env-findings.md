# U(1)-Sz uniform-sector env — flag-gated Gate-B measurement findings (#615)

**Date:** 2026-06-18
**Branch:** `feat/615-u1sz-uniform-sector-env` (off `origin/main` post-#614)
**Tracking issue:** #615 · **Parent:** #566 · **Predecessor spike:** #610 (PR #614, NO-GO-by-obstruction)
**Spec:** `docs/superpowers/specs/2026-06-17-u1sz-uniform-sector-env-measurement-design.md`
**Plan:** `docs/superpowers/plans/2026-06-17-u1sz-uniform-sector-env-measurement.md`
**Verdict:** **NO-GO.** The mechanism works — the #610 cold-backward obstruction is *structurally solved for one sweep* — but the measured charge-mask reduction (21.63%, flat in χ) misses the 25% Gate-B bar, the load-bearing #609 premise is **refuted**, and an A100 runtime check shows the lever is irrelevant against a ~24× #566 dispatch wall (and the full implicit-AD path doesn't even run under keep). Do **not** merge the flag to `main`; do **not** fund the full feature. The branch preserves the proven single-sweep mechanism for the record.

---

## What was built (opt-in, default-off; branch-local)

A `keep_sectors: frozenset[int] | None = None` flag (default `None` ⇒ byte-identical) realized as a **process-local context** read at trace time (`tenax.algorithms._ctm_uniform_sector`), mirroring the codebase's existing module toggles. Under `keep_sectors={-1,0,1}`:
- **Env-init** seeds every chi bond uniformly to `{-1:3, 0:4, 1:3}` (dim 10), D² legs preserved (`_ctm_tensor_init.py`).
- **Projector truncation** drops non-keep sectors at all four allocation sites (`linalg._truncated_svd_symmetric_traced`, `_ctm_projector._eigh/_svd_projector_symmetric`, `_ctm_tensor_projector_2x2._retruncate_by_base_charges`).
- **A-consistency (the key to making the backward trace):** the backward/multisite `_get_base_charges` returns the FULL D² pattern (tile-then-restrict, matching env-init) while the forward paired-moves `_get_base_charges` keeps restrict-then-tile (the two sweep paths need *opposite* policies — documented in-code), and the χ-backfill is **suppressed under keep** so `chi_new` = keep-target sum (=10), identical to env-init. This makes every chi bond (init and truncated) `{-1:3,0:4,1:3}`, so the 2×2 sweep's mixed-generation contractions pair equal per-sector dims.

Default-off is proven byte-identical: the flag-off backward op-histogram is unchanged (TOTAL 63612 / charge-mask 7398) and the core suite (977 passed) is unaffected.

## The three-gate result

### Gate A — sector census (PASS, as in #610)
Under keep the converged forward env drops corners 5→3, edges 19→9 (env blocks ~96→48, 2×), energy finite/sane. Reproduced exactly. (Faithfulness guard passes at D=3 χ=12.)

### Mechanism gate — cold backward VJP builds (PASS — the #610 obstruction is solved for one sweep)
The #610 spike could not even trace the 2×2 multisite backward under the drop (`ValueError: Size of label 'd' (4)!=(5)` in `_build_enlarged_corner`). Under the structural flag the **cold** backward VJP (`backward_vjp_jaxpr`, `jax.clear_caches()` first) **builds** (3506 eqns vs 6610 flag-off). Locked as a regression test (`test_cold_backward_vjp_builds_under_keep`). This is the structural advance #615 set out to make.

### Gate B — backward charge-mask re-profile (FAIL: 21.63% < 25%, and flat in χ)

| metric (D=3 χ=12, flag-off → keep) | flag-off | keep | reduction |
|---|---|---|---|
| **charge-mask / index-arith** | 7398 | 5798 | **21.63%** |
| TOTAL backward ops | 63612 | 45664 | 28.21% |

- **Below the 25% charge-mask bar.** The χ-sweep shows χ=12 and χ=16 are **byte-identical** (op count is fixed by the D=3 sector structure, independent of χ) ⇒ the reduction is **flat at 21.63%** and will not cross 25% at larger χ. (χ≥24 can't be measured: the *baseline* flag-off backward fails at χ=24 — a pre-existing D=3 large-χ fragility, unrelated to this work.)
- **The #609 premise is REFUTED.** Every bucket shrinks roughly uniformly (~0.59–0.87×, ~0.72× overall) — the charge-mask cluster is *not* a disproportionately-reducible "third"; it falls in proportion with the broad sector drop. #610's *inferred* ~40–50% was optimistic by ~2×.

### Gate C proxy — A100 warm-step runtime (decisive NO-GO)

| arm (D=3 χ=12, depth 8, warm step) | warm_ms |
|---|---|
| dense | **776** |
| u1sz fragmented (keep off) | **18,643** (~24× slower than dense) |
| u1sz keep {-1,0,1} | **errored** — `ValueError: ... float64[36] vs float64[34]` in the implicit-AD VJP |

- u1sz is **~24× slower than dense** at warm-step — the bottleneck is the **#566 eager per-block dispatch** (32 blocks), not the backward op count. A 28% op cut (or 21.6% charge-mask) is irrelevant against a 24× wall; it cannot approach dense parity.
- **The keep flag's full implicit fixed-point AD does not run.** Tactic A makes the *single-sweep* VJP trace, but the full `ctm_energy_implicit` value_and_grad needs the env pytree shape to be a **fixed-point invariant** under keep (the forward fixed-point and the adjoint must agree); under keep an env leaf comes out `[34]` where `[36]` is expected. End-to-end runtime under keep would need a *further* structural change (a fixed-point-invariant keep env), beyond the single-sweep mechanism built here.

## Verdict and recommendation

**NO-GO.** The de-fragmentation lever is real and the #610 single-sweep obstruction is structurally solved, but: (1) the measured charge-mask reduction is 21.63% — below the 25% bar and flat in χ; (2) the premise that charge-mask is disproportionately reducible is refuted; (3) the runtime gap to dense is ~24× and dominated by #566 eager dispatch, which a ~28% op cut cannot close; and (4) the full implicit-AD path isn't even runnable under keep without further structural work. Funding the full feature (Gate C / accuracy spine / tactic B sweep rewrite) is not justified on this evidence.

**Disposition:** the opt-in, default-off flag stays **branch-local** (not merged to `main`), preserving the proven single-sweep uniform-sector mechanism and the measurement tooling for the record. For D≥3 U(1)-Sz today, the **dense** path remains pragmatic (memory `u1sz-perf-study-d3-findings`). The binding wall for D≥3 U(1)-Sz is **#566 eager per-block dispatch** (the ~24× warm-step gap), not the backward charge-mask cluster — any future effort should target that, not env de-fragmentation.

## Artifacts (all on the branch)

- `src/` flag (branch-local, default-off): `_ctm_uniform_sector.py` (context), `_ctm_tensor_init.py` (env-init seed), `linalg.py` + `_ctm_projector.py` + `_ctm_tensor_projector_2x2.py` (truncation), `_ctm_tensor_convergence.py` + `_ctm_tensor_paired_moves.py` (split base_charges policy), `_ctm_energy_ad.py` (param + VJP-cache key).
- `tests/test_u1sz_uniform_sector_615.py` — identity (default-off), env-init keep-seed, forward block-count drop, **cold backward VJP builds**, faithfulness guard.
- `examples/probe_backward_jaxpr_566.py` (`--defrag` → real flag), `probe_u1sz_off_615.txt`, `probe_u1sz_on_615.txt`.
- `examples/profile_ctm_ad_wall_566.py` (`--defrag`), `profile_d3chi12_baseline_615.json`, `/tmp/timing_615.txt` (warm-step arms).

## Risks / caveats recorded
- **`E` from `compute_energy_ctm_tensor` is a random-init scale** (≈ −0.06, not a ground state); the faithfulness window (−2, 0) is a sanity gate, not an accuracy claim. (Moot — Gate C accuracy not reached.)
- **Verdict scope:** NO-GO is for env de-fragmentation as a runtime lever for D≥3 U(1)-Sz; the single-sweep mechanism is sound and preserved. The dominant wall is #566, surfaced cleanly by the ~24× warm-step gap.
