# #672 frontier re-derivation at `recipe=2x2` — findings

**Date:** 2026-08-09
**Issue:** #747 (last remaining item), re-deriving PR #672 + the #673 ceiling addendum
**Harness:** `examples/bench_ctm_frontier_grad.py` + `tests/_frontier_grad_probe.py`
(both taught a `--recipe` knob for this run) driven by
`scripts/bench_672_rederivation.py`; tables from `scripts/analyze_672_rederivation.py`
**Raw data:** `runs/672_rederivation/{results.jsonl,gate.jsonl}`
**Hardware:** A100-SXM4-80GB, GPUs 0/1/2, f64

## Why this was re-run

PR #672 concluded *"split-CTM on 1 GPU dominates dense multi-GPU everywhere on the
frontier"* — D=10 split χ≥128 vs dense-2GPU χ16, D=12 split χ96 vs dense unable to
compile. **Both arms ran `recipe="1x1"`**, whose corner-pair projector collapses the
CTM environment to a rank-1 corner (#723/#726). A rank-1 corner is a χ_eff=1
mean-field boundary, so every (D,χ) reach number described how cheaply that
degenerate object scales rather than the reach of a converged CTM (#747).

#747 Run 1 re-ran the *forward* D=8 split path at `2x2` and found the χ ceiling
identical to dense and the memory advantage 1.02–1.03× rather than ~12×. That
result does not formally transfer here: #672 measured `jax.value_and_grad` peaks
through the implicit-AD `custom_vjp`, and nobody had measured split-vs-dense
**backward** peaks at `2x2`.

## Protocol

Matching the original where it was sound, fixing it where it was not.

| aspect | original (2026-07-01) | this run |
|---|---|---|
| recipe | `1x1` dense arm (hard-coded), split arm unset | **`2x2` both arms**, explicit |
| allocator | BFC (`PREALLOCATE=false` only) | **`cuda_async`** + `PREALLOCATE=false` |
| peak metric | `peak_bytes_in_use`, **device 0 only** | same, but **max over all devices** |
| isolation | one config per process | same (one cell per subprocess) |
| collapse gate | none | `rank(C1)` via `ctm_corner_rank` |
| autotuner walls | reported as ceilings | **re-probed** with `--xla_gpu_autotune_level=0` |
| warmup | none | none (kept, for comparability) |
| x64 / max_iter / seed | on / 30 / 0 | unchanged |

Three of those changes are corrections, not preferences:

- **Allocator.** #747 established that BFC fragmentation inflated the D=8 peaks and
  that comparing a `cuda_async` arm against a BFC arm is what produced the original
  ~12× memory error. Both arms here use `cuda_async`.
- **Peak across devices.** A review comment on PR #673 (`discussion_r3506756167`,
  marked resolved on GitHub but never actioned in code) noted that
  `spike_split_shard_ctm_gate.py` read `jax.devices()[0]` while using the value as
  *the mesh's* per-device peak, which under-reports and flatters the sharded arm.
  The #672 frontier harness has the same defect
  (`bench_ctm_frontier_grad.py:39`); fixed here in `peak_gb()`.
- **Autotuner walls.** The 2026-07-02 addendum to the #673 handoff (PR #678) showed
  the `(χD)` giant-transpose/gemm autotuner failure can be a *soft* wall that
  `--xla_gpu_autotune_level=0` bypasses — though only the D=10 ceiling actually
  lifted (χ128→χ160); D=8 χ=256 merely converted to an OOM. Reporting such a cell
  as a ceiling risks reporting a compiler setting, so every autotuner-raised
  failure here is re-probed.

### The harness could not do this comparison before

`tests/_frontier_grad_probe.py:59` pinned `recipe="1x1"` on the dense arm while the
split arm passed no recipe at all. Since #746 both `ctm_energy_implicit` and
`ctm_energy_split_implicit` default to `"2x2"`. So **re-running the original harness
unmodified today would have silently compared dense-`1x1` against split-`2x2`** — a
worse comparison than the one being re-derived. Both arms now thread `--recipe`.

The knob was verified live rather than assumed, which matters because #795 reports
`ctm_energy_split_implicit` silently falling back to the 1x1 sweep on an
unrecognised recipe string. This check was run out-of-band on CPU and is **not**
in `runs/672_rederivation/` — it is reproduced here for the record:

| path | recipe | E (D=3, χ=8) | corner_rank |
|---|---|---|---|
| split | `1x1` | −0.0000576710 | **1** |
| split | `2x2` | −0.0000568515 | **8** |
| dense | `1x1` | −0.0000576710 | **1** |
| dense | `2x2` | −0.0000568523 | **8** |

Note the two paths are **bit-identical at `1x1`** — the collapse makes them the same
computation — and separate at `2x2`. That is itself evidence the original benchmark
was comparing two spellings of one degenerate object.

## Caveats that bound what these numbers can say

- **No warmup: every wall time includes XLA compile.** The original harness timed a
  single cold call and published no timings; the same is true here. Nothing in this
  document is a runtime claim.
- **`OK` is not "converged".** The probe checks no convergence flag, so a cell that
  exhausts `max_iter=30` still reports `OK`. #767 (split `2x2` AD forward can return
  a non-converged environment silently) is live.
- **Units.** Peaks are decimal GB (`bytes/1e9`), matching the original harness; XLA's
  own OOM messages are GiB. Both appear below, as they did in the original table.
- The probe's gate is `diag(0.25,−0.25,−0.25,0.25)`, an Ising-like diagonal, and the
  state is a deliberately near-product well-conditioned init. Energies here are
  memory-study bookkeeping, not physics — in particular their χ-invariance is
  expected and is **not** the collapse signature.

## The control: the recipe is isolated as the cause

Before any reach claim, the recorded cells were re-run at `1x1` on **this** box,
**this** code, **this** allocator. If code drift or the BFC→`cuda_async` switch
were responsible for the differences below, these would not reproduce.

| cell | recorded 2026-07-01 (BFC) | control today (`cuda_async`) | Δ |
|---|---|---|---|
| **split** D=8 χ=128 | 16.93 GB | 16.63 GB | −1.8% |
| **split** D=8 χ=224 (#673 ceiling) | 51.2 GB | 50.90 GB | −0.6% |
| **split** D=10 χ=128 | 39.89 GB | 39.85 GB | −0.1% |
| **split** D=12 χ=96 | 46.35 GB | 46.29 GB | −0.1% |
| **dense** D=10 χ=16 (1-GPU) | 28.01 GB | 25.96 GB | **−7.3%** |
| **dense** D=10 χ=24 (1-GPU) | **OOM** (29.0 GiB) | **OK @ 41.57 GB** | — |
| **dense** D=12 χ=16 (1-GPU) | **autotuner-fail** | **OK @ 83.03 GB** | — |
| **dense** D=12 χ=24 (1-GPU) | autotuner-fail | OOM | — |

Recorded values are the handoff table's, not PR #672's rounded body figures.

**The two arms behave differently under the allocator change, and this matters.**
Every *split* control reproduces to within 1.8%, so the split reach collapse
reported below is cleanly attributable to the recipe. The *dense* controls do not:
at an unchanged `1x1` recipe, moving BFC→`cuda_async` alone shifts dense-1GPU D=10
from an OOM at χ=24 to running it at 41.57 GB. That is the same BFC-fragmentation
artifact #747 identified at D=8, now confirmed on the dense arm.

The D=12 row settles a question the `2x2` grid alone could not. #672 recorded
`autotuner-fail` for dense D=12 χ=16 on **all three** dense configs and concluded
split was "the *only* path that runs D=12". At `1x1` — the original recipe — that
cell runs here at 83.03 GB. **So that claim falls to the allocator/toolchain, not
to the recipe.** An earlier draft of this document attributed it to the recipe;
that was wrong, and only the control exposed it.

**Consequence, stated plainly: the dense arm's improvement over the recorded
figures is confounded between recipe and allocator and is not apportioned here.**
The split-vs-dense comparisons in this document are unaffected, because both arms
in every `2x2` cell ran the same allocator on the same box — the confound is only
between *this* run and the *2026-07-01* run, and only on the dense side.

One caveat on the sources rather than on this run: #673's ceiling addendum records
the *same* split D=10 χ=128 cell as **37.3 GB** on the same day the frontier
handoff recorded 39.89 GB — a 7% internal inconsistency in the 2026-07-01 data.
The control agrees with the frontier figure; against #673's it is +6.8%.

## Collapse gate

`rank(C1)` at production D, both paths — `2x2` environments are genuine, so these
are peaks of real CTM environments and not of a rank-1 corner:

| cell | corner_rank |
|---|---|
| D=8 dense χ=32 / split χ=32 | 31 / 32 |
| D=10 dense χ=16 / split χ=16 | 16 / 16 |
| D=10 dense χ=32 / split χ=32 | 31 / 32 |

(A 31 is one singular value below the 1e-10 tolerance, not a collapse.) The six
cells above are in `gate.jsonl`.

**The gate is an inference, not a per-cell certification.** It was measured at
χ≤32 and D∈{8,10} — not at any *ceiling* cell (D=8 χ=96, D=10 χ=48, D=12 χ=32) and
not at D=12 at all — and the split gate calls `ctm_split_tensor` rather than the
`ctm_energy_split_implicit` path the benchmark differentiates. The recipe is a
global switch and phase-fixing is rank-preserving, so it very probably transfers;
but it has not been demonstrated on the cells whose peaks the reach table reports. For contrast, the same probe at `1x1` returns rank
**1** on both paths — and the two paths are then *bit-identical* in energy and
gradient, which is direct evidence that the original benchmark compared two
spellings of one degenerate object. Those `1x1` rank checks were run out-of-band
(D=3/D=4, CPU and GPU) and are not in `gate.jsonl`.

## Reach at `2x2`

Every arm below is walled — the rung above each entry was run and failed, so no
cell is a grid cap.

| D | dense 1-GPU | dense 2-GPU | split 1-GPU | split vs dense-2GPU |
|---|---|---|---|---|
| 8 | χ=64 @ 45.22 GB | χ=96 @ 71.76 GB | **χ=96 @ 77.51 GB** | **tie** |
| 10 | χ=48 @ 79.88 GB | χ=48 @ 68.81 GB | **χ=48 @ 78.39 GB** | **tie** |
| 12 | χ=16 @ 67.37 GB | χ=16 @ 76.77 GB | **χ=32 @ 77.54 GB** | **2×** |

Against the recorded `1x1` split reaches (χ=224 / χ=128 / χ=96) and the recorded
split-vs-dense-2GPU gaps (4× / 8× / "the only path that runs D=12").

Split's χ ceiling is **2.3–3× below** the recorded figures (224→96, 128→48,
96→32). Two honest qualifications on that arithmetic:

- It is not a capability *regression*. A `1x1` χ and a `2x2` χ are not
  commensurable — at `1x1` the corner is rank-1 regardless of χ — so the recorded
  numbers were never reach numbers to begin with. They are **void**, not 2.3×
  optimistic.
- The recorded D=10 χ≥128 was itself annotated a **grid cap, not a wall**, in the
  2026-07-01 handoff, so 128→48 understates the true `1x1` figure.

Every ceiling here also carries **±1 ladder rung** (the ladder is
{16,24,32,48,64,96,128,224}), so the χ ratios below are one-significant-figure
statements, not measurements to two digits.

Note the dense 2-GPU column is not uniformly better than 1-GPU: at D=8 sharding
buys a full rung (χ=64→96), at D=10 it buys nothing, and at D=12 it is a net
**loss** — 76.77 GB against 67.37 GB for the same χ=16 ceiling, i.e. the GSPMD
overhead exceeds its relief.

**Every wall here is real.** All four autotuner-raised failures (dense D=10 χ=64,
split D=10 χ=64, split D=12 χ=48, split D=8 χ=128) were re-probed with
`--xla_gpu_autotune_level=0` and every one converts to a plain OOM — so these are
memory ceilings, not compiler artifacts. The `1x1` ceilings of #673 were reported
as autotuner walls with 30+ GB of headroom, and its later addendum showed at least
the D=10 one was genuinely soft (χ128 lifted to χ160). Nothing here lifts.

## The split advantage erodes with χ and dies at the wall

Peak at matched (D=10, χ), 1 GPU:

| χ | split | dense | dense/split |
|---|---|---|---|
| 16 | 8.73 | 23.21 | **2.66×** |
| 24 | 19.61 | 46.38 | 2.37× |
| 32 | 34.85 | 47.96 | 1.38× |
| 48 | 78.39 | 79.88 | **1.02×** |

Split obeys χ² to within 0.3% across the whole range (8.73→19.61→34.85→78.39 vs
8.73×{1, 2.25, 4, 9}); dense does not — it flattens. So the advantage is largest
where memory is not the binding constraint and **gone at the ceiling**.

That 1.02× at the wall independently reproduces the 1.02–1.03× #747 Run 1 measured
on the D=8 *forward* path, by a different route on a different quantity. The two
results agree.

## Verdict on each recorded claim

| recorded claim | verdict |
|---|---|
| "split-CTM 1-GPU dominates dense multi-GPU **everywhere** on the frontier" | **REFUTED** — it ties dense-2GPU at D=8 and D=10 |
| D=10: split χ≥128 vs dense-2GPU χ16 (~8× the χ) | **REFUTED** — χ=48 vs χ=48, a dead heat |
| D=8: split χ≥128 ≈ dense-2GPU χ32, "4× the χ" | **REFUTED** — χ=96 vs χ=96, also a dead heat |
| D=12: split χ96; dense "cannot compile even χ16", split "the *only* path that runs D=12" | **REFUTED, but by the allocator not the recipe** — dense D=12 χ=16 runs at **`1x1`** too (83.03 GB) once BFC is replaced. Split's own χ=96→32 |
| split-1GPU ceilings χ=224 (D8) / χ=128 (D10), *compile*-bound with headroom | **REFUTED** — χ=96 / χ=48, and **memory**-bound |
| dense GSPMD shard relief ~1.1–1.4×/device, never extends D-reach | **CONFIRMED on reach, wider than recorded on relief** — the full matched span is **0.88×–1.77×** (1.10–1.62× at D=10 alone). D=12 χ=16 is **0.88×**, i.e. sharding *costs* memory there. Ceiling unchanged by sharding at every D |
| multi-GPU CTM-AD is a NO-GO (SVD-replication, path-agnostic) | **UNAFFECTED** — mechanism-level, independent of recipe |
| "the memory win over the fused path is a large-D (D≳16) effect" (`CHANGELOG.md` v0.8.1; `README.md:26` says the same in its own words) | **NOT CONTRADICTED** — no D≥16 cell was run, so it cannot be confirmed; but it is the conservative framing and the D≤12 data is consistent with it |

The 8× D=10 gap closed from **both** ends: split's recorded reach was inflated by
the recipe, and dense's was suppressed — though the dense half is the allocator's
doing at least as much as the recipe's (see the control). Dense-2GPU goes from a
recorded χ=16 to a measured χ=48 at D=10, and from χ=32 to χ=96 at D=8.

## What survives

Split-CTM is still the better path **on equal hardware**: against dense on one GPU
it buys 1.5× in χ at D=8, ties at D=10, and buys 2× at D=12, plus up to 2.66× in
peak at small χ. Against the *two-GPU* dense apparatus — the comparison #672
actually made — it ties at D=8 and D=10 and wins only at D=12, where it delivers
2× the χ on half the hardware. That last cell is the one place the original
verdict still holds in spirit.

The D-dependence is **non-monotonic** on this sample (1.5× / 1.0× / 2.0× at
D=8/10/12), so "the advantage grows with D" is not something these three points
establish; they are merely consistent with the `D≳16` hedge already in `README.md`
and the v0.8.1 changelog.

## Roadmap impact (#663)

v1.0 leans on the refuted figures in four places. The *decision* — multi-GPU
deferred post-1.0 — is unaffected, because it rests on the SVD-replication
mechanism, not on this comparison. What needs rewording is the **positive** claim:

- Decision log: "Split-CTM on 1 GPU dominates dense on 2 GPUs across the entire
  (D,χ) frontier" → false at D=10, overstated elsewhere.
- Scope item 2: "frontier: **D=10 χ≥128, D=12 χ96 on one A100**" → the measured
  `2x2` figures are **D=10 χ=48, D=12 χ=32**.
- Post-1.0: "phase-1 frontier grid capped at 128; true ceiling higher" → the true
  ceiling is *lower*, and it is memory-bound rather than compile-bound.
- `CHANGELOG.md:175–180` (v0.8.2): "split-CTM on 1 GPU beats dense on 2 across the
  whole (D,χ) frontier" (line 177) — retract the clause; the rest of the entry,
  including the multi-GPU NO-GO and the HOTRG note, stands.

v1.0 still has a large-D story, but it is "split-CTM buys 1.5–2× in χ over dense on
equal hardware at D=8 and D=12, and matches a 2-GPU dense run on one GPU", not
"split-CTM reaches χ≥128 at D=10 and dominates dense multi-GPU everywhere".

## If someone extends this

- The ladder is {16,24,32,48,64,96,128,224}, so every ceiling is ±1 rung. Refining
  near each wall (72/80 at D=8, 40 at D=10/D=12) would tighten the χ ratios, which
  are currently one-significant-figure.
- No gate was run at a ceiling cell or at D=12. Gating D=8 χ=96 and D=12 χ=32
  would close the inference noted above.
- The dense-arm confound could be apportioned by re-running the full 2026-07-01
  dense grid at `1x1` under `cuda_async`; two cells of that exist here and both
  moved, so the rest probably would too.
- `dense flattens in χ` (23.21 → 46.38 → 47.96 → 79.88 at D=10) fits no χ²·D⁶
  model. It is most likely XLA's scheduler rematerializing harder as the buffer
  approaches capacity — worth confirming, because it is what makes the
  split advantage vanish at the ceiling.

