# QR-CTMRG "re-open multi-GPU" — shardability probe (a): findings

> **CORRECTION (2026-07-02, later same day):** the narrow HLO claim below (the
> reduced-corner decomposition all-gathers only a small χD²×χ operand) is correct and
> reproducible — but it does **NOT** translate into full-sweep per-device relief. The
> follow-up gate found the full reduced-corner 1×1 *sweep* does not shard (real
> high-water ~1.0×, NO-GO). See
> `docs/superpowers/handoffs/2026-07-02-570-reduced-corner-shard-gate.md`. Read this
> probe together with that correction, not as a standalone GO.


**Date:** 2026-07-02
**Trigger:** "explore QR-CTM" session. The QR-CTMRG kickoff brief flagged two payoffs
for arXiv:2505.00494's reduced-corner scheme: (1) large-χ single-GPU speed, (2) *possibly
re-open multi-GPU* by moving the decomposition off the dominant intermediate. This settles
de-risk step **(a)**: *does `jnp.linalg.qr` on a sharded matrix shard, or replicate like SVD?*
**Probe:** `examples/probe_qr_shard_reopen_570.py`. **Hardware:** 4× fake CPU device + 4× A100
(compile-only; NCCL 4-way clique deadlocks on this box due to the DGX-Display GPU, but the
GSPMD *partitioning* decision is a compile pass — CPU and GPU HLO agree exactly).

## Verdict: payoff-2 is STRUCTURALLY ALIVE — but the lever is the reduced-corner SCHEME, not QR

Two clean, GPU-confirmed facts:

1. **Neither QR nor tall-skinny SVD is shardable in XLA.** Both all-gather their input to a
   replicated copy before the (cuSOLVER) decomposition, then return a replicated output. The
   literal question "(a) is `jnp.linalg.qr` shardable?" is **NO** — same replication barrier as
   SVD. QR is *not* a sharding lever.

2. **But the reduced-corner scheme shrinks the replicated operand by D².** The decomposition's
   forced all-gather is confined to the *small* reduced corner; the dominant χ²D⁶ absorption
   contraction (which #632 proved shards cleanly) never has to be replicated to feed it.

### The decisive HLO (χ=128, D²=16 → χD²=2048; 4 devices; identical CPU & A100)

Input sharded on the tall (χD²) axis; measure the biggest `all-gather` XLA inserts:

| pattern | decomp operand | biggest all-gather | vs dense |
|---|---|---:|---:|
| `square-svd` (dense 2×2 analog) | χD² × χD² | **f64[2048,2048] = 33.55 MB** | 1× |
| `tall-svd` (reduced 1×1, method=svd) | χD² × χ | f64[2048,128] = **2.10 MB** | **16× smaller** |
| `tall-qr`  (reduced 1×1, method=qr) | χD² × 2χ | f64[2048,256] = 4.19 MB | 8× smaller |

GPU HLO for `square-svd` literally shows `all-gather-done = f64[2048,2048]` feeding
`custom-call(cusolver_gesvd)` on the full replicated matrix — this **is** the #632/#663/#672/#673
NO-GO mechanism ("projector SVD forces the dominant operand replicated"). The reduced corner
all-gathers only χD²×χ, which is **D²× smaller** (16× at D=4; 64× at D=8; 100× at D=10).

### QR vs SVD is nearly irrelevant for sharding

`tall-qr` all-gathers *more* than `tall-svd` (2χ-wide concat of both corners vs χ). So the
favorable multi-GPU profile belongs to the **reduced-corner `recipe="1x1"` scheme** — which
already exists with `projector_method="svd"` (#595/#597) — **not** to QR. This matches the
2026-06-22 end-to-end finding (QR's only clean win is ~1.4× memory ceiling, not speed).

## Why this dissolves the #632 multi-GPU NO-GO (in principle)

- #632 rung-1 measured the CTM **contraction shards fine** (2.00× per-device without the SVD).
- The SVD collapsed it to ~1.20× by all-gathering the dominant intermediate (dense χ²D⁶ or
  split (χD)²). Root cause: the SVD operand *is* (a reshape of) the dominant intermediate.
- The reduced-corner scheme makes the SVD/QR operand the **reduced** corner (χD²×χ), which is
  D²× smaller than the dominant absorption. So the decomposition still replicates — but it now
  replicates a *small* thing, leaving the χ²D⁶ absorption free to stay sharded.

The obstacle that was declared *path-agnostic* (dense + split both replicate the dominant
operand) is **not** scheme-agnostic: it depended on the decomposition operand *being* the
dominant intermediate. The reduced-corner scheme breaks that identity.

## What this does NOT prove (the real gate remains)

- **No end-to-end per-device relief was demonstrated.** The probe's peak-memory metric was a
  construction artifact: the toy producer materializes a replicated `(χ,D²,D²,χ)` buffer *before*
  the sharded contraction with `a`, so that buffer set the peak (production threads the sharded
  axis through the whole absorption via `ctm_sharding._MOVE_SURVIVING_AXIS`; the toy injects it
  too late). The collective-operand evidence is robust; the peak number is not.
- The reduced corner is still **all-gathered** (replicated χD²×χ per device) + a replicated tiny
  SVD/eigh. At very large χ this fixed replicated cost grows ∝ χ²D²; whether it becomes the new
  bottleneck vs the sharded χ²D⁶ absorption needs the real measurement.
- A residual **all-reduce** forms the reduced corner when the reduction contracts the sharded
  axis (cheap, χ²D²-class).

## Recommendation / next gate

**GO to the build gate (post-1.0; #663 defers multi-GPU).** The single structural obstacle that
killed multi-GPU CTM-AD is removed by the reduced-corner scheme. Next, the concrete gate:

1. Wire GSPMD sharding (reuse `ctm_sharding.py` `_MOVE_SURVIVING_AXIS` threading) through the
   **actual reduced-corner `recipe="1x1"` move** (SVD variant first — simpler, better sharding
   profile than QR), forward only.
2. Measure per-device peak 1-GPU vs N-GPU at large D (D=8/10). GO if it recovers the ~2× rung-1
   (no-SVD) profile — i.e. the reduced corner's replication does *not* re-collapse it.
3. Only then: repeat for the backward (rung-2), and reconsider QR purely for its memory-ceiling
   headroom on top.

QR-specific work is **not** the lever. The lever is *reduced-corner scheme + GSPMD*.

## Artifacts

- `examples/probe_qr_shard_reopen_570.py` — pattern probe (`full`/`reduced-svd`/`reduced-qr`)
  + isolated tall-vs-square decomposition (`--only-isolated`). CPU (fake devices) or GPU.
- Ties to: [[qr-ctmrg-multigpu-largechi-lever]], [[632-gspmd-svd-replication-rootcause]],
  [[632-frontier-split-vs-dense-multigpu]], #663 (v1.0 roadmap, multi-GPU deferred),
  `2026-06-22-qr-vs-svd-projector-largechi-findings.md`.
