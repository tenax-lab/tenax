# Reduced-corner 1×1 CTM move — GSPMD shard gate: **NO-GO** (corrected)

**Date:** 2026-07-02.
**Status:** This supersedes an earlier draft of this file that reported "GO, 4× relief".
That verdict was **WRONG** — it came from a `memory_analysis` measurement artifact
(dead-code elimination). The corrected verdict from **real multi-GPU high-water**
measurement is **NO-GO**: the reduced-corner 1×1 CTM *sweep* does not achieve
meaningful multi-GPU per-device relief (~1.0×), the same practical outcome as #632.

## What happened (the methodology error)

The gate `examples/gate_reduced_corner_shard_570.py` measured per-device peak via
`compiled.memory_analysis().temp_size_in_bytes`, returning a **single leaf** (`T4`)
from the move. XLA's DCE then pruned everything except the one sharded absorption that
`T4` depends on, so the ratio looked like a clean 4× (and 2× on 2 devices). Returning
the **full env** (so nothing is pruned), or measuring **real `peak_bytes_in_use`
high-water** on GPU, both collapse the relief to ~1×.

Lesson (again, cf. `[[570-svd-vjp-wall]]` "op-count ≠ HLO"): **`memory_analysis` +
single-leaf return is not a faithful peak proxy — it is DCE-sensitive.** Use real
`memory_stats()['peak_bytes_in_use']` high-water, one config per process, exactly as
`examples/bench_ctm_sharding_memory.py` does. The isolated HLO collective scan
(`probe_qr_shard_reopen_570.py`) was *correct* about its narrow claim (the reduced-corner
decomposition all-gathers only a small operand) — but that narrow fact does **not**
translate into full-sweep per-device relief.

## The real numbers (2× A100, D=10 χ=32, `peak_bytes_in_use`, one mode/process)

> **Corrected 2026-07-03 (Codex review of #680, P1):** the earlier "replicated"
> baseline still passed `device_mesh=mesh` into the sweep and applied the manual
> `constrain_double_layer_for_move` in the absorb/move cases, so it compared two GSPMD
> layouts rather than sharding-off vs on. Fixed (`mode=="repl"` now uses
> `device_mesh=None` and no manual constraint). Re-measured numbers below; the **NO-GO
> conclusion is unchanged** (the full sweep still gets ~1×). The clean baseline is
> *higher* for the single move, so the move relief is actually better than first
> reported — but that only sharpens that the erosion is a **sweep** (cross-move) effect.

| layer | replicated (mesh=None) | sharded | relief |
|---|---:|---:|---:|
| **absorb** (isolated χ²D⁶ contraction) | 9.80 GB | 5.71 GB | **1.72×** ✓ |
| **move** (full `_ctm_tensor_move_left`, all env live) | 17.19 GB | 9.40 GB | **1.83×** ✓ |
| **sweep** (full 4-direction 1×1 sweep) | 17.19 GB | 17.60 GB | **0.98×** ✗ |

- The **isolated absorption shards** (1.72×) and the **single move shards well** (1.83×) —
  the GSPMD mechanism works at the move level (consistent with #632 rung-1's "contraction
  shards 2.00× without SVD").
- But the win **vanishes at the full sweep** (0.98×, marginally *worse* than replicated
  from sharding overhead). Across the 4 directions each move's output env comes back
  replicated (the projector/isometry produces replicated env), so moves 2–4 run on
  replicated env; the internal `_shard_a` only shards the small double-layer, not the env.
  The env sharding does not persist across moves.
- The single move shards ~1.8×, but that does **not survive** the 4-direction sweep
  (~1×). There is no N× multi-GPU reach here for the CTM-AD path.

## Why the premise looked good but fails

The `probe_qr_shard_reopen_570.py` finding — reduced-corner decomposition all-gathers only
a χD²×χ operand, not the χ²D⁶ intermediate — is **true and reproducible**. But the
projector decomposition was never the *whole* story: the full move's `_apply_projector_
with_reembed` + fuse/unfuse and the sweep's per-move `a` reshards reintroduce enough
replicated/copied work that per-device peak does not drop. The #632 conclusion (projector
SVD replicates the dominant operand → multi-GPU NO-GO) was framed around the 2×2 SVD, but
the *practical* multi-GPU-doesn't-help outcome holds for the 1×1 reduced-corner path too,
for a different (broader) reason.

## Code state (no productization needed / warranted)

`main` **already** applies `_shard_a` (`constrain_double_layer_for_move`) in the 1×1 branch
of `_ctm_tensor_sweep_multisite` (added generically in #632 rung-1, `4ce7564`). So there is
no hook to add — and it does not deliver relief. No production change is warranted.

## Disposition

- **NO-GO.** The reduced-corner 1×1 sweep is not a multi-GPU lever. Confirms #663's
  "multi-GPU deferred post-1.0" and `[[632-frontier-split-vs-dense-multigpu]]`
  (split-1GPU dominates). Large-D reach stays: **split-CTM on 1 GPU**.
- If anyone revisits: the ceiling to beat is the **move-level ~1.8×** (fix the projector/
  reembed sharding loss first); the sweep additionally needs env-sharding persistence and
  a way to avoid re-sharding `a` four ways. Given the 1.4× ceiling, low priority.

## Artifacts (characterization; measure with real high-water, not memory_analysis)

- `examples/probe_qr_shard_reopen_570.py` — isolated decomposition HLO collective scan
  (correct narrow claim: reduced corner all-gathers small).
- `examples/gate_reduced_corner_shard_570.py` — **memory_analysis-based; DCE-artifacted,
  do not trust its relief number**; kept only with this caveat.
- `examples/bench_1x1_shard_highwater.py` — the corrected real-high-water layered probe
  (absorb / move / sweep). This is the faithful measurement.
- Ties: `2026-07-02-570-qr-ctm-shardability-probe.md`, [[qr-ctmrg-multigpu-largechi-lever]],
  [[632-gspmd-svd-replication-rootcause]], #663.
