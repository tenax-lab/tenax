# HOTRG/TRG multi-GPU feasibility — **HOTRG = GO**, TRG = not useful

**Date:** 2026-07-03.
**Trigger:** #663 flagged TRG/HOTRG coarse-graining multi-GPU as the *one* multi-GPU lead
NOT refuted by the CTM-AD NO-GO (TRG/HOTRG are forward-only — no AD-through-SVD backward,
which was the wall that replicated the dominant CTM intermediate). This probe tests it.
**Method:** real `peak_bytes_in_use` high-water, full output tensor live (no DCE — the
mistake that produced a false CTM "GO" on 2026-07-02), one mode per process, 2×A100
(NCCL 4-way deadlocks on the DGX-Display box). Probe: `examples/probe_hotrg_multigpu.py`.

## Verdict

| algorithm | χ=32 repl | χ=32 shard | relief | χ=40 repl | χ=40 shard |
|---|---:|---:|---:|---:|---:|
| **HOTRG** | 25.80 GB | 12.91 GB | **2.00×** ✓ | **OOM** | **32.9 GB (fits)** ✓ |
| TRG | 0.084 GB | 0.113 GB | 0.75× (worse) | — | — |

- **HOTRG multi-GPU = GO.** Sharding the `up` leg gives **ideal 2.00× per-device relief**
  on 2 GPUs, numerically identical result, and **extends the χ ceiling**: χ=40 replicated
  OOMs (RESOURCE_EXHAUSTED, 30.5 GiB alloc = χ⁶) but sharded fits at 32.9 GB/device.
- **TRG = not a multi-GPU target.** Its peak is tiny (84 MB at χ=32) — TRG never forms a
  χ⁶ intermediate (it is χ⁴/χ⁵-memory), so there is nothing to shard at practical χ, and
  sharding adds overhead. (Plain TRG also has the CDL accuracy floor → use HOTRG anyway.)

## Why HOTRG works where CTM-AD did not

1. **Forward-only.** Free energy is a forward computation — no backward SVD-VJP. The CTM
   NO-GO (2026-07-02) was caused by the *backward* projector-SVD replicating the dominant
   operand; that mechanism is simply absent here.
2. **χ⁶-dominant / χ⁴-tiny-SVD structure.** tenax HOTRG explicitly forms
   `T_merged = (up,down,left,U,D,right) = χ⁶` (`_hotrg_step_horizontal` step 3) — the memory
   wall. The HOSVD operand is only `χ²×χ² = χ⁴` (8 MB at χ=32 vs 8.6 GB for χ⁶). Sharding a
   surviving leg (`up`) keeps the χ⁶ contraction at 1/N; the χ⁴ SVD replicates but is ~1000×
   smaller, so its replication is free. No dominant-intermediate all-gather.
3. **Single clean step** — no 4-direction sweep re-sharding one tensor to different axes
   (which eroded the CTM sweep to ~1×).

## Practical value

HOTRG's single-GPU χ ceiling is set by χ⁶: ~χ=40 (33 GB) on an 80 GB A100, hard-capped ~χ=44
(≈98 GB peak). Sharding across N GPUs multiplies the reachable χ⁶ by N → higher χ for
high-accuracy critical-point / large-χ HOTRG studies. This is a **real** large-χ lever for
HOTRG, unlike every CTM-AD multi-GPU attempt.

## Caveats / next steps

- Feasibility only (single step, `hotrg()` does not yet take a `device_mesh`). Productizing =
  add a `device_mesh` to `HOTRGConfig` and apply a `with_sharding_constraint` on a surviving
  leg inside `_hotrg_step_{horizontal,vertical}` (the alternating steps rotate which leg
  survives — shard the one that carries into the χ⁶ merge each step).
- 4-GPU untested (this box's DGX-Display GPU deadlocks 4-way NCCL); the clean 2.00×/N suggests
  it scales, which would reach χ≈48 (χ⁶≈98 GB / 4 ≈ 25 GB/device).
- A better long-term fix is to **not materialize χ⁶** (contract the isometries into the merge
  so the largest intermediate is χ⁵/χ⁴) — that raises the single-GPU ceiling directly and is
  orthogonal to sharding.

## Artifacts

- `examples/probe_hotrg_multigpu.py` — real high-water shard probe (HOTRG/TRG, per leg/χ/mode).
- Ties: #663 (v1.0 roadmap: "TRG/HOTRG multi-GPU untested, post-1.0"),
  `2026-07-02-570-reduced-corner-shard-gate.md` (the CTM NO-GO this contrasts with).
