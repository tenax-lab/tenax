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

## Productized (same PR)

`HOTRGConfig(device_mesh=<1-D Mesh>)` now shards dense HOTRG. Implementation
(`src/tenax/algorithms/hotrg.py`):

- Each step re-shards its input `up` leg via `with_sharding_constraint`
  (`_shard_leg`); `up` is present in both the horizontal and vertical χ⁶
  `T_merged`, so the eager contractions produce a sharded χ⁶ **without ever
  materializing it replicated**. Re-sharding each step handles the leg rotation
  across the horizontal/vertical alternation.
- **Eager, not jit.** `truncated_svd` does host-side rank truncation
  (`linalg.py:1793`, `np.array(s)`) and is **not jit-safe** — jitting the step
  crashes (`reshape (4,4)->(2,2,8)` on the first rank-deficient SVD). Eager
  `with_sharding_constraint` works and keeps χ⁶ sharded; the trade-off is ~49 GB
  vs a jitted ~33 GB at χ=40 (eager keeps extra copies), still a clear win.
- Guards: no-op for `device_mesh=None`, for non-`DenseTensor` (block-sparse HOTRG
  is small), and when a leg dim isn't divisible by the device count (early
  small-bond steps — no memory pressure there anyway).

Measured on the **full productized** `hotrg()` (2×A100, real high-water, β=0.3):

| χ (num_steps=8) | nomesh | mesh | note |
|---|---|---|---|
| 32 | 2.17 GB | **1.10 GB (2.0×)** | identical free energy |
| 40 | **FAILS** (autotuner RESOURCE_EXHAUSTED on the χ⁶ transpose) | **49 GB, fits, f=−2.6339** | **ceiling extension** |
| 44+ | fails | autotuner wall (even sharded) | compile-level cap |

So multi-GPU extends the reachable χ (single-GPU wall ~χ=40 → reachable with 2
GPUs) at 2× per-device relief and the same free energy. **Caveat:** beyond ~χ=44
an XLA autotuner wall on the giant `(χ³×χ³)` gemm/transpose caps *both* paths
(same class as the split-CTM D=12 wall; `--xla_gpu_autotune_level=0` may push it).
Tests: `tests/test_hotrg_sharding.py` (parity on fake CPU devices) + 22 existing
hotrg tests still pass.

## Artifacts

- `examples/probe_hotrg_multigpu.py` — real high-water shard probe (HOTRG/TRG, per leg/χ/mode).
- Ties: #663 (v1.0 roadmap: "TRG/HOTRG multi-GPU untested, post-1.0"),
  `2026-07-02-570-reduced-corner-shard-gate.md` (the CTM NO-GO this contrasts with).
