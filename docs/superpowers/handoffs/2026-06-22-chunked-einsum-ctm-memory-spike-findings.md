# Chunked-einsum single-GPU memory lever for the dense CTM — Spike Findings

**Date:** 2026-06-22
**Provenance:** ports the `qiyang-ustc/omeinsum` (PyTorch) chunked-einsum idea to JAX.
**Spec:** `docs/superpowers/specs/2026-06-22-chunked-einsum-ctm-memory-spike-design.md`.
**Hardware:** single A100-80GB, f64.
**Verdict: GATE = GO — and stronger than expected.** Chunking a CTM-shaped `χ²·D⁶` contraction
via `jax.lax.map(batch_size=K)` cuts per-device peak ~K **at ~1.0× steady-state runtime** (no tax),
lifts the single-GPU D-ceiling **+2 in D**, and dodges the XLA giant-gemm autotuner wall. It's a
near-free single-device memory lever, complementary to (composes with) the #632 GSPMD sharding.

## Question

Does `lax.map(f, x, batch_size=K)` actually reduce the peak memory of the dominant dense-CTM
intermediate (`χ²·D⁶`) by ~K on one A100 — i.e. does XLA *respect* the chunk and stream it rather
than re-fuse — and at what runtime cost?

## Microbenchmark

`examples/spike_chunked_einsum.py` — an isolated contraction with the CTM peak's memory shape and a
clean free batch axis `B = χ²`:

    X:(B,D²,D²)  Y:(D²,D²,D²)   full:   M = einsum('bik,kjl->bijl')  # (B,D²,D²,D²) = B·D⁶ peak
                                chunked: lax.map(per_b, X, batch_size=B/K)  # peak ≈ (B/K)·D⁶

> **Measurement trap (re-learned, then fixed):** `peak_bytes_in_use` is a cumulative high-water mark
> JAX never resets — running all variants/Ds in one process reported an identical contaminated peak
> and a false OOM. **One variant × one D per process** is mandatory (the script is now single-shot).

## Results (χ=48 → B=2304, K=4 chunks, clean per-process)

| D | `full` peak | `chunked` peak | **peak ratio** | ceiling |
|---|---:|---:|---|---|
| 10 | 37.43 GB | **9.59 GB** | **3.9×** (≈ ideal 4×) | both OK |
| 12 | 56.14 GB | 28.08 GB | 2.0× | both OK |
| 14 | **autotuner-FAIL** | **35.90 GB** | `full` can't run | **chunked only** |
| 16 | autotuner-FAIL | autotuner-FAIL | — | neither |

- **C0 — XLA respects the chunk: YES.** D=10 peak drops **3.9×** for K=4 (near-ideal). The ratio
  fades to 2.0× at D=12 — a constant-factor floor (un-chunked inputs/outputs/XLA temps); more chunks
  help but the floor remains. Parity exact (`max|chunked−full| = 0.0`).
- **C0b — steady-state runtime tax: ~NONE.** Compile-subtracted warm time at D=10: `full` 49.0 ms vs
  `chunked(4)` 49.3 ms → **1.01×**. Chunking a *free* batch axis does **no recomputation** (same
  total FLOPs, streamed), so it's runtime-neutral. (The pre-spike fear of a K× forward tax was
  wrong — K× applies only to `remat`/backward recompute, not forward chunking.)
- **C1 — ceiling lift: +2 in D on ONE GPU.** `full` ceiling = D=12; `chunked` ceiling = D=14. Bonus:
  `full` dies at D=14 by an **XLA autotuner failure** on the giant `gemm_fusion_dot` (`f64[451584,
  38416]`), not pure OOM — chunking's smaller per-chunk gemms autotune fine, so chunking *sidesteps*
  the autotuner wall too. (At D=16 even the chunk gemm is too big → more chunks needed.)

## Gate decision: **GO**

| sub-gate | threshold | measured | |
|---|---|---|---|
| C0 peak reduction | ≥ ~4× at K=4 | 3.9× (D=10) | ✅ |
| C0b runtime tax | bounded (≤ K×) | **1.01×** (free-axis, no recompute) | ✅✅ |
| C1 ceiling | ≥ +1 in D | **+2** (D=12→14) + dodges autotuner | ✅ |

Chunking is a viable, near-free single-GPU peak-memory lever for CTM-shaped contractions. Unlike
GSPMD (N^(1/6), needs N devices), it works on one device and **composes** with sharding (chunk
on-device × shard across devices → multiplicative headroom).

## Honest caveats (what the microbenchmark does NOT prove)

1. **Production CTM may not chunk this cleanly.** The microbenchmark has a *clean free batch axis*
   (`χ²`). The real CTM peak is an intermediate inside a fused multi-einsum step (`_ctm_compiled_
   moves.py`) where the boundary-χ axis is entangled with contracted legs — the chunkable axis may
   not be exposed without restructuring the step. **This is the load-bearing follow-up unknown.**
2. **Constant-factor floor:** peak reduction was 3.9× (D=10) but 2.0× (D=12) — inputs/outputs/temps
   aren't chunked. Real gains are sub-K.
3. **Autotuner wall persists** at very large per-chunk gemms (D=16); needs more chunks, and is a
   separate XLA issue.
4. **Backward not gated here.** The `remat` (per-chunk checkpoint) path — omeinsum's other feature —
   *does* carry a recompute tax (it buys backward-activation memory); only the forward was measured.

## Recommendation

**GO to a follow-up production spike**, gated on caveat #1: can the dense CTM step (`_ctm_*`) expose
a chunkable free axis (boundary χ) on its peak intermediate without a costly restructure? If yes,
wire `lax.map(batch_size)` there and measure the real forward-CTM peak + warm step time, then compose
with GSPMD. If the peak axis can't be cleanly exposed, the lever stays a microbenchmark win. The
strategic picture is unchanged for *truly* large D (runtime-bound; eager/YASTN), but chunking is the
cheapest lever yet to push the single-GPU dense ceiling a couple of D values — at ~no runtime cost.

## Artifacts (branch `spike/chunked-einsum-ctm`)

- `examples/spike_chunked_einsum.py` — single-shot microbenchmark (`--variant`, one D/process).
