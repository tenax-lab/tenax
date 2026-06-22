# Design: chunked-einsum single-GPU memory lever for the dense CTM — feasibility gate

**Date:** 2026-06-22
**Status:** Approved (brainstorming) — gate-first feasibility spike.
**Provenance:** Inspired by `qiyang-ustc/omeinsum` (PyTorch: chunked einsum + per-chunk
checkpoint + multi-GPU dispatch). This spike ports the **chunking** idea to JAX and tests it as a
**single-device** peak-memory lever for the dense iPEPS CTM.
**Relation to #632:** orthogonal/complementary to the GSPMD sharding (#632 rung 1/2). GSPMD spreads
one intermediate across N devices (capped ~N^(1/6) in D); chunking streams it through *time* on one
device (peak ÷ K, at ~K× sequential compute). They compose.

## The question this gate answers

Does `jax.lax.map(f, x, batch_size=K)` (+ optional `jax.checkpoint`) actually reduce the **peak
device memory** of the dominant dense-CTM intermediate (the `χ²·D⁶` enlarged-corner/absorption
contraction) by ~K on a single A100 — i.e. does XLA *respect* the chunking and stream it rather than
re-fuse to the full materialization — and at what **runtime** cost? GO/NO-GO before any production
wiring.

## Why a gate, and the load-bearing risk

The whole lever rests on XLA honoring the chunk boundary. XLA aggressively fuses; `lax.map` lowers
to `scan`, which *should* keep only `batch_size` elements live, but XLA may unroll/re-fuse and
materialize the full intermediate anyway → **no memory win**. This is not safely inferable; measure
it. Second risk: dense large-D CTM is already **runtime-bound** (`570-dense-largeD-study`: D=3 χ=16
≈ 108 min), so a K× compute tax may make the larger D it *fits* too slow to use — the gate must
report the runtime multiplier, not just the memory drop.

## Experiment (isolated microbenchmark — cheapest faithful test)

Mimic the CTM peak with a contraction whose intermediate scales as `χ²·D⁶` and whose batch axis
(`B = χ²`) is a free output index (so it chunks cleanly):

    X : (B=χ², D², D²)      Y : (D², D², D²)
    full(X,Y):    M = einsum('bik,kjl->bijl', X, Y)   # (B, D², D², D²) = B·D⁶  ← the peak
                  return M.sum((2,3))                  # (B, D²)
    chunked(X,Y,K):  lax.map(per_b, X, batch_size=K)   # peak ≈ (K·D⁶) → full_peak ÷ (B/K)
    chunked_remat:   jax.checkpoint(per_b) variant      # backward-memory angle (secondary)

`per_b` computes one B-row's contribution; `lax.map(..., batch_size=K)` processes K rows per scan
step. This is the exact memory shape of the rung-1 peak, with a clean chunk axis.

## Measurements (single A100, f64, XLA_PYTHON_CLIENT_PREALLOCATE=false, one config/process)

- **C0 — does chunking reduce peak?** Per-device `peak_bytes_in_use` of `full` vs `chunked(K)` at a
  fixed (D, χ) large enough to be GBs. Expect chunked ≈ full ÷ (B/K). If peak does **not** drop →
  XLA re-fused → NO-GO.
- **C0b — runtime tax.** wall(chunked)/wall(full) vs the chunk count B/K.
- **C1 — ceiling lift.** Largest D where `full` runs vs where `chunked` runs (full OOMs, chunked
  fits).

## GO / NO-GO

- **GO** ⟺ (C0) chunked peak drops materially (≥ ~4× at a B/K≈4 chunking, i.e. XLA respects it)
  **and** (C1) chunked runs at a D where `full` OOMs on one GPU **and** (C0b) the runtime tax is
  bounded (≈ chunk-count×, not catastrophically super-linear). → worth a follow-up: wire chunking
  into the real CTM peak einsum (and/or combine with GSPMD).
- **NO-GO** ⟺ XLA re-fuses (no peak drop), or the runtime tax makes the larger-D runtime prohibitive
  (likely, since dense large-D is already runtime-bound) → document; chunking is not the lever for
  the CTM peak; GSPMD (#632) + eager/YASTN stand.

## Components

- **Create** `examples/spike_chunked_einsum.py` — standalone microbenchmark: `full` / `chunked(K)` /
  `chunked_remat`, parity check (chunked == full to ~1e-10) + per-device peak + wall, CLI
  `--D --chi --batch --shard?`. Throwaway.

## Out of scope (follow-ups only if GO)

- Wiring chunking into the production `_ctm_*` step einsums.
- Composing chunking with GSPMD sharding (chunk on-device × shard across devices).
- Chunked backward / `remat` integration with the implicit-AD adjoint.
- A general `tenax.contraction` chunked-einsum utility.
