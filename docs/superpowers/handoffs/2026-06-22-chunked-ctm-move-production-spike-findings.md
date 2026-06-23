# Chunked-einsum in the REAL dense-CTM move — Production-follow-up Spike Findings

**Date:** 2026-06-22
**Parent:** the chunked-einsum microbenchmark spike
(`2026-06-22-chunked-einsum-ctm-memory-spike-findings.md`, GO). That spike's open question:
*does the production CTM step expose a chunkable free axis on its `χ²·D⁶` peak, or only an
idealized microbenchmark?* This resolves it.
**Hardware:** single A100-80GB, f64.
**Verdict: GATE = GO — the real CTM move IS chunkable.** Streaming the left-move edge path over the
boundary-χ axis (accumulating through the projector) reproduces the real projector output to **2e-18**,
cuts the move's peak by ~the chunk count, takes the move from *OOM-at-D=10* to *runs-at-D=12*, scales
**better at large χ** (peak ~linear vs `full`'s quadratic), at a **1.4×** steady-state runtime tax.

## What was tested (faithful, not idealized)

`examples/spike_chunked_ctm_move.py` mirrors `_ctm_compiled_moves._compiled_move_left`'s edge path:
`T4_a = einsum('ijk,lmjn->iklmn', T4, a)` (the `(χ,χ,D²,D²,D²)=χ²·D⁶` **peak**) → transpose →
reshape to the grown edge `T4g` → `T_new = P1^H @ T4g @ P2` via the **real** `_apply_projector_raw`.

The boundary-χ axis (`t4_d=i`) is free in `T4_a` but becomes **contracted** through the projector
(`fl=(i,l)`), so chunking = **stream over i + accumulate the projector contraction**. The chunked-fused
move (`lax.map(per_i, …, batch_size=K)` then sum) does exactly that. (Projectors `P1/P2` are random of
the real shape — the projector **SVD** itself is *not* exercised; see the large-χ caveat.)

## Results

### Parity — exact
`max|chunked − full| = 2.2e-18` (rel 8.6e-16). The chunked-fused move == the real move.

### Peak (χ=48, K=4 chunks) — `full` OOMs where `chunked` runs

| D | `full` move | `chunked` move |
|---|---:|---:|
| 10 | **OOM** (+34 GiB) | **14.65 GB** ✅ |
| 12 | autotuner-FAIL | 44.75 GB ✅ |
| 14 | FAIL | FAIL (autotuner) |

The **real** move OOMs at D=10/χ=48 (it materializes `T4_a` **and** its transpose — ~2–4× the
microbenchmark's single intermediate); chunked runs. Ceiling lift ≥ +2 in D.

### Steady-state runtime tax — 1.4× (bounded)
Compile-subtracted warm, χ=24 D=10: `full` 20.2 ms vs `chunked(4)` 28.8 ms → **1.42×**. Higher than the
microbenchmark's 1.0× because the projector tensordots + transpose run **per chunk**; still far from K×.
(At χ=24 **D=12 the `full` move already OOMs** — chunked is the only one that runs.)

### Large χ — chunking wins MORE (the answer to "how about large χ?")
D=8, **fixed** batch=16 (so n_chunks = χ/16 grows with χ):

| χ | `full` peak | `chunked` peak |
|---|---:|---:|
| 48 | 14.65 GB | 4.99 GB |
| 64 | 25.93 GB (×1.77 ≈ (64/48)²) | 6.63 GB |
| 96 | **OOM** | 13.09 GB |
| 128 | OOM / autotuner | 21.69 GB |

`full`'s edge peak grows **quadratically** (`χ²·D⁶`) and OOMs by χ=96; `chunked` with fixed batch grows
**~linearly** (`batch·χ·D⁶`) and still runs at χ=128. The chunking advantage **widens** with χ — which is
exactly the regime you want (large χ = better CTM accuracy). This is the strongest result of the spike.

## Gate decision: **GO**

The production CTM move exposes a chunkable axis; chunking it is correct, cuts the dominant `χ²·D⁶` peak
by ~the chunk count, lifts the single-GPU ceiling, scales sub-quadratically in χ, and costs ~1.4× warm.
The microbenchmark's load-bearing caveat is **resolved**.

## Caveats / scope

1. **Projector SVD is a SEPARATE large-χ bottleneck** this lever does NOT touch. Edge-chunking bounds the
   *edge-growth* peak (`χ²·D⁶`); at large χ the real CTM also pays the projector SVD on a `χD²×χD²` matrix
   — `(χD²)²` memory + `(χD²)³` compute (D=8 χ=128 → 8192² = 0.5 GB / ~0.5 s; χ=256 → 16384² = 2 GB /
   ~5 s). That is the **rung-3 distributed-SVD** problem. So: chunking solves the *dominant* edge wall at
   large χ; the SVD wall takes over only at very large χ and is a different lever.
2. **One move, edge path only.** This is the left move's edge growth + projector-apply. Production needs
   all four moves chunked (symmetric), the **corner** accumulation (`C_new = P^H @ Cg`, also over the χ
   axis — small), and wiring into the sweep loop. Mechanically straightforward; not done here.
3. **Autotuner wall persists** (D=14, χ=128): very large per-chunk gemms still fail XLA autotuning →
   more chunks (smaller batch) needed; orthogonal XLA issue.

## Recommendation

**GO to a production implementation:** add a chunked path to the four `_compiled_move_*` edge/corner
contractions (opt-in `n_chunks`/`batch` knob), defaulting off. It is a single-GPU memory knob that
**composes** with the #632 GSPMD sharding (chunk on-device × shard across devices) and is most valuable
at large χ. It does not change the truly-large-D *runtime* verdict (eager/YASTN), but it is the cheapest
lever yet to push the single-GPU dense ceiling — especially in χ.

## Artifacts (branch `spike/chunked-einsum-ctm`)

- `examples/spike_chunked_ctm_move.py` — faithful chunked-fused left-move edge path; parity + per-device
  peak (`--variant full|chunked|parity`, `--batch`, one variant/process).
