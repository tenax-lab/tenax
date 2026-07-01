# Chunked dense-CTM backward — Increment 2 gate findings

**Date:** 2026-07-01
**Branch:** `feat/632-chunk-ctm-absorb-inc1`
**Parent:** Increment 1 (forward 1×1 chunked absorb, #668) — `2026-07-01-chunk-ctm-absorb-increment1-build.md`
**Harness:** `examples/spike_chunk_backward_gate.py`
**Hardware:** single A100-SXM4-80GB (GPU 1; ~64 GiB usable limit), `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
**Verdict: GATE = NO-GO.** Chunking the implicit-AD *backward* is numerically exact but *increases*
peak memory — it defeats XLA's existing rematerialization of the χ²·D⁶ absorb intermediate and
adds `lax.map` stacked-residual + VJP-transpose overhead. The backward must stay monolith; large-D
backward memory relief comes from GSPMD sharding (rung-2), not chunking.

## Question

Increment 1 chunks the forward χ²·D⁶ edge absorption via `lax.map` over the boundary-χ axis. Does
that compose with the implicit-AD **backward** (`_ctm_energy_ad.py`, `custom_vjp` + fixed-point
Neumann/GMRES adjoint) — i.e. when `ctm_chunk_size` is threaded into `jit_step_bwd`, does
`jax.vjp` through the chunked `lax.map` keep the adjoint's `Jᵀ` matvec memory-bounded (÷K), or does
it re-materialize / regress? The user's stated risk: "if the backward re-materializes χ²·D⁶ it
becomes the new wall."

## Architecture finding (why chunk ≠ shard for the backward)

- **device_mesh sharding reaches the backward automatically** via GSPMD propagating from the
  sharded env residuals the forward saves (rung-2, `2026-06-21-632-rung2-backward-sharding-gate`).
  Line 1020 (`jit_step_bwd = _make_jit_ctm_step(neighbors, recipe)`) was never touched for sharding.
- **Chunking cannot propagate that way.** It is a `lax.map` control structure that must be
  *explicitly present* in the differentiated `jit_step_bwd`. So the gate wiring under test was the
  minimal (and only) src change:
  ```python
  jit_step_bwd = _make_jit_ctm_step(neighbors, recipe, ctm_chunk_size=ctm_chunk_size)
  jit_step_bwd = partial(jit_step_bwd, chunk_size=ctm_chunk_size)
  ```
  (`recipe="1x1"`, dense; `None` chunk size = byte-for-byte no-op.)

## Correctness — PASS (but not the open question)

Grads are exact whether the backward is chunked or monolith, because the forward fixed point is
bit-identical (Increment 1 forward parity ≤1e-12) and the adjoint of the chunked sweep = adjoint of
the monolith sweep at that fixed point — the *same* linear operator `Jᵀ`, differing only by FP
reassociation of the `lax.map` sum.

| config | `value_and_grad` D=2 χ=6 recipe=1×1 | grad max\|Δ\| vs chunk-off |
|---|---|---|
| chunk-on forward + **chunked** backward (gate wiring) | `chunk_engaged=True` | **2.1e-25** |
| chunk-on forward + **monolith** backward (merged Inc-1) | `chunk_engaged=True` | **0.0** |

Correctness was never the risk — **peak memory** is.

## Memory — NO-GO (the decisive result)

Per-device peak of the full `value_and_grad` (forward CTM + implicit-AD backward),
`recipe="1x1"`, well-conditioned state (identical χ²·D⁶ intermediate sizes; fast clean adjoint):

| D | χ | monolith (OFF) | chunked | outcome |
|---|---|---|---|---|
| 8 | 32 | **18.37 GB** | 26.11 (K=16) / 24.77 (K=8) / 23.56 (K=1) GB | chunk **+28–42 %** |
| 10 | 16 | **28.01 GB (fits)** | **OOM (>64)** | chunk turns a fit into **OOM** |
| 10 | 24 | OOM (tried +29 GiB) | OOM (57.5 GiB pre-remat) | both fail |
| 10 | 32 | OOM (65.9→57 remat) | OOM (**79.5**→79 remat blocked) | chunk **+13.6 GiB** pre-remat |
| 12 | 32 | OOM (tried +52 GiB) | OOM (autotuner transpose `f64[144,32,144,144,32]`) | both fail |

- **Chunk-size trend (D=8 χ=32):** peak *decreases* with more chunks (26.11→24.77→23.56 as K→1) —
  the mechanism works *directionally* — but every chunked variant sits **above** the monolith.
  A large fixed overhead dominates the small chunkable gain.
- **Headline (D=10 χ=16):** monolith fits comfortably at **28 GB**; the chunked backward **OOMs**.
- **Forward-only isolation (D=8 χ=32):** OFF and chunk=8 are **identical (7.61 GB)** — the overhead
  is *purely in the backward*; the forward `lax.map` neither helps nor hurts here (the ~2 GB absorb
  intermediate is below the pipeline's ~7.6 GB waterline).

## Mechanism

The XLA rematerialization log is the smoking gun. At D=10 χ=32:
```
OFF     : ...only reduced to 56.57GiB, down from 65.90GiB originally   (remat recoups ~9 GiB ≈ one χ²D⁶ array)
chunk=8 : ...only reduced to 78.87GiB, down from 79.49GiB originally   (remat recoups ~0.6 GiB — blocked)
```
XLA **already rematerializes** the χ²·D⁶ absorb intermediate in the monolith backward (recompute
instead of store). The `lax.map` chunking:
1. **Blocks that rematerialization** — the scan structure forces stacked per-chunk residuals XLA
   cannot recompute away.
2. **Adds VJP-transpose intermediates** — the chunked reverse pass builds transposes
   (`f64[144,32,144,144,32]` = D²·χ·D²·D²·χ) that are themselves χ²·D⁶-scale and even fail XLA
   autotuning at D=12.

Net: the chunked backward is *strictly worse* than the monolith. The feared "re-materializes χ²·D⁶"
is real, but the remedy is **not** chunking — XLA's automatic activation-checkpointing already does
it better than an explicit `lax.map` can.

Also note the **irreducible ~56 GiB floor** (D=10 χ=32, same in both configs after remat): the
D≥10 single-GPU backward wall is *not* the (remat-able) absorb intermediate but the env/linearized-
sweep/adjoint structure, which chunking the absorb cannot touch.

## Decision & production state

- **Reverted the gate wiring.** Line 1020 stays `jit_step_bwd = _make_jit_ctm_step(neighbors, recipe)`
  (backward monolith). The only production change is a **comment** documenting why the backward is
  not chunked. `ctm_chunk_size` keeps its documented **forward-only** scope (commit 4ad4d91).
- Merged behavior (forward-chunk + monolith-backward) gives **exact** grads (guarded by new CI test
  `tests/test_ctm_chunk_backward_grad.py`).

## Consequences for #632

- **Increment 2 (chunked backward): NO-GO — do not build.** The `optimize_gs_ad` chunk-backward
  benchmark that depended on it is **moot**.
- **chunk∘shard multiply thesis holds for the isolated forward move only** (Increment 1 gate), *not*
  the `value_and_grad` pipeline: (a) the forward absorb is below the CTM-convergence waterline until
  very large D, and (b) the backward is XLA-remat'd / floor-bound. The **backward's large-D memory
  lever is GSPMD sharding (rung-2, merged)** — chunking is not additive there.
- Open (separate, lower priority): does the *forward*-chunk help the `optimize_gs_ad` forward at
  D≥12 where the forward absorb may exceed the convergence waterline? Not tested here; Increment 1's
  isolated-move gate overstated the pipeline benefit.

## Reproduce

```bash
# correctness (CPU): grad chunk-on vs off, recipe=1x1
JAX_PLATFORMS=cpu uv run python examples/spike_chunk_backward_gate.py --mode correctness --D 2 --chi 6 --chunk 2
# memory (single A100, one config per process for a clean peak):
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/spike_chunk_backward_gate.py --mode memory --D 10 --chi 16 --chunk 0   # OFF: 28 GB
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/spike_chunk_backward_gate.py --mode memory --D 10 --chi 16 --chunk 8   # chunk: OOM
# forward-only isolation:
... --mode memory --no-grad --D 8 --chi 32 --chunk 0   # 7.61 GB
... --mode memory --no-grad --D 8 --chi 32 --chunk 8   # 7.61 GB (identical)
```
