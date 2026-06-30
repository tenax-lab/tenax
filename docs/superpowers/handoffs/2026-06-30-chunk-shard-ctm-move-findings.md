# Chunk × Shard dense-CTM move — gate findings

**Date:** 2026-06-30
**Hardware:** 2× A100-80GB (devices 0,2), f64, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, one variant/process.
**Spec:** `docs/superpowers/specs/2026-06-30-chunk-shard-ctm-move-gate-design.md`
**Plan:** `docs/superpowers/plans/2026-06-30-chunk-shard-ctm-move-gate.md`
**Harness:** `examples/spike_chunk_shard_ctm_move.py` (arrays passed as jit args → no constant-folding;
faithful runtime peak).
**Verdict: GATE = STRONG GO.** Chunking and GSPMD sharding compose multiplicatively on the real edge
contraction; sharding is *more* effective on the chunked path than the monolith; and chunking
**dodges the autotuner wall** that blocks monolithic sharding — extending reach past either lever alone.

## G4 — parity (correctness)
`chunkshard` vs `full`, D=6 χ=16 batch=8, N=2: **rel = 1.1e-15** (≤ 1e-14). The sharded+chunked move
is bit-faithful to the real move.

## G1 — composition (D=8 χ=48 batch=12, N=2, K=4)

| variant | per-device peak (GB) | vs full |
|---|---:|---|
| full | 10.10 | — |
| chunked | 3.78 | ÷2.67 |
| sharded | 7.54 | **÷1.34** |
| chunkshard | 2.10 | ÷4.8 |

- The two reliefs **multiply**: full→chunked ÷2.67, chunked→chunkshard ÷1.80 → ÷4.8 total.
  `chunkshard` ≈ `chunked`/N within 1.11× (2.10 vs 3.78/2=1.89); ≈ `full`/(N·K) within 1.67×.
- **Key signal:** sharding the **monolith** gives only **÷1.34** (reproduces the weak ~2× #632 wall),
  but sharding the **chunked** move gives **÷1.80 — near-ideal ÷2**. Chunking makes sharding more
  efficient. **G1 = GO.**

## G2 — reach (χ=64 batch=16, N=2)

| D | chunked | sharded | chunkshard |
|---|---|---|---|
| 10 | 25.41 GB | **OOM (30.5 GB alloc)** | 13.52 GB |
| 12 | **FAIL (OOM)** | **FAIL (autotuner)** | **41.90 GB ✅** |
| 14 | FAIL | FAIL | FAIL |

At **D=12, chunkshard runs (41.9 GB) where BOTH chunked-alone and sharded-alone fail.** Reach extends
past either single lever. **G2 = GO.** (D=14 needs more chunks/devices — orthogonal.)

## G3 — layout + autotuner-dodge (the deep question)

- **G3a (layout):** chunkshard, D=8 χ=48 → `all-gather=0 (touching full n=64: 0)`. The sharded `n=D²`
  axis stays sharded inside `lax.map` — no de-shard. **GO.**
- **G3b (autotuner-dodge):** D=12 χ=64 → `sharded` **HLO FAILED — Autotuning failed for
  `%loop_transpose_fusion = f64[64,144,64,144,72]`** (the monolithic giant op), while `chunkshard`
  compiles cleanly: `all-gather=0 (full n=144: 0) temp=24.6 GB`. **Chunking unblocks sharding.
  STRONG GO.**

## Verdict & recommendation

**GATE = STRONG GO** — all four gates pass, plus the autotuner-dodge. This is the best-targeted
multi-GPU dense-CTM lever to date: it attacks the *confirmed* wall (the `χ²·D⁶`/`χ³D³` absorption
contraction — see `632-rung3-rsvd-projector` attribution gate, which killed rSVD precisely because
the SVDs are not the wall), the chunk and shard reliefs compose, and chunking removes the monolithic
giant-gemm autotuner barrier that capped GSPMD sharding at ~1.34×.

**Honest bounds:** still a "fit a bigger calculation" lever, not a throughput change. The per-device
relief is ÷(N·K)-ish (here ÷4.8 at N=2 K=4), so reach grows ~linearly in N·K but each is sub-ideal
(chunked ÷2.67 not ÷4, sharding ÷1.80 not ÷2). The large-D *runtime* verdict (eager/YASTN per
`fermionic-large-d-tooling`) is unchanged. This is forward-only, left-move edge path, random
projectors (the projector SVD — separate, small — is not exercised).

**Build (recommended):** add an opt-in `n_chunks`/`batch` knob to the four `_compiled_move_*`
edge/corner contractions, composed with the `ctm_sharding` mesh; AD/backward through the
chunked-sharded move; then a D=10–12 / large-χ multi-GPU `optimize_gs_ad` benchmark vs shard-only.

## Artifacts (branch `feat/632-chunk-shard-ctm-move`)
- `examples/spike_chunk_shard_ctm_move.py` — four-variant harness (`--variant full|chunked|sharded|
  chunkshard|parity`, `--D --chi --batch --mesh-n`, `--hlo`). One variant/process.
