# Design: GSPMD-sharded dense CTM for large-D iPEPS (rung 1)

**Date:** 2026-06-19
**Status:** Approved (brainstorming), pending implementation plan.
**Context:** Symmetric/block-sparse CTM is a confirmed no-go at large D (host-bound
per-block eager dispatch; #566/#570/#618). This design pursues large D on the
**dense** path by spreading memory across multiple GPUs.

## Goal

Make the **dense** iPEPS CTM **fit and run at D=6–8, χ≈16–24 on a single node with
2–4 NVLink GPUs**, where it currently OOMs on one GPU. The binding constraint is
**memory** (throughput is secondary). Rung 1 covers the **forward CTM convergence
+ energy evaluation**; AD backward and larger D are explicit follow-ups.

## Decomposition (multi-rung project; this spec is rung 1 only)

1. **Rung 1 (this spec):** GSPMD-shard the dense forward CTM + energy so a D=6–8
   case that OOMs on 1 GPU fits on 2–4 GPUs.
2. **Rung 2 (follow-up):** shard the AD backward (implicit Krylov adjoint) so the
   full `optimize_gs_ad` fits.
3. **Rung 3 (later, D≥10):** a genuinely distributed truncated SVD, only once the
   `(χD² × χD²)` SVD matrix itself becomes large.

## Key sizing insight (why this is tractable)

At D=6–8 / χ≈24, the **projector SVD matrix** `(χD² × χD²) ≈ 1536² ≈ 38 MB` is
*small* and is **not** the OOM source. The memory hogs are the **contraction
intermediates** — the double-layer tensor `a` (`D⁸` elements) and the
enlarged-corner / absorption intermediates carrying `D²` legs. Therefore rung 1
**shards the contractions and replicates the small SVD**, avoiding any distributed
linear algebra.

## Tensor layout (verified against current code)

- Corner `C`: `(χ, χ)` — `_ctm_tensor_init.py` `CTMTensorEnv` (`.C1`).
- Edge `T1..T4`: `(χ, D², χ)` — 3-leg, the middle `D²` leg is the fused
  double-layer virtual bond (`_ctm_tensor_init.py:42`).
- Double-layer `a`: `(D², D², D², D²)` — `_build_double_layer_tensor`
  (`_ctm_tensor_init.py:84`); the `D⁸` object.

## Approach (chosen: A — GSPMD input-sharding)

Rejected alternatives:
- **B — explicit `shard_map` + hand-written collectives / blocked SVD:** full
  control but a large rewrite of the CTM step; overkill while the SVD is small.
- **C — single-GPU algorithmic memory reduction:** limited headroom, won't reach
  D=8 χ=24, doesn't scale with device count. (A complementary lever, out of scope.)

## Architecture

A **1-D JAX device mesh** (`axis="d"`, size = number of GPUs) over the node. The
dense CTM step is left **algorithmically unchanged**; a sharding layer commits the
env tensors and double-layer tensor to `NamedSharding`s, and the CTM step + energy
fn are `jax.jit`-wrapped with explicit `in_shardings`/`out_shardings`. XLA-GSPMD
partitions every contraction across devices and inserts NVLink collectives
automatically; per-device peak memory drops ≈1/N for the sharded objects. The
projector SVD is auto-replicated (XLA all-gathers a sharded matrix before an
un-partitionable SVD), which is cheap at this size. No change to physics,
projector math, or convergence logic.

## Sharding scheme (the load-bearing choice)

Shard the **virtual `D²` axis** (not χ: χ≈16–24 is small and divides poorly across
4 devices; `D²` = 36 at D=6 / 64 at D=8 divides cleanly by 2 and 4):

| Tensor | Shape | Sharding | Per-device |
|---|---|---|---|
| double-layer `a` | `(D², D², D², D²)` | a **per-move surviving** `D²` axis over `d` (see resolution below) | `D⁸/N` (primary win) |
| edge `T` | `(χ, D², χ)` | the `D²` axis over `d` | `χ²D²/N` |
| corner `C` | `(χ, χ)` | replicated (tiny ~χ²) | full (cheap) |
| projector SVD matrix | `(χD², χD²)` | auto-replicated at the SVD | ~38 MB (cheap) |

GSPMD propagates these shardings through the enlarged-corner and absorption
einsums, so the large intermediates (all carrying `D²` legs) are partitioned. For
the double-layer `a`, one explicit `with_sharding_constraint` per move pins it to a
leg that survives that move (the env edges/corners need no per-move annotation).
All-gathers around the SVD ride NVLink — acceptable because the goal is memory, not
speed.

**Risk / empirical tuning point — RESOLVED 2026-06-19 by the Task-3 spike.** The
exact axis choice was empirically validated on the four real edge-grow einsums
(`_ctm_compiled_moves.py`) under 4 fake CPU devices. Finding: the double-layer `a`
has four `D²` legs `(u2, d2, l2, r2)` and each directional move contracts a
*different* one of them, so **no single fixed sharded axis survives all four
moves** — the move that contracts the sharded leg all-reduces, returning that
move's dominant `χ²·D⁶` intermediate fully **replicated** (full per-device size).
The empty-intersection of the four survivor sets makes this unavoidable for any
static single-axis scheme. Measured worst-case per-device fraction across the four
moves (N=4):

| Scheme | Worst-case per-device fraction | Ever replicates? |
|---|---|---|
| single fixed `D²` axis (1-D mesh) | `1.0` (one move replicates) | yes |
| two `D²` legs on a 2-D `(2,2)` mesh | `0.5` (any leg pair, identical) | no |
| **per-move resharding (1-D mesh, size N)** | **`1/N`** on all four moves | no |

**Decision: per-move resharding.** Before each directional move the double-layer
tensor is constrained (`jax.lax.with_sharding_constraint`) to shard a `D²` leg that
*survives* that move, so the dominant `χ²·D⁶` intermediate stays `≈1/N` on every
move. Cost: one all-to-all reshard of the small `a` (`D⁸`, ~134 MB at D=8) per move
over NVLink — acceptable because the goal is memory, not speed. The static 2-D-mesh
two-leg scheme (guaranteed ≥2× reduction, no resharding) is the simpler partial-win
alternative but does **not** reach `1/N`; it is not adopted in rung 1.

## Components

- **`src/tenax/algorithms/ctm_sharding.py`** (new, small, single responsibility —
  "given a mesh, produce the sharding for each CTM tensor"):
  - `build_ctm_mesh(devices=None) -> Mesh` — 1-D mesh over `jax.devices()`.
  - `double_layer_sharding(mesh)`, `edge_sharding(mesh)`, `corner_sharding(mesh)`
    returning `NamedSharding`s.
  - `shard_env(env, mesh)` / `shard_double_layer(a, mesh)` — `jax.device_put`
    helpers committing tensors to those shardings.
- **Sharded env init:** after `_ctm_tensor_init` builds the initial corners/edges
  and `a`, commit them to their shardings when a mesh is configured.
- **JIT wrapper:** wrap `_jit_ctm_step` (forward) and the energy fn with
  `in_shardings`/`out_shardings` so envs remain sharded across Python CTM-loop
  iterations. The step body is unchanged.
- **Opt-in surface:** rung 1 exposes the opt-in **only** as a direct
  `python_loop_ctm_converge(device_mesh=...)` keyword argument. Default `None` →
  today's single-device path, bit-for-bit unchanged; when a mesh is passed, init +
  step route through the sharding layer. The `CTMConfig` / `iPEPSConfig`
  config-level surface is deferred to rung 2, when the full `optimize_gs_ad`
  forward+backward path is sharded — wiring a config field in rung 1 would shard
  only the forward and leave the AD backward to OOM.

## Data flow (one CTM sweep)

sharded envs + sharded `a` → per move, constrain `a` to a surviving-leg sharding →
enlarged-corner / edge-grow contraction (GSPMD-partitioned, dominant `χ²·D⁶`
intermediate sharded ≈1/N) → form projector matrix → all-gather → replicated
truncated SVD on each device → projectors → re-shard → corner/edge absorption
(partitioned) → sharded envs out. The Python loop (`python_loop_ctm_converge`) is
untouched; it carries sharded arrays. The `_max_eps`/`_max_S` convergence scalars
(`_ctm_loop_core.py:187-188`) are tiny replicated scalars — unaffected.

## AD interaction (forward only this rung)

Rung 1 ships and verifies the **forward CTM + energy** sharded. The implicit-AD
path is left intact and **must not be precluded**: the jit'd backward fns inherit
input shardings, so rung 2 (sharding the Krylov adjoint) is a natural follow-on.
No backward sharding work in rung 1; the only requirement is that rung-1 code
introduces no single-device hard assumptions that would block rung 2.

## Testing & success criteria

1. **Correctness (CI-able on CPU, no real GPUs).** Using
   `XLA_FLAGS=--xla_force_host_platform_device_count=N` to expose N fake devices,
   the sharded forward CTM + energy must match the single-device result to ~1e-10
   at D=2–3, χ=8 (deterministic). This is the load-bearing test; marker-gated so
   CI runs it without GPUs.
2. **No-regression.** With the flag off, the path is identical to today
   (bit-for-bit), guarded by an existing dense-CTM test.
3. **Memory feasibility (GPU, throwaway benchmark — the headline deliverable).**
   Measure the single-GPU OOM ceiling (the D where 1 GPU dies), then show the same
   D fits on 2–4 GPUs with per-device peak ≈1/N. Throughput recorded but not a
   gate.

## Out of scope / follow-ups

- AD backward sharding (rung 2).
- Distributed truncated SVD for D≥10 (rung 3); the planned GKL bidiagonalization
  truncated SVD is the natural vehicle.
- Multisite 2×2 plaquette path (its `χ⁴D⁸` intermediate is a separate, harder
  case).
- Multi-node / inter-node sharding; TPU.
- Throughput optimization (comms minimization, sharding-axis auto-tuning).
