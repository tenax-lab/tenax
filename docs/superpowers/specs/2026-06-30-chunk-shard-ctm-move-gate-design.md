# Design: chunk × shard the dense-CTM move — feasibility gate

**Date:** 2026-06-30
**Status:** Approved (brainstorming) — gate-first; build deferred behind the GO.
**Parents (both GO, neither productized — `grep lax.map src/` is empty):**
- chunked CTM move (single-GPU): `docs/superpowers/handoffs/2026-06-22-chunked-ctm-move-production-spike-findings.md`
  — streaming the real left-move edge contraction `T4_a = einsum('ijk,lmjn->iklmn', T4, a)`
  (`(χ,χ,D²,D²,D²)=χ²·D⁶` peak) over the boundary-χ axis: parity 2e-18, peak ÷ ~chunk-count,
  D10-OOM→D12-runs, **sub-quadratic in χ**, 1.4× warm.
- GSPMD sharding (multi-GPU): `ctm_sharding.py` + `632-multigpu-dense-ctm-measured` — shard the D²
  axis, ~2× per-device relief, capped ~N^(1/6), throughput unchanged.
**Motivation (why now):** the rSVD line was killed by the attribution gate
(`632-rung3-rsvd-projector`): the split/1×1 per-device **peak is the `χ²·D⁶`/`χ³D³` absorption
contraction**, not any SVD (SVDs are χD = 100s× smaller). Chunked-einsum attacks *exactly* that
contraction; sharding attacks it too. Composing them is the best-targeted lever for the confirmed
wall — and is **untested**: each was measured alone.

## The one question this gate answers

On the real CTM edge contraction, does composing **chunking** (stream the free boundary-χ axis on one
device) with **GSPMD sharding** (split a free D² axis across N devices) (a) cut per-device peak
**multiplicatively (÷N·K)**, and (b) let the **sharded** contraction dodge the monolithic giant-gemm
autotuner wall — so multi-GPU reach exceeds *either* lever alone and the relief beats the weak ~2×?
GO/NO-GO before wiring an `n_chunks` knob into the production moves.

## Why this is the binding uncertainty (and why it might fail)

Chunk axis (`i=t4_d`, χ) and shard axis (`a`'s `r2=n`, a *free* D² output leg = `ctm_sharding`'s
left-move surviving axis) are **orthogonal** — so on paper peak ÷ (N·K). Load-bearing risks, none
safely inferable:
1. **XLA may de-shard inside `lax.map`.** `lax.map` lowers to `scan`; the partitioner could gather
   the sharded `a` per scan step (→ peak ÷ K only, no ÷ N). Must measure the per-chunk layout.
2. **Sharding may not dodge the autotuner.** Each device's shard of a monolithic gemm is still a
   giant gemm; if `sharded`-alone autotuner-FAILs where `chunk×shard` (small per-chunk gemms) runs,
   that's the payoff. If both fail (or both run), chunking adds nothing to sharding.
3. **Fuse-of-sharded-axis.** `n` must stay un-fused through `reshape(D2, χ·D2, D2)` and the projector
   so its sharding survives — chosen precisely because `n` is the standalone last axis (`fr=(k,m)` is
   the fused one, `n` is not). Verify the constraint isn't elided.

## Experiment (extend the existing faithful spike — cheapest real test)

`examples/spike_chunk_shard_ctm_move.py` = `spike_chunked_ctm_move.py` + a mesh. `a` committed
`NamedSharding(mesh, P(None, None, None, "d"))` (shard `r2=n`); `T4`/projectors replicated (random —
the projector **SVD is not exercised**, isolating the contraction). Four variants, **one per process**
(`peak_bytes_in_use` is cumulative): `full` (replicated monolith) · `chunked` (1-GPU, ÷K) · `sharded`
(monolith, `a` sharded, ÷N) · `chunkshard` (chunked + `a` sharded, ÷N·K). CLI `--D --chi --batch
--mesh-n --variant`. Mesh = devices 0,2 (clean A100s); `D²` divisible by N (D=8→64/2).

## Measurements (2× A100, f64, XLA_PYTHON_CLIENT_PREALLOCATE=false)

- **G1 — composition.** Per-device `peak_bytes_in_use`, all four variants, fixed (D,χ,K,N). GO needs
  `chunkshard ≈ chunked / N` (within ~1.5×) and `≈ full/(N·K)` — i.e. the two reliefs multiply.
- **G2 — reach.** Largest D (and largest χ at fixed batch) that RUNS per variant. GO needs
  `chunkshard` to run where **both** `chunked` and `sharded` fail (OOM or autotuner-FAIL).
- **G3 — layout + autotuner-dodge (the deep question).** Optimized-HLO of `sharded` vs `chunkshard`:
  (a) the per-chunk `χ·D⁶` intermediate stays **sharded** (no all-gather of the full `n=D²` axis);
  (b) does `sharded` hit autotuner-FAIL / replication at a (D,χ) where `chunkshard` succeeds? A yes to
  (b) is the strong result: chunking **unblocks** sharding rather than merely adding to it.
- **G4 — parity.** `chunkshard` vs `full`: `max|Δ|/‖full‖ ≤ 1e-14` (forward; backward deferred —
  spike is the forward edge path with random projectors).

## GO / NO-GO

- **GO** ⟺ G1 composes (÷N·K within ~1.5×) **and** G2 reach extends past both single levers **and**
  G4 parity exact. **Strong GO** additionally if G3(b) shows chunking dodges the autotuner wall.
  → build: opt-in `n_chunks`/`batch` knob on the four `_compiled_move_*` edge/corner contractions,
  composed with the existing `ctm_sharding` mesh; D=10–12 / large-χ multi-GPU benchmark.
- **NO-GO** ⟺ peak doesn't drop ÷N under chunking (XLA de-shards inside `lax.map`, G1), or no reach
  gain over chunk-alone (G2), or parity breaks (G4). → document; chunking stays a single-GPU lever,
  multi-GPU stays the weak ~2× GSPMD; large-D stays eager/YASTN per `fermionic-large-d-tooling`.

## Honest ceiling (set expectations)

A **"fit a bigger calculation"** lever: extends the reachable (D,χ) by ~N·K in memory at ~1.4× warm ×
sharding overhead. It does **not** change throughput or the large-D runtime verdict. Its upside over
the dead rSVD line is that it hits the *confirmed* wall (the contraction) and — if G3(b) holds — could
be the first thing to push multi-GPU relief past ~2×.

## Components

- **Create** `examples/spike_chunk_shard_ctm_move.py` — the four-variant mesh harness above
  (per-device peak + HLO gather/shard check + parity), CLI `--D --chi --batch --mesh-n --variant`.
  Throwaway/spike. (Derived from `spike_chunked_ctm_move.py`; keep that one untouched.)
- **Create** `docs/superpowers/handoffs/2026-06-30-chunk-shard-ctm-move-findings.md` — measured G1–G4
  + GO/NO-GO.
- **No `src/` change in the gate** (the spike is standalone raw-array). The production move-knob
  wiring is build-time, behind the GO.

## Out of scope (deferred to the build, only if GO)

- Wiring `n_chunks` into the four production `_compiled_move_*` (and the corner accumulation) + the
  sweep loop; the `optimize_gs_ad`/config surface.
- All four moves (gate does the left-move edge path only, per the parent spike).
- Backward/AD through the chunked-sharded move (forward-only gate; random projectors).
- The projector-SVD wall and the unresolved **1×1 projector size** conflict (handoff claims `(χD²)²`;
  the split corner measured `χ×χ`) — a separate lever, isolated out by using random projectors here.
