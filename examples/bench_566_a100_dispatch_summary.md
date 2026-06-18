# #566 batched-dispatch D-sweep — A100 record (2026-06-18)

Does the live batched-dispatch gate `TENAX_BATCH_BLOCKSPARSE` (off→on) speed up a
real symmetric/fermionic iPEPS CTM-AD step, and does it cross over to a *win* at
large bond dimension D the way YASTN's abelian-symmetric iPEPS does at D≈8?

Harness: `examples/bench_symmetric_ad_batching_566.py` — production
`jax.value_and_grad` path on a **FermionParity** 16-block iPEPS site tensor
(`_build_initial_fpeps_tensor`), default `ad_backward_method="vjp"` (iterative
Neumann backward), x64, on one NVIDIA A100-SXM4-80GB. χ=3D unless noted.

## Result — decisive NO-GO, no crossover

| D | χ | blocks | compile off→on | warm step off→on | warm ratio | notes |
|---|----|--------|----------------|------------------|-----------|-------|
| 2 | 6  | 16 | 159→219s (0.73×) | 60.0→67.0s | **0.90×** | clean (run alone) |
| 3 | 9  | 16 | 1173→2053s (0.57×) | 1433→2535s | 0.57× | **contaminated** — seed-dependent slow Neumann backward (see below) |
| 4 | 12 | 16 | 168→227s (0.74×) | 62.8→70.4s | **0.89×** | clean |
| 6 | 18 | 16 | 188→272s (0.69×) | 72.1→79.2s | **0.91×** | clean |
| 8 | 16 | 16 | 209→278s (0.75×) | 77.9→84.7s | **0.92×** | clean; χ=2D because χ=3D=24 OOMs 80 GB |

`speedup > 1.0` ⇒ batching faster. **It is < 1.0 at every D, on both axes.**

Raw JSON: `bench_566_a100_stackedfwd.json` (D=2), `bench_566_a100_D{3,4,6}.json`,
`bench_566_a100_D8chi16.json`.

## Reading

* **No crossover through D=8.** The warm off/on ratio is flat at ~0.90× — batched
  dispatch is consistently ~8–11% *slower*, never faster, and trends nowhere near a
  win. Compile is also always slower under batching (0.69–0.75×).
* **The warm step is host-orchestration-bound, not kernel-bound.** During warm
  steps the **GPU sits at 0% util**; warm time grows only 60→78s from D=2→8 while
  the block count is *fixed* at 16 (FermionParity is structural — only block *size*
  grows with D). The cost is the eager CTM-convergence Python loop + eager
  `_fuse_indices_symmetric` + the Neumann vjp backward Python loop. Batching the
  *contraction* collapses only the in-jit contraction primitives, adding
  stack/segment overhead with nothing on the GPU to offset.
* **Why this differs from YASTN.** YASTN's per-block loop *is* the cheap C-level
  (NumPy/PyTorch) inner loop, so collapsing it into batched kernels pays by D=8.
  Here the equivalent cost is host Python orchestration, which batched contraction
  kernels don't touch. The eager↔batched dial is the wrong axis; the lever is the
  *cost of one block's unit of work*.
* **Memory wall.** D=8 at χ=3D=24 OOMs on 80 GB with the Neumann backward (2138
  allocator-retry dumps); only χ=2D=16 fits. Echoes YASTN needing its ~30× memory
  gain to reach large D.

## Caveats

* **D=3 row is contaminated and kept only for the record.** Both seed 42 and seed
  123 produced ~24-min warm steps at D=3 — a seed-dependent *slow-converging
  Neumann backward* (data-dependent iteration count; the harness disables the
  Arnoldi precheck). This is a backward-iteration-count artifact, **not** a
  dispatch property. The true D=3 warm step is bracketed by D=2/D=4 (~60–63s); the
  ratio direction (batching slower) still holds.
* This is the **deprecated** `ctm_tensor_converge` + vjp-Neumann path. The modern
  `ctm_energy_implicit` (CTM fixed-point implicit differentiation) + env warm-start
  is the faster production path; this benchmark isolates the *dispatch* question on
  the multi-block path that runs end-to-end.

## Bottom line

Batched dispatch never wins the warm step on this path; dense stays pragmatic for
D≥3. The only lever that addresses the measured (host-orchestration) wall is making
the symmetric CTM sweep fully jittable as one graph — the `PaddedBlockArray` +
`lax.scan` approach already proven in `_jit_sweep.py` for DMRG, ported to CTM.
