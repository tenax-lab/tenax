# Even-D padded-`vmap` CTM-AD port — feasibility spike design

**Date:** 2026-06-19
**Issue:** #566 (block-sparse `SymmetricTensor` AD compile + warm walls)
**Status:** DESIGN — not yet executed.
**Predecessors:** `566-padded-vmap-evenD` memory +
`examples/probe_padded_vmap_566_summary.md` (the measured even-D uniformity that
re-opened this lever); `566-cadjoint-nogo-closure` (the closure this corrects).

## 1. Why this, and why now

The #566 effort accumulated ~10 NO-GOs and a closure ("dense is pragmatic at
D≥3; every tractable JAX lever exhausted"). A follow-up probe
(`examples/probe_padded_vmap_566.py`, A100) **refuted one premise of that
closure** for the even-D fermionic case:

- The feared obstacle to the un-built `PaddedBlockArray` + `vmap`/`lax.scan`
  port — *"padding heterogeneous charge blocks to uniform reconstructs dense"* —
  is **false at even D**. The entire converged 1×1 fermionic CTM environment
  (corners χ×χ, edges χ×D²×χ) is **block-shape-uniform** at even D: χ splits
  evenly across parity sectors (χ=8 → 4+4, χ=16 → 8+8), so corners are 2 uniform
  blocks and edges 4 uniform blocks (`n_shapes = 1`). A padded-`vmap`
  representation therefore has **zero padding waste, O(1) compile in block count,
  and the full 2× Z₂ sparsity**.
- This holds only at **even D**. Odd D=3 fragments the site tensor to 16 distinct
  shapes (`padded/dense = 1.58`, worse than dense); diverse-charge symmetries
  (U(1)-Sz) fragment similarly (`566-u1sz-stacking-nogo`).
- What was actually measured-NO-GO is the **partial** realization: the batched
  contraction (#568, `TENAX_BATCH_BLOCKSPARSE`) batches the *contraction* but
  leaves `_fuse_indices_symmetric` and the Python sweep/convergence loop eager —
  the host-bound warm wall (#618, #627: warm ~0.90×, compile −21% capping).

So the one **un-refuted** lever is the **full** port: contraction + **padded
`_fuse_indices_symmetric`** + truncated SVD + fixed-point as ONE jitted graph
over padded uniform-block stacks, at even D. This spike decides GO/NO-GO on it
cheaply, highest-risk-first, without building the whole thing.

## 2. The measured wall and the prerequisite

Baselines (`examples/profile_566_a100_Dsweep.json`, A100, x64, implicit path):

| sym       | D | χ  | blk | fwd_cmp | vg_cmp     | bwd_cmp |
|-----------|---|----|-----|---------|------------|---------|
| fermionic | 2 | 8  | 16  | 63.8s   | 206.4s     | 142.6s  |
| fermionic | 3 | 12 | 16  | 527.8s  | 2111.3s    | 1583.5s |
| dense     | 3 | 12 | 1   | 27.8s   | 40.6s      | 12.8s   |

The wall is **compile**, in both directions (fwd sweep-step ~528s at D3 alone;
bwd dominates vg). The warm step is host-bound eager per-block dispatch dominated
by `_fuse_indices_symmetric` (#618).

The prerequisite the probe established: at even D the env is shape-uniform, so the
per-block loop is `vmap`-able with zero waste. The chain-breaker is the fusion:
`_fuse_indices_symmetric` (`src/tenax/algorithms/_tensor_utils.py:231`) builds
**static** charge scatter-maps (numpy, host-side) then scatters each block's data
into the fused block. At even D the uniform blocks yield **uniform scatter-maps**,
so the per-block data scatter collapses to **one `vmap`'d gather/scatter** over
the stack. Verifying that is Gate 0.

## 3. Architecture

A standalone `examples/spike_evenD_padded_vmap_566.py`. **Zero production edits**
through Gates 0–2 (the spike monkeypatches / wraps; production wiring is deferred
to a post-GO build). It introduces a **padded uniform-block stack** rep for the
1×1 fermionic CTM tensors at even D:

- Each symmetric tensor → one dense array `(n_blocks, *max_block_shape)` (at even
  D, `max_block_shape == every block_shape`, so the pad is a no-op) plus the
  existing static `(block_keys, block_shapes, indices)` metadata.
- **What already exists and is reused:** the batched contraction
  `_contract_symmetric_batched` (#568) groups same-shape combos into a batched
  einsum + `segment_sum` — at even D that is exactly one batched op; truncated SVD
  over a stack is `jnp.linalg.svd` batched over the leading axis (native).
- **What is new (the crux):** a padded/stacked-aware `fuse` that, for the uniform
  even-D stack, performs the scatter as one `vmap`'d op using the static
  scatter-map, instead of a Python loop over blocks.
- **The fixed-point:** the Python convergence loop → `lax.scan`/`while_loop` over
  the jitted sweep step (Gate 2). The backward → the existing implicit adjoint
  (`ctm_energy_implicit`'s `custom_vjp`), whose VJP through a `scan` graph is
  native JAX (Gate 3).

## 4. Gate 0 — padded fusion (the chain-breaker)

Port `_fuse_indices_symmetric` to operate on the even-D uniform stack: emit ONE
`vmap`'d scatter over `(n_blocks, *block_shape)` using the static scatter-map,
producing the fused stack.

- **Correctness:** numerically match the eager `_fuse_indices_symmetric`
  (reconstruct both to dense via `todense()` on the *small* fused tensor and
  compare) on the real fermionic env tensors at D=2 χ=8 AND D=4 χ=16, all four
  edge/corner fuse sites used in a sweep.
- **Compile-flat:** cold-`jit` the padded fuse; capture `vg`/fwd compile +
  `n_compiles` at D=2 vs D=4. The op count must be **block-count-independent**.

**GO:** max-abs dense diff **< 1e-10** AND padded-fuse compile is **flat D=2→D=4**
(ratio < 1.5×) at a few seconds. **NO-GO ⇒ stop**: the chain-breaker is unbreakable
in-graph; the whole port is dead and dense stays pragmatic. ~1–2 days.

## 5. Gate 1 — forward sweep step (compile collapse = Gate A)

Assemble ONE jitted forward CTM sweep step for 1×1 even-D fermionic from {padded
fuse (Gate 0), batched contraction (#568), batched truncated SVD/projectors},
operating end-to-end on the padded stack.

- **Gate A (compile):** cold-`jit` the sweep step; measure compile + `n_compiles`
  at D=2 χ=8 vs D=4 χ=16. Contrast against the eager-per-block baseline
  (fwd_cmp ferm D2 = 63.8s; the D3 point is 528s).
- **Warm + correctness:** one warm step matches the production eager sweep
  (dense-compare on the small output) and is timed (feeds Gate B).

**Gate A GO:** sweep-step compile **< 30s** AND **flat D=2→D=4** (ratio < 2×) —
i.e. the per-block emission is gone. **NO-GO ⇒ stop** (record: even-D uniformity
is necessary but the fused sweep still emits per-block ops). ~+2–3 days.

## 6. Gate 2 — forward energy + Gate B (warm beats dense)

Wrap the Gate-1 step in a `lax.scan`/`while_loop` fixed-point and compute the
energy (forward only, no AD yet).

- **Gate B (warm):** warm energy step vs **dense** at D=4 χ=16 (dense is the
  pragmatic baseline; runtime-bound ~χ^1.7, `570-dense-largeD-study`). The
  padded-`vmap` path does ~0.5× dense FLOPs (Z₂ sparsity); the question is whether
  it saturates the GPU enough to realize a net win.

**Gate B FULL GO:** padded-`vmap` warm step **≤ dense** warm step at D=4 (any
real win; the partial form was 0.90× = slower). **Gate B NO-GO ⇒ PARTIAL GO**:
the compile/dev-CI/cold-start win (Gate A) stands and is worth landing, but there
is no production warm speedup — stop before backward. ~+3–5 days.

## 7. Gate 3 — implicit backward (only if Gates A+B GO)

Add the existing implicit adjoint over the `scan` graph and measure full
`value_and_grad` compile (the 2111s baseline) + warm at D=4. Validate the gradient
against production `grad(ctm_energy_implicit)` at D=2 χ=8 (affordable reference),
max-abs < 1e-6.

**GO:** vg compile collapses (flat D2→D4, ≪ 2111s) AND gradient matches. A
double-GO here ⇒ open a production-integration design (a new even-D `adjoint_method`
/ path). ~+1 week.

## 8. Scope, deferred work, honest value proposition

- **In scope:** 1×1 cell, fermionic (the validated AD path, and the symmetry that
  is shape-uniform at even D), even D ∈ {2, 4} with a D=6/8 projection.
- **Out of scope:** odd D (converges to dense — measured), U(1)-Sz (fragments),
  multisite cells, production integration, and the warm-step host round-trip past
  what Gate B measures.
- **Honest value:** Gate A alone (compile collapse) helps dev iteration / CI /
  cold-start (the persistent cache only saves unchanged code). Gate B is the
  production-speedup bet — and it is where the partial form already failed (0.90×).
  The staged structure books the cheaper, more-likely win (compile) even if the
  warm bet doesn't land.

## 9. Risks

- **Gate 0 is the real risk.** If the padded scatter can't be expressed as a
  single in-graph `vmap`'d op (e.g. the scatter-map structure forces per-block
  indexing), the chain breaks and the port dies. Highest-risk-first by design.
- **Balanced-sector assumption.** Even-D uniformity assumes the SVD keeps χ/2 per
  parity sector. A polarized state → ≤2 distinct env shapes → modest bounded waste
  (env block counts 2–4). The spike uses the physical fermionic ground-state init,
  where the probe measured balance; note any drift if Gate-1 sweeps polarize χ.
- **Gate B saturation.** 0.5× FLOPs ≠ 0.5× wall if the padded-`vmap` ops are too
  small to saturate the A100 (the #627 host-bound regime). Gate B measures this
  directly; PARTIAL GO is the honest fallback.
- **Float reorder** vmap-vs-eager ~1e-12, ≪ the 1e-10 / 1e-6 gates.

## 10. Cost and outcome

- **Gate 0:** ~1–2 days, A100. Cheapest kill.
- **Gates 1–2:** ~+1 week cumulative.
- **Gate 3:** ~+1 week, only on Gate A+B GO.
- Decisive staged GO/NO-GO; each gate is a recordable result on its own.

A100 env per project notes; harness reuse: `examples/profile_ctm_ad_wall_566.py`
(`make_site_and_gate`, `_install_compile_capture`, `_cold`),
`examples/probe_padded_vmap_566.py` (block-stat + env-converge helpers),
`src/tenax/algorithms/_tensor_utils.py:231` (`_fuse_indices_symmetric`),
`src/tenax/contraction/contractor.py:241` (`_contract_symmetric_batched`).
