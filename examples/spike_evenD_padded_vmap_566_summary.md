# Even-D padded-`vmap` CTM-AD port — spike findings

**Platform:** NVIDIA A100-SXM4-80GB · x64 · 2026-06-20
**Design:** `docs/superpowers/specs/2026-06-19-566-evenD-padded-vmap-ctm-port-spike-design.md`
**Plan:** `docs/superpowers/plans/2026-06-19-566-evenD-padded-vmap-ctm-port-spike.md`
**Code:** `examples/spike_evenD_padded_vmap_566.py`

## Question

Can a fully-jitted even-D fermionic CTM sweep over padded uniform-block stacks
collapse the #566 compile wall? The probe (`probe_padded_vmap_566_summary.md`)
showed the even-D fermionic environment is block-shape-uniform, so a padded
representation has zero padding waste — the structural prerequisite. The spike
gates whether that actually collapses the compile, highest-risk-first.

## Gate 0 — padded global-scatter fuse: **GO**

The fusion "chain-breaker" IS expressible as O(1) ops. `padded_fuse` reproduces
the eager `_fuse_indices_symmetric` **exactly** (max-abs 0.0, fermionic + U(1)-Sz,
D=2/3/4) via one gather + one scatter over the flat buffer with statically
precomputed indices.

- **jaxpr op-count collapse (decisive):** fermionic site **165 → 17 eqns (9.7×)**
  at every D; padded is CONSTANT at 17 regardless of block count/symmetry, while
  eager scales (u1sz 89→325 for 8→32 blocks).
- **Verified safe drop-in in the sweep:** same-env 4-config check — padded/BATCH-off
  = 0.0 exact, padded/BATCH-on = 2.2e-14 (float reorder).
- Single-fuse compile wall-time is ~0.1s (XLA's overhead floor) — too small to
  reveal the wall; the op-count is the real signal.

## Gate 1 / Gate A — does the sweep-step compile collapse? **NO-GO**

Measured ONE production fermionic `_ctm_tensor_sweep_multisite` step, baseline vs
levered (padded_fuse + `TENAX_BATCH_BLOCKSPARSE=1`), attributing the jaxpr by
primitive (A100, recipe=2x2):

| config | D | compile_s | total_eqns | dominant primitives |
|--------|---|-----------|-----------|---------------------|
| baseline | 2 | 5.89 | 9680 | reshape 2584, slice 1236, transpose 864, dot_general 848 |
| levered  | 2 | 4.71 | 9384 | reshape 2504, slice 1860, broadcast 980, … gather 224 |
| baseline | 4 | 9.35 | **9680** | (identical eqn count to D=2) |
| levered  | 4 | 6.01 | 9384 | |

**Why NO-GO — the fuse was the wrong lever for the *compile* wall:**

1. **The fuse is only ~3% of a sweep step.** padded_fuse removes 9680→9384 eqns.
   The mass is `reshape/slice/transpose/dot_general` — the block-sparse
   contraction/plumbing, not the fuse.
2. **Compile is array-size-bound, not op-count-bound** (within fermionic): eqn
   count is IDENTICAL at D=2 and D=4 (9680, block count fixed) yet compile grows
   5.89s → 9.35s.
3. **The tractable levers cap.** padded_fuse + batched contraction give only
   0.80×/0.64× sweep compile — echoing the #627 "batched caps" prior, not a
   collapse.

**The deep reason dense stays pragmatic:** even a *full* padded-`vmap` rewrite
(batching reshape/slice/transpose/contraction too) is ceilinged at **dense-like**
compile, because the residual scaling is array-size codegen — which dense has too.
So the full port is a massive rewrite to merely *match* dense, not beat it.

## Salvage lead (untested): the WARM wall

`padded_fuse` is correct, O(1)-op, and a verified drop-in. It does NOT help
*compile*, but the fermionic **warm** forward is host-bound and *fuse-dominated*
(#618) — so padded_fuse is a plausible lever for fermionic **warm runtime** at
D≥3. This spike did not test it (Gate A was compile). It is the one place the
spike's `padded_fuse` might still pay off in tenax.

## Conclusion

The even-D padded-`vmap` thesis is **holed for the compile wall**. The even-D
uniformity (real) and the fuse op-count collapse (real, Gate 0) do not collapse
the sweep compile, which is dominated by broad block-sparse plumbing + array-size
codegen. This converges with [[566-cadjoint-nogo-closure]] with a sharper
mechanism: the fermionic compile wall is NOT fuse-op-count-bound.

**Recommendation:** dense stays pragmatic for bosonic D≥3; for *fermionic* (which
has no dense fallback — the graded path is mandatory), tenax/JAX is the small-D
tool (≤3–4), and **large-D fermionic belongs in an eager framework (YASTN fPEPS /
peps-torch)** with no compile wall. See [[fermionic-large-d-tooling]].
