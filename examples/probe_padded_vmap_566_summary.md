# #566 padded-`vmap` probe — findings

**Platform:** NVIDIA A100-SXM4-80GB · x64 · 2026-06-19
**Script:** `examples/probe_padded_vmap_566.py`

## Question

Can the block-sparse path be factored into a JAX-native form that gives **O(1)
compile in charge-block count** *without* converging to dense cost? The candidate
is the one un-built lever in the (superseded) #566 plan: `PaddedBlockArray` +
`vmap`/`lax.scan` over blocks — make every per-block op ONE batched op (O(1)
compile) by padding blocks to a common shape. The feared obstacle: padding
heterogeneous charge blocks to uniform fills the forbidden sectors back in with
zeros, so padded-block-sparse ≈ dense (the "dense is pragmatic" intuition).

We measure the two orthogonal axes the obstacle splits into.

## (ii) Padding waste — does single-stack `vmap` (K=1 op) beat dense?

For the real fermionic CTM tensors. `n_shapes` = #distinct block shapes (= op
count of the **batched** path #568, zero padding waste). `padded` = `n_blocks ×
max_block_vol` (single-stack `vmap`: K=1 op, O(1) compile, pads all to max).

| tensor | D | n_blocks | n_shapes | sparse/dense | pad/sparse | **pad/dense** |
|--------|---|---------|----------|--------------|------------|---------------|
| site_A | 2 | 16 | **1**  | 0.500 | 1.00 | **0.500 WIN** |
| dbl_ac | 2 | 8  | **1**  | 0.500 | 1.00 | **0.500 WIN** |
| site_A | 3 | 16 | **16** | 0.500 | 3.16 | **1.580 ~dense+** |
| dbl_ac | 3 | 8  | **8**  | 0.500 | 1.52 | 0.762 |
| site_A | 4 | 16 | **1**  | 0.500 | 1.00 | **0.500 WIN** |
| dbl_ac | 4 | 8  | **1**  | 0.500 | 1.00 | **0.500 WIN** |

The driver is **block-shape heterogeneity**, which for fermionic (Z₂) is
**D-parity-dependent**: at **even D** every leg's even/odd sub-dims are balanced
(D/2 each) ⇒ all blocks share one shape ⇒ zero padding waste; at **odd D=3** the
2/1 split makes every block a distinct shape ⇒ padding the site tensor to a single
stack *exceeds* dense (1.58×).

## (ii-env) Converged environment — the dominant cost

The #566 compile wall is dominated by the **environment** tensors at χ, not the
site/double-layer. Converged 1×1 fermionic CTM, even D:

| env | D | χ | n_blocks | n_shapes | pad/sparse | **pad/dense** |
|-----|---|---|---------|----------|-----------|---------------|
| C1, C2 (corner χ×χ)    | 2 | 8  | 2 | **1** | 1.00 | **0.500 WIN** |
| T1, T2 (edge χ×D²×χ)   | 2 | 8  | 4 | **1** | 1.00 | **0.500 WIN** |
| C1, C2                 | 4 | 16 | 2 | **1** | 1.00 | **0.500 WIN** |
| T1, T2                 | 4 | 16 | 4 | **1** | 1.00 | **0.500 WIN** |

**The full environment is shape-uniform at even D.** χ splits evenly across
parity sectors (8→4+4, 16→8+8), so corners are 2 uniform χ/2×χ/2 blocks and edges
4 uniform blocks. A padded-`vmap`/`scan` representation therefore has **zero
padding waste, O(1) compile in block count, and keeps the full 2× Z₂ sparsity** —
the "converges to dense" obstacle is **refuted for even-D fermionic**.

*Caveat:* this assumes balanced parity-sector occupation (the SVD keeps χ/2 per
sector), which held here. A polarized state could allocate χ unevenly → ≤2 distinct
shapes → modest waste; but env block counts are tiny (2–4), so the waste is bounded
(≪ the odd-D site fragmentation).

## (i) Compile flatness — mechanism check

Per-block matmul on `dbl_ac`, cold `jit`, unrolled (K=n_blocks ops) vs `vmap`
single-stack (K=1 op):

| D | n_blocks | unrolled compile | vmap compile |
|---|---------|------------------|--------------|
| 2 | 8 | 0.169s | 0.168s |
| 3 | 8 | 0.436s (2.58× vs D2) | 0.261s (1.56× vs D2) |

`vmap` is flatter than unrolled, confirming the op-count→compile link. (This
micro-scale — 8 tiny blocks — does not reproduce the full-sweep wall, e.g. 2111s
at D3 χ12; it is a mechanism check, not a wall reproduction.)

## Conclusion — the conclusion changes

1. **"Padded-`vmap` converges to dense" is FALSE for even-D fermionic.** The
   entire CTM environment is block-shape-uniform at even D (measured χ=8, 16), so
   the padded-`vmap`/`scan` port has zero padding waste, O(1) compile, and the full
   2× sparsity win. It converges to dense only at **odd D=3** (site fragments to
   all-distinct shapes) and for diverse-charge symmetries (U(1)-Sz, cf.
   `566-u1sz-stacking-nogo`).

2. **What was actually measured-NO-GO is the *partial* realization**, not the
   lever. The batched contraction (#568, `TENAX_BATCH_BLOCKSPARSE`) groups
   same-shape blocks — at even D that is exactly the single-stack `vmap` — yet
   #618/#627 found it doesn't win (warm ~0.90×, compile −21% capping). Reason: it
   batches **only the contraction**, leaving `_fuse_indices_symmetric` and the
   Python sweep/convergence loop as eager per-block host work (the measured warm
   wall, #618).

3. **The genuinely-untested lever** is the FULL port the plan called the
   "structural lever": contraction + **padded `_fuse_indices_symmetric`** + SVD +
   fixed-point as ONE jitted `lax.scan` graph over padded-`vmap` blocks. At even-D
   fermionic its feared obstacle (padding waste) is now measured to be **zero**, so
   "every tractable JAX lever exhausted" (`566-cadjoint-nogo-closure`) was an
   **overclaim**. The gating risk has shifted from padding waste to the
   padded-fusion engineering (the plan's "chain-breaker", Task 5b) and whether one
   jitted scan graph collapses both the compile (2111s) and the host-bound warm
   wall (#618).

## Recommendation

- **Odd D=3 / U(1)-Sz:** dense stays pragmatic (padding reconstructs dense).
- **Even-D fermionic (D=2,4,6,8 — the large-D regime that matters):** the
  padded-`vmap`+`scan` port is **un-refuted** and worth a focused spike, gated on
  the padded-`_fuse_indices_symmetric` crux. This is the JAX-native analog of
  YASTN's "one big kernel", and even D is exactly where YASTN's sym-vs-dense
  crossover lives (~D8).
