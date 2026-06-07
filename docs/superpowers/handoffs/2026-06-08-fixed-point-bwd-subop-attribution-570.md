# Fixed-point backward sub-op attribution — the wall is the block-sparse SVD VJP (#570)

**Date:** 2026-06-08
**Context:** The #200 P-B4 measurement (PR #587) settled that the symmetric-iPEPS
CTM-AD compile wall is the implicit-diff backward `_jit_fused_fixed_point_bwd` and
is **backend-invariant** (per-block / stacked / cuTensorNet all ~549 s at D=4/χ=12
on A100). The NO-GO handoff prescribed one next step: *profile WHICH sub-ops of the
backward dominate* before committing #570 work. This is that profile — CPU,
trace-only, source-attributed.
**Tool:** `examples/probe_bwd_subop_attribution_570.py`
**Verdict:** **#570 is confirmed.** The χ-scaling driver of the backward op count is
the **block-sparse SVD VJP** — 61% of the entire fused backward at the D=4/χ=12 bar,
and the *only* term that grows with χ. Everything else is χ-flat.

---

## Method (why this differs from the existing primitive histogram)

`probe_backward_jaxpr_566.py` buckets ops by *primitive type*, which **hides** the
SVD cost: an SVD-VJP emits dozens of `dot_general` / `transpose` / `div` /
`broadcast` ops that get miscounted under contraction/transpose/elementwise. This
probe attributes **every** op to the **source function** that emitted it, via the
equation's traceback (`eqn.source_info.traceback.frames`), recursing into all
sub-jaxprs. So the full dense-algebra footprint of the SVD differentiation is
counted where it belongs. Ordered categories (decomp checked first, so an op
emitted *inside* svd is credited to svd, not the enclosing projector):
`svd_vjp` / `eigh_vjp` / `projector` / `contraction` / `structural` / `other`.

Trace-only (no XLA compile) → runs in seconds at any D/χ; the op-count breakdown is
platform-independent. **Validation:** the probe's D=4/χ=12 total (150,663 ops)
reproduces the independently-recorded backward op count, confirming the unit.

Profiled unit = the full fused backward (env-sweep VJP **+** params-sweep VJP,
`--full`), fermionic FermionParity site, `projector_method="svd"` (the default).

## Data (full fused backward, fermionic)

### D=2 χ-sweep

| χ | TOTAL | svd_vjp | eigh_vjp | projector | contraction | structural | decomp% | svd kernels |
|---:|------:|--------:|---------:|----------:|------------:|-----------:|--------:|-----:|
| 4  | 59,079 | 20.7% | 0% | 15.7% | 7.9% | **50.2%** | 36.5% | 48 |
| 6  | 63,543 | 25.4% | 0% | 14.6% | 7.3% | 46.6% | 40.0% | 48 |
| 8  | 68,007 | 29.5% | 0% | 13.7% | 6.8% | 43.6% | 43.2% | 48 |
| 12 | 76,935 | 36.2% | 0% | 12.1% | 6.0% | 38.5% | 48.3% | 48 |
| 16 | 85,863 | 41.5% | 0% | 10.8% | 5.4% | 34.5% | 52.4% | 48 |
| 24 | 103,719 | **49.4%** | 0% | 9.0% | 4.5% | 28.6% | 58.4% | 48 |

### D=4 χ-sweep (the A100 bar is χ=12)

| χ | TOTAL | svd_vjp | eigh_vjp | projector | contraction | structural | decomp% | svd kernels |
|---:|------:|--------:|---------:|----------:|------------:|-----------:|--------:|-----:|
| 8  | 117,159 | 53.8% | 0% | 7.9% | 4.0% | 25.3% | 61.8% | 48 |
| **12** | **150,663** | **61.3%** | 0% | 6.2% | 3.1% | 19.7% | 67.5% | 48 |
| 16 | 184,167 | 66.1% | 0% | 5.1% | 2.5% | 16.1% | 71.1% | 48 |

## The finding (five facts)

1. **The SVD VJP is the χ-scaling driver.** Its absolute op count grows
   12,240→51,280 (4.2×) over χ=4→24 at D=2 — that growth is **87% of the entire
   backward's op growth**. At D=4 it reaches **61% of the whole fused backward at
   χ=12** (66% at χ=16). It is the only category whose share rises with χ.

2. **Everything else is χ-flat in absolute op count.** projector 9,304,
   contraction 4,653, structural 29,634 — **constant across all χ AND identical at
   D=2 and D=4.** They scale with *block count* (16, same for even D), not with D or
   χ. They are the per-block representation scaffolding, not a χ lever.

3. **It is specifically the SVD, not "SVD/eigh/projector".** `eigh_vjp = 0` at the
   default SVD projector (eigh only appears for `projector_method="eigh"`/C4v), and
   the projector *algebra* (Fishman matmuls around the SVD) is χ-flat. The A100
   "block-sparse SVD/eigh/projector VJP" is, concretely, the **SVD VJP**.

4. **Compile-time under-counts it further.** The 48 SVD custom-calls are χ-flat in
   *count*, but each is disproportionately expensive to *compile* (XLA SVD lowering
   ≫ a generic elementwise op) and grows more expensive at large χ. So in
   compile-*time* terms the SVD dominance exceeds even its 61% op-share — consistent
   with the A100 wall being ~100% `_jit_fused_fixed_point_bwd`.

5. **Mechanism (why op count grows with χ at all).** The symmetric SVD
   (`_truncated_svd_symmetric_traced`) emits ops **per surviving charge sector** on
   the χ-bond; the number of sectors that survive truncation grows with χ → more
   per-sector SVD+VJP emission. So the SVD-VJP growth is *itself* per-block (charge-
   sector) emission, now on the χ-bond rather than the D-bonds.

## Implications for #570

The two #570 sub-levers map cleanly onto this data:

- **QR projector instead of SVD** → directly replaces the 61%-and-growing `svd_vjp`
  term. QR's VJP has no degeneracy/Lorentzian machinery and batches cleanly across
  sectors, so it should compile far cheaper than the truncated-symmetric-SVD VJP.
  **This is the highest-leverage compile fix at large χ.** (Tenax already supports
  `projector_method="qr"` — needs energy/variational parity validation.)
- **Truncated backprop** → shrinks the whole backward proportionally (fewer
  differentiated sweeps); orthogonal, helps every category including the SVD term.

Two caveats the data makes explicit:

- **At small χ / large D the structural per-block cost dominates** (50% at D=2 χ=4),
  and QR does **not** touch it (block↔dense pack/unpack is representation cost, not
  decomposition). But it is **χ-flat**, so it is a shrinking fraction exactly in the
  large-χ regime #570 targets — at D=4/χ=12 structural is already only 20%. The
  large-χ wall is genuinely the SVD; the structural cost is the *separate* #566
  per-block-emission axis (stacked backend = necessary-but-insufficient, already
  measured).
- **Batching the per-sector SVDs** (the χ-flat 48 kernels emit per-sector) is a
  third lever orthogonal to QR — cf. the gated `TENAX_BATCH_BLOCKSPARSE` batched-SVD
  (#572, default-off). Worth re-evaluating specifically for the SVD term now that we
  know it is the χ driver.

## Reproduce

```bash
JAX_PLATFORMS=cpu uv run python examples/probe_bwd_subop_attribution_570.py \
    --mode raw     --D 2 --chi 6 --full      # top (file::function) emitters
JAX_PLATFORMS=cpu uv run python examples/probe_bwd_subop_attribution_570.py \
    --mode buckets --D 2 --chi 4 6 8 12 16 24 --full
JAX_PLATFORMS=cpu uv run python examples/probe_bwd_subop_attribution_570.py \
    --mode buckets --D 4 --chi 8 12 16 --full
```
