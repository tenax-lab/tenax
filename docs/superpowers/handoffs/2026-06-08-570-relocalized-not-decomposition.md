# #570 re-localized: the "SVD VJP" wall is per-sector STRUCTURAL emission, not decomposition

**Date:** 2026-06-08
**Context:** PR #589 localized the CTM-AD compile wall to the `svd_vjp` bucket (61% of
the fused backward at D=4/χ=12). Two follow-up experiments tested the obvious
decomposition levers; **both are falsified**, and drilling into the bucket shows why.

## Experiment 1 — cheaper decomposition (SVD→eigh): FALSIFIED
`truncated_svd_via_eigh_ad` (eigh of the Gram matrix + reconstruct `U=M·V/s`) as a
drop-in for the full `(U,s,Vh)`: backward = **274 vs 265 ops** (no win). Mechanism:
`eigh(MᴴM)` has eigenvalues `S²`, so its F-matrix `1/(sᵢ²−sⱼ²)` is identical to the
SVD's — no fundamental saving. (Spec
`2026-06-08-svd-via-eigh-fishman-projector-570.md`, marked FALSIFIED.)

## Experiment 2 — batch the per-sector decomposition (#572 flag): FALSIFIED
`TENAX_BATCH_BLOCKSPARSE=1` at D=4/χ=12: svd **kernels 48→24** (sectors *do* batch),
but **svd_vjp ops 92,368→91,760 (−0.7%)** and TOTAL **150,663→154,858 (+2.8%)** from
stacking overhead. Batching the decomposition does not shrink the bucket.

## Why — drill-down of the svd_vjp bucket (92,368 ops, D=4/χ=12)

Top *innermost* emitters within the bucket:

| op | share | what it is |
|---|---:|---|
| `broadcast_in_dim` | 20.6% | per-sector scaling / block expand |
| `scatter_mul` | 10.8% | reverse-mode of per-sector slicing |
| `_gauge_fix_symmetric_svd` | 10.3% | per-sector gauge fix |
| `squeeze` | 10.2% | block pack/unpack |
| `reshape` | 7.2% | block pack/unpack |
| `slice` | 7.2% | block pack/unpack |
| `concatenate` | 3.6% | sector → bond assembly |
| `argmax`/`abs`/`lt`/`gt`/`where`/`select` | ~17% | `_fix_svd_signs` sign logic |
| `svd` primitive / F-matrix `dot_general` | ~0% | the decomposition itself |

The bucket is **per-sector block pack/unpack & scaling (~60%) + per-sector
gauge-fixing (~25%)**. The decomposition math is negligible. This is the **#566
per-block structural emission**, localized in `_truncated_svd_symmetric_traced`,
`_compute_2x2_projector_symmetric`, and `_gauge_fix_symmetric_svd` — emitted once
per charge sector, growing with χ because more sectors survive truncation.

(It was attributed to `svd_vjp` because those structural ops are emitted lexically
*inside* the SVD/projector functions — correct attribution; the name "SVD VJP" is
just misleading about the underlying cost.)

## Consequence for #570

The decomposition is **not** the lever. QR-CTMRG-as-cheaper-decomposition and
decomposition-batching are both dead for the *compile* cost. The wall is the
per-sector structural block-assembly + gauge-fixing — the same representation cost
the stacked-block contraction backend (PR #586) fixed for contractions but which
still runs per-sector in the SVD/projector wrapper. The viable levers are:

1. **Truncated backprop (recommended next).** Differentiate through K CTM sweeps
   instead of the full fixed point. It reduces the compile/runtime backward
   *regardless of per-sweep cost* — orthogonal to the representation issue, and a
   well-scoped change validated against the exact fixed-point adjoint
   (variational-bound + energy/grad parity). This is the Yang/Corboz #570 lever
   that actually bites here.
2. **Stacked-block representation extended to the SVD/projector wrapper.** Batch the
   per-sector pack/unpack + gauge-fix across sectors (one stacked op, not a per-
   sector loop) — the #566 deep restructure, now precisely localized. High blast
   radius; the larger prize but the harder build.
3. **Batch/vectorize `_gauge_fix_symmetric_svd` + `_fix_svd_signs` across sectors**
   (~25% of the bucket) — a smaller, contained sub-lever independent of (2).

## Reproduce
```bash
# bucket attribution (flag off vs on):
JAX_PLATFORMS=cpu TENAX_BATCH_BLOCKSPARSE=0 uv run python \
  examples/probe_bwd_subop_attribution_570.py --mode buckets --D 4 --chi 12 --full
JAX_PLATFORMS=cpu TENAX_BATCH_BLOCKSPARSE=1 uv run python \
  examples/probe_bwd_subop_attribution_570.py --mode buckets --D 4 --chi 12 --full
# drill-down (innermost emitters within the svd_vjp bucket): see the script in
# this PR's commit message / the inline analysis.
```
