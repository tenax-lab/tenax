# Design — vectorize `_gauge_fix_symmetric_svd` (per-column → per-sector)

**Date:** 2026-06-08 · **Issue:** #566 (contained sub-lever) · **Branch:** `perf/gauge-fix-symmetric-svd-vectorize-566`

## Context

The #570 compile-wall investigation localized the fermionic CTM-AD compile cost to **#566
per-sector structural emission** inside the SVD/projector wrapper: ~60% block pack/unpack +
**~25% gauge-fix/sign-logic**, ~0% decomposition (`docs/superpowers/handoffs/2026-06-08-570-relocalized-not-decomposition.md`,
`…-batching-compile-finding.md`). Of the three originally-listed vectorization targets, two are
already done — `_fuse_indices_symmetric`'s scatter (vectorized to one scatter-per-block in #569
Milestone A) and `_fix_svd_signs` (already `argmax(axis=0)` + broadcast). The remaining real
target is **`_gauge_fix_symmetric_svd`** (`src/tenax/algorithms/_ctm_tensor_projector_2x2.py`).

## Problem

`_gauge_fix_symmetric_svd` loops over **every bond column `j` (≈χ)** and per column emits an
`argmax` + `concatenate` + a per-block `.at[..., local].multiply(...)` (lines ~104–141). The
emitted-op count therefore grows with χ — this is the gauge-fix slice of the wall and one of
the `scatter_mul`/`argmax` contributors #589 measured. The loop is purely a representation cost;
the math is per-sector.

## Approach (A — per-sector batched)

Hoist the loop from columns (≈χ, many) to **bond-charge sectors** (`n_sectors`, few — 2 for Z₂).
Interface is unchanged: `_gauge_fix_symmetric_svd(U_T, Vh_T) -> (U_out, Vh_out)`. Pure internal
rewrite; no call sites, signatures, or output semantics change.

For each bond charge `q` that has U-blocks (iterating `u_blocks_by_q[q]` in the **same order** the
current code concatenates):

1. Reshape each U-block (shape `(...non-bond..., n_q)`) to `(rows_i, n_q)` and `concatenate` along
   axis 0 → `M_q` of shape `(R, n_q)`, **byte-identical layout to the current per-column
   `candidates` concatenation** (so `argmax` tie-breaking is identical).
2. Vectorized over all `n_q` columns at once:
   - `idx = jnp.argmax(jnp.abs(M_q), axis=0)` → `(n_q,)`
   - `best = M_q[idx, jnp.arange(n_q)]`
   - `phase = jnp.where(jnp.abs(best) > 0, best / jnp.maximum(jnp.abs(best), 1e-30), 1)`
   - real/complex split preserved: `conj_phase`/`bare_phase` as today.
3. Multiply each U-block (charge `q`) by `conj_phase` broadcast over its column axis (one
   broadcast-multiply per block, replacing per-column `.at[].multiply`); multiply each Vh-block
   (charge `q`) by `bare_phase` broadcast over its row axis.
4. Charges with no U-blocks are skipped (preserves the current `continue`).

This replaces ≈χ `argmax`/`concatenate`/scatter-multiply emissions with ≈`n_sectors` batched ones.
Sector grouping is derived from **static numpy charges** (`bond_idx.charges`), so it is
JIT/trace-safe and does not introduce data-dependent control flow.

### Why not B/C
- **B (fully global, masked):** one `(R, χ)` masked structure with no sector loop needs padding for
  ragged per-sector row counts; `n_sectors` is tiny so the per-sector loop already captures the win
  without padding complexity.
- **C (multiply-only):** keeps the per-column `argmax`/`concatenate` loop — the bulk — so it barely
  moves compile. Rejected.

## Correctness

**Bar: bit-identical** `U_out` and `Vh_out` versus the current implementation on random multi-sector
SVD outputs (order-preservation in step 1 guarantees identical `argmax`), plus **gradient** parity
(the function is on the AD-traced projector path). Edge cases preserved: empty/absent bond sector,
single sector, fully-degenerate columns, real vs complex128, zero `best` (phase→1).

Invariants that must continue to hold (the reason this gauge fix exists):
- `U @ diag(s) @ Vh == M` per sector (reconstruction unchanged — phases cancel).
- max-|U| row of each kept column is real-positive.
- the 2×2 closure `P_bot · P_top = I` (no intervening matrix to absorb `conj(phase)²`).

## Testing (TDD)

1. **Reference-parity test (new).** Freeze the current loop as an inline reference in the test;
   assert the vectorized impl matches it **bit-identically** for `U_out`, `Vh_out`, and
   `jax.grad` of a scalar gauge-invariant loss, across U(1) / Z₂ / FermionParity tensors, real and
   complex128, including the edge cases above. Write this first (red), then implement.
2. **Existing tests must pass unchanged:** `tests/test_ctm_2x2_projector_symmetric.py`
   (`test_gauge_fix_symmetric_svd_preserves_reconstruction`, `…_real_positive_max_row`), and the
   broader symmetric-CTM AD suite (`tests/stacked/`, `tests/test_block_sparse_ctm_ad.py`).
3. Mark new tests `core` if they fit the fast tier (mechanism-level, small tensors).

## Measurement

Before/after on `examples/profile_570_sweepvjp_compile.py` (D=4, χ ∈ {8,12,16}, `--full`): expect
the gauge-fix slice to shrink and the reduction to **grow with χ** (more columns collapsed).
Report HLO instruction count + compile-time delta. This is a structural win independent of
`TENAX_BATCH_BLOCKSPARSE`.

## Scope / non-goals

- **In scope:** rewrite of `_gauge_fix_symmetric_svd` only; its parity tests; a before/after
  compile measurement.
- **Out of scope:** the remaining per-block transpose/reshape/scatter in `_fuse_indices_symmetric`
  (cross-block batching = #566 fix #1), and the sweep-level stacked representation. No interface or
  default-behavior changes; always-on (correctness-neutral, no gate).

## Risks

- **`argmax` tie order:** mitigated by preserving the exact U-block concatenation order; covered by
  the bit-identical parity test.
- **Ragged per-sector row counts across U-blocks:** handled by `concatenate` along the flattened
  row axis (all blocks of a sector share `n_q` columns); no padding needed.
- **Dtype promotion:** preserve the static `is_complex` check so real blocks are not promoted.
