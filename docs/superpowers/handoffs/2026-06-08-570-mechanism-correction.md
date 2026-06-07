# #570 correction — the AD CTM projector is block-sparse (per-sector) SVD, not dense; "lever-3" is already implemented

**Date:** 2026-06-08 · **Corrects:** [`2026-06-07-570-svd-vjp-compile-finding.md`](2026-06-07-570-svd-vjp-compile-finding.md) (lever-1), [`2026-06-08-570-lever2-truncated-backprop-nogo.md`](2026-06-08-570-lever2-truncated-backprop-nogo.md) (lever-2) · **Issue:** #570

## What was wrong

Both prior #570 finding docs described the CTM-AD compile wall as a **dense** SVD VJP on the
χ-sized corner matrix, and cited the "Task 2.2 dense fallback" in `_ctm_projector.py` as the
mechanism — concluding that a **lever-3** ("implement a block-sparse AD-traced SVD VJP") was
the remaining work. That mechanism is **incorrect**, and lever-3 (as scoped) is **moot**.

## Ground truth (code-level + verified by trace)

The production fermionic CTM-AD path uses the default `recipe="2x2"` plaquette projector.
For SymmetricTensor inputs, `_compute_2x2_projector` (`_ctm_tensor_projector_2x2.py:399-416`)
routes **all** cases — including tracer-bearing AD-backward — to the **block-sparse**
`_compute_2x2_projector_symmetric` ("tracer-safe end-to-end, Issue #435"). That calls
`tenax.linalg.svd`, which under tracing dispatches to `_truncated_svd_symmetric_traced`
(`linalg.py`), running a **per-sector** `truncated_svd_ad` on each charge block.

So under AD the backward differentiates a **per-sector block-sparse SVD**, never a dense
`jnp.linalg.svd` on the full corner. The dense `_compute_2x2_projector` →
`_gauge_fixed_svd` → `jnp.linalg.svd` branch is reached **only for all-DenseTensor inputs**.
The "Task 2.2 dense fallback" (`_ctm_projector.py:~946-962`) lives in the **1×1** recipe
(`_compute_projector_tensor`), which the production path **does not use**.

Verification (D=2, χ=12, fermionic, `apply_Jt` traced backward):
- The 24 `svd` primitives operate on **(24,24) per-sector blocks**, not one dense χ-corner.
- The corner `C1` is a SymmetricTensor with **2 charge sectors** of (6,6) — block-sparse.
- (Method/qr/eigh still trace byte-identical because the 2×2 symmetric path has no
  `qr`/`eigh` branch — it always uses `tensor_svd`. The `projector_method`-invariance
  conclusion is unchanged; only "dense" was wrong.)

## What stands unchanged

All *conclusions* of both prior docs hold:
- The wall is the **decomposition (SVD) VJP** inside `_jit_fused_fixed_point_bwd`,
  super-linear in χ (one sweep-VJP unit 50.7 s → 522.9 s as χ 6 → 36).
- **Lever-1 (QR projector) is a no-op as a config flip** (the 2×2 path has no QR).
- **Lever-2 (truncated backprop) is a NO-GO for the compile wall** (best 0.88× implicit).

Only the dense-vs-block-sparse *mechanism* description is corrected.

## Implication — the levers, re-scoped (again)

- **Lever-3 ("implement block-sparse SVD VJP") is moot** — it already exists and *is* the
  wall. The per-sector block-SVD VJP, summed over sectors × projectors × the fixed-point
  backward, is the irreducible cost.
- The remaining structural lever to cut compile is **batching the equal-shaped per-sector
  SVD-VJP units** into one vmapped graph (fewer XLA modules) — the #566/#569
  `TENAX_BATCH_BLOCKSPARSE` axis. It is **built and benchmarked for runtime** ("never a net
  win" through D=6, A100) but its effect on the **compile wall** was never measured (the
  #569 summary is warm-step only). That is the one open, low-cost question (gate ON vs OFF,
  `total_compile_s` / `env_hlo_instr` via `examples/profile_570_sweepvjp_compile.py`).
- If batching doesn't cut compile either, the wall is **intrinsic** to differentiating N
  per-sector block-sparse SVDs in XLA, and #570 should conclude.

## How this happened (so it doesn't recur)

The lever-1 doc inferred the dense mechanism from the *1×1* recipe's "Task 2.2" comment
without confirming which recipe the production path runs, and read "SVD-only" (true) as
"dense SVD" (false). The fix was a 10-minute code trace (dispatch in
`_ctm_tensor_projector_2x2.py`) + a jaxpr shape probe. Lesson: when attributing a cost to a
specific code path, confirm the dispatch actually reaches it before writing the mechanism.
