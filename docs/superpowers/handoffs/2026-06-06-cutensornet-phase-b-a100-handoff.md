# cuTensorNet backend (Phase B) — A100/CUDA handoff (#200)

**Date:** 2026-06-06
**Depends on:** PR #586 (Phase A seam — branch `docs/symmetric-ctm-ad-stacked-design-566`).
**Goal:** a GPU `CuTensorNetBackend` that does the **forward** block-sparse contraction as a
single op, slotting into the existing seam. The plan, dispatch, and the **hand-written VJP are
already done and proven on CPU (float64 + complex128)** — Phase B is the forward kernel + the
A100 measurement.

You do NOT need #586 merged — branch Phase B off `docs/symmetric-ctm-ad-stacked-design-566`
(you get the whole seam). Wait only for #586's CI to go green before building on it.

---

## 0. What's already done (do not rebuild)

| Piece | Where | Status |
|---|---|---|
| Static plan | `blocksparse_plan.py` `build_block_contract_plan` → `BlockContractPlan` | done; no tensor data |
| Pure-JAX reference backend | `blocksparse_backend.py` `StackedJaxBackend` | done; your value oracle |
| Backend protocol + dispatch | `blocksparse_backend.py` `BlockSparseContractBackend`, `select_backend`, `_select_cutensornet` (stub → returns None) | done; plug in here |
| Hand-written VJP | `blocksparse_vjp.py` `backward_contraction(subscripts, fwd_operands, out_cotangent_stack, plan, wrt)` | **done; reuse verbatim** (real+complex128, fp 1e-12) |
| Opaque-backend template | `blocksparse_backend.py` `MockFFIBackend._execute_opaque` (lines ~154-196) | **copy this custom_vjp pattern** |
| Accuracy spine | `tests/stacked/_harness.py` (bit / bounded-fp / gauge-invariant) | use for validation |
| Profiler | `examples/profile_ctm_ad_wall_566.py` | use for the A100 measurement |

---

## 1. Prereqs on the A100 box

- CUDA + cuQuantum / cuTensorNet (`cuquantum-python` or the `nvidia-cuquantum-cu12` wheels).
- JAX with the matching CUDA build; Tenax forces `jax_enable_x64=True` on import.
- `git fetch && git checkout -b feat/cutensornet-backend-200 origin/docs/symmetric-ctm-ad-stacked-design-566`
- Sanity: `JAX_PLATFORMS=cuda uv run python -c "import jax; print(jax.devices())"` shows the GPU.

---

## 2. The contract the backend must honour

`BlockSparseContractBackend` (`blocksparse_backend.py:36`):

```python
def available(self) -> bool          # CUDA present + cuTensorNet importable
def supports(self, tensors, plan) -> bool   # dtype/shape it can run (else dispatch falls back)
def execute(self, operand_stacks, plan) -> Any   # -> canonical stacked output array
```

- `operand_stacks`: per forward operand, its stacked block array `(n_blocks, *block_shape)`
  (from `tensor.stacked_blocks()`; for even-D single-shape-group there's exactly one group/operand).
- **Output contract:** return the canonical-ordered stacked output array of shape
  `(len(plan.out_block_keys), *plan.out_block_shapes[0])`, rows in **`plan.out_block_keys` order**.
  The contractor's assembly and `backward_contraction` both assume this canonical row order.

`BlockContractPlan` (`blocksparse_plan.py:63`) gives the kernel everything static:
`out_indices`, `out_block_keys`, `out_block_shapes`, `out_block_offsets`, `total_size`,
`batched_subscripts`, and `groups: tuple[PlanGroup]` where each `PlanGroup` has
`operand_rows` (which stacked rows of each operand feed the batched contraction),
`segment_ids` + `num_segments` (output accumulation), and `canonical_perm`.

---

## 3. The ONE anticipated seam change: per-operand metadata on the plan

`execute(operand_stacks, plan)` does **not** carry the per-operand block metadata
(`indices`, `block_keys`, `block_shapes`, `block_offsets`) that the **backward** needs to rebuild
operands for `backward_contraction`. `MockFFIBackend` worked around this by capturing it at
construction (`operand_meta`, `blocksparse_backend.py:123`). For a real backend, make it clean:

- **Recommended (a):** add per-operand metadata to `BlockContractPlan`
  (`operand_indices`, `operand_block_keys`, `operand_block_shapes`, `operand_block_offsets` —
  tuples parallel to the inputs). `build_block_contract_plan` already has the tensors, so this is
  a small additive edit; both `CuTensorNetBackend` and `StackedJaxBackend` then read it from the
  plan, and `backward_contraction` can take the plan instead of separately-passed `fwd_operands`.
- Alternative (b): widen the protocol to `execute(tensors, operand_stacks, plan)`.

Do (a). This is the single API edit Phase B drives back into the seam — make it on the Phase B
branch, keep `tests/stacked/` green, and it folds back cleanly.

---

## 4. Build order (de-risk the novel/GPU-specific parts first)

**P-B0 — cuTensorNet FFI smoke (highest unknown).** Can you call cuTensorNet from JAX as a
`custom_call`/FFI at all on this stack? Do a trivial *dense* pairwise contraction, forward only,
GPU. This answers the riskiest question (FFI plumbing + library availability) before touching the
seam. Mine `src/tenax/contraction/cutensor_blocksparse.py` (`contract_blocksparse`, `is_available`)
for the cuTensorNet **calling pattern** — but note it is the **legacy, NON-seam-conformant** path
(`TENAX_USE_CUTENSOR_BLOCKSPARSE`, eager, full-tensor `custom_vjp`, non-fermionic); **do not reuse
it as the backend**, only as an API reference.

**P-B1 — forward block-sparse kernel.** For the even-D double-layer plan (build it from
`tests/stacked` ferm_D2/D4), feed `operand_stacks` + `plan` to cuTensorNet and produce the
canonical stacked output. Two viable shapes: (a) cuTensorNet does the grouped block-sparse
contraction natively; (b) you drive the per-`PlanGroup` batched contraction via its API and
`segment_sum`-accumulate (mirror `stacked_execute`). Either way the **output must be the canonical
stacked array**. **Validate VALUE** against `StackedJaxBackend().execute(operand_stacks, plan)` at
fp tier (`assert_tiered(..., tier="fp")`, 1e-12), real + complex128.

**P-B2 — wrap in `jax.custom_vjp` + reuse `backward_contraction`.** Copy
`MockFFIBackend._execute_opaque` (`blocksparse_backend.py:154-196`) verbatim in structure: forward
= your cuTensorNet `execute`; `opaque_fwd` residual = operand stacks only (no forward intermediates
→ autodiff can't leak); `opaque_bwd` = `backward_contraction(subscripts, fwd_operands,
out_cotangent_stack, plan, wrt)` for each operand. Register the backend in `_select_cutensornet`
(`blocksparse_backend.py:266`), guarded by `available()`.

**P-B3 — validate against the spine.** Swap `MockFFIBackend` → `CuTensorNetBackend` in the
`tests/stacked/test_vjp_seam.py` pattern: assert `value` and `jax.grad` match per-block AND
`StackedJaxBackend`, **real and complex128**, ferm_D2/D4, fp tier. Run the whole
`tests/stacked/` on GPU. (Note: GPU `segment_sum` reduction order can drift ~5e-7 — that lives in
the bounded-fp tier, not bit-identical; the energy/grad assertions should still hold at 1e-12 in
f64. Never compare raw SVD factors — N/A here, contraction only.)

**P-B4 — THE MEASUREMENT (the point of #200).**
```bash
# baseline (per-block) vs cuTensorNet, fermionic, A100, x64, implicit/fixed_point
TENAX_BLOCKSPARSE_BACKEND=perblock    uv run python examples/profile_ctm_ad_wall_566.py \
  --D 2 3 4 --sym fermionic --depth 8 --reps 3 --json profile_perblock.json
TENAX_BLOCKSPARSE_BACKEND=cutensornet uv run python examples/profile_ctm_ad_wall_566.py \
  --D 2 3 4 --sym fermionic --depth 8 --reps 3 --json profile_cutensornet.json
```
Report `vg_cmp` (compile) + warm-step for both. Reference baselines (A100, per-block):
compile D2/3/4 ≈ 206 / 2111 / 2379 s; warm step ≈ 4.5 / 5.5 / 9.5 s. **Targets:** compile collapses
toward dense (~40-60 s, χ/D-flat) — the original ≥10×-at-D4 framing now applies to the kernel; warm
step approaches dense (~0.5-2.3 s) — the #195/#200 tiny-kernel fix.

---

## 5. Scope notes / gotchas

- **Default OFF.** cuTensorNet is selected ONLY via `TENAX_BLOCKSPARSE_BACKEND=cutensornet` (and
  `available()`); never auto-selected, so the default path stays byte-identical.
- **Odd-D / ragged is cuTensorNet's real advantage** (native ragged blocks) over the pure-JAX
  stacked path — BUT `build_block_contract_plan` currently returns `None` for multi-shape-group
  (odd-D, U(1)), so those fall back to per-block today. Exercising odd-D needs a ragged-plan
  extension (a follow-up). For the first measurement, **even-D D2/D4 suffices to prove the lever**;
  odd-D D3 is where cuTensorNet should later beat both paths.
- **complex128:** `backward_contraction` applies **no conjugation** and is proven correct for
  complex (the VJP of a bilinear contraction transposes the unconjugated surviving operand;
  conjugation enters only via non-holomorphic leaf ops, outside the seam). cuTensorNet iPEPS/CTM
  runs complex128 — validate it.
- **Canonical order:** if cuTensorNet does its own accumulation, the returned rows MUST be in
  `plan.out_block_keys` order (assembly + backward assume it).
- **Energies:** compare only at **full CTM convergence** (P1d lesson — early-iteration SVD-projector
  gauge freedom makes raw env data differ under 1e-16 perturbations while the gauge-invariant energy
  converges identically).

---

## 6. Success criteria (the #200 gate)

1. **Correctness:** `CuTensorNetBackend` value + grad == per-block within fp 1e-12, real AND
   complex128, ferm_D2/D4 (the `test_vjp_seam` pattern on GPU).
2. **Compile:** fermionic `vg_cmp` at D=4 collapses toward dense (from ~2379 s — order-of-magnitude
   reduction).
3. **Runtime:** warm step approaches dense (#195 tiny-kernel launches eliminated).

If 1 holds but 2/3 disappoint, that's a real finding about cuTensorNet block-sparse overhead on this
problem size — bring the numbers back before scaling to Pallas/TPU.
