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

**P-B0 — JAX↔GPU contraction bridge smoke. ✅ DONE (A100, commit `df1b92f` on branch
`feat/cutensornet-backend-200`; `examples/probe_200_cutensornet_smoke.py`).** A dense `ij,jk->ik`
routed through cupy's cuTENSOR inside one `jax.pure_callback` returns a JAX array under jit as
**exactly 1 opaque op in the jaxpr**, f64 bit-exact, c128 8.88e-16. Plumbing notes: `pure_callback`
needs `JAX_PLATFORMS=cuda,cpu` (cuda alone has no CPU device to stage host inputs) + `JAX_ENABLE_X64=1`;
cuQuantum installed transiently (`cuquantum-cu13` + `nvmath-python` + `cupy-cuda13x`), **not yet in
`pyproject`**. (`src/tenax/contraction/cutensor_blocksparse.py` remains a NON-seam-conformant API
reference only — do not reuse as the backend.)

### Decision surfaced by P-B0: `pure_callback` vs `jax.ffi` (read before P-B1)
P-B0 used the **callback** bridge (cheapest faithful route). It splits the two #200 goals:
- **Compile (#566): callback likely WINS** — 1 op/contraction, so the per-block structural graph
  (incl. the fixed-point backward) collapses. Host round-trip does not affect compile-graph size.
- **Runtime (#195): callback likely LOSES (maybe badly)** — `pure_callback` does a
  device→host→device round-trip *per contraction*, and a CTM sweep is ~60–97 contractions ×
  many iterations. That is the opposite of the tiny-kernel fix. The on-device win needs
  **`jax.ffi.ffi_call`** (a C++/CUDA handler calling cuTensorNet on the device buffers, no transfer).

**Strategy:** use `pure_callback` as the **correctness + compile-win scaffold** through P-B1→P-B3;
let the **P-B4 warm-step measurement be the go/no-go for building the `jax.ffi` on-device handler**
(almost certainly needed for runtime — but measure it, don't assume). Keep the backend's `execute`
backend-shaped so swapping callback→FFI later is internal to `CuTensorNetBackend.execute`.

**P-B1 — forward block-sparse kernel (still callback).** Build a `CuTensorNetBackend.execute(
operand_stacks, plan)` that produces the **canonical stacked output array** (rows in
`plan.out_block_keys` order). Generalize P-B0's callback from dense `ij,jk->ik` to the block-sparse
forward: for each `PlanGroup`, gather the operand rows (`group.operand_rows`), run the batched
group contraction via cuTENSOR inside ONE `pure_callback` (or one per group — fewer is better),
and accumulate by `segment_ids`/`num_segments` then reorder by `canonical_perm` — i.e. the same
shape as `stacked_execute` (`blocksparse_plan.py`), but the einsum runs on the GPU kernel. The plan
is now self-contained (commit `34347b2`): everything you need is on it — `subscripts`,
`batched_subscripts`, `groups`, `out_block_*`, and `operand_*` metadata.
Validate, even-D `ferm_D2`/`ferm_D4`, real + complex128:
1. **VALUE** == `StackedJaxBackend().execute(operand_stacks, plan)` at fp tier
   (`tests/stacked/_harness.py::assert_tiered(..., tier="fp")`, 1e-12) — `StackedJaxBackend` is the
   correctness oracle, identical block structure.
2. **OP-COUNT** (the compile-collapse premise, on a REAL block-sparse contraction not just dense):
   `len(jax.make_jaxpr(lambda s: backend.execute(s, plan))(operand_stacks).jaxpr.eqns)` should be
   O(#groups) callback ops — NOT O(#blocks) structural ops. Confirm it's a handful, independent of
   block count. This is the direct evidence the kernel route fixes #566.
Forward + value/op-count only here; VJP is P-B2.

**P-B2 — wrap in `jax.custom_vjp` + reuse `backward_contraction`.** Copy
`MockFFIBackend._execute_opaque` verbatim in structure (`blocksparse_backend.py`): forward = your
cuTensorNet `execute`; `opaque_fwd` residual = operand stacks only (no forward intermediates →
autodiff can't leak); `opaque_bwd` = `backward_contraction(operand_stacks_res, out_cotangent_stack,
plan, wrt)` for each operand. **NB (post-`34347b2`):** the signature is now plan-sourced —
`backward_contraction(operand_stacks, out_cotangent_stack, plan, wrt)`, NO `subscripts`/`fwd_operands`
args (it reads `plan.subscripts` + `plan.operand_*` and rebuilds operands itself). So the backend
needs nothing but `(operand_stacks, plan)` — the protocol gap is already closed. The backward is
itself a contraction, so under the **callback** route it becomes another `pure_callback` (the
transposed contraction on GPU); under **FFI** it's another `ffi_call`. Register in
`_select_cutensornet`, guarded by `available()` (cupy/cuQuantum import check).

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
Run **all three** backends at the **same χ** (use the profiler default χ-factor 3, matching the
A100 stacked-vs-per-block run §C, so the numbers are apples-to-apples — do NOT override `--chi-factor`):

```bash
for be in perblock stacked cutensornet; do
  TENAX_BLOCKSPARSE_BACKEND=$be uv run python examples/profile_ctm_ad_wall_566.py \
    --D 2 4 --sym fermionic --depth 8 --reps 3 --json profile_$be.json
done
```

Report `bwd_cmp` + `vg_cmp` (compile) **and** warm-step for all three. Measured A100 baselines to
beat (fermionic, **χ = 3·D**, from §C of `profile_566_a100_summary.md`):

| D=4, χ=12 | per-block | **stacked** (current best) | dense floor (χ=16) |
|---|---|---|---|
| `bwd_cmp` | 880 s | **546 s** | ~13 s |
| `vg_cmp`  | 1418 s | **1061 s** | ~39 s |

(Per-block at the higher χ=4·D is worse still: D=4 `vg_cmp` ≈ 2379 s, §A.) Warm-step baseline not yet
measured for stacked/per-block — capture it in this same run. **Targets:** `bwd_cmp` collapses from
the **546 s stacked baseline toward the ~13 s dense floor** (that ~42× residual is the remaining
order-of-magnitude #200 must deliver); warm step approaches dense (#195/#200 tiny-kernel fix).

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
2. **Compile (the bar to beat):** fermionic `bwd_cmp` at D=4, χ=12 drops from the **stacked
   baseline 546 s toward the ~13 s dense floor** (per-block is 880 s; stacked already banked −38%).
   The remaining ~42× from stacked→dense is the order-of-magnitude #200 must deliver; beating only
   the 546 s stacked number by a little is NOT success — the goal is minutes→seconds. Equivalently
   `vg_cmp` 1061 s → toward dense ~39 s.
3. **Runtime:** warm step approaches dense (#195 tiny-kernel launches eliminated). NB: the
   stacked-vs-per-block warm-step is not yet measured — capture it in the same P-B4 run so
   cuTensorNet's runtime has a stacked baseline too, not just per-block.

If 1 holds but 2/3 disappoint, that's a real finding about cuTensorNet block-sparse overhead on this
problem size — bring the numbers back before scaling to Pallas/TPU.
